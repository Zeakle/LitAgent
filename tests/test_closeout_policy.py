"""Keep the release policy bounded without changing historical evaluation rules."""

import copy
import json
import sys

import pytest
from test_evaluation_dimensions_identity import dimension_envelope
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.closeout import (
    PROFILE,
    profile_fields,
    profile_flags,
    recheck_prompt,
    recovery_allowed,
    result_summary,
)
from litagent.config import load_config
from litagent.eval.inventory_recovery import (
    DEFINITION_RECOVERY_INSTRUCTION,
    MIXED_CONTRACT,
    inventory_identity,
    validate_recovered_inventory,
)
from litagent.frozen_writing import plan_frozen_budget
from litagent.safety.planned_accounting import _plan
from litagent.writing_v2 import RANGE_WRITING_PLAN
from scripts import run_frozen_writing as campaign


def scope():
    """Construct the complete real envelope, including both independent parents."""
    return dimension_envelope(
        writer_source_view="source-only-writer-v1",
        semantic_conflict_policy="host-unverified-continue-v1",
        closeout_profile=PROFILE,
    )


@pytest.mark.parametrize("bad", ["", False, 1, {}, "unknown"])
def test_unknown_closeout_profile_cannot_select_an_execution_policy(bad):
    with pytest.raises(ValueError, match="closeout_profile_invalid"):
        profile_flags(bad)


def test_closeout_plan_uses_complete_existing_stack_and_distinct_identity(
    frozen_bundle,
):
    flags = profile_flags(PROFILE)
    plan = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v7", **flags)
    old = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        **{k: v for k, v in flags.items() if k != "closeout_profile"},
    )
    assert plan["status"] == "admitted"
    assert plan["identity"] != old["identity"]
    assert profile_fields(plan["envelope"])["closeout_profile"] == PROFILE
    assert profile_fields(old["envelope"]) == {}
    assert plan["maximum_tokens"] == 4500000
    assert plan["maximum_requests"] == 64
    assert plan["repair_limit"] == 2
    _plan(plan)
    with pytest.raises(ValueError, match="closeout_joint_budget_required"):
        plan_frozen_budget(
            frozen_bundle,
            budget_policy="planned-budget-v7",
            **{**flags, "round_resource_contract": None},
        )


def test_closeout_cli_reaches_actual_prepare_and_freezes_every_policy(
    case, tmp_path, monkeypatch
):
    """Run CLI, manifest production, planning and validation with portable inputs."""
    config = load_config()
    monkeypatch.chdir(tmp_path)
    archives = tmp_path / "artifacts/runs"
    archives.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        archive = {
            "run_id": run_id,
            "query": case.packet["query"],
            "events": [],
            "nodes": {
                "selection": {
                    "kind": "evidence.retrieve",
                    "output": case.selection.to_dict(),
                },
                "extract": {
                    "kind": "worker",
                    "task_id": "extract",
                    "agent_type": "extractor",
                    "output": [{"paper_id": "evit"}],
                },
            },
        }
        (archives / (run_id + ".json")).write_text(json.dumps(archive))
    baseline = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = tmp_path / "portable_source.py"
    source.write_text('"""Portable closeout preparation fixture."""\n')
    monkeypatch.setattr(
        campaign, "source_paths", lambda: [source.relative_to(tmp_path)]
    )
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    monkeypatch.setattr(campaign, "load_config", lambda: config)
    root = tmp_path / "prepared"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_writing",
            "--root",
            str(root),
            "--prepare",
            "--closeout-profile",
            PROFILE,
        ],
    )
    campaign.main()
    selected_config = campaign.profile_config(config, campaign.ROUND_PROFILE)
    manifest = campaign.validate(root, selected_config)
    for key, expected in profile_flags(PROFILE).items():
        assert manifest[key] == expected
    assert manifest["writing_plan"] == RANGE_WRITING_PLAN
    assert (
        manifest["writing_parameters"]["semantic_conflict_policy"]
        == "host-unverified-continue-v1"
    )
    for plan in manifest["budget_plans"].values():
        assert plan["status"] == "admitted"
        assert profile_fields(plan["envelope"])["closeout_profile"] == PROFILE
        assert plan["repair_limit"] == 2
    assert not (root / "state.json").exists() and not (root / ".live.lock").exists()


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "started"])
def test_sent_inventory_attempt_is_not_released_by_failure_or_new_report(status):
    envelope = scope()
    state = {"attempts": []}
    assert recovery_allowed(envelope, state)
    state["attempts"].append(
        {"purpose": MIXED_CONTRACT, "status": status, "source_report_sha256": "old"}
    )
    assert not recovery_allowed(envelope, state)
    state["attempts"][0]["source_report_sha256"] = "new"
    assert not recovery_allowed(envelope, state)


def test_two_prose_attempts_do_not_count_as_inventory_recovery():
    assert recovery_allowed(scope(), {"attempts": [{}, {}]})
    assert recovery_allowed(None, {})


@pytest.mark.parametrize("key", ["closeout_profile", "closeout_profile_identity"])
def test_incomplete_profile_pair_is_rejected(key):
    envelope = scope()
    envelope.pop(key)
    with pytest.raises(ValueError, match="closeout_profile_invalid"):
        profile_fields(envelope)


def test_unchanged_inventory_can_be_rechecked_only_on_explicit_new_path():
    claims = [
        dict(
            source_fragment_id="f",
            kind="scientific",
            claim_text="A relates to B.",
            evidence_ids=["e"],
        )
    ]
    feedback = {
        "contract": MIXED_CONTRACT,
        "previous_inventory_sha256": inventory_identity(claims),
        "seen_inventory_sha256": [],
    }
    with pytest.raises(ValueError, match="inventory_recovery_no_progress"):
        validate_recovered_inventory(claims, feedback)
    before = copy.deepcopy(claims)
    validate_recovered_inventory(claims, feedback, allow_unchanged=True)
    assert claims == before


def test_new_recheck_prompt_does_not_change_the_historical_prompt():
    assert (
        recheck_prompt(DEFINITION_RECOVERY_INSTRUCTION, None)
        == DEFINITION_RECOVERY_INSTRUCTION
    )
    current = recheck_prompt(DEFINITION_RECOVERY_INSTRUCTION, scope())
    assert "cycling inventories is no progress" not in current
    assert "single inventory-recovery slot" in current
    assert "fresh independent assessment" in current


@pytest.mark.parametrize(
    "invalid", ["skipped", "partial", "inventory", "global", "overall"]
)
def test_incomplete_evaluation_is_never_presented_as_completed(invalid):
    metric = {
        "skipped": False,
        "score": 0.5,
        "details": {
            "coverage": {"complete": True},
            "inventory_check": {"status": "passed"},
            "stages": {"global_check": "completed"},
        },
    }
    assert result_summary({"evaluation": {"faithfulness": metric}})["evaluation_valid"]
    if invalid == "skipped":
        metric["skipped"] = True
    elif invalid == "partial":
        metric["details"]["coverage"]["complete"] = False
    elif invalid == "inventory":
        metric["details"]["inventory_check"]["status"] = "failed"
    elif invalid == "global":
        metric["details"]["stages"]["global_check"] = "invalid_response"
    else:
        metric["details"]["overall_valid"] = False
    summary = result_summary({"evaluation": {"faithfulness": metric}})
    assert summary["evaluation"] == "not_complete"
    assert not summary["evaluation_valid"]
