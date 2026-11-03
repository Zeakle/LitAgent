"""Bind explicit feedback selection without upgrading historical writer identities."""

import copy
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.frozen_writing import plan_frozen_budget
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import (
    PlannedBudget,
    _digest,
    make_writer_envelope,
    request_scope,
    validate_writer_envelope,
    wire_reservation,
)
from litagent.workbench import RunOptions
from litagent.writing_v2 import writer_budget_plan
from scripts import run_frozen_writing as campaign


def options_for(policy="planned-budget-v7", **changes):
    """Use the same public choices as the pre-feedback three-domain certificate."""
    return RunOptions(
        **{
            "report_language": "zh-CN",
            "human_review": False,
            "review_structure": "academic_v2",
            "auto_audit_mode": "off",
            "auto_audit_max_versions": 1,
            "quality_policy": "strict_auto_v2",
            "verification": {"judge": False, "jev": False},
            "body_quality": "analysis_v1",
            "budget_policy": policy,
            "capacity_policy": (
                "legacy" if policy == "planned-budget-v5" else "adaptive-capacity-v1"
            ),
            "classification_review": policy == "planned-budget-v7",
            "generation_strategy": "body_summary_v3",
            **changes,
        }
    )


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            "planned-budget-v5",
            "540408b7ccc41272a9f935a3594a1b5a8c5e5a56452afe299b0d8ad186124103",
        ),
        (
            "planned-budget-v6",
            "b714b44c0e50f5c875a79e1267254bf84908f7c1a45f5810eceff1c6a96a540d",
        ),
        (
            "planned-budget-v7",
            "159b5e3d00e58fad526f5e8d63e66baf876626884fe01345a9d72598f429bc20",
        ),
    ],
)
def test_pre_feedback_fingerprints_remain_pinned(policy, expected):
    """The new inactive field cannot invalidate v5, v6 or existing v7 checkpoints."""
    options = options_for(policy)
    assert not options.repair_feedback
    assert "repair_feedback" not in options.model_dump(mode="json")
    assert options.fingerprint() == expected
    assert options_for(policy, repair_feedback=False).fingerprint() == expected


@pytest.mark.parametrize("wrong", [None, 0, 1, "true", [], {}])
def test_internal_feedback_option_is_strictly_typed(wrong):
    with pytest.raises(ValidationError):
        options_for(repair_feedback=wrong)


@pytest.mark.parametrize("policy", ["planned-budget-v5", "planned-budget-v6"])
def test_feedback_cannot_upgrade_an_older_budget_policy(policy):
    with pytest.raises(ValueError, match="repair_feedback_requires_planned_budget_v7"):
        options_for(policy, repair_feedback=True)


def test_feedback_identity_changes_only_explicit_runs_and_plans(case, monkeypatch):
    """Both plan and run identity bind the actual versioned feedback projection."""
    import litagent.writing_flat as flat

    common = {"methodology": {}, "budget_policy": "planned-budget-v7"}
    old = writer_budget_plan(case.packet, case.aliases, **common)
    current = writer_budget_plan(
        case.packet, case.aliases, **common, repair_feedback=True
    )
    assert "repair_feedback_contract" not in old["envelope"]
    assert current["envelope"]["repair_feedback_contract"] == "host-repair-feedback-v1"
    assert old["identity"] != current["identity"]
    assert current["repair_limit"] == 2
    old_options = options_for()
    current_options = options_for(repair_feedback=True)
    before = current_options.fingerprint()
    assert before != old_options.fingerprint()
    original_identity = flat.repair_feedback_identity()
    monkeypatch.setattr(flat, "repair_feedback_identity", lambda: "f" * 64)
    assert original_identity != flat.repair_feedback_identity()
    assert current_options.fingerprint() != before
    assert writer_budget_plan(case.packet, case.aliases, **common) == old
    assert (
        writer_budget_plan(case.packet, case.aliases, **common, repair_feedback=True)[
            "identity"
        ]
        != current["identity"]
    )


def test_new_feedback_does_not_become_a_public_request_parameter():
    from litagent.api import SurveyRequest

    assert "repair_feedback" not in SurveyRequest.model_fields
    assert "repair_feedback" not in SurveyRequest.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SurveyRequest(query="Explain the methods.", repair_feedback=True)


@pytest.mark.parametrize(
    "field", ["REPAIR_FEEDBACK_AGGREGATE_RULE", "REPAIR_FEEDBACK_INSTRUCTION"]
)
def test_actual_feedback_rule_revision_invalidates_only_selected_policy(
    case, monkeypatch, field
):
    """A changed rule or prompt must invalidate the new run and budget together."""
    import litagent.writing_flat as flat

    common = {"methodology": {}, "budget_policy": "planned-budget-v7"}
    old = writer_budget_plan(case.packet, case.aliases, **common)
    selected = writer_budget_plan(
        case.packet, case.aliases, **common, repair_feedback=True
    )
    old_run = options_for().fingerprint()
    selected_run = options_for(repair_feedback=True).fingerprint()
    monkeypatch.setattr(flat, field, getattr(flat, field) + " revised")
    assert options_for().fingerprint() == old_run
    assert options_for(repair_feedback=True).fingerprint() != selected_run
    assert writer_budget_plan(case.packet, case.aliases, **common) == old
    assert (
        writer_budget_plan(case.packet, case.aliases, **common, repair_feedback=True)[
            "identity"
        ]
        != selected["identity"]
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_forwards_only_an_explicit_internal_flag(enabled):
    graph = TaskGraph()
    graph.add_task(SubTask("write", "Write a report", "synthesis"))
    LitAgent._apply_run_options_to_graph(graph, options_for(repair_feedback=enabled))
    value = graph.tasks["write"].input_data
    assert ("repair_feedback" in value) is enabled
    if enabled:
        assert value["repair_feedback"] is True


def test_envelope_marker_requires_the_adaptive_round_contract():
    kwargs = dict(
        body_units=66,
        assertions=128,
        body_bytes=32000,
        source_identity="a" * 64,
        builder_identity="b" * 64,
        contract="flat-writer-envelope-v1",
    )
    old = make_writer_envelope(**kwargs, capacity=CapacityConfig(), round_budget=True)
    assert "repair_feedback_contract" not in old
    current = make_writer_envelope(
        **kwargs,
        capacity=CapacityConfig(),
        round_budget=True,
        repair_feedback=True,
    )
    assert current["identity"] != old["identity"]
    assert validate_writer_envelope(current) == current
    for changes in ({"capacity": None}, {"round_budget": False}):
        with pytest.raises(ValueError, match="repair_feedback_policy_invalid"):
            make_writer_envelope(
                **kwargs,
                **{
                    "capacity": CapacityConfig(),
                    "round_budget": True,
                    "repair_feedback": True,
                    **changes,
                },
            )
    for marker in [None, False, [], {}, "host-repair-feedback-v2"]:
        invalid = copy.deepcopy(current)
        invalid["repair_feedback_contract"] = marker
        invalid["identity"] = _digest(
            {key: value for key, value in invalid.items() if key != "identity"}
        )
        with pytest.raises(ValueError, match="repair_feedback_invalid"):
            validate_writer_envelope(invalid)


@pytest.mark.parametrize("wrong", [None, 0, 1, "true", [], {}])
def test_frozen_budget_rejects_untyped_feedback_before_reading_material(wrong):
    with pytest.raises(ValueError, match="repair_feedback_policy_invalid"):
        plan_frozen_budget({}, budget_policy="planned-budget-v7", repair_feedback=wrong)


def test_manifest_requires_explicit_true_and_round_profile():
    old = {"writing_profile": campaign.ROUND_PROFILE}
    assert campaign.repair_feedback_options(old) == {}
    assert "repair_feedback" not in campaign.manifest_options(old)
    new = {**old, "repair_feedback": True}
    assert campaign.repair_feedback_options(new) == {"repair_feedback": True}
    assert campaign.manifest_options(new)["repair_feedback"] is True
    for value in [False, None, 1, "true", {}, []]:
        with pytest.raises(ValueError, match="repair_feedback_configuration_invalid"):
            campaign.repair_feedback_options({**old, "repair_feedback": value})
    with pytest.raises(ValueError, match="repair_feedback_configuration_invalid"):
        campaign.repair_feedback_options(
            {"writing_profile": campaign.FLAT_PROFILE, "repair_feedback": True}
        )


def test_cli_rejects_live_feedback_override_before_loading_config(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["frozen", "--root", str(tmp_path), "--live", "--repair-feedback"],
    )
    monkeypatch.setattr(
        campaign,
        "load_config",
        lambda: pytest.fail("Live override reached configuration or transport"),
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_prepare_only_forwards_the_new_flag_when_explicit(
    monkeypatch, tmp_path, enabled
):
    calls = []
    monkeypatch.setattr(campaign, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(campaign, "profile_config", lambda config, profile: config)
    monkeypatch.setattr(
        campaign,
        "prepare",
        lambda *args, **kwargs: calls.append(kwargs) or {"status": "offline-fixture"},
    )
    argv = ["frozen", "--root", str(tmp_path), "--prepare"]
    if enabled:
        argv.extend(["--writing-profile", campaign.ROUND_PROFILE, "--repair-feedback"])
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    assert len(calls) == 1
    assert ("repair_feedback" in calls[0]) is enabled
    if enabled:
        assert calls[0]["repair_feedback"] is True


def test_old_and_new_feedback_checkpoint_plans_cannot_cross_restore(case):
    common = {"methodology": {}, "budget_policy": "planned-budget-v7"}
    old = writer_budget_plan(case.packet, case.aliases, **common)
    current = writer_budget_plan(
        case.packet, case.aliases, **common, repair_feedback=True
    )
    for plan, conflicting in [(old, current), (current, old)]:
        owner = PlannedBudget(plan)
        snapshot = owner.snapshot()
        with pytest.raises(ValueError, match="identity_conflict"):
            owner.restore(PlannedBudget(conflicting).snapshot())
        assert owner.snapshot() == snapshot


def test_feedback_plan_keeps_unknown_usage_held_and_forbids_replay(case):
    """The new envelope identity cannot refund or resend an uncertain request."""
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        repair_feedback=True,
    )
    owner = PlannedBudget(plan)
    amount = wire_reservation([{"role": "user", "content": "Offline request."}], 20000)
    with request_scope("repair", "unknown-repair"):
        owner.reserve(amount)
        owner.bind_wire({"offline_request": "unique"})
        owner.begin_request()
        owner.charge_unknown_usage(amount)
        owner.release(amount)
    snapshot = owner.snapshot()
    entry = snapshot["entries"]["unknown-repair"]
    assert entry["status"] == "interrupted_no_replay"
    assert entry["reserved_tokens"] == amount and entry["actual_tokens"] is None
    assert snapshot["requests_sent"] == 1 and snapshot["used_tokens"] == 0
    restored = CostBudget.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    with request_scope("repair", "unknown-repair"):
        with pytest.raises(ValueError, match="no_replay"):
            restored.begin_request()
    assert restored.snapshot() == snapshot
