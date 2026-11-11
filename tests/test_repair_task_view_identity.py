"""Bind the finite repair task view without changing older wire selections."""

import copy
import json
from types import SimpleNamespace

import pytest
from model_config_helpers import load_previous_default_config as load_config
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_structured_repair_identity import SELECTION, guidance_envelope, rehash
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.safety.planned_budget import CONTEXT_LIMIT, validate_writer_envelope
from litagent.writing_v2 import RANGE_WRITING_PLAN, prepare_planned_input
from scripts import run_frozen_writing as campaign

REVISION = "host-repair-task-view-v1"
RULE = "complete-locked-request-reservation-v1"
FIELDS = {
    "repair_guidance_wire_revision",
    "repair_guidance_wire_identity",
    "repair_guidance_capacity_rule",
}
FLAGS = {**SELECTION, "repair_guidance_contract": "host-structured-repair-v1"}


def scope(revision=REVISION):
    """Use the real complete parent and the actual selected envelope factory."""
    return guidance_envelope(repair_guidance_wire_revision=revision)


def test_task_view_uses_one_existing_selector_and_explicit_aggregate_rule():
    parent, legacy, new = (
        scope(None),
        scope(flat.REPAIR_GUIDANCE_WIRE_REVISION),
        scope(),
    )
    assert not FIELDS.intersection(parent)
    assert "repair_guidance_capacity_rule" not in legacy
    assert new["repair_guidance_capacity_rule"] == RULE
    assert new["repair_guidance_wire_identity"] == flat.repair_guidance_wire_identity(
        REVISION
    )
    assert (
        legacy["repair_guidance_wire_identity"] == flat.repair_guidance_wire_identity()
    )
    assert new["repair_guidance_json_bytes"] == new["diagnostic_json_bytes"] == 12288
    assert new["request_caps"]["repair"] == CONTEXT_LIMIT == 117964
    assert {k: v for k, v in new.items() if k not in FIELDS | {"identity"}} == {
        k: v for k, v in parent.items() if k != "identity"
    }


@pytest.mark.parametrize("field", sorted(FIELDS))
def test_task_view_rejects_missing_pair_or_rule_after_rehash(field):
    value = scope()
    value.pop(field)
    rehash(value)
    with pytest.raises(
        ValueError, match="writer_envelope_repair_guidance_wire_invalid"
    ):
        validate_writer_envelope(value)


@pytest.mark.parametrize("bad", [None, False, True, 0, 1, "", "future", [], {}])
def test_task_view_rejects_invalid_admission_rule_after_rehash(bad):
    value = scope()
    value["repair_guidance_capacity_rule"] = bad
    rehash(value)
    with pytest.raises(
        ValueError, match="writer_envelope_repair_guidance_wire_invalid"
    ):
        validate_writer_envelope(value)


@pytest.mark.parametrize("revision", [None, "host-unit-ref-guidance-v1"])
def test_aggregate_rule_never_silently_promotes_older_envelopes(revision):
    value = scope(revision)
    value["repair_guidance_capacity_rule"] = RULE
    rehash(value)
    with pytest.raises(
        ValueError, match="writer_envelope_repair_guidance_wire_invalid"
    ):
        validate_writer_envelope(value)


@pytest.mark.parametrize("bad", [False, True, 0, 1, "", "future", [], {}])
def test_unknown_selector_rejects_before_frozen_input_is_read(bad):
    with pytest.raises(ValueError, match="repair_guidance_wire_policy_invalid"):
        plan_frozen_budget(
            {},
            budget_policy="planned-budget-v7",
            repair_guidance_wire_revision=bad,
            **FLAGS,
        )


def test_actual_frozen_plan_selects_new_identity_without_larger_request_caps(
    frozen_bundle,
):
    plans = [
        plan_frozen_budget(
            frozen_bundle,
            budget_policy="planned-budget-v7",
            repair_guidance_wire_revision=revision,
            **FLAGS,
        )
        for revision in (None, "host-unit-ref-guidance-v1", REVISION)
    ]
    assert len({p["identity"] for p in plans}) == 3
    assert len({p["envelope"]["builder_identity"] for p in plans}) == 3
    for plan in plans:
        assert plan["status"] == "admitted"
        assert plan["maximum_tokens"] == 4500000 and plan["maximum_requests"] == 64
        assert plan["scope_bounds"]["shared_repairs"] == 2
        assert plan["stages"]["repair"]["request_cap"] == 117964
    assert plans[0]["envelope"]["request_caps"] == plans[2]["envelope"]["request_caps"]


def test_legacy_factory_still_calls_zero_argument_identity(monkeypatch):
    expected = scope(flat.REPAIR_GUIDANCE_WIRE_REVISION)
    monkeypatch.setattr(flat, "repair_guidance_wire_identity", lambda: "a" * 64)
    changed = scope(flat.REPAIR_GUIDANCE_WIRE_REVISION)
    assert changed["repair_guidance_wire_identity"] == "a" * 64
    assert changed["source_map_identity"] == expected["source_map_identity"]
    assert changed["repair_guidance_identity"] == expected["repair_guidance_identity"]


@pytest.mark.asyncio
async def test_task_view_never_starts_without_a_frozen_plan():
    with pytest.raises(ValueError, match="repair_guidance_wire_frozen_plan_required"):
        await run_frozen(
            SimpleNamespace(_wired=False),
            {},
            "unstarted-task-view",
            repair_guidance_wire_revision=REVISION,
        )


def actual_repair_start(case, envelope):
    """Build the actual planned wire augmentation without sending a request."""
    _, payload = prepare_planned_input(
        "Repair only the authorized source-bound units.",
        {
            "current_body": case.body,
            "cause": "capacity",
            "repair_input_contract": "flat-repair-input-v1",
        },
        case.packet,
        case.aliases,
        RANGE_WRITING_PLAN,
        envelope,
    )
    return {
        "messages": [
            {"role": "system", "content": ""},
            {"role": "user", "content": json.dumps(payload)},
        ]
    }


def test_historical_task_view_uses_pinned_rule_without_calling_current_codec(
    case, monkeypatch
):
    envelope = scope()
    start = actual_repair_start(case, envelope)
    payload = json.loads(start["messages"][1]["content"])
    assert "repair_guidance_json_bytes" not in payload["writing_envelope"]
    assert payload["writing_envelope"]["repair_guidance_capacity_rule"] == RULE
    monkeypatch.setattr(
        flat,
        "repair_guidance_wire_identity",
        lambda *a, **k: pytest.fail("historical cost invoked current runtime identity"),
    )
    campaign._historical_writer_envelope(start, envelope)


@pytest.mark.parametrize("mutation", ["rule", "legacy_cap", "request_cap", "output"])
def test_historical_task_view_cannot_hide_a_changed_admission_rule(case, mutation):
    envelope = scope()
    start = actual_repair_start(case, envelope)
    payload = json.loads(start["messages"][1]["content"])
    if mutation == "rule":
        payload["writing_envelope"]["repair_guidance_capacity_rule"] = False
    elif mutation == "legacy_cap":
        payload["writing_envelope"]["repair_guidance_json_bytes"] = 12288
    elif mutation == "request_cap":
        payload["repair_feedback_limits"]["maximum_request_reservation"] += 1
    else:
        payload["repair_feedback_limits"]["reserved_output_tokens"] = 20000.0
    start["messages"][1]["content"] = json.dumps(payload)
    with pytest.raises(ValueError, match="frozen_accounting_mixed_"):
        campaign._historical_writer_envelope(start, envelope)


@pytest.mark.parametrize("revision", [None, "host-unit-ref-guidance-v1", REVISION])
def test_actual_prepare_validate_keeps_choice_and_rule_on_the_manifest(
    case, tmp_path, monkeypatch, revision
):
    """Replace file/Git inputs only; use actual campaign and frozen consumers."""
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
    source.write_text('"""Portable task-view preparation fixture."""\n')
    monkeypatch.setattr(
        campaign, "source_paths", lambda: [source.relative_to(tmp_path)]
    )
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: "a" * 40)
    config = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    root = tmp_path / "prepared"
    prepared = campaign.prepare(
        root,
        config,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        repair_guidance_wire_revision=revision,
        **FLAGS,
    )
    manifest = campaign.validate(root, config)
    assert manifest["identity"] == prepared["identity"]
    assert not FIELDS.intersection(manifest["options"])
    assert ("repair_guidance_capacity_rule" in manifest) is (revision == REVISION)
    if revision == REVISION:
        assert manifest["repair_guidance_capacity_rule"] == RULE
        assert campaign.repair_guidance_wire_options(manifest) == {
            "repair_guidance_wire_revision": REVISION
        }
        changed = copy.deepcopy(manifest)
        changed.pop("repair_guidance_capacity_rule")
        with pytest.raises(
            ValueError, match="frozen_repair_guidance_wire_configuration_invalid"
        ):
            campaign.repair_guidance_wire_options(changed)
    for case_row in manifest["cases"]:
        plan = manifest["budget_plans"][case_row["case_id"]]
        assert plan == plan_frozen_budget(
            case_row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=config.capacity,
            repair_guidance_wire_revision=revision,
            **FLAGS,
        )
    assert not any(
        (root / n).exists()
        for n in ("state.json", "execution.lock", "launch-intent.json")
    )
