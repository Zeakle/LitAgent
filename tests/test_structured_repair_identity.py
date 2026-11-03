"""Keep structured repair inputs opt-in, pinned and separate from fee authority."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_reference_set_identity import envelope as reference_envelope
from test_support_receipt_identity import FLAGS, selected_envelope
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.config import load_config
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.quality import digest
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import (
    PlannedBudget,
    _digest,
    envelope_budget_plan,
    validate_writer_envelope,
)
from litagent.writing_flat import (
    STRUCTURED_REPAIR_GUIDANCE_CONTRACT,
    structured_repair_guidance_identity,
)
from litagent.writing_v2 import RANGE_WRITING_PLAN
from scripts import run_frozen_writing as campaign

CONTRACT = STRUCTURED_REPAIR_GUIDANCE_CONTRACT
FIELDS = {
    "repair_guidance_contract",
    "repair_guidance_identity",
    "repair_guidance_json_bytes",
}
SELECTION = {
    **FLAGS,
    "support_validation_revision": "host-support-receipts-v1",
    "repair_mapping_contract": "capacity-source-map-v3",
}


def guidance_envelope(**changes):
    """Use the actual factory on the complete opt-in source-map parent."""
    return reference_envelope(
        **{
            **SELECTION,
            "global_findings_contract": "anchored-global-findings-v2",
            "summary_constraints": True,
            "repair_mapping_contract": "capacity-source-map-v3",
            "repair_guidance_contract": CONTRACT,
            **changes,
        }
    )


def rehash(value):
    """Model an attacker recomputing a digest without repairing field semantics."""
    value["identity"] = _digest({k: v for k, v in value.items() if k != "identity"})
    return value


def budget_plan(scope):
    """Build the complete actual two-repair reservation without sending a request."""
    return envelope_budget_plan(
        scope,
        identity="portable-frozen-input",
        first_request={
            "messages": [{"role": "user", "content": "Frozen packet."}],
            "response_format": {"type": "json_object"},
        },
        repairs=2,
    )


def test_none_omits_fields_and_does_not_promote_existing_v3():
    old = selected_envelope(repair_mapping_contract="capacity-source-map-v3")
    assert guidance_envelope(repair_guidance_contract=None) == old
    assert not FIELDS.intersection(old)
    new = guidance_envelope()
    assert new["repair_guidance_contract"] == CONTRACT
    assert new["repair_guidance_identity"] == structured_repair_guidance_identity()
    assert new["repair_guidance_json_bytes"] == new["diagnostic_json_bytes"] == 12288
    assert new["source_map_identity"] == old["source_map_identity"]
    assert new["identity"] != old["identity"]
    assert validate_writer_envelope(new) == new


def test_guidance_capacity_uses_the_effective_parent_not_a_new_request_cap():
    scope = guidance_envelope(capacity=CapacityConfig(diagnostic_json_bytes=4096))
    assert scope["repair_guidance_json_bytes"] == scope["diagnostic_json_bytes"] == 4096
    assert scope["request_caps"]["repair"] == 117964
    assert accounting_envelope(scope).diagnostic_json_bytes == 4096


@pytest.mark.parametrize("bad", [False, True, 0, 1, "", "future", [], {}])
def test_selector_is_not_coerced(bad):
    with pytest.raises(ValueError, match="repair_guidance_policy_invalid"):
        guidance_envelope(repair_guidance_contract=bad)


@pytest.mark.parametrize(
    "change",
    [
        {"contract": "budgeted-writer-envelope-v1"},
        {"round_budget": False},
        {"round_budget": 1},
        {"capacity": None},
        {"repair_mapping_contract": "capacity-edit-map-v2"},
        {"global_findings_contract": None},
        {"summary_constraints": False},
        {"inventory_recovery": False},
        {"mixed_inventory_recovery": False},
        {"classification_review": False},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"transport_revision": None},
        {"support_validation_revision": None},
    ],
)
def test_factory_requires_every_parent(change):
    with pytest.raises(ValueError, match="repair_guidance_policy_invalid"):
        guidance_envelope(**change)


@pytest.mark.parametrize("missing", sorted(FIELDS))
def test_partial_marker_groups_fail_even_after_rehash(missing):
    scope = guidance_envelope()
    scope.pop(missing)
    rehash(scope)
    before = copy.deepcopy(scope)
    with pytest.raises(ValueError, match="writer_envelope_repair_guidance_invalid"):
        validate_writer_envelope(scope)
    with pytest.raises(ValueError, match="envelope_repair_guidance_invalid"):
        accounting_envelope(scope)
    assert scope == before


@pytest.mark.parametrize(
    "change",
    [
        {"repair_guidance_contract": None},
        {"repair_guidance_contract": "future"},
        {"repair_guidance_identity": None},
        {"repair_guidance_identity": "not-a-hash"},
        {"repair_guidance_identity": "F" * 64},
        {"repair_guidance_json_bytes": None},
        {"repair_guidance_json_bytes": True},
        {"repair_guidance_json_bytes": 12288.0},
        {"repair_guidance_json_bytes": 0},
        {"repair_guidance_json_bytes": 12287},
        {"repair_guidance_json_bytes": 12289},
    ],
)
def test_malformed_pinned_guidance_is_not_accepted_as_fee_data(change):
    value = rehash({**guidance_envelope(), **change})
    with pytest.raises(ValueError):
        validate_writer_envelope(value)
    with pytest.raises(ValueError):
        accounting_envelope(value)


@pytest.mark.parametrize(
    "field",
    [
        "capacity",
        "repair_mapping_contract",
        "global_findings_contract",
        "summary_constraints_contract",
        "inventory_recovery_contract",
        "classification_review_contract",
        "repair_feedback_contract",
        "reference_contract",
        "transport_revision",
        "support_validation_revision",
    ],
)
def test_rehashed_guidance_cannot_strip_its_parent(field):
    value = guidance_envelope()
    value.pop(field)
    rehash(value)
    with pytest.raises(ValueError):
        validate_writer_envelope(value)
    with pytest.raises(ValueError):
        accounting_envelope(value)


def test_semantic_drift_rejects_runtime_but_preserves_pinned_costs(monkeypatch):
    import litagent.writing_flat as flat

    old = guidance_envelope(repair_guidance_contract=None)
    selected = guidance_envelope()
    plan = budget_plan(selected)
    assert plan["status"] == "admitted"
    owner = PlannedBudget(plan)
    snapshot = owner.snapshot()
    globals_before = (
        flat.source_map_wire_identity(),
        flat.transport_identity(),
        flat.repair_feedback_identity(),
    )

    def changed():
        return "f" * 64

    monkeypatch.setattr(flat, "structured_repair_guidance_identity", changed)
    assert guidance_envelope(repair_guidance_contract=None) == old
    assert globals_before == (
        flat.source_map_wire_identity(),
        flat.transport_identity(),
        flat.repair_feedback_identity(),
    )
    assert guidance_envelope()["identity"] != selected["identity"]
    with pytest.raises(ValueError, match="writer_envelope_repair_guidance_invalid"):
        validate_writer_envelope(selected)
    with pytest.raises(ValueError, match="writer_envelope_repair_guidance_invalid"):
        PlannedBudget(plan)
    before = copy.deepcopy(snapshot)
    restored = validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert restored == before == snapshot
    assert restored is not snapshot


@pytest.mark.parametrize("selected", [False, True])
def test_full_plan_keeps_the_same_two_http_and_scientific_reservations(selected):
    scope = guidance_envelope(repair_guidance_contract=CONTRACT if selected else None)
    plan = budget_plan(scope)
    assert plan["contract"] == "planned-budget-v7"
    assert plan["scope_bounds"]["shared_repairs"] == 2
    assert plan["stages"]["repair"]["count"] == 2
    assert plan["stages"]["assessment"]["count"] == 24
    assert plan["stages"]["assessment"]["output_tokens"] == 16384
    assert plan["stages"]["repair"]["output_tokens"] == 20000
    assert plan["maximum_tokens"] == 4500000
    assert plan["maximum_requests"] == 64
    snapshot = PlannedBudget(plan).snapshot()
    assert validate_accounting_snapshot(snapshot, expected_plan=plan) == snapshot


def test_guidance_is_not_a_public_run_or_survey_parameter():
    from litagent.api import SurveyRequest
    from litagent.workbench import RunOptions

    assert "repair_guidance_contract" not in SurveyRequest.model_fields
    assert "repair_guidance_contract" not in RunOptions.model_fields
    with pytest.raises(ValidationError):
        SurveyRequest(query="Explain mechanisms.", repair_guidance_contract=CONTRACT)


@pytest.mark.parametrize("selected", [False, True])
def test_frozen_plan_uses_the_explicit_selection_without_mutating_input(
    frozen_bundle, selected
):
    before = copy.deepcopy(frozen_bundle)
    selector = {"repair_guidance_contract": CONTRACT} if selected else {}
    old = plan_frozen_budget(
        frozen_bundle, budget_policy="planned-budget-v7", **SELECTION
    )
    actual = plan_frozen_budget(
        frozen_bundle, budget_policy="planned-budget-v7", **SELECTION, **selector
    )
    assert ("repair_guidance_contract" in actual["envelope"]) is selected
    assert actual["input_identity"] == old["input_identity"]
    assert (
        actual["envelope"]["source_map_identity"]
        == old["envelope"]["source_map_identity"]
    )
    assert actual["stages"].keys() == old["stages"].keys()
    assert frozen_bundle == before


@pytest.mark.asyncio
async def test_new_frozen_runtime_requires_a_plan_before_touching_the_client(
    frozen_bundle,
):
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="repair_guidance_frozen_plan_required"):
        await run_frozen(
            agent,
            frozen_bundle,
            "no-model-structured-repair",
            budget_policy="planned-budget-v7",
            repair_guidance_contract=CONTRACT,
            **SELECTION,
        )
    assert agent.__dict__ == {"_wired": False}


@pytest.mark.parametrize("field", sorted(FIELDS))
@pytest.mark.parametrize("bad", [None, False, 12288.0, "wrong"])
def test_historical_wire_guidance_fields_match_pinned_json_types(field, bad):
    scope = guidance_envelope()
    payload = {
        "current_body": {},
        "repair_input_contract": "flat-repair-input-v1",
        "cause": "evaluation",
        "writing_envelope": {
            key: scope[key]
            for key in (
                "contract",
                "inventory_recovery_contract",
                "reference_contract",
                "transport_revision",
                "support_validation_revision",
                *sorted(FIELDS),
            )
        },
    }

    def start(value):
        return {
            "messages": [
                {"role": "system", "content": "fixed"},
                {"role": "user", "content": json.dumps(value)},
            ]
        }

    campaign._historical_writer_envelope(start(payload), scope)
    payload["writing_envelope"][field] = bad
    with pytest.raises(ValueError, match="writer_repair_guidance_envelope_invalid"):
        campaign._historical_writer_envelope(start(payload), scope)


def test_nonrepair_wire_has_no_guidance_fields_even_for_selected_run():
    scope = guidance_envelope()
    payload = {
        "writing_envelope": {
            key: scope[key]
            for key in (
                "contract",
                "inventory_recovery_contract",
                "reference_contract",
                "transport_revision",
                "support_validation_revision",
            )
        }
    }
    start = lambda: {
        "messages": [{"content": "fixed"}, {"content": json.dumps(payload)}]
    }
    campaign._historical_writer_envelope(start(), scope)
    payload["writing_envelope"]["repair_guidance_contract"] = CONTRACT
    with pytest.raises(ValueError, match="writer_repair_guidance_envelope_invalid"):
        campaign._historical_writer_envelope(start(), scope)


def test_real_prepare_read_bundle_validate_and_strip_resistance(
    case, tmp_path, monkeypatch
):
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
    historical = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    historical.parent.mkdir(parents=True)
    historical.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = tmp_path / "portable_source.py"
    source.write_text('"""Portable frozen source fixture."""\n')
    monkeypatch.setattr(
        campaign, "source_paths", lambda: [source.relative_to(tmp_path)]
    )
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    cfg = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    root = tmp_path / "prepared"
    prepared = campaign.prepare(
        root,
        cfg,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        repair_guidance_contract=CONTRACT,
        **SELECTION,
    )
    manifest = campaign.validate(root, cfg)
    assert prepared["identity"] == manifest["identity"]
    assert (
        hashlib.sha256(source.read_bytes()).hexdigest()
        == manifest["source_hashes"][str(source.relative_to(tmp_path))]
    )
    assert manifest["repair_guidance_contract"] == CONTRACT
    assert manifest["writing_parameters"]["repair_guidance_contract"] == CONTRACT
    assert "repair_guidance_contract" not in manifest["options"]
    for row in manifest["cases"]:
        assert manifest["budget_plans"][row["case_id"]] == plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=cfg.capacity,
            repair_guidance_contract=CONTRACT,
            **SELECTION,
        )
    assert not (root / "state.json").exists()
    assert not (root / ".live.lock").exists()
    manifest.pop("repair_guidance_contract")
    manifest["writing_parameters"].pop("repair_guidance_contract")
    manifest["identity"] = campaign.digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    (root / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        campaign.validate(root, cfg)


def test_prepare_rejects_bad_parent_before_any_artifact(tmp_path):
    root = tmp_path / "must-not-exist"
    cfg = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    with pytest.raises(
        ValueError, match="frozen_repair_guidance_configuration_invalid"
    ):
        campaign.prepare(
            root,
            cfg,
            writing_profile=campaign.ROUND_PROFILE,
            writing_plan=RANGE_WRITING_PLAN,
            repair_guidance_contract=CONTRACT,
        )
    assert not root.exists()


def test_live_cli_cannot_override_frozen_guidance_before_config(monkeypatch, tmp_path):
    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_writing",
            "--live",
            "--root",
            str(tmp_path),
            "--repair-guidance-contract",
            CONTRACT,
        ],
    )
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("configuration touched")
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2
