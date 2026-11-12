"""Bind optional source lineage to trusted frozen plans without upgrading history."""

import copy
from types import SimpleNamespace

import pytest
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_support_receipt_identity import FLAGS, selected_envelope
from test_writer_v2 import case  # noqa: F401

from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.orchestrator.task_graph import SubTask
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_budget import _digest, validate_writer_envelope
from litagent.writing_v2 import WriterError, synthesize
from scripts import run_frozen_writing as campaign

CONTRACT = "capacity-source-map-v3"
SELECTION = {**FLAGS, "support_validation_revision": "host-support-receipts-v1"}


def selected_manifest(**changes):
    """Keep the selection outside public run options."""
    return {
        "writing_profile": campaign.ROUND_PROFILE,
        **SELECTION,
        "repair_mapping_contract": CONTRACT,
        **changes,
    }


def test_none_and_explicit_old_map_preserve_original_envelope_bytes():
    old = selected_envelope()
    assert selected_envelope(repair_mapping_contract=None) == old
    assert selected_envelope(repair_mapping_contract="capacity-edit-map-v2") == old
    assert not {"source_map_identity", "source_map_json_bytes"} & old.keys()


def test_source_identity_changes_only_selected_plan_and_rejects_current_drift(
    monkeypatch,
):
    import litagent.writing_flat as flat

    old = selected_envelope()
    new = selected_envelope(repair_mapping_contract=CONTRACT)
    assert new["source_map_json_bytes"] == new["diagnostic_json_bytes"] == 12288
    assert new["identity"] != old["identity"]
    assert new["body_max_units"] == old["body_max_units"] == 66
    assert new["model_assertions"] == old["model_assertions"] == 128
    assert validate_writer_envelope(new) == new
    opaque = copy.deepcopy(new)
    monkeypatch.setattr(flat, "source_map_wire_identity", lambda: "f" * 64)
    assert selected_envelope() == old
    with pytest.raises(ValueError, match="writer_envelope_source_map_invalid"):
        validate_writer_envelope(new)
    # A pure fee reader authenticates pinned data, not today's executable semantics.
    assert accounting_envelope(opaque).max_body_units == 66
    assert opaque == new


@pytest.mark.parametrize(
    "changes",
    [
        {"source_map_identity": None},
        {"source_map_json_bytes": True},
        {"source_map_json_bytes": 12289},
        {"source_map_json_bytes": 12287},
        {"support_validation_revision": None},
        {"repair_mapping_contract": "capacity-edit-map-v2"},
    ],
)
def test_rehashed_source_envelope_cannot_hide_missing_ownership_or_limits(changes):
    value = selected_envelope(repair_mapping_contract=CONTRACT)
    value.update(changes)
    value["identity"] = _digest({k: v for k, v in value.items() if k != "identity"})
    before = copy.deepcopy(value)
    with pytest.raises(ValueError):
        validate_writer_envelope(value)
    with pytest.raises(ValueError):
        accounting_envelope(value)
    assert value == before


@pytest.mark.parametrize("value", [False, True, 0, 1, "", "unknown", [], {}])
def test_unknown_selector_is_not_coerced(value):
    with pytest.raises(ValueError, match="repair_mapping_policy_invalid"):
        selected_envelope(repair_mapping_contract=value)


def test_frozen_budget_recomputes_explicit_map_without_mutating_input(frozen_bundle):
    before = copy.deepcopy(frozen_bundle)
    old = plan_frozen_budget(
        frozen_bundle, budget_policy="planned-budget-v7", **SELECTION
    )
    assert (
        plan_frozen_budget(
            frozen_bundle,
            budget_policy="planned-budget-v7",
            repair_mapping_contract=None,
            **SELECTION,
        )
        == old
    )
    new = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        repair_mapping_contract=CONTRACT,
        **SELECTION,
    )
    assert new["status"] == old["status"] == "admitted"
    assert new["identity"] != old["identity"]
    assert new["envelope"]["source_identity"] == old["envelope"]["source_identity"]
    assert new["envelope"]["repair_mapping_contract"] == CONTRACT
    assert frozen_bundle == before


@pytest.mark.asyncio
async def test_frozen_runtime_requires_prepared_plan_before_wiring_any_client(
    frozen_bundle,
):
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="repair_mapping_frozen_plan_required"):
        await run_frozen(
            agent,
            frozen_bundle,
            "no-send-source-map",
            budget_policy="planned-budget-v7",
            repair_mapping_contract=CONTRACT,
            **SELECTION,
        )
    assert agent.__dict__ == {"_wired": False}


@pytest.mark.asyncio
async def test_planner_input_cannot_enable_source_map_without_trusted_provider():
    task = SubTask(
        "synthesis",
        "Untrusted planner selection",
        "synthesis",
        input_data={
            "generation_strategy": "body_summary_v3",
            "budget_policy": "planned-budget-v7",
            "repair_mapping_contract": CONTRACT,
        },
    )
    with pytest.raises(WriterError, match="repair_mapping_frozen_plan_required"):
        await synthesize(SimpleNamespace(), task, [], None, {}, {})


def test_source_selector_is_not_public_and_does_not_change_public_options():
    from litagent.api import SurveyRequest
    from litagent.runner import RunOptions

    assert "repair_mapping_contract" not in SurveyRequest.model_fields
    assert "repair_mapping_contract" not in RunOptions.model_fields
    old = selected_manifest()
    old.pop("repair_mapping_contract")
    assert campaign.manifest_options(old) == campaign.manifest_options(
        selected_manifest()
    )
    assert campaign.repair_mapping_options(old) == {}
    assert campaign.repair_mapping_options(selected_manifest()) == {
        "repair_mapping_contract": CONTRACT
    }
    with pytest.raises(
        ValueError, match="frozen_support_validation_configuration_invalid"
    ):
        campaign.repair_mapping_options(
            selected_manifest(support_validation_revision=None)
        )


def test_source_map_keeps_shared_content_guidance_in_real_chapter_request(case):
    from litagent.capacity_repair import CONTENT_GUIDANCE
    from litagent.writing_v2 import (
        BODY,
        RANGE_WRITING_PLAN,
        chapter_request_input,
        prepare_planned_input,
    )

    envelope = selected_envelope(repair_mapping_contract=CONTRACT)
    system, payload, _ = chapter_request_input(
        case.packet, case.aliases, BODY[0], {}, {}, RANGE_WRITING_PLAN, envelope
    )
    system, _ = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, envelope
    )
    assert system.count(CONTENT_GUIDANCE) == 1


def test_runtime_recomputes_stored_source_binding_hash_for_json_type_aliases(case):
    import json

    from litagent.capacity_repair import capacity_targets, original_unit_directory
    from litagent.writing_flat import encode, parse
    from litagent.writing_v2 import (
        check_source_map_response,
        repair_request_input,
        source_map_response_fields,
    )

    envelope = selected_envelope(repair_mapping_contract=CONTRACT)
    targets = capacity_targets(case.body, {})
    tid = next(iter(targets))
    sources = original_unit_directory(targets)[tid]
    canonical = {
        "outcome": "patched",
        "explanation": "Source-bound replacement without any semantic verdict.",
        "patches": [
            {"target_id": tid, "sentences": copy.deepcopy(targets[tid]["original"])}
        ],
        "source_relations": [
            {"target_id": tid, "replacement_index": index, "source_unit_ids": [uid]}
            for index, uid in enumerate(sources)
        ],
        "deleted_sources": [],
    }
    _, _, schema = repair_request_input(
        case.packet,
        case.body,
        targets,
        case.aliases,
        cause="capacity",
        previous_failure=None,
        envelope=envelope,
        wire_contract="flat-writer-v1",
    )
    parsed = parse(
        json.dumps(encode(canonical, schema)),
        schema,
        reference_contract="ordered-reference-set-v1",
        transport_revision="host-lossless-transport-v1",
    )
    record = {
        **source_map_response_fields(envelope, parsed),
        "source_relation_bindings": copy.deepcopy(parsed["source_relation_bindings"]),
    }
    check_source_map_response(envelope, record, parsed)
    damaged = copy.deepcopy(record)
    damaged["source_relation_bindings"]["relations"][0]["canonical_path"][1] = False
    assert damaged["source_relation_bindings"] == record["source_relation_bindings"]
    assert (
        damaged["source_relation_bindings_sha256"]
        == record["source_relation_bindings_sha256"]
    )
    with pytest.raises(WriterError, match="writer_source_map_receipt_invalid"):
        check_source_map_response(envelope, damaged, parsed)
