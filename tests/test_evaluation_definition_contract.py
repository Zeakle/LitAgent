"""Align explicit evaluation definitions without changing old verdict contracts."""

import copy
import hashlib
import json

import pytest
from test_evaluation_wire_projection import canonical  # noqa: F401

import litagent.eval.scoped as scoped
from litagent.body_quality import (
    APPLICABILITY_CONTRACT,
    APPLICABILITY_INSTRUCTION,
    LEGACY_SUMMARY_SELECTION,
    WRITER_CONTENT_CONTRACT,
    writer_content_instruction,
)
from litagent.capacity import CapacityConfig
from litagent.eval.batched import (
    ANCHORED_FINDINGS_CONTRACT,
    _validate_decision_consistency,
    assessment_schema,
    batch_prompt,
    build_global_payload,
    global_prompt,
)
from litagent.eval.inventory_recovery import (
    DEFINITION_RECOVERY_INSTRUCTION,
    MIXED_CONTRACT,
    MIXED_STABILITY_INSTRUCTION,
    recovery_instruction,
)
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import (
    make_writer_envelope,
    validate_writer_envelope,
)

V1 = scoped.SUPPORT_VALIDATION_REVISION
V2 = scoped.SUPPORT_VALIDATION_REVISION_V2
OLD_V1_IDENTITY = "1cafb459f74f09ce5276aacebabc85bb8dbac3ee092ace9de01aa3039313f607"


def envelope(revision=V2):
    """Use the real complete selected parent, never a self-rehashed marker."""
    return make_writer_envelope(
        body_units=66,
        assertions=128,
        body_bytes=32000,
        source_identity="a" * 64,
        builder_identity="b" * 64,
        contract="flat-writer-envelope-v1",
        capacity=CapacityConfig(),
        round_budget=True,
        global_findings_contract=ANCHORED_FINDINGS_CONTRACT,
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        summary_constraints=True,
        classification_review=True,
        repair_feedback=True,
        reference_contract="ordered-reference-set-v1",
        transport_revision="host-lossless-transport-v1",
        **({"support_validation_revision": revision} if revision else {}),
    )


def request_payload(canonical, stage="assessment", revision=V2):
    """Retain complete public source text and native host identifiers."""
    value = copy.deepcopy(canonical)
    value["source_units"][0]["source_fragments"][0]["text"] += " [E:e1]"
    value["writer_envelope"] = envelope(revision)
    if stage == "assessment":
        value["evaluation_input_contract"] = "flat-evaluation-input-v1"
        value["schema"] = assessment_schema(value)
        return value
    if stage == "inventory":
        return {
            "writer_envelope": value["writer_envelope"],
            "inventory_output_contract": "host-flat-fragment-inventory-v1",
            "evaluation_input_contract": "flat-evaluation-input-v1",
            "source_units": value["source_units"],
            "allowed_citation_aliases": ["e1"],
            "schema": scoped.flat_inventory_schema(value["source_units"]),
        }
    value.update(
        global_findings_contract=ANCHORED_FINDINGS_CONTRACT,
        coverage_requirements=[],
        writer_content_contract=WRITER_CONTENT_CONTRACT,
        summary_constraints_contract="effective-summary-limits-v1",
        source_applicability_contract=APPLICABILITY_CONTRACT,
        evaluation_input_contract="flat-evaluation-input-v1",
        report_input_contract="host-report-input-v1",
    )
    # Use the real stage whitelist. Assessment papers are deliberately absent
    # from global input; report-side consistency is not a second paper verdict.
    inventory = [
        {
            "source_unit_id": unit["source_unit_id"],
            "source_fragment_id": fragment["source_fragment_id"],
            "assertion_count": sum(
                claim["source_fragment_id"] == fragment["source_fragment_id"]
                for claim in value["assertions"]
            ),
            "exclusion_reason": "",
        }
        for unit in value["source_units"]
        for fragment in unit["source_fragments"]
    ]
    return build_global_payload(value, {"fragment_inventory": inventory})


def prompt_for(stage, payload):
    """Exercise the same concrete producers used before actual transport."""
    if stage == "inventory":
        return scoped.FLAT_INVENTORY_PROMPT + scoped.LOCATED_INVENTORY_INSTRUCTION
    return batch_prompt(payload) if stage == "assessment" else global_prompt(payload)


def without_descriptions(value):
    """Compare validation shape, allowing only the authorized annotations."""
    if isinstance(value, dict):
        return {
            key: without_descriptions(item)
            for key, item in value.items()
            if key != "description"
        }
    if isinstance(value, list):
        return [without_descriptions(item) for item in value]
    return value


@pytest.mark.parametrize("stage", ["inventory", "assessment", "global"])
def test_final_native_constructor_receives_one_shared_definition_set(canonical, stage):
    payload = request_payload(canonical, stage)
    before = copy.deepcopy(payload)
    messages, native = scoped.build_evaluation_request(
        prompt_for(stage, payload), payload
    )
    system, wire = messages[0]["content"], json.loads(messages[1]["content"])
    schema = native["json_schema"]["schema"]
    assert system.startswith(scoped.evaluation_definition_prompt(stage, payload))
    assert "or loses a scientific premise" not in system
    assert scoped.CLASSIFICATION_RULES not in system
    assert scoped.SUPPORT_VALIDATION_INSTRUCTION not in system
    for concept in scoped._EVALUATION_STAGE_DEFINITIONS_V2[stage]:
        definition = scoped.EVALUATION_DEFINITIONS_V2[concept]
        if stage == "assessment" and concept == "classification":
            assert (
                "classification_valid tests their agreement"
                not in schema["description"]
            )
            assert (
                "The host derives agreement with the assigned kind;"
                in schema["description"]
            )
        else:
            assert schema["description"].count(definition) == 1
        assert definition not in system
    assert strict_request_reservation(messages, 16384, native) > 16384
    assert payload == before
    if stage == "assessment":
        assert wire["original_supports"] == payload["original_supports"]
        assert (
            schema["$defs"]["PassageDecision"]["properties"]["reason_code"][
                "description"
            ]
            == scoped.decision_status_instruction()
        )
    elif stage == "inventory":
        assert schema["properties"]["fragments"]["required"] == ["f1"]
        assert "HOST inventory aliases" in system
    else:
        assert (
            "HOST ANALYSIS OWNERSHIP" not in system
            or "analysis_claim_ids_by_section" in system
        )
        assert "positive confirmations" not in system
        assert "positive confirmations" in schema["description"]
        assert (
            "[issues]"
            in schema["$defs"]["AnchoredGlobalIssue"]["properties"]["code"][
                "description"
            ]
        )


@pytest.mark.parametrize("stage", ["inventory", "assessment", "global"])
def test_descriptions_do_not_mutate_output_fields_enums_or_guards(canonical, stage):
    payload = request_payload(canonical, stage)
    original = copy.deepcopy(
        scoped.Assessment.model_json_schema()
        if stage == "assessment"
        else payload["schema"]
    )
    before = copy.deepcopy(original)
    described = scoped.evaluation_definition_schema(original, stage=stage)
    assert original == before
    assert without_descriptions(described) == without_descriptions(original)
    assert described is not original
    if stage == "assessment":
        kinds = described["$defs"]["PassageDecision"]["properties"]["correct_kind"][
            "enum"
        ]
        assert set(kinds) == {"scientific", "runtime_method", "report_structure"}
        assert set(
            described["$defs"]["PassageDecision"]["properties"]["status"]["enum"]
        ) == {"supported", "unsupported", "unverifiable"}


def test_both_actual_alias_candidates_are_described_before_full_reservation(
    canonical, monkeypatch
):
    payload = request_payload(canonical)
    measured = []
    original = scoped.strict_request_reservation

    def measure(messages, output, native):
        measured.append(copy.deepcopy(native["json_schema"]["schema"]))
        return original(messages, output, native)

    monkeypatch.setattr(scoped, "strict_request_reservation", measure)
    scoped.build_evaluation_request(batch_prompt(payload), payload)
    assert len(measured) == 2
    assert all(s["description"].startswith("[atomic]") for s in measured)
    for schema in measured:
        assert (
            schema["$defs"]["AspectCheck"]["properties"]["supported"]["description"]
            == "Selected basis establishes this aspect under [aspects]."
        )


def test_json_fallback_contains_the_same_definitions(canonical, monkeypatch):
    payload = request_payload(canonical)
    payload.pop("evaluation_input_contract")
    payload.pop("assessment_input_contract")
    monkeypatch.setattr(scoped, "evaluation_response_format", lambda payload: None)
    messages, native = scoped.build_evaluation_request(batch_prompt(payload), payload)
    assert native is None
    wire = json.loads(messages[1]["content"])
    assert wire["schema"] == scoped.evaluation_definition_schema(
        payload["schema"], stage="assessment", host_kind_agreement=True
    )
    assert wire["original_supports"] == payload["original_supports"]


def test_v2_definition_change_isolates_v1_identity_and_actual_wire(
    canonical, monkeypatch
):
    old = request_payload(canonical, revision=V1)
    none = request_payload(canonical, revision=None)
    selected = request_payload(canonical)
    old_wire = scoped.build_evaluation_request(batch_prompt(old), old)
    none_wire = scoped.build_evaluation_request(batch_prompt(none), none)
    old_identity = scoped.support_validation_identity()
    current_identity = scoped.support_validation_identity(V2)
    definitions = copy.deepcopy(scoped.EVALUATION_DEFINITIONS_V2)
    definitions["atomic"] += " Controlled definition mutation."
    monkeypatch.setattr(scoped, "EVALUATION_DEFINITIONS_V2", definitions)
    assert scoped.support_validation_identity() == old_identity == OLD_V1_IDENTITY
    assert scoped.support_validation_identity(V1) == OLD_V1_IDENTITY
    assert scoped.support_validation_identity(V2) != current_identity
    assert scoped.build_evaluation_request(batch_prompt(old), old) == old_wire
    assert scoped.build_evaluation_request(batch_prompt(none), none) == none_wire
    with pytest.raises(ValueError, match="support_validation"):
        validate_writer_envelope(selected["writer_envelope"])


def test_no_envelope_keeps_exact_legacy_native_wire(canonical):
    payload = copy.deepcopy(canonical)
    payload["source_units"][0]["source_fragments"][0]["text"] += " [E:e1]"
    payload["evaluation_input_contract"] = "flat-evaluation-input-v1"
    messages, schema = scoped.build_evaluation_request(batch_prompt(payload), payload)
    digest = hashlib.sha256(
        json.dumps(
            [messages, schema], ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert digest == "e05fc00c282d73b2a7fc2ab44b5cf9589d0cd3116722db1aa4995e2acf2dee29"
    assert scoped.support_validation_identity() == OLD_V1_IDENTITY


def test_v1_identity_monkeypatch_keeps_its_zero_argument_consumer(monkeypatch):
    value = envelope(V1)
    original = scoped.support_validation_identity()
    monkeypatch.setattr(scoped, "support_validation_identity", lambda: original)
    assert scoped._support_revision(value) == V1


@pytest.mark.parametrize("stage", [None, False, 1, [], {}, "unknown"])
def test_unknown_stage_is_typed_in_both_helpers(canonical, stage):
    payload = request_payload(canonical)
    with pytest.raises(ValueError, match="evaluation_definition_stage_invalid"):
        scoped.evaluation_definition_prompt(stage, payload)
    with pytest.raises(ValueError, match="evaluation_definition_schema_invalid"):
        scoped.evaluation_definition_schema(payload["schema"], stage=stage)


@pytest.mark.parametrize("revision", [None, False, True, 1, [], {}, "unknown"])
def test_identity_never_coerces_disabled_or_unknown_revision(revision):
    with pytest.raises(
        ValueError, match="evaluation_support_validation_revision_invalid"
    ):
        scoped.support_validation_identity(revision)


@pytest.mark.parametrize("stage", ["inventory", "assessment", "global"])
@pytest.mark.parametrize("fault", ["missing_definition", "missing_field", "wrong_type"])
def test_malformed_stage_schema_is_not_silently_skipped(canonical, stage, fault):
    payload = request_payload(canonical, stage)
    schema = payload["schema"]
    name, fields = next(iter(scoped._EVALUATION_SCHEMA_FIELDS_V2[stage].items()))
    field = next(iter(fields))
    if fault == "missing_definition":
        del schema["$defs"][name]
    elif fault == "missing_field":
        del schema["$defs"][name]["properties"][field]
    else:
        schema["$defs"][name]["properties"][field] = None
    with pytest.raises(ValueError, match="evaluation_definition_schema_invalid"):
        scoped.evaluation_definition_schema(schema, stage=stage)


@pytest.mark.parametrize("prompt", ["unrecognized task", None, False, [], {}])
def test_inventory_builder_never_discards_unknown_extra_instructions(canonical, prompt):
    payload = request_payload(canonical, "inventory")
    with pytest.raises(ValueError, match="evaluation_definition_prompt_invalid"):
        scoped.build_evaluation_request(prompt, payload)
    malformed = copy.deepcopy(payload)
    malformed["assessment_output_contract"] = "host-batch-assessment-v1"
    with pytest.raises(ValueError, match="evaluation_definition_stage_invalid"):
        scoped.build_evaluation_request(scoped.FLAT_INVENTORY_PROMPT, malformed)


def test_stable_recovery_uses_shared_definitions_without_rewriting_v1(canonical):
    value = request_payload(canonical, "inventory")
    value["inventory_recovery"] = {
        "contract": MIXED_CONTRACT,
        "preserved_fragments": [],
    }
    prompt = scoped.evaluation_definition_prompt("inventory", value)
    assert prompt.count(DEFINITION_RECOVERY_INSTRUCTION) == 1
    assert "not validation" in prompt
    assert "fresh actual-citation assessment" in prompt
    assert MIXED_STABILITY_INSTRUCTION not in prompt
    assert (
        recovery_instruction(MIXED_CONTRACT, stable=True) == MIXED_STABILITY_INSTRUCTION
    )
    assert (
        recovery_instruction(
            MIXED_CONTRACT, stable=True, support_validation_revision=V1
        )
        == MIXED_STABILITY_INSTRUCTION
    )
    assert (
        recovery_instruction(
            MIXED_CONTRACT, stable=True, support_validation_revision=V2
        )
        == DEFINITION_RECOVERY_INSTRUCTION
    )
    with pytest.raises(ValueError, match="inventory_recovery_policy_invalid"):
        recovery_instruction(MIXED_CONTRACT, support_validation_revision=V2)


def test_consistency_guard_keeps_a_negative_status_with_positive_aspects_invalid():
    claim = {
        "claim_id": "c1",
        "source_unit_id": "u1",
        "source_fragment_id": "f1",
        "kind": "scientific",
        "required_support_aspects": ["predicate"],
        "evidence": [{"evidence_alias": "e1", "material_ref": "e1"}],
    }
    decision = scoped.PassageDecision.model_validate(
        {
            "claim_id": "c1",
            "correct_kind": "scientific",
            "atomic": True,
            "classification_valid": True,
            "source_faithful": True,
            "status": "unsupported",
            "reason_code": "attribution_mismatch",
            "explanation": "Controlled contradictory fields, not a scientific gold label.",
            "support_basis": [{"support_id": "p1"}],
            "aspect_checks": [
                {
                    "aspect": "predicate",
                    "supported": True,
                    "support_ids": ["p1"],
                    "explanation": "Controlled positive aspect.",
                }
            ],
        }
    )
    before = decision.model_dump()
    with pytest.raises(ValueError, match="evaluation_decision_consistency_invalid"):
        _validate_decision_consistency(
            claim,
            decision,
            {
                "support_passages": {"p1": ["e1", "s1", 0, 3]},
                "original_supports": {
                    "s1": {"text": "A supported controlled relation."}
                },
            },
        )
    assert decision.model_dump() == before


def test_real_global_whitelist_and_sender_use_review_duties_not_absent_papers(
    canonical,
):
    """Keep actual global review duties separate from absent paper support."""
    payload = request_payload(canonical, "global")
    before = copy.deepcopy(payload)
    assert (
        not {"cited_materials", "original_supports", "support_passages"}
        & payload.keys()
    )
    prompt = global_prompt(payload, native_unit_ownership=True)
    messages, schema = scoped.build_evaluation_request(prompt, payload)
    system, wire = messages[0]["content"], json.loads(messages[1]["content"])
    assert (
        not {"cited_materials", "original_supports", "support_passages"} & wire.keys()
    )
    assert system.count(scoped.EVALUATION_GLOBAL_REVIEW_CRITERIA_V2) == 1
    assert system.count(scoped.EVALUATION_GLOBAL_APPLICABILITY_V2) == 1
    assert writer_content_instruction("effective-summary-limits-v1") not in system
    assert APPLICABILITY_INSTRUCTION not in system
    assert "global input has no paper originals" in system
    normalized = " ".join(system.split())
    assert "only assessment checks them against paper originals" in normalized
    assert "report, inventory and summaries" in normalized
    assert "never omit a claim, weaken a qualifier" in " ".join(system.split())
    assert "hide science as organization" in system
    assert "Never relabel such text" in system
    assert (
        scoped.EVALUATION_DEFINITIONS_V2["atomic"]
        in schema["json_schema"]["schema"]["description"]
    )
    assert payload == before


@pytest.mark.parametrize("marker", [None, "effective-summary-limits-v1"])
def test_v2_global_summary_marker_has_an_explicit_compatible_selection(
    canonical, marker
):
    payload = request_payload(canonical, "global")
    payload["summary_constraints_contract"] = marker
    messages, _ = scoped.build_evaluation_request(global_prompt(payload), payload)
    system = messages[0]["content"]
    assert scoped.EVALUATION_GLOBAL_REVIEW_CRITERIA_V2 in system
    assert (LEGACY_SUMMARY_SELECTION in system) is (marker is None)
    assert "punctuation alone proves no defect" in system


@pytest.mark.parametrize("marker", [False, True, 0, 1, "unknown", [], {}])
def test_v2_bad_summary_selection_is_not_silently_removed_by_global_whitelist(
    canonical, marker
):
    upstream = request_payload(canonical)
    upstream.update(
        writer_content_contract=WRITER_CONTENT_CONTRACT,
        summary_constraints_contract=marker,
    )
    before = copy.deepcopy(upstream)
    with pytest.raises(
        ValueError, match="evaluation_summary_constraints_contract_invalid"
    ):
        build_global_payload(upstream, {})
    current = request_payload(canonical, "global")
    current["summary_constraints_contract"] = marker
    with pytest.raises(
        ValueError, match="evaluation_summary_constraints_contract_invalid"
    ):
        global_prompt(current)
    assert upstream == before


@pytest.mark.parametrize(
    "field,constant",
    [
        ("writer_content_contract", "EVALUATION_GLOBAL_REVIEW_CRITERIA_V2"),
        ("source_applicability_contract", "EVALUATION_GLOBAL_APPLICABILITY_V2"),
    ],
)
def test_v2_absent_optional_global_duty_does_not_invent_the_stage_segment(
    canonical, field, constant
):
    payload = request_payload(canonical, "global")
    payload.pop(field)
    messages, _ = scoped.build_evaluation_request(global_prompt(payload), payload)
    assert getattr(scoped, constant) not in messages[0]["content"]


@pytest.mark.parametrize("revision", [None, V1])
def test_old_global_and_current_assessment_keep_original_applicability(
    canonical, revision
):
    payload = request_payload(canonical, "global", revision)
    messages, _ = scoped.build_evaluation_request(global_prompt(payload), payload)
    assert APPLICABILITY_INSTRUCTION in messages[0]["content"]
    assert (
        writer_content_instruction("effective-summary-limits-v1")
        in messages[0]["content"]
    )
    assert scoped.EVALUATION_GLOBAL_APPLICABILITY_V2 not in messages[0]["content"]
    assessed = request_payload(canonical)
    assessed["source_applicability_contract"] = APPLICABILITY_CONTRACT
    messages, _ = scoped.build_evaluation_request(batch_prompt(assessed), assessed)
    wire = json.loads(messages[1]["content"])
    assert APPLICABILITY_INSTRUCTION in messages[0]["content"]
    assert wire["original_supports"] == assessed["original_supports"]
    assert scoped.EVALUATION_GLOBAL_APPLICABILITY_V2 not in messages[0]["content"]


@pytest.mark.parametrize(
    "constant",
    ["EVALUATION_GLOBAL_REVIEW_CRITERIA_V2", "EVALUATION_GLOBAL_APPLICABILITY_V2"],
)
def test_new_global_wording_changes_only_explicit_v2_identity_and_actual_wire(
    canonical, monkeypatch, constant
):
    old = [request_payload(canonical, "global", revision) for revision in (None, V1)]
    old_wires = [
        scoped.build_evaluation_request(global_prompt(payload), payload)
        for payload in old
    ]
    selected = request_payload(canonical, "global")
    current = scoped.build_evaluation_request(global_prompt(selected), selected)
    before = scoped.support_validation_identity(V2)
    monkeypatch.setattr(
        scoped, constant, getattr(scoped, constant) + " Controlled semantic variation."
    )
    assert scoped.support_validation_identity() == OLD_V1_IDENTITY
    assert scoped.support_validation_identity(V2) != before
    for payload, expected in zip(old, old_wires, strict=True):
        assert (
            scoped.build_evaluation_request(global_prompt(payload), payload) == expected
        )
    with pytest.raises(ValueError, match="support_validation"):
        validate_writer_envelope(selected["writer_envelope"])
    new = request_payload(canonical, "global")
    actual = scoped.build_evaluation_request(global_prompt(new), new)
    assert getattr(scoped, constant) in actual[0][0]["content"]
    assert actual != current
