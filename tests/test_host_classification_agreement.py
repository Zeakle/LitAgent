"""Derive classification agreement without promoting science or historical raw.

Declared decisions are transport fixtures, not scientific gold. Integration
cases replace HTTP only and retain real schemas, SDK, checkpoints and consumers.
"""

import copy
import json

import httpx
import pytest
import test_mixed_inventory_pipeline as mixed
import test_report_repair_eligibility as semantic
import test_runtime_classification_pipeline as runtime
from jsonschema import Draft202012Validator
from test_flat_writer_pipeline import FlatGateway
from test_source_map_pipeline import products, sent_with
from test_support_receipts_pipeline import SELECTION
from test_writer_v2 import case  # noqa: F401

from litagent.eval import scoped
from litagent.eval.batched import assessment_schema, decode_assessment_response
from litagent.quality import digest
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.writing_v2 import writer_budget_plan

KINDS = ("scientific", "runtime_method", "report_structure")
REVISION = "host-support-receipts-v2"
PREVIOUS_V2_IDENTITY = (
    "16ec9f6bcf1a85a5647b52c79183fbe247e47cfbcbac65b96121790bf5420d0c"
)
V1_IDENTITY = "1cafb459f74f09ce5276aacebabc85bb8dbac3ee092ace9de01aa3039313f607"

# Exact original050 decision objects; the historical response is never modified.
ORIGINAL050_BATCH_RESPONSE_SHA256 = (
    "ac94de07aafd62cbd513a9261e1b739b99b986cfee9d891a2f2fc123abb7c7ea"
)
ORIGINAL050_DECISIONS = {
    "claim_100": {
        "claim_id": "claim_100",
        "correct_kind": "report_structure",
        "atomic": True,
        "classification_valid": True,
        "source_faithful": True,
        "status": "unsupported",
        "reason_code": "uncited_claim",
        "explanation": "This is a report-organization instruction about how the "
        "review should separate same-benchmark relative results "
        "from cross-dataset generalization. It is faithfully "
        "extracted from fragment f3, but fragment f3 has no "
        "citations, so there is no paper basis for this "
        "methodological instruction.",
        "support_basis": [],
        "aspect_checks": [
            {
                "aspect": "predicate",
                "supported": False,
                "support_ids": [],
                "explanation": "No cited source is supplied for this "
                "report-level comparison rule.",
            },
            {
                "aspect": "conditions",
                "supported": False,
                "support_ids": [],
                "explanation": "No cited source is supplied for the "
                "required distinction condition.",
            },
        ],
    },
    "claim_102": {
        "claim_id": "claim_102",
        "correct_kind": "report_structure",
        "atomic": True,
        "classification_valid": True,
        "source_faithful": True,
        "status": "unsupported",
        "reason_code": "uncited_claim",
        "explanation": "This is a report-organization conclusion that the "
        "review should not merge the available abstract-level "
        "conclusions into a global ranking. It is faithfully "
        "extracted from fragment f3, but fragment f3 has no "
        "citations, so the report-level decision is not "
        "paper-supported.",
        "support_basis": [],
        "aspect_checks": [
            {
                "aspect": "predicate",
                "supported": False,
                "support_ids": [],
                "explanation": "No cited source is supplied for the "
                "report's choice not to produce a "
                "global ranking.",
            },
            {
                "aspect": "conditions",
                "supported": False,
                "support_ids": [],
                "explanation": "No cited source is supplied for the "
                "condition under which the ranking is "
                "withheld.",
            },
        ],
    },
}


def selected_envelope(case, *, revision=REVISION, complete=True):
    """Obtain an authentic server envelope from the unchanged plan producer."""
    flags = {**SELECTION, "support_validation_revision": revision}
    if not complete:
        flags["mixed_inventory_recovery"] = False
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **flags,
    )
    assert plan["status"] == "admitted"
    envelope = plan["envelope"]
    validate_writer_envelope(envelope)
    return envelope


def packet(case, *, assigned="scientific", correct="scientific", envelope=True):
    """Declare one bound negative decision without supplying scientific proof."""
    text = "The report compares supplied methods."
    claim = {
        "claim_id": "claim_0",
        "kind": assigned,
        "claim_text": text,
        "source_unit_id": "unit_0",
        "source_fragment_id": "unit_0:f0",
        "source_quote": text,
        "source_start": 0,
        "source_end": len(text),
        "evidence_ids": [],
        "evidence": [],
        "required_support_aspects": [],
        "local_error": None,
    }
    payload = {
        "body_analysis_contract": "body-analysis-v1",
        "assessment_output_contract": "host-batch-assessment-v1",
        "assertions": [claim],
        "source_units": [{"source_unit_id": "unit_0", "text": text}],
        "cited_materials": {},
        "original_supports": {},
        "support_passages": {},
    }
    if envelope:
        payload["writer_envelope"] = selected_envelope(case)
    value = {
        "units": [
            {
                "source_unit_id": "unit_0",
                "complete": True,
                "explanation": "Controlled complete inventory, not scientific proof.",
            }
        ],
        "decisions": [
            {
                "claim_id": "claim_0",
                "correct_kind": correct,
                "atomic": True,
                "source_faithful": True,
                "status": "unsupported",
                "reason_code": "insufficient_support",
                "explanation": "No supplied paper proof in this declared fixture.",
                "support_basis": [],
                "aspect_checks": [],
            }
        ],
        "conflicts": [],
    }
    return payload, value


def without_annotations(value):
    """Compare schema validation rules independently of explanatory text."""
    if isinstance(value, dict):
        return {
            key: without_annotations(child)
            for key, child in value.items()
            if key not in {"title", "description"}
        }
    if isinstance(value, list):
        return [without_annotations(child) for child in value]
    return value


@pytest.mark.parametrize("assigned", KINDS)
@pytest.mark.parametrize("correct", KINDS)
def test_all_kind_pairs_derive_only_agreement(case, assigned, correct):
    payload, value = packet(case, assigned=assigned, correct=correct)
    raw = json.dumps(value)
    before = copy.deepcopy((payload, value))
    Draft202012Validator(assessment_schema(payload)).validate(value)
    [decision] = decode_assessment_response(raw, payload).decisions
    assert decision.classification_valid is (assigned == correct)
    canonical = decision.model_dump()
    canonical.pop("classification_valid")
    assert canonical == value["decisions"][0]
    assert (payload, value) == before and json.loads(raw) == value
    [checked] = scoped.validate_assessment_decisions(
        payload["assertions"],
        [decision],
        ledger={},
        aliases={},
        support_passages={},
        context={"_independent_classification": True},
        strict=True,
    )
    assert checked["kind"] == assigned and checked["status"] != "supported"
    if assigned != correct:
        assert checked["status"] == "unverifiable"
        assert checked["local_error"] == "evaluation_classification_or_source_invalid"


def test_native_schema_only_removes_redundant_model_field(case):
    payload, value = packet(case)
    legacy = {key: copy.deepcopy(child) for key, child in payload.items()}
    legacy.pop("writer_envelope")
    old_schema, new_schema = assessment_schema(legacy), assessment_schema(payload)
    before = copy.deepcopy(payload)
    decision = new_schema["$defs"]["PassageDecision"]
    assert "classification_valid" not in decision["properties"]
    assert "classification_valid" not in decision["required"]
    assert decision["additionalProperties"] is False
    reconstructed = copy.deepcopy(new_schema)
    reconstructed["$defs"]["PassageDecision"]["properties"]["classification_valid"] = (
        old_schema["$defs"]["PassageDecision"]["properties"]["classification_valid"]
    )
    reconstructed["$defs"]["PassageDecision"]["required"] = old_schema["$defs"][
        "PassageDecision"
    ]["required"]
    assert without_annotations(reconstructed) == without_annotations(old_schema)
    payload["schema"] = new_schema
    messages, native = scoped.build_evaluation_request(
        scoped.evaluation_definition_prompt("assessment", payload), payload
    )
    assert payload == {**before, "schema": new_schema}
    actual = native["json_schema"]["schema"]
    assert without_annotations(actual) == without_annotations(new_schema)
    Draft202012Validator(actual).validate(value)
    assert json.loads(messages[1]["content"])["assertions"][0]["kind"] == "scientific"
    bad = copy.deepcopy(value)
    bad["decisions"][0]["classification_valid"] = False
    assert not Draft202012Validator(actual).is_valid(bad)
    with pytest.raises(ValueError, match="classification_field_invalid"):
        decode_assessment_response(json.dumps(bad), payload)


@pytest.mark.parametrize("extra", [True, False, None, 0, "false"])
def test_provider_cannot_supply_the_host_derived_field(case, extra):
    payload, value = packet(case)
    value["decisions"][0]["classification_valid"] = extra
    assert not Draft202012Validator(assessment_schema(payload)).is_valid(value)
    with pytest.raises(ValueError, match="classification_field_invalid"):
        decode_assessment_response(json.dumps(value), payload)


@pytest.mark.parametrize("correct", [None, "", "organization", 0, True, [], {}])
def test_missing_or_invalid_correct_kind_is_not_inferred(case, correct):
    payload, value = packet(case)
    value["decisions"][0]["correct_kind"] = correct
    with pytest.raises(ValueError, match="classification_kind_invalid"):
        decode_assessment_response(json.dumps(value), payload)


def test_omitted_correct_kind_is_not_inferred(case):
    payload, value = packet(case)
    value["decisions"][0].pop("correct_kind")
    with pytest.raises(ValueError, match="classification_kind_invalid"):
        decode_assessment_response(json.dumps(value), payload)


@pytest.mark.parametrize("fault", ["missing", "unknown", "duplicate", "null", "int"])
def test_raw_claim_id_coverage_is_complete_and_exact(case, fault):
    payload, value = packet(case)
    if fault == "missing":
        value["decisions"] = []
    elif fault == "duplicate":
        value["decisions"] *= 2
    else:
        value["decisions"][0]["claim_id"] = {
            "unknown": "claim_999",
            "null": None,
            "int": 0,
        }[fault]
    with pytest.raises(ValueError, match="evaluation_decision_coverage_invalid"):
        decode_assessment_response(json.dumps(value), payload)


@pytest.mark.parametrize(
    "fault",
    [
        "root_list",
        "root_null",
        "root_string",
        "missing_decisions",
        "decisions_null",
        "decision_null",
        "extra_root",
        "extra_decision",
        "missing_atomic",
        "missing_units",
    ],
)
def test_invalid_response_shapes_remain_protocol_errors(case, fault):
    payload, value = packet(case)
    if fault.startswith("root_"):
        value = {"root_list": [], "root_null": None, "root_string": "report"}[fault]
    elif fault == "missing_decisions":
        value.pop("decisions")
    elif fault == "decisions_null":
        value["decisions"] = None
    elif fault == "decision_null":
        value["decisions"] = [None]
    elif fault == "extra_root":
        value["approved"] = True
    elif fault == "extra_decision":
        value["decisions"][0]["approved"] = True
    elif fault == "missing_atomic":
        value["decisions"][0].pop("atomic")
    else:
        value.pop("units")
    with pytest.raises(ValueError, match=r"^evaluation_"):
        decode_assessment_response(json.dumps(value), payload)


@pytest.mark.parametrize(
    "field,value",
    [
        ("atomic", 0),
        ("atomic", "true"),
        ("atomic", None),
        ("source_faithful", 1),
        ("source_faithful", "false"),
        ("source_faithful", None),
        ("status", 1),
        ("reason_code", False),
        ("explanation", 1),
        ("support_basis", {}),
        ("aspect_checks", {}),
    ],
)
def test_raw_diagnostic_types_are_not_coerced(case, field, value):
    payload, response = packet(case)
    response["decisions"][0][field] = value
    assert not Draft202012Validator(assessment_schema(payload)).is_valid(response)
    with pytest.raises(ValueError, match="evaluation_assessment_response_invalid"):
        decode_assessment_response(json.dumps(response), payload)


@pytest.mark.parametrize("value", [0, "true", None])
def test_unit_complete_type_is_not_coerced(case, value):
    payload, response = packet(case)
    response["units"][0]["complete"] = value
    with pytest.raises(ValueError, match="evaluation_assessment_response_invalid"):
        decode_assessment_response(json.dumps(response), payload)


@pytest.mark.parametrize("where", ["root", "decision", "nested"])
def test_duplicate_json_keys_are_rejected_before_derivation(case, where):
    payload, value = packet(case)
    if where == "nested":
        value["decisions"][0]["aspect_checks"] = [
            {
                "aspect": "predicate",
                "supported": False,
                "support_ids": [],
                "explanation": "Negative.",
            }
        ]
    raw = json.dumps(value)
    if where == "root":
        raw = '{"units": [], ' + raw[1:]
    elif where == "decision":
        raw = raw.replace('"atomic": true', '"atomic": false, "atomic": true', 1)
    else:
        raw = raw.replace(
            '"supported": false', '"supported": true, "supported": false', 1
        )
    with pytest.raises(ValueError, match="evaluation_assessment_duplicate_key"):
        decode_assessment_response(raw, payload)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_json_numeric_constants_are_rejected(case, constant):
    payload, value = packet(case)
    raw = json.dumps(value).replace('"atomic": true', '"atomic": ' + constant, 1)
    with pytest.raises(ValueError, match="evaluation_assessment_json_invalid"):
        decode_assessment_response(raw, payload)


@pytest.mark.parametrize("raw", [None, {}, [], True, b"{}", "{not-json"])
def test_non_string_or_invalid_json_has_specific_error(case, raw):
    payload, _ = packet(case)
    with pytest.raises(ValueError, match="evaluation_assessment_json_invalid"):
        decode_assessment_response(raw, payload)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "wrong_id_type", "unknown_kind"]
)
def test_host_assertion_binding_cannot_be_guessed(case, fault):
    payload, value = packet(case)
    if fault == "missing":
        payload.pop("assertions")
    elif fault == "duplicate":
        payload["assertions"] *= 2
    elif fault == "wrong_id_type":
        payload["assertions"][0]["claim_id"] = False
    else:
        payload["assertions"][0]["kind"] = "organization"
    with pytest.raises(
        ValueError, match="evaluation_assessment_input_claim_identity_invalid"
    ):
        decode_assessment_response(json.dumps(value), payload)


@pytest.mark.parametrize("revision", [None, "host-support-receipts-v1", REVISION])
def test_unselected_schema_keeps_original_boolean_and_false_negative(case, revision):
    payload, value = packet(case, envelope=False)
    if revision == "host-support-receipts-v1":
        payload["writer_envelope"] = selected_envelope(case, revision=revision)
    elif revision == REVISION:
        # This generic revision-reader input is not a funded writer envelope and
        # cannot authorize execution. It must retain the old non-selected schema.
        payload["writer_envelope"] = {
            "support_validation_revision": revision,
            "support_validation_identity": scoped.support_validation_identity(revision),
        }
        assert scoped._support_revision(payload["writer_envelope"]) == revision
    assert not scoped._runtime_classification_round_enabled(
        payload.get("writer_envelope")
    )
    schema = assessment_schema(payload)
    assert "classification_valid" in schema["$defs"]["PassageDecision"]["required"]
    value["decisions"][0]["classification_valid"] = False
    Draft202012Validator(schema).validate(value)
    result = decode_assessment_response(json.dumps(value), payload)
    [decision] = result.decisions
    assert decision.correct_kind == payload["assertions"][0]["kind"]
    assert decision.classification_valid is False
    [checked] = scoped.validate_assessment_decisions(
        payload["assertions"],
        [decision],
        ledger={},
        aliases={},
        support_passages={},
        context={},
        strict=True,
    )
    assert checked["status"] == "unverifiable"
    assert checked["local_error"] == "evaluation_classification_or_source_invalid"
    value["decisions"][0].pop("classification_valid")
    with pytest.raises(ValueError, match="classification_valid"):
        decode_assessment_response(json.dumps(value), payload)


def test_funded_producer_still_rejects_partial_selected_policy(case):
    with pytest.raises(ValueError, match="support_validation_policy_invalid"):
        selected_envelope(case, complete=False)


@pytest.mark.parametrize("claim_id", ["claim_100", "claim_102"])
def test_original050_flag_kind_conflict_is_never_migrated_to_new_wire(case, claim_id):
    payload, value = packet(case, correct="report_structure")
    payload["assertions"][0]["claim_id"] = claim_id
    value["decisions"] = [copy.deepcopy(ORIGINAL050_DECISIONS[claim_id])]
    raw, before = json.dumps(value), copy.deepcopy(value)
    with pytest.raises(ValueError, match="classification_field_invalid"):
        decode_assessment_response(raw, payload)
    assert json.loads(raw) == before
    old = copy.deepcopy(payload)
    old["writer_envelope"]["support_validation_identity"] = PREVIOUS_V2_IDENTITY
    old["writer_envelope"]["identity"] = digest(
        {
            key: child
            for key, child in old["writer_envelope"].items()
            if key != "identity"
        }
    )
    with pytest.raises(ValueError, match="support_validation_identity_invalid"):
        decode_assessment_response(raw, old)
    assert scoped.support_validation_identity(REVISION) != PREVIOUS_V2_IDENTITY
    assert scoped.support_validation_identity() == V1_IDENTITY


@pytest.mark.parametrize(
    "local_error",
    [
        "evaluation_atomic_classification_unverified",
        "evaluation_source_scope_unverified",
    ],
)
def test_derived_true_does_not_clear_prior_inventory_error(case, local_error):
    payload, value = packet(case)
    payload["assertions"][0]["local_error"] = local_error
    result = decode_assessment_response(json.dumps(value), payload)
    [checked] = scoped.validate_assessment_decisions(
        payload["assertions"],
        result.decisions,
        ledger={},
        aliases={},
        support_passages={},
        context={"_independent_classification": True},
        strict=True,
    )
    assert checked["classification_valid"] is True
    assert checked["local_error"] == local_error
    assert checked["status"] == "unverifiable"


class RecordingAgreementHTTP:
    """Record actual wire and inject faults only after the normal mock producer."""

    def __init__(self, *args, agreement_fault=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.agreement_fault = agreement_fault
        self.agreement_responses = []

    async def __call__(self, request):
        response = await super().__call__(request)
        body = json.loads(request.content)
        messages = body.get("input", body.get("messages"))
        payload = json.loads(messages[1]["content"])
        if "cited_materials" not in payload:
            return response
        schema = (
            body["text"]["format"]["schema"]
            if "input" in body
            else body["response_format"]["json_schema"]["schema"]
        )
        provider = response.json()
        raw = provider["choices"][0]["message"]["content"]
        value = json.loads(raw)
        definition = schema["$defs"]["PassageDecision"]
        assert "classification_valid" not in definition["properties"]
        assert all("classification_valid" not in row for row in value["decisions"])
        if self.agreement_fault is not None and not self.agreement_responses:
            if self.agreement_fault == "extra_flag":
                value["decisions"][0]["classification_valid"] = False
            elif self.agreement_fault == "unknown_id":
                value["decisions"][0]["claim_id"] = "not-a-host-claim"
            raw = json.dumps(value)
            provider["choices"][0]["message"]["content"] = raw
            response = httpx.Response(200, json=provider)
        self.agreement_responses.append(
            {
                "schema": copy.deepcopy(schema),
                "raw": raw,
                "payload": copy.deepcopy(payload),
            }
        )
        if self.agreement_fault is None:
            Draft202012Validator(schema).validate(value)
        else:
            assert not Draft202012Validator(schema).is_valid(value)
        return response


class AgreementGateway(RecordingAgreementHTTP, FlatGateway):
    """Keep generation, reviewer and science consumers on their real entry path."""


class RecoveryAgreementGateway(RecordingAgreementHTTP, runtime.RuntimeGateway):
    """Keep mixed inventory recovery on the original bounded shared owner."""


class RepairAgreementGateway(RecordingAgreementHTTP, semantic.SemanticGateway):
    """Keep semantic repair and raw replay behind actual external responses."""


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "extra_flag", "unknown_id"])
async def test_actual_sender_native_wire_and_checkpoint_canonical_agree(
    case, tmp_path, monkeypatch, fault
):
    semantic.prepare_case(case)
    case.body["methods"][0]["sentences"][0]["text"] = case.semantic_corrected
    runtime.select_policy(monkeypatch, current_stack=True)
    gateway = AgreementGateway(case, agreement_fault=fault)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    _, journal, ledger = products(tmp_path)
    assert gateway.agreement_responses
    assert ledger["usage_uncertain"] is False
    assert not journal["candidates"]
    details = report["evaluation"]["faithfulness"]["details"]
    if fault is None:
        assert details["stages"]["global_check"] == "completed"
        assert details["coverage"]["complete"] is True
        assert report["delivery"]["publishable"] is True
        for progress_path in (tmp_path / "evaluation").glob("*/progress.json"):
            progress = json.loads(progress_path.read_text())["progress"]
            for batch in progress["batches"]:
                assert batch["status"] == "completed"
                assert all(
                    type(row["classification_valid"]) is bool
                    for row in batch["decisions"]
                )
            for row in progress["requests"]:
                if row["phase"] == "assessment":
                    assert all(
                        "classification_valid" not in item
                        for item in json.loads(row["response"])["decisions"]
                    )
    else:
        assert details["coverage"]["complete"] is False
        assert report["delivery"]["publishable"] is False
        assert not sent_with(gateway, "targets")
        assert not sent_with(gateway, "global_output_contract")
        assert report["metadata"]["targeted_repair_state"]["attempts"] == []


@pytest.mark.asyncio
async def test_actual_mixed_recovery_uses_same_derived_classification_decoder(
    case, tmp_path, monkeypatch
):
    runtime.select_policy(monkeypatch, current_stack=True)
    gateway = RecoveryAgreementGateway(case)
    report, _ = await mixed.execute(case, tmp_path, monkeypatch, gateway=gateway)
    initial = runtime.initial_details(report)
    [row] = [item for item in initial["assertions"] if item["kind"] == "runtime_method"]
    assert row["correct_kind"] == "scientific" and row["classification_valid"] is False
    assert initial["inventory_check"]["status"] == "failed"
    assert initial["stages"]["global_check"] == "completed"
    assert gateway.recovery_requests == 1
    assert all(
        "classification_valid" not in item
        for response in gateway.agreement_responses
        for item in json.loads(response["raw"])["decisions"]
    )
    assert report["delivery"]["publishable"] is True
    assert (
        report["evaluation"]["faithfulness"]["details"]["coverage"]["complete"] is True
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_actual_repair_replay_and_full_candidates_use_same_wire_contract(
    case, tmp_path, monkeypatch, failing_candidates
):
    semantic.prepare_case(case)
    monkeypatch.setattr(semantic, "SemanticGateway", RepairAgreementGateway)
    report, gateway = await semantic.run_semantic(
        case, tmp_path, monkeypatch, failing_candidates=failing_candidates
    )
    _, journal, ledger = products(tmp_path)
    expected = min(failing_candidates + 1, 2)
    assert gateway.repair_count == len(sent_with(gateway, "targets")) == expected
    assert len(journal["candidates"]) == expected
    assert gateway.inventories == expected + 1
    assert len(sent_with(gateway, "global_output_contract")) == expected + 1
    assert ledger["usage_uncertain"] is False
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    assert report["delivery"]["publishable"] is (failing_candidates < 2)
    initial = runtime.initial_details(report)
    assert initial["coverage"]["complete"] is False
    assert initial["inventory_check"]["status"] == "passed"
    assert initial["stages"]["global_check"] == "completed"
    assert all(
        "classification_valid" not in item
        for response in gateway.agreement_responses
        for item in json.loads(response["raw"])["decisions"]
    )
    assert all(
        type(item["classification_valid"]) is bool
        for batch in initial["stages"]["batches"]
        for item in batch["decisions"]
    )
    assert [row["committed"] for row in journal["candidates"]] == (
        [False] * (expected - 1) + [True] if failing_candidates < 2 else [False, False]
    )
