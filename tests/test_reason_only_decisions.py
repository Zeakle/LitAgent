"""Keep single-code decisions strict without upgrading historical verdicts."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator
from test_claim_aspect_contract import aspect_payload  # noqa: F401
from test_claim_response_contract import permissions  # noqa: F401
from test_claim_response_contract import complete_reply
from test_evaluation_dimensions_contract import native_for, nested_reply
from test_evaluation_dimensions_identity import dimension_envelope
from test_evaluation_wire_projection import canonical  # noqa: F401

from litagent.eval import batched
from litagent.eval import decision_output as codec
from litagent.eval import scoped
from litagent.eval.dimensions import _decision_schemas
from litagent.eval.results import CONTRACT as RESULT_CONTRACT
from litagent.quality import digest
from litagent.safety.planned_accounting import _envelope as historical_envelope
from litagent.safety.planned_budget import validate_writer_envelope


@pytest.fixture
def reason_payload(aspect_payload):
    """Use the actual owner constructor with the complete independent parent stack."""
    value = copy.deepcopy(aspect_payload)
    value["writer_envelope"] = dimension_envelope(
        evaluation_result_contract=RESULT_CONTRACT
    )
    value["schema"] = batched.assessment_schema(value)
    return value


def wire_reply(payload, alias_wire=False):
    """Produce fresh shape controls, never rewrite a historical provider response."""
    original, nested_format, aliases = native_for(payload, alias_wire)
    value = nested_reply(complete_reply(payload, original, aliases))
    for decision in value["decisions"]:
        decision["evidence_check"].pop("status")
        decision["evidence_check"].pop("support_basis")
        decision["evidence_check"]["aspect_checks"] = {
            row["aspect"]: {key: val for key, val in row.items() if key != "aspect"}
            for row in decision["evidence_check"]["aspect_checks"]
        }
    return value, nested_format, aliases


def rehash(value):
    """Rehash data without authenticating a tampered policy."""
    value["identity"] = digest({k: v for k, v in value.items() if k != "identity"})
    return value


def test_new_owner_binds_codec_but_old_owner_retains_its_identity(reason_payload):
    owner = reason_payload["writer_envelope"]
    assert codec.selected(owner)
    assert owner[codec.FIELDS[1]] == codec.contract_identity()
    assert validate_writer_envelope(owner) == owner
    historical_envelope(owner)
    old = dimension_envelope()
    assert not set(codec.FIELDS).intersection(old)
    assert not codec.selected(old)
    assert codec.decode_response("old invalid JSON", {"writer_envelope": old}) == (
        "old invalid JSON"
    )


@pytest.mark.parametrize("field", codec.FIELDS)
@pytest.mark.parametrize("value", [None, False, "forged"])
def test_half_or_forged_owner_is_not_a_wire_selector(reason_payload, field, value):
    owner = copy.deepcopy(reason_payload["writer_envelope"])
    owner[field] = value
    rehash(owner)
    with pytest.raises(ValueError, match="assessment_decision"):
        validate_writer_envelope(owner)
    with pytest.raises(ValueError):
        historical_envelope(owner)


def test_data_only_reader_does_not_use_current_codec_identity(
    reason_payload, monkeypatch
):
    owner = copy.deepcopy(reason_payload["writer_envelope"])
    owner[codec.FIELDS[1]] = "f" * 64
    rehash(owner)
    monkeypatch.setattr(codec, "contract_identity", lambda: "0" * 64)
    historical_envelope(owner)
    with pytest.raises(ValueError):
        validate_writer_envelope(owner)


@pytest.mark.parametrize("alias_wire", [False, True])
def test_both_wire_candidates_derive_redundant_host_fields(reason_payload, alias_wire):
    reply, before, aliases = wire_reply(reason_payload, alias_wire)
    frozen = copy.deepcopy(before)
    projected = codec.project_format(before, reason_payload)
    validator = Draft202012Validator(projected["json_schema"]["schema"])
    Draft202012Validator.check_schema(projected["json_schema"]["schema"])
    validator.validate(reply)
    assert before == frozen
    # Both the native schema and unchanged host decoder preserve claim ownership.
    invalid = copy.deepcopy(reply)
    invalid["decisions"][0]["evidence_check"]["status"] = "supported"
    assert not validator.is_valid(invalid)
    invalid = copy.deepcopy(reply)
    invalid["decisions"][0]["evidence_check"]["aspect_checks"]["predicate"][
        "support_ids"
    ] = ["foreign-passage"]
    assert not validator.is_valid(invalid)
    canonical_reply = batched.decode_assessment_response(
        json.dumps(reply), reason_payload
    )
    assert len(canonical_reply.decisions) == len(reply["decisions"])


@pytest.mark.parametrize("alias_wire", [False, True])
def test_actual_sender_projects_both_candidates_before_measurement(
    reason_payload, monkeypatch, alias_wire
):
    before = copy.deepcopy(reason_payload)
    measured = []

    def reservation(messages, output, response_format):
        wire = json.loads(messages[1]["content"])
        measured.append((wire, copy.deepcopy(response_format)))
        return int(("assessment_input_contract" in wire) is not alias_wire)

    monkeypatch.setattr(scoped, "strict_request_reservation", reservation)
    messages, native = scoped.build_evaluation_request(
        batched.batch_prompt(reason_payload), reason_payload
    )
    wire = json.loads(messages[1]["content"])
    assert ("assessment_input_contract" in wire) is alias_wire
    assert wire["assessment_decision_contract"] == codec.CONTRACT
    assert wire["original_supports"] == reason_payload["original_supports"]
    assert len(measured) == 2
    for _, candidate in measured:
        for branch in _decision_schemas(candidate["json_schema"]["schema"]):
            group = branch["properties"]["evidence_check"]
            while "$ref" in group:
                group = candidate["json_schema"]["schema"]["$defs"][
                    group["$ref"].rsplit("/", 1)[-1]
                ]
            assert set(group["properties"]) == {
                "reason_code",
                "aspect_checks",
            }
    assert codec.INSTRUCTION.strip() in messages[0]["content"]
    assert reason_payload == before


@pytest.mark.parametrize("reason,status", list(codec.REASON_TO_STATUS.items()))
def test_one_code_derives_only_the_compatible_host_status(
    reason_payload, reason, status
):
    reply, _, _ = wire_reply(reason_payload)
    reply["decisions"][0]["evidence_check"]["reason_code"] = reason
    raw = json.dumps(reply)
    result = batched.decode_assessment_response(raw, reason_payload)
    assert result.decisions[0].status == status
    assert result.decisions[0].reason_code == reason
    scoped.validate_decision_status(result.decisions[0])
    assert json.loads(raw) == reply


def test_derived_support_does_not_override_report_negatives(reason_payload):
    reply, _, _ = wire_reply(reason_payload)
    first = reply["decisions"][0]
    first["report_check"].update(atomic=False, source_faithful=False)
    first["report_check"]["correct_kind"] = "report_structure"
    first["evidence_check"]["reason_code"] = "supported"
    decision = batched.decode_assessment_response(
        json.dumps(reply), reason_payload
    ).decisions[0]
    assert decision.status == "supported"
    assert not decision.atomic and not decision.source_faithful
    assert decision.classification_valid is False


@pytest.mark.parametrize(
    "fault", ["mixed_status", "unknown", "missing", "type", "extra"]
)
def test_invalid_or_mixed_output_is_not_repaired(reason_payload, fault):
    reply, _, _ = wire_reply(reason_payload)
    evidence = reply["decisions"][0]["evidence_check"]
    if fault == "mixed_status":
        evidence["status"] = "unsupported"
    elif fault == "unknown":
        evidence["reason_code"] = "probably_supported"
    elif fault == "missing":
        del evidence["reason_code"]
    elif fault == "type":
        evidence["reason_code"] = True
    else:
        evidence["explanation"] = "An undeclared field."
    with pytest.raises(ValueError, match="assessment_decision_response_invalid"):
        batched.decode_assessment_response(json.dumps(reply), reason_payload)


@pytest.mark.parametrize("raw", ['{"units":[],"units":[]}', '{"units":NaN}', "{"])
def test_non_json_or_duplicate_fields_never_gain_a_verdict(reason_payload, raw):
    with pytest.raises(ValueError, match="assessment_decision_response_invalid"):
        batched.decode_assessment_response(raw, reason_payload)


def test_real_059_pairs_remain_rejected_without_reclassification():
    path = Path(__file__).parent / "fixtures/run059-decision-pairs.json"
    raw = path.read_bytes()
    fixture = json.loads(raw)
    assert fixture["scientific_gold_standard"] is False
    assert len(fixture["decisions"]) == 3
    for row in fixture["decisions"]:
        with pytest.raises(ValueError, match="evaluation_decision_status_invalid"):
            scoped.validate_decision_status(SimpleNamespace(**row))
    assert path.read_bytes() == raw


def test_real_059_archive_and_result_contract_still_read_without_migration():
    from litagent.eval.results import EvaluationRunResult

    path = Path("artifacts/runs/frozen-writer-single-20261104-059-few-shot-metric.json")
    if not path.exists():
        pytest.skip("Private real-run artifact is not distributed")
    raw = path.read_bytes()
    report = json.loads(raw)["report"]
    details = report["evaluation"]["faithfulness"]["details"]
    run = EvaluationRunResult.from_details(
        details, hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    assert run.metrics()["faithfulness"]["score"] is None
    assert run.metrics()["internal_consistency"]["score"] is None
    assert path.read_bytes() == raw
