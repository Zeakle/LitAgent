"""Align native aspect coverage with strict host ownership and historical failures."""

import copy
import itertools
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_claim_aspect_contract import aspect_payload  # noqa: F401
from test_claim_response_contract import permissions  # noqa: F401
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_reason_only_decisions import reason_payload, wire_reply  # noqa: F401

from litagent.eval import batched
from litagent.eval import decision_output as codec
from litagent.eval import scoped
from litagent.eval.dimensions import _decision_schemas
from litagent.eval.results import contract_identity as result_identity
from litagent.safety.planned_accounting import _envelope
from litagent.safety.planned_budget import validate_writer_envelope


def selected_format(payload, alias_wire=False):
    """Build the actual permission-bound projection in either sender namespace."""
    reply, original, aliases = wire_reply(payload, alias_wire)
    native = codec.project_format(original, payload)
    validator = Draft202012Validator(native["json_schema"]["schema"])
    validator.check_schema(validator.schema)
    return reply, validator, aliases


@pytest.mark.parametrize("alias_wire", [False, True])
@pytest.mark.parametrize(
    "required",
    [
        list(names)
        for count in range(5)
        for names in itertools.combinations(
            ("predicate", "conditions", "quantities", "causal_relation"), count
        )
    ],
)
def test_every_required_set_has_exact_native_keys_and_host_order(
    reason_payload, required, alias_wire
):
    payload = copy.deepcopy(reason_payload)
    claim = payload["assertions"][0]
    claim["required_support_aspects"] = required
    if not required:
        claim.update(kind="report_structure", evidence=[])
    payload["schema"] = batched.assessment_schema(payload)
    reply, validator, _ = selected_format(payload, alias_wire)
    checks = reply["decisions"][0]["evidence_check"]["aspect_checks"]
    assert set(checks) == set(required)
    validator.validate(reply)
    reply["decisions"][0]["evidence_check"]["aspect_checks"] = dict(
        reversed(list(checks.items()))
    )
    result = batched.decode_assessment_response(json.dumps(reply), payload)
    assert [check.aspect for check in result.decisions[0].aspect_checks] == required


def test_065_bad_array_stays_invalid_without_deduplication(reason_payload):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/run065-aspect-coverage.json").read_text()
    )
    assert fixture["scientific_gold_standard"] is False
    assert fixture["returned_aspects"] == ["predicate", "conditions", "conditions"]
    reply, validator, _ = selected_format(reason_payload)
    checks = reply["decisions"][0]["evidence_check"]["aspect_checks"]
    array = [{"aspect": name, **checks[name]} for name in fixture["returned_aspects"]]
    # The historical native enum admitted this shape; the historical host did not.
    old_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"aspect": {"enum": fixture["required_aspects"]}},
        },
    }
    Draft202012Validator(old_schema).validate(array)
    with pytest.raises(scoped.ScopedEvaluationError, match="support_selection_invalid"):
        codec._derive_basis(array, reason_payload["assertions"][0])
    reply["decisions"][0]["evidence_check"]["aspect_checks"] = array
    assert not validator.is_valid(reply)
    with pytest.raises(scoped.ScopedEvaluationError, match="support_selection_invalid"):
        batched.decode_assessment_response(json.dumps(reply), reason_payload)


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "list", "string_bool", "extra_value", "wrong_ids"]
)
def test_native_and_host_reject_structural_faults(reason_payload, fault):
    reply, validator, _ = selected_format(reason_payload)
    checks = reply["decisions"][0]["evidence_check"]["aspect_checks"]
    if fault == "missing":
        checks.pop("conditions")
    elif fault == "extra":
        checks["unknown"] = copy.deepcopy(checks["predicate"])
    elif fault == "list":
        checks["predicate"] = []
    elif fault == "string_bool":
        checks["predicate"]["supported"] = "true"
    elif fault == "extra_value":
        checks["predicate"]["aspect"] = "predicate"
    else:
        checks["predicate"]["support_ids"] = "source"
    assert not validator.is_valid(reply)
    with pytest.raises((ValueError, scoped.ScopedEvaluationError)):
        batched.decode_assessment_response(json.dumps(reply), reason_payload)


def test_duplicate_json_aspect_key_is_not_silently_overwritten(reason_payload):
    reply, _, _ = selected_format(reason_payload)
    raw = json.dumps(reply).replace(
        '"predicate": {', '"predicate": {}, "predicate": {', 1
    )
    with pytest.raises(ValueError, match="assessment_decision_response_invalid"):
        batched.decode_assessment_response(raw, reason_payload)


@pytest.mark.parametrize(
    "version", ["reason-only-decision-v1", "reason-only-decision-v2"]
)
def test_old_owners_are_read_only_and_cannot_select_new_live_wire(
    reason_payload, version
):
    owner = copy.deepcopy(reason_payload["writer_envelope"])
    owner[codec.FIELDS[0]] = version
    from litagent.quality import digest

    owner["identity"] = digest({k: v for k, v in owner.items() if k != "identity"})
    _envelope(owner)
    with pytest.raises(ValueError, match="assessment_decision_contract_invalid"):
        validate_writer_envelope(owner)


def test_schema_compression_preserves_literal_and_property_names():
    schema = {
        "type": "object",
        "title": "annotation",
        "properties": {
            "description": {
                "type": "string",
                "description": "annotation",
                "enum": [{"title": "literal", "description": "literal"}],
            },
            "title": {"const": {"description": "literal"}},
        },
        "required": ["description", "title"],
    }
    result = codec._without_schema_annotations(schema)
    assert "title" not in result
    assert set(result["properties"]) == {"description", "title"}
    assert result["properties"]["description"]["enum"] == [
        {"title": "literal", "description": "literal"}
    ]
    assert result["properties"]["title"]["const"] == {"description": "literal"}


def test_final_native_permissions_cannot_disagree_with_host_claim(reason_payload):
    _, original, _ = wire_reply(reason_payload)
    native = copy.deepcopy(original)
    schema = native["json_schema"]["schema"]
    branch = next(iter(_decision_schemas(schema)))
    group = codec._resolve(schema, branch["properties"]["evidence_check"])
    checks = codec._resolve(schema, group["properties"]["aspect_checks"])
    item = codec._resolve(schema, checks["items"])
    codec._resolve(schema, item["properties"]["aspect"])["enum"] = ["predicate"]
    with pytest.raises(ValueError, match="assessment_decision_format_invalid"):
        codec.project_format(native, reason_payload)


def test_wire_revision_does_not_change_scientific_result_identity(monkeypatch):
    baseline = result_identity()
    monkeypatch.setattr(codec, "INSTRUCTION", codec.INSTRUCTION + "\nWire-only test.")
    result_identity.cache_clear()
    assert result_identity() == baseline
