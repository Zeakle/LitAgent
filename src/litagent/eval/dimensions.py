"""Separate report checks from evidence checks without changing their verdicts."""

import copy
import hashlib
import inspect
import json

CONTRACT = "separated-evaluation-dimensions-v1"
FIELDS = ("evaluation_dimensions_contract", "evaluation_dimensions_identity")
_PARENTS = {
    "contract": "flat-writer-envelope-v1",
    "global_findings_contract": "anchored-global-findings-v2",
    "summary_constraints_contract": "effective-summary-limits-v1",
    "repair_feedback_contract": "host-repair-feedback-v1",
    "reference_contract": "ordered-reference-set-v1",
    "transport_revision": "host-lossless-transport-v1",
    "support_validation_revision": "host-support-receipts-v2",
    "inventory_recovery_contract": "bounded-inventory-recovery-v2",
    "classification_review_contract": "independent-assertion-kinds-v1",
    "evaluation_response_contract": "host-bound-evaluation-v2",
}
_REPORT_FIELDS = ("correct_kind", "atomic", "source_faithful", "explanation")
_EVIDENCE_FIELDS = ("status", "reason_code", "support_basis", "aspect_checks")
_KINDS = ("scientific", "runtime_method", "report_structure")
_STATUSES = ("supported", "unsupported", "unverifiable")
_REASONS = (
    "supported",
    "insufficient_support",
    "uncited_claim",
    "attribution_mismatch",
    "record_mismatch",
    "unverifiable",
)
_ASPECTS = ("predicate", "conditions", "quantities", "causal_relation")
DIMENSIONS_INSTRUCTION = """
SEPARATED EVALUATION DIMENSIONS:
Each decision contains claim_id, report_check and evidence_check exactly once.
report_check judges kind, atomicity and extraction fidelity against the owning
REPORT fragment and its same-unit qualifiers. Its explanation concerns those
report checks only. Paper support, missing citations and dispersed evidence do not
decide these flags. Keep every genuine report defect; do not force flags true.
evidence_check independently judges the listed assertion using only its own
allowed original passages. Keep scientific support reasons in the existing aspect
explanations. A faithful extraction may lack paper support; paper entailment cannot
repair an extraction that changed the report's entity, number, condition or scope.
For runtime_method use trusted records; for report_structure use the actual report.
Preserve those existing rules without adding scientific aspects or exemptions.
Neither group grants a passing verdict to the other. The host still requires all
report checks and complete scientific support before delivery.
"""


def dimensions_identity() -> str:
    """Bind the new representation and relation boundary independently of old IDs."""
    from litagent.eval.batched import _decode_assessment_dimensions
    from litagent.eval.relation_scopes import relation_scope_identity
    from litagent.eval.scoped import (
        _RELATION_SCOPES,
        _dimension_relation_issue,
        _protected_entities,
        _source_entity_forms,
    )

    value = [
        CONTRACT,
        FIELDS,
        _PARENTS,
        _REPORT_FIELDS,
        _EVIDENCE_FIELDS,
        _KINDS,
        _STATUSES,
        _REASONS,
        _ASPECTS,
        DIMENSIONS_INSTRUCTION,
        relation_scope_identity(),
        [
            (scope, pattern)
            for scope, pattern in _RELATION_SCOPES.items()
            if scope != "parameter_count"
        ],
        *[
            inspect.getsource(function)
            for function in (
                selected_dimensions,
                dimensions_fields,
                _format_error,
                _decision_schemas,
                _project_decision,
                _share_dimension_groups,
                project_dimensions_format,
                _response_error,
                _unique_object,
                _invalid_constant,
                _string,
                _object,
                _validate_rows,
                _flatten_decision,
                decode_dimensions_response,
                _decode_assessment_dimensions,
                _dimension_relation_issue,
                _source_entity_forms,
                _protected_entities,
            )
        ],
    ]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def selected_dimensions(envelope) -> bool:
    """Authenticate the opt-in pair after the caller authenticates its owner."""
    if envelope is None:
        return False
    if type(envelope) is not dict:
        raise ValueError("evaluation_dimensions_policy_invalid")
    if not any(key in envelope for key in FIELDS):
        return False
    # Owner validation checks the parent identities. Re-entering it here would
    # recurse through this selector during envelope construction and validation.
    if (
        type(envelope.get(FIELDS[0])) is not str
        or envelope[FIELDS[0]] != CONTRACT
        or type(envelope.get(FIELDS[1])) is not str
        or envelope[FIELDS[1]] != dimensions_identity()
        or any(envelope.get(key) != value for key, value in _PARENTS.items())
        or type(envelope.get("capacity")) is not dict
    ):
        raise ValueError("evaluation_dimensions_policy_invalid")
    return True


def dimensions_fields(envelope) -> dict:
    """Return only the authenticated optional ownership pair."""
    return (
        {key: envelope[key] for key in FIELDS} if selected_dimensions(envelope) else {}
    )


def _format_error():
    """Reject an unfamiliar schema without weakening its validation space."""
    raise ValueError("evaluation_dimensions_format_invalid")


def _decision_schemas(schema):
    """Resolve only the final local decision branches, including shared refs."""
    if type(schema) is not dict or type(schema.get("$defs")) is not dict:
        _format_error()
    try:
        item = schema["properties"]["decisions"]["items"]
    except (KeyError, TypeError):
        _format_error()
    definitions = schema["$defs"]
    result, visited = [], set()

    def visit(node, trail):
        if type(node) is not dict:
            _format_error()
        if "$ref" in node:
            ref = node["$ref"]
            if (
                set(node) != {"$ref"}
                or type(ref) is not str
                or not ref.startswith("#/$defs/")
                or ref in trail
            ):
                _format_error()
            name = ref.removeprefix("#/$defs/")
            if not name or "/" in name or "~" in name or name not in definitions:
                _format_error()
            visit(definitions[name], {*trail, ref})
        elif "anyOf" in node:
            if (
                set(node) != {"anyOf"}
                or type(node["anyOf"]) is not list
                or not node["anyOf"]
            ):
                _format_error()
            for branch in node["anyOf"]:
                visit(branch, trail)
        elif id(node) not in visited:
            visited.add(id(node))
            result.append(node)

    visit(item, set())
    if not result:
        _format_error()
    return result


def _project_decision(node):
    """Move each existing field exactly once and retain its original schema."""
    if (
        node.get("type") != "object"
        or node.get("additionalProperties") is not False
        or set(node)
        - {
            "type",
            "title",
            "description",
            "additionalProperties",
            "properties",
            "required",
        }
        or type(node.get("properties")) is not dict
        or type(node.get("required")) is not list
    ):
        _format_error()
    properties = node["properties"]
    expected = {"claim_id", *_REPORT_FIELDS, *_EVIDENCE_FIELDS}
    if "classification_valid" in properties:
        expected.add("classification_valid")
    required = node["required"]
    if (
        set(properties) != expected
        or any(type(field) is not str for field in required)
        or len(required) != len(set(required))
        or set(required) != expected
        or any(type(value) is not dict for value in properties.values())
    ):
        _format_error()
    report_fields = {*_REPORT_FIELDS, "classification_valid"}
    report = {key: value for key, value in properties.items() if key in report_fields}
    evidence = {
        key: value for key, value in properties.items() if key in _EVIDENCE_FIELDS
    }
    node["properties"] = {
        "claim_id": properties["claim_id"],
        "report_check": {
            "type": "object",
            "properties": report,
            "required": [key for key in required if key in report],
            "additionalProperties": False,
        },
        "evidence_check": {
            "type": "object",
            "properties": evidence,
            "required": [key for key in required if key in evidence],
            "additionalProperties": False,
        },
    }
    node["required"] = ["claim_id", "report_check", "evidence_check"]


def _share_dimension_groups(schema):
    """Share only identical complete group schemas when the wire becomes smaller."""
    groups = {}
    for branch in _decision_schemas(schema):
        for field in ("report_check", "evidence_check"):
            value = branch["properties"][field]
            key = json.dumps(value, ensure_ascii=False, sort_keys=True)
            groups.setdefault((field, key), []).append((branch, field, value))
    size = len(json.dumps(schema, ensure_ascii=False).encode())
    definitions = schema["$defs"]
    for occurrences in groups.values():
        if len(occurrences) < 2:
            continue
        number = 0
        while f"DimensionsGroup{number}" in definitions:
            number += 1
        name = f"DimensionsGroup{number}"
        definitions[name] = copy.deepcopy(occurrences[0][2])
        for branch, field, _ in occurrences:
            branch["properties"][field] = {"$ref": f"#/$defs/{name}"}
        proposed = len(json.dumps(schema, ensure_ascii=False).encode())
        if proposed < size:
            size = proposed
        else:
            # Restoring the original objects retains every permission and ref;
            # sharing is an encoding choice, never an input-capacity exemption.
            for branch, field, value in occurrences:
                branch["properties"][field] = value
            definitions.pop(name)


def project_dimensions_format(response_format, payload) -> dict:
    """Project both final native candidates after their original permission binders."""
    if type(payload) is not dict:
        _format_error()
    if not selected_dimensions(payload.get("writer_envelope")):
        return response_format
    if (
        payload.get("assessment_output_contract") != "host-batch-assessment-v1"
        or type(response_format) is not dict
        or response_format.get("type") != "json_schema"
        or type(response_format.get("json_schema")) is not dict
        or response_format["json_schema"].get("name") != "host_batch_assessment_v1"
        or response_format["json_schema"].get("strict") is not True
    ):
        _format_error()
    result = copy.deepcopy(response_format)
    schema = result["json_schema"].get("schema")
    for node in _decision_schemas(schema):
        _project_decision(node)
    _share_dimension_groups(schema)
    return result


def _response_error():
    """Reject malformed nested decisions without changing any original diagnostic."""
    raise ValueError("evaluation_dimensions_response_invalid")


def _unique_object(pairs):
    """Reject duplicate keys before a JSON object can overwrite a diagnostic."""
    result = {}
    for key, value in pairs:
        if key in result:
            _response_error()
        result[key] = value
    return result


def _invalid_constant(value):
    """Keep non-JSON numeric constants out of the canonical response."""
    _response_error()


def _string(value, *, nonempty=False):
    """Apply the existing strict string type without normalization."""
    if type(value) is not str or nonempty and not value:
        _response_error()


def _object(value, fields):
    """Reject absent, mixed or additional fields at an explicit wire boundary."""
    if type(value) is not dict or set(value) != set(fields):
        _response_error()


def _validate_rows(rows, kind):
    """Preserve the original strict unit, conflict, basis and aspect shapes."""
    if type(rows) is not list:
        _response_error()
    for row in rows:
        if kind == "units":
            _object(row, ("source_unit_id", "complete", "explanation"))
            _string(row["source_unit_id"])
            _string(row["explanation"], nonempty=True)
            if type(row["complete"]) is not bool:
                _response_error()
        elif kind == "conflicts":
            _object(row, ("claim_a", "claim_b", "reason"))
            _string(row["claim_a"])
            _string(row["claim_b"])
            _string(row["reason"], nonempty=True)
        elif kind == "support_basis":
            _object(row, ("support_id",))
            _string(row["support_id"])
        elif kind == "aspect_checks":
            _object(row, ("aspect", "supported", "support_ids", "explanation"))
            if row["aspect"] not in _ASPECTS or type(row["supported"]) is not bool:
                _response_error()
            if type(row["support_ids"]) is not list:
                _response_error()
            for support_id in row["support_ids"]:
                _string(support_id)
            _string(row["explanation"], nonempty=True)
        else:
            _response_error()


def _flatten_decision(decision):
    """Restore field ownership without deriving or repairing a scientific verdict."""
    _object(decision, ("claim_id", "report_check", "evidence_check"))
    _string(decision["claim_id"], nonempty=True)
    report = decision["report_check"]
    evidence = decision["evidence_check"]
    report_fields = set(_REPORT_FIELDS)
    if type(report) is dict and "classification_valid" in report:
        report_fields.add("classification_valid")
    _object(report, report_fields)
    _object(evidence, _EVIDENCE_FIELDS)
    if report["correct_kind"] not in _KINDS:
        _response_error()
    for key in ("atomic", "source_faithful", "classification_valid"):
        if key in report and type(report[key]) is not bool:
            _response_error()
    _string(report["explanation"], nonempty=True)
    if evidence["status"] not in _STATUSES or evidence["reason_code"] not in _REASONS:
        _response_error()
    _validate_rows(evidence["support_basis"], "support_basis")
    if len(evidence["support_basis"]) > 16:
        _response_error()
    _validate_rows(evidence["aspect_checks"], "aspect_checks")
    # The original decoder still authenticates IDs, derives kind agreement and
    # enforces the selected schema. No report flag is inferred from evidence.
    return {"claim_id": decision["claim_id"], **report, **evidence}


def decode_dimensions_response(content, payload) -> str:
    """Flatten selected nested JSON while the caller retains the untouched raw."""
    if type(payload) is not dict:
        _response_error()
    if not selected_dimensions(payload.get("writer_envelope")):
        return content
    if type(content) is not str:
        _response_error()
    try:
        value = json.loads(
            content,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (json.JSONDecodeError, TypeError, RecursionError) as exc:
        raise ValueError("evaluation_dimensions_response_invalid") from exc
    _object(value, ("units", "decisions", "conflicts"))
    _validate_rows(value["units"], "units")
    _validate_rows(value["conflicts"], "conflicts")
    if type(value["decisions"]) is not list:
        _response_error()
    canonical = {
        **value,
        "decisions": [_flatten_decision(d) for d in value["decisions"]],
    }
    return json.dumps(canonical, ensure_ascii=False, allow_nan=False)
