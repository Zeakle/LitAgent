"""Use one model verdict code while retaining canonical host decision fields."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from functools import lru_cache

from litagent.eval.scoped import DECISION_STATUS_REASONS

CONTRACT = "reason-only-decision-v3"
FIELDS = ("assessment_decision_contract", "assessment_decision_identity")
REASON_TO_STATUS = {
    reason: status
    for status, reasons in DECISION_STATUS_REASONS.items()
    for reason in reasons
}
INSTRUCTION = (
    "\nEVIDENCE VERDICT: evidence_check contains only reason_code and aspect_checks. "
    "aspect_checks is an object with exactly this claim's required_support_aspects "
    "as keys. Each value contains only supported, support_ids and explanation; "
    "do not output an aspect field. Each selects only this claim's "
    "allowed_support_ids. Within each aspect, support_ids must contain no duplicates; "
    "the same source may be used in different aspects. Use at "
    "most 16 distinct IDs across its aspects. True requires a nonempty selection; "
    "false may retain partial evidence or select none. The host derives status "
    "from reason_code and the total basis from these selections. Do not output "
    "status or support_basis. Neither derivation grants scientific support or a "
    "report-check exemption.\n"
)
_PROMPT_REPLACEMENTS = (
    ("using selected basis IDs", "using this claim's allowed_support_ids"),
    ("using IDs from the selected basis", "using this claim's allowed_support_ids"),
    (
        "from this assertion's selected basis",
        "from this assertion's allowed_support_ids",
    ),
    (
        "from this claim's selected support_basis",
        "from this claim's allowed_support_ids",
    ),
)


@lru_cache(maxsize=1)
def contract_identity() -> str:
    """Bind the codec without changing historical result or dimension identities."""
    from litagent.eval.batched import (
        _decode_reason_only_assessment,
        limitations_check,
        restore_batch_support_ids,
    )
    from litagent.eval.global_input import input_identity
    from litagent.eval.global_permissions import permission_identity
    from litagent.eval.scoped import support_diagnostics_identity

    value = [
        CONTRACT,
        FIELDS,
        REASON_TO_STATUS,
        INSTRUCTION,
        _PROMPT_REPLACEMENTS,
        permission_identity(),
        input_identity(),
        support_diagnostics_identity(),
        inspect.getsource(limitations_check),
        inspect.getsource(_decode_reason_only_assessment),
        inspect.getsource(restore_batch_support_ids),
        *[
            inspect.getsource(function)
            for function in (
                selected,
                _resolve,
                _without_schema_annotations,
                _keyed_aspects,
                project_format,
                project_prompt,
                _selection_error,
                _derive_basis,
                validate_selections,
                _response_error,
                _unique_object,
                _invalid_constant,
                decode_response,
            )
        ],
    ]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def fields() -> dict:
    """Produce an internal pair only for a newly frozen owner."""
    return {FIELDS[0]: CONTRACT, FIELDS[1]: contract_identity()}


def selected(envelope) -> bool:
    """Require an explicit authenticated pair and both existing parent policies."""
    if not isinstance(envelope, dict) or not any(k in envelope for k in FIELDS):
        return False
    from litagent.eval.dimensions import selected_dimensions
    from litagent.eval.results import selected as independent_results

    if (
        envelope.get(FIELDS[0]) != CONTRACT
        or envelope.get(FIELDS[1]) != contract_identity()
        or not independent_results(envelope)
        or not selected_dimensions(envelope)
    ):
        raise ValueError("assessment_decision_contract_invalid")
    return True


def _resolve(schema, node):
    """Follow only local definitions in the final permission-bound schema."""
    seen = set()
    while isinstance(node, dict) and "$ref" in node:
        ref = node["$ref"]
        if (
            set(node) != {"$ref"}
            or not isinstance(ref, str)
            or not ref.startswith("#/$defs/")
            or ref in seen
        ):
            raise ValueError("assessment_decision_format_invalid")
        seen.add(ref)
        name = ref.removeprefix("#/$defs/")
        if (
            not name
            or "/" in name
            or "~" in name
            or name not in schema.get("$defs", {})
        ):
            raise ValueError("assessment_decision_format_invalid")
        node = schema["$defs"][name]
    if not isinstance(node, dict):
        raise ValueError("assessment_decision_format_invalid")
    return node


def _without_schema_annotations(node):
    """Remove annotations without interpreting property names or literal data."""
    if isinstance(node, list):
        return [_without_schema_annotations(value) for value in node]
    if not isinstance(node, dict):
        return node
    result = {}
    for key, value in node.items():
        if key in {"title", "description"}:
            continue
        if key in {"properties", "$defs", "patternProperties", "dependentSchemas"}:
            result[key] = {
                name: _without_schema_annotations(child)
                for name, child in value.items()
            }
        elif key in {"const", "enum", "default", "examples"}:
            result[key] = copy.deepcopy(value)
        else:
            result[key] = _without_schema_annotations(value)
    return result


def _keyed_aspects(schema, checks, pool):
    """Bind exact aspect keys while sharing only identical value permissions."""
    item = _resolve(schema, checks.get("items"))
    expected = {"aspect", "supported", "support_ids", "explanation"}
    if (
        checks.get("type") != "array"
        or set(item.get("properties", {})) != expected
        or set(item.get("required", [])) != expected
        or item.get("additionalProperties") is not False
    ):
        raise ValueError("assessment_decision_format_invalid")
    names = (
        []
        if checks.get("maxItems") == 0
        else _resolve(schema, item["properties"]["aspect"]).get("enum")
    )
    if (
        type(names) is not list
        or any(type(name) is not str for name in names)
        or len(names) != len(set(names))
    ):
        raise ValueError("assessment_decision_format_invalid")
    values = _without_schema_annotations(item)
    del values["properties"]["aspect"]
    values["required"] = [name for name in values["required"] if name != "aspect"]
    properties = {}
    if names:
        key = json.dumps(values, ensure_ascii=False, sort_keys=True)
        if key not in pool:
            index = len(pool)
            while f"KeyedAspectValue{index}" in schema["$defs"]:
                index += 1
            pool[key] = f"KeyedAspectValue{index}"
            schema["$defs"][pool[key]] = values
        properties = {name: {"$ref": f"#/$defs/{pool[key]}"} for name in names}
    return {
        "type": "object",
        "properties": properties,
        "required": list(names),
        "additionalProperties": False,
    }


def project_format(response_format, payload) -> dict:
    """Emit aspect selections once after all source permissions are bound."""
    if type(payload) is not dict:
        raise ValueError("assessment_decision_format_invalid")
    if not selected(payload.get("writer_envelope")):
        return response_format
    from litagent.eval.batched import _prune_response_definitions
    from litagent.eval.dimensions import _decision_schemas

    if (
        payload.get("assessment_output_contract") != "host-batch-assessment-v1"
        or not isinstance(response_format, dict)
        or not isinstance(response_format.get("json_schema"), dict)
        or response_format["json_schema"].get("name") != "host_batch_assessment_v1"
        or not isinstance(response_format["json_schema"].get("schema"), dict)
    ):
        raise ValueError("assessment_decision_format_invalid")
    result = copy.deepcopy(response_format)
    schema = result["json_schema"]["schema"]
    seen, pool, covered = {}, {}, set()
    claims = {claim["claim_id"]: claim for claim in payload["assertions"]}
    for branch in _decision_schemas(schema):
        group = _resolve(schema, branch.get("properties", {}).get("evidence_check"))
        if id(group) in seen:
            names = seen[id(group)]
            ids = _resolve(schema, branch["properties"]["claim_id"])["enum"]
            if any(
                cid not in claims
                or set(names) != set(claims[cid]["required_support_aspects"])
                for cid in ids
            ):
                raise ValueError("assessment_decision_format_invalid")
            covered.update(ids)
            continue
        expected = {"status", "reason_code", "support_basis", "aspect_checks"}
        if (
            not isinstance(group.get("properties"), dict)
            or not isinstance(group.get("required"), list)
            or set(group["properties"]) != expected
            or set(group.get("required", [])) != expected
            or group.get("additionalProperties") is not False
        ):
            raise ValueError("assessment_decision_format_invalid")
        reason = _resolve(schema, group["properties"]["reason_code"])
        if not isinstance(reason.get("enum"), list) or reason["enum"] != list(
            REASON_TO_STATUS
        ):
            raise ValueError("assessment_decision_format_invalid")
        reason["description"] = (
            "Single verdict code. supported requires complete actual support; "
            "unverifiable means unresolved. The host derives status."
        )
        del group["properties"]["status"]
        del group["properties"]["support_basis"]
        group["required"] = [
            k for k in group["required"] if k not in {"status", "support_basis"}
        ]
        checks = _resolve(schema, group["properties"]["aspect_checks"])
        keyed = _keyed_aspects(schema, checks, pool)
        names = keyed["required"]
        ids = _resolve(schema, branch["properties"]["claim_id"])["enum"]
        if any(
            cid not in claims
            or set(names) != set(claims[cid]["required_support_aspects"])
            for cid in ids
        ):
            raise ValueError("assessment_decision_format_invalid")
        covered.update(ids)
        seen[id(group)] = names
        group["properties"]["aspect_checks"] = keyed
    if covered != set(claims):
        raise ValueError("assessment_decision_format_invalid")
    _prune_response_definitions(schema)
    result["json_schema"]["schema"] = _without_schema_annotations(schema)
    return result


def project_prompt(prompt: str, payload: dict) -> str:
    """Remove redundant selection instructions only on the new frozen wire."""
    if not selected(payload.get("writer_envelope")):
        return prompt
    for original, replacement in _PROMPT_REPLACEMENTS:
        prompt = prompt.replace(original, replacement)
    return prompt + INSTRUCTION


def _selection_error(claim, subreason):
    """Expose structural source faults before they reach scientific repair."""
    from litagent.eval.scoped import ScopedEvaluationError

    raise ScopedEvaluationError(
        "evaluation_assessment_support_selection_invalid",
        {
            "claim_id": claim.get("claim_id"),
            "source_unit_id": claim.get("source_unit_id"),
            "source_fragment_id": claim.get("source_fragment_id"),
            "subreason": subreason,
        },
    )


def _derive_basis(aspects, claim) -> list[dict]:
    """Union only explicit per-aspect selections in the host's required order."""
    required = claim["required_support_aspects"]
    if type(aspects) is not list:
        _selection_error(claim, "aspect_type")
    by_name = {}
    for check in aspects:
        if type(check) is not dict or set(check) != {
            "aspect",
            "supported",
            "support_ids",
            "explanation",
        }:
            _selection_error(claim, "aspect_fields")
        name, ids = check["aspect"], check["support_ids"]
        if type(name) is not str or name not in required or name in by_name:
            _selection_error(claim, "aspect_coverage")
        if type(check["supported"]) is not bool or type(ids) is not list:
            _selection_error(claim, "aspect_type")
        if any(type(pid) is not str or not pid for pid in ids):
            _selection_error(claim, "support_identity")
        if len(ids) != len(set(ids)) or (check["supported"] and not ids):
            _selection_error(claim, "support_selection")
        by_name[name] = ids
    if set(by_name) != set(required):
        _selection_error(claim, "aspect_coverage")
    ids = list(dict.fromkeys(pid for name in required for pid in by_name[name]))
    if len(ids) > 16:
        _selection_error(claim, "support_capacity")
    return [{"support_id": pid} for pid in ids]


def validate_selections(result, payload) -> None:
    """Check canonical ownership after the existing alias and type consumers."""
    if not selected(payload.get("writer_envelope")):
        return
    claims = {claim["claim_id"]: claim for claim in payload["assertions"]}
    for decision in result.decisions:
        claim = claims[decision.claim_id]
        expected = _derive_basis(
            [a.model_dump() for a in decision.aspect_checks], claim
        )
        if [b.model_dump() for b in decision.support_basis] != expected:
            _selection_error(claim, "basis_derivation")
        aliases = {e["evidence_alias"] for e in claim["evidence"]}
        for basis in expected:
            passage = payload["support_passages"].get(basis["support_id"])
            if (
                claim["kind"] != "scientific"
                or not passage
                or passage[0] not in aliases
                or passage[1] not in payload["original_supports"]
            ):
                _selection_error(claim, "support_ownership")


def _response_error():
    """Reject mixed revisions and malformed values instead of repairing them."""
    raise ValueError("assessment_decision_response_invalid")


def _unique_object(pairs):
    """Keep a duplicate key from replacing a provider's original decision."""
    value = {}
    for key, item in pairs:
        if key in value:
            _response_error()
        value[key] = item
    return value


def _invalid_constant(value):
    """Reject non-JSON numbers before deriving any host field."""
    _response_error()


def decode_response(content, payload) -> str:
    """Derive redundant host fields without changing any scientific judgment."""
    if type(payload) is not dict:
        _response_error()
    if not selected(payload.get("writer_envelope")):
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
        raise ValueError("assessment_decision_response_invalid") from exc
    if (
        type(value) is not dict
        or set(value) != {"units", "decisions", "conflicts"}
        or type(value["decisions"]) is not list
    ):
        _response_error()
    claims = {claim["claim_id"]: claim for claim in payload["assertions"]}
    seen = set()
    for decision in value["decisions"]:
        if type(decision) is not dict or set(decision) != {
            "claim_id",
            "report_check",
            "evidence_check",
        }:
            _response_error()
        evidence = decision["evidence_check"]
        if type(evidence) is not dict or set(evidence) != {
            "reason_code",
            "aspect_checks",
        }:
            _response_error()
        cid = decision["claim_id"]
        if type(cid) is not str or cid not in claims or cid in seen:
            _response_error()
        seen.add(cid)
        checks = evidence["aspect_checks"]
        required = claims[cid]["required_support_aspects"]
        if type(checks) is not dict or set(checks) != set(required):
            _selection_error(claims[cid], "aspect_coverage")
        for check in checks.values():
            if (
                type(check) is not dict
                or set(check) != {"supported", "support_ids", "explanation"}
                or type(check["supported"]) is not bool
                or type(check["support_ids"]) is not list
                or type(check["explanation"]) is not str
            ):
                _selection_error(claims[cid], "aspect_fields")
        # Ordering and names belong to the frozen host claim, not model prose.
        evidence["aspect_checks"] = [
            {"aspect": name, **checks[name]} for name in required
        ]
        reason = evidence["reason_code"]
        if type(reason) is not str or reason not in REASON_TO_STATUS:
            _response_error()
        # This is a declared representation, not a correction of a raw verdict.
        # Never infer support from report kind, atomicity, or evidence identifiers.
        evidence["status"] = REASON_TO_STATUS[reason]
        evidence["support_basis"] = _derive_basis(
            evidence["aspect_checks"], claims[cid]
        )
    if seen != set(claims):
        _response_error()
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
