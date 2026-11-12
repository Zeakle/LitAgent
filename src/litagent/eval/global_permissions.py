"""Project existing global finding permissions into the actual native schema."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json

ANCHOR_PERMISSIONS = {
    "missing_assertion": ("unit",),
    "contradiction": ("unit", "claim"),
    "category_count": ("unit", "claim"),
    "repetition": ("unit", "claim"),
    "analysis": ("unit", "claim"),
    "core_coverage": ("unit", "claim"),
}
REFERENCE_INSTRUCTION = (
    "Select the minimum sufficient scientific claim IDs for each coverage, "
    "limitations or analysis judgment, covering every requested dimension and "
    "chapter duty. Do not copy the full inventory by default; this never reduces "
    "full-report checking."
)


def permission_identity() -> str:
    """Freeze request constraints without changing historical result identities."""
    from litagent.eval.batched import normalize_global_findings
    from litagent.writing_flat import global_identity_maps

    value = [
        ANCHOR_PERMISSIONS,
        REFERENCE_INSTRUCTION,
        *[
            inspect.getsource(function)
            for function in (
                project_format,
                project_prompt,
                _reference_catalogs,
                effective_permissions,
                _invalid_schema,
                normalize_global_findings,
                global_identity_maps,
            )
        ],
    ]
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def effective_permissions(payload):
    """Keep the historical catalog while selecting one owner for new global checks."""
    from litagent.eval.responsibilities import FORBIDDEN_CODES
    from litagent.eval.results import selected

    return {
        code: scopes
        for code, scopes in ANCHOR_PERMISSIONS.items()
        if not selected(payload.get("writer_envelope")) or code not in FORBIDDEN_CODES
    }


def _invalid_schema():
    """Reject an unexpected producer before loosening any permission."""
    raise ValueError("evaluation_global_permission_schema_invalid")


def _reference_catalogs(payload: dict) -> dict:
    """Derive exact output permissions from host input, never from model answers."""
    from litagent.body_quality import analysis_claim_ids_by_section
    from litagent.writing_flat import global_identity_maps

    try:
        sources, assertions = payload["source_units"], payload["assertions"]
        requirements = payload.get("coverage_requirements", [])
        if any(type(rows) is not list for rows in (sources, assertions, requirements)):
            _invalid_schema()
        for rows, field in (
            (sources, "source_unit_id"),
            (assertions, "claim_id"),
            (requirements, "requirement_id"),
        ):
            ids = [row[field] for row in rows]
            if any(type(v) is not str or not v.strip() for v in ids) or len(ids) != len(
                set(ids)
            ):
                _invalid_schema()
        units, claims = global_identity_maps(payload)
        if not units or any(
            c["kind"] not in ("scientific", "runtime_method", "report_structure")
            or c["source_unit_id"] not in units
            for c in assertions
        ):
            _invalid_schema()
        structure = payload["report_structure"]
        if type(structure) is not list:
            _invalid_schema()
        structure_ids = [row["source_unit_id"] for row in structure]
        if (
            any(type(uid) is not str for uid in structure_ids)
            or len(structure_ids) != len(set(structure_ids))
            or set(structure_ids) != set(units)
        ):
            _invalid_schema()
        grouped = analysis_claim_ids_by_section(assertions, structure)
        if grouped != payload["analysis_claim_ids_by_section"]:
            _invalid_schema()
        science = [
            claims[c["claim_id"]] for c in assertions if c["kind"] == "scientific"
        ]
        return {
            "u": list(units.values()),
            "c": list(claims.values()),
            "s": science,
            "n": [
                claims[c["claim_id"]] for c in assertions if c["kind"] != "scientific"
            ],
            "r": [r["requirement_id"] for r in requirements],
            **{
                "chapter" + str(i): [claims[cid] for cid in ids]
                for i, ids in enumerate(grouped.values())
            },
            "limits": [claims[cid] for cid in grouped["open_problems"]],
        }
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise ValueError("evaluation_global_permission_schema_invalid") from exc


def project_prompt(prompt: str, payload: dict) -> str:
    """Request sufficient witnesses without changing the full scientific denominator."""
    from litagent.eval.decision_output import selected

    if not selected(payload.get("writer_envelope")):
        return prompt
    if type(prompt) is not str:
        _invalid_schema()
    return prompt + "\n" + REFERENCE_INSTRUCTION + "\n"


def project_format(response_format, payload) -> dict:
    """Bind all global references while retaining existing chapter permissions."""
    from litagent.eval.decision_output import _resolve, selected

    if type(payload) is not dict:
        _invalid_schema()
    if not selected(payload.get("writer_envelope")):
        return response_format
    if (
        payload.get("global_output_contract") != "host-global-completeness-v1"
        or payload.get("global_findings_contract") != "anchored-global-findings-v2"
        or payload.get("evaluation_input_contract") != "flat-evaluation-input-v1"
        or type(response_format) is not dict
        or type(response_format.get("json_schema")) is not dict
        or response_format["json_schema"].get("name") != "host_global_completeness_v1"
        or type(response_format["json_schema"].get("schema")) is not dict
    ):
        _invalid_schema()

    from litagent.body_quality import ROLES
    from litagent.eval.batched import _prune_response_definitions

    catalogs = _reference_catalogs(payload)
    result = copy.deepcopy(response_format)
    schema = result["json_schema"]["schema"]
    root_properties = schema.get("properties")
    if type(root_properties) is not dict:
        _invalid_schema()
    issues = root_properties.get("issues")
    if type(issues) is not dict:
        _invalid_schema()
    try:
        issue = _resolve(schema, issues.get("items"))
    except ValueError as exc:
        raise ValueError("evaluation_global_permission_schema_invalid") from exc
    properties = issue.get("properties")
    if (
        type(properties) is not dict
        or type(properties.get("code")) is not dict
        or type(properties.get("anchor_ids")) is not dict
    ):
        _invalid_schema()
    codes = properties["code"].get("enum")
    permissions = effective_permissions(payload)
    anchors = properties["anchor_ids"]
    expected = {"code", "anchor_ids", "explanation"}
    if (
        issues.get("type") != "array"
        or set(properties) != expected
        or type(issue.get("required")) is not list
        or set(issue["required"]) != expected
        or issue.get("additionalProperties") is not False
        or type(codes) is not list
        or any(type(code) is not str for code in codes)
        or len(codes) != len(set(codes))
        or set(codes) != set(permissions)
        or anchors.get("type") != "array"
        or anchors.get("items") != {"type": "string"}
        or type(anchors.get("minItems", 0)) is not int
    ):
        _invalid_schema()

    pool = {}

    def reference(key):
        values = tuple(catalogs[key])
        if not values:
            return {"type": "string"}
        if values not in pool:
            name = "_" + key
            if name in schema.setdefault("$defs", {}):
                _invalid_schema()
            # Scientific and non-scientific IDs form an exact disjoint union.
            # Sharing it avoids a second nearly identical large enum for issues.
            definition = (
                {"anyOf": [reference("s"), reference("n")]}
                if key == "c" and catalogs["s"] and catalogs["n"]
                else {"type": "string", "enum": list(values)}
            )
            schema["$defs"][name] = definition
            pool[values] = name
        return {"$ref": "#/$defs/" + pool[values]}

    def properties(region, *, array=True):
        try:
            node = schema["properties"][region]
            if array:
                if node.get("type") != "array":
                    _invalid_schema()
                node = node["items"]
            result = _resolve(schema, node)["properties"]
            if type(result) is not dict:
                _invalid_schema()
            return result
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("evaluation_global_permission_schema_invalid") from exc

    def bind_scalar(props, field, key):
        original = props[field]
        if (
            type(original) is not dict
            or original.get("type") != "string"
            or "enum" in original
            and original["enum"] != catalogs[key]
        ):
            _invalid_schema()
        props[field] = {
            **{k: v for k, v in original.items() if k not in {"type", "enum"}},
            **reference(key),
        }

    def bind_array(props, field, key):
        node = props[field]
        if type(node) is not dict or node.get("type") != "array":
            _invalid_schema()
        bind_scalar(node, "items", key)
        if not catalogs[key]:
            node["maxItems"] = 0

    try:
        bind_scalar(properties("units"), "source_unit_id", "u")
        for field in ("claim_a", "claim_b"):
            bind_scalar(properties("conflicts"), field, "s")
        if not catalogs["s"]:
            schema["properties"]["conflicts"]["maxItems"] = 0
        bind_scalar(properties("coverage"), "requirement_id", "r")
        bind_array(properties("coverage"), "claim_ids", "s")
        if not catalogs["r"]:
            schema["properties"]["coverage"]["maxItems"] = 0
        bind_array(properties("limitations", array=False), "claim_ids", "limits")
        branches = schema["properties"]["analyses"]["items"]["anyOf"]
        if type(branches) is not list or len(branches) != len(ROLES):
            _invalid_schema()
        for i, (branch, section) in enumerate(zip(branches, ROLES, strict=True)):
            props = branch["properties"]
            if props["section_key"] != {"type": "string", "const": section}:
                _invalid_schema()
            bind_array(props, "claim_ids", "chapter" + str(i))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("evaluation_global_permission_schema_invalid") from exc

    grouped = {}
    for code in codes:
        grouped.setdefault(permissions[code], []).append(code)
    branches = []
    for scopes, allowed_codes in grouped.items():
        branch = copy.deepcopy(issue)
        branch["properties"]["code"]["enum"] = allowed_codes
        branch_anchors = branch["properties"]["anchor_ids"]
        branch_anchors["items"] = (
            {"anyOf": [reference("u"), reference("c")]}
            if "claim" in scopes and catalogs["c"]
            else reference("u")
        )
        branch_anchors["minItems"] = max(1, anchors.get("minItems", 0))
        branches.append(branch)
    # Only the output permission changes. Original prose, claims, and their
    # canonical ownership stay untouched and still face the existing decoder.
    issues["items"] = {"anyOf": branches}

    def strip_annotations(node):
        if isinstance(node, dict):
            node.pop("title", None)
            node.pop("description", None)
            for child in node.values():
                strip_annotations(child)
        elif isinstance(node, list):
            for child in node:
                strip_annotations(child)

    strip_annotations(schema)
    _prune_response_definitions(schema)
    return result
