"""Adapt bounded writer collections without changing canonical scientific values."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

CONTRACT = "required-head-wire-v1"


def _writer():
    """Reuse existing writer validators without an import-time dependency cycle."""
    from litagent import writing_v2

    return writing_v2


def _canonical(value):
    """Keep JSON types significant and reject non-JSON floating point values."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _shape_error(code, path):
    """Attach the original schema location to a rejected transport shape."""
    exc = ValueError(code)
    exc.schema_path = list(path)
    raise exc


def _width(schema, path=()):
    """Require a real finite maximum rather than deriving one from examples."""
    bounds = []
    if "maxItems" in schema:
        maximum = schema["maxItems"]
        if type(maximum) is not int or maximum < 0:
            _shape_error("array_maximum_invalid", (*path, "maxItems"))
        bounds.append(maximum)
    if schema.get("items") is False:
        bounds.append(len(schema.get("prefixItems", [])))
    if not bounds:
        _shape_error("array_maximum_missing", path)
    return min(bounds)


def _item_schema(schema, index):
    """Keep heterogeneous prefix positions tied to their original indices."""
    prefix = schema.get("prefixItems", [])
    child = prefix[index] if index < len(prefix) else schema.get("items")
    if not isinstance(child, dict):
        raise ValueError("array_item_schema_missing")
    return child


def _wrapped(schema, path=()):
    """Only explicitly nonempty arrays change representation."""
    if schema.get("type") != "array":
        return False
    minimum = schema.get("minItems", 0)
    if type(minimum) is not int or minimum not in (0, 1):
        _shape_error("array_minimum_unsupported", (*path, "minItems"))
    return minimum == 1


def _transform(schema, path=()):
    """Preserve ordinary values and zero-minimum collections recursively."""
    if _wrapped(schema, path):
        maximum = _width(schema, path)
        if maximum < 1:
            _shape_error("array_bounds_conflict", path)
        slots = [
            _transform(
                _item_schema(schema, i),
                (
                    (*path, "prefixItems", i)
                    if i < len(schema.get("prefixItems", []))
                    else (*path, "items")
                ),
            )
            for i in range(maximum)
        ]
        tail = {"type": "array", "items": False}
        if len(slots) > 1:
            tail["prefixItems"] = slots[1:]
        return {
            "type": "object",
            "properties": {"head": slots[0], "tail": tail},
            "required": ["head", "tail"],
            "additionalProperties": False,
        }
    result = copy.deepcopy(schema)
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        # The codec owns every possible property path. An open or optional
        # object would let schema validation admit paths that cannot be mapped.
        if schema.get("additionalProperties") is not False:
            _shape_error("object_must_be_closed", (*path, "additionalProperties"))
        if set(schema.get("required", [])) != set(properties):
            _shape_error("object_required_mismatch", (*path, "required"))
        result["properties"] = {
            name: _transform(child, (*path, "properties", name))
            for name, child in properties.items()
        }
    elif schema.get("type") == "array":
        if isinstance(schema.get("items"), dict):
            result["items"] = _transform(schema["items"], (*path, "items"))
        if "prefixItems" in schema:
            result["prefixItems"] = [
                _transform(child, (*path, "prefixItems", index))
                for index, child in enumerate(schema["prefixItems"])
            ]
    return result


def _share_enums(schema):
    """Remove repeated identifier domains without removing any allowed values."""
    definitions = {}
    names = {}

    def visit(node):
        if isinstance(node, list):
            return [visit(child) for child in node]
        if not isinstance(node, dict):
            return node
        if node.get("type") == "string" and "enum" in node:
            signature = _canonical(node)
            name = names.get(signature)
            if name is None:
                name = "wire_enum_" + str(len(names) + 1)
                names[signature] = name
                definitions[name] = copy.deepcopy(node)
            return {"$ref": "#/$defs/" + name}
        return {
            key: copy.deepcopy(value) if key == "enum" else visit(value)
            for key, value in node.items()
        }

    value = visit(schema)
    if definitions:
        value.setdefault("$defs", {}).update(definitions)
    return value


@lru_cache(maxsize=128)
def _prepare(stage, serialized):
    """Cache schema-only contracts without caching any paper or authored text."""
    host = json.loads(serialized)
    Draft202012Validator.check_schema(host)
    wire = _transform(host)
    Draft202012Validator.check_schema(wire)
    native = _share_enums(
        _writer().provider_response_format(stage, wire)["json_schema"]["schema"]
    )
    Draft202012Validator.check_schema(native)
    return {"contract": CONTRACT, "wire_schema": wire, "native_schema": native}


def prepare(stage, host_schema):
    """Return isolated schemas or a stable error at the original schema path."""
    writer = _writer()
    if not isinstance(host_schema, dict):
        raise writer.WriterError(
            "writer_native_wire_contract_invalid",
            diagnostics=[{"phase": "contract", "schema_path": []}],
        )
    try:
        return copy.deepcopy(_prepare(stage, _canonical(host_schema)))
    except (SchemaError, ValueError, TypeError) as exc:
        path = list(getattr(exc, "absolute_path", getattr(exc, "schema_path", [])))
        if not path and isinstance(exc, writer.WriterError) and exc.diagnostics:
            path = list(exc.diagnostics[0].get("path", []))
        raise writer.WriterError(
            "writer_native_wire_contract_invalid",
            diagnostics=[
                {"phase": "contract", "reason": type(exc).__name__, "schema_path": path}
            ],
        ) from exc


def _map(value, schema, *, encoding):
    """Change containers only, preserving text, order, references and duplicates."""
    if _wrapped(schema):
        values = value if encoding else [value["head"], *value["tail"]]
        mapped = [
            _map(item, _item_schema(schema, i), encoding=encoding)
            for i, item in enumerate(values)
        ]
        return {"head": mapped[0], "tail": mapped[1:]} if encoding else mapped
    if schema.get("type") == "array":
        return [
            _map(item, _item_schema(schema, i), encoding=encoding)
            for i, item in enumerate(value)
        ]
    if schema.get("type") == "object":
        return {
            key: _map(item, schema["properties"][key], encoding=encoding)
            for key, item in value.items()
        }
    return copy.deepcopy(value)


def encode(value, host_schema):
    """Change containers only after validating the complete canonical schema."""
    contract = prepare("synthesis", host_schema)
    Draft202012Validator(host_schema).validate(value)
    encoded = _map(value, host_schema, encoding=True)
    Draft202012Validator(contract["native_schema"]).validate(encoded)
    return encoded


def layout(schema):
    """Describe actual wire containers, without examples or source enumeration."""

    def describe(node):
        if "$ref" in node:
            return "defined " + node["$ref"].rsplit("/", 1)[-1]
        kind = node["type"]
        if kind == "object":
            return (
                "object{"
                + ", ".join(
                    json.dumps(k) + ": " + describe(v)
                    for k, v in node["properties"].items()
                )
                + "}"
            )
        if kind == "array":
            if "prefixItems" not in node:
                return (
                    "empty array"
                    if node.get("items") is False
                    else "array<" + describe(node["items"]) + ">"
                )
            groups = []
            for child in node["prefixItems"]:
                shape = describe(child)
                if groups and groups[-1][1] == shape:
                    groups[-1][0] += 1
                else:
                    groups.append([1, shape])
            return (
                "bounded array<"
                + ", ".join(f"up to {n} consecutive {shape}" for n, shape in groups)
                + "> (no additional items)"
            )
        return kind + (" (native enum)" if "enum" in node else "")

    definitions = schema.get("$defs", {})
    return (
        "Native output type map (not example JSON): "
        + describe(schema)
        + ". Definitions: "
        + "; ".join(name + "=" + describe(value) for name, value in definitions.items())
        + ". Return one JSON object matching the native schema. A required head and tail object represents a nonempty collection; head is its first existing item and tail contains only its remaining items in order. An empty tail is legal. Preserve every text value, actual reference, condition and paragraph boundary; do not flatten containers or output the type map. The host restores canonical arrays and still validates text, references, coverage and scientific support. Structural validity is not scientific verification."
    )


def _sha(raw):
    """Keep UTF-8 identity separate from Python character offsets."""
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fail(code):
    """Reject an invalid wire boundary without changing scientific outcomes."""
    raise _writer().WriterError("writer_native_wire_" + code)


def _indexed_tokens(raw):
    """Index validated string-only JSON at its received character positions."""
    scan = _writer().scan_native_strings(raw)
    tokens = scan["tokens"]
    nodes = {}
    cursor = 0

    def consume(kind):
        nonlocal cursor
        if cursor >= len(tokens) or tokens[cursor]["kind"] != kind:
            _fail("token_grammar_invalid")
        token = tokens[cursor]
        cursor += 1
        return token

    def visit(path):
        nonlocal cursor
        if cursor >= len(tokens) or path in nodes:
            _fail("path_coverage_invalid")
        start = tokens[cursor]
        kind = start["kind"]
        if kind == "string":
            token = consume("string")
            if token["is_key"]:
                _fail("value_expected")
            nodes[path] = {"kind": kind, "start": token["start"], "end": token["end"]}
            return
        if kind not in {"{", "["}:
            _fail("value_type_invalid")
        consume(kind)
        closing = "}" if kind == "{" else "]"
        first = True
        index = 0
        while cursor < len(tokens) and tokens[cursor]["kind"] != closing:
            if not first:
                consume(",")
            first = False
            if kind == "{":
                token = consume("string")
                if not token["is_key"]:
                    _fail("key_expected")
                consume(":")
                child = token["value"]
            else:
                child = index
                index += 1
            visit((*path, child))
        end = consume(closing)["end"]
        nodes[path] = {"kind": kind, "start": start["start"], "end": end}

    visit(())
    if cursor != len(tokens):
        _fail("trailing_content")
    return nodes, scan


def _eof_candidate(raw):
    """Reuse the current scanner limits and only append its unique closing stack."""
    scan = _writer().scan_native_strings(raw)
    stack = []
    for token in scan["tokens"]:
        kind = token["kind"]
        if kind in {"{", "["}:
            stack.append(kind)
        elif kind in {"}", "]"}:
            stack.pop()
    if not stack:
        _fail("eof_ineligible")
    suffix = "".join("}" if kind == "{" else "]" for kind in reversed(stack))
    return raw + suffix, suffix


def _bind_canonical(value, schema, nodes, raw, candidate_raw):
    """Bind schema-declared paths, never matching strings or raw member order."""
    units = []
    leaves = []

    def node(path, kind=None):
        found = nodes.get(path)
        if found is None or (kind is not None and found["kind"] != kind):
            _fail("declared_path_missing")
        return found

    def reference_spans(refs, ref_schema, wire_path):
        spans = []
        for index, ref in enumerate(refs):
            path = (
                (*wire_path, "head")
                if _wrapped(ref_schema) and index == 0
                else (
                    (*wire_path, "tail", index - 1)
                    if _wrapped(ref_schema)
                    else (*wire_path, index)
                )
            )
            found = node(path, "string")
            if json.loads(candidate_raw[found["start"] : found["end"]]) != ref:
                _fail("reference_token_mismatch")
            spans.append(
                {"wire_path": list(path), "start": found["start"], "end": found["end"]}
            )
        return spans

    def visit(current, current_schema, canonical_path, wire_path):
        kind = current_schema.get("type")
        if kind == "array":
            node(wire_path, "{" if _wrapped(current_schema) else "[")
            for index, child in enumerate(current):
                child_path = (
                    (*wire_path, "head")
                    if _wrapped(current_schema) and index == 0
                    else (
                        (*wire_path, "tail", index - 1)
                        if _wrapped(current_schema)
                        else (*wire_path, index)
                    )
                )
                visit(
                    child,
                    _item_schema(current_schema, index),
                    (*canonical_path, index),
                    child_path,
                )
            return
        if kind == "object":
            found = node(wire_path, "{")
            fields = current_schema.get("properties", {})
            ref_field = next(
                (
                    key
                    for key in ("fact_card_refs", "body_unit_refs")
                    if set(fields) == {"text", key}
                ),
                None,
            )
            if ref_field is not None:
                if found["end"] > len(raw):
                    _fail("original_unit_incomplete")
                original = _writer()._parse_json(raw[found["start"] : found["end"]])
                if _canonical(
                    _decode_declared_wire(original, current_schema)
                ) != _canonical(current):
                    _fail("original_unit_mismatch")
                _writer()._valid_text(current["text"])
                text_node = node((*wire_path, "text"), "string")
                units.append(
                    {
                        "canonical_path": list(canonical_path),
                        "wire_path": list(wire_path),
                        "start": found["start"],
                        "end": found["end"],
                        "text_span": {
                            "start": text_node["start"],
                            "end": text_node["end"],
                        },
                        "reference_spans": reference_spans(
                            current[ref_field],
                            fields[ref_field],
                            (*wire_path, ref_field),
                        ),
                        "canonical_unit_sha256": _writer().digest(current),
                    }
                )
            for key, child_schema in fields.items():
                if key in current:
                    visit(
                        current[key],
                        child_schema,
                        (*canonical_path, key),
                        (*wire_path, key),
                    )
            return
        if kind != "string" or not isinstance(current, str):
            _fail("unsupported_leaf_type")
        found = node(wire_path, "string")
        if found["end"] > len(raw):
            _fail("original_string_incomplete")
        original = json.loads(raw[found["start"] : found["end"]])
        if original != current:
            _fail("original_string_mismatch")
        if canonical_path == ("title",):
            _writer()._valid_text(current)
        leaves.append(
            {
                "canonical_path": list(canonical_path),
                "wire_path": list(wire_path),
                "start": found["start"],
                "end": found["end"],
            }
        )

    visit(value, schema, (), ())
    raw_leaf_ranges = {
        (record["start"], record["end"])
        for record in nodes.values()
        if record["kind"] == "string"
    }
    assigned = {(record["start"], record["end"]) for record in leaves}
    if len(assigned) != len(leaves) or assigned != raw_leaf_ranges:
        _fail("leaf_coverage_invalid")
    physical = sorted((record["start"], record["end"]) for record in units)
    if any(left[1] > right[0] for left, right in zip(physical, physical[1:])):
        _fail("unit_ranges_overlap")
    if len(units) > 128:
        _fail("unit_capacity_exceeded")
    return units, leaves


def _collection_scope(wire_path, host_schema):
    """Resolve a wire collection to its declared canonical capacity boundary."""
    schema, canonical, index = host_schema, [], 0
    while index < len(wire_path):
        part = wire_path[index]
        if schema.get("type") == "object":
            if part not in schema.get("properties", {}):
                return None
            schema = schema["properties"][part]
            canonical.append(part)
            index += 1
        elif schema.get("type") == "array":
            if _wrapped(schema):
                if part == "tail" and index + 1 == len(wire_path):
                    return canonical, schema, 1
                if part == "head":
                    item, consumed = 0, 1
                elif part == "tail" and index + 1 < len(wire_path):
                    offset = wire_path[index + 1]
                    if type(offset) is not int:
                        return None
                    item, consumed = offset + 1, 2
                else:
                    return None
            else:
                if type(part) is not int:
                    return None
                item, consumed = part, 1
            try:
                schema = _item_schema(schema, item)
            except ValueError:
                return None
            canonical.append(item)
            index += consumed
        else:
            return None
    return canonical, schema, 0


def _decode_declared_wire(value, host_schema):
    """Preserve stable error families instead of inspecting authored error text."""
    if not isinstance(host_schema, dict):
        _fail("contract_invalid")
    contract = prepare("synthesis", host_schema)
    errors = list(Draft202012Validator(contract["native_schema"]).iter_errors(value))
    wire_diagnostics = [
        {
            "phase": "wire",
            "wire_path": list(error.absolute_path),
            "validator": error.validator,
        }
        for error in errors
    ]
    capacity_diagnostics = []
    for error in errors:
        if (
            error.validator != "items"
            or error.validator_value is not False
            or not isinstance(error.instance, list)
        ):
            continue
        scope = _collection_scope(list(error.absolute_path), host_schema)
        if scope is None:
            continue
        canonical_path, original_schema, tail_offset = scope
        if (
            original_schema.get("items") is False
            and "prefixItems" in original_schema
            and len(error.instance) + tail_offset > len(original_schema["prefixItems"])
        ):
            capacity_diagnostics.append(
                {
                    "path": canonical_path,
                    "wire_path": list(error.absolute_path),
                    "actual": len(error.instance) + tail_offset,
                    "maximum": len(original_schema["prefixItems"]),
                    "phase": "wire",
                }
            )
    if capacity_diagnostics:
        raise _writer().WriterError(
            "writer_native_capacity_exceeded", "output_scope", capacity_diagnostics
        )
    if errors and any(error.validator != "enum" for error in errors):
        raise _writer().WriterError(
            "writer_native_wire_schema_invalid", diagnostics=wire_diagnostics
        )
    # Enum-only errors leave every container shape valid. Restore their exact
    # canonical positions so the real validator can preserve reference failures
    # as quality diagnostics. No invalid value is returned as accepted prose.
    canonical = _map(value, host_schema, encoding=False)
    try:
        canonical = _writer().parse_response(_canonical(canonical), host_schema)
    except _writer().WriterError as exc:
        exc.diagnostics = [
            {**diagnostic, "phase": "canonical"} for diagnostic in exc.diagnostics
        ]
        exc.wire_diagnostics = wire_diagnostics
        raise
    if errors:
        raise _writer().WriterError(
            "writer_native_wire_schema_invalid", diagnostics=wire_diagnostics
        )
    return canonical


def parse(raw, host_schema, *, allow_eof=False):
    """Validate received wire values and bind canonical units to original spans."""
    if not isinstance(host_schema, dict):
        _fail("contract_invalid")
    if not isinstance(raw, str) or not raw or len(raw) > 1000000:
        _fail("raw_limit")
    try:
        raw.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise _writer().WriterError("writer_native_wire_raw_unicode_invalid") from exc
    if type(allow_eof) is not bool:
        _fail("eof_flag_invalid")
    # Bound depth, token count, containers and complete strings before Python's
    # general JSON decoder can recurse into hostile output.
    _writer().scan_native_strings(raw)
    candidate_raw, suffix = raw, ""
    try:
        wire_value = _writer()._parse_json(raw)
    except _writer().WriterError as exc:
        if not allow_eof or exc.code != "writer_json_invalid":
            raise
        roots = set(host_schema.get("properties", {})) - {"title"}
        if not roots or not (
            roots <= set(_writer().BODY) or roots <= {"abstract", "conclusion"}
        ):
            _fail("eof_stage_ineligible")
        candidate_raw, suffix = _eof_candidate(raw)
        wire_value = _writer()._parse_json(candidate_raw)
    canonical = _decode_declared_wire(wire_value, host_schema)
    nodes, scan = _indexed_tokens(candidate_raw)
    units, leaves = _bind_canonical(canonical, host_schema, nodes, raw, candidate_raw)
    if suffix and not units:
        _fail("eof_no_original_units")
    bindings = {
        "contract": CONTRACT,
        "raw_sha256": _sha(raw),
        "candidate_sha256": _sha(candidate_raw),
        "host_schema_sha256": _writer().digest(host_schema),
        "native_schema_sha256": _writer().digest(
            prepare("synthesis", host_schema)["native_schema"]
        ),
        "units": units,
        "leaves": leaves,
        "scanner_tokens": len(scan["tokens"]),
        "coverage_complete": True,
        "scientific_assessment": "not_verified",
    }
    return {
        "canonical": canonical,
        "bindings": bindings,
        "normalization": (
            {
                "suffix": suffix,
                "raw_sha256": _sha(raw),
                "candidate_sha256": _sha(candidate_raw),
                "original_units_complete": True,
            }
            if suffix
            else None
        ),
        "raw_sha256": _sha(raw),
        "candidate_sha256": _sha(candidate_raw),
        "candidate_raw": candidate_raw,
    }
