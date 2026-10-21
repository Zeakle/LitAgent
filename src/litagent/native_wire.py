"""Translate explicit array transport without changing report unit values."""

from __future__ import annotations

import copy
import json
import re

from jsonschema import Draft202012Validator

from litagent.body_quality import NATIVE_ARRAY_WIRE_CONTRACT as CONTRACT
from litagent.report_audit import digest

BODY = ("introduction", "taxonomy", "methods", "experiments", "open_problems")
SECTIONS = (*BODY, "abstract", "conclusion")
BODY_FIELDS = {"kind", "text", "fact_card_ids", "scope_notice_id"}
SUMMARY_FIELDS = {"text", "body_references"}


def closed(value, keys):
    """Reject extra, missing and mixed transport fields before projecting."""
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("array_wire_shape_invalid")


def unique(pairs):
    """Reject duplicate JSON keys rather than selecting their last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("array_wire_duplicate_key")
        result[key] = value
    return result


def inline_local_schema(schema: dict) -> dict:
    """Expand acyclic root definitions without changing assertion semantics."""
    if not isinstance(schema, dict):
        raise ValueError("native_schema_inline_invalid")
    Draft202012Validator.check_schema(schema)
    definitions = schema.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValueError("native_schema_inline_invalid")
    schema_maps = {"properties", "patternProperties", "dependentSchemas"}
    single_schemas = {
        "items",
        "additionalProperties",
        "unevaluatedProperties",
        "unevaluatedItems",
        "propertyNames",
        "contains",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    }
    schema_arrays = {"prefixItems", "allOf", "anyOf", "oneOf"}
    visited = 0

    def expand(node, stack=(), depth=0):
        """Resolve schema positions only; annotation and literal data stay exact."""
        nonlocal visited
        visited += 1
        if depth > 64 or visited > 10000:
            raise ValueError("native_schema_inline_capacity_exceeded")
        if isinstance(node, bool):
            return node
        if not isinstance(node, dict) or "$defs" in node:
            raise ValueError("native_schema_inline_invalid")
        if any(
            key in node
            for key in (
                "$id",
                "$anchor",
                "$dynamicAnchor",
                "$dynamicRef",
                "$recursiveRef",
            )
        ):
            # Scoped and dynamic resolution needs a resolver, not substitution.
            raise ValueError("native_schema_inline_reference_invalid")
        if "$ref" in node:
            ref = node["$ref"]
            match = (
                re.fullmatch(r"#/\$defs/([^/]+)", ref) if isinstance(ref, str) else None
            )
            if set(node) != {"$ref"} or match is None:
                raise ValueError("native_schema_inline_reference_invalid")
            name = match[1].replace("~1", "/").replace("~0", "~")
            if name not in definitions or name in stack:
                raise ValueError("native_schema_inline_reference_invalid")
            return expand(definitions[name], (*stack, name), depth + 1)
        result = {}
        for key, value in node.items():
            if key in schema_maps:
                result[key] = {
                    name: expand(child, stack, depth + 1)
                    for name, child in value.items()
                }
            elif key in single_schemas:
                result[key] = expand(value, stack, depth + 1)
            elif key in schema_arrays:
                result[key] = [expand(child, stack, depth + 1) for child in value]
            else:
                result[key] = copy.deepcopy(value)
        return result

    expanded = expand({key: value for key, value in schema.items() if key != "$defs"})
    Draft202012Validator.check_schema(expanded)
    return expanded


def array_wire_schema(native_schema):
    """Project known v22 containers; retain unit permissions and source definitions."""
    source = copy.deepcopy(native_schema)
    Draft202012Validator.check_schema(source)
    try:
        closed(source["properties"], {"title", "sections"})
        sections = source["properties"]["sections"]["properties"]
        closed(sections, SECTIONS)
        for section in sections.values():
            closed(section["properties"], {"first_block", "additional_blocks"})
            ref = section["properties"]["first_block"]
            if section["properties"]["additional_blocks"] != {
                "type": "array",
                "items": ref,
            }:
                raise ValueError("array_wire_source_schema_invalid")
            name = ref["$ref"].removeprefix("#/$defs/")
            block = source["$defs"][name]
            properties = block["properties"]
            closed(
                properties,
                {"heading", "type", "first_sentence", "additional_sentences"},
            )
            # First-position restrictions remain mandatory in the original local
            # schema after projection; ordinary items are only a transport shape.
            sentence = properties["additional_sentences"]["items"]
            block["properties"] = {
                "heading": properties["heading"],
                "type": properties["type"],
                "sentences": {"type": "array", "items": sentence},
            }
            block["required"] = list(block["properties"])
            section["properties"] = {"blocks": {"type": "array", "items": ref}}
            section["required"] = ["blocks"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("array_wire_source_schema_invalid") from exc
    Draft202012Validator.check_schema(source)
    return source


def _check_unit(unit, section, *, validate_references=True):
    """Validate known value types while leaving scientific support unverified."""
    closed(unit, BODY_FIELDS if section in BODY else SUMMARY_FIELDS)
    if not isinstance(unit["text"], str):
        raise ValueError("array_wire_unit_invalid")
    if section in BODY:
        if (
            not isinstance(unit["kind"], str)
            or unit["kind"] not in {"scientific", "organization", "gap"}
            or not isinstance(unit["scope_notice_id"], str)
            or not isinstance(unit["fact_card_ids"], list)
            or any(not isinstance(cid, str) for cid in unit["fact_card_ids"])
        ):
            raise ValueError("array_wire_unit_invalid")
    elif not isinstance(unit["body_references"], list) or (
        validate_references and not unit["body_references"]
    ):
        raise ValueError("array_wire_reference_invalid")


def array_from_native(document, *, body_only=False):
    """Encode canonical containers without editing unit values or ordering."""
    closed(document, {"title", "sections"})
    closed(document["sections"], BODY if body_only else SECTIONS)
    result = {"title": document["title"], "sections": {}}
    for key, section in document["sections"].items():
        closed(section, {"first_block", "additional_blocks"})
        if not isinstance(section["additional_blocks"], list):
            raise ValueError("array_wire_shape_invalid")
        blocks = []
        for block in [section["first_block"], *section["additional_blocks"]]:
            closed(block, {"heading", "type", "first_sentence", "additional_sentences"})
            if not isinstance(block["additional_sentences"], list):
                raise ValueError("array_wire_shape_invalid")
            units = [block["first_sentence"], *block["additional_sentences"]]
            for unit in units:
                _check_unit(unit, key)
            blocks.append(
                {
                    "heading": block["heading"],
                    "type": block["type"],
                    "sentences": copy.deepcopy(units),
                }
            )
        result["sections"][key] = {"blocks": blocks}
    return result


def canonical_from_array(document, *, validate_references=True, body_only=False):
    """Restore exact indices and field values, recording only a format projection."""
    if isinstance(document, str):
        document = json.loads(document, object_pairs_hook=unique)
    closed(document, {"title", "sections"})
    if not isinstance(document["title"], str):
        raise ValueError("array_wire_shape_invalid")
    closed(document["sections"], BODY if body_only else SECTIONS)
    result, positions = {"title": document["title"], "sections": {}}, []
    for key, section in document["sections"].items():
        closed(section, {"blocks"})
        if not isinstance(section["blocks"], list) or not section["blocks"]:
            raise ValueError("array_wire_shape_invalid")
        blocks = []
        for bi, block in enumerate(section["blocks"]):
            closed(block, {"heading", "type", "sentences"})
            if not isinstance(block["heading"], str) or block["type"] not in {
                "paragraph",
                "list",
            }:
                raise ValueError("array_wire_shape_invalid")
            units = block["sentences"]
            if not isinstance(units, list) or not units:
                raise ValueError("array_wire_shape_invalid")
            for si, unit in enumerate(units):
                _check_unit(unit, key, validate_references=validate_references)
                positions.append(
                    {
                        "wire_path": f"sections.{key}.blocks.{bi}.sentences.{si}",
                        "canonical_path": f"sections.{key}."
                        + ("first_block" if bi == 0 else f"additional_blocks.{bi-1}")
                        + "."
                        + (
                            "first_sentence"
                            if si == 0
                            else f"additional_sentences.{si-1}"
                        ),
                        "unit_sha256": digest(unit),
                    }
                )
            blocks.append(
                {
                    "heading": block["heading"],
                    "type": block["type"],
                    "first_sentence": copy.deepcopy(units[0]),
                    "additional_sentences": copy.deepcopy(units[1:]),
                }
            )
        result["sections"][key] = {
            "first_block": blocks[0],
            "additional_blocks": blocks[1:],
        }
    for key in (() if body_only else ("abstract", "conclusion")):
        for block in document["sections"][key]["blocks"]:
            for unit in block["sentences"]:
                if not validate_references:
                    # The canonical compiler owns repairable target existence,
                    # duplicate selections and semantic kind checks. The caller
                    # still validates the complete wire schema before rendering.
                    continue
                seen = set()
                for ref in unit["body_references"]:
                    closed(ref, {"section_key", "block_index", "sentence_index"})
                    k, bi, si = (
                        ref["section_key"],
                        ref["block_index"],
                        ref["sentence_index"],
                    )
                    if (
                        k not in BODY
                        or type(bi) is not int
                        or type(si) is not int
                        or min(bi, si) < 0
                        or (k, bi, si) in seen
                    ):
                        raise ValueError("array_wire_reference_invalid")
                    seen.add((k, bi, si))
                    if validate_references:
                        try:
                            target = document["sections"][k]["blocks"][bi]["sentences"][
                                si
                            ]
                        except IndexError as exc:
                            raise ValueError("array_wire_reference_invalid") from exc
                        if target["kind"] != "scientific":
                            raise ValueError("array_wire_reference_invalid")
    return result, {
        "contract": CONTRACT,
        "wire_document_sha256": digest(document),
        "canonical_sha256": digest(result),
        "positions": positions,
        "scientific_verdict": "not_evaluated",
    }
