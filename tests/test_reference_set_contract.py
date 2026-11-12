"""Keep exact received reference occurrences under an explicit ordered-set rule."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from litagent.writing_flat import (
    REFERENCE_SET_CONTRACT,
    encode,
    parse,
    prepare,
    reference_set_identity,
    restore_reference_input,
)
from litagent.writing_v2 import WriterError, body_section_schema, digest, schema_for


def chapter():
    """Provide three permitted IDs so a legal duplicate stays within raw bounds."""
    schema = body_section_schema("methods", ["c1", "c2", "c3"], maximum=8)
    wire = {
        "units": [
            {
                "text": "A source-supported finding with its conditions.",
                "fact_card_refs": ["c1", "c2", "c1"],
                "paragraph_start": True,
            }
        ]
    }
    return schema, wire


def selected(raw, schema, **kwargs):
    """Use the explicit marker, never infer permission from response content."""
    return parse(raw, schema, reference_contract=REFERENCE_SET_CONTRACT, **kwargs)


def test_real_036_raw_remains_rejected_by_old_contract():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/public-reference-duplicates-036.json"
        ).read_text()
    )
    raw, schema = fixture["raw"], fixture["schema"]
    assert hashlib.sha256(raw.encode()).hexdigest() == fixture["raw_sha256"]
    with pytest.raises(WriterError) as caught:
        parse(raw, schema)
    assert caught.value.code == "writer_flat_schema_invalid"
    assert {"path": ["units", 5, "fact_card_refs"], "reason": "uniqueItems"} in (
        caught.value.diagnostics
    )
    result = selected(raw, schema)
    affected = result["bindings"]["units"][5]
    assert affected["received_reference_ids"] == ["c1", "c9", "c18", "c1"]
    assert [s["canonical_index"] for s in affected["reference_spans"]] == [0, 1, 2, 0]
    assert result["bindings"]["scientific_assessment"] == "not_verified"
    assert restore_reference_input(result["canonical"], result["bindings"]) == (
        result["received_canonical"]
    )
    assert fixture["raw"] == raw


def test_exact_duplicates_preserve_text_units_raw_spans_and_received_hashes():
    schema, wire = chapter()
    wire["units"].append({**copy.deepcopy(wire["units"][0]), "paragraph_start": False})
    raw = json.dumps(wire)
    result = selected(raw, schema)
    assert len(result["canonical"]["methods"][0]["sentences"]) == 2
    assert result["bindings"]["reference_contract"] == REFERENCE_SET_CONTRACT
    assert result["bindings"]["reference_contract_identity"] == reference_set_identity()
    for index, record in enumerate(result["bindings"]["units"]):
        received = result["received_canonical"]["methods"][0]["sentences"][index]
        canonical = result["canonical"]["methods"][0]["sentences"][index]
        assert received["fact_card_refs"] == ["c1", "c2", "c1"]
        assert canonical["fact_card_refs"] == ["c1", "c2"]
        assert canonical["text"] == received["text"] == wire["units"][index]["text"]
        assert record["received_unit_sha256"] == digest(received)
        assert record["canonical_unit_sha256"] == digest(canonical)
        assert len(record["reference_spans"]) == 3
        for expected, span in zip(
            received["fact_card_refs"], record["reference_spans"], strict=True
        ):
            assert json.loads(raw[span["start"] : span["end"]]) == expected
    assert (
        restore_reference_input(result["canonical"], result["bindings"])
        == result["received_canonical"]
    )


@pytest.mark.parametrize("refs", [["c1", "c1", "c1", "c1"], "c1", None, [True], [1]])
def test_original_array_bounds_and_types_precede_normalization(refs):
    schema, wire = chapter()
    wire["units"][0]["fact_card_refs"] = refs
    with pytest.raises(WriterError) as caught:
        selected(json.dumps(wire), schema)
    assert caught.value.code == "writer_flat_schema_invalid"


@pytest.mark.parametrize("unknown", ["unknown", " c1", "C1", "c1 "])
def test_unknown_id_is_not_removed_or_guessed(unknown):
    schema, wire = chapter()
    wire["units"][0]["fact_card_refs"] = ["c1", unknown, "c1"]
    with pytest.raises(WriterError) as caught:
        selected(json.dumps(wire), schema)
    assert caught.value.code == "writer_reference_invalid"
    assert caught.value.candidate["methods"][0]["sentences"][0]["fact_card_refs"] == [
        "c1",
        unknown,
        "c1",
    ]


@pytest.mark.parametrize("fault", ["extra", "duplicate_key", "boolean", "units"])
def test_rule_does_not_repair_other_invalid_response_fields(fault):
    schema, wire = chapter()
    if fault == "extra":
        wire["fact_card_refs"] = []
    elif fault == "boolean":
        wire["units"][0]["paragraph_start"] = "true"
    elif fault == "units":
        wire["units"] = wire["units"] * 9
    raw = json.dumps(wire)
    if fault == "duplicate_key":
        raw = raw.replace('"text":', '"text":"shadow", "text":', 1)
    with pytest.raises(WriterError):
        selected(raw, schema)


@pytest.mark.parametrize("stage", ["summary", "repair", "summary_repair"])
def test_summary_and_both_patch_kinds_preserve_received_occurrences(stage):
    references = ["u1", "u2", "u3"] if stage != "repair" else ["c1", "c2", "c3"]
    field = "fact_card_refs" if stage == "repair" else "body_unit_refs"
    unit = {
        "text": "The same complete finding.",
        field: [references[0], references[1], references[0]],
    }
    if stage == "summary":
        schema = schema_for("summary", references)
        canonical = {
            key: [{"sentences": [copy.deepcopy(unit)]}]
            for key in ("abstract", "conclusion")
        }
    else:
        schema = schema_for(
            "repair", references, ["target"], summary_repair=stage == "summary_repair"
        )
        canonical = {
            "outcome": "patched",
            "explanation": "Preserve scientific premises.",
            "patches": [{"target_id": "target", "sentences": [copy.deepcopy(unit)]}],
        }
    # Encoding remains strict; construct the received wire only after a valid encode.
    unique = copy.deepcopy(canonical)
    paths = (
        [p["sentences"][0] for p in unique["patches"]]
        if stage != "summary"
        else [unique[k][0]["sentences"][0] for k in unique]
    )
    for value in paths:
        value[field] = references[:2]
    wire = encode(unique, schema)
    wire_units = wire["patches"] if stage != "summary" else [wire[k][0] for k in wire]
    for value in wire_units:
        value[field] = unit[field]
    result = selected(json.dumps(wire), schema)
    assert result["received_canonical"] == canonical
    assert result["canonical"] == unique
    assert restore_reference_input(unique, result["bindings"]) == canonical


def test_passthrough_arrays_never_receive_reference_uniqueness_exception():
    schema = {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    with pytest.raises(WriterError):
        selected('{"items":["c1","c1"]}', schema)
    result = selected('{"items":["c1","c2"]}', schema)
    assert (
        result["canonical"] == result["received_canonical"] == {"items": ["c1", "c2"]}
    )
    assert (
        restore_reference_input(result["canonical"], result["bindings"])
        == result["canonical"]
    )


def test_passthrough_field_name_does_not_create_prose_unit_bindings():
    schema = {
        "type": "object",
        "properties": {
            "patches": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["patches"],
        "additionalProperties": False,
    }
    result = selected('{"patches":["plain"]}', schema)
    assert result["bindings"]["reference_reply_kind"] == "passthrough"
    assert restore_reference_input(result["canonical"], result["bindings"]) == {
        "patches": ["plain"]
    }


@pytest.mark.parametrize(
    "fault",
    [
        "path",
        "duplicate_path",
        "wire_path",
        "boolean_wire_path",
        "negative_wire_path",
        "boolean_occurrence_path",
        "boolean_canonical_path",
        "index",
        "span",
        "text_span",
        "text_reference_overlap",
        "received",
        "received_hash",
        "canonical_hash",
        "document_hash",
        "missing",
    ],
)
def test_restore_rejects_wrong_paths_hashes_occurrences_or_coverage(fault):
    schema, wire = chapter()
    wire["units"].append({**copy.deepcopy(wire["units"][0]), "paragraph_start": False})
    result = selected(json.dumps(wire), schema)
    bindings = copy.deepcopy(result["bindings"])
    record = bindings["units"][0]
    if fault == "path":
        record["canonical_path"][1] = 9
    elif fault == "duplicate_path":
        bindings["units"][1]["canonical_path"] = record["canonical_path"]
    elif fault == "wire_path":
        record["wire_path"] = ["wrong", 0]
    elif fault == "boolean_wire_path":
        record["wire_path"][-1] = False
    elif fault == "negative_wire_path":
        record["wire_path"][-1] = -1
    elif fault == "boolean_occurrence_path":
        record["reference_spans"][0]["wire_path"][-1] = False
    elif fault == "boolean_canonical_path":
        record["canonical_path"][1] = False
    elif fault == "index":
        record["reference_spans"][-1]["canonical_index"] = 1
    elif fault == "span":
        record["reference_spans"][-1]["start"] = record["end"] + 1
    elif fault == "text_span":
        record["text_span"]["start"] = record["end"]
    elif fault == "text_reference_overlap":
        record["text_span"] = {
            key: record["reference_spans"][0][key] for key in ("start", "end")
        }
    elif fault == "received":
        record["received_reference_ids"][-1] = "c2"
    elif fault == "received_hash":
        record["received_unit_sha256"] = "0" * 64
    elif fault == "canonical_hash":
        record["canonical_unit_sha256"] = "0" * 64
    elif fault == "document_hash":
        bindings["received_canonical_sha256"] = "0" * 64
    elif fault == "missing":
        bindings["units"].pop()
    with pytest.raises(WriterError, match="writer_reference_binding_invalid"):
        restore_reference_input(result["canonical"], bindings)


def test_empty_transitions_eof_and_legacy_result_shape_remain_separate():
    schema, wire = chapter()
    wire["units"][0]["fact_card_refs"] = []
    raw = json.dumps(wire)
    assert "received_canonical" not in parse(raw, schema)
    result = selected(raw[:-2], schema, allow_eof=True)
    assert result["normalization"]["suffix"] == "]}"
    assert result["received_canonical"] == result["canonical"]
    assert result["bindings"]["units"][0]["reference_spans"] == []
    assert (
        restore_reference_input(result["canonical"], result["bindings"])
        == result["canonical"]
    )


@pytest.mark.parametrize("marker", ["", "future", True, [], {}])
def test_unknown_reference_contract_is_a_typed_error(marker):
    schema, wire = chapter()
    with pytest.raises(WriterError, match="writer_reference_contract_invalid"):
        parse(json.dumps(wire), schema, reference_contract=marker)


def test_preparation_and_host_schema_do_not_implicitly_relax_uniqueness():
    schema, wire = chapter()
    before = copy.deepcopy(schema)
    plan = prepare(schema)
    assert (
        plan["wire_schema"]["properties"]["units"]["items"]["properties"][
            "fact_card_refs"
        ]["uniqueItems"]
        is True
    )
    selected(json.dumps(wire), schema)
    assert schema == before
