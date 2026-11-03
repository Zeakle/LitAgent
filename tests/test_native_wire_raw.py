"""Preserve received character spans and existing EOF rejection boundaries."""

import copy
import hashlib
import json

import pytest

from litagent import writing_native as native
from litagent import writing_v2 as writer

RAW = r'{"taxonomy":{"head":{"sentences":{"head":{"text":"同\u4e00\/Ω","fact_card_refs":{"head":"c1","tail":["c2"]}},"tail":[{"text":"same","fact_card_refs":{"head":"c2","tail":[]}}]}},"tail":[{"sentences":{"head":{"text":"same","fact_card_refs":{"head":"c1","tail":[]}},"tail":[]}}]}}'


def dumped(value):
    """Keep synthetic field order visible while serializing legal JSON."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def reversed_wrappers(value):
    """Move tail before head without changing the logical collection order."""
    if isinstance(value, dict):
        keys = ("tail", "head") if set(value) == {"head", "tail"} else value
        return {key: reversed_wrappers(value[key]) for key in keys}
    if isinstance(value, list):
        return [reversed_wrappers(child) for child in value]
    return value


@pytest.fixture
def schema():
    """Use an actual heterogeneous bounded chapter schema."""
    return writer.body_section_schema("taxonomy", ["c1", "c2"], maximum=5)


def test_received_spans_bind_duplicate_text_unicode_and_each_reference(schema):
    """Decoded text is never used to reconstruct received source coordinates."""
    result = native.parse(RAW, schema)
    assert len(RAW) == 276
    assert result["raw_sha256"] == hashlib.sha256(RAW.encode()).hexdigest()
    assert result["candidate_raw"] == RAW
    assert result["normalization"] is None
    units = result["bindings"]["units"]
    assert [record["canonical_path"] for record in units] == [
        ["taxonomy", 0, "sentences", 0],
        ["taxonomy", 0, "sentences", 1],
        ["taxonomy", 1, "sentences", 0],
    ]
    assert [record["wire_path"] for record in units] == [
        ["taxonomy", "head", "sentences", "head"],
        ["taxonomy", "head", "sentences", "tail", 0],
        ["taxonomy", "tail", 0, "sentences", "head"],
    ]
    assert [(record["start"], record["end"]) for record in units] == [
        (41, 107),
        (116, 172),
        (205, 261),
    ]
    assert [
        (record["text_span"]["start"], record["text_span"]["end"]) for record in units
    ] == [(49, 61), (124, 130), (213, 219)]
    assert [
        (ref["start"], ref["end"]) for unit in units for ref in unit["reference_spans"]
    ] == [(87, 91), (100, 104), (156, 160), (245, 249)]
    assert result["canonical"]["taxonomy"][0]["sentences"][0]["text"] == "同一/Ω"
    assert RAW[49:61] == r'"同\u4e00\/Ω"'
    leaves = result["bindings"]["leaves"]
    assert len(leaves) == len({(leaf["start"], leaf["end"]) for leaf in leaves}) == 7
    assert result["bindings"]["coverage_complete"] is True


def test_tail_before_head_uses_canonical_order_with_nonmonotonic_spans(schema):
    """Object member order cannot reverse paragraphs or sentences."""
    raw = dumped(reversed_wrappers(json.loads(RAW)))
    result = native.parse(raw, schema)
    assert result["canonical"] == native.parse(RAW, schema)["canonical"]
    starts = [unit["start"] for unit in result["bindings"]["units"]]
    assert starts != sorted(starts)
    for unit in result["bindings"]["units"]:
        assert set(json.loads(raw[unit["start"] : unit["end"]])) == {
            "text",
            "fact_card_refs",
        }


def test_json_looking_prose_does_not_create_a_second_unit():
    """An untrusted quoted example remains one authored string token."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=1)
    text = 'The paper quotes {"text":"fake","fact_card_refs":["c999"]} as an example.'
    value = {"taxonomy": [{"sentences": [{"text": text, "fact_card_refs": ["c1"]}]}]}
    parsed = native.parse(dumped(native.encode(value, schema)), schema)
    assert parsed["canonical"] == value
    assert len(parsed["bindings"]["units"]) == 1


@pytest.mark.parametrize(
    "fault",
    ["missing_head", "empty_old_array", "extra_root", "extra_sibling", "empty_refs"],
)
def test_invalid_wire_cannot_borrow_text_from_a_different_location(schema, fault):
    """Matching strings elsewhere never authorize an unknown or empty container."""
    wire = json.loads(RAW)
    if fault == "missing_head":
        del wire["taxonomy"]["head"]
    elif fault == "empty_old_array":
        wire["taxonomy"] = []
    elif fault == "extra_root":
        wire["unrelated"] = "same"
    elif fault == "extra_sibling":
        wire["taxonomy"]["head"]["unrelated"] = {"text": "same"}
    else:
        wire["taxonomy"]["head"]["sentences"]["head"]["fact_card_refs"] = []
    with pytest.raises(writer.WriterError):
        native.parse(dumped(wire), schema, allow_eof=True)


def test_eof_only_closes_outer_containers_and_retains_original_spans(schema):
    """A normalization candidate is independently identified from the original."""
    original = RAW[:-1]
    parsed = native.parse(original, schema, allow_eof=True)
    complete = native.parse(RAW, schema)
    assert parsed["canonical"] == complete["canonical"]
    assert parsed["bindings"]["units"] == complete["bindings"]["units"]
    assert (
        parsed["candidate_raw"] == original + parsed["normalization"]["suffix"] == RAW
    )
    assert parsed["normalization"]["original_units_complete"] is True
    assert parsed["raw_sha256"] != parsed["candidate_sha256"]
    with pytest.raises(writer.WriterError, match="writer_json_invalid"):
        native.parse(original, schema)


@pytest.mark.parametrize("close_reference", [False, True])
def test_a_valid_closed_candidate_cannot_authorize_an_incomplete_original_unit(
    close_reference,
):
    """Even complete scientific strings do not constitute an intact unit object."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=1)
    partial = '{"taxonomy":{"tail":[],"head":{"sentences":{"tail":[],"head":{"text":"A","fact_card_refs":{"head":"c1","tail":[]}'
    if not close_reference:
        partial = partial[:-1]
    suffix = "}" * (5 if close_reference else 6)
    value = json.loads(partial + suffix)
    assert value["taxonomy"]["head"]["sentences"]["head"]["text"] == "A"
    with pytest.raises(
        writer.WriterError, match="writer_native_wire_original_unit_incomplete"
    ):
        native.parse(partial, schema, allow_eof=True)


def test_empty_sentence_tail_can_close_after_a_complete_original_head():
    """An unfinished layout container is distinct from an unfinished unit."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=1)
    raw = '{"taxonomy":{"tail":[],"head":{"sentences":{"head":{"text":"A","fact_card_refs":{"head":"c1","tail":[]}},"tail":['
    result = native.parse(raw, schema, allow_eof=True)
    assert result["normalization"]["suffix"] == "]}}}}"
    assert result["bindings"]["units"][0]["end"] <= len(raw)


@pytest.mark.parametrize(
    "raw",
    [
        RAW.replace('"text":', '"text":"ignored","text":', 1),
        RAW.replace('"text":', r'"te\u0078t":"ignored","text":', 1),
        RAW.replace('"head":', '"head":{},"head":', 1),
        RAW.replace(',"tail":', '"tail":', 1),
        RAW[:-1] + "]",
        RAW[:-1] + ",}",
        RAW + "{}",
        RAW + '"same"',
        RAW[: RAW.index('"same"') + 3],
        RAW.replace('"same"', r'"\ud800"', 1),
    ],
)
def test_ambiguous_or_incomplete_raw_cannot_enter_canonical_consumers(schema, raw):
    """The adapter never drops duplicates, repairs inner syntax or ignores suffixes."""
    with pytest.raises(writer.WriterError):
        native.parse(raw, schema, allow_eof=True)


@pytest.mark.parametrize(
    "literal", ["NaN", "Infinity", "-Infinity", "true", "false", "null", "1"]
)
def test_non_string_constants_are_rejected_by_the_actual_raw_boundary(schema, literal):
    """The scanner owns this restriction; the old JSON parser need not reject NaN."""
    raw = RAW.replace(r'"同\u4e00\/Ω"', literal, 1)
    assert raw != RAW
    with pytest.raises(writer.WriterError):
        native.parse(raw, schema)


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        " " * 1000001,
        "\ud800",
        "[" * 3000,
        "[" * 3000 + "]" * 3000,
        dumped({"x": [{}] * 1024}),
        dumped({"x": ["v"] * 8200}),
    ],
)
def test_raw_limits_run_before_generic_decoder_or_schema_work(schema, raw):
    """Pathological raw input raises a checked error rather than RecursionError."""
    with pytest.raises(writer.WriterError):
        native.parse(raw, schema, allow_eof=True)


def test_unknown_reference_keeps_quality_candidate_and_both_coordinate_paths(schema):
    """The extra wrapper must not turn a repairable reference into protocol failure."""
    value = json.loads(RAW)
    value["taxonomy"]["head"]["sentences"]["head"]["fact_card_refs"]["head"] = "outside"
    with pytest.raises(writer.WriterError) as failure:
        native.parse(dumped(value), schema)
    error = failure.value
    assert error.code == "writer_reference_invalid" and error.family == "quality"
    assert (
        error.candidate["taxonomy"][0]["sentences"][0]["fact_card_refs"][0] == "outside"
    )
    assert error.diagnostics == [
        {
            "path": ["taxonomy", 0, "sentences", 0, "fact_card_refs", 0],
            "phase": "canonical",
        }
    ]
    assert error.wire_diagnostics[0]["wire_path"] == [
        "taxonomy",
        "head",
        "sentences",
        "head",
        "fact_card_refs",
        "head",
    ]


@pytest.mark.parametrize(
    "actual,family,maximum", [(3, "protocol", 2), (4, "output_scope", 3)]
)
def test_combined_maxItems_and_tuple_bounds_keep_original_failure_precedence(
    actual, family, maximum
):
    """A tighter maxItems does not manufacture an old tuple overflow."""
    schema = writer.body_section_schema("taxonomy", ["c1"], maximum=7)
    schema["properties"]["taxonomy"]["maxItems"] = 2
    canonical = {"taxonomy": [{"sentences": [{"text": "A", "fact_card_refs": ["c1"]}]}]}
    wire = native.encode(canonical, schema)
    wire["taxonomy"]["tail"] = [
        copy.deepcopy(wire["taxonomy"]["head"]) for _ in range(actual - 1)
    ]
    with pytest.raises(writer.WriterError) as old:
        writer.parse_response(
            dumped({"taxonomy": canonical["taxonomy"] * actual}), schema
        )
    with pytest.raises(writer.WriterError) as new:
        native.parse(dumped(wire), schema)
    assert new.value.family == old.value.family == family
    if family == "output_scope":
        assert new.value.code == old.value.code == "writer_native_capacity_exceeded"
        assert new.value.diagnostics[0]["actual"] == actual
        assert new.value.diagnostics[0]["maximum"] == maximum
        assert new.value.diagnostics[0]["path"] == ["taxonomy"]


@pytest.mark.parametrize("summary_repair", [False, True])
def test_repair_eof_is_not_authorized_by_the_helper_switch(summary_repair):
    """The experiment flag preserves the narrow schema family boundary."""
    refs = ["u1"] if summary_repair else ["c1"]
    schema = writer.schema_for(
        "repair", refs, ["target"], summary_repair=summary_repair
    )
    value = {
        "outcome": "declined",
        "explanation": "Insufficient support.",
        "patches": [],
    }
    raw = dumped(native.encode(value, schema))
    with pytest.raises(
        writer.WriterError, match="writer_native_wire_eof_stage_ineligible"
    ):
        native.parse(raw[:-1], schema, allow_eof=True)
