"""Bind the native summary layout to the complete report's actual consumer."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_body_analysis import analysis_document, render_analysis
from test_native_array_integration import request
from test_staged_native_generation import split
from test_writing_passages import facts

from litagent.native_wire import canonical_from_array
from litagent.quality import SUMMARY_RULES, project_grounded_response
from litagent.staged_generation import (
    decode_summaries,
    encode_summaries,
    prepare_summary_request,
    summary_layout,
    summary_reference_binding,
)


def material(language="en"):
    """Use the real producer for canonical data, permissions and full schema."""
    cards = facts()
    body, summaries = split(analysis_document(cards, language))
    messages, rf = request(cards, language, array_wire=True)
    binding = summary_reference_binding(body)
    layout = summary_layout(rf, SUMMARY_RULES)
    return cards, body, summaries, messages, rf, binding, layout


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("blocks,units", [(1, 1), (1, 6), (2, 6)])
def test_native_layout_reaches_complete_canonical_schema(language, blocks, units):
    cards, body, summaries, messages, rf, binding, layout = material(language)
    prepared, native = prepare_summary_request(
        body,
        binding["body_unit_count"],
        messages,
        rf,
        short_refs=True,
        compact_body=True,
        host_layout=True,
    )
    packet = json.loads(prepared[1]["content"])
    assert packet["summary_layout"] == layout
    assert native["json_schema"]["name"] == "grounded_summaries_v3_layout"
    assert "heading or type fields" in prepared[0]["content"]
    wire = encode_summaries(summaries, binding, layout=layout)
    for key, rows in wire.items():
        rows[:] = [
            dict(copy.deepcopy(rows[0]), block_index=min(i, blocks - 1))
            for i in range(units)
        ]
        assert all(set(row) == {"block_index", "text", "body_refs"} for row in rows)
    Draft202012Validator(native["json_schema"]["schema"]).validate(wire)
    before = copy.deepcopy(wire)
    decoded = decode_summaries(wire, binding, body=body, layout=layout)
    combined = {
        "title": body["title"],
        "sections": {**body["sections"], **decoded["sections"]},
    }
    result, proof = project_grounded_response(json.dumps(combined), rf)
    canonical, _ = canonical_from_array(combined)
    assert json.loads(result) == canonical
    assert proof["scientific_verdict"] == "not_evaluated"
    assert wire == before
    assert body == split(analysis_document(cards, language))[0]
    if units == 1:
        assert render_analysis(canonical, cards, language)
    for key, section in decoded["sections"].items():
        restored = [unit for block in section["blocks"] for unit in block["sentences"]]
        assert [unit["text"] for unit in restored] == [row["text"] for row in wire[key]]
        assert all(
            b["heading"] == "" and b["type"] == "paragraph" for b in section["blocks"]
        )


@pytest.mark.parametrize(
    "fault",
    [
        "heading",
        "type",
        "third_block",
        "seven_units",
        "boolean",
        "starts_one",
        "backwards",
        "unknown_ref",
        "extra",
        "text_type",
        "mutated_body",
    ],
)
def test_layout_rejects_injections_limits_and_reference_drift(fault):
    _, body, summaries, _, _, binding, layout = material()
    wire = encode_summaries(summaries, binding, layout=layout)
    row = wire["abstract"][0]
    if fault in {"heading", "type", "extra"}:
        row[fault] = "injected"
    elif fault == "third_block":
        row["block_index"] = 2
    elif fault == "seven_units":
        wire["abstract"] *= 7
    elif fault == "boolean":
        row["block_index"] = True
    elif fault == "starts_one":
        row["block_index"] = 1
    elif fault == "backwards":
        wire["abstract"] = [copy.deepcopy(row) for _ in range(3)]
        wire["abstract"][1]["block_index"] = 1
    elif fault == "unknown_ref":
        row["body_refs"] = ["b999"]
    elif fault == "text_type":
        row["text"] = []
    else:
        body["title"] += " changed"
    with pytest.raises((ValueError, ValidationError)):
        decode_summaries(wire, binding, body=body, layout=layout)


@pytest.mark.parametrize("fault", ["rules", "heading", "type", "missing"])
def test_trusted_source_contract_cannot_silently_drift(fault):
    _, _, _, _, rf, _, _ = material()
    rules = copy.deepcopy(SUMMARY_RULES)
    if fault == "rules":
        rules["max_blocks"] = 7
    else:
        block = rf["json_schema"]["schema"]["properties"]["sections"]["properties"][
            "abstract"
        ]["properties"]["blocks"]["items"]["properties"]
        if fault == "missing":
            del block["heading"]
        else:
            block[fault]["const"] = "摘要" if fault == "heading" else "list"
    with pytest.raises(ValueError):
        summary_layout(rf, rules)


def test_legacy_142_shape_remains_a_failure_instead_of_being_sanitized():
    # Only observed layout metadata is replayed; no private response prose is public.
    _, body, summaries, messages, rf, binding, layout = material()
    wire = encode_summaries(summaries, binding)
    for key, count in (("abstract", 4), ("conclusion", 3)):
        wire[key] = [
            dict(
                copy.deepcopy(wire[key][0]),
                block_index=i,
                heading="摘要" if key == "abstract" else "结论",
            )
            for i in range(count)
        ]
    _, legacy_rf = prepare_summary_request(
        body, binding["body_unit_count"], messages, rf, short_refs=True
    )
    Draft202012Validator(legacy_rf["json_schema"]["schema"]).validate(wire)
    legacy = decode_summaries(wire, binding, body=body)
    raw = json.dumps(
        {"title": body["title"], "sections": {**body["sections"], **legacy["sections"]}}
    )
    with pytest.raises(ValueError, match="array_wire_contract_invalid") as error:
        project_grounded_response(raw, rf)
    assert isinstance(error.value.__cause__, ValidationError)
    assert error.value.__cause__.validator == "const"
    with pytest.raises(ValidationError):
        decode_summaries(wire, binding, body=body, layout=layout)
    with pytest.raises(ValueError, match="block_metadata_invalid"):
        encode_summaries(legacy, binding, layout=layout)


def test_old_paid_summary_executor_remains_closed_to_new_contract(tmp_path):
    from scripts import probe_frozen_summary as frozen

    assert frozen.EXPECTED_CONTRACT == "native-body-summary-v7"
    with pytest.raises(ValueError, match="migration_required"):
        frozen.prepare(tmp_path / "new", tmp_path / "parent", tmp_path / "tip")
