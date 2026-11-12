"""Validate single-declaration block metadata without weakening scientific gates."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_body_analysis import analysis_document
from test_native_array_integration import request
from test_staged_native_generation import split
from test_writing_passages import facts

from litagent import staged_generation as staged
from litagent.native_wire import BODY
from litagent.quality import _writing_scopes
from litagent.report_audit import digest
from litagent.workbench import RunOptions


@pytest.fixture
def parts():
    """Keep the normal writer inputs and exact chapter permissions."""
    cards = facts()
    messages, rf = request(cards, array_wire=True)
    body, _ = split(analysis_document(cards))
    binding = staged.card_reference_binding(
        cards, _writing_scopes(cards, "academic_v2", "en", None)
    )
    prepared, body_rf = staged.prepare_body_request(messages, rf)
    return cards, body, binding, prepared, body_rf


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_directory_roundtrips_every_block_value_and_scientific_position(language):
    cards = facts()
    messages, rf = request(cards, language, array_wire=True)
    body, _ = split(analysis_document(cards, language))
    first = body["sections"]["methods"]["blocks"][0]
    first["sentences"].append(copy.deepcopy(first["sentences"][0]))
    second = copy.deepcopy(first)
    second["heading"] = "Conditions" if language == "en" else "实验条件"
    second["type"] = "list"
    body["sections"]["methods"]["blocks"].append(second)
    binding = staged.card_reference_binding(
        cards, _writing_scopes(cards, "academic_v2", language, None)
    )
    before = copy.deepcopy(body)
    wire = staged.encode_typed_body(body, binding)
    prepared, schema = staged.prepare_body_request(messages, rf)
    Draft202012Validator(schema["json_schema"]["schema"]).validate(wire)
    assert schema["json_schema"]["name"] == "grounded_body_v5_blocks"
    assert wire["block_layout"]["methods"] == [
        {"heading": b["heading"], "type": b["type"]}
        for b in body["sections"]["methods"]["blocks"]
    ]
    assert [r["block_index"] for r in wire["methods"]] == [0, 0, 1, 1]
    assert all("heading" not in r and "type" not in r for r in wire["methods"])
    assert staged.expand_body(staged.decode_typed_body(wire, binding)) == before
    restored, _ = staged.validate_body(
        json.dumps(wire), schema, cards, "academic_v2", language, None
    )
    assert restored == before == body
    proof = staged.body_row_projection(restored, binding, typed_units=True)
    assert proof["wire_document_sha256"] == digest(wire)
    position = next(p for p in proof["positions"] if p["wire_path"] == "methods.2")
    assert position["array_path"] == "sections.methods.blocks.1.sentences.0"
    packet = json.loads(prepared[1]["content"])
    assert packet["body_layout_contract"] == staged.BODY_LAYOUT_CONTRACT
    assert "exactly once in block_layout" in prepared[0]["content"]
    assert "must repeat the exact heading" not in prepared[0]["content"]


@pytest.mark.parametrize(
    "fault",
    [
        "missing_directory",
        "extra_section",
        "missing_section",
        "empty_directory",
        "unused_block",
        "duplicate_row_heading",
        "duplicate_row_type",
        "missing_heading",
        "extra_metadata",
        "heading_type",
        "block_type",
        "block_type_object",
        "boolean_index",
        "negative_index",
        "skip_index",
        "backward_index",
        "unknown_reference",
        "free_notice",
        "unknown_row_field",
    ],
)
def test_new_wire_fails_closed_on_ambiguity_and_loss(parts, fault):
    cards, body, binding, _, rf = parts
    wire = staged.encode_typed_body(body, binding)
    row = wire["methods"][0]
    metadata = wire["block_layout"]["methods"][0]
    if fault == "missing_directory":
        del wire["block_layout"]
    elif fault == "extra_section":
        wire["block_layout"]["abstract"] = [copy.deepcopy(metadata)]
    elif fault == "missing_section":
        del wire["block_layout"]["methods"]
    elif fault == "empty_directory":
        wire["block_layout"]["methods"] = []
    elif fault == "unused_block":
        wire["block_layout"]["methods"].append(copy.deepcopy(metadata))
    elif fault == "duplicate_row_heading":
        row["heading"] = metadata["heading"]
    elif fault == "duplicate_row_type":
        row["type"] = metadata["type"]
    elif fault == "missing_heading":
        del metadata["heading"]
    elif fault == "extra_metadata":
        metadata["text"] = "Unbound claim."
    elif fault == "heading_type":
        metadata["heading"] = []
    elif fault == "block_type":
        metadata["type"] = "table"
    elif fault == "block_type_object":
        metadata["type"] = {}
    elif fault == "boolean_index":
        row["block_index"] = False
    elif fault == "negative_index":
        row["block_index"] = -1
    elif fault == "skip_index":
        row["block_index"] = 1
        wire["block_layout"]["methods"].append(copy.deepcopy(metadata))
    elif fault == "backward_index":
        wire["block_layout"]["methods"].append(copy.deepcopy(metadata))
        wire["methods"] += [dict(row, block_index=1), copy.deepcopy(row)]
    elif fault == "unknown_reference":
        row["fact_card_refs"] = ["invented"]
    elif fault == "free_notice":
        wire["scope_notices"] = [
            {
                "section_key": "methods",
                "after_row": 0,
                "scope_notice_id": "review_scope",
                "text": "A field-wide gap.",
            }
        ]
    else:
        row["extra"] = "discard?"
    before = copy.deepcopy(wire)
    with pytest.raises((ValueError, ValidationError)):
        staged.validate_body(json.dumps(wire), rf, cards, "academic_v2", "en", None)
    assert wire == before


def test_same_titles_do_not_merge_distinct_blocks(parts):
    _, body, binding, _, _ = parts
    body["sections"]["methods"]["blocks"] *= 2
    wire = staged.encode_typed_body(body, binding)
    assert len(wire["block_layout"]["methods"]) == 2
    assert staged.expand_body(staged.decode_typed_body(wire, binding)) == body


def test_old_v4_contract_is_explicit_and_never_downgrades_new_wire(parts):
    cards, body, binding, _, current = parts
    messages, rf = request(cards, array_wire=True)
    _, legacy = staged.prepare_body_request(messages, rf, block_layout=False)
    old = staged.encode_typed_body(body, binding, block_layout=False)
    assert legacy["json_schema"]["name"] == "grounded_body_v4_typed"
    assert (
        staged.validate_body(json.dumps(old), legacy, cards, "academic_v2", "en", None)[
            0
        ]
        == body
    )
    with pytest.raises(ValidationError):
        staged.validate_body(json.dumps(old), current, cards, "academic_v2", "en", None)
    new = staged.encode_typed_body(body, binding)
    with pytest.raises(ValidationError):
        staged.validate_body(json.dumps(new), legacy, cards, "academic_v2", "en", None)


def test_actual_failure_projection_remains_a_rejection():
    fixture = json.loads(
        Path("tests/fixtures/body-metadata-conflict-145.json").read_text()
    )
    assert fixture["source_response_sha256"] == (
        "4cb2122c4e6951dcf27524601e821d0228a6095c6d029007670062884af43aba"
    )
    rows = fixture["redacted_flat_rows"]
    assert sum(row["kind"] == "scientific" for k in BODY for row in rows[k]) == 34
    assert sum(row["kind"] != "scientific" for k in BODY for row in rows[k]) == 2
    assert rows["methods"][0]["block_index"] == rows["methods"][1]["block_index"]
    assert rows["methods"][0]["heading"] != rows["methods"][1]["heading"]
    original = copy.deepcopy(rows)
    with pytest.raises(
        staged.BodyRowError, match="flat_body_block_metadata_conflict"
    ) as error:
        staged.expand_body(rows)
    assert error.value.absolute_path == ["methods", 1, "heading"]
    assert rows == original


def test_staged_identity_changes_without_changing_single_stage(monkeypatch):
    single = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    current = single.model_copy(update={"generation_strategy": "body_summary_v1"})
    single_sha, current_sha = single.fingerprint(), current.fingerprint()
    monkeypatch.setattr(staged, "CONTRACT", "native-body-summary-v9")
    assert single.fingerprint() == single_sha
    assert current.fingerprint() != current_sha
    cards = facts()
    messages, rf = request(cards, array_wire=True)
    packet, legacy = staged.prepare_body_request(messages, rf)
    assert legacy["json_schema"]["name"] == "grounded_body_v4_typed"
    assert "body_layout_contract" not in json.loads(packet[1]["content"])


@pytest.mark.parametrize("field", ["heading", "type"])
@pytest.mark.parametrize("value", [None, [], {}, False])
def test_direct_decoder_rejects_metadata_types_with_located_errors(parts, field, value):
    _, body, binding, _, _ = parts
    wire = staged.encode_typed_body(body, binding)
    wire["block_layout"]["methods"][0][field] = value
    with pytest.raises(staged.BodyRowError, match="body_layout_invalid") as error:
        staged.decode_typed_body(wire, binding)
    assert error.value.absolute_path == ["block_layout", "methods", 0]


def test_untyped_legacy_request_cannot_mix_block_directory(parts):
    cards, _, _, _, _ = parts
    messages, rf = request(cards, array_wire=True)
    with pytest.raises(ValueError, match="body_layout_requires_typed_units"):
        staged.prepare_body_request(messages, rf, typed_units=False, block_layout=True)
