"""Keep summary quality gates while using a provider-compatible array encoding."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_body_analysis import analysis_document, render_analysis
from test_writing_passages import facts

from litagent.body_quality import SUMMARY_ARRAY_WIRE_CONTRACT
from litagent.quality import (
    SUMMARY_RULES,
    GroundedWritingError,
    grounded_messages,
    prepare_grounded_request,
)
from litagent.workbench import RunOptions


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_analysis_array_wire_retains_required_units_and_host_limits(language):
    cards = facts()
    messages, response_format = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            language,
            "EViT speed",
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
    )
    packet = json.loads(messages[1]["content"])
    schema = response_format["json_schema"]["schema"]
    assert packet["summary_array_wire_contract"] == SUMMARY_ARRAY_WIRE_CONTRACT
    assert packet["summary_rules"]["max_blocks"] == SUMMARY_RULES["max_blocks"] == 2
    for key in ("abstract", "conclusion"):
        section = schema["properties"]["sections"]["properties"][key]
        assert section["required"] == ["first_block", "additional_blocks"]
        extra = section["properties"]["additional_blocks"]
        assert set(extra) == {"type", "items"} and extra["type"] == "array"
        assert extra["items"] == section["properties"]["first_block"]
    document = analysis_document(cards, language)
    Draft202012Validator(schema).validate(document)
    draft, bindings = render_analysis(document, cards, language)
    assert "[E:a]" in draft and bindings["summary_derivations"]


@pytest.mark.parametrize("section_key", ["abstract", "conclusion"])
def test_wire_permitted_excess_blocks_still_fail_host_summary_limit(section_key):
    cards = facts()
    _, response_format = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            "en",
            "EViT speed",
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
    )
    document = analysis_document(cards)
    section = document["sections"][section_key]
    block = copy.deepcopy(section["first_block"])
    variants = [
        "EViT improves speed by 50% on DeiT-S.",
        "On DeiT-S, EViT improves speed by 50%.",
        "The reported EViT speed improvement on DeiT-S is 50%.",
    ]
    blocks = []
    for text in variants:
        item = copy.deepcopy(block)
        item["first_sentence"]["text"] = text
        item["additional_sentences"] = []
        blocks.append(item)
    section.update(first_block=blocks[0], additional_blocks=blocks[1:])
    Draft202012Validator(response_format["json_schema"]["schema"]).validate(document)
    before = copy.deepcopy(document)
    with pytest.raises(GroundedWritingError) as failed:
        render_analysis(document, cards)
    assert any(
        d["reason_code"] == f"strict_writing_summary_layout_invalid:{section_key}"
        for d in failed.value.diagnostics
    )
    assert document == before


def test_non_analysis_native_schema_keeps_legacy_array_contract():
    messages, response_format = prepare_grounded_request(
        grounded_messages(
            facts(),
            "academic_v2",
            "en",
            "EViT speed",
            use_full_sources=True,
            use_scope_notices=True,
        )
    )
    assert "summary_array_wire_contract" not in json.loads(messages[1]["content"])
    for key in ("abstract", "conclusion"):
        section = response_format["json_schema"]["schema"]["properties"]["sections"][
            "properties"
        ][key]
        assert section["properties"]["additional_blocks"]["items"] is False
        assert len(section["properties"]["additional_blocks"]["prefixItems"]) == 1


def test_new_wire_version_changes_only_analysis_run_identity(monkeypatch):
    import litagent.body_quality as body

    legacy = RunOptions()
    analysis = RunOptions(
        quality_policy="strict_auto_v2",
        review_structure="academic_v2",
        body_quality="analysis_v1",
    )
    old_legacy, old_analysis = legacy.fingerprint(), analysis.fingerprint()
    monkeypatch.setattr(
        body, "SUMMARY_ARRAY_WIRE_CONTRACT", "summary-array-items-future"
    )
    assert legacy.fingerprint() == old_legacy
    assert analysis.fingerprint() != old_analysis


@pytest.mark.parametrize(
    "key,error_key",
    [
        ("survey_native", "survey_json_error"),
        ("original_schema_probe", "original_probe_error"),
    ],
)
def test_received_invalid_json_is_not_repaired_or_given_a_scientific_score(
    key, error_key
):
    fixture = json.loads(
        Path("tests/fixtures/native-schema-json-failure-real.json").read_text()
    )
    original = copy.deepcopy(fixture)
    with pytest.raises(json.JSONDecodeError) as failed:
        json.loads(fixture[key])
    assert failed.value.pos == fixture[error_key]["position"]
    assert fixture == original
    valid_probe = json.loads(fixture["ordinary_array_probe"])
    assert valid_probe["title"] == "PROTOCOL_PROBE"
    # A provider protocol fixture has no evaluated facts and cannot be a report.
    with pytest.raises(ValueError):
        render_analysis(valid_probe, facts())
