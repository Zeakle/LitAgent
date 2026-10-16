"""Keep provider summary layout constraints identical to the host's rules."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_native_writing_schema import packet, section_definition
from test_strict_quality import to_native

from litagent.quality import (
    SUMMARY_RULES,
    prepare_grounded_request,
    render_grounded_report,
)


@pytest.mark.parametrize("key", ["abstract", "conclusion"])
@pytest.mark.parametrize("mode", ["heading", "list", "three_blocks"])
def test_native_summary_rejects_host_forbidden_layout(key, mode):
    cards, selection, report, messages = packet()
    native = to_native(report)
    block = native["sections"][key]["first_block"]
    if mode == "heading":
        block["heading"] = "Summary"
    elif mode == "list":
        block["type"] = "list"
    else:
        block["additional_sentences"] = []
        native["sections"][key]["additional_blocks"] = [copy.deepcopy(block)] * 2
    _, fmt = prepare_grounded_request(messages)
    assert list(Draft202012Validator(fmt["json_schema"]["schema"]).iter_errors(native))
    raw = json.dumps(native)
    with pytest.raises(ValueError, match="strict_writing_summary_layout_invalid"):
        render_grounded_report(
            raw,
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )
    assert json.dumps(native) == raw


@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_native_summary_rules_and_body_subheadings_remain_distinct(structure, language):
    cards, selection, report, messages = packet(structure, language)
    native = to_native(report)
    block = native["sections"]["methods"]["first_block"]
    block["heading"] = "Mechanism" if language == "en" else "机制分析"
    block["type"] = "list"
    _, fmt = prepare_grounded_request(messages)
    schema = fmt["json_schema"]["schema"]
    Draft202012Validator(schema).validate(native)
    for key in ("abstract", "conclusion"):
        definition = section_definition(schema, key, "block")["properties"]
        assert definition["heading"]["const"] == ""
        assert definition["type"]["const"] == SUMMARY_RULES["block_type"]
        continuation = schema["properties"]["sections"]["properties"][key][
            "properties"
        ]["additional_blocks"]
        assert len(continuation["prefixItems"]) == SUMMARY_RULES["max_blocks"] - 1
    body = render_grounded_report(
        json.dumps(native),
        cards,
        structure,
        language,
        evidence_selection=selection,
        require_native=True,
    )
    assert block["heading"] in body and body.count("[E:") == 7


def test_native_optional_blocks_consume_shared_limit_instead_of_copying_it(monkeypatch):
    cards, selection, report, messages = packet()
    monkeypatch.setitem(SUMMARY_RULES, "max_blocks", 1)
    # Rebuild the packet so the host summary policy and native producer share it.
    from litagent.quality import grounded_messages

    messages = grounded_messages(
        cards, "academic_v2", "en", "EViT", evidence_selection=selection
    )
    _, fmt = prepare_grounded_request(messages)
    native = to_native(report)
    native["sections"]["abstract"]["additional_blocks"] = [
        copy.deepcopy(native["sections"]["abstract"]["first_block"])
    ]
    assert list(Draft202012Validator(fmt["json_schema"]["schema"]).iter_errors(native))
    with pytest.raises(ValueError, match="strict_writing_summary_layout_invalid"):
        render_grounded_report(
            json.dumps(native),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )
