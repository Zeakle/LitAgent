"""Keep array prompts unambiguous without weakening source or scientific checks."""

import hashlib
import json
from pathlib import Path

import pytest
from test_writing_passages import facts

from litagent.body_quality import NATIVE_PROMPT_COMPOSITION_CONTRACT
from litagent.native_wire import array_wire_schema, inline_local_schema
from litagent.quality import (
    grounded_messages,
    prepare_grounded_request,
    project_grounded_response,
)
from litagent.workbench import RunOptions


def messages(language, draft=""):
    """Exercise the shared producer used by first generation and revisions."""
    return grounded_messages(
        facts(),
        "academic_v2",
        language,
        "Explain EViT on DeiT-S",
        draft=draft,
        feedback="Preserve the original conditions." if draft else "",
        use_full_sources=True,
        use_scope_notices=True,
        analysis_mode=True,
        require_unit_citation_scope=True,
    )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("draft", ["", "Current draft [E:a]"])
def test_array_prompt_has_one_analysis_contract_and_one_generation_order(
    language, draft
):
    original = messages(language, draft)
    frozen = json.loads(json.dumps(original))
    prepared, _ = prepare_grounded_request(original, array_wire=True)
    text = prepared[0]["content"]
    assert text.count("BODY ANALYSIS CONTRACT body-analysis-v1:") == 1
    assert text.count("SOURCE APPLICABILITY CONTRACT source-applicability-v1:") == 1
    assert "Use supplied section_keys in order." not in text
    assert "Use generation_section_order for output" in text
    assert "host's final display order only" in text
    assert "all summary science is independently verified" in text
    assert "sources[source_ref].source_quote in full" in text
    assert original == frozen
    old, _ = prepare_grounded_request(original)
    assert old[0]["content"].count("BODY ANALYSIS CONTRACT body-analysis-v1:") == 2
    assert "Use supplied section_keys in order." in old[0]["content"]


def test_prompt_change_preserves_complete_packet_and_schema_permissions():
    original = messages("zh-CN")
    old, old_format = prepare_grounded_request(original)
    new, new_format = prepare_grounded_request(original, array_wire=True)
    old_packet = json.loads(old[1]["content"])
    new_packet = json.loads(new[1]["content"])
    additions = {
        "native_wire_contract",
        "native_schema_encoding_contract",
        "native_prompt_composition_contract",
    }
    assert {k: v for k, v in new_packet.items() if k not in additions} == old_packet
    assert (
        new_packet["native_prompt_composition_contract"]
        == NATIVE_PROMPT_COMPOSITION_CONTRACT
    )
    assert new_format["json_schema"]["strict"] is True
    expected = array_wire_schema(old_format["json_schema"]["schema"])
    for definition in expected["$defs"].values():
        references = definition.get("properties", {}).get("body_references", {})
        properties = references.get("items", {}).get("properties", {})
        for key, description in (
            ("block_index", "Use the zero-based blocks array index."),
            ("sentence_index", "Use the zero-based sentences array index."),
        ):
            if key in properties:
                properties[key]["description"] = description
    assert new_format["json_schema"]["schema"] == inline_local_schema(expected)


def test_new_prompt_identity_does_not_change_legacy_fingerprint(monkeypatch):
    import litagent.body_quality as body

    legacy = RunOptions()
    strict = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    a, b = legacy.fingerprint(), strict.fingerprint()
    monkeypatch.setattr(body, "NATIVE_PROMPT_COMPOSITION_CONTRACT", "future-prompt")
    assert legacy.fingerprint() == a and strict.fingerprint() != b


def test_latest_real_missing_quote_still_fails_at_exact_original_offset():
    fixture = json.loads(
        Path("tests/fixtures/native-inline-production-failure-real.json").read_text()
    )
    raw = fixture["raw_response"]
    before = hashlib.sha256(raw.encode()).hexdigest()
    events = []
    _, response_format = prepare_grounded_request(messages("zh-CN"), array_wire=True)
    with pytest.raises(ValueError, match="array_wire_contract_invalid") as failure:
        project_grounded_response(
            raw, response_format, trace_hook=lambda *args: events.append(args)
        )
    error = failure.value.__cause__
    assert isinstance(error, json.JSONDecodeError)
    assert (error.pos, error.lineno, error.colno) == (8287, 1, 8288)
    assert hashlib.sha256(raw.encode()).hexdigest() == before == fixture["raw_sha256"]
    assert events[-1][0] == "writing.wire.failed"
    assert events[-1][1]["candidate_sha256"] == before
    assert events[-1][1]["json_position"] == 8287
    assert not any(event == "writing.wire.projected" for event, _ in events)
