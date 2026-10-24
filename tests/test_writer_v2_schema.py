"""Keep provider compatibility independent from authoritative host validation."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.llm.client import LLMResponse
from litagent.writing_v2 import (
    PROVIDER_SCHEMA_CONTRACT,
    WriterError,
    WriterSession,
    body_units,
    parse_response,
    provider_response_format,
    schema_for,
)


def structural_schema(node):
    """Compare field ownership separately from locally enforced cardinality."""
    omitted = {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "uniqueItems",
        "description",
    }
    result = {k: copy.deepcopy(v) for k, v in node.items() if k not in omitted}
    if "properties" in node:
        result["properties"] = {
            k: structural_schema(v) for k, v in node["properties"].items()
        }
    if "items" in node:
        result["items"] = structural_schema(node["items"])
    return result


@pytest.mark.parametrize("stage", ["body", "summary", "repair"])
def test_provider_projection_keeps_exact_closed_shape_and_enums(case, stage):
    """No field, required member, type or source permission changes in transit."""
    host = schema_for(
        stage, body_units(case.body) if stage == "summary" else case.aliases, ["target"]
    )
    before = copy.deepcopy(host)
    native = provider_response_format(stage, host)
    assert host == before
    assert native["json_schema"]["strict"] is True
    assert structural_schema(native["json_schema"]["schema"]) == structural_schema(host)
    serialized = json.dumps(native)
    assert '"description"' in serialized
    assert '"minItems":' not in serialized and '"uniqueItems":' not in serialized


def test_keyword_named_property_is_data_and_is_not_removed():
    """A field called minItems must survive the schema traversal."""
    host = {
        "type": "object",
        "properties": {"minItems": {"type": "string", "minLength": 2}},
        "required": ["minItems"],
        "additionalProperties": False,
    }
    native = provider_response_format("body", host)["json_schema"]["schema"]
    assert native["required"] == ["minItems"]
    assert native["properties"]["minItems"]["type"] == "string"
    assert "minLength" in native["properties"]["minItems"]["description"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["empty_section", "empty_refs", "duplicate_refs", "long_text", "long_title"],
)
async def test_host_rejects_business_constraint_violations_after_native_acceptance(
    case, tmp_path, fault
):
    """Removing a native decoder constraint never removes its host delivery check."""
    body = copy.deepcopy(case.body)
    unit = body["methods"][0]["sentences"][0]
    if fault == "empty_section":
        body["methods"] = []
    elif fault == "empty_refs":
        unit["fact_card_refs"] = []
    elif fault == "duplicate_refs":
        unit["fact_card_refs"] *= 2
    elif fault == "long_text":
        unit["text"] = "x" * 2001
    else:
        body["title"] = "x" * 301
    host = schema_for("body", case.aliases)
    native = provider_response_format("synthesis", host)
    assert Draft202012Validator(native["json_schema"]["schema"]).is_valid(body)
    provider = Provider(case)
    provider.chat = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps(body),
            model="offline",
            finish_reason="stop",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "usage_available": True,
            },
        )
    )
    session = WriterSession(
        provider, tmp_path / "r.json", "r", case.packet, case.aliases
    )
    with pytest.raises(WriterError, match="schema_invalid"):
        await session.call("body", "system", case.packet, host)
    assert provider.chat.await_args.kwargs["response_format"] == native
    entry = session.journal["calls"]["body"]
    assert entry["provider_schema_contract"] == PROVIDER_SCHEMA_CONTRACT
    assert entry["raw"] == json.dumps(body)
    assert entry["schema_sha256"] != entry["provider_schema_sha256"]
    with pytest.raises(WriterError, match="no_replay"):
        await session.call("body", "system", case.packet, host)
    assert provider.chat.await_count == 1


@pytest.mark.asyncio
async def test_unknown_native_keyword_is_journaled_without_http(case, tmp_path):
    """Future constraints need a reviewed mapping rather than automatic omission."""
    host = schema_for("body", case.aliases)
    host["properties"]["title"]["pattern"] = "^review"
    provider = Provider(case)
    session = WriterSession(
        provider, tmp_path / "r.json", "r", case.packet, case.aliases
    )
    with pytest.raises(WriterError, match="provider_schema_keyword_invalid"):
        await session.call("body", "system", case.packet, host)
    assert not provider.calls
    entry = session.journal["calls"]["body"]
    assert entry["status"] == "rejected_not_sent"
    assert entry["failure"]["code"] == "writer_provider_schema_keyword_invalid"
    assert session.journal["failure"]["request_sent"] is False
    assert session.journal["failure"]["stage"] == "synthesis"


def test_original_malformed_json_stays_invalid_under_full_host_schema(case):
    """The transport projection never repairs or accepts truncated response bytes."""
    raw = json.dumps(case.body)[:-1]
    with pytest.raises(WriterError, match="json_invalid"):
        parse_response(raw, schema_for("body", case.aliases))
