"""Keep EOF compilation separate from scientific validation and model repairs."""

import copy
import hashlib
import json

import pytest
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.llm.client import LLMResponse
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    WriterError,
    WriterSession,
    body_section_schema,
    intact_unit_footprint,
    normalize_eof_response,
    parse_response,
)


def chapter(case):
    """Return a fully populated chapter with a closed final scientific unit."""
    schema = body_section_schema("taxonomy", case.aliases)
    value = {"taxonomy": copy.deepcopy(case.body["taxonomy"])}
    return json.dumps(value, ensure_ascii=False), schema, value


def test_eof_closure_preserves_every_scientific_byte_and_binding(case):
    """Only final container delimiters may differ from the original response."""
    raw, schema, value = chapter(case)
    prefix = raw[:-2]
    with pytest.raises(WriterError, match="writer_json_invalid"):
        parse_response(prefix, schema)
    parsed, receipt = normalize_eof_response(prefix, schema)
    assert parsed == value
    assert receipt["suffix"] == "]}"
    assert receipt["policy"] == EOF_CONTAINER_POLICY
    assert receipt["candidate_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert receipt["raw_sha256"] == hashlib.sha256(prefix.encode()).hexdigest()
    assert receipt["units"] == intact_unit_footprint(prefix, schema)["units"]
    assert receipt["scientific_assessment"] == "not_verified"


@pytest.mark.parametrize(
    "fault",
    [
        "unfinished_string",
        "unfinished_escape",
        "missing_comma",
        "missing_colon",
        "wrong_closer",
        "duplicate_key",
        "unknown_ref",
        "unknown_field",
        "missing_paragraph",
        "incomplete_unit",
        "extra_container",
        "empty",
        "depth",
        "size",
    ],
)
def test_eof_closure_rejects_non_container_repairs(case, fault):
    """A valid prefix cannot authorize new scientific strings or invented layout."""
    raw, schema, value = chapter(case)
    prefix = raw[:-2]
    if fault == "unfinished_string":
        prefix = raw[: raw.index("function.") + 4]
    elif fault == "unfinished_escape":
        prefix = raw[: raw.index("function.") + 4] + "\\"
    elif fault == "missing_comma":
        prefix = prefix.replace(', "fact_card_refs"', ' "fact_card_refs"')
    elif fault == "missing_colon":
        prefix = prefix.replace('"text":', '"text"')
    elif fault == "wrong_closer":
        prefix += "}"
    elif fault == "duplicate_key":
        prefix = prefix.replace('"text":', '"text":"extra", "text":')
    elif fault == "unknown_ref":
        prefix = prefix.replace(
            value["taxonomy"][0]["sentences"][0]["fact_card_refs"][0], "outside"
        )
    elif fault == "unknown_field":
        prefix = prefix.replace('"text":', '"extra":"invented", "text":')
    elif fault == "missing_paragraph":
        prefix = json.dumps({"taxonomy": value["taxonomy"][0]["sentences"]})[:-2]
    elif fault == "incomplete_unit":
        prefix = raw[:-5]
    elif fault == "extra_container":
        prefix = "[" + prefix
    elif fault == "empty":
        prefix = ""
    elif fault == "depth":
        prefix = "[" * 65
    else:
        prefix = " " * 1000001
    with pytest.raises(WriterError):
        normalize_eof_response(prefix, schema)


def test_eof_closure_does_not_normalize_already_valid_json(case):
    """The normal parser remains the authority for original complete responses."""
    raw, schema, value = chapter(case)
    assert parse_response(raw, schema) == value
    with pytest.raises(WriterError, match="eof_ineligible"):
        normalize_eof_response(raw, schema)


@pytest.mark.asyncio
async def test_session_persists_raw_and_derived_candidate_without_a_repair(
    case, tmp_path
):
    """One received response yields one receipt and no semantic modification."""
    raw, schema, value = chapter(case)
    responses = []

    class Transport:
        async def chat(self, *args, **kwargs):
            responses.append(kwargs)
            return LLMResponse(
                content=raw[:-2],
                usage={
                    "usage_available": True,
                    "prompt_tokens": 20,
                    "completion_tokens": 10,
                },
                finish_reason="stop",
                model="test",
            )

    path = tmp_path / "repairs.json"
    session = WriterSession(
        Transport(),
        path,
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
    )
    result = await session.call("body:taxonomy", "write", {}, schema)
    assert result == value and len(responses) == 1
    assert session.state["attempts"] == []
    entry = json.loads(session.journal_path.read_text())["calls"]["body:taxonomy"]
    assert entry["raw"] == raw[:-2]
    assert entry["normalized_raw"] == raw
    assert entry["syntax_normalization"]["suffix"] == "]}"
    assert entry["original_syntax_failure"]["code"] == "writer_json_invalid"
    with pytest.raises(WriterError, match="no_replay"):
        await session.call("body:taxonomy", "write", {}, schema)
    restored = WriterSession(
        Transport(),
        path,
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
    )
    with pytest.raises(WriterError, match="no_replay"):
        await restored.call("body:taxonomy", "write", {}, schema)
    with pytest.raises(WriterError, match="identity_conflict"):
        WriterSession(Transport(), path, "r", case.packet, case.aliases)
    assert len(responses) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "name,stage",
    [
        ("reviewer", "reviewer"),
        ("repair_0", "synthesis"),
        ("summary_0", "summary"),
        ("body:madeup", "synthesis"),
    ],
)
async def test_non_initial_responses_never_receive_eof_closure(
    case, tmp_path, name, stage
):
    """Reusing an enabled journal never weakens judgment or semantic repair parsing."""
    raw, schema, _ = chapter(case)

    class Transport:
        async def chat(self, *args, **kwargs):
            return LLMResponse(
                content=raw[:-2],
                usage={
                    "usage_available": True,
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                },
                finish_reason="stop",
                model="test",
            )

    session = WriterSession(
        Transport(),
        tmp_path / "r",
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
    )
    with pytest.raises(WriterError, match="writer_json_invalid"):
        await session.call(name, "write", {}, schema, stage=stage)
    assert "normalized_raw" not in session.journal["calls"][name]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unknown_usage", "length", "error", "cancel"])
async def test_invalid_response_state_is_never_compiled(case, tmp_path, fault):
    """A syntactically repairable body cannot bypass transport and cost guards."""
    import asyncio

    raw, schema, _ = chapter(case)

    class Transport:
        async def chat(self, *args, **kwargs):
            if fault == "cancel":
                raise asyncio.CancelledError()
            response = LLMResponse(
                content=raw[:-2],
                usage={
                    "usage_available": fault != "unknown_usage",
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                },
                finish_reason="length" if fault == "length" else "stop",
                model="test",
            )
            if fault == "error":
                response.provider_error = "provider_failed"
            return response

    session = WriterSession(
        Transport(),
        tmp_path / "r",
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
    )
    with pytest.raises((WriterError, asyncio.CancelledError)):
        await session.call("body:taxonomy", "write", {}, schema)
    assert "syntax_normalization" not in session.journal["calls"]["body:taxonomy"]
    assert session.state["attempts"] == []


@pytest.mark.parametrize("policy", ["unknown", False, 1, {}])
def test_invalid_policy_rejected_without_creating_a_journal(case, tmp_path, policy):
    """Only the server-defined policy may change initial compilation semantics."""
    path = tmp_path / "r"
    with pytest.raises(WriterError, match="syntax_policy_invalid"):
        WriterSession(
            Provider(case), path, "r", case.packet, case.aliases, syntax_policy=policy
        )
    assert not path.exists()


@pytest.mark.parametrize("suffix", ["}", "]}"])
def test_only_complete_existing_units_can_enter_eof_compilation(case, suffix):
    """The outer array may be open, but every scientific unit must already close."""
    raw, schema, value = chapter(case)
    assert raw.endswith(suffix)
    parsed, receipt = normalize_eof_response(raw[: -len(suffix)], schema)
    assert parsed == value and receipt["suffix"] == suffix


@pytest.mark.asyncio
async def test_normalization_does_not_consume_or_expand_two_semantic_attempts(
    case, tmp_path
):
    """Two declined semantic requests are counted even after free local compilation."""
    raw, schema, _ = chapter(case)
    provider = Provider(case, decline=True)

    class Transport:
        async def chat(self, messages, **kwargs):
            if (
                "taxonomy"
                in kwargs["response_format"]["json_schema"]["schema"]["properties"]
            ):
                return LLMResponse(
                    content=raw[:-2],
                    usage={
                        "usage_available": True,
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                    },
                    finish_reason="stop",
                    model="test",
                )
            return await provider.chat(messages, **kwargs)

    session = WriterSession(
        Transport(),
        tmp_path / "r",
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
    )
    await session.call("body:taxonomy", "write", {}, schema)
    targets = {
        "methods:0:0": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
        }
    }
    for expected in (1, 2):
        with pytest.raises(WriterError, match="declined"):
            await session.patch(case.body, targets, cause="evaluation", claim_count=5)
        assert len(session.state["attempts"]) == expected
    with pytest.raises(WriterError, match="repair_limit"):
        await session.patch(case.body, targets, cause="evaluation", claim_count=5)
    assert len(provider.calls) == 2
