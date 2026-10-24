"""Exercise configured thinking through actual native producers and SDK requests."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import AdversarialConfig
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.writing_v2 import WriterError, WriterSession, review, schema_for


def assert_mode(kwargs, mode):
    """Provider default omits the parameter rather than sending a made-up type."""
    if mode == "provider_default":
        assert "extra_body" not in kwargs
    else:
        assert kwargs["extra_body"] == {"thinking": {"type": mode}}
    key = "max_tokens" if mode == "disabled" else "max_completion_tokens"
    other = "max_completion_tokens" if key == "max_tokens" else "max_tokens"
    assert key in kwargs and other not in kwargs


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["enabled", "disabled", "provider_default"])
async def test_actual_synthesis_and_reviewer_honor_independent_modes(
    case, tmp_path, mode
):
    """The shared journal freezes generation while Reviewer uses its own config."""
    provider = Provider(case)
    synthesis = SynthesisWorker(
        provider,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=case.selection)
        ),
        thinking_mode=mode,
        run_id_provider=lambda: "r",
        repair_state_path_provider=lambda: tmp_path / "r.json",
    )
    reviewer = ReviewerWorker(
        provider, config=AdversarialConfig(thinking_mode="disabled")
    )
    worker = AdversarialReviewWorker(provider, synthesis, reviewer, thinking_mode=mode)
    task = SubTask(
        "adversarial_review",
        "Offline flow",
        "adversarial_review",
        input_data={
            "query": case.packet["query"],
            "report_language": "en",
            "review_structure": "academic_v2",
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "generation_strategy": "body_summary_v2",
            "verification": {"judge": False, "jev": False},
            "upstream_results": {},
        },
    )
    result = await worker.execute(task)
    assert len(provider.calls) == 7
    for _, kwargs in provider.calls[:6]:
        assert_mode(kwargs, mode)
    assert_mode(provider.calls[-1][1], "disabled")
    assert result["writer_document"]["generation_thinking_mode"] == mode
    j = json.loads((tmp_path / "r.json.writer-v2.json").read_text())
    assert j["generation_thinking_mode"] == mode
    assert j["calls"]["summary_initial"]["thinking_mode"] == mode
    assert j["calls"]["reviewer"]["thinking_mode"] == "disabled"


@pytest.mark.asyncio
async def test_historical_document_without_mode_keeps_disabled_generation_identity(
    case, tmp_path
):
    """Missing historical metadata means old actual disabled behavior, not new config."""
    from test_native_report_input import source

    value = source(case)
    provider = Provider(case)
    session = WriterSession(
        provider, tmp_path / "r.json", "r", case.packet, case.aliases
    )
    session.save()
    worker = SimpleNamespace(
        _llm=provider,
        _writer_path=tmp_path / "r.json",
        _writer_run_id="r",
        _config=AdversarialConfig(thinking_mode="enabled"),
    )
    task = SubTask(
        "reviewer",
        "Historical review",
        "reviewer",
        input_data={
            "query": case.packet["query"],
            "upstream_results": {"synthesis": value},
        },
    )
    await review(worker, task)
    assert_mode(provider.calls[-1][1], "enabled")
    assert value["writer_document"].get("generation_thinking_mode") is None
    j = json.loads(session.journal_path.read_text())
    assert (
        j["generation_thinking_mode"] == "disabled"
        and j["calls"]["reviewer"]["thinking_mode"] == "enabled"
    )


@pytest.mark.parametrize("mode", ["invalid", None, False, {}, []])
def test_invalid_generation_mode_cannot_open_or_send(case, tmp_path, mode):
    """Reject invalid mode types before any durable attempt or HTTP can exist."""
    p = Provider(case)
    with pytest.raises(WriterError, match="thinking_mode_invalid"):
        WriterSession(
            p, tmp_path / "r.json", "r", case.packet, case.aliases, thinking_mode=mode
        )
    assert not p.calls and not (tmp_path / "r.json.writer-v2.json").exists()


def test_generation_mode_changes_checkpoint_identity_without_overwriting_old(
    case, tmp_path
):
    """A new requested reasoning policy cannot reopen an older native journal."""
    p = Provider(case)
    s = WriterSession(
        p, tmp_path / "r.json", "r", case.packet, case.aliases, thinking_mode="enabled"
    )
    s.save()
    before = s.journal_path.read_bytes()
    with pytest.raises(WriterError, match="checkpoint_identity_conflict"):
        WriterSession(
            p,
            tmp_path / "r.json",
            "r",
            case.packet,
            case.aliases,
            thinking_mode="disabled",
        )
    assert s.journal_path.read_bytes() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["enabled", "provider_default"])
async def test_summary_and_shared_repair_keep_frozen_mode(case, tmp_path, mode):
    """All Pro calls inherit the source mode without spending a separate repair pool."""
    p = Provider(case, decline=True)
    s = WriterSession(
        p, tmp_path / "r.json", "r", case.packet, case.aliases, thinking_mode=mode
    )
    await s.summaries(case.body, "summary_initial")
    targets = {
        "t": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
        }
    }
    with pytest.raises(WriterError, match="repair_declined"):
        await s.patch(case.body, targets, cause="evaluation", claim_count=5)
    assert len(p.calls) == 2 and len(s.state["attempts"]) == 1
    for _, kwargs in p.calls:
        assert_mode(kwargs, mode)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["enabled", "disabled", "provider_default"])
@pytest.mark.parametrize("fault", ["valid", "reason_only", "length"])
async def test_real_sdk_preserves_mode_schema_identity_and_counts_reasoning_once(
    case, tmp_path, mode, fault
):
    """Mock only HTTP; reasoning stays outside the scientific result and its prompts."""
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    sent = []
    events = []
    secret_thought = "PRIVATE_REASONING_NOT_EVIDENCE"

    def transport(request):
        payload = json.loads(request.content)
        sent.append(payload)
        if mode == "provider_default":
            assert "thinking" not in payload
        else:
            assert payload["thinking"] == {"type": mode}
        key = "max_tokens" if mode == "disabled" else "max_completion_tokens"
        other = "max_completion_tokens" if key == "max_tokens" else "max_tokens"
        assert payload["model"] == "pro" and payload[key] == 20000
        assert other not in payload
        assert payload["response_format"]["type"] == "json_schema"
        assert budget._reserved_tokens == strict_request_reservation(
            payload["messages"], 20000, payload["response_format"]
        )
        return httpx.Response(
            200,
            json={
                "id": "thinking-fixture",
                "object": "chat.completion",
                "created": 0,
                "model": "pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": (
                                "" if fault == "reason_only" else json.dumps(case.body)
                            ),
                            "reasoning_content": secret_thought,
                        },
                        "finish_reason": "length" if fault == "length" else "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                    "completion_tokens_details": {"reasoning_tokens": 70},
                },
            },
        )

    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "lite",
        api_key="offline",
        cost_budget=budget,
        trace_hook=lambda event, data: events.append((event, data)),
    )
    await owner.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    s = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode=mode,
    )
    try:
        if fault != "valid":
            with pytest.raises(
                WriterError,
                match="output_truncated" if fault == "length" else "json_invalid",
            ):
                await s.call(
                    "body",
                    "Write source-backed science.",
                    case.packet,
                    schema_for("body", case.aliases),
                )
        else:
            assert (
                await s.call(
                    "body",
                    "Write source-backed science.",
                    case.packet,
                    schema_for("body", case.aliases),
                )
                == case.body
            )
        entry = s.journal["calls"]["body"]
        extra = None if mode == "provider_default" else {"thinking": {"type": mode}}
        key = "max_tokens" if mode == "disabled" else "max_completion_tokens"
        limit = 20000 if mode == "disabled" else {key: 20000}
        assert entry["thinking_mode"] == mode and entry["input_sha256"] == digest(
            [sent[0]["messages"], sent[0]["response_format"], limit, extra]
        )
        assert entry["output_limit_parameter"] == key
        for event, data in events:
            if event in {"llm.start", "llm.complete"}:
                assert data["output_limit_parameter"] == key
                assert data["max_tokens"] == 20000
        assert entry["returned_reasoning_characters"] == len(secret_thought)
        assert secret_thought not in json.dumps(
            s.journal
        ) and secret_thought not in json.dumps(events)
        assert budget.used == 120 and budget.snapshot()["requests_sent"] == 1
        assert budget._reserved_tokens == 0 and len(sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_invalid_call_override_is_not_sent_or_charged(case, tmp_path):
    """Reviewer overrides cannot smuggle unsupported modes into provider kwargs."""
    p = Provider(case)
    p._cost_budget = CostBudget()
    s = WriterSession(p, tmp_path / "r.json", "r", case.packet, case.aliases)
    before = p._cost_budget.snapshot()
    with pytest.raises(WriterError, match="thinking_mode_invalid"):
        await s.call(
            "reviewer",
            "Review only.",
            {},
            schema_for("body", case.aliases),
            thinking_mode=False,
        )
    assert not p.calls and not s.state["attempts"]
    assert s.journal["calls"]["reviewer"]["status"] == "rejected_not_sent"
    assert s.journal["failure"]["request_sent"] is False
    assert p._cost_budget.snapshot() == before


@pytest.mark.asyncio
async def test_real_007_structure_stays_invalid_under_enabled_mode(case, tmp_path):
    """Configuring reasoning cannot accept or patch the historical malformed bytes."""
    from pathlib import Path

    from litagent.writing_v2 import body_section_schema

    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/native-open-problems-real-007.json"
        ).read_text()
    )
    assert (
        fixture["fixture_kind"]
        == "sanitized_protocol_structure_not_scientific_evidence"
    )
    p = Provider(case)
    original = p.chat

    async def transport(messages, **kwargs):
        result = await original(messages, **kwargs)
        result.content = fixture["raw"]
        return result

    p.chat = transport
    s = WriterSession(
        p, tmp_path / "r.json", "r", case.packet, case.aliases, thinking_mode="enabled"
    )
    with pytest.raises(WriterError, match="json_invalid"):
        await s.call(
            "body:open_problems",
            "Source-backed limitations only.",
            case.packet,
            body_section_schema("open_problems", case.aliases),
        )
    assert len(p.calls) == 1 and s.journal["failure"]["response_received"] is True
    assert s.journal["calls"]["body:open_problems"]["raw"] == fixture["raw"]
    assert_mode(p.calls[0][1], "enabled")
