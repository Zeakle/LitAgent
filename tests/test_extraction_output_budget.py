"""Reject truncated extraction and reserve complete strict output before transport."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_observability import _tracer_with_mock
from test_strict_auto_recovery import extraction, paper

from litagent.agents.extraction_strategy import LLMStrategy, _parse_extraction_response
from litagent.agents.extractor import ExtractorWorker
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.observability.recorder import RedactingTraceHook
from litagent.orchestrator.task_graph import SubTask
from litagent.safety.budget import CostBudget


@pytest.mark.parametrize(
    "reason,content,tokens,code",
    [
        ("length", '{"value":', 8192, "strict_extraction_output_truncated"),
        ("length", "{}", 20, "strict_extraction_output_truncated"),
        ("content_filter", "{}", 20, "strict_extraction_output_incomplete"),
        ("tool_calls", "{}", 20, "strict_extraction_output_incomplete"),
        ("unknown", "{}", 20, "strict_extraction_output_incomplete"),
        (None, '{"value":', 2048, "strict_extraction_output_truncated_inferred"),
        (None, "{}", 2048, "strict_extraction_completion_unverified"),
        ("stop", '{"value":', 2048, "strict_extraction_json_invalid"),
        (None, '{"value":', 15, "strict_extraction_json_invalid"),
    ],
)
def test_invalid_or_unfinished_output_cannot_create_evidence(
    reason, content, tokens, code
):
    response = LLMResponse(
        content, usage={"completion_tokens": tokens}, finish_reason=reason
    )
    before = copy.deepcopy(response)
    with pytest.raises(ValueError, match=f"^{code}$"):
        _parse_extraction_response(
            response, max_tokens=8192 if reason == "length" else 2048, strict=True
        )
    assert response == before


@pytest.mark.parametrize("reason", [None, "stop"])
def test_large_complete_output_and_optional_historical_reason_remain_readable(reason):
    data = {"claim_records": [{"text": "Complete source-bound finding."}] * 40}
    response = LLMResponse(
        json.dumps(data), usage={"completion_tokens": 3000}, finish_reason=reason
    )
    assert _parse_extraction_response(response, max_tokens=8192, strict=True) == data
    response.finish_reason = "stop"
    response.usage["completion_tokens"] = 8192
    assert _parse_extraction_response(response, max_tokens=8192, strict=True) == data


def test_legacy_parsing_does_not_apply_new_finish_gate_or_repair_json():
    assert (
        _parse_extraction_response(
            LLMResponse("{}", finish_reason="length"), max_tokens=2048, strict=False
        )
        == {}
    )
    with pytest.raises(json.JSONDecodeError):
        _parse_extraction_response(
            LLMResponse('{"value":'), max_tokens=2048, strict=False
        )


def skills():
    """Supply only offline skill metadata."""
    return SimpleNamespace(to_metadata_text_for=lambda *args, **kwargs: "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "policy,expected", [("legacy", 1536), ("strict_v1", 8192), ("strict_auto_v1", 8192)]
)
async def test_actual_strategy_requests_strict_output_without_changing_legacy(
    policy, expected
):
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(json.dumps(extraction()), finish_reason="stop")
    result = await LLMStrategy(llm, skills(), max_tokens=1536).extract(paper(policy))
    assert result["claim_records"]
    llm.chat.assert_awaited_once()
    assert llm.chat.await_args.kwargs["max_tokens"] == expected
    if policy != "legacy":
        assert llm.chat.await_args.kwargs["transport_max_attempts"] == 1
    assert "EViT improves speed." in llm.chat.await_args.args[0][1]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["length", "content_filter", None])
async def test_actual_strategy_does_not_retry_or_salvage_truncated_response(reason):
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        '{"value":', usage={"completion_tokens": 8192}, finish_reason=reason
    )
    with pytest.raises(ValueError, match="strict_extraction_output_"):
        await LLMStrategy(llm, skills()).extract(paper())
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_strategy_preflight_cannot_borrow_writing_reserve_or_shrink_output():
    budget = CostBudget(500000)
    budget.configure_strict()
    budget.record({"prompt_tokens": 195000})
    llm = SimpleNamespace(_cost_budget=budget, chat=AsyncMock())
    with pytest.raises(ValueError, match="evaluation_run_budget_exceeded"):
        await LLMStrategy(llm, skills()).extract(paper())
    llm.chat.assert_not_awaited()
    assert budget.used == 195000 and budget.remaining() == 5000


@pytest.mark.asyncio
async def test_complete_strict_request_preserves_context_ceiling(monkeypatch):
    monkeypatch.setattr(
        "litagent.agents.extraction_strategy.build_extraction_context",
        lambda candidate: ("x" * 120000, ["abstract"]),
    )
    llm = AsyncMock()
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        await LLMStrategy(llm, skills()).extract(paper())
    llm.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["stop", "length", None])
async def test_sdk_completion_reason_usage_and_redacted_trace_consume_same_fields(
    reason,
):
    budget = CostBudget(500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    tracer, log = _tracer_with_mock()
    local = []
    redacted = []

    def sink(event, data):
        redacted.append((event, data))
        tracer._handle(event, data)

    hook = RedactingTraceHook(sink, "metadata_only")

    def record(event, data):
        local.append((event, data))
        hook(event, data)

    client = OpenAICompatibleClient(
        "https://example.test/v1",
        "offline",
        api_key="offline",
        cost_budget=budget,
        trace_hook=record,
    )
    real_sdk = client._client
    message = SimpleNamespace(
        content=json.dumps(extraction()), tool_calls=None, reasoning_content=None
    )
    choice = SimpleNamespace(message=message)
    if reason is not None:
        choice.finish_reason = reason
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[choice],
            usage=SimpleNamespace(prompt_tokens=25, completion_tokens=35),
            model="offline",
        )
    )
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    try:
        response = await client.chat(
            [{"role": "user", "content": "public source"}], max_tokens=8192
        )
    finally:
        await real_sdk.close()
    create.assert_awaited_once()
    assert response.finish_reason == reason
    assert budget.used == 60 and budget.snapshot()["requests_sent"] == 1
    assert budget.remaining() == 200000 - 60
    complete = next(d for event, d in local if event == "llm.complete")
    public = next(d for event, d in redacted if event == "llm.complete")
    assert complete["finish_reason"] == public["finish_reason"] == reason
    assert complete["max_tokens"] == public["max_tokens"] == 8192
    assert "content" not in public and "messages" not in public
    metadata = [d["metadata"] for event, d in log if event == "update"][-1]
    assert metadata["finish_reason"] == reason and metadata["max_tokens"] == 8192
    if reason == "length":
        with pytest.raises(ValueError, match="strict_extraction_output_truncated"):
            _parse_extraction_response(response, max_tokens=8192, strict=True)
        assert budget.used == 60


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["truncated", "cancelled"])
async def test_actual_extractor_stops_queued_transports_on_truncation_or_cancel(mode):
    budget = CostBudget(500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    create = AsyncMock(
        side_effect=asyncio.CancelledError() if mode == "cancelled" else None
    )
    client = OpenAICompatibleClient(
        "https://example.test/v1", "offline", api_key="offline", cost_budget=budget
    )
    real_sdk = client._client
    if mode == "truncated":
        create.return_value = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    finish_reason="length",
                    message=SimpleNamespace(
                        content="{}", tool_calls=None, reasoning_content=None
                    ),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=25, completion_tokens=35),
            model="offline",
        )
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    candidates = [{**paper(), "paper_id": str(i)} for i in range(3)]
    loader = AsyncMock()
    loader.enrich.return_value = candidates
    worker = ExtractorWorker(
        LLMStrategy(client, skills()),
        max_concurrent=1,
        fulltext_loader=loader,
        failure_callback=budget.stop_after_worker_failure,
    )
    task = SubTask(
        "extract",
        "offline",
        "extractor",
        input_data={
            "quality_policy": "strict_auto_v1",
            "upstream_results": {"relevance_gate": candidates},
        },
    )
    try:
        with pytest.raises(
            asyncio.CancelledError if mode == "cancelled" else ValueError
        ):
            await worker.execute(task)
    finally:
        await real_sdk.close()
    create.assert_awaited_once()
    assert budget.is_exceeded()
    if mode == "truncated":
        assert budget.used == 60 and budget.stop_reason == "strict_worker_failed"
    else:
        assert budget.stop_reason == "strict_usage_unknown" and budget.used > 8192


def test_strict_config_reports_actual_cap_and_preserves_legacy_summary():
    config = load_config()
    before = build_config_summary(config)
    assert before["effective"]["extractor"]["max_tokens"] == 2048
    assert "strict_max_output_tokens" not in before["effective"]["extractor"]
    for policy in ["strict_v1", "strict_auto_v1"]:
        strict = build_config_summary(config, policy)
        assert strict["effective"]["extractor"]["strict_max_output_tokens"] == 8192
        assert (
            "strict-extraction-output-v1"
            in strict["effective"]["quality_policy"]["contracts"]
        )
        assert strict["fingerprint"] != before["fingerprint"]
