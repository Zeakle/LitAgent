"""Check total-output limits through the actual SDK and shared strict budget."""

import json

import httpx
import pytest
from openai import AsyncOpenAI

from litagent.llm.client import OpenAICompatibleClient
from litagent.safety.budget import CostBudget, strict_request_reservation


async def client(tmp_path, *, allowance=500000, fault=None):
    """Own an offline SDK transport, keeping all reservation behavior real."""
    budget = CostBudget(max_tokens=allowance)
    budget.configure_strict(halt_on_unknown_usage=True)
    sent, events = [], []

    def transport(request):
        body = json.loads(request.content)
        sent.append(body)
        assert budget._reserved_tokens == strict_request_reservation(
            body["messages"], 2048
        )
        if fault == "timeout":
            raise httpx.ReadTimeout("offline timeout", request=request)
        return httpx.Response(
            200,
            json={
                "id": "bounded-completion",
                "object": "chat.completion",
                "created": 0,
                "model": "pro",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": (
                    None
                    if fault == "missing_usage"
                    else {
                        "prompt_tokens": 30,
                        "completion_tokens": 80,
                        "total_tokens": 110,
                        "completion_tokens_details": {"reasoning_tokens": 60},
                    }
                ),
            },
        )

    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "pro",
        api_key="offline",
        max_tokens=20000,
        cost_budget=budget,
        trace_hook=lambda e, d: events.append((e, d)),
    )
    await owner.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    return owner, budget, sent, events


@pytest.mark.asyncio
async def test_total_output_limit_replaces_default_answer_cap_and_reserves_exactly(
    tmp_path,
):
    """A smaller explicit total cap must reach HTTP without a default max_tokens."""
    owner, budget, sent, events = await client(tmp_path)
    try:
        await owner.chat(
            [{"role": "user", "content": "source"}], max_completion_tokens=2048
        )
        assert len(sent) == 1 and sent[0]["max_completion_tokens"] == 2048
        assert "max_tokens" not in sent[0]
        assert budget.used == 110 and budget._reserved_tokens == 0
        assert budget.snapshot()["requests_sent"] == 1
        assert all(
            data["output_limit_parameter"] == "max_completion_tokens"
            for event, data in events
            if event in {"llm.start", "llm.complete"}
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_completion_tokens": 2048, "max_tokens": 20000},
        {"max_completion_tokens": 0},
        {"max_completion_tokens": -1},
        {"max_completion_tokens": True},
        {"max_completion_tokens": None},
        {"max_completion_tokens": 1.5},
        {"max_completion_tokens": "2048"},
    ],
)
async def test_invalid_total_cap_rejected_before_attempt_or_unknown_charge(
    tmp_path, kwargs
):
    """Local argument errors must not start or charge a provider request."""
    owner, budget, sent, events = await client(tmp_path)
    before = budget.snapshot()
    try:
        with pytest.raises(ValueError, match="output_limit"):
            await owner.chat([{"role": "user", "content": "source"}], **kwargs)
        assert not sent and not events and budget.snapshot() == before
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_full_total_cap_budget_refusal_happens_before_transport(tmp_path):
    """A valid total cap still consumes the complete strict stage reservation."""
    owner, budget, sent, events = await client(tmp_path, allowance=7000)
    try:
        with pytest.raises(ValueError, match="budget"):
            await owner.chat(
                [{"role": "user", "content": "source"}], max_completion_tokens=2048
            )
        assert not sent and not events
        assert budget.used == budget.snapshot()["requests_sent"] == 0
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["timeout", "missing_usage"])
async def test_unknown_response_keeps_full_total_cap_hold_without_retry(
    tmp_path, fault
):
    """An uncertain paid response cannot refund its reasoning-inclusive reservation."""
    owner, budget, sent, events = await client(tmp_path, fault=fault)
    messages = [{"role": "user", "content": "source"}]
    reserved = strict_request_reservation(messages, 2048)
    try:
        from openai import APITimeoutError

        expected = APITimeoutError if fault == "timeout" else ValueError
        with pytest.raises(expected):
            await owner.chat(messages, max_completion_tokens=2048)
        assert len(sent) == budget.snapshot()["requests_sent"] == 1
        assert budget.used == reserved and budget._reserved_tokens == 0
        assert budget.stop_reason == "strict_usage_unknown"
        with pytest.raises(ValueError, match="strict_usage_unknown"):
            await owner.chat(messages, max_completion_tokens=2048)
        assert budget.used == reserved and len(sent) == 1
        assert not any(event == "llm.retry" for event, _ in events)
    finally:
        await owner.close()


def test_output_cap_metadata_survives_redaction_and_langfuse():
    """Only the safe output field name and numeric cap leave the local recorder."""
    from unittest.mock import MagicMock

    from litagent.observability.recorder import RedactingTraceHook
    from litagent.observability.tracing import LangFuseTracer

    tracer = LangFuseTracer(host="", public_key="", secret_key="")
    tracer._client, tracer._root = object(), MagicMock()
    hook = RedactingTraceHook(tracer, payload_mode="metadata_only")
    data = {
        "operation_id": "cap",
        "model": "pro",
        "max_tokens": 20000,
        "output_limit_parameter": "max_completion_tokens",
        "messages": ["PRIVATE"],
        "content": "PRIVATE",
        "reasoning_content": "PRIVATE",
        "usage_available": True,
        "prompt_tokens": 30,
        "completion_tokens": 80,
        "total_tokens": 110,
    }
    hook("llm.start", data)
    generation = tracer._root.start_observation.return_value
    hook("llm.complete", data)
    metadata = generation.update.call_args.kwargs["metadata"]
    assert metadata["output_limit_parameter"] == "max_completion_tokens"
    assert metadata["max_tokens"] == 20000
    assert generation.update.call_args.kwargs["usage_details"]["total_tokens"] == 110
    assert "PRIVATE" not in str(tracer._root.mock_calls)
