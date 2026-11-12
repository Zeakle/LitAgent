"""Verify Ark admission, cooldown and provider rejection diagnostics."""

import asyncio
import time

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI

import litagent.llm.client as transport
from litagent.llm.client import (
    OpenAICompatibleClient,
    ProviderResponseError,
    RequestDeadlineExceeded,
)
from litagent.observability.context import deadline_scope
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.safety.budget import CostBudget

ARK = "https://ark.cn-beijing.volces.com/api/v3"
MESSAGES = [{"role": "user", "content": "offline"}]


async def offline_client(base, handler, *, budget=None, trace=None):
    """Keep the real SDK and replace only HTTP."""
    client = OpenAICompatibleClient(
        base,
        "offline",
        api_key="offline",
        cost_budget=budget,
        trace_hook=trace,
        request_timeouts={"default": 10},
    )
    await client.close()
    client._client = AsyncOpenAI(
        base_url=base,
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api", "status", "code"),
    [
        ("responses", 403, "AccountOverdueError"),
        ("responses", 429, "RequestBurstTooFast"),
        ("chat", 429, "RequestBurstTooFast"),
    ],
)
async def test_responses_rejection_has_public_code_and_keeps_unknown_hold(
    monkeypatch, tmp_path, api, status, code
):
    """A known provider rejection does not silently settle unknown billing."""
    pacer = transport._ArkPacer(interval=0, cooldown=60)
    monkeypatch.setattr(transport, "_ark_pacer", pacer)
    sent = []
    recorder = RunRecorder("offline", "offline", ArchiveRepository(tmp_path))
    budget = CostBudget(500000)
    budget.configure_strict(halt_on_unknown_usage=True)

    def handler(request):
        sent.append(request)
        return httpx.Response(
            status,
            headers={"x-request-id": "offline-rejection-42"},
            json={"error": {"code": code, "message": "PRIVATE_PROVIDER_MESSAGE"}},
        )

    client = await offline_client(ARK, handler, budget=budget, trace=recorder)
    try:
        if api == "responses":
            with pytest.raises(ProviderResponseError, match="strict_usage_incomplete"):
                await client.chat(
                    MESSAGES,
                    provider_api=api,
                    max_output_tokens=200,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "fixture",
                            "strict": True,
                            "schema": {"type": "object", "additionalProperties": False},
                        },
                    },
                )
        else:
            with pytest.raises(APIStatusError):
                await client.chat(MESSAGES)
        assert len(sent) == 1
        assert budget.used > 0 and budget.stop_reason == "strict_usage_unknown"
        assert recorder.provider_failures() == [
            {
                "task_id": "",
                "provider_http_status": status,
                "provider_error_code": code,
                "provider_request_id": "offline-rejection-42",
                "usage_available": False,
            }
        ]
        assert "PRIVATE_PROVIDER_MESSAGE" not in repr(recorder.provider_failures())
        assert (pacer._next_start > time.monotonic() + 50) is (status == 429)
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_ark_clients_share_spacing_and_leave_other_hosts_unpaced(monkeypatch):
    """Concurrent clients must share the same send interval across models."""
    pacer = transport._ArkPacer(interval=0.025)
    monkeypatch.setattr(transport, "_ark_pacer", pacer)
    sent, starts = [], []

    def trace(event, data):
        if event == "llm.start":
            starts.append(time.monotonic())

    def handler(request):
        sent.append(request)
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "offline",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "ok"},
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    first = await offline_client(ARK, handler, trace=trace)
    second = await offline_client(ARK, handler, trace=trace)
    other = await offline_client("https://offline.invalid/v1", handler)
    try:
        assert first._pacer is second._pacer is pacer
        assert other._pacer is None
        await asyncio.gather(first.chat(MESSAGES), second.chat(MESSAGES))
        assert len(sent) == len(starts) == 2 and starts[1] - starts[0] >= 0.020
    finally:
        await asyncio.gather(first.close(), second.close(), other.close())


@pytest.mark.asyncio
async def test_cooldown_respects_deadline_before_budget_reservation(monkeypatch):
    """A queue wait beyond the task deadline consumes no provider request."""
    pacer = transport._ArkPacer()
    pacer.defer_after_429()
    monkeypatch.setattr(transport, "_ark_pacer", pacer)
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)

    def handler(request):
        pytest.fail("request should not reach the provider")

    client = await offline_client(ARK, handler, budget=budget)
    try:
        async with deadline_scope(30):
            with pytest.raises(
                RequestDeadlineExceeded, match="request_deadline_insufficient"
            ):
                await client.chat(MESSAGES)
        assert budget.used == 0 and budget._reserved_tokens == 0
        assert budget.stop_reason is None
    finally:
        await client.close()
