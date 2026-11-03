"""Verify configurable deadlines at real graph and SDK boundaries without network."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_native_array_integration import request
from test_staged_native_generation import provider_body, provider_summaries, split
from test_writing_passages import facts

from litagent.agents.planner import SurveyPlanner
from litagent.config import AppConfig, LLMConfig, OrchestratorConfig, load_config
from litagent.contracts import build_config_summary
from litagent.llm.client import (
    OpenAICompatibleClient,
    RequestDeadlineExceeded,
    chat_with_timeout,
)
from litagent.llm.routing import ModelBoundClient
from litagent.observability import context
from litagent.observability.context import deadline_scope, remaining_deadline_seconds
from litagent.orchestrator.scheduler import Scheduler, Worker
from litagent.orchestrator.task_graph import SubTask, TaskGraph, TaskStatus
from litagent.runner import Infra, LitAgent
from litagent.safety.budget import CostBudget
from litagent.staged_generation import generate_staged_report


async def client_with_transport(handler, *, timeouts=None):
    """Replace only HTTP transport while keeping preparation and accounting real."""
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    events = []
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "offline",
        api_key="offline",
        cost_budget=budget,
        transport_max_attempts=1,
        request_timeouts=timeouts or {"default": 180, "synthesis": 300, "summary": 180},
        trace_hook=lambda e, d: events.append((e, d)),
    )
    await client._client.close()
    client._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client, budget, events


def response(content="{}"):
    """Return complete usage so successful local calls never create unknown holds."""
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
                    "message": {"role": "assistant", "content": content},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        },
    )


@pytest.mark.asyncio
async def test_default_timeouts_reach_actual_runner_graph(monkeypatch):
    cfg = load_config()
    monkeypatch.setenv("LLM_API_KEY", "offline-test-key")
    cfg.observability.enabled = False
    cfg.mcp_servers = {}
    agent = LitAgent(cfg)
    monkeypatch.setattr(agent, "_connect_infra", AsyncMock(return_value=Infra()))
    await agent._wire()
    try:
        graph = await agent._planner.plan("arxiv:2010.11929")
        expected = {
            "search": 120000,
            "recall": 120000,
            "dedup": 120000,
            "relevance_gate": 180000,
            "graph": 120000,
            "extractor": 1200000,
            "adversarial_review": 1800000,
        }
        for task in graph.tasks.values():
            assert task.timeout_ms == expected[task.agent_type]
        assert agent._scheduler._timeout_ms == 4200000
        assert agent._llm._request_timeouts == cfg.llm.request_timeouts_seconds
        assert agent._extraction_strategy._automatic_timeout_seconds == 210
    finally:
        await agent.cleanup()


@pytest.mark.asyncio
async def test_omitted_configuration_preserves_legacy_graph_limits():
    graph = await SurveyPlanner().plan("arxiv:2010.11929")
    assert graph.get_task("extract").timeout_ms == 300000
    assert graph.get_task("adversarial_review").timeout_ms == 720000
    assert graph.get_task("relevance_gate").timeout_ms == 120000


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), float("inf"), 601, "180"])
def test_invalid_request_timeout_values_are_rejected(value):
    with pytest.raises(ValueError):
        LLMConfig(request_timeouts_seconds={"synthesis": value})


@pytest.mark.parametrize("key", ["generation", "arbitrary", "endpoint"])
def test_unknown_timeout_stage_is_rejected(key):
    with pytest.raises(ValueError):
        LLMConfig(request_timeouts_seconds={key: 180})
    with pytest.raises(ValueError):
        OrchestratorConfig(task_timeouts_ms={key: 1000})


@pytest.mark.parametrize("fault", ["strategy", "worker", "evaluation", "total", "dag"])
def test_inconsistent_parent_deadlines_are_rejected(fault):
    data = load_config().model_dump()
    if fault == "strategy":
        data["extractor"]["automatic_timeout_seconds"] = 185
    elif fault == "worker":
        data["extractor"]["task_timeout_ms"] = 200000
    elif fault == "evaluation":
        data["llm"]["request_timeouts_seconds"]["evaluation"] = 300
    elif fault == "total":
        data["eval"]["total_timeout_seconds"] = 100
    else:
        data["orchestrator"]["timeout_ms"] = 900000
    with pytest.raises(ValueError, match="timeout"):
        AppConfig.model_validate(data)


def test_request_timeout_changes_resume_identity_without_changing_model():
    cfg = load_config()
    before = build_config_summary(cfg, "strict_auto_v2")
    cfg.llm.request_timeouts_seconds["synthesis"] = 310
    after = build_config_summary(cfg, "strict_auto_v2")
    assert before["fingerprint"] != after["fingerprint"]
    assert before["effective"]["llm"]["model"] == after["effective"]["llm"]["model"]


@pytest.mark.asyncio
async def test_nested_deadlines_inherit_earliest_and_restore_after_failure(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(context, "monotonic", lambda: now[0])
    assert remaining_deadline_seconds() is None
    async with deadline_scope(300):
        now[0] = 295.321
        async with deadline_scope(210):
            assert remaining_deadline_seconds() == pytest.approx(4.679)
        with pytest.raises(RuntimeError):
            async with deadline_scope(1):
                assert remaining_deadline_seconds() == 1
                raise RuntimeError("offline")
        assert remaining_deadline_seconds() == pytest.approx(4.679)
    assert remaining_deadline_seconds() is None


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_seconds, sent", [(300, False), (1200, True)])
async def test_last_extraction_window_is_admitted_only_when_it_fits(
    monkeypatch, worker_seconds, sent
):
    now, calls = [0.0], []
    monkeypatch.setattr(context, "monotonic", lambda: now[0])

    def transport(req):
        calls.append(req)
        return response()

    client, budget, events = await client_with_transport(transport)
    try:
        async with deadline_scope(worker_seconds):
            now[0] = 295.321
            async with deadline_scope(210):
                if sent:
                    await chat_with_timeout(
                        client,
                        [{"role": "user", "content": "offline"}],
                        stage="extraction",
                        max_tokens=100,
                    )
                else:
                    with pytest.raises(RequestDeadlineExceeded):
                        await chat_with_timeout(
                            client,
                            [{"role": "user", "content": "offline"}],
                            stage="extraction",
                            max_tokens=100,
                        )
        assert len(calls) == int(sent)
        assert budget.snapshot()["requests_sent"] == int(sent)
        assert budget.used == (20 if sent else 0)
        assert budget.snapshot()["usage_uncertain"] is False
        if not sent:
            assert [e for e, _ in events] == ["llm.not_sent"]
            assert events[0][1]["reason_code"] == "request_deadline_insufficient"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_sent_cancellation_keeps_unknown_usage_and_never_retries():
    started = asyncio.Event()

    async def transport(req):
        started.set()
        await asyncio.Event().wait()

    client, budget, events = await client_with_transport(transport)
    try:
        task = asyncio.create_task(
            chat_with_timeout(
                client,
                [{"role": "user", "content": "offline"}],
                stage="extraction",
                max_tokens=100,
            )
        )
        await asyncio.wait_for(started.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert budget.snapshot()["requests_sent"] == 1
        assert budget.used > 0 and budget.snapshot()["usage_uncertain"]
        assert len([e for e, _ in events if e == "llm.start"]) == 1
    finally:
        await client.close()
    assert remaining_deadline_seconds() is None


@pytest.mark.asyncio
async def test_real_staged_body_and_summary_send_distinct_sdk_timeouts(tmp_path):
    cards = facts()
    document = analysis_document(cards)
    body, summaries = split(document)
    messages, rf = request(cards, array_wire=True)
    calls = []

    def transport(req):
        wire = json.loads(req.content)
        calls.append((wire, req.extensions["timeout"]))
        content = (
            provider_body(body, cards)
            if len(calls) == 1
            else provider_summaries(summaries, wire)
        )
        return response(json.dumps(content))

    client, budget, _ = await client_with_transport(transport)
    view = ModelBoundClient(client, "offline", "generation")
    token = context.set_task_id("synthesis:staged")
    try:
        async with deadline_scope(1800):
            candidate, _ = await generate_staged_report(
                view,
                messages,
                rf,
                cards=cards,
                structure="academic_v2",
                language="en",
                selection=None,
                checkpoint_path=tmp_path / "run.json",
            )
        assert json.loads(candidate) == document
        assert [x[1]["read"] for x in calls] == [300, 180]
        assert all("timeout" not in x[0] for x in calls)
        assert budget.snapshot()["requests_sent"] == 2
    finally:
        context.reset_task_id(token)
        await client.close()


@pytest.mark.asyncio
async def test_scheduler_inherits_deadline_before_new_http_request():
    calls = []
    client, budget, events = await client_with_transport(lambda r: calls.append(r))

    class CallingWorker(Worker):
        agent_type = "deadline_test"

        async def execute(self, task):
            return await chat_with_timeout(
                client,
                [{"role": "user", "content": "offline"}],
                stage="extraction",
                max_tokens=100,
            )

    graph = TaskGraph()
    graph.add_task(
        SubTask("root", "offline", "deadline_test", timeout_ms=300000, max_retries=0)
    )
    try:
        await Scheduler([CallingWorker()], timeout_ms=100, cost_budget=budget).run(
            graph
        )
        assert graph.get_task("root").status is TaskStatus.FAILED
        assert graph.get_task("root").error == "request_deadline_insufficient"
        assert not calls and budget.snapshot()["requests_sent"] == 0
        assert any(e == "llm.not_sent" for e, _ in events)
        assert remaining_deadline_seconds() is None
    finally:
        await client.close()
