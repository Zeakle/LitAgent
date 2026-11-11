"""Trace real request deadlines without changing billing or scientific gates."""

import asyncio
import json

import httpx
import pytest
from openai import APITimeoutError
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_observability import _tracer_with_mock
from test_step_timeouts import client_with_transport
from test_writer_v2 import case  # noqa: F401

from litagent.llm.client import LLMRequestTimeout, chat_with_timeout
from litagent.observability.recorder import (
    ArchiveRepository,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.orchestrator.scheduler import Scheduler, Worker
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.safety.budget import CostBudget
from litagent.writing_v2 import WriterError, classify_failure, writer_budget_plan


def arguments(api):
    """Choose only the actual API's output field and preserve zero retries."""
    result = {
        "provider_api": api,
        "max_output_tokens" if api == "responses" else "max_tokens": 100,
        "transport_max_attempts": 1,
    }
    if api == "responses":
        result["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "timeout_probe",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        }
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["chat", "responses"])
@pytest.mark.parametrize("fault", ["request", "cancel", "transport"])
async def test_sdk_failure_trace_and_recorder_keep_provenance_and_unknown_usage(
    tmp_path, api, fault
):
    started, ended, calls = asyncio.Event(), asyncio.Event(), []

    async def transport(request):
        calls.append(request)
        started.set()
        try:
            if fault == "transport":
                raise httpx.ReadTimeout("controlled transport timeout", request=request)
            await asyncio.Event().wait()
        finally:
            ended.set()

    seconds = 1 if fault == "request" else 5
    client, budget, events = await client_with_transport(
        transport, timeouts={"default": seconds, "synthesis": seconds}
    )
    tracer, log = _tracer_with_mock()
    redacted = RedactingTraceHook(tracer, payload_mode="metadata_only")
    recorder = RunRecorder("deadline", "offline", ArchiveRepository(tmp_path))
    original_hook = client._trace_hook

    def hook(event, data):
        original_hook(event, data)
        recorder(event, data)
        redacted(event, data)

    client._trace_hook = hook
    try:
        task = asyncio.create_task(
            chat_with_timeout(
                client,
                [{"role": "user", "content": "PRIVATE_SCIENTIFIC_SOURCE"}],
                stage="synthesis",
                **arguments(api),
            )
        )
        if fault == "cancel":
            await asyncio.wait_for(started.wait(), 1)
            task.cancel()
            expected = asyncio.CancelledError
        else:
            expected = LLMRequestTimeout if fault == "request" else APITimeoutError
        with pytest.raises(expected) as caught:
            await task
        if fault == "request":
            assert classify_failure(caught.value)["code"] == "llm_request_timeout"
            assert caught.value.diagnostics[0]["request_stage"] == "synthesis"
        assert ended.is_set() and len(calls) == 1, {
            "transport_started": started.is_set(),
            "transport_finished": ended.is_set(),
            "calls": len(calls),
            "events": [(e, d.get("error_type")) for e, d in events],
        }
        wire = json.loads(calls[0].content)
        assert (
            not {"request_stage", "request_timeout_seconds", "timeout_scope"}
            & wire.keys()
        )
        assert calls[0].extensions["timeout"]["read"] == seconds
        failures = [d for e, d in events if e == "llm.failed"]
        assert len(failures) == 1
        failure = failures[0]
        assert failure["provider_api"] == api
        assert failure["request_stage"] == "synthesis"
        assert failure["request_timeout_seconds"] == seconds
        reason = {
            "request": "llm_request_timeout",
            "cancel": "cancelled",
            "transport": "provider_transport_timeout",
        }[fault]
        assert failure["reason_code"] == reason
        update = [data for action, data in log if action == "update"][-1]
        assert update["metadata"]["provider_api"] == api
        assert update["metadata"]["request_stage"] == "synthesis"
        assert update["metadata"]["request_timeout_seconds"] == seconds
        assert update["metadata"]["reason_code"] == reason
        assert update["metadata"]["usage_available"] is False
        assert update["metadata"]["usage_complete"] is False
        assert "usage_details" not in update
        assert "PRIVATE_SCIENTIFIC_SOURCE" not in repr(log)
        assert not tracer._operations
        node = recorder.snapshot()["nodes"]["llm:" + failure["operation_id"]]
        assert node["metadata"]["request_stage"] == "synthesis"
        assert node["metadata"]["provider_api"] == api
        assert node["usage"] is None
        assert budget.snapshot()["requests_sent"] == 1
        assert budget.snapshot()["usage_uncertain"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_an_inner_timeout_is_not_relabelled_as_an_expired_request():
    original = TimeoutError("inner operation timeout")

    class Inner:
        async def chat(self, messages, **kwargs):
            raise original

    with pytest.raises(TimeoutError) as caught:
        await chat_with_timeout(Inner(), [], stage="synthesis", default=1)
    assert caught.value is original
    assert not isinstance(caught.value, LLMRequestTimeout)


@pytest.mark.asyncio
async def test_scheduler_reports_real_request_deadline_instead_of_worker_limit():
    class Client:
        async def chat(self, messages, **kwargs):
            await asyncio.Event().wait()

    class CallingWorker(Worker):
        @property
        def agent_type(self):
            return "timeout-probe"

        async def execute(self, task):
            return await chat_with_timeout(Client(), [], stage="summary", default=0.01)

    graph, events = TaskGraph(), []
    graph.add_task(
        SubTask("root", "offline", "timeout-probe", timeout_ms=1000, max_retries=0)
    )
    await Scheduler(
        [CallingWorker()], trace_hook=lambda e, d: events.append((e, d))
    ).run(graph)
    failure = next(d for e, d in events if e == "worker.failed")
    assert failure["reason_code"] == "llm_request_timeout"
    assert failure["error_type"] == "LLMRequestTimeout"
    assert "summary" in failure["error"] and "1000ms" not in failure["error"]
    assert failure["diagnostics"][0]["request_timeout_seconds"] == 0.01


@pytest.mark.asyncio
async def test_real_writer_timeout_retains_planned_hold_and_restart_never_resends(
    case, tmp_path
):
    received, finished = [], asyncio.Event()

    async def transport(request):
        received.append(request)
        try:
            await asyncio.Event().wait()
        finally:
            finished.set()

    owner, budget, events = await owner_for(transport)
    owner._request_timeouts = {"synthesis": 1, "default": 1}
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )
    budget.activate_plan(plan, path=tmp_path / "ledger.json")
    session = make_session(owner, case, tmp_path)
    try:
        with pytest.raises(LLMRequestTimeout):
            await session.initial_body()
        assert len(received) == 1 and finished.is_set()
        failure = session.journal["failure"]
        assert failure["code"] == "llm_request_timeout"
        assert failure["diagnostics"][0]["request_stage"] == "synthesis"
        assert failure["response_received"] is False
        snapshot = budget.snapshot()
        assert snapshot["requests_sent"] == 1 and snapshot["usage_uncertain"]
        entry = next(iter(snapshot["entries"].values()))
        assert entry["status"] == "interrupted_no_replay"
        assert entry["actual_tokens"] is None and entry["reserved_tokens"] > 0
        restored = CostBudget.from_snapshot(snapshot)
        assert restored.snapshot() == snapshot
        restarted = make_session(owner, case, tmp_path)
        with pytest.raises((WriterError, ValueError)):
            await restarted.initial_body()
        assert len(received) == 1
        assert budget.snapshot() == snapshot
        assert len([d for e, d in events if e == "llm.failed"]) == 1
    finally:
        await owner.close()


def test_historical_failed_responses_event_keeps_api_without_inventing_timeout():
    tracer, log = _tracer_with_mock()
    hook = RedactingTraceHook(tracer, payload_mode="metadata_only")
    hook(
        "llm.start",
        {
            "operation_id": "old",
            "provider_api": "responses",
            "messages": [{"role": "user", "content": "PRIVATE_SOURCE"}],
        },
    )
    hook(
        "llm.failed",
        {
            "operation_id": "old",
            "provider_api": "responses",
            "max_tokens": 20000,
            "output_limit_parameter": "max_output_tokens",
            "error_type": "CancelledError",
            "usage_available": False,
        },
    )
    update = [data for action, data in log if action == "update"][-1]
    assert update["metadata"]["provider_api"] == "responses"
    assert update["metadata"]["max_tokens"] == 20000
    assert "request_timeout_seconds" not in update["metadata"]
    assert "usage_details" not in update and "PRIVATE_SOURCE" not in repr(log)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outer,inner,expected", [(0.01, 0.1, "synthesis"), (0.1, 0.01, "summary")]
)
async def test_nested_requests_keep_the_earliest_owned_deadline(outer, inner, expected):
    ended = asyncio.Event()

    class Inner:
        async def chat(self, messages, **kwargs):
            try:
                await asyncio.Event().wait()
            finally:
                ended.set()

    class Outer:
        async def chat(self, messages, **kwargs):
            return await chat_with_timeout(Inner(), [], stage="summary", default=inner)

    with pytest.raises(LLMRequestTimeout) as caught:
        await chat_with_timeout(Outer(), [], stage="synthesis", default=outer)
    assert ended.is_set()
    assert caught.value.diagnostics[0]["request_stage"] == expected
    assert caught.value.diagnostics[0]["request_timeout_seconds"] == min(outer, inner)


def test_api_import_does_not_read_any_local_history():
    import os
    import subprocess
    import sys

    code = """
from litagent.observability.recorder import ArchiveRepository

def forbidden(self):
    raise AssertionError('API import scanned run history')

ArchiveRepository.list = forbidden
from litagent import api
assert isinstance(api.app.state.flow_repository, ArchiveRepository)
assert api.app.state.flow_archive_index == {}
print('API imported without history scan')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "LITAGENT_DEMO_MODE": "live"},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert "without history scan" in result.stdout


@pytest.mark.asyncio
async def test_live_lifespan_still_indexes_history_once(monkeypatch):
    """Use the live profile and load the archive index once during startup."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from fastapi import FastAPI

    from litagent import api

    listed = []

    def list_archives():
        listed.append(True)
        return [{"run_id": "offline-archive"}]

    interactive = SimpleNamespace(initialize=AsyncMock(), shutdown=AsyncMock())
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "live")
    monkeypatch.delenv("LITAGENT_CONFIG", raising=False)
    monkeypatch.setattr(api, "presentation_enabled", lambda: False)
    monkeypatch.setattr(api, "InteractiveRunService", lambda *a, **k: interactive)
    application = FastAPI()
    application.state.tasks = {}
    application.state.flow_runs = {}
    application.state.flow_repository = SimpleNamespace(list=list_archives)
    application.state.interactive_repository = object()
    application.state.flow_archive_index = {}
    async with api.lifespan(application):
        assert listed == [True]
        assert application.state.flow_archive_index == {
            "offline-archive": {"run_id": "offline-archive"}
        }
        assert application.state.config.safety.planned_run_limits.profile == (
            "agent-format-repair-v1"
        )
        assert application.state.report_audit_service is None
        interactive.initialize.assert_awaited_once()
    interactive.shutdown.assert_awaited_once()
