"""Distinguish failed operations from task deadlines without leaking work."""

import asyncio

import pytest

from litagent.orchestrator.scheduler import CancellationToken, Scheduler, Worker
from litagent.orchestrator.task_graph import SubTask, TaskGraph, TaskStatus
from litagent.safety.budget import CostBudget


class TimeoutWorker(Worker):
    """Expose an inner timeout or an indefinitely running operation."""

    def __init__(self, mode, message=""):
        self.mode = mode
        self.message = message
        self.started = asyncio.Event()
        self.finished = asyncio.Event()
        self.calls = 0

    @property
    def agent_type(self):
        return "test_timeout"

    async def execute(self, task):
        self.calls += 1
        self.started.set()
        try:
            if self.mode == "inner":
                raise TimeoutError(self.message)
            if self.mode == "success":
                return {"complete": True}
            await asyncio.Event().wait()
        finally:
            self.finished.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("token_present", [False, True])
@pytest.mark.parametrize("mode", ["inner", "deadline", "success"])
async def test_actual_graph_reports_only_the_deadline_that_expired(token_present, mode):
    token = CancellationToken() if token_present else None
    worker = TimeoutWorker(mode, "transport timeout at 180s")
    graph = TaskGraph()
    task = SubTask("root", "offline", worker.agent_type, timeout_ms=20, max_retries=0)
    graph.add_task(task)
    graph.add_task(
        SubTask("child", "dependent", worker.agent_type), depends_on=["root"]
    )
    events = []
    scheduler = Scheduler([worker], trace_hook=lambda e, d: events.append((e, d)))
    result = await scheduler.run(graph, token)
    assert worker.finished.is_set()
    if mode == "success":
        assert result == {"root": {"complete": True}, "child": {"complete": True}}
        assert not [e for e, _ in events if e == "worker.failed"]
        return
    assert result == {} and worker.calls == 1
    assert task.status is TaskStatus.FAILED
    assert graph.get_task("child").status is TaskStatus.SKIPPED
    failure = next(d for e, d in events if e == "worker.failed")
    assert failure["error_type"] == "TimeoutError"
    if mode == "inner":
        assert task.error == "transport timeout at 180s"
        assert failure["reason_code"] == "worker_inner_timeout"
        assert "20ms" not in failure["error"]
    else:
        assert task.error == "Timeout after 20ms"
        assert failure["reason_code"] == "worker_timeout"


@pytest.mark.asyncio
async def test_empty_inner_timeout_is_not_assigned_the_outer_300s_duration():
    worker = TimeoutWorker("inner")
    graph = TaskGraph()
    graph.add_task(
        SubTask("root", "offline", worker.agent_type, timeout_ms=300000, max_retries=0)
    )
    events = []
    scheduler = Scheduler([worker], trace_hook=lambda e, d: events.append((e, d)))
    await scheduler.run(graph)
    failure = next(d for e, d in events if e == "worker.failed")
    assert failure["error"] == "Worker operation timed out"
    assert failure["reason_code"] == "worker_inner_timeout"
    assert "300000" not in failure["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("token_present", [False, True])
async def test_parent_deadline_cancels_and_reaps_the_owned_worker(token_present):
    worker = TimeoutWorker("block")
    token = CancellationToken() if token_present else None
    graph = TaskGraph()
    graph.add_task(
        SubTask("root", "offline", worker.agent_type, timeout_ms=5000, max_retries=0)
    )
    events = []
    scheduler = Scheduler(
        [worker], timeout_ms=20, trace_hook=lambda e, d: events.append((e, d))
    )
    assert await scheduler.run(graph, token) == {}
    assert worker.started.is_set() and worker.finished.is_set()
    assert graph.get_task("root").status is TaskStatus.CANCELLED
    assert graph.get_task("root").error == "orchestration_timeout"
    assert not [e for e, _ in events if e == "worker.failed"]


@pytest.mark.asyncio
async def test_cooperative_cancel_is_not_reported_as_any_timeout():
    worker = TimeoutWorker("block")
    token = CancellationToken()
    graph = TaskGraph()
    graph.add_task(SubTask("root", "offline", worker.agent_type, max_retries=0))
    events = []
    run = asyncio.create_task(
        Scheduler([worker], trace_hook=lambda e, d: events.append((e, d))).run(
            graph, token
        )
    )
    await asyncio.wait_for(worker.started.wait(), 1)
    token.cancel()
    assert await asyncio.wait_for(run, 1) == {}
    assert worker.finished.is_set()
    assert graph.get_task("root").error == "user_cancelled"
    assert not [e for e, _ in events if e == "worker.failed"]


@pytest.mark.asyncio
async def test_automatic_inner_timeout_keeps_stop_and_never_dispatches_queued_work():
    worker = TimeoutWorker("inner", "inner timeout")
    budget = CostBudget(500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.record({"prompt_tokens": 50, "completion_tokens": 10})
    graph = TaskGraph()
    graph.add_task(SubTask("root", "offline", worker.agent_type, max_retries=0))
    graph.add_task(SubTask("queued", "offline", worker.agent_type, max_retries=0))
    assert (
        await Scheduler([worker], max_concurrent=1, cost_budget=budget).run(graph) == {}
    )
    assert worker.calls == 1 and budget.stop_reason == "strict_worker_failed"
    assert budget.used == 60
    assert graph.get_task("queued").status is TaskStatus.SKIPPED
    assert graph.get_task("queued").error == "strict_worker_failed"
