"""Verify resumable graph and scheduler boundaries without external services."""

from types import MethodType

import pytest

from litagent.config import AgentConfig, AppConfig, LoggingConfig
from litagent.interaction import RunAdvanceResult
from litagent.orchestrator.scheduler import Scheduler, SchedulerOutcome, Worker
from litagent.orchestrator.task_graph import SubTask, TaskGraph, TaskStatus
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget
from litagent.workbench import HumanDecision, RunOptions


class EchoWorker(Worker):
    @property
    def agent_type(self):
        return "echo"

    async def execute(self, task):
        return {"task_id": task.task_id}


def graph_with_boundary():
    graph = TaskGraph()
    graph.add_task(SubTask("relevance_gate", "rank", "echo"))
    graph.add_task(SubTask("extract", "extract", "echo"), depends_on=["relevance_gate"])
    graph.add_task(
        SubTask("graph_analysis", "graph", "echo"),
        depends_on=["relevance_gate"],
    )
    return graph


def test_task_graph_roundtrip_preserves_pending_boundary():
    graph = graph_with_boundary()
    graph.mark_running("relevance_gate")
    graph.mark_done("relevance_gate", [{"paper_id": "p1"}])
    restored = TaskGraph.from_snapshot(graph.to_snapshot())
    assert restored.get_task("relevance_gate").status == TaskStatus.DONE
    assert restored.get_task("relevance_gate").result == [{"paper_id": "p1"}]
    assert [task.task_id for task in restored.get_ready_tasks()] == [
        "extract",
        "graph_analysis",
    ]


@pytest.mark.parametrize(
    "snapshot",
    [
        {"schema_version": 1, "tasks": [], "dependencies": {"x": []}},
        {
            "schema_version": 1,
            "tasks": [
                {
                    "task_id": "x",
                    "description": "x",
                    "agent_type": "echo",
                    "status": "pending",
                }
            ],
            "dependencies": {"x": ["missing"]},
        },
    ],
)
def test_task_graph_rejects_corrupt_snapshots(snapshot):
    with pytest.raises(ValueError):
        TaskGraph.from_snapshot(snapshot)


@pytest.mark.asyncio
async def test_scheduler_stops_after_boundary_without_finalizing_graph():
    graph = graph_with_boundary()
    completed = []

    async def on_complete(_graph):
        completed.append(True)

    scheduler = Scheduler([EchoWorker()], on_complete=on_complete)
    outcome = await scheduler.run_until(graph, stop_after="relevance_gate")
    assert outcome.status == "paused"
    assert graph.get_task("relevance_gate").status == TaskStatus.DONE
    assert graph.get_task("extract").status == TaskStatus.PENDING
    assert graph.get_task("graph_analysis").status == TaskStatus.PENDING
    assert completed == []

    completed_outcome = await scheduler.run_until(graph)
    assert completed_outcome.status == "completed"
    assert graph.is_complete()
    assert completed == [True]


def test_cost_budget_roundtrip_keeps_consumed_calls():
    budget = CostBudget(max_tokens=100)
    budget.record({"prompt_tokens": 12, "completion_tokens": 8})
    restored = CostBudget.from_snapshot(budget.snapshot())
    assert restored.used == 20
    assert restored.call_count == 1
    assert restored.remaining() == 80


@pytest.mark.asyncio
async def test_runner_resumes_two_boundaries_without_replanning_or_early_trust_writes():
    graph = TaskGraph()
    graph.add_task(SubTask("dedup", "deduplicate", "fake"))
    graph.add_task(SubTask("relevance_gate", "rank", "fake"), depends_on=["dedup"])
    graph.add_task(SubTask("extract", "extract", "fake"), depends_on=["relevance_gate"])
    graph.add_task(
        SubTask("graph_analysis", "graph", "fake"),
        depends_on=["relevance_gate"],
    )
    graph.add_task(
        SubTask("adversarial_review", "write", "fake"),
        depends_on=["extract", "graph_analysis"],
    )

    class FakePlanner:
        calls = 0

        async def plan(self, query):
            self.calls += 1
            return graph

    class FakeScheduler:
        calls = 0

        async def run_until(self, active, *, stop_after=None, cancellation=None):
            self.calls += 1
            if stop_after:
                active.mark_running("dedup")
                active.mark_done(
                    "dedup",
                    [
                        {"paper_id": "p1", "title": "One"},
                        {"paper_id": "p2", "title": "Two"},
                    ],
                )
                active.mark_running("relevance_gate")
                active.mark_done("relevance_gate", [{"paper_id": "p1", "title": "One"}])
                return SchedulerOutcome("paused", active.get_results(), stop_after)
            for task_id in ("extract", "graph_analysis", "adversarial_review"):
                active.mark_running(task_id)
                active.mark_done(task_id, {"task_id": task_id})
            return SchedulerOutcome("completed", active.get_results())

    config = AppConfig(agent=AgentConfig(), logging=LoggingConfig())
    agent = LitAgent(config)
    agent._wired = True
    agent._cost_budget = CostBudget(max_tokens=config.safety.max_cost_tokens)
    agent._planner = FakePlanner()
    agent._scheduler = FakeScheduler()
    finalized = []

    async def prepare(self, **_kwargs):
        return (
            {
                "survey": "# Report",
                "metadata": {},
                "partial": False,
                "evaluation": {
                    "citation_accuracy": {"passed": True, "score": 1.0},
                    "faithfulness": {"passed": True, "score": 1.0},
                },
                "quality": {"status": "passed"},
                "delivery": {
                    "status": "ready",
                    "publishable": True,
                    "reason_codes": [],
                },
            },
            [],
        )

    async def finalize(self, **kwargs):
        finalized.append(kwargs["allow_trusted_writes"])
        return kwargs["report_data"]

    agent._prepare_report = MethodType(prepare, agent)
    agent._finalize_report = MethodType(finalize, agent)
    options = RunOptions(report_language="en", human_review=True)

    first: RunAdvanceResult = await agent.advance(
        "query", options=options, run_id="run-1"
    )
    assert first.status == "awaiting_human"
    assert first.review.stage == "paper_selection"
    assert finalized == []

    second = await agent.advance(
        "query",
        options=options,
        checkpoint=first.checkpoint,
        decision=HumanDecision(
            request_id=first.review.request_id,
            revision=first.review.revision,
            action="confirm",
            selected_paper_ids=("p1",),
        ),
    )
    assert second.status == "awaiting_human"
    assert second.review.stage == "final_report"
    assert agent._planner.calls == 1
    assert finalized == []

    terminal = await agent.advance(
        "query",
        options=options,
        checkpoint=second.checkpoint,
        decision=HumanDecision(
            request_id=second.review.request_id,
            revision=second.review.revision,
            action="approve",
        ),
    )
    assert terminal.status == "completed"
    assert terminal.report["delivery"]["publishable"] is True
    assert agent._planner.calls == 1
    assert agent._scheduler.calls == 2
    assert finalized == [True]
