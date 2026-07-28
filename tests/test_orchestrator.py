"""Tests for task graphs, scheduling, and survey planning."""

import asyncio
from typing import Any

import pytest

from litagent.orchestrator.task_graph import TaskGraph, SubTask, TaskStatus
from litagent.orchestrator.scheduler import Scheduler, Worker
from litagent.agents.planner import SurveyPlanner


class MockWorker(Worker):
    """Worker that returns deterministic task metadata."""

    def __init__(self, agent_type: str, delay: float = 0):
        self._type = agent_type
        self._delay = delay

    @property
    def agent_type(self) -> str:
        return self._type

    async def execute(self, task: SubTask) -> Any:
        if self._delay:
            await asyncio.sleep(self._delay)
        return {"agent": self._type, "task_id": task.task_id}


class FailingWorker(Worker):
    """Worker test double that always raises."""

    def __init__(self, agent_type: str):
        self._type = agent_type

    @property
    def agent_type(self) -> str:
        return self._type

    async def execute(self, task: SubTask) -> Any:
        raise RuntimeError(f"Worker '{self._type}' failed")


class TestTaskGraph:
    """Tests task-graph state and dependency handling."""

    def test_add_and_get(self):
        g = TaskGraph()
        t = SubTask(task_id="t1", description="test", agent_type="search")
        g.add_task(t)
        assert g.get_task("t1") is t

    def test_get_ready_no_deps(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="t2", description="b", agent_type="search"))
        ready = g.get_ready_tasks()
        assert len(ready) == 2

    def test_get_ready_with_deps(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(
            SubTask(task_id="t2", description="b", agent_type="extract"),
            depends_on=["t1"],
        )
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t1"

    def test_ready_after_dep_done(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(
            SubTask(task_id="t2", description="b", agent_type="extract"),
            depends_on=["t1"],
        )
        g.mark_done("t1", {"papers": 10})
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t2"

    def test_mark_failed_skips_downstream(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(
            SubTask(task_id="t2", description="b", agent_type="extract"),
            depends_on=["t1"],
        )
        g.add_task(
            SubTask(task_id="t3", description="c", agent_type="synthesis"),
            depends_on=["t2"],
        )
        g.mark_failed("t1", "API down")
        assert g.get_task("t2").status == TaskStatus.SKIPPED
        assert g.get_task("t3").status == TaskStatus.SKIPPED

    def test_is_complete(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        assert not g.is_complete()
        g.mark_done("t1", "ok")
        assert g.is_complete()

    def test_is_complete_with_mixed_status(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="t2", description="b", agent_type="search"))
        g.mark_done("t1", "ok")
        g.mark_failed("t2", "error")
        assert g.is_complete()

    def test_get_results(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="t2", description="b", agent_type="search"))
        g.mark_done("t1", {"papers": 5})
        g.mark_failed("t2", "error")
        results = g.get_results()
        assert "t1" in results
        assert "t2" not in results

    def test_priority_ordering(self):
        g = TaskGraph()
        g.add_task(
            SubTask(task_id="low", description="a", agent_type="search", priority=2)
        )
        g.add_task(
            SubTask(task_id="high", description="b", agent_type="search", priority=0)
        )
        ready = g.get_ready_tasks()
        assert ready[0].task_id == "high"

    def test_finalize_incomplete_cancels_running_and_skips_pending(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="running", description="a", agent_type="search"))
        g.add_task(
            SubTask(task_id="pending", description="b", agent_type="extract"),
            depends_on=["running"],
        )
        g.mark_running("running")

        transitions = g.finalize_incomplete("orchestration_timeout")

        assert transitions == {"cancelled": ["running"], "skipped": ["pending"]}
        assert g.get_task("running").status == TaskStatus.CANCELLED
        assert g.get_task("running").error == "orchestration_timeout"
        assert g.get_task("pending").status == TaskStatus.SKIPPED
        assert g.is_complete()

    def test_execution_summary_reports_non_done_tasks(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="done", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="failed", description="b", agent_type="extract"))
        g.mark_done("done", {})
        g.mark_failed("failed", "worker_timeout")

        summary = g.execution_summary()

        assert summary["status"] == "incomplete"
        assert summary["counts"] == {"done": 1, "failed": 1}
        assert summary["failed_task_ids"] == ["failed"]


class TestScheduler:
    """Tests scheduler execution and failure propagation."""

    @pytest.mark.asyncio
    async def test_simple_sequential(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(
            SubTask(task_id="t2", description="b", agent_type="extract"),
            depends_on=["t1"],
        )
        scheduler = Scheduler(workers=[MockWorker("search"), MockWorker("extract")])
        results = await scheduler.run(g)
        assert "t1" in results
        assert "t2" in results

    @pytest.mark.asyncio
    async def test_parallel_dispatch(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="s1", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="s2", description="b", agent_type="search"))
        g.add_task(SubTask(task_id="s3", description="c", agent_type="search"))
        scheduler = Scheduler(workers=[MockWorker("search")])
        results = await scheduler.run(g)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_missing_worker(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="unknown"))
        scheduler = Scheduler(workers=[])
        results = await scheduler.run(g)
        assert len(results) == 0
        assert g.get_task("t1").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_worker_failure_cascades(self):
        g = TaskGraph()
        g.add_task(
            SubTask(task_id="t1", description="a", agent_type="bad", max_retries=0)
        )
        g.add_task(
            SubTask(task_id="t2", description="b", agent_type="search"),
            depends_on=["t1"],
        )
        scheduler = Scheduler(workers=[FailingWorker("bad"), MockWorker("search")])
        results = await scheduler.run(g)
        assert g.get_task("t1").status == TaskStatus.FAILED
        assert g.get_task("t2").status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_timeout_returns_partial(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="fast", description="a", agent_type="search"))
        g.add_task(
            SubTask(
                task_id="slow",
                description="b",
                agent_type="slow",
                timeout_ms=100,
                max_retries=0,
            )
        )
        scheduler = Scheduler(
            workers=[MockWorker("search"), MockWorker("slow", delay=5)],
            timeout_ms=2000,
        )
        results = await scheduler.run(g)
        assert "fast" in results
        assert g.get_task("slow").status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_global_timeout_cancels_running_and_emits_terminal(self):
        g = TaskGraph()
        g.add_task(
            SubTask(
                task_id="slow",
                description="a",
                agent_type="slow",
                timeout_ms=5000,
                max_retries=0,
            )
        )
        events = []
        scheduler = Scheduler(
            workers=[MockWorker("slow", delay=5)],
            timeout_ms=20,
            trace_hook=lambda event, data: events.append((event, data)),
        )

        assert await scheduler.run(g) == {}
        assert g.get_task("slow").status == TaskStatus.CANCELLED
        cancelled = [data for event, data in events if event == "worker.cancelled"]
        assert cancelled == [
            {
                "task_id": "slow",
                "agent_type": "slow",
                "error": "orchestration_timeout",
            }
        ]


class TestSurveyPlanner:
    """Tests survey-plan task construction."""

    @pytest.fixture(autouse=True)
    def _no_ss_key(self, monkeypatch):

        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_plan_creates_dag(self):
        planner = SurveyPlanner()
        graph = await planner.plan("few-shot learning in CV")

        assert len(graph.tasks) == 8
        assert "report" not in graph.tasks
        assert graph.tasks["adversarial_review"].agent_type == "adversarial_review"

    @pytest.mark.asyncio
    async def test_search_tasks_are_parallel(self):
        planner = SurveyPlanner()
        graph = await planner.plan("test query")
        ready = graph.get_ready_tasks()
        search_tasks = [t for t in ready if t.agent_type == "search"]
        recall_tasks = [t for t in ready if t.agent_type == "recall"]
        assert len(search_tasks) == 2
        assert len(recall_tasks) == 1

    @pytest.mark.asyncio
    async def test_adversarial_review_depends_on_extract_and_graph(self):
        planner = SurveyPlanner()
        graph = await planner.plan("test")
        ready_ids = {t.task_id for t in graph.get_ready_tasks()}
        assert "adversarial_review" not in ready_ids

    @pytest.mark.asyncio
    async def test_query_propagated_to_search_input(self):
        planner = SurveyPlanner()
        graph = await planner.plan("vision transformers")
        t = graph.get_task("search_arxiv_q0")
        assert t.input_data["query"] == "vision transformers"


class TestRelevanceGateDAG:
    """Tests relevance-gate dependencies in the task graph."""

    @pytest.fixture(autouse=True)
    def _no_ss_key(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_dag_contains_relevance_gate(self):
        from litagent.agents.planner import SurveyPlanner

        planner = SurveyPlanner()
        graph = await planner.plan("test query")
        rg = graph.get_task("relevance_gate")
        assert rg is not None
        assert rg.agent_type == "relevance_gate"

    @pytest.mark.asyncio
    async def test_relevance_gate_depends_only_on_dedup(self):
        from litagent.agents.planner import SurveyPlanner

        planner = SurveyPlanner()
        graph = await planner.plan("test")
        deps = graph._deps.get("relevance_gate", set())
        assert deps == {"dedup"}

    @pytest.mark.asyncio
    async def test_extract_and_graph_depend_on_relevance_gate(self):
        from litagent.agents.planner import SurveyPlanner

        planner = SurveyPlanner()
        graph = await planner.plan("test")
        assert graph._deps.get("extract") == {"relevance_gate"}
        assert graph._deps.get("graph_analysis") == {"relevance_gate"}

    @pytest.mark.asyncio
    async def test_extract_has_zero_retries(self):
        from litagent.agents.planner import SurveyPlanner

        planner = SurveyPlanner()
        graph = await planner.plan("test")
        extract = graph.get_task("extract")
        assert extract.max_retries == 0
        assert extract.timeout_ms == 300000


class TestSchedulerPriority:
    """Tests scheduler priority ordering."""

    @pytest.mark.asyncio
    async def test_scheduler_starts_lower_priority_task_first(self):
        from litagent.orchestrator.scheduler import Scheduler, Worker
        from litagent.orchestrator.task_graph import TaskGraph, SubTask

        started: list[str] = []

        class RecordWorker(Worker):
            """Worker that records task start order."""

            def __init__(self, name):
                self._name = name

            @property
            def agent_type(self):
                return self._name

            async def execute(self, task):
                started.append(task.task_id)
                return task.task_id

        graph = TaskGraph()
        graph.add_task(
            SubTask(task_id="low_pri", description="low", agent_type="w", priority=10)
        )
        graph.add_task(
            SubTask(task_id="high_pri", description="high", agent_type="w", priority=0)
        )

        scheduler = Scheduler(
            workers=[RecordWorker("w")], max_concurrent=1, timeout_ms=5000
        )
        await scheduler.run(graph)

        assert started[0] == "high_pri"
        assert len(started) == 2
