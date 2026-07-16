import pytest
import asyncio
from typing import Any

from litagent.orchestrator.task_graph import TaskGraph, SubTask, TaskStatus
from litagent.orchestrator.scheduler import Scheduler, Worker
from litagent.orchestrator.message_bus import MessageBus, AgentMessage, MessageType
from litagent.agents.planner import SurveyPlanner


class MockWorker(Worker):
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
    def __init__(self, agent_type: str):
        self._type = agent_type

    @property
    def agent_type(self) -> str:
        return self._type

    async def execute(self, task: SubTask) -> Any:
        raise RuntimeError(f"Worker '{self._type}' failed")


class TestTaskGraph:
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
        g.add_task(SubTask(task_id="t2", description="b", agent_type="extract"), depends_on=["t1"])
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t1"

    def test_ready_after_dep_done(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="t2", description="b", agent_type="extract"), depends_on=["t1"])
        g.mark_done("t1", {"papers": 10})
        ready = g.get_ready_tasks()
        assert len(ready) == 1
        assert ready[0].task_id == "t2"

    def test_mark_failed_skips_downstream(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="t2", description="b", agent_type="extract"), depends_on=["t1"])
        g.add_task(SubTask(task_id="t3", description="c", agent_type="synthesis"), depends_on=["t2"])
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
        g.add_task(SubTask(task_id="low", description="a", agent_type="search", priority=2))
        g.add_task(SubTask(task_id="high", description="b", agent_type="search", priority=0))
        ready = g.get_ready_tasks()
        assert ready[0].task_id == "high"


class TestScheduler:
    @pytest.mark.asyncio
    async def test_simple_sequential(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="t1", description="a", agent_type="search"))
        g.add_task(SubTask(task_id="t2", description="b", agent_type="extract"), depends_on=["t1"])
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
        g.add_task(SubTask(task_id="t1", description="a", agent_type="bad", max_retries=0))
        g.add_task(SubTask(task_id="t2", description="b", agent_type="search"), depends_on=["t1"])
        scheduler = Scheduler(workers=[FailingWorker("bad"), MockWorker("search")])
        results = await scheduler.run(g)
        assert g.get_task("t1").status == TaskStatus.FAILED
        assert g.get_task("t2").status == TaskStatus.SKIPPED

    @pytest.mark.asyncio
    async def test_timeout_returns_partial(self):
        g = TaskGraph()
        g.add_task(SubTask(task_id="fast", description="a", agent_type="search"))
        g.add_task(SubTask(
            task_id="slow", description="b", agent_type="slow",
            timeout_ms=100, max_retries=0,
        ))
        scheduler = Scheduler(
            workers=[MockWorker("search"), MockWorker("slow", delay=5)],
            timeout_ms=2000,
        )
        results = await scheduler.run(g)
        assert "fast" in results


class TestMessageBus:
    @pytest.mark.asyncio
    async def test_send_and_receive(self):
        bus = MessageBus()
        bus.register("worker_1")
        msg = AgentMessage(
            type=MessageType.TASK_ASSIGN, sender="orchestrator", receiver="worker_1",
            task_id="t1", payload={"query": "test"},
        )
        await bus.send(msg)
        received = await bus.receive("worker_1", timeout=1.0)
        assert received is not None
        assert received.task_id == "t1"

    @pytest.mark.asyncio
    async def test_receive_timeout(self):
        bus = MessageBus()
        bus.register("worker_1")
        received = await bus.receive("worker_1", timeout=0.1)
        assert received is None

    @pytest.mark.asyncio
    async def test_send_to_unregistered(self):
        bus = MessageBus()
        msg = AgentMessage(type=MessageType.TASK_ASSIGN, sender="orch", receiver="ghost")
        await bus.send(msg)

    @pytest.mark.asyncio
    async def test_broadcast(self):
        bus = MessageBus()
        bus.register("w1")
        bus.register("w2")
        bus.register("sender")
        msg = AgentMessage(
            type=MessageType.SHARED_DISCOVERY, sender="sender", receiver="",
            payload={"finding": "important"},
        )
        await bus.broadcast(msg)
        r1 = await bus.receive("w1", timeout=0.5)
        r2 = await bus.receive("w2", timeout=0.5)
        r_sender = await bus.receive("sender", timeout=0.1)
        assert r1 is not None
        assert r2 is not None
        assert r_sender is None

    @pytest.mark.asyncio
    async def test_pending_count(self):
        bus = MessageBus()
        bus.register("w1")
        assert bus.pending_count("w1") == 0
        await bus.send(AgentMessage(type=MessageType.TASK_ASSIGN, sender="orch", receiver="w1"))
        assert bus.pending_count("w1") == 1


class TestSurveyPlanner:
    """无 llm 构造 → _decompose 降级到单 query。这里验证「规则/降级骨架」；
    完整 LLM 分解形态见 tests/test_planner_decompose.py。"""

    @pytest.fixture(autouse=True)
    def _no_ss_key(self, monkeypatch):
        # 锁定源数为 arxiv+hf（2 源），避免宿主环境设了 SS key 导致源数飘
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_plan_creates_dag(self):
        planner = SurveyPlanner()
        graph = await planner.plan("few-shot learning in CV")
        # 降级单 query × 2 源 = 2 search + 5 下游(dedup/extract/graph/adversarial/report)
        assert len(graph.tasks) == 7

    @pytest.mark.asyncio
    async def test_search_tasks_are_parallel(self):
        planner = SurveyPlanner()
        graph = await planner.plan("test query")
        ready = graph.get_ready_tasks()
        search_tasks = [t for t in ready if t.agent_type == "search"]
        assert len(search_tasks) == 2   # arxiv+hf，单 query

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
        t = graph.get_task("search_arxiv_q0")   # 新 id 方案：search_{源}_q{i}
        assert t.input_data["query"] == "vision transformers"


# ═══════════════════════════════════════════════════════════
# 13.7.1-B — Scheduler 按 priority 调度
# ═══════════════════════════════════════════════════════════

class TestSchedulerPriority:
    """Scheduler 在 max_concurrent=1 时启动更低 priority task。"""

    @pytest.mark.asyncio
    async def test_scheduler_starts_lower_priority_task_first(self):
        from litagent.orchestrator.scheduler import Scheduler, Worker
        from litagent.orchestrator.task_graph import TaskGraph, SubTask

        started: list[str] = []

        class RecordWorker(Worker):
            def __init__(self, name):
                self._name = name

            @property
            def agent_type(self):
                return self._name

            async def execute(self, task):
                started.append(task.task_id)
                return task.task_id

        graph = TaskGraph()
        graph.add_task(SubTask(task_id="low_pri", description="low",
                               agent_type="w", priority=10))
        graph.add_task(SubTask(task_id="high_pri", description="high",
                               agent_type="w", priority=0))

        scheduler = Scheduler(
            workers=[RecordWorker("w")], max_concurrent=1, timeout_ms=5000)
        await scheduler.run(graph)

        assert started[0] == "high_pri"
        assert len(started) == 2
