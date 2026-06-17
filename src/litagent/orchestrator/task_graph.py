"""TaskGraph——SubTask DAG + 拓扑排序就绪检测。"""


from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from litagent.logging import get_logger


logger = get_logger('orchestrator.task_graph')


class TaskStatus(str, Enum):
    PENDING = 'pending'
    RUNNING = 'running'
    DONE = 'done'
    FAILED = 'failed'
    SKIPPED = 'skipped'


@dataclass
class SubTask:
    """DAG 中的一个节点
    
    Attributes:
        task_id: 唯一标识(e.g. "search_arxiv", "extract_001")
        description: (Langfuse trace可用)
        agent_type: 分派到哪个worker
        input_data
        status
        priority: 同层内排序
        timeout_ms: 单任务超时时间
        max_retries
        result: 执行结果(Worker返回值)
        error: 错误信息
    """
    task_id: str
    description: str
    agent_type: str
    input_data: dict[str, Any] = field(default_factory=dict)
    output_schema: dict = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    timeout_ms: int = 120000
    max_retries: int = 2
    result: Any = None
    error: str | None = None


class TaskGraph:
    """SubTask 的有向无环图（DAG）。

    职责：
    1. 维护任务和依赖关系
    2. 拓扑排序——get_ready_tasks() 返回所有前置已完成的任务
    3. 状态转移——mark_done / mark_failed + 级联跳过
    4. 完成检测——is_complete()

    Scheduler 调用此类驱动编排循环。TaskGraph 本身不做 I/O。
    """

    def __init__(self):
        self._tasks: dict[str, SubTask] = {}
        # dependencies: 存储每个任务前置依赖列表
        self._deps: dict[str, set[str]] = {}

    
    def add_task(self, task: SubTask, depends_on: list[str] | None = None) -> None:
        """添加任务+前置依赖"""
        self._tasks[task.task_id] = task
        self._deps[task.task_id] = set(depends_on or [])


    def get_task(self, task_id: str) -> SubTask | None:
        return self._tasks.get(task_id)


    def get_ready_tasks(self) -> list[SubTask]:
        """返回所有前置任务完成(Done)且自身状态为Pending的任务

        按priority排序。Scheduler每轮调一次。
        依赖了不存在的task_id视为未满足
        """
        ready = []
        for tid, task in self._tasks.items():
            if task.status != TaskStatus.PENDING:
                continue
            
            # 取前置任务
            deps = self._deps.get(tid, set())
            # 判断前置是否存在且all done
            all_done = all(
                d in self._tasks and self._tasks[d].status == TaskStatus.DONE
                for d in deps
            )

            if all_done:
                ready.append(task)

        return sorted(ready, key=lambda t: t.priority)

    
    def mark_running(self, task_id: str) -> None:
        self._tasks[task_id].status = TaskStatus.RUNNING

    
    def mark_done(self, task_id: str, result: Any) -> None:
        task = self._tasks[task_id]
        task.status = TaskStatus.DONE
        task.result = result
        logger.debug(f"Task '{task_id}' done")

    
    def mark_failed(self, task_id: str, error: str) -> None:
        """标记失败 + 级联跳过所有下游任务"""
        task = self._tasks[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        logger.warning(f"Task '{task_id}' failed: {error}")
        self._skip_downstream(task_id)


    def _skip_downstream(self, failed_id: str) -> None:
        """递归跳过所有依赖与failed_id的任务"""
        for tid, deps in self._deps.items():
            if failed_id in deps and self._tasks[tid].status == TaskStatus.PENDING:
                self._tasks[tid].status = TaskStatus.SKIPPED
                self._tasks[tid].error = f"Skipped dependency '{failed_id}' unavailable"
                logger.debug(f"Task '{tid}' skipped due to '{failed_id}' failure")
                self._skip_downstream(tid)

    
    def is_complete(self) -> bool:
        """所有任务都终结(DONE/FAILED/SKIPPED)"""
        terminal = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.SKIPPED}
        return all(t.status in terminal for t in self._tasks.values())


    def get_results(self) -> dict[str, Any]:
        """返回所有成功任务的结果。"""
        return {
            tid: t.result
            for tid, t in self._tasks.items()
            if t.status == TaskStatus.DONE
        }

        
    @property
    def tasks(self) -> dict[str, SubTask]:
        return self._tasks
