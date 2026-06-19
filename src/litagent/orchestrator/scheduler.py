"""Scheduler——编排循环 + Worker 接口。"""


from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Any

from litagent.orchestrator.task_graph import TaskGraph, SubTask, TaskStatus
from litagent.logging import get_logger


logger = get_logger('orchestrator.scheduler')


class Worker(ABC):
    """所有Agent的基类"""

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """与 Subtask.agent_type匹配的标识"""
        ...

    
    @abstractmethod
    async def execute(self, task: SubTask) -> Any:
        """执行一个子任务，返回结果。异常由scheduler捕获"""
        ...


class Scheduler:
    """编排调度器-驱动TaskGraph从开始到完成

    循环逻辑：
    1. get_ready_tasks() -> 获取所有前置已完成任务
    2. 并行分派到对应的Worker(asyncio.gather)
    3. 收集结果，mark_done / mark_failed
    4. 重复直到is_complete()

    Args:
        workers: Worker 实例列表，按agent_type索引
        max_concurrent: 最大并行任务数
        timeout_ms: 整体编排超时
    """

    def __init__(
        self,
        workers: list[Worker],
        max_concurrent: int = 5,
        timeout_ms: int = 600000,
    ):
        self._workers: dict[str, Worker] = {w.agent_type: w for w in workers}
        self._semaphore = asyncio.Semaphore(max_concurrent)  # 并发限流器--控制同时运行的任务数量
        self._timeout_ms = timeout_ms

    
    async def run(self, graph: TaskGraph) -> dict[str, Any]:
        """运行编排循环，返回所有成功的结果。

        整体超时后返回部分结果（不崩溃）
        """
        try:
            return await asyncio.wait_for(
                self._loop(graph),
                timeout=self._timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning("Orchestration timeout, returning partial results")
            return graph.get_results()

    
    async def _loop(self, graph: TaskGraph) -> dict[str, Any]:
        """编排主循环

        找就绪任务->并行执行->等完成
        """
        while not graph.is_complete():
            ready = graph.get_ready_tasks()
            if not ready:
                await asyncio.sleep(0.05)  # 无就绪任务，暂停后重新获取
                continue

            # self._dispatch()返回coroutine
            coros = [self._dispatch(graph, task) for task in ready]

            # 并行执行所有corountine， return_exception保证异常不终止执行
            # gather让所有协程同时启动
            # semaphore保证同时只有max_concurrent在执行
            await asyncio.gather(*coros, return_exceptions=True)

        return graph.get_results()

    
    async def _dispatch(self, graph: TaskGraph, task: SubTask) -> None:
        """分配单个任务到Worker--带信号限流 + 上游结果注入 + 超时 + 充实"""
        async with self._semaphore:
            # match对应的worker
            worker = self._workers.get(task.agent_type)
            if not worker:
                graph.mark_failed(task.task_id, f"No worker for type '{task.agent_type}'")
                return

            self._inject_upstream_results(graph, task)
            # 将task的status改为running
            graph.mark_running(task.task_id)

            last_error = None
            for attempt in range(task.max_retries + 1):
                try:
                    # wait_for: 给async操作架超时限制，超时就抛TimeoutError
                    result = await asyncio.wait_for(
                        worker.execute(task),
                        timeout=task.timeout_ms / 1000,
                    )
                    graph.mark_done(task.task_id, result)
                    return
                except asyncio.TimeoutError:
                    last_error = f'Timeout after {task.timeout_ms}ms'
                except Exception as e:
                    last_error = str(e)

                # 指数退避重试 -- 失败后的过一段时间再试
                if attempt < task.max_retries:
                    wait = 2 ** attempt
                    logger.debug(f"Retry {attempt + 1} for '{task.task_id}', waiting {wait}s")
                    await asyncio.sleep(wait)
            
            graph.mark_failed(task.task_id, last_error or "Unknown error")


    def _inject_upstream_results(self, graph: TaskGraph, task: SubTask) -> None:
        """将上游任务的 result 注入到当前任务的 input_data["upstream_results"]。

        这样 Worker.execute() 可以通过 task.input_data["upstream_results"]
        访问前置任务的结果，无需持有 TaskGraph 引用。
        """
        deps = graph._deps.get(task.task_id, set())
        upstream = {}
        # 把前置任务结果放入task.input['upstream_result']
        for dep_id in deps:
            dep_task = graph.get_task(dep_id)
            if dep_task and dep_task.status == TaskStatus.DONE:
                upstream[dep_id] = dep_task.result
        if upstream:
            task.input_data['upstream_results'] = upstream