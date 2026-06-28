"""Scheduler——编排循环 + Worker 接口。"""


from __future__ import annotations
import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable

from litagent.safety.budget import CostBudget
from litagent.orchestrator.message_bus import AgentMessage, MessageBus, MessageType
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


class CancellationToken:
    def __init__(self):
        # Event() 异步信号标记，类似boolean开关，支持await
        self._event = asyncio.Event()

    
    def cancel(self) -> None:
        self._event.set()

    
    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


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
        on_complete: Callable | None = None,
        bus: MessageBus | None = None,
        cost_budget: CostBudget | None = None
    ):
        self._workers: dict[str, Worker] = {w.agent_type: w for w in workers}
        self._semaphore = asyncio.Semaphore(max_concurrent)  # 并发限流器--控制同时运行的任务数量
        self._timeout_ms = timeout_ms
        self._on_complete = on_complete
        self._replan_count = 0
        self._bus = bus
        if bus:
            bus.register('orchestrator')
        self._cost_budget = cost_budget

    
    async def run(self, graph: TaskGraph, cancellation: CancellationToken | None = None) -> dict[str, Any]:
        """运行编排循环，返回所有成功的结果。

        整体超时后返回部分结果（不崩溃）
        """
        try:
            return await asyncio.wait_for(
                self._loop(graph, cancellation),
                timeout=self._timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning("Orchestration timeout, returning partial results")
            return graph.get_results()
        finally:
            if self._on_complete:
                try:
                    await self._on_complete(graph)
                except Exception as e:
                    logger.warning(f"on_complete failed: {e}")

    
    async def _loop(self, graph: TaskGraph, cancellation: CancellationToken | None = None) -> dict[str, Any]:
        """编排主循环

        找就绪任务->并行执行->等完成
        """
        while not graph.is_complete():
            # 整体cost超限 -> 停发新任务，聚合已有结果
            if self._cost_budget and self._cost_budget.is_exceeded():
                logger.warning(
                    f"Cost budget exceeded ({self._cost_budget.used} tokens), "
                    f"stopping dispatch, returning partial results"
                )
                return graph.get_results()

            if cancellation and cancellation.is_cancelled:
                logger.info("Cancelled by user, returning partial results")
                return graph.get_results()

            if self._bus:
                msg = await self._bus.receive('orchestrator', timeout=0.05)
                if msg and msg.type == MessageType.REPLAN_REQUEST:
                    self._handle_replan(graph, msg)
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

            validation_attempts = 0
            for attempt in range(task.max_retries + 1):
                try:
                    # wait_for: 给async操作架超时限制，超时就抛TimeoutError
                    result = await asyncio.wait_for(
                        worker.execute(task),
                        timeout=task.timeout_ms / 1000,
                    )

                    if task.output_schema and 'type' in task.output_schema:
                        from pydantic import TypeAdapter, ValidationError
                        try:
                            # TypeAdapter 类型校验器，给定schema判断数据是否符合schema
                            adapter = TypeAdapter(task.output_schema)
                            adapter.validate_python(result)
                        except ValidationError as e:
                            validation_attempts += 1
                            if validation_attempts <= 3:
                                continue
                            last_error = f"Schema validation exhausted {e}"
                            raise

                    graph.mark_done(task.task_id, result)

                    # search文档太少，replan + search
                    if (self._bus and task.agent_type == 'search' and isinstance(result, list) and len(result) < 3):
                        await self._bus.broadcast(
                            AgentMessage(
                                type=MessageType.REPLAN_REQUEST,
                                sender='orchestrator',
                                receiver='orchestrator',
                                task_id=task.task_id,
                                payload={
                                    'query': task.input_data.get('query', ''),
                                    'count': len(result)
                                }
                            )
                        )
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


    def _handle_replan(self, graph, msg):
        """扩展query -> 追加task -> 重连downstream dep"""
        if self._replan_count >= 3:
            logger.warning("Replan limit(#)")
            return
        
        self._replan_count += 1
        original = msg.payload['query']
        parts = original.split()
        expanded = f"{original} Or broader: {parts[0]}" if parts else original
        new_id = f'search_replan_{hash(expanded) & 0xFFFF:04x}'
        graph.add_task(SubTask(
            task_id = new_id, description=f'Replan {expanded}',
            agent_type='search', priority=0,
            input_data={'query': expanded, 'source': 'semantic_scholar'}
        ))
        for tid in graph.tasks:
            if msg.task_id in graph._deps.get(tid, set()):
                graph.add_dependency(tid, new_id)
        logger.info(f"Replan: added {new_id}")