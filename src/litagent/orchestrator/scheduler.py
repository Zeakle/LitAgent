"""Schedule dependency-ready tasks with concurrency, retries, and timeouts."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable

from litagent.logging import get_logger
from litagent.observability.context import reset_task_id, set_task_id
from litagent.orchestrator.task_graph import SubTask, TaskGraph, TaskStatus
from litagent.safety.budget import CostBudget

logger = get_logger("orchestrator.scheduler")


class Worker(ABC):
    """Define the worker contract consumed by the scheduler."""

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Return the worker agent type."""
        ...

    @abstractmethod
    async def execute(self, task: SubTask) -> Any:
        """Execute the worker task."""
        ...


class CancellationToken:
    """Expose cooperative cancellation state to an orchestration run."""

    def __init__(self):
        """Initialize the cancellation token."""
        self._event = asyncio.Event()

    def cancel(self) -> None:
        """Request cancellation."""
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        """Return whether cancellation was requested."""
        return self._event.is_set()

    async def wait(self) -> None:
        """Wait until cancellation is requested."""
        await self._event.wait()


class Scheduler:
    """Execute ready DAG tasks under concurrency and budget limits."""

    def __init__(
        self,
        workers: list[Worker],
        max_concurrent: int = 5,
        timeout_ms: int = 600000,
        on_complete: Callable | None = None,
        cost_budget: CostBudget | None = None,
        trace_hook: Callable | None = None,
    ):
        """Initialize the scheduler."""
        self._workers: dict[str, Worker] = {w.agent_type: w for w in workers}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._timeout_ms = timeout_ms
        self._on_complete = on_complete
        self._cost_budget = cost_budget
        self._trace_hook = trace_hook

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event without allowing hook failures to escape."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    async def run(
        self, graph: TaskGraph, cancellation: CancellationToken | None = None
    ) -> dict[str, Any]:
        """Run a graph within the orchestration deadline."""
        try:
            return await asyncio.wait_for(
                self._loop(graph, cancellation),
                timeout=self._timeout_ms / 1000,
            )
        except asyncio.TimeoutError:
            logger.warning("Orchestration timeout, returning partial results")
            self._finalize_incomplete(graph, "orchestration_timeout")
            return graph.get_results()
        finally:
            if self._on_complete:
                try:
                    await self._on_complete(graph)
                except Exception as e:
                    logger.warning(f"on_complete failed: {e}")

    async def _loop(
        self, graph: TaskGraph, cancellation: CancellationToken | None = None
    ) -> dict[str, Any]:
        """Dispatch ready tasks until the graph reaches a terminal state."""
        while not graph.is_complete():

            if self._cost_budget and self._cost_budget.is_exceeded():
                logger.warning(
                    f"Cost budget exceeded ({self._cost_budget.used} tokens), "
                    f"stopping dispatch, returning partial results"
                )
                self._finalize_incomplete(graph, "cost_budget_exceeded")
                return graph.get_results()

            if cancellation and cancellation.is_cancelled:
                logger.info("Cancelled by user, returning partial results")
                self._finalize_incomplete(graph, "user_cancelled")
                return graph.get_results()

            ready = graph.get_ready_tasks()
            if not ready:
                await asyncio.sleep(0.05)
                continue

            coros = [self._dispatch(graph, task, cancellation) for task in ready]

            # Each dispatcher records its own failure, so siblings remain independent.
            await asyncio.gather(*coros, return_exceptions=True)

            if cancellation and cancellation.is_cancelled:
                self._finalize_incomplete(graph, "user_cancelled")
                return graph.get_results()

        return graph.get_results()

    def _finalize_incomplete(self, graph: TaskGraph, reason: str) -> None:
        """Move unfinished graph tasks into terminal states."""
        transitions = graph.finalize_incomplete(reason)
        for task_id in transitions["cancelled"]:
            task = graph.get_task(task_id)
            self._emit(
                "worker.cancelled",
                {
                    "task_id": task_id,
                    "agent_type": task.agent_type if task else "",
                    "error": reason,
                },
            )

    async def _dispatch(
        self,
        graph: TaskGraph,
        task: SubTask,
        cancellation: CancellationToken | None = None,
    ) -> None:
        """Execute one task with tracing, validation, retries, and timeout."""
        async with self._semaphore:

            if cancellation and cancellation.is_cancelled:
                return

            worker = self._workers.get(task.agent_type)
            if not worker:
                graph.mark_failed(
                    task.task_id, f"No worker for type '{task.agent_type}'"
                )
                return

            self._inject_upstream_results(graph, task)
            self._emit(
                "worker.input",
                {
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "input": task.input_data,
                },
            )

            graph.mark_running(task.task_id)
            self._emit(
                "worker.start",
                {
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "description": task.description,
                },
            )
            _ctx_token = set_task_id(task.task_id)

            try:
                last_error = None
                last_error_type = "WorkerError"
                for attempt in range(task.max_retries + 1):
                    try:

                        result = await self._await_worker(
                            worker.execute(task),
                            cancellation=cancellation,
                            timeout=task.timeout_ms / 1000,
                        )

                        if task.output_schema and "type" in task.output_schema:
                            from pydantic import TypeAdapter, ValidationError

                            try:

                                adapter = TypeAdapter(task.output_schema)
                                adapter.validate_python(result)
                            except ValidationError as e:
                                last_error = f"Schema validation exhausted {e}"
                                last_error_type = type(e).__name__
                                raise

                        graph.mark_done(task.task_id, result)
                        self._emit(
                            "worker.complete",
                            {
                                "task_id": task.task_id,
                                "agent_type": task.agent_type,
                                "output": result,
                            },
                        )

                        return
                    except asyncio.CancelledError:
                        # Cooperative user cancellation is finalized once by
                        # _loop after every sibling dispatcher has unwound.
                        if cancellation and cancellation.is_cancelled:
                            return
                        raise
                    except asyncio.TimeoutError:
                        last_error = f"Timeout after {task.timeout_ms}ms"
                        last_error_type = "TimeoutError"
                    except Exception as e:
                        last_error = str(e)
                        last_error_type = type(e).__name__

                    if attempt < task.max_retries:
                        wait = 2**attempt
                        self._emit(
                            "worker.retry",
                            {
                                "task_id": task.task_id,
                                "agent_type": task.agent_type,
                                "name": task.agent_type,
                                "attempt": attempt + 2,
                                "max_attempts": task.max_retries + 1,
                                "reason_code": (
                                    "worker_timeout"
                                    if last_error_type == "TimeoutError"
                                    else (
                                        "worker_output_invalid"
                                        if last_error_type == "ValidationError"
                                        else "worker_execution_failed"
                                    )
                                ),
                                "error_type": last_error_type,
                                "backoff_ms": wait * 1000,
                            },
                        )
                        logger.debug(
                            f"Retry {attempt + 1} for '{task.task_id}', waiting {wait}s"
                        )
                        try:
                            await self._await_worker(
                                asyncio.sleep(wait),
                                cancellation=cancellation,
                            )
                        except asyncio.CancelledError:
                            if cancellation and cancellation.is_cancelled:
                                return
                            raise

                graph.mark_failed(task.task_id, last_error or "Unknown error")
                self._emit(
                    "worker.failed",
                    {
                        "task_id": task.task_id,
                        "agent_type": task.agent_type,
                        "error": last_error or "Unknown Error",
                    },
                )
            finally:
                reset_task_id(_ctx_token)

    @staticmethod
    async def _await_worker(
        awaitable,
        *,
        cancellation: CancellationToken | None,
        timeout: float | None = None,
    ):
        """Wait for work while allowing cooperative cancellation to preempt it."""
        work = asyncio.create_task(awaitable)
        if cancellation is None:
            return await asyncio.wait_for(work, timeout=timeout)

        cancel_waiter = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {work, cancel_waiter},
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if work in done:
                return await work
            if cancel_waiter in done:
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                raise asyncio.CancelledError

            work.cancel()
            await asyncio.gather(work, return_exceptions=True)
            raise asyncio.TimeoutError
        finally:
            cancel_waiter.cancel()
            await asyncio.gather(cancel_waiter, return_exceptions=True)

    def _inject_upstream_results(self, graph: TaskGraph, task: SubTask) -> None:
        """Inject successful dependency results into the task input."""
        deps = graph._deps.get(task.task_id, set())
        upstream = {}

        for dep_id in deps:
            dep_task = graph.get_task(dep_id)
            if dep_task and dep_task.status == TaskStatus.DONE:
                upstream[dep_id] = dep_task.result
        if upstream:
            task.input_data["upstream_results"] = upstream
