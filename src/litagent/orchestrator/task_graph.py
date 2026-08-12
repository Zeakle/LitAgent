"""Model survey subtasks, dependencies, and terminal execution state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from litagent.logging import get_logger

logger = get_logger("orchestrator.task_graph")


class TaskStatus(str, Enum):
    """Define lifecycle states for task-graph execution."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


@dataclass
class SubTask:
    """Represent one schedulable task and its execution state."""

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
    """Track task dependencies and enforce terminal state propagation."""

    def __init__(self):
        """Initialize the task graph."""
        self._tasks: dict[str, SubTask] = {}

        self._deps: dict[str, set[str]] = {}

    def add_task(self, task: SubTask, depends_on: list[str] | None = None) -> None:
        """Add or replace a task with its dependency IDs."""
        self._tasks[task.task_id] = task
        self._deps[task.task_id] = set(depends_on or [])

    def get_task(self, task_id: str) -> SubTask | None:
        """Return a task by ID."""
        return self._tasks.get(task_id)

    def get_ready_tasks(self) -> list[SubTask]:
        """Return pending tasks whose declared dependencies are complete."""
        ready = []
        for tid, task in self._tasks.items():
            if task.status != TaskStatus.PENDING:
                continue

            deps = self._deps.get(tid, set())

            all_done = all(
                d in self._tasks and self._tasks[d].status == TaskStatus.DONE
                for d in deps
            )

            if all_done:
                ready.append(task)

        return sorted(ready, key=lambda t: t.priority)

    def mark_running(self, task_id: str) -> None:
        """Mark the task as running."""
        self._tasks[task_id].status = TaskStatus.RUNNING

    def mark_done(self, task_id: str, result: Any) -> None:
        """Mark the task as completed."""
        task = self._tasks[task_id]
        task.status = TaskStatus.DONE
        task.result = result
        logger.debug(f"Task '{task_id}' done")

    def mark_failed(self, task_id: str, error: str) -> None:
        """Mark a task failed and recursively skip its dependents."""
        task = self._tasks[task_id]
        task.status = TaskStatus.FAILED
        task.error = error
        logger.warning(f"Task '{task_id}' failed: {error}")
        self._skip_downstream(task_id)

    def finalize_incomplete(self, reason: str) -> dict[str, list[str]]:
        """Put every unfinished task into a stable terminal state."""
        cancelled: list[str] = []
        skipped: list[str] = []
        for task_id, task in self._tasks.items():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED
                task.error = reason
                cancelled.append(task_id)
            elif task.status == TaskStatus.PENDING:
                task.status = TaskStatus.SKIPPED
                task.error = reason
                skipped.append(task_id)
        return {"cancelled": cancelled, "skipped": skipped}

    def execution_summary(self) -> dict[str, Any]:
        """Return the canonical execution status consumed by runner and replay."""
        counts: dict[str, int] = {}
        task_ids: dict[str, list[str]] = {
            "failed": [],
            "cancelled": [],
            "skipped": [],
        }
        for task_id, task in self._tasks.items():
            status = task.status.value
            counts[status] = counts.get(status, 0) + 1
            if status in task_ids:
                task_ids[status].append(task_id)
        incomplete = any(task_ids.values()) or any(
            task.status
            not in {
                TaskStatus.DONE,
                TaskStatus.FAILED,
                TaskStatus.SKIPPED,
                TaskStatus.CANCELLED,
            }
            for task in self._tasks.values()
        )
        return {
            "status": "incomplete" if incomplete else "complete",
            "counts": counts,
            "failed_task_ids": task_ids["failed"],
            "cancelled_task_ids": task_ids["cancelled"],
            "skipped_task_ids": task_ids["skipped"],
        }

    def _skip_downstream(self, failed_id: str) -> None:
        """Recursively skip pending tasks that depend on an unavailable task."""
        for tid, deps in self._deps.items():
            if failed_id in deps and self._tasks[tid].status == TaskStatus.PENDING:
                self._tasks[tid].status = TaskStatus.SKIPPED
                self._tasks[tid].error = f"Skipped dependency '{failed_id}' unavailable"
                logger.debug(f"Task '{tid}' skipped due to '{failed_id}' failure")
                self._skip_downstream(tid)

    def is_complete(self) -> bool:
        """Return whether every task is in a terminal state."""
        terminal = {
            TaskStatus.DONE,
            TaskStatus.FAILED,
            TaskStatus.SKIPPED,
            TaskStatus.CANCELLED,
        }
        return all(t.status in terminal for t in self._tasks.values())

    def get_results(self) -> dict[str, Any]:
        """Return results from completed tasks only."""
        return {
            tid: t.result
            for tid, t in self._tasks.items()
            if t.status == TaskStatus.DONE
        }

    @property
    def tasks(self) -> dict[str, SubTask]:
        """Return the tasks in insertion order."""
        return self._tasks

    @property
    def dependencies(self) -> dict[str, set[str]]:
        """Read-only graph dependency view for observability and replay."""
        return self._deps
