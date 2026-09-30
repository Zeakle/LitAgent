"""Model survey subtasks, dependencies, and terminal execution state."""

from __future__ import annotations

import json
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

    def to_snapshot(self) -> dict[str, Any]:
        """Serialize graph state at a quiescent execution boundary."""
        snapshot = {
            "schema_version": 1,
            "tasks": [
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "input_data": task.input_data,
                    "output_schema": task.output_schema,
                    "status": task.status.value,
                    "priority": task.priority,
                    "timeout_ms": task.timeout_ms,
                    "max_retries": task.max_retries,
                    "result": task.result,
                    "error": task.error,
                }
                for task in self._tasks.values()
            ],
            "dependencies": {
                task_id: sorted(dependencies)
                for task_id, dependencies in self._deps.items()
            },
        }
        try:
            json.dumps(snapshot)
        except (TypeError, ValueError) as exc:
            raise ValueError("task_graph_not_json_serializable") from exc
        return snapshot

    @classmethod
    def from_snapshot(cls, data: dict[str, Any]) -> "TaskGraph":
        """Restore and validate a graph without converting corruption to new work."""
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("unsupported_task_graph_snapshot")
        raw_tasks = data.get("tasks")
        dependencies = data.get("dependencies")
        if not isinstance(raw_tasks, list) or not isinstance(dependencies, dict):
            raise ValueError("invalid_task_graph_snapshot")

        graph = cls()
        seen: set[str] = set()
        for raw in raw_tasks:
            if not isinstance(raw, dict):
                raise ValueError("invalid_task_record")
            task_id = raw.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in seen:
                raise ValueError("invalid_or_duplicate_task_id")
            seen.add(task_id)
            try:
                status = TaskStatus(raw.get("status", "pending"))
            except ValueError as exc:
                raise ValueError("invalid_task_status") from exc
            if status == TaskStatus.RUNNING:
                raise ValueError("running_task_not_resumable")
            if status == TaskStatus.DONE and "result" not in raw:
                raise ValueError("done_task_missing_result")
            graph._tasks[task_id] = SubTask(
                task_id=task_id,
                description=str(raw.get("description") or ""),
                agent_type=str(raw.get("agent_type") or ""),
                input_data=dict(raw.get("input_data") or {}),
                output_schema=dict(raw.get("output_schema") or {}),
                status=status,
                priority=int(raw.get("priority", 0)),
                timeout_ms=int(raw.get("timeout_ms", 120000)),
                max_retries=int(raw.get("max_retries", 2)),
                result=raw.get("result"),
                error=raw.get("error"),
            )

        if set(dependencies) != seen:
            raise ValueError("dependency_keys_mismatch")
        for task_id, raw_deps in dependencies.items():
            if not isinstance(raw_deps, list) or any(
                not isinstance(dep, str) or dep not in seen for dep in raw_deps
            ):
                raise ValueError("unknown_dependency")
            graph._deps[task_id] = set(raw_deps)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError("cyclic_task_graph")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in graph._deps[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in seen:
            visit(task_id)
        return graph

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
