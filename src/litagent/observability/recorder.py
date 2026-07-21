"""Local, full-payload execution recording for the flow-demo archive."""

from __future__ import annotations

import copy
import json
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from litagent.logging import get_logger


logger = get_logger("observability.recorder")


class CompositeTraceHook:
    """Fan out events while keeping the local recorder independent of sinks."""

    def __init__(self, *hooks: Any) -> None:
        self._hooks = [hook for hook in hooks if hook is not None]

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        for hook in self._hooks:
            hook(event, data)

    def capture_graph(self, graph: Any) -> None:
        for hook in self._hooks:
            capture = getattr(hook, "capture_graph", None)
            if capture:
                capture(graph)

    def flush(self) -> None:
        for hook in self._hooks:
            flush = getattr(hook, "flush", None)
            if callable(flush):
                flush()


class RedactingTraceHook:
    """Forwards only operational metadata to a remote observability sink."""

    _ALLOWED_FIELDS = {
        "operation_id", "task_id", "agent_type", "model", "elapsed_ms",
        "usage", "error", "status", "attempt", "output_size", "top_k",
        "candidate_count", "result_count", "score", "quality", "delivery",
    }

    def __init__(self, sink: Any) -> None:
        self._sink = sink

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        self._sink(event, {key: value for key, value in data.items() if key in self._ALLOWED_FIELDS})

    def flush(self) -> None:
        flush = getattr(self._sink, "flush", None)
        if callable(flush):
            flush()


class ArchiveRepository:
    """File-backed archive store. Artifacts stay local and are never tracked."""

    def __init__(self, root: Path | str = "artifacts/runs") -> None:
        self._root = Path(root)

    def path_for(self, run_id: str) -> Path:
        return self._root / f"{run_id}.json"

    def save(self, artifact: dict[str, Any]) -> Path:
        run_id = str(artifact["run_id"])
        self._root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(run_id)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self._root, delete=False, suffix=".tmp"
        ) as handle:
            json.dump(artifact, handle, ensure_ascii=False, indent=2, default=_json_default)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(target)
        return target

    def get(self, run_id: str) -> dict[str, Any] | None:
        path = self.path_for(run_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable flow archive %s: %s", path, exc)
            return None

    def list(self) -> list[dict[str, Any]]:
        if not self._root.is_dir():
            return []
        artifacts = [artifact for path in self._root.glob("*.json") if (artifact := self._read(path))]
        return sorted(artifacts, key=lambda item: item.get("completed_at", ""), reverse=True)

    def _read(self, path: Path) -> dict[str, Any] | None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable flow archive %s: %s", path, exc)
            return None
        return value if isinstance(value, dict) and value.get("run_id") else None


class RunRecorder:
    """Trace hook that preserves full local payloads and assembles replay nodes."""

    def __init__(self, run_id: str, query: str, repository: ArchiveRepository) -> None:
        now = _now()
        self.run_id = run_id
        self._repository = repository
        self._artifact: dict[str, Any] = {
            "version": 1,
            "run_id": run_id,
            "query": query,
            "status": "running",
            "started_at": now,
            "completed_at": None,
            "events": [],
            "nodes": {},
            "graph": {"tasks": {}, "dependencies": {}},
            "report": None,
            "error": None,
        }

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        payload = _snapshot(data)
        record = {"sequence": len(self._artifact["events"]), "at": _now(), "event": event, "data": payload}
        self._artifact["events"].append(record)
        self._apply_event(event, payload, record["at"])

    def capture_graph(self, graph: Any) -> None:
        tasks = getattr(graph, "tasks", {})
        dependencies = getattr(graph, "dependencies", {})
        self._artifact["graph"] = {
            "tasks": {
                task_id: {
                    "task_id": task.task_id,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "input": _snapshot(task.input_data),
                    "status": str(task.status.value if hasattr(task.status, "value") else task.status),
                    "priority": task.priority,
                    "timeout_ms": task.timeout_ms,
                    "max_retries": task.max_retries,
                }
                for task_id, task in tasks.items()
            },
            "dependencies": {task_id: sorted(values) for task_id, values in dependencies.items()},
        }

    def finalize(self, report: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
        self._artifact["status"] = "failed" if error else "completed"
        self._artifact["completed_at"] = _now()
        self._artifact["report"] = _snapshot(report) if report is not None else None
        self._artifact["error"] = error
        self._repository.save(self._artifact)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return _snapshot(self._artifact)

    def _apply_event(self, event: str, data: dict[str, Any], at: str) -> None:
        node_key, lifecycle = _node_identity(event, data)
        if node_key is None:
            return
        nodes: dict[str, dict[str, Any]] = self._artifact["nodes"]
        node = nodes.setdefault(node_key, {
            "id": node_key,
            "kind": event.rsplit(".", 1)[0],
            "task_id": data.get("task_id", ""),
            "name": data.get("name") or data.get("agent_type") or event.rsplit(".", 1)[0],
            "status": "pending",
            "started_at": None,
            "completed_at": None,
            "input": None,
            "output": None,
            "error": None,
            "elapsed_ms": None,
            "usage": None,
        })
        if event == "worker.input":
            node["input"] = data.get("input", {})
            return
        if lifecycle == "start":
            node["status"] = "running"
            node["started_at"] = at
            node["input"] = data.get("input", data.get("messages", data.get("args", node["input"])))
            return
        if lifecycle == "complete":
            node["status"] = "completed"
            node["completed_at"] = at
            node["elapsed_ms"] = data.get("elapsed_ms")
            node["output"] = data.get("output", data.get("content", data.get("result", data.get("output_size"))))
            if event == "llm.complete":
                node["usage"] = {
                    "prompt_tokens": data.get("prompt_tokens", 0),
                    "completion_tokens": data.get("completion_tokens", 0),
                    "total_tokens": data.get("total_tokens", 0),
                }
            return
        if lifecycle in {"failed", "error"}:
            node["status"] = "failed"
            node["completed_at"] = at
            node["elapsed_ms"] = data.get("elapsed_ms")
            node["error"] = data.get("error") or data.get("error_type") or "unknown_error"


def _node_identity(event: str, data: dict[str, Any]) -> tuple[str | None, str | None]:
    if event == "worker.input":
        return f"worker:{data.get('task_id', '')}", None
    if event.startswith("worker."):
        return f"worker:{data.get('task_id', '')}", event.rsplit(".", 1)[-1]
    if event.startswith(("llm.", "tool.", "rag.search.", "memory.", "claims.")):
        operation_id = data.get("operation_id")
        if operation_id:
            return f"{event.rsplit('.', 1)[0]}:{operation_id}", event.rsplit(".", 1)[-1]
    if event == "subspan.start":
        return f"subspan:{data.get('task_id', '')}", "start"
    if event == "subspan.end":
        return f"subspan:{data.get('task_id', '')}", "complete"
    return None, None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _snapshot(value: Any) -> Any:
    try:
        return copy.deepcopy(value)
    except Exception:
        return _json_default(value)


def _json_default(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "value"):
        return value.value
    return f"<{type(value).__name__}>"
