"""Local, full-payload execution recording for the flow-demo archive."""

from __future__ import annotations

import dataclasses
import json
import os
import re
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
        """Forward an initial task-graph snapshot to capable hooks."""
        for hook in self._hooks:
            capture = getattr(hook, "capture_graph", None)
            if capture:
                capture(graph)

    def capture_graph_state(self, graph: Any) -> None:
        """Forward a terminal task-graph snapshot to capable hooks."""
        for hook in self._hooks:
            capture = getattr(hook, "capture_graph_state", None)
            if capture:
                capture(graph)

    def flush(self) -> None:
        """Flush every hook that exposes a flush operation."""
        for hook in self._hooks:
            flush = getattr(hook, "flush", None)
            if callable(flush):
                flush()


class RedactingTraceHook:
    """Forward trace payloads after recursively removing credentials."""

    _ALLOWED_FIELDS = {
        "operation_id",
        "task_id",
        "agent_type",
        "model",
        "elapsed_ms",
        "usage",
        "error",
        "status",
        "attempt",
        "output_size",
        "top_k",
        "candidate_count",
        "result_count",
        "score",
        "quality",
        "delivery",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "error_type",
        "error_code",
        "name",
        "layer",
        "source",
        "count",
    }

    def __init__(self, sink: Any, payload_mode: str = "full_redacted") -> None:
        self._sink = sink
        self._payload_mode = payload_mode

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        if self._payload_mode == "metadata_only":
            payload = {
                key: _redact_trace_value(value)
                for key, value in data.items()
                if key in self._ALLOWED_FIELDS
            }
        else:
            payload = _redact_trace_value(data)
        self._sink(event, payload)

    def flush(self) -> None:
        """Flush the wrapped trace sink when supported."""
        flush = getattr(self._sink, "flush", None)
        if callable(flush):
            flush()


class ArchiveRepository:
    """Store completed run artifacts on the local filesystem."""

    def __init__(self, root: Path | str = "artifacts/runs") -> None:
        self._root = Path(root)

    def path_for(self, run_id: str) -> Path:
        """Return the archive path for a run."""
        return self._root / f"{run_id}.json"

    def save(self, artifact: dict[str, Any]) -> Path:
        """Persist an artifact atomically and return its final path."""
        run_id = str(artifact["run_id"])
        self._root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(run_id)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self._root, delete=False, suffix=".tmp"
        ) as handle:
            json.dump(
                artifact, handle, ensure_ascii=False, indent=2, default=_json_default
            )
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        temporary.replace(target)
        return target

    def get(self, run_id: str) -> dict[str, Any] | None:
        """Return a valid archived run, or None when unavailable."""
        path = self.path_for(run_id)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable flow archive %s: %s", path, exc)
            return None

    def list(self) -> list[dict[str, Any]]:
        """List valid archives from newest to oldest."""
        if not self._root.is_dir():
            return []
        artifacts = [
            artifact
            for path in self._root.glob("*.json")
            if (artifact := self._read(path))
        ]
        return sorted(
            artifacts, key=lambda item: item.get("completed_at", ""), reverse=True
        )

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
        self._started_monotonic = time.perf_counter()
        self.run_id = run_id
        self._repository = repository
        self._artifact: dict[str, Any] = {
            "version": 2,
            "run_id": run_id,
            "query": query,
            "status": "running",
            "started_at": now,
            "completed_at": None,
            "elapsed_ms": None,
            "session_id": None,
            "config": {},
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "task_statuses": {},
            "quality": None,
            "delivery": None,
            "events": [],
            "nodes": {},
            "graph": {"tasks": {}, "dependencies": {}},
            "report": None,
            "error": None,
        }

    def __call__(self, event: str, data: dict[str, Any]) -> None:
        payload = _snapshot(data)
        record = {
            "sequence": len(self._artifact["events"]),
            "at": _now(),
            "event": event,
            "data": payload,
        }
        self._artifact["events"].append(record)
        if event == "survey.start":
            self._artifact["session_id"] = payload.get("session_id")
        if event == "llm.complete":
            usage = self._artifact["usage"]
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                usage[key] += int(payload.get(key, 0) or 0)
        self._apply_event(event, payload, record["at"])

    def set_config_summary(self, summary: dict[str, Any]) -> None:
        """Store a serializable summary of effective run configuration."""
        self._artifact["config"] = _snapshot(summary)

    def capture_graph(self, graph: Any) -> None:
        """Capture the planned graph definition before execution."""
        tasks = getattr(graph, "tasks", {})
        dependencies = getattr(graph, "dependencies", {})
        self._artifact["graph"] = {
            "tasks": {
                task_id: {
                    "task_id": task.task_id,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "input": _snapshot(task.input_data),
                    "status": str(
                        task.status.value
                        if hasattr(task.status, "value")
                        else task.status
                    ),
                    "priority": task.priority,
                    "timeout_ms": task.timeout_ms,
                    "max_retries": task.max_retries,
                    "error": task.error,
                }
                for task_id, task in tasks.items()
            },
            "dependencies": {
                task_id: sorted(values) for task_id, values in dependencies.items()
            },
        }

    def capture_graph_state(self, graph: Any) -> None:
        """Synchronize terminal task states after execution."""
        if not self._artifact["graph"]["tasks"]:
            self.capture_graph(graph)
            return
        for task_id, task in getattr(graph, "tasks", {}).items():
            target = self._artifact["graph"]["tasks"].setdefault(
                task_id,
                {
                    "task_id": task.task_id,
                    "description": task.description,
                    "agent_type": task.agent_type,
                    "input": _snapshot(task.input_data),
                    "priority": task.priority,
                    "timeout_ms": task.timeout_ms,
                    "max_retries": task.max_retries,
                },
            )
            target["status"] = str(
                task.status.value if hasattr(task.status, "value") else task.status
            )
            target["error"] = task.error

    def finalize(
        self, report: dict[str, Any] | None = None, error: str | None = None
    ) -> dict[str, Any]:
        """Finalize, persist, and return the terminal run artifact."""
        self._artifact["status"] = "failed" if error else "completed"
        self._artifact["completed_at"] = _now()
        self._artifact["elapsed_ms"] = int(
            (time.perf_counter() - self._started_monotonic) * 1000
        )
        self._artifact["report"] = _snapshot(report) if report is not None else None
        self._artifact["error"] = error
        report_data = report or {}
        self._artifact["quality"] = _snapshot(report_data.get("quality"))
        self._artifact["delivery"] = _snapshot(report_data.get("delivery"))
        statuses: dict[str, int] = {}
        unfinished_graph_tasks: list[str] = []
        for task in self._artifact["graph"]["tasks"].values():
            status = task.get("status", "unknown")
            statuses[status] = statuses.get(status, 0) + 1
            if status in {"pending", "running"}:
                unfinished_graph_tasks.append(task.get("task_id", ""))
        self._artifact["task_statuses"] = statuses
        # A completed archive must not claim success with unfinished graph tasks.
        if not error and unfinished_graph_tasks:
            self._artifact["status"] = "failed"
            self._artifact["error"] = "incomplete_graph_state"
        if not error:
            self._close_unfinished_nodes()
        self._repository.save(self._artifact)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        """Return the current run snapshot."""
        return _snapshot(self._artifact)

    def _apply_event(self, event: str, data: dict[str, Any], at: str) -> None:
        node_key, lifecycle = _node_identity(event, data)
        if node_key is None:
            return
        nodes: dict[str, dict[str, Any]] = self._artifact["nodes"]
        node = nodes.setdefault(
            node_key,
            {
                "id": node_key,
                "kind": event.rsplit(".", 1)[0],
                "task_id": data.get("task_id", ""),
                "name": data.get("name")
                or data.get("agent_type")
                or event.rsplit(".", 1)[0],
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "input": None,
                "output": None,
                "error": None,
                "elapsed_ms": None,
                "usage": None,
                "metadata": {},
            },
        )
        if event == "worker.input":
            node["input"] = data.get("input", {})
            return
        if lifecycle == "start":
            node["status"] = "running"
            node["started_at"] = at
            event_input = _event_input(event, data)
            if event_input is not None:
                node["input"] = event_input
            return
        if lifecycle == "complete":
            node["status"] = "completed"
            node["completed_at"] = at
            node["elapsed_ms"] = _event_elapsed_ms(data, node["started_at"], at)
            node["output"] = _event_output(event, data)
            node["metadata"] = _event_metadata(event, data)
            if event == "llm.complete":
                node["usage"] = {
                    "prompt_tokens": data.get("prompt_tokens", 0),
                    "completion_tokens": data.get("completion_tokens", 0),
                    "total_tokens": data.get("total_tokens", 0),
                }
            return
        if lifecycle in {"failed", "error", "cancelled"}:
            node["status"] = "cancelled" if lifecycle == "cancelled" else "failed"
            node["completed_at"] = at
            node["elapsed_ms"] = _event_elapsed_ms(data, node["started_at"], at)
            node["error"] = (
                data.get("error") or data.get("error_type") or "unknown_error"
            )

    def _close_unfinished_nodes(self) -> None:
        completed_at = self._artifact["completed_at"]
        for node in self._artifact["nodes"].values():
            if node["status"] in {"pending", "running"}:
                node["status"] = "cancelled"
                node["completed_at"] = completed_at
                node["elapsed_ms"] = _event_elapsed_ms(
                    {}, node.get("started_at"), completed_at
                )
                node["error"] = "terminal_event_missing"


def _node_identity(event: str, data: dict[str, Any]) -> tuple[str | None, str | None]:
    if event == "worker.input":
        return f"worker:{data.get('task_id', '')}", None
    if event.startswith("worker."):
        return f"worker:{data.get('task_id', '')}", event.rsplit(".", 1)[-1]
    if event.startswith(
        ("llm.", "tool.", "rag.search.", "memory.", "claims.", "evidence.retrieve")
    ):
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
    return json.loads(json.dumps(value, ensure_ascii=False, default=_json_default))


def _event_input(event: str, data: dict[str, Any]) -> Any:
    if event == "worker.start":
        return None
    if event == "llm.start":
        return {
            key: data[key]
            for key in ("model", "messages", "max_tokens", "temperature")
            if key in data
        }
    if event == "tool.start":
        return data.get("args", {})
    ignored = {"operation_id", "task_id", "elapsed_ms"}
    payload = {key: value for key, value in data.items() if key not in ignored}
    return payload or None


def _event_output(event: str, data: dict[str, Any]) -> Any:
    if "output" in data:
        return data["output"]
    if event == "llm.complete":
        content = data.get("content", "")
        tool_calls = data.get("tool_calls", [])
        if tool_calls:
            return {"content": content, "tool_calls": tool_calls}
        return content
    ignored = {
        "operation_id",
        "task_id",
        "name",
        "agent_type",
        "model",
        "elapsed_ms",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    }
    payload = {key: value for key, value in data.items() if key not in ignored}
    return payload or None


def _event_metadata(event: str, data: dict[str, Any]) -> dict[str, Any]:
    if event == "llm.complete":
        return {"tool_calls": data.get("tool_calls", [])}
    if "output" in data:
        ignored = {
            "operation_id",
            "task_id",
            "name",
            "agent_type",
            "output",
            "elapsed_ms",
        }
        return {key: value for key, value in data.items() if key not in ignored}
    return {}


def _event_elapsed_ms(
    data: dict[str, Any],
    started_at: str | None,
    completed_at: str | None,
) -> int | None:
    explicit = data.get("elapsed_ms")
    if explicit is not None:
        return int(explicit)
    if not started_at or not completed_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        completed = datetime.fromisoformat(completed_at)
    except (TypeError, ValueError):
        return None
    return max(0, int((completed - started).total_seconds() * 1000))


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "value"):
        return value.value
    return f"<{type(value).__name__}>"


_SECRET_KEYS = {
    "authorization",
    "proxy_authorization",
    "cookie",
    "set_cookie",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "auth_token",
    "private_key",
    "token",
    "session_token",
}
_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:authorization\s*[:=]\s*bearer\s+\S+|bearer\s+[a-z0-9._-]{8,}|"
    r"(?<![a-z0-9])(?:sk|rk|pk)-[a-z0-9_-]{16,}(?![a-z0-9_-])|"
    r"(?:api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|"
    r"auth[_-]?token|password|secret)\s*[:=]\s*\S+|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)


def _normalized_key(key: Any) -> str:
    return str(key).strip().lower().replace("-", "_")


def _is_secret_key(key: Any) -> bool:
    normalized = _normalized_key(key)
    return normalized in _SECRET_KEYS or any(
        normalized.endswith(f"_{suffix}")
        for suffix in (
            "api_key",
            "access_token",
            "refresh_token",
            "session_token",
            "auth_token",
            "client_secret",
        )
    )


def _redact_trace_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _redact_trace_value(item)
            for key, item in value.items()
            if not _is_secret_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_redact_trace_value(item) for item in value]
    if isinstance(value, str) and _SECRET_VALUE_RE.search(value):
        return _SECRET_VALUE_RE.sub("<redacted>", value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return _redact_trace_value(_snapshot(value))
