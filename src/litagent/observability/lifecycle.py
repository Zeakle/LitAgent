"""I/O lifecycle 追踪辅助——统一 *.start / *.complete / *.failed 配对契约。

所有 await 型 I/O 在调用前 emit start，成功或失败路径只 emit 一次终态。
BaseException 捕获保证 CancelledError 也能关闭 observation。
"""

from __future__ import annotations
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable, Mapping

from litagent.observability.context import get_task_id


_SENSITIVE_KEY_TOKENS = ('key', 'token', 'secret', 'password',
                         'authorization', 'cookie', 'session')


_RESERVED_FIELDS = frozenset({
    "operation_id", "task_id", "elapsed_ms", "error_code", "error_type",
})


_SECRET_VALUE_RE = re.compile(
    r"(?i)(?:bearer\s+\S+|(?:sk|rk|pk)-[a-z0-9_-]{8,}|"
    r"(?:api[_-]?key|token|secret|password)\s*[=:]\s*\S+)"
)


_MAX_STR = 200
_MAX_DEPTH = 5
_MAX_ITEMS = 50


def sanitize_input(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """递归清洗——返回有界、可序列化的 trace payload，不含敏感值。"""
    if data is None:
        return {}
    return _sanitize_mapping(data, depth=0)


def _safe_key(key: Any) -> str:
    """避免对非字符串键调用 __str__（可能触发用户代码）。"""
    return key if isinstance(key, str) else f"<{type(key).__name__}>"


def _sanitize_string(value: str) -> str:
    if _SECRET_VALUE_RE.search(value):
        return "<redacted>"
    return value[:_MAX_STR] + "..." if len(value) > _MAX_STR else value


def _sanitize_mapping(d: Mapping[str, Any], depth: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, (k, v) in enumerate(d.items()):
        if i >= _MAX_ITEMS:
            out['_truncated'] = f'{len(d) - _MAX_ITEMS} more keys omitted'
            break

        safe_key = _safe_key(k)

        if _is_sensitive_key(safe_key):
            continue

        out[safe_key] = _sanitize_value(v, depth)
    return out


def _is_sensitive_key(key: str) -> bool:
    low = key.lower()
    return any(t in low for t in _SENSITIVE_KEY_TOKENS)


def _sanitize_value(v: Any, depth: int) -> Any:
    if v is None or isinstance(v, (bool, int, float)):
        return v

    if isinstance(v, str):
        return _sanitize_string(v)

    if isinstance(v, Mapping):
        if depth >= _MAX_DEPTH:
            return f'<{type(v).__name__} depth-limit>'
        return _sanitize_mapping(v, depth + 1)

    if isinstance(v, (list, tuple, set)):
        if depth >= _MAX_DEPTH:
            return f'<{type(v).__name__} depth-limit>'

        items: list[Any] = []
        for i, item in enumerate(v):
            if i >= _MAX_ITEMS:
                items.append(f'<{len(v) - _MAX_ITEMS} more items omitted>')
                break
            items.append(_sanitize_value(item, depth + 1))
        return items

    # Unknown data type
    try:
        return f'<{type(v).__name__}>'
    except Exception:
        return '<unknown>'


def _sanitize_unreserved(data: Mapping[str, Any] | None) -> dict[str, Any]:
    """清洗后剔除框架保留字段——防 producer 注入同名键覆盖。"""
    return {
        key: value
        for key, value in sanitize_input(data).items()
        if key not in _RESERVED_FIELDS
    }


@asynccontextmanager
async def traced_io(emit: Callable[[str, dict], None], namespace: str, input_data: dict[str, Any] | None = None):
    """Tracing一次真实外部 I/O"""
    operation_id = uuid.uuid4().hex
    task_id = get_task_id()
    started = time.perf_counter()

    start = {"operation_id": operation_id, "task_id": task_id}
    start.update(_sanitize_unreserved(input_data))
    emit(f"{namespace}.start", start)

    outcome: dict[str, Any] = {}
    try:
        yield outcome
    except BaseException as exc:
        emit(f"{namespace}.failed", {
            "operation_id": operation_id,
            "task_id": task_id,
            "elapsed_ms": int((time.perf_counter() - started) * 1000),
            "error_code": "io_failed",
            "error_type": type(exc).__name__,
        })
        raise

    complete = {
        "operation_id": operation_id,
        "task_id": task_id,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
    }
    complete.update(_sanitize_unreserved(outcome))
    emit(f"{namespace}.complete", complete)
