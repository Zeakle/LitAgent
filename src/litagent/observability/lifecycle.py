"""I/O lifecycle 追踪辅助——统一 *.start / *.complete / *.failed 配对契约。

所有 await 型 I/O 在调用前 emit start，成功或失败路径只 emit 一次终态。
BaseException 捕获保证 CancelledError 也能关闭 observation。
"""

from __future__ import annotations
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Callable

from litagent.observability.context import get_task_id


_SENSITIVE_KEY_PARTS = ('key', 'token', 'secret', 'password')
_MAX_STR = 200


def sanitize_input(data: dict[str, Any]) -> dict[str, Any]:
    """截断长字符串 + 剔除疑似密钥字段——trace input 不含密钥/全文。

    容器值统一 repr 截断——嵌套 dict/list 里的长文本/密钥不逐层深挖，直接压平。
    """
    out: dict[str, Any] = {}
    for k, v in data.items():
        if any(part in k.lower() for part in _SENSITIVE_KEY_PARTS):
            continue
        if isinstance(v, str) and len(v) > _MAX_STR:
            out[k] = v[:_MAX_STR] + '...'
        elif isinstance(v, (dict, list, tuple)):
            r = repr(v)
            out[k] = r if len(r) <= _MAX_STR else r[:_MAX_STR] + '...'
        else:
            out[k] = v
    return out


@asynccontextmanager
async def traced_io(emit: Callable[[str, dict], None], namespace: str, input_data: dict[str, Any] | None = None):
    """Tracing一次真实外部 I/O"""
    op_id = uuid.uuid4().hex
    t0 = time.perf_counter()

    emit(f'{namespace}.start', {
        'operation_id': op_id,
        'task_id': get_task_id(),
        **sanitize_input(input_data or {}),
    })

    outcome: dict[str, Any] = {}
    try:
        yield outcome
    except BaseException as e:
        emit(f'{namespace}.failed', {
            'operation_id': op_id,
            'task_id': get_task_id(),
            'elapsed_ms': int((time.perf_counter() - t0) * 1000),
            'error_type': type(e).__name__,
            'error': str(e)[:512]
        })
        raise

    emit(f'{namespace}.complete', {
        'operation_id': op_id,
        'task_id': get_task_id(),
        'elapsed_ms': int((time.perf_counter() - t0) * 1000),
        **outcome,
    })