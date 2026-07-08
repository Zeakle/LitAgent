from __future__ import annotations
from typing import Any

from litagent.logging import get_logger


logger = get_logger('observability.tracing')


class LangFuseTracer:
    """消费 trace_hook 事件，产生 LangFuse trace + per-worker span。

    用法:
        tracer = LangFuseTracer(host=..., public_key=..., secret_key=...)
        agent = LitAgent(config, trace_hook=tracer)   # tracer 可直接当 hook 调
        ...
        tracer.flush()   # 短生命周期脚本结束前必须 flush

    事件流:
        survey.start   → 建 trace + root span
        worker.start   → 建子 span（存进 self._spans[task_id]）
        llm.call       → 在对应 Worker span 下建 generation（token/tool_calls/耗时）
        tool.call / rag.search / memory.recall / memory.write / claims.op
                       → 底层 I/O 点事件，挂对应 Worker span（无则 fallback root）
        worker.complete/failed → 关对应子 span
        survey.complete/error  → 清扫残留 span + 关 root span
    """

    def __init__(self, host: str, public_key: str, secret_key: str):
        self._client = None
        self._root = None
        self._spans: dict[str, Any] = {}  # task_id -> span

        try:
            if not (public_key and secret_key):
                logger.warning("LangFuse keys missing, tracing disabled")
                self._client = None
            else:
                from langfuse import Langfuse   # 延迟 import：langfuse 未装时 no-op 不崩
                self._client = Langfuse(
                    host=host, public_key=public_key, secret_key=secret_key
                )
                logger.info(f'LangFuse tracer initialized {host}')
        except Exception as e:
            logger.warning("LangFuse unavailable, tracing disabled: %s", e)
            self._client = None

    
    def __call__(self, event: str, data: dict[str, Any]) -> None:
        if not self._client:
            return

        try:
            self._handle(event, data)
        except Exception as e:
            logger.debug(f"Tracer error on {event}, {e}")


    def _handle(self, event: str, data: dict[str, Any]) -> None:
        if event == 'survey.start':
            self._root = self._client.start_observation(
                name='survey', as_type='span',
                input={'query': data.get('query', '')},
            )
        elif event == 'worker.start':
            if not self._root:
                return 
            tid = data.get('task_id', '')
            self._spans[tid] = self._root.start_observation(
                name=data.get('agent_type', tid), as_type='span',
                input={'description': data.get('description', '')},
            )
        elif event in ('worker.complete', 'worker.failed'):
            tid = data.get('task_id', '')
            span = self._spans.pop(tid, None)
            if span:
                if event == 'worker.failed':
                    span.update(level="ERROR", status_message=data.get("error", ""))
                span.end()
        elif event == 'llm.call':
            tid = data.get('task_id', '')
            parent = self._spans.get(tid) or self._root
            if not parent:
                return
            gen = parent.start_observation(
                name="llm.call", as_type="generation",   # generation = LLM 专用 span 类型
                model=data.get("model", ""),
                input=data.get("messages", []),
                output=data.get("content", ""),
            )

            gen.update(usage_details={
                "input": data.get("prompt_tokens", 0),
                "output": data.get("completion_tokens", 0),
                "total": data.get("total_tokens", 0),
            })

            tcs = data.get("tool_calls", [])
            if tcs:
                gen.update(metadata={"tool_calls": tcs})
            gen.end()
        elif event in ("tool.call", "rag.search", "memory.recall", "memory.write", "claims.op"):
            # 底层 I/O 事件（点事件）：挂在对应 Worker span 下（get_task_id → _spans[tid]）
            # 无对应 Worker span（如 consolidate 在 worker context 外）→ fallback 到 root
            tid = data.get("task_id", "")
            parent = self._spans.get(tid) or self._root
            if parent is None:
                return
            name = {
                "tool.call": data.get("name", "tool"),
                "rag.search": "rag.search",
                "memory.recall": "memory.recall",
                "memory.write": f"memory.write.{data.get('layer', '')}",
                "claims.op": f"claims.{data.get('op', '')}",
            }[event]
            sp = parent.start_observation(
                name=name,
                as_type="tool" if event == "tool.call" else "span",
                input={k: v for k, v in data.items() if k != "task_id"},
            )
            if data.get("success") is False or data.get("error"):
                sp.update(level="ERROR", status_message=data.get("error", ""))
            sp.end()
        elif event == 'survey.complete':
            self._close_orphans()
            if self._root is not None:
                self._root.update(
                    output={
                        "rounds": data.get("rounds", 0),
                        "accepted": data.get("accepted", False),
                    }
                )
                self._root.end()
                self._root = None
        elif event == "survey.error":
            self._close_orphans()
            if self._root is not None:
                self._root.update(level="ERROR", status_message=data.get("error", ""))
                self._root.end()
                self._root = None


    def _close_orphans(self) -> None:
        for tid, span in list(self._spans.items()):
            try:
                span.update(level="WARNING",
                            status_message="span not closed (cancelled/abandoned)")
                span.end()
            except Exception:
                pass
        self._spans.clear()


    def flush(self) -> None:
        """强制上报——短生命周期脚本/请求结束前调。"""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.debug("Flush error: %s", e)