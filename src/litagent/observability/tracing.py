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
        self._operations: dict[str, Any] = {}
        self._worker_inputs: dict[str, Any] = {}

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
                as_type='span',
                name='survey',
                input={'query': data.get('query', '')},
            )
            self._root.update_trace(
                session_id=data.get('session_id', '')
            )

        elif event == 'worker.input':
            self._worker_inputs[data.get('task_id', '')] = data.get('input', {})

        elif event == 'worker.start':
            if not self._root:
                return 
            tid = data.get('task_id', '')
            worker_input = self._worker_inputs.pop(tid, None)
            self._spans[tid] = self._root.start_observation(
                name=data.get('agent_type', tid), as_type='span',
                input=(worker_input if worker_input is not None else {
                    'description': data.get('description', '')
                }),
            )

        elif event in ('worker.complete', 'worker.failed', 'worker.cancelled'):
            tid = data.get('task_id', '')
            span = self._spans.pop(tid, None)
            if span:
                if event == 'worker.failed':
                    span.update(level="ERROR", status_message=data.get("error", ""))
                elif event == 'worker.cancelled':
                    span.update(level="WARNING", status_message=data.get("error", "cancelled"))
                elif 'output' in data:
                    self._safe_update_output(span, data['output'])
                span.end()

        elif event == 'llm.start':
            parent = self._spans.get(data.get('task_id', '')) or self._root
            if parent is None:
                return
            gen = parent.start_observation(
                as_type='generation',
                name=f"llm:{data.get('model', '')}",
                input=data.get('messages', []),
                model=data.get('model', ''),
            )
            self._operations[data['operation_id']] = gen

        elif event == 'llm.complete':
            op_id = data.get('operation_id')
            if op_id and op_id in self._operations:
                gen = self._operations.pop(op_id)
                output: Any = data.get('content', '')
                if data.get('tool_calls'):
                    output = {
                        'content': data.get('content', ''),
                        'tool_calls': data['tool_calls'],
                    }
                gen.update(
                    output=output,
                    usage_details={
                        'prompt_tokens': data.get('prompt_tokens', 0),
                        'completion_tokens': data.get('completion_tokens', 0),
                        'total_tokens': data.get('total_tokens', 0),
                    },
                    metadata={
                        'elapsed_ms': data.get('elapsed_ms', 0)
                    }
                )
                gen.end()

        elif event == 'llm.failed':
            op_id = data.get('operation_id')
            if op_id and op_id in self._operations:
                gen = self._operations.pop(op_id)
                gen.update(
                    level='ERROR',
                    status_message=data.get('error', ''),
                    metadata={'elapsed_ms': data.get('elapsed_ms', 0)},
                )
                gen.end()

        elif event == 'subspan.start':
            # 嵌套子 span：挂到 parent_task_id 对应的 span 下（非 root）。
            # 用于 adversarial 内部直接调用的 synthesis/reviewer，让它们的
            # llm.call 归到各自子 span，而非全扁平挂到 adversarial。
            parent_tid = data.get('parent_task_id', '')
            sub_tid = data.get('task_id', '')
            parent = self._spans.get(parent_tid) or self._root
            if parent is None:
                return
            self._spans[sub_tid] = parent.start_observation(
                name=data.get('name', sub_tid), as_type='span',
                input={'round': data.get('round', 0)},
            )

        elif event == 'rag.search.start':
            parent = self._spans.get(data.get('task_id', '')) or self._root
            if parent is None:
                return
            span = parent.start_observation(
                as_type='span',
                name="rag.search",
                input={
                    "query": data.get("query", ""),
                    "top_k": data.get("top_k", 0),
                },
            )
            self._operations[data['operation_id']] = span

        elif event == 'rag.search.complete':
            op_id = data.get('operation_id')
            if op_id and op_id in self._operations:
                span = self._operations.pop(op_id)
                span.update(
                    output={
                        'count': data.get('count', 0),
                        'results': data.get('results', []),
                    },
                    metadata={'elapsed_ms': data.get('elapsed_ms', 0)},
                )
                span.end()

        elif event == 'rag.search.failed':
            op_id = data.get('operation_id')
            if op_id and op_id in self._operations:
                span = self._operations.pop(op_id)
                span.update(
                    level='ERROR',
                    status_message=data.get('error', ''),
                    metadata={'elapsed_ms': data.get('elapsed_ms', 0)}
                )
                span.end()

        elif event == 'subspan.end':
            sub_tid = data.get('task_id', '')
            span = self._spans.pop(sub_tid, None)
            if span:
                if data.get('error'):
                    span.update(level="ERROR", status_message=data.get('error', ''))
                elif 'output' in data:
                    self._safe_update_output(span, data['output'])
                span.end()

        elif self._is_io_lifecycle(event):
            self._handle_io_event(event, data)

        elif event == 'survey.complete':
            self._close_orphans()
            if self._root is not None:
                quality_status = data.get('quality_status', 'unverified')
                self._root.update(
                    output={
                        "rounds": data.get("rounds", 0),
                        "accepted": data.get("accepted", False),
                        'quality_status': quality_status,
                        'total_tokens': data.get('total_tokens', 0),
                        'delivery_status': data.get('delivery_status', '')
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

    
    _IO_NAMESPACES = ('tool', 'claims', 'memory', 'evidence')

    
    def _is_io_lifecycle(self, event: str) -> bool:
        root = event.split('.', 1)[0]
        return (root in self._IO_NAMESPACES and event.endswith(('.start', '.complete', '.failed')))


    def _handle_io_event(self, event: str, data: dict[str, Any]) -> None:
        namespace, phase = event.rsplit('.', 1)
        op_id = data.get('operation_id')
        if not op_id:
            return

        if phase == 'start':
            parent = self._spans.get(data.get('task_id', '')) or self._root
            if not parent:
                return

            if namespace == 'tool':
                name = f"tool:{data.get('name', '')}"
                as_type = 'tool'
            elif namespace == 'memory.write':
                name = f"memory.write.{data.get('layer', '')}"
                as_type = 'span'
            else:
                name = namespace
                as_type = 'span'
            
            self._operations[op_id] = parent.start_observation(
                as_type=as_type,
                name=name,
                input={k: v for k, v in data.items() if k not in ('operation_id', 'task_id')},
            )

            return
        
        obs = self._operations.pop(op_id, None)

        if not obs:
            return

        if phase == "failed":
            # 只消费稳定码和异常类名，不复读 raw error 字段
            status = data.get("error_code") or data.get("error_type") or "io_failed"
            obs.update(
                level="ERROR",
                status_message=status,
                metadata={
                    "elapsed_ms": data.get("elapsed_ms", 0),
                    "error_type": data.get("error_type", ""),
                    "error_code": data.get("error_code", "io_failed"),
                },
            )
        else:
            obs.update(
                output={
                    key: value for key, value in data.items()
                    if key not in ("operation_id", "task_id", "elapsed_ms")
                },
                metadata={"elapsed_ms": data.get("elapsed_ms", 0)},
            )
        obs.end()



    def _safe_update_output(self, span, output) -> None:
        """填 span output，序列化失败则降级为 repr 截断。"""
        try:
            span.update(output=output)
        except Exception:
            try:
                span.update(output={"serialization_error": type(output).__name__})
            except Exception:
                pass   # 追踪失败绝不影响业务


    def _close_orphans(self) -> None:
        for tid, span in list(self._spans.items()):
            try:
                span.update(level="WARNING",
                            status_message="span not closed (cancelled/abandoned)")
                span.end()
            except Exception:
                pass
        self._spans.clear()
        self._worker_inputs.clear()
        for op_id, obs in list(self._operations.items()):
            try:
                obs.update(level="WARNING", status_message="Orphan operation (closed by cleanup)")
                obs.end()
            except Exception:
                pass
        self._operations.clear()


    def flush(self) -> None:
        """强制上报——短生命周期脚本/请求结束前调。"""
        if self._client is not None:
            try:
                self._client.flush()
            except Exception as e:
                logger.debug("Flush error: %s", e)
