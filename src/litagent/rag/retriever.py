"""Coordinate vector retrieval, optional reranking, and trace emission."""

import time
import uuid

from litagent.rag.interfaces import ScoredDoc, Reranker, VectorStore
from litagent.observability.context import get_task_id
from litagent.logging import get_logger

logger = get_logger("rag.retriever")


class HybridRetriever:
    """Retrieve an expanded candidate set and optionally rerank it."""

    def __init__(
        self, store: VectorStore, reranker: Reranker | None = None, trace_hook=None
    ):
        self._store = store
        self._reranker = reranker
        self._trace_hook = trace_hook

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event without allowing hook failures to escape."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    async def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        """Retrieve documents and emit one terminal lifecycle trace event."""
        op_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        self._emit(
            "rag.search.start",
            {
                "operation_id": op_id,
                "task_id": get_task_id(),
                "query": query,
                "top_k": top_k,
            },
        )

        try:
            results = await self._store.search(query, top_k * 2)
            if len(results) > top_k and self._reranker is not None:
                results = self._reranker.rerank(query, results)
            results = results[:top_k]

            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._emit(
                "rag.search.complete",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "count": len(results),
                    "results": [
                        {
                            "content": item.doc.page_content,
                            "metadata": item.doc.metadata,
                            "score": item.score,
                        }
                        for item in results
                    ],
                    "elapsed_ms": elapsed_ms,
                },
            )
            return results
        # Trace cancellation and other base exceptions before propagating them.
        except BaseException as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._emit(
                "rag.search.failed",
                {
                    "operation_id": op_id,
                    "task_id": get_task_id(),
                    "elapsed_ms": elapsed_ms,
                    "error_type": type(e).__name__,
                    "error": str(e)[:512],
                },
            )
            raise
