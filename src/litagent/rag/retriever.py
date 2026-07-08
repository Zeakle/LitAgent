"""Hybrid Retriever——Qdrant hybrid search + optional Rerank。"""

import time

from litagent.rag.interfaces import Retriever, ScoredDoc, Reranker, VectorStore
from litagent.observability.context import get_task_id
from litagent.logging import get_logger

logger = get_logger('rag.retriever')


class HybridRetriever(Retriever):
    """Qdrant dual-index hybrid search + cross-encoder rerank。

    Qdrant 内部已做 RRF fusion——这里只负责：
    1. 调 vector_store.search()（内含 dense+sparse+RRF）
    2. 可选 rerank
    """

    def __init__(self, store: VectorStore, reranker: Reranker, trace_hook=None):
        self._store = store
        self._reranker = reranker
        self._trace_hook = trace_hook


    def _emit(self, event: str, data: dict) -> None:
        """触发 trace hook"""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")


    async def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        t0 = time.monotonic()
        results = await self._store.search(query, top_k * 2)
        if len(results) > top_k:
            results = self._reranker.rerank(query, results)
        results = results[:top_k]
        self._emit("rag.search", {
            "task_id": get_task_id(),
            "query": query[:200],
            "count": len(results),
            "elapsed_ms": (time.monotonic() - t0) * 1000,
        })
        return results