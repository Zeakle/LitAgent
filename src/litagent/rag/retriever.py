"""Hybrid Retriever——Qdrant hybrid search + optional Rerank。"""


from litagent.rag.interfaces import Retriever, ScoredDoc, Reranker
from litagent.rag.vector_store import VectorStore
from litagent.logging import get_logger

logger = get_logger('rag.retriever')


class HybridRetriever(Retriever):
    """Qdrant dual-index hybrid search + cross-encoder rerank。

    Qdrant 内部已做 RRF fusion——这里只负责：
    1. 调 vector_store.search()（内含 dense+sparse+RRF）
    2. 可选 rerank
    """

    def __init__(self, store: VectorStore, reranker: Reranker):
        self._store = store
        self._reranker = reranker


    async def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        results = await self._store.search(query, top_k * 2)
        if len(results) > top_k:
            results = self._reranker.rerank(query, results)
        return results[:top_k]