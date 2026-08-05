"""Coordinate vector retrieval, optional reranking, and trace emission."""

import time
import uuid

from litagent.logging import get_logger
from litagent.observability.context import get_task_id
from litagent.rag.interfaces import Reranker, ScoredDoc, VectorStore
from litagent.rag.models import (
    ContentChunk,
    ContentScope,
    ScoredChunkHit,
    ScoredPaperHit,
)

logger = get_logger("rag.retriever")


def aggregate_chunk_hits(
    hits: list[ScoredChunkHit],
    *,
    top_k: int,
    max_chunks_per_paper: int = 4,
) -> list[ScoredPaperHit]:
    if top_k <= 0 or max_chunks_per_paper <= 0:
        raise ValueError("top_k and max_chunks_per_paper must be positive")
    grouped: dict[str, list[ScoredChunkHit]] = {}
    for hit in hits:
        grouped.setdefault(hit.chunk.paper_id, []).append(hit)

    papers: list[ScoredPaperHit] = []
    for paper_hits in grouped.values():
        ordered = sorted(paper_hits, key=lambda item: item.score, reverse=True)
        first = ordered[0]
        chunks: list[ContentChunk] = []
        seen: set[str] = set()
        for hit in ordered:
            if hit.chunk.chunk_key in seen:
                continue
            seen.add(hit.chunk.chunk_key)
            chunks.append(hit.chunk)
            if len(chunks) >= max_chunks_per_paper:
                break
        abstract = next(
            (chunk.text for chunk in chunks if chunk.chunk_key == "abstract"),
            first.abstract,
        )
        scope = (
            ContentScope.SELECTED_FULLTEXT
            if any(
                chunk.content_scope is ContentScope.SELECTED_FULLTEXT
                for chunk in chunks
            )
            else ContentScope.ABSTRACT if abstract else ContentScope.METADATA_ONLY
        )
        papers.append(
            ScoredPaperHit(
                paper_id=first.chunk.paper_id,
                title=first.title,
                abstract=abstract,
                authors=first.authors,
                year=first.year,
                sources=first.sources,
                warnings=first.warnings,
                content_scope=scope,
                chunks=chunks,
                score=first.score,
                collection=first.collection,
                corpus_version=first.corpus_version,
                schema_version=first.schema_version,
                parser_version=first.parser_version,
                chunking_version=first.chunking_version,
                embedding_model=first.embedding_model,
            )
        )
    return sorted(papers, key=lambda item: item.score, reverse=True)[:top_k]


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

    async def search_papers(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int | None = None,
        max_chunks_per_paper: int = 4,
    ) -> list[ScoredPaperHit]:
        """Retrieve versioned chunks and aggregate them into parent papers."""
        requested = max(candidate_k or top_k * max_chunks_per_paper, top_k)
        hits = await self._store.search_chunks(query, requested)
        return aggregate_chunk_hits(
            hits,
            top_k=top_k,
            max_chunks_per_paper=max_chunks_per_paper,
        )
