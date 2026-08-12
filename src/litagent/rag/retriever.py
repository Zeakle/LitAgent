"""Coordinate vector retrieval, optional reranking, and trace emission."""

import time
import uuid

from litagent.config import RetrievalMode
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
        ordered = sorted(
            paper_hits,
            key=lambda item: (
                -item.score,
                item.chunk.chunk_key,
                item.chunk.content_hash,
            ),
        )
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
                chunk_strategy=first.chunk_strategy,
                chunk_size=first.chunk_size,
                chunk_overlap=first.chunk_overlap,
                embedding_backend=first.embedding_backend,
                embedding_model=first.embedding_model,
                embedding_document_adapter=first.embedding_document_adapter,
            )
        )
    # Qdrant does not guarantee order for equal ANN/RRF scores. A stable
    # paper-id tie break keeps runtime output and repeated benchmarks aligned.
    return sorted(papers, key=lambda item: (-item.score, item.paper_id))[:top_k]


class HybridRetriever:
    """Coordinate typed retrieval, parent aggregation, reranking, and trace."""

    def __init__(
        self,
        store: VectorStore,
        reranker: Reranker | None = None,
        trace_hook=None,
    ) -> None:
        self._store = store
        self._reranker = reranker
        self._trace_hook = trace_hook

    def _emit(self, event: str, data: dict) -> None:
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as exc:
                logger.debug("Trace hook failed for '%s': %s", event, exc)

    async def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        """Preserve the legacy Document retrieval surface."""
        operation_id = uuid.uuid4().hex
        started = time.perf_counter()
        self._emit(
            "rag.search.start",
            {
                "operation_id": operation_id,
                "task_id": get_task_id(),
                "query": query,
                "top_k": top_k,
                "retrieval_mode": "legacy_rrf",
            },
        )
        try:
            results = await self._store.search(query, top_k * 2)
            if len(results) > top_k and self._reranker is not None:
                results = self._reranker.rerank(query, results)
            results = results[:top_k]
            self._emit(
                "rag.search.complete",
                {
                    "operation_id": operation_id,
                    "task_id": get_task_id(),
                    "count": len(results),
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return results
        except BaseException as exc:
            self._emit(
                "rag.search.failed",
                {
                    "operation_id": operation_id,
                    "task_id": get_task_id(),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:512],
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            raise

    async def search_papers(
        self,
        query: str,
        top_k: int = 20,
        candidate_k: int | None = None,
        max_chunks_per_paper: int = 4,
        *,
        mode: RetrievalMode = RetrievalMode.RRF,
        strict: bool = False,
    ) -> list[ScoredPaperHit]:
        """Run a typed chunk search and return unique ranked papers."""
        if top_k <= 0 or max_chunks_per_paper <= 0:
            raise ValueError("top_k and max_chunks_per_paper must be positive")
        requested = max(candidate_k or top_k * max_chunks_per_paper, top_k)
        operation_id = uuid.uuid4().hex
        started = time.perf_counter()
        store_mode = RetrievalMode.RRF if mode is RetrievalMode.RRF_RERANK else mode
        self._emit(
            "rag.search.start",
            {
                "operation_id": operation_id,
                "task_id": get_task_id(),
                "query": query,
                "retrieval_mode": mode.value,
                "strict": strict,
                "top_k": top_k,
                "requested_chunk_candidates": requested,
            },
        )
        rerank_applied = False
        degradation_reason: str | None = None
        try:
            hits = await self._store.search_chunks(
                query,
                requested,
                mode=store_mode,
            )
            papers = aggregate_chunk_hits(
                hits,
                top_k=requested,
                max_chunks_per_paper=max_chunks_per_paper,
            )
            if mode is RetrievalMode.RRF_RERANK:
                if self._reranker is None:
                    if strict:
                        raise RuntimeError("reranker_unavailable")
                    degradation_reason = "reranker_unavailable"
                else:
                    try:
                        papers = self._reranker.rerank_papers(query, papers)
                        rerank_applied = True
                    except Exception as exc:
                        if strict:
                            raise RuntimeError("reranker_failed") from exc
                        degradation_reason = "reranker_failed"
                if degradation_reason:
                    self._emit(
                        "rag.search.degraded",
                        {
                            "operation_id": operation_id,
                            "task_id": get_task_id(),
                            "retrieval_mode": mode.value,
                            "degradation_reason": degradation_reason,
                        },
                    )

            results = papers[:top_k]
            first = results[0] if results else None
            self._emit(
                "rag.search.complete",
                {
                    "operation_id": operation_id,
                    "task_id": get_task_id(),
                    "retrieval_mode": mode.value,
                    "strict": strict,
                    "requested_chunk_candidates": requested,
                    "raw_hit_count": len(hits),
                    "parent_paper_count": len(papers),
                    "result_count": len(results),
                    "reranker_model": getattr(
                        self._reranker,
                        "model_name",
                        None,
                    ),
                    "rerank_applied": rerank_applied,
                    "degradation_reason": degradation_reason,
                    "collection_identity": first.collection if first else None,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return results
        except BaseException as exc:
            self._emit(
                "rag.search.failed",
                {
                    "operation_id": operation_id,
                    "task_id": get_task_id(),
                    "retrieval_mode": mode.value,
                    "strict": strict,
                    "reason_code": str(exc)[:128],
                    "error_type": type(exc).__name__,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            raise
