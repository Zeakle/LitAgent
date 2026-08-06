"""Recall indexed papers for survey planning."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from litagent.config import RetrievalMode
from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.models import PaperCandidate
from litagent.rag.retriever import HybridRetriever

logger = get_logger("agents.recall")


class RecallWorker(Worker):
    """Retrieve related papers from the configured hybrid index."""

    def __init__(self, retriever: HybridRetriever | None, *, trace_hook=None):
        self._retriever = retriever
        self._trace_hook = trace_hook

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "recall"

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception:
                logger.debug("Recall trace hook failed", exc_info=True)

    async def execute(self, task: SubTask) -> list[dict[str, Any]]:
        """Retrieve papers or return an explicitly traced degraded result."""
        query = str(task.input_data.get("query") or "")
        top_k = int(task.input_data.get("top_k", 20))
        candidate_k = int(task.input_data.get("candidate_k", max(top_k, 40)))
        max_chunks = int(task.input_data.get("max_representative_chunks", 4))
        mode = RetrievalMode(task.input_data.get("retrieval_mode", "rrf"))
        if not self._retriever or not query:
            return []

        operation_id = uuid.uuid4().hex
        started = time.perf_counter()
        trace_input = {
            "operation_id": operation_id,
            "task_id": task.task_id,
            "query": query,
            "top_k": top_k,
            "candidate_k": candidate_k,
            "max_representative_chunks": max_chunks,
            "retrieval_mode": mode.value,
            "strict": False,
        }
        self._emit("rag.recall.start", trace_input)
        try:
            hits = await self._retriever.search_papers(
                query,
                top_k=top_k,
                candidate_k=candidate_k,
                max_chunks_per_paper=max_chunks,
                mode=mode,
                strict=False,
            )
        except asyncio.CancelledError:
            self._emit(
                "rag.recall.failed",
                {
                    **trace_input,
                    "reason_code": "recall_cancelled",
                    "error_type": "CancelledError",
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            raise
        except Exception as exc:
            logger.warning("Recall failed for %r: %s", query, exc)
            self._emit(
                "rag.recall.failed",
                {
                    **trace_input,
                    "reason_code": "recall_failed",
                    "error_type": type(exc).__name__,
                    "elapsed_ms": int((time.perf_counter() - started) * 1000),
                },
            )
            return []

        candidates = [PaperCandidate.from_scored_hit(hit).to_dag_dict() for hit in hits]
        first = hits[0] if hits else None
        self._emit(
            "rag.recall.complete",
            {
                **trace_input,
                "result_count": len(candidates),
                "collection": first.collection if first else None,
                "corpus_version": first.corpus_version if first else None,
                "schema_version": first.schema_version if first else None,
                "parser_version": first.parser_version if first else None,
                "chunking_version": first.chunking_version if first else None,
                "chunk_strategy": first.chunk_strategy if first else None,
                "embedding_backend": first.embedding_backend if first else None,
                "embedding_model": first.embedding_model if first else None,
                "content_scopes": sorted(
                    {item["content_scope"] for item in candidates}
                ),
                "empty_index": not candidates,
                "results": candidates,
                "elapsed_ms": int((time.perf_counter() - started) * 1000),
            },
        )
        return candidates
