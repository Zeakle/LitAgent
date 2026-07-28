"""Recall indexed papers for survey planning."""

from __future__ import annotations

from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.retriever import HybridRetriever
from litagent.logging import get_logger

logger = get_logger("agents.recall")


class RecallWorker(Worker):
    """Retrieve related papers from the configured hybrid index."""

    def __init__(self, retriever: HybridRetriever | None):
        self._retriever = retriever

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "recall"

    async def execute(self, task: SubTask) -> list[dict[str, Any]]:
        """Retrieve related papers or return an empty degraded result."""
        query = task.input_data.get("query", "")
        top_k = task.input_data.get("top_k", 20)

        if not self._retriever or not query:
            return []

        try:
            scored_docs = await self._retriever.search(query, top_k=top_k)
        except Exception as e:
            logger.warning(f"Recall failed for {query!r}: {e}")
            return []

        return [
            {
                "paper_id": sd.doc.metadata.get("arxiv_id", ""),
                "title": sd.doc.metadata.get("title", ""),
                "abstract": sd.doc.page_content[:500],
                "source": "rag_index",
                "score": sd.score,
            }
            for sd in scored_docs
        ]
