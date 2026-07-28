"""Deduplicate paper-search results by normalized title."""

from __future__ import annotations

from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger

logger = get_logger("agent.dedup")


class DedupWorker(Worker):
    """Merge upstream paper lists and remove duplicate titles."""

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "dedup"

    async def execute(self, task: SubTask) -> Any:
        """Merge upstream results and keep one paper per normalized title."""
        upstream = task.input_data.get("upstream_results", {})

        all_papers: list[dict] = []
        for task_id, papers in upstream.items():
            if isinstance(papers, list):
                all_papers.extend(papers)

        deduped = self._dedup_by_title(all_papers)
        logger.info(f"Dedup: {len(all_papers)} -> {len(deduped)} papers")
        return deduped

    def _dedup_by_title(self, papers: list[dict]) -> list[dict]:
        """Prefer the most-cited paper when normalized titles collide."""
        seen: dict[str, dict] = {}
        for p in papers:
            key = p.get("title", "").lower().strip()
            if not key:
                continue
            if key in seen:
                if p.get("citation_count", 0) > seen[key].get("citation_count", 0):
                    seen[key] = p
            else:
                seen[key] = p

        return list(seen.values())
