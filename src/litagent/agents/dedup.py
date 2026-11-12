"""Deduplicate paper-search results by normalized title."""

from __future__ import annotations

from typing import Any

from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.models import (
    ContentChunk,
    ContentScope,
    PaperCandidate,
    merge_paper_candidates,
)

logger = get_logger("agent.dedup")


class DedupWorker(Worker):
    """Merge upstream paper lists and remove duplicate titles."""

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "dedup"

    async def execute(self, task: SubTask) -> Any:
        """Merge upstream candidates and preserve richer content."""
        upstream = task.input_data.get("upstream_results", {})
        candidates = []

        for papers in upstream.values():
            if not isinstance(papers, list):
                continue
            for paper in papers:
                if not isinstance(paper, dict):
                    continue
                try:
                    candidates.append(PaperCandidate.model_validate(paper))
                except ValueError:
                    title = str(paper.get("title") or "").strip()
                    paper_id = str(paper.get("paper_id") or "").strip()
                    if not title or not paper_id:
                        continue
                    abstract = str(paper.get("abstract") or "").strip()
                    candidates.append(
                        PaperCandidate(
                            paper_id=paper_id,
                            title=title,
                            abstract=abstract,
                            citation_count=max(
                                int(paper.get("citation_count") or 0),
                                0,
                            ),
                            source=str(paper.get("source") or "legacy"),
                            content_scope=(
                                ContentScope.ABSTRACT
                                if abstract
                                else ContentScope.METADATA_ONLY
                            ),
                            chunks=(
                                [
                                    ContentChunk.from_text(
                                        paper_id=paper_id,
                                        chunk_key="abstract",
                                        text=abstract,
                                        section="abstract",
                                        content_scope=ContentScope.ABSTRACT,
                                    )
                                ]
                                if abstract
                                else []
                            ),
                        )
                    )
        merged = merge_paper_candidates(candidates)
        logger.info("Dedup: %d -> %d papers", len(candidates), len(merged))
        return [candidate.to_dag_dict() for candidate in merged]

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
