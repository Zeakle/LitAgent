"""Assign citation-based tiers and summarize ranked papers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask

logger = get_logger("agents.graph")


class GraphWorker(Worker):
    """Build a citation-tier summary from ranked papers."""

    TIER1_THRESHOLD = 500
    TIER2_THRESHOLD = 50

    def __init__(self, max_papers: int = 50) -> None:
        """Initialize the graph worker."""
        if max_papers <= 0:
            raise ValueError("max_papers must be positive")

        self._max_papers = max_papers

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "graph"

    async def execute(self, task: SubTask) -> Any:
        """Assign paper tiers and return aggregate tier metadata."""
        upstream = task.input_data.get("upstream_results", {})
        papers = self._get_papers_from_upstream(upstream)

        tiered = self._assign_tiers(papers)

        tier_counts = {"tier1": 0, "tier2": 0, "tier3": 0}
        for p in tiered:
            tier_counts[f"tier{p['tier']}"] += 1
        logger.info(f"Graph: {len(tiered)} papers, tiers: {tier_counts}")

        return {
            "papers": tiered,
            "tier_counts": tier_counts,
            "seminal_papers": [p for p in tiered if p["tier"] == 1],
        }

    def _get_papers_from_upstream(self, upstream: dict) -> list[dict]:
        """Return papers from upstream."""
        if not isinstance(upstream, Mapping):
            return []

        if "relevance_gate" in upstream:
            candidates = upstream["relevance_gate"]
        else:
            candidates = upstream.get("dedup", [])

        if not isinstance(candidates, list):
            return []

        return [paper for paper in candidates if isinstance(paper, dict)][
            : self._max_papers
        ]

    def _assign_tiers(self, papers: list[dict]) -> list[dict]:
        """Assign each paper a tier from its citation count."""
        result = []
        for p in papers:
            cc = p.get("citation_count", 0) or 0
            if cc >= self.TIER1_THRESHOLD:
                tier = 1
            elif cc >= self.TIER2_THRESHOLD:
                tier = 2
            else:
                tier = 3
            result.append({**p, "tier": tier})
        return result
