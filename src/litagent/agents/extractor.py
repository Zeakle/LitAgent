"""Extract structured evidence from ranked papers."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from litagent.agents.extraction_strategy import ExtractionStrategy
from litagent.evidence import build_evidence_items
from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.safety.injection import InjectionDetector, InjectionRisk

logger = get_logger("agents.extractor")


class ExtractorWorker(Worker):
    """Extract papers concurrently into run-scoped, untrusted evidence."""

    def __init__(
        self,
        strategy: ExtractionStrategy,
        max_concurrent: int = 5,
        detector: InjectionDetector = None,
        max_papers: int = 50,
    ):
        """Initialize the extractor worker."""
        if max_papers <= 0:
            raise ValueError("max_papers must be positive")

        self._strategy = strategy
        self._sem = asyncio.Semaphore(max_concurrent)
        self._detector = detector
        self._max_papers = max_papers

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "extractor"

    async def execute(self, task: SubTask) -> Any:
        """Extract, enrich, and index papers from upstream ranking results."""
        upstream = task.input_data.get("upstream_results", {})
        papers = self._get_papers_from_upstream(upstream)

        if self._detector:
            safe = []
            for p in papers:
                # Scan every candidate chunk, not only title/abstract. HIGH
                # candidates are skipped; suspicious chunks should already have
                # been excluded during ingestion.
                content_parts = [
                    str(p.get("title") or ""),
                    str(p.get("abstract") or ""),
                ]
                chunks = p.get("chunks")
                if isinstance(chunks, list):
                    content_parts.extend(
                        str(chunk.get("text") or "")
                        for chunk in chunks
                        if isinstance(chunk, Mapping)
                    )
                res = self._detector.scan(
                    " ".join(part for part in content_parts if part)
                )
                if res.risk == InjectionRisk.HIGH:
                    logger.warning(
                        f"Skip paper {p.get('paper_id', '?')}: injection in content"
                    )
                    continue
                safe.append(p)
            papers = safe

        results = await asyncio.gather(
            *[self._extract_one(p) for p in papers], return_exceptions=True
        )

        extractions = []
        for paper, r in zip(papers, results):
            if isinstance(r, asyncio.CancelledError):
                raise r

            if isinstance(r, BaseException):
                logger.warning(
                    f"Extraction fully failed for paper="
                    f"{paper.get('paper_id', '?')} "
                    f"error_type={type(r).__name__}",
                )
                continue

            r.update(
                {
                    "paper_id": paper.get("paper_id", ""),
                    "title": paper.get("title", ""),
                    "abstract": paper.get("abstract", ""),
                    "citation_count": paper.get("citation_count", 0),
                    "source": paper.get("source", ""),
                    "content_scope": paper.get("content_scope", ""),
                    "chunks": paper.get("chunks", []),
                    "provenance": paper.get("provenance", []),
                    "warnings": paper.get("warnings", []),
                    "retrieval": paper.get("retrieval"),
                }
            )

            r["evidence_items"] = build_evidence_items(r)
            extractions.append(r)

        # No global Claims write here: extraction output is run-scoped and
        # untrusted until Evaluation/Delivery finishes. Trusted promotion is
        # handled by ClaimsPromoter after the final delivery decision.
        return extractions

    def _get_papers_from_upstream(self, upstream: dict) -> list[dict]:
        """Use the explicit DAG contract; dedup is a migration/test fallback only."""
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

    async def _extract_one(self, paper: dict) -> dict:
        """Extract one paper under the configured concurrency limit."""
        async with self._sem:
            return await self._strategy.extract(paper)
