"""Extract structured evidence from ranked papers."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

from litagent.agents.extraction_strategy import ExtractionStrategy
from litagent.body_quality import (
    EXTRACTION_CLAIM_CONTRACT,
    EXTRACTION_FIELD_SPAN_CONTRACT,
)
from litagent.evidence import SOURCE_FIELDS_IDENTITY, build_evidence_items
from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.safety.injection import InjectionDetector, InjectionRisk
from litagent.workbench import is_strict_policy

logger = get_logger("agents.extractor")


class ExtractorWorker(Worker):
    """Extract papers concurrently into run-scoped, untrusted evidence."""

    def __init__(
        self,
        strategy: ExtractionStrategy,
        max_concurrent: int = 5,
        detector: InjectionDetector = None,
        max_papers: int = 50,
        fulltext_loader=None,
        failure_callback: Callable[[], None] | None = None,
        strict_paper_batch_size: int = 1,
        strict_field_spans: bool = False,
        strict_grounded_claims: bool = False,
    ):
        """Initialize the extractor worker."""
        if max_papers <= 0:
            raise ValueError("max_papers must be positive")
        if type(strict_paper_batch_size) is not int or strict_paper_batch_size not in {
            1,
            2,
        }:
            raise ValueError("unsupported strict_paper_batch_size")

        self._strategy = strategy
        self._sem = asyncio.Semaphore(max_concurrent)
        self._detector = detector
        self._max_papers = max_papers
        self._fulltext_loader = fulltext_loader
        self._failure_callback = failure_callback
        if type(strict_field_spans) is not bool:
            raise ValueError("strict_field_spans must be boolean")
        if type(strict_grounded_claims) is not bool or (
            strict_grounded_claims and strict_field_spans
        ):
            raise ValueError("invalid strict_grounded_claims mode")
        self._strict_grounded_claims = strict_grounded_claims
        self._strict_field_spans = strict_field_spans
        self._strict_paper_batch_size = strict_paper_batch_size
        self._batch_sem = asyncio.Semaphore(min(max_concurrent, 2))

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "extractor"

    async def execute(self, task: SubTask) -> Any:
        """Extract, enrich, and index papers from upstream ranking results."""
        upstream = task.input_data.get("upstream_results", {})
        papers = self._get_papers_from_upstream(upstream)
        strict = is_strict_policy(task.input_data.get("quality_policy"))
        if strict:
            if self._fulltext_loader is None:
                raise ValueError("strict_fulltext_loader_unavailable")
            papers = await self._fulltext_loader.enrich(
                papers, task.input_data.get("coverage_requirements", [])
            )
            for paper in papers:
                paper["quality_policy"] = task.input_data["quality_policy"]

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

        batched = (
            self._strict_paper_batch_size == 2
            and task.input_data.get("quality_policy") == "strict_auto_v2"
            and task.input_data.get("body_quality") == "analysis_v1"
        )
        if batched:
            if self._strict_grounded_claims:
                papers = [
                    {**p, "_claim_contract": EXTRACTION_CLAIM_CONTRACT} for p in papers
                ]
            if self._strict_field_spans:
                papers = [
                    {**p, "_field_span_contract": EXTRACTION_FIELD_SPAN_CONTRACT}
                    for p in papers
                ]
            ids = [paper.get("paper_id") for paper in papers]
            if any(not isinstance(pid, str) or not pid for pid in ids) or len(
                set(ids)
            ) != len(ids):
                raise ValueError("extraction_batch_paper_identity_invalid")
            groups = [papers[i : i + 2] for i in range(0, len(papers), 2)]
            grouped = await asyncio.gather(
                *[self._extract_batch(group) for group in groups],
                return_exceptions=True,
            )
            results = []
            for group, response in zip(groups, grouped):
                if isinstance(response, BaseException):
                    results.extend([response] * len(group))
                elif not isinstance(response, list) or len(response) != len(group):
                    raise ValueError("extraction_batch_member_coverage_invalid")
                else:
                    results.extend(response)
        else:
            results = await asyncio.gather(
                *[self._extract_one(p) for p in papers], return_exceptions=True
            )

        extractions = []
        for paper, r in zip(papers, results):
            if isinstance(r, asyncio.CancelledError):
                raise r

            if isinstance(r, BaseException):
                if task.input_data.get("quality_policy") in {
                    "strict_auto_v1",
                    "strict_auto_v2",
                }:
                    raise r
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
            if strict:
                r["fulltext_acquisition"] = paper.get("fulltext_acquisition")
                r["quality_policy"] = task.input_data["quality_policy"]
                if r.get("record_identity") != SOURCE_FIELDS_IDENTITY:
                    fields = {
                        (c.get("source_chunk_key"), c.get("text")): c.get(
                            "fact_fields", {}
                        )
                        for c in r.get("claim_records", [])
                    }
                    for item in r["evidence_items"]:
                        item["fact_fields"] = fields.get(
                            (item.get("chunk_key"), item.get("text")), {}
                        )
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
            try:
                return await self._strategy.extract(paper)
            except Exception:
                # Stop queued transports immediately, before gather settles other
                # in-flight responses. Known responses still retain their usage.
                if (
                    paper.get("quality_policy") in {"strict_auto_v1", "strict_auto_v2"}
                    and self._failure_callback is not None
                ):
                    self._failure_callback()
                raise

    async def _extract_batch(self, papers: list[dict]) -> list[dict]:
        """Stop queued transports before releasing a failed batch's semaphore."""
        async with self._batch_sem:
            try:
                result = await self._strategy.extract_batch(papers)
                if (
                    not isinstance(result, list)
                    or len(result) != len(papers)
                    or any(not isinstance(item, dict) for item in result)
                ):
                    raise ValueError("extraction_batch_member_coverage_invalid")
                return result
            except Exception:
                if self._failure_callback is not None:
                    self._failure_callback()
                raise
