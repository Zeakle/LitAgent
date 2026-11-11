"""Extract structured evidence from ranked papers."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Callable, Mapping
from typing import Any

from litagent.agents.extraction_strategy import ExtractionStrategy
from litagent.body_quality import (
    EXTRACTION_CLAIM_CONTRACT,
    EXTRACTION_FIELD_SPAN_CONTRACT,
    check_preparation_policy,
    checked_preparation_chunks,
)
from litagent.evidence import (
    SOURCE_FIELDS_IDENTITY,
    build_evidence_items,
    collect_ledger,
)
from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.safety.injection import InjectionDetector, InjectionRisk
from litagent.workbench import extraction_admission_enabled, is_strict_policy

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
        extraction_verifier=None,
        run_id_provider=None,
        evidence_preparation_policy=None,
        preparation_budget_path_provider=None,
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
        self._extraction_verifier = extraction_verifier
        self._run_id_provider = run_id_provider
        self._preparation_budget_path_provider = preparation_budget_path_provider
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
        self._evidence_preparation_policy = check_preparation_policy(
            evidence_preparation_policy
        )
        loader_policy = getattr(fulltext_loader, "__dict__", {}).get(
            "preparation_policy"
        )
        if loader_policy != self._evidence_preparation_policy:
            raise ValueError("evidence_preparation_components_mismatch")
        if self._evidence_preparation_policy and (
            strict_paper_batch_size != 2 or not strict_grounded_claims
        ):
            raise ValueError("evidence_preparation_components_mismatch")
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
        focused = strict and self._evidence_preparation_policy is not None
        if focused and task.input_data.get("quality_policy") != "strict_auto_v2":
            raise ValueError("evidence_preparation_requires_strict_auto_v2")
        # Search content cannot activate a host preparation mode.
        papers = [
            {
                k: v
                for k, v in paper.items()
                if k
                not in {
                    "_evidence_preparation",
                    "_preparation_request_bound",
                    "_preparation_request_scope",
                    "_extraction_question",
                }
            }
            for paper in papers
        ]
        if strict:
            if self._fulltext_loader is None:
                raise ValueError("strict_fulltext_loader_unavailable")
            papers = await self._fulltext_loader.enrich(
                papers, task.input_data.get("coverage_requirements", [])
            )
            for paper in papers:
                paper["quality_policy"] = task.input_data["quality_policy"]
                if (
                    task.input_data["quality_policy"] == "strict_auto_v2"
                    and getattr(self._fulltext_loader, "prefer_method_sections", False)
                    is True
                ):
                    paper["_extraction_question"] = task.input_data.get("query")

        if focused:
            from litagent.agents.extraction_strategy import focused_extraction_context

            for paper in papers:
                reason = (paper.get("fulltext_acquisition") or {}).get("reason_code")
                if reason in {
                    "asset_hash_mismatch",
                    "source_identity_mismatch",
                    "not_pdf",
                    "evidence_preparation_source_identity_invalid",
                    "evidence_preparation_source_invalid",
                }:
                    raise ValueError("evidence_preparation_integrity_failed:" + reason)
                paper["chunks"] = checked_preparation_chunks(paper)
                preparation = {
                    "policy": self._evidence_preparation_policy,
                    "question": task.input_data.get("query"),
                    "requirements": copy.deepcopy(
                        task.input_data.get("coverage_requirements", [])
                    ),
                }
                _, _, receipt = focused_extraction_context(
                    paper, preparation["question"], preparation["requirements"]
                )
                paper["_evidence_preparation"] = preparation
                paper["preparation_context"] = receipt

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
            if focused:
                if not callable(self._preparation_budget_path_provider):
                    raise ValueError("evidence_preparation_budget_path_unavailable")
                budget_path = self._preparation_budget_path_provider()
                if budget_path is None:
                    raise ValueError("evidence_preparation_budget_path_unavailable")
                bounds = self._strategy.preflight_batches(
                    groups, budget_path=budget_path
                )
                if len(bounds) != len(groups):
                    raise ValueError("evidence_preparation_budget_identity_invalid")
                for group, bound in zip(groups, bounds):
                    for paper in group:
                        paper["_preparation_request_bound"] = bound
                        paper["preparation_context"]["request_reservation"] = bound
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
                if focused:
                    r["preparation_context"] = copy.deepcopy(
                        paper["preparation_context"]
                    )
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

        if task.input_data.get(
            "quality_policy"
        ) == "strict_auto_v2" and extraction_admission_enabled(task.input_data):
            if self._extraction_verifier is None or self._run_id_provider is None:
                raise ValueError("extraction_admission_unavailable")
            candidates = collect_ledger(extractions)
            admitted, receipt = await self._extraction_verifier.verify_candidates(
                candidates,
                run_id=self._run_id_provider(),
                **(
                    {
                        "resource_policy": task.input_data["verification"][
                            "extraction_resource_policy"
                        ]
                    }
                    if task.input_data["verification"].get("extraction_resource_policy")
                    else {}
                ),
                **(
                    {
                        "confidence_threshold": task.input_data["verification"][
                            "extraction_confidence_threshold"
                        ]
                    }
                    if task.input_data["verification"].get(
                        "extraction_confidence_threshold"
                    )
                    is not None
                    else {}
                ),
                **(
                    {
                        "failure_policy": task.input_data["verification"][
                            "extraction_failure_policy"
                        ]
                    }
                    if task.input_data["verification"].get("extraction_failure_policy")
                    else {}
                ),
            )
            if any(
                eid not in candidates or item != candidates[eid]
                for eid, item in admitted.items()
            ):
                raise ValueError("extraction_admission_ledger_changed")
            if not admitted:
                raise ValueError("extraction_no_supported_candidates")
            for extraction in extractions:
                original = extraction["evidence_items"]
                extraction["candidate_evidence_items"] = copy.deepcopy(original)
                extraction["candidate_claims"] = copy.deepcopy(
                    extraction.get("claims", [])
                )
                extraction["candidate_claim_records"] = copy.deepcopy(
                    extraction.get("claim_records", [])
                )
                extraction["evidence_items"] = [
                    item for item in original if item["evidence_id"] in admitted
                ]
                extraction["claims"] = list(
                    dict.fromkeys(item["text"] for item in extraction["evidence_items"])
                )
                if "claim_records" in extraction:
                    # Equal text does not identify equal sources or field associations.
                    extraction["claim_records"] = [
                        record
                        for record in extraction["claim_records"]
                        if any(
                            item["evidence_id"] in admitted
                            for item in build_evidence_items(
                                {**extraction, "claim_records": [record]}
                            )
                        )
                    ]
            # One run-owned receipt covers the complete candidate batch. Do not
            # duplicate paid source material into every paper record.
            extractions[0]["extraction_verification"] = receipt

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
