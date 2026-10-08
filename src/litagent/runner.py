"""Assemble configured components into the executable survey pipeline."""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal

from litagent.agents import adversarial
from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.dedup import DedupWorker
from litagent.agents.extraction_strategy import (
    ExtractionStrategy,
    LLMStrategy,
    RegexStrategy,
    ResilientExtractionStrategy,
)
from litagent.agents.extractor import ExtractorWorker
from litagent.agents.graph import GraphWorker
from litagent.agents.planner import SurveyPlanner
from litagent.agents.recall import RecallWorker
from litagent.agents.relevance_gate import RelevanceGateWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.search import SearchWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import AppConfig
from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import EvidenceSelection, EvidenceSelector
from litagent.contracts import build_config_summary, normalize_survey_result
from litagent.eval.base import (
    CTX_CLAIMS,
    CTX_EVIDENCE,
    CTX_PAPERS,
    CTX_QUERY,
    CTX_REFERENCED_EVIDENCE,
    CTX_REFERENCED_EVIDENCE_IDS,
    CTX_SELECTED_EVIDENCE,
    CTX_UNRESOLVED_EVIDENCE_IDS,
)
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.eval.scoped import check_request_budget
from litagent.evidence import (
    build_citation_aliases,
    collect_ledger,
    extract_evidence_refs,
    extract_evidence_refs_ordered,
    find_malformed_evidence_refs,
)
from litagent.exceptions import ConfigError
from litagent.interaction import (
    ExecutionCheckpoint,
    RunAdvanceResult,
    apply_paper_decision,
    interaction_source_fingerprint,
    paper_id,
    report_version_from_checkpoint,
)
from litagent.llm.client import BaseLLMClient, OpenAICompatibleClient
from litagent.logging import get_logger, setup_logging
from litagent.mcp.bridge import MCPBridge
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.manager import MemoryManager
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.working import WorkingMemory
from litagent.observability.citations import (
    decode_citations_traced,
    normalize_citations_traced,
)
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.scheduler import CancellationToken, Scheduler, Worker
from litagent.orchestrator.task_graph import TaskGraph
from litagent.rag.claim_promotion import (
    ClaimPromotionSummary,
    ClaimsPromoter,
    is_publishable_report,
)
from litagent.rag.claims_index import ClaimsIndex
from litagent.rag.corpus import CollectionIdentity
from litagent.rag.embedder import build_retrieval_embedder
from litagent.rag.interfaces import Reranker
from litagent.rag.reranker import CrossEncoderReranker
from litagent.rag.retriever import HybridRetriever
from litagent.rag.vector_store import QdrantVectorStore
from litagent.review_structure import (
    assemble_methodology,
    build_methodology,
    check_methodology,
    validate_authored_methodology,
)
from litagent.safety.budget import CostBudget
from litagent.safety.injection import InjectionDetector
from litagent.skills.manager import SkillManager
from litagent.tools.builtin.extract import register_extract_tools
from litagent.tools.builtin.search import register_search_tools
from litagent.tools.executor import ToolExecutor
from litagent.tools.registry import ToolRegistry
from litagent.workbench import (
    HumanDecision,
    ReportVersion,
    ResumeIdentity,
    ReviewRequest,
    RunOptions,
    can_publish,
    validate_decision,
)

logger = get_logger("runner")


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*$")
_PLACEHOLDER_LINE_RE = re.compile(
    r"(?im)^\s*(?:\.\.\.|…|\[no changes to this section\.\]|"
    r"\[omitted\]|todo|tbd)\s*$"
)


@dataclass(frozen=True)
class RewriteValidation:
    """Describe structural validation of a rewritten survey candidate."""

    accepted: bool
    reason_codes: tuple[str, ...]
    referenced_evidence_ids: tuple[str, ...]
    unknown_evidence_ids: tuple[str, ...]
    unexpected_removed_evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RewriteOutcome:
    """Capture the transactional outcome of one rewrite attempt."""

    attempted: bool
    committed: bool
    candidate: str | None
    validation: RewriteValidation | None
    candidate_evaluation: Mapping[str, Any] | None
    candidate_quality: Mapping[str, Any] | None
    reason_code: str


@dataclass(frozen=True)
class RunPolicy:
    """Control cross-run state reads and writes for one LitAgent instance."""

    memory_reads: bool = True
    memory_writes: bool = True
    claims_reads: bool = True
    claims_writes: bool = True

    @classmethod
    def isolated_benchmark(cls) -> "RunPolicy":
        """Return a policy that prevents benchmark runs from sharing state."""
        return cls(
            memory_reads=False,
            memory_writes=False,
            claims_reads=False,
            claims_writes=False,
        )


@dataclass(frozen=True)
class ExternalInputReplay:
    """Supply immutable planner and external-search outputs to one run."""

    query: str
    decomposition: tuple[str, ...]
    search_outputs: Mapping[str, Mapping[str, Any]]
    fingerprint: str


def derive_delivery(
    partial: bool,
    quality: Mapping[str, Any] | None,
    degradation_reason_codes: Sequence[str] = (),
) -> dict[str, Any]:
    """Map execution completeness and quality to the delivery policy."""
    q_status = (quality or {}).get("status", "unverified")
    valid_quality_status = q_status in {"passed", "failed", "unverified"}
    degradation_codes = list(dict.fromkeys(degradation_reason_codes))

    critical_degradation = any(
        code in {"all_external_sources_unavailable", "all_external_sources_empty"}
        for code in degradation_codes
    )

    reason_codes: list[str] = []
    if partial:
        reason_codes.append("partial_execution")
    if not valid_quality_status:
        reason_codes.append("quality_invalid")
    elif q_status == "failed":
        reason_codes.append("quality_failed")
    elif q_status == "unverified":
        reason_codes.append("quality_unverified")
    reason_codes.extend(degradation_codes)

    if partial:
        status = "partial"
    elif q_status == "failed":
        status = "blocked"
    elif not valid_quality_status or q_status == "unverified" or critical_degradation:
        status = "needs_review"
    else:
        status = "ready"

    return {
        "status": status,
        "publishable": status == "ready",
        "reason_codes": list(dict.fromkeys(reason_codes)),
    }


def _cleanup_incomplete_wiring(func):
    """Release resources when private or public wiring exits exceptionally."""

    @wraps(func)
    async def wrapped(self, *args, **kwargs):
        """Close partially initialized resources when wiring fails."""
        try:
            return await func(self, *args, **kwargs)
        except BaseException:
            await self.cleanup()
            raise

    return wrapped


@dataclass
class Infra:
    """Hold optional infrastructure services and shared connection handles."""

    memory: MemoryManager | None = None
    claims_index: ClaimsIndex | None = None
    retriever: HybridRetriever | None = None
    reranker: Reranker | None = None
    relevance_reranker: Reranker | None = None
    evidence_reranker: Reranker | None = None

    _working_memory: WorkingMemory | None = None
    _qdrant_client: Any = None
    _pg_pool: Any = None
    _closed: bool = False

    async def close(self) -> None:
        """Close owned handles once; borrowers never close shared handles."""
        if self._closed:
            return
        self._closed = True

        resources = (
            ("PostgreSQL pool", self._pg_pool),
            ("Qdrant client", self._qdrant_client),
            ("Working Memory", self._working_memory),
        )

        self._pg_pool = None
        self._qdrant_client = None
        self._working_memory = None

        for label, resource in resources:
            if resource is None:
                continue
            try:
                await resource.close()
            except BaseException as exc:
                logger.debug("%s close error: %s", label, exc)


TraceHook = Callable[[str, dict[str, Any]], Any]
"""Trace callback invoked with a stable event name and payload."""


class _RunCancelled(Exception):
    """Stop the survey at an explicit cooperative cancellation boundary."""


async def _await_with_cancellation(awaitable, cancellation: CancellationToken | None):
    """Await one phase while allowing the run token to preempt pending work."""
    if cancellation is None:
        return await awaitable
    if cancellation.is_cancelled:
        if asyncio.iscoroutine(awaitable):
            awaitable.close()
        raise _RunCancelled

    work = asyncio.create_task(awaitable)
    cancel_waiter = asyncio.create_task(cancellation.wait())
    try:
        done, _ = await asyncio.wait(
            {work, cancel_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        # Preserve a business result that completed in the same event-loop tick.
        if work in done:
            return await work
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        raise _RunCancelled
    finally:
        cancel_waiter.cancel()
        await asyncio.gather(cancel_waiter, return_exceptions=True)


class LitAgent:
    """Assemble components, execute surveys, and release shared resources."""

    def __init__(
        self,
        config: AppConfig,
        trace_hook: TraceHook | None = None,
        *,
        run_policy: RunPolicy | None = None,
        external_inputs: ExternalInputReplay | None = None,
    ) -> None:
        """Initialize the LitAgent runtime and optional integrations."""
        self._config = config
        self._config_summary = build_config_summary(config)
        self._session_id = str(uuid.uuid4())[:8]
        self._trace_hook = trace_hook
        self._run_policy = run_policy or RunPolicy()
        self._external_inputs = external_inputs

        self._cost_budget: CostBudget | None = None
        self._llm: BaseLLMClient | None = None
        self._registry = None
        self._executor: ToolExecutor | None = None
        self._budget_manager: BudgetManager | None = None

        self._infra: Infra = Infra()

        self._skill_manager: SkillManager | None = None
        self._extraction_strategy: ExtractionStrategy | None = None
        self._mcp_bridge = None

        self._search: SearchWorker | None = None
        self._dedup: DedupWorker | None = None
        self._relevance_gate: RelevanceGateWorker | None = None
        self._extractor: ExtractorWorker | None = None
        self._graph: GraphWorker | None = None
        self._evidence_selector: EvidenceSelector | None = None
        self._synthesis: SynthesisWorker | None = None
        self._reviewer: ReviewerWorker | None = None
        self._adversarial: AdversarialReviewWorker | None = None

        self._scheduler: Scheduler | None = None
        self._planner: SurveyPlanner | None = None

        self._evaluators = []

        self._wired = False

    async def __aenter__(self) -> "LitAgent":
        """Enter the LitAgent asynchronous context."""
        await self._wire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Exit the LitAgent context and close owned resources."""
        await self.cleanup()

    @staticmethod
    def _apply_run_options_to_graph(graph: TaskGraph, options: RunOptions) -> None:
        """Attach immutable generation choices to every scheduler task."""
        started_at = time.time()
        for task in graph.tasks.values():
            task.input_data["report_language"] = options.report_language
            if options.review_structure != "legacy":
                task.input_data.setdefault("search_started_at", started_at)
                task.input_data["review_structure"] = options.review_structure

    def _resume_identity(self, run_id: str, options: RunOptions) -> ResumeIdentity:
        """Bind interactive checkpoints to behavior and configuration."""
        return ResumeIdentity(
            run_id=run_id,
            source_fingerprint=interaction_source_fingerprint(),
            config_fingerprint=self._config_summary["fingerprint"],
            options=options,
        )

    async def advance(
        self,
        query: str,
        *,
        options: RunOptions,
        checkpoint: ExecutionCheckpoint | None = None,
        decision: HumanDecision | None = None,
        cancellation: CancellationToken | None = None,
        run_id: str | None = None,
    ) -> RunAdvanceResult:
        """Advance one interactive run between its two safe human boundaries."""
        if not options.human_review:
            raise ValueError("interactive_run_requires_human_review")
        if not self._wired:
            await self._wire()

        if checkpoint is None:
            if decision is not None:
                raise ValueError("decision_without_checkpoint")
            active_run_id = run_id or self._session_id
            self._session_id = active_run_id
            graph = await _await_with_cancellation(
                self._planner.plan(query), cancellation
            )
            self._apply_run_options_to_graph(graph, options)
            if hasattr(self._trace_hook, "capture_graph"):
                self._trace_hook.capture_graph(graph)
            outcome = await self._scheduler.run_until(
                graph, stop_after="relevance_gate", cancellation=cancellation
            )
            if outcome.status != "paused":
                return RunAdvanceResult(status="interrupted")
            if hasattr(self._trace_hook, "capture_graph_state"):
                self._trace_hook.capture_graph_state(graph)
            dedup = graph.get_task("dedup")
            relevance = graph.get_task("relevance_gate")
            candidates = (
                dedup.result if dedup and isinstance(dedup.result, list) else []
            )
            selected = (
                relevance.result
                if relevance and isinstance(relevance.result, list)
                else []
            )
            candidate_ids = tuple(
                item_id
                for item in candidates
                if isinstance(item, Mapping) and (item_id := paper_id(item))
            )
            if not candidate_ids:
                return RunAdvanceResult(status="interrupted")
            request_id = uuid.uuid4().hex
            request = ReviewRequest(
                request_id=request_id,
                run_id=active_run_id,
                stage="paper_selection",
                revision=1,
                candidate_paper_ids=candidate_ids,
                automatic_selected_paper_ids=tuple(
                    item_id
                    for item in selected
                    if isinstance(item, Mapping) and (item_id := paper_id(item))
                ),
            )
            search_outcomes, degradation_codes = self._summarize_search_outcomes(
                self._search.get_source_outcomes() if self._search else ()
            )
            created = ExecutionCheckpoint(
                identity=self._resume_identity(active_run_id, options),
                query=query,
                options=options,
                boundary="paper_selection",
                graph=graph.to_snapshot(),
                results=graph.get_results(),
                budget=self._cost_budget.snapshot(),
                search_outcomes=tuple(search_outcomes),
                degradation_codes=tuple(degradation_codes),
                review_request_id=request_id,
            )
            self._emit(
                "human.requested",
                {
                    "run_id": active_run_id,
                    "request_id": request_id,
                    "stage": "paper_selection",
                    "revision": 1,
                },
            )
            return RunAdvanceResult(
                status="awaiting_human", checkpoint=created, review=request
            )

        self._session_id = checkpoint.identity.run_id
        checkpoint.identity.require_compatible(
            self._resume_identity(self._session_id, options)
        )
        if checkpoint.query != query or checkpoint.options != options:
            raise ValueError("checkpoint_input_mismatch")
        self._cost_budget.restore(checkpoint.budget)
        if decision is None:
            raise ValueError("checkpoint_requires_decision")

        if checkpoint.boundary == "paper_selection":
            if decision.action == "cancel":
                return RunAdvanceResult(status="cancelled")
            checkpoint = apply_paper_decision(checkpoint, decision)
            graph = TaskGraph.from_snapshot(checkpoint.graph)
            outcome = await self._scheduler.run_until(graph, cancellation=cancellation)
            if outcome.status != "completed":
                return RunAdvanceResult(status="interrupted")
            report_data, degradation_codes = await self._prepare_report(
                query=query,
                graph=graph,
                results=graph.get_results(),
                options=options,
                search_outcomes=list(checkpoint.search_outcomes),
                degradation_codes=list(checkpoint.degradation_codes),
                cancellation=cancellation,
            )
            automatic_delivery = dict(report_data["delivery"])
            report_data.setdefault("metadata", {})[
                "automatic_delivery"
            ] = automatic_delivery
            report_data["metadata"]["human_review"] = {"status": "pending"}
            report_data["metadata"]["report_revision"] = 1
            report_data["delivery"] = {
                "status": (
                    "needs_review"
                    if automatic_delivery.get("status") == "ready"
                    else automatic_delivery.get("status")
                ),
                "publishable": False,
                "reason_codes": list(
                    dict.fromkeys(
                        [
                            *automatic_delivery.get("reason_codes", []),
                            "human_review_pending",
                        ]
                    )
                ),
            }
            revision = 1
            version = ReportVersion(
                run_id=self._session_id,
                revision=revision,
                report_language=options.report_language,
                survey=report_data["survey"],
                human_status="pending",
            )
            request_id = uuid.uuid4().hex
            request = ReviewRequest(
                request_id=request_id,
                run_id=self._session_id,
                stage="final_report",
                revision=2,
                report_revision=revision,
                report_sha256=version.content_sha256,
            )
            created = ExecutionCheckpoint(
                identity=checkpoint.identity,
                query=query,
                options=options,
                boundary="final_report",
                graph=graph.to_snapshot(),
                results=graph.get_results(),
                budget=self._cost_budget.snapshot(),
                search_outcomes=checkpoint.search_outcomes,
                degradation_codes=tuple(degradation_codes),
                report_data=report_data,
                report_revision=revision,
                review_request_id=request_id,
            )
            return RunAdvanceResult(
                status="awaiting_human", checkpoint=created, review=request
            )

        request = ReviewRequest(
            request_id=checkpoint.review_request_id,
            run_id=self._session_id,
            stage="final_report",
            revision=decision.revision,
            revision_attempts=checkpoint.revision_attempts,
            report_revision=checkpoint.report_revision,
            report_sha256=report_version_from_checkpoint(checkpoint).content_sha256,
        )
        validate_decision(request, decision)
        if decision.action == "cancel":
            return RunAdvanceResult(status="cancelled")
        report_data = dict(checkpoint.report_data or {})
        metadata = report_data.setdefault("metadata", {})
        automatic_delivery = dict(metadata.get("automatic_delivery") or {})
        if decision.action == "reject":
            metadata["human_review"] = {"status": "rejected"}
            report_data["delivery"] = {
                "status": "blocked",
                "publishable": False,
                "reason_codes": list(
                    dict.fromkeys(
                        [*automatic_delivery.get("reason_codes", []), "human_rejected"]
                    )
                ),
            }
            finalized = await self._finalize_report(
                query=query,
                report_data=report_data,
                results=checkpoint.results,
                degradation_codes=checkpoint.degradation_codes,
                cancellation=cancellation,
                allow_trusted_writes=False,
            )
            return RunAdvanceResult(status="rejected", report=finalized)
        if decision.action == "revise":
            revised, revision_result = await self._attempt_human_revision(
                query=query,
                report_data=report_data,
                results=checkpoint.results,
                feedback=decision.feedback,
                report_language=options.report_language,
            )
            revised_metadata = revised.setdefault("metadata", {})
            revised_metadata["human_revision"] = revision_result
            revised_metadata["human_review"] = {"status": "pending"}
            automatic_delivery = dict(revised.get("delivery") or automatic_delivery)
            revised_metadata["automatic_delivery"] = automatic_delivery
            revised["delivery"] = {
                "status": (
                    "needs_review"
                    if automatic_delivery.get("status") == "ready"
                    else automatic_delivery.get("status", "blocked")
                ),
                "publishable": False,
                "reason_codes": list(
                    dict.fromkeys(
                        [
                            *automatic_delivery.get("reason_codes", []),
                            "human_review_pending",
                        ]
                    )
                ),
            }
            next_revision = checkpoint.report_revision + 1
            revised_metadata["report_revision"] = next_revision
            next_version = ReportVersion(
                run_id=self._session_id,
                revision=next_revision,
                report_language=options.report_language,
                survey=revised["survey"],
                human_status="pending",
            )
            next_request_id = uuid.uuid4().hex
            next_request = ReviewRequest(
                request_id=next_request_id,
                run_id=self._session_id,
                stage="final_report",
                revision=decision.revision + 1,
                revision_attempts=1,
                report_revision=next_revision,
                report_sha256=next_version.content_sha256,
            )
            next_checkpoint = checkpoint.model_copy(
                update={
                    "budget": self._cost_budget.snapshot(),
                    "report_data": revised,
                    "report_revision": next_revision,
                    "revision_attempts": 1,
                    "review_request_id": next_request_id,
                }
            )
            return RunAdvanceResult(
                status="awaiting_human",
                checkpoint=next_checkpoint,
                review=next_request,
            )

        metadata["human_review"] = {"status": "approved"}
        publishable = can_publish(
            bool(automatic_delivery.get("publishable")), options, "approved"
        )
        report_data["delivery"] = {
            **automatic_delivery,
            "publishable": publishable,
        }
        finalized = await self._finalize_report(
            query=query,
            report_data=report_data,
            results=checkpoint.results,
            degradation_codes=checkpoint.degradation_codes,
            cancellation=cancellation,
            allow_trusted_writes=publishable,
        )
        return RunAdvanceResult(status="completed", report=finalized)

    async def run(
        self,
        query: str,
        cancellation: CancellationToken | None = None,
        *,
        options: RunOptions | None = None,
    ) -> dict[str, Any]:
        """Run planning, scheduling, evaluation, and delivery for one query."""
        if not isinstance(query, str) or not query.strip() or len(query) > 2000:
            raise ValueError("invalid_research_question")
        options = options or RunOptions()
        if options.human_review:
            raise ValueError("human_review_requires_interactive_service")
        if self._external_inputs is not None and query != self._external_inputs.query:
            raise ValueError("external input replay query does not match run query")
        if cancellation and cancellation.is_cancelled:
            return self._build_cancelled_report(query)
        if not self._wired:
            await self._wire()
        if cancellation and cancellation.is_cancelled:
            return self._build_cancelled_report(query)

        self._emit(
            "survey.start",
            {
                "query": query,
                "session_id": self._session_id,
                "report_language": options.report_language,
            },
        )
        if self._external_inputs is not None:
            # Replay inputs are immutable before any profile-specific worker runs.
            self._emit(
                "external_inputs_ready",
                {"fingerprint": self._external_inputs.fingerprint},
            )
        logger.info("Starting survey: %s", query)

        graph: TaskGraph | None = None
        results: dict[str, Any] = {}
        report_data: dict[str, Any] | None = None
        degradation_codes: list[str] = []
        search_outcomes: list[dict[str, Any]] = []

        try:
            if self._search is not None:
                self._search.reset_source_outcomes()

            graph = await _await_with_cancellation(
                self._planner.plan(query), cancellation
            )
            self._apply_run_options_to_graph(graph, options)
            if hasattr(self._trace_hook, "capture_graph"):
                self._trace_hook.capture_graph(graph)
            results = await self._scheduler.run(graph, cancellation=cancellation)
            if hasattr(self._trace_hook, "capture_graph_state"):
                self._trace_hook.capture_graph_state(graph)

            # Collect search outcomes after Scheduler completes.
            search_outcomes, degradation_codes = self._summarize_search_outcomes(
                self._search.get_source_outcomes() if self._search else ()
            )

            report_data, degradation_codes = await self._prepare_report(
                query=query,
                graph=graph,
                results=results,
                options=options,
                search_outcomes=search_outcomes,
                degradation_codes=degradation_codes,
                cancellation=cancellation,
            )
            return await self._finalize_report(
                query=query,
                report_data=report_data,
                results=results,
                degradation_codes=degradation_codes,
                cancellation=cancellation,
            )

        except _RunCancelled:
            if graph is not None:
                graph.finalize_incomplete("user_cancelled")
                if hasattr(self._trace_hook, "capture_graph_state"):
                    self._trace_hook.capture_graph_state(graph)
            cancelled_report = self._build_cancelled_report(
                query,
                graph=graph,
                results=results,
                report_data=report_data,
                search_outcomes=search_outcomes,
                degradation_codes=degradation_codes,
            )
            self._emit(
                "survey.complete",
                {
                    "query": query,
                    "cancelled": True,
                    "quality_status": cancelled_report["quality"]["status"],
                    "delivery_status": cancelled_report["delivery"]["status"],
                    "total_tokens": self._cost_budget.used if self._cost_budget else 0,
                    "config_fingerprint": cancelled_report["config_fingerprint"],
                },
            )
            return cancelled_report

        except Exception as e:
            self._emit("survey.error", {"query": query, "error": str(e)})
            raise

    async def _prepare_report(
        self,
        *,
        query: str,
        graph: TaskGraph,
        results: Mapping[str, Any],
        options: RunOptions,
        search_outcomes: list[dict[str, Any]],
        degradation_codes: list[str],
        cancellation: CancellationToken | None,
    ) -> tuple[dict[str, Any], list[str]]:
        """Assemble, evaluate, and optionally rewrite before trust side effects."""
        if cancellation and cancellation.is_cancelled:
            raise _RunCancelled
        report_data = self._extract_report(dict(results), query)
        adversarial_result = results.get("adversarial_review")
        if isinstance(adversarial_result, Mapping):
            degradation_codes.extend(
                str(item)
                for item in adversarial_result.get("degradation_reasons", [])
                if isinstance(item, str) and item
            )
        degradation_codes = list(dict.fromkeys(degradation_codes))
        execution = self._derive_execution(
            graph,
            budget_exceeded=(
                self._cost_budget.is_exceeded() if self._cost_budget else False
            ),
            final_output_present=bool(
                isinstance(results.get("adversarial_review"), dict)
                and results["adversarial_review"].get("final_draft")
            ),
        )
        report_data["partial"] = execution["partial"]
        metadata = report_data.setdefault("metadata", {})
        metadata["execution"] = execution
        metadata["search_source_outcomes"] = search_outcomes
        metadata["degradation_reason_codes"] = degradation_codes
        metadata["report_language"] = options.report_language
        metadata["run_options_fingerprint"] = options.fingerprint()
        metadata["review_structure"] = options.review_structure
        metadata["review_methodology"] = (
            adversarial_result.get("review_methodology", {})
            if isinstance(adversarial_result, Mapping)
            else {}
        )

        initial_evaluation = await _await_with_cancellation(
            self._evaluate(
                query=query,
                survey=report_data["survey"],
                results=results,
                phase="initial",
            ),
            cancellation,
        )
        initial_quality = self._derive_quality(initial_evaluation)
        report_data.update(
            {"evaluation": initial_evaluation, "quality": initial_quality}
        )
        if initial_quality["status"] == "failed":
            rewrite = await _await_with_cancellation(
                self._attempt_evidence_rewrite(
                    query=query,
                    report_data=report_data,
                    results=results,
                    initial_evaluation=initial_evaluation,
                    initial_quality=initial_quality,
                    report_language=options.report_language,
                ),
                cancellation,
            )
            metadata["evidence_rewrite"] = {
                "attempted": rewrite.attempted,
                "committed": rewrite.committed,
                "reason_code": rewrite.reason_code,
                "validation_reason_codes": (
                    list(rewrite.validation.reason_codes) if rewrite.validation else []
                ),
                "candidate_quality": rewrite.candidate_quality,
            }
        report_data["delivery"] = derive_delivery(
            report_data["partial"], report_data["quality"], degradation_codes
        )
        return report_data, degradation_codes

    async def _finalize_report(
        self,
        *,
        query: str,
        report_data: dict[str, Any],
        results: Mapping[str, Any],
        degradation_codes: Sequence[str],
        cancellation: CancellationToken | None,
        allow_trusted_writes: bool = True,
    ) -> dict[str, Any]:
        """Apply trusted side effects and emit one normalized terminal report."""
        if (
            allow_trusted_writes
            and self._run_policy.claims_writes
            and self._infra.claims_index is not None
        ):
            promotion = await _await_with_cancellation(
                ClaimsPromoter(self._infra.claims_index).promote(
                    run_id=self._session_id,
                    domain=query,
                    report_data=report_data,
                    extractions=self._collect_extractions(results),
                ),
                cancellation,
            )
        elif not allow_trusted_writes:
            promotion = ClaimPromotionSummary(
                "skipped", 0, 0, 0, "human_approval_missing"
            )
        elif not self._run_policy.claims_writes:
            promotion = ClaimPromotionSummary(
                "skipped", 0, 0, 0, "claims_write_disabled"
            )
        else:
            promotion = ClaimPromotionSummary(
                "skipped", 0, 0, 0, "claims_index_unavailable"
            )
        metadata = report_data.setdefault("metadata", {})
        metadata["claims_promotion"] = promotion.to_dict()
        if allow_trusted_writes:
            metadata["memory"] = await _await_with_cancellation(
                self._finalize_memory(report_data=report_data), cancellation
            )
        else:
            metadata["memory"] = {
                "status": "skipped",
                "reason_code": "human_approval_missing",
            }

        report_data = normalize_survey_result(
            report_data, config_fingerprint=self._config_summary["fingerprint"]
        )
        final_quality = report_data["quality"]
        self._emit(
            "survey.complete",
            {
                "query": query,
                "rounds": metadata.get("total_rounds", 0),
                "accepted": metadata.get("accepted", False),
                "quality_status": final_quality["status"],
                "total_tokens": self._cost_budget.used if self._cost_budget else 0,
                "delivery_status": report_data["delivery"]["status"],
                "config_fingerprint": report_data["config_fingerprint"],
            },
        )
        logger.info("Survey complete: %d chars", len(report_data.get("survey", "")))
        return report_data

    @staticmethod
    def _derive_execution(
        graph: TaskGraph | None,
        budget_exceeded: bool,
        final_output_present: bool,
        run_cancelled: bool = False,
    ) -> dict[str, Any]:
        """Derive the final execution status from graph outcomes."""
        summary = (
            graph.execution_summary()
            if graph is not None
            else {
                "status": "incomplete",
                "counts": {},
                "failed_task_ids": [],
                "cancelled_task_ids": [],
                "skipped_task_ids": [],
            }
        )
        reason_codes: list[str] = []
        if run_cancelled:
            reason_codes.append("run_cancelled")
        if budget_exceeded:
            reason_codes.append("cost_budget_exceeded")
        if summary["failed_task_ids"]:
            reason_codes.append("task_failed")
        if summary["cancelled_task_ids"]:
            reason_codes.append("task_cancelled")
        if summary["skipped_task_ids"]:
            reason_codes.append("task_skipped")
        if not final_output_present:
            reason_codes.append("final_output_missing")

        partial = bool(reason_codes)
        return {
            **summary,
            "status": "incomplete" if partial else "complete",
            "partial": partial,
            "reason_codes": reason_codes,
        }

    def _build_cancelled_report(
        self,
        query: str,
        *,
        graph: TaskGraph | None = None,
        results: Mapping[str, Any] | None = None,
        report_data: Mapping[str, Any] | None = None,
        search_outcomes: Sequence[Mapping[str, Any]] = (),
        degradation_codes: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Build the canonical partial result for cooperative cancellation."""
        result_map = dict(results or {})
        payload = (
            dict(report_data)
            if report_data is not None
            else self._extract_report(result_map, query)
        )
        final_output_present = bool(
            isinstance(result_map.get("adversarial_review"), dict)
            and result_map["adversarial_review"].get("final_draft")
        )
        execution = self._derive_execution(
            graph,
            budget_exceeded=(
                self._cost_budget.is_exceeded() if self._cost_budget else False
            ),
            final_output_present=final_output_present,
            run_cancelled=True,
        )
        payload["partial"] = True
        payload.setdefault("evaluation", {})
        payload.setdefault(
            "quality",
            {
                "status": "unverified",
                "failed_metrics": [],
                "unverified_metrics": [],
            },
        )
        metadata = payload.setdefault("metadata", {})
        metadata["execution"] = execution
        metadata["search_source_outcomes"] = list(search_outcomes)
        metadata["claims_promotion"] = {
            "status": "skipped",
            "attempted_count": 0,
            "promoted_count": 0,
            "rejected_count": 0,
            "reason_code": "run_cancelled",
        }
        metadata["memory"] = {
            "attempted": False,
            "content_saved": False,
            "consolidated": False,
            "reason_code": "run_cancelled",
            "error_type": None,
        }
        payload["delivery"] = derive_delivery(
            True, payload["quality"], degradation_codes
        )
        return normalize_survey_result(
            payload, config_fingerprint=self._config_summary["fingerprint"]
        )

    @staticmethod
    def _summarize_search_outcomes(outcomes) -> tuple[list[dict], list[str]]:
        """Serialize search outcomes and collect degradation codes."""
        serialized: list[dict] = []
        degradation_codes: list[str] = []
        unavailable = {"rate_limited", "timeout", "failed"}

        for outcome in outcomes:
            item = {
                "task_id": outcome.task_id,
                "source": outcome.source,
                "status": outcome.status.value,
                "result_count": outcome.result_count,
                "elapsed_ms": outcome.elapsed_ms,
                "reason_code": outcome.reason_code,
                "error_type": outcome.error_type,
                "from_fallback": outcome.from_fallback,
            }
            serialized.append(item)
            if outcome.reason_code:
                degradation_codes.append(outcome.reason_code)

        if serialized and all(item["status"] in unavailable for item in serialized):
            degradation_codes.append("all_external_sources_unavailable")
        elif serialized and all(item["result_count"] == 0 for item in serialized):
            degradation_codes.append("all_external_sources_empty")

        return serialized, list(dict.fromkeys(degradation_codes))

    def _extract_report(self, results: dict[str, Any], query: str) -> dict[str, Any]:
        """Extract the sole final-report payload from scheduler results."""
        graph_result = results.get("graph_analysis")
        graph_data = graph_result if isinstance(graph_result, dict) else {}

        base_metadata: dict[str, Any] = {
            "query": query,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_rounds": 0,
            "final_score": 0.0,
            "accepted": False,
            "config_summary": self._config_summary["effective"],
        }

        adv_result = results.get("adversarial_review")
        if isinstance(adv_result, dict) and adv_result.get("final_draft"):
            metadata = {**base_metadata}
            metadata["generated_at"] = time.time()
            metadata["total_rounds"] = adv_result.get("total_rounds", 0)
            metadata["final_score"] = adv_result.get("final_score", 0.0)
            metadata["accepted"] = adv_result.get("accepted", False)
            selection = adv_result.get("evidence_selection")
            if isinstance(selection, Mapping) and selection.get("valid") is False:
                error = selection.get("error")
                metadata["evidence_selection_error"] = dict(
                    error
                    if isinstance(error, Mapping)
                    else {"reason_code": "evidence_selection_invalid"}
                )
            return {
                "survey": adv_result["final_draft"],
                "metadata": metadata,
                "review_history": self._format_review_history(
                    adv_result.get("rounds", [])
                ),
                "graph_data": graph_data,
            }

        return {
            "survey": f"Survey incomplete. Partial results from {len(results)} tasks.",
            "metadata": base_metadata,
            "review_history": [],
            "graph_data": graph_data,
        }

    @staticmethod
    def _format_review_history(rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Format bounded adversarial-review history for evaluation."""
        history = []
        for item in rounds:
            review = item.get("review", {})
            history.append(
                {
                    "round": item.get("round", 0),
                    "review": {
                        "score": review.get("score", 0),
                        "verdict": review.get("verdict", "unknown"),
                        "weaknesses": review.get("weaknesses", []),
                        "issues": review.get("issues", []),
                    },
                }
            )
        return history

    @_cleanup_incomplete_wiring
    async def _wire(self) -> None:
        """Construct all configured components once in dependency order."""
        if self._wired:
            return

        cfg = self._config
        setup_logging(cfg.logging)
        self._emit("wire.start", {"session_id": self._session_id})

        logger.info("Wiring LitAgent components (session %s)...", self._session_id)

        obs = getattr(cfg, "observability", None)
        if obs and obs.enabled and self._trace_hook is None:
            langfuse = LangFuseTracer(
                host=obs.langfuse_host,
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            )
            self._trace_hook = RedactingTraceHook(
                langfuse,
                payload_mode=obs.payload_mode,
            )

        self._cost_budget = CostBudget(
            max_tokens=cfg.safety.max_cost_tokens, warn_ratio=cfg.safety.cost_warn_ratio
        )

        try:
            self._llm = OpenAICompatibleClient(
                base_url=cfg.llm.base_url,
                model=cfg.llm.model,
                max_tokens=cfg.llm.max_tokens,
                temperature=cfg.llm.temperature,
                cost_budget=self._cost_budget,
                trace_hook=self._trace_hook,
            )
            logger.info(f"LLM client: {cfg.llm.model} @ {cfg.llm.base_url}")
        except Exception as e:
            raise ConfigError(f"Failed to create LLM client: {e}") from e

        try:
            self._registry = ToolRegistry()
            register_search_tools(self._registry)
            register_extract_tools(self._registry)
            logger.info("ToolRegistry: %d tools registered", len(self._registry))
        except Exception as e:
            raise ConfigError(f"Failed to register tools: {e}") from e

        self._budget_manager = BudgetManager(
            max_tokens=cfg.context.max_tokens,
            compact_threshold=cfg.context.compact_threshold,
        )

        # Publish the owner before connecting so cancellation can find and close
        # handles created before _connect_infra() returns.
        self._infra = Infra()
        self._infra = await self._connect_infra(cfg, infra=self._infra)

        skills_dir = str(Path(__file__).resolve().parent / "skills")
        self._skill_manager = SkillManager(skills_dir=skills_dir)
        registered_mcp_names: list[str] = []
        if cfg.mcp_servers:
            bridge = MCPBridge(registry=self._registry)
            self._mcp_bridge = bridge
            try:
                registered_mcp_names = await bridge.connect_all(cfg.mcp_servers)
                logger.info("MCP: %d tools registered", len(registered_mcp_names))
            except Exception as exc:
                logger.warning(
                    "MCP bridge failed, continuing without MCP tools: %s",
                    exc,
                )
                # connect_all may have opened earlier servers before a later one failed.
                try:
                    await bridge.disconnect_all()
                except Exception as close_exc:
                    logger.debug("MCP rollback disconnect error: %s", close_exc)
                self._mcp_bridge = None

        allowed_names = {
            *cfg.safety.tool_policy.allowed_names,
            *registered_mcp_names,
        }
        self._executor = ToolExecutor(
            registry=self._registry,
            cb_fail_threshold=cfg.resilience.cb_fail_threshold,
            cb_cooldown_seconds=cfg.resilience.cb_cooldown_seconds,
            trace_hook=self._trace_hook,
            allowed_names=allowed_names,
            allowed_categories=cfg.safety.tool_policy.allowed_categories,
        )

        regex_strategy = RegexStrategy(self._executor)
        if cfg.extractor.enable_llm:
            llm_strategy = LLMStrategy(
                self._llm,
                self._skill_manager,
                max_tokens=cfg.extractor.max_tokens,
                thinking_mode=cfg.extractor.thinking_mode,
            )
            self._extraction_strategy = ResilientExtractionStrategy(
                llm_strategy,
                regex_strategy,
                per_paper_timeout_ms=cfg.extractor.per_paper_timeout_ms,
            )
        else:
            self._extraction_strategy = regex_strategy

        workers: list[Worker] = []

        self._search = SearchWorker(
            executor=self._executor,
            memory_manager=(
                self._infra.memory if self._run_policy.memory_writes else None
            ),
            trace_hook=self._trace_hook,
            replay_outputs=(
                self._external_inputs.search_outputs
                if self._external_inputs is not None
                else None
            ),
            replay_fingerprint=(
                self._external_inputs.fingerprint
                if self._external_inputs is not None
                else None
            ),
        )
        workers.append(self._search)

        self._recall = RecallWorker(
            retriever=self._infra.retriever,
            trace_hook=self._trace_hook,
        )
        workers.append(self._recall)

        self._dedup = DedupWorker()
        workers.append(self._dedup)

        self._relevance_gate = RelevanceGateWorker(
            reranker=self._infra.relevance_reranker,
            max_papers=cfg.relevance.max_papers,
            min_papers=cfg.relevance.min_papers,
            cross_encoder_min_score=cfg.relevance.cross_encoder_min_score,
            lexical_min_score=cfg.relevance.lexical_min_score,
            trace_hook=self._trace_hook,
        )
        workers.append(self._relevance_gate)

        self._extractor = ExtractorWorker(
            strategy=self._extraction_strategy,
            max_concurrent=cfg.extractor.max_concurrent,
            detector=InjectionDetector(),
            max_papers=cfg.extractor.max_papers,
        )
        workers.append(self._extractor)

        self._graph = GraphWorker(max_papers=cfg.extractor.max_papers)
        workers.append(self._graph)
        self._evidence_selector = EvidenceSelector(
            reranker=self._infra.evidence_reranker,
            budget=self._budget_manager,
            top_k_per_section=cfg.context.evidence_top_k_per_section,
            max_items=cfg.context.evidence_max_items,
            per_paper_cap=cfg.context.evidence_per_paper_cap,
            trace_hook=self._trace_hook,
        )

        self._synthesis = SynthesisWorker(
            llm=self._llm,
            memory=self._infra.memory if self._run_policy.memory_reads else None,
            budget=self._budget_manager,
            skill_manager=self._skill_manager,
            agent_config=cfg.agent,
            methodology_provider=lambda task: build_methodology(
                str(task.input_data.get("query", "")),
                task.input_data.get("methodology_results", {}),
                cfg.rag.content_mode,
                source_outcomes=self._summarize_search_outcomes(
                    self._search.get_source_outcomes() if self._search else ()
                )[0],
                started_at=task.input_data.get("search_started_at"),
            ),
            evidence_selector=self._evidence_selector,
            context_config=cfg.context,
            trace_hook=self._trace_hook,
        )
        workers.append(self._synthesis)

        self._reviewer = ReviewerWorker(
            llm=self._llm,
            claims_index=(
                self._infra.claims_index if self._run_policy.claims_reads else None
            ),
            budget=self._budget_manager,
            skill_manager=self._skill_manager,
            config=cfg.adversarial,
            context_config=cfg.context,
            trusted_claim_recall_top_k=cfg.rag.trusted_claim_recall_top_k,
            trusted_claim_context_max_chars=cfg.rag.trusted_claim_context_max_chars,
        )
        workers.append(self._reviewer)

        self._adversarial = AdversarialReviewWorker(
            llm=self._llm,
            synthesis=self._synthesis,
            reviewer=self._reviewer,
            max_rounds=cfg.adversarial.max_rounds,
            pass_threshold=cfg.adversarial.pass_threshold,
            thinking_mode=cfg.adversarial.thinking_mode,
            trace_hook=self._trace_hook,
        )
        workers.append(self._adversarial)

        logger.info("Workers: %d registered", len(workers))

        self._scheduler = Scheduler(
            workers=workers,
            max_concurrent=cfg.orchestrator.max_concurrent,
            timeout_ms=cfg.orchestrator.timeout_ms,
            cost_budget=self._cost_budget,
            trace_hook=self._trace_hook,
        )

        self._planner = SurveyPlanner(
            llm=self._llm,
            config=cfg.planner,
            trace_hook=self._trace_hook,
            memory_manager=(
                self._infra.memory if self._run_policy.memory_reads else None
            ),
            rag_config=cfg.rag,
            adversarial_timeout_ms=cfg.adversarial.task_timeout_ms,
            decomposition_override=(
                self._external_inputs.decomposition
                if self._external_inputs is not None
                else None
            ),
            replay_fingerprint=(
                self._external_inputs.fingerprint
                if self._external_inputs is not None
                else None
            ),
        )

        eval_mt = cfg.eval.max_tokens
        self._evaluators = [
            CitationEvaluator(
                self._llm,
                max_tokens=eval_mt,
                thinking_mode=cfg.eval.thinking_mode,
                timeout_seconds=cfg.eval.timeout_seconds,
            ),
            ConsistencyEvaluator(
                self._llm,
                max_tokens=eval_mt,
                thinking_mode=cfg.eval.thinking_mode,
                timeout_seconds=cfg.eval.timeout_seconds,
            ),
            RagasFaithfulnessEvaluator(self._config, llm=self._llm),
        ]

        self._wired = True

        self._emit(
            "wire.complete",
            {
                "session_id": self._session_id,
                "worker_count": len(workers),
            },
        )
        logger.info("LitAgent wiring complete (session %s)", self._session_id)

    async def _finalize_memory(
        self,
        *,
        report_data: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Promote trusted content to long-term memory after delivery decision."""
        result: dict[str, Any] = {
            "attempted": False,
            "content_saved": False,
            "consolidated": False,
            "reason_code": "",
            "error_type": None,
        }
        if not self._run_policy.memory_writes:
            result["reason_code"] = "memory_write_disabled"
            return result
        memory = self._infra.memory
        if memory is None:
            result["reason_code"] = "memory_unavailable"
            return result

        if not is_publishable_report(report_data):
            result["reason_code"] = "content_consolidation_skipped"
            return result

        result["attempted"] = True
        state = {
            "messages": [
                {
                    "type": "human",
                    "content": f"Survey session {self._session_id}",
                },
                {
                    "type": "ai",
                    "content": str(report_data.get("survey") or ""),
                },
            ],
        }

        try:
            await memory.save_state(self._session_id, state)
            result["content_saved"] = True
            episode = await memory.consolidate(self._session_id, llm=self._llm)
            if episode is None:
                result["reason_code"] = "consolidation_returned_none"
                return result
            result["consolidated"] = True
            result["reason_code"] = "consolidated"
            return result
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            result["reason_code"] = (
                "consolidation_failed"
                if result["content_saved"]
                else "working_save_failed"
            )
            result["error_type"] = type(exc).__name__
            logger.warning(
                "Memory finalize degraded reason=%s error_type=%s",
                result["reason_code"],
                result["error_type"],
            )
            return result

    @staticmethod
    def _create_reranker(model_name: str | None = None) -> Reranker | None:
        """Create the configured reranker implementation."""
        try:
            return (
                CrossEncoderReranker(model_name)
                if model_name
                else CrossEncoderReranker()
            )
        except Exception as exc:
            logger.warning(
                "CrossEncoder unavailable reason=model_init_failed error_type=%s",
                type(exc).__name__,
            )
            return None

    async def _create_reranker_async(
        self,
        model_name: str | None = None,
    ) -> Reranker | None:
        """Offload synchronous model loading to a thread."""
        if model_name is None:
            return await asyncio.to_thread(self._create_reranker)
        return await asyncio.to_thread(self._create_reranker, model_name)

    async def _connect_rankers(self, cfg: AppConfig, infra: Infra) -> None:
        """Load shared models with independent product/retrieval ownership."""
        # Product rankers are independent of optional database availability.
        rankers: dict[str, Reranker | None] = {}
        for component, enabled, model in (
            (
                "relevance_reranker",
                cfg.relevance.reranker_enabled,
                cfg.relevance.reranker_model,
            ),
            (
                "evidence_reranker",
                cfg.context.evidence_reranker_enabled,
                cfg.context.evidence_reranker_model,
            ),
            (
                "reranker",
                cfg.rag.enabled and cfg.rag.reranker_enabled,
                cfg.rag.reranker_model,
            ),
        ):
            ranker = None
            if enabled:
                if model not in rankers:
                    try:
                        rankers[model] = await self._create_reranker_async(model)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        rankers[model] = None
                        logger.warning(
                            "Ranker load failed model=%s error=%s",
                            model,
                            type(exc).__name__,
                        )
                ranker = rankers[model]
            setattr(infra, component, ranker)
            if ranker is None:
                self._emit(
                    "ranking.degraded",
                    {
                        "component": component,
                        "model": model,
                        "reason_code": (
                            "ranker_unavailable" if enabled else "ranker_disabled"
                        ),
                        "method": "lexical_fallback",
                    },
                )

    async def _connect_infra(
        self, cfg: AppConfig, *, infra: Infra | None = None
    ) -> Infra:
        """Connect optional backends independently and return those available."""
        mem_cfg = cfg.memory
        if infra is None:
            infra = Infra()
        await self._connect_rankers(cfg, infra)

        try:
            working = await WorkingMemory.connect(mem_cfg)
            infra._working_memory = working
            logger.info("Infra: Redis connected (Working Memory)")
        except Exception as exc:
            logger.warning(
                "Infra: Redis unavailable - Working Memory disabled (%s)",
                exc,
            )
            working = None

        qdrant_client = None
        episodic = None
        try:
            from qdrant_client import AsyncQdrantClient
            from qdrant_client.models import Distance, VectorParams

            qdrant_client = AsyncQdrantClient(url=mem_cfg.qdrant_url)
            infra._qdrant_client = qdrant_client
            dim = await self._get_embedding_dim_async()

            for collection_name in ("episodes", cfg.rag.claims_collection):
                try:
                    await qdrant_client.get_collection(collection_name)
                except Exception:
                    await qdrant_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=dim,
                            distance=Distance.COSINE,
                        ),
                    )
                    logger.info(
                        "Infra: Created Qdrant collection '%s'", collection_name
                    )

            episodic = EpisodicMemory(qdrant_client)
            infra.claims_index = ClaimsIndex(
                qdrant_client,
                trace_hook=self._trace_hook,
                collection_name=cfg.rag.claims_collection,
            )

            try:
                if not cfg.rag.enabled:
                    raise ConfigError("RAG disabled by configuration")
                paper_identity = CollectionIdentity.from_config(cfg.rag)
                paper_embedder = build_retrieval_embedder(cfg.rag)
                paper_dim = await asyncio.to_thread(lambda: paper_embedder.dim)
                vector_store = await QdrantVectorStore.ensure_compatible(
                    qdrant_client,
                    paper_identity.collection_name,
                    paper_dim,
                    identity=paper_identity,
                    embedder=paper_embedder,
                )
                infra.retriever = HybridRetriever(
                    vector_store,
                    infra.reranker,
                    trace_hook=self._trace_hook,
                )
            except Exception as exc:
                logger.warning("Infra: RAG unavailable (%s)", exc)
            logger.info("Infra: Qdrant connected (Episodic + Claims + Papers)")
        except Exception as exc:
            logger.warning(
                "Infra: Qdrant unavailable - RAG/Claims/Episodic disabled (%s)",
                exc,
            )
            if qdrant_client is not None:
                try:
                    await qdrant_client.close()
                except Exception as close_exc:
                    logger.debug("Qdrant rollback close error: %s", close_exc)
            infra._qdrant_client = None
            infra.claims_index = None
            infra.retriever = None
            infra.reranker = None
            episodic = None

        pg_pool = None
        semantic = None
        procedural = None
        try:
            import asyncpg

            pg_pool = await asyncpg.create_pool(
                mem_cfg.pg_url,
                min_size=2,
                max_size=10,
            )
            infra._pg_pool = pg_pool
            semantic = SemanticMemory(pg_pool)
            procedural = ProceduralMemory(pg_pool)
            await procedural.ensure_tables()
            logger.info("Infra: PostgreSQL connected (Semantic + Procedural)")
        except Exception as exc:
            logger.warning(
                "Infra: PostgreSQL unavailable - Semantic/Procedural disabled (%s)",
                exc,
            )
            if pg_pool is not None:
                try:
                    await pg_pool.close()
                except Exception as close_exc:
                    logger.debug("PostgreSQL rollback close error: %s", close_exc)
            infra._pg_pool = None
            semantic = None
            procedural = None

        if working is not None and episodic is not None:
            infra.memory = MemoryManager(
                working=working,
                episodic=episodic,
                semantic=semantic,
                procedural=procedural,
                trace_hook=self._trace_hook,
            )
            logger.info("Infra: Memory Manager assembled (4-layer)")
        elif working is not None:
            logger.warning(
                "Infra: MemoryManager skipped — Episodic unavailable (Qdrant)"
            )
        else:
            logger.warning("Infra: MemoryManager skipped — Working unavailable (Redis)")

        return infra

    @staticmethod
    def _get_embedding_dim() -> int:
        """Return the active embedder dimension when available."""
        from litagent.rag.embedder import get_embedder

        return get_embedder().dim

    async def _get_embedding_dim_async(self) -> int:
        """Offload synchronous embedding model init to a thread."""
        return await asyncio.to_thread(self._get_embedding_dim)

    async def _evaluate(
        self,
        *,
        query: str,
        survey: str,
        results: Mapping[str, Any],
        phase: Literal["initial", "post_rewrite", "post_human_revision"],
        parent_task_id: str = "",
    ) -> dict[str, Any]:
        """Run evaluators concurrently and serialize results by metric."""
        if phase not in {"initial", "post_rewrite", "post_human_revision"}:
            raise ValueError(f"unsupported evaluation phase: {phase}")
        if not self._evaluators:
            return {}

        task_id = f"evaluation.{phase}"
        self._emit(
            "subspan.start",
            {
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                "name": task_id,
                "phase": phase,
            },
        )
        token = set_task_id(task_id)
        output: dict[str, Any] = {}

        try:
            adversarial_result = results.get("adversarial_review")
            selection = (
                adversarial_result.get("evidence_selection")
                if isinstance(adversarial_result, Mapping)
                else None
            )
            if isinstance(selection, Mapping) and selection.get("valid") is False:
                # Retain the diagnostic draft without spending on invalid inputs.
                output = {
                    evaluator.metric_name: {
                        "score": 0.0,
                        "passed": True,
                        "skipped": True,
                        "details": {"skipped_reason": "evidence_selection_invalid"},
                    }
                    for evaluator in self._evaluators
                }
                return output
            context = self._build_eval_context(
                query=query,
                survey=survey,
                results=results,
            )
            scoped = next(
                (
                    ev
                    for ev in self._evaluators
                    if isinstance(ev, RagasFaithfulnessEvaluator)
                ),
                None,
            )
            shared = (
                asyncio.create_task(scoped.evaluate(survey, context))
                if scoped
                else None
            )

            async def run_metric(evaluator):
                """Share the two-stage scientific inventory with consistency checking."""
                if evaluator is scoped:
                    return await shared
                metric_context = context
                if isinstance(evaluator, ConsistencyEvaluator):
                    metric_context = {
                        **context,
                        "scoped_faithfulness_result": await shared if shared else None,
                    }
                return await evaluator.evaluate(survey, metric_context)

            evaluated = await asyncio.gather(
                *[run_metric(evaluator) for evaluator in self._evaluators],
                return_exceptions=True,
            )

            for result in evaluated:
                if isinstance(result, asyncio.CancelledError):
                    raise result

            for evaluator, result in zip(self._evaluators, evaluated):
                if isinstance(result, BaseException):
                    logger.warning(
                        f"Evaluator {evaluator.metric_name} failed "
                        f"error_type = {type(result).__name__}"
                    )
                    continue

                output[result.metric] = {
                    "score": result.score,
                    "passed": result.passed,
                    "skipped": result.skipped,
                    "details": {
                        **result.details,
                        **(
                            {"threshold": evaluator._threshold}
                            if context.get("evaluation_version")
                            == "survey-evaluation-v2"
                            and hasattr(evaluator, "_threshold")
                            else {}
                        ),
                    },
                }
            return output
        finally:
            reset_task_id(token)
            self._emit(
                "subspan.end",
                {
                    "task_id": task_id,
                    "output": output,
                    "phase": phase,
                },
            )

    @staticmethod
    def _collect_extractions(results: dict[str, Any]) -> list[dict]:
        """Collect normalized paper extractions from task results."""
        for result in results.values():
            if (
                isinstance(result, list)
                and result
                and isinstance(result[0], dict)
                and "claims" in result[0]
            ):
                return result
        return []

    def _build_eval_context(
        self, *, query: str, survey: str, results: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Build paper, claim, and evidence context from extraction output."""
        extractions = self._collect_extractions(dict(results))
        ledger = collect_ledger(extractions)
        ordered_refs = extract_evidence_refs_ordered(survey)

        referenced: list[dict[str, Any]] = []
        unresolved: list[str] = []
        for evidence_id in ordered_refs:
            item = ledger.get(evidence_id)
            if isinstance(item, dict):
                referenced.append(item)
            else:
                unresolved.append(evidence_id)

        selected: list[dict[str, Any]] = []
        selected_seen: set[str] = set()
        adversarial = results.get("adversarial_review")
        selection_summary = (
            adversarial.get("evidence_selection")
            if isinstance(adversarial, Mapping)
            else None
        )
        if (
            isinstance(selection_summary, Mapping)
            and selection_summary.get("valid") is True
        ):
            sections = selection_summary.get("section_evidence_ids")
            if isinstance(sections, Mapping):
                for raw_ids in sections.values():
                    if not isinstance(raw_ids, list):
                        continue
                    for evidence_id in raw_ids:
                        if (
                            isinstance(evidence_id, str)
                            and evidence_id not in selected_seen
                            and isinstance(ledger.get(evidence_id), dict)
                        ):
                            selected_seen.add(evidence_id)
                            selected.append(ledger[evidence_id])

        claims = [
            {"text": claim}
            for extraction in extractions
            for claim in extraction.get("claims", [])
            if isinstance(claim, str) and claim.strip()
        ]

        from litagent.audit_context import SurveyAuditContext, check_structure
        from litagent.eval.scoped import EVALUATION_VERSION
        from litagent.review_structure import check_report_language

        structure = (
            (adversarial or {}).get("review_structure", "legacy")
            if isinstance(adversarial, Mapping)
            else "legacy"
        )
        methodology = (
            (adversarial or {}).get("review_methodology", {})
            if isinstance(adversarial, Mapping)
            else {}
        )
        language = (
            (adversarial or {}).get("report_language", "en")
            if isinstance(adversarial, Mapping)
            else "en"
        )
        audit_context = SurveyAuditContext(
            query=query,
            report_language=language,
            review_structure=structure,
            review_methodology=methodology,
        )
        return {
            "evaluation_version": EVALUATION_VERSION,
            "review_structure": structure,
            "report_language": language,
            "language_checks": check_report_language(
                survey, language, methodology, evidence=ledger
            ),
            "review_methodology": methodology,
            "structure_checks": check_structure(
                survey,
                audit_context,
                selected_evidence_ids=[i["evidence_id"] for i in selected],
            ),
            "method_checks": check_methodology(
                survey, structure, language, methodology
            ),
            CTX_QUERY: query,
            CTX_PAPERS: extractions,
            CTX_CLAIMS: claims,
            CTX_EVIDENCE: ledger,
            CTX_REFERENCED_EVIDENCE_IDS: ordered_refs,
            CTX_REFERENCED_EVIDENCE: referenced,
            CTX_UNRESOLVED_EVIDENCE_IDS: unresolved,
            CTX_SELECTED_EVIDENCE: selected,
        }

    @staticmethod
    def _passed_metric_regressed(
        initial: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> bool:
        """Reject rewrites that degrade previously passing metrics."""
        for metric, initial_result in initial.items():
            if not isinstance(initial_result, Mapping):
                continue
            if (
                initial_result.get("passed") is True
                and initial_result.get("skipped") is not True
            ):
                candidate_result = candidate.get(metric)
                if (
                    not isinstance(candidate_result, Mapping)
                    or candidate_result.get("passed") is not True
                    or candidate_result.get("skipped") is True
                ):
                    return True
        return False

    @staticmethod
    def _rewrite_material_packet(
        items: Sequence[Mapping[str, Any]], *, candidate_count: int | None = None
    ) -> dict[str, Any]:
        """Send complete repair sources without host locators or repeated text."""
        evidence, supports, source_keys = [], {}, {}
        for item in items:
            eid = str(item["evidence_id"])
            paper_id = str(item.get("paper_id") or "")
            # Missing identity cannot authorize sharing across unrelated items.
            support = str(item.get("supporting_text") or item.get("text") or "")
            key = (paper_id or f"unknown:{eid}", support)
            sid = source_keys.setdefault(key, f"s{len(source_keys) + 1}")
            supports[sid] = {"paper_id": paper_id, "text": support}
            value = {
                "evidence_id": eid,
                "paper_id": paper_id,
                "paper_title": item.get("paper_title") or "",
                "text": item.get("text") or "",
                "original_support_ref": sid,
            }
            if item.get("method_entities"):
                value["method_entities"] = item["method_entities"]
            evidence.append(value)
        total = len(items) if candidate_count is None else candidate_count
        return {
            "material_contract": "repair-original-supports-v1",
            "coverage": {
                "candidate_items": total,
                "supplied_items": len(items),
                "omitted_items": max(0, total - len(items)),
                "instruction": "Only supplied items may justify repair; omitted items are unavailable.",
            },
            "evidence": evidence,
            "original_supports": supports,
        }

    def _build_rewrite_evidence(
        self,
        *,
        diagnostics: Sequence[Mapping[str, Any]],
        evidence_ledger: Mapping[str, Mapping[str, Any]],
        selected_evidence_ids: Sequence[str],
        max_items: int,
        max_chars: int,
    ) -> list[dict[str, Any]]:
        """Collect evidence items focused on diagnostic failures, bounded by budget."""
        ordered_ids: list[str] = []
        seen: set[str] = set()
        has_uncited_claim = False

        for diagnostic in diagnostics:
            if diagnostic.get("reason_code") == "uncited_claim":
                has_uncited_claim = True
            raw_ids = diagnostic.get("evidence_ids")
            if not isinstance(raw_ids, list):
                continue
            for evidence_id in raw_ids:
                if isinstance(evidence_id, str) and evidence_id not in seen:
                    seen.add(evidence_id)
                    ordered_ids.append(evidence_id)

        if has_uncited_claim:
            # Distribute the bounded repair pool across diagnostic subjects rather
            # than spending it on the evidence selector's unrelated prefix.
            repair_queries = [
                set(
                    re.findall(
                        r"[a-z0-9]+",
                        f"{d.get('claim_text', '')} {d.get('source_quote', '')}".lower(),
                    )
                )
                for d in diagnostics
                if d.get("reason_code") == "uncited_claim"
            ]
            candidates = [
                eid
                for eid in selected_evidence_ids
                if isinstance(evidence_ledger.get(eid), Mapping)
            ]
            ranked = [
                sorted(
                    candidates,
                    key=lambda eid: -EvidenceSelector._lexical_score(
                        tokens, evidence_ledger[eid]
                    ),
                )
                for tokens in repair_queries
                if tokens
            ]
            for rank in range(len(candidates)):
                for query_order in ranked:
                    eid = query_order[rank]
                    if eid not in seen:
                        seen.add(eid)
                        ordered_ids.append(eid)
            for evidence_id in selected_evidence_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    ordered_ids.append(evidence_id)

        selected: list[dict[str, Any]] = []
        for evidence_id in ordered_ids:
            raw = evidence_ledger.get(evidence_id)
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            if item.get("evidence_id") != evidence_id or not item.get("text"):
                continue
            if len(selected) >= max_items:
                break
            packet = self._rewrite_material_packet(
                [
                    {**entry, "evidence_id": f"e{index + 1}"}
                    for index, entry in enumerate([*selected, item])
                ],
                candidate_count=len(ordered_ids),
            )
            rendered_size = len(
                json.dumps(packet, ensure_ascii=False, separators=(",", ":"))
            )
            # Skip the entire oversized source, never truncate supporting text.
            if rendered_size > max_chars:
                continue
            selected.append(item)
        return selected

    async def _attempt_evidence_rewrite(
        self,
        *,
        query: str,
        report_data: dict[str, Any],
        results: Mapping[str, Any],
        initial_evaluation: Mapping[str, Any],
        initial_quality: Mapping[str, Any],
        report_language: Literal["en", "zh-CN"] = "en",
    ) -> RewriteOutcome:
        """Attempt evidence-grounded rewrite with full validation and atomic commit."""
        original_survey = str(report_data.get("survey") or "")
        if not original_survey:
            return RewriteOutcome(
                False,
                False,
                None,
                None,
                None,
                None,
                "empty_original_survey",
            )

        diagnostics = (
            initial_evaluation.get("faithfulness", {})
            .get("details", {})
            .get("unsupported_claims")
        )
        if not isinstance(diagnostics, list) or not diagnostics:
            return RewriteOutcome(
                False,
                False,
                None,
                None,
                None,
                None,
                "no_repairable_diagnostics",
            )

        eval_context = self._build_eval_context(
            query=query,
            survey=original_survey,
            results=results,
        )
        ledger = eval_context[CTX_EVIDENCE]
        selected_ids = [
            item["evidence_id"]
            for item in eval_context[CTX_SELECTED_EVIDENCE]
            if isinstance(item, Mapping) and item.get("evidence_id")
        ]
        rewrite_cfg = self._config.eval.rewrite
        rewrite_items = self._build_rewrite_evidence(
            diagnostics=diagnostics,
            evidence_ledger=ledger,
            selected_evidence_ids=selected_ids,
            max_items=rewrite_cfg.max_items,
            max_chars=rewrite_cfg.max_chars,
        )
        original_refs = extract_evidence_refs(original_survey)
        # Only a uniquely located citation occurrence can be removed.
        removable_refs = {
            eid
            for diagnostic in diagnostics
            if isinstance(diagnostic, Mapping)
            and diagnostic.get("alignment_status") == "exact"
            and diagnostic.get("citation_scope") == "exact"
            and isinstance(diagnostic.get("source_quote"), str)
            for eid in diagnostic.get("evidence_ids", [])
            if isinstance(eid, str)
            and f"[E:{eid}]" in diagnostic["source_quote"]
            and original_survey.count(f"[E:{eid}]") == 1
        }
        reference_policy = {
            "must_preserve_evidence_ids": sorted(original_refs - removable_refs),
            "allowed_removed_evidence_ids": sorted(removable_refs),
            "instruction": "Repair only diagnosed claims. Preserve every must_preserve reference. Never invent IDs.",
        }
        aliases = build_citation_aliases(
            list(
                dict.fromkeys(
                    [item["evidence_id"] for item in rewrite_items]
                    + extract_evidence_refs_ordered(original_survey)
                )
            )
        )
        alias_by_id = aliases.evidence_id_to_alias
        candidate_ids = {
            eid
            for item in diagnostics
            if isinstance(item, Mapping)
            for eid in item.get("evidence_ids", [])
            if isinstance(eid, str)
        }
        if any(item.get("reason_code") == "uncited_claim" for item in diagnostics):
            candidate_ids.update(selected_ids)
        model_items = self._rewrite_material_packet(
            [
                {**item, "evidence_id": alias_by_id[item["evidence_id"]]}
                for item in rewrite_items
            ],
            candidate_count=len(candidate_ids),
        )
        diagnostic_keys = (
            "claim_id",
            "kind",
            "claim_text",
            "source_unit_id",
            "source_fragment_id",
            "source_quote",
            "source_start",
            "source_end",
            "section",
            "reason_code",
            "explanation",
            "alignment_status",
            "citation_scope",
        )
        model_diagnostics = [
            {
                **{key: item[key] for key in diagnostic_keys if key in item},
                "evidence_ids": [
                    alias_by_id[eid]
                    for eid in item.get("evidence_ids", [])
                    if eid in alias_by_id
                ],
                "source_offsets_context": "frozen_original_before_aliasing",
                "repair_instruction": (
                    "Verify complete original support, then attach its citation directly "
                    "to this assertion if entailed; otherwise state the specific evidence gap."
                    if item.get("reason_code") == "uncited_claim"
                    else "Repair the diagnosed fact/attribution against its original support "
                    "or trusted run record; adding a citation alone is insufficient."
                ),
            }
            for item in diagnostics
            if isinstance(item, Mapping)
        ]
        model_policy = {
            **reference_policy,
            "must_preserve_evidence_ids": [
                alias_by_id[eid]
                for eid in reference_policy["must_preserve_evidence_ids"]
            ],
            "allowed_removed_evidence_ids": [
                alias_by_id[eid]
                for eid in reference_policy["allowed_removed_evidence_ids"]
            ],
        }

        # Ask the model to repair the unsupported evidence claims.
        self._emit(
            "subspan.start",
            {
                "task_id": "evidence_rewrite",
                "parent_task_id": "",
                "name": "evidence_rewrite",
                "round": 0,
            },
        )
        token = set_task_id("evidence_rewrite")
        candidate: str | None = None
        llm_error_type: str | None = None
        reason_code = "llm_call_failed"
        reservation = 0
        budget_reserved = False
        request_budget = getattr(self._llm, "_cost_budget", None)
        if not isinstance(request_budget, CostBudget):
            request_budget = None

        try:
            chat_kwargs: dict[str, Any] = {
                "max_tokens": self._config.eval.max_tokens,
                "transport_max_attempts": 1,
            }
            if self._config.eval.thinking_mode != "provider_default":
                chat_kwargs["extra_body"] = {
                    "thinking": {"type": self._config.eval.thinking_mode}
                }
            async with asyncio.timeout(self._config.eval.timeout_seconds):
                messages = self._synthesis.rewrite_with_evidence(
                    original_survey,
                    json.dumps(model_items, ensure_ascii=False, separators=(",", ":")),
                    json.dumps(model_diagnostics, ensure_ascii=False, indent=2),
                    query=query,
                    aliases=aliases,
                    report_language=report_language,
                    **(
                        {
                            "review_structure": report_data.get("metadata", {}).get(
                                "review_structure"
                            ),
                            "review_methodology": report_data.get("metadata", {}).get(
                                "review_methodology", {}
                            ),
                        }
                        if report_data.get("metadata", {}).get(
                            "review_structure", "legacy"
                        )
                        != "legacy"
                        else {}
                    ),
                ) + [{"role": "user", "content": json.dumps(model_policy)}]
                reservation = check_request_budget(
                    self._llm,
                    messages,
                    self._config.eval.max_tokens,
                    self._config.eval.context_tokens,
                )
                if request_budget is not None:
                    request_budget.reserve(reservation)
                    budget_reserved = True
                response = await self._llm.chat(messages, **chat_kwargs)
                if request_budget is not None and not response.usage.get(
                    "usage_available", bool(response.usage)
                ):
                    request_budget.record(
                        {"prompt_tokens": reservation, "completion_tokens": 0}
                    )
                    raise ValueError("rewrite_usage_incomplete")
            candidate = (response.content or "").strip()
        except asyncio.CancelledError:
            reason_code = "cancelled"
            llm_error_type = "CancelledError"
            raise
        except TimeoutError:
            reason_code = "llm_timeout"
            llm_error_type = "TimeoutError"
        except ValueError as exc:
            known_errors = {
                "evaluation_context_budget_exceeded": "rewrite_context_budget_exceeded",
                "evaluation_run_budget_exceeded": "rewrite_run_budget_exceeded",
                "rewrite_usage_incomplete": "rewrite_usage_incomplete",
            }
            reason_code = known_errors.get(str(exc), "llm_call_failed")
            llm_error_type = "ValueError"
        except Exception as exc:
            reason_code = "llm_call_failed"
            llm_error_type = type(exc).__name__
        finally:
            if budget_reserved:
                request_budget.release(reservation)
            reset_task_id(token)

        if candidate is None:
            self._emit(
                "subspan.end",
                {
                    "task_id": "evidence_rewrite",
                    "output": {"accepted": False, "reason": reason_code},
                },
            )
            return RewriteOutcome(
                True,
                False,
                None,
                None,
                None,
                None,
                reason_code,
            )

        # Validate the candidate before running evaluators again.
        candidate = decode_citations_traced(
            candidate,
            aliases,
            trace_hook=self._trace_hook,
            parent_task_id="evidence_rewrite",
            stage="evidence_rewrite",
        ).text
        candidate = normalize_citations_traced(
            candidate,
            {item["evidence_id"] for item in rewrite_items if item.get("evidence_id")},
            trace_hook=self._trace_hook,
            parent_task_id="evidence_rewrite",
            stage="evidence_rewrite",
        ).text
        try:
            validate_authored_methodology(
                candidate,
                report_data.get("metadata", {}).get("review_structure", "legacy"),
            )
        except ValueError as exc:
            self._emit(
                "subspan.end",
                {
                    "task_id": "evidence_rewrite",
                    "output": {"accepted": False, "reason": str(exc)},
                },
            )
            return RewriteOutcome(True, False, candidate, None, None, None, str(exc))
        candidate = assemble_methodology(
            candidate,
            report_data.get("metadata", {}).get("review_structure", "legacy"),
            report_language,
            report_data.get("metadata", {}).get("review_methodology", {}),
        )
        validation = self._validate_rewrite_candidate(
            original=original_survey,
            candidate=candidate,
            evidence_ledger=ledger,
            allowed_removed_refs=sorted(removable_refs),
        )
        if not validation.accepted:
            self._emit(
                "subspan.end",
                {
                    "task_id": "evidence_rewrite",
                    "output": {
                        "accepted": False,
                        "reason": "validation_failed",
                        "reason_codes": list(validation.reason_codes),
                        "unexpected_removed_evidence_ids": list(
                            validation.unexpected_removed_evidence_ids
                        ),
                    },
                },
            )
            logger.warning(
                "Evidence rewrite rejected reason_codes=%s",
                list(validation.reason_codes),
            )
            return RewriteOutcome(
                True,
                False,
                candidate,
                validation,
                None,
                None,
                "validation_failed",
            )

        # Re-evaluate the validated candidate as a separate trace phase.
        candidate_evaluation = await self._evaluate(
            query=query,
            survey=candidate,
            results=results,
            phase="post_rewrite",
            parent_task_id="evidence_rewrite",
        )
        candidate_quality = self._derive_quality(candidate_evaluation)

        if self._passed_metric_regressed(initial_evaluation, candidate_evaluation):
            self._emit(
                "subspan.end",
                {
                    "task_id": "evidence_rewrite",
                    "output": {
                        "accepted": False,
                        "reason": "metric_regressed",
                        "candidate_evaluation": candidate_evaluation,
                    },
                },
            )
            return RewriteOutcome(
                True,
                False,
                candidate,
                validation,
                candidate_evaluation,
                candidate_quality,
                "metric_regressed",
            )

        if candidate_quality["status"] != "passed":
            self._emit(
                "subspan.end",
                {
                    "task_id": "evidence_rewrite",
                    "output": {
                        "accepted": False,
                        "reason": "quality_still_failed",
                        "candidate_quality": candidate_quality,
                    },
                },
            )
            return RewriteOutcome(
                True,
                False,
                candidate,
                validation,
                candidate_evaluation,
                candidate_quality,
                "quality_still_failed",
            )

        # Commit the survey, evaluation, and quality fields together.
        report_data["survey"] = candidate
        report_data["evaluation"] = candidate_evaluation
        report_data["quality"] = candidate_quality

        self._emit(
            "subspan.end",
            {
                "task_id": "evidence_rewrite",
                "output": {
                    "accepted": True,
                    "reason": "committed",
                    "candidate_quality": candidate_quality,
                },
            },
        )
        logger.info("Evidence rewrite committed")
        return RewriteOutcome(
            True,
            True,
            candidate,
            validation,
            candidate_evaluation,
            candidate_quality,
            "committed",
        )

    async def _attempt_human_revision(
        self,
        *,
        query: str,
        report_data: dict[str, Any],
        results: Mapping[str, Any],
        feedback: str,
        report_language: Literal["en", "zh-CN"],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Generate one feedback revision and commit it only after every gate."""
        original = str(report_data.get("survey") or "")
        metadata = report_data.get("metadata") or {}
        eval_context = self._build_eval_context(
            query=query, survey=original, results=results
        )
        selected_items = {
            str(item["evidence_id"]): dict(item)
            for item in eval_context[CTX_SELECTED_EVIDENCE]
            if isinstance(item, Mapping) and item.get("evidence_id")
        }
        adversarial_result = results.get("adversarial_review")
        summary = (
            adversarial_result.get("evidence_selection", {})
            if isinstance(adversarial_result, Mapping)
            else {}
        )
        raw_sections = summary.get("section_evidence_ids", {})
        sections = {
            str(section): [
                evidence_id
                for evidence_id in raw_ids
                if isinstance(evidence_id, str) and evidence_id in selected_items
            ]
            for section, raw_ids in raw_sections.items()
            if isinstance(raw_ids, list)
        }
        selection = EvidenceSelection(
            candidate_count=int(summary.get("candidate_count", len(selected_items))),
            selected_items=selected_items,
            section_evidence_ids=sections,
            method_by_section=dict(summary.get("method_by_section") or {}),
            omitted_count=int(summary.get("omitted_count", 0)),
            estimated_tokens=int(summary.get("estimated_tokens", 0)),
        )
        if not selected_items:
            return report_data, {
                "committed": False,
                "reason_code": "human_revision_evidence_missing",
            }

        messages = self._synthesis.revise(
            original,
            feedback,
            selection,
            query=query,
            report_language=report_language,
            **(
                {
                    "review_structure": report_data.get("metadata", {}).get(
                        "review_structure"
                    ),
                    "review_methodology": report_data.get("metadata", {}).get(
                        "review_methodology", {}
                    ),
                }
                if report_data.get("metadata", {}).get("review_structure", "legacy")
                != "legacy"
                else {}
            ),
        )
        self._emit(
            "subspan.start",
            {
                "task_id": "human_revision",
                "parent_task_id": "",
                "name": "human_revision",
                "round": 1,
            },
        )

        def finish_revision(payload: dict[str, Any]) -> dict[str, Any]:
            """Close the human-revision span before returning a stable outcome."""
            self._emit(
                "subspan.end",
                {"task_id": "human_revision", "output": payload},
            )
            return payload

        token = set_task_id("human_revision")
        try:
            kwargs: dict[str, Any] = {"max_tokens": self._config.eval.max_tokens}
            if self._config.eval.thinking_mode != "provider_default":
                kwargs["extra_body"] = {
                    "thinking": {"type": self._config.eval.thinking_mode}
                }
            async with asyncio.timeout(self._config.eval.timeout_seconds):
                response = await self._llm.chat(messages, **kwargs)
            candidate = (response.content or "").strip()
        except asyncio.CancelledError:
            finish_revision(
                {"committed": False, "reason_code": "human_revision_cancelled"}
            )
            raise
        except TimeoutError:
            return report_data, finish_revision(
                {
                    "committed": False,
                    "reason_code": "human_revision_timeout",
                }
            )
        except Exception as exc:
            return report_data, finish_revision(
                {
                    "committed": False,
                    "reason_code": "human_revision_llm_failed",
                    "error_type": type(exc).__name__,
                }
            )
        finally:
            reset_task_id(token)

        candidate = normalize_citations_traced(
            candidate,
            selected_items,
            trace_hook=self._trace_hook,
            parent_task_id="human_revision",
            stage="human_revision",
        ).text
        try:
            validate_authored_methodology(
                candidate,
                report_data.get("metadata", {}).get("review_structure", "legacy"),
            )
        except ValueError as exc:
            return report_data, finish_revision(
                {"committed": False, "reason_code": str(exc)}
            )
        candidate = assemble_methodology(
            candidate,
            report_data.get("metadata", {}).get("review_structure", "legacy"),
            report_language,
            report_data.get("metadata", {}).get("review_methodology", {}),
        )
        original_refs = extract_evidence_refs(original)
        validation = self._validate_rewrite_candidate(
            original=original,
            candidate=candidate,
            evidence_ledger=eval_context[CTX_EVIDENCE],
            allowed_removed_refs=sorted(original_refs),
        )
        if not validation.accepted:
            return report_data, finish_revision(
                {
                    "committed": False,
                    "reason_code": "human_revision_validation_failed",
                    "validation_reason_codes": list(validation.reason_codes),
                }
            )

        review = await self._reviewer.review_revision(
            candidate,
            {},
            selection,
            query=query,
            report_language=report_language,
            **(
                {
                    "review_structure": metadata.get("review_structure", "legacy"),
                    "review_methodology": metadata.get("review_methodology"),
                }
                if metadata.get("review_structure", "legacy") != "legacy"
                else {}
            ),
        )
        if not self._adversarial._review_passes(review):
            return report_data, finish_revision(
                {
                    "committed": False,
                    "reason_code": "human_revision_review_failed",
                    "review": review,
                }
            )
        evaluation = await self._evaluate(
            query=query,
            survey=candidate,
            results=results,
            phase="post_human_revision",
            parent_task_id="human_revision",
        )
        quality = self._derive_quality(evaluation)
        if quality["status"] != "passed" or self._passed_metric_regressed(
            report_data.get("evaluation", {}), evaluation
        ):
            return report_data, finish_revision(
                {
                    "committed": False,
                    "reason_code": "human_revision_quality_failed",
                    "candidate_quality": quality,
                }
            )

        updated = dict(report_data)
        updated["survey"] = candidate
        updated["evaluation"] = evaluation
        updated["quality"] = quality
        updated["delivery"] = derive_delivery(
            bool(updated.get("partial")),
            quality,
            updated.get("metadata", {}).get("degradation_reason_codes", []),
        )
        return updated, finish_revision(
            {
                "committed": True,
                "reason_code": "human_revision_committed",
                "review": review,
                "candidate_quality": quality,
            }
        )

    @staticmethod
    def _derive_quality(evaluation: dict[str, dict]) -> dict[str, Any]:
        """Derive the aggregate quality decision from evaluation metrics."""
        required = ["citation_accuracy", "faithfulness"]
        failed: list[str] = []
        unverified: list[str] = []

        for metric in required:
            ev = evaluation.get(metric, {})
            if not ev or ev.get("skipped", False):
                unverified.append(metric)
            elif not ev.get("passed", False):
                failed.append(metric)

        details = evaluation.get("faithfulness", {}).get("details", {})
        if details.get("evaluation_version") == "survey-evaluation-v2":
            for name in ("structure_checks", "method_checks"):
                check = details.get(name, {})
                if check.get("status") == "failed":
                    failed.append(name)
                elif check.get("status") not in {"passed", "not_applicable"}:
                    unverified.append(name)
            language = details.get("language_checks", {})
            if language.get("contract") == "report-prose-language-v1":
                if language.get("status") == "failed":
                    failed.append("language_checks")
                elif language.get("status") != "passed":
                    unverified.append("language_checks")
            for name in (
                "critical_attribution_errors",
                "method_assertion_errors",
                "organization_errors",
            ):
                if details.get(name):
                    failed.append(name)
        if failed:
            return {
                "status": "failed",
                "failed_metrics": failed,
                "unverified_metrics": unverified,
            }
        if unverified:
            return {
                "status": "unverified",
                "failed_metrics": [],
                "unverified_metrics": unverified,
            }

        return {"status": "passed", "failed_metrics": [], "unverified_metrics": []}

    async def cleanup(self) -> None:
        """Disconnect MCP and close each owned infrastructure handle once."""
        self._emit("cleanup.start", {"session_id": self._session_id})
        logger.info("Cleaning up LitAgent (session %s)...", self._session_id)

        if self._mcp_bridge is not None:
            bridge = self._mcp_bridge
            self._mcp_bridge = None
            try:
                await bridge.disconnect_all()
            except BaseException as exc:
                logger.debug("MCP disconnect error: %s", exc)

        infra = self._infra
        self._infra = Infra()
        await infra.close()

        self._wired = False
        self._emit("cleanup.complete", {"session_id": self._session_id})

        # Flush after cleanup.complete so terminal lifecycle events reach LangFuse.
        if self._trace_hook and hasattr(self._trace_hook, "flush"):
            try:
                self._trace_hook.flush()
            except BaseException as exc:
                logger.debug("Tracer flush error: %s", exc)

        logger.info("LitAgent cleanup complete")

    @staticmethod
    def _markdown_sections(text: str) -> list[tuple[int, str, str]]:
        """Return heading level, normalized H1/H2 heading, and direct body."""
        sections: list[tuple[int, str, str]] = []
        current_level: int | None = None
        current_heading: str | None = None
        body: list[str] = []

        # Preserve order while normalizing heading whitespace and case.
        for line in (text or "").splitlines():
            match = _MARKDOWN_HEADING_RE.match(line)
            if match:
                if current_heading is not None and current_level is not None:
                    sections.append(
                        (current_level, current_heading, "\n".join(body).strip())
                    )
                current_level = len(match.group(1))
                current_heading = " ".join(match.group(2).lower().split())
                body = []
            elif current_heading is not None:
                body.append(line)

        if current_heading is not None and current_level is not None:
            sections.append((current_level, current_heading, "\n".join(body).strip()))
        return sections

    def _validate_rewrite_candidate(
        self,
        *,
        original: str,
        candidate: str,
        evidence_ledger: Mapping[str, Mapping[str, Any]],
        allowed_removed_refs: Sequence[str] = (),
    ) -> RewriteValidation:
        """Validate a rewritten survey against structural constraints."""
        reasons: list[str] = []
        stripped = (candidate or "").strip()
        if not stripped:
            return RewriteValidation(
                False,
                ("empty_candidate",),
                (),
                (),
            )
        if stripped == (original or "").strip():
            reasons.append("identical_to_original")
        if _PLACEHOLDER_LINE_RE.search(stripped):
            if re.search(r"(?m)^\s*(?:\.\.\.|…)\s*$", stripped):
                reasons.append("ellipsis_placeholder")
            else:
                reasons.append("placeholder_detected")

        original_sections = self._markdown_sections(original)
        candidate_sections = self._markdown_sections(candidate)
        original_headings = [
            (level, heading) for level, heading, _ in original_sections
        ]
        candidate_headings = [
            (level, heading) for level, heading, _ in candidate_sections
        ]

        for heading in original_headings:
            if heading not in candidate_headings:
                reasons.append(f"missing_section:{heading}")
        retained_order = [
            heading
            for heading in candidate_headings
            if heading in set(original_headings)
        ]
        if retained_order != [
            heading
            for heading in original_headings
            if heading in set(candidate_headings)
        ]:
            reasons.append("section_order_changed")
        # H1 is the document title; only H2 entries represent content sections.
        if any(level == 2 and not body for level, _, body in candidate_sections):
            reasons.append("empty_section_detected")

        rewrite_cfg = self._config.eval.rewrite
        if original.strip():
            if (
                len(stripped)
                < len(original.strip()) * rewrite_cfg.min_body_length_ratio
            ):
                reasons.append("body_length_collapse")
            original_paragraphs = [
                part for part in original.split("\n\n") if part.strip()
            ]
            candidate_paragraphs = [
                part for part in candidate.split("\n\n") if part.strip()
            ]
            if (
                original_paragraphs
                and len(candidate_paragraphs)
                < len(original_paragraphs) * rewrite_cfg.min_paragraph_ratio
            ):
                reasons.append("paragraph_collapse")

        if find_malformed_evidence_refs(candidate):
            reasons.append("malformed_evidence_refs")
        refs = extract_evidence_refs_ordered(candidate)
        unknown = tuple(
            evidence_id for evidence_id in refs if evidence_id not in evidence_ledger
        )
        if unknown:
            reasons.append("unknown_evidence_refs")

        removable = set(allowed_removed_refs)
        removed = extract_evidence_refs(original) - set(refs) - removable
        if removed:
            reasons.append("unexpected_removed_evidence_refs")

        return RewriteValidation(
            accepted=not reasons,
            reason_codes=tuple(dict.fromkeys(reasons)),
            referenced_evidence_ids=tuple(refs),
            unknown_evidence_ids=unknown,
            unexpected_removed_evidence_ids=tuple(sorted(removed)),
        )

    def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Emit a trace event without allowing hook failures to escape."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data or {})
            except Exception as e:
                logger.debug("Trace hook failed for event '%s': %s", event, e)
