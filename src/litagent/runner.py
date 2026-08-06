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
from litagent.context.evidence_selector import EvidenceSelector
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
from litagent.evidence import (
    collect_ledger,
    extract_evidence_refs,
    extract_evidence_refs_ordered,
)
from litagent.exceptions import ConfigError
from litagent.llm.client import BaseLLMClient, OpenAICompatibleClient
from litagent.logging import get_logger, setup_logging
from litagent.mcp.bridge import MCPBridge
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.manager import MemoryManager
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.working import WorkingMemory
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.scheduler import Scheduler, Worker
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
from litagent.safety.budget import CostBudget
from litagent.safety.injection import InjectionDetector
from litagent.skills.manager import SkillManager
from litagent.tools.builtin.extract import register_extract_tools
from litagent.tools.builtin.search import register_search_tools
from litagent.tools.executor import ToolExecutor
from litagent.tools.registry import get_registry

logger = get_logger("runner")


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,2})\s+(.+?)\s*$")
_PLACEHOLDER_LINE_RE = re.compile(
    r"(?im)^\s*(?:\.\.\.|…|\[no changes to this section\.\]|"
    r"\[omitted\]|todo|tbd)\s*$"
)


@dataclass(frozen=True)
class RewriteValidation:
    accepted: bool
    reason_codes: tuple[str, ...]
    referenced_evidence_ids: tuple[str, ...]
    unknown_evidence_ids: tuple[str, ...]


@dataclass(frozen=True)
class RewriteOutcome:
    attempted: bool
    committed: bool
    candidate: str | None
    validation: RewriteValidation | None
    candidate_evaluation: Mapping[str, Any] | None
    candidate_quality: Mapping[str, Any] | None
    reason_code: str


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


class LitAgent:
    """Assemble components, execute surveys, and release shared resources."""

    def __init__(self, config: AppConfig, trace_hook: TraceHook | None = None) -> None:
        self._config = config
        self._config_summary = build_config_summary(config)
        self._session_id = str(uuid.uuid4())[:8]
        self._trace_hook = trace_hook

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
        await self._wire()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.cleanup()

    async def run(self, query: str) -> dict[str, Any]:
        """Run planning, scheduling, evaluation, and delivery for one query."""
        if not self._wired:
            await self._wire()

        self._emit("survey.start", {"query": query, "session_id": self._session_id})
        logger.info("Starting survey: %s", query)

        try:
            # Reset source outcomes from prior runs so cross-run contamination is avoided.
            if self._search is not None:
                self._search.reset_source_outcomes()

            graph = await self._planner.plan(query)
            if hasattr(self._trace_hook, "capture_graph"):
                self._trace_hook.capture_graph(graph)
            results = await self._scheduler.run(graph)
            if hasattr(self._trace_hook, "capture_graph_state"):
                self._trace_hook.capture_graph_state(graph)

            # Collect search outcomes after Scheduler completes.
            search_outcomes, degradation_codes = self._summarize_search_outcomes(
                self._search.get_source_outcomes() if self._search else ()
            )

            report_data = self._extract_report(results, query)
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
            report_data.setdefault("metadata", {})["execution"] = execution
            report_data.setdefault("metadata", {})[
                "search_source_outcomes"
            ] = search_outcomes

            # Phase 1: initial evaluation.
            initial_evaluation = await self._evaluate(
                query=query,
                survey=report_data["survey"],
                results=results,
                phase="initial",
            )
            initial_quality = self._derive_quality(initial_evaluation)
            report_data.update(
                {
                    "evaluation": initial_evaluation,
                    "quality": initial_quality,
                }
            )

            # Phase 2: evidence rewrite on quality failure.
            if initial_quality["status"] == "failed":
                rewrite_outcome = await self._attempt_evidence_rewrite(
                    query=query,
                    report_data=report_data,
                    results=results,
                    initial_evaluation=initial_evaluation,
                    initial_quality=initial_quality,
                )
                report_data.setdefault("metadata", {})["evidence_rewrite"] = {
                    "attempted": rewrite_outcome.attempted,
                    "committed": rewrite_outcome.committed,
                    "reason_code": rewrite_outcome.reason_code,
                    "validation_reason_codes": (
                        list(rewrite_outcome.validation.reason_codes)
                        if rewrite_outcome.validation
                        else []
                    ),
                    "candidate_quality": rewrite_outcome.candidate_quality,
                }

            # Phase 3: delivery + trusted claim promotion + memory + emit.
            final_quality = report_data["quality"]
            report_data["delivery"] = derive_delivery(
                report_data["partial"],
                final_quality,
                degradation_codes,
            )
            promotion = (
                await ClaimsPromoter(self._infra.claims_index).promote(
                    run_id=self._session_id,
                    domain=query,
                    report_data=report_data,
                    extractions=self._collect_extractions(results),
                )
                if self._infra.claims_index is not None
                else ClaimPromotionSummary(
                    "skipped", 0, 0, 0, "claims_index_unavailable"
                )
            )
            report_data.setdefault("metadata", {})[
                "claims_promotion"
            ] = promotion.to_dict()
            report_data["metadata"]["memory"] = await self._finalize_memory(
                report_data=report_data
            )

            report_data = normalize_survey_result(
                report_data, config_fingerprint=self._config_summary["fingerprint"]
            )

            self._emit(
                "survey.complete",
                {
                    "query": query,
                    "rounds": report_data.get("metadata", {}).get("total_rounds", 0),
                    "accepted": report_data.get("metadata", {}).get("accepted", False),
                    "quality_status": final_quality["status"],
                    "total_tokens": self._cost_budget.used if self._cost_budget else 0,
                    "delivery_status": report_data["delivery"]["status"],
                    "config_fingerprint": report_data["config_fingerprint"],
                },
            )

            logger.info("Survey complete: %d chars", len(report_data.get("survey", "")))
            return report_data

        except Exception as e:
            self._emit("survey.error", {"query": query, "error": str(e)})
            raise

    @staticmethod
    def _derive_execution(
        graph: TaskGraph,
        budget_exceeded: bool,
        final_output_present: bool,
    ) -> dict[str, Any]:
        summary = graph.execution_summary()
        reason_codes: list[str] = []
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

    @staticmethod
    def _summarize_search_outcomes(outcomes) -> tuple[list[dict], list[str]]:
        """Convert SearchSourceOutcome objects into JSON-safe metadata and degradation codes."""
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
            self._registry = get_registry()
            register_search_tools()
            register_extract_tools()
            logger.info("ToolRegistry: %d tools registered", len(self._registry))
        except Exception as e:
            raise ConfigError(f"Failed to register tools: {e}") from e

        self._executor = ToolExecutor(
            registry=self._registry,
            cb_fail_threshold=cfg.resilience.cb_fail_threshold,
            cb_cooldown_seconds=cfg.resilience.cb_cooldown_seconds,
            trace_hook=self._trace_hook,
        )

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
        if cfg.mcp_servers:
            bridge = MCPBridge()
            self._mcp_bridge = bridge
            try:
                registered = await bridge.connect_all(cfg.mcp_servers)
                logger.info("MCP: %d tools registered", len(registered))
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

        regex_strategy = RegexStrategy(self._executor)
        if cfg.extractor.enable_llm:
            llm_strategy = LLMStrategy(self._llm, self._skill_manager)
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
            memory_manager=self._infra.memory,
            trace_hook=self._trace_hook,
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
            reranker=self._infra.reranker,
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
            reranker=self._infra.reranker,
            budget=self._budget_manager,
            top_k_per_section=cfg.context.evidence_top_k_per_section,
            max_items=cfg.context.evidence_max_items,
            per_paper_cap=cfg.context.evidence_per_paper_cap,
            trace_hook=self._trace_hook,
        )

        self._synthesis = SynthesisWorker(
            llm=self._llm,
            memory=self._infra.memory,
            budget=self._budget_manager,
            skill_manager=self._skill_manager,
            agent_config=cfg.agent,
            evidence_selector=self._evidence_selector,
            context_config=cfg.context,
        )
        workers.append(self._synthesis)

        self._reviewer = ReviewerWorker(
            llm=self._llm,
            claims_index=self._infra.claims_index,
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
            memory_manager=self._infra.memory,
            rag_config=cfg.rag,
        )

        eval_mt = cfg.eval.max_tokens
        self._evaluators = [
            CitationEvaluator(self._llm, max_tokens=eval_mt),
            ConsistencyEvaluator(self._llm, max_tokens=eval_mt),
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

    async def _connect_infra(
        self,
        cfg: AppConfig,
        *,
        infra: Infra | None = None,
    ) -> Infra:
        """Connect optional backends independently and return those available."""
        mem_cfg = cfg.memory
        if infra is None:
            infra = Infra()

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
                infra.reranker = (
                    await self._create_reranker_async(cfg.rag.reranker_model)
                    if cfg.rag.reranker_enabled
                    else None
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
        phase: Literal["initial", "post_rewrite"],
        parent_task_id: str = "",
    ) -> dict[str, Any]:
        """Run evaluators concurrently and serialize results by metric."""
        if phase not in {"initial", "post_rewrite"}:
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
            context = self._build_eval_context(
                query=query,
                survey=survey,
                results=results,
            )
            evaluated = await asyncio.gather(
                *[
                    evaluator.evaluate(survey, context)
                    for evaluator in self._evaluators
                ],
                return_exceptions=True,
            )

            for result in evaluated:
                if isinstance(result, asyncio.CancelledError):
                    raise result

            for evaluator, result in zip(self._evaluators, evaluated):
                if isinstance(result, BaseException):
                    logger.warning(
                        f"Evaluator {evaluator.metric_name} failed error_type = {type(result).__name__}"
                    )
                    continue

                output[result.metric] = {
                    "score": result.score,
                    "passed": result.passed,
                    "skipped": result.skipped,
                    "details": result.details,
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

        return {
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
        """Rewrite must only repair failures, never degrade previously passing metrics."""
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
            for evidence_id in selected_evidence_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    ordered_ids.append(evidence_id)

        selected: list[dict[str, Any]] = []
        used_chars = 0
        for evidence_id in ordered_ids:
            raw = evidence_ledger.get(evidence_id)
            if not isinstance(raw, Mapping):
                continue
            item = dict(raw)
            rendered_size = (
                len(str(item.get("evidence_id") or ""))
                + len(str(item.get("paper_title") or ""))
                + len(str(item.get("text") or ""))
                + 16
            )
            if rendered_size > max_chars:
                continue
            if len(selected) >= max_items:
                break
            if used_chars + rendered_size > max_chars:
                break
            selected.append(item)
            used_chars += rendered_size
        return selected

    async def _attempt_evidence_rewrite(
        self,
        *,
        query: str,
        report_data: dict[str, Any],
        results: Mapping[str, Any],
        initial_evaluation: Mapping[str, Any],
        initial_quality: Mapping[str, Any],
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

        # ── LLM rewrite ──────────────────────────────────────────────
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

        try:
            response = await self._llm.chat(
                self._synthesis.rewrite_with_evidence(
                    original_survey,
                    json.dumps(rewrite_items, ensure_ascii=False),
                    json.dumps(diagnostics, ensure_ascii=False, indent=2),
                ),
                max_tokens=self._config.eval.max_tokens,
            )
            candidate = (response.content or "").strip()
        except asyncio.CancelledError:
            reason_code = "cancelled"
            llm_error_type = "CancelledError"
            raise
        except Exception as exc:
            reason_code = "llm_call_failed"
            llm_error_type = type(exc).__name__
        finally:
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

        # ── validation ───────────────────────────────────────────────
        validation = self._validate_rewrite_candidate(
            original=original_survey,
            candidate=candidate,
            evidence_ledger=ledger,
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

        # ── re-evaluate ──────────────────────────────────────────────
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

        # ── atomic commit ────────────────────────────────────────────
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

    @staticmethod
    def _derive_quality(evaluation: dict[str, dict]) -> dict[str, Any]:
        required = ["citation_accuracy", "faithfulness"]
        failed: list[str] = []
        unverified: list[str] = []

        for metric in required:
            ev = evaluation.get(metric, {})
            if not ev or ev.get("skipped", False):
                unverified.append(metric)
            elif not ev.get("passed", False):
                failed.append(metric)

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

        # 保留出现顺序并标准化 heading 空白/大小写，供相对结构比较。
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
        )

    def _emit(self, event: str, data: dict[str, Any] | None = None) -> None:
        """Emit a trace event without allowing hook failures to escape."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data or {})
            except Exception as e:
                logger.debug("Trace hook failed for event '%s': %s", event, e)
