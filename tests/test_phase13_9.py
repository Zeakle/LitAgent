"""Regression contracts for Phase 13.9 behavior."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from langchain_core.documents import Document

from litagent.config import AgentConfig, AppConfig, LoggingConfig, load_config
from litagent.eval.base import EvalResult
from litagent.llm.client import LLMResponse
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.interfaces import ScoredDoc
from litagent.runner import LitAgent, derive_delivery
from litagent.tools.base import ToolDefinition
from litagent.tools.executor import ToolExecutor, ToolResult
from litagent.tools.registry import ToolRegistry


def _agent() -> LitAgent:
    return LitAgent(AppConfig(agent=AgentConfig(), logging=LoggingConfig()))


def _evidence_item(evidence_id: str, text: str, *, title: str = "Paper") -> dict:
    return {
        "evidence_id": evidence_id,
        "paper_id": evidence_id.split(":", 1)[0],
        "paper_title": title,
        "text": text,
        "source_locator": "extracted_claim",
        "confidence": None,
    }


def _extraction(evidence_items: list[dict]) -> dict:
    return {
        "paper_id": "p1",
        "title": "Paper",
        "abstract": "Abstract",
        "claims": [item["text"] for item in evidence_items],
        "evidence_items": evidence_items,
    }


class TestEvidenceAlignmentContract:
    """Tests evaluation evidence alignment."""

    def test_ordered_refs_preserve_first_occurrence(self):
        from litagent.evidence import extract_evidence_refs_ordered

        refs = extract_evidence_refs_ordered(
            "[E:p2:claim:0] then [E:p1:claim:0] and [E:p2:claim:0]"
        )

        assert refs == ["p2:claim:0", "p1:claim:0"]

    def test_eval_context_keeps_complete_ledger_dict_and_resolves_selected_ids(self):
        first = _evidence_item("p1:claim:0", "first")
        second = _evidence_item("p1:claim:1", "second")
        results = {
            "extract": [_extraction([first, second])],
            "adversarial_review": {
                "evidence_selection": {
                    "valid": True,
                    "section_evidence_ids": {
                        "methods": ["p1:claim:1"],
                        "introduction": ["p1:claim:0", "p1:claim:1"],
                    },
                },
            },
        }

        context = _agent()._build_eval_context(
            query="few-shot learning",
            survey="Supported [E:p1:claim:1].",
            results=results,
        )

        assert isinstance(context["evidence"], dict)
        assert list(context["evidence"]) == ["p1:claim:0", "p1:claim:1"]
        assert context["referenced_evidence_ids"] == ["p1:claim:1"]
        assert [item["evidence_id"] for item in context["selected_evidence"]] == [
            "p1:claim:1",
            "p1:claim:0",
        ]

    def test_reference_after_sixty_thousand_characters_is_resolved(self):
        items = [
            _evidence_item(f"p{i}:claim:0", f"prefix-{i}-" + ("x" * 1100))
            for i in range(60)
        ]
        tail = _evidence_item("tail:claim:0", "TAIL_SUPPORT")
        extractions = [_extraction([item]) for item in [*items, tail]]
        serialized_prefix_size = sum(len(item["text"]) for item in items)
        assert serialized_prefix_size > 60_000

        context = _agent()._build_eval_context(
            query="few-shot learning",
            survey="Tail fact [E:tail:claim:0].",
            results={"extract": extractions},
        )

        assert context["unresolved_evidence_ids"] == []
        assert context["referenced_evidence"][0]["text"] == "TAIL_SUPPORT"


class TestFaithfulnessContextContract:
    """Tests faithfulness-context resolution."""

    def test_resolver_uses_only_report_referenced_evidence(self):
        from litagent.eval.base import (
            CTX_CLAIMS,
            CTX_EVIDENCE,
            CTX_REFERENCED_EVIDENCE,
            CTX_SELECTED_EVIDENCE,
        )
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator

        referenced = _evidence_item("p1:claim:0", "referenced support")
        unrelated = _evidence_item("p2:claim:0", "unrelated support")
        evaluator = RagasFaithfulnessEvaluator(load_config())
        resolved = evaluator._resolve_contexts(
            {
                CTX_EVIDENCE: {
                    referenced["evidence_id"]: referenced,
                    unrelated["evidence_id"]: unrelated,
                },
                CTX_REFERENCED_EVIDENCE: [referenced],
                CTX_SELECTED_EVIDENCE: [unrelated],
                CTX_CLAIMS: [{"text": "must not be used"}],
            }
        )

        assert resolved.evidence_ids == ("p1:claim:0",)
        assert len(resolved.texts) == 1
        assert "referenced support" in resolved.texts[0]
        assert "must not be used" not in resolved.texts[0]
        assert resolved.fallback_used is False

    def test_resolver_falls_back_only_to_selected_evidence(self):
        from litagent.eval.base import (
            CTX_CLAIMS,
            CTX_EVIDENCE,
            CTX_REFERENCED_EVIDENCE,
            CTX_SELECTED_EVIDENCE,
        )
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator

        selected = _evidence_item("p1:claim:0", "selected")
        evaluator = RagasFaithfulnessEvaluator(load_config())
        resolved = evaluator._resolve_contexts(
            {
                CTX_EVIDENCE: {selected["evidence_id"]: selected},
                CTX_REFERENCED_EVIDENCE: [],
                CTX_SELECTED_EVIDENCE: [selected],
                CTX_CLAIMS: [{"text": "unselected claim"}],
            }
        )

        assert resolved.evidence_ids == ("p1:claim:0",)
        assert resolved.fallback_used is True
        assert resolved.fallback_reason == "report_has_no_evidence_refs"
        assert all("unselected claim" not in text for text in resolved.texts)

    @pytest.mark.asyncio
    async def test_diagnostic_uses_tail_reference_instead_of_ledger_prefix(self):
        from litagent.eval.base import (
            CTX_EVIDENCE,
            CTX_REFERENCED_EVIDENCE,
        )
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator

        prefix_items = {
            f"p{i}:claim:0": _evidence_item(
                f"p{i}:claim:0", f"prefix-{i}-" + ("x" * 1100)
            )
            for i in range(60)
        }
        tail = _evidence_item("tail:claim:0", "TAIL_SUPPORT_MUST_REACH_PROMPT")
        ledger = {**prefix_items, tail["evidence_id"]: tail}

        llm = MagicMock()
        llm.chat = AsyncMock(
            return_value=SimpleNamespace(content='{"unsupported_claims": []}')
        )
        evaluator = RagasFaithfulnessEvaluator(load_config(), llm=llm)

        result = await evaluator._diagnose(
            "Tail fact [E:tail:claim:0].",
            {
                CTX_EVIDENCE: ledger,
                CTX_REFERENCED_EVIDENCE: [tail],
            },
        )

        assert result == {"unsupported_claims": []}
        messages = llm.chat.await_args.args[0]
        prompt = "\n".join(str(message["content"]) for message in messages)
        assert "TAIL_SUPPORT_MUST_REACH_PROMPT" in prompt
        assert "prefix-0-" not in prompt


def _rewrite_report() -> dict:
    initial_evaluation = {
        "citation_accuracy": {
            "score": 1.0,
            "passed": True,
            "skipped": False,
            "details": {},
        },
        "faithfulness": {
            "score": 0.4,
            "passed": False,
            "skipped": False,
            "details": {
                "unsupported_claims": [
                    {
                        "claim_text": "old claim",
                        "evidence_ids": ["p1:claim:0"],
                        "reason_code": "insufficient_support",
                    }
                ],
            },
        },
        "internal_consistency": {
            "score": 0.9,
            "passed": True,
            "skipped": False,
            "details": {},
        },
    }
    return {
        "survey": (
            "## Introduction\nOld [E:p1:claim:0].\n\n"
            "## Methods\nOld methods [E:p1:claim:0]."
        ),
        "evaluation": initial_evaluation,
        "quality": {
            "status": "failed",
            "failed_metrics": ["faithfulness"],
            "unverified_metrics": [],
        },
        "metadata": {},
    }


class TestTransactionalRewriteContract:
    """Tests transactional evidence rewrites."""

    @staticmethod
    def _configure_agent(post_evaluation: dict) -> LitAgent:
        from litagent.runner import RewriteValidation

        agent = _agent()
        agent._synthesis = MagicMock()
        agent._synthesis.rewrite_with_evidence.return_value = [
            {"role": "user", "content": "rewrite"}
        ]
        agent._llm = MagicMock()
        agent._llm.chat = AsyncMock(
            return_value=LLMResponse(
                content=(
                    "## Introduction\nNew [E:p1:claim:0].\n\n"
                    "## Methods\nNew methods [E:p1:claim:0]."
                ),
                model="test",
            )
        )
        agent._build_rewrite_evidence = MagicMock(
            return_value=[_evidence_item("p1:claim:0", "support")]
        )
        agent._validate_rewrite_candidate = MagicMock(
            return_value=RewriteValidation(
                accepted=True,
                reason_codes=(),
                referenced_evidence_ids=("p1:claim:0",),
                unknown_evidence_ids=(),
            )
        )
        agent._evaluate = AsyncMock(return_value=post_evaluation)
        return agent

    @pytest.mark.asyncio
    async def test_failed_post_evaluation_keeps_initial_state_triplet(self):
        report = _rewrite_report()
        initial_survey = report["survey"]
        initial_evaluation = report["evaluation"]
        initial_quality = report["quality"]
        agent = self._configure_agent(initial_evaluation)

        outcome = await agent._attempt_evidence_rewrite(
            query="few-shot learning",
            report_data=report,
            results={
                "extract": [_extraction([_evidence_item("p1:claim:0", "support")])]
            },
            initial_evaluation=initial_evaluation,
            initial_quality=initial_quality,
        )

        assert outcome.committed is False
        assert report["survey"] == initial_survey
        assert report["evaluation"] is initial_evaluation
        assert report["quality"] is initial_quality
        assert outcome.candidate_evaluation == initial_evaluation

    @pytest.mark.asyncio
    async def test_passed_candidate_commits_survey_evaluation_and_quality_together(
        self,
    ):
        post_evaluation = {
            "citation_accuracy": {
                "score": 1.0,
                "passed": True,
                "skipped": False,
                "details": {},
            },
            "faithfulness": {
                "score": 0.9,
                "passed": True,
                "skipped": False,
                "details": {},
            },
            "internal_consistency": {
                "score": 0.9,
                "passed": True,
                "skipped": False,
                "details": {},
            },
        }
        report = _rewrite_report()
        initial_evaluation = report["evaluation"]
        initial_quality = report["quality"]
        agent = self._configure_agent(post_evaluation)

        outcome = await agent._attempt_evidence_rewrite(
            query="few-shot learning",
            report_data=report,
            results={
                "extract": [_extraction([_evidence_item("p1:claim:0", "support")])]
            },
            initial_evaluation=initial_evaluation,
            initial_quality=initial_quality,
        )

        assert outcome.committed is True
        assert report["survey"].startswith("## Introduction\nNew")
        assert report["evaluation"] == post_evaluation
        assert report["quality"]["status"] == "passed"
        assert outcome.candidate_quality == report["quality"]


class _StaticEvaluator:
    """Evaluator that returns a fixed result."""

    metric_name = "faithfulness"

    async def evaluate(self, survey: str, context: dict) -> EvalResult:
        return EvalResult(metric="faithfulness", score=0.9, passed=True, details={})


class _CancellingEvaluator:
    """Evaluator that propagates cancellation."""

    metric_name = "faithfulness"

    async def evaluate(self, survey: str, context: dict) -> EvalResult:
        raise asyncio.CancelledError


class TestEvaluationTraceContract:
    """Tests evaluation trace identity and lifecycle."""

    @pytest.mark.asyncio
    async def test_evaluation_phase_has_unique_identity_and_parent(self):
        events: list[tuple[str, dict]] = []
        agent = LitAgent(
            AppConfig(agent=AgentConfig(), logging=LoggingConfig()),
            trace_hook=lambda event, data: events.append((event, data)),
        )
        agent._evaluators = [_StaticEvaluator()]

        await agent._evaluate(
            query="few-shot learning",
            survey="Survey",
            results={},
            phase="post_rewrite",
            parent_task_id="evidence_repair",
        )

        assert events[0] == (
            "subspan.start",
            {
                "task_id": "evaluation.post_rewrite",
                "parent_task_id": "evidence_repair",
                "name": "evaluation.post_rewrite",
                "phase": "post_rewrite",
            },
        )
        assert events[-1][0] == "subspan.end"
        assert events[-1][1]["task_id"] == "evaluation.post_rewrite"

    @pytest.mark.asyncio
    async def test_evaluation_does_not_swallow_cancellation(self):
        agent = _agent()
        agent._evaluators = [_CancellingEvaluator()]

        with pytest.raises(asyncio.CancelledError):
            await agent._evaluate(
                query="few-shot learning",
                survey="Survey",
                results={},
                phase="initial",
            )

    def test_artifact_keeps_both_evaluation_nodes(self, tmp_path):
        recorder = RunRecorder("run-evals", "query", ArchiveRepository(tmp_path))
        for task_id in ("evaluation.initial", "evaluation.post_rewrite"):
            recorder(
                "subspan.start",
                {
                    "task_id": task_id,
                    "parent_task_id": (
                        "evidence_repair" if task_id.endswith("post_rewrite") else ""
                    ),
                    "name": task_id,
                },
            )
            recorder(
                "subspan.end",
                {
                    "task_id": task_id,
                    "output": {"faithfulness": {"score": 0.9}},
                },
            )

        nodes = recorder.snapshot()["nodes"]
        assert "subspan:evaluation.initial" in nodes
        assert "subspan:evaluation.post_rewrite" in nodes


class TestMemoryTrustContract:
    """Tests memory writes against trust state."""

    @pytest.mark.asyncio
    async def test_blocked_run_makes_no_content_memory_calls(self):
        agent = _agent()
        memory = MagicMock()
        memory.save_state = AsyncMock()
        memory.consolidate = AsyncMock()
        agent._infra.memory = memory

        result = await agent._finalize_memory(
            report_data={
                "survey": "blocked",
                "delivery": {"status": "blocked", "publishable": False},
                "metadata": {"execution": {"partial": False}},
            }
        )

        assert result["attempted"] is False
        memory.save_state.assert_not_awaited()
        memory.consolidate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_consolidate_none_is_not_reported_as_success(self):
        agent = _agent()
        memory = MagicMock()
        memory.save_state = AsyncMock()
        memory.consolidate = AsyncMock(return_value=None)
        agent._infra.memory = memory

        result = await agent._finalize_memory(
            report_data={
                "survey": "ready",
                "delivery": {"status": "ready", "publishable": True},
                "metadata": {"execution": {"partial": False}},
            }
        )

        assert result["attempted"] is True
        assert result["content_saved"] is True
        assert result["consolidated"] is False
        assert result["reason_code"] == "consolidation_returned_none"

    @pytest.mark.asyncio
    async def test_memory_finalization_does_not_swallow_cancellation(self):
        agent = _agent()
        memory = MagicMock()
        memory.save_state = AsyncMock()
        memory.consolidate = AsyncMock(side_effect=asyncio.CancelledError)
        agent._infra.memory = memory

        with pytest.raises(asyncio.CancelledError):
            await agent._finalize_memory(
                report_data={
                    "survey": "ready",
                    "delivery": {"status": "ready", "publishable": True},
                    "metadata": {"execution": {"partial": False}},
                }
            )


class _IdentityReranker:
    """Reranker that preserves input order."""

    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        scores = [0.9, 0.2, 0.8]
        for doc, score in zip(docs, scores):
            doc.score = score
        return sorted(docs, key=lambda item: item.score, reverse=True)


class TestRelevanceWorkerContract:
    """Tests relevance-worker output contracts."""

    @pytest.mark.asyncio
    async def test_worker_preserves_complete_paper_dict_and_rank_order(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        papers = [
            {"paper_id": "p0", "title": "A", "abstract": "a", "source": "arxiv"},
            {"paper_id": "p1", "title": "B", "abstract": "b", "source": "hf"},
            {"paper_id": "p2", "title": "C", "abstract": "c", "source": "s2"},
        ]
        worker = RelevanceGateWorker(
            reranker=_IdentityReranker(),
            max_papers=3,
            min_papers=0,
            cross_encoder_min_score=0.5,
            lexical_min_score=0.0,
        )
        result = await worker.execute(
            SubTask(
                task_id="relevance_gate",
                description="gate",
                agent_type="relevance_gate",
                input_data={
                    "query": "few-shot",
                    "upstream_results": {"dedup": papers},
                },
            )
        )

        assert [paper["paper_id"] for paper in result] == ["p0", "p2"]
        assert result[0]["abstract"] == "a"
        assert result[1]["source"] == "s2"
        assert all("relevance_score" in paper for paper in result)


class TestToolAndSearchDegradationContract:
    """Tests tool and search degradation contracts."""

    @pytest.mark.asyncio
    async def test_tool_executor_classifies_timeout_without_consumer_string_matching(
        self,
    ):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="slow", description="slow", timeout_ms=5, max_retries=0
            ),
            lambda: asyncio.sleep(0.1),
        )
        result = await ToolExecutor(registry).execute("slow", {})

        assert result.error_code == "tool_timeout"
        assert result.error_type == "TimeoutError"

    @pytest.mark.asyncio
    async def test_tool_executor_classifies_http_429(self):
        request = httpx.Request("GET", "https://example.test")
        response = httpx.Response(429, request=request)

        async def rate_limited():
            raise httpx.HTTPStatusError(
                "rate limited", request=request, response=response
            )

        registry = ToolRegistry()
        registry.register(
            ToolDefinition(name="remote", description="remote", max_retries=0),
            rate_limited,
        )
        result = await ToolExecutor(registry).execute("remote", {})

        assert result.error_code == "http_rate_limited"
        assert result.error_type == "HTTPStatusError"

    @pytest.mark.asyncio
    async def test_search_error_is_not_classified_as_empty(self):
        from litagent.agents.search import SearchSourceStatus, SearchWorker

        tool_result = ToolResult(
            name="search_arxiv",
            args={},
            output=None,
            error="Timeout after 30000ms",
            error_code="tool_timeout",
            error_type="TimeoutError",
        )
        executor = MagicMock()
        executor.execute = AsyncMock(return_value=tool_result)
        worker = SearchWorker(executor=executor)

        papers = await worker.execute(
            SubTask(
                task_id="search_arxiv_q0",
                description="search",
                agent_type="search",
                input_data={"source": "arxiv", "query": "few-shot"},
            )
        )

        outcome = worker.get_source_outcomes()[0]
        assert papers == []
        assert outcome.status is SearchSourceStatus.TIMEOUT
        assert outcome.reason_code == "provider_timeout"

    @pytest.mark.asyncio
    async def test_unknown_source_does_not_silently_call_arxiv(self):
        from litagent.agents.search import SearchSourceStatus, SearchWorker

        executor = MagicMock()
        executor.execute = AsyncMock()
        worker = SearchWorker(executor=executor)

        papers = await worker.execute(
            SubTask(
                task_id="search_unknown_q0",
                description="search",
                agent_type="search",
                input_data={"source": "unknown", "query": "few-shot"},
            )
        )

        assert papers == []
        executor.execute.assert_not_awaited()
        outcome = worker.get_source_outcomes()[0]
        assert outcome.status is SearchSourceStatus.FAILED
        assert outcome.reason_code == "invalid_search_source"

    @pytest.mark.asyncio
    async def test_search_outcomes_can_be_reset_between_runs(self):
        from litagent.agents.search import SearchWorker

        executor = MagicMock()
        executor.execute = AsyncMock()
        worker = SearchWorker(executor=executor)

        await worker.execute(
            SubTask(
                task_id="search_unknown_q0",
                description="search",
                agent_type="search",
                input_data={"source": "unknown", "query": "few-shot"},
            )
        )
        assert len(worker.get_source_outcomes()) == 1

        worker.reset_source_outcomes()

        assert worker.get_source_outcomes() == ()

    def test_all_sources_unavailable_maps_delivery_to_needs_review(self):
        delivery = derive_delivery(
            partial=False,
            quality={
                "status": "passed",
                "failed_metrics": [],
                "unverified_metrics": [],
            },
            degradation_reason_codes=["all_external_sources_unavailable"],
        )

        assert delivery["status"] == "needs_review"
        assert delivery["publishable"] is False
        assert "all_external_sources_unavailable" in delivery["reason_codes"]

    def test_runner_aggregates_all_unavailable_sources(self):
        from litagent.agents.search import SearchSourceOutcome, SearchSourceStatus

        outcomes = [
            SearchSourceOutcome(
                task_id="search_arxiv_q0",
                source="arxiv",
                status=SearchSourceStatus.TIMEOUT,
                result_count=0,
                elapsed_ms=30_000,
                reason_code="provider_timeout",
                error_type="TimeoutError",
            ),
            SearchSourceOutcome(
                task_id="search_s2_q0",
                source="semantic_scholar",
                status=SearchSourceStatus.RATE_LIMITED,
                result_count=0,
                elapsed_ms=200,
                reason_code="provider_rate_limited",
                error_type="HTTPStatusError",
            ),
        ]

        serialized, reasons = LitAgent._summarize_search_outcomes(outcomes)

        assert [item["status"] for item in serialized] == [
            "timeout",
            "rate_limited",
        ]
        assert "all_external_sources_unavailable" in reasons


class TestPhase139Configuration:
    """Tests Phase 13.9 configuration boundaries."""

    def test_lexical_threshold_matches_actual_zero_to_three_score_range(self):
        from litagent.config import RelevanceConfig

        config = RelevanceConfig(lexical_min_score=2.5)
        assert config.lexical_min_score == 2.5

    def test_relevance_minimum_cannot_exceed_maximum(self):
        from pydantic import ValidationError

        from litagent.config import RelevanceConfig

        with pytest.raises(ValidationError):
            RelevanceConfig(min_papers=20, max_papers=10)


class TestNonBlockingModelInitialization:
    """Tests non-blocking model initialization."""

    @pytest.mark.asyncio
    async def test_reranker_load_does_not_block_event_loop(self, monkeypatch):
        agent = _agent()

        def slow_create():
            time.sleep(0.05)
            return "reranker"

        monkeypatch.setattr(agent, "_create_reranker", slow_create)
        ticked = asyncio.Event()

        async def ticker():
            await asyncio.sleep(0.005)
            ticked.set()

        result, _ = await asyncio.gather(
            agent._create_reranker_async(),
            ticker(),
        )

        assert ticked.is_set()
        assert result == "reranker"
