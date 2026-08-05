"""Tests for application assembly, runner contracts, and delivery policy."""

from __future__ import annotations

import asyncio
import json
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import (
    AdversarialConfig,
    AgentConfig,
    AppConfig,
    ContextConfig,
    ExtractorConfig,
    LLMConfig,
    LoggingConfig,
    MemoryConfig,
    ObservabilityConfig,
    OrchestratorConfig,
    ResilienceConfig,
    SafetyConfig,
)
from litagent.llm.client import BaseLLMClient, LLMResponse, MockLLMClient
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.working import WorkingMemory
from litagent.observability.recorder import RedactingTraceHook
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.rag.claims_index import ClaimsIndex
from litagent.rag.interfaces import Reranker, ScoredDoc, VectorStore
from litagent.rag.retriever import HybridRetriever
from litagent.runner import Infra, LitAgent


def _minimal_config(**overrides) -> AppConfig:
    """Build a minimal application configuration for isolated tests."""
    return AppConfig(
        agent=AgentConfig(max_loops=3),
        logging=LoggingConfig(level="WARNING"),
        memory=MemoryConfig(
            redis_url=overrides.pop("redis_url", "redis://localhost:9999"),
            qdrant_url=overrides.pop("qdrant_url", "http://localhost:9999"),
            pg_url=overrides.pop(
                "pg_url", "postgresql://none:none@localhost:9999/none"
            ),
        ),
        context=ContextConfig(),
        orchestrator=OrchestratorConfig(timeout_ms=30000, max_concurrent=3),
        llm=LLMConfig(base_url="https://api.deepseek.com", model="deepseek-v4-flash"),
        adversarial=AdversarialConfig(max_rounds=1, pass_threshold=0.5),
        safety=SafetyConfig(max_cost_tokens=100000),
        resilience=ResilienceConfig(cb_fail_threshold=3, cb_cooldown_seconds=10),
        extractor=overrides.pop("extractor", ExtractorConfig(max_concurrent=2)),
        **overrides,
    )


def _minimal_agent() -> LitAgent:
    """Return an unwired agent for pure runner contract tests."""
    return LitAgent(_minimal_config())


class TestAdversarialReviewWorker:
    """Tests adversarial worker dependency injection."""

    def test_accepts_injected_workers(self):
        """The adversarial worker accepts injected collaborators."""
        llm = MockLLMClient(["draft text"])
        synthesis = SynthesisWorker(llm)
        reviewer = ReviewerWorker(llm)

        worker = AdversarialReviewWorker(
            llm=llm,
            synthesis=synthesis,
            reviewer=reviewer,
            max_rounds=2,
            pass_threshold=0.7,
        )
        assert worker._synthesis is synthesis
        assert worker._reviewer is reviewer
        assert worker._max_rounds == 2
        assert worker._pass_threshold == 0.7

    def test_agent_type(self):
        llm = MockLLMClient()
        worker = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm),
            reviewer=ReviewerWorker(llm),
        )
        assert worker.agent_type == "adversarial_review"


class TestBackendClose:
    """Tests backend resource cleanup."""

    @pytest.mark.asyncio
    async def test_working_memory_close(self):
        """Working-memory cleanup closes its Redis client."""
        redis = MagicMock()
        redis.aclose = AsyncMock()
        wm = WorkingMemory(redis, MemoryConfig())
        await wm.close()
        redis.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_episodic_memory_close(self):
        """Episodic-memory cleanup closes its database pool."""
        client = MagicMock()
        client.close = AsyncMock()
        em = EpisodicMemory(client)
        await em.close()
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_semantic_memory_close(self):
        """Semantic-memory cleanup closes its vector store."""
        pool = MagicMock()
        pool.close = AsyncMock()
        sm = SemanticMemory(pool)
        await sm.close()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_procedural_memory_close(self):
        """Procedural-memory cleanup closes its database pool."""
        pool = MagicMock()
        pool.close = AsyncMock()
        pm = ProceduralMemory(pool)
        await pm.close()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_claims_index_close(self):
        """Claims-index cleanup closes its vector store."""
        client = MagicMock()
        client.close = AsyncMock()
        ci = ClaimsIndex(client)
        await ci.close()
        client.close.assert_awaited_once()


class TestLitAgentWiring:
    """Tests LitAgent dependency wiring."""

    @pytest.mark.asyncio
    async def test_wires_without_infra(self):
        """Wiring degrades cleanly when infrastructure is unavailable."""
        config = _minimal_config()
        agent = LitAgent(config)

        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm = MockLLMClient(["test"])
            mock_llm_cls.return_value = mock_llm

            await agent._wire()

        assert agent._llm is not None
        assert agent._executor is not None
        assert agent._scheduler is not None
        assert agent._planner is not None
        # Unreachable test endpoints must disable optional infrastructure.
        assert agent._infra.memory is None
        assert agent._infra.claims_index is None
        assert agent._infra.retriever is None
        assert agent._wired is True

        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_all_workers_registered(self):
        """Wiring registers every required worker."""
        config = _minimal_config()
        agent = LitAgent(config)

        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = MockLLMClient(["test"])
            await agent._wire()

        assert agent._search is not None
        assert agent._dedup is not None
        assert agent._extractor is not None
        assert agent._graph is not None
        assert agent._synthesis is not None
        assert agent._reviewer is not None
        assert agent._adversarial is not None
        assert not hasattr(agent, "_report")

        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_extractor_can_disable_llm_strategy(self):
        from litagent.agents.extraction_strategy import RegexStrategy

        config = _minimal_config(
            extractor=ExtractorConfig(max_concurrent=2, enable_llm=False)
        )
        agent = LitAgent(config)
        with (
            patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls,
            patch.object(agent, "_connect_infra", new=AsyncMock(return_value=Infra())),
        ):
            mock_llm_cls.return_value = MockLLMClient(["test"])
            await agent._wire()

        assert isinstance(agent._extraction_strategy, RegexStrategy)
        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_automatic_langfuse_uses_redacting_boundary(self):
        config = _minimal_config(
            observability=ObservabilityConfig(
                enabled=True, payload_mode="full_redacted"
            )
        )
        agent = LitAgent(config)
        with (
            patch("litagent.runner.LangFuseTracer") as tracer_cls,
            patch("litagent.runner.OpenAICompatibleClient") as llm_cls,
            patch.object(agent, "_connect_infra", new=AsyncMock(return_value=Infra())),
        ):
            llm_cls.return_value = MockLLMClient(["test"])
            await agent._wire()

        assert isinstance(agent._trace_hook, RedactingTraceHook)
        assert agent._trace_hook._sink is tracer_cls.return_value
        assert agent._trace_hook._payload_mode == "full_redacted"
        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """The async context manager wires and closes the agent."""
        config = _minimal_config()
        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = MockLLMClient(["test"])
            async with LitAgent(config) as agent:
                assert agent._wired is True
            assert agent._wired is False

    @pytest.mark.asyncio
    async def test_wire_idempotent(self):
        """Repeated wiring is idempotent."""
        config = _minimal_config()
        agent = LitAgent(config)

        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = MockLLMClient(["test"])
            await agent._wire()
            await agent._wire()

        assert agent._wired is True
        await agent.cleanup()


class StubLLMClient(BaseLLMClient):
    """LLM client that returns prompt-specific deterministic responses."""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._call_count = 0
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        self._call_count += 1
        self.calls.append(messages)
        content = self._responses.get(
            "default",
            '{"draft": "A survey draft about the query.", "score": 0.9, '
            '"verdict": "accept", "weaknesses": [], "issues": []}',
        )
        return LLMResponse(content=content, model="stub")


class TestLitAgentRun:
    """Tests end-to-end runner orchestration with test doubles."""

    @pytest.mark.asyncio
    async def test_minimal_run_returns_report(self):
        """A minimal run returns the normalized report contract."""
        config = _minimal_config()
        agent = LitAgent(config)
        stub_llm = StubLLMClient()

        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = stub_llm

            await agent._wire()

            async def fake_search(task):
                return [{"paper_id": "p1", "title": "Test Paper", "abstract": "test"}]

            async def fake_extract(task):
                return [{"claims": ["claim 1"], "paper_id": "p1"}]

            async def fake_graph(task):
                return {"nodes": 1, "edges": 0}

            agent._search.execute = fake_search
            agent._extractor.execute = fake_extract
            agent._graph.execute = fake_graph

            report = await agent.run("test query")

        assert "survey" in report
        assert "metadata" in report
        assert report["metadata"]["query"] == "test query"
        assert "partial" in report

        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_run_without_wire_calls_wire(self):
        """Running an unwired agent wires it automatically."""
        config = _minimal_config()
        agent = LitAgent(config)
        stub_llm = StubLLMClient()

        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = stub_llm
            await agent._wire()

            async def fake_search(task):
                return []

            async def fake_extract(task):
                return []

            async def fake_graph(task):
                return {}

            agent._search.execute = fake_search
            agent._extractor.execute = fake_extract
            agent._graph.execute = fake_graph

            report = await agent.run("auto wire test")

        assert agent._wired is True
        assert "survey" in report

        await agent.cleanup()


class TestExtractReport:
    """Tests final report extraction and normalization."""

    def test_builds_report_from_adversarial_output(self):
        """The runner owns final report assembly and normalizes review history."""
        config = _minimal_config()
        agent = LitAgent(config)
        results = {
            "adversarial_review": {
                "final_draft": "adversarial draft",
                "total_rounds": 2,
                "final_score": 0.7,
                "accepted": False,
                "rounds": [
                    {
                        "round": 2,
                        "review": {
                            "score": 0.7,
                            "verdict": "revise",
                            "weaknesses": ["missing baseline"],
                            "issues": [{"section": "Methods", "issue": "thin"}],
                        },
                    }
                ],
            },
            "graph_analysis": {
                "papers": [{"title": "Paper A", "tier": 1}],
                "tier_counts": {"tier1": 1, "tier2": 0, "tier3": 0},
                "seminal_papers": [{"title": "Paper A", "tier": 1}],
            },
        }
        out = agent._extract_report(results, "test query")
        assert out["survey"] == "adversarial draft"
        assert out["metadata"]["accepted"] is False
        assert isinstance(out["metadata"]["generated_at"], float)
        assert out["review_history"] == [
            {
                "round": 2,
                "review": {
                    "score": 0.7,
                    "verdict": "revise",
                    "weaknesses": ["missing baseline"],
                    "issues": [{"section": "Methods", "issue": "thin"}],
                },
            }
        ]
        assert "Survey incomplete" not in out["survey"]

    def test_preserves_graph_analysis_contract(self):
        """Report extraction preserves the graph-analysis contract."""
        config = _minimal_config()
        agent = LitAgent(config)
        graph_output = {
            "papers": [
                {"title": "Paper A", "tier": 1},
                {"title": "Paper B", "tier": 2},
            ],
            "tier_counts": {"tier1": 1, "tier2": 1, "tier3": 0},
            "seminal_papers": [{"title": "Paper A", "tier": 1}],
        }
        results = {
            "graph_analysis": graph_output,
            "adversarial_review": {
                "final_draft": "x",
                "nodes": 999,
            },
        }
        out = agent._extract_report(results, "test query")

        assert out["graph_data"] == graph_output
        assert "papers" in out["graph_data"]
        assert "tier_counts" in out["graph_data"]
        assert "seminal_papers" in out["graph_data"]

        assert "nodes" not in out["graph_data"]


class TestRunnerEnsureTables:
    """Tests database table initialization during wiring."""

    @pytest.mark.asyncio
    async def test_ensure_tables_called_when_pg_available(self):
        from unittest.mock import AsyncMock, patch

        import asyncpg

        config = _minimal_config()
        agent = LitAgent(config)
        mock_pool = MagicMock(spec=asyncpg.Pool)
        ensure_tables = AsyncMock()
        with (
            patch(
                "litagent.runner.WorkingMemory.connect",
                new=AsyncMock(side_effect=RuntimeError("redis unavailable")),
            ),
            patch(
                "qdrant_client.AsyncQdrantClient",
                side_effect=RuntimeError("qdrant unavailable"),
            ),
            patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)),
            patch.object(ProceduralMemory, "ensure_tables", new=ensure_tables),
        ):
            await agent._connect_infra(config)

        ensure_tables.assert_awaited_once()


class TestCLI:
    """Tests CLI output and validation."""

    def test_config_validate_ok(self):
        """Configuration validation reports success."""
        import argparse

        from litagent.cli import _cmd_config

        ns = argparse.Namespace(config=None, validate_only=True)
        with pytest.raises(SystemExit) as exc:
            _cmd_config(ns)
        assert exc.value.code == 0

    def test_tools_json_output(self):
        """Tool listing produces valid JSON."""
        import argparse

        from litagent.cli import _cmd_tools

        old_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            _cmd_tools(argparse.Namespace(format="json"))
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        data = json.loads(output)
        assert isinstance(data, list)
        tool_names = {t["name"] for t in data}
        assert "search_arxiv" in tool_names
        assert "extract_claims" in tool_names

    def test_cli_help(self):
        """The CLI exposes help output."""
        import argparse

        from litagent.cli import main

        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["litagent", "--help"]):
                main()
        assert exc.value.code == 0


class TestQualityGate:
    """Tests quality-gate state derivation."""

    def test_quality_failed_when_citation_fails(self):
        from litagent.runner import LitAgent

        ev = {
            "citation_accuracy": {"passed": False, "skipped": False},
            "faithfulness": {"passed": True, "skipped": False},
        }
        q = LitAgent._derive_quality(ev)
        assert q["status"] == "failed"
        assert "citation_accuracy" in q["failed_metrics"]

    def test_quality_unverified_when_skipped(self):
        from litagent.runner import LitAgent

        ev = {
            "citation_accuracy": {"passed": True, "skipped": True},
            "faithfulness": {"passed": True, "skipped": False},
        }
        q = LitAgent._derive_quality(ev)
        assert q["status"] == "unverified"

    def test_quality_passed_only_when_both_pass(self):
        from litagent.runner import LitAgent

        ev = {
            "citation_accuracy": {"passed": True, "skipped": False},
            "faithfulness": {"passed": True, "skipped": False},
        }
        q = LitAgent._derive_quality(ev)
        assert q["status"] == "passed"
        assert q["failed_metrics"] == []

    def test_partial_remains_execution_interruption_only(self):
        """Quality failures do not redefine execution completeness."""
        from litagent.runner import LitAgent

        q = LitAgent._derive_quality({})
        assert q["status"] == "unverified"


class TestExecutionSummary:
    """Tests execution-summary derivation."""

    def test_failed_task_marks_report_partial(self):
        graph = TaskGraph()
        graph.add_task(
            SubTask(task_id="extract", description="e", agent_type="extractor")
        )
        graph.mark_failed("extract", "worker_timeout")

        execution = LitAgent._derive_execution(
            graph, budget_exceeded=False, final_output_present=False
        )

        assert execution["partial"] is True
        assert execution["status"] == "incomplete"
        assert execution["failed_task_ids"] == ["extract"]
        assert execution["reason_codes"] == ["task_failed", "final_output_missing"]

    def test_quality_does_not_affect_execution_completeness(self):
        graph = TaskGraph()
        graph.add_task(
            SubTask(
                task_id="adversarial_review",
                description="a",
                agent_type="adversarial_review",
            )
        )
        graph.mark_done("adversarial_review", {"final_draft": "draft"})

        execution = LitAgent._derive_execution(
            graph, budget_exceeded=False, final_output_present=True
        )

        assert execution["partial"] is False
        assert execution["status"] == "complete"
        assert execution["reason_codes"] == []


class TestDeriveDelivery:
    """Tests delivery-state derivation."""

    def test_partial_wins_over_quality(self):
        from litagent.runner import derive_delivery

        d = derive_delivery(True, {"status": "failed"})
        assert d["status"] == "partial"
        assert d["publishable"] is False
        assert "partial_execution" in d["reason_codes"]
        assert "quality_failed" in d["reason_codes"]

    def test_quality_failed_blocks(self):
        from litagent.runner import derive_delivery

        d = derive_delivery(False, {"status": "failed"})
        assert d["status"] == "blocked"
        assert d["publishable"] is False
        assert d["reason_codes"] == ["quality_failed"]

    def test_quality_unverified_needs_review(self):
        from litagent.runner import derive_delivery

        d = derive_delivery(False, {"status": "unverified"})
        assert d["status"] == "needs_review"
        assert d["publishable"] is False
        assert d["reason_codes"] == ["quality_unverified"]

    def test_passed_and_complete_is_ready(self):
        from litagent.runner import derive_delivery

        d = derive_delivery(False, {"status": "passed"})
        assert d["status"] == "ready"
        assert d["publishable"] is True
        assert d["reason_codes"] == []

    def test_missing_quality_treated_as_unverified(self):
        from litagent.runner import derive_delivery

        d = derive_delivery(False, None)
        assert d["status"] == "needs_review"

    def test_unknown_quality_status_fails_closed(self):
        from litagent.runner import derive_delivery

        d = derive_delivery(False, {"status": "pass"})

        assert d["status"] == "needs_review"
        assert d["publishable"] is False
        assert d["reason_codes"] == ["quality_invalid"]


class TestCLIDeliveryContract:
    """Tests CLI behavior for each delivery state."""

    @staticmethod
    def _report(delivery_status, publishable, **extra):
        return {
            "survey": "survey body",
            "metadata": {"query": "q"},
            "review_history": [],
            "partial": extra.pop("partial", False),
            "quality": extra.pop(
                "quality",
                {"status": "passed", "failed_metrics": [], "unverified_metrics": []},
            ),
            "delivery": {
                "status": delivery_status,
                "publishable": publishable,
                "reason_codes": extra.pop("reason_codes", []),
            },
        }

    def test_exit_code_zero_when_ready(self):
        from litagent.cli import _delivery_exit_code

        assert _delivery_exit_code(self._report("ready", True)) == 0

    def test_exit_code_nonzero_when_blocked(self):
        from litagent.cli import _delivery_exit_code

        assert _delivery_exit_code(self._report("blocked", False)) != 0

    def test_exit_code_nonzero_when_needs_review(self):
        from litagent.cli import _delivery_exit_code

        assert _delivery_exit_code(self._report("needs_review", False)) != 0

    def test_exit_code_derives_for_legacy_report(self):
        """Legacy reports derive an exit code from normalized delivery."""
        from litagent.cli import _delivery_exit_code

        legacy = {
            "survey": "x",
            "partial": False,
            "quality": {
                "status": "failed",
                "failed_metrics": ["faithfulness"],
                "unverified_metrics": [],
            },
        }
        assert _delivery_exit_code(legacy) != 0

    def test_banner_blocked(self):
        from litagent.cli import _format_report_markdown

        report = self._report(
            "blocked",
            False,
            quality={
                "status": "failed",
                "failed_metrics": ["faithfulness"],
                "unverified_metrics": [],
            },
        )
        out = _format_report_markdown(report)
        assert "NOT PUBLISHABLE" in out
        assert "faithfulness" in out
        assert "survey body" in out

    def test_banner_needs_review(self):
        from litagent.cli import _format_report_markdown

        out = _format_report_markdown(
            self._report(
                "needs_review",
                False,
                quality={
                    "status": "unverified",
                    "failed_metrics": [],
                    "unverified_metrics": ["faithfulness"],
                },
            )
        )
        assert "Quality unverified" in out

    def test_banner_partial(self):
        from litagent.cli import _format_report_markdown

        out = _format_report_markdown(self._report("partial", False, partial=True))
        assert "Partial results" in out

    def test_no_banner_when_ready(self):
        from litagent.cli import _format_report_markdown

        out = _format_report_markdown(self._report("ready", True))
        assert "⚠" not in out


class TestRelevanceGateWiring:
    """Tests relevance-gate dependency wiring."""

    @pytest.mark.asyncio
    async def test_reranker_shared_with_retriever_and_gate(self):
        config = _minimal_config()
        agent = LitAgent(config)
        fake_reranker = MagicMock(spec=Reranker)
        fake_store = MagicMock(spec=VectorStore)
        infra = Infra(
            reranker=fake_reranker,
            retriever=HybridRetriever(fake_store, fake_reranker),
        )
        with (
            patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls,
            patch.object(agent, "_connect_infra", new=AsyncMock(return_value=infra)),
        ):
            mock_llm_cls.return_value = MockLLMClient(["test"])
            await agent._wire()
        assert agent._infra.reranker is fake_reranker
        assert agent._infra.retriever._reranker is fake_reranker
        assert agent._relevance_gate._reranker is fake_reranker
        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_wiring_succeeds_when_reranker_fails(self):
        config = _minimal_config()
        agent = LitAgent(config)
        with patch(
            "litagent.runner.CrossEncoderReranker",
            side_effect=RuntimeError("model load failed"),
        ):
            assert agent._create_reranker() is None

        with (
            patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls,
            patch.object(agent, "_connect_infra", new=AsyncMock(return_value=Infra())),
        ):
            mock_llm_cls.return_value = MockLLMClient(["test"])
            await agent._wire()
        assert agent._relevance_gate is not None
        assert agent._relevance_gate._reranker is None
        assert agent._wired is True
        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_retriever_without_reranker_returns_vector_top_k(self):
        from langchain_core.documents import Document

        store = MagicMock(spec=VectorStore)
        store.search = AsyncMock(
            return_value=[
                ScoredDoc(Document(page_content=f"doc-{index}"), float(index))
                for index in range(4)
            ]
        )
        retriever = HybridRetriever(store, reranker=None)
        result = await retriever.search("query", top_k=2)
        assert [item.doc.page_content for item in result] == ["doc-0", "doc-1"]


class TestExtractorConfig:
    """Tests extractor configuration boundaries."""

    def test_extractor_config_defaults(self):
        from litagent.config import ExtractorConfig

        cfg = ExtractorConfig()
        assert cfg.max_papers == 50
        assert cfg.per_paper_timeout_ms == 20000

    def test_extractor_config_rejects_zero_max_papers(self):
        from litagent.config import ExtractorConfig

        with pytest.raises(Exception):
            ExtractorConfig(max_papers=0)

    def test_per_paper_timeout_rejects_negative(self):
        from litagent.config import ExtractorConfig

        with pytest.raises(Exception):
            ExtractorConfig(per_paper_timeout_ms=-1)


class TestEvalContextAlignment:
    """Tests evaluation-context evidence alignment."""

    def _make_results(self, extractions=None):
        return {
            "extract": extractions
            or [
                {
                    "paper_id": "p1",
                    "title": "T1",
                    "claims": ["claim A"],
                    "evidence_items": [
                        {
                            "evidence_id": "p1:claim:0",
                            "paper_id": "p1",
                            "paper_title": "T1",
                            "text": "evidence A",
                        },
                        {
                            "evidence_id": "p1:claim:1",
                            "paper_id": "p1",
                            "paper_title": "T1",
                            "text": "evidence B",
                        },
                    ],
                },
                {
                    "paper_id": "p2",
                    "title": "T2",
                    "claims": ["claim C"],
                    "evidence_items": [
                        {
                            "evidence_id": "p2:claim:0",
                            "paper_id": "p2",
                            "paper_title": "T2",
                            "text": "evidence C",
                        },
                    ],
                },
            ]
        }

    def test_referenced_evidence_resolves_correctly(self):
        agent = _minimal_agent()
        ctx = agent._build_eval_context(
            query="few-shot learning",
            survey="claim [E:p1:claim:0] and [E:p2:claim:0]",
            results=self._make_results(),
        )
        assert len(ctx["referenced_evidence"]) == 2
        assert ctx["unresolved_evidence_ids"] == []

    def test_unknown_ref_preserved_as_unresolved(self):
        agent = _minimal_agent()
        ctx = agent._build_eval_context(
            query="few-shot learning",
            survey="claim [E:p1:claim:0] and [E:p99:claim:99]",
            results=self._make_results(),
        )
        assert len(ctx["referenced_evidence"]) == 1
        assert "p99:claim:99" in ctx["unresolved_evidence_ids"]

    def test_empty_survey_returns_empty_referenced(self):
        agent = _minimal_agent()
        ctx = agent._build_eval_context(
            query="few-shot learning",
            survey="",
            results=self._make_results(),
        )
        assert ctx["referenced_evidence_ids"] == []

    def test_tail_evidence_not_lost_by_prefix_truncation(self):
        """Evidence referenced near the report tail remains resolvable."""
        extractions = []
        for i in range(70):
            eid = f"p{i}:claim:0"
            evidence_text = f"evidence {i} " + ("x" * 1000)
            extractions.append(
                {
                    "paper_id": f"p{i}",
                    "title": f"T{i}",
                    "claims": [f"c{i}"],
                    "evidence_items": [
                        {
                            "evidence_id": eid,
                            "paper_id": f"p{i}",
                            "paper_title": f"T{i}",
                            "text": evidence_text,
                        },
                    ],
                }
            )
        agent = _minimal_agent()
        ctx = agent._build_eval_context(
            query="few-shot learning",
            survey="claim [E:p69:claim:0]",
            results={"extract": extractions},
        )
        assert len(ctx["referenced_evidence"]) == 1
        assert ctx["unresolved_evidence_ids"] == []
        assert ctx["referenced_evidence"][0]["text"].startswith("evidence 69")
        assert (
            sum(len(item["evidence_items"][0]["text"]) for item in extractions[:-1])
            > 60_000
        )


class TestRewriteValidation:
    """Tests validation of evidence-based rewrites."""

    def _make_agent(self):
        from litagent.config import AgentConfig, AppConfig, LoggingConfig
        from litagent.runner import LitAgent

        return LitAgent(AppConfig(agent=AgentConfig(), logging=LoggingConfig()))

    def _ledger(self):
        item = {"evidence_id": "p1:claim:0", "paper_title": "T", "text": "ev"}
        return {item["evidence_id"]: item}

    @staticmethod
    def _original():
        return (
            "## Introduction\nOriginal intro [E:p1:claim:0].\n\n"
            "## Methods\nOriginal methods [E:p1:claim:0].\n\n"
            "## Open Problems\nOriginal limitations [E:p1:claim:0]."
        )

    def test_rejects_empty_candidate(self):
        agent = self._make_agent()
        v = agent._validate_rewrite_candidate(
            original=self._original(),
            candidate="",
            evidence_ledger=self._ledger(),
        )
        assert v.accepted is False
        assert "empty_candidate" in v.reason_codes

    def test_rejects_placeholder_text(self):
        agent = self._make_agent()
        v = agent._validate_rewrite_candidate(
            original=self._original(),
            evidence_ledger=self._ledger(),
            candidate="[No changes to this section.]\n## Intro\nText",
        )
        assert "placeholder_detected" in v.reason_codes

    def test_rejects_ellipsis_placeholder(self):
        agent = self._make_agent()
        v = agent._validate_rewrite_candidate(
            original=self._original(),
            evidence_ledger=self._ledger(),
            candidate="## Intro\nReal\n\n...\n\n## Next",
        )
        assert "ellipsis_placeholder" in v.reason_codes

    def test_rejects_unknown_evidence_refs(self):
        agent = self._make_agent()
        v = agent._validate_rewrite_candidate(
            original=self._original(),
            evidence_ledger=self._ledger(),
            candidate="Good [E:p1:claim:0]. Bad [E:fake:id].",
        )
        assert "unknown_evidence_refs" in v.reason_codes
        assert "fake:id" in v.unknown_evidence_ids

    def test_accepts_valid_candidate(self):
        agent = self._make_agent()
        v = agent._validate_rewrite_candidate(
            original=self._original(),
            evidence_ledger=self._ledger(),
            candidate=(
                "## Introduction\nImproved intro [E:p1:claim:0].\n\n"
                "## Methods\nImproved methods [E:p1:claim:0].\n\n"
                "## Open Problems\nImproved limitations [E:p1:claim:0]."
            ),
        )
        assert v.accepted is True

    def test_accepts_document_title_without_direct_body(self):
        agent = self._make_agent()
        original = (
            "# Survey Title\n\n"
            "## Introduction\nOriginal intro [E:p1:claim:0].\n\n"
            "## Methods\nOriginal methods [E:p1:claim:0]."
        )
        candidate = (
            "# Survey Title\n\n"
            "## Introduction\nImproved intro [E:p1:claim:0].\n\n"
            "## Methods\nImproved methods [E:p1:claim:0]."
        )

        v = agent._validate_rewrite_candidate(
            original=original,
            candidate=candidate,
            evidence_ledger=self._ledger(),
        )

        assert v.accepted is True
        assert "empty_section_detected" not in v.reason_codes

    def test_rejects_empty_content_section(self):
        agent = self._make_agent()
        original = (
            "# Survey Title\n\n"
            "## Introduction\nOriginal intro [E:p1:claim:0].\n\n"
            "## Methods\nOriginal methods [E:p1:claim:0]."
        )
        candidate = (
            "# Survey Title\n\n"
            "## Introduction\nImproved intro [E:p1:claim:0].\n\n"
            "## Methods\n"
        )

        v = agent._validate_rewrite_candidate(
            original=original,
            candidate=candidate,
            evidence_ledger=self._ledger(),
        )

        assert v.accepted is False
        assert "empty_section_detected" in v.reason_codes

    def test_rejects_missing_section(self):
        agent = self._make_agent()
        v = agent._validate_rewrite_candidate(
            original=self._original(),
            evidence_ledger=self._ledger(),
            candidate=(
                "## Introduction\nImproved intro [E:p1:claim:0].\n\n"
                "## Open Problems\nImproved limitations [E:p1:claim:0]."
            ),
        )
        assert v.accepted is False
        assert any("missing_section" in rc for rc in v.reason_codes)


class TestMemoryTrustOrdering:
    """Tests memory writes against delivery trust state."""

    @pytest.mark.asyncio
    async def test_blocked_delivery_skips_consolidation(self):
        agent = _minimal_agent()
        agent._infra.memory = MagicMock()
        result = await agent._finalize_memory(
            report_data={
                "delivery": {"status": "blocked", "publishable": False},
                "metadata": {"execution": {"partial": False}},
                "survey": "test",
            },
        )
        assert not result["content_saved"]

    @pytest.mark.asyncio
    async def test_partial_skips_even_if_publishable(self):
        agent = _minimal_agent()
        agent._infra.memory = MagicMock()
        result = await agent._finalize_memory(
            report_data={
                "delivery": {"status": "ready", "publishable": True},
                "metadata": {"execution": {"partial": True}},
                "survey": "test",
            },
        )
        assert not result["content_saved"]

    @pytest.mark.asyncio
    async def test_publishable_saves_and_consolidates(self):
        agent = _minimal_agent()
        agent._infra.memory = MagicMock()
        agent._infra.memory.save_state = AsyncMock()
        agent._infra.memory.consolidate = AsyncMock()
        result = await agent._finalize_memory(
            report_data={
                "partial": False,
                "quality": {"status": "passed"},
                "delivery": {"status": "ready", "publishable": True},
                "metadata": {"execution": {"partial": False}},
                "survey": "test",
            },
        )
        assert result["content_saved"] is True
        assert result["consolidated"] is True

    @pytest.mark.asyncio
    async def test_invalid_quality_never_reaches_memory(self):
        agent = _minimal_agent()
        agent._infra.memory = MagicMock()

        result = await agent._finalize_memory(
            report_data={
                "partial": False,
                "quality": {"status": "pass"},
                "delivery": {"status": "ready", "publishable": True},
                "metadata": {"execution": {"partial": False}},
                "survey": "test",
            }
        )

        assert result["content_saved"] is False
        assert result["reason_code"] == "content_consolidation_skipped"
