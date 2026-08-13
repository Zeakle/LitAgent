"""Tests for LitAgent worker behavior and contracts."""

import asyncio
import json
from typing import Any

import pytest

from litagent.agents.dedup import DedupWorker
from litagent.agents.extractor import ExtractorWorker
from litagent.agents.graph import GraphWorker
from litagent.agents.search import SearchWorker
from litagent.orchestrator.task_graph import SubTask
from litagent.tools.executor import ToolExecutor, ToolResult
from litagent.tools.registry import ToolRegistry


def _mock_executor() -> ToolExecutor:
    """Return a tool executor backed by an isolated registry."""
    return ToolExecutor(ToolRegistry(), allowed_names=set())


class TestInjectedBuiltinToolRegistry:
    """Tests that built-in tools can be scoped to one agent run."""

    def test_search_and_regex_extraction_use_declared_allowed_tools(self):
        from litagent.tools.builtin.extract import register_extract_tools
        from litagent.tools.builtin.search import register_search_tools
        from litagent.tools.registry import get_registry, reset_registry

        reset_registry()
        registry = ToolRegistry()
        register_search_tools(registry)
        register_extract_tools(registry)

        expected_names = {
            "search_arxiv",
            "search_semantic_scholar",
            "search_huggingface",
            "extract_claims",
            "extract_metrics",
            "extract_methods",
            "extract_datasets",
        }
        assert {tool.name for tool in registry.list_all()} == expected_names
        assert len(get_registry()) == 0
        for definition in registry.list_all():
            assert definition.category.value == "read"
            assert definition.parameters["type"] == "object"
            assert definition.parameters["additionalProperties"] is False
            assert definition.parameters["required"]


class TestSearchWorker:
    """Tests external paper search."""

    @pytest.mark.asyncio
    async def test_agent_type(self):
        w = SearchWorker(_mock_executor())
        assert w.agent_type == "search"

    @pytest.mark.asyncio
    async def test_parse_arxiv_xml(self):
        from litagent.tools.builtin.search import _parse_arxiv_xml

        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/1703.05175v1</id>
            <title>Prototypical Networks for Few-shot Learning</title>
            <summary>We propose prototypical networks.</summary>
          </entry>
        </feed>"""
        papers = _parse_arxiv_xml(xml)
        assert len(papers) == 1
        assert "Prototypical" in papers[0]["title"]
        assert papers[0]["source"] == "arxiv"

    @pytest.mark.asyncio
    async def test_execute_unknown_source_fails_closed(self):
        w = SearchWorker(_mock_executor())
        task = SubTask(
            task_id="t1",
            description="test",
            agent_type="search",
            input_data={"source": "unknown", "query": "test"},
        )
        result = await w.execute(task)
        assert result == []
        outcome = w.get_source_outcomes()[0]
        assert outcome.reason_code == "invalid_search_source"


class TestDedupWorker:
    """Tests paper deduplication."""

    @pytest.mark.asyncio
    async def test_dedup_by_title(self):
        w = DedupWorker()
        task = SubTask(
            task_id="dedup",
            description="dedup",
            agent_type="dedup",
            input_data={
                "upstream_results": {
                    "s1": [
                        {"title": "Paper A", "paper_id": "a1", "citation_count": 10},
                        {"title": "Paper B", "paper_id": "b1"},
                    ],
                    "s2": [
                        {"title": "paper a", "paper_id": "a2", "citation_count": 50},
                        {"title": "Paper C", "paper_id": "c1"},
                    ],
                }
            },
        )
        result = await w.execute(task)
        assert len(result) == 3
        paper_a = [p for p in result if "a" in p["title"].lower()][0]
        assert paper_a.get("citation_count", 0) == 50

    @pytest.mark.asyncio
    async def test_dedup_empty_upstream(self):
        w = DedupWorker()
        task = SubTask(
            task_id="dedup",
            description="dedup",
            agent_type="dedup",
            input_data={"upstream_results": {}},
        )
        result = await w.execute(task)
        assert result == []


class TestExtractorWorker:
    """Tests paper metadata extraction."""

    @pytest.mark.asyncio
    async def test_extract_claims(self):
        from litagent.tools.builtin.extract import extract_claims

        claims = await extract_claims(
            "We achieve state-of-the-art results on miniImageNet."
        )
        assert len(claims) >= 1

    @pytest.mark.asyncio
    async def test_extract_datasets(self):
        from litagent.tools.builtin.extract import extract_datasets

        datasets = await extract_datasets("we evaluate on miniimagenet and cifar-100")
        assert len(datasets) >= 1

    @pytest.mark.asyncio
    async def test_extract_empty_paper(self):
        from litagent.tools.builtin.extract import extract_claims, extract_metrics

        claims = await extract_claims("")
        metrics = await extract_metrics("")
        assert claims == []
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_extract_metrics(self):
        from litagent.tools.builtin.extract import extract_metrics

        metrics = await extract_metrics("We achieve accuracy of 93.2% on miniImageNet.")
        assert "accuracy" in metrics


class TestGraphWorker:
    """Tests graph-analysis output."""

    @pytest.mark.asyncio
    async def test_tier_assignment(self):
        w = GraphWorker()
        task = SubTask(
            task_id="graph",
            description="graph",
            agent_type="graph",
            input_data={
                "upstream_results": {
                    "dedup": [
                        {"paper_id": "p1", "title": "Seminal", "citation_count": 1000},
                        {"paper_id": "p2", "title": "Good", "citation_count": 100},
                        {"paper_id": "p3", "title": "New", "citation_count": 5},
                    ]
                }
            },
        )
        result = await w.execute(task)
        assert result["tier_counts"]["tier1"] == 1
        assert result["tier_counts"]["tier2"] == 1
        assert result["tier_counts"]["tier3"] == 1
        assert len(result["seminal_papers"]) == 1

    @pytest.mark.asyncio
    async def test_empty_upstream(self):
        w = GraphWorker()
        task = SubTask(
            task_id="graph",
            description="graph",
            agent_type="graph",
            input_data={"upstream_results": {}},
        )
        result = await w.execute(task)
        assert result["papers"] == []

    @pytest.mark.asyncio
    async def test_agent_type(self):
        w = GraphWorker()
        assert w.agent_type == "graph"


from unittest.mock import AsyncMock, MagicMock, patch

from litagent.tools.executor import ToolResult


class TestSearchWorkerProfile:
    """Tests search-source profile recording."""

    @pytest.mark.asyncio
    async def test_records_success_profile(self):
        """Successful searches update the source profile."""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(
                name="test_tool",
                args={},
                output=[{"title": "P", "paper_id": "2401.00001", "source": "arxiv"}],
                error=None,
            )
        )
        memory = AsyncMock()
        sw = SearchWorker(executor=executor, memory_manager=memory)
        await sw.execute(
            SubTask(
                task_id="t1",
                description="search",
                agent_type="search",
                input_data={"source": "arxiv", "query": "t"},
            )
        )
        memory.record_search_source_execution.assert_called_once()
        kwargs = memory.record_search_source_execution.call_args.kwargs
        assert kwargs["subject"] == "arxiv"
        assert kwargs["success"] is True
        assert kwargs["empty_result"] is False

    @pytest.mark.asyncio
    async def test_records_empty_result_separately(self):
        """Empty searches are recorded separately from failures."""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(name="test_tool", args={}, output=[], error=None)
        )
        memory = AsyncMock()
        sw = SearchWorker(executor=executor, memory_manager=memory)
        await sw.execute(
            SubTask(
                task_id="t2",
                description="search",
                agent_type="search",
                input_data={"source": "arxiv", "query": "t"},
            )
        )
        kwargs = memory.record_search_source_execution.call_args.kwargs
        assert kwargs["success"] is True
        assert kwargs["empty_result"] is True

    @pytest.mark.asyncio
    async def test_records_failure_profile(self):
        """Failed searches update the source profile."""
        executor = MagicMock(spec=ToolExecutor)
        failure = ToolResult(
            name="test_tool",
            args={},
            output=[],
            error="Timeout after 30000ms",
            error_code="tool_timeout",
            error_type="TimeoutError",
        )
        executor.execute = AsyncMock(return_value=failure)
        memory = AsyncMock()
        sw = SearchWorker(executor=executor, memory_manager=memory)
        await sw.execute(
            SubTask(
                task_id="t3",
                description="search",
                agent_type="search",
                input_data={"source": "huggingface", "query": "t"},
            )
        )
        kwargs = memory.record_search_source_execution.call_args.kwargs
        assert kwargs["subject"] == "huggingface"
        assert kwargs["success"] is False
        assert kwargs["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_memory_failure_does_not_break_search(self):
        """Profile-write failures do not break search results."""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(
                name="test_tool",
                args={},
                output=[{"title": "X", "paper_id": "2401.00002", "source": "arxiv"}],
                error=None,
            )
        )
        memory = AsyncMock()
        memory.record_search_source_execution.side_effect = RuntimeError("db down")
        sw = SearchWorker(executor=executor, memory_manager=memory)
        result = await sw.execute(
            SubTask(
                task_id="t4",
                description="search",
                agent_type="search",
                input_data={"source": "arxiv", "query": "t"},
            )
        )
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_noop_when_memory_is_none(self):
        """Search remains functional without a memory manager."""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(
                name="test_tool",
                args={},
                output=[{"title": "X", "paper_id": "2401.00002", "source": "arxiv"}],
                error=None,
            )
        )
        sw = SearchWorker(executor=executor, memory_manager=None)
        result = await sw.execute(
            SubTask(
                task_id="t5",
                description="search",
                agent_type="search",
                input_data={"source": "arxiv", "query": "t"},
            )
        )
        assert len(result) > 0


from unittest.mock import AsyncMock, MagicMock

from litagent.memory.manager import MemoryManager


class TestRankSearchSources:
    """Tests reliability-based search-source ranking."""

    @staticmethod
    def _memory_with_profiles(profiles=None, error=None):
        procedural = MagicMock()
        procedural.get_profiles = AsyncMock(
            side_effect=error,
            return_value=profiles or [],
        )
        return MemoryManager(
            working=MagicMock(),
            episodic=MagicMock(),
            procedural=procedural,
        )

    @pytest.mark.asyncio
    async def test_rank_orders_sufficient_profiles_by_reliability(self):
        memory = self._memory_with_profiles(
            [
                {
                    "subject": "arxiv",
                    "success_count": 90,
                    "failure_count": 1,
                    "empty_result_count": 5,
                    "rate_limit_count": 0,
                    "timeout_count": 0,
                    "execution_count": 96,
                },
                {
                    "subject": "huggingface",
                    "success_count": 2,
                    "failure_count": 8,
                    "empty_result_count": 0,
                    "rate_limit_count": 2,
                    "timeout_count": 5,
                    "execution_count": 15,
                },
                {
                    "subject": "semantic_scholar",
                    "success_count": 10,
                    "failure_count": 10,
                    "empty_result_count": 5,
                    "rate_limit_count": 5,
                    "timeout_count": 1,
                    "execution_count": 30,
                },
            ]
        )
        result = await memory.rank_search_sources(
            ["huggingface", "semantic_scholar", "arxiv"]
        )
        assert set(result) == {"huggingface", "semantic_scholar", "arxiv"}
        assert result[0] == "arxiv"

    @pytest.mark.asyncio
    async def test_rank_preserves_input_order_when_samples_insufficient(self):
        memory = self._memory_with_profiles(
            [
                {
                    "subject": "hf",
                    "success_count": 1,
                    "failure_count": 0,
                    "empty_result_count": 0,
                    "rate_limit_count": 0,
                    "timeout_count": 0,
                    "execution_count": 1,
                },
            ]
        )
        result = await memory.rank_search_sources(["arxiv", "hf"], min_samples=5)
        assert result == ["arxiv", "hf"]

    @pytest.mark.asyncio
    async def test_rank_preserves_input_order_when_profile_read_fails(self):
        memory = self._memory_with_profiles(error=RuntimeError("db down"))
        result = await memory.rank_search_sources(["arxiv", "huggingface"])
        assert result == ["arxiv", "huggingface"]


class TestPlannerPriorityMapping:
    """Tests mapping source rank to planner priority."""

    @pytest.mark.asyncio
    async def test_planner_maps_rank_to_search_task_priority(self):
        from litagent.agents.planner import SurveyPlanner
        from litagent.config import PlannerConfig
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(['{"sub_queries": ["a"]}'])
        memory = AsyncMock(spec=MemoryManager)
        memory.rank_search_sources = AsyncMock(return_value=["huggingface", "arxiv"])

        cfg = PlannerConfig(use_procedural_profiles=True)
        planner = SurveyPlanner(llm=llm, config=cfg, memory_manager=memory)
        graph = await planner.plan("test")

        search_tasks = [t for t in graph.tasks.values() if t.agent_type == "search"]
        hf_tasks = [t for t in search_tasks if "huggingface" in t.task_id]
        arxiv_tasks = [t for t in search_tasks if "arxiv" in t.task_id]
        assert all(t.priority == 0 for t in hf_tasks)
        assert all(t.priority == 1 for t in arxiv_tasks)
        assert len(set(t.priority for t in hf_tasks)) == 1

    @pytest.mark.asyncio
    async def test_planner_disabled_profile_ranking_keeps_default_sources(self):
        from litagent.agents.planner import SurveyPlanner
        from litagent.config import PlannerConfig
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(['{"sub_queries": ["a"]}'])
        memory = AsyncMock(spec=MemoryManager)

        cfg = PlannerConfig(use_procedural_profiles=False)
        planner = SurveyPlanner(llm=llm, config=cfg, memory_manager=memory)
        graph = await planner.plan("test")

        search_sources = {
            t.input_data["source"]
            for t in graph.tasks.values()
            if t.agent_type == "search"
        }
        assert "arxiv" in search_sources
        assert "huggingface" in search_sources
        memory.rank_search_sources.assert_not_called()


class TestSynthesisEvidenceBoundary:
    """Tests synthesis evidence boundaries."""

    def test_prompt_contains_evidence_boundary_instructions(self):
        from litagent.agents.synthesis import SYNTHESIS_INSTRUCTIONS

        assert "SCOPED EVIDENCE SUMMARY" in SYNTHESIS_INSTRUCTIONS
        assert "evidence not provided" in SYNTHESIS_INSTRUCTIONS
        assert "NEVER invent evidence IDs" in SYNTHESIS_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_execute_uses_configured_react_loop_limit(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.config import AgentConfig

        react = MagicMock()
        react.run = AsyncMock(return_value='{"draft": "ok"}')
        with patch(
            "litagent.agents.synthesis.ReActRunner", return_value=react
        ) as runner_cls:
            worker = SynthesisWorker(
                llm=MagicMock(), agent_config=AgentConfig(max_loops=2)
            )
            await worker.execute(
                SubTask(
                    task_id="synthesis",
                    description="synthesize",
                    agent_type="synthesis",
                    input_data={"query": "few-shot", "upstream_results": {}},
                )
            )

        assert runner_cls.call_args.kwargs["config"].max_loops == 2


class TestReviewerStructuredOutput:
    """Tests structured reviewer output."""

    @pytest.mark.asyncio
    async def test_reviewer_uses_one_structured_llm_call_not_react_loop(self):
        import json as _json
        from unittest.mock import AsyncMock, MagicMock

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import BaseLLMClient, LLMResponse

        llm = MagicMock(spec=BaseLLMClient)
        review_json = _json.dumps(
            {
                "score": 0.7,
                "strengths": ["clear"],
                "weaknesses": ["short"],
                "issues": [],
                "missing_coverage": [],
                "verdict": "revise",
            }
        )
        llm.chat = AsyncMock(
            return_value=LLMResponse(content=review_json, model="test")
        )

        reviewer = ReviewerWorker(llm=llm)
        result = await reviewer.execute(
            SubTask(
                task_id="r1",
                description="review",
                agent_type="reviewer",
                input_data={"upstream_results": {"synthesis": {"draft": "test draft"}}},
            )
        )
        assert result["score"] == 0.7
        llm.chat.assert_called_once()
        assert llm.chat.call_args.kwargs.get("response_format") == {
            "type": "json_object"
        }

    @pytest.mark.asyncio
    async def test_reviewer_invalid_json_returns_parse_error_diagnostic(self):
        from unittest.mock import AsyncMock, MagicMock

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import BaseLLMClient, LLMResponse

        llm = MagicMock(spec=BaseLLMClient)
        llm.chat = AsyncMock(return_value=LLMResponse(content="not json", model="test"))

        reviewer = ReviewerWorker(llm=llm)
        result = await reviewer.execute(
            SubTask(
                task_id="r2",
                description="review",
                agent_type="reviewer",
                input_data={"upstream_results": {"synthesis": {"draft": "test draft"}}},
            )
        )
        assert "parse_error" in result


class TestQueryIntent:
    """Tests query-intent parsing."""

    def test_topic(self):
        from litagent.agents.planner import QueryIntent, classify_query_intent

        assert classify_query_intent("few-shot learning in CV") == QueryIntent.TOPIC
        assert (
            classify_query_intent("attention is all you need 2017") == QueryIntent.TOPIC
        )

    def test_arxiv_new_style(self):
        from litagent.agents.planner import QueryIntent, classify_query_intent

        assert classify_query_intent("2401.00001") == QueryIntent.ARXIV_ID
        assert classify_query_intent("2401.00001v2") == QueryIntent.ARXIV_ID
        assert classify_query_intent("arXiv:2401.00001") == QueryIntent.ARXIV_ID

    def test_arxiv_legacy(self):
        from litagent.agents.planner import QueryIntent, classify_query_intent

        assert classify_query_intent("cs.CL/0301001") == QueryIntent.ARXIV_ID
        assert classify_query_intent("hep-th/9901001v1") == QueryIntent.ARXIV_ID

    def test_doi(self):
        from litagent.agents.planner import QueryIntent, classify_query_intent

        assert classify_query_intent("10.1038/nature12373") == QueryIntent.DOI
        assert classify_query_intent("doi:10.1145/3292500.3330701") == QueryIntent.DOI

    def test_url_wins_over_embedded_id(self):
        from litagent.agents.planner import QueryIntent, classify_query_intent

        assert (
            classify_query_intent("https://arxiv.org/abs/2401.00001") == QueryIntent.URL
        )
        assert classify_query_intent("http://example.com/paper") == QueryIntent.URL


class TestPlannerRecallTasks:
    """Tests planner recall-task construction."""

    @pytest.fixture(autouse=True)
    def _no_ss_key(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_n_subqueries_yield_n_recall_tasks(self):
        """Each normalized subquery produces one recall task."""
        from litagent.agents.planner import SurveyPlanner
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(['{"sub_queries": ["angle a", "angle b"]}'])
        planner = SurveyPlanner(llm=llm)
        graph = await planner.plan("few-shot learning")

        recall_tasks = [t for t in graph.tasks.values() if t.agent_type == "recall"]
        search_tasks = [t for t in graph.tasks.values() if t.agent_type == "search"]
        assert len(recall_tasks) == 3
        assert len(search_tasks) == 6

        for t in recall_tasks:
            assert "query" in t.input_data
            assert "top_k" in t.input_data
            assert "query_index" in t.input_data

    @pytest.mark.asyncio
    async def test_arxiv_id_skips_decompose_but_keeps_one_recall(self):
        from unittest.mock import AsyncMock, MagicMock

        from litagent.agents.planner import SurveyPlanner
        from litagent.llm.client import BaseLLMClient

        llm = MagicMock(spec=BaseLLMClient)
        llm.chat = AsyncMock()
        planner = SurveyPlanner(llm=llm)
        graph = await planner.plan("2401.00001")

        llm.chat.assert_not_called()
        recall_tasks = [t for t in graph.tasks.values() if t.agent_type == "recall"]
        assert len(recall_tasks) == 1
        assert recall_tasks[0].input_data["query"] == "2401.00001"

    @pytest.mark.asyncio
    async def test_external_task_carries_mode_and_source(self):
        from litagent.agents.planner import SurveyPlanner

        planner = SurveyPlanner()
        graph = await planner.plan("test topic")
        for t in graph.tasks.values():
            if t.agent_type == "search":
                assert t.input_data["mode"] == "external"
                assert "source" in t.input_data
                assert "query" in t.input_data

    @pytest.mark.asyncio
    async def test_dedup_fans_in_external_and_recall(self):
        """Deduplication waits for external search and recall tasks."""
        from litagent.agents.planner import SurveyPlanner

        planner = SurveyPlanner()
        graph = await planner.plan("test topic")

        for t in list(graph.tasks.values()):
            if t.agent_type == "search":
                graph.mark_done(t.task_id, [])
        ready_ids = {t.task_id for t in graph.get_ready_tasks()}
        assert "dedup" not in ready_ids

        for t in list(graph.tasks.values()):
            if t.agent_type == "recall":
                graph.mark_done(t.task_id, [])
        ready_ids = {t.task_id for t in graph.get_ready_tasks()}
        assert "dedup" in ready_ids


class TestRecallWorker:
    """Tests semantic-memory recall."""

    @staticmethod
    def _scored_hit():
        from litagent.rag.models import ContentChunk, ContentScope, ScoredPaperHit

        chunk = ContentChunk.from_text(
            paper_id="arxiv:2401.00001",
            chunk_key="abstract",
            text="abstract text",
            section="abstract",
            content_scope=ContentScope.ABSTRACT,
        )
        return ScoredPaperHit(
            paper_id="arxiv:2401.00001",
            title="T",
            abstract="abstract text",
            content_scope=ContentScope.ABSTRACT,
            chunks=[chunk],
            score=0.9,
            collection="papers",
            corpus_version="v1",
            schema_version="paper-v1",
            parser_version="pymupdf-v1",
            chunking_version="page-block-v1",
            embedding_model="all-MiniLM-L6-v2",
        )

    @pytest.mark.asyncio
    async def test_no_retriever_returns_empty(self):
        from litagent.agents.recall import RecallWorker

        w = RecallWorker(retriever=None)
        assert w.agent_type == "recall"
        result = await w.execute(
            SubTask(
                task_id="recall_q0",
                description="r",
                agent_type="recall",
                input_data={"query": "x", "top_k": 5},
            )
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_retriever_error_returns_empty(self):
        """Retriever failures degrade to an empty result."""
        from litagent.agents.recall import RecallWorker

        retriever = MagicMock()
        retriever.search = AsyncMock(side_effect=RuntimeError("qdrant 400"))
        w = RecallWorker(retriever=retriever)
        result = await w.execute(
            SubTask(
                task_id="recall_q0",
                description="r",
                agent_type="recall",
                input_data={"query": "x", "top_k": 5},
            )
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_result_keeps_dedup_contract(self):
        from litagent.agents.recall import RecallWorker

        retriever = MagicMock()
        retriever.search_papers = AsyncMock(return_value=[self._scored_hit()])
        w = RecallWorker(retriever=retriever)
        result = await w.execute(
            SubTask(
                task_id="recall_q0",
                description="r",
                agent_type="recall",
                input_data={"query": "x", "top_k": 5},
            )
        )
        assert len(result) == 1
        p = result[0]
        assert p["paper_id"] == "arxiv:2401.00001"
        assert p["title"] == "T"
        assert p["source"] == "rag_index"
        assert p["score"] == 0.9
        assert len(p["abstract"]) <= 500

    @pytest.mark.asyncio
    async def test_retriever_calls_equal_normalized_subquery_count(self, monkeypatch):
        """Recall invokes the retriever once per normalized subquery."""
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        from litagent.agents.planner import SurveyPlanner
        from litagent.agents.recall import RecallWorker
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(['{"sub_queries": ["a", "b"]}'])
        planner = SurveyPlanner(llm=llm)
        graph = await planner.plan("topic")

        retriever = MagicMock()
        retriever.search_papers = AsyncMock(return_value=[])
        w = RecallWorker(retriever=retriever)
        for t in graph.tasks.values():
            if t.agent_type == "recall":
                await w.execute(t)
        assert retriever.search_papers.await_count == 3


class TestSearchWorkerPureExternal:
    """Tests that search remains external-only."""

    def test_search_worker_has_no_retriever(self):
        import inspect

        sig = inspect.signature(SearchWorker.__init__)
        assert "retriever" not in sig.parameters

    @pytest.mark.asyncio
    async def test_returns_api_papers_only(self):
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(
                name="t",
                args={},
                output=[{"title": "P", "source": "arxiv", "paper_id": "2401.00001"}],
                error=None,
            )
        )
        sw = SearchWorker(executor=executor, memory_manager=None)
        result = await sw.execute(
            SubTask(
                task_id="s",
                description="s",
                agent_type="search",
                input_data={"source": "arxiv", "query": "q"},
            )
        )
        assert result == [
            {
                "paper_id": "arxiv:2401.00001",
                "title": "P",
                "abstract": "",
                "authors": [],
                "citation_count": 0,
                "source": "arxiv",
                "content_scope": "metadata_only",
                "chunks": [],
                "provenance": [],
                "warnings": [],
            }
        ]


class TestEvidenceLedger:
    """Tests evidence-ledger identity and formatting."""

    def test_ids_stable_and_unique(self):
        from litagent.evidence import build_evidence_items

        ext = {
            "paper_id": "p1",
            "title": "T",
            "abstract": "abs",
            "claims": ["c one", "c two"],
        }
        items = build_evidence_items(ext)
        ids = [it["evidence_id"] for it in items]
        assert ids == ["p1:claim:0", "p1:claim:1", "p1:abstract"]
        assert len(ids) == len(set(ids))
        assert build_evidence_items(ext) == items
        assert items[0]["text"] == "c one"
        assert items[0]["source_locator"] == "extracted_claim"
        assert items[2]["source_locator"] == "abstract"
        assert items[0]["confidence"] is None

    def test_missing_paper_id_uses_title_hash(self):
        from litagent.evidence import build_evidence_items

        ext = {"paper_id": "", "title": "Some Paper", "abstract": "", "claims": ["c"]}
        a = build_evidence_items(ext)
        b = build_evidence_items(dict(ext))
        assert a[0]["evidence_id"] == b[0]["evidence_id"]
        assert a[0]["evidence_id"].startswith("t")

    def test_collect_and_find_unknown_refs(self):
        from litagent.evidence import (
            build_evidence_items,
            collect_ledger,
            find_unknown_refs,
        )

        ext = {"paper_id": "p1", "title": "T", "abstract": "abs", "claims": ["c"]}
        ext["evidence_items"] = build_evidence_items(ext)
        ledger = collect_ledger([ext])
        assert "p1:claim:0" in ledger

        text = "Good [E:p1:claim:0]. Bad [E:fake:claim:9]."
        assert find_unknown_refs(text, ledger) == ["fake:claim:9"]
        assert find_unknown_refs("no refs here", ledger) == []

    def test_format_ledger_lines(self):
        from litagent.evidence import (
            build_evidence_items,
            collect_ledger,
            format_ledger,
        )

        ext = {"paper_id": "p1", "title": "T", "abstract": "", "claims": ["c one"]}
        ext["evidence_items"] = build_evidence_items(ext)
        text = format_ledger(collect_ledger([ext]))
        assert "[E:p1:claim:0]" in text
        assert "(T)" in text
        assert "c one" in text

    def test_format_ledger_truncates_at_max_chars(self):
        """Ledger rendering respects its character limit."""
        from litagent.evidence import (
            build_evidence_items,
            collect_ledger,
            format_ledger,
        )

        ext = {
            "paper_id": "p1",
            "title": "T",
            "abstract": "",
            "claims": [f"claim number {i} " + "x" * 80 for i in range(50)],
        }
        ext["evidence_items"] = build_evidence_items(ext)
        text = format_ledger(collect_ledger([ext]), max_chars=500)
        assert len(text) < 700
        assert "omitted" in text

        full = format_ledger(collect_ledger([ext]))
        assert "omitted" not in full


class TestSynthesisEvidenceCitation:
    """Tests evidence-citation prompts and payloads."""

    def test_instructions_contain_citation_rules(self):
        from litagent.agents.synthesis import SYNTHESIS_INSTRUCTIONS

        assert "EVIDENCE BOUNDARIES" in SYNTHESIS_INSTRUCTIONS
        assert "[E:" in SYNTHESIS_INSTRUCTIONS
        assert "NEVER invent" in SYNTHESIS_INSTRUCTIONS

    def test_revise_instructions_forbid_new_papers(self):
        from litagent.agents.synthesis import REVISE_INSTRUCTIONS

        assert "known gap" in REVISE_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_evidence_layer_renders_in_execute(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.evidence import build_evidence_items
        from litagent.llm.client import MockLLMClient

        ext = {"paper_id": "p1", "title": "T", "abstract": "abs", "claims": ["c"]}
        ext["evidence_items"] = build_evidence_items(ext)
        llm = MockLLMClient(responses=["Survey draft."])
        fake_sel = FakeEvidenceSelector()
        w = SynthesisWorker(llm=llm, evidence_selector=fake_sel)
        task = SubTask(
            task_id="synthesis",
            description="s",
            agent_type="synthesis",
            input_data={
                "query": "test",
                "upstream_results": {
                    "extract": [ext],
                    "graph_analysis": {"papers": [], "tier_counts": {}},
                },
            },
        )
        result = await w.execute(task)
        assert "<evidence_ledger>" not in result["draft"]
        assert "evidence_selection" in result
        assert len(result["evidence_selection"]["selected_items"]) > 0

    def test_rewrite_messages_carry_draft_ledger_diagnostics(self):
        from litagent.agents.synthesis import REWRITE_INSTRUCTIONS, SynthesisWorker

        assert "NEVER add new paper titles" in REWRITE_INSTRUCTIONS
        w = SynthesisWorker(llm=MagicMock())
        messages = w.rewrite_with_evidence(
            "the draft", "[E:p1:claim:0] (T) c", '[{"claim_text": "bad"}]'
        )
        assert messages[0]["role"] == "system"
        user = messages[1]["content"]
        assert "<current_draft>" in user
        assert "<evidence_ledger>" in user
        assert "<unsupported_claims>" in user
        assert "the draft" in user


class TestFaithfulnessDiagnostic:
    """Tests faithfulness diagnostics."""

    @staticmethod
    def _cfg():
        from litagent.config import AgentConfig, AppConfig, LoggingConfig

        return AppConfig(agent=AgentConfig(), logging=LoggingConfig())

    @staticmethod
    def _ledger():
        from litagent.evidence import build_evidence_items, collect_ledger

        ext = {"paper_id": "p1", "title": "T", "abstract": "abs", "claims": ["c"]}
        ext["evidence_items"] = build_evidence_items(ext)
        return collect_ledger([ext])

    @classmethod
    def _context(cls):
        from litagent.eval.base import (
            CTX_EVIDENCE,
            CTX_REFERENCED_EVIDENCE,
            CTX_REFERENCED_EVIDENCE_IDS,
        )

        ledger = cls._ledger()
        return {
            CTX_EVIDENCE: ledger,
            CTX_REFERENCED_EVIDENCE: list(ledger.values()),
            CTX_REFERENCED_EVIDENCE_IDS: list(ledger),
        }

    @pytest.mark.asyncio
    async def test_diagnose_returns_unsupported_claims(self):
        import json as _json

        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
        from litagent.llm.client import BaseLLMClient, LLMResponse

        llm = MagicMock(spec=BaseLLMClient)
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content=_json.dumps(
                    {
                        "unsupported_claims": [
                            {
                                "claim_text": "X beats Y",
                                "evidence_ids": ["p1:claim:0"],
                                "reason": "overstates",
                            }
                        ]
                    }
                ),
                model="test",
            )
        )
        ev = RagasFaithfulnessEvaluator(self._cfg(), llm=llm)
        out = await ev._diagnose("survey", self._context())
        assert out["unsupported_claims"][0]["claim_text"] == "X beats Y"
        assert out["unsupported_claims"][0]["evidence_ids"] == ["p1:claim:0"]

        assert llm.chat.call_args.kwargs.get("response_format") == {
            "type": "json_object"
        }

    @pytest.mark.asyncio
    async def test_diagnose_malformed_marks_skipped_not_empty(self):
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
        from litagent.llm.client import BaseLLMClient, LLMResponse

        llm = MagicMock(spec=BaseLLMClient)
        llm.chat = AsyncMock(
            return_value=LLMResponse(
                content='{"unsupported_claims": "not a list"}', model="test"
            )
        )
        ev = RagasFaithfulnessEvaluator(self._cfg(), llm=llm)
        out = await ev._diagnose("survey", self._context())
        assert "unsupported_claims" not in out
        assert "diagnostic_skipped" in out

    @pytest.mark.asyncio
    async def test_diagnose_without_llm_or_ledger_marks_skipped(self):
        from litagent.eval.base import CTX_EVIDENCE
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator

        ev = RagasFaithfulnessEvaluator(self._cfg())
        out = await ev._diagnose("survey", self._context())
        assert "diagnostic_skipped" in out

        ev2 = RagasFaithfulnessEvaluator(self._cfg(), llm=MagicMock())
        out2 = await ev2._diagnose("survey", {CTX_EVIDENCE: {}})
        assert "diagnostic_skipped" in out2


class TestRelevanceGateWorker:
    """Tests relevance filtering and degradation."""

    @staticmethod
    def _papers(n: int = 10):
        return [
            {
                "paper_id": f"p{i}",
                "title": f"Paper {i}",
                "abstract": f"abstract text {i}",
            }
            for i in range(n)
        ]

    class _FakeReranker:
        """Reranker that assigns deterministic scores."""

        def rerank(self, query, docs):
            assert query
            assert all(hasattr(item.doc, "page_content") for item in docs)
            for index, item in enumerate(docs):
                item.score = float(index)
            return sorted(docs, key=lambda item: item.score, reverse=True)

    @pytest.mark.asyncio
    async def test_cross_encoder_sorts_and_caps(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        w = RelevanceGateWorker(
            reranker=self._FakeReranker(),
            max_papers=5,
            min_papers=0,
        )
        result = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "test",
                    "upstream_results": {"dedup": self._papers(10)},
                },
            )
        )
        assert len(result) == 5
        assert [paper["paper_id"] for paper in result] == [
            "p9",
            "p8",
            "p7",
            "p6",
            "p5",
        ]
        assert result[0]["relevance_score"] >= result[-1]["relevance_score"]
        assert result[0]["relevance_method"] == "cross_encoder"
        assert "relevance_rank" in result[0]
        assert result[0]["relevance_rank"] == 0

    @pytest.mark.asyncio
    async def test_ignores_other_upstream_keys(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        w = RelevanceGateWorker(reranker=self._FakeReranker(), max_papers=50)
        result = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "test",
                    "upstream_results": {
                        "dedup": self._papers(3),
                        "search_arxiv_q0": [{"title": "should be ignored"}],
                    },
                },
            )
        )
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_reranker_error_falls_back_to_lexical(self):
        class _BadReranker:
            """Reranker test double that always raises."""

            def rerank(self, query, docs):
                raise RuntimeError("model load failed")

        from litagent.agents.relevance_gate import RelevanceGateWorker

        events = []
        w = RelevanceGateWorker(
            reranker=_BadReranker(),
            max_papers=50,
            trace_hook=lambda event, data: events.append((event, data)),
        )
        result = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "paper",
                    "upstream_results": {"dedup": self._papers(5)},
                },
            )
        )
        assert len(result) <= 5
        assert result[0]["relevance_method"] == "lexical_fallback"
        assert events[0] == (
            "relevance.gate.degraded",
            {
                "reason_code": "reranker_error",
                "error_type": "RuntimeError",
            },
        )

    @pytest.mark.asyncio
    async def test_reranker_error_respects_configured_cap(self):
        class _BadReranker:
            """Reranker test double that always raises."""

            def rerank(self, query, docs):
                raise RuntimeError("model load failed")

        from litagent.agents.relevance_gate import RelevanceGateWorker

        worker = RelevanceGateWorker(
            reranker=_BadReranker(),
            max_papers=3,
            min_papers=0,
        )
        result = await worker.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "paper",
                    "upstream_results": {
                        "dedup": self._papers(10),
                    },
                },
            )
        )
        assert len(result) == 3
        assert all(item["relevance_method"] == "lexical_fallback" for item in result)

    @pytest.mark.asyncio
    async def test_lexical_fallback_stable(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        w = RelevanceGateWorker(reranker=None, max_papers=50)
        a = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "paper 0",
                    "upstream_results": {"dedup": self._papers(5)},
                },
            )
        )
        b = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "paper 0",
                    "upstream_results": {"dedup": self._papers(5)},
                },
            )
        )
        assert len(a) == len(b)
        assert [p["paper_id"] for p in a] == [p["paper_id"] for p in b]
        assert a[0]["relevance_method"] == "lexical_fallback"

    @pytest.mark.asyncio
    async def test_empty_papers_returns_empty(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        w = RelevanceGateWorker(reranker=None)
        result = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={"query": "t", "upstream_results": {"dedup": []}},
            )
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_no_title_or_abstract_filtered(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        w = RelevanceGateWorker(reranker=None)
        papers = [
            {"paper_id": "p1"},
            {"paper_id": "p2", "title": "T"},
        ]
        result = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={"query": "t", "upstream_results": {"dedup": papers}},
            )
        )
        assert len(result) == 1
        assert result[0]["paper_id"] == "p2"

    @pytest.mark.asyncio
    async def test_non_dict_candidates_are_filtered(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        worker = RelevanceGateWorker(reranker=None)
        result = await worker.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "t",
                    "upstream_results": {
                        "dedup": [
                            None,
                            "bad",
                            ["bad"],
                            {"paper_id": "p2", "title": "T"},
                        ],
                    },
                },
            )
        )
        assert [paper["paper_id"] for paper in result] == ["p2"]

    @pytest.mark.asyncio
    async def test_empty_query_uses_lexical(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        w = RelevanceGateWorker(reranker=self._FakeReranker(), max_papers=50)
        result = await w.execute(
            SubTask(
                task_id="rg",
                description="t",
                agent_type="relevance_gate",
                input_data={
                    "query": "",
                    "upstream_results": {"dedup": self._papers(5)},
                },
            )
        )
        assert result[0]["relevance_method"] == "lexical_fallback"

    def test_rejects_non_positive_max_papers(self):
        from litagent.agents.relevance_gate import RelevanceGateWorker

        with pytest.raises(ValueError):
            RelevanceGateWorker(reranker=None, max_papers=0)


class TestExtractorResilience:
    """Tests extractor ordering and failure handling."""

    @pytest.mark.asyncio
    async def test_reads_relevance_gate_first(self):
        from litagent.agents.extractor import ExtractorWorker

        strategy = MagicMock()
        strategy.extract = AsyncMock(return_value={"claims": [], "metrics": {}})
        w = ExtractorWorker(strategy=strategy, max_papers=50)
        papers = [{"paper_id": "p1", "title": "T", "abstract": "A"}]
        result = await w.execute(
            SubTask(
                task_id="extract",
                description="e",
                agent_type="extractor",
                input_data={
                    "upstream_results": {
                        "relevance_gate": papers,
                        "dedup": [{"paper_id": "should_not_be_used"}],
                    }
                },
            )
        )
        assert isinstance(result, list)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_falls_back_to_dedup(self):
        from litagent.agents.extractor import ExtractorWorker

        strategy = MagicMock()
        strategy.extract = AsyncMock(return_value={"claims": [], "metrics": {}})
        w = ExtractorWorker(strategy=strategy, max_papers=50)
        papers = [{"paper_id": "d1", "title": "D", "abstract": "a"}]
        result = await w.execute(
            SubTask(
                task_id="extract",
                description="e",
                agent_type="extractor",
                input_data={"upstream_results": {"dedup": papers}},
            )
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_ignores_unrelated_list_keys(self):
        from litagent.agents.extractor import ExtractorWorker

        strategy = MagicMock()
        strategy.extract = AsyncMock(return_value={"claims": [], "metrics": {}})
        w = ExtractorWorker(strategy=strategy, max_papers=50)
        result = await w.execute(
            SubTask(
                task_id="extract",
                description="e",
                agent_type="extractor",
                input_data={
                    "upstream_results": {
                        "search_arxiv_q0": [{"title": "irrelevant"}],
                    }
                },
            )
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_max_papers_caps_extractor_calls(self):
        from litagent.agents.extractor import ExtractorWorker

        strategy = MagicMock()
        strategy.extract = AsyncMock(return_value={"claims": [], "metrics": {}})
        w = ExtractorWorker(strategy=strategy, max_papers=3)
        papers = [
            {"paper_id": f"p{i}", "title": f"P{i}", "abstract": "a"} for i in range(10)
        ]
        await w.execute(
            SubTask(
                task_id="extract",
                description="e",
                agent_type="extractor",
                input_data={"upstream_results": {"relevance_gate": papers}},
            )
        )
        assert strategy.extract.await_count == 3

    @pytest.mark.asyncio
    async def test_empty_gate_does_not_fall_back_to_dedup(self):
        strategy = MagicMock()
        strategy.extract = AsyncMock(return_value={"claims": [], "metrics": {}})
        worker = ExtractorWorker(strategy=strategy)
        result = await worker.execute(
            SubTask(
                task_id="extract",
                description="e",
                agent_type="extractor",
                input_data={
                    "upstream_results": {
                        "relevance_gate": [],
                        "dedup": [{"paper_id": "must-not-return", "title": "T"}],
                    }
                },
            )
        )
        assert result == []
        strategy.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_graph_and_extractor_preserve_gate_order(self):
        papers = [
            {"paper_id": "p2", "title": "Second", "citation_count": 2},
            {"paper_id": "p1", "title": "First", "citation_count": 1},
        ]
        upstream = {"relevance_gate": papers, "dedup": list(reversed(papers))}
        strategy = MagicMock()
        strategy.extract = AsyncMock(
            side_effect=lambda _paper: {"claims": [], "metrics": {}},
        )

        extractor_result = await ExtractorWorker(strategy=strategy).execute(
            SubTask(
                task_id="extract",
                description="e",
                agent_type="extractor",
                input_data={"upstream_results": upstream},
            )
        )
        graph_result = await GraphWorker().execute(
            SubTask(
                task_id="graph",
                description="g",
                agent_type="graph",
                input_data={"upstream_results": upstream},
            )
        )

        assert [paper["paper_id"] for paper in extractor_result] == ["p2", "p1"]
        assert [paper["paper_id"] for paper in graph_result["papers"]] == ["p2", "p1"]

    @pytest.mark.asyncio
    async def test_graph_empty_gate_does_not_fall_back_to_dedup(self):
        result = await GraphWorker().execute(
            SubTask(
                task_id="graph",
                description="g",
                agent_type="graph",
                input_data={
                    "upstream_results": {
                        "relevance_gate": [],
                        "dedup": [{"paper_id": "must-not-return", "title": "T"}],
                    }
                },
            )
        )
        assert result["papers"] == []

    @pytest.mark.asyncio
    async def test_worker_propagates_cancelled_error(self):
        strategy = MagicMock()
        strategy.extract = AsyncMock(side_effect=asyncio.CancelledError())
        worker = ExtractorWorker(strategy=strategy)
        with pytest.raises(asyncio.CancelledError):
            await worker.execute(
                SubTask(
                    task_id="extract",
                    description="e",
                    agent_type="extractor",
                    input_data={
                        "upstream_results": {
                            "relevance_gate": [{"paper_id": "p1", "title": "T"}],
                        }
                    },
                )
            )

    def test_workers_reject_non_positive_max_papers(self):
        with pytest.raises(ValueError):
            ExtractorWorker(strategy=MagicMock(), max_papers=0)
        with pytest.raises(ValueError):
            GraphWorker(max_papers=0)

    @pytest.mark.asyncio
    async def test_agent_type(self):
        from litagent.agents.extractor import ExtractorWorker

        w = ExtractorWorker(strategy=MagicMock())
        assert w.agent_type == "extractor"


class TestResilientExtractionStrategy:
    """Tests resilient extraction fallbacks."""

    @staticmethod
    def _strategy(llm_extract, regex_result=None, timeout_ms=100):
        from litagent.agents.extraction_strategy import ResilientExtractionStrategy

        llm = MagicMock()
        llm.extract = AsyncMock(side_effect=llm_extract)
        regex = MagicMock()
        regex.extract = AsyncMock(
            return_value=regex_result
            or {
                "claims": [],
                "metrics": {},
                "methods": [],
                "datasets": [],
            }
        )
        return ResilientExtractionStrategy(llm, regex, timeout_ms), llm, regex

    @pytest.mark.asyncio
    async def test_success_calls_llm_once(self):
        strategy, llm, regex = self._strategy(None)
        llm.extract.side_effect = None
        llm.extract.return_value = {"claims": [], "metrics": {}}
        result = await strategy.extract({"paper_id": "p1"})
        assert llm.extract.await_count == 1
        regex.extract.assert_not_awaited()
        assert result["extraction_mode"] == "llm"

    @pytest.mark.asyncio
    async def test_timeout_calls_llm_once_and_falls_back(self):
        async def slow_extract(_paper):
            await asyncio.sleep(1)

        strategy, llm, regex = self._strategy(slow_extract, timeout_ms=1)
        result = await strategy.extract({"paper_id": "p1"})
        assert llm.extract.await_count == 1
        assert regex.extract.await_count == 1
        assert result["degradation_reason"] == "llm_timeout"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("error", "reason"),
        [
            (json.JSONDecodeError("bad", "x", 0), "invalid_llm_output"),
            (RuntimeError("provider unavailable"), "llm_error"),
        ],
    )
    async def test_errors_fall_back_with_stable_reason(self, error, reason):
        strategy, llm, regex = self._strategy(error)
        result = await strategy.extract({"paper_id": "p1"})
        assert llm.extract.await_count == 1
        assert regex.extract.await_count == 1
        assert result["extraction_mode"] == "regex_fallback"
        assert result["degradation_reason"] == reason

    @pytest.mark.asyncio
    async def test_cancelled_error_propagates_without_fallback(self):
        strategy, llm, regex = self._strategy(asyncio.CancelledError())
        with pytest.raises(asyncio.CancelledError):
            await strategy.extract({"paper_id": "p1"})
        assert llm.extract.await_count == 1
        regex.extract.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_provider_error_text_is_not_logged(self, caplog):
        strategy, _, _ = self._strategy(RuntimeError("secret-value-must-not-leak"))
        with caplog.at_level("WARNING"):
            await strategy.extract({"paper_id": "p1"})
        assert "secret-value-must-not-leak" not in caplog.text
        assert "error_type=RuntimeError" in caplog.text


class TestExtractorEvidenceItems:
    """Tests extracted evidence-item construction."""

    @pytest.mark.asyncio
    async def test_extractions_include_evidence_items(self):
        from litagent.agents.extractor import ExtractorWorker

        strategy = MagicMock()
        strategy.extract = AsyncMock(return_value={"claims": ["c1"], "metrics": {}})
        w = ExtractorWorker(strategy=strategy)
        result = await w.execute(
            SubTask(
                task_id="extract",
                description="e",
                agent_type="extractor",
                input_data={
                    "upstream_results": {
                        "dedup": [{"paper_id": "p1", "title": "T", "abstract": "abs"}]
                    }
                },
            )
        )
        assert result[0]["claims"] == ["c1"]
        ids = [it["evidence_id"] for it in result[0]["evidence_items"]]
        assert "p1:claim:0" in ids
        assert "p1:abstract" in ids


from litagent.context.evidence_selector import EvidenceSelection, EvidenceSelector


class FakeEvidenceSelector:
    """Evidence selector that returns a fixed selection."""

    def __init__(self, selection=None):
        self._selection = selection or EvidenceSelection(
            candidate_count=2,
            selected_items={
                "p1:claim:0": {
                    "evidence_id": "p1:claim:0",
                    "paper_id": "p1",
                    "paper_title": "ProtoNet",
                    "text": "achieves SOTA",
                    "source_locator": "extracted_claim",
                    "confidence": None,
                },
                "p1:claim:1": {
                    "evidence_id": "p1:claim:1",
                    "paper_id": "p1",
                    "paper_title": "ProtoNet",
                    "text": "uses episodic training",
                    "source_locator": "extracted_claim",
                    "confidence": None,
                },
            },
            section_evidence_ids={
                "introduction": ["p1:claim:0"],
                "methods": ["p1:claim:1"],
                "taxonomy": [],
                "experiments": [],
                "open_problems": [],
            },
            method_by_section={
                "introduction": "cross_encoder",
                "methods": "cross_encoder",
                "taxonomy": "lexical_fallback",
                "experiments": "lexical_fallback",
                "open_problems": "lexical_fallback",
            },
            omitted_count=0,
            estimated_tokens=120,
        )
        self.select_call_count = 0

    async def select(self, query, ledger, sections=None, *, max_tokens):
        self.select_call_count += 1
        return self._selection


class _ExplodingSelector:
    """Evidence selector test double that always raises."""

    async def select(self, query, ledger, sections=None, *, max_tokens):
        raise RuntimeError("selector crash")


class TestSynthesisEvidencePipeline:
    """Tests evidence selection in the synthesis pipeline."""

    @staticmethod
    def _task():
        return SubTask(
            task_id="synthesis",
            description="synthesize",
            agent_type="synthesis",
            input_data={
                "query": "few-shot learning",
                "upstream_results": {
                    "extract": [
                        {
                            "paper_id": "p1",
                            "title": "ProtoNet",
                            "abstract": "Few-shot classification.",
                            "claims": ["achieves SOTA"],
                            "metrics": {"accuracy": "93.2%"},
                            "evidence_items": [
                                {
                                    "evidence_id": "p1:claim:0",
                                    "paper_id": "p1",
                                    "paper_title": "ProtoNet",
                                    "text": "achieves SOTA",
                                    "source_locator": "extracted_claim",
                                    "confidence": None,
                                },
                            ],
                        },
                    ],
                    "graph_analysis": {
                        "papers": [
                            {"paper_id": "p1", "tier": 1, "citation_count": 1000}
                        ],
                        "tier_counts": {"tier1": 1, "tier2": 0, "tier3": 0},
                    },
                },
            },
        )

    @pytest.mark.asyncio
    async def test_execute_returns_draft_and_evidence_selection(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(responses=["This is a survey about few-shot learning."])
        fake_sel = FakeEvidenceSelector()
        w = SynthesisWorker(llm=llm, evidence_selector=fake_sel)
        result = await w.execute(self._task())

        assert "draft" in result
        assert "evidence_selection" in result
        assert fake_sel.select_call_count == 1

        ev = result["evidence_selection"]
        restored = EvidenceSelection.from_dict(ev)
        assert restored.candidate_count >= 0
        assert len(restored.selected_items) > 0

    @pytest.mark.asyncio
    async def test_selector_failure_returns_empty_selection_and_error(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(responses=["Scoped evidence summary."])
        w = SynthesisWorker(llm=llm, evidence_selector=_ExplodingSelector())
        result = await w.execute(self._task())

        assert "draft" in result
        assert result.get("error") == "evidence_selection_failed"
        assert "degradation_reasons" in result
        ev = result["evidence_selection"]

        restored = EvidenceSelection.from_dict(ev)
        assert restored.candidate_count > 0
        assert restored.omitted_count == restored.candidate_count
        assert restored.selected_items == {}

    @pytest.mark.asyncio
    async def test_no_selector_records_unavailable_degradation(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(responses=["Survey draft."])
        w = SynthesisWorker(llm=llm)
        result = await w.execute(self._task())

        assert "evidence_selection" in result
        assert "degradation_reasons" in result
        assert "evidence_selector_unavailable" in result["degradation_reasons"]
        restored = EvidenceSelection.from_dict(result["evidence_selection"])
        assert restored.candidate_count > 0
        assert restored.omitted_count == restored.candidate_count

    @pytest.mark.asyncio
    async def test_plain_text_draft_not_json_parse_error(self):
        """Plain-text drafts are not treated as JSON parse failures."""
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(responses=["A plain text survey about few-shot learning."])
        fake_sel = FakeEvidenceSelector()
        w = SynthesisWorker(llm=llm, evidence_selector=fake_sel)
        result = await w.execute(self._task())

        assert result["draft"] == "A plain text survey about few-shot learning."
        assert "JSON parse" not in result.get("error", "")
        assert "JSON parse" not in str(result.get("degradation_reasons", []))

    @pytest.mark.asyncio
    async def test_llm_forged_evidence_selection_is_overwritten(self):
        """Model-supplied evidence selections cannot override trusted data."""
        import json as _json

        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        fake_llm_output = _json.dumps(
            {
                "draft": "Trusted draft text.",
                "evidence_selection": {"candidate_count": 9999},
            }
        )
        llm = MockLLMClient(responses=[fake_llm_output])
        fake_sel = FakeEvidenceSelector()
        w = SynthesisWorker(llm=llm, evidence_selector=fake_sel)
        result = await w.execute(self._task())

        assert result["draft"] == "Trusted draft text."
        ev = result["evidence_selection"]
        assert ev["candidate_count"] != 9999

    def test_paper_catalog_excludes_claims_metrics_abstract(self):
        """The paper catalog excludes untrusted evidence fields."""
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        w = SynthesisWorker(llm=MockLLMClient())
        extractions = [
            {
                "paper_id": "p1",
                "title": "Test Paper",
                "abstract": "should not appear",
                "claims": ["should not appear"],
                "metrics": {"should": "not appear"},
                "evidence_items": [
                    {
                        "evidence_id": "p1:claim:0",
                        "paper_id": "p1",
                        "paper_title": "Test Paper",
                        "text": "valid evidence",
                    },
                ],
            },
        ]
        sel = EvidenceSelection(
            candidate_count=1,
            selected_items={
                "p1:claim:0": {
                    "evidence_id": "p1:claim:0",
                    "paper_id": "p1",
                    "paper_title": "Test Paper",
                    "text": "valid evidence",
                },
            },
            section_evidence_ids={"introduction": ["p1:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=0,
            estimated_tokens=30,
        )
        ctx = w._build_papers_context(
            extractions,
            {"papers": [{"paper_id": "p1", "tier": 2}]},
            sel,
        )
        assert "paper_id=p1" in ctx
        assert "tier=2" in ctx
        assert "Test Paper" in ctx
        assert "[E:p1:claim:0]" in ctx
        assert "should not appear" not in ctx

    def test_paper_catalog_no_title_hash_cross_wire(self):
        """Title-derived IDs do not cross-wire catalog entries."""
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        w = SynthesisWorker(llm=MockLLMClient())

        extractions = [
            {
                "paper_id": "",
                "title": "Paper A",
                "evidence_items": [
                    {
                        "evidence_id": "tA:claim:0",
                        "paper_id": "",
                        "paper_title": "Paper A",
                        "text": "evidence from A",
                    },
                ],
            },
            {
                "paper_id": "",
                "title": "Paper B",
                "evidence_items": [
                    {
                        "evidence_id": "tB:claim:0",
                        "paper_id": "",
                        "paper_title": "Paper B",
                        "text": "evidence from B",
                    },
                ],
            },
        ]
        sel = EvidenceSelection(
            candidate_count=2,
            selected_items={
                "tA:claim:0": {
                    "evidence_id": "tA:claim:0",
                    "paper_id": "",
                    "paper_title": "Paper A",
                    "text": "evidence from A",
                },
                "tB:claim:0": {
                    "evidence_id": "tB:claim:0",
                    "paper_id": "",
                    "paper_title": "Paper B",
                    "text": "evidence from B",
                },
            },
            section_evidence_ids={"introduction": ["tA:claim:0", "tB:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=0,
            estimated_tokens=60,
        )
        ctx = w._build_papers_context(extractions, {}, sel)

        assert "[E:tA:claim:0]" in ctx
        assert "[E:tB:claim:0]" in ctx
        assert "Paper A" in ctx
        assert "Paper B" in ctx

    def test_revise_includes_evidence(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        w = SynthesisWorker(llm=MockLLMClient())
        sel = EvidenceSelection(
            candidate_count=1,
            selected_items={
                "p1:claim:0": {
                    "evidence_id": "p1:claim:0",
                    "paper_title": "T",
                    "text": "some evidence",
                },
            },
            section_evidence_ids={"introduction": ["p1:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=0,
            estimated_tokens=30,
        )
        messages = w.revise("draft text", "needs more support", evidence_selection=sel)
        user = messages[1]["content"]
        assert "draft text" in user
        assert "needs more support" in user
        assert "<evidence_plan>" in user
        assert "[E:p1:claim:0]" in user

    def test_revise_malformed_dict_raises(self):
        """Revision rejects malformed evidence selections."""
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        w = SynthesisWorker(llm=MockLLMClient())
        with pytest.raises(ValueError):
            w.revise("draft", "feedback", evidence_selection={"candidate_count": -1})

    def test_revise_accepts_dict_evidence_selection(self):
        """Revision accepts serialized evidence selections."""
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.llm.client import MockLLMClient

        w = SynthesisWorker(llm=MockLLMClient())
        sel_dict = EvidenceSelection(
            candidate_count=1,
            selected_items={
                "p1:claim:0": {
                    "evidence_id": "p1:claim:0",
                    "paper_title": "T",
                    "text": "ev",
                },
            },
            section_evidence_ids={"introduction": ["p1:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=0,
            estimated_tokens=30,
        ).to_dict()
        messages = w.revise("draft", "feedback", evidence_selection=sel_dict)
        assert "<evidence_plan>" in messages[1]["content"]


class TestReviewerEvidenceCompliance:
    """Tests reviewer evidence-compliance enforcement."""

    @staticmethod
    def _selection():
        return EvidenceSelection(
            candidate_count=2,
            selected_items={
                "p1:claim:0": {
                    "evidence_id": "p1:claim:0",
                    "paper_id": "p1",
                    "paper_title": "ProtoNet",
                    "text": "achieves SOTA",
                },
                "p1:claim:1": {
                    "evidence_id": "p1:claim:1",
                    "paper_id": "p1",
                    "paper_title": "ProtoNet",
                    "text": "uses episodic training",
                },
            },
            section_evidence_ids={
                "introduction": ["p1:claim:0"],
                "methods": ["p1:claim:1"],
            },
            method_by_section={
                "introduction": "cross_encoder",
                "methods": "cross_encoder",
            },
            omitted_count=0,
            estimated_tokens=80,
        )

    @pytest.mark.asyncio
    async def test_execute_reads_evidence_selection_from_upstream(self):
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.85,
                "strengths": ["good"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "test draft [E:p1:claim:0]",
                        "evidence_selection": self._selection().to_dict(),
                    }
                }
            },
        )
        result = await w.execute(task)
        assert "evidence_compliance" in result
        assert result["evidence_compliance"]["passed"] is True

    @pytest.mark.asyncio
    async def test_missing_evidence_selection_flags_and_caps_score(self):
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.95,
                "strengths": ["great"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "test draft",
                    }
                }
            },
        )
        result = await w.execute(task)
        assert result.get("evidence_context_missing") is True
        assert result["score"] <= 0.79
        assert result["verdict"] != "accept"

    @pytest.mark.asyncio
    async def test_draft_unknown_ref_forces_fail_closed(self):
        """Unknown draft references force a failed review."""
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.95,
                "strengths": ["good"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "Great results [E:p99:claim:99].",
                        "evidence_selection": self._selection().to_dict(),
                    }
                }
            },
        )
        result = await w.execute(task)
        assert result["evidence_compliance"]["passed"] is False
        assert "p99:claim:99" in result["evidence_compliance"]["unknown_evidence_ids"]
        assert result["score"] < 0.8
        assert result["verdict"] != "accept"

    @pytest.mark.asyncio
    async def test_model_reports_unknown_ids_forces_fail(self):
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.9,
                "strengths": ["good"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": ["p99:claim:0"],
                },
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "test [E:p1:claim:0]",
                        "evidence_selection": self._selection().to_dict(),
                    }
                }
            },
        )
        result = await w.execute(task)
        assert result["evidence_compliance"]["passed"] is False
        assert "p99:claim:0" in result["evidence_compliance"]["unknown_evidence_ids"]

    @pytest.mark.asyncio
    async def test_missing_evidence_compliance_field_fails_closed(self):
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.95,
                "strengths": ["great"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "test [E:p1:claim:0]",
                        "evidence_selection": self._selection().to_dict(),
                    }
                }
            },
        )
        result = await w.execute(task)
        assert result["evidence_compliance"]["passed"] is False

    @pytest.mark.asyncio
    async def test_unsupported_claims_forces_fail(self):
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.9,
                "strengths": [],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [
                        {
                            "section": "intro",
                            "claim": "SOTA",
                            "reason": "no evidence",
                            "action": "delete",
                        },
                    ],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "test [E:p1:claim:0]",
                        "evidence_selection": self._selection().to_dict(),
                    }
                }
            },
        )
        result = await w.execute(task)
        assert result["evidence_compliance"]["passed"] is False
        assert len(result["evidence_compliance"]["unsupported_claims"]) >= 1

    @pytest.mark.asyncio
    async def test_review_revision_receives_same_evidence(self):
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.85,
                "strengths": ["fixed"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        result = await w.review_revision(
            revised_draft="revised [E:p1:claim:0]",
            previous_review={"score": 0.5, "verdict": "revise"},
            evidence_selection=self._selection(),
        )
        assert result["evidence_compliance"]["passed"] is True

    @pytest.mark.asyncio
    async def test_review_revision_accepts_dict_selection(self):
        import json as _json

        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import MockLLMClient

        review_json = _json.dumps(
            {
                "score": 0.8,
                "strengths": [],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm=llm)
        result = await w.review_revision(
            "draft v2",
            {"score": 0.5},
            evidence_selection=self._selection().to_dict(),
        )
        assert "score" in result

    def test_reviewer_prompt_forbids_ledger_external_suggestions(self):
        from litagent.agents.reviewer import REVIEWER_INSTRUCTIONS

        assert "ONLY describe evidence present in the" in REVIEWER_INSTRUCTIONS
        assert "MUST NOT" in REVIEWER_INSTRUCTIONS

    def test_reviewer_prompt_contains_evidence_compliance_schema(self):
        from litagent.agents.reviewer import REVIEWER_INSTRUCTIONS

        assert "evidence_compliance" in REVIEWER_INSTRUCTIONS
        assert '"passed"' in REVIEWER_INSTRUCTIONS
        assert "unknown_evidence_ids" in REVIEWER_INSTRUCTIONS
        assert "unsupported_claims" in REVIEWER_INSTRUCTIONS

    def test_reviewer_prompt_declares_evidence_untrusted(self):
        from litagent.agents.reviewer import REVIEWER_INSTRUCTIONS

        assert "UNTRUSTED DATA" in REVIEWER_INSTRUCTIONS

    def test_synthesis_prompt_contains_evidence_boundary(self):
        from litagent.agents.synthesis import SYNTHESIS_INSTRUCTIONS

        assert "EVIDENCE BOUNDARIES" in SYNTHESIS_INSTRUCTIONS
        assert "ONLY allowed fact source" in SYNTHESIS_INSTRUCTIONS
        assert "NEVER invent evidence IDs" in SYNTHESIS_INSTRUCTIONS

    def test_synthesis_prompt_declares_evidence_untrusted(self):
        from litagent.agents.synthesis import SYNTHESIS_INSTRUCTIONS

        assert "UNTRUSTED DATA" in SYNTHESIS_INSTRUCTIONS

    def test_revise_prompt_mentions_same_evidence(self):
        from litagent.agents.synthesis import REVISE_INSTRUCTIONS

        assert "SAME" in REVISE_INSTRUCTIONS
        assert "known gap" in REVISE_INSTRUCTIONS


class TestRelevanceSelectRanked:
    """Tests ranked relevance selection."""

    @staticmethod
    def _scored_docs(scores):
        from langchain_core.documents import Document

        from litagent.rag.interfaces import ScoredDoc

        return [
            ScoredDoc(
                doc=Document(
                    page_content=f"text{i}",
                    metadata={"_relevance_source_index": i},
                ),
                score=s,
            )
            for i, s in enumerate(scores)
        ]

    def test_threshold_filters_low_scores(self):
        from litagent.agents.relevance_gate import _select_ranked

        docs = self._scored_docs([0.9, 0.5, 0.3, 0.8])
        sel = _select_ranked(
            docs,
            min_score=0.6,
            min_papers=0,
            max_papers=10,
            mode="threshold",
        )
        assert sel.threshold_qualified_count == 2
        assert sel.output_count == 2
        assert sel.filtered_count == 2
        assert [d.score for d in sel.selected] == [0.9, 0.8]
        assert sel.mode == "threshold"

    def test_null_threshold_is_rank_cap_only(self):
        from litagent.agents.relevance_gate import _select_ranked

        docs = self._scored_docs([0.9, 0.5, 0.3, 0.8])
        sel = _select_ranked(
            docs,
            min_score=None,
            min_papers=0,
            max_papers=3,
            mode="rank_cap_only",
        )
        assert sel.threshold_qualified_count == 4
        assert sel.output_count == 3
        assert sel.mode == "rank_cap_only"

    def test_minimum_refill_when_qualified_too_few(self):
        from litagent.agents.relevance_gate import _select_ranked

        docs = self._scored_docs([0.9, 0.5, 0.3, 0.8, 0.2])
        sel = _select_ranked(
            docs,
            min_score=0.7,
            min_papers=4,
            max_papers=10,
            mode="threshold",
        )
        assert sel.threshold_qualified_count == 2
        assert sel.refilled_count == 2
        assert sel.output_count == 4
        assert [d.score for d in sel.selected] == [0.9, 0.8, 0.5, 0.3]

    def test_maximum_cap_is_always_enforced(self):
        from litagent.agents.relevance_gate import _select_ranked

        docs = self._scored_docs([0.9, 0.8, 0.7, 0.6, 0.5])
        sel = _select_ranked(
            docs,
            min_score=None,
            min_papers=0,
            max_papers=2,
            mode="rank_cap_only",
        )
        assert sel.output_count == 2
        assert len(sel.selected) == 2
        assert [d.doc.metadata["_relevance_source_index"] for d in sel.selected] == [
            0,
            1,
        ]


class TestSearchDegradationContract:
    """Tests stable search-degradation reason codes."""

    def test_source_outcome_has_stable_reason_codes(self):
        from litagent.agents.search import SearchSourceStatus

        valid = {"success", "empty", "rate_limited", "timeout", "failed"}
        assert set(s.value for s in SearchSourceStatus) == valid

    def test_reason_codes_are_stable_strings(self):
        from litagent.agents.search import SearchSourceOutcome, SearchSourceStatus

        outcome = SearchSourceOutcome(
            task_id="search_arxiv_q0",
            source="arxiv",
            status=SearchSourceStatus.RATE_LIMITED,
            result_count=0,
            elapsed_ms=100,
            reason_code="provider_rate_limited",
            error_type="HTTPStatusError",
            from_fallback=False,
        )
        assert outcome.reason_code == "provider_rate_limited"
        assert outcome.source == "arxiv"
        assert outcome.status == SearchSourceStatus.RATE_LIMITED
