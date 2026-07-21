import pytest
from typing import Any

from litagent.orchestrator.task_graph import SubTask
from litagent.agents.search import SearchWorker
from litagent.agents.dedup import DedupWorker
from litagent.agents.extractor import ExtractorWorker
from litagent.agents.graph import GraphWorker
from litagent.tools.executor import ToolExecutor, ToolResult
from litagent.tools.registry import ToolRegistry


def _mock_executor() -> ToolExecutor:
    """创建返回空结果的 mock executor。"""
    return ToolExecutor(ToolRegistry())


class TestSearchWorker:
    @pytest.mark.asyncio
    async def test_agent_type(self):
        w = SearchWorker(_mock_executor())
        assert w.agent_type == "search"

    @pytest.mark.asyncio
    async def test_parse_arxiv_xml(self):
        from litagent.tools.builtin.search import _parse_arxiv_xml
        xml = '''<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/1703.05175v1</id>
            <title>Prototypical Networks for Few-shot Learning</title>
            <summary>We propose prototypical networks.</summary>
          </entry>
        </feed>'''
        papers = _parse_arxiv_xml(xml)
        assert len(papers) == 1
        assert "Prototypical" in papers[0]["title"]
        assert papers[0]["source"] == "arxiv"

    @pytest.mark.asyncio
    async def test_execute_unknown_source_fallback(self):
        w = SearchWorker(_mock_executor())
        task = SubTask(task_id="t1", description="test", agent_type="search",
                       input_data={"source": "unknown", "query": "test"})
        result = await w.execute(task)
        assert isinstance(result, list)


class TestDedupWorker:
    @pytest.mark.asyncio
    async def test_dedup_by_title(self):
        w = DedupWorker()
        task = SubTask(task_id="dedup", description="dedup", agent_type="dedup",
                       input_data={"upstream_results": {
                           "s1": [
                               {"title": "Paper A", "paper_id": "a1", "citation_count": 10},
                               {"title": "Paper B", "paper_id": "b1"},
                           ],
                           "s2": [
                               {"title": "paper a", "paper_id": "a2", "citation_count": 50},
                               {"title": "Paper C", "paper_id": "c1"},
                           ],
                       }})
        result = await w.execute(task)
        assert len(result) == 3
        paper_a = [p for p in result if "a" in p["title"].lower()][0]
        assert paper_a.get("citation_count", 0) == 50

    @pytest.mark.asyncio
    async def test_dedup_empty_upstream(self):
        w = DedupWorker()
        task = SubTask(task_id="dedup", description="dedup", agent_type="dedup",
                       input_data={"upstream_results": {}})
        result = await w.execute(task)
        assert result == []


class TestExtractorWorker:
    @pytest.mark.asyncio
    async def test_extract_claims(self):
        from litagent.tools.builtin.extract import extract_claims
        claims = await extract_claims("We achieve state-of-the-art results on miniImageNet.")
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
    @pytest.mark.asyncio
    async def test_tier_assignment(self):
        w = GraphWorker()
        task = SubTask(task_id="graph", description="graph", agent_type="graph",
                       input_data={"upstream_results": {
                           "dedup": [
                               {"paper_id": "p1", "title": "Seminal", "citation_count": 1000},
                               {"paper_id": "p2", "title": "Good", "citation_count": 100},
                               {"paper_id": "p3", "title": "New", "citation_count": 5},
                           ]
                       }})
        result = await w.execute(task)
        assert result["tier_counts"]["tier1"] == 1
        assert result["tier_counts"]["tier2"] == 1
        assert result["tier_counts"]["tier3"] == 1
        assert len(result["seminal_papers"]) == 1

    @pytest.mark.asyncio
    async def test_empty_upstream(self):
        w = GraphWorker()
        task = SubTask(task_id="graph", description="graph", agent_type="graph",
                       input_data={"upstream_results": {}})
        result = await w.execute(task)
        assert result["papers"] == []

    @pytest.mark.asyncio
    async def test_agent_type(self):
        w = GraphWorker()
        assert w.agent_type == "graph"


# ═══════════════════════════════════════════════════════════
# 13.7.1 — SearchWorker profile 写入
# ═══════════════════════════════════════════════════════════

from unittest.mock import MagicMock, AsyncMock, patch
from litagent.tools.executor import ToolResult


class TestSearchWorkerProfile:
    """13.7.1：SearchWorker 写入 source profile。全部用 mock，无需 DB。"""

    @pytest.mark.asyncio
    async def test_records_success_profile(self):
        """工具无 error 且有结果 → success=True, empty_result=False。"""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(name="test_tool", args={}, output=[{"title": "P"}], error=None))
        memory = AsyncMock()
        sw = SearchWorker(executor=executor, memory_manager=memory)
        await sw.execute(SubTask(task_id="t1", description="search", agent_type="search",
                                  input_data={"source": "arxiv", "query": "t"}))
        memory.record_search_source_execution.assert_called_once()
        kwargs = memory.record_search_source_execution.call_args.kwargs
        assert kwargs["subject"] == "arxiv"
        assert kwargs["success"] is True
        assert kwargs["empty_result"] is False

    @pytest.mark.asyncio
    async def test_records_empty_result_separately(self):
        """工具无 error 但返回空 → success=True, empty_result=True。"""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(name="test_tool", args={}, output=[], error=None))
        memory = AsyncMock()
        sw = SearchWorker(executor=executor, memory_manager=memory)
        await sw.execute(SubTask(task_id="t2", description="search", agent_type="search",
                                  input_data={"source": "arxiv", "query": "t"}))
        kwargs = memory.record_search_source_execution.call_args.kwargs
        assert kwargs["success"] is True
        assert kwargs["empty_result"] is True

    @pytest.mark.asyncio
    async def test_records_failure_profile(self):
        """executor 返回 error → success=False。"""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(name="test_tool", args={}, output=[], error="timeout"))
        memory = AsyncMock()
        sw = SearchWorker(executor=executor, memory_manager=memory)
        await sw.execute(SubTask(task_id="t3", description="search", agent_type="search",
                                  input_data={"source": "huggingface", "query": "t"}))
        kwargs = memory.record_search_source_execution.call_args.kwargs
        assert kwargs["subject"] == "huggingface"
        assert kwargs["success"] is False
        assert kwargs["error_type"] == "timeout"

    @pytest.mark.asyncio
    async def test_memory_failure_does_not_break_search(self):
        """memory 写抛异常 → search 正常返回，不传播。"""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(name="test_tool", args={}, output=[{"title": "X"}], error=None))
        memory = AsyncMock()
        memory.record_search_source_execution.side_effect = RuntimeError("db down")
        sw = SearchWorker(executor=executor, memory_manager=memory)
        result = await sw.execute(SubTask(task_id="t4", description="search", agent_type="search",
                                           input_data={"source": "arxiv", "query": "t"}))
        assert len(result) > 0  # 正常产出，不崩

    @pytest.mark.asyncio
    async def test_noop_when_memory_is_none(self):
        """memory_manager=None → search 正常返回，不崩。"""
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(
            return_value=ToolResult(name="test_tool", args={}, output=[{"title": "X"}], error=None))
        sw = SearchWorker(executor=executor, memory_manager=None)
        result = await sw.execute(SubTask(task_id="t5", description="search", agent_type="search",
                                           input_data={"source": "arxiv", "query": "t"}))
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════
# 13.7.1-B — rank_search_sources + Planner priority 映射
# ═══════════════════════════════════════════════════════════

from unittest.mock import AsyncMock, MagicMock
from litagent.memory.manager import MemoryManager


class TestRankSearchSources:
    """13.7.1-B：MemoryManager.rank_search_sources 排序契约。"""

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
        memory = self._memory_with_profiles([
            {"subject": "arxiv", "success_count": 90, "failure_count": 1,
             "empty_result_count": 5, "rate_limit_count": 0, "timeout_count": 0,
             "execution_count": 96},
            {"subject": "huggingface", "success_count": 2, "failure_count": 8,
             "empty_result_count": 0, "rate_limit_count": 2, "timeout_count": 5,
             "execution_count": 15},
            {"subject": "semantic_scholar", "success_count": 10, "failure_count":10,
             "empty_result_count": 5, "rate_limit_count": 5, "timeout_count": 1,
             "execution_count": 30},
        ])
        result = await memory.rank_search_sources(
            ["huggingface", "semantic_scholar", "arxiv"])
        assert set(result) == {"huggingface", "semantic_scholar", "arxiv"}
        assert result[0] == "arxiv"

    @pytest.mark.asyncio
    async def test_rank_preserves_input_order_when_samples_insufficient(self):
        memory = self._memory_with_profiles([
            {"subject": "hf", "success_count": 1, "failure_count": 0,
             "empty_result_count": 0, "rate_limit_count": 0, "timeout_count": 0,
             "execution_count": 1},
        ])
        result = await memory.rank_search_sources(["arxiv", "hf"], min_samples=5)
        assert result == ["arxiv", "hf"]

    @pytest.mark.asyncio
    async def test_rank_preserves_input_order_when_profile_read_fails(self):
        memory = self._memory_with_profiles(error=RuntimeError("db down"))
        result = await memory.rank_search_sources(["arxiv", "huggingface"])
        assert result == ["arxiv", "huggingface"]


class TestPlannerPriorityMapping:
    """13.7.1-B：SurveyPlanner 把 rank 映射到 SubTask.priority。"""

    @pytest.mark.asyncio
    async def test_planner_maps_rank_to_search_task_priority(self):
        from litagent.agents.planner import SurveyPlanner
        from litagent.config import PlannerConfig
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(['{"sub_queries": ["a"]}'])
        memory = AsyncMock(spec=MemoryManager)
        memory.rank_search_sources = AsyncMock(
            return_value=["huggingface", "arxiv"])

        cfg = PlannerConfig(use_procedural_profiles=True)
        planner = SurveyPlanner(llm=llm, config=cfg, memory_manager=memory)
        graph = await planner.plan("test")

        search_tasks = [t for t in graph.tasks.values()
                        if t.agent_type == "search"]
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

        search_sources = {t.input_data["source"]
                          for t in graph.tasks.values()
                          if t.agent_type == "search"}
        assert "arxiv" in search_sources
        assert "huggingface" in search_sources
        memory.rank_search_sources.assert_not_called()


# ═══════════════════════════════════════════════════════════
# 13.7.2-C1 — Synthesis evidence boundary + Reviewer structured output
# ═══════════════════════════════════════════════════════════

class TestSynthesisEvidenceBoundary:
    """13.7.2-C1：Synthesis evidence 边界约束。"""

    def test_prompt_contains_evidence_boundary_instructions(self):
        from litagent.agents.synthesis import SYNTHESIS_INSTRUCTIONS
        assert "SCOPED EVIDENCE SUMMARY" in SYNTHESIS_INSTRUCTIONS
        assert "evidence not provided" in SYNTHESIS_INSTRUCTIONS
        assert "fabricate" in SYNTHESIS_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_execute_uses_configured_react_loop_limit(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.config import AgentConfig

        react = MagicMock()
        react.run = AsyncMock(return_value='{"draft": "ok"}')
        with patch("litagent.agents.synthesis.ReActRunner", return_value=react) as runner_cls:
            worker = SynthesisWorker(
                llm=MagicMock(), agent_config=AgentConfig(max_loops=2)
            )
            await worker.execute(SubTask(
                task_id="synthesis",
                description="synthesize",
                agent_type="synthesis",
                input_data={"query": "few-shot", "upstream_results": {}},
            ))

        assert runner_cls.call_args.kwargs["config"].max_loops == 2


class TestReviewerStructuredOutput:
    """13.7.2-C1：Reviewer 结构化 JSON 调用。"""

    @pytest.mark.asyncio
    async def test_reviewer_uses_one_structured_llm_call_not_react_loop(self):
        from unittest.mock import AsyncMock, MagicMock
        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import BaseLLMClient, LLMResponse
        import json as _json

        llm = MagicMock(spec=BaseLLMClient)
        review_json = _json.dumps({
            "score": 0.7, "strengths": ["clear"], "weaknesses": ["short"],
            "issues": [], "missing_coverage": [], "verdict": "revise",
        })
        llm.chat = AsyncMock(return_value=LLMResponse(
            content=review_json, model="test"))

        reviewer = ReviewerWorker(llm=llm)
        result = await reviewer.execute(
            SubTask(task_id="r1", description="review", agent_type="reviewer",
                    input_data={"upstream_results": {
                        "synthesis": {"draft": "test draft"}
                    }}))
        assert result["score"] == 0.7
        llm.chat.assert_called_once()
        assert llm.chat.call_args.kwargs.get("response_format") == {
            "type": "json_object"}

    @pytest.mark.asyncio
    async def test_reviewer_invalid_json_returns_parse_error_diagnostic(self):
        from unittest.mock import AsyncMock, MagicMock
        from litagent.agents.reviewer import ReviewerWorker
        from litagent.llm.client import BaseLLMClient, LLMResponse

        llm = MagicMock(spec=BaseLLMClient)
        llm.chat = AsyncMock(return_value=LLMResponse(
            content="not json", model="test"))

        reviewer = ReviewerWorker(llm=llm)
        result = await reviewer.execute(
            SubTask(task_id="r2", description="review", agent_type="reviewer",
                    input_data={"upstream_results": {
                        "synthesis": {"draft": "test draft"}
                    }}))
        assert "parse_error" in result


# ═══════════════════════════════════════════════════════════
# 13.7.3.1 — Query Intent + 独立 Recall
# ═══════════════════════════════════════════════════════════

class TestQueryIntent:
    """classify_query_intent 纯规则分类——无网络无 LLM。"""

    def test_topic(self):
        from litagent.agents.planner import classify_query_intent, QueryIntent
        assert classify_query_intent("few-shot learning in CV") == QueryIntent.TOPIC
        assert classify_query_intent("attention is all you need 2017") == QueryIntent.TOPIC

    def test_arxiv_new_style(self):
        from litagent.agents.planner import classify_query_intent, QueryIntent
        assert classify_query_intent("2401.00001") == QueryIntent.ARXIV_ID
        assert classify_query_intent("2401.00001v2") == QueryIntent.ARXIV_ID
        assert classify_query_intent("arXiv:2401.00001") == QueryIntent.ARXIV_ID

    def test_arxiv_legacy(self):
        from litagent.agents.planner import classify_query_intent, QueryIntent
        assert classify_query_intent("cs.CL/0301001") == QueryIntent.ARXIV_ID
        assert classify_query_intent("hep-th/9901001v1") == QueryIntent.ARXIV_ID

    def test_doi(self):
        from litagent.agents.planner import classify_query_intent, QueryIntent
        assert classify_query_intent("10.1038/nature12373") == QueryIntent.DOI
        assert classify_query_intent("doi:10.1145/3292500.3330701") == QueryIntent.DOI

    def test_url_wins_over_embedded_id(self):
        from litagent.agents.planner import classify_query_intent, QueryIntent
        assert classify_query_intent("https://arxiv.org/abs/2401.00001") == QueryIntent.URL
        assert classify_query_intent("http://example.com/paper") == QueryIntent.URL


class TestPlannerRecallTasks:
    """13.7.3.1：每个规范化 sub-query 恰好一个 recall task；ID/DOI/URL 不调 LLM。"""

    @pytest.fixture(autouse=True)
    def _no_ss_key(self, monkeypatch):
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)

    @pytest.mark.asyncio
    async def test_n_subqueries_yield_n_recall_tasks(self):
        """LLM 返回 2 个 → merge 原始 query 后 3 个规范化 sub-query → 3 recall（非 N×源数）。"""
        from litagent.agents.planner import SurveyPlanner
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(['{"sub_queries": ["angle a", "angle b"]}'])
        planner = SurveyPlanner(llm=llm)
        graph = await planner.plan("few-shot learning")

        recall_tasks = [t for t in graph.tasks.values() if t.agent_type == "recall"]
        search_tasks = [t for t in graph.tasks.values() if t.agent_type == "search"]
        assert len(recall_tasks) == 3       # [原始, a, b]
        assert len(search_tasks) == 6       # 2 源 × 3 sub-query
        # recall 输入契约
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

        llm.chat.assert_not_called()        # 不送 decomposition LLM
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
        """dedup 依赖 external + recall 全部完成才 ready。"""
        from litagent.agents.planner import SurveyPlanner
        planner = SurveyPlanner()
        graph = await planner.plan("test topic")

        # 只完成 search，不完成 recall → dedup 不 ready
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
    """13.7.3.1：RecallWorker——基础设施适配，可恢复失败一律返回 []。"""

    @staticmethod
    def _scored_doc():
        class _Doc:
            metadata = {"arxiv_id": "2401.00001", "title": "T"}
            page_content = "abstract text " * 100

        class _SD:
            doc = _Doc()
            score = 0.9

        return _SD()

    @pytest.mark.asyncio
    async def test_no_retriever_returns_empty(self):
        from litagent.agents.recall import RecallWorker
        w = RecallWorker(retriever=None)
        assert w.agent_type == "recall"
        result = await w.execute(SubTask(task_id="recall_q0", description="r",
                                         agent_type="recall",
                                         input_data={"query": "x", "top_k": 5}))
        assert result == []

    @pytest.mark.asyncio
    async def test_retriever_error_returns_empty(self):
        """recall 失败不拖累 dedup fan-in（external results 仍能进 dedup）。"""
        from litagent.agents.recall import RecallWorker
        retriever = MagicMock()
        retriever.search = AsyncMock(side_effect=RuntimeError("qdrant 400"))
        w = RecallWorker(retriever=retriever)
        result = await w.execute(SubTask(task_id="recall_q0", description="r",
                                         agent_type="recall",
                                         input_data={"query": "x", "top_k": 5}))
        assert result == []

    @pytest.mark.asyncio
    async def test_result_keeps_dedup_contract(self):
        from litagent.agents.recall import RecallWorker
        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[self._scored_doc()])
        w = RecallWorker(retriever=retriever)
        result = await w.execute(SubTask(task_id="recall_q0", description="r",
                                         agent_type="recall",
                                         input_data={"query": "x", "top_k": 5}))
        assert len(result) == 1
        p = result[0]
        assert p["paper_id"] == "2401.00001"
        assert p["title"] == "T"
        assert p["source"] == "rag_index"
        assert p["score"] == 0.9
        assert len(p["abstract"]) <= 500

    @pytest.mark.asyncio
    async def test_retriever_calls_equal_normalized_subquery_count(self, monkeypatch):
        """两源并行时 retriever 调用数 = 规范化 sub-query 数（不是 ×源数）。"""
        monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
        from litagent.agents.planner import SurveyPlanner
        from litagent.agents.recall import RecallWorker
        from litagent.llm.client import MockLLMClient

        llm = MockLLMClient(['{"sub_queries": ["a", "b"]}'])
        planner = SurveyPlanner(llm=llm)
        graph = await planner.plan("topic")     # 规范化后 3 个 sub-query

        retriever = MagicMock()
        retriever.search = AsyncMock(return_value=[])
        w = RecallWorker(retriever=retriever)
        for t in graph.tasks.values():
            if t.agent_type == "recall":
                await w.execute(t)
        assert retriever.search.await_count == 3


class TestSearchWorkerPureExternal:
    """13.7.3.1：SearchWorker 只做外部 API，不再内嵌 RAG。"""

    def test_search_worker_has_no_retriever(self):
        import inspect
        sig = inspect.signature(SearchWorker.__init__)
        assert "retriever" not in sig.parameters

    @pytest.mark.asyncio
    async def test_returns_api_papers_only(self):
        executor = MagicMock(spec=ToolExecutor)
        executor.execute = AsyncMock(return_value=ToolResult(
            name="t", args={}, output=[{"title": "P", "source": "arxiv"}], error=None))
        sw = SearchWorker(executor=executor, memory_manager=None)
        result = await sw.execute(SubTask(task_id="s", description="s", agent_type="search",
                                          input_data={"source": "arxiv", "query": "q"}))
        assert result == [{"title": "P", "source": "arxiv"}]


# ═══════════════════════════════════════════════════════════
# 13.7.3.4 — Evidence Ledger + Synthesis 引用 + Faithfulness 诊断
# ═══════════════════════════════════════════════════════════

class TestEvidenceLedger:
    def test_ids_stable_and_unique(self):
        from litagent.evidence import build_evidence_items
        ext = {"paper_id": "p1", "title": "T", "abstract": "abs",
               "claims": ["c one", "c two"]}
        items = build_evidence_items(ext)
        ids = [it["evidence_id"] for it in items]
        assert ids == ["p1:claim:0", "p1:claim:1", "p1:abstract"]
        assert len(ids) == len(set(ids))                      # 无重复
        assert build_evidence_items(ext) == items             # 跨调用稳定
        assert items[0]["text"] == "c one"                    # 指向同一 text
        assert items[0]["source_locator"] == "extracted_claim"
        assert items[2]["source_locator"] == "abstract"
        assert items[0]["confidence"] is None                 # 不伪造置信度

    def test_missing_paper_id_uses_title_hash(self):
        from litagent.evidence import build_evidence_items
        ext = {"paper_id": "", "title": "Some Paper", "abstract": "", "claims": ["c"]}
        a = build_evidence_items(ext)
        b = build_evidence_items(dict(ext))
        assert a[0]["evidence_id"] == b[0]["evidence_id"]     # 同 title 跨 run 稳定
        assert a[0]["evidence_id"].startswith("t")

    def test_collect_and_find_unknown_refs(self):
        from litagent.evidence import build_evidence_items, collect_ledger, find_unknown_refs
        ext = {"paper_id": "p1", "title": "T", "abstract": "abs", "claims": ["c"]}
        ext["evidence_items"] = build_evidence_items(ext)
        ledger = collect_ledger([ext])
        assert "p1:claim:0" in ledger

        text = "Good [E:p1:claim:0]. Bad [E:fake:claim:9]."
        assert find_unknown_refs(text, ledger) == ["fake:claim:9"]
        assert find_unknown_refs("no refs here", ledger) == []

    def test_format_ledger_lines(self):
        from litagent.evidence import build_evidence_items, collect_ledger, format_ledger
        ext = {"paper_id": "p1", "title": "T", "abstract": "", "claims": ["c one"]}
        ext["evidence_items"] = build_evidence_items(ext)
        text = format_ledger(collect_ledger([ext]))
        assert "[E:p1:claim:0]" in text
        assert "(T)" in text
        assert "c one" in text

    def test_format_ledger_truncates_at_max_chars(self):
        """诊断/rewrite prompt 场景：ledger 无界 → 按行截断并标注省略。"""
        from litagent.evidence import build_evidence_items, collect_ledger, format_ledger
        ext = {"paper_id": "p1", "title": "T", "abstract": "",
               "claims": [f"claim number {i} " + "x" * 80 for i in range(50)]}
        ext["evidence_items"] = build_evidence_items(ext)
        text = format_ledger(collect_ledger([ext]), max_chars=500)
        assert len(text) < 700              # 500 + 省略标注行
        assert "omitted" in text
        # 无截断时全量输出
        full = format_ledger(collect_ledger([ext]))
        assert "omitted" not in full


class TestSynthesisEvidenceCitation:
    """13.7.3.4：Synthesis 的 ledger 层 + 引用规则 + rewrite messages。"""

    def test_instructions_contain_citation_rules(self):
        from litagent.agents.synthesis import SYNTHESIS_INSTRUCTIONS
        assert "EVIDENCE CITATION RULES" in SYNTHESIS_INSTRUCTIONS
        assert "[E:" in SYNTHESIS_INSTRUCTIONS
        assert "NEVER invent" in SYNTHESIS_INSTRUCTIONS

    def test_revise_instructions_forbid_new_papers(self):
        from litagent.agents.synthesis import REVISE_INSTRUCTIONS
        assert "known gap" in REVISE_INSTRUCTIONS

    @pytest.mark.asyncio
    async def test_ledger_layer_renders_evidence(self):
        from litagent.agents.synthesis import SynthesisWorker
        from litagent.evidence import build_evidence_items
        ext = {"paper_id": "p1", "title": "T", "abstract": "abs", "claims": ["c"]}
        ext["evidence_items"] = build_evidence_items(ext)
        w = SynthesisWorker(llm=MagicMock())
        out = await w._ledger_layer({"extractions": [ext]})
        assert "<evidence_ledger>" in out
        assert "[E:p1:claim:0]" in out

    def test_rewrite_messages_carry_draft_ledger_diagnostics(self):
        from litagent.agents.synthesis import SynthesisWorker, REWRITE_INSTRUCTIONS
        assert "NEVER add new paper titles" in REWRITE_INSTRUCTIONS
        w = SynthesisWorker(llm=MagicMock())
        messages = w.rewrite_with_evidence("the draft", "[E:p1:claim:0] (T) c",
                                           '[{"claim_text": "bad"}]')
        assert messages[0]["role"] == "system"
        user = messages[1]["content"]
        assert "<current_draft>" in user
        assert "<evidence_ledger>" in user
        assert "<unsupported_claims>" in user
        assert "the draft" in user


class TestFaithfulnessDiagnostic:
    """13.7.3.4：faithfulness 未过阈值 → 定位 unsupported claims；无法评估不伪造空列表。"""

    @staticmethod
    def _cfg():
        from litagent.config import AppConfig, AgentConfig, LoggingConfig
        return AppConfig(agent=AgentConfig(), logging=LoggingConfig())

    @staticmethod
    def _ledger():
        from litagent.evidence import build_evidence_items, collect_ledger
        ext = {"paper_id": "p1", "title": "T", "abstract": "abs", "claims": ["c"]}
        ext["evidence_items"] = build_evidence_items(ext)
        return collect_ledger([ext])

    @pytest.mark.asyncio
    async def test_diagnose_returns_unsupported_claims(self):
        import json as _json
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
        from litagent.eval.base import CTX_EVIDENCE
        from litagent.llm.client import BaseLLMClient, LLMResponse

        llm = MagicMock(spec=BaseLLMClient)
        llm.chat = AsyncMock(return_value=LLMResponse(
            content=_json.dumps({"unsupported_claims": [
                {"claim_text": "X beats Y", "evidence_ids": ["p1:claim:0"],
                 "reason": "overstates"}]}),
            model="test"))
        ev = RagasFaithfulnessEvaluator(self._cfg(), llm=llm)
        out = await ev._diagnose("survey", {CTX_EVIDENCE: self._ledger()})
        assert out["unsupported_claims"][0]["claim_text"] == "X beats Y"
        assert out["unsupported_claims"][0]["evidence_ids"] == ["p1:claim:0"]
        # 诊断走 json_object 结构化调用
        assert llm.chat.call_args.kwargs.get("response_format") == {"type": "json_object"}

    @pytest.mark.asyncio
    async def test_diagnose_malformed_marks_skipped_not_empty(self):
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
        from litagent.eval.base import CTX_EVIDENCE
        from litagent.llm.client import BaseLLMClient, LLMResponse

        llm = MagicMock(spec=BaseLLMClient)
        llm.chat = AsyncMock(return_value=LLMResponse(
            content='{"unsupported_claims": "not a list"}', model="test"))
        ev = RagasFaithfulnessEvaluator(self._cfg(), llm=llm)
        out = await ev._diagnose("survey", {CTX_EVIDENCE: self._ledger()})
        assert "unsupported_claims" not in out       # 不用空列表掩盖失败
        assert "diagnostic_skipped" in out

    @pytest.mark.asyncio
    async def test_diagnose_without_llm_or_ledger_marks_skipped(self):
        from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
        from litagent.eval.base import CTX_EVIDENCE

        ev = RagasFaithfulnessEvaluator(self._cfg())     # 无 llm
        out = await ev._diagnose("survey", {CTX_EVIDENCE: self._ledger()})
        assert "diagnostic_skipped" in out

        ev2 = RagasFaithfulnessEvaluator(self._cfg(), llm=MagicMock())
        out2 = await ev2._diagnose("survey", {CTX_EVIDENCE: {}})   # 无 ledger
        assert "diagnostic_skipped" in out2


class TestExtractorEvidenceItems:
    """13.7.3.4：Extractor 每篇论文产出 evidence_items（claims 兼容保留）。"""

    @pytest.mark.asyncio
    async def test_extractions_include_evidence_items(self):
        from litagent.agents.extractor import ExtractorWorker

        strategy = MagicMock()
        strategy.extract = AsyncMock(return_value={"claims": ["c1"], "metrics": {}})
        w = ExtractorWorker(strategy=strategy)
        result = await w.execute(SubTask(
            task_id="extract", description="e", agent_type="extractor",
            input_data={"upstream_results": {"dedup": [
                {"paper_id": "p1", "title": "T", "abstract": "abs"}]}}))
        assert result[0]["claims"] == ["c1"]                  # 旧 consumer 兼容
        ids = [it["evidence_id"] for it in result[0]["evidence_items"]]
        assert "p1:claim:0" in ids
        assert "p1:abstract" in ids
