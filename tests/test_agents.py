import pytest
from typing import Any

from litagent.orchestrator.task_graph import SubTask
from litagent.agents.search import SearchWorker
from litagent.agents.dedup import DedupWorker
from litagent.agents.extractor import ExtractorWorker
from litagent.agents.graph import GraphWorker
from litagent.agents.report import ReportWorker
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


class TestReportWorker:
    @pytest.mark.asyncio
    async def test_generates_report(self):
        w = ReportWorker()
        task = SubTask(
            task_id="report", description="report", agent_type="report",
            input_data={"upstream_results": {
                "adversarial_review": {
                    "final_draft": "This is the final survey.",
                    "rounds": [
                        {"round": 1, "review": {"score": 0.5, "verdict": "revise", "weaknesses": ["incomplete"], "issues": []}},
                        {"round": 2, "review": {"score": 0.9, "verdict": "accept", "weaknesses": [], "issues": []}},
                    ],
                    "total_rounds": 2,
                    "final_score": 0.9,
                    "accepted": True,
                },
            }},
        )
        result = await w.execute(task)
        assert result["survey"] == "This is the final survey."
        assert result["metadata"]["accepted"] is True
        assert result["metadata"]["total_rounds"] == 2
        assert len(result["review_history"]) == 2
        assert result["review_history"][0]["score"] == 0.5
        assert result["review_history"][1]["verdict"] == "accept"

    @pytest.mark.asyncio
    async def test_empty_upstream(self):
        w = ReportWorker()
        task = SubTask(
            task_id="report", description="report", agent_type="report",
            input_data={"upstream_results": {}},
        )
        result = await w.execute(task)
        assert result["survey"] == ""
        assert result["metadata"]["accepted"] is False

    @pytest.mark.asyncio
    async def test_agent_type(self):
        w = ReportWorker()
        assert w.agent_type == "report"


# ═══════════════════════════════════════════════════════════
# 13.7.1 — SearchWorker profile 写入
# ═══════════════════════════════════════════════════════════

from unittest.mock import MagicMock, AsyncMock
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
