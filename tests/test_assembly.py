"""Phase 12 tests — Assembly + CLI。

覆盖:
  - AdversarialReviewWorker 新签名（接受注入 Worker）
  - LitAgent wiring + minimal run (MockLLM)
  - Graceful degradation (infra 不可用不崩)
  - Memory/RAG backends close()
  - CLI config/tools subcommands
"""

from __future__ import annotations
import asyncio
import json
import sys
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litagent.config import AppConfig, AgentConfig, LoggingConfig, MemoryConfig
from litagent.config import ContextConfig, OrchestratorConfig, LLMConfig
from litagent.config import AdversarialConfig, SafetyConfig, ResilienceConfig, ExtractorConfig
from litagent.llm.client import BaseLLMClient, LLMResponse, MockLLMClient
from litagent.agents.synthesis import SynthesisWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.runner import LitAgent, Infra
from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.procedural import ProceduralMemory
from litagent.rag.claims_index import ClaimsIndex


# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def _minimal_config(**overrides) -> AppConfig:
    """构建最小可用 AppConfig，所有 infra 端点指向无效地址（触发降级）。"""
    return AppConfig(
        agent=AgentConfig(max_loops=3, per_tool_timeout_ms=5000, per_loop_timeout_ms=30000),
        logging=LoggingConfig(level="WARNING"),
        memory=MemoryConfig(
            redis_url=overrides.pop("redis_url", "redis://localhost:9999"),
            qdrant_url=overrides.pop("qdrant_url", "http://localhost:9999"),
            pg_url=overrides.pop("pg_url", "postgresql://none:none@localhost:9999/none"),
        ),
        context=ContextConfig(max_tokens=8000),
        orchestrator=OrchestratorConfig(timeout_ms=30000, max_concurrent=3),
        llm=LLMConfig(base_url="https://api.deepseek.com", model="deepseek-v4-flash"),
        adversarial=AdversarialConfig(max_rounds=1, pass_threshold=0.5),
        safety=SafetyConfig(max_cost_tokens=100000),
        resilience=ResilienceConfig(cb_fail_threshold=3, cb_cooldown_seconds=10),
        extractor=ExtractorConfig(max_concurrent=2),
        **overrides,
    )


# ═══════════════════════════════════════════════════════════
# 12.1 — AdversarialReviewWorker refactored constructor
# ═══════════════════════════════════════════════════════════

class TestAdversarialReviewWorker:
    def test_accepts_injected_workers(self):
        """新签名接受预构建 SynthesisWorker + ReviewerWorker。"""
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


# ═══════════════════════════════════════════════════════════
# 12.2 — close() methods on memory/RAG backends
# ═══════════════════════════════════════════════════════════

class TestBackendClose:
    @pytest.mark.asyncio
    async def test_working_memory_close(self):
        """WorkingMemory.close() 调用 redis.close()。"""
        redis = MagicMock()
        redis.close = AsyncMock()
        wm = WorkingMemory(redis, MemoryConfig())
        await wm.close()
        redis.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_episodic_memory_close(self):
        """EpisodicMemory.close() 调用 qdrant_client.close()。"""
        client = MagicMock()
        client.close = AsyncMock()
        em = EpisodicMemory(client)
        await em.close()
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_semantic_memory_close(self):
        """SemanticMemory.close() 调用 pool.close()。"""
        pool = MagicMock()
        pool.close = AsyncMock()
        sm = SemanticMemory(pool)
        await sm.close()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_procedural_memory_close(self):
        """ProceduralMemory.close() 调用 pool.close()。"""
        pool = MagicMock()
        pool.close = AsyncMock()
        pm = ProceduralMemory(pool)
        await pm.close()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_claims_index_close(self):
        """ClaimsIndex.close() 调用 client.close()。"""
        client = MagicMock()
        client.close = AsyncMock()
        ci = ClaimsIndex(client)
        await ci.close()
        client.close.assert_awaited_once()


# ═══════════════════════════════════════════════════════════
# 12.3 — LitAgent assembly
# ═══════════════════════════════════════════════════════════

class TestLitAgentWiring:
    @pytest.mark.asyncio
    async def test_wires_without_infra(self):
        """无 Redis/Qdrant/PG 时 wiring 不崩，所有 infra 组件为 None。"""
        config = _minimal_config()
        agent = LitAgent(config)

        # 跳过 OpenAI client 真实连接——直接用 _wire 中 mock LLM client
        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm = MockLLMClient(["test"])
            mock_llm_cls.return_value = mock_llm

            await agent._wire()

        assert agent._llm is not None
        assert agent._executor is not None
        assert agent._scheduler is not None
        assert agent._planner is not None
        # Infra should all be None (Redis/Qdrant/PG unreachable at localhost:9999)
        assert agent._infra.memory is None
        assert agent._infra.claims_index is None
        assert agent._infra.retriever is None
        assert agent._wired is True

        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_all_workers_registered(self):
        """Wiring 后 8 个 Worker 全部创建。"""
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
        assert agent._report is not None

        await agent.cleanup()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """async with LitAgent(config) 自动 wire + cleanup。"""
        config = _minimal_config()
        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = MockLLMClient(["test"])
            async with LitAgent(config) as agent:
                assert agent._wired is True
            assert agent._wired is False  # cleanup resets

    @pytest.mark.asyncio
    async def test_wire_idempotent(self):
        """重复 _wire() 不崩（幂等守护）。"""
        config = _minimal_config()
        agent = LitAgent(config)

        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = MockLLMClient(["test"])
            await agent._wire()
            await agent._wire()  # second call should be no-op

        assert agent._wired is True
        await agent.cleanup()


# ═══════════════════════════════════════════════════════════
# 12.3 — LitAgent.run() with MockLLM
# ═══════════════════════════════════════════════════════════

class StubLLMClient(BaseLLMClient):
    """可控响应的 LLM Client——每个 Worker 调不同 prompt 返回不同内容。"""

    def __init__(self, responses: dict[str, str] | None = None):
        self._responses = responses or {}
        self._call_count = 0
        self.calls: list[list[dict]] = []

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        self._call_count += 1
        self.calls.append(messages)
        # Return structured JSON based on call context
        content = self._responses.get(
            "default",
            '{"draft": "A survey draft about the query.", "score": 0.9, "verdict": "accept", "weaknesses": [], "issues": []}',
        )
        return LLMResponse(content=content, model="stub")


class TestLitAgentRun:
    @pytest.mark.asyncio
    async def test_minimal_run_returns_report(self):
        """最小化 run()——MockLLM + no infra → 返回 report dict。"""
        config = _minimal_config()
        agent = LitAgent(config)
        stub_llm = StubLLMClient()

        with patch("litagent.runner.OpenAICompatibleClient") as mock_llm_cls:
            mock_llm_cls.return_value = stub_llm

            await agent._wire()

            # 把搜索/extract worker 的 execute 替换为 stub（避免真实网络调用）
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
        """未显式 _wire() 时 run() 自动调用。"""
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


# ═══════════════════════════════════════════════════════════
# 13.7.0 — Report/Evaluation 契约加固
# ═══════════════════════════════════════════════════════════

class TestExtractReport:
    """_extract_report 按 DAG 契约直接键查找，不遍历不 duck-typing。"""

    def test_prefers_report_worker_output(self):
        """results["report"] 最高优先级，赢过 final_draft 和兜底文案。"""
        config = _minimal_config()
        agent = LitAgent(config)
        results = {
            "report": {
                "survey": "report worker survey",
                "metadata": {"total_rounds": 3, "final_score": 0.9, "accepted": True},
                "review_history": [{"score": 0.9}],
            },
            "adversarial_review": {
                "final_draft": "adversarial draft",  # 应该被 report 覆盖
                "total_rounds": 2,
                "final_score": 0.7,
                "accepted": False,
                "rounds": [{"score": 0.7}],
            },
            "graph_analysis": {
                "papers": [{"title": "Paper A", "tier": 1}],
                "tier_counts": {"tier1": 1, "tier2": 0, "tier3": 0},
                "seminal_papers": [{"title": "Paper A", "tier": 1}],
            },
        }
        out = agent._extract_report(results, "test query")
        assert out["survey"] == "report worker survey"      # ← report 赢
        assert out["metadata"]["accepted"] is True           # ← report metadata 不被覆盖
        assert out["review_history"] == [{"score": 0.9}]     # ← report review_history
        assert "Survey incomplete" not in out["survey"]      # ← 不触发 false positive

    def test_preserves_graph_analysis_contract(self):
        """graph_data 只来自 results["graph_analysis"]，字段为 GraphWorker 真实 schema。"""
        config = _minimal_config()
        agent = LitAgent(config)
        graph_output = {
            "papers": [{"title": "Paper A", "tier": 1}, {"title": "Paper B", "tier": 2}],
            "tier_counts": {"tier1": 1, "tier2": 1, "tier3": 0},
            "seminal_papers": [{"title": "Paper A", "tier": 1}],
        }
        results = {
            "report": {
                "survey": "a survey",
                "metadata": {},
                "review_history": [],
            },
            "graph_analysis": graph_output,
            # 注入碰巧含 nodes key 的非 graph 数据，验证不被误判
            "adversarial_review": {
                "final_draft": "x",
                "nodes": 999,  # 旧逻辑会误抓这个
            },
        }
        out = agent._extract_report(results, "test query")
        # graph_data 必须是 graph_analysis 的真实输出，不是含 nodes 的 adversarial
        assert out["graph_data"] == graph_output
        assert "papers" in out["graph_data"]
        assert "tier_counts" in out["graph_data"]
        assert "seminal_papers" in out["graph_data"]
        # 不应包含旧代码的 nodes 注入
        assert "nodes" not in out["graph_data"]


# ═══════════════════════════════════════════════════════════
# 13.7.1-C — Runner ensure_tables 调用
# ═══════════════════════════════════════════════════════════

class TestRunnerEnsureTables:
    """runner._connect_infra 在 PG 可用时调 ensure_tables。"""

    @pytest.mark.asyncio
    async def test_ensure_tables_called_when_pg_available(self):
        from unittest.mock import AsyncMock, patch
        import asyncpg

        config = _minimal_config()
        agent = LitAgent(config)
        mock_pool = MagicMock(spec=asyncpg.Pool)
        ensure_tables = AsyncMock()
        with (
            patch("litagent.runner.WorkingMemory.connect", new=AsyncMock(
                side_effect=RuntimeError("redis unavailable")
            )),
            patch("qdrant_client.AsyncQdrantClient", side_effect=RuntimeError(
                "qdrant unavailable"
            )),
            patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)),
            patch.object(ProceduralMemory, "ensure_tables", new=ensure_tables),
        ):
            await agent._connect_infra(config)

        ensure_tables.assert_awaited_once()


# ═══════════════════════════════════════════════════════════
# 12.4 — CLI
# ═══════════════════════════════════════════════════════════

class TestCLI:
    def test_config_validate_ok(self):
        """--validate-only 有效 config → exit 0 + stdout 'OK'。"""
        from litagent.cli import _cmd_config
        import argparse

        ns = argparse.Namespace(config=None, validate_only=True)
        with pytest.raises(SystemExit) as exc:
            _cmd_config(ns)
        assert exc.value.code == 0

    def test_tools_json_output(self):
        """tools --format json → valid JSON 输出。"""
        from litagent.cli import _cmd_tools
        import argparse

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
        """--help 输出包含三个子命令。"""
        from litagent.cli import main
        import argparse

        with pytest.raises(SystemExit) as exc:
            with patch("sys.argv", ["litagent", "--help"]):
                main()
        assert exc.value.code == 0


# ═══════════════════════════════════════════════════════════
# 13.7.2-C2 — Quality Gate
# ═══════════════════════════════════════════════════════════

class TestQualityGate:
    """13.7.2-C2：_derive_quality 质量判定。"""

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
        """partial 只表示超时/预算中断，不被 quality 改写。"""
        from litagent.runner import LitAgent
        q = LitAgent._derive_quality({})
        assert q["status"] == "unverified"
