import pytest
import json
from litagent.llm.client import MockLLMClient, LLMResponse
from litagent.agents.synthesis import SynthesisWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.orchestrator.task_graph import SubTask


def _make_task_with_data() -> SubTask:
    return SubTask(
        task_id="adversarial_review",
        description="test",
        agent_type="adversarial_review",
        input_data={"upstream_results": {
            "extract": [
                {"paper_id": "p1", "title": "ProtoNet", "abstract": "Few-shot classification.",
                 "claims": ["achieves SOTA"], "metrics": {"accuracy": "93.2%"}},
            ],
            "graph_analysis": {
                "papers": [{"paper_id": "p1", "tier": 1, "citation_count": 1000}],
                "tier_counts": {"tier1": 1, "tier2": 0, "tier3": 0},
                "seminal_papers": [{"paper_id": "p1", "tier": 1}],
            },
        }},
    )


class TestMockLLMClient:
    @pytest.mark.asyncio
    async def test_returns_preset(self):
        client = MockLLMClient(responses=["hello", "world"])
        r1 = await client.chat([{"role": "user", "content": "hi"}])
        r2 = await client.chat([{"role": "user", "content": "hi"}])
        assert r1.content == "hello"
        assert r2.content == "world"

    @pytest.mark.asyncio
    async def test_repeats_last_response(self):
        client = MockLLMClient(responses=["only"])
        r1 = await client.chat([{"role": "user", "content": "1"}])
        r2 = await client.chat([{"role": "user", "content": "2"}])
        assert r1.content == "only"
        assert r2.content == "only"


class TestSynthesisWorker:
    @pytest.mark.asyncio
    async def test_generates_draft(self):
        llm = MockLLMClient(responses=["This is a survey about few-shot learning."])
        w = SynthesisWorker(llm)
        task = _make_task_with_data()
        result = await w.execute(task)
        assert "draft" in result
        assert len(result["draft"]) > 0

    @pytest.mark.asyncio
    async def test_revise_builds_messages(self):
        llm = MockLLMClient()
        w = SynthesisWorker(llm)
        messages = w.revise("draft text", "needs more citations")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "draft text" in messages[1]["content"]


class TestReviewerWorker:
    @pytest.mark.asyncio
    async def test_parses_json_review(self):
        review_json = json.dumps({
            "score": 0.85,
            "strengths": ["good coverage"],
            "weaknesses": ["missing recent papers"],
            "issues": [],
            "missing_coverage": [],
            "verdict": "accept",
        })
        llm = MockLLMClient(responses=[review_json])
        w = ReviewerWorker(llm)
        task = SubTask(task_id="review", description="test", agent_type="reviewer",
                       input_data={"upstream_results": {"synthesis": {"draft": "test draft"}}})
        result = await w.execute(task)
        assert result["score"] == 0.85
        assert result["verdict"] == "accept"

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        """13.7.2 契约更新：无效 JSON → score 0.0 + parse_error 诊断（不再静默 0.3）。"""
        llm = MockLLMClient(responses=["This is not JSON"])
        w = ReviewerWorker(llm)
        task = SubTask(task_id="review", description="test", agent_type="reviewer",
                       input_data={"upstream_results": {"synthesis": {"draft": "test"}}})
        result = await w.execute(task)
        assert result["score"] == 0.0
        assert result["verdict"] == "revise"
        assert "parse_error" in result


class TestAdversarialReviewWorker:
    @pytest.mark.asyncio
    async def test_accepts_on_high_score(self):
        review = json.dumps({
            "score": 0.9, "strengths": ["excellent"],
            "weaknesses": [], "issues": [], "missing_coverage": [],
            "verdict": "accept",
        })
        llm = MockLLMClient(responses=[
            "Great survey draft about few-shot learning.",
            review,
        ])
        w = AdversarialReviewWorker(
            llm=llm, synthesis=SynthesisWorker(llm), reviewer=ReviewerWorker(llm),
            max_rounds=3, pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is True
        assert result["total_rounds"] == 1

    @pytest.mark.asyncio
    async def test_revises_on_low_score(self):
        low_review = json.dumps({
            "score": 0.5, "strengths": [], "weaknesses": ["incomplete"],
            "issues": [{"section": "intro", "issue": "missing motivation", "severity": "major"}],
            "missing_coverage": ["MAML"], "verdict": "revise",
        })
        high_review = json.dumps({
            "score": 0.85, "strengths": ["improved"], "weaknesses": [],
            "issues": [], "missing_coverage": [], "verdict": "accept",
        })
        llm = MockLLMClient(responses=[
            "Initial draft.",
            low_review,
            "Revised draft.",
            high_review,
        ])
        w = AdversarialReviewWorker(
            llm=llm, synthesis=SynthesisWorker(llm), reviewer=ReviewerWorker(llm),
            max_rounds=3, pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is True
        assert result["total_rounds"] == 2

    @pytest.mark.asyncio
    async def test_max_rounds_reached(self):
        low_review = json.dumps({
            "score": 0.4, "strengths": [], "weaknesses": ["still bad"],
            "issues": [], "missing_coverage": [], "verdict": "reject",
        })
        llm = MockLLMClient(responses=[
            "Draft v1.", low_review,
            "Draft v2.", low_review,
            "Draft v3.", low_review,
        ])
        w = AdversarialReviewWorker(
            llm=llm, synthesis=SynthesisWorker(llm), reviewer=ReviewerWorker(llm),
            max_rounds=3, pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is False
        assert result["total_rounds"] == 3

    @pytest.mark.asyncio
    async def test_reviewer_failure_returns_fallback_draft(self):
        """对抗循环中断（reviewer 抛异常，如 reasoning_content 400）→ 保底返回初稿。

        验证方案二B：synthesis 初稿成功后，reviewer 失败不该丢弃已生成的 draft。
        """
        llm = MockLLMClient(responses=["Initial draft that must survive."])
        synthesis = SynthesisWorker(llm)

        class _ExplodingReviewer(ReviewerWorker):
            async def execute(self, task):
                raise RuntimeError("reasoning_content must be passed back (simulated 400)")

        w = AdversarialReviewWorker(
            llm=llm, synthesis=synthesis, reviewer=_ExplodingReviewer(llm),
            max_rounds=3, pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        # 初稿被保底返回，非空
        assert result["final_draft"]                 # 非空
        assert "Initial draft" in result["final_draft"]
        # 对抗未完成 → accepted False，rounds 空（reviewer 首轮就崩）
        assert result["accepted"] is False
        assert result["total_rounds"] == 0
        assert result["final_score"] == 0
