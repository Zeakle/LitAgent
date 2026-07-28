"""Tests for adversarial synthesis, review, and acceptance."""

import json

import pytest

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
        input_data={
            "upstream_results": {
                "extract": [
                    {
                        "paper_id": "p1",
                        "title": "ProtoNet",
                        "abstract": "Few-shot classification.",
                        "claims": ["achieves SOTA"],
                        "metrics": {"accuracy": "93.2%"},
                    },
                ],
                "graph_analysis": {
                    "papers": [{"paper_id": "p1", "tier": 1, "citation_count": 1000}],
                    "tier_counts": {"tier1": 1, "tier2": 0, "tier3": 0},
                    "seminal_papers": [{"paper_id": "p1", "tier": 1}],
                },
            }
        },
    )


class TestMockLLMClient:
    """Tests deterministic mock LLM responses."""

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
    """Tests draft generation and revision."""

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
        from litagent.context.evidence_selector import EvidenceSelection

        llm = MockLLMClient()
        w = SynthesisWorker(llm)
        sel = EvidenceSelection(
            candidate_count=1,
            selected_items={
                "p1:claim:0": {
                    "evidence_id": "p1:claim:0",
                    "paper_title": "T",
                    "text": "ev",
                }
            },
            section_evidence_ids={"introduction": ["p1:claim:0"]},
            method_by_section={"introduction": "cross_encoder"},
            omitted_count=0,
            estimated_tokens=30,
        )
        messages = w.revise(
            "draft text", "needs more citations", evidence_selection=sel
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "draft text" in messages[1]["content"]


class TestReviewerWorker:
    """Tests structured reviewer output."""

    @pytest.mark.asyncio
    async def test_parses_json_review(self):
        review_json = json.dumps(
            {
                "score": 0.85,
                "strengths": ["good coverage"],
                "weaknesses": ["missing recent papers"],
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
        w = ReviewerWorker(llm)
        sel = _EVIDENCE_SELECTION.to_dict()
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "test draft",
                        "evidence_selection": sel,
                    }
                }
            },
        )
        result = await w.execute(task)
        assert result["score"] == 0.85
        assert result["verdict"] == "accept"

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        """Invalid reviewer JSON produces a safe fallback result."""
        llm = MockLLMClient(responses=["This is not JSON"])
        w = ReviewerWorker(llm)
        sel = _EVIDENCE_SELECTION.to_dict()
        task = SubTask(
            task_id="review",
            description="test",
            agent_type="reviewer",
            input_data={
                "upstream_results": {
                    "synthesis": {
                        "draft": "test",
                        "evidence_selection": sel,
                    }
                }
            },
        )
        result = await w.execute(task)
        assert result["score"] == 0.0
        assert result["verdict"] == "revise"
        assert "parse_error" in result


class TestAdversarialReviewWorker:
    """Tests adversarial review orchestration."""

    @pytest.mark.asyncio
    async def test_accepts_on_high_score(self):
        """A passing review ends the adversarial loop."""
        import json as _json

        from litagent.context.evidence_selector import EvidenceSelection

        fake_sel = FakeEvidenceSelector()
        review = _json.dumps(
            {
                "score": 0.9,
                "strengths": ["excellent"],
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
        llm = MockLLMClient(
            responses=[
                "Great survey draft about few-shot learning [E:p1:claim:0].",
                review,
            ]
        )
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is True
        assert result["total_rounds"] == 1

    @pytest.mark.asyncio
    async def test_revises_on_low_score(self):
        """A failing review triggers a draft revision."""
        import json as _json

        fake_sel = FakeEvidenceSelector()
        low_review = _json.dumps(
            {
                "score": 0.5,
                "strengths": [],
                "weaknesses": ["incomplete"],
                "issues": [
                    {
                        "section": "intro",
                        "issue": "missing motivation",
                        "severity": "major",
                    }
                ],
                "missing_coverage": [],
                "verdict": "revise",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        high_review = _json.dumps(
            {
                "score": 0.85,
                "strengths": ["improved"],
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
        llm = MockLLMClient(
            responses=[
                "Initial draft [E:p1:claim:0].",
                low_review,
                "Revised draft [E:p1:claim:0].",
                high_review,
            ]
        )
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is True
        assert result["total_rounds"] == 2

    @pytest.mark.asyncio
    async def test_max_rounds_reached(self):
        """The review loop stops at the configured round limit."""
        import json as _json

        fake_sel = FakeEvidenceSelector()
        low_review = _json.dumps(
            {
                "score": 0.4,
                "strengths": [],
                "weaknesses": ["still bad"],
                "issues": [],
                "missing_coverage": [],
                "verdict": "reject",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(
            responses=[
                "Draft v1 [E:p1:claim:0].",
                low_review,
                "Draft v2 [E:p1:claim:0].",
                low_review,
                "Draft v3 [E:p1:claim:0].",
                low_review,
            ]
        )
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is False
        assert result["total_rounds"] == 3

    @pytest.mark.asyncio
    async def test_reviewer_failure_returns_fallback_draft(self):
        """A reviewer failure preserves the initial draft."""
        llm = MockLLMClient(responses=["Initial draft that must survive."])
        synthesis = SynthesisWorker(llm)

        class _ExplodingReviewer(ReviewerWorker):
            """Reviewer test double that always raises."""

            async def execute(self, task):
                raise RuntimeError(
                    "reasoning_content must be passed back (simulated 400)"
                )

        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=synthesis,
            reviewer=_ExplodingReviewer(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)

        assert result["final_draft"]
        assert "Initial draft" in result["final_draft"]

        assert result["accepted"] is False
        assert result["total_rounds"] == 0
        assert result["final_score"] == 0


from litagent.context.evidence_selector import EvidenceSelection

_EVIDENCE_SELECTION = EvidenceSelection(
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
    estimated_tokens=80,
)


class FakeEvidenceSelector:
    """Evidence selector that returns a fixed selection."""

    def __init__(self, selection=None):
        self._selection = selection or _EVIDENCE_SELECTION
        self.select_call_count = 0

    async def select(self, query, ledger, sections=None, *, max_tokens):
        self.select_call_count += 1
        return self._selection


class TestAdversarialEvidenceGate:
    """Tests the score, verdict, and evidence-compliance acceptance gate."""

    @pytest.mark.asyncio
    async def test_all_three_pass_is_accepted(self):
        """Acceptance requires all three gate conditions to pass."""
        import json as _json

        fake_sel = FakeEvidenceSelector()
        review = _json.dumps(
            {
                "score": 0.9,
                "strengths": ["great"],
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
        llm = MockLLMClient(
            responses=[
                "Great survey [E:p1:claim:0].",
                review,
            ]
        )
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is True
        assert result["total_rounds"] == 1
        assert "evidence_selection" in result

    @pytest.mark.asyncio
    async def test_high_score_and_accept_but_evidence_failed_is_rejected(self):
        """Failed evidence compliance overrides score and verdict."""
        import json as _json

        fake_sel = FakeEvidenceSelector()
        review = _json.dumps(
            {
                "score": 0.95,
                "strengths": ["great"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": False,
                    "unsupported_claims": [
                        {
                            "section": "intro",
                            "claim": "fake",
                            "reason": "no evidence",
                            "action": "delete",
                        },
                    ],
                    "unknown_evidence_ids": ["p99:claim:0"],
                },
            }
        )
        llm = MockLLMClient(
            responses=[
                "Survey with unsupported claims.",
                review,
                "Revised draft.",
                review,
                "Final draft.",
                review,
            ]
        )
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is False
        assert result["total_rounds"] == 3

    @pytest.mark.asyncio
    async def test_score_above_threshold_but_reject_verdict_not_accepted(self):
        """A rejecting verdict overrides a passing score."""
        import json as _json

        fake_sel = FakeEvidenceSelector()
        reject_review = _json.dumps(
            {
                "score": 0.9,
                "strengths": [],
                "weaknesses": ["fatal flaws"],
                "issues": [{"section": "intro", "issue": "wrong", "severity": "major"}],
                "missing_coverage": [],
                "verdict": "reject",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(
            responses=[
                "Survey draft.",
                reject_review,
                "Revised.",
                reject_review,
                "Final.",
                reject_review,
            ]
        )
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is False
        assert result["total_rounds"] == 3

    @pytest.mark.asyncio
    async def test_missing_compliance_is_not_accepted(self):
        """Missing evidence compliance fails closed."""
        import json as _json

        fake_sel = FakeEvidenceSelector()
        review_no_compliance = _json.dumps(
            {
                "score": 0.95,
                "strengths": ["great"],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
            }
        )
        llm = MockLLMClient(
            responses=[
                "Survey.",
                review_no_compliance,
                "R2.",
                review_no_compliance,
                "R3.",
                review_no_compliance,
            ]
        )
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is False

    @pytest.mark.asyncio
    async def test_reviewer_first_round_failure_returns_fallback(self):
        """A first-round reviewer failure returns the initial draft."""
        fake_sel = FakeEvidenceSelector()
        llm = MockLLMClient(responses=["Initial draft that must survive."])
        synthesis = SynthesisWorker(llm, evidence_selector=fake_sel)

        class _ExplodingReviewer(ReviewerWorker):
            """Reviewer test double that always raises."""

            async def execute(self, task):
                raise RuntimeError("simulated failure")

        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=synthesis,
            reviewer=_ExplodingReviewer(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert "Initial draft" in result["final_draft"]
        assert result["accepted"] is False
        assert result["total_rounds"] == 0

    @pytest.mark.asyncio
    async def test_selection_section_ids_identical_across_rounds(self):
        """Every review round receives the same selected evidence IDs."""
        import json as _json

        from litagent.agents.synthesis import SynthesisWorker as _SW
        from litagent.agents.reviewer import ReviewerWorker as _RW

        calls: list[dict] = []

        class _SpySynthesis(_SW):
            """Synthesis worker that records revision evidence."""

            def revise(self, draft, review_comments, evidence_selection):
                sel = (
                    EvidenceSelection.from_dict(evidence_selection)
                    if not isinstance(evidence_selection, EvidenceSelection)
                    else evidence_selection
                )
                calls.append(
                    {
                        "method": "revise",
                        "section_ids": {
                            k: list(v) for k, v in sel.section_evidence_ids.items()
                        },
                    }
                )
                return super().revise(draft, review_comments, evidence_selection)

        class _SpyReviewer(_RW):
            """Reviewer that records revision evidence."""

            async def review_revision(
                self, revised_draft, previous_review, evidence_selection
            ):
                sel = (
                    EvidenceSelection.from_dict(evidence_selection)
                    if not isinstance(evidence_selection, EvidenceSelection)
                    else evidence_selection
                )
                calls.append(
                    {
                        "method": "review_revision",
                        "section_ids": {
                            k: list(v) for k, v in sel.section_evidence_ids.items()
                        },
                    }
                )
                return await super().review_revision(
                    revised_draft,
                    previous_review,
                    evidence_selection,
                )

        low_review = _json.dumps(
            {
                "score": 0.5,
                "strengths": [],
                "weaknesses": ["missing"],
                "issues": [],
                "missing_coverage": [],
                "verdict": "revise",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        high_review = _json.dumps(
            {
                "score": 0.9,
                "strengths": ["improved"],
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
        llm = MockLLMClient(
            responses=[
                "Draft v1 [E:p1:claim:0].",
                low_review,
                "Revised draft.",
                high_review,
            ]
        )
        fake_sel = FakeEvidenceSelector()
        syn = _SpySynthesis(llm, evidence_selector=fake_sel)
        rev = _SpyReviewer(llm)
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=syn,
            reviewer=rev,
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)
        assert result["accepted"] is True

        assert len(calls) >= 2
        first_ids = calls[0]["section_ids"]
        for c in calls[1:]:
            assert (
                c["section_ids"] == first_ids
            ), f"section IDs drift: {c['method']} differs from first call"

    @pytest.mark.asyncio
    async def test_unsupported_claims_appear_in_revision_feedback(self):
        """Revision feedback includes unsupported and unknown claims."""
        import json as _json

        from litagent.agents.synthesis import SynthesisWorker as _SW

        revise_msgs: list = []

        class _SpySynthesis(_SW):
            """Synthesis worker that records revision feedback."""

            def revise(self, draft, review_comments, evidence_selection):
                revise_msgs.append(review_comments)
                return super().revise(draft, review_comments, evidence_selection)

        review_with_issues = _json.dumps(
            {
                "score": 0.3,
                "strengths": [],
                "weaknesses": ["many issues"],
                "issues": [],
                "missing_coverage": [],
                "verdict": "revise",
                "evidence_compliance": {
                    "passed": False,
                    "unsupported_claims": [
                        {
                            "section": "intro",
                            "claim": "SOTA",
                            "reason": "no evidence",
                            "action": "delete",
                        },
                    ],
                    "unknown_evidence_ids": ["p99:claim:0"],
                },
            }
        )
        final_review = _json.dumps(
            {
                "score": 0.9,
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
        llm = MockLLMClient(
            responses=[
                "Draft.",
                review_with_issues,
                "Revised draft.",
                final_review,
            ]
        )
        fake_sel = FakeEvidenceSelector()
        syn = _SpySynthesis(llm, evidence_selector=fake_sel)
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=syn,
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        await w.execute(task)

        assert len(revise_msgs) >= 1
        feedback = revise_msgs[0]

        assert "SOTA" in feedback
        assert "delete" in feedback

        assert "[E:p99:claim:0]" in feedback

    @pytest.mark.asyncio
    async def test_revision_empty_text_keeps_previous_draft(self):
        """An empty revision cannot replace the previous draft."""
        import json as _json

        low_review = _json.dumps(
            {
                "score": 0.4,
                "strengths": [],
                "weaknesses": ["bad"],
                "issues": [],
                "missing_coverage": [],
                "verdict": "revise",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
            }
        )
        llm = MockLLMClient(
            responses=[
                "Original draft that must be preserved.",
                low_review,
                "",
            ]
        )
        fake_sel = FakeEvidenceSelector()
        w = AdversarialReviewWorker(
            llm=llm,
            synthesis=SynthesisWorker(llm, evidence_selector=fake_sel),
            reviewer=ReviewerWorker(llm),
            max_rounds=3,
            pass_threshold=0.8,
        )
        task = _make_task_with_data()
        result = await w.execute(task)

        assert "Original draft that must be preserved." in result["final_draft"]
