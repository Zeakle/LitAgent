"""Verify failed evidence contracts stop automatic work through real consumers."""

import hashlib
from unittest.mock import AsyncMock, MagicMock

import pytest

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import AppConfig
from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval.base import EvalResult
from litagent.llm.client import LLMResponse
from litagent.orchestrator.scheduler import Scheduler
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.rag.claim_promotion import ClaimsPromoter, is_publishable_report
from litagent.report_audit import ReportAuditService
from litagent.runner import LitAgent
from litagent.workbench import RunOptions


class BrokenSelectionSynthesis(SynthesisWorker):
    """Simulate the observed producer omission after real draft generation."""

    async def execute(self, task):
        """Remove provenance while retaining the generated draft and evidence."""
        result = await super().execute(task)
        result["evidence_selection"]["method_by_section"].pop("methods")
        self.original_draft = result["draft"]
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["legacy", "academic_v1", "method_transparent_v1"])
async def test_broken_producer_stops_review_evaluation_rewrite_and_trust(
    profile,
    tmp_path,
):
    """Keep diagnostic citations intact and reject downstream paid/trusted work."""
    config = AppConfig(agent={}, logging={}, observability={"enabled": False})
    item = {"evidence_id": "p:claim:0", "paper_id": "p", "text": "Supported finding"}
    selector = AsyncMock()
    selector.select.return_value = EvidenceSelection(
        1,
        {"p:claim:0": item},
        {"methods": ["p:claim:0"]},
        {"methods": "lexical_fallback"},
        0,
        20,
    )
    llm = MagicMock()
    llm.chat = AsyncMock(
        return_value=LLMResponse(content="# Title\n\nSupported finding [E:e1].")
    )
    synthesis = BrokenSelectionSynthesis(llm, evidence_selector=selector)
    worker = AdversarialReviewWorker(llm, synthesis, ReviewerWorker(llm))
    graph = TaskGraph()
    graph.add_task(
        SubTask(
            task_id="adversarial_review",
            description="test",
            agent_type="adversarial_review",
            input_data={"query": "finding", "review_structure": profile},
        )
    )
    results = await Scheduler([worker]).run(graph)
    results["extract"] = [{"paper_id": "p", "claims": [], "evidence_items": [item]}]
    agent = LitAgent(config)
    evaluators = []
    for metric in ("citation_accuracy", "faithfulness", "internal_consistency"):
        evaluator = MagicMock(metric_name=metric)
        evaluator.evaluate = AsyncMock(
            side_effect=AssertionError("invalid evidence reached evaluator")
        )
        evaluators.append(evaluator)
    agent._evaluators = evaluators
    agent._attempt_evidence_rewrite = AsyncMock()
    events = []
    agent._trace_hook = lambda event, data: events.append((event, data))

    report, degradation = await agent._prepare_report(
        query="finding",
        graph=graph,
        results=results,
        options=RunOptions(review_structure=profile),
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )

    assert llm.chat.await_count == 1
    assert report["survey"] == synthesis.original_draft
    assert "[E:p:claim:0]" in report["survey"]
    assert report["partial"] is False
    assert report["metadata"]["accepted"] is False
    assert report["metadata"]["evidence_selection_error"] == {
        "reason_code": "evidence_selection_invalid",
        "message": "section 'methods' has evidence ids but no retrieval method",
    }
    assert report["quality"]["status"] == "unverified"
    assert report["delivery"]["status"] == "needs_review"
    assert report["delivery"]["publishable"] is False
    assert "evidence_selection_invalid" in degradation
    for evaluator in evaluators:
        evaluator.evaluate.assert_not_awaited()
        metric = report["evaluation"][evaluator.metric_name]
        assert metric["skipped"] is True
        assert metric["details"]["skipped_reason"] == "evidence_selection_invalid"
    agent._attempt_evidence_rewrite.assert_not_awaited()
    assert not is_publishable_report(report)
    index = MagicMock()
    index.upsert_trusted = AsyncMock()
    promotion = await ClaimsPromoter(index).promote(
        run_id="guard-test",
        domain="finding",
        report_data=report,
        extractions=results["extract"],
    )
    assert promotion.status == "skipped"
    assert promotion.reason_code == "run_not_publishable"
    index.upsert_trusted.assert_not_awaited()
    service = ReportAuditService(config, tmp_path)
    service.submit = AsyncMock()
    snapshot = {
        "source_run_id": "guard-test",
        "source_report_revision": 1,
        "source_report_sha256": hashlib.sha256(report["survey"].encode()).hexdigest(),
        "report": report,
    }
    automatic = await service.submit_automatic(
        snapshot, mode="judge_jev", max_versions=1
    )
    assert automatic["status"] == "skipped"
    assert automatic["reason_codes"] == ["automatic_quality_not_passed"]
    assert automatic["budget_reserved"] is False
    service.submit.assert_not_awaited()
    terminal = next(data for event, data in events if event == "subspan.end")
    assert terminal["output"] == report["evaluation"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "summary", [None, {"valid": True}, {"valid": True, "selected_count": 0}]
)
async def test_valid_or_legacy_summary_keeps_existing_evaluation(summary):
    """Do not interpret a missing historical summary or legal emptiness as corruption."""
    agent = LitAgent(AppConfig(agent={}, logging={}))
    evaluator = MagicMock(metric_name="faithfulness")
    evaluator.evaluate = AsyncMock(return_value=EvalResult("faithfulness", 0.9, True))
    agent._evaluators = [evaluator]
    results = (
        {}
        if summary is None
        else {"adversarial_review": {"evidence_selection": summary}}
    )

    output = await agent._evaluate(
        query="finding", survey="draft", results=results, phase="initial"
    )

    evaluator.evaluate.assert_awaited_once()
    assert output["faithfulness"]["skipped"] is False
    assert output["faithfulness"]["score"] == 0.9


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["initial", "post_rewrite", "post_human_revision"])
async def test_invalid_selection_guard_applies_to_all_evaluation_phases(phase):
    """Prevent alternate evaluation entry points from spending on broken selections."""
    agent = LitAgent(AppConfig(agent={}, logging={}))
    evaluator = MagicMock(metric_name="faithfulness")
    evaluator.evaluate = AsyncMock()
    agent._evaluators = [evaluator]
    results = {"adversarial_review": {"evidence_selection": {"valid": False}}}

    output = await agent._evaluate(
        query="finding", survey="draft", results=results, phase=phase
    )

    evaluator.evaluate.assert_not_awaited()
    assert output["faithfulness"]["skipped"] is True
    assert agent._derive_quality(output)["status"] == "unverified"
