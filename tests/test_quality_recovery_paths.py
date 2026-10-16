"""Exercise complete diagnostic repair and preserve reserved evaluation capacity."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_chapter_scope_contract import materials
from test_quantitative_attribution import evaluate_claim
from test_strict_quality import draft, to_native, valid_details
from test_support_passages import selected

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.eval.scoped import (
    build_source_fragments,
    build_support_passages,
    support_passage_table,
    validate_support_basis,
)
from litagent.observability.context import reset_task_id, set_task_id
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import check_grounded_request_budget, render_grounded_report
from litagent.report_render import build_report_source_units
from litagent.review_structure import assemble_methodology
from litagent.runner import LitAgent, RewriteOutcome
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions


def completed_unverified(body):
    """Bind one unresolved scientific diagnostic to its actual current source."""
    unit = build_report_source_units(body)[0]
    fragment = build_source_fragments(unit)[0]
    details = valid_details()
    details.update(
        quality_policy="strict_auto_v1",
        source_report_sha256=hashlib.sha256(body.encode()).hexdigest(),
    )
    details["coverage"].update(
        complete=False,
        source_units=1,
        processed_units=1,
        model_source_units=1,
        model_processed_units=1,
    )
    details["stages"]["requests"] = [
        {"phase": "inventory", "status": "response_received"}
    ]
    assertion = details["assertions"][0]
    assertion.update(
        kind="scientific",
        status="unverifiable",
        claim_text="EViT improves speed.",
        source_unit_id=unit.unit_id,
        source_fragment_id=fragment["source_fragment_id"],
        source_start=unit.start + fragment["start"],
        source_end=unit.start + fragment["end"],
        source_quote=fragment["source_quote"],
        evidence_ids=["a"],
    )
    details["stages"]["inventory_assertions"] = [copy.deepcopy(assertion)]
    details["unsupported_claims"] = [
        {
            **assertion,
            "diagnostic_source": "survey-evaluation-v3",
            "citation_scope": "exact",
            "alignment_status": "exact",
            "reason_code": "support_entity_unverified",
        }
    ]
    return {
        "citation_accuracy": {"passed": True},
        "faithfulness": {"passed": True, "skipped": True, "details": details},
        "internal_consistency": {"passed": True, "skipped": True},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "defect",
    [
        None,
        "missing_batch",
        "cancelled",
        "invalid_global",
        "stale_quote",
        "wrong_references",
        "missing_inventory",
    ],
)
async def test_completed_bound_unverified_diagnostics_enter_repair_only_when_valid(
    defect,
):
    body = "EViT improves speed. [E:a]"
    evaluation = completed_unverified(body)
    details = evaluation["faithfulness"]["details"]
    if defect == "missing_batch":
        details["stages"]["batches"] = []
    if defect == "cancelled":
        details["stages"]["batches"][0]["status"] = "interrupted_no_replay"
    if defect == "invalid_global":
        details["stages"]["global_check"] = "interrupted_no_replay"
    if defect == "stale_quote":
        details["unsupported_claims"][0]["source_quote"] = "Old body. [E:a]"
    if defect == "wrong_references":
        details["unsupported_claims"][0]["evidence_ids"] = ["other"]
    if defect == "missing_inventory":
        details["stages"]["requests"] = []
    agent = LitAgent(load_config())
    graph = TaskGraph()
    graph.add_task(SubTask("adversarial_review", "write", "adversarial_review"))
    graph.mark_running("adversarial_review")
    graph.mark_done("adversarial_review", {})
    agent._extract_report = lambda *args: {"survey": body, "metadata": {}}
    agent._evaluate = AsyncMock(return_value=evaluation)
    agent._attempt_evidence_rewrite = AsyncMock(
        return_value=RewriteOutcome(
            False, False, None, None, None, None, "rewrite_run_budget_exceeded"
        )
    )
    result, _ = await agent._prepare_report(
        query="EViT",
        graph=graph,
        results={"adversarial_review": {"final_draft": body}},
        options=RunOptions(quality_policy="strict_auto_v1", report_language="en"),
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )
    assert agent._attempt_evidence_rewrite.await_count == (1 if defect is None else 0)
    assert result["survey"] == body and not result["delivery"]["publishable"]


@pytest.mark.parametrize("strict", [False, True])
def test_original_support_choice_is_explicit_without_changing_legacy_ids(strict):
    source = "Background is broad. EViT uses token reorganization."
    passages = build_support_passages(
        {"a": {"text": "candidate", "supporting_text": source}},
        {"a": "e1"},
        original_only=strict,
    )
    if strict:
        assert next(iter(passages)) == "e1:full_original"
        assert passages["e1:full_original"]["source_end"] == len(source)
        assert any(":window:" in p for p in passages)
    else:
        assert next(iter(passages)) == "e1:p0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["stage_budget", "real_call_error", "spoofed_budget_message", "unknown_usage"],
)
async def test_only_proven_no_send_writing_stage_refusal_preserves_evaluation(failure):
    facts, selection, report = materials()
    body = render_grounded_report(
        json.dumps(to_native(report)),
        facts,
        "academic_v2",
        "en",
        evidence_selection=selection,
        require_native=True,
    )
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    token = set_task_id("adversarial_review")
    try:
        budget.record(
            {
                "prompt_tokens": 285000 if failure == "stage_budget" else 0,
                "completion_tokens": 0,
            }
        )
    finally:
        reset_task_id(token)
    if failure == "unknown_usage":
        budget.charge_unknown_usage(1)
    llm = SimpleNamespace(
        _cost_budget=budget,
        chat=AsyncMock(
            side_effect=(
                ValueError("evaluation_run_budget_exceeded")
                if failure == "spoofed_budget_message"
                else OSError("controlled transport failure")
            )
        ),
    )
    synthesis = SynthesisWorker(llm)
    synthesis.execute = AsyncMock(
        return_value={
            "draft": body,
            "quality_policy": "strict_auto_v1",
            "fact_cards": facts,
            "evidence_selection": selection.to_dict(),
        }
    )
    reviewer = SimpleNamespace(
        execute=AsyncMock(
            return_value={
                "score": 0.7,
                "verdict": "revise",
                "evidence_compliance": {"passed": False},
                "bound_feedback": [{"message": "Fix actual evidence attribution"}],
            }
        )
    )
    worker = AdversarialReviewWorker(llm, synthesis, reviewer, max_rounds=2)
    task = SubTask(
        "adversarial_review",
        "write",
        "adversarial_review",
        input_data={
            "query": "EViT",
            "quality_policy": "strict_auto_v1",
            "report_language": "en",
            "review_structure": "academic_v2",
        },
    )
    result = await worker.execute(task)
    assert (
        result["final_draft"] == assemble_methodology(body, "academic_v2", "en", {})
        and not result["accepted"]
    )
    if failure == "stage_budget":
        assert llm.chat.await_count == 0 and budget.stop_reason is None
        assert "writing_phase_budget_exhausted" in result["degradation_reasons"]
        token = set_task_id("evaluation.initial")
        try:
            assert budget.remaining() == 215000
        finally:
            reset_task_id(token)
    else:
        assert budget.stop_reason is not None


def test_preflight_error_has_host_no_send_identity():
    llm = SimpleNamespace(_cost_budget=CostBudget())
    llm._cost_budget.configure_strict(halt_on_unknown_usage=True)
    token = set_task_id("adversarial_review")
    try:
        llm._cost_budget.record({"prompt_tokens": 285000, "completion_tokens": 0})
        with pytest.raises(ValueError) as exc:
            check_grounded_request_budget(
                llm, [{"role": "user", "content": "complete"}], {"type": "json_object"}
            )
        assert exc.value.not_sent and exc.value.permits_evaluation(llm._cost_budget)
    finally:
        reset_task_id(token)


@pytest.mark.asyncio
async def test_actual_evaluator_produces_completed_source_bound_repair_evidence():
    body, _, details = await evaluate_claim(
        "Model-A achieves average accuracy of 90%.",
        "Model-A achieves average accuracy of 80%. Model-A achieves peak accuracy of 90%.",
    )
    assert details["score"] is None
    evaluation = {"faithfulness": {"details": details}}
    assert LitAgent._completed_bound_repair(evaluation, body)
    assert not LitAgent._completed_bound_repair(evaluation, body.replace("90%", "95%"))


CASES = json.loads(
    (Path(__file__).parent / "fixtures/binding-v2-passage-cases.json").read_text()
)["cases"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["claim"]["claim_id"])
def test_real_complete_original_basis_keeps_correct_methods_and_wrong_citation_blocked(
    case,
):
    claim = case["claim"]
    ledger = case["source"]
    aliases = {eid: f"e{i}" for i, eid in enumerate(ledger)}
    decision, passages = selected(claim, ledger, aliases)
    result = validate_support_basis(
        claim, decision, ledger, aliases, passages=passages, original_only=True
    )
    assert result["status"] == case["expected_guard"]
    if case["expected_guard"] == "passed":
        first = build_support_passages(ledger, aliases, original_only=True)
        selected_id = next(iter(first))
        assert selected_id.endswith(":full_original")
        eid = first[selected_id]["evidence_id"]
        assert first[selected_id]["source_end"] == len(ledger[eid]["supporting_text"])


def test_long_original_is_preserved_and_window_display_is_not_its_support_range():
    source = "A clearly stated scientific condition is present. " * 150
    ledger = {"a": {"text": "candidate", "supporting_text": source}}
    passages = build_support_passages(ledger, {"a": "e1"}, original_only=True)
    assert passages and not any(i.endswith(":full_original") for i in passages)
    for p in passages.values():
        assert (
            p["display_preview"]
            == source[p["source_start"] : min(p["source_end"], p["source_start"] + 80)]
        )
        assert p["source_end"] - p["source_start"] <= 6000
    table = support_passage_table(
        passages, {"a": "e1"}, {"a": {"original_support_ref": "s1"}}
    )
    assert all(row[4]["display_preview"] for row in table.values())
    assert ledger["a"]["supporting_text"] == source
