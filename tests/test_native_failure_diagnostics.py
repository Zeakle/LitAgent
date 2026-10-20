"""Replay mixed native failures without losing source permissions or paid attempts."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import (
    GroundedWritingError,
    apply_native_repairs,
    begin_targeted_repair,
    grounded_messages,
    prepare_grounded_request,
    prepare_native_repair_request,
    render_grounded_report_traced,
    save_targeted_repairs,
    targeted_repair_state,
)
from litagent.runner import LitAgent
from litagent.workbench import RunOptions, VerificationOptions


@pytest.fixture
def received():
    """Keep the received scientific prose, sources, and rejected patches unchanged."""
    return json.loads(
        (Path(__file__).parent / "fixtures/native-mixed-errors-real.json").read_text()
    )


def render(received, candidate):
    """Use the same full-source and section permissions as the real writing path."""
    return render_grounded_report_traced(
        candidate,
        received["cards"],
        "academic_v2",
        "zh-CN",
        evidence_selection=received["selection"],
        require_native=True,
        require_full_sources=True,
        require_scope_notices=True,
        analysis_mode=True,
    )


def test_received_mixed_errors_are_all_available_before_paid_repair(received):
    with pytest.raises(GroundedWritingError) as failed:
        render(received, received["candidate"])
    diagnostics = failed.value.diagnostics
    codes = {d["reason_code"] for d in diagnostics}
    assert "strict_writing_scope_notice_invalid" in codes
    assert "strict_writing_out_of_scope_fact_card" in codes
    outside = [
        d
        for d in diagnostics
        if d["reason_code"] == "strict_writing_out_of_scope_fact_card"
    ]
    assert {d["section_key"] for d in outside} == {
        "introduction",
        "methods",
        "open_problems",
    }
    assert all(d["raw_json_path"] and d["allowed_fact_card_ids"] for d in outside)
    messages, _ = prepare_grounded_request(
        grounded_messages(
            received["cards"],
            "academic_v2",
            "zh-CN",
            "Compare metric learning",
            evidence_selection=received["selection"],
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
    )
    _, schema, binding = prepare_native_repair_request(
        messages,
        received["candidate"],
        diagnostics,
        cards=received["cards"],
        structure="academic_v2",
        language="zh-CN",
        evidence_selection=received["selection"],
    )
    assert binding["version"] == "complete-native-unit-repair-v5"
    assert len(binding["targets"]) == 7
    assert {target["path"] for target in binding["targets"].values()} == {
        d["raw_json_path"]
        for d in diagnostics
        if d["reason_code"]
        in {
            "strict_writing_scope_notice_invalid",
            "strict_writing_out_of_scope_fact_card",
        }
    }
    for key, target in binding["targets"].items():
        if target["must_remain_scientific"]:
            fields = schema["json_schema"]["schema"]["properties"]["patches"][
                "properties"
            ][key]["properties"]
            assert fields["kind"]["enum"] == ["scientific"]
            assert fields["scope_notice_id"]["enum"] == ["none"]


def test_received_patch_cannot_hide_protected_out_of_scope_units(received):
    candidate, _ = apply_native_repairs(
        received["candidate"],
        received["repair_response"],
        received["repair_binding"],
        cards=received["cards"],
    )
    with pytest.raises(GroundedWritingError) as failed:
        render(received, candidate)
    assert any(
        d["reason_code"] == "strict_writing_out_of_scope_fact_card"
        for d in failed.value.diagnostics
    )


@pytest.mark.asyncio
async def test_actual_synthesis_retains_invalid_repair_after_complete_input_fits(
    received, tmp_path
):
    selected = EvidenceSelection(
        candidate_count=len(received["cards"]),
        selected_items=received["selected_items"],
        section_evidence_ids=copy.deepcopy(
            received["selection"]["section_evidence_ids"]
        ),
        method_by_section={},
        omitted_count=0,
        estimated_tokens=10000,
    )
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                received["candidate"],
                usage={"prompt_tokens": 25000, "completion_tokens": 5000},
            )
        )
    )
    from litagent.native_wire import array_from_native

    original_response = llm.chat.return_value
    llm.chat.side_effect = [
        LLMResponse(
            json.dumps(
                array_from_native(json.loads(received["candidate"])), ensure_ascii=False
            ),
            usage=dict(original_response.usage),
        ),
        original_response,
    ]
    path = tmp_path / "repairs.json"
    events = []
    worker = SynthesisWorker(
        llm,
        evidence_selector=SimpleNamespace(select=AsyncMock(return_value=selected)),
        run_id_provider=lambda: "mixed-failure",
        repair_state_path_provider=lambda: path,
        trace_hook=lambda event, data: events.append((event, data)),
    )
    task = SubTask(
        "adversarial_review",
        "write",
        "synthesis",
        input_data={
            "query": "Compare metric-learning approaches for few-shot image classification.",
            "quality_policy": "strict_auto_v2",
            "review_structure": "academic_v2",
            "report_language": "zh-CN",
            "body_quality": "analysis_v1",
            "verification": {"judge": False, "jev": False},
        },
    )
    with pytest.raises(ValueError, match="strict_native_repair_response_invalid"):
        await worker.execute(task)
    assert llm.chat.await_count == 2 and path.exists()
    saved = json.loads(path.read_text())
    assert len(saved["state"]["attempts"]) == 1
    assert saved["state"]["attempts"][0]["status"] == "failed"
    assert saved["native_attempt_0"]["response"] == received["candidate"]
    assert not any(event == "writing.request.rejected" for event, _ in events)
    diagnostics = next(
        data["diagnostics"]
        for event, data in events
        if event == "writing.validation.failed"
    )
    assert any(
        d["reason_code"] == "strict_writing_unit_citation_scope_ambiguous"
        for d in diagnostics
    )
    assert (
        sum(
            d["reason_code"] == "strict_writing_out_of_scope_fact_card"
            for d in diagnostics
        )
        == 5
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("has_final", [False, True])
async def test_report_outlet_preserves_failed_worker_ledger_without_refund(
    tmp_path, has_final
):
    agent = LitAgent(load_config())
    agent._targeted_repair_state_path = tmp_path / "state.json"
    state = targeted_repair_state(None, agent._session_id)
    entry = begin_targeted_repair(state, "native_initial", "a" * 64)
    entry.update(status="failed", error_code="strict_writing_out_of_scope_fact_card")
    save_targeted_repairs(agent._targeted_repair_path(), state)
    agent._evaluate = AsyncMock(side_effect=AssertionError("No report to evaluate"))
    agent._evaluators = [
        SimpleNamespace(metric_name=name)
        for name in ("citation_accuracy", "faithfulness", "internal_consistency")
    ]
    graph = TaskGraph()
    graph.add_task(SubTask("adversarial_review", "write", "adversarial_review"))
    graph.mark_running("adversarial_review")
    results = {}
    if has_final:
        results["adversarial_review"] = {"final_draft": "# A report"}
        graph.mark_done("adversarial_review", results["adversarial_review"])
    else:
        graph.mark_failed("adversarial_review", entry["error_code"])
    kwargs = dict(
        query="few-shot",
        graph=graph,
        results=results,
        options=RunOptions(
            quality_policy="strict_auto_v2",
            verification=VerificationOptions(judge=False, jev=False),
        ),
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )
    if has_final:
        with pytest.raises(
            ValueError, match="strict_targeted_repair_checkpoint_conflict"
        ):
            await agent._prepare_report(**kwargs)
    else:
        result, _ = await agent._prepare_report(**kwargs)
        assert result["metadata"]["targeted_repair_state"] == state
        assert result["partial"] and result["quality"]["status"] == "unverified"
        assert result["delivery"]["publishable"] is False
        assert "task_failed" in result["metadata"]["execution"]["reason_codes"]
        assert all(
            v["skipped"]
            and not v["passed"]
            and v["details"]["skipped_reason"] == "final_report_unavailable"
            for v in result["evaluation"].values()
        )
    agent._evaluate.assert_not_awaited()
    assert json.loads(agent._targeted_repair_path().read_text())["state"] == state
