"""Exercise native repair, the real worker export, and the shared durable limit."""

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_extraction_verification import verify
from test_scope_notice_contract import notice_document as full_document
from test_strict_quality import valid_details

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import (
    begin_targeted_repair,
    build_fact_cards,
    load_targeted_repairs,
    save_targeted_repairs,
    targeted_repair_state,
)
from litagent.runner import LitAgent, RewriteOutcome
from litagent.workbench import RunOptions


async def setup_worker(tmp_path, monkeypatch, fault=None):
    selected, receipt, _ = await verify(
        tmp_path / "jev", monkeypatch, run_id="native-test"
    )
    cards = build_fact_cards(selected.selected_items)
    candidate = full_document(cards)
    candidate["sections"]["open_problems"]["first_block"]["first_sentence"].update(
        text="Current research lacks systematic comparisons.",
        kind="gap",
        fact_card_ids=[],
    )
    candidate["sections"]["conclusion"]["first_block"]["first_sentence"][
        "body_reference"
    ]["section_key"] = "open_problems"
    response = {
        "can_repair": fault != "unavailable",
        "patches": {
            "u0": {
                "text": "EViT improves speed by 50% on DeiT-S.",
                "kind": "scientific",
                "scope_notice_id": "none",
                "fact_card_ids": list(cards),
            }
        },
    }
    usage = {
        "prompt_tokens": 10,
        "completion_tokens": 2,
        "total_tokens": 12,
        "usage_available": True,
    }
    outputs = [
        LLMResponse(json.dumps(candidate), usage=usage),
        (
            asyncio.CancelledError()
            if fault == "cancelled"
            else LLMResponse(
                json.dumps(response), usage={} if fault == "usage" else usage
            )
        ),
    ]
    llm = SimpleNamespace(chat=AsyncMock(side_effect=outputs))
    selector = SimpleNamespace(select=AsyncMock(return_value=selected))
    gate = SimpleNamespace(verify=AsyncMock(return_value=(selected, receipt)))
    path = tmp_path / "repair-state.json"
    events = []

    def factory():
        return SynthesisWorker(
            llm,
            evidence_selector=selector,
            extraction_verifier=gate,
            run_id_provider=lambda: "native-test",
            repair_state_path_provider=lambda: path,
            trace_hook=lambda event, data: events.append((event, copy.deepcopy(data))),
        )

    task = SubTask(
        "adversarial_review",
        "write",
        "synthesis",
        input_data={
            "query": "EViT",
            "report_language": "en",
            "review_structure": "academic_v2",
            "quality_policy": "strict_auto_v2",
        },
    )
    return factory, task, llm, path, events, gate


@pytest.mark.asyncio
async def test_actual_synthesis_repairs_once_and_cold_restart_does_not_resend(
    tmp_path, monkeypatch
):
    factory, task, llm, path, events, gate = await setup_worker(tmp_path, monkeypatch)
    result = await factory().execute(task)
    state = result["targeted_repair_state"]
    assert len(state["attempts"]) == 1 and state["attempts"][0]["status"] == "completed"
    assert llm.chat.await_count == 2 and gate.verify.await_count == 1
    assert all(
        call.kwargs["transport_max_attempts"] == 1
        and call.kwargs["max_tokens"] == 20000
        for call in llm.chat.await_args_list
    )
    saved = json.loads(path.read_text())
    assert (
        saved["state"] == state
        and saved["native_candidate"] != saved["repaired_native_candidate"]
    )
    assert saved["native_response"] and saved["native_binding"]
    assert result["draft"].count("EViT improves speed by 50%") == 7
    new_task = copy.deepcopy(task)
    new_task.input_data.pop("targeted_repair_state")
    with pytest.raises(ValueError, match="restart_no_replay"):
        await factory().execute(new_task)
    assert llm.chat.await_count == 2 and gate.verify.await_count == 1
    assert [
        data["status"] for event, data in events if event == "targeted_repair.state"
    ] == ["started_no_replay", "completed"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unavailable", "usage", "cancelled"])
async def test_failed_unknown_or_cancelled_repair_consumes_a_slot(
    tmp_path, monkeypatch, fault
):
    factory, task, llm, path, events, _ = await setup_worker(
        tmp_path, monkeypatch, fault
    )
    expected = asyncio.CancelledError if fault == "cancelled" else ValueError
    with pytest.raises(expected):
        await factory().execute(task)
    state = load_targeted_repairs(path, "native-test")
    assert len(state["attempts"]) == 1
    assert state["attempts"][0]["status"] == (
        "cancelled" if fault == "cancelled" else "failed"
    )
    with pytest.raises(ValueError, match="restart_no_replay"):
        await factory().execute(task)
    assert llm.chat.await_count == 2


@pytest.mark.asyncio
async def test_native_preflight_refusal_has_no_repair_request(tmp_path, monkeypatch):
    from litagent import quality

    factory, task, llm, path, _, _ = await setup_worker(tmp_path, monkeypatch)
    real = quality.check_grounded_request_budget
    calls = []

    def budget(*args, **kwargs):
        calls.append(kwargs.get("stage"))
        if len(calls) == 2:
            raise ValueError("strict_context_budget_exceeded")
        return real(*args, **kwargs)

    monkeypatch.setattr(quality, "check_grounded_request_budget", budget)
    with pytest.raises(ValueError, match="context_budget"):
        await factory().execute(task)
    assert llm.chat.await_count == 1 and not path.exists()


@pytest.mark.asyncio
async def test_real_worker_export_and_report_outlet_preserve_counter(
    tmp_path, monkeypatch
):
    factory, task, llm, path, _, _ = await setup_worker(tmp_path, monkeypatch)
    reviewer = SimpleNamespace(
        execute=AsyncMock(
            return_value={
                "score": 1.0,
                "verdict": "accept",
                "evidence_compliance": {"passed": True},
                "language_compliance": {"passed": True},
            }
        )
    )
    result = await AdversarialReviewWorker(
        llm, factory(), reviewer, max_rounds=1
    ).execute(task)
    agent = LitAgent(load_config())
    agent._session_id = "native-test"
    agent._targeted_repair_state_path = path
    agent._evaluate = AsyncMock(
        return_value={
            "citation_accuracy": {"passed": True},
            "faithfulness": {"passed": True},
        }
    )
    graph = TaskGraph()
    graph.add_task(SubTask("adversarial_review", "write", "adversarial_review"))
    graph.mark_running("adversarial_review")
    graph.mark_done("adversarial_review", result)
    report, _ = await agent._prepare_report(
        query="EViT",
        graph=graph,
        results={"adversarial_review": result},
        options=RunOptions(quality_policy="strict_auto_v2"),
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )
    assert (
        report["metadata"]["targeted_repair_state"]
        == result["targeted_repair_state"]
        == load_targeted_repairs(path, "native-test")
    )
    assert report["survey"] == result["final_draft"]


@pytest.mark.asyncio
@pytest.mark.parametrize("used,expected", [(0, 2), (1, 1), (2, 0)])
async def test_prepare_report_cannot_reset_slots_when_evaluation_still_fails(
    tmp_path, used, expected
):
    agent = LitAgent(load_config())
    agent._targeted_repair_state_path = tmp_path / "state.json"
    state = targeted_repair_state(None, agent._session_id)
    for index in range(used):
        entry = begin_targeted_repair(
            state, "native_initial" if index == 0 else "evaluation", str(index) * 64
        )
        entry["status"] = "completed" if index == 0 else "failed"
    save_targeted_repairs(agent._targeted_repair_path(), state)
    details = valid_details()
    details["assertions"][0]["status"] = "unsupported"
    evaluation = {
        "citation_accuracy": {"passed": True},
        "faithfulness": {"passed": False, "details": details},
    }
    agent._evaluate = AsyncMock(return_value=evaluation)
    agent._attempt_evidence_rewrite = AsyncMock(
        side_effect=[
            RewriteOutcome(
                True,
                False,
                f"candidate {n}",
                None,
                evaluation,
                {"status": "failed"},
                "quality_still_failed",
            )
            for n in range(expected)
        ]
    )
    graph = TaskGraph()
    graph.add_task(SubTask("adversarial_review", "write", "adversarial_review"))
    graph.mark_running("adversarial_review")
    graph.mark_done("adversarial_review", {})
    result, _ = await agent._prepare_report(
        query="EViT",
        graph=graph,
        results={
            "adversarial_review": {
                "final_draft": "original",
                "targeted_repair_state": state,
            }
        },
        options=RunOptions(quality_policy="strict_auto_v2"),
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )
    assert (
        agent._attempt_evidence_rewrite.await_count == expected
        and result["survey"] == "original"
    )
    assert not result["delivery"]["publishable"]


def test_durable_unknown_response_and_counter_refund_are_rejected(tmp_path):
    path = tmp_path / "state.json"
    state = targeted_repair_state(None, "run")
    entry = begin_targeted_repair(state, "native_initial", "a" * 64)
    save_targeted_repairs(path, state)
    with pytest.raises(ValueError, match="restart_no_replay"):
        load_targeted_repairs(path, "run")
    entry["status"] = "failed"
    save_targeted_repairs(path, state)
    with pytest.raises(ValueError, match="checkpoint_conflict"):
        save_targeted_repairs(path, targeted_repair_state(None, "run"))
    entry = begin_targeted_repair(state, "evaluation", "b" * 64)
    entry["status"] = "failed"
    with pytest.raises(ValueError, match="limit_exceeded"):
        begin_targeted_repair(state, "evaluation", "c" * 64)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "usage", "invalid", "cancelled"])
async def test_actual_evaluation_repair_consumes_the_second_slot_even_on_failure(
    tmp_path, fault
):
    from test_fragment_repair import case

    from litagent.eval.base import CTX_EVIDENCE, CTX_SELECTED_EVIDENCE
    from litagent.quality import digest

    body, evaluation, cards, _, response, _ = case()
    for patch in response["patches"].values():
        patch["first_sentence"].pop("support_quotes")
    if fault == "invalid":
        next(iter(response["patches"].values()))["first_sentence"]["fact_card_ids"] = []
    usage = (
        {}
        if fault == "usage"
        else {"prompt_tokens": 10, "completion_tokens": 1, "usage_available": True}
    )
    llm = SimpleNamespace(
        chat=(
            AsyncMock(side_effect=asyncio.CancelledError())
            if fault == "cancelled"
            else AsyncMock(return_value=LLMResponse(json.dumps(response), usage=usage))
        )
    )
    agent = LitAgent(load_config())
    agent._llm = llm
    agent._synthesis = SynthesisWorker(llm)
    agent._targeted_repair_state_path = tmp_path / "state.json"
    state = targeted_repair_state(None, agent._session_id)
    entry = begin_targeted_repair(state, "native_initial", "a" * 64)
    entry["status"] = "completed"
    save_targeted_repairs(agent._targeted_repair_path(), state)
    ledger = {
        c["evidence_id"]: {
            "paper_id": c["paper_id"],
            "text": c["source_quote"],
            "supporting_text": c["source_quote"],
        }
        for c in cards.values()
    }
    agent._build_eval_context = lambda **kwargs: {
        CTX_EVIDENCE: ledger,
        CTX_SELECTED_EVIDENCE: list(ledger.values()),
    }
    agent._evaluate = AsyncMock(
        return_value={"faithfulness": {"skipped": True, "details": {}}}
    )
    # This hand-authored short fixture exercises call accounting, not a support verdict.
    report = {
        "survey": body,
        "metadata": {
            "review_structure": "academic_v2",
            "quality_policy": "strict_auto_v2",
            "fact_cards": cards,
            "evidence_snapshot_sha256": digest(cards),
            "targeted_repair_state": state,
            "writing_bindings": {
                "report_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "locations": [],
            },
        },
    }
    kwargs = dict(
        query="EViT",
        report_data=report,
        results={},
        initial_evaluation=evaluation,
        initial_quality={"status": "unverified"},
        report_language="en",
    )
    if fault == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await agent._attempt_evidence_rewrite(**kwargs)
    else:
        outcome = await agent._attempt_evidence_rewrite(**kwargs)
        assert not outcome.committed
    assert llm.chat.await_count == 1 and report["survey"] == body
    stored = load_targeted_repairs(agent._targeted_repair_path(), agent._session_id)
    assert (
        len(stored["attempts"]) == 2 and stored["attempts"][1]["stage"] == "evaluation"
    )
    assert stored["attempts"][1]["status"] == (
        "completed"
        if fault is None
        else "cancelled" if fault == "cancelled" else "failed"
    ), (outcome.reason_code if fault != "cancelled" else "cancelled")
    refused = await agent._attempt_evidence_rewrite(**kwargs)
    assert (
        refused.reason_code == "strict_targeted_repair_limit_exceeded"
        and llm.chat.await_count == 1
    )


def test_repair_event_is_retained_locally_and_redacted_externally(tmp_path):
    from litagent.observability.recorder import (
        ArchiveRepository,
        RedactingTraceHook,
        RunRecorder,
    )
    from litagent.quality import emit_targeted_repair

    recorder = RunRecorder(
        "run", "private question", ArchiveRepository(tmp_path / "runs")
    )
    sent = []
    sink = RedactingTraceHook(
        lambda event, data: sent.append((event, data)), payload_mode="metadata_only"
    )
    state = targeted_repair_state(None, "run")
    begin_targeted_repair(state, "native_initial", "b" * 64)

    def hook(event, data):
        recorder(event, data)
        sink(event, data)

    emit_targeted_repair(hook, state, "synthesis")
    assert recorder.snapshot()["targeted_repair_state"] == state
    assert "state" not in sent[0][1] and sent[0][1]["count"] == 1
    from litagent.demo import project_flow_demo_view

    assert (
        project_flow_demo_view(recorder.snapshot(), source="archive")[
            "targeted_repair_state"
        ]
        == state
    )


@pytest.mark.parametrize(
    "reason",
    [
        "strict_native_repair_response_invalid",
        "strict_targeted_repair_checkpoint_conflict",
        "strict_targeted_repair_restart_no_replay",
    ],
)
def test_new_protocol_errors_stop_but_unrepairable_science_is_not_a_protocol_error(
    reason,
):
    from litagent.benchmark.quality_suite import _technical_failure

    assert _technical_failure([reason]) == reason
    assert _technical_failure(["strict_native_repair_unavailable"]) is None


@pytest.mark.asyncio
async def test_unrelated_first_draft_error_is_preserved_without_an_extra_call(
    tmp_path, monkeypatch
):
    factory, task, llm, path, _, _ = await setup_worker(tmp_path, monkeypatch)
    raw = full_document(
        build_fact_cards(
            factory()._extraction_verifier.verify.return_value[0].selected_items
        )
    )
    raw["sections"]["taxonomy"]["first_block"]["first_sentence"]["fact_card_ids"] = [
        "f:invented"
    ]
    llm.chat.side_effect = [LLMResponse(json.dumps(raw), usage={"total_tokens": 1})]
    with pytest.raises(ValueError, match="strict_writing_passage_binding_invalid"):
        await factory().execute(task)
    assert llm.chat.await_count == 1 and not path.exists()
