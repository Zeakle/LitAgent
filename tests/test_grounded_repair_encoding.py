"""Keep complete frozen evidence and make native repair references resolvable."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_native_writing_schema import packet
from test_strict_automatic import frozen, passing_audit

from litagent.eval.automatic import automatic_acceptance
from litagent.quality import check_grounded_request_budget, prepare_grounded_request
from litagent.safety.budget import CostBudget, strict_request_reservation


def repair():
    cards, _, _, messages = packet()
    body = json.loads(messages[1]["content"])
    body["current_draft"] = "EViT improves speed. [E:a]"
    body["located_feedback"] = json.dumps(
        [
            {
                "claim_id": "c",
                "source_quote": body["current_draft"],
                "evidence_ids": ["a"],
                "source_start": 0,
                "source_end": 33,
                "source_offsets_context": "frozen_original_before_aliasing",
            }
        ]
    )
    messages[1]["content"] = json.dumps(body, ensure_ascii=False)
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "must_preserve_evidence_ids": ["a"],
                    "allowed_removed_evidence_ids": ["b"],
                    "instruction": "Preserve required references.",
                }
            ),
        }
    )
    return cards, messages


def test_native_repair_keeps_original_sources_and_uses_actual_card_identity():
    cards, messages = repair()
    original = copy.deepcopy(messages)
    before = json.loads(messages[1]["content"])
    prepared, fmt = prepare_grounded_request(messages)
    after = json.loads(prepared[1]["content"])
    assert messages == original
    assert all(len(ref) < 8 for ref in after["sources"])
    assert list(after["sources"].values()) == list(before["sources"].values())
    for cid, card in cards.items():
        assert (
            after["sources"][after["cards"][cid]["source_ref"]]["source_quote"]
            == card["source_quote"]
        )
        assert (
            after["cards"][cid]["unverified_fields"]
            == before["cards"][cid]["unverified_fields"]
        )
    a = next(cid for cid, card in cards.items() if card["evidence_id"] == "a")
    b = next(cid for cid, card in cards.items() if card["evidence_id"] == "b")
    assert after["current_draft"] == f"EViT improves speed. [F:{a}]"
    feedback = json.loads(after["located_feedback"])[0]
    assert feedback["source_quote"] == after["current_draft"]
    assert feedback["fact_card_ids"] == [a] and "evidence_ids" not in feedback
    assert feedback["source_start"] == 0 and feedback["source_end"] == 33
    policy = json.loads(prepared[2]["content"])
    assert policy["must_preserve_fact_card_ids"] == [a]
    assert policy["allowed_removed_fact_card_ids"] == [b]
    assert "evidence_ids" not in str(policy)
    schema = fmt["json_schema"]["schema"]
    assert a in str(schema) and b in str(schema)


@pytest.mark.parametrize(
    "fault",
    ["unknown_draft", "unknown_policy", "malformed", "ambiguous", "missing_source"],
)
def test_repair_projection_rejects_unresolvable_references_before_call(fault):
    _, messages = repair()
    body = json.loads(messages[1]["content"])
    if fault == "unknown_draft":
        body["current_draft"] = "Finding [E:unknown]"
    elif fault == "malformed":
        body["current_draft"] = "Finding [E:a; E:b]"
    elif fault == "ambiguous":
        for card in body["cards"].values():
            card["evidence_id"] = "a"
    elif fault == "missing_source":
        next(iter(body["cards"].values()))["source_ref"] = "unknown"
    else:
        policy = json.loads(messages[2]["content"])
        policy["must_preserve_evidence_ids"] = ["unknown"]
        messages[2]["content"] = json.dumps(policy)
    messages[1]["content"] = json.dumps(body)
    before = copy.deepcopy(messages)
    with pytest.raises(ValueError, match="strict_writing_reference"):
        prepare_grounded_request(messages)
    assert messages == before


@pytest.mark.parametrize("bad_audit", [None, {}])
def test_unrun_audits_do_not_claim_a_binding_error_or_completed_failure(bad_audit):
    snapshot = frozen()
    evaluation = copy.deepcopy(snapshot["report"]["evaluation"])
    evaluation["faithfulness"]["passed"] = False
    result = automatic_acceptance(snapshot, evaluation, bad_audit)
    assert not result["publishable"]
    assert result["gates"] == {
        "evaluation": "failed",
        "judge": "blocked",
        "jev": "blocked",
    }
    assert "audit_binding_mismatch" not in result["failure_codes"]
    assert (
        "judge_not_run" in result["failure_codes"]
        and "jev_not_run" in result["failure_codes"]
    )


def test_present_incomplete_or_mismatched_audit_is_still_a_binding_failure():
    snapshot = frozen()
    for audit in [
        {"audit_id": "already-sent"},
        passing_audit(snapshot) | {"source_report_sha256": "stale"},
    ]:
        result = automatic_acceptance(snapshot, snapshot["report"]["evaluation"], audit)
        assert (
            not result["publishable"]
            and "audit_binding_mismatch" in result["failure_codes"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_wait", [True, False])
async def test_submitted_audit_without_result_is_never_reported_as_unrun(
    tmp_path, monkeypatch, failed_wait
):
    from litagent.config import load_config
    from litagent.eval.base import CTX_EVIDENCE
    from litagent.runner import LitAgent

    snapshot = frozen()
    report = copy.deepcopy(snapshot["report"])
    service = AsyncMock()
    service.submit.return_value = {"audit_id": "already-submitted"}
    if failed_wait:
        service.wait.side_effect = OSError("Response not available")
    else:
        service.wait.return_value = {}
    agent = LitAgent(load_config(), audit_service=service)
    agent._session_id = "audit-attempt"
    agent._build_eval_context = lambda **kw: {CTX_EVIDENCE: snapshot["ledger"]}
    monkeypatch.chdir(tmp_path)
    await agent._automatic_review(report, "EViT", {}, None)
    result = report["metadata"]["automatic_acceptance"]
    assert not result["publishable"] and result["audit_id"] == "already-submitted"
    assert result["gates"]["judge"] == result["gates"]["jev"] == "failed"
    assert "judge_not_run" not in result["failure_codes"]
    service.submit.assert_awaited_once()


def test_complete_material_pressure_keeps_output_reserve_and_overlimit_refusal():
    from litagent.quality import build_fact_cards, grounded_messages

    ledger = {
        f"e{index}": {
            "paper_id": f"paper{index % 33}",
            "supporting_text": "Exact source data " * 85 + str(index % 33),
        }
        for index in range(53)
    }
    cards = build_fact_cards(ledger)
    raw = grounded_messages(cards, "academic_v2", "en", "Compare reported mechanisms")
    prepared, fmt = prepare_grounded_request(raw)
    packet = json.loads(prepared[1]["content"])
    original = json.loads(raw[1]["content"])
    restored = copy.deepcopy(prepared)
    expanded = copy.deepcopy(packet)
    expanded["sources"] = original["sources"]
    for cid, card in expanded["cards"].items():
        card["source_ref"] = original["cards"][cid]["source_ref"]
    restored[1]["content"] = json.dumps(
        expanded, ensure_ascii=False, separators=(",", ":")
    )
    old_reservation = strict_request_reservation(restored, 20000, fmt)
    new_reservation = strict_request_reservation(prepared, 20000, fmt)
    assert old_reservation - new_reservation > 5000
    assert list(packet["sources"].values()) == list(original["sources"].values())
    llm = SimpleNamespace(_cost_budget=CostBudget(), chat=AsyncMock())
    llm._cost_budget.configure_strict()
    before = llm._cost_budget.snapshot()
    check_grounded_request_budget(llm, prepared, fmt, context_tokens=new_reservation)
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_grounded_request_budget(
            llm, restored, fmt, context_tokens=new_reservation
        )
    oversized = copy.deepcopy(prepared)
    oversized[1]["content"] += " complete material" * 2000
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_grounded_request_budget(
            llm, oversized, fmt, context_tokens=new_reservation
        )
    assert llm._cost_budget.snapshot() == before
    llm.chat.assert_not_awaited()


@pytest.mark.parametrize(
    "fault", ["feedback_collision", "extra_policy", "forged_marker"]
)
def test_reference_projection_never_drops_conflicting_or_unknown_contract_fields(fault):
    _, messages = repair()
    body = json.loads(messages[1]["content"])
    if fault == "feedback_collision":
        feedback = json.loads(body["located_feedback"])
        feedback[0]["fact_card_ids"] = ["forged"]
        body["located_feedback"] = json.dumps(feedback)
    elif fault == "forged_marker":
        body["current_draft"] += "[F:forged]"
    else:
        policy = json.loads(messages[2]["content"])
        policy["must_preserve_claims"] = ["c"]
        messages[2]["content"] = json.dumps(policy)
    messages[1]["content"] = json.dumps(body)
    with pytest.raises(ValueError, match="strict_writing_reference"):
        prepare_grounded_request(messages)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_transport_markers_cannot_leak_into_any_rendered_scientific_unit(
    native, language
):
    from test_strict_quality import to_native

    from litagent.quality import render_grounded_report

    cards, selection, report, _ = packet(language=language)
    report["sections"][1]["blocks"][0]["sentences"][0]["text"] += " [F:forged]"
    raw = json.dumps(to_native(report) if native else report)
    with pytest.raises(ValueError, match="strict_writing_host_content_invalid"):
        render_grounded_report(
            raw,
            cards,
            "academic_v2",
            language,
            evidence_selection=selection,
            require_native=native,
        )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_duplicate_template_subheading_is_not_rendered_and_science_is_preserved(
    language,
):
    from litagent.quality import render_grounded_report
    from litagent.review_structure import structure_sections

    cards, selection, report, _ = packet(language=language)
    baseline = render_grounded_report(
        json.dumps(report), cards, "academic_v2", language, evidence_selection=selection
    )
    for section, (_, title) in zip(
        report["sections"], structure_sections("academic_v2", language)
    ):
        if section["section_key"] not in {"abstract", "conclusion"}:
            section["blocks"][0]["heading"] = title
    body = render_grounded_report(
        json.dumps(report), cards, "academic_v2", language, evidence_selection=selection
    )
    assert body == baseline
    assert body.count("[E:") == baseline.count("[E:") == 7


def test_pending_benchmark_slot_is_not_counted_as_terminal_scientific_failure():
    from test_strict_automatic import observation

    from litagent.benchmark.quality_suite import summarize

    state = {
        "observations": [
            {"phase": "pilot", **observation(False)},
            {"phase": "pilot", "status": "started_no_replay"},
        ]
    }
    original = copy.deepcopy(state)
    counts = summarize(state)["pilot"]
    assert counts["planned"] == 3 and counts["run"] == 2 and counts["not_run"] == 1
    assert counts["pending"] == counts["terminal"] == counts["blocked"] == 1
    assert counts["automatic_passed"] == 0 and counts["pass_rate_planned"] == 0
    assert state == original
