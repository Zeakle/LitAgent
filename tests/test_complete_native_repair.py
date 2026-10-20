"""Exercise complete local repair through real synthesis, review and SDK transport."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_writing_passages import facts

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    GroundedWritingError,
    apply_native_repairs,
    begin_targeted_repair,
    grounded_messages,
    prepare_grounded_request,
    prepare_native_repair_request,
    render_grounded_report_traced,
    targeted_repair_state,
)
from litagent.safety.budget import CostBudget, strict_request_reservation


def source_selection(cards):
    """Keep a single public source explicitly available in each body section."""
    items = {
        c["evidence_id"]: {
            "evidence_id": c["evidence_id"],
            "paper_id": c["paper_id"],
            "text": c["source_quote"],
            "supporting_text": c["source_quote"],
            "chunk_key": "abstract",
        }
        for c in cards.values()
    }
    sections = {
        key: list(items)
        for key in (
            "introduction",
            "taxonomy",
            "methods",
            "experiments",
            "open_problems",
        )
    }
    return EvidenceSelection(
        len(items),
        items,
        sections,
        {k: "lexical_fallback" for k in sections},
        0,
        1000,
    )


def original_with_local_error(cards):
    """Keep the scientific body intact and add one invalid free-text scope claim."""
    document = analysis_document(cards)
    notice = {
        "text": "",
        "kind": "organization",
        "fact_card_ids": [],
        "scope_notice_id": "review_scope",
    }
    document["sections"]["introduction"]["first_block"]["additional_sentences"].append(
        notice
    )
    document["sections"]["experiments"]["first_block"]["additional_sentences"].append(
        {
            "text": "Current research lacks systematic comparisons.",
            "kind": "gap",
            "fact_card_ids": [],
            "scope_notice_id": "none",
        }
    )
    return document


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "protocol", "unknown_usage", "cancelled"])
async def test_bounded_revisions_use_original_and_preserve_every_response(
    tmp_path, fault
):
    cards = facts()
    original = original_with_local_error(cards)
    first_patch = {
        "can_repair": True,
        "patches": {
            "u0": {
                "text": "",
                "kind": "organization",
                "fact_card_ids": [],
                "scope_notice_id": "review_scope",
            }
        },
    }
    final_patch = {
        "can_repair": True,
        "patches": {
            "u0": {
                "text": "EViT improves speed by 50% on DeiT-S.",
                "kind": "scientific",
                "fact_card_ids": list(cards),
                "scope_notice_id": "none",
            }
        },
    }
    reviewer = {
        "score": 0.9,
        "verdict": "accept",
        "evidence_compliance": {"passed": True},
    }
    from litagent.native_wire import array_from_native

    outputs = [array_from_native(original), first_patch, final_patch, reviewer]
    wire, events = [], []

    def transport(request):
        payload = json.loads(request.content)
        wire.append(payload)
        i = len(wire) - 1
        if i == 1 and fault == "cancelled":
            raise asyncio.CancelledError()
        content = (
            "invalid JSON" if i == 1 and fault == "protocol" else json.dumps(outputs[i])
        )
        response = {
            "id": f"offline-{i}",
            "object": "chat.completion",
            "created": 0,
            "model": payload["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
        }
        if not (i == 1 and fault == "unknown_usage"):
            response["usage"] = {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            }
        return httpx.Response(200, json=response)

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    owner = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "lite",
        api_key="offline",
        cost_budget=budget,
        trace_hook=lambda event, data: events.append((event, data)),
    )
    await owner._client.close()
    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/v1",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    pro = ModelBoundClient(owner, "pro", "generation")
    lite = ModelBoundClient(owner, "lite", "evaluation")
    path = tmp_path / "repairs.json"
    synthesis = SynthesisWorker(
        pro,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=source_selection(cards))
        ),
        run_id_provider=lambda: "bounded-native",
        repair_state_path_provider=lambda: path,
        thinking_mode="disabled",
    )
    worker = AdversarialReviewWorker(pro, synthesis, ReviewerWorker(lite), max_rounds=1)
    task = SubTask(
        "adversarial_review",
        "write",
        "adversarial_review",
        input_data={
            "query": "Explain EViT speed on DeiT-S",
            "quality_policy": "strict_auto_v2",
            "review_structure": "academic_v2",
            "body_quality": "analysis_v1",
            "report_language": "en",
            "verification": {"judge": False, "jev": False},
        },
    )
    try:
        if fault:
            with pytest.raises((ValueError, asyncio.CancelledError)):
                await worker.execute(task)
            assert len(wire) == 2
        else:
            result = await worker.execute(task)
            assert [p["model"] for p in wire] == ["pro", "pro", "pro", "lite"]
            assert budget.used == 120 and budget.snapshot()["requests_sent"] == 4
            first = json.loads(wire[0]["messages"][1]["content"])
            assert first["section_source_contract"] == "chapter-source-menu-v1"
            assert all(p.get("thinking", {}).get("type") != "enabled" for p in wire)
            repair1 = json.loads(wire[1]["messages"][1]["content"])
            repair2 = json.loads(wire[2]["messages"][1]["content"])
            assert (
                repair1["current_candidate"] == repair2["current_candidate"] == original
            )
            assert repair2["previous_local_failure"]["attempt_index"] == 0
            assert (
                repair2["previous_local_failure"]["diagnostics"][0]["reason_code"]
                == "strict_writing_scope_notice_repeated"
            )
            state = result["targeted_repair_state"]
            assert [a["status"] for a in state["attempts"]] == ["failed", "completed"]
            with pytest.raises(ValueError, match="limit_exceeded"):
                begin_targeted_repair(state, "evaluation", "b" * 64)
            assert result["writing_bindings"]["report_sha256"]
            assert all(
                "candidate_card_ids" in d
                for d in result["analysis_plan"]["section_duties"]
            )
        saved = json.loads(path.read_text())
        assert len(saved["state"]["attempts"]) == (1 if fault else 2)
        if fault not in {"cancelled", "unknown_usage"}:
            assert saved["native_attempt_0"]["response"]
        if fault is None:
            assert (
                saved["native_attempt_0"]["applied_candidate"]
                != saved["native_attempt_1"]["applied_candidate"]
            )
            assert (
                saved["native_attempt_0"]["candidate"]
                == saved["native_attempt_1"]["candidate"]
            )
        if fault == "unknown_usage":
            assert budget.snapshot()["usage_uncertain"] and budget.stop_reason
            unknown = [
                d for e, d in events if e == "llm.complete" and not d["usage_available"]
            ]
            assert (
                len(unknown) == 1 and json.loads(unknown[0]["content"]) == first_patch
            )
        before = len(wire)
        with pytest.raises(ValueError, match="restart_no_replay"):
            await worker.execute(copy.deepcopy(task))
        assert len(wire) == before
    finally:
        await owner.close()


def real_request():
    """Prepare complete permissions from the untouched received multi-error fixture."""
    f = json.loads(
        (Path(__file__).parent / "fixtures/native-mixed-errors-real.json").read_text()
    )
    kwargs = dict(
        evidence_selection=f["selection"],
        require_native=True,
        require_full_sources=True,
        require_scope_notices=True,
        analysis_mode=True,
    )
    with pytest.raises(GroundedWritingError) as failed:
        render_grounded_report_traced(
            f["candidate"], f["cards"], "academic_v2", "zh-CN", **kwargs
        )
    messages, _ = prepare_grounded_request(
        grounded_messages(
            f["cards"],
            "academic_v2",
            "zh-CN",
            f["query"],
            coverage=f["coverage_requirements"],
            evidence_selection=f["selection"],
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
    )
    prepared, schema, binding = prepare_native_repair_request(
        messages,
        f["candidate"],
        failed.value.diagnostics,
        cards=f["cards"],
        structure="academic_v2",
        language="zh-CN",
        evidence_selection=f["selection"],
    )
    return f, messages, failed.value.diagnostics, prepared, schema, binding


@pytest.mark.parametrize(
    "attack",
    [
        "notice",
        "foreign_card",
        "empty_cards",
        "extra_unit",
        "changed_original",
        "changed_diagnostic",
    ],
)
def test_complete_repair_never_licenses_unrelated_or_unverified_science(attack):
    f, messages, diagnostics, _, _, binding = real_request()
    # This checks local authorization only; these patches have no scientific verdict.
    patches = {
        key: {
            "text": "Local codec candidate.",
            "kind": "scientific",
            "fact_card_ids": target["allowed_fact_card_ids"][:1],
            "scope_notice_id": "none",
        }
        for key, target in binding["targets"].items()
    }
    key = next(k for k, t in binding["targets"].items() if t["must_remain_scientific"])
    candidate = f["candidate"]
    if attack == "notice":
        patches[key].update(
            text="",
            kind="organization",
            fact_card_ids=[],
            scope_notice_id="review_scope",
        )
    elif attack == "foreign_card":
        patches[key]["fact_card_ids"] = [
            next(
                c
                for c in f["cards"]
                if c not in binding["targets"][key]["allowed_fact_card_ids"]
            )
        ]
    elif attack == "empty_cards":
        patches[key]["fact_card_ids"] = []
    elif attack == "extra_unit":
        patches["u999"] = copy.deepcopy(patches[key])
    elif attack == "changed_original":
        doc = json.loads(candidate)
        doc["title"] += " changed"
        candidate = json.dumps(doc)
    elif attack == "changed_diagnostic":
        diagnostics[0]["candidate_text"] += " changed"
        with pytest.raises(ValueError, match="diagnostic_invalid"):
            prepare_native_repair_request(
                messages,
                candidate,
                diagnostics,
                cards=f["cards"],
                structure="academic_v2",
                language="zh-CN",
                evidence_selection=f["selection"],
            )
        return
    with pytest.raises(ValueError):
        apply_native_repairs(
            candidate,
            json.dumps({"can_repair": True, "patches": patches}),
            binding,
            cards=f["cards"],
        )


def test_source_menu_preserves_exact_per_section_permission_and_complete_originals():
    f, messages, _, _, _, _ = real_request()
    packet = json.loads(messages[1]["content"])
    for section, rows in packet["section_source_index"].items():
        assert [f["cards"][r["fact_card_id"]]["evidence_id"] for r in rows] == f[
            "selection"
        ]["section_evidence_ids"][section]
        for row in rows:
            assert (
                packet["sources"][row["source_ref"]]["source_quote"]
                == f["cards"][row["fact_card_id"]]["source_quote"]
            )
    assert (
        "Missing evidence must be disclosed as a gap sentence"
        not in messages[0]["content"]
    )
    assert "FINAL NATIVE CONTRACT" in messages[0]["content"]
    assert all(
        "candidate_sources_ref" in d and "candidate_card_ids" not in d
        for d in packet["analysis_plan"]["section_duties"]
    )


def test_real_complete_repair_fits_the_original_context_and_writing_reserve():
    """Use all original requirements, sources and the unchanged maximum output."""
    f, _, _, prepared, schema, binding = real_request()
    reservation = strict_request_reservation(prepared, 20000, schema)
    assert len(binding["targets"]) == 7
    assert reservation <= int(131072 * 0.9)
    assert reservation <= 300000 - 179439
    packet = json.loads(prepared[1]["content"])
    assert packet["research_question"] == f["query"]
    assert packet["coverage_requirements"] == f["coverage_requirements"]
    for cid, card in f["cards"].items():
        assert (
            packet["sources"][packet["cards"][cid]["source_ref"]]["source_quote"]
            == card["source_quote"]
        )


@pytest.mark.parametrize("fault", ["missing", "unhashable", "wrong_permissions"])
def test_malformed_preparation_plan_is_a_contract_error(fault):
    cards = facts()
    messages = grounded_messages(
        cards,
        "academic_v2",
        "en",
        "EViT speed",
        use_full_sources=True,
        use_scope_notices=True,
        analysis_mode=True,
    )
    packet = json.loads(messages[1]["content"])
    if fault == "missing":
        del packet["analysis_plan"]["section_duties"]
    elif fault == "unhashable":
        packet["analysis_plan"]["section_duties"][0]["section_key"] = []
    else:
        packet["analysis_plan"]["section_duties"][0]["candidate_card_ids"] = []
    messages[1]["content"] = json.dumps(packet)
    with pytest.raises(ValueError, match="strict_writing_schema_contract_invalid"):
        prepare_grounded_request(messages)


@pytest.mark.asyncio
async def test_preflight_failure_with_a_prior_attempt_does_not_loop():
    """A local refusal cannot become a new attempt or spin on the inherited entry."""
    failure = GroundedWritingError("original", [{"reason_code": "local_refusal"}])
    llm = SimpleNamespace(chat=AsyncMock())
    worker = SynthesisWorker(llm)
    worker._repair_native_once = AsyncMock(
        side_effect=[failure, AssertionError("looped")]
    )
    state = targeted_repair_state(None, "inherited")
    entry = begin_targeted_repair(state, "native_initial", "a" * 64)
    entry.update(status="failed", usage={"usage_available": True})
    before = copy.deepcopy(state)
    task = SubTask(
        "adversarial_review",
        "write",
        "synthesis",
        input_data={"body_quality": "analysis_v1"},
    )
    with pytest.raises(GroundedWritingError, match="local_refusal"):
        await worker._repair_native_initial(
            task, [], "original", failure, {}, None, "academic_v2", "en", state, {}
        )
    assert worker._repair_native_once.await_count == 1 and state == before
    llm.chat.assert_not_awaited()
