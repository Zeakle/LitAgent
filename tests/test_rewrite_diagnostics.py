"""Verify bounded original materials and exact diagnostics reach one repair call."""

import copy
import json
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.llm.client import LLMResponse
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget


def item(eid, paper="p", support="Original support; conditions apply."):
    """Supply large host metadata that must stay out of model materials."""
    return {
        "evidence_id": eid,
        "paper_id": paper,
        "paper_title": "Original paper",
        "text": "EViT accelerates DeiT-S by 50%.",
        "supporting_text": support,
        "source_spans": [{"host_only": "x" * 20000}],
        "bbox": [1, 2, 3, 4],
        "content_hash": "host_hash",
    }


def serialized(packet):
    """Measure the actual Unicode material payload, including JSON structure."""
    return json.dumps(packet, ensure_ascii=False, separators=(",", ":"))


@pytest.mark.parametrize(
    "papers,expected", [(["p", "p"], 1), (["p", "q"], 2), (["", ""], 2)]
)
def test_complete_original_support_shares_only_within_identified_paper(
    papers, expected
):
    support = '完整原文 with Unicode, quotes " and a qualifier: on ImageNet only.'
    items = [item(f"p{i}:claim:0", paper, support) for i, paper in enumerate(papers)]
    before = copy.deepcopy(items)
    packet = LitAgent._rewrite_material_packet(items)
    assert len(packet["original_supports"]) == expected
    for record in packet["evidence"]:
        source = packet["original_supports"][record["original_support_ref"]]
        assert source["text"] == support
        assert source["paper_id"] == record["paper_id"]
    assert items == before
    assert "host_only" not in serialized(packet)
    assert "content_hash" not in serialized(packet)


def test_rewrite_counts_actual_support_and_skips_whole_oversized_source():
    agent = LitAgent(load_config())
    items = [
        item("huge", support="全文" * 20000),
        item("small", support="Small full original"),
    ]
    ledger = {i["evidence_id"]: i for i in items}
    selected = agent._build_rewrite_evidence(
        diagnostics=[{"reason_code": "uncited_claim", "evidence_ids": []}],
        evidence_ledger=ledger,
        selected_evidence_ids=list(ledger),
        max_items=30,
        max_chars=1000,
    )
    assert [i["evidence_id"] for i in selected] == ["small"]
    packet = agent._rewrite_material_packet(selected, candidate_count=2)
    assert len(serialized(packet)) <= 1000
    assert packet["coverage"]["omitted_items"] == 1
    assert packet["original_supports"]["s1"]["text"] == "Small full original"
    assert items[0]["supporting_text"] == "全文" * 20000


def test_material_cap_includes_json_identity_and_shared_original_text():
    agent = LitAgent(load_config())
    items = [item(str(i), support="Same original " * 40) for i in range(5)]
    ledger = {i["evidence_id"]: i for i in items}
    aliased = [{**entry, "evidence_id": f"e{i + 1}"} for i, entry in enumerate(items)]
    limit = len(
        serialized(agent._rewrite_material_packet(aliased[:2], candidate_count=5))
    )
    selected = agent._build_rewrite_evidence(
        diagnostics=[{"reason_code": "uncited_claim", "evidence_ids": []}],
        evidence_ledger=ledger,
        selected_evidence_ids=list(ledger),
        max_items=30,
        max_chars=limit,
    )
    assert len(selected) == 2
    modeled = [
        {**entry, "evidence_id": f"e{i + 1}"} for i, entry in enumerate(selected)
    ]
    assert (
        len(serialized(agent._rewrite_material_packet(modeled, candidate_count=5)))
        == limit
    )
    assert (
        len(serialized(agent._rewrite_material_packet(aliased[:3], candidate_count=5)))
        > limit
    )


@pytest.mark.parametrize("uncited", [False, True])
def test_repair_evidence_prioritizes_actual_refs_and_never_expands_unrequested_pool(
    uncited,
):
    agent = LitAgent(load_config())
    items = [item("prefix"), item("tail", "other")]
    selected = agent._build_rewrite_evidence(
        diagnostics=[
            {
                "reason_code": "uncited_claim" if uncited else "attribution_mismatch",
                "evidence_ids": ["tail"],
            }
        ],
        evidence_ledger={i["evidence_id"]: i for i in items},
        selected_evidence_ids=["prefix", "tail"],
        max_items=30,
        max_chars=15000,
    )
    assert [i["evidence_id"] for i in selected] == (
        ["tail", "prefix"] if uncited else ["tail"]
    )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_uncited_repair_pool_serves_diagnostic_subjects_before_unrelated_prefix(
    language,
):
    agent = LitAgent(load_config())
    items = [
        {
            **item("unrelated"),
            "paper_title": "Face quality",
            "text": "Face quality is measured.",
        },
        {
            **item("evovit", "evovit"),
            "paper_title": "Evo-ViT",
            "text": "Evo-ViT evolves tokens.",
        },
        {
            **item("evit", "evit"),
            "paper_title": "EViT",
            "text": "EViT accelerates DeiT-S.",
        },
        {
            **item("convit", "convit"),
            "paper_title": "ConViT",
            "text": "ConViT outperforms DeiT.",
        },
    ]
    ledger = {i["evidence_id"]: i for i in items}
    diagnostics = [
        {"reason_code": "uncited_claim", "evidence_ids": [], "claim_text": text}
        for text in (
            ["EViT accelerates DeiT-S", "ConViT outperforms DeiT"]
            if language == "en"
            else ["EViT加速DeiT-S", "ConViT优于DeiT"]
        )
    ]
    selected = agent._build_rewrite_evidence(
        diagnostics=diagnostics,
        evidence_ledger=ledger,
        selected_evidence_ids=list(ledger),
        max_items=2,
        max_chars=15000,
    )
    assert [i["evidence_id"] for i in selected] == ["evit", "convit"]


def report_and_agent(*, budget=None):
    """Use the real synthesis builder and runner, without creating a provider client."""
    survey = "# Review\n\n## Methods\n\nEViT accelerates DeiT-S by 50%.\n\nA different supported finding [E:p:claim:0].\n"
    quote = "EViT accelerates DeiT-S by 50%."
    start = survey.index(quote)
    diagnostic = {
        "claim_id": "claim_0",
        "kind": "scientific",
        "claim_text": quote,
        "source_quote": quote,
        "source_start": start,
        "source_end": start + len(quote),
        "section": "Methods",
        "source_unit_id": "unit:0",
        "source_fragment_id": "unit:0:fragment:0",
        "evidence_ids": [],
        "alignment_status": "exact",
        "citation_scope": "exact",
        "reason_code": "uncited_claim",
        "explanation": "No direct citation.",
        "source_spans": "host_diagnostic_metadata",
    }
    report = {
        "survey": survey,
        "metadata": {},
        "evaluation": {
            "faithfulness": {
                "passed": False,
                "details": {"unsupported_claims": [diagnostic]},
            }
        },
        "quality": {"status": "failed"},
        "approval": {"status": "not_requested"},
    }
    agent = LitAgent(load_config())
    agent._llm = SimpleNamespace(
        chat=AsyncMock(return_value=LLMResponse(content=survey)), _cost_budget=budget
    )
    agent._synthesis = SynthesisWorker(agent._llm)
    agent._evaluate = AsyncMock()
    results = {
        "extract": [
            {
                "paper_id": "p",
                "claims": ["EViT accelerates DeiT-S by 50%."],
                "evidence_items": [item("p:claim:0")],
            }
        ],
        "adversarial_review": {
            "evidence_selection": {
                "valid": True,
                "section_evidence_ids": {"methods": ["p:claim:0"]},
            }
        },
    }
    return report, agent, results


async def attempt(report, agent, results):
    """Exercise one transaction using frozen original diagnostics."""
    return await agent._attempt_evidence_rewrite(
        query="Compare methods",
        report_data=report,
        results=results,
        initial_evaluation=report["evaluation"],
        initial_quality=report["quality"],
    )


@pytest.mark.asyncio
async def test_actual_repair_messages_preserve_anchors_aliases_and_noop_rejection():
    report, agent, results = report_and_agent()
    before = copy.deepcopy(report)
    outcome = await attempt(report, agent, results)
    assert outcome.reason_code == "validation_failed"
    assert "identical_to_original" in outcome.validation.reason_codes
    assert not outcome.committed and report == before
    agent._evaluate.assert_not_awaited()
    agent._llm.chat.assert_awaited_once()
    messages = agent._llm.chat.await_args.args[0]
    assert agent._llm.chat.await_args.kwargs["transport_max_attempts"] == 1
    user = messages[1]["content"]
    diagnostics = json.loads(
        re.search(r"<unsupported_claims>\s*(.*?)\s*</unsupported_claims>", user, re.S)[
            1
        ]
    )
    assert diagnostics[0]["source_quote"] == "EViT accelerates DeiT-S by 50%."
    start, end = diagnostics[0]["source_start"], diagnostics[0]["source_end"]
    assert before["survey"][start:end] == diagnostics[0]["source_quote"]
    assert diagnostics[0]["source_offsets_context"] == "frozen_original_before_aliasing"
    assert "Verify complete original support" in diagnostics[0]["repair_instruction"]
    assert diagnostics[0]["evidence_ids"] == []
    assert "host_diagnostic_metadata" not in user
    packet = json.loads(
        re.search(r"<evidence_ledger>\s*(.*?)\s*</evidence_ledger>", user, re.S)[1]
    )
    assert packet["evidence"][0]["evidence_id"] == "e1"
    assert (
        packet["original_supports"]["s1"]["text"]
        == item("p:claim:0")["supporting_text"]
    )
    policy = json.loads(messages[2]["content"])
    assert policy["must_preserve_evidence_ids"] == ["e1"]
    assert policy["allowed_removed_evidence_ids"] == []
    assert "retain that assertion and put its citation" in messages[0]["content"]
    assert "ONLY the C clause" in messages[0]["content"]


@pytest.mark.asyncio
async def test_cited_diagnostic_aliases_preserve_frozen_anchor_and_shared_reference():
    report, agent, results = report_and_agent()
    diagnostic = report["evaluation"]["faithfulness"]["details"]["unsupported_claims"][
        0
    ]
    quote = "Evo-ViT accelerates DeiT-S by 50% [E:p:claim:0]."
    report["survey"] = report["survey"].replace(diagnostic["source_quote"], quote)
    diagnostic.update(
        source_quote=quote,
        claim_text="Evo-ViT accelerates DeiT-S by 50%",
        evidence_ids=["p:claim:0"],
        reason_code="attribution_mismatch",
        source_end=diagnostic["source_start"] + len(quote),
    )
    agent._llm.chat.return_value = LLMResponse(content=report["survey"])
    before = copy.deepcopy(report)
    outcome = await attempt(report, agent, results)
    assert not outcome.committed and report == before
    messages = agent._llm.chat.await_args.args[0]
    user = messages[1]["content"]
    modeled = json.loads(
        re.search(r"<unsupported_claims>\s*(.*?)\s*</unsupported_claims>", user, re.S)[
            1
        ]
    )[0]
    assert modeled["source_quote"] == quote.replace("[E:p:claim:0]", "[E:e1]")
    assert (
        modeled["source_quote"]
        in re.search(r"<current_draft>\s*(.*?)\s*</current_draft>", user, re.S)[1]
    )
    assert modeled["evidence_ids"] == ["e1"]
    assert report["survey"][modeled["source_start"] : modeled["source_end"]] == quote
    policy = json.loads(messages[2]["content"])
    assert policy["must_preserve_evidence_ids"] == ["e1"]
    assert policy["allowed_removed_evidence_ids"] == []
    agent._evaluate.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["context", "budget"])
async def test_repair_full_request_budget_stops_before_provider_call(cause):
    budget = CostBudget(max_tokens=10) if cause == "budget" else None
    report, agent, results = report_and_agent(budget=budget)
    if cause == "context":
        report["survey"] += "\n\n" + "科学材料" * 16000
    before = copy.deepcopy(report)
    outcome = await attempt(report, agent, results)
    assert (
        outcome.reason_code
        == f"rewrite_{'run' if cause == 'budget' else 'context'}_budget_exceeded"
    )
    agent._llm.chat.assert_not_awaited()
    agent._evaluate.assert_not_awaited()
    assert report == before and not outcome.committed
    if budget:
        assert budget.remaining() == 10 and budget.call_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", ["cancel", "missing_usage"])
async def test_repair_reservation_released_after_cancellation_or_unknown_usage(cause):
    budget = CostBudget(max_tokens=500000)
    report, agent, results = report_and_agent(budget=budget)
    before = copy.deepcopy(report)
    if cause == "cancel":
        import asyncio

        agent._llm.chat.side_effect = asyncio.CancelledError
        with pytest.raises(asyncio.CancelledError):
            await attempt(report, agent, results)
        assert budget.remaining() == 500000 and budget.call_count == 0
    else:
        outcome = await attempt(report, agent, results)
        assert (
            outcome.reason_code == "rewrite_usage_incomplete" and not outcome.committed
        )
        assert 0 < budget.used < 500000 and budget.remaining() == 500000 - budget.used
        assert budget.call_count == 1
    assert report == before
    agent._evaluate.assert_not_awaited()
