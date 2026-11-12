"""Distinguish chapter placements from the union of selected evidence."""

from collections import Counter

import pytest
from test_evidence_selector import _FakeReranker, _make_ledger

from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import DEFAULT_SECTIONS, EvidenceSelector


@pytest.mark.asyncio
@pytest.mark.parametrize("per_section", [12, 20])
async def test_shared_chapter_rankings_reuse_evidence_instead_of_multiplying_it(
    per_section,
):
    selector = EvidenceSelector(
        _FakeReranker(),
        BudgetManager(),
        top_k_per_section=per_section,
        per_paper_cap=4,
    )
    result = await selector.select(
        "few-shot methods", _make_ledger(100), max_tokens=16000, reuse_selected=True
    )
    assert len(result.selected_items) == per_section
    assert all(len(ids) == per_section for ids in result.section_evidence_ids.values())
    assert (
        sum(map(len, result.section_evidence_ids.values()))
        == len(DEFAULT_SECTIONS) * per_section
    )
    reasons = Counter(d["reason_code"] for d in result.ranking_decisions)
    assert reasons["selected"] == per_section
    assert reasons["reused_selected"] == 4 * per_section
    assert "global_item_limit" not in reasons
    assert "token_budget" not in reasons


@pytest.mark.asyncio
async def test_twenty_recommendations_still_obey_global_paper_and_token_limits():
    result = await EvidenceSelector(
        _FakeReranker(), BudgetManager(), max_items=8, per_paper_cap=2
    ).select(
        "few-shot methods", _make_ledger(100), max_tokens=16000, reuse_selected=True
    )
    assert len(result.selected_items) == 8
    assert max(map(len, result.section_evidence_ids.values())) <= 20
    assert (
        max(Counter(x["paper_id"] for x in result.selected_items.values()).values())
        <= 2
    )
    assert result.estimated_tokens <= 16000
    assert any(
        d["reason_code"] == "global_item_limit" for d in result.ranking_decisions
    )
