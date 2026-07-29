"""Tests for RAGAS faithfulness evaluation."""

import pytest

from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.config import load_config


def _ev(threshold=0.8):
    return RagasFaithfulnessEvaluator(load_config(), threshold)


def _evidence_context(text: str = "ProtoNet is supported") -> dict:
    item = {
        "evidence_id": "p1:claim:0",
        "paper_id": "p1",
        "paper_title": "Paper",
        "text": text,
        "source_locator": "extracted_claim",
        "confidence": None,
    }
    return {
        "query": "few-shot learning",
        "evidence": {item["evidence_id"]: item},
        "referenced_evidence_ids": [item["evidence_id"]],
        "referenced_evidence": [item],
        "unresolved_evidence_ids": [],
        "selected_evidence": [item],
    }


@pytest.mark.asyncio
async def test_empty_survey_skips():
    r = await _ev().evaluate("  ", _evidence_context())
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_no_contexts_skips():
    """Evaluation is skipped when evidence contexts are absent."""
    r = await _ev().evaluate("some survey", {})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_ragas_not_installed_skips(monkeypatch):
    """A missing RAGAS dependency produces a skipped evaluation."""
    ev = _ev()

    async def _boom(*a, **k):
        raise ImportError("No module named 'ragas'")

    monkeypatch.setattr(ev, "_score_ragas", _boom)
    r = await ev.evaluate("survey", _evidence_context("ProtoNet 93.2%"))
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_score_from_ragas(monkeypatch):
    """The evaluator returns the score reported by RAGAS."""
    ev = _ev(threshold=0.8)

    async def _fake(*a, **k):
        return 0.9

    monkeypatch.setattr(ev, "_score_ragas", _fake)
    r = await ev.evaluate("survey", _evidence_context("claim"))
    assert r.score == 0.9 and r.passed is True
    assert r.details["contexts_count"] == 1


@pytest.mark.asyncio
async def test_nan_skips(monkeypatch):
    """A non-finite RAGAS score produces a skipped evaluation."""
    ev = _ev()

    async def _nan(*a, **k):
        return float("nan")

    monkeypatch.setattr(ev, "_score_ragas", _nan)
    r = await ev.evaluate("survey", _evidence_context("claim"))
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_referenced_evidence_preferred_over_selected():
    ev = _ev()
    context = _evidence_context("referenced")
    context["selected_evidence"] = [
        {
            **context["referenced_evidence"][0],
            "evidence_id": "p2:claim:0",
            "text": "selected fallback",
        }
    ]
    resolved = ev._resolve_contexts(context)
    assert resolved.evidence_ids == ("p1:claim:0",)
    assert "referenced" in resolved.texts[0]
    assert "selected fallback" not in resolved.texts[0]


@pytest.mark.asyncio
async def test_no_report_refs_falls_back_to_selected_evidence():
    ev = _ev()
    context = _evidence_context("selected")
    context["referenced_evidence_ids"] = []
    context["referenced_evidence"] = []
    resolved = ev._resolve_contexts(context)
    assert resolved.evidence_ids == ("p1:claim:0",)
    assert resolved.fallback_used is True
    assert resolved.fallback_reason == "report_has_no_evidence_refs"
