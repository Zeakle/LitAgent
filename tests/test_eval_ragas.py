"""Tests for RAGAS faithfulness evaluation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litagent.config import load_config
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator


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


@pytest.mark.asyncio
async def test_score_ragas_uses_configured_provider_adapter():
    class Adapter:
        def __init__(self):
            self.kwargs = None

        async def score_ragas_faithfulness(self, **kwargs):
            self.kwargs = kwargs
            return 0.91, {"attempts": 1}

    adapter = Adapter()
    evaluator = RagasFaithfulnessEvaluator(load_config(), llm=adapter)

    result = await evaluator.evaluate("survey", _evidence_context())

    assert result.score == 0.91
    assert result.details["provider"] == {"attempts": 1}
    assert adapter.kwargs["thinking_mode"] == "disabled"
    assert adapter.kwargs["max_tokens"] == load_config().eval.max_tokens


@pytest.mark.asyncio
async def test_diagnostic_uses_configured_provider_controls():
    llm = AsyncMock()
    llm.chat.return_value = SimpleNamespace(content='{"unsupported_claims": []}')
    config = load_config()
    evaluator = RagasFaithfulnessEvaluator(config, llm=llm)

    result = await evaluator._diagnose("survey", _evidence_context())

    assert result == {"unsupported_claims": []}
    assert llm.chat.await_args.kwargs["max_tokens"] == config.eval.max_tokens
    assert llm.chat.await_args.kwargs["extra_body"] == {
        "thinking": {"type": "disabled"}
    }


@pytest.mark.asyncio
async def test_diagnostic_timeout_returns_stable_skip_reason():
    config = load_config()
    config = config.model_copy(
        update={"eval": config.eval.model_copy(update={"timeout_seconds": 0.01})}
    )
    llm = AsyncMock()

    async def never_returns(*args, **kwargs):
        await asyncio.sleep(1)

    llm.chat.side_effect = never_returns
    evaluator = RagasFaithfulnessEvaluator(config, llm=llm)

    result = await evaluator._diagnose("survey", _evidence_context())

    assert result == {"diagnostic_skipped": "diagnostic_timeout"}
