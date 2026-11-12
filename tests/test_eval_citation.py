"""Tests for citation-accuracy evaluation."""

import asyncio
import json

import pytest

from litagent.eval.base import CTX_EVIDENCE, CTX_PAPERS
from litagent.eval.citation import CitationEvaluator
from litagent.llm.client import LLMResponse


class _StubLLM:
    """LLM test double that returns a fixed citation response."""

    def __init__(self, payload: dict | None = None, raise_exc=False):
        self._payload = payload
        self._raise = raise_exc

    async def chat(self, messages, **kwargs):
        if self._raise:
            raise RuntimeError("simulated LLM failure")
        return LLMResponse(content=json.dumps(self._payload), model="stub")


class _SequenceLLM:
    """Return a sequence of raw contents while recording request options."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, BaseException):
            raise value
        return LLMResponse(content=value, model="stub")


PAPERS = {CTX_PAPERS: [{"title": "ProtoNet"}, {"title": "MAML"}]}


@pytest.mark.asyncio
async def test_all_citations_real():
    """Known citations receive a full score."""
    llm = _StubLLM(
        {
            "cited": [
                {"title": "ProtoNet", "in_source": True},
                {"title": "MAML", "in_source": True},
            ]
        }
    )
    ev = CitationEvaluator(llm, threshold=0.8)
    r = await ev.evaluate("draft mentions ProtoNet and MAML", PAPERS)
    assert r.score == 1.0 and r.passed is True
    assert r.details["real_count"] == 2 and r.details["fabricated"] == []


@pytest.mark.asyncio
async def test_fabricated_citation_lowers_score():
    """Fabricated citations reduce the score."""
    llm = _StubLLM(
        {
            "cited": [
                {"title": "ProtoNet", "in_source": True},
                {"title": "FakeNet 2099", "in_source": False},
            ]
        }
    )
    ev = CitationEvaluator(llm, threshold=0.8)
    r = await ev.evaluate("draft", PAPERS)
    assert r.score == 0.5 and r.passed is False
    assert r.details["fabricated"] == ["FakeNet 2099"]


@pytest.mark.asyncio
async def test_no_citations_is_full_score():
    """A report without citations receives a full score."""
    llm = _StubLLM({"cited": []})
    ev = CitationEvaluator(llm, threshold=0.8)
    r = await ev.evaluate("generic text", PAPERS)
    assert r.score == 1.0 and r.skipped is False


@pytest.mark.asyncio
async def test_no_source_papers_skips():
    """Evaluation is skipped when source papers are unavailable."""
    llm = _StubLLM({"cited": []})
    ev = CitationEvaluator(llm, threshold=0.8)
    r = await ev.evaluate("draft", {})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_llm_failure_skips():
    """LLM failures produce a skipped evaluation."""
    ev = CitationEvaluator(_StubLLM(raise_exc=True), threshold=0.8)
    r = await ev.evaluate("draft", PAPERS)
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_cited_null_skips():
    """A null citation list produces a skipped evaluation."""
    ev = CitationEvaluator(_StubLLM({"cited": None}), threshold=0.8)
    r = await ev.evaluate("draft", PAPERS)
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_cited_list_of_strings_no_crash():
    """Malformed string citations do not crash evaluation."""
    ev = CitationEvaluator(_StubLLM({"cited": ["ProtoNet", "FakeNet"]}), threshold=0.8)
    r = await ev.evaluate("draft", PAPERS)
    assert r.score == 0.0
    assert r.details["cited_count"] == 2
    assert set(r.details["fabricated"]) == {"ProtoNet", "FakeNet"}


@pytest.mark.asyncio
async def test_evidence_ids_are_checked_without_calling_llm():
    llm = _SequenceLLM([])
    evaluator = CitationEvaluator(llm)
    context = {
        CTX_EVIDENCE: {
            "known": {"evidence_id": "known"},
        }
    }

    result = await evaluator.evaluate(
        "Supported [E:known]. Unknown [E:missing].", context
    )

    assert result.score == 0.5
    assert result.details["mode"] == "evidence_ledger"
    assert result.details["unknown_evidence_ids"] == ["missing"]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_malformed_evidence_markers_lower_deterministic_citation_score():
    llm = _SequenceLLM([])
    evaluator = CitationEvaluator(llm)
    context = {CTX_EVIDENCE: {"e1": {"evidence_id": "e1"}}}

    result = await evaluator.evaluate("Claim [E:e1; E:e2].", context)

    assert result.score == 0.0
    assert result.passed is False
    assert result.details["cited_count"] == 1
    assert result.details["malformed_evidence_refs"] == ["[E:e1; E:e2]"]
    assert llm.calls == []


@pytest.mark.asyncio
async def test_legacy_audit_retries_empty_response_once_with_thinking_disabled():
    llm = _SequenceLLM(["", '{"cited": []}'])
    evaluator = CitationEvaluator(llm, thinking_mode="disabled")

    result = await evaluator.evaluate("generic text", PAPERS)

    assert result.score == 1
    assert result.details["attempts"] == 2
    assert len(llm.calls) == 2
    assert llm.calls[0]["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.asyncio
async def test_legacy_audit_does_not_swallow_cancellation():
    evaluator = CitationEvaluator(_SequenceLLM([asyncio.CancelledError()]))

    with pytest.raises(asyncio.CancelledError):
        await evaluator.evaluate("generic text", PAPERS)


@pytest.mark.asyncio
async def test_legacy_audit_timeout_is_bounded_and_skipped():
    class HangingLLM:
        async def chat(self, messages, **kwargs):
            await asyncio.sleep(1)

    evaluator = CitationEvaluator(HangingLLM(), timeout_seconds=0.01)

    result = await evaluator.evaluate("generic text", PAPERS)

    assert result.skipped is True
    assert result.details["skipped_reason"] == (
        "legacy_citation_audit_failed:TimeoutError"
    )
    assert result.details["attempts"] == 2
