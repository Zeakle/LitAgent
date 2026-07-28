"""Tests for citation-accuracy evaluation."""

import json

import pytest

from litagent.eval.citation import CitationEvaluator
from litagent.eval.base import CTX_PAPERS
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
