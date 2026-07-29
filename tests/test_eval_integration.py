"""Tests for evaluation aggregation and context construction."""

import pytest

from litagent.runner import LitAgent
from litagent.config import load_config
from litagent.eval.base import EvalResult, CTX_PAPERS, CTX_CLAIMS


class _StubEval:
    """Evaluator test double that returns a fixed result."""

    def __init__(self, metric, score, skipped=False):
        self._m, self._s, self._sk = metric, score, skipped

    @property
    def metric_name(self):
        return self._m

    async def evaluate(self, survey, context):
        if self._sk:
            return EvalResult.skip(self._m, "stub skip")
        return EvalResult(metric=self._m, score=self._s, passed=self._s >= 0.8)


@pytest.mark.asyncio
async def test_evaluate_aggregates_results():
    """Evaluation results are aggregated by metric name."""
    agent = LitAgent(load_config())
    agent._evaluators = [
        _StubEval("citation_accuracy", 0.9),
        _StubEval("internal_consistency", 0.7),
        _StubEval("faithfulness", 0.85, skipped=True),
    ]
    results = {"extract": [{"title": "P", "abstract": "a", "claims": ["c1"]}]}
    out = await agent._evaluate(query="q", survey="survey text", results=results, phase="initial")
    assert out["citation_accuracy"]["score"] == 0.9
    assert out["internal_consistency"]["passed"] is False
    assert out["faithfulness"]["skipped"] is True


@pytest.mark.asyncio
async def test_evaluate_no_evaluators_empty():
    """An empty evaluator set produces an empty result."""
    agent = LitAgent(load_config())
    agent._evaluators = []
    assert await agent._evaluate(query="q", survey="s", results={}, phase="initial") == {}


@pytest.mark.asyncio
async def test_one_evaluator_raises_others_survive():
    """One evaluator failure does not suppress other results."""

    class _Boom:
        """Evaluator test double that always raises."""

        @property
        def metric_name(self):
            return "boom"

        async def evaluate(self, s, c):
            raise RuntimeError("boom")

    agent = LitAgent(load_config())
    agent._evaluators = [_Boom(), _StubEval("citation_accuracy", 0.9)]
    out = await agent._evaluate(query="q", survey="s", results={"extract": [{"title": "P", "claims": []}]}, phase="initial")
    assert "citation_accuracy" in out and "boom" not in out


def test_build_context_from_extractions():
    """Evaluation context is built from extractor output."""
    agent = LitAgent(load_config())
    results = {"extract": [{"title": "P1", "abstract": "a1", "claims": ["x", "y"]}]}
    ctx = agent._build_eval_context(query="q", survey="s", results=results)
    assert ctx[CTX_PAPERS][0]["title"] == "P1"
    assert ctx[CTX_CLAIMS] == [{"text": "x"}, {"text": "y"}]


def test_build_context_picks_extractor_not_search():
    """Extractor output takes precedence over search output."""
    agent = LitAgent(load_config())
    results = {
        "search": [{"title": "SP", "abstract": "sa", "source": "arxiv"}],
        "dedup": [{"title": "SP", "abstract": "sa"}],
        "extract": [{"title": "EP", "abstract": "ea", "claims": ["c1"]}],
    }
    ctx = agent._build_eval_context(query="q", survey="s", results=results)
    assert ctx[CTX_PAPERS][0]["title"] == "EP"
    assert ctx[CTX_CLAIMS] == [{"text": "c1"}]
