"""Tests for shared evaluation result and threshold behavior."""

import pytest

from litagent.eval.base import EvalResult, Evaluator


def test_eval_result_fields():
    r = EvalResult(metric="citation", score=0.9, passed=True)
    assert r.details == {} and r.skipped is False


def test_eval_result_skip_is_neutral():
    """Skipped evaluations are neutral rather than failed."""
    r = EvalResult.skip("faithfulness", "ragas not installed")
    assert r.skipped is True
    assert r.passed is True
    assert r.score == 0.0
    assert r.details["skipped_reason"] == "ragas not installed"


def test_details_not_shared():
    """Evaluation detail mappings are not shared between instances."""
    a = EvalResult(metric="x", score=1.0, passed=True)
    b = EvalResult(metric="y", score=1.0, passed=True)
    a.details["k"] = "v"
    assert b.details == {}


class _DummyEvaluator(Evaluator):
    """Evaluator test double with a fixed metric name."""

    @property
    def metric_name(self) -> str:
        return "dummy"

    async def evaluate(self, survey, context):
        return self._make_result(0.75)


@pytest.mark.asyncio
async def test_evaluator_make_result_applies_threshold():
    """Result construction applies the configured threshold."""
    ev = _DummyEvaluator(threshold=0.8)
    r = await ev.evaluate("draft", {})
    assert r.score == 0.75
    assert r.passed is False
    assert r.metric == "dummy"


@pytest.mark.asyncio
async def test_evaluator_threshold_override():
    ev = _DummyEvaluator(threshold=0.7)
    r = await ev.evaluate("draft", {})
    assert r.passed is True
