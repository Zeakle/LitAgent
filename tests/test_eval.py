import pytest
from litagent.eval.base import EvalResult, Evaluator


def test_eval_result_fields():
    r = EvalResult(metric="citation", score=0.9, passed=True)
    assert r.details == {} and r.skipped is False   # 默认值


def test_eval_result_skip_is_neutral():
    """skip 的结果：skipped=True 但 passed=True（不拖累 CI）。"""
    r = EvalResult.skip("faithfulness", "ragas not installed")
    assert r.skipped is True
    assert r.passed is True                          # 关键：跳过不算失败
    assert r.score == 0.0
    assert r.details["skipped_reason"] == "ragas not installed"


def test_details_not_shared():
    """两个 EvalResult 的 details 是独立 dict（default_factory 生效）。"""
    a = EvalResult(metric="x", score=1.0, passed=True)
    b = EvalResult(metric="y", score=1.0, passed=True)
    a.details["k"] = "v"
    assert b.details == {}                            # b 不受 a 影响


class _DummyEvaluator(Evaluator):
    @property
    def metric_name(self) -> str:
        return "dummy"

    async def evaluate(self, survey, context):
        return self._make_result(0.75)


@pytest.mark.asyncio
async def test_evaluator_make_result_applies_threshold():
    """_make_result 用评估器阈值算 passed。"""
    ev = _DummyEvaluator(threshold=0.8)
    r = await ev.evaluate("draft", {})
    assert r.score == 0.75
    assert r.passed is False                          # 0.75 < 0.8
    assert r.metric == "dummy"


@pytest.mark.asyncio
async def test_evaluator_threshold_override():
    ev = _DummyEvaluator(threshold=0.7)
    r = await ev.evaluate("draft", {})
    assert r.passed is True                           # 0.75 >= 0.7
