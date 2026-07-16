import pytest
from litagent.runner import LitAgent
from litagent.config import load_config
from litagent.eval.base import EvalResult, CTX_PAPERS, CTX_CLAIMS


class _StubEval:
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
    """三评估器结果聚合进 dict，key 是 metric 名。"""
    agent = LitAgent(load_config())
    agent._evaluators = [_StubEval("citation_accuracy", 0.9),
                         _StubEval("internal_consistency", 0.7),
                         _StubEval("faithfulness", 0.85, skipped=True)]
    results = {"extract": [{"title": "P", "abstract": "a", "claims": ["c1"]}]}
    out = await agent._evaluate("survey text", results)
    assert out["citation_accuracy"]["score"] == 0.9
    assert out["internal_consistency"]["passed"] is False
    assert out["faithfulness"]["skipped"] is True


@pytest.mark.asyncio
async def test_evaluate_no_evaluators_empty():
    """无评估器 → 空 dict，不崩。"""
    agent = LitAgent(load_config())
    agent._evaluators = []
    assert await agent._evaluate("s", {}) == {}


@pytest.mark.asyncio
async def test_one_evaluator_raises_others_survive():
    """一个评估器抛异常 → 其他正常聚合（return_exceptions 生效）。"""
    class _Boom:
        @property
        def metric_name(self):
            return "boom"
        async def evaluate(self, s, c):
            raise RuntimeError("boom")
    agent = LitAgent(load_config())
    agent._evaluators = [_Boom(), _StubEval("citation_accuracy", 0.9)]
    out = await agent._evaluate("s", {"extract": [{"title": "P", "claims": []}]})
    assert "citation_accuracy" in out and "boom" not in out


def test_build_context_from_extractions():
    """_build_eval_context 从 extractor result 提 papers + claims。"""
    agent = LitAgent(load_config())
    results = {"extract": [{"title": "P1", "abstract": "a1", "claims": ["x", "y"]}]}
    ctx = agent._build_eval_context(results)
    assert ctx[CTX_PAPERS][0]["title"] == "P1"
    assert ctx[CTX_CLAIMS] == [{"text": "x"}, {"text": "y"}]


def test_build_context_picks_extractor_not_search():
    """search/dedup result 也是 list[dict] 有 title，但只有 extractor 带 claims。
    验证判别式取带 claims 的那个，不误取先插入的 search。"""
    agent = LitAgent(load_config())
    results = {
        "search": [{"title": "SP", "abstract": "sa", "source": "arxiv"}],
        "dedup": [{"title": "SP", "abstract": "sa"}],
        "extract": [{"title": "EP", "abstract": "ea", "claims": ["c1"]}],
    }
    ctx = agent._build_eval_context(results)
    assert ctx[CTX_PAPERS][0]["title"] == "EP"
    assert ctx[CTX_CLAIMS] == [{"text": "c1"}]
