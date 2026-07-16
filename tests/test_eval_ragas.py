import pytest
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.eval.base import CTX_CLAIMS
from litagent.config import load_config


def _ev(threshold=0.8):
    return RagasFaithfulnessEvaluator(load_config(), threshold)


@pytest.mark.asyncio
async def test_empty_survey_skips():
    r = await _ev().evaluate("  ", {CTX_CLAIMS: [{"text": "x"}]})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_no_contexts_skips():
    """无 claims 也无 papers → skip。"""
    r = await _ev().evaluate("some survey", {})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_ragas_not_installed_skips(monkeypatch):
    """模拟 ragas 未装：_score_ragas 抛 ImportError → skip，不崩。"""
    ev = _ev()
    async def _boom(*a, **k):
        raise ImportError("No module named 'ragas'")
    monkeypatch.setattr(ev, "_score_ragas", _boom)
    r = await ev.evaluate("survey", {CTX_CLAIMS: [{"text": "ProtoNet 93.2%"}]})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_score_from_ragas(monkeypatch):
    """_score_ragas 返回分数 → _make_result 正常算 passed。"""
    ev = _ev(threshold=0.8)
    async def _fake(*a, **k):
        return 0.9
    monkeypatch.setattr(ev, "_score_ragas", _fake)
    r = await ev.evaluate("survey", {CTX_CLAIMS: [{"text": "c"}]})
    assert r.score == 0.9 and r.passed is True
    assert r.details["contexts_count"] == 1


@pytest.mark.asyncio
async def test_nan_skips(monkeypatch):
    """ragas 返回 nan → skip，不当 0 分。"""
    ev = _ev()
    async def _nan(*a, **k):
        return float("nan")
    monkeypatch.setattr(ev, "_score_ragas", _nan)
    r = await ev.evaluate("survey", {CTX_CLAIMS: [{"text": "c"}]})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_claims_preferred_over_papers():
    """_build_contexts：有 claims 用 claims 文本。"""
    ev = _ev()
    ctx = {CTX_CLAIMS: [{"text": "claim1"}, {"text": "claim2"}]}
    assert ev._build_contexts(ctx) == ["claim1", "claim2"]


@pytest.mark.asyncio
async def test_build_contexts_falls_back_to_papers():
    """无 claims → 回退用 papers 的 abstract（验证第 43 行修复）。"""
    from litagent.eval.base import CTX_PAPERS
    ev = _ev()
    ctx = {CTX_PAPERS: [{"abstract": "abs1"}, {"abstract": "abs2"}, {"no_abstract": "x"}]}
    assert ev._build_contexts(ctx) == ["abs1", "abs2"]
