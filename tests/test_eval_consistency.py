import json
import pytest
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.llm.client import LLMResponse


class _StubLLM:
    def __init__(self, payload=None, raise_exc=False):
        self._payload = payload
        self._raise = raise_exc

    async def chat(self, messages, **kwargs):
        if self._raise:
            raise RuntimeError("simulated LLM failure")
        return LLMResponse(content=json.dumps(self._payload), model="stub")


@pytest.mark.asyncio
async def test_no_conflicts_full_score():
    ev = ConsistencyEvaluator(_StubLLM({"conflicts": []}), threshold=0.8)
    r = await ev.evaluate("consistent survey", {})
    assert r.score == 1.0 and r.passed is True and r.skipped is False


@pytest.mark.asyncio
async def test_one_conflict_deducts_penalty():
    """1 个矛盾 → 1 - 0.2 = 0.8（边界 pass）。"""
    ev = ConsistencyEvaluator(
        _StubLLM({"conflicts": [{"claim_a": "93.2%", "claim_b": "89%", "reason": "same model"}]}),
        threshold=0.8, penalty=0.2)
    r = await ev.evaluate("survey", {})
    assert abs(r.score - 0.8) < 1e-9
    assert r.details["conflict_count"] == 1


@pytest.mark.asyncio
async def test_many_conflicts_floored_at_zero():
    """6 个矛盾 × 0.2 = 1.2 → 钳到 0。"""
    conflicts = [{"claim_a": f"a{i}", "claim_b": f"b{i}", "reason": "x"} for i in range(6)]
    ev = ConsistencyEvaluator(_StubLLM({"conflicts": conflicts}), threshold=0.8, penalty=0.2)
    r = await ev.evaluate("survey", {})
    assert r.score == 0.0 and r.passed is False


@pytest.mark.asyncio
async def test_empty_survey_skips():
    ev = ConsistencyEvaluator(_StubLLM({"conflicts": []}), threshold=0.8)
    r = await ev.evaluate("   ", {})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_llm_failure_skips():
    ev = ConsistencyEvaluator(_StubLLM(raise_exc=True), threshold=0.8)
    r = await ev.evaluate("survey", {})
    assert r.skipped is True and r.passed is True


@pytest.mark.asyncio
async def test_malformed_conflicts_skips():
    """{"conflicts": null} → 非 list → skip（不当 0 矛盾满分）。"""
    ev = ConsistencyEvaluator(_StubLLM({"conflicts": None}), threshold=0.8)
    r = await ev.evaluate("survey", {})
    assert r.skipped is True and r.passed is True
