"""Phase 10 resilience tests — CircuitBreaker (10.4) + Resilient 提取 (10.5) + ToolExecutor 接入。

含回归测试（⟲）：
- C1: ResilientExtractionStrategy LLM 失败 → 降级 regex（_regex / _regex_strategy 名错）
- C2: CircuitBreaker 冷却期内保持 OPEN（opened_at / _opened_at 不一致）
- C3: HALF_OPEN 转换不崩（logger.debu typo）
"""

import pytest

from litagent.errors.circuit_breaker import CircuitBreaker, CircuitState
from litagent.agents.extraction_strategy import ResilientExtractionStrategy
from litagent.tools.registry import ToolRegistry
from litagent.tools.base import ToolDefinition, ToolCategory
from litagent.tools.executor import ToolExecutor


# ── 10.4 CircuitBreaker ──

class TestCircuitBreaker:
    def test_closed_allows(self):
        cb = CircuitBreaker(fail_threshold=3, cooldown_seconds=60)
        assert cb.allow()
        assert cb.state == CircuitState.CLOSED

    def test_trips_after_threshold(self):
        cb = CircuitBreaker(fail_threshold=3, cooldown_seconds=60)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_blocks_within_cooldown(self):
        # ⟲ C2：trip 后冷却期内 allow() 必须 False（属性名不一致会让它恒 True）
        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

    def test_half_open_after_cooldown_no_crash(self):
        # ⟲ C3：冷却到期转 HALF_OPEN 不应崩（logger.debu typo）；只放一个试探
        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=0)  # 立即冷却到期
        cb.record_failure()
        cb.record_failure()
        assert cb.allow() is True               # 第一个试探放行（且不崩）
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow() is False              # 第二个被拒（单试探）

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=0)
        cb.record_failure()
        cb.record_failure()
        cb.allow()                              # → HALF_OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=0)
        cb.record_failure()
        cb.record_failure()
        cb.allow()                              # → HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_fail_count(self):
        cb = CircuitBreaker(fail_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()                     # 清零
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # 重新计数，未到阈值

    def test_probe_resets_across_cycles(self):
        # _probe_in_flight 在 _trip 复位 → 第二轮冷却后还能再放试探
        cb = CircuitBreaker(fail_threshold=1, cooldown_seconds=0)
        cb.record_failure()                     # → OPEN
        assert cb.allow() is True               # cycle1 probe
        cb.record_failure()                     # HALF_OPEN fail → OPEN（probe 复位）
        assert cb.allow() is True               # cycle2 probe again


# ── 10.5 ResilientExtractionStrategy ──

class _RaisingLLM:
    async def extract(self, paper):
        raise RuntimeError("llm down")


class _StubRegex:
    async def extract(self, paper):
        return {"claims": ["fallback claim"], "metrics": {}, "methods": [], "datasets": []}


class TestResilientExtraction:
    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_regex(self):
        # ⟲ C1：LLM 抛错 → 降级到 regex（_regex / _regex_strategy 名错会崩，论文被丢）
        strat = ResilientExtractionStrategy(_RaisingLLM(), _StubRegex())
        out = await strat.extract({"paper_id": "x"})
        assert out["claims"] == ["fallback claim"]


# ── ToolExecutor 熔断接入 ──

_call_count = 0


async def _boom(**kwargs):
    global _call_count
    _call_count += 1
    raise RuntimeError("boom")


class TestExecutorCircuitBreaker:
    @pytest.mark.asyncio
    async def test_open_circuit_skips_real_call(self):
        global _call_count
        _call_count = 0
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(name="boom", description="always fails",
                           category=ToolCategory.READ, max_retries=0),
            _boom,
        )
        ex = ToolExecutor(reg, cb_fail_threshold=2, cb_cooldown_seconds=60)

        await ex.execute("boom", {})            # fail 1
        await ex.execute("boom", {})            # fail 2 → 跳闸
        assert _call_count == 2

        res = await ex.execute("boom", {})       # 熔断 OPEN → 快速失败，不调用 tool
        assert res.error == "Circuit breaker open"
        assert _call_count == 2                  # 没有再真实调用
