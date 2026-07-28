"""Tests for circuit-breaking and extraction fallback behavior."""

import pytest

from litagent.errors.circuit_breaker import CircuitBreaker, CircuitState
from litagent.agents.extraction_strategy import ResilientExtractionStrategy
from litagent.tools.registry import ToolRegistry
from litagent.tools.base import ToolDefinition, ToolCategory
from litagent.tools.executor import ToolExecutor


class TestCircuitBreaker:
    """Tests circuit-breaker state transitions."""

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

        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow() is False

    def test_half_open_after_cooldown_no_crash(self):

        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=0)
        cb.record_failure()
        cb.record_failure()
        assert cb.allow() is True
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow() is False

    def test_half_open_success_closes(self):
        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=0)
        cb.record_failure()
        cb.record_failure()
        cb.allow()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(fail_threshold=2, cooldown_seconds=0)
        cb.record_failure()
        cb.record_failure()
        cb.allow()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_fail_count(self):
        cb = CircuitBreaker(fail_threshold=3, cooldown_seconds=60)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_probe_resets_across_cycles(self):

        cb = CircuitBreaker(fail_threshold=1, cooldown_seconds=0)
        cb.record_failure()
        assert cb.allow() is True
        cb.record_failure()
        assert cb.allow() is True


class _RaisingLLM:
    """LLM extractor that always raises."""

    async def extract(self, paper):
        raise RuntimeError("llm down")


class _StubRegex:
    """Regex extractor that returns a fixed result."""

    async def extract(self, paper):
        return {
            "claims": ["fallback claim"],
            "metrics": {},
            "methods": [],
            "datasets": [],
        }


class TestResilientExtraction:
    """Tests extraction fallback behavior."""

    @pytest.mark.asyncio
    async def test_llm_failure_degrades_to_regex(self):

        strat = ResilientExtractionStrategy(_RaisingLLM(), _StubRegex())
        out = await strat.extract({"paper_id": "x"})
        assert out["claims"] == ["fallback claim"]


_call_count = 0


async def _boom(**kwargs):
    global _call_count
    _call_count += 1
    raise RuntimeError("boom")


class TestExecutorCircuitBreaker:
    """Tests tool-executor circuit breaking."""

    @pytest.mark.asyncio
    async def test_open_circuit_skips_real_call(self):
        global _call_count
        _call_count = 0
        reg = ToolRegistry()
        reg.register(
            ToolDefinition(
                name="boom",
                description="always fails",
                category=ToolCategory.READ,
                max_retries=0,
            ),
            _boom,
        )
        ex = ToolExecutor(reg, cb_fail_threshold=2, cb_cooldown_seconds=60)

        await ex.execute("boom", {})
        await ex.execute("boom", {})
        assert _call_count == 2

        res = await ex.execute("boom", {})
        assert res.error == "Circuit breaker open"
        assert _call_count == 2
