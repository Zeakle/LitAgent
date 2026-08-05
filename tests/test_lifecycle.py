"""Tests for lifecycle events, sanitization, and trace producers."""

from __future__ import annotations

import asyncio

import pytest

from litagent.observability.lifecycle import sanitize_input, traced_io
from litagent.observability.tracing import LangFuseTracer
from litagent.tools.base import RateLimitConfig, ToolDefinition
from litagent.tools.executor import ToolExecutor
from litagent.tools.registry import ToolRegistry


class TestSanitizeInput:
    """Tests recursive trace-input sanitization."""

    def test_truncates_long_strings(self):
        out = sanitize_input({"query": "x" * 500})
        assert len(out["query"]) < 500
        assert out["query"].endswith("...")

    def test_drops_secret_like_keys(self):
        out = sanitize_input({"api_key": "sk-123", "token": "t", "query": "q"})
        assert "api_key" not in out
        assert "token" not in out
        assert out["query"] == "q"

    def test_none_returns_empty_dict(self):
        assert sanitize_input(None) == {}

    def test_scalars_kept_containers_recursed(self):
        """Sanitization preserves scalars and recurses into containers."""
        out = sanitize_input({"top_k": 20, "flags": [1, 2], "ok": True, "none": None})
        assert out["top_k"] == 20
        assert out["flags"] == [1, 2]
        assert out["ok"] is True
        assert out["none"] is None

    def test_nested_secrets_dropped_at_any_depth(self):
        """Secret-like keys are removed at every nesting depth."""
        out = sanitize_input(
            {
                "headers": {
                    "Authorization": "Bearer sk-live",
                    "Accept": "application/json",
                },
                "cookies": {"session_token": "abc"},
                "batch": [{"api_key": "sk-1", "query": "safe q"}],
            }
        )
        flat = repr(out)
        assert "sk-live" not in flat
        assert "abc" not in flat
        assert "sk-1" not in flat

        assert out["headers"]["Accept"] == "application/json"
        assert "cookies" not in out
        assert out["batch"][0]["query"] == "safe q"
        assert "api_key" not in out["batch"][0]

    def test_depth_limit_bounds_nesting(self):
        d: dict = {"v": "leaf"}
        for _ in range(10):
            d = {"nest": d}
        out = sanitize_input(d)
        assert "depth-limit" in repr(out)
        assert "leaf" not in repr(out)

    def test_item_limit_bounds_collections(self):
        big_list = list(range(200))
        big_dict = {f"k{i}": i for i in range(200)}
        out = sanitize_input({"lst": big_list, "map": big_dict})
        assert len(out["lst"]) <= 51
        assert any("omitted" in str(x) for x in out["lst"])
        assert "_truncated" in out["map"]

    def test_cyclic_structure_does_not_raise(self):
        d: dict = {"q": "ok"}
        d["self"] = d
        out = sanitize_input(d)
        assert out["q"] == "ok"

    def test_custom_object_becomes_type_placeholder(self):
        class Weird:
            """Object whose representation must never be evaluated."""

            def __repr__(self):
                raise RuntimeError("repr must not be called")

        out = sanitize_input({"obj": Weird()})
        assert out["obj"] == "<Weird>"


class TestTracedIO:
    """Tests paired lifecycle events for traced I/O."""

    @pytest.mark.asyncio
    async def test_success_emits_start_then_complete(self):
        events = []
        async with traced_io(
            lambda e, d: events.append((e, d)), "claims.add", {"count": 2}
        ) as outcome:
            outcome["count"] = 2
        assert [e for e, _ in events] == ["claims.add.start", "claims.add.complete"]
        start, comp = events[0][1], events[1][1]
        assert start["operation_id"] == comp["operation_id"]
        assert start["count"] == 2
        assert comp["count"] == 2
        assert "elapsed_ms" in comp

    @pytest.mark.asyncio
    async def test_exception_emits_failed_and_reraises(self):
        events = []
        with pytest.raises(ValueError):
            async with traced_io(
                lambda e, d: events.append((e, d)),
                "memory.write",
                {"layer": "episodic"},
            ):
                raise ValueError("boom")
        assert [e for e, _ in events] == ["memory.write.start", "memory.write.failed"]
        failed = events[1][1]
        assert failed["error_code"] == "io_failed"
        assert failed["error_type"] == "ValueError"
        assert "error" not in failed
        assert "elapsed_ms" in failed

    @pytest.mark.asyncio
    async def test_cancellation_emits_failed_terminal(self):
        """Cancellation emits exactly one failed terminal event."""
        events = []

        async def body():
            async with traced_io(
                lambda e, d: events.append((e, d)), "memory.recall", {}
            ):
                await asyncio.sleep(5)

        t = asyncio.create_task(body())
        await asyncio.sleep(0.02)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert [e for e, _ in events] == ["memory.recall.start", "memory.recall.failed"]
        assert events[1][1]["error_type"] == "CancelledError"

    @pytest.mark.asyncio
    async def test_input_is_sanitized(self):
        events = []
        async with traced_io(
            lambda e, d: events.append((e, d)),
            "claims.search",
            {"query": "q" * 500, "api_key": "sk"},
        ):
            pass
        start = events[0][1]
        assert "api_key" not in start
        assert len(start["query"]) < 500

    @pytest.mark.asyncio
    async def test_complete_sanitizes_outcome_and_preserves_framework_fields(self):
        events = []
        async with traced_io(
            lambda event, data: events.append((event, data)), "claims.add"
        ) as outcome:
            outcome.update(
                {
                    "headers": {"Authorization": "Bearer should-not-appear"},
                    "batch": [{"token": "should-not-appear", "count": 2}],
                    "operation_id": "attacker-value",
                    "elapsed_ms": -1,
                }
            )
        complete = events[-1][1]
        assert "should-not-appear" not in repr(complete)
        assert complete["batch"] == [{"count": 2}]
        assert complete["operation_id"] != "attacker-value"
        assert complete["elapsed_ms"] >= 0

    @pytest.mark.asyncio
    async def test_failed_terminal_never_contains_exception_text(self):
        events = []
        with pytest.raises(RuntimeError):
            async with traced_io(
                lambda event, data: events.append((event, data)), "claims.search"
            ):
                raise RuntimeError("Authorization: Bearer should-not-appear")
        failed = events[-1][1]
        assert failed["error_code"] == "io_failed"
        assert failed["error_type"] == "RuntimeError"
        assert "should-not-appear" not in repr(failed)

    @pytest.mark.asyncio
    async def test_hostile_input_does_not_block_wrapped_io(self):
        """Hostile objects cannot block the wrapped I/O call."""
        ran = []
        cyclic: dict = {}
        cyclic["self"] = cyclic

        class Weird:
            """Object whose representation always raises."""

            def __repr__(self):
                raise RuntimeError("no repr")

        events = []
        async with traced_io(
            lambda e, d: events.append((e, d)), "tool", {"cyc": cyclic, "obj": Weird()}
        ):
            ran.append(True)
        assert ran == [True]
        assert [e for e, _ in events] == ["tool.start", "tool.complete"]


def _executor_with_events(registry=None, **kw):
    events = []
    ex = ToolExecutor(
        registry or ToolRegistry(), trace_hook=lambda e, d: events.append((e, d)), **kw
    )
    return ex, events


class TestToolExecutorLifecycle:
    """Tests tool-executor lifecycle event pairing."""

    @pytest.mark.asyncio
    async def test_success_pairs_start_complete(self):
        registry = ToolRegistry()

        async def ok_tool(**kwargs):
            return [1, 2]

        registry.register(ToolDefinition(name="ok", description="d"), ok_tool)
        ex, events = _executor_with_events(registry)
        result = await ex.execute("ok", {"query": "x"})
        assert result.error is None
        assert [e for e, _ in events] == ["tool.start", "tool.complete"]
        start, comp = events[0][1], events[1][1]
        assert start["operation_id"] == comp["operation_id"]
        assert comp["output_size"] == 2
        assert comp["from_cache"] is False

    @pytest.mark.asyncio
    async def test_unregistered_tool_emits_failed(self):
        ex, events = _executor_with_events()
        result = await ex.execute("nope", {})
        assert result.error is not None
        assert [e for e, _ in events] == ["tool.start", "tool.failed"]
        failed = events[1][1]
        assert "not registered" in failed["error"]
        assert failed["error_code"] == "tool_not_registered"
        assert failed["error_type"] == "KeyError"

    @pytest.mark.asyncio
    async def test_cache_hit_still_pairs(self):
        registry = ToolRegistry()

        async def cached_tool(**kwargs):
            return ["v"]

        registry.register(
            ToolDefinition(name="c", description="d", cache_ttl_ms=60000), cached_tool
        )
        ex, events = _executor_with_events(registry)
        await ex.execute("c", {"query": "same"})
        await ex.execute("c", {"query": "same"})
        names = [e for e, _ in events]
        assert names == ["tool.start", "tool.complete", "tool.start", "tool.complete"]
        assert events[3][1]["from_cache"] is True

    @pytest.mark.asyncio
    async def test_rate_limit_emits_failed(self):
        registry = ToolRegistry()

        async def rl_tool(**kwargs):
            return []

        registry.register(
            ToolDefinition(
                name="rl",
                description="d",
                rate_limit=RateLimitConfig(max_calls=1, window_seconds=60),
            ),
            rl_tool,
        )
        ex, events = _executor_with_events(registry)
        await ex.execute("rl", {})
        await ex.execute("rl", {})
        names = [e for e, _ in events]
        assert names == ["tool.start", "tool.complete", "tool.start", "tool.failed"]
        assert "Rate limit" in events[3][1]["error"]

    @pytest.mark.asyncio
    async def test_retry_exhausted_emits_failed(self):
        registry = ToolRegistry()

        async def bad_tool(**kwargs):
            raise RuntimeError("api down")

        registry.register(
            ToolDefinition(name="bad", description="d", max_retries=0), bad_tool
        )
        ex, events = _executor_with_events(registry)
        result = await ex.execute("bad", {})
        assert result.error is not None
        assert [e for e, _ in events] == ["tool.start", "tool.failed"]
        assert "api down" in events[1][1]["error"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_emits_failed(self):
        registry = ToolRegistry()

        async def bad_tool(**kwargs):
            raise RuntimeError("down")

        registry.register(
            ToolDefinition(name="cb", description="d", max_retries=0), bad_tool
        )
        ex, events = _executor_with_events(registry, cb_fail_threshold=1)
        await ex.execute("cb", {})
        await ex.execute("cb", {})
        names = [e for e, _ in events]
        assert names == ["tool.start", "tool.failed", "tool.start", "tool.failed"]
        assert "Circuit breaker" in events[3][1]["error"]

    @pytest.mark.asyncio
    async def test_cancellation_emits_failed_terminal(self):
        """Cancellation emits exactly one failed terminal event."""
        registry = ToolRegistry()

        async def slow_tool(**kwargs):
            await asyncio.sleep(5)

        registry.register(
            ToolDefinition(name="slow", description="d", max_retries=0), slow_tool
        )
        ex, events = _executor_with_events(registry)
        t = asyncio.create_task(ex.execute("slow", {}))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert [e for e, _ in events] == ["tool.start", "tool.failed"]
        assert events[1][1]["error_type"] == "CancelledError"

    @pytest.mark.asyncio
    async def test_start_args_sanitized(self):
        ex, events = _executor_with_events()
        await ex.execute("nope", {"api_key": "sk-1", "query": "q" * 500})
        args = events[0][1]["args"]
        assert "api_key" not in args
        assert len(args["query"]) < 500


class _MockSpan:
    """Span test double that records lifecycle calls."""

    def __init__(self, tag="span", log=None):
        self.tag = tag
        self.log = log if log is not None else []
        self.ended = False

    def start_observation(self, **kwargs):
        child = _MockSpan(tag=kwargs.get("as_type", "span"), log=self.log)
        self.log.append(("start", kwargs))
        return child

    def update(self, **kwargs):
        self.log.append(("update", kwargs))

    def end(self):
        self.ended = True
        self.log.append(("end", {}))


def _tracer_with_mock():
    tracer = LangFuseTracer(host="", public_key="", secret_key="")
    log = []
    tracer._client = object()
    tracer._root = _MockSpan(tag="root", log=log)
    return tracer, log


class TestTracerIOLifecycle:
    """Tests tracer consumption of I/O lifecycle events."""

    def test_start_complete_pairing(self):
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "tool.start",
            {"operation_id": "op1", "task_id": "", "name": "search_arxiv", "args": {}},
        )
        assert "op1" in tracer._operations
        tracer._handle(
            "tool.complete",
            {
                "operation_id": "op1",
                "task_id": "",
                "name": "search_arxiv",
                "elapsed_ms": 12,
                "from_cache": False,
            },
        )
        assert tracer._operations == {}

        updates = [k for act, k in log if act == "update"]
        assert any("output" in k and "operation_id" not in k["output"] for k in updates)

    def test_failed_sets_error_level(self):
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "claims.add.start", {"operation_id": "op2", "task_id": "", "count": 3}
        )
        tracer._handle(
            "claims.add.failed",
            {
                "operation_id": "op2",
                "task_id": "",
                "elapsed_ms": 5,
                "error_code": "io_failed",
                "error_type": "RuntimeError",
                "error": "Authorization: Bearer should-not-appear",
            },
        )
        assert tracer._operations == {}
        updates = [k for act, k in log if act == "update"]
        assert any(
            k.get("level") == "ERROR" and k.get("status_message") == "io_failed"
            for k in updates
        )

        assert any(
            k.get("metadata", {}).get("error_type") == "RuntimeError" for k in updates
        )

    def test_memory_write_named_by_layer(self):
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "memory.write.start",
            {"operation_id": "op3", "task_id": "", "layer": "episodic"},
        )
        starts = [k for act, k in log if act == "start"]
        assert any(k.get("name") == "memory.write.episodic" for k in starts)

    def test_tool_start_uses_tool_type(self):
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "tool.start",
            {"operation_id": "op4", "task_id": "", "name": "search_arxiv", "args": {}},
        )
        starts = [k for act, k in log if act == "start"]
        assert any(
            k.get("as_type") == "tool" and k.get("name") == "tool:search_arxiv"
            for k in starts
        )

    def test_concurrent_tasks_do_not_cross_parents(self):
        """Concurrent tasks retain distinct parent spans."""
        tracer, _ = _tracer_with_mock()
        span_a, span_b = _MockSpan(log=[]), _MockSpan(log=[])
        tracer._spans["ta"] = span_a
        tracer._spans["tb"] = span_b
        tracer._handle(
            "tool.start",
            {"operation_id": "opA", "task_id": "ta", "name": "t", "args": {}},
        )
        tracer._handle(
            "tool.start",
            {"operation_id": "opB", "task_id": "tb", "name": "t", "args": {}},
        )
        assert len([1 for act, _ in span_a.log if act == "start"]) == 1
        assert len([1 for act, _ in span_b.log if act == "start"]) == 1

    def test_terminal_without_start_is_ignored(self):
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "tool.complete", {"operation_id": "ghost", "task_id": "", "elapsed_ms": 1}
        )
        assert log == []

    def test_orphan_operations_closed_on_survey_complete(self):
        """Survey completion closes orphaned operation spans."""
        tracer, _ = _tracer_with_mock()
        tracer._handle(
            "tool.start",
            {"operation_id": "op9", "task_id": "", "name": "t", "args": {}},
        )
        orphan = tracer._operations["op9"]
        tracer._handle("survey.complete", {})
        assert tracer._operations == {}
        assert orphan.ended


from unittest.mock import AsyncMock, MagicMock


class _FakeEmbedder:
    """Embedder that returns a fixed four-dimensional vector."""

    dim = 4

    def embed(self, text):
        if isinstance(text, list):
            return [[0.0] * 4 for _ in text]
        return [0.0] * 4


class TestClaimsIndexProducer:
    """Tests claims-index lifecycle event production."""

    @staticmethod
    def _trusted_claim(text: str):
        from litagent.rag.claims_index import TrustedClaim

        return TrustedClaim(
            claim_id=f"claim-{text}",
            text=text,
            supporting_text="support",
            run_id="run-1",
            domain="test",
            paper_id="p1",
            evidence_id="e1",
            chunk_key="abstract",
            section="abstract",
            content_scope="abstract",
            content_hash="hash",
            evidence_version="v2",
            quality_status="passed",
            delivery_status="ready",
        )

    @pytest.mark.asyncio
    async def test_add_emits_lifecycle_pair(self, monkeypatch):
        import litagent.rag.claims_index as ci_mod
        from litagent.rag.claims_index import ClaimsIndex

        monkeypatch.setattr(ci_mod, "get_embedder", lambda: _FakeEmbedder())
        events = []
        client = MagicMock()
        client.upsert = AsyncMock()
        ci = ClaimsIndex(client, trace_hook=lambda e, d: events.append((e, d)))
        await ci.upsert_trusted([self._trusted_claim("c1"), self._trusted_claim("c2")])
        assert [e for e, _ in events] == [
            "claims.promote.start",
            "claims.promote.complete",
        ]
        assert events[1][1]["count"] == 2

    @pytest.mark.asyncio
    async def test_add_failure_emits_failed_and_raises(self, monkeypatch):
        import litagent.rag.claims_index as ci_mod
        from litagent.rag.claims_index import ClaimsIndex

        monkeypatch.setattr(ci_mod, "get_embedder", lambda: _FakeEmbedder())
        events = []
        client = MagicMock()
        client.upsert = AsyncMock(side_effect=RuntimeError("qdrant down"))
        ci = ClaimsIndex(client, trace_hook=lambda e, d: events.append((e, d)))
        with pytest.raises(RuntimeError):
            await ci.upsert_trusted([self._trusted_claim("c1")])
        assert [e for e, _ in events] == [
            "claims.promote.start",
            "claims.promote.failed",
        ]

    @pytest.mark.asyncio
    async def test_search_emits_lifecycle_pair(self, monkeypatch):
        import litagent.rag.claims_index as ci_mod
        from litagent.rag.claims_index import ClaimsIndex

        monkeypatch.setattr(ci_mod, "get_embedder", lambda: _FakeEmbedder())
        events = []
        hits = MagicMock()
        hits.points = []
        client = MagicMock()
        client.query_points = AsyncMock(return_value=hits)
        ci = ClaimsIndex(client, trace_hook=lambda e, d: events.append((e, d)))
        await ci.search("q")
        assert [e for e, _ in events] == [
            "claims.search.start",
            "claims.search.complete",
        ]
        assert events[1][1]["count"] == 0


class TestMemoryManagerProducer:
    """Tests memory-manager lifecycle event production."""

    @pytest.mark.asyncio
    async def test_recall_emits_lifecycle_pair(self):
        from litagent.memory.manager import MemoryManager

        events = []
        episodic = MagicMock()
        episodic.search = AsyncMock(return_value=[])
        mm = MemoryManager(
            working=MagicMock(),
            episodic=episodic,
            trace_hook=lambda e, d: events.append((e, d)),
        )
        out = await mm.recall("query text")
        assert out == {"episodes": [], "facts": []}
        assert [e for e, _ in events] == [
            "memory.recall.start",
            "memory.recall.complete",
        ]
        assert events[1][1]["episodes"] == 0
        assert events[1][1]["facts"] == 0

    @pytest.mark.asyncio
    async def test_procedural_write_failure_emits_failed_then_swallows(self):
        """Procedural write failures emit terminal events and degrade safely."""
        from litagent.memory.manager import MemoryManager

        events = []
        procedural = MagicMock()
        procedural.upsert_profile = AsyncMock(side_effect=RuntimeError("pg down"))
        mm = MemoryManager(
            working=MagicMock(),
            episodic=MagicMock(),
            procedural=procedural,
            trace_hook=lambda e, d: events.append((e, d)),
        )
        await mm.record_search_source_execution(subject="arxiv", success=True)
        assert [e for e, _ in events] == ["memory.write.start", "memory.write.failed"]
        assert events[0][1]["layer"] == "procedural"

    @pytest.mark.asyncio
    async def test_procedural_write_success_pair(self):
        from litagent.memory.manager import MemoryManager

        events = []
        procedural = MagicMock()
        procedural.upsert_profile = AsyncMock()
        mm = MemoryManager(
            working=MagicMock(),
            episodic=MagicMock(),
            procedural=procedural,
            trace_hook=lambda e, d: events.append((e, d)),
        )
        await mm.record_search_source_execution(
            subject="arxiv", success=True, duration_ms=42
        )
        assert [e for e, _ in events] == ["memory.write.start", "memory.write.complete"]
        assert events[1][1]["source"] == "arxiv"
        assert events[1][1]["duration_ms"] == 42


class TestMemoryLayerCanonical:
    """Tests canonical memory-layer names across producers and tracing."""

    @staticmethod
    def _mm_for_consolidate(events, store_side_effect=None, monkeypatch=None):
        import litagent.memory.manager as mgr_mod
        from litagent.memory.manager import MemoryManager

        episode = MagicMock()
        episode.extracted_facts = []

        async def fake_consolidate_session(state, session_id, llm=None):
            return episode

        monkeypatch.setattr(mgr_mod, "consolidate_session", fake_consolidate_session)

        working = MagicMock()
        working.get = AsyncMock(
            return_value={"messages": [{"type": "ai", "content": "x"}]}
        )
        episodic = MagicMock()
        episodic.store = AsyncMock(return_value="ep1", side_effect=store_side_effect)
        return MemoryManager(
            working=working,
            episodic=episodic,
            trace_hook=lambda e, d: events.append((e, d)),
        )

    def test_layer_constants_are_canonical(self):
        from litagent.memory.manager import (
            LAYER_EPISODIC,
            LAYER_PROCEDURAL,
            LAYER_SEMANTIC,
        )

        assert LAYER_EPISODIC == "episodic"
        assert LAYER_SEMANTIC == "semantic"
        assert LAYER_PROCEDURAL == "procedural"

    @pytest.mark.asyncio
    async def test_consolidate_success_emits_episodic_pair(self, monkeypatch):
        events = []
        mm = self._mm_for_consolidate(events, monkeypatch=monkeypatch)
        await mm.consolidate("s1")
        writes = [(e, d) for e, d in events if e.startswith("memory.write")]
        assert [e for e, _ in writes] == ["memory.write.start", "memory.write.complete"]
        assert writes[0][1]["layer"] == "episodic"

    @pytest.mark.asyncio
    async def test_consolidate_store_failure_emits_failed_with_canonical_layer(
        self, monkeypatch
    ):
        events = []
        mm = self._mm_for_consolidate(
            events,
            store_side_effect=RuntimeError("qdrant down"),
            monkeypatch=monkeypatch,
        )
        result = await mm.consolidate("s1")
        assert result is not None
        writes = [(e, d) for e, d in events if e.startswith("memory.write")]
        assert [e for e, _ in writes] == ["memory.write.start", "memory.write.failed"]
        assert writes[0][1]["layer"] == "episodic"

    def test_tracer_span_name_uses_canonical_layer(self):
        """Tracer span names use the canonical memory layer."""
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "memory.write.start",
            {"operation_id": "opL", "task_id": "", "layer": "episodic"},
        )
        starts = [k for act, k in log if act == "start"]
        assert any(k.get("name") == "memory.write.episodic" for k in starts)
        assert not any("eepisodic" in str(k.get("name", "")) for k in starts)
