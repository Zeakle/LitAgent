"""Tests for tracing context, span lifecycle, and LLM events."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from litagent.observability.context import set_task_id, get_task_id, reset_task_id
from litagent.observability.tracing import LangFuseTracer


class TestContextVar:
    """Tests task-ID context isolation."""

    @pytest.mark.asyncio
    async def test_isolates_concurrent_tasks(self):
        """Concurrent tasks retain independent task IDs."""
        seen = {}

        async def worker(tid):
            tok = set_task_id(tid)
            await asyncio.sleep(0.01)
            seen[tid] = get_task_id()
            reset_task_id(tok)

        await asyncio.gather(worker("a"), worker("b"))
        assert seen == {"a": "a", "b": "b"}

    def test_default_empty(self):
        assert get_task_id() == ""

    def test_set_get_reset(self):
        tok = set_task_id("x")
        assert get_task_id() == "x"
        reset_task_id(tok)
        assert get_task_id() == ""


class TestTracerNoOp:
    """Tests disabled tracer behavior."""

    def test_noop_without_keys(self):
        """Missing credentials leave tracing disabled."""
        tracer = LangFuseTracer(host="", public_key="", secret_key="")
        tracer("survey.start", {"query": "x"})
        tracer("worker.start", {"task_id": "t1", "agent_type": "search"})
        tracer(
            "tool.start",
            {"operation_id": "op", "task_id": "t1", "name": "search_arxiv", "args": {}},
        )
        tracer("survey.complete", {})
        tracer.flush()

    def test_call_returns_early_when_disabled(self):
        tracer = LangFuseTracer(host="", public_key="", secret_key="")
        assert tracer._client is None


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
    root = _MockSpan(tag="root", log=log)
    tracer._client = object()
    tracer._root = root
    return tracer, log


class TestTracerHandlers:
    """Tests lifecycle-event handling by the tracer."""

    def test_worker_input_is_attached_and_cancel_closes_span(self):
        tracer, log = _tracer_with_mock()
        worker_input = {"query": "few-shot", "papers": [{"title": "Paper"}]}

        tracer._handle(
            "worker.input",
            {
                "task_id": "t1",
                "agent_type": "extractor",
                "input": worker_input,
            },
        )
        tracer._handle(
            "worker.start",
            {
                "task_id": "t1",
                "agent_type": "extractor",
                "description": "extract",
            },
        )
        span = tracer._spans["t1"]
        starts = [kwargs for action, kwargs in log if action == "start"]
        assert starts[-1]["input"] == worker_input

        tracer._handle(
            "worker.cancelled",
            {
                "task_id": "t1",
                "agent_type": "extractor",
                "error": "user_cancelled",
            },
        )
        assert span.ended
        assert "t1" not in tracer._spans

    def test_worker_span_nests_and_closes(self):
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "search"})
        assert "t1" in tracer._spans
        tracer._handle("worker.complete", {"task_id": "t1"})
        assert "t1" not in tracer._spans

    def test_llm_lifecycle_nests_under_worker(self):
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "synthesis"})
        worker_span = tracer._spans["t1"]
        tracer._handle(
            "llm.start",
            {
                "operation_id": "op-1",
                "task_id": "t1",
                "model": "mock",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        tracer._handle(
            "llm.complete",
            {
                "operation_id": "op-1",
                "content": "hello",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "elapsed_ms": 12,
            },
        )

        starts = [k for act, k in worker_span.log if act == "start"]
        assert any(k.get("as_type") == "generation" for k in starts)
        updates = [k for act, k in worker_span.log if act == "update"]
        assert any(k.get("output") == "hello" for k in updates)
        assert any(
            k.get("usage_details", {}).get("total_tokens") == 15 for k in updates
        )

    def test_tool_call_generation_has_structured_output(self):
        tracer, _ = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "synthesis"})
        worker_span = tracer._spans["t1"]
        tracer._handle(
            "llm.start",
            {
                "operation_id": "op-tool",
                "task_id": "t1",
                "model": "mock",
                "messages": [{"role": "user", "content": "load skill"}],
            },
        )
        tool_calls = [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "load_skill", "arguments": {"name": "cv"}},
            }
        ]
        tracer._handle(
            "llm.complete",
            {
                "operation_id": "op-tool",
                "content": "",
                "tool_calls": tool_calls,
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
                "elapsed_ms": 12,
            },
        )

        updates = [kwargs for action, kwargs in worker_span.log if action == "update"]
        assert any(
            kwargs.get("output") == {"content": "", "tool_calls": tool_calls}
            for kwargs in updates
        )

    def test_rag_span_has_full_input_and_output(self):
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "rag.search.start",
            {
                "operation_id": "rag-1",
                "task_id": "",
                "query": "few shot",
                "top_k": 5,
            },
        )
        tracer._handle(
            "rag.search.complete",
            {
                "operation_id": "rag-1",
                "task_id": "",
                "count": 1,
                "results": [{"title": "Paper"}],
                "elapsed_ms": 7,
            },
        )

        starts = [k for act, k in log if act == "start"]
        updates = [k for act, k in log if act == "update"]
        assert starts[-1]["input"] == {"query": "few shot", "top_k": 5}
        assert updates[-1]["output"]["results"] == [{"title": "Paper"}]

    def test_tool_lifecycle_emits_tool_span(self):
        """Tool lifecycle events produce a tool span."""
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "search"})
        worker_span = tracer._spans["t1"]
        tracer._handle(
            "tool.start",
            {
                "operation_id": "op-t",
                "task_id": "t1",
                "name": "search_arxiv",
                "args": {},
            },
        )
        assert "op-t" in tracer._operations
        tracer._handle(
            "tool.complete",
            {
                "operation_id": "op-t",
                "task_id": "t1",
                "name": "search_arxiv",
                "elapsed_ms": 20,
            },
        )
        assert "op-t" not in tracer._operations
        starts = [k for act, k in worker_span.log if act == "start"]
        assert any(k.get("as_type") == "tool" for k in starts)

    def test_io_event_orphan_falls_back_to_root(self):
        """Orphaned I/O events attach to the root span."""
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "memory.write.start",
            {"operation_id": "op-m", "task_id": "", "layer": "episodic"},
        )

        assert any(act == "start" for act, _ in log)
        tracer._handle(
            "memory.write.complete",
            {
                "operation_id": "op-m",
                "task_id": "",
                "elapsed_ms": 5,
                "episode_id": "e1",
            },
        )
        assert tracer._operations == {}

    def test_orphan_spans_closed_on_survey_complete(self):
        """Survey completion closes orphaned spans."""
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "search"})
        orphan = tracer._spans["t1"]
        tracer._handle("survey.complete", {})
        assert orphan.ended
        assert tracer._spans == {}

    def test_survey_complete_closes_root(self):
        tracer, log = _tracer_with_mock()
        root = tracer._root
        tracer._handle("survey.complete", {"rounds": 3, "accepted": True})
        assert root.ended
        assert tracer._root is None

    def test_survey_complete_output_includes_quality_and_delivery(self):
        """Survey output includes quality and delivery metadata."""
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "survey.complete",
            {
                "rounds": 1,
                "accepted": True,
                "quality_status": "passed",
                "delivery_status": "ready",
            },
        )
        updates = [k for act, k in log if act == "update"]
        assert any(
            k.get("output", {}).get("quality_status") == "passed" for k in updates
        )
        assert any(
            k.get("output", {}).get("delivery_status") == "ready" for k in updates
        )

    def test_subspan_nests_under_parent_worker(self):
        """Subspans attach to their parent worker span."""
        tracer, log = _tracer_with_mock()

        tracer._handle(
            "worker.start", {"task_id": "adv", "agent_type": "adversarial_review"}
        )
        parent_span = tracer._spans["adv"]

        tracer._handle(
            "subspan.start",
            {
                "task_id": "adv:synthesis:r1",
                "parent_task_id": "adv",
                "name": "synthesis.r1",
                "round": 1,
            },
        )

        parent_starts = [k for act, k in parent_span.log if act == "start"]
        assert any(k.get("name") == "synthesis.r1" for k in parent_starts)
        assert "adv:synthesis:r1" in tracer._spans

    def test_subspan_end_removes_from_spans(self):
        """Ending a subspan removes it from active spans."""
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "worker.start", {"task_id": "adv", "agent_type": "adversarial_review"}
        )
        tracer._handle(
            "subspan.start",
            {
                "task_id": "adv:reviewer:r1",
                "parent_task_id": "adv",
                "name": "reviewer.r1",
            },
        )
        tracer._handle("subspan.end", {"task_id": "adv:reviewer:r1"})
        assert "adv:reviewer:r1" not in tracer._spans

    def test_subspan_empty_parent_falls_back_to_root(self):
        """Subspans without parents attach to the root span."""
        tracer, log = _tracer_with_mock()
        tracer._handle(
            "subspan.start",
            {"task_id": "consolidate", "parent_task_id": "", "name": "consolidate"},
        )

        root_starts = [k for act, k in log if act == "start"]
        assert any(k.get("name") == "consolidate" for k in root_starts)
        assert "consolidate" in tracer._spans


class _FakeMsg:
    """Minimal assistant message response."""

    content = "hello"
    tool_calls = None


class _FakeChoice:
    """Minimal completion choice response."""

    message = _FakeMsg()


class _FakeUsage:
    """Fixed token-usage response."""

    prompt_tokens = 10
    completion_tokens = 5


class _FakeCompletion:
    """Fixed chat-completion response."""

    model = "deepseek-v4-flash"
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeOpenAI:
    """OpenAI-compatible client that returns a fixed completion."""

    class chat:
        """OpenAI-compatible chat namespace."""

        class completions:
            """OpenAI-compatible completions namespace."""

            @staticmethod
            async def create(**kwargs):
                return _FakeCompletion()


class TestClientEmitsLLMCall:
    """Tests LLM lifecycle events from the client."""

    @pytest.mark.asyncio
    async def test_client_emits_llm_lifecycle(self):
        """client.chat() emits paired lifecycle events with real timing metadata."""
        from litagent.llm.client import OpenAICompatibleClient

        events = []
        client = OpenAICompatibleClient(
            base_url="x",
            model="m",
            trace_hook=lambda e, d: events.append((e, d)),
        )
        client._client = _FakeOpenAI()
        tok = set_task_id("t42")
        try:
            await client.chat([{"role": "user", "content": "hi"}])
        finally:
            reset_task_id(tok)
        assert [event for event, _ in events] == ["llm.start", "llm.complete"]
        d = events[1][1]
        assert d["task_id"] == "t42"
        assert d["total_tokens"] == 15
        assert d["content"] == "hello"
        assert "elapsed_ms" in d

    @pytest.mark.asyncio
    async def test_client_emits_full_structured_tool_calls(self):
        from types import SimpleNamespace

        from litagent.llm.client import OpenAICompatibleClient

        message = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="load_skill",
                        arguments='{"name":"cv","api_key":"secret"}',
                    ),
                )
            ],
        )
        completion = SimpleNamespace(
            model="deepseek-v4-flash",
            choices=[SimpleNamespace(message=message)],
            usage=_FakeUsage(),
        )
        fake_openai = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=AsyncMock(return_value=completion)),
            ),
        )
        events = []
        client = OpenAICompatibleClient(
            base_url="x",
            model="m",
            trace_hook=lambda event, data: events.append((event, data)),
        )
        client._client = fake_openai

        response = await client.chat([{"role": "user", "content": "load skill"}])

        assert response.tool_calls[0]["function"]["arguments"] == (
            '{"name":"cv","api_key":"secret"}'
        )
        emitted = events[-1][1]["tool_calls"][0]
        assert emitted["id"] == "call-1"
        assert emitted["function"] == {
            "name": "load_skill",
            "arguments": {"name": "cv", "api_key": "secret"},
        }

    @pytest.mark.asyncio
    async def test_client_no_trace_hook_no_crash(self):
        """The client remains usable without a trace hook."""
        from litagent.llm.client import OpenAICompatibleClient

        client = OpenAICompatibleClient(base_url="x", model="m")
        client._client = _FakeOpenAI()
        resp = await client.chat([{"role": "user", "content": "hi"}])
        assert resp.content == "hello"

    def test_react_does_not_emit_llm_call(self):
        """The ReAct layer does not duplicate client LLM events."""
        from pathlib import Path

        import litagent.agent.react as react_mod

        src = Path(react_mod.__file__).read_text(encoding="utf-8")
        assert "'llm.call'" not in src and '"llm.call"' not in src
