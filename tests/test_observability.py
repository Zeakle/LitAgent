"""Phase 13.0 LangFuse observability tests（不需真 LangFuse）。"""

import asyncio
import pytest

from litagent.observability.context import set_task_id, get_task_id, reset_task_id
from litagent.observability.tracing import LangFuseTracer


# ═══════════════════════════════════════════════════
# contextvar 并发隔离（全覆盖追踪的地基）
# ═══════════════════════════════════════════════════

class TestContextVar:
    @pytest.mark.asyncio
    async def test_isolates_concurrent_tasks(self):
        """两个并发 task 各自 set task_id，互不串。"""
        seen = {}

        async def worker(tid):
            tok = set_task_id(tid)
            await asyncio.sleep(0.01)     # 让出，模拟并发交错
            seen[tid] = get_task_id()     # 应该还是自己的 tid
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


# ═══════════════════════════════════════════════════
# LangFuseTracer no-op（无 key / langfuse 未装）
# ═══════════════════════════════════════════════════

class TestTracerNoOp:
    def test_noop_without_keys(self):
        """空 key → client=None → 所有调用不崩。"""
        tracer = LangFuseTracer(host="", public_key="", secret_key="")
        tracer("survey.start", {"query": "x"})
        tracer("worker.start", {"task_id": "t1", "agent_type": "search"})
        tracer("tool.call", {"task_id": "t1", "name": "search_arxiv", "success": True})
        tracer("survey.complete", {})
        tracer.flush()   # 全程无异常

    def test_call_returns_early_when_disabled(self):
        tracer = LangFuseTracer(host="", public_key="", secret_key="")
        assert tracer._client is None


# ═══════════════════════════════════════════════════
# _handle span 重组（mock client，不连真 LangFuse）
# ═══════════════════════════════════════════════════

class _MockSpan:
    """记录 start_observation / update / end 调用。"""
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
    tracer._client = object()          # 骗过 __call__ 的 not client 检查
    tracer._root = root
    return tracer, log


class TestTracerHandlers:
    def test_worker_span_nests_and_closes(self):
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "search"})
        assert "t1" in tracer._spans
        tracer._handle("worker.complete", {"task_id": "t1"})
        assert "t1" not in tracer._spans   # 关掉后从字典移除

    def test_llm_lifecycle_nests_under_worker(self):
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "synthesis"})
        worker_span = tracer._spans["t1"]
        tracer._handle("llm.start", {"operation_id": "op-1", "task_id": "t1", "model": "mock",
                                      "messages": [{"role": "user", "content": "hi"}]})
        tracer._handle("llm.complete", {"operation_id": "op-1", "content": "hello",
                                         "prompt_tokens": 10, "completion_tokens": 5,
                                         "total_tokens": 15, "elapsed_ms": 12})
        # 在 worker span 下建了 generation
        starts = [k for act, k in worker_span.log if act == "start"]
        assert any(k.get("as_type") == "generation" for k in starts)

    def test_tool_call_emits_tool_span(self):
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "search"})
        worker_span = tracer._spans["t1"]
        tracer._handle("tool.call", {"task_id": "t1", "name": "search_arxiv",
                                     "success": True, "elapsed_ms": 20})
        starts = [k for act, k in worker_span.log if act == "start"]
        assert any(k.get("as_type") == "tool" for k in starts)

    def test_io_event_orphan_falls_back_to_root(self):
        """task_id 无对应 Worker span（如 consolidate）→ fallback 到 root，不崩。"""
        tracer, log = _tracer_with_mock()
        tracer._handle("memory.write", {"task_id": "", "layer": "episodic", "success": True})
        # root 上建了 span（fallback）
        assert any(act == "start" for act, _ in log)

    def test_orphan_spans_closed_on_survey_complete(self):
        """worker.start 后没等到 complete（模拟取消）→ survey.complete 清扫。"""
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "t1", "agent_type": "search"})
        orphan = tracer._spans["t1"]
        tracer._handle("survey.complete", {})
        assert orphan.ended             # orphan 被 end
        assert tracer._spans == {}      # 字典清空

    def test_survey_complete_closes_root(self):
        tracer, log = _tracer_with_mock()
        root = tracer._root
        tracer._handle("survey.complete", {"rounds": 3, "accepted": True})
        assert root.ended
        assert tracer._root is None

    def test_subspan_nests_under_parent_worker(self):
        """subspan.start 挂到 parent_task_id 对应的 span 下，不是 root（方案 A）。"""
        tracer, log = _tracer_with_mock()
        # 先建 adversarial worker span
        tracer._handle("worker.start", {"task_id": "adv", "agent_type": "adversarial_review"})
        parent_span = tracer._spans["adv"]
        # 在其下建 synthesis 子 span
        tracer._handle("subspan.start", {"task_id": "adv:synthesis:r1",
                                         "parent_task_id": "adv",
                                         "name": "synthesis.r1", "round": 1})
        # 子 span 由 parent（非 root）spawn
        parent_starts = [k for act, k in parent_span.log if act == "start"]
        assert any(k.get("name") == "synthesis.r1" for k in parent_starts)
        assert "adv:synthesis:r1" in tracer._spans          # 存进 _spans

    def test_subspan_end_removes_from_spans(self):
        """subspan.end 关闭子 span 并从 _spans 移除（配对逻辑）。"""
        tracer, log = _tracer_with_mock()
        tracer._handle("worker.start", {"task_id": "adv", "agent_type": "adversarial_review"})
        tracer._handle("subspan.start", {"task_id": "adv:reviewer:r1",
                                         "parent_task_id": "adv", "name": "reviewer.r1"})
        tracer._handle("subspan.end", {"task_id": "adv:reviewer:r1"})
        assert "adv:reviewer:r1" not in tracer._spans        # 已移除

    def test_subspan_empty_parent_falls_back_to_root(self):
        """parent_task_id 为空（consolidate 场景）→ 挂 root，不崩。"""
        tracer, log = _tracer_with_mock()
        tracer._handle("subspan.start", {"task_id": "consolidate",
                                         "parent_task_id": "", "name": "consolidate"})
        # root 上 spawn 了 consolidate span
        root_starts = [k for act, k in log if act == "start"]
        assert any(k.get("name") == "consolidate" for k in root_starts)
        assert "consolidate" in tracer._spans


# ═══════════════════════════════════════════════════
# 13.0++ llm.call emit 下沉到 client（全覆盖 + 无双计）
# ═══════════════════════════════════════════════════

class _FakeMsg:
    content = "hello"
    tool_calls = None

class _FakeChoice:
    message = _FakeMsg()

class _FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5

class _FakeCompletion:
    model = "deepseek-v4-flash"
    choices = [_FakeChoice()]
    usage = _FakeUsage()

class _FakeOpenAI:
    """替身 openai client：chat.completions.create 返回固定响应。"""
    class chat:
        class completions:
            @staticmethod
            async def create(**kwargs):
                return _FakeCompletion()


class TestClientEmitsLLMCall:
    @pytest.mark.asyncio
    async def test_client_emits_llm_lifecycle(self):
        """client.chat() emits paired lifecycle events with real timing metadata."""
        from litagent.llm.client import OpenAICompatibleClient
        events = []
        client = OpenAICompatibleClient(
            base_url="x", model="m",
            trace_hook=lambda e, d: events.append((e, d)),
        )
        client._client = _FakeOpenAI()          # 绕开真实 openai
        tok = set_task_id("t42")
        try:
            await client.chat([{"role": "user", "content": "hi"}])
        finally:
            reset_task_id(tok)
        assert [event for event, _ in events] == ["llm.start", "llm.complete"]
        d = events[1][1]
        assert d["task_id"] == "t42"
        assert d["total_tokens"] == 15          # 10 + 5
        assert d["content"] == "hello"          # content 会映射为 span output
        assert "elapsed_ms" in d

    @pytest.mark.asyncio
    async def test_client_no_trace_hook_no_crash(self):
        """无 trace_hook 时 chat 正常返回，不崩。"""
        from litagent.llm.client import OpenAICompatibleClient
        client = OpenAICompatibleClient(base_url="x", model="m")
        client._client = _FakeOpenAI()
        resp = await client.chat([{"role": "user", "content": "hi"}])
        assert resp.content == "hello"

    def test_react_does_not_emit_llm_call(self):
        """删掉 react emit 后，react.py 源码里不应再有 llm.call emit（防双计回归）。

        emit 已下沉到 client；react 若再 emit 一次 → LangFuse 每次调用两个
        generation、token 双计。直接断言源码文本比构造整个 LangGraph 更稳。
        """
        from pathlib import Path
        import litagent.agent.react as react_mod
        src = Path(react_mod.__file__).read_text(encoding="utf-8")
        assert "'llm.call'" not in src and '"llm.call"' not in src
