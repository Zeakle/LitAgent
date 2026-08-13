"""Tests for ReAct agent validation and message conversion."""

import pytest

from litagent.agent.react import (
    _route_after_validate,
    build_react_graph,
)
from litagent.agent.state import AgentState
from litagent.agent.validation import validate_tool_call
from litagent.config import AgentConfig
from litagent.llm.client import BaseLLMClient
from tests.conftest import make_mock_agent_node, make_mock_tool


class TestValidateToolCall:
    """Tests tool-call validation."""

    def test_valid_action(self):
        result = validate_tool_call({"name": "search_arxiv", "args": {"q": "test"}})
        assert result["_validation_result"]["valid"] is True

    def test_missing_name(self):
        result = validate_tool_call({"args": {"q": "test"}})
        assert result["_validation_result"]["valid"] is False
        assert "name" in str(result["_validation_result"]["errors"])

    def test_missing_args(self):
        result = validate_tool_call({"name": "search"})
        assert result["_validation_result"]["valid"] is False

    def test_empty_name(self):
        result = validate_tool_call({"name": "", "args": {}})
        assert result["_validation_result"]["valid"] is False

    def test_args_not_dict(self):
        result = validate_tool_call({"name": "search", "args": "not_a_dict"})
        assert result["_validation_result"]["valid"] is False


class TestRouteAfterValidate:
    """Tests routing after tool-call validation."""

    def test_valid_sends_to_tools(self):
        state = {"_validation_result": {"valid": True}}
        assert _route_after_validate(state) == "tools"

    def test_invalid_sends_to_agent(self):
        state = {"_validation_result": {"valid": False, "errors": ["bad"]}}
        assert _route_after_validate(state) == "agent"

    def test_invalid_twice_then_exceeded(self):
        """Repeated invalid calls eventually stop tool execution."""
        state = {"_validation_result": {"valid": False, "retry_count": 3}}
        assert _route_after_validate(state) == "tools"


class TestReactGraph:
    """Tests ReAct graph construction and loop limits."""

    def test_build_graph_compiles(self):
        """The ReAct graph compiles with a mock node."""
        mock = make_mock_agent_node([])
        graph = build_react_graph(mock, [], AgentConfig())
        compiled = graph.compile()
        result = compiled.invoke(
            {
                "messages": [],
                "tools": [],
                "current_thought": "",
                "current_action": None,
                "loop_count": 0,
                "final_answer": None,
            }
        )
        assert result.get("final_answer") == "Task completed."

    def test_max_loops_stops_graph(self):
        """The graph stops at the configured loop limit."""
        cfg = AgentConfig(max_loops=2)
        mock = make_mock_agent_node(
            [{"current_action": {"name": "search", "args": {}}}] * 5
        )
        graph = build_react_graph(mock, [make_mock_tool("search", "mock result")], cfg)
        compiled = graph.compile()
        result = compiled.invoke(
            {
                "messages": [],
                "tools": [],
                "current_thought": "",
                "current_action": None,
                "loop_count": 0,
                "final_answer": None,
            }
        )

        assert result.get("loop_count") == 2


@pytest.fixture(autouse=True)
def _ensure_registry():
    """Reset and populate the tool registry for each test."""
    from litagent.tools.registry import get_registry

    registry = get_registry()
    if len(registry) == 0:
        from litagent.tools.builtin.search import register_search_tools

        register_search_tools()


class TestValidateToolCallWithGraphTools:
    """Tests validation against graph-scoped tool names."""

    def test_worker_tool_accepted(self):
        """A graph-scoped worker tool is accepted."""
        result = validate_tool_call(
            {"name": "recall_memory", "args": {"query": "attention"}},
            graph_tool_names={"recall_memory", "lookup_claims"},
        )
        assert result["_validation_result"]["valid"] is True

    def test_unknown_tool_still_rejected(self):
        """An unknown graph tool remains invalid."""
        result = validate_tool_call(
            {"name": "nonexistent_tool", "args": {}}, graph_tool_names={"recall_memory"}
        )
        assert result["_validation_result"]["valid"] is False
        assert "not registered" in str(result["_validation_result"]["errors"])

    def test_explicit_empty_graph_tool_set_rejects_every_tool(self):
        """An empty graph capability is fail-closed, not an absent policy."""
        result = validate_tool_call(
            {"name": "search_arxiv", "args": {"q": "x"}},
            graph_tool_names=set(),
        )

        assert result["_validation_result"]["valid"] is False
        assert "not registered" in str(result["_validation_result"]["errors"])

    def test_graph_tool_names_none_backward_compat(self):
        """Registry-only validation remains backward compatible."""
        result = validate_tool_call({"name": "search_arxiv", "args": {"q": "x"}})
        assert result["_validation_result"]["valid"] is True


class TestClientToRunnableMessages:
    """Tests conversion from client messages to runnable messages."""

    @pytest.mark.asyncio
    async def test_tool_message_includes_tool_call_id(self):
        """Tool messages preserve their tool-call IDs."""
        from langchain_core.messages import ToolMessage

        from litagent.agent.react import _client_to_runnable
        from litagent.llm.client import BaseLLMClient

        client = _CaptureFormattedClient("ok")
        runnable = _client_to_runnable(client, tools=None)

        state = {
            "messages": [
                ToolMessage(content="search result", tool_call_id="call_abc123")
            ]
        }
        await runnable.ainvoke(state)
        formatted = client.last_formatted
        tool_msg = next(m for m in formatted if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "call_abc123"
        assert tool_msg["content"] == "search result"

    @pytest.mark.asyncio
    async def test_aimessage_includes_tool_calls(self):
        """Assistant messages preserve structured tool calls."""
        from langchain_core.messages import AIMessage

        from litagent.agent.react import _client_to_runnable

        client = _CaptureFormattedClient("ok")
        runnable = _client_to_runnable(client, tools=None)

        state = {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_123",
                            "name": "recall_memory",
                            "args": {"query": "attention"},
                        }
                    ],
                )
            ]
        }
        await runnable.ainvoke(state)
        formatted = client.last_formatted
        assistant = next(m for m in formatted if m["role"] == "assistant")
        assert "tool_calls" in assistant
        tc = assistant["tool_calls"][0]
        assert tc["id"] == "call_123"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "recall_memory"
        assert "attention" in tc["function"]["arguments"]

        assert isinstance(tc["function"]["arguments"], str)
        import json as _json

        assert _json.loads(tc["function"]["arguments"]) == {"query": "attention"}

        assert assistant["content"] is None

    @pytest.mark.asyncio
    async def test_aimessage_no_tool_calls_still_works(self):
        """Plain assistant messages remain valid without tool calls."""
        from langchain_core.messages import AIMessage

        from litagent.agent.react import _client_to_runnable

        client = _CaptureFormattedClient("done")
        runnable = _client_to_runnable(client, tools=None)

        state = {"messages": [AIMessage(content="Here is the answer.")]}
        result = await runnable.ainvoke(state)
        formatted = client.last_formatted
        assistant = next(m for m in formatted if m["role"] == "assistant")
        assert "tool_calls" not in assistant
        assert assistant["content"] == "Here is the answer."
        assert result["final_answer"] == "done"


class _CaptureFormattedClient(BaseLLMClient):
    """LLM client that records its formatted input messages."""

    def __init__(self, response_content: str = "ok"):
        self.response_content = response_content
        self.last_formatted: list[dict] = []

    async def chat(self, messages: list[dict], **kwargs):
        self.last_formatted = messages
        from litagent.llm.client import LLMResponse

        return LLMResponse(content=self.response_content, usage={})
