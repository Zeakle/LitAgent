import pytest
from litagent.agent.state import AgentState
from litagent.agent.react import (
    build_react_graph,
    _route_after_validate,
)
from litagent.agent.validation import validate_tool_call
from litagent.config import AgentConfig
from litagent.llm.client import BaseLLMClient
from tests.conftest import make_mock_agent_node, make_mock_tool


# ── Validation Tests ──

class TestValidateToolCall:
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


# ── Validation Routing Test ──

class TestRouteAfterValidate:
    def test_valid_sends_to_tools(self):
        state = {"_validation_result": {"valid": True}}
        assert _route_after_validate(state) == "tools"

    def test_invalid_sends_to_agent(self):
        state = {"_validation_result": {"valid": False, "errors": ["bad"]}}
        assert _route_after_validate(state) == "agent"

    def test_invalid_twice_then_exceeded(self):
        """retry_count 已达 3，超过修正上限，返回 tools"""
        state = {"_validation_result": {"valid": False, "retry_count": 3}}
        assert _route_after_validate(state) == "tools"


# ── Graph Integration Tests ──

class TestReactGraph:
    def test_build_graph_compiles(self):
        """确保图结构正确可编译"""
        mock = make_mock_agent_node([])
        graph = build_react_graph(mock, [], AgentConfig())
        compiled = graph.compile()
        result = compiled.invoke({
            "messages": [],
            "tools": [],
            "current_thought": "",
            "current_action": None,
            "loop_count": 0,
            "final_answer": None,
        })
        assert result.get("final_answer") == "Task completed."

    def test_max_loops_stops_graph(self):
        """max_loops=2 时图不报错正常终止（即使每轮都有 tool_call）"""
        cfg = AgentConfig(max_loops=2)
        mock = make_mock_agent_node(
            [{"current_action": {"name": "search", "args": {}}}] * 5
        )
        graph = build_react_graph(mock, [make_mock_tool("search", "mock result")], cfg)
        compiled = graph.compile()
        result = compiled.invoke({
            "messages": [],
            "tools": [],
            "current_thought": "",
            "current_action": None,
            "loop_count": 0,
            "final_answer": None,
        })
        # 图正常终止（没有抛异常），loop_count 停在了 max_loops 位置
        assert result.get("loop_count") == 2


# ── Bugfix 11.6: validate_tool_call with graph_tool_names ──

@pytest.fixture(autouse=True)
def _ensure_registry():
    """确保 Registry 非空——否则 validate 不检查 tool 名（len(registry)>0 guard）。"""
    from litagent.tools.registry import get_registry
    registry = get_registry()
    if len(registry) == 0:
        from litagent.tools.builtin.search import register_search_tools
        register_search_tools()


class TestValidateToolCallWithGraphTools:
    def test_worker_tool_accepted(self):
        """Worker tool（如 recall_memory）在 graph_tool_names 中 → 合法"""
        result = validate_tool_call(
            {"name": "recall_memory", "args": {"query": "attention"}},
            graph_tool_names={"recall_memory", "lookup_claims"}
        )
        assert result["_validation_result"]["valid"] is True

    def test_unknown_tool_still_rejected(self):
        """不在 Registry 也不在 graph_tool_names → 拒绝"""
        result = validate_tool_call(
            {"name": "nonexistent_tool", "args": {}},
            graph_tool_names={"recall_memory"}
        )
        assert result["_validation_result"]["valid"] is False
        assert "not registered" in str(result["_validation_result"]["errors"])

    def test_graph_tool_names_none_backward_compat(self):
        """不传 graph_tool_names → 向后兼容（旧行为不变）"""
        result = validate_tool_call(
            {"name": "search_arxiv", "args": {"q": "x"}}
        )
        assert result["_validation_result"]["valid"] is True


# ── Bugfix 11.6: _client_to_runnable message conversion ──

class TestClientToRunnableMessages:
    """验证 _client_to_runnable 的 _call 函数正确转换 LangChain → OpenAI 消息格式"""

    @pytest.mark.asyncio
    async def test_tool_message_includes_tool_call_id(self):
        """ToolMessage.tool_call_id 必须出现在 OpenAI 格式中"""
        from langchain_core.messages import ToolMessage
        from litagent.agent.react import _client_to_runnable
        from litagent.llm.client import BaseLLMClient

        client = _CaptureFormattedClient("ok")
        runnable = _client_to_runnable(client, tools=None)

        state = {
            'messages': [
                ToolMessage(content="search result", tool_call_id="call_abc123")
            ]
        }
        await runnable.ainvoke(state)
        formatted = client.last_formatted
        tool_msg = next(m for m in formatted if m['role'] == 'tool')
        assert tool_msg['tool_call_id'] == 'call_abc123'
        assert tool_msg['content'] == 'search result'

    @pytest.mark.asyncio
    async def test_aimessage_includes_tool_calls(self):
        """AIMessage.tool_calls 在下一轮中保留（转为 OpenAI API 格式）"""
        from langchain_core.messages import AIMessage
        from litagent.agent.react import _client_to_runnable

        client = _CaptureFormattedClient("ok")
        runnable = _client_to_runnable(client, tools=None)

        state = {
            'messages': [
                AIMessage(
                    content="",
                    tool_calls=[{
                        'id': 'call_123',
                        'name': 'recall_memory',
                        'args': {'query': 'attention'}
                    }]
                )
            ]
        }
        await runnable.ainvoke(state)
        formatted = client.last_formatted
        assistant = next(m for m in formatted if m['role'] == 'assistant')
        assert 'tool_calls' in assistant
        tc = assistant['tool_calls'][0]
        assert tc['id'] == 'call_123'
        assert tc['type'] == 'function'
        assert tc['function']['name'] == 'recall_memory'
        assert 'attention' in tc['function']['arguments']
        # arguments 必须是 JSON 字符串（OpenAI API 要求），不是 dict
        assert isinstance(tc['function']['arguments'], str)
        import json as _json
        assert _json.loads(tc['function']['arguments']) == {'query': 'attention'}
        # 带 tool_calls 时 content 应为 None（OpenAI 契约）
        assert assistant['content'] is None

    @pytest.mark.asyncio
    async def test_aimessage_no_tool_calls_still_works(self):
        """AIMessage 没有 tool_calls → 正常返回，无 tool_calls 字段"""
        from langchain_core.messages import AIMessage
        from litagent.agent.react import _client_to_runnable

        client = _CaptureFormattedClient("done")
        runnable = _client_to_runnable(client, tools=None)

        state = {
            'messages': [
                AIMessage(content="Here is the answer.")
            ]
        }
        result = await runnable.ainvoke(state)
        formatted = client.last_formatted
        assistant = next(m for m in formatted if m['role'] == 'assistant')
        assert 'tool_calls' not in assistant
        assert assistant['content'] == 'Here is the answer.'
        assert result['final_answer'] == 'done'


class _CaptureFormattedClient(BaseLLMClient):
    """测试用 client，捕获传给 chat() 的 formatted messages"""

    def __init__(self, response_content: str = "ok"):
        self.response_content = response_content
        self.last_formatted: list[dict] = []

    async def chat(self, messages: list[dict], **kwargs):
        self.last_formatted = messages
        from litagent.llm.client import LLMResponse
        return LLMResponse(content=self.response_content, usage={})
