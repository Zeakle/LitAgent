from typing import Literal, AsyncIterator
import json

from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from litagent.agent.state import AgentState
from litagent.agent.validation import validate_tool_call
from litagent.config import AgentConfig
from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger

logger = get_logger('agent.react')


def build_react_graph(
    agent_node: Runnable,
    tools: list,
    config: AgentConfig
) -> StateGraph:
    """构建 ReAct Loop StateGraph。

    这是所有 Worker 共享的执行引擎。每个 Worker 调用此函数时
    传入自己的 agent_node（含 System Prompt + Model）和 tools。

    Graph 结构:
        agent_node → validate → tools → agent_node (循环)
                  ↘ END (final_answer)

    Args:
        agent_node: LLM 调用节点，输入 state 返回 state
        tools: 本 Worker 可用的工具列表
        config: Agent 运行时配置

    Returns:
        未编译的 StateGraph（binder 在 Worker 层做，这里只定义图结构）
    """
    workflow = StateGraph(AgentState)

    # 兼容 LangChain BaseTool (.name) 和普通函数 (.__name__)
    graph_tool_names: set[str] = set()
    for t in (tools or []):
        name = getattr(t, 'name', None) or getattr(t, '__name__', None)
        if name:
            graph_tool_names.add(name)

    # _route_after_agent 闭包捕获 config.max_loops，只读 state，返回方向
    def _route_after_agent(state: AgentState) -> Literal['validate', 'end']:
        if state.get('final_answer') is not None:
            return 'end'

        if state.get('loop_count', 0) >= config.max_loops:
            logger.warning(f'Max loops ({config.max_loops}) exceeded, forcing termination')
            return 'end'

        if state.get("current_action") is not None:
            return 'validate'

        return 'end'


    # step 节点——递增 loop_count（引擎负责，不依赖 Worker 层）
    def _step_node(state: AgentState) -> dict:
        result = {'loop_count': state.get('loop_count', 0) + 1}

        # Dead loop Detection
        action = state.get('current_action')
        history: list[str] = state.get('_last_tool_calls', [])
        if action is not None:
            history.append(json.dumps(action, sort_keys=True))
            history = history[-3:]
            result['_last_tool_calls'] = history
            if len(history) >= 3 and len(set(history)) == 1:
                result['final_answer'] = 'Dead loop detected: same tool call 3 times'
                logger.warning('Dead Loop Detected')
        else:
            result['_last_tool_calls'] = history
        
        return result


    # 节点注册
    workflow.add_node("step", _step_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node('validate', _make_validate_node(graph_tool_names))
    workflow.add_node('tools', ToolNode(tools))
    workflow.set_entry_point('step')
    workflow.add_edge('step', 'agent')
    
    workflow.add_conditional_edges(
        'agent',
        _route_after_agent,
        {
            'validate': 'validate',
            'end': END
        }
    )

    workflow.add_conditional_edges(
        'validate',
        _route_after_validate,
        {
            'tools': 'tools',
            'agent': 'agent'
        }
    )

    workflow.add_edge('tools', 'step')  # 回到 step（不是 agent），每轮递增 loop_count

    return workflow


# 模块级纯函数：只读 state，不做修改
def _route_after_validate(state: AgentState) -> Literal['tools', 'agent']:
    """校验节点后的路由（只读 state，不修改）。"""
    vr = state.get('_validation_result', {})
    if vr.get('valid', True):
        return 'tools'
    if vr.get('retry_count', 0) < 3:
        return 'agent'
    # 修正次数超限，进 tools（让 ToolNode 报错，Observation 包含错误信息）
    return 'tools'


def _make_validate_node(graph_tool_names: set[str] | None = None):
    """创建校验节点（retry_count 在节点内递增，不在路由函数里改 state）。"""

    def validate_node(state: AgentState) -> dict:
        action = state.get("current_action")
        if action is None:
            return {"_validation_result": {"valid": True}}

        result = validate_tool_call(action, graph_tool_names=graph_tool_names)
        vr = result["_validation_result"]

        # 校验失败：继承上次的 retry_count 并 +1（在节点里做，不在路由函数里）
        if not vr.get("valid", True):
            prev = state.get("_validation_result", {})
            vr["retry_count"] = prev.get("retry_count", 0) + 1

        return {"_validation_result": vr}

    return validate_node


async def astream_tokens(graph: StateGraph, input_state: dict) -> AsyncIterator[str]:
    compiled = graph.compile()
    async for event in compiled.stream_events(input_state, version='v2'):
        if event['event'] == 'on_chat_model_stream':
            chunk = event['data']['chunk']
            if hasattr(chunk, 'content') and chunk.content:
                yield chunk.content


def _client_to_runnable(llm_client: BaseLLMClient, tools: list | None = None):
    """将 BaseLLMClient 包装为 LangChain Runnable。
    
    tools 参数: LangChain BaseTool 列表，转换为 OpenAI API tools 格式。
               如果不传 tools，LLM 不会返回 tool_calls。
    """
    tools_spec = _tools_to_api_format(tools) if tools else None

    async def _call(state: dict) -> dict:
        messages = state.get('messages', [])
        formatted = []
        for m in messages:
            if hasattr(m, 'content'):
                msg = {'role': _lc_role(m), 'content': m.content}

                # ToolMessage: 必须带tool_call_id
                if hasattr(m, 'tool_call_id') and m.tool_call_id:
                    msg['tool_call_id'] = m.tool_call_id

                # AIMessage: 保留tool_calls，从 LangChain 格式 {id, name, args}
                # 转回 OpenAI API 格式 {id, type, function: {name, arguments: json_str}}
                if hasattr(m, 'tool_calls') and m.tool_calls:
                    api_tool_calls = []
                    for tc in m.tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get('id', '')
                            tc_name = tc.get('name', '')
                            tc_args = tc.get('args', {})
                        else:
                            tc_id = getattr(tc, 'id', '')
                            tc_name = getattr(tc, 'name', '')
                            tc_args = getattr(tc, 'args', {})
                        api_tool_calls.append({
                            'id': tc_id,
                            'type': 'function',
                            'function': {
                                'name': tc_name,
                                'arguments': json.dumps(tc_args) if isinstance(tc_args, dict) else str(tc_args),
                            },
                        })
                    msg['tool_calls'] = api_tool_calls
                    # OpenAI 要求 assistant turn 带 tool_calls 时 content 为 null 而非 ""
                    if not msg['content']:
                        msg['content'] = None
                    # deepseek thinking 模式：带 tool_calls 的历史 assistant 重发时，
                    # 必须原样带回 reasoning_content，否则第 2 轮起 400
                    rc = m.additional_kwargs.get('reasoning_content', '') if hasattr(m, 'additional_kwargs') else ''
                    if rc:
                        msg['reasoning_content'] = rc

                formatted.append(msg)
            elif isinstance(m, dict):
                formatted.append(m)

        resp = await llm_client.chat(formatted, tools=tools_spec)

        # reasoning_content 存进 additional_kwargs（LangChain 非标字段口袋），
        # 供下一轮回传给 deepseek（thinking 模式要求带 tool_calls 的历史 assistant 原样带回）
        ai_kwargs = {}
        if getattr(resp, 'reasoning_content', ''):
            ai_kwargs['reasoning_content'] = resp.reasoning_content
        ai_msg = AIMessage(content=resp.content or "", additional_kwargs=ai_kwargs)
        result: dict = {'messages': [ai_msg]}

        if resp.tool_calls:
            tc = resp.tool_calls[0]
            tc_name = tc['function']['name']
            tc_args = _parse_tool_args(tc['function']['arguments'])

            # ToolNode读 AIMessages.tool_calls 的值，需要转换为 LangChain 的格式
            ai_msg.tool_calls = [{
                'id': tc.get('id', ''),
                'name': tc_name,
                'args': tc_args,
            }]

            # 将 tool_calls 转换为 current_action, ReAct Loop 需要这个字段
            result['current_action'] = {
                'name': tc_name,
                'args': tc_args,
            }
        else:
            # 没有 tool_call：本轮就是最终答案
            result['final_answer'] = resp.content or ''

        return result

    return RunnableLambda(_call)


def _tools_to_api_format(tools: list) -> list[dict]:
    """LangChain BaseTool -> OpenAI API tools格式"""
    spec = []
    for t in tools:
        spec.append({
            'type': 'function',
            'function': {
                'name': t.name,
                'description': t.description,
                'parameters': t.args_schema.schema() if t.args_schema else {},
            },
        })
    return spec


def _lc_role(msg) -> str:
    """LangChain Message -> OpenAI role message"""
    type_name = getattr(msg, 'type', 'unknown')
    role_map = {
        'human': 'user',
        'ai': 'assistant',
        'system': 'system',
        'tool': 'tool'
    }
    return role_map.get(type_name, 'user')


def _parse_tool_args(args) -> dict:
    """解析 tool argument, API可能返回JSON字符串或dict"""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        import json as _json
        try:
            return _json.loads(args)
        except _json.JSONDecodeError:
            return {'raw': args}
    return {'raw': str(args)}


class ReActRunner:
    """ReAct Loop 封装

    工作流程:
        1. 将 LLM client 包装为 Runnable
        2. build_react_graph(agent_node, tools, config)
        3. compile → ainvoke（或 astream_tokens）
        4. 返回
    """

    def __init__(self, llm_client: BaseLLMClient, tools: list | None = None, config: AgentConfig | None = None):
        self._llm = llm_client
        self._tools = tools or []
        self._config = config or AgentConfig()


    async def run(self, system_prompt: str, user_message: str) -> str:
        """运行ReAct循环, 返回final_answer"""
        agent_node = _client_to_runnable(self._llm, tools=self._tools)

        graph = build_react_graph(agent_node, self._tools, self._config)

        compiled = graph.compile()
        initial_state = {
            'messages': [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
                ]
        }

        result = await compiled.ainvoke(initial_state)

        return result.get('final_answer', '')


    async def run_streaming(
        self,
        system_prompt: str,
        user_message: str,
    ):
        """Streaming ReAct"""
        agent_node = _client_to_runnable(self._llm, tools=self._tools)
        graph = build_react_graph(agent_node, self._tools, self._config)

        initial_state = {
            'messages': [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
                ]
        }

        async for token in astream_tokens(graph, initial_state):
            yield token