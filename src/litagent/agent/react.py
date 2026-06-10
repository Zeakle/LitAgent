import json
from typing import Literal

from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from litagent.agent.state import AgentState
from litagent.agent.validation import validate_tool_call
from litagent.config import AgentConfig, AppConfig
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

    # _route_after_agent 需要闭包捕获 config.max_loops，写在 build 函数内部
    def _route_after_agent(state: AgentState) -> Literal['validate', 'end']:
        if state.get('loop_count', 0) >= config.max_loops:
            logger.warning(f'Max loops ({config.max_loops}) exceeded, forcing termination')
            state['final_answer'] = (
                "Agent stopped: maximum loop count exceeded. "
                "Partial results available in previous messages."
            )
            return 'end'

        # 有工具调用
        if state.get("current_action") is not None:
            return 'validate'
        
        return 'end'


    # 节点注册
    workflow.add_node("agent", agent_node)
    workflow.add_node('validate', _make_validate_node(config))
    workflow.add_node('tools', ToolNode(tools))

    workflow.set_entry_point('agent')
    
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

    workflow.add_edge('tools', 'agent')

    return workflow


def _route_after_validate(state: AgentState) -> Literal['tools', 'agent']:
    """校验节点后的路由。

    tool_call 校验通过 → 进 tools 执行
    校验失败 + 修正计数未超限 → 回 agent 让 LLM 修正
    校验失败 + 修正计数超限 → 回 agent 但下次会因 loop_count 超限而终止
    """
    validation_result = state.get('_validation_result', {})
    if validation_result.get('valid', True):
        return 'tools'
    else:
        validation_result.setdefault('retry_count', 0)
        validation_result['retry_count'] += 1

        if validation_result['retry_count'] < 3:
            return 'agent'
        # 修正次数超限，进tools，(让ToolNode报错)
        return 'tools'


def _make_validate_node(config: AgentConfig):
    """创建校验节点 (闭包封装config)"""

    def validate_node(state: AgentState) -> dict:
        action = state.get("current_action")
        if action is None:
            return {"_validation_result": {"valid": True}}
        return validate_tool_call(action)
    
    return validate_node