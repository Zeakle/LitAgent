from typing import Literal
import json

from langchain_core.runnables import Runnable
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from litagent.agent.state import AgentState
from litagent.agent.validation import validate_tool_call
from litagent.config import AgentConfig
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
    workflow.add_node('validate', _make_validate_node())
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


def _make_validate_node():
    """创建校验节点（retry_count 在节点内递增，不在路由函数里改 state）。"""

    def validate_node(state: AgentState) -> dict:
        action = state.get("current_action")
        if action is None:
            return {"_validation_result": {"valid": True}}

        result = validate_tool_call(action)
        vr = result["_validation_result"]

        # 校验失败：继承上次的 retry_count 并 +1（在节点里做，不在路由函数里）
        if not vr.get("valid", True):
            prev = state.get("_validation_result", {})
            vr["retry_count"] = prev.get("retry_count", 0) + 1

        return {"_validation_result": vr}

    return validate_node