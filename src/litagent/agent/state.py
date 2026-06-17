from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from litagent.tools.base import ToolDefinition


class AgentState(TypedDict):
    """ReAct Loop 的共享状态。

    每个 Worker 内部跑 ReAct Loop 时共用这个 State 结构。
    区别在于 System Prompt、加载的 Tools、注入的 Context。
    """
    # 全量对话历史。用 add_messages 做 reducer，新消息自动追加不覆盖
    messages: Annotated[list, add_messages]

    # 当前 Worker 可用的工具列表。TODO: Phase 3 换成 ToolDefinition
    tools: list[ToolDefinition]

    # 当前轮的思考过程（仅本轮消费，不进下一轮 observation）
    current_thought: str

    # 当前轮 LLM 返回的 tool_call
    current_action: dict | None

    # ReAct 循环计数，用于 max_loops 终止和死循环检测
    loop_count: int

    # 输出。有值意味着循环终止
    final_answer: str | None

    # 内部：校验节点与路由函数之间的通信通道（不暴露给 Worker 层）
    _validation_result: dict  # {"valid": bool, "errors": [...], "retry_count": int}