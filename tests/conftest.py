"""Phase 2 test helpers — mock agent node and tools."""

from langchain_core.messages import AIMessage


def make_mock_agent_node(responses: list[dict], final_text: str = "Task completed."):
    """创建一个 Mock Agent 节点，按顺序返回预设的 tool_call + final text.

    loop_count 由引擎的 step_node 递增，mock 不再管理。
    """
    call_count = [0]

    def mock_node(state: dict) -> dict:
        idx = call_count[0]
        call_count[0] += 1
        update = {}

        if idx < len(responses):
            action = responses[idx].get("current_action")
            update["current_action"] = action
            update["current_thought"] = responses[idx].get("current_thought", "")
            if action:
                update["messages"] = [
                    AIMessage(content="", tool_calls=[{"name": action["name"], "args": action.get("args", {}), "id": f"mock_{idx}"}])
                ]
        else:
            update["current_action"] = None
            update["messages"] = [AIMessage(content=final_text)]
            update["final_answer"] = final_text

        return update

    return mock_node


def make_mock_tool(name: str, result: str):
    """创建一个 Mock Tool，返回固定结果。"""

    def tool_func(**kwargs) -> str:
        return result

    tool_func.__name__ = name
    tool_func.__doc__ = f"Mock tool: {name}"
    return tool_func
