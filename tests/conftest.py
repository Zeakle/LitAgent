"""Shared helpers for mock agent nodes and tools."""

from langchain_core.messages import AIMessage


def make_mock_agent_node(responses: list[dict], final_text: str = "Task completed."):
    """Return an agent node that replays deterministic responses."""
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
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": action["name"],
                                "args": action.get("args", {}),
                                "id": f"mock_{idx}",
                            }
                        ],
                    )
                ]
        else:
            update["current_action"] = None
            update["messages"] = [AIMessage(content=final_text)]
            update["final_answer"] = final_text

        return update

    return mock_node


def make_mock_tool(name: str, result: str):
    """Return a named tool that always produces the given result."""

    def tool_func(**kwargs) -> str:
        return result

    tool_func.__name__ = name
    tool_func.__doc__ = f"Mock tool: {name}"
    return tool_func
