"""Define the shared state contract for the ReAct graph."""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from litagent.tools.base import ToolDefinition


class AgentState(TypedDict):
    """Define values carried between ReAct graph nodes."""

    messages: Annotated[list, add_messages]

    tools: list[ToolDefinition]

    current_thought: str

    current_action: dict | None

    loop_count: int

    final_answer: str | None

    # Shape: {"valid": bool, "errors": [...], "retry_count": int}.
    _validation_result: dict

    _last_tool_calls: list[str]
