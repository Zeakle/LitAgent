"""Build and run the LangGraph-based ReAct loop."""

import json
from typing import AsyncIterator, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from litagent.agent.state import AgentState
from litagent.agent.validation import validate_tool_call
from litagent.config import AgentConfig
from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger

logger = get_logger("agent.react")


def build_react_graph(
    agent_node: Runnable, tools: list, config: AgentConfig
) -> StateGraph:
    """Build an uncompiled ReAct state graph for the supplied tools."""
    workflow = StateGraph(AgentState)

    graph_tool_names: set[str] = set()
    for t in tools or []:
        name = getattr(t, "name", None) or getattr(t, "__name__", None)
        if name:
            graph_tool_names.add(name)

    def _route_after_agent(state: AgentState) -> Literal["validate", "end"]:
        """Route an agent step to tool execution or termination."""
        if state.get("final_answer") is not None:
            return "end"

        if state.get("loop_count", 0) >= config.max_loops:
            logger.warning(
                f"Max loops ({config.max_loops}) exceeded, forcing termination"
            )
            return "end"

        if state.get("current_action") is not None:
            return "validate"

        return "end"

    def _step_node(state: AgentState) -> dict:
        """Run one model step and append its message."""
        result = {"loop_count": state.get("loop_count", 0) + 1}

        # Stop after three identical consecutive tool calls.
        action = state.get("current_action")
        history: list[str] = state.get("_last_tool_calls", [])
        if action is not None:
            history.append(json.dumps(action, sort_keys=True))
            history = history[-3:]
            result["_last_tool_calls"] = history
            if len(history) >= 3 and len(set(history)) == 1:
                result["final_answer"] = "Dead loop detected: same tool call 3 times"
                logger.warning("Dead Loop Detected")
        else:
            result["_last_tool_calls"] = history

        return result

    workflow.add_node("step", _step_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("validate", _make_validate_node(graph_tool_names))
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("step")
    workflow.add_edge("step", "agent")

    workflow.add_conditional_edges(
        "agent", _route_after_agent, {"validate": "validate", "end": END}
    )

    workflow.add_conditional_edges(
        "validate", _route_after_validate, {"tools": "tools", "agent": "agent"}
    )

    workflow.add_edge("tools", "step")

    return workflow


def _route_after_validate(state: AgentState) -> Literal["tools", "agent"]:
    """Choose the next node from the stored tool-call validation result."""
    vr = state.get("_validation_result", {})
    if vr.get("valid", True):
        return "tools"
    if vr.get("retry_count", 0) < 3:
        return "agent"

    return "tools"


def _make_validate_node(graph_tool_names: set[str] | None = None):
    """Create a graph node that validates the pending tool call."""

    def validate_node(state: AgentState) -> dict:
        """Validate pending tool calls before tool execution."""
        action = state.get("current_action")
        if action is None:
            return {"_validation_result": {"valid": True}}

        result = validate_tool_call(action, graph_tool_names=graph_tool_names)
        vr = result["_validation_result"]

        if not vr.get("valid", True):
            prev = state.get("_validation_result", {})
            vr["retry_count"] = prev.get("retry_count", 0) + 1

        return {"_validation_result": vr}

    return validate_node


async def astream_tokens(graph: StateGraph, input_state: dict) -> AsyncIterator[str]:
    """Yield chat-model content chunks emitted while running the graph."""
    compiled = graph.compile()
    async for event in compiled.stream_events(input_state, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content


def _client_to_runnable(llm_client: BaseLLMClient, tools: list | None = None):
    """Adapt an LLM client to a LangGraph state-update runnable."""
    tools_spec = _tools_to_api_format(tools) if tools else None

    async def _call(state: dict) -> dict:
        """Forward graph messages to the configured LLM client."""
        messages = state.get("messages", [])
        formatted = []
        for m in messages:
            if hasattr(m, "content"):
                msg = {"role": _lc_role(m), "content": m.content}

                if hasattr(m, "tool_call_id") and m.tool_call_id:
                    msg["tool_call_id"] = m.tool_call_id

                if hasattr(m, "tool_calls") and m.tool_calls:
                    api_tool_calls = []
                    for tc in m.tool_calls:
                        if isinstance(tc, dict):
                            tc_id = tc.get("id", "")
                            tc_name = tc.get("name", "")
                            tc_args = tc.get("args", {})
                        else:
                            tc_id = getattr(tc, "id", "")
                            tc_name = getattr(tc, "name", "")
                            tc_args = getattr(tc, "args", {})
                        api_tool_calls.append(
                            {
                                "id": tc_id,
                                "type": "function",
                                "function": {
                                    "name": tc_name,
                                    "arguments": (
                                        json.dumps(tc_args)
                                        if isinstance(tc_args, dict)
                                        else str(tc_args)
                                    ),
                                },
                            }
                        )
                    msg["tool_calls"] = api_tool_calls

                    if not msg["content"]:
                        msg["content"] = None

                    rc = (
                        m.additional_kwargs.get("reasoning_content", "")
                        if hasattr(m, "additional_kwargs")
                        else ""
                    )
                    if rc:
                        msg["reasoning_content"] = rc

                formatted.append(msg)
            elif isinstance(m, dict):
                formatted.append(m)

        resp = await llm_client.chat(formatted, tools=tools_spec)

        ai_kwargs = {}
        if getattr(resp, "reasoning_content", ""):
            ai_kwargs["reasoning_content"] = resp.reasoning_content
        ai_msg = AIMessage(content=resp.content or "", additional_kwargs=ai_kwargs)
        result: dict = {"messages": [ai_msg]}

        if resp.tool_calls:
            tc = resp.tool_calls[0]
            tc_name = tc["function"]["name"]
            tc_args = _parse_tool_args(tc["function"]["arguments"])

            ai_msg.tool_calls = [
                {
                    "id": tc.get("id", ""),
                    "name": tc_name,
                    "args": tc_args,
                }
            ]

            result["current_action"] = {
                "name": tc_name,
                "args": tc_args,
            }
        else:

            result["final_answer"] = resp.content or ""

        return result

    return RunnableLambda(_call)


def _tools_to_api_format(tools: list) -> list[dict]:
    """Convert LangChain tools to OpenAI-compatible function schemas."""
    spec = []
    for t in tools:
        spec.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.args_schema.schema() if t.args_schema else {},
                },
            }
        )
    return spec


def _lc_role(msg) -> str:
    """Map a LangChain message type to an OpenAI-compatible role."""
    type_name = getattr(msg, "type", "unknown")
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    return role_map.get(type_name, "user")


def _parse_tool_args(args) -> dict:
    """Normalize tool arguments to a dictionary."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        import json as _json

        try:
            return _json.loads(args)
        except _json.JSONDecodeError:
            return {"raw": args}
    return {"raw": str(args)}


class ReActRunner:
    """Run a ReAct loop over a configured LLM and tool set."""

    def __init__(
        self,
        llm_client: BaseLLMClient,
        tools: list | None = None,
        config: AgentConfig | None = None,
    ):
        """Initialize the ReAct runner."""
        self._llm = llm_client
        self._tools = tools or []
        self._config = config or AgentConfig()

    async def run(self, system_prompt: str, user_message: str) -> str:
        """Run the ReAct loop and return its final answer."""
        agent_node = _client_to_runnable(self._llm, tools=self._tools)

        graph = build_react_graph(agent_node, self._tools, self._config)

        compiled = graph.compile()
        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        }

        result = await compiled.ainvoke(
            initial_state,
            config={"recursion_limit": self._config.max_loops * 5 + 5},
        )

        return result.get("final_answer", "")

    async def run_streaming(
        self,
        system_prompt: str,
        user_message: str,
    ):
        """Yield response tokens from the ReAct loop."""
        agent_node = _client_to_runnable(self._llm, tools=self._tools)
        graph = build_react_graph(agent_node, self._tools, self._config)

        initial_state = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message),
            ]
        }

        async for token in astream_tokens(graph, initial_state):
            yield token
