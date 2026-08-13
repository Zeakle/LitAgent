"""Shared helpers for mock agent nodes and tools."""

import os
import socket

import pytest
from langchain_core.messages import AIMessage


def pytest_configure(config: pytest.Config) -> None:
    """Keep the default test process deterministic and offline."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Leave marker selection explicit instead of guessing from file names."""
    del config, items


@pytest.fixture(autouse=True)
def _forbid_unmarked_network(request: pytest.FixtureRequest, monkeypatch):
    """Fail tests that cross the network without integration/live markers."""
    if request.node.get_closest_marker(
        "integration"
    ) or request.node.get_closest_marker("live"):
        yield
        return

    attempts: list[str] = []
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def is_asyncio_wakeup(address) -> bool:
        """Allow only Windows asyncio's private high-port loopback socketpair."""
        return (
            isinstance(address, tuple)
            and len(address) >= 2
            and address[0] in {"127.0.0.1", "::1"}
            and isinstance(address[1], int)
            and address[1] >= 49152
        )

    def blocked_connect(_socket, address):
        if is_asyncio_wakeup(address):
            return original_connect(_socket, address)
        attempts.append(repr(address))
        raise RuntimeError("network access requires integration or live marker")

    def blocked_connect_ex(_socket, address):
        if is_asyncio_wakeup(address):
            return original_connect_ex(_socket, address)
        attempts.append(repr(address))
        raise RuntimeError("network access requires integration or live marker")

    def blocked_create_connection(address, *args, **kwargs):
        if is_asyncio_wakeup(address):
            return original_create_connection(address, *args, **kwargs)
        attempts.append(repr(address))
        raise RuntimeError("network access requires integration or live marker")

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_connect_ex)
    monkeypatch.setattr(socket, "create_connection", blocked_create_connection)
    yield

    if attempts:
        pytest.fail(f"unmarked test attempted network access: {attempts}")


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
