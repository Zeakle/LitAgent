"""Phase 9 MCP module tests."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from contextlib import AsyncExitStack

from litagent.mcp.connection import MCPConnection
from litagent.mcp.bridge import MCPBridge
from litagent.config import MCPServerConfig
from litagent.exceptions import MCPError


# ── MCPConnection ──

class TestMCPConnectionConstruction:
    def test_stdio_factory_sets_transport(self):
        conn = MCPConnection.stdio("python", ["test.py"])
        assert conn._transport == "stdio"
        assert conn._kwargs["command"] == "python"

    def test_stdio_factory_default_args_env(self):
        conn = MCPConnection.stdio("echo")
        assert conn._kwargs["args"] == []
        assert conn._kwargs["env"] == {}

    def test_streamable_http_factory_sets_transport(self):
        conn = MCPConnection.streamable_http("http://localhost:8080/mcp")
        assert conn._transport == "streamable-http"
        assert conn._kwargs["url"] == "http://localhost:8080/mcp"

    def test_streamable_http_factory_default_headers(self):
        conn = MCPConnection.streamable_http("http://localhost:8080/mcp")
        assert conn._kwargs["headers"] == {}

    def test_unknown_transport_raises_on_enter(self):
        """ValueError is raised in __aenter__, not __init__."""
        conn = MCPConnection("invalid_transport", some_arg="x")
        # __init__ succeeds (stores transport string), __aenter__ validates
        assert conn._transport == "invalid_transport"


class TestMCPConnectionExitStack:
    def test_has_exit_stack(self):
        conn = MCPConnection.stdio("echo")
        assert isinstance(conn._exit_stack, AsyncExitStack)


# ── MCPBridge ──

class TestMCPBridgeEmpty:
    @pytest.mark.asyncio
    async def test_connect_all_empty_servers(self):
        bridge = MCPBridge()
        result = await bridge.connect_all({})
        assert result == []


class TestMCPBridgeDisabledServers:
    @pytest.mark.asyncio
    async def test_connect_all_skips_disabled(self):
        bridge = MCPBridge()
        cfg = MCPServerConfig(enabled=False, command="echo", args=[])
        servers = {"disabled_server": cfg}
        result = await bridge.connect_all(servers)
        assert result == []


class TestMCPBridgeDisconnect:
    @pytest.mark.asyncio
    async def test_disconnect_all_on_empty(self):
        bridge = MCPBridge()
        # should not raise
        await bridge.disconnect_all()


# ── MCPError ──

class TestMCPError:
    def test_is_litagent_error(self):
        from litagent.exceptions import LitAgentError
        assert issubclass(MCPError, LitAgentError)

    def test_error_message(self):
        e = MCPError("test error")
        assert str(e) == "test error"


# ── MCPServerConfig ──

class TestMCPServerConfig:
    def test_defaults(self):
        cfg = MCPServerConfig()
        assert cfg.transport == "stdio"
        assert cfg.enabled is True
        assert cfg.args == []
        assert cfg.command is None

    def test_disabled_server(self):
        cfg = MCPServerConfig(enabled=False)
        assert cfg.enabled is False

    def test_http_config(self):
        cfg = MCPServerConfig(
            transport="streamable-http",
            url="http://localhost:8080/mcp",
            headers={"Authorization": "Bearer xyz"},
        )
        assert cfg.transport == "streamable-http"
        assert cfg.url == "http://localhost:8080/mcp"
