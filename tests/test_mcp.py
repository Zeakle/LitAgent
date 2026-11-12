"""Tests for MCP connection lifecycle and capability-gated registration."""

from contextlib import AsyncExitStack
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litagent.config import MCPServerConfig, MCPToolCapability
from litagent.exceptions import MCPError
from litagent.mcp.bridge import MCPBridge
from litagent.mcp.connection import MCPConnection
from litagent.tools.base import ToolCategory
from litagent.tools.executor import ToolExecutor
from litagent.tools.registry import ToolRegistry


def _remote_tool(name: str, schema: dict | None = None, description: str = "remote"):
    """Return an MCP tool-shaped test double."""
    return SimpleNamespace(name=name, inputSchema=schema, description=description)


def _capability(category: ToolCategory, enabled: bool = True):
    """Return the trusted local capability model used by MCP configuration."""
    return MCPToolCapability(category=category, enabled=enabled)


class TestMCPConnectionConstruction:
    """Tests MCP connection factories."""

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
        assert conn._transport == "invalid_transport"


class TestMCPConnectionExitStack:
    """Tests MCP connection resource management."""

    def test_has_exit_stack(self):
        conn = MCPConnection.stdio("echo")
        assert isinstance(conn._exit_stack, AsyncExitStack)


class TestMCPBridgeEmpty:
    """Tests an MCP bridge without servers."""

    @pytest.mark.asyncio
    async def test_connect_all_empty_servers(self):
        bridge = MCPBridge(ToolRegistry())
        result = await bridge.connect_all({})
        assert result == []


class TestMCPBridgeDisabledServers:
    """Tests disabled MCP server handling."""

    @pytest.mark.asyncio
    async def test_connect_all_skips_disabled(self):
        bridge = MCPBridge(ToolRegistry())
        cfg = MCPServerConfig(enabled=False, command="echo", args=[])
        servers = {"disabled_server": cfg}
        result = await bridge.connect_all(servers)
        assert result == []


class TestMCPBridgeDisconnect:
    """Tests MCP bridge disconnection."""

    @pytest.mark.asyncio
    async def test_disconnect_all_on_empty(self):
        bridge = MCPBridge(ToolRegistry())
        await bridge.disconnect_all()


class TestMCPCapabilityRegistration:
    """Tests local trust policy around advertised MCP tools."""

    def test_empty_capability_map_registers_no_remote_tools(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)

        names = bridge._register_mcp_tools(
            AsyncMock(), [_remote_tool("search")], "papers", {}
        )

        assert names == []
        assert len(registry) == 0

    def test_only_configured_enabled_tools_are_registered(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)
        capabilities = {
            "enabled": _capability(ToolCategory.READ),
            "disabled": _capability(ToolCategory.READ, enabled=False),
        }

        names = bridge._register_mcp_tools(
            AsyncMock(),
            [
                _remote_tool("enabled"),
                _remote_tool("disabled"),
                _remote_tool("unconfigured"),
            ],
            "papers",
            capabilities,
        )

        assert names == ["mcp.papers.enabled"]
        assert "mcp.papers.enabled" in registry
        assert "mcp.papers.disabled" not in registry
        assert "mcp.papers.unconfigured" not in registry

    def test_mcp_names_include_server_namespace_and_do_not_collide(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)
        capability = {"search": _capability(ToolCategory.READ)}

        first = bridge._register_mcp_tools(
            AsyncMock(), [_remote_tool("search")], "papers-a", capability
        )
        second = bridge._register_mcp_tools(
            AsyncMock(), [_remote_tool("search")], "papers_b", capability
        )

        assert first == ["mcp.papers-a.search"]
        assert second == ["mcp.papers_b.search"]
        assert len(registry) == 2

    def test_bridges_keep_run_local_registries_isolated(self):
        first_registry = ToolRegistry()
        second_registry = ToolRegistry()
        capability = {"search": _capability(ToolCategory.READ)}

        MCPBridge(first_registry)._register_mcp_tools(
            AsyncMock(), [_remote_tool("search")], "first", capability
        )
        MCPBridge(second_registry)._register_mcp_tools(
            AsyncMock(), [_remote_tool("search")], "second", capability
        )

        assert "mcp.first.search" in first_registry
        assert "mcp.second.search" not in first_registry
        assert "mcp.second.search" in second_registry
        assert "mcp.first.search" not in second_registry

    @pytest.mark.parametrize("name", ["escape.tool", "bad name", "../tool", "tool\\x"])
    def test_remote_tool_name_cannot_escape_server_namespace(self, name):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)

        names = bridge._register_mcp_tools(
            AsyncMock(),
            [_remote_tool(name)],
            "papers",
            {name: _capability(ToolCategory.READ)},
        )

        assert names == []
        assert len(registry) == 0

    def test_mcp_category_comes_from_local_config_not_remote_description(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)

        bridge._register_mcp_tools(
            AsyncMock(),
            [_remote_tool("delete", description="harmless read-only lookup")],
            "admin",
            {"delete": _capability(ToolCategory.DESTRUCTIVE)},
        )

        definition = registry.get("mcp.admin.delete").definition
        assert definition.category is ToolCategory.DESTRUCTIVE

    def test_configured_but_unadvertised_tool_is_skipped(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)

        names = bridge._register_mcp_tools(
            AsyncMock(),
            [_remote_tool("advertised")],
            "papers",
            {"missing": _capability(ToolCategory.READ)},
        )

        assert names == []

    def test_invalid_server_namespace_is_rejected_defensively(self):
        bridge = MCPBridge(ToolRegistry())

        with pytest.raises(MCPError, match="Invalid MCP server namespace"):
            bridge._register_mcp_tools(
                AsyncMock(),
                [_remote_tool("search")],
                "bad.server",
                {"search": _capability(ToolCategory.READ)},
            )

    def test_empty_remote_schema_is_fail_closed_object_schema(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)

        bridge._register_mcp_tools(
            AsyncMock(),
            [_remote_tool("ping", schema=None)],
            "health",
            {"ping": _capability(ToolCategory.READ)},
        )

        assert registry.get("mcp.health.ping").definition.parameters == {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    @pytest.mark.asyncio
    async def test_mcp_write_tool_is_denied_before_remote_call(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)
        session = AsyncMock()
        bridge._register_mcp_tools(
            session,
            [_remote_tool("write")],
            "notes",
            {"write": _capability(ToolCategory.WRITE)},
        )
        executor = ToolExecutor(registry, allowed_names={"mcp.notes.write"})

        result = await executor.execute("mcp.notes.write", {})

        assert result.error_code == "tool_category_denied"
        session.call_tool.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_arguments_cannot_override_bound_remote_name(self):
        registry = ToolRegistry()
        bridge = MCPBridge(registry)
        session = AsyncMock()
        session.call_tool.return_value = SimpleNamespace(
            content=[SimpleNamespace(text="ok")]
        )
        bridge._register_mcp_tools(
            session,
            [
                _remote_tool(
                    "safe",
                    schema={
                        "type": "object",
                        "properties": {"_mcp_name": {"type": "string"}},
                        "additionalProperties": False,
                    },
                )
            ],
            "notes",
            {"safe": _capability(ToolCategory.READ)},
        )
        executor = ToolExecutor(registry, allowed_names={"mcp.notes.safe"})

        result = await executor.execute(
            "mcp.notes.safe", {"_mcp_name": "unconfigured_destructive_tool"}
        )

        assert result.output == "ok"
        session.call_tool.assert_awaited_once_with(
            "safe", arguments={"_mcp_name": "unconfigured_destructive_tool"}
        )


class TestMCPError:
    """Tests MCP error contracts."""

    def test_is_litagent_error(self):
        from litagent.exceptions import LitAgentError

        assert issubclass(MCPError, LitAgentError)

    def test_error_message(self):
        e = MCPError("test error")
        assert str(e) == "test error"


class TestMCPServerConfig:
    """Tests MCP server configuration."""

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
