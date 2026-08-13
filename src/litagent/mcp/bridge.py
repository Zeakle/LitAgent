"""Connect MCP servers and register their advertised tools locally."""

import re
from collections.abc import Mapping
from typing import Any

from mcp import ClientSession

from litagent.config import MCPServerConfig
from litagent.exceptions import MCPError
from litagent.logging import get_logger
from litagent.mcp.connection import MCPConnection
from litagent.tools.base import ToolCategory, ToolDefinition
from litagent.tools.registry import ToolRegistry, get_registry

logger = get_logger("mcp.bridge")

_MCP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")
_EMPTY_OBJECT_SCHEMA = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


class MCPBridge:
    """Manage MCP sessions and expose remote tools through the registry."""

    def __init__(self, registry: ToolRegistry | None = None):
        """Initialize the bridge with a run-local registry when provided."""
        self._registry = registry if registry is not None else get_registry()
        self._sessions: dict[str, ClientSession] = {}
        self._connections: dict[str, MCPConnection] = {}

    async def connect_all(self, servers: dict[str, MCPServerConfig]) -> list[str]:
        """Connect enabled servers independently and return registered names."""
        registered = []

        for server_name, cfg in servers.items():
            if hasattr(cfg, "enabled") and not cfg.enabled:
                logger.debug(f"MCP {server_name} disabled, skipping")
                continue

            try:
                names = await self._connect_one(server_name, cfg)
                registered.extend(names)
                logger.info(f"MCP {server_name}: {len(names)} tools registered")
            except Exception as e:
                # One unavailable server must not block other MCP integrations.
                logger.warning(f"MCP '{server_name}' failed: {e}")

        return registered

    async def _connect_one(self, name: str, cfg: MCPServerConfig) -> list[str]:
        """Open one MCP session and register its advertised tools."""
        if not _MCP_NAME_PATTERN.fullmatch(name):
            raise MCPError(f"Invalid MCP server namespace: '{name}'")

        transport_type = getattr(cfg, "transport", "stdio")

        if transport_type == "stdio":
            conn = MCPConnection.stdio(
                command=cfg.command,
                args=getattr(cfg, "args", []),
                env=getattr(cfg, "env", None),
                sandboxed=getattr(cfg, "sandboxed", False),
                sandbox_network=getattr(cfg, "sandbox_network", "none"),
            )
        elif transport_type in ("streamable-http", "http"):
            conn = MCPConnection.streamable_http(
                url=cfg.url, headers=getattr(cfg, "headers", None)
            )
        else:
            raise MCPError(f"Unknown transport: '{transport_type}' for server '{name}'")

        session = await conn.__aenter__()
        self._connections[name] = conn
        self._sessions[name] = session

        response = await session.list_tools()
        tools = response.tools
        logger.debug(f"MCP '{name}': found {len(tools)} tools")

        capabilities = getattr(cfg, "allowed_tools", {}) or {}
        advertised_names = {
            tool.name for tool in tools if isinstance(getattr(tool, "name", None), str)
        }
        for tool_name, capability in capabilities.items():
            if self._capability_field(capability, "enabled", True):
                if tool_name not in advertised_names:
                    logger.warning(
                        "MCP '%s': configured tool '%s' was not advertised",
                        name,
                        tool_name,
                    )

        return self._register_mcp_tools(session, tools, name, capabilities)

    def _register_mcp_tools(
        self,
        session: ClientSession,
        tools: list[Any],
        server_name: str,
        capabilities: Mapping[str, Any],
    ) -> list[str]:
        """Register the trusted intersection of capabilities and advertised tools."""
        if not _MCP_NAME_PATTERN.fullmatch(server_name):
            raise MCPError(f"Invalid MCP server namespace: '{server_name}'")

        names: list[str] = []

        for tool in tools:
            remote_name = getattr(tool, "name", None)
            if not isinstance(remote_name, str) or not _MCP_NAME_PATTERN.fullmatch(
                remote_name
            ):
                logger.warning(
                    "MCP '%s': skipping invalid remote tool name (%s)",
                    server_name,
                    "invalid_remote_tool_name",
                )
                continue

            capability = capabilities.get(remote_name)
            if capability is None or not self._capability_field(
                capability, "enabled", True
            ):
                logger.debug(
                    "MCP '%s': tool '%s' is not enabled by local capability policy",
                    server_name,
                    remote_name,
                )
                continue

            category_value = self._capability_field(capability, "category", None)
            try:
                category = (
                    category_value
                    if isinstance(category_value, ToolCategory)
                    else ToolCategory(category_value)
                )
            except (TypeError, ValueError):
                logger.warning(
                    "MCP '%s': tool '%s' has an invalid local category",
                    server_name,
                    remote_name,
                )
                continue

            reg_name = f"mcp.{server_name}.{remote_name}"

            # A closure factory prevents tool arguments from overriding the bound
            # session or remote name through specially crafted schema properties.
            def _make_call(bound_name: str):
                async def _call_mcp(**kwargs) -> str:
                    """Call one bound MCP tool and normalize its content blocks."""

                    result = await session.call_tool(bound_name, arguments=kwargs)

                    texts = [
                        block.text for block in result.content if hasattr(block, "text")
                    ]
                    return "\n".join(texts) if texts else str(result)

                return _call_mcp

            schema = getattr(tool, "inputSchema", None)
            definition = ToolDefinition(
                name=reg_name,
                description=(
                    f"[MCP:{server_name}] "
                    f"{getattr(tool, 'description', None) or remote_name}"
                ),
                parameters=schema if schema else dict(_EMPTY_OBJECT_SCHEMA),
                category=category,
                timeout_ms=30000,
                max_retries=1,
            )

            self._registry.register(definition, _make_call(remote_name))
            names.append(reg_name)
            logger.debug(f"  Registered: {reg_name} ({remote_name})")

        return names

    @staticmethod
    def _capability_field(capability: Any, field: str, default: Any) -> Any:
        """Read capability fields from Pydantic models or mapping test doubles."""
        if isinstance(capability, Mapping):
            return capability.get(field, default)
        return getattr(capability, field, default)

    async def disconnect_all(self) -> None:
        """Close all connections without stopping on individual failures."""
        for name in list(self._connections.keys()):
            try:
                await self._connections[name].__aexit__(None, None, None)
                logger.debug(f"Disconnected MCP '{name}'")
            except Exception as e:
                logger.warning(f"Error disconnecting MCP '{name}' : {e}")

        self._connections.clear()
        self._sessions.clear()
