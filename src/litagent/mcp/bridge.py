"""Connect MCP servers and register their advertised tools locally."""

import json

from mcp import ClientSession

from litagent.config import MCPServerConfig
from litagent.mcp.connection import MCPConnection
from litagent.tools.registry import get_registry
from litagent.tools.base import ToolDefinition, ToolCategory
from litagent.exceptions import MCPError
from litagent.logging import get_logger

logger = get_logger("mcp.bridge")


class MCPBridge:
    """Manage MCP sessions and expose remote tools through the registry."""

    def __init__(self):

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

        return self._register_mcp_tools(session, tools, name)

    def _register_mcp_tools(
        self, session: ClientSession, tools: list, server_name: str
    ) -> list[str]:
        """Wrap advertised MCP tools as local tool definitions."""
        registry = get_registry()
        names = []

        for tool in tools:
            mcp_name = tool.name
            reg_name = f"mcp_{mcp_name}"

            # Bind loop values as defaults so each wrapper targets its own tool.
            async def _call_mcp(
                _session: ClientSession = session, _mcp_name: str = mcp_name, **kwargs
            ) -> str:

                result = await _session.call_tool(_mcp_name, arguments=kwargs)

                # Preserve text blocks; stringify non-text-only MCP results.
                texts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        texts.append(block.text)

                return "\n".join(texts) if texts else str(result)

            definition = ToolDefinition(
                name=reg_name,
                description=f"[MCP:{server_name}] {tool.description or mcp_name}",
                parameters=(
                    tool.inputSchema
                    if tool.inputSchema
                    else {"type": "object", "properties": {}}
                ),
                category=ToolCategory.READ,
                timeout_ms=30000,
                max_retries=1,
            )

            registry.register(definition, _call_mcp)
            names.append(reg_name)
            logger.debug(f"  Registered: {reg_name} ({mcp_name})")

        return names

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
