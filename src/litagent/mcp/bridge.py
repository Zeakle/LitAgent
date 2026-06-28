"""MCP → ToolRegistry 桥接层。

职责: 读配置 → 连 MCP server → 拉 tools → 注册到 ToolRegistry。
注册后, MCP tools 和内置 tools 在 ToolExecutor 眼里无差别。
"""

import json
from mcp import ClientSession
from litagent.config import MCPServerConfig
from litagent.mcp.connection import MCPConnection
from litagent.tools.registry import get_registry
from litagent.tools.base import ToolDefinition, ToolCategory
from litagent.exceptions import MCPError
from litagent.logging import get_logger


logger = get_logger('mcp.bridge')


class MCPBridge:
    """MCP 工具桥接器。

    MCPBridge 是 MCP 模块的「总入口」。做三件事:
    1. 读 config.mcp_servers, 知道要连哪些 server
    2. 逐个连接, 获取每个 server 的 tool 列表
    3. 把每个 tool 包装成 ToolDefinition 注册到全局 ToolRegistry

    生命周期:
        bridge = MCPBridge()
        await bridge.connect_all(config.mcp_servers)  # 启动时
        # ... 应用运行, Worker 调用 MCP tools ...
        await bridge.disconnect_all()                  # 关闭时
    """

    def __init__(self):
        # 长连接方案: session在connect_all时建立, disconnect_all才关闭
        # 而不是每次 call_tool都重新握手
        self._sessions: dict[str, ClientSession] = {}
        self._connections: dict[str, MCPConnection] = {}


    async def connect_all(self, servers: dict[str, MCPServerConfig]) -> list[str]:
        """连接所有配置的MCP servers, 注册它们的tools

        一个 server 连接失败不影响其他 server

        Args:
            servers: config.mcp_servers

        Returns:
            成功注册的tool名列表(含mcp_前缀)
        """
        registered = []

        for server_name, cfg in servers.items():
            if hasattr(cfg, 'enabled') and not cfg.enabled:
                logger.debug(f"MCP {server_name} disabled, skipping")
                continue

            try:
                names = await self._connect_one(server_name, cfg)
                registered.extend(names)
                logger.info(
                    f"MCP {server_name}: {len(names)} tools registered"
                )
            except Exception as e:
                logger.warning(f"MCP '{server_name}' failed: {e}")
        
        return registered

    
    async def _connect_one(self, name: str, cfg: MCPServerConfig) -> list[str]:
        """connect one MCP server: 建连接 -> 拉tools -> register

        长连接: session保持活跃，disconnect_all才关
        """
        # 根据transport_type创建MCPConnection
        transport_type = getattr(cfg, 'transport', 'stdio')

        if transport_type == 'stdio':
            conn = MCPConnection.stdio(
                command=cfg.command,
                args=getattr(cfg, 'args', []),
                env=getattr(cfg, 'env', None),
            )
        elif transport_type in ('streamable-http', 'http'):
            conn = MCPConnection.streamable_http(
                url=cfg.url,
                headers=getattr(cfg, 'headers', None)
            )
        else:
            raise MCPError(
                f"Unknown transport: '{transport_type}' for server '{name}'"
            )

        session = await conn.__aenter__()
        self._connections[name] = conn
        self._sessions[name] = session

        # 拉取server提供的tools
        # response.tools是MCP的MCPTool对象列表
        # 每个都有 .name / .description / .inputSchema 属性
        response = await session.list_tools()
        tools = response.tools
        logger.debug(f"MCP '{name}': found {len(tools)} tools")

        return self._register_mcp_tools(session, tools, name)

    
    def _register_mcp_tools(self, session: ClientSession, tools: list, server_name: str) -> list[str]:
        """把 MCP tool 列表转换为 ToolDefinition 并注册到 Registry。

        对每个 MCP tool:
        1. 创建一个 wrapper 函数 (_call_mcp)
        2. wrapper 内部调用 session.call_tool() → 走 MCP 协议执行
        3. 转成 ToolDefinition 注册到 ToolRegistry

        当 Worker 调 ToolExecutor.execute("mcp_search", {...}) 时:
          → ToolExecutor 从 Registry 取出 _call_mcp
          → _call_mcp 内部调 session.call_tool("search", arguments={...})
          → MCP server 执行并返回结果
          → Worker 拿到结果, 完全不知道底层是 MCP
        """
        registry = get_registry()
        names = []

        for tool in tools:
            mcp_name = tool.name          # MCP server 给的名字, 如 "search"
            reg_name = f"mcp_{mcp_name}"  # Registry 里的名字, 如 "mcp_search"
                                          # 加 mcp_ 前缀防止和内置 tool 重名

            async def _call_mcp(_session: ClientSession = session,
                               _mcp_name: str = mcp_name, **kwargs) -> str:
                # 调 MCP server 的工具
                result = await _session.call_tool(_mcp_name, arguments=kwargs)

                # result.content 是官方 SDK 的 ContentBlock 对象列表
                # 可能类型: TextContent (type="text", text="...") /
                #           ImageContent / EmbeddedResource
                # 用 hasattr 检查, 不用 dict 访问 —— 这是 SDK 对象, 不是 dict
                texts = []
                for block in result.content:
                    if hasattr(block, 'text'):
                        texts.append(block.text)

                # 有文本内容就join，没有就返回原始对象的字符串表示
                return '\n'.join(texts) if texts else str(result)
        
            # ── ToolDefinition ──
            # MCP tool 对象字段 → ToolDefinition 字段的映射:
            #   tool.name          → definition.name (加 mcp_ 前缀)
            #   tool.description   → definition.description
            #   tool.inputSchema   → definition.parameters (JSON Schema 原样传入)
            definition = ToolDefinition(
                name=reg_name,
                description=f"[MCP:{server_name}] {tool.description or mcp_name}",
                # 某些 MCP server 不提供 inputSchema → 兜底空 schema
                parameters=tool.inputSchema if tool.inputSchema else {
                    "type": "object", "properties": {}
                },
                category=ToolCategory.READ,
                timeout_ms=30000,
                max_retries=1,
            )

            # 注册到全局 ToolRegistry
            registry.register(definition, _call_mcp)
            names.append(reg_name)
            logger.debug(f"  Registered: {reg_name} ({mcp_name})")
        
        return names


    async def disconnect_all(self) -> None:
        """关闭所有MCP 连接

        每个 MCPConnection 的 __aexit__ 会触发 AsyncExitStack.aclose(),
        逆序关闭 session 和 transport。
        """
        for name in list(self._connections.keys()):
            try:
                await self._connections[name].__aexit__(None, None, None)
                logger.debug(f"Disconnected MCP '{name}'")
            except Exception as e:
                logger.warning(f"Error disconnecting MCP '{name}' : {e}")

        self._connections.clear()
        self._sessions.clear()