"""MCP 连接薄封装。基于 Anthropic 官方 mcp SDK。"""

from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from litagent.logging import get_logger

logger = get_logger('mcp_connection')


class MCPConnection:
    """统一 stdio / streamable-http 的 MCP 客户端连接。

    本质是一个 async context manager:
    - 进入 (__aenter__): 建 transport → 建 session → 握手 → 返回 session
    - 退出 (__aexit__): AsyncExitStack 自动逆序清理 session → transport

    两种构造方式的区别在于 transport:

    stdio: 启动本地子进程
      → MCPConnection.stdio("npx", ["@anthropic/mcp-server-filesystem", "/tmp"])
      → 内部: npx @anthropic/mcp-server-filesystem /tmp
      → 通信: 往子进程 stdin 写 JSON, 从 stdout 读 JSON
      → 适合: MCP server 是本地命令行程序

    streamable-http: 连远程 HTTP 服务
      → MCPConnection.streamable_http("http://localhost:8080/mcp")
      → 内部: POST http://localhost:8080/mcp
      → 通信: HTTP 请求体发 JSON, 响应体收 JSON
      → 适合: MCP server 是远程 web 服务
    """

    def __init__(self, transport: str, **kwargs):
        # transport: 'stdio' or 'streamable-http'
        # kwargs: 传给对应transport的参数 (command/args/env 或 url/headers)
        self._transport = transport
        self._kwargs = kwargs

        # 批量管理异步资源
        self._exit_stack = AsyncExitStack()


    @classmethod
    def stdio(cls, command: str, args: list[str] | None = None,
              env: dict[str, str] | None = None) -> "MCPConnection":
        """创建 stdio 连接。

        Args:
            command: 可执行文件 (npx / python / uvx)
            args: 命令行参数
            env: 传给子进程的额外环境变量 (如 API_KEY=xxx)
        """ 
        return cls("stdio", command=command, args=args or [], env = env or {})


    @classmethod
    def streamable_http(cls, url: str, headers: dict[str, str] | None = None) -> "MCPConnection":
        """创建 streamable-http 连接。

        Args:
            url: MCP server 的 HTTP 端点 (如 http://localhost:8080/mcp)
            headers: 额外 HTTP 头 (如 Authorization: Bearer xxx)
        """
        return cls("streamable-http", url=url, headers=headers or {})


    async def __aenter__(self) -> ClientSession:
        """__aenter__：进入async with时自动调用
        建立全链路: transport -> session -> 握手 -> 返回就绪session

        官方SDK把transport和session分开创建(两层async with)
        此处用 AsyncExitStack把两层压入一个__aenter__
        """

        # step1 打开transport
        if self._transport == 'stdio':
            # StdioServerParameters: 告诉官方 SDK 启动什么子进程,传什么参
            params = StdioServerParameters(
                command=self._kwargs['command'],
                args=self._kwargs.get('args', []),
                env=self._kwargs.get('env', {}),
            )

            # stdio_client(params): 启动子进程，返回(read_stream, write_stream)
            # enter_async_context: 将这个context manager 托管给 exit_stack
            # self._exit_stack销毁时，自动关闭该stdio进程
            read, write = await self._exit_stack.enter_async_context(stdio_client(params))
        elif self._transport == 'streamable-http':
            # streamablehttp_client(url): 建立HTTP连接，返回(read, write, get_session_id(跨会话使用))
            read, write, *_ = await self._exit_stack.enter_async_context(
                streamable_http_client(
                    url=self._kwargs['url'],
                    headers=self._kwargs.get('headers')
                )
            )
        else:
            raise ValueError(
                f"Unknown transport: '{self._transport}'"
                f"Must be 'stdio' or 'streamable-http'."
            )

        # Step2: 在transport之上建立ClientSession(JSON-RPC协议层)
        # ClientSession 负责: JSON-RPC 序列化、 请求-响应匹配、握手、错误处理
        session = await self._exit_stack.enter_async_context(
            ClientSession(read, write)
        )

        # Step3: MCP握手
        # SDK内部自动完成initialize请求 -> 收响应 -> 发initialized通知
        await session.initialize()
        logger.debug(f"MCP connected via {self._transport}")

        return session

    
    async def __aexit__(self, *exc) -> None:
        """退出时统一清理

        AsyncExitStack.aclose() 按进入的逆序退出所有 context manager:
        1. 先关 ClientSession
        2. 再关 transport（停止子进程 / 关 HTTP 连接）
        """
        await self._exit_stack.aclose()