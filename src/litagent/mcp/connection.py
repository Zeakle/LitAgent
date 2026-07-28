"""Open MCP sessions over stdio or streamable HTTP transports."""

from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from litagent.logging import get_logger

logger = get_logger("mcp_connection")


class MCPConnection:
    """Manage an MCP transport and session with one async exit stack."""

    def __init__(self, transport: str, **kwargs):
        self._transport = transport
        self._kwargs = kwargs

        self._exit_stack = AsyncExitStack()

    @classmethod
    def stdio(
        cls,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        sandboxed: bool = False,
        sandbox_network: str = "none",
        image: str = "agent-sandbox:latest",
    ) -> "MCPConnection":
        """Configure a stdio connection, optionally wrapped in Docker."""
        if sandboxed:
            from litagent.sandbox.docker_cmd import build_sandbox_command

            command, args = build_sandbox_command(
                command, args, image=image, network=sandbox_network, env=env
            )
            # The Docker command carries the environment; the host client does not.
            env = {}

        return cls("stdio", command=command, args=args or [], env=env or {})

    @classmethod
    def streamable_http(
        cls, url: str, headers: dict[str, str] | None = None
    ) -> "MCPConnection":
        """Configure a streamable HTTP connection."""
        return cls("streamable-http", url=url, headers=headers or {})

    async def __aenter__(self) -> ClientSession:
        """Open the transport, initialize MCP, and return the session."""

        if self._transport == "stdio":

            params = StdioServerParameters(
                command=self._kwargs["command"],
                args=self._kwargs.get("args", []),
                env=self._kwargs.get("env", {}),
            )

            read, write = await self._exit_stack.enter_async_context(
                stdio_client(params)
            )
        elif self._transport == "streamable-http":

            read, write, *_ = await self._exit_stack.enter_async_context(
                streamable_http_client(
                    url=self._kwargs["url"], headers=self._kwargs.get("headers")
                )
            )
        else:
            raise ValueError(
                f"Unknown transport: '{self._transport}'"
                f"Must be 'stdio' or 'streamable-http'."
            )

        # Keep transport and session contexts on one stack for ordered cleanup.
        session = await self._exit_stack.enter_async_context(ClientSession(read, write))

        # Complete the MCP handshake before exposing the session.
        await session.initialize()
        logger.debug(f"MCP connected via {self._transport}")

        return session

    async def __aexit__(self, *exc) -> None:
        """Close the MCP session and transport resources."""
        await self._exit_stack.aclose()
