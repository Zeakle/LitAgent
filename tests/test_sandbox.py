"""Tests for Docker sandbox command and MCP isolation."""

import pytest

from litagent.sandbox.docker_cmd import build_sandbox_command
from litagent.mcp.connection import MCPConnection


class TestBuildSandboxCommand:
    """Tests hardened sandbox command construction."""

    def test_returns_docker_command(self):
        cmd, args = build_sandbox_command("npx", ["srv"])
        assert cmd == "docker"
        assert args[0] == "run"

    def test_original_command_after_image(self):
        """The original command follows the container image."""
        cmd, args = build_sandbox_command("npx", ["server-fs", "/workspace"])
        img_idx = args.index("agent-sandbox:latest")
        assert args.index("npx") > img_idx
        assert args.index("server-fs") > img_idx
        assert args.index("/workspace") > img_idx

    def test_hardening_flags_present(self):
        _, args = build_sandbox_command("npx", [])

        assert args[args.index("--network") + 1] == "none"

        assert args[args.index("--cap-drop") + 1] == "ALL"

        assert "--read-only" in args

        assert "--rm" in args

        assert "-i" in args

        assert args[args.index("--security-opt") + 1] == "no-new-privileges:true"

    def test_resource_limits(self):
        _, args = build_sandbox_command("npx", [])
        assert args[args.index("--memory") + 1] == "512m"
        assert args[args.index("--memory-swap") + 1] == "512m"
        assert args[args.index("--cpus") + 1] == "1"
        assert args[args.index("--pids-limit") + 1] == "100"

    def test_tmpfs_noexec(self):
        """Temporary storage is mounted with execution disabled."""
        _, args = build_sandbox_command("npx", [])
        tmpfs_spec = args[args.index("--tmpfs") + 1]
        assert "/tmp" in tmpfs_spec
        assert "noexec" in tmpfs_spec
        assert "nosuid" in tmpfs_spec
        assert "nodev" in tmpfs_spec

    def test_network_default_none(self):
        _, args = build_sandbox_command("npx", [])
        assert args[args.index("--network") + 1] == "none"

    def test_network_override_bridge(self):
        _, args = build_sandbox_command("npx", [], network="bridge")
        assert args[args.index("--network") + 1] == "bridge"

    def test_custom_image(self):
        _, args = build_sandbox_command("npx", [], image="my-sandbox:v2")
        assert "my-sandbox:v2" in args

    def test_env_injected_as_e_flags(self):
        """Environment variables are passed as Docker flags."""
        _, args = build_sandbox_command(
            "npx", [], env={"API_KEY": "secret", "DEBUG": "1"}
        )
        assert "-e" in args
        assert "API_KEY=secret" in args
        assert "DEBUG=1" in args

    def test_env_none_no_e_flags(self):
        _, args = build_sandbox_command("npx", [])
        assert "-e" not in args

    def test_none_args_handled(self):
        cmd, args = build_sandbox_command("npx")
        assert cmd == "docker"
        assert "npx" in args


class TestMCPConnectionSandboxed:
    """Tests sandboxed MCP stdio connections."""

    def test_sandboxed_wraps_docker(self):
        conn = MCPConnection.stdio("npx", ["srv"], sandboxed=True)
        assert conn._kwargs["command"] == "docker"
        assert "npx" in conn._kwargs["args"]
        assert "srv" in conn._kwargs["args"]

    def test_not_sandboxed_unchanged(self):
        """Unsandboxed commands remain unchanged."""
        conn = MCPConnection.stdio("npx", ["srv"])
        assert conn._kwargs["command"] == "npx"
        assert conn._kwargs["args"] == ["srv"]

    def test_transport_still_stdio(self):
        """Sandbox wrapping preserves the stdio transport."""
        conn = MCPConnection.stdio("npx", ["srv"], sandboxed=True)
        assert conn._transport == "stdio"

    def test_env_injected_and_cleared(self):
        """Sandboxed environments are injected into Docker only."""
        conn = MCPConnection.stdio("npx", [], env={"K": "v"}, sandboxed=True)
        assert "K=v" in conn._kwargs["args"]
        assert conn._kwargs["env"] == {}

    def test_sandbox_network_passed(self):
        conn = MCPConnection.stdio("npx", [], sandboxed=True, sandbox_network="bridge")
        assert (
            conn._kwargs["args"][conn._kwargs["args"].index("--network") + 1]
            == "bridge"
        )

    def test_custom_image_passed(self):
        conn = MCPConnection.stdio("npx", [], sandboxed=True, image="custom:latest")
        assert "custom:latest" in conn._kwargs["args"]


@pytest.mark.integration
class TestSandboxIntegration:
    """Tests sandbox integration against Docker."""

    @pytest.mark.asyncio
    async def test_sandboxed_mcp_roundtrip(self):
        """A sandboxed MCP process completes a protocol round trip."""
        conn = MCPConnection.stdio(
            "npx",
            ["@modelcontextprotocol/server-filesystem", "/workspace"],
            sandboxed=True,
        )
        async with conn as session:
            response = await session.list_tools()
            assert len(response.tools) > 0

    @pytest.mark.asyncio
    async def test_network_isolation(self):
        """Network-isolated containers cannot reach external URLs."""
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "agent-sandbox:latest",
            "python",
            "-c",
            "import urllib.request; "
            "urllib.request.urlopen('http://example.com', timeout=5)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()

        assert proc.returncode != 0
