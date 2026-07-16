"""Phase 12 Docker Sandbox tests.

单元测试（不需 Docker）：
  - build_sandbox_command 参数组装
  - MCPConnection.stdio(sandboxed=) 命令包装
集成测试（需 Docker，标 @pytest.mark.integration，CI 跳过）：
  - 真跑容器 JSON-RPC roundtrip
"""

import pytest

from litagent.sandbox.docker_cmd import build_sandbox_command
from litagent.mcp.connection import MCPConnection


# ═══════════════════════════════════════════════════════
# build_sandbox_command —— 纯函数，参数组装
# ═══════════════════════════════════════════════════════

class TestBuildSandboxCommand:
    def test_returns_docker_command(self):
        cmd, args = build_sandbox_command("npx", ["srv"])
        assert cmd == "docker"
        assert args[0] == "run"

    def test_original_command_after_image(self):
        """command + args 必须在 image 之后（docker CLI 要求的顺序）。"""
        cmd, args = build_sandbox_command("npx", ["server-fs", "/workspace"])
        img_idx = args.index("agent-sandbox:latest")
        assert args.index("npx") > img_idx
        assert args.index("server-fs") > img_idx
        assert args.index("/workspace") > img_idx

    def test_hardening_flags_present(self):
        _, args = build_sandbox_command("npx", [])
        # 网络隔离
        assert args[args.index("--network") + 1] == "none"
        # 能力隔离
        assert args[args.index("--cap-drop") + 1] == "ALL"
        # 文件系统只读
        assert "--read-only" in args
        # 退出即删
        assert "--rm" in args
        # stdin 保持（JSON-RPC 通道）
        assert "-i" in args
        # 提权防护
        assert args[args.index("--security-opt") + 1] == "no-new-privileges:true"

    def test_resource_limits(self):
        _, args = build_sandbox_command("npx", [])
        assert args[args.index("--memory") + 1] == "512m"
        assert args[args.index("--memory-swap") + 1] == "512m"  # = memory → 禁 swap
        assert args[args.index("--cpus") + 1] == "1"
        assert args[args.index("--pids-limit") + 1] == "100"

    def test_tmpfs_noexec(self):
        """/tmp 是只读 rootfs 下唯一可写点，必须 noexec 防写二进制再执行。"""
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
        """env 通过 docker run -e 注入容器（沙箱后子进程是 docker CLI）。"""
        _, args = build_sandbox_command("npx", [], env={"API_KEY": "secret", "DEBUG": "1"})
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


# ═══════════════════════════════════════════════════════
# MCPConnection.stdio(sandboxed=) —— 命令包装
# ═══════════════════════════════════════════════════════

class TestMCPConnectionSandboxed:
    def test_sandboxed_wraps_docker(self):
        conn = MCPConnection.stdio("npx", ["srv"], sandboxed=True)
        assert conn._kwargs["command"] == "docker"
        assert "npx" in conn._kwargs["args"]
        assert "srv" in conn._kwargs["args"]

    def test_not_sandboxed_unchanged(self):
        """sandboxed=False → 行为和 Phase 9 完全一致（向后兼容）。"""
        conn = MCPConnection.stdio("npx", ["srv"])
        assert conn._kwargs["command"] == "npx"
        assert conn._kwargs["args"] == ["srv"]

    def test_transport_still_stdio(self):
        """sandboxed 只换 command，transport 类型仍是 stdio。"""
        conn = MCPConnection.stdio("npx", ["srv"], sandboxed=True)
        assert conn._transport == "stdio"

    def test_env_injected_and_cleared(self):
        """env 进 docker -e 后，不再传给 docker CLI 进程本身（防传两遍）。"""
        conn = MCPConnection.stdio("npx", [], env={"K": "v"}, sandboxed=True)
        assert "K=v" in conn._kwargs["args"]
        assert conn._kwargs["env"] == {}

    def test_sandbox_network_passed(self):
        conn = MCPConnection.stdio("npx", [], sandboxed=True, sandbox_network="bridge")
        assert conn._kwargs["args"][conn._kwargs["args"].index("--network") + 1] == "bridge"

    def test_custom_image_passed(self):
        conn = MCPConnection.stdio("npx", [], sandboxed=True, image="custom:latest")
        assert "custom:latest" in conn._kwargs["args"]


# ═══════════════════════════════════════════════════════
# 集成测试 —— 需真实 Docker（CI 跳过）
# ═══════════════════════════════════════════════════════

@pytest.mark.integration
class TestSandboxIntegration:
    """需要 `docker build -t agent-sandbox:latest docker/` 和 Docker daemon。

    运行: pytest tests/test_sandbox.py -v -m integration
    """

    @pytest.mark.asyncio
    async def test_sandboxed_mcp_roundtrip(self):
        """沙箱模式跑预装的 filesystem server，验证 JSON-RPC 握手 + list_tools。"""
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
        """--network none → 容器内无法访问外网。"""
        import asyncio

        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "--rm", "--network", "none", "agent-sandbox:latest",
            "python", "-c",
            "import urllib.request; urllib.request.urlopen('http://example.com', timeout=5)",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        # 断网 → urlopen 应失败（非 0 退出）
        assert proc.returncode != 0
