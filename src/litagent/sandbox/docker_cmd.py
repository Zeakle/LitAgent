"""把 MCP server 命令包装成硬化 docker run命令"""


def build_sandbox_command(
    command: str,
    args: list[str] | None = None,
    image: str = 'agent-sandbox:latest',
    network: str = 'none',
    env: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """返回 ("docker", [run 参数列表])。"""
    args = args or []
    docker_args = [
        "run",
        "--rm",                                       # 退出即删
        "-i",                                         # 保持 stdin（JSON-RPC 通道）
        "--network", network,
        "--cpus", "1",
        "--memory", "512m",
        "--memory-swap", "512m",                      # = memory，禁 swap
        "--pids-limit", "100",
        "--security-opt", "no-new-privileges:true",
        "--cap-drop", "ALL",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,nodev,size=256m",
    ]
    for k, v in (env or {}).items():
        docker_args.extend(["-e", f"{k}={v}"])
    docker_args.extend([image, command, *args])
    return "docker", docker_args