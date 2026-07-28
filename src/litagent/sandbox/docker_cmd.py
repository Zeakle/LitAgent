"""Build hardened Docker command lines for sandboxed tool execution."""


def build_sandbox_command(
    command: str,
    args: list[str] | None = None,
    image: str = "agent-sandbox:latest",
    network: str = "none",
    env: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Return the Docker executable and isolated container arguments."""
    args = args or []
    docker_args = [
        "run",
        "--rm",
        "-i",
        "--network",
        network,
        "--cpus",
        "1",
        "--memory",
        "512m",
        "--memory-swap",
        "512m",
        "--pids-limit",
        "100",
        "--security-opt",
        "no-new-privileges:true",
        "--cap-drop",
        "ALL",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,nodev,size=256m",
    ]
    for k, v in (env or {}).items():
        docker_args.extend(["-e", f"{k}={v}"])
    docker_args.extend([image, command, *args])
    return "docker", docker_args
