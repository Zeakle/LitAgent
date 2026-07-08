import os
from pathlib import Path
from dotenv import load_dotenv

import yaml
from pydantic import BaseModel, Field, ValidationError

load_dotenv()


class AgentConfig(BaseModel):
    """Agent配置"""
    max_loops: int = 15
    per_tool_timeout_ms: int = 30000
    per_loop_timeout_ms: int = 600000


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class MemoryConfig(BaseModel):
    redis_url: str = "redis://localhost:6379"
    qdrant_url: str = "http://localhost:6333"
    working_ttl_seconds: int = 1800
    pg_url: str = "postgresql://litagent:litagent@localhost:5432/litagent"


class ContextConfig(BaseModel):
    max_tokens: int = Field(default=16000, gt=0)
    compact_threshold: float = Field(default=0.7, gt=0, le=1.0)


class OrchestratorConfig(BaseModel):
    timeout_ms: int = Field(default=600000, gt=0)
    max_concurrent: int = Field(default=5, gt=0)


class LLMConfig(BaseModel):
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    max_tokens: int = Field(default=4096, gt=0)
    temperature: float = Field(default=0.1, ge=0, le=2.0)

class AdversarialConfig(BaseModel):
    max_rounds: int = Field(default=3, gt=0, le=5)
    pass_threshold: float = Field(default=0.8, gt=0, le=1.0)


class MCPServerConfig(BaseModel):
    transport: str = 'stdio'
    command: str | None = None
    args: list[str] = []
    env: dict | None = None
    url: str | None = None
    headers: dict | None = None
    enabled: bool = True
    sandboxed: bool = False
    sandbox_network: str = 'none'


class SafetyConfig(BaseModel):
    max_cost_tokens: int = Field(default=500_000, gt=0)
    cost_warn_ratio: float = Field(default=0.8, gt=0, le=1.0)


class ResilienceConfig(BaseModel):
    cb_fail_threshold: int = Field(default=5, gt=0)
    cb_cooldown_seconds: int = Field(default=60, gt=0)


class ExtractorConfig(BaseModel):
    max_concurrent: int = Field(default=5, gt=0)
    enable_llm: bool = True


class ObservabilityConfig(BaseModel):
    enabled: bool = False
    langfuse_host: str = 'http://localhost:3000'


class AppConfig(BaseModel):
    """应用顶层配置"""
    agent: AgentConfig
    logging: LoggingConfig
    memory: MemoryConfig = MemoryConfig()
    context: ContextConfig = ContextConfig()
    orchestrator: OrchestratorConfig = OrchestratorConfig()
    llm: LLMConfig = LLMConfig()
    adversarial: AdversarialConfig = AdversarialConfig()
    mcp_servers: dict[str, MCPServerConfig] = {}
    safety: SafetyConfig = SafetyConfig()
    resilience: ResilienceConfig = ResilienceConfig()
    extractor: ExtractorConfig = ExtractorConfig()
    observability: ObservabilityConfig = ObservabilityConfig()


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise FileNotFoundError("Cannot find project root (no pyproject.toml found)")


def _find_config_file(config_path: str | None = None) -> Path:
    """确定配置文件路径。

    优先级:
    1. 参数传入的 config_path
    2. 环境变量 LITAGENT_CONFIG
    3. 项目根目录下的 config/default.yaml
    """
    if config_path:
        path = Path(config_path)
    elif os.environ.get("LITAGENT_CONFIG"):
        path = Path(os.environ["LITAGENT_CONFIG"])
    else:
        path = _find_project_root() / "config" / "default.yaml"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return path


def _apply_env_overrides(data: dict, prefix: str = "LITAGENT_") -> dict:
    """用环境变量覆盖 YAML 中的值。

    映射规则:
      LITAGENT_AGENT_MAX_LOOPS  → data["agent"]["max_loops"]
      LITAGENT_LOGGING_LEVEL    → data["logging"]["level"]

    按 SECTION_KEY 的格式解析: 第一个下划线分隔 section 和 key。
    LITAGENT_AGENT_MAX_LOOPS → section="agent", key="max_loops"
    """
    for env_key, env_val in os.environ.items():
        if not env_key.startswith(prefix):
            continue

        # 去掉前缀
        rest = env_key[len(prefix):].lower()

        # 第一个下划线分隔 section 和 key
        parts = rest.split("_", 1)
        if len(parts) != 2:
            continue

        section, key = parts
        if section in data:
            # 尝试类型转换（YAML 里是 int，环境变量是 str）
            original = data[section].get(key)
            if isinstance(original, int):
                data[section][key] = int(env_val)
            elif isinstance(original, float):
                data[section][key] = float(env_val)
            else:
                data[section][key] = env_val

    return data



def load_config(config_path: str | None = None) -> AppConfig:
    """加载配置：读取 YAML → 环境变量覆盖 → Pydantic 校验。

    整个应用的唯一配置入口。所有模块通过 load_config() 获取配置实例。

    用法:
        from litagent.config import load_config
        config = load_config()
        print(config.agent.max_loops)  # IDE 有自动补全
    """
    # 1. 找到配置文件
    config_file = _find_config_file(config_path)

    # 2. 读取 YAML
    with open(config_file, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    # 3. 环境变量覆盖
    raw = _apply_env_overrides(raw)

    # 4. Pydantic 校验
    try:
        return AppConfig(**raw)
    except ValidationError as e:
        from litagent.exceptions import ConfigError
        raise ConfigError(f"Invalid config: {e}") from e
