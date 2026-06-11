from enum import Enum
from typing import Any, Callable, Literal
from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """(str, Enum) 让枚举成员天生等价于字符串"""
    READ = 'read'
    WRITE = 'write'
    DESTRUCTIVE = 'destructive'


class RateLimitConfig(BaseModel):
    max_calls: int
    window_seconds: int = 60


class FallbackStep(BaseModel):
    """降级链中的一个步骤"""
    type: Literal['cached', 'default_value', 'skip', 'alternative_tool']
    alternative_tool: str | None = None
    default_result: Any = None


class ToolDefinition(BaseModel):
    """工具的完整定义——对外（LLM context）和对内（Registry）共用。

    序列化给 LLM 时只输出 name/description/parameters 三个字段。
    其他字段用于内部执行控制。
    """
    name: str
    description: str
    parameters: dict = Field(default_factory=dict)
    category: ToolCategory = ToolCategory.READ
    timeout_ms: int = 30000
    max_retries: int = 2
    rate_limit: RateLimitConfig | None = None
    fallback: list[FallbackStep] = Field(default_factory=list)
    cache_ttl_ms: int = 0
    version: str = '1.0.0'

    def to_llm_format(self) -> dict:
        """给 LLM 看的 tool schema（OpenAI/Anthropic 格式）"""
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }


class RegisteredTool:
    """Registry 内部存储的完整工具——ToolDefinition + 可调用函数"""

    def __init__(self, definition: ToolDefinition, func: Callable):
        self.definition = definition
        self.func = func