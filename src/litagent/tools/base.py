"""Define tool metadata, execution policy, and callable wrappers."""

from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field


class ToolCategory(str, Enum):
    """Classify tools by read, write, or destructive effects."""

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"


class RateLimitConfig(BaseModel):
    """Configure a rolling-window tool call limit."""

    max_calls: int
    window_seconds: int = 60


class FallbackStep(BaseModel):
    """Describe one ordered recovery action after tool failure."""

    type: Literal["cached", "default_value", "skip", "alternative_tool"]
    alternative_tool: str | None = None
    default_result: Any = None


class ToolDefinition(BaseModel):
    """Define the schema and execution policy for a callable tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    category: ToolCategory = ToolCategory.READ
    timeout_ms: int = 30000
    max_retries: int = 2
    rate_limit: RateLimitConfig | None = None
    fallback: list[FallbackStep] = Field(default_factory=list)
    cache_ttl_ms: int = 0
    version: str = "1.0.0"

    def to_llm_format(self) -> dict:
        """Return the tool metadata exposed to an LLM."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class RegisteredTool:
    """Bind a tool definition to its callable implementation."""

    def __init__(self, definition: ToolDefinition, func: Callable):
        """Initialize the registered tool."""
        self.definition = definition
        self.func = func
