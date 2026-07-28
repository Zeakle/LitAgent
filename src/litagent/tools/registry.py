"""Register callable tools and expose their schemas to workers and LLMs."""

from typing import Callable

from litagent.tools.base import ToolDefinition, RegisteredTool, ToolCategory
from litagent.logging import get_logger

logger = get_logger("tools.registry")


class ToolRegistry:
    """Store registered tools by unique name."""

    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, definition: ToolDefinition, func: Callable) -> None:
        """Add or replace a callable tool definition."""
        name = definition.name
        if name in self._tools:
            logger.warning(f"Tool '{name}' is being overwritten")
        self._tools[name] = RegisteredTool(definition, func)
        logger.debug(f"Registered tool: {name} (v{definition.version})")

    def get(self, name: str) -> RegisteredTool:
        """Return a registered tool or raise ``KeyError`` when absent."""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_all(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return [t.definition for t in self._tools.values()]

    def list_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """Return tool definitions in one effect category."""
        return [d for d in self.list_all() if d.category == category]

    def to_llm_format(self, names: list[str] | None = None) -> list[dict]:
        """Return selected tool definitions in the LLM-facing schema."""
        tools = self._tools.values()
        if names:
            tools = [self._tools[n] for n in names if n in self._tools]
        return [tool.definition.to_llm_format() for tool in tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


_registry: ToolRegistry | None = None


def get_registry() -> ToolRegistry:
    """Return the process-wide tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    """Replace the process-wide registry with an empty instance."""
    global _registry
    _registry = ToolRegistry()
