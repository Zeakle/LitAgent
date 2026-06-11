from typing import Callable
from litagent.tools.base import ToolDefinition, RegisteredTool, ToolCategory
from litagent.logging import get_logger

logger = get_logger('tools.registry')


class ToolRegistry:
    """全局工具注册，启动时全量加载，运行时不可变"""

    def __init__(self):
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, definition: ToolDefinition, func: Callable) -> None:
        """注册一个工具。同名工具后注册的覆盖先注册的。"""
        name = definition.name
        if name in self._tools:
            logger.warning(f"Tool '{name}' is being overwritten")
        self._tools[name] = RegisteredTool(definition, func)
        logger.debug(f"Registered tool: {name} (v{definition.version})")

    def get(self, name:str) -> RegisteredTool:
        """按名称获取工具。不存在抛KeyError"""
        if name not in self._tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self._tools[name]
    
    def list_all(self) -> list[ToolDefinition]:
        """获取所有已注册工具的definition列表"""
        return [t.definition for t in self._tools.values()]
    
    def list_by_category(self, category: ToolCategory) -> list[ToolDefinition]:
        """按类别筛选"""
        return [d for d in self.list_all() if d.category==category]

    def to_llm_format(self, names: list[str] | None = None) -> list[dict]:
        """转为 LLM 可见的 tool schema 列表。names=None 时输出全部。"""
        tools = self._tools.values()
        if names:
            tools = [self._tools[n] for n in names if n in self._tools]
        return [tool.definition.to_llm_format() for tool in tools]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# 全局单例
_registry: ToolRegistry | None = None

def get_registry() -> ToolRegistry:
    """获取全局 ToolRegistry 单例"""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


def reset_registry() -> None:
    """重置全局 Registry（仅测试用）"""
    global _registry
    _registry = ToolRegistry()