"""Context Pipeline——分层组装 context，BudgetManager 控制预算。"""


from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any, Awaitable

from litagent.context.budget import BudgetManager
from litagent.logging import get_logger


logger = get_logger('context.pipeline')


@dataclass
class ContextLayer:
    """Context 中的一个层。

    Attributes:
        name: 层名称（如 "system_prompt", "tools", "papers", "memory"）
        priority: 优先级，数字越小越先处理，越先分配预算
        max_tokens: 本层最大 token 数
        builder: async (state: dict) -> str 的回调，返回该层的文本内容
    """
    name: str
    priority: int
    max_tokens: int
    builder: Callable[[dict[str, Any]], Awaitable[str]]  # 一个async函数，接受字典，返回字符串


class ContextPipeline:
    """分层组装 context。

    按 priority 顺序构建每个 layer，高优先级层先拿预算。
    超预算时低优先级层被截断或跳过。

    用法（Phase 8 Worker 接入时）：
        pipeline = ContextPipeline(budget)
        pipeline.add_layer(ContextLayer("system", 0, 2000, build_system))
        pipeline.add_layer(ContextLayer("tools", 1, 1000, build_tools))
        pipeline.add_layer(ContextLayer("papers", 2, 8000, build_papers))
        result = await pipeline.build(state)
    """

    def __init__(self, budget: BudgetManager):
        self._layers: list[ContextLayer] = []
        self._budget = budget


    def add_layer(self, layer: ContextLayer) -> ContextPipeline:
        """添加一个层，返回self支持链式调用"""
        self._layers.append(layer)
        return self

    
    async def build(self, state: dict[str, Any]) -> tuple[str, int]:
        """按优先级构建所有层，返回 (完整 context, 总 token 数)。

        高优先级层先执行，先消耗预算。
        当剩余预算不够下一层的 max_tokens 时，尝试用剩余空间截断。
        剩余预算为 0 时跳过后续层。
        builder 
        """
        sorted_layers = sorted(self._layers, key=lambda la: la.priority)

        parts: list[str] = []  # each layer生成的文本
        total_used = 0
        
        for layer in sorted_layers:
            remaining = self._budget.remaining(total_used)
            if remaining == 0:
                logger.debug(f"Skipping layer '{layer.name}': no budget remaining")
                break

            budget_for_layer = min(layer.max_tokens, remaining)

            try:
                content = await layer.builder(state)
            except Exception as e:
                logger.warning(f"Layer '{layer.name}' builder failed: {e}, skipping")
                continue
            
            if not content:
                continue

            token_count = self._budget.count_tokens(content)
            if token_count > budget_for_layer:
                content = self._budget.truncate(content, budget_for_layer)
                # 更新truncated后的token count
                token_count = self._budget.count_tokens(content)

            parts.append(content)
            total_used += token_count
        
        result = '\n\n'.join(parts)

        if self._budget.needs_compact(total_used):
            logger.warning(
                f"Context at {total_used}/{self._budget.max_tokens} tokens "
                f"({total_used / self._budget.max_tokens:.0%}), exceeds compact threshold"
            )

        return result, total_used