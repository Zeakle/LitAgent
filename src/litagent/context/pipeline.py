"""Assemble prioritized context layers within a token budget."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any, Awaitable

from litagent.context.budget import BudgetManager
from litagent.logging import get_logger

logger = get_logger("context.pipeline")


@dataclass
class ContextLayer:
    """Describe one asynchronous context producer and its budget."""

    name: str
    priority: int
    max_tokens: int
    builder: Callable[[dict[str, Any]], Awaitable[str]]


class ContextPipeline:
    """Build ordered context layers under a shared token budget."""

    def __init__(self, budget: BudgetManager):
        self._layers: list[ContextLayer] = []
        self._budget = budget

    def add_layer(self, layer: ContextLayer) -> ContextPipeline:
        """Register a context layer and return this pipeline."""
        self._layers.append(layer)
        return self

    async def build(self, state: dict[str, Any]) -> tuple[str, int]:
        """Build layers by priority and return text with token usage."""
        sorted_layers = sorted(self._layers, key=lambda la: la.priority)

        parts: list[str] = []
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

                token_count = self._budget.count_tokens(content)

            parts.append(content)
            total_used += token_count

        result = "\n\n".join(parts)

        if self._budget.needs_compact(total_used):
            logger.warning(
                f"Context at {total_used}/{self._budget.max_tokens} tokens "
                f"({total_used / self._budget.max_tokens:.0%}), "
                "exceeds compact threshold"
            )

        return result, total_used
