"""Estimate and enforce context token budgets."""

from __future__ import annotations

from abc import ABC, abstractmethod

from litagent.logging import get_logger

logger = get_logger("context.budget")


class TokenCounter(ABC):
    """Define the token-counting interface used by context assembly."""

    @abstractmethod
    def count(self, text: str) -> int:
        """Count tokens in the supplied text."""
        ...

    def chars_per_token(self) -> int:
        """Return the configured character-to-token ratio."""
        return 4


class CharBasedCounter(TokenCounter):
    """Estimate token counts from a fixed character ratio."""

    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        """Estimate the token count for the supplied text."""
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def chars_per_token(self) -> int:
        """Return the configured character-to-token ratio."""
        return self.CHARS_PER_TOKEN


class BudgetManager:
    """Apply token-counting, compaction, and truncation policies."""

    def __init__(
        self,
        max_tokens: int = 16000,
        compact_threshold: float = 0.7,
        counter: TokenCounter | None = None,
    ):
        """Initialize the budget manager."""
        self._max_tokens = max_tokens
        self._compact_threshold = compact_threshold
        self._counter = counter or CharBasedCounter()

    @property
    def max_tokens(self) -> int:
        """Return the configured token budget."""
        return self._max_tokens

    def count_tokens(self, text: str) -> int:
        """Count tokens in the supplied text."""
        return self._counter.count(text)

    def needs_compact(self, current_tokens: int) -> bool:
        """Return whether usage exceeds the compaction threshold."""
        if self._max_tokens <= 0:
            return False
        return current_tokens / self._max_tokens > self._compact_threshold

    def remaining(self, used: int) -> int:
        """Return unused tokens, clamped at zero."""
        return max(0, self._max_tokens - used)

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text, preferring a nearby sentence boundary."""
        if self.count_tokens(text) <= max_tokens:
            return text

        char_limit = max_tokens * self._counter.chars_per_token()
        truncated = text[:char_limit]

        last_period = truncated.rfind(". ")
        last_newline = truncated.rfind("\n")
        cut_point = max(last_period, last_newline)

        if cut_point > char_limit * 0.5:
            truncated = truncated[: cut_point + 1]

        return truncated
