"""Track aggregate LLM token use against a run budget."""

from __future__ import annotations

from litagent.logging import get_logger

logger = get_logger("safety.budget")


class CostBudget:
    """Track token consumption and emit a single threshold warning."""

    def __init__(self, max_tokens: int = 500_000, warn_ratio: float = 0.8):
        self._max_tokens = max_tokens
        self._warn_ratio = warn_ratio
        self._used_tokens = 0
        self._call_count = 0
        self._warned = False

    def record(self, usage: dict) -> None:
        """Add one provider usage record to the budget."""
        self._used_tokens += usage.get("prompt_tokens", 0)
        self._used_tokens += usage.get("completion_tokens", 0)
        self._call_count += 1

        if (
            not self._warned
            and self._used_tokens >= self._max_tokens * self._warn_ratio
        ):
            self._warned = True
            logger.warning(
                f"Cost budget at {self._used_tokens}/{self._max_tokens} tokens "
                f"({self._used_tokens / self._max_tokens:.0%})"
            )

    def is_exceeded(self) -> bool:
        """Return whether token use reached the hard limit."""
        return self._used_tokens >= self._max_tokens

    def remaining(self) -> int:
        """Return the non-negative remaining token allowance."""
        return max(0, self._max_tokens - self._used_tokens)

    @property
    def used(self) -> int:
        """Return the consumed budget."""
        return self._used_tokens

    @property
    def call_count(self) -> int:
        """Return the number of recorded LLM calls."""
        return self._call_count
