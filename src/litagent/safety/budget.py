"""Track aggregate LLM token use against a run budget."""

from __future__ import annotations

from litagent.logging import get_logger

logger = get_logger("safety.budget")


class CostBudget:
    """Track token consumption and emit a single threshold warning."""

    def __init__(self, max_tokens: int = 500_000, warn_ratio: float = 0.8):
        """Initialize the cost budget."""
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

    def snapshot(self) -> dict:
        """Serialize cumulative usage for a safe-boundary resume."""
        return {
            "schema_version": 1,
            "max_tokens": self._max_tokens,
            "warn_ratio": self._warn_ratio,
            "used_tokens": self._used_tokens,
            "call_count": self._call_count,
            "warned": self._warned,
        }

    @classmethod
    def from_snapshot(cls, data: dict) -> "CostBudget":
        """Restore usage without granting a fresh budget."""
        if not isinstance(data, dict) or data.get("schema_version") != 1:
            raise ValueError("invalid_cost_budget_snapshot")
        values = {
            key: data.get(key) for key in ("max_tokens", "used_tokens", "call_count")
        }
        if any(not isinstance(value, int) or value < 0 for value in values.values()):
            raise ValueError("invalid_cost_budget_values")
        warn_ratio = data.get("warn_ratio")
        if not isinstance(warn_ratio, (int, float)) or not 0 < warn_ratio <= 1:
            raise ValueError("invalid_cost_budget_warn_ratio")
        budget = cls(max_tokens=values["max_tokens"], warn_ratio=float(warn_ratio))
        budget._used_tokens = values["used_tokens"]
        budget._call_count = values["call_count"]
        budget._warned = bool(data.get("warned", False))
        return budget

    def restore(self, data: dict) -> None:
        """Restore into the object already shared with LLM clients and scheduler."""
        restored = self.from_snapshot(data)
        if restored._max_tokens != self._max_tokens:
            raise ValueError("cost_budget_limit_mismatch")
        self._warn_ratio = restored._warn_ratio
        self._used_tokens = restored._used_tokens
        self._call_count = restored._call_count
        self._warned = restored._warned

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
