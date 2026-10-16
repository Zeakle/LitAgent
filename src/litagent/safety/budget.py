"""Track aggregate LLM token use against a run budget."""

from __future__ import annotations

import json

from litagent.logging import get_logger

logger = get_logger("safety.budget")

STRICT_EXTRACTION_OUTPUT_TOKENS = 8192


def strict_request_reservation(messages, max_tokens, response_format=None) -> int:
    """Include native schemas in the same conservative request reservation."""
    size = len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))
    if response_format and response_format.get("type") == "json_schema":
        size += len(json.dumps(response_format, ensure_ascii=False).encode("utf-8"))
    return size + 1024 + max_tokens


class CostBudget:
    """Track token consumption and emit a single threshold warning."""

    def __init__(self, max_tokens: int = 500_000, warn_ratio: float = 0.8):
        """Initialize the cost budget."""
        self._max_tokens = max_tokens
        self._warn_ratio = warn_ratio
        self._used_tokens = 0
        self._call_count = 0
        self._warned = False
        self._reserved_tokens = 0
        self.strict = False
        self._strict_phase = 0
        self._requests_sent = 0
        self._phase_usage = [0, 0, 0]
        self._usage_uncertain = False
        self._halt_on_unknown_usage = False
        self._stop_reason: str | None = None

    def configure_strict(self, *, halt_on_unknown_usage: bool = False) -> None:
        """Reserve downstream capacity within the existing total allowance."""
        if self._max_tokens > 500000:
            raise ValueError("strict_total_budget_limit")
        self.strict = True
        self._halt_on_unknown_usage |= halt_on_unknown_usage

    def _phase(self) -> int:
        """Classify actual call contexts and prohibit backward budget borrowing."""
        from litagent.observability.context import get_task_id

        task = get_task_id() or ""
        if task.startswith("evaluation") or task in {
            "evidence_rewrite",
            "human_revision",
        }:
            return 2
        if (
            task.startswith("adversarial_review")
            or task.startswith("synthesis")
            or task.startswith("reviewer")
        ):
            return 1
        return 0

    def begin_request(self) -> None:
        """Count transport attempts before sending, including uncertain failures."""
        if self.strict:
            if self.stop_reason:
                raise ValueError(self.stop_reason)
            if self._requests_sent >= 64:
                raise ValueError("strict_request_budget_exceeded")
            self._requests_sent += 1

    def charge_unknown_usage(self, reservation: int) -> None:
        """Keep a conservative charge when a response omits provider accounting."""
        self._used_tokens += reservation
        self._usage_uncertain = True
        if self.strict:
            self._phase_usage[self._phase()] += reservation

    def stop_after_worker_failure(self) -> None:
        """Stop automatic execution after a technical worker failure, without refund."""
        if self._halt_on_unknown_usage:
            self._stop_reason = "strict_worker_failed"

    @property
    def stop_reason(self) -> str | None:
        """Expose a durable reason without interpreting an unknown response as zero."""
        if self._halt_on_unknown_usage and self._usage_uncertain:
            return "strict_usage_unknown"
        return self._stop_reason

    def record(self, usage: dict) -> None:
        """Add one provider usage record to the budget."""
        self._used_tokens += usage.get("prompt_tokens", 0)
        self._used_tokens += usage.get("completion_tokens", 0)
        self._call_count += 1
        if self.strict:
            self._phase_usage[self._phase()] += usage.get(
                "prompt_tokens", 0
            ) + usage.get("completion_tokens", 0)

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
            **(
                {
                    "strict_policy": "staged-budget-v1",
                    "strict_phase": self._strict_phase,
                    "requests_sent": self._requests_sent,
                    "phase_usage": list(self._phase_usage),
                    "usage_uncertain": self._usage_uncertain,
                    **(
                        {
                            "halt_on_unknown_usage": True,
                            "stop_reason": self._stop_reason,
                        }
                        if self._halt_on_unknown_usage
                        else {}
                    ),
                }
                if self.strict
                else {}
            ),
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
        if data.get("strict_policy"):
            if (
                data["strict_policy"] != "staged-budget-v1"
                or type(data.get("requests_sent")) is not int
                or not 0 <= data["requests_sent"] <= 64
                or type(data.get("strict_phase")) is not int
                or not 0 <= data["strict_phase"] <= 2
            ):
                raise ValueError("invalid_strict_budget_snapshot")
            budget.configure_strict(
                halt_on_unknown_usage=data.get("halt_on_unknown_usage") is True
            )
            budget._requests_sent = data["requests_sent"]
            budget._strict_phase = data["strict_phase"]
            phases = data.get("phase_usage")
            if (
                not isinstance(phases, list)
                or len(phases) != 3
                or any(type(v) is not int or v < 0 for v in phases)
                or sum(phases) != values["used_tokens"]
            ):
                raise ValueError("invalid_strict_budget_usage")
            budget._phase_usage = list(phases)
            budget._usage_uncertain = bool(data.get("usage_uncertain"))
            if data.get("stop_reason") not in {None, "strict_worker_failed"}:
                raise ValueError("invalid_strict_budget_stop_reason")
            budget._stop_reason = data.get("stop_reason")
        return budget

    def restore(self, data: dict) -> None:
        """Restore into the object already shared with LLM clients and scheduler."""
        restored = self.from_snapshot(data)
        if restored._max_tokens != self._max_tokens:
            raise ValueError("cost_budget_limit_mismatch")
        if self.strict != restored.strict:
            raise ValueError("strict_budget_identity_mismatch")
        self._warn_ratio = restored._warn_ratio
        self._used_tokens = restored._used_tokens
        self._call_count = restored._call_count
        self._warned = restored._warned
        self._requests_sent = restored._requests_sent
        self._strict_phase = restored._strict_phase
        self._phase_usage = list(restored._phase_usage)
        self._usage_uncertain = restored._usage_uncertain
        self._halt_on_unknown_usage = restored._halt_on_unknown_usage
        self._stop_reason = restored._stop_reason

    def is_exceeded(self) -> bool:
        """Return whether token use reached the hard limit."""
        return self._used_tokens >= self._max_tokens or self.stop_reason is not None

    def remaining(self, *, advance: bool = True) -> int:
        """Return available capacity, optionally without advancing the strict phase."""
        ceiling = self._max_tokens
        if self.stop_reason:
            return 0
        if self.strict:
            phase = self._phase()
            if phase < self._strict_phase:
                return 0
            if advance:
                self._strict_phase = phase
            ceiling = int(self._max_tokens * (0.4, 0.6, 1.0)[phase])
        return max(0, ceiling - self._used_tokens - self._reserved_tokens)

    def reserve(self, tokens: int) -> None:
        """Reserve a request upper bound before yielding to concurrent callers."""
        if self.stop_reason:
            raise ValueError(self.stop_reason)
        if tokens < 0 or tokens > self.remaining():
            raise ValueError("evaluation_run_budget_exceeded")
        self._reserved_tokens += tokens

    def release(self, tokens: int) -> None:
        """Release a completed request's reservation without changing actual usage."""
        if tokens < 0 or tokens > self._reserved_tokens:
            raise ValueError("invalid_budget_reservation")
        self._reserved_tokens -= tokens

    @property
    def used(self) -> int:
        """Return the consumed budget."""
        return self._used_tokens

    @property
    def call_count(self) -> int:
        """Return the number of recorded LLM calls."""
        return self._call_count
