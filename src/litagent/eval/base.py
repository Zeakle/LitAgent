"""Define shared evaluation contracts and context keys."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

CTX_PAPERS = "papers"
CTX_CLAIMS = "claims"
CTX_EVIDENCE = "evidence"
CTX_QUERY = "query"
CTX_REFERENCED_EVIDENCE = "referenced_evidence"
CTX_REFERENCED_EVIDENCE_IDS = "referenced_evidence_ids"
CTX_UNRESOLVED_EVIDENCE_IDS = "unresolved_evidence_ids"
CTX_SELECTED_EVIDENCE = "selected_evidence"


@dataclass
class EvalResult:
    """Store a normalized evaluation result."""

    metric: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    @classmethod
    def skip(cls, metric: str, reason: str) -> "EvalResult":
        """Create a skipped evaluation result."""
        return cls(
            metric=metric,
            score=0.0,
            passed=True,
            details={"skipped_reason": reason},
            skipped=True,
        )


class Evaluator(ABC):
    """Define the evaluator interface and threshold handling."""

    def __init__(self, threshold: float = 0.8):
        self._threshold = threshold

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the evaluator metric name."""
        ...

    @abstractmethod
    async def evaluate(self, survey: str, context: dict[str, Any]) -> EvalResult:
        """Score a survey using the supplied evaluation context."""
        ...

    def _make_result(self, score: float, details: dict | None = None) -> EvalResult:
        return EvalResult(
            metric=self.metric_name,
            score=score,
            passed=score >= self._threshold,
            details=details or {},
        )
