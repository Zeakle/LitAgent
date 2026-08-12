"""Implement a cooldown-based circuit-breaker state machine."""

from __future__ import annotations

import time
from enum import Enum

from litagent.logging import get_logger

logger = get_logger("errors.circuit_breaker")


class CircuitState(str, Enum):
    """Define closed, open, and half-open circuit-breaker states."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Block calls after repeated failures and allow one cooldown probe."""

    def __init__(self, fail_threshold: int = 5, cooldown_seconds: int = 60):
        """Initialize the circuit breaker."""
        self._fail_threshold = fail_threshold
        self._cooldown = cooldown_seconds
        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    def allow(self) -> bool:
        """Return whether the next call may proceed."""
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self._cooldown:
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                logger.debug("Circuit half-open: trying one probe")
                return True
            return False

        return False

    def record_success(self) -> None:
        """Close the circuit and clear consecutive failures."""
        self._fail_count = 0
        self._probe_in_flight = False
        self._state = CircuitState.CLOSED

    def record_failure(self) -> None:
        """Record a failure and open the circuit when required."""
        self._fail_count += 1

        if self._state == CircuitState.HALF_OPEN:
            self._trip()
            return

        if self._fail_count >= self._fail_threshold:
            self._trip()

    def _trip(self) -> None:
        """Open the circuit and start its cooldown."""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._probe_in_flight = False
        logger.warning(
            f"Circuit OPEN after {self._fail_count} consecutive failures, "
            f"cooling down {self._cooldown}s"
        )

    @property
    def state(self) -> CircuitState:
        """Return the current circuit state."""
        return self._state
