"""Propagate the active worker task ID through asynchronous call context."""

from contextlib import asynccontextmanager
from contextvars import ContextVar
from time import monotonic

_current_task_id: ContextVar[str] = ContextVar("current_task_id", default="")
_deadline: ContextVar[float | None] = ContextVar("task_deadline", default=None)


@asynccontextmanager
async def deadline_scope(timeout_seconds: float):
    """Inherit the earliest monotonic deadline and restore it on every exit path."""
    parent = _deadline.get()
    deadline = monotonic() + timeout_seconds
    token = _deadline.set(min(parent, deadline) if parent is not None else deadline)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_deadline_seconds() -> float | None:
    """Return remaining parent capacity without extending an inherited deadline."""
    deadline = _deadline.get()
    return None if deadline is None else max(0.0, deadline - monotonic())


def set_task_id(task_id: str):
    """Set the active task ID and return its reset token."""
    return _current_task_id.set(task_id)


def get_task_id() -> str:
    """Return the active task ID."""
    return _current_task_id.get()


def reset_task_id(token) -> None:
    """Restore the previous task identifier."""
    _current_task_id.reset(token)
