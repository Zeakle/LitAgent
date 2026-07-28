"""Propagate the active worker task ID through asynchronous call context."""

from contextvars import ContextVar

_current_task_id: ContextVar[str] = ContextVar("current_task_id", default="")


def set_task_id(task_id: str):
    """Set the active task ID and return its reset token."""
    return _current_task_id.set(task_id)


def get_task_id() -> str:
    """Return the active task ID."""
    return _current_task_id.get()


def reset_task_id(token) -> None:
    """Restore the previous task identifier."""
    _current_task_id.reset(token)
