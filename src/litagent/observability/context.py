from contextvars import ContextVar

# 异步全局上下文
_current_task_id: ContextVar[str] = ContextVar('current_task_id', default='')


def set_task_id(task_id: str):
    """Scheduler 在 dispatch 时调，返回 token 供 reset。"""
    return _current_task_id.set(task_id)


def get_task_id() -> str:
    """底层组件 emit 时调，拿当前 Worker 的 task_id。"""
    return _current_task_id.get()


def reset_task_id(token) -> None:
    _current_task_id.reset(token)