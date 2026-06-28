"""Circuit Breaker——连续失败熔断，避免对挂掉的服务无效重试。"""


from __future__ import annotations
import time
from enum import Enum
from litagent.logging import get_logger


logger = get_logger('errors.circuit_breaker')


class CircuitState(str, Enum):
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'


class CircuitBreaker:
    """单个 tool 的熔断状态机
    
    用法:
        cb = CircuitBreaker(fail_threshold=5, cooldown_seconds=60)
        if not cb.allow():
            return "circuit open"       # 快速失败
        try:
            result = await call()
            cb.record_success()
        except Exception:
            cb.record_failure()
    """

    def __init__(self, fail_threshold: int = 5, cooldown_seconds: int = 60):
        self._fail_threshold = fail_threshold
        self._cooldown = cooldown_seconds
        self._state = CircuitState.CLOSED
        self._fail_count = 0
        self._opened_at = 0.0
        self._probe_in_flight = False

    
    def allow(self) -> bool:
        """当前是否允许发起调用

        CLOSED → True
        OPEN   → 冷却未到 False；冷却到了转 HALF_OPEN 放「一个」试探
        HALF_OPEN → False（已有试探在飞，其余拒绝）

        关键：HALF_OPEN 只放一个试探请求。
        异步并发下（Scheduler max_concurrent=5），冷却到期瞬间会有多个
        execute() 同时调 allow()。若无条件返回 True，会一次放进一批，
        服务没恢复就又被打爆，失去"试探"的意义。
        用 _probe_in_flight 标志保证只放第一个。
        asyncio 单线程下 allow() 内无 await，不会被打断，标志够用，无需锁。
        """
        if self._state == CircuitState.CLOSED:
            return True

        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self._cooldown:
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                logger.debug("Circuit half-open: trying one probe")
                return True
            return False

        # HALF_OPEN: 已有试探在飞，其余请求快速失败
        return False


    def record_success(self) -> None:
        """调用成功，清零，回到CLOSED"""
        self._fail_count = 0
        self._probe_in_flight = False
        self._state = CircuitState.CLOSED

    
    def record_failure(self) -> None:
        """调用失败：累加；HALF_OPEN 失败或攒够阈值 → OPEN。"""
        self._fail_count += 1

        # HALF_OPEN 时一失败立刻回 OPEN
        if self._state == CircuitState.HALF_OPEN:
            self._trip()
            return

        if self._fail_count >= self._fail_threshold:
            self._trip()


    def _trip(self) -> None:
        """跳闸函数"""
        self._state = CircuitState.OPEN
        self._opened_at = time.monotonic()
        self._probe_in_flight = False   # 复位，下次冷却到期可再放试探
        logger.warning(
            f"Circuit OPEN after {self._fail_count} consecutive failures, "
            f"cooling down {self._cooldown}s"
        )

    
    @property
    def state(self) -> CircuitState:
        return self._state