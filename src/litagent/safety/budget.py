"""Cost Budget——综述任务级别的 token 预算追踪。"""


from __future__ import annotations
from litagent.logging import get_logger


logger = get_logger('safety.budget')


class CostBudget:
    """累计 LLM token消耗,超预算后拦截后续调用

        用法（Scheduler 持有一个实例）：
        budget = CostBudget(max_tokens=500_000)
        # 每次 LLM 调用后
        budget.record(resp.usage)
        # 派发新任务前
        if budget.is_exceeded():
            # 停止派发，走部分综述
    """

    def __init__(self, max_tokens: int = 500_000, warn_ratio: float = 0.8):
        self._max_tokens = max_tokens
        self._warn_ratio = warn_ratio
        self._used_tokens = 0
        self._call_count = 0
        self._warned = False

    
    def record(self, usage: dict) -> None:
        """usage count

        Args:
            usage: LLMResponse.usage，含 prompt_tokens / completion_tokens
        """
        self._used_tokens += usage.get("prompt_tokens", 0)
        self._used_tokens += usage.get('completion_tokens', 0)
        self._call_count += 1

        # 接近上限时警告一次
        if not self._warned and self._used_tokens >= self._max_tokens * self._warn_ratio:
            self._warned = True
            logger.warning(
                f"Cost budget at {self._used_tokens}/{self._max_tokens} tokens "
                f"({self._used_tokens / self._max_tokens:.0%})"
            )

    
    def is_exceeded(self) -> bool:
        """是否超预算"""
        return self._used_tokens >= self._max_tokens


    def remaining(self) -> int:
        """剩余可用"""
        return max(0, self._max_tokens - self._used_tokens)


    @property
    def used(self) -> int:
        return self._used_tokens

    
    @property
    def call_count(self) -> int:
        return self._call_count