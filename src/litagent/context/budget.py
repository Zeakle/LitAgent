"""BudgetManager——token 预算分配 + 70% 阈值 + 截断。"""


from __future__ import annotations
from abc import ABC, abstractmethod

from litagent.logging import get_logger


logger = get_logger('context.budget')


class TokenCounter(ABC):
    """Token计数器接口。"""

    @abstractmethod
    def count(self, text: str) -> int:
        ...

    def chars_per_token(self) -> int:
        return 4


class CharBasedCounter(TokenCounter):
    """字符估算：~4 字符 ≈ 1 token（英文/混合文本的经验值）。

    误差 ±15%，但 BudgetManager 是预算管理不是精确计费，足够用。
    """
    CHARS_PER_TOKEN = 4

    def count(self, text: str) -> int:
        if not text:
            return 0
        return max(1, len(text) // self.CHARS_PER_TOKEN)

    def chars_per_token(self) -> int:
        return self.CHARS_PER_TOKEN


class BudgetManager:
    """Token Budget Manager

    职责：
    1. 计数：count_tokens()
    2. 阈值检测：needs_compact()——超过 70% 时返回 True
    3. 截断：truncate()——按句子边界截断到指定 token 数
    4. 余量查询：remaining()


    Args:
        max_tokens: 总预算
        compact_threshold: compact触发阈值
        counter: TokenCounter实例
    """

    def __init__(
        self,
        max_tokens: int = 16000,
        compact_threshold: float = 0.7,
        counter: TokenCounter | None = None,
    ):
        self._max_tokens = max_tokens
        self._compact_threshold = compact_threshold
        self._counter = counter or CharBasedCounter()

    
    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    
    def count_tokens(self, text: str) -> int:
        return self._counter.count(text)

    
    def needs_compact(self, current_tokens: int) -> bool:
        """当前token是否超过阈值"""
        if self._max_tokens <= 0:
            return False
        return current_tokens / self._max_tokens > self._compact_threshold

    
    def remaining(self, used: int) -> int:
        """剩余可用token数"""
        return max(0, self._max_tokens - used)

    
    def truncate(self, text: str, max_tokens: int) -> str:
        """截断文本到 max_tokens 以内。尝试在句子边界截断。

        策略：先按字符数粗截，再向前找最近的句号/换行作为截断点。
        如果找
        """
        if self.count_tokens(text) <= max_tokens:
            return text

        char_limit = max_tokens * self._counter.chars_per_token()
        truncated = text[:char_limit]

        last_period = truncated.rfind('. ')  # 最后一个". "的index
        last_newline = truncated.rfind('\n')
        cut_point = max(last_period, last_newline)

        if cut_point > char_limit * 0.5:
            truncated = truncated[: cut_point+1]

        return truncated
