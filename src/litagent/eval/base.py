"""评估模块基础契约：EvalResult 数据结构 + Evaluator 基类。"""


from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# context dict 的规范键——pipeline 生产、各评估器消费，跨模块约定必须对齐。
# 用常量而非裸字符串，防止生产端/消费端拼写漂移（见 SUGGESTION：emit/consume 字段名对齐）。
CTX_PAPERS = "papers"     # 源论文列表（citation 验存在性、ragas 验忠实度）
CTX_CLAIMS = "claims"     # 源 claims 列表（ragas 可选用）
CTX_EVIDENCE = 'evidence'


@dataclass
class EvalResult:
    """单个评估指标的结果。三个评估器（citation/consistency/ragas）统一产出此结构。

    Attributes:
        metric: 指标名，如 "citation_accuracy" / "faithfulness"
        score: 0-1 归一化分数
        passed: 是否达阈值（评估器内部 score>=threshold 算出）
        details: 评估器特有细节（假引用列表、矛盾对等），供 Report 附录和 debug
        skipped: True=评估未运行（依赖缺失/API 挂），中性，不参与 CI 判定
    """
    metric: str
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    skipped: bool = False

    @classmethod
    def skip(cls, metric: str, reason: str) -> "EvalResult":
        return cls(metric=metric, score=0.0, passed=True, details={"skipped_reason": reason}, skipped=True)


class Evaluator(ABC):
    """评估器基类。子类实现 evaluate()，持有自己的阈值。

    统一异步接口——citation 要调 Semantic Scholar API、ragas/consistency 要调 LLM，
    都是 I/O，async 一致。
    """

    def __init__(self, threshold: float = 0.8):
        self._threshold = threshold

    
    @property
    @abstractmethod
    def metric_name(self) -> str:
        ...

    
    @abstractmethod
    async def evaluate(self, survey: str, context: dict[str, Any]) -> EvalResult:
        """评估一篇综述。

        Args:
            survey: 综述全文（final draft）
            context: 评估所需上下文，各评估器取自己要的键（键名用模块常量 CTX_*）：
                     - citation: context[CTX_PAPERS]（源论文列表，验引用存在性）
                     - consistency: 无需额外（只看 survey 内部）
                     - ragas: context[CTX_PAPERS] 或 context[CTX_CLAIMS]（源 claims 验忠实度）

        Returns:
            EvalResult。失败/依赖缺失时返回 EvalResult.skip(...)，不抛异常。
        """
        ...


    def _make_result(self, score: float, details: dict | None = None) -> EvalResult:
        return EvalResult(
            metric=self.metric_name,
            score=score,
            passed=score >= self._threshold,
            details=details or {},
        )