"""Citation Accuracy 评估器——检测引用幻觉（综述引用源列表外的论文）。"""

from __future__ import annotations
import json
from typing import Any

from litagent.eval.base import Evaluator, EvalResult, CTX_PAPERS
from litagent.llm.client import BaseLLMClient
from litagent.context.templates import wrap_xml
from litagent.logging import get_logger


logger = get_logger("eval.citation")


_CITATION_PROMPT = """You are a citation auditor. Given a survey and the list of \
SOURCE papers it was written from, identify every paper the survey cites/mentions, \
and judge whether each one appears in the source list (real) or is fabricated \
(not in the source list).

Match by paper title semantically (ignore formatting, minor wording differences).

Return ONLY a JSON object:
{
  "cited": [
    {"title": "<paper title as mentioned in survey>", "in_source": true/false}
  ]
}
If the survey mentions no specific papers, return {"cited": []}."""


class CitationEvaluator(Evaluator):
    """LLM 判定综述引用的真实率（真实引用数 / 总引用数）。

    context 需要 CTX_PAPERS：源论文列表（每项含 title）。
    LLM 挂 / JSON 解析失败 → skip（中性降级，不抛异常）。
    综述未引用任何论文 → score=1.0（无幻觉可言）。
    """

    def __init__(self, llm: BaseLLMClient, threshold: float = 0.8, max_tokens: int = 16384):
        super().__init__(threshold)
        self._llm = llm
        self._max_tokens = max_tokens

    @property
    def metric_name(self) -> str:
        return "citation_accuracy"

    async def evaluate(self, survey: str, context: dict[str, Any]) -> EvalResult:
        papers = context.get(CTX_PAPERS, []) or []
        source_titles = [p.get("title", "") for p in papers if p.get("title")]
        if not source_titles:
            # 无源论文列表 → 无法判定 → skip（不是评估失败，是前提缺失）
            return EvalResult.skip(self.metric_name, "no source papers in context")

        user_msg = "\n\n".join([
            wrap_xml("survey", survey),
            wrap_xml("source_papers", "\n".join(f"- {t}" for t in source_titles)),
        ])
        try:
            resp = await self._llm.chat(
                [{"role": "system", "content": _CITATION_PROMPT},
                 {"role": "user", "content": user_msg}],
                response_format={"type": "json_object"},
                max_tokens=self._max_tokens,   # reasoning 模型：拆引用的 reasoning 长，需大额度免 content 被挤空
            )
            cited = json.loads(resp.content).get("cited")
        except Exception as e:
            logger.warning(f"Citation eval LLM/parse failed: {e}")
            return EvalResult.skip(self.metric_name, f"llm_or_parse_error: {e}")

        if not isinstance(cited, list):
            return EvalResult.skip(self.metric_name, "malformed 'cited' (not a list)")
        
        total = len(cited)
        if total == 0:
            # 综述没引用任何具体论文 → 无幻觉 → 满分（details 标注）
            return self._make_result(1.0, {'cited_count': 0, 'note': 'no citations found'})
        
        real = sum(1 for c in cited if isinstance(c, dict) and c.get('in_source') is True)
        fabricated = [
            c.get('title', '') if isinstance(c, dict) else str(c)
            for c in cited
            if not (isinstance(c, dict) and c.get('in_source') is True)
        ]

        score = real / total
        return self._make_result(score, {
            'cited_count': total,
            'real_count': real,
            'fabricated': fabricated
        })