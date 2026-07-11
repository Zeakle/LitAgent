"""RAGAS Faithfulness 评估器——综述内容是否忠于源论文 claims（内容幻觉检测）。

真接 ragas 官方库（可选依赖）。未装 ragas / langchain-openai → skip 降级。
"""

from __future__ import annotations
import os
from typing import Any

from litagent.eval.base import Evaluator, EvalResult, CTX_CLAIMS, CTX_PAPERS
from litagent.config import AppConfig
from litagent.logging import get_logger

logger = get_logger('eval.ragas')


class RagasFaithfulnessEvaluator(Evaluator):
    """接 ragas 官方 faithfulness。综述=response，源 claims=retrieved_contexts。

    context 取 CTX_CLAIMS（源 claims 文本列表）；缺则回退用 CTX_PAPERS 的 abstract。
    ragas / langchain-openai 未装 → skip。ragas 评估异常 → skip。永不抛。
    """

    def __init__(self, config: AppConfig, threshold: float = 0.8):
        super().__init__(threshold)
        self._cfg = config


    @property
    def metric_name(self) -> str:
        return 'faithfulness'


    def _build_contexts(self, context: dict[str, Any]) -> list[str]:
        """源支持材料：优先 claims 文本，回退 papers 的 abstract。"""
        claims = context.get(CTX_CLAIMS) or []
        texts = [c.get('text', '') if isinstance(c, dict) else str(c) for c in claims]
        texts = [t for t in texts if t]
        if texts:
            return texts
        
        papers = context.get(CTX_PAPERS) or []
        return [p.get('abstract', '') for p in papers if isinstance(p, dict) and p.get('abstract')]


    async def _score_ragas(self, survey: str, contexts: list[str], query: str) -> float:
        """跑 ragas collections API faithfulness（async 单样本打分）。返回分数。"""

        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness

        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        client = AsyncOpenAI(base_url=self._cfg.llm.base_url, api_key=api_key)
        llm = llm_factory(self._cfg.llm.model, client=client)
        scorer = Faithfulness(llm=llm)

        result = await scorer.ascore(
            user_input=query or 'literature survey',
            response=survey,
            retrieved_contexts=contexts
        )

        return float(getattr(result, 'value', result))

    
    async def evaluate(self, survey: str, context: dict[str, Any]) -> EvalResult:
        if not survey or not survey.strip():
            return EvalResult.skip(self.metric_name, 'empty_survey')

        contexts = self._build_contexts(context)
        if not contexts:
            # 无源 claims/abstract → 无法验证忠实度 → skip（前提缺失）
            return EvalResult.skip(self.metric_name, "no source claims/abstracts in context")

        query = context.get('query', "")
        try:
            score = await self._score_ragas(survey, contexts, query)
        except ImportError as e:
            logger.warning(f"ragas not installed: {e}")
            return EvalResult.skip(self.metric_name, "ragas not installed")
        except Exception as e:
            logger.warning(f"ragas eval failed: {e}")
            return EvalResult.skip(self.metric_name, f"ragas_error: {e}")

        if score != score:  # ragas返回nan，判断失败
            return EvalResult.skip(self.metric_name, 'ragas returned nan')
        return self._make_result(score, {'contexts_count': len(contexts)})