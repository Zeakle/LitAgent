"""RAGAS Faithfulness 评估器——综述内容是否忠于源论文 claims（内容幻觉检测）。

真接 ragas 官方库（可选依赖）。未装 ragas / langchain-openai → skip 降级。
"""

from __future__ import annotations
import os
import json
from typing import Any

from litagent.eval.base import Evaluator, EvalResult, CTX_CLAIMS, CTX_PAPERS, CTX_EVIDENCE
from litagent.context.templates import wrap_xml
from litagent.llm.client import BaseLLMClient
from litagent.config import AppConfig
from litagent.logging import get_logger

logger = get_logger('eval.ragas')


_DIAGNOSTIC_PROMPT = """You are a faithfulness auditor. Given a survey and an evidence \
ledger (each line "[E:<id>] (<paper title>) <text>"), list every factual claim in the \
survey that is NOT supported by any ledger item.

Return ONLY a JSON object:
{"unsupported_claims": [
  {"claim_text": "<claim as written in the survey>",
   "evidence_ids": ["<related ledger ids the claim overstates, [] if none>"],
   "reason": "<one-sentence why it is unsupported>"}
]}
If every claim is supported, return {"unsupported_claims": []}."""


class RagasFaithfulnessEvaluator(Evaluator):
    """接 ragas 官方 faithfulness。综述=response，源 claims=retrieved_contexts。

    context 取 CTX_CLAIMS（源 claims 文本列表）；缺则回退用 CTX_PAPERS 的 abstract。
    ragas / langchain-openai 未装 → skip。ragas 评估异常 → skip。永不抛。
    """

    def __init__(self, config: AppConfig, threshold: float = 0.8,
                 llm: BaseLLMClient | None = None):
        super().__init__(threshold)
        self._cfg = config
        self._llm = llm


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

    
    async def _diagnose(self, survey: str, context: dict[str, Any]) -> dict[str, Any]:
        """score 未过阈值时定位 unsupported claims。

        契约：诊断成功 → {'unsupported_claims': [...]}（可为 []=真没找到）；
        无法诊断 → {'diagnostic_skipped': <原因>}，绝不用空列表掩盖「无法评估」。
        """
        ledger = context.get(CTX_EVIDENCE) or {}
        if not self._llm or not ledger:
            return {'diagnostic_skipped': 'no llm or evidence ledger'}

        from litagent.evidence import format_ledger
        user_msg = '\n\n'.join([
            wrap_xml('survey', survey),
            wrap_xml('evidence_ledger', format_ledger(ledger, max_chars=20000))
        ])

        try:
            resp = await self._llm.chat(
                [
                    {'role': 'system', 'content': _DIAGNOSTIC_PROMPT},
                    {'role': 'user', 'content': user_msg}
                ],
                response_format={'type': 'json_object'},
                max_tokens=self._cfg.eval.max_tokens
            )
            raw = json.loads(resp.content).get('unsupported_claims')
        except Exception as e:
            logger.warning(f"Faithfulness diagnostic failed: {e}")
            return {'diagnostic_skipped': f'diagnostic_error: {e}'}

        if not isinstance(raw, list):
            return {'diagnostic_skipped': "malformed 'unsupported_claims' (not a list)"}

        cleaned = []
        for c in raw:
            if not isinstance(c, dict) or not c.get('claim_text'):
                continue
            eids = c.get('evidence_ids')
            cleaned.append({
                'claim_text': c.get('claim_text', ''),
                'evidence_ids': eids if isinstance(eids, list) else [],
                'reason': c.get('reason', ''),
            })
        return {'unsupported_claims': cleaned}


    async def _score_ragas(self, survey: str, contexts: list[str], query: str) -> float:
        """跑 ragas collections API faithfulness（async 单样本打分）。返回分数。"""

        from openai import AsyncOpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import Faithfulness

        api_key = os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        client = AsyncOpenAI(base_url=self._cfg.llm.base_url, api_key=api_key)
        # max_tokens 从 eval config 取：reasoning 模型下 ragas 的拆 claim + 验证 reasoning 长，
        # 默认额度易被截断。经 llm_factory 的 **kwargs 尽力透传；若该版本不透传，ragas 仍会因截断 skip（降级保证不崩）。
        llm = llm_factory(self._cfg.llm.model, client=client, max_tokens=self._cfg.eval.max_tokens)
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

        result = self._make_result(score, {'contexts_count': len(contexts)})

        if not result.passed:
            result.details.update(await self._diagnose(survey, context))
        
        return result