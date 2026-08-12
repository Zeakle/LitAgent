"""Evaluate internal contradictions in survey claims."""

from __future__ import annotations

import json
from typing import Any

from litagent.context.templates import wrap_xml
from litagent.eval.base import EvalResult, Evaluator
from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger

logger = get_logger("eval.consistency")


_CONSISTENCY_PROMPT = """You are a consistency checker for academic surveys. \
Find pairs of QUANTITATIVE claims in the survey that CONTRADICT each other — \
e.g. the same model/dataset reported with different accuracy numbers, or \
conflicting statements about which method is best on the same benchmark.

Only flag genuine contradictions about the SAME entity/metric/setting. \
Different models or different datasets are NOT contradictions.

Return ONLY a JSON object:
{
  "conflicts": [
    {"claim_a": "<statement 1>", "claim_b": "<statement 2>", \
"reason": "<why they conflict>"}
  ]
}
If no contradictions, return {"conflicts": []}."""


class ConsistencyEvaluator(Evaluator):
    """Penalize contradictory quantitative claims identified by an LLM."""

    def __init__(
        self,
        llm: BaseLLMClient,
        threshold: float = 0.8,
        penalty: float = 0.2,
        max_tokens: int = 16384,
    ):
        """Initialize the consistency evaluator."""
        super().__init__(threshold)
        self._llm = llm
        self._penalty = penalty
        self._max_tokens = max_tokens

    @property
    def metric_name(self) -> str:
        """Return the evaluator metric name."""
        return "internal_consistency"

    async def evaluate(self, survey: str, context: dict[str, Any]) -> EvalResult:
        """Score internal consistency from detected quantitative conflicts."""
        if not survey or not survey.strip():
            return EvalResult.skip(self.metric_name, "empty_survey")

        try:
            resp = await self._llm.chat(
                [
                    {"role": "system", "content": _CONSISTENCY_PROMPT},
                    {"role": "user", "content": wrap_xml("survey", survey)},
                ],
                response_format={"type": "json_object"},
                max_tokens=self._max_tokens,
            )
            conflicts = json.loads(resp.content).get("conflicts")
        except Exception as e:
            logger.warning(f"Consistency eval LLM/parse failed: {e}")
            return EvalResult.skip(self.metric_name, f"llm_or_parse_error: {e}")

        if not isinstance(conflicts, list):
            return EvalResult.skip(
                self.metric_name, "malformed 'conflicts' (not a list)"
            )

        n = len(conflicts)
        score = max(0.0, 1.0 - n * self._penalty)

        return self._make_result(score, {"conflict_count": n, "conflicts": conflicts})
