"""Evaluate whether cited papers occur in the source set."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from litagent.context.templates import wrap_xml
from litagent.eval.base import CTX_EVIDENCE, CTX_PAPERS, EvalResult, Evaluator
from litagent.evidence import (
    extract_evidence_refs_ordered,
    find_malformed_evidence_refs,
)
from litagent.llm.client import BaseLLMClient
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
    """Measure citation accuracy with an LLM citation audit."""

    def __init__(
        self,
        llm: BaseLLMClient,
        threshold: float = 0.8,
        max_tokens: int = 16384,
        thinking_mode: str = "provider_default",
        timeout_seconds: float = 180.0,
    ):
        """Initialize the citation evaluator."""
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        super().__init__(threshold)
        self._llm = llm
        self._max_tokens = max_tokens
        self._thinking_mode = thinking_mode
        self._timeout_seconds = timeout_seconds

    @property
    def metric_name(self) -> str:
        """Return the evaluator metric name."""
        return "citation_accuracy"

    async def evaluate(self, survey: str, context: dict[str, Any]) -> EvalResult:
        """Score cited papers against titles in the evaluation context."""
        ledger = context.get(CTX_EVIDENCE)
        if isinstance(ledger, dict) and ledger:
            refs = extract_evidence_refs_ordered(survey)
            malformed = find_malformed_evidence_refs(survey)
            unknown = [evidence_id for evidence_id in refs if evidence_id not in ledger]
            total = len(refs) + len(malformed)
            real = len(refs) - len(unknown)
            score = real / total if total else 1.0
            result = self._make_result(
                score,
                {
                    "mode": "evidence_ledger",
                    "cited_count": total,
                    "real_count": real,
                    "unknown_evidence_ids": unknown,
                    "malformed_evidence_refs": malformed,
                    "note": "no evidence references found" if not total else "",
                },
            )
            if malformed:
                result.passed = False
                result.details["reason_codes"] = ["malformed_evidence_refs"]
            return result

        papers = context.get(CTX_PAPERS, []) or []
        source_titles = [p.get("title", "") for p in papers if p.get("title")]
        if not source_titles:
            return EvalResult.skip(self.metric_name, "no source papers in context")

        user_msg = "\n\n".join(
            [
                wrap_xml("survey", survey),
                wrap_xml("source_papers", "\n".join(f"- {t}" for t in source_titles)),
            ]
        )
        cited: list[Any] | None = None
        last_error_type = "UnknownError"
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        attempts = 0
        for attempts in range(1, 3):
            try:
                kwargs: dict[str, Any] = {
                    "response_format": {"type": "json_object"},
                    "max_tokens": self._max_tokens,
                }
                if self._thinking_mode != "provider_default":
                    kwargs["extra_body"] = {"thinking": {"type": self._thinking_mode}}
                async with asyncio.timeout(self._timeout_seconds):
                    resp = await self._llm.chat(
                        [
                            {"role": "system", "content": _CITATION_PROMPT},
                            {"role": "user", "content": user_msg},
                        ],
                        **kwargs,
                    )
                usage["prompt_tokens"] += int(resp.usage.get("prompt_tokens") or 0)
                usage["completion_tokens"] += int(
                    resp.usage.get("completion_tokens") or 0
                )
                if not (resp.content or "").strip():
                    raise ValueError("empty_response")
                parsed = json.loads(resp.content)
                raw_cited = parsed.get("cited") if isinstance(parsed, dict) else None
                if not isinstance(raw_cited, list):
                    raise ValueError("malformed_cited")
                cited = raw_cited
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_error_type = type(exc).__name__
                logger.warning(
                    "Citation legacy audit failed attempt=%d error_type=%s",
                    attempts,
                    last_error_type,
                )

        if cited is None:
            result = EvalResult.skip(
                self.metric_name,
                f"legacy_citation_audit_failed:{last_error_type}",
            )
            result.details.update({"attempts": attempts, "usage": usage})
            return result

        total = len(cited)
        real = sum(
            1 for c in cited if isinstance(c, dict) and c.get("in_source") is True
        )
        fabricated = [
            c.get("title", "") if isinstance(c, dict) else str(c)
            for c in cited
            if not (isinstance(c, dict) and c.get("in_source") is True)
        ]

        score = real / total if total else 1.0
        malformed = find_malformed_evidence_refs(survey)
        result = self._make_result(
            score,
            {
                "mode": "legacy_llm",
                "cited_count": total,
                "real_count": real,
                "fabricated": fabricated,
                "attempts": attempts,
                "usage": usage,
                "malformed_evidence_refs": malformed,
                "note": "no citations found" if not total else "",
            },
        )
        if malformed:
            result.passed = False
            result.details["reason_codes"] = ["malformed_evidence_refs"]
        return result
