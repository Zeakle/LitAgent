"""Evaluate report faithfulness against referenced evidence with RAGAS."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from litagent.config import AppConfig
from litagent.context.templates import wrap_xml
from litagent.eval.base import (
    CTX_EVIDENCE,
    CTX_QUERY,
    CTX_REFERENCED_EVIDENCE,
    CTX_REFERENCED_EVIDENCE_IDS,
    CTX_SELECTED_EVIDENCE,
    CTX_UNRESOLVED_EVIDENCE_IDS,
    EvalResult,
    Evaluator,
)
from litagent.evidence import extract_evidence_refs_ordered
from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger
from litagent.report_render import align_claim_to_source, build_report_source_units

logger = get_logger("eval.ragas")


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


@dataclass(frozen=True)
class ResolvedFaithfulnessContext:
    """Store bounded evidence and fallback metadata for faithfulness scoring."""

    texts: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str | None


def _format_evidence_item(item: Mapping[str, Any]) -> str:
    """Format one evidence item for RAGAS evaluation."""
    return (
        f"[E:{item.get('evidence_id', '')}] "
        f"({item.get('paper_title', '')}; "
        f"{item.get('source_locator', '')}) "
        f"{item.get('text', '')}"
    )


class RagasFaithfulnessEvaluator(Evaluator):
    """Score faithfulness with RAGAS and diagnose failed evaluations."""

    def __init__(
        self,
        config: AppConfig,
        threshold: float = 0.8,
        llm: BaseLLMClient | None = None,
    ):
        """Initialize the RAGAS faithfulness evaluator."""
        super().__init__(threshold)
        self._cfg = config
        self._llm = llm

    @property
    def metric_name(self) -> str:
        """Return the evaluator metric name."""
        return "faithfulness"

    def _resolve_contexts(
        self, context: Mapping[str, Any]
    ) -> ResolvedFaithfulnessContext:
        """Resolve and bound the evidence used for faithfulness scoring."""
        referenced = context.get(CTX_REFERENCED_EVIDENCE)
        referenced_ids = context.get(CTX_REFERENCED_EVIDENCE_IDS)
        selected = context.get(CTX_SELECTED_EVIDENCE)

        report_has_refs = isinstance(referenced_ids, list) and bool(referenced_ids)
        fallback_used = False
        fallback_reason: str | None = None

        if isinstance(referenced, list) and referenced:
            source_items = referenced
        elif report_has_refs:
            source_items = []
        elif isinstance(selected, list) and selected:
            source_items = selected
            fallback_used = True
            fallback_reason = "report_has_no_evidence_refs"
        else:
            source_items = []

        max_items = self._cfg.eval.faithfulness_max_evidence_items
        max_chars = self._cfg.eval.faithfulness_max_context_chars
        texts: list[str] = []
        evidence_ids: list[str] = []
        seen: set[str] = set()
        used_chars = 0

        for raw in source_items:
            if not isinstance(raw, Mapping):
                continue
            evidence_id = raw.get("evidence_id")
            if not isinstance(evidence_id, str) or not evidence_id:
                continue
            if evidence_id in seen:
                continue
            rendered = _format_evidence_item(raw)
            if not raw.get("text"):
                continue
            if len(texts) >= max_items:
                break
            if used_chars + len(rendered) > max_chars:
                break
            seen.add(evidence_id)
            evidence_ids.append(evidence_id)
            texts.append(rendered)
            used_chars += len(rendered)

        return ResolvedFaithfulnessContext(
            texts=tuple(texts),
            evidence_ids=tuple(evidence_ids),
            fallback_used=fallback_used,
            fallback_reason=fallback_reason,
        )

    async def _diagnose(self, survey: str, context: dict[str, Any]) -> dict[str, Any]:
        """Diagnose unsupported claims against the resolved evidence ledger."""
        resolved = self._resolve_contexts(context)
        ledger = context.get(CTX_EVIDENCE)
        if not self._llm:
            return {"diagnostic_skipped": "diagnostic_llm_unavailable"}
        if not isinstance(ledger, Mapping) or not ledger:
            return {"diagnostic_skipped": "evidence_ledger_unavailable"}
        if not resolved.texts:
            return {"diagnostic_skipped": "referenced_evidence_unavailable"}

        user_msg = "\n\n".join(
            [
                wrap_xml("survey", survey),
                wrap_xml("evidence_ledger", "\n".join(resolved.texts)),
            ]
        )

        try:
            chat_kwargs: dict[str, Any] = {
                "response_format": {"type": "json_object"},
                "max_tokens": self._cfg.eval.max_tokens,
            }
            if self._cfg.eval.thinking_mode != "provider_default":
                chat_kwargs["extra_body"] = {
                    "thinking": {"type": self._cfg.eval.thinking_mode}
                }
            async with asyncio.timeout(self._cfg.eval.timeout_seconds):
                response = await self._llm.chat(
                    [
                        {"role": "system", "content": _DIAGNOSTIC_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    **chat_kwargs,
                )
            parsed = json.loads(response.content or "{}")
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return {"diagnostic_skipped": "diagnostic_timeout"}
        except Exception as exc:
            logger.warning(
                "Faithfulness diagnostic failed error_type=%s",
                type(exc).__name__,
            )
            return {
                "diagnostic_skipped": "diagnostic_error",
                "diagnostic_error_type": type(exc).__name__,
            }

        raw_claims = parsed.get("unsupported_claims")
        if not isinstance(raw_claims, list):
            return {"diagnostic_skipped": "malformed_diagnostic_output"}

        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        ledger_ids = set(ledger)

        for raw in raw_claims:
            if not isinstance(raw, Mapping):
                continue

            claim_text = str(raw.get("claim_text") or "").strip()
            if not claim_text:
                continue

            raw_ids = raw.get("evidence_ids")
            evidence_ids: list[str] = []
            if isinstance(raw_ids, list):
                for item in raw_ids:
                    if isinstance(item, str) and item:
                        evidence_ids.append(item)
            key = (claim_text, tuple(evidence_ids))
            if key in seen:
                continue
            seen.add(key)

            unknown = [item for item in evidence_ids if item not in ledger_ids]
            if unknown:
                reason_code = "unknown_evidence_ref"
            elif not evidence_ids:
                reason_code = "uncited_claim"
            else:
                reason_code = "insufficient_support"

            normalized.append(
                {
                    "claim_text": claim_text,
                    "evidence_ids": evidence_ids,
                    "reason_code": reason_code,
                    "explanation": str(
                        raw.get("reason") or raw.get("explanation") or ""
                    ),
                }
            )

        return {"unsupported_claims": normalized}

    async def _score_ragas(
        self, survey: str, contexts: list[str], query: str
    ) -> tuple[float, dict[str, Any]]:
        """Score via the same configured provider adapter used by LitAgent."""
        adapter = getattr(self._llm, "score_ragas_faithfulness", None)
        if not callable(adapter):
            raise RuntimeError("ragas_provider_adapter_unavailable")
        async with asyncio.timeout(self._cfg.eval.timeout_seconds):
            return await adapter(
                user_input=query or "literature survey",
                response=survey,
                retrieved_contexts=contexts,
                max_tokens=self._cfg.eval.max_tokens,
                thinking_mode=self._cfg.eval.thinking_mode,
            )

    async def evaluate(self, survey: str, context: dict[str, Any]) -> EvalResult:
        """Score a survey against report-referenced or selected evidence."""
        if not survey.strip():
            return EvalResult.skip(self.metric_name, "survey is empty")

        resolved = self._resolve_contexts(context)
        if not resolved.texts:
            return EvalResult.skip(
                self.metric_name,
                "no report-referenced or selected evidence in context",
            )

        query = str(context.get(CTX_QUERY) or "")
        try:
            scored = await self._score_ragas(survey, list(resolved.texts), query)
            if isinstance(scored, tuple):
                score, score_metadata = scored
            else:
                # Preserve compatibility for injected test evaluators.
                score, score_metadata = float(scored), {}
        except ImportError:
            return EvalResult.skip(self.metric_name, "ragas not installed")
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            return EvalResult.skip(self.metric_name, "ragas_timeout")
        except Exception as exc:
            error_type = type(exc).__name__
            if error_type == "IncompleteOutputException":
                reason = "ragas_incomplete_output"
            elif isinstance(exc, (json.JSONDecodeError, TypeError, ValueError)):
                reason = "ragas_malformed_output"
            elif str(exc) == "ragas_provider_adapter_unavailable":
                reason = "ragas_provider_adapter_unavailable"
            else:
                reason = f"ragas_provider_error:{error_type}"
            logger.warning("ragas eval failed error_type=%s", error_type)
            return EvalResult.skip(self.metric_name, reason)

        if score != score:
            return EvalResult.skip(self.metric_name, "ragas returned nan")

        details = {
            "contexts_count": len(resolved.texts),
            "evidence_ids": list(resolved.evidence_ids),
            "fallback_used": resolved.fallback_used,
            "fallback_reason": resolved.fallback_reason,
            "provider": score_metadata,
            "unresolved_evidence_ids": list(
                context.get(CTX_UNRESOLVED_EVIDENCE_IDS) or []
            ),
        }
        result = self._make_result(score, details)
        verdicts = score_metadata.get("verdicts")
        if isinstance(verdicts, list):
            result.details["factual_claim_count"] = len(verdicts)
            result.details["unsupported_claims"] = self._diagnostics_from_verdicts(
                survey, verdicts, set(resolved.evidence_ids)
            )
        elif not result.passed:
            result.details["diagnostic_skipped"] = "legacy_score_without_verdicts"
        return result

    @staticmethod
    def _diagnostics_from_verdicts(
        survey: str, verdicts: list[dict[str, Any]], known_ids: set[str]
    ) -> list[dict[str, Any]]:
        """Preserve every negative verdict without guessing original citations."""
        units = build_report_source_units(survey)
        diagnostics = []
        for item in verdicts:
            if item.get("verdict") != 0:
                continue
            claim = str(item.get("statement") or "").strip()
            alignment = align_claim_to_source(claim, units)
            diagnostics.append(
                {
                    "claim_text": claim,
                    "statement_id": item.get("statement_id"),
                    "evidence_ids": [
                        ref for ref in alignment.evidence_ids if ref in known_ids
                    ],
                    "reason_code": "insufficient_support",
                    "explanation": str(item.get("reason") or ""),
                    "source_quote": alignment.source_quote,
                    "source_unit_id": alignment.source_unit_id,
                    "source_start": alignment.start,
                    "source_end": alignment.end,
                    "citation_scope": alignment.citation_scope,
                    "alignment_status": alignment.status,
                    "diagnostic_source": "ragas_verdict",
                }
            )
        return diagnostics
