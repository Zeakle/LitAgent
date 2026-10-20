"""Promote only final, locator-backed evidence into cross-run Claims."""

from __future__ import annotations

import asyncio
import hashlib
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from litagent.evidence import extract_evidence_refs
from litagent.quality import build_fact_cards
from litagent.rag.claims_index import ClaimsIndex, TrustedClaim

_CLAIM_NAMESPACE = uuid.UUID("a9506bd8-35d8-59e9-8180-2f0f018c99e5")


@dataclass(frozen=True)
class ClaimPromotionSummary:
    """Summarize trusted-claim promotion for one completed run."""

    status: str
    attempted_count: int
    promoted_count: int
    rejected_count: int
    reason_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the claim-promotion summary."""
        return asdict(self)


def is_publishable_report(report_data: Mapping[str, Any]) -> bool:
    """Apply the single fail-closed trust policy used by all long-term writes."""
    quality = report_data.get("quality")
    delivery = report_data.get("delivery")
    metadata = report_data.get("metadata")
    execution: Mapping[str, Any] = {}
    if isinstance(metadata, Mapping):
        raw_execution = metadata.get("execution")
        # Execution metadata participates in the trust decision. A malformed
        # value is untrusted instead of being silently ignored.
        if raw_execution is not None and not isinstance(raw_execution, Mapping):
            return False
        if isinstance(raw_execution, Mapping):
            execution = raw_execution
    partial = report_data.get("partial")
    if partial is None:
        partial = execution.get("partial")
    return (
        partial is False
        and execution.get("partial") in (None, False)
        and isinstance(quality, Mapping)
        and quality.get("status") == "passed"
        and isinstance(delivery, Mapping)
        and delivery.get("status") == "ready"
        and delivery.get("publishable") is True
    )


def _to_trusted_claim(
    *,
    run_id: str,
    domain: str,
    evidence: Mapping[str, Any],
    quality_status: str,
    delivery_status: str,
) -> TrustedClaim | None:
    """Convert eligible evidence into a trusted claim."""
    required = (
        "evidence_id",
        "paper_id",
        "claim_text",
        "supporting_text",
        "chunk_key",
        "section",
        "content_scope",
        "content_hash",
    )
    if evidence.get("evidence_version") != "v2" or any(
        not evidence.get(k) for k in required
    ):
        return None
    stable = "|".join(
        (
            run_id,
            str(evidence["evidence_id"]),
            str(evidence["claim_text"]).strip().casefold(),
        )
    )
    try:
        return TrustedClaim(
            claim_id=str(uuid.uuid5(_CLAIM_NAMESPACE, stable)),
            text=str(evidence["claim_text"]),
            supporting_text=str(evidence["supporting_text"]),
            run_id=run_id,
            domain=domain,
            paper_id=str(evidence["paper_id"]),
            evidence_id=str(evidence["evidence_id"]),
            chunk_key=str(evidence["chunk_key"]),
            section=str(evidence["section"]),
            page=evidence.get("page"),
            block_index=evidence.get("block_index"),
            bbox=evidence.get("bbox"),
            source_spans=evidence.get("source_spans") or [],
            content_scope=str(evidence["content_scope"]),
            content_hash=str(evidence["content_hash"]),
            evidence_version="v2",
            quality_status=quality_status,
            delivery_status=delivery_status,
            confidence=evidence.get("confidence"),
        )
    except (TypeError, ValueError):
        # One malformed item is rejected; it cannot fail the already-complete Survey.
        return None


def _verified_report_items(
    report_data: Mapping[str, Any], raw_items: list
) -> list[dict]:
    """Promote assessed final statements, never unverified extractor candidates."""
    survey = report_data.get("survey")
    evaluation = report_data.get("evaluation")
    if not isinstance(survey, str) or not isinstance(evaluation, Mapping):
        return []
    metric = evaluation.get("faithfulness")
    details = metric.get("details") if isinstance(metric, Mapping) else None
    if not isinstance(details, Mapping):
        return []
    coverage = details.get("coverage") or {}
    stages = details.get("stages") or {}
    assertions = details.get("assertions")
    card_map = details.get("fact_card_map")
    if (
        details.get("evaluation_version") != "survey-evaluation-v3"
        or details.get("source_report_sha256")
        != hashlib.sha256(survey.encode()).hexdigest()
        or not isinstance(coverage, Mapping)
        or coverage.get("complete") is not True
        or type(coverage.get("source_units")) is not int
        or coverage["source_units"] <= 0
        or coverage.get("processed_units") != coverage["source_units"]
        or not isinstance(stages, Mapping)
        or stages.get("global_check") != "completed"
        or not isinstance(assertions, list)
        or not assertions
        or stages.get("expected_assertions") != len(assertions)
        or stages.get("processed_assertions") != len(assertions)
        or not isinstance(card_map, Mapping)
        or not card_map
    ):
        return []
    ledger = {}
    for item in raw_items:
        eid = item.get("evidence_id")
        if not isinstance(eid, str) or not eid:
            continue
        if eid in ledger and ledger[eid] != item:
            return []
        ledger[eid] = item
    cards = build_fact_cards(ledger)
    # Compare the same provenance projection persisted by Evaluation, including ID.
    provenance_keys = (
        "evidence_id",
        "paper_id",
        "content_sha256",
        "source_locator",
        "source_version",
        "chunk_key",
        "page",
        "block_index",
        "content_scope",
        "missing_fields",
    )
    evaluated_eids = set()
    for cid, saved in card_map.items():
        card = cards.get(cid)
        if (
            not card
            or not isinstance(saved, Mapping)
            or saved != {k: card.get(k) for k in provenance_keys}
        ):
            return []
        evaluated_eids.add(card["evidence_id"])
    selected = []
    seen_ids = set()
    for assertion in assertions:
        if not isinstance(assertion, Mapping):
            return []
        cid = assertion.get("claim_id")
        if not isinstance(cid, str) or cid in seen_ids:
            return []
        seen_ids.add(cid)
        if assertion.get("kind") != "scientific":
            continue
        # A corrupted scientific result invalidates the complete promotion set.
        if (
            assertion.get("status") != "supported"
            or assertion.get("correct_kind") != "scientific"
            or assertion.get("atomic") is not True
            or assertion.get("classification_valid") is not True
            or assertion.get("source_faithful") is not True
            or assertion.get("local_error")
        ):
            return []
        start, end = assertion.get("source_start"), assertion.get("source_end")
        quote = assertion.get("source_quote")
        eids = assertion.get("evidence_ids")
        text = assertion.get("claim_text")
        validation = assertion.get("support_validation")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(survey)
            or survey[start:end] != quote
            or not isinstance(eids, list)
            or not eids
            or any(not isinstance(eid, str) for eid in eids)
            or len(set(eids)) != len(eids)
            or not set(eids).issubset(evaluated_eids)
            or extract_evidence_refs(quote) != set(eids)
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(validation, Mapping)
            or validation.get("status") != "passed"
        ):
            return []
        spans = validation.get("spans")
        if not isinstance(spans, list) or not spans:
            return []
        for span in spans:
            if not isinstance(span, Mapping) or span.get("evidence_id") not in eids:
                return []
            source = ledger[span["evidence_id"]].get("supporting_text")
            a, b = span.get("source_start"), span.get("source_end")
            if (
                span.get("source") != "original_support"
                or not isinstance(source, str)
                or type(a) is not int
                or type(b) is not int
                or not 0 <= a < b <= len(source)
                or source[a:b] != span.get("quote")
            ):
                return []
        sources = {
            (
                ledger[eid].get("paper_id"),
                ledger[eid].get("chunk_key"),
                ledger[eid].get("supporting_text"),
                ledger[eid].get("content_hash"),
            )
            for eid in eids
        }
        # TrustedClaim has a single source; retain cross-source comparisons locally.
        if len(sources) != 1:
            continue
        text = re.sub(r"\[E:[^\[\]\s;,]+\]", "", text).strip()
        if text:
            selected.append({**ledger[eids[0]], "claim_text": text, "confidence": None})
    return selected


class ClaimsPromoter:
    """Promote evidence-backed claims from publishable survey runs."""

    def __init__(self, index: ClaimsIndex) -> None:
        """Initialize the claims promoter."""
        self._index = index

    async def promote(
        self,
        *,
        run_id: str,
        domain: str,
        report_data: Mapping[str, Any],
        extractions: Sequence[Mapping[str, Any]],
    ) -> ClaimPromotionSummary:
        """Promote eligible claims into the trusted index."""
        if not is_publishable_report(report_data):
            return ClaimPromotionSummary("skipped", 0, 0, 0, "run_not_publishable")
        quality_status = str(report_data["quality"].get("status") or "")
        delivery_status = str(report_data["delivery"].get("status") or "")
        raw_items = [
            item
            for extraction in extractions
            for item in extraction.get("evidence_items", [])
            if isinstance(item, Mapping) and item.get("claim_text")
        ]
        grounded = any(
            extraction.get("claim_contract") == "host-grounded-claims-v1"
            for extraction in extractions
        )
        # New evaluated reports cannot fall back to the historical candidate path.
        evaluation = report_data.get("evaluation")
        metric = (
            evaluation.get("faithfulness") if isinstance(evaluation, Mapping) else None
        )
        details = metric.get("details") if isinstance(metric, Mapping) else None
        grounded = grounded or (
            isinstance(details, Mapping)
            and details.get("evaluation_version") == "survey-evaluation-v3"
        )
        metadata = report_data.get("metadata")
        grounded = grounded or (
            isinstance(metadata, Mapping)
            and metadata.get("quality_policy")
            in {"strict_v1", "strict_auto_v1", "strict_auto_v2"}
        )
        try:
            eligible = (
                _verified_report_items(report_data, raw_items)
                if grounded
                else raw_items
            )
        except (TypeError, ValueError, KeyError):
            # Invalid persisted proof must never reopen the legacy promotion path.
            eligible = []
        claims = [
            claim
            for item in eligible
            if (
                claim := _to_trusted_claim(
                    run_id=run_id,
                    domain=domain,
                    evidence=item,
                    quality_status=quality_status,
                    delivery_status=delivery_status,
                )
            )
            is not None
        ]
        claims = list({claim.claim_id: claim for claim in claims}.values())
        attempted = len(eligible) if grounded and eligible else len(raw_items)
        if not claims:
            return ClaimPromotionSummary(
                "skipped", attempted, 0, attempted, "no_promotable_claims"
            )
        try:
            promoted = await self._index.upsert_trusted(claims)
            return ClaimPromotionSummary(
                "succeeded", attempted, promoted, attempted - len(claims)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ClaimPromotionSummary(
                "degraded", attempted, 0, attempted, "claims_upsert_failed"
            )
