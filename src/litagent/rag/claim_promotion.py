"""Promote only final, locator-backed evidence into cross-run Claims."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from litagent.rag.claims_index import ClaimsIndex, TrustedClaim

_CLAIM_NAMESPACE = uuid.UUID("a9506bd8-35d8-59e9-8180-2f0f018c99e5")


@dataclass(frozen=True)
class ClaimPromotionSummary:
    status: str
    attempted_count: int
    promoted_count: int
    rejected_count: int
    reason_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
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


class ClaimsPromoter:
    def __init__(self, index: ClaimsIndex) -> None:
        self._index = index

    async def promote(
        self,
        *,
        run_id: str,
        domain: str,
        report_data: Mapping[str, Any],
        extractions: Sequence[Mapping[str, Any]],
    ) -> ClaimPromotionSummary:
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
        claims = [
            claim
            for item in raw_items
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
        if not claims:
            return ClaimPromotionSummary(
                "skipped", len(raw_items), 0, len(raw_items), "no_promotable_claims"
            )
        try:
            promoted = await self._index.upsert_trusted(claims)
            return ClaimPromotionSummary(
                "succeeded", len(raw_items), promoted, len(raw_items) - len(claims)
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            return ClaimPromotionSummary(
                "degraded", len(raw_items), 0, len(raw_items), "claims_upsert_failed"
            )
