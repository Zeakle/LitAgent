"""Persist validated live-search abstracts without affecting survey delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from litagent.rag.models import ContentChunk, ContentScope, PaperCandidate


@dataclass(frozen=True)
class WritebackSummary:
    """Describe a non-authoritative live writeback attempt."""

    status: str
    accepted_count: int
    rejected_count: int
    reason_code: str | None = None


async def writeback_candidates(
    service,
    raw_candidates: Sequence[dict[str, Any]],
) -> WritebackSummary:
    """Validate dedup output and degrade instead of failing the survey."""
    accepted = []
    rejected = 0
    for raw in raw_candidates:
        try:
            candidate = PaperCandidate.model_validate(raw)
            if (
                candidate.source == "rag.index"
                or not candidate.title
                or not candidate.abstract
            ):
                raise ValueError("candidate is not writeback eligible")
            if not candidate.chunks:
                candidate = candidate.model_copy(
                    update={
                        "content_scope": ContentScope.ABSTRACT,
                        "chunks": [
                            ContentChunk.from_text(
                                paper_id=candidate.paper_id,
                                chunk_key="abstract",
                                text=candidate.abstract,
                                section="abstract",
                                content_scope=ContentScope.ABSTRACT,
                            )
                        ],
                    }
                )
            accepted.append(candidate)
        except (TypeError, ValueError):
            rejected += 1
    if not accepted:
        return WritebackSummary("skipped", 0, rejected, "no_valid_candidates")
    try:
        await service.sync_candidates(accepted)
        return WritebackSummary("succeeded", len(accepted), rejected)
    except Exception:
        # Corpus writeback is explicitly outside report correctness.
        return WritebackSummary(
            "degraded",
            len(accepted),
            rejected,
            "writeback_failed",
        )
