"""Trace deterministic citation formatting independently of LLM generations."""

from collections.abc import Callable, Collection
from dataclasses import asdict
from time import perf_counter
from uuid import uuid4

from litagent.evidence import CitationNormalizationResult, normalize_evidence_citations
from litagent.logging import get_logger

logger = get_logger("observability.citations")


def normalize_citations_traced(
    text: str,
    valid_evidence_ids: Collection[str],
    *,
    trace_hook: Callable[[str, dict], None] | None,
    parent_task_id: str,
    stage: str,
    round_num: int | None = None,
) -> CitationNormalizationResult:
    """Preserve raw and normalized text in one uniquely paired local subspan."""
    node_id = f"{parent_task_id}:citation_normalization:{uuid4().hex}"

    def emit(event: str, data: dict) -> None:
        """Keep observation failures from changing deterministic report output."""
        if trace_hook is not None:
            try:
                trace_hook(event, data)
            except Exception:
                logger.debug("Citation normalization trace hook failed", exc_info=True)

    emit(
        "subspan.start",
        {
            "task_id": node_id,
            "parent_task_id": parent_task_id,
            "name": "citation_normalization",
            "stage": stage,
            "round": round_num,
            "input": {"text": text, "valid_evidence_ids": sorted(valid_evidence_ids)},
        },
    )
    started = perf_counter()
    try:
        result = normalize_evidence_citations(text, valid_evidence_ids)
    except Exception as exc:
        emit("subspan.end", {"task_id": node_id, "error": type(exc).__name__})
        raise
    emit(
        "subspan.end",
        {
            "task_id": node_id,
            "output": asdict(result),
            "elapsed_ms": int((perf_counter() - started) * 1000),
        },
    )
    return result
