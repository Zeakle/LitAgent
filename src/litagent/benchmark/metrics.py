"""Compute deterministic, paper-level benchmark metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

from litagent.benchmark.models import (
    IngestionBenchmarkResult,
    IngestionBenchmarkSummary,
    IngestionCaseObservation,
    RetrievalBenchmarkSummary,
    RetrievalCaseMetrics,
    RetrievalCaseResult,
)


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: Sequence[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_retrieval_case(
    *,
    retrieved_paper_ids: Sequence[str],
    relevant_paper_ids: set[str],
    recall_ks: tuple[int, ...] = (5, 10, 20),
) -> RetrievalCaseMetrics:
    """Evaluate one ranking after paper-level de-duplication."""
    if not relevant_paper_ids:
        raise ValueError("relevant_paper_ids must not be empty")
    if not recall_ks or any(value <= 0 for value in recall_ks):
        raise ValueError("recall_ks must contain positive values")

    raw = list(retrieved_paper_ids)
    ranking = _unique(raw)
    recall = {
        k: len(set(ranking[:k]) & relevant_paper_ids) / len(relevant_paper_ids)
        for k in recall_ks
    }
    first_rank = next(
        (
            index
            for index, paper_id in enumerate(ranking[:10], start=1)
            if paper_id in relevant_paper_ids
        ),
        None,
    )
    mrr = 1.0 / first_rank if first_rank else 0.0
    dcg = sum(
        1.0 / math.log2(index + 1)
        for index, paper_id in enumerate(ranking[:10], start=1)
        if paper_id in relevant_paper_ids
    )
    ideal_count = min(len(relevant_paper_ids), 10)
    ideal_dcg = sum(1.0 / math.log2(index + 1) for index in range(1, ideal_count + 1))
    duplicate_ratio = (len(raw) - len(ranking)) / len(raw) if raw else 0.0
    return RetrievalCaseMetrics(
        recall_at_k=recall,
        mrr_at_10=mrr,
        ndcg_at_10=dcg / ideal_dcg if ideal_dcg else 0.0,
        duplicate_paper_ratio=duplicate_ratio,
    )


def aggregate_retrieval_cases(
    cases: Sequence[RetrievalCaseResult],
) -> RetrievalBenchmarkSummary:
    """Average independent query metrics and retain measured latency."""
    if not cases:
        raise ValueError("retrieval benchmark requires at least one case")
    recall_keys = sorted(cases[0].metrics.recall_at_k)
    for case in cases:
        if sorted(case.metrics.recall_at_k) != recall_keys:
            raise ValueError("retrieval cases use different recall cutoffs")
    return RetrievalBenchmarkSummary(
        case_count=len(cases),
        recall_at_k={
            k: _mean([case.metrics.recall_at_k[k] for case in cases])
            for k in recall_keys
        },
        mrr_at_10=_mean([case.metrics.mrr_at_10 for case in cases]),
        ndcg_at_10=_mean([case.metrics.ndcg_at_10 for case in cases]),
        duplicate_paper_ratio=_mean(
            [case.metrics.duplicate_paper_ratio for case in cases]
        ),
        empty_result_rate=_mean(
            [1.0 if not case.retrieved_paper_ids else 0.0 for case in cases]
        ),
        latency_p50_ms=_percentile([case.elapsed_ms for case in cases], 0.50),
        latency_p95_ms=_percentile([case.elapsed_ms for case in cases], 0.95),
    )


def aggregate_ingestion_cases(
    cases: Sequence[IngestionCaseObservation],
    *,
    run_id: str,
    dataset_fingerprint: str,
) -> IngestionBenchmarkResult:
    """Aggregate dirty-data outcomes without collapsing them to one score."""
    if not cases:
        raise ValueError("ingestion benchmark requires at least one case")
    expected_indexable = [
        case for case in cases if case.expected_outcome in {"indexed", "metadata_only"}
    ]
    expected_quarantine = [
        case for case in cases if case.expected_outcome == "quarantined"
    ]
    predicted_quarantine = [
        case for case in cases if case.actual_outcome == "quarantined"
    ]
    true_quarantine = [
        case for case in predicted_quarantine if case.expected_outcome == "quarantined"
    ]
    outcome_matches = [
        case.actual_outcome == case.expected_outcome for case in cases
    ]
    reason_matches = [
        set(case.expected_reason_codes).issubset(case.actual_reason_codes)
        for case in cases
    ]
    contract_matches = [
        outcome_match and reason_match
        for outcome_match, reason_match in zip(
            outcome_matches,
            reason_matches,
            strict=True,
        )
    ]
    metadata_values = [
        case.metadata_correct
        for case in cases
        if case.metadata_correct is not None
    ]
    locator_values = [
        case.locator_preserved
        for case in cases
        if case.locator_preserved is not None
    ]
    duplicate_values = [
        case.duplicates_suppressed
        for case in cases
        if case.duplicates_suppressed is not None
    ]
    incremental_values = [
        case.incremental_update_correct
        for case in cases
        if case.incremental_update_correct is not None
    ]
    summary = IngestionBenchmarkSummary(
        case_count=len(cases),
        expected_outcome_accuracy=_mean(outcome_matches),
        reason_code_accuracy=_mean(reason_matches),
        contract_accuracy=_mean(contract_matches),
        indexable_parse_success=_mean(
            [
                case.actual_outcome in {"indexed", "metadata_only"}
                for case in expected_indexable
            ]
        ),
        quarantine_precision=(
            len(true_quarantine) / len(predicted_quarantine)
            if predicted_quarantine
            else 0.0
        ),
        quarantine_recall=(
            len(true_quarantine) / len(expected_quarantine)
            if expected_quarantine
            else 0.0
        ),
        metadata_accuracy=_mean(metadata_values),
        locator_preservation=_mean(locator_values),
        duplicate_suppression=_mean(duplicate_values),
        incremental_update_accuracy=_mean(incremental_values),
        metadata_case_count=len(metadata_values),
        locator_case_count=len(locator_values),
        duplicate_case_count=len(duplicate_values),
        incremental_case_count=len(incremental_values),
        latency_p50_ms=_percentile([case.elapsed_ms for case in cases], 0.50),
        latency_p95_ms=_percentile([case.elapsed_ms for case in cases], 0.95),
    )
    return IngestionBenchmarkResult(
        run_id=run_id,
        status="succeeded" if all(contract_matches) else "failed",
        dataset_fingerprint=dataset_fingerprint,
        cases=list(cases),
        summary=summary,
        reason_codes=[
            f"case_contract_mismatch:{case.case_id}"
            for case, matched in zip(cases, contract_matches, strict=True)
            if not matched
        ],
    )
