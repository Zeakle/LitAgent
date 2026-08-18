"""Persist complete local benchmark results as JSON and Markdown."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class BenchmarkArtifactPaths:
    """Identify the JSON and Markdown files for one benchmark run."""

    json_path: Path
    markdown_path: Path


def _atomic_write(path: Path, text: str) -> None:
    """Atomically replace an artifact file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _markdown(payload: Mapping[str, Any]) -> str:
    """Render benchmark results as Markdown."""
    if payload.get("benchmark_type") == "survey":
        return _survey_markdown(payload)
    run_id = str(payload.get("run_id") or "unknown")
    status = str(payload.get("status") or "unknown")
    summary = payload.get("summary") or {}
    lines = [
        f"# RAG Benchmark {run_id}",
        "",
        f"- Status: `{status}`",
        f"- Dataset: `{payload.get('dataset_fingerprint', 'unknown')}`",
        f"- Profile: `{payload.get('profile_fingerprint', 'unknown')}`",
        f"- Collection: `{payload.get('collection_identity', 'unknown')}`",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    recall = summary.get("recall_at_k") or {}
    for key, value in sorted(recall.items(), key=lambda item: int(item[0])):
        lines.append(f"| Recall@{key} | {float(value):.6f} |")
    for field, label in (
        ("mrr_at_10", "MRR@10"),
        ("ndcg_at_10", "nDCG@10"),
        ("duplicate_paper_ratio", "Duplicate paper ratio"),
        ("empty_result_rate", "Empty result rate"),
        ("expected_outcome_accuracy", "Expected outcome accuracy"),
        ("reason_code_accuracy", "Reason code accuracy"),
        ("contract_accuracy", "Case contract accuracy"),
        ("indexable_parse_success", "Indexable parse success"),
        ("quarantine_precision", "Quarantine precision"),
        ("quarantine_recall", "Quarantine recall"),
        ("metadata_accuracy", "Metadata accuracy"),
        ("locator_preservation", "Locator preservation"),
        ("duplicate_suppression", "Duplicate suppression"),
        ("incremental_update_accuracy", "Incremental update accuracy"),
        ("latency_p50_ms", "Latency p50 ms"),
        ("latency_p95_ms", "Latency p95 ms"),
    ):
        if field in summary:
            lines.append(f"| {label} | {float(summary[field]):.6f} |")
    reason_codes = payload.get("reason_codes") or []
    if reason_codes:
        lines.extend(["", "## Failure Reasons", ""])
        lines.extend(f"- `{reason}`" for reason in reason_codes)
    return "\n".join(lines) + "\n"


def _survey_markdown(payload: Mapping[str, Any]) -> str:
    """Render separated quality, delivery, and cost Survey metrics."""
    lines = [
        f"# Survey Benchmark {payload.get('run_id', 'unknown')}",
        "",
        f"- Status: `{payload.get('status', 'unknown')}`",
        f"- Stage: `{payload.get('stage', 'unknown')}`",
        f"- Dataset: `{payload.get('dataset_fingerprint', 'unknown')}`",
        f"- Formal eligible: `{payload.get('formal_eligible', False)}`",
        f"- Recommended profile: `{payload.get('recommended_profile_id') or 'none'}`",
    ]
    fields = (
        ("coverage_evidence", "Evidence coverage"),
        ("citation_recall", "Citation recall"),
        ("citation_precision", "Citation precision"),
        ("topic_coverage", "Topic coverage"),
        ("faithfulness", "Faithfulness"),
        ("unsupported_claim_rate", "Unsupported claim rate"),
        ("contradiction_rate", "Contradiction rate"),
        ("delivery_accuracy", "Delivery accuracy"),
        ("total_tokens", "Product tokens"),
        ("latency_p50_ms", "Latency p50 ms"),
        ("latency_p95_ms", "Latency p95 ms"),
    )
    summaries = list(payload.get("profile_summaries") or [])
    if not summaries and payload.get("summary"):
        summaries = [dict(payload["summary"], profile_id="summary")]
    for summary in summaries:
        lines.extend(
            [
                "",
                f"## Profile `{summary.get('profile_id', 'unknown')}`",
                "",
                "| Metric | Value |",
                "|---|---:|",
            ]
        )
        for field, label in fields:
            value = summary.get(field)
            rendered = "unavailable" if value is None else f"{float(value):.6f}"
            lines.append(f"| {label} | {rendered} |")
        distribution = summary.get("delivery_distribution") or {}
        if distribution:
            rendered_distribution = ", ".join(
                f"{status}={count}" for status, count in sorted(distribution.items())
            )
            lines.append(f"| Delivery distribution | {rendered_distribution} |")
        lines.append(
            f"| Judge tokens | {int(summary.get('judge_total_tokens') or 0)} |"
        )
    comparisons = payload.get("profile_comparisons") or []
    if comparisons:
        lines.extend(["", "## Pairwise Deltas", ""])
        for comparison in comparisons:
            pair = (
                f"{comparison.get('left_profile_id')} - "
                f"{comparison.get('right_profile_id')}"
            )
            if not comparison.get("comparable"):
                lines.append(f"- `{pair}`: non-comparable external inputs")
                continue
            deltas = comparison.get("metric_deltas") or {}
            rendered = ", ".join(
                f"{name}={value:.6f}" if value is not None else f"{name}=unavailable"
                for name, value in sorted(deltas.items())
            )
            lines.append(f"- `{pair}`: {rendered}")
    lines.extend(
        [
            "",
            "## Interpretation Boundaries",
            "",
            "- Judge metrics are separate from runtime evaluation and product token cost.",
            "- Missing Judge or evaluator metrics remain unavailable rather than being scored as zero.",
            "- Profile deltas require matching external-input fingerprints.",
        ]
    )
    reasons = payload.get("reason_codes") or []
    if reasons:
        lines.extend(["", "## Reason Codes", ""])
        lines.extend(f"- `{reason}`" for reason in reasons)
    return "\n".join(lines) + "\n"


class BenchmarkArtifactRepository:
    """Own atomic local benchmark artifact persistence."""

    def __init__(self, root: Path) -> None:
        """Initialize the benchmark artifact repository."""
        self._root = root

    @property
    def root(self) -> Path:
        """Return the owned artifact root for related private assets."""
        return self._root

    def write(
        self,
        result: BaseModel | Mapping[str, Any],
    ) -> BenchmarkArtifactPaths:
        """Persist one benchmark result as JSON and Markdown artifacts."""
        payload = (
            result.model_dump(mode="json")
            if isinstance(result, BaseModel)
            else dict(result)
        )
        run_id = str(payload.get("run_id") or "").strip()
        if not run_id or any(char in run_id for char in ("/", "\\", "..")):
            raise ValueError("benchmark artifact requires a safe run_id")
        json_path = self._root / f"{run_id}.json"
        markdown_path = self._root / f"{run_id}.md"
        _atomic_write(
            json_path,
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        )
        _atomic_write(markdown_path, _markdown(payload))
        return BenchmarkArtifactPaths(json_path, markdown_path)
