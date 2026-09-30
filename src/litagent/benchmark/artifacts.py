"""Persist complete local benchmark results as JSON and Markdown."""

from __future__ import annotations

import json
import os
import re
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
    if payload.get("benchmark_type") == "citation_audit":
        summary = payload.get("summary") or {}
        usage = payload.get("usage") or {}
        lines = [
            f"# Citation Audit {payload.get('run_id')}",
            "",
            f"- Status: `{payload.get('status')}`",
            f"- Source run: `{payload.get('source_run_id')}`",
            f"- Model: `{payload.get('actual_model') or payload.get('requested_model')}`",
            f"- Inventory: `{payload.get('inventory_sha256')}`",
            "- Experimental sidecar only; product delivery and quality are unchanged.",
            "",
            "## Protocol",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Planned questions | {summary.get('planned_questions', 0)} |",
            f"| Valid questions | {summary.get('valid_questions', 0)} |",
            f"| Protocol completion | {summary.get('protocol_completion_rate')} |",
            f"| Uncertain rate | {summary.get('uncertain_rate')} |",
            f"| Input tokens | {usage.get('input_tokens')} |",
            f"| Usage complete | {usage.get('usage_complete', False)} |",
        ]
        reasons = payload.get("reason_codes") or []
        if reasons:
            lines.extend(["", "## Reason Codes", ""])
            lines.extend(f"- `{reason}`" for reason in reasons)
        lines.extend(["", "## Decisions", ""])
        for decision in payload.get("decisions") or []:
            target = decision.get("evidence_id") or "citation set"
            lines.append(
                f"- `{decision.get('claim_id')}` / `{target}`: "
                f"`{decision.get('choice')}` (confidence={decision.get('confidence')})"
            )
        return "\n".join(lines) + "\n"
    if payload.get("benchmark_type") == "survey_corrected":
        lines = [
            f"# Corrected Survey Diagnostic {payload.get('run_id')}",
            "",
            "Historical offline recomputation only; not a new Survey or formal baseline.",
            f"Source run: `{payload.get('source_run_id')}`",
            f"Source SHA-256: `{payload.get('source_sha256')}`",
            f"Metrics version: `{payload.get('metrics_version')}`",
            "",
            "## Profile summaries",
            "",
            "| Profile | N | Coverage | Citation recall | Judge faithfulness | Product tokens |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for item in payload.get("profile_summaries", []):
            lines.append(
                f"| {item['profile_id']} | {item['observation_count']} | "
                f"{item.get('coverage_evidence')} | {item.get('citation_recall')} | "
                f"{item.get('faithfulness')} | {item.get('product_total_tokens')} |"
            )
        lines.extend(["", "## Paired comparisons", ""])
        for item in payload.get("profile_comparisons", []):
            lines.extend(
                [
                    f"### {item['left_profile_id']} vs {item['right_profile_id']}",
                    "",
                    f"Comparable: `{item['comparable']}`; shared cohorts: `{item['shared_cohort_count']}`.",
                    "",
                    "```json",
                    json.dumps(
                        {
                            "paired_counts": item.get("metric_paired_counts"),
                            "exclusions": item.get("metric_exclusion_reasons"),
                            "left_minus_right": item.get("metric_deltas"),
                            "reason_codes": item.get("reason_codes"),
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                    "```",
                ]
            )
        lines.extend(
            [
                "",
                "Old runs have no external_inputs_ready boundary; controlled latency is unavailable.",
                "Historical RAGAS usage cannot be recovered; cost superiority is not inferred.",
            ]
        )
        return "\n".join(lines) + "\n"
    if payload.get("benchmark_type") == "survey_rejudge":
        lines = [
            f"# Survey Judge Recovery {payload.get('run_id')}",
            "",
            f"- Status: `{payload.get('status')}`",
            "- Diagnostic only; not a formal baseline.",
            f"- Source SHA-256: `{payload.get('source_sha256')}`",
            "",
            "| Product | Runtime Faithfulness | Judge Claims | Judge Faithfulness | Judge Tokens | Usage Complete |",
            "|---|---|---|---|---|---|",
        ]
        for item in payload.get("observations", []):
            metrics = item.get("metrics") or {}
            runtime = metrics.get("runtime_evaluation") or {}
            judge = item.get("judge") or {}
            usage = item.get("judge_usage") or {}
            lines.append(
                f"| {item.get('run_id')} | {runtime.get('faithfulness')} | "
                f"{judge.get('factual_claim_count')} | {judge.get('faithfulness')} | "
                f"{usage.get('total_tokens')} | {usage.get('usage_complete')} |"
            )
        lines.extend(
            [
                "",
                "## Accounting",
                "",
                "```json",
                json.dumps(payload.get("accounting", {}), indent=2),
                "```",
                "",
                "Claim inventories and evidence scopes can differ; scores are not interchangeable.",
            ]
        )
        return "\n".join(lines) + "\n"
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
    if payload.get("observations"):
        lines.extend(
            [
                "",
                "## Runtime / Judge Audit",
                "",
                "| Case / Profile | Runtime claims | Runtime faithfulness | Citation diagnostics | Judge claims | Judge faithfulness | Judge status / reasons | Usage complete |",
                "|---|---:|---:|---|---:|---:|---|---|",
            ]
        )
        for observation in payload["observations"]:
            metrics = observation.get("metrics") or {}
            runtime = (metrics.get("runtime_evaluation") or {}).get(
                "faithfulness"
            ) or {}
            citations = metrics.get("citations") or {}
            judge = observation.get("judge") or {}
            count = (runtime.get("details") or {}).get("factual_claim_count")
            citation_diagnostics = [
                *(
                    f"unknown:{value}"
                    for value in citations.get("unknown_evidence_ids") or []
                ),
                *(
                    f"malformed:{value}"
                    for value in citations.get("malformed_evidence_refs") or []
                ),
            ]
            judge_reasons = judge.get("reason_codes") or []
            lines.append(
                f"| {observation.get('case_id')} / {observation.get('profile_id')} "
                f"| {count} | {runtime.get('score')} "
                f"| {', '.join(citation_diagnostics) or 'none'} "
                f"| {judge.get('factual_claim_count')} | {judge.get('faithfulness')} "
                f"| {judge.get('status', 'unavailable')}: {', '.join(judge_reasons) or 'none'} "
                f"| {(metrics.get('cost') or {}).get('usage_complete', False)} |"
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

    def _checkpoint_path(self, namespace: str, key: str) -> Path:
        """Resolve one checkpoint path without allowing path traversal."""
        if (
            re.fullmatch(r"[A-Za-z0-9_-]+", namespace) is None
            or re.fullmatch(r"[0-9a-f]{64}", key) is None
        ):
            raise ValueError("benchmark checkpoint requires a safe checkpoint path")
        return self._root / "checkpoints" / namespace / f"{key}.json"

    def write_checkpoint(
        self,
        namespace: str,
        key: str,
        payload: BaseModel | Mapping[str, Any],
    ) -> Path:
        """Atomically persist one resumable benchmark checkpoint."""
        checkpoint = (
            payload.model_dump(mode="json")
            if isinstance(payload, BaseModel)
            else dict(payload)
        )
        path = self._checkpoint_path(namespace, key)
        _atomic_write(
            path,
            json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True),
        )
        return path

    def read_checkpoint(
        self, namespace: str, key: str, *, strict: bool = False
    ) -> dict[str, Any] | None:
        """Load a valid checkpoint, treating corruption as a cache miss."""
        path = self._checkpoint_path(namespace, key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            if strict:
                raise ValueError(f"corrupt checkpoint: {namespace}/{key}") from exc
            return None
        if strict and not isinstance(payload, dict):
            raise ValueError(f"invalid checkpoint: {namespace}/{key}")
        return payload if isinstance(payload, dict) else None

    def list_checkpoints(self, namespace: str) -> list[dict[str, Any]]:
        """Read all accounting records, refusing to hide corrupt paid-run state."""
        directory = self._checkpoint_path(namespace, "0" * 64).parent
        return [
            self.read_checkpoint(namespace, path.stem, strict=True)
            for path in sorted(directory.glob("*.json"))
        ]
