"""Project run artifacts into a stable, read-only flow-demo contract."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping

from litagent.report_render import (
    build_reader_report,
    paper_catalog_from_nodes,
    reader_report_projection,
    render_report_projection,
)
from litagent.workbench import ReportVersion

BUNDLED_SAMPLE_PATH = (
    Path(__file__).resolve().parent / "static" / "flow-demo" / "samples.json"
)

DemoSource = Literal["sample", "archive", "live"]


def load_bundled_samples() -> dict[str, dict[str, Any]]:
    """Load immutable synthetic artifacts shipped with the demo UI."""
    try:
        payload = json.loads(BUNDLED_SAMPLE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("bundled_demo_samples_unavailable") from exc

    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("samples"), list
    ):
        raise RuntimeError("bundled_demo_samples_invalid")

    samples: dict[str, dict[str, Any]] = {}
    for item in payload["samples"]:
        if not isinstance(item, dict) or not item.get("run_id"):
            raise RuntimeError("bundled_demo_samples_invalid")
        run_id = str(item["run_id"])
        if run_id in samples:
            raise RuntimeError("bundled_demo_samples_invalid")
        samples[run_id] = item
    return copy.deepcopy(samples)


def _mapping(value: Any) -> dict[str, Any]:
    """Return a shallow mapping copy without trusting archive field types."""
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    """Return a list copy for optional historical archive collections."""
    return list(value) if isinstance(value, (list, tuple)) else []


def _total_tokens(artifact: Mapping[str, Any]) -> int:
    """Prefer the artifact total and derive it from nodes for older archives."""
    usage = _mapping(artifact.get("usage"))
    if usage.get("total_tokens") is not None:
        return int(usage.get("total_tokens") or 0)
    total = 0
    for node in _mapping(artifact.get("nodes")).values():
        total += int(_mapping(_mapping(node).get("usage")).get("total_tokens") or 0)
    return total


def _normalize_evidence(item: Any, fallback_id: str = "") -> dict[str, Any] | None:
    """Normalize one selected evidence item for bounded UI rendering."""
    value = _mapping(item)
    if not value:
        return None
    evidence_id = str(value.get("evidence_id") or fallback_id)
    if not evidence_id:
        return None
    locator = value.get("source_locator") or value.get("locator") or ""
    return {
        "evidence_id": evidence_id,
        "paper_id": str(value.get("paper_id") or ""),
        "paper_title": str(value.get("paper_title") or value.get("title") or ""),
        "text": str(value.get("text") or value.get("claim_text") or ""),
        "content_scope": str(value.get("content_scope") or "unknown"),
        "locator": locator,
        "confidence": value.get("confidence"),
    }


def _extract_evidence(nodes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read selected evidence first and fall back to extractor evidence items."""
    for node_id, raw_node in nodes.items():
        node = _mapping(raw_node)
        if not (
            str(node_id).startswith("evidence.retrieve")
            or str(node.get("kind", "")).startswith("evidence.retrieve")
        ):
            continue
        selected = _mapping(node.get("output")).get("selected_items")
        if isinstance(selected, Mapping):
            values = [
                _normalize_evidence(item, str(evidence_id))
                for evidence_id, item in selected.items()
            ]
        else:
            values = [_normalize_evidence(item) for item in _list(selected)]
        normalized = [item for item in values if item is not None]
        if normalized:
            return normalized

    extract_node = _mapping(nodes.get("worker:extract"))
    normalized: list[dict[str, Any]] = []
    for extraction in _list(extract_node.get("output")):
        for item in _list(_mapping(extraction).get("evidence_items")):
            evidence = _normalize_evidence(item)
            if evidence is not None:
                normalized.append(evidence)
    return normalized


def extract_report_evidence(nodes: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose the same host-selected evidence used by reading and audit views."""
    return _extract_evidence(nodes)


def project_flow_demo_view(
    artifact: Mapping[str, Any], *, source: DemoSource, citation_style: str = "numbered"
) -> dict[str, Any]:
    """Project one raw RunArtifact into the stable browser-facing view."""
    report = _mapping(artifact.get("report"))
    quality = _mapping(report.get("quality") or artifact.get("quality"))
    delivery = _mapping(report.get("delivery") or artifact.get("delivery"))
    graph = _mapping(artifact.get("graph"))
    nodes = _mapping(artifact.get("nodes"))
    trace = _mapping(artifact.get("trace"))
    metadata = _mapping(report.get("metadata"))
    receipt = metadata.get("extraction_verification") or artifact.get(
        "extraction_verification"
    )
    pipeline = dict(artifact.get("pipeline_verification") or {})
    if pipeline:
        pipeline["writing_diagnostics"] = [
            diagnostic
            for event in artifact.get("events", [])
            if event.get("event") == "writing.validation.failed"
            for diagnostic in event.get("data", {}).get("diagnostics", [])
        ]
    if not report and receipt:
        # A sidecar is evidence of a pre-writing attempt, never a verified report.
        metadata = {
            "quality_policy": pipeline.get("quality_policy"),
            "extraction_verification": receipt,
        }
        quality = {"status": "unverified"}
        delivery = {
            "status": "blocked",
            "publishable": False,
            "reason_codes": [artifact.get("error") or "report_not_generated"],
        }
    survey = str(report.get("survey") or "")
    evidence = _extract_evidence(nodes)
    evidence_by_id = {
        item["evidence_id"]: item
        for item in evidence
        if isinstance(item.get("evidence_id"), str) and item["evidence_id"]
    }
    if survey:
        reader = build_reader_report(
            ReportVersion(
                run_id=str(artifact.get("run_id") or "archive"),
                revision=1,
                report_language=metadata.get("report_language") or "en",
                survey=survey,
            ),
            evidence_ledger=evidence_by_id,
            paper_catalog=paper_catalog_from_nodes(nodes),
            citation_style=citation_style,
        )
        rendered_report = reader_report_projection(reader)
    else:
        rendered_report = render_report_projection(survey)

    summary = {
        "run_id": str(artifact.get("run_id") or ""),
        "query": str(artifact.get("query") or metadata.get("query") or ""),
        "status": str(artifact.get("status") or "unknown"),
        "started_at": artifact.get("started_at"),
        "completed_at": artifact.get("completed_at"),
        "elapsed_ms": artifact.get("elapsed_ms"),
        "total_tokens": _total_tokens(artifact),
        "partial": bool(report.get("partial", False)),
        "quality_status": str(quality.get("status") or "unverified"),
        "delivery_status": str(delivery.get("status") or "pending"),
        "publishable": bool(delivery.get("publishable", False)),
        "reason_codes": _list(delivery.get("reason_codes")),
        "error": artifact.get("error"),
    }

    projection = {
        "schema_version": 1,
        "source": source,
        "provenance": str(artifact.get("provenance") or source),
        "formal_benchmark_evidence": bool(
            artifact.get("formal_benchmark_evidence", source != "sample")
        ),
        "summary": summary,
        "dag": {
            "tasks": _mapping(graph.get("tasks")),
            "dependencies": _mapping(graph.get("dependencies")),
            "nodes": nodes,
        },
        "timeline": _list(artifact.get("events")),
        "report": {
            "survey": survey,
            "review_history": _list(report.get("review_history")),
            "metadata": metadata,
        },
        "rendered_report": rendered_report,
        "evidence": evidence,
        "graph": _mapping(report.get("graph_data")),
        "evaluation": _mapping(report.get("evaluation")),
        "pipeline_verification": pipeline,
        "targeted_repair_state": metadata.get("targeted_repair_state")
        or artifact.get("targeted_repair_state"),
        "quality": quality,
        "delivery": delivery,
        "trace": {
            "trace_id": trace.get("trace_id"),
            "trace_url": trace.get("trace_url"),
        },
    }
    invalid_count = 0

    def browser_value(value: Any) -> Any:
        nonlocal invalid_count
        if isinstance(value, float) and not math.isfinite(value):
            invalid_count += 1
            return None
        if isinstance(value, dict):
            return {key: browser_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [browser_value(item) for item in value]
        return value

    projection = browser_value(projection)
    projection["display_warnings"] = (
        [{"code": "nonfinite_numeric_values", "count": invalid_count}]
        if invalid_count
        else []
    )
    return projection
