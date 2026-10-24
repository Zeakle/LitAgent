"""Project run artifacts into a stable, read-only flow-demo contract."""

from __future__ import annotations

import copy
import hashlib
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
    result = {
        "evidence_id": evidence_id,
        "paper_id": str(value.get("paper_id") or ""),
        "paper_title": str(value.get("paper_title") or value.get("title") or ""),
        "text": str(value.get("text") or value.get("claim_text") or ""),
        "content_scope": str(value.get("content_scope") or "unknown"),
        "locator": locator,
        "confidence": value.get("confidence"),
    }
    # These are existing source fields, not metadata inferred from a model claim.
    for key in (
        "supporting_text",
        "source_locator",
        "chunk_key",
        "section",
        "page",
        "block_index",
        "bbox",
        "source_spans",
        "content_hash",
        "raw_content_hash",
        "evidence_version",
        "fact_fields",
        "record_identity",
    ):
        if key in value:
            result[key] = copy.deepcopy(value[key])
    return result


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


def extract_report_evidence(
    nodes: Mapping[str, Any], *, report: Mapping[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Expose the same host-selected evidence used by reading and audit views."""
    if not report:
        return _extract_evidence(nodes)
    metadata = _mapping(report.get("metadata"))
    bindings = _mapping(metadata.get("writing_bindings"))
    if (
        bindings.get("writer_contract") != "grounded-writer-v1"
        and metadata.get("generation_strategy") != "body_summary_v2"
    ):
        return _extract_evidence(nodes)
    survey = report.get("survey")
    if (
        bindings.get("writer_contract") != "grounded-writer-v1"
        or not isinstance(survey, str)
        or hashlib.sha256(survey.encode()).hexdigest() != bindings.get("report_sha256")
    ):
        return []
    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.quality import digest, validate_fact_snapshot

    cards = _mapping(metadata.get("fact_cards"))
    snapshot = metadata.get("evidence_snapshot_sha256")
    if not cards or any(not isinstance(card, Mapping) for card in cards.values()):
        return []
    candidates = {}
    for node_id, node in nodes.items():
        node = _mapping(node)
        output = _mapping(node.get("output"))
        if str(node_id).startswith("evidence.retrieve") or str(
            node.get("kind", "")
        ).startswith("evidence.retrieve"):
            pool = _mapping(output.get("selected_items"))
        elif output.get("generation_strategy") == "body_summary_v2":
            selection = _mapping(output.get("evidence_selection"))
            if (
                selection.get("valid") is False
                or selection.get("writing_scope_policy") != "frozen_union_v1"
                or output.get("evidence_snapshot_sha256") != snapshot
                or digest(_mapping(output.get("fact_cards"))) != snapshot
            ):
                continue
            try:
                pool = EvidenceSelection.from_dict(selection).selected_items
            except (ValueError, TypeError, KeyError):
                continue
        else:
            continue
        if not pool or any(not isinstance(item, Mapping) for item in pool.values()):
            continue
        try:
            validate_fact_snapshot(cards, pool, snapshot)
        except (ValueError, TypeError, KeyError):
            continue
        if {card.get("evidence_id") for card in cards.values()} != set(pool):
            continue
        candidates[digest(pool)] = pool
    # Ambiguous worker snapshots are not a reason to choose the latest source.
    if len(candidates) != 1:
        return []
    return [
        evidence
        for eid, item in next(iter(candidates.values())).items()
        if (evidence := _normalize_evidence(item, eid)) is not None
    ]


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
    evaluation_unavailable = (
        _mapping(
            _mapping(_mapping(report.get("evaluation")).get("faithfulness")).get(
                "details"
            )
        ).get("skipped_reason")
        == "final_report_unavailable"
    )
    if evaluation_unavailable:
        # A failed writer's placeholder is not evaluated academic prose.
        pipeline = {
            **pipeline,
            "assessment_scope": "generation_only",
            "stages": {
                **pipeline.get("stages", {}),
                "writing": "failed",
                "evaluation": "not_run",
            },
        }
    if pipeline or any(
        event.get("event") == "writing.wire.failed"
        for event in artifact.get("events", [])
    ):
        pipeline["writing_diagnostics"] = [
            {
                **diagnostic,
                "phase": event.get("data", {}).get("phase"),
                "candidate_sha256": event.get("data", {}).get("candidate_sha256"),
            }
            for event in artifact.get("events", [])
            if event.get("event") == "writing.validation.failed"
            for diagnostic in event.get("data", {}).get("diagnostics", [])
        ]
        # Wire failures locate a rejected provider response, never a sentence in
        # the placeholder report. Historical artifacts are projected read-only.
        for event in artifact.get("events", []):
            if event.get("event") != "writing.wire.failed":
                continue
            data = _mapping(event.get("data"))
            positions = {
                key: data.get(key)
                for key in ("json_line", "json_column", "json_position")
            }
            path = "native_response"
            if all(type(value) is int and value >= 0 for value in positions.values()):
                path += f" (line={positions['json_line']}, column={positions['json_column']}, character={positions['json_position']})"
            elif isinstance(data.get("schema_path"), list):
                path += "." + ".".join(str(part) for part in data["schema_path"])
            pipeline["writing_diagnostics"].append(
                {
                    "location": "native_response",
                    "raw_json_path": path,
                    "candidate_text": "",
                    "reason_code": data.get(
                        "reason_code", "array_wire_contract_invalid"
                    ),
                    "subreason": data.get("subreason")
                    or data.get("error_type")
                    or data.get("reason_code"),
                    "phase": data.get("phase"),
                    "candidate_sha256": data.get("candidate_sha256")
                    or data.get("raw_response_sha256"),
                    **{
                        key: value
                        for key, value in positions.items()
                        if type(value) is int
                    },
                }
            )
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
    evidence = extract_report_evidence(nodes, report=report)
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
