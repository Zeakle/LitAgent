"""Compute deterministic metrics from complete local Survey artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from litagent.benchmark.survey_models import (
    CitationMetrics,
    CoverageMetrics,
    PaperFlowCoverage,
    RuntimeCostMetrics,
    SurveyBenchmarkCase,
    SurveyCaseAggregate,
    SurveyCaseObservation,
    SurveyProfileSummary,
    SurveyRunMetrics,
)
from litagent.evidence import (
    extract_evidence_refs_ordered,
    find_malformed_evidence_refs,
)
from litagent.rag.models import normalize_paper_identity

METRICS_VERSION = "survey-metrics-v4-paired-downstream"


def _mean_optional(values: Sequence[float | int | None]) -> float | None:
    """Average only available numeric observations."""
    available = [float(value) for value in values if value is not None]
    return sum(available) / len(available) if available else None


def _percentile_optional(
    values: Sequence[float | int | None], q: float
) -> float | None:
    """Return an interpolated percentile over available values."""
    available = sorted(float(value) for value in values if value is not None)
    if not available:
        return None
    position = (len(available) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return available[lower]
    weight = position - lower
    return available[lower] * (1 - weight) + available[upper] * weight


def _unique(values: Sequence[str]) -> list[str]:
    """Return strings in stable first-seen order."""
    return list(dict.fromkeys(value for value in values if value))


def _worker_output(artifact: Mapping[str, Any], task_id: str) -> Any:
    """Return one terminal Worker output from an artifact node."""
    nodes = artifact.get("nodes")
    if not isinstance(nodes, Mapping):
        return None
    direct = nodes.get(f"worker:{task_id}")
    if isinstance(direct, Mapping):
        return direct.get("output")
    for node in nodes.values():
        if isinstance(node, Mapping) and node.get("task_id") == task_id:
            return node.get("output")
    return None


def _worker_outputs_with_prefix(
    artifact: Mapping[str, Any], prefix: str
) -> tuple[bool, list[Any]]:
    """Return whether a worker stage was observed and all matching outputs."""
    nodes = artifact.get("nodes")
    if not isinstance(nodes, Mapping):
        return False, []
    outputs: list[Any] = []
    observed = False
    for node in nodes.values():
        if not isinstance(node, Mapping):
            continue
        task_id = str(node.get("task_id") or "")
        if task_id.startswith(prefix):
            observed = True
            outputs.append(node.get("output"))
    return observed, outputs


def _paper_ids(value: Any) -> list[str]:
    """Collect stable paper IDs from nested worker output."""
    found: list[str] = []

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            paper_id = item.get("paper_id")
            if isinstance(paper_id, str) and paper_id:
                found.append(normalize_paper_identity(paper_id))
            for nested in item.values():
                if isinstance(nested, (Mapping, list, tuple)):
                    visit(nested)
        elif isinstance(item, (list, tuple)):
            for nested in item:
                visit(nested)

    visit(value)
    return _unique(found)


def _evidence_ledger(artifact: Mapping[str, Any]) -> dict[str, str]:
    """Map evidence IDs to paper IDs from extractor output."""
    output = _worker_output(artifact, "extract")
    if output is None:
        output = _worker_output(artifact, "extractor")
    ledger: dict[str, str] = {}
    for extraction in output if isinstance(output, list) else []:
        if not isinstance(extraction, Mapping):
            continue
        fallback_paper_id = str(extraction.get("paper_id") or "")
        for item in extraction.get("evidence_items") or []:
            if not isinstance(item, Mapping):
                continue
            evidence_id = str(item.get("evidence_id") or "")
            paper_id = str(item.get("paper_id") or fallback_paper_id)
            if evidence_id and paper_id and evidence_id not in ledger:
                ledger[evidence_id] = normalize_paper_identity(paper_id)
    return ledger


def artifact_evidence_ledger(
    artifact: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Return complete evidence items keyed by stable ID for Judge input."""
    output = _worker_output(artifact, "extract")
    if output is None:
        output = _worker_output(artifact, "extractor")
    ledger: dict[str, dict[str, Any]] = {}
    for extraction in output if isinstance(output, list) else []:
        if not isinstance(extraction, Mapping):
            continue
        fallback_paper_id = str(extraction.get("paper_id") or "")
        for item in extraction.get("evidence_items") or []:
            if not isinstance(item, Mapping):
                continue
            evidence_id = str(item.get("evidence_id") or "")
            if evidence_id and evidence_id not in ledger:
                snapshot = dict(item)
                snapshot.setdefault("paper_id", fallback_paper_id)
                ledger[evidence_id] = snapshot
    return ledger


def _coverage(required: set[str], observed: Sequence[str]) -> tuple[float, list[str]]:
    """Return required-paper recall and stable missing IDs."""
    required = {normalize_paper_identity(value) for value in required}
    observed_set = {normalize_paper_identity(value) for value in observed}
    missing = sorted(required - observed_set)
    return (len(required & observed_set) / len(required), missing)


def paper_flow_coverage(
    artifact: Mapping[str, Any], required_paper_ids: set[str]
) -> PaperFlowCoverage:
    """Track required-paper survival across observable production stages."""
    if not required_paper_ids:
        raise ValueError("required_paper_ids must not be empty")

    ledger = _evidence_ledger(artifact)
    report_observed = isinstance(artifact.get("report"), Mapping)
    report = artifact.get("report") if report_observed else {}
    survey = str(report.get("survey") or "")
    refs = extract_evidence_refs_ordered(survey)

    stage_values: dict[str, list[str] | None] = {}
    recall_observed, recall_outputs = _worker_outputs_with_prefix(artifact, "recall_")
    stage_values["recall"] = _paper_ids(recall_outputs) if recall_observed else None

    for stage, task_id in (
        ("dedup", "dedup"),
        ("relevance_gate", "relevance_gate"),
    ):
        output = _worker_output(artifact, task_id)
        stage_values[stage] = _paper_ids(output) if output is not None else None

    extraction_output = _worker_output(artifact, "extract")
    if extraction_output is None:
        extraction_output = _worker_output(artifact, "extractor")
    stage_values["evidence"] = (
        _unique(list(ledger.values())) if extraction_output is not None else None
    )

    adversarial_output = _worker_output(artifact, "adversarial_review")
    selected_ids: list[str] | None = None
    if isinstance(adversarial_output, Mapping):
        selection = adversarial_output.get("evidence_selection")
        if isinstance(selection, Mapping):
            sections = selection.get("section_evidence_ids")
            if isinstance(sections, Mapping):
                selected_ids = _unique(
                    [
                        evidence_id
                        for values in sections.values()
                        if isinstance(values, list)
                        for evidence_id in values
                        if isinstance(evidence_id, str)
                    ]
                )
    stage_values["synthesis_context"] = (
        _unique([ledger[item] for item in selected_ids if item in ledger])
        if selected_ids is not None
        else None
    )
    stage_values["final_citation"] = (
        _unique([ledger[item] for item in refs if item in ledger])
        if report_observed and "survey" in report
        else None
    )

    scores: dict[str, float | None] = {}
    missing_by_stage: dict[str, list[str]] = {}
    unavailable: list[str] = []
    for stage, observed_ids in stage_values.items():
        if observed_ids is None:
            scores[stage] = None
            missing_by_stage[stage] = sorted(required_paper_ids)
            unavailable.append(stage)
            continue
        score, missing = _coverage(required_paper_ids, observed_ids)
        scores[stage] = score
        missing_by_stage[stage] = missing

    return PaperFlowCoverage(
        **scores,
        missing_by_stage=missing_by_stage,
        unavailable_stages=unavailable,
    )


def _event_count(artifact: Mapping[str, Any], *events: str) -> int:
    """Count exact lifecycle events in the persisted timeline."""
    allowed = set(events)
    return sum(
        1
        for item in artifact.get("events") or []
        if isinstance(item, Mapping) and item.get("event") in allowed
    )


def _stage_latencies(artifact: Mapping[str, Any]) -> dict[str, int]:
    """Measure each stage's wall-clock span without summing parallel workers."""
    stage_tasks = {
        "search_recall": ("search_", "recall_"),
        "dedup": ("dedup",),
        "relevance": ("relevance_gate",),
        "extract_graph": ("extract", "extractor", "graph_analysis"),
        "synthesis_review": ("adversarial_review",),
        "evaluation": ("evaluation.",),
        "rewrite": ("evidence_rewrite",),
    }
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    nodes = artifact.get("nodes")
    if not isinstance(nodes, Mapping):
        return {}
    for node in nodes.values():
        if not isinstance(node, Mapping):
            continue
        task_id = str(node.get("task_id") or "")
        elapsed = node.get("elapsed_ms")
        if not isinstance(elapsed, (int, float)):
            continue
        for stage, prefixes in stage_tasks.items():
            if any(
                task_id == prefix or task_id.startswith(prefix) for prefix in prefixes
            ):
                grouped[stage].append(node)
                break
    latencies: dict[str, int] = {}
    for stage, stage_nodes in grouped.items():
        timestamp_pairs = []
        for node in stage_nodes:
            try:
                timestamp_pairs.append(
                    (
                        datetime.fromisoformat(str(node["started_at"])),
                        datetime.fromisoformat(str(node["completed_at"])),
                    )
                )
            except (KeyError, TypeError, ValueError):
                timestamp_pairs = []
                break
        if timestamp_pairs:
            started = min(pair[0] for pair in timestamp_pairs)
            completed = max(pair[1] for pair in timestamp_pairs)
            latencies[stage] = max(0, int((completed - started).total_seconds() * 1000))
        else:
            latencies[stage] = max(
                int(node.get("elapsed_ms") or 0) for node in stage_nodes
            )
    return {stage: elapsed for stage, elapsed in latencies.items() if elapsed}


def _controlled_downstream_ms(artifact: Mapping[str, Any]) -> int | None:
    """Measure one recorded wall-clock interval, never infer it by span sums."""
    start = None
    for event in artifact.get("events", []):
        if not isinstance(event, Mapping):
            continue
        if event.get("event") == "external_inputs_ready":
            try:
                start = datetime.fromisoformat(str(event["at"]))
            except (KeyError, ValueError):
                return None
        elif event.get("event") == "survey.complete" and start is not None:
            try:
                end = datetime.fromisoformat(str(event["at"]))
                elapsed = (end - start).total_seconds() * 1000
            except (KeyError, ValueError, TypeError):
                return None
            return round(elapsed) if elapsed >= 0 else None
    return None


def evaluate_survey_artifact(
    *,
    case: SurveyBenchmarkCase,
    artifact: Mapping[str, Any],
    judged_paper_ids: set[str] | None = None,
) -> SurveyRunMetrics:
    """Project one complete RunArtifact into hand-checkable metrics."""
    required = {normalize_paper_identity(p) for p in case.required_paper_ids}
    relevant = {normalize_paper_identity(p) for p in case.relevant_paper_ids}
    judged = (
        {normalize_paper_identity(p) for p in judged_paper_ids}
        if judged_paper_ids is not None
        else relevant
    )
    if not relevant.issubset(judged):
        raise ValueError("judged_paper_ids must include every relevant paper")
    dedup_ids = _paper_ids(_worker_output(artifact, "dedup"))
    relevance_ids = _paper_ids(_worker_output(artifact, "relevance_gate"))
    ledger = _evidence_ledger(artifact)
    evidence_ids = _unique(list(ledger.values()))

    dedup_coverage, dedup_missing = _coverage(required, dedup_ids)
    relevance_coverage, relevance_missing = _coverage(required, relevance_ids)
    evidence_coverage, evidence_missing = _coverage(required, evidence_ids)

    report = (
        artifact.get("report") if isinstance(artifact.get("report"), Mapping) else {}
    )
    survey = str(report.get("survey") or "")
    refs = extract_evidence_refs_ordered(survey)
    malformed_refs = find_malformed_evidence_refs(survey)
    unknown_refs = [ref for ref in refs if ref not in ledger]
    cited_papers = _unique([ledger[ref] for ref in refs if ref in ledger])
    reviewed_citations = [paper_id for paper_id in cited_papers if paper_id in judged]
    relevant_citations = [
        paper_id for paper_id in reviewed_citations if paper_id in relevant
    ]
    unjudged = [paper_id for paper_id in cited_papers if paper_id not in judged]
    citation_recall = len(required & set(cited_papers)) / len(required)
    citation_precision = (
        len(relevant_citations) / len(reviewed_citations)
        if reviewed_citations
        else None
    )

    usage = artifact.get("usage") if isinstance(artifact.get("usage"), Mapping) else {}
    delivery = (
        report.get("delivery") if isinstance(report.get("delivery"), Mapping) else {}
    )
    delivery_status = str(delivery.get("status") or "unknown")
    missing_usage = sum(
        1
        for event in artifact.get("events", [])
        if event.get("event") in {"llm.complete", "llm.failed"}
        and (
            event.get("data", {}).get("usage_available") is False
            or event.get("event") == "llm.failed"
            or (
                isinstance(event.get("data", {}).get("content"), dict)
                and "faithfulness" in event["data"]["content"]
            )
            or not {"prompt_tokens", "completion_tokens"}.issubset(
                event.get("data", {})
            )
        )
    )
    return SurveyRunMetrics(
        coverage=CoverageMetrics(
            dedup=dedup_coverage,
            relevance_gate=relevance_coverage,
            evidence=evidence_coverage,
            missing_by_stage={
                "dedup": dedup_missing,
                "relevance_gate": relevance_missing,
                "evidence": evidence_missing,
            },
        ),
        paper_flow=paper_flow_coverage(artifact, required),
        citations=CitationMetrics(
            recall=citation_recall,
            precision=citation_precision,
            cited_paper_ids=cited_papers,
            missing_required_paper_ids=sorted(required - set(cited_papers)),
            unjudged_paper_ids=unjudged,
            unknown_evidence_ids=unknown_refs,
            malformed_evidence_refs=malformed_refs,
            unknown_reference_rate=(
                (len(unknown_refs) + len(malformed_refs))
                / (len(refs) + len(malformed_refs))
                if refs or malformed_refs
                else 0.0
            ),
        ),
        delivery_status=delivery_status,
        delivery_accurate=delivery_status in case.allowed_delivery_statuses,
        partial=bool(report.get("partial", False)),
        elapsed_ms=max(0, int(artifact.get("elapsed_ms") or 0)),
        controlled_downstream_ms=_controlled_downstream_ms(artifact),
        stage_latency_ms=_stage_latencies(artifact),
        cost=RuntimeCostMetrics(
            usage_complete=missing_usage == 0,
            missing_usage_calls=missing_usage,
            prompt_tokens=max(0, int(usage.get("prompt_tokens") or 0)),
            completion_tokens=max(0, int(usage.get("completion_tokens") or 0)),
            total_tokens=max(0, int(usage.get("total_tokens") or 0)),
            llm_calls=_event_count(artifact, "llm.complete", "llm.failed"),
            tool_calls=_event_count(artifact, "tool.complete", "tool.failed"),
            worker_calls=_event_count(
                artifact,
                "worker.complete",
                "worker.failed",
                "worker.cancelled",
            ),
            worker_retries=_event_count(artifact, "worker.retry"),
            tool_retries=_event_count(artifact, "tool.retry"),
            worker_failures=_event_count(artifact, "worker.failed", "worker.cancelled"),
            tool_failures=_event_count(artifact, "tool.failed"),
        ),
        runtime_evaluation=dict(report.get("evaluation") or {}),
    )


def external_input_fingerprint(artifact: Mapping[str, Any]) -> str:
    """Fingerprint decomposition and external search outputs for comparability."""
    decomposition: dict[str, Any] | None = None
    search_outputs: dict[str, dict[str, Any]] = {}
    nodes = artifact.get("nodes")
    if isinstance(nodes, Mapping):
        for node_id, node in nodes.items():
            if not isinstance(node, Mapping):
                continue
            task_id = str(node.get("task_id") or "")
            kind = str(node.get("kind") or "")
            is_subspan = kind == "subspan" or str(node_id).startswith("subspan:")
            is_worker = kind == "worker" or str(node_id).startswith("worker:")
            if is_subspan and task_id in {
                "query_decomposition",
                "planner.query_decomposition",
            }:
                decomposition = {
                    "task_id": "query_decomposition",
                    "input": node.get("input"),
                    "output": node.get("output"),
                }
            elif is_worker and task_id.startswith("search_"):
                search_outputs[task_id] = {
                    "task_id": task_id,
                    "input": node.get("input"),
                    "output": node.get("output"),
                }
    projection = {
        "decomposition": decomposition,
        "search_outputs": [search_outputs[key] for key in sorted(search_outputs)],
    }
    encoded = json.dumps(
        projection, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _aggregate_case(group: Sequence[SurveyCaseObservation]) -> SurveyCaseAggregate:
    """Aggregate repeated observations for one profile/case pair."""
    first = group[0]
    valid = [item for item in group if item.status == "completed" and item.metrics]
    return SurveyCaseAggregate(
        case_id=first.case_id,
        domain=first.domain,
        profile_id=first.profile_id,
        observation_count=len(group),
        valid_observation_rate=len(valid) / len(group),
        coverage_evidence=_mean_optional(
            [item.metrics.coverage.evidence for item in valid if item.metrics]
        ),
        citation_recall=_mean_optional(
            [item.metrics.citations.recall for item in valid if item.metrics]
        ),
        citation_precision=_mean_optional(
            [item.metrics.citations.precision for item in valid if item.metrics]
        ),
        delivery_accuracy=_mean_optional(
            [float(item.metrics.delivery_accurate) for item in valid if item.metrics]
        ),
        topic_coverage=_mean_optional(
            [item.judge.topic_coverage for item in valid if item.judge]
        ),
        faithfulness=_mean_optional(
            [item.judge.faithfulness for item in valid if item.judge]
        ),
        unsupported_claim_rate=_mean_optional(
            [item.judge.unsupported_claim_rate for item in valid if item.judge]
        ),
        contradiction_rate=_mean_optional(
            [item.judge.contradiction_rate for item in valid if item.judge]
        ),
        total_tokens=_mean_optional(
            [item.metrics.cost.total_tokens for item in valid if item.metrics]
            if all(item.metrics.cost.usage_complete for item in valid if item.metrics)
            else []
        ),
        elapsed_ms=_mean_optional(
            [item.metrics.elapsed_ms for item in valid if item.metrics]
        ),
        controlled_downstream_ms=_mean_optional(
            [item.metrics.controlled_downstream_ms for item in valid if item.metrics]
        ),
    )


def _aggregate_profile(
    profile_id: str,
    cases: Sequence[SurveyCaseAggregate],
    observations: Sequence[SurveyCaseObservation],
) -> SurveyProfileSummary:
    """Macro-average case aggregates for one profile."""
    profile_observations = [
        item for item in observations if item.profile_id == profile_id
    ]
    valid_metrics = [
        item.metrics
        for item in profile_observations
        if item.status == "completed" and item.metrics is not None
    ]
    delivery_distribution: dict[str, int] = {}
    for metrics in valid_metrics:
        delivery_distribution[metrics.delivery_status] = (
            delivery_distribution.get(metrics.delivery_status, 0) + 1
        )
    stage_names = {
        stage for metrics in valid_metrics for stage in metrics.stage_latency_ms
    }
    worker_calls = sum(metrics.cost.worker_calls for metrics in valid_metrics)
    tool_calls = sum(metrics.cost.tool_calls for metrics in valid_metrics)
    worker_failures = sum(metrics.cost.worker_failures for metrics in valid_metrics)
    tool_failures = sum(metrics.cost.tool_failures for metrics in valid_metrics)
    return SurveyProfileSummary(
        profile_id=profile_id,
        case_count=len(cases),
        observation_count=len(profile_observations),
        valid_observation_rate=(
            sum(
                item.status == "completed" and item.metrics is not None
                for item in profile_observations
            )
            / len(profile_observations)
            if profile_observations
            else 0.0
        ),
        coverage_evidence=_mean_optional([item.coverage_evidence for item in cases]),
        citation_recall=_mean_optional([item.citation_recall for item in cases]),
        citation_precision=_mean_optional([item.citation_precision for item in cases]),
        delivery_accuracy=_mean_optional([item.delivery_accuracy for item in cases]),
        topic_coverage=_mean_optional([item.topic_coverage for item in cases]),
        faithfulness=_mean_optional([item.faithfulness for item in cases]),
        unsupported_claim_rate=_mean_optional(
            [item.unsupported_claim_rate for item in cases]
        ),
        contradiction_rate=_mean_optional([item.contradiction_rate for item in cases]),
        total_tokens=(
            _mean_optional([item.total_tokens for item in cases])
            if all(item.total_tokens is not None for item in cases)
            else None
        ),
        latency_p50_ms=_percentile_optional([item.elapsed_ms for item in cases], 0.50),
        latency_p95_ms=_percentile_optional([item.elapsed_ms for item in cases], 0.95),
        controlled_downstream_p95_ms=_percentile_optional(
            [item.controlled_downstream_ms for item in cases], 0.95
        ),
        delivery_distribution=delivery_distribution,
        stage_latency_p95_ms={
            stage: value
            for stage in sorted(stage_names)
            if (
                value := _percentile_optional(
                    [metrics.stage_latency_ms.get(stage) for metrics in valid_metrics],
                    0.95,
                )
            )
            is not None
        },
        product_total_tokens=sum(
            metrics.cost.total_tokens for metrics in valid_metrics
        ),
        judge_total_tokens=sum(
            item.judge_usage.total_tokens for item in profile_observations
        ),
        llm_call_count=sum(metrics.cost.llm_calls for metrics in valid_metrics),
        tool_call_count=tool_calls,
        worker_retry_count=sum(
            metrics.cost.worker_retries for metrics in valid_metrics
        ),
        tool_retry_count=sum(metrics.cost.tool_retries for metrics in valid_metrics),
        worker_failure_rate=(worker_failures / worker_calls if worker_calls else None),
        tool_failure_rate=(tool_failures / tool_calls if tool_calls else None),
    )


def aggregate_survey_cases(
    observations: Sequence[SurveyCaseObservation],
) -> tuple[list[SurveyCaseAggregate], SurveyProfileSummary]:
    """Aggregate one or more profiles with case-first weighting."""
    if not observations:
        raise ValueError("survey benchmark requires observations")
    profiles = {item.profile_id for item in observations}
    if len(profiles) != 1:
        raise ValueError("aggregate_survey_cases accepts exactly one profile")
    grouped: dict[tuple[str, str], list[SurveyCaseObservation]] = defaultdict(list)
    for observation in observations:
        grouped[(observation.profile_id, observation.case_id)].append(observation)
    cases = [
        _aggregate_case(grouped[key])
        for key in sorted(grouped, key=lambda value: (value[0], value[1]))
    ]
    profile_id = next(iter(profiles))
    return cases, _aggregate_profile(profile_id, cases, observations)


def aggregate_profiles(
    observations: Sequence[SurveyCaseObservation],
) -> tuple[list[SurveyCaseAggregate], list[SurveyProfileSummary]]:
    """Aggregate all profiles while preserving case-first weighting."""
    all_cases: list[SurveyCaseAggregate] = []
    summaries: list[SurveyProfileSummary] = []
    for profile_id in sorted({item.profile_id for item in observations}):
        profile_observations = [
            item for item in observations if item.profile_id == profile_id
        ]
        cases, summary = aggregate_survey_cases(profile_observations)
        all_cases.extend(cases)
        summaries.append(summary)
    return all_cases, summaries
