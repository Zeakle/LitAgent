"""Export public execution views from authenticated Survey benchmark archives."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from litagent.benchmark.survey_latency import (
    CASES,
    CAUSE_LABELS,
    PROFILES,
    SUMMARY_CHECKPOINT_SCOPE,
    WORKBENCH_MARKERS,
    _evaluation_score,
    write_bytes,
)
from scripts.survey_benchmark_public_content import project_report, project_step
from scripts.view_survey_latency import read_campaign, retry_rows

RESULTS = "benchmarks/survey/results/latency-faithfulness-95/execution-data"
ARCHIVES = "artifacts/benchmarks/survey"
PREFIX = "latency-faithfulness-95-"
RETRIES = ("summary-repair-pilot", "summary-checkpoint", "summary-rebuild")
ARCHIVE_ID = re.compile(r"[a-f0-9]{12}")
TASK_LABEL = re.compile(
    r"search_(?:arxiv|huggingface)_q[0-9]{1,2}|recall_q[0-9]{1,2}|"
    r"dedup|relevance_gate|extract|graph_analysis|adversarial_review"
)
SUBTASK_LABEL = re.compile(
    r"query_decomposition|evaluation\.(?:initial|post_rewrite)|evidence_rewrite|"
    r"adversarial_review:(?:synthesis|reviewer):r[0-9]{1,2}"
)
AGENT_TYPES = {
    "search",
    "recall",
    "dedup",
    "relevance_gate",
    "extractor",
    "graph",
    "adversarial_review",
}
STATES = {
    "completed",
    "done",
    "pending",
    "running",
    "failed",
    "error",
    "skipped",
    "cancelled",
    "unknown",
    "partial",
    "passed",
    "unverified",
    "timeout",
}
DELIVERY = {"ready", "blocked", "partial", "pending"}
OBSERVATION_KINDS = {"worker", "llm", "tool", "rag.search", "evidence.retrieve"}
OPERATION_KINDS = OBSERVATION_KINDS - {"worker"}
TOOL_NAMES = {"search_arxiv", "search_huggingface"}
SPAN_NAME = re.compile(
    r"planner\.query_decomposition|evaluation\.(?:initial|post_rewrite)|"
    r"(?:synthesis|reviewer)\.r[0-9]{1,2}"
)
LIFECYCLE = {"start", "end", "complete", "failed", "retry", "cancelled", "error"}
MILESTONES = {
    "survey.start",
    "survey.complete",
    "worker.input",
    "subspan.start",
    "subspan.end",
    "rag.recall.start",
    "rag.recall.complete",
    "search.source.complete",
    "relevance.gate.complete",
    "fulltext.start",
    "fulltext.complete",
    "external_inputs_ready",
    "evaluation.batch",
    "targeted_repair.state",
    "automatic_acceptance",
    "benchmark.accounting",
    "benchmark.summary_checkpoint",
    "benchmark.summary_capacity_continuation",
    "benchmark.external_inputs.replayed",
}
PUBLIC_EVENTS = MILESTONES | {
    f"{kind}.{state}" for kind in OBSERVATION_KINDS for state in LIFECYCLE
}
USAGE_COUNTS = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "billed_tokens",
    "reconciled_billed_tokens",
    "paid_requests",
    "llm_calls",
    "tool_calls",
    "worker_calls",
    "worker_retries",
    "tool_retries",
    "worker_failures",
    "tool_failures",
    "llm_failures",
    "llm_retries",
    "missing_usage_calls",
)
STAGE_LABELS = {
    "写作／审阅",
    "规划",
    "搜索／召回",
    "去重／筛选",
    "全文获取",
    "抽取（含全文获取）",
    "修订（含准备）",
    "evaluation.initial",
    "evaluation.post_rewrite",
    "evidence_rewrite",
}


def integer(value):
    """Retain only nonnegative counts and durations."""
    return value if type(value) is int and value >= 0 else None


def timestamp(value):
    """Normalize timezone-aware archive timestamps without copying free text."""
    if not isinstance(value, str):
        return None
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return None
    return stamp.astimezone(timezone.utc).isoformat() if stamp.tzinfo else None


def label(value, known, default=None):
    """Keep a string only if it belongs to a public vocabulary."""
    return value if isinstance(value, str) and value in known else default


def task_id(value):
    """Recognize workflow-generated task IDs instead of exposing raw identifiers."""
    if isinstance(value, str) and (
        TASK_LABEL.fullmatch(value) or SUBTASK_LABEL.fullmatch(value)
    ):
        return value
    return None


def scoped_identity(row):
    """Project the run and cohort identifiers used by the public selector."""
    run_id = row["run_id"]
    case = row["case_id"]
    profile = row["profile_id"]
    repetition = integer(row["repetition"])
    replay_id = f"replay-{case}-{profile}-r{repetition}"
    if (
        case not in CASES
        or profile not in PROFILES
        or repetition is None
        or not (
            ARCHIVE_ID.fullmatch(run_id)
            or re.fullmatch(re.escape(replay_id) + r"(?:-retry-[a-f0-9]{12})?", run_id)
        )
    ):
        raise ValueError("archive identity is outside the public benchmark scope")
    return {
        "run_id": run_id,
        "case_id": case,
        "profile_id": profile,
        "repetition": repetition,
        "status": label(row.get("status"), STATES, "unknown"),
        "delivery": label(row.get("delivery"), DELIVERY),
    }


def project_metrics(row):
    """Retain published scores, usage, and the recorded latency measurements."""
    evaluation = row.get("evaluation") or {}
    metrics = {}
    for key in ("faithfulness", "citation_accuracy", "internal_consistency"):
        item = evaluation.get(key) or {}
        score = _evaluation_score(item)
        threshold = item.get("threshold")
        metrics[key] = {
            "score": score if type(score) in (int, float) and 0 <= score <= 1 else None,
            "status": label(item.get("status"), STATES, "unverified"),
            "threshold": (
                threshold
                if type(threshold) in (int, float) and 0 <= threshold <= 1
                else None
            ),
            **{
                key: item[key]
                for key in (
                    "supported_count",
                    "factual_claim_count",
                    "unsupported_count",
                    "unverified_count",
                    "conflict_count",
                )
                if integer(item.get(key)) is not None
            },
        }

    usage = row.get("usage") or {}
    metrics["usage"] = {key: integer(usage.get(key)) for key in USAGE_COUNTS}
    metrics["usage"]["reported_tokens"] = integer(usage.get("total_tokens"))
    metrics["usage"]["usage_uncertain"] = usage.get("usage_uncertain") is True
    metrics["usage"]["usage_complete"] = usage.get("usage_complete") is True

    stages = {}
    for stage, elapsed in (row.get("stages_ms") or {}).items():
        if stage not in STAGE_LABELS and not task_id(stage):
            continue
        if isinstance(elapsed, list):
            stages[stage] = [integer(value) for value in elapsed]
        else:
            stages[stage] = integer(elapsed)

    browser = row.get("browser_ms") or {}
    clock = label(row.get("timing_scope"), {SUMMARY_CHECKPOINT_SCOPE})
    downstream = integer(row.get("downstream_ms"))
    elapsed = downstream
    basis = clock or "controlled_downstream"
    if row["source"] in {"workbench", "workbench-diagnostic"}:
        elapsed = integer(browser.get("final_report"))
        basis = "workbench_final_report"
        if elapsed is None:
            elapsed = integer(browser.get("terminal"))
            basis = "workbench_terminal"
    metrics.update(
        stages_ms=stages,
        timing_scope=clock,
        downstream_ms=downstream,
        elapsed_ms=integer(row.get("elapsed_ms")),
        browser_ms={key: integer(browser.get(key)) for key in WORKBENCH_MARKERS},
        display_elapsed_ms=elapsed,
        duration_basis=basis if elapsed is not None else None,
    )
    return metrics


def project_execution(row, archive, public_run_ids):
    """Build the Workbench view from authenticated reports and step observations."""
    summary = scoped_identity(row)
    run_id = summary["run_id"]
    summary.update(
        started_at=timestamp(archive["started_at"]),
        completed_at=timestamp(archive["completed_at"]),
    )

    tasks, step_io = {}, {}
    for key, item in archive["graph"]["tasks"].items():
        if not TASK_LABEL.fullmatch(key) or item.get("agent_type") not in AGENT_TYPES:
            raise ValueError("unrecognized benchmark task")
        tasks[key] = {
            "task_id": key,
            "agent_type": item["agent_type"],
            "status": label(item.get("status"), STATES, "unknown"),
        }
        if entry := project_step(input=item.get("input"), status="planned"):
            step_io[key] = entry
    dependencies = {}
    for key, required in archive["graph"]["dependencies"].items():
        if key not in tasks or any(dependency not in tasks for dependency in required):
            raise ValueError("invalid benchmark dependency")
        dependencies[key] = required

    pairs = {}
    nodes = {}
    for private_key, item in archive["nodes"].items():
        kind, name = item.get("kind"), item.get("name")
        key = task_id(item.get("task_id"))
        if kind not in OBSERVATION_KINDS or not key:
            continue
        operation = None
        if kind == "worker":
            if key not in tasks:
                raise ValueError("worker observation is outside its graph")
            node_id = f"worker:{key}"
            name = tasks[key]["agent_type"]
        else:
            private_operation = private_key.removeprefix(kind + ":")
            operation = pairs.setdefault(
                (kind, private_operation), f"op-{len(pairs) + 1}"
            )
            node_id = f"{kind}:{operation}"
            name = label(name, TOOL_NAMES, kind) if kind == "tool" else kind
        node = {
            "id": node_id,
            "kind": kind,
            "name": name,
            "task_id": key,
            "status": label(item.get("status"), STATES, "unknown"),
            "started_at": timestamp(item.get("started_at")),
            "completed_at": timestamp(item.get("completed_at")),
            "elapsed_ms": integer(item.get("elapsed_ms")),
        }
        if operation:
            node["operation_id"] = operation
        raw_usage = item.get("usage") or {}
        if raw_usage:
            node["usage"] = {
                field: integer(raw_usage.get(field)) for field in USAGE_COUNTS[:3]
            }
        source = (item.get("metadata") or {}).get("checkpoint_source_run_id")
        if kind == "worker" and source in public_run_ids:
            node["metadata"] = {"checkpoint_source_run_id": source}
        nodes[node_id] = node
        if entry := project_step(
            input=item.get("input"),
            output=item.get("output"),
            status="reused" if kind == "worker" and source else "recorded",
            withhold_output=kind == "llm",
        ):
            step_io[node_id] = entry
            if kind == "worker":
                step_io.pop(key, None)

    events, active_spans = [], {}
    span_number = 0
    for item in archive["events"]:
        event = item["event"]
        if event not in PUBLIC_EVENTS:
            continue
        at = timestamp(item.get("at"))
        if at is None:
            continue
        raw = item["data"]
        key = task_id(raw.get("task_id"))
        data = {"task_id": key} if key else {}
        parent = raw.get("parent_task_id")
        if isinstance(parent, str) and parent in tasks:
            data["parent_task_id"] = parent
        for field in (
            "attempt",
            "round",
            "elapsed_ms",
            "processed_assertions",
            "count",
        ):
            if integer(raw.get(field)) is not None:
                data[field] = raw[field]
        kind = event.rsplit(".", 1)[0]
        if kind in OPERATION_KINDS and isinstance(raw.get("operation_id"), str):
            private_operation = raw["operation_id"]
            data["operation_id"] = pairs.setdefault(
                (kind, private_operation), f"op-{len(pairs) + 1}"
            )

        if event == "subspan.start" and key and isinstance(raw.get("name"), str):
            span_number += 1
            name = raw["name"] if SPAN_NAME.fullmatch(raw["name"]) else key
            operation = f"span-{span_number}"
            node_id = f"subspan:{operation}"
            data["operation_id"] = operation
            node = {
                "id": node_id,
                "kind": "subspan",
                "name": name,
                "task_id": key,
                "status": "running",
                "operation_id": operation,
                "started_at": at,
                "metadata": {
                    field: data[field]
                    for field in ("parent_task_id", "round")
                    if field in data
                },
            }
            active_spans[key] = node
            nodes[node_id] = node
            if entry := project_step(input=raw.get("input")):
                step_io[node_id] = entry
        elif event == "subspan.end" and key in active_spans:
            node = active_spans.pop(key)
            data["operation_id"] = node["operation_id"]
            elapsed = datetime.fromisoformat(at) - datetime.fromisoformat(
                node["started_at"]
            )
            node.update(
                status=(
                    "cancelled"
                    if raw.get("error_type") == "CancelledError"
                    else "failed" if raw.get("error") else "completed"
                ),
                completed_at=at,
                elapsed_ms=(
                    integer(raw.get("elapsed_ms"))
                    if raw.get("elapsed_ms") is not None
                    else integer(int(elapsed.total_seconds() * 1000))
                ),
            )
            if entry := project_step(output=raw.get("output")):
                previous = step_io.setdefault(node["id"], {"status": "recorded"})
                omitted = previous.get("omitted", []) + entry.pop("omitted", [])
                previous.update(entry)
                if omitted:
                    previous["omitted"] = omitted
        if event == "benchmark.summary_checkpoint":
            source = label(raw.get("source_run_id"), public_run_ids)
            if source:
                summary["checkpoint_source_run_id"] = source
                data["source_run_id"] = source
        if event == "benchmark.summary_capacity_continuation":
            source = label(raw.get("run_id"), public_run_ids)
            if source:
                summary["continued_from_run_id"] = source
                data["source_run_id"] = source
        sequence = len(events) + 1
        events.append({"sequence": sequence, "event": event, "at": at, "data": data})
        if event == "llm.complete" and "operation_id" in data:
            model_step = step_io.get(f"llm:{data['operation_id']}")
            if model_step is not None:
                output = {
                    field: raw[field]
                    for field in (*USAGE_COUNTS[:3], "finish_reason", "output_size")
                    if field in raw
                }
                if entry := project_step(output=output or None):
                    model_step.update(entry)
        if event in MILESTONES - {"worker.input", "subspan.start", "subspan.end"} or (
            event.endswith((".retry", ".failed", ".cancelled", ".error"))
        ):
            event_payload = {
                field: value
                for field, value in raw.items()
                if field not in {"output", "input", "operation_id", "session_id"}
            }
            if event.endswith(".start") or event == "external_inputs_ready":
                entry = project_step(input=event_payload)
            else:
                entry = project_step(output=event_payload)
            if entry:
                step_io[f"event:sequence:{sequence}"] = entry

    quality = (archive.get("quality") or {}).get("status")
    reasons = [
        code
        for code in CAUSE_LABELS
        if any(code in value for value in row.get("reasons", ()))
    ]
    return {
        "identity": {
            "key": f"archive:{run_id}",
            "source": "archive",
            "runId": run_id,
            "status": summary["status"],
        },
        "summary": summary,
        "graph": {"tasks": tasks, "dependencies": dependencies, "nodes": nodes},
        "events": events,
        "delivery": {"status": summary["delivery"], "reason_codes": reasons},
        "quality": {"status": label(quality, STATES)} if quality in STATES else None,
        "metrics": project_metrics(row),
        "report": project_report(archive.get("report"), row.get("report_sha256")),
        "step_io": step_io,
    }


def campaign_entries(root):
    """Return each authenticated observation and archive in benchmark table order."""
    baseline = read_campaign(root / (PREFIX + "final-run"))
    attempts = []
    for name in RETRIES:
        campaign = read_campaign(root / (PREFIX + name))
        attempts.append((campaign, retry_rows(baseline, campaign)))

    for row in baseline.rows:
        run_id = row.get("run_id")
        if run_id not in baseline.archives:
            raise ValueError(
                "published benchmark requires complete baseline observations"
            )
        yield row, baseline.archives[run_id]
        for campaign, retried in attempts:
            if run_id in retried:
                row = retried[run_id]
                yield row, campaign.archives[row["run_id"]]

    diagnostics = read_campaign(root / (PREFIX + "recheck"))
    for row in diagnostics.rows:
        run_id = row.get("run_id")
        if run_id in diagnostics.archives:
            yield {**row, "source": "workbench-diagnostic"}, diagnostics.archives[
                run_id
            ]


def export(archives, output):
    """Generate all views before writing any public file."""
    entries = list(campaign_entries(archives))
    run_ids = {row["run_id"] for row, _ in entries}
    if not entries or len(entries) != len(run_ids):
        raise ValueError("duplicate or missing public run identity")
    views = {
        row["run_id"] + ".json": project_execution(row, artifact.read(), run_ids)
        for row, artifact in entries
    }
    views["index.json"] = {"runs": [scoped_identity(row) for row, _ in entries]}
    if output.exists() and {path.name for path in output.iterdir()} - views.keys():
        raise ValueError("output directory contains unexpected files")
    for name, view in views.items():
        content = json.dumps(
            view, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        )
        write_bytes(output / name, (content + "\n").encode())


def main():
    """Publish the data projection when the original local archives are available."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, default=Path(ARCHIVES))
    parser.add_argument("--output", type=Path, default=Path(RESULTS))
    args = parser.parse_args()
    export(args.archives, args.output)


if __name__ == "__main__":
    main()
