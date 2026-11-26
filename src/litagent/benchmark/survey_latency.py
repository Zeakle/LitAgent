"""Measure the isolated Survey latency matrix and render its run ledger."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import urlencode

from litagent.benchmark.survey_metrics import (
    _controlled_downstream_ms,
    _percentile_optional,
    evaluate_survey_artifact,
)
from litagent.eval.results import BENCHMARK_FAITHFULNESS_POLICY, BENCHMARK_POLICY_FIELD

CASES = ("few-shot-metric", "vit-foundations", "nerf-foundations")
PROFILES = ("external-only", "external-abstract", "external-fulltext")
WORKBENCH_MARKERS = ("first_progress", "first_draft", "terminal", "final_report")
PRODUCT_METRICS = ("citation_accuracy", "faithfulness", "internal_consistency")
CONTRACT = "survey-latency-v1"
SUMMARY_CHECKPOINT_SCOPE = "summary_checkpoint"
VIEWER_URL = "http://127.0.0.1:8765"
VIEW_HEADERS = (
    "运行",
    "工作台",
    "严格交付",
    "Faithfulness",
    "引用 ID 有效性",
    "内部一致性",
    "耗时（秒）",
    "Tokens",
)
COHORT_LABELS = {
    "workbench": "工作台",
    "replay": "受控回放",
    "workbench-diagnostic": "诊断复验",
    "retry": "修复复跑",
    SUMMARY_CHECKPOINT_SCOPE: "摘要阶段续跑",
}
STATUS_LABELS = {
    "ready": "通过",
    "blocked": "未通过",
    "completed": "已完成",
    "timeout": "超时",
    "cancelled": "已取消",
}
CAUSE_LABELS = {
    "writer_summary_capacity_exceeded": "摘要或结论超长",
    "writer_capacity_source_map_invalid": "正文压缩来源归属错误",
    "AccountOverdueError": "方舟账户拒绝请求",
    "extract_request_timeout": "文献抽取超时",
    "automatic_capacity_binding_invalid": "预算账本绑定错误",
    "benchmark_usage_unknown": "模型用量未结算",
    "strict_usage_incomplete": "模型用量未结算",
    "benchmark_budget_exhausted": "预算用尽",
    "automatic_report_partial": "报告未完成",
    "evaluation_not_passed": "评估未通过",
}
VIEW_NOTE = (
    "工作台耗时取提交至终稿可读，终稿缺失时取提交至终态。"
    "受控回放耗时取外部输入就绪至 Survey 完成。"
    "Tokens 为供应商已报告用量，部分回执缺失时单独标记。"
    "— 表示缺测，Judge 未运行。"
)
CHECKPOINT_NOTE = "摘要阶段续跑耗时取摘要修订开始至 Survey 完成，单独展示。"


@dataclass(frozen=True)
class RunSlot:
    """Identify one planned observation independently of the actual run ID."""

    source: str
    case_id: str
    profile_id: str
    repetition: int

    @property
    def key(self) -> str:
        """Return a stable key for files and table rows."""
        return f"{self.source}-{self.case_id}-{self.profile_id}-r{self.repetition}"


def run_slots() -> list[RunSlot]:
    """Prepare the browser cohort and the two balanced replay repetitions."""
    slots = [RunSlot("workbench", case, PROFILES[1], 1) for case in CASES]
    for repetition in (1, 2):
        for index, case in enumerate(CASES):
            order = PROFILES[index:] + PROFILES[:index]
            if repetition == 2:
                order = tuple(reversed(order))
            slots.extend(
                RunSlot("replay", case, profile, repetition) for profile in order
            )
    return slots


def _time(event):
    """Parse one timezone-aware recorder timestamp."""
    try:
        timestamp = datetime.fromisoformat(event["at"])
    except (KeyError, TypeError, ValueError):
        return None
    return timestamp if timestamp.utcoffset() is not None else None


def summary_checkpoint_duration_ms(events):
    """Measure the real continuation from its checkpoint to final delivery."""
    start = None
    for event in events:
        if event.get("event") == "benchmark.summary_checkpoint":
            start = _time(event)
        elif event.get("event") == "survey.complete" and start is not None:
            end = _time(event)
            if end is not None and end >= start:
                return round((end - start).total_seconds() * 1000)
            return None
    return None


def _span_group(task_id, event_name):
    """Classify recorded task and subspan identities for this experiment."""
    if event_name.startswith("fulltext."):
        return "全文获取"
    if task_id in {"query_decomposition", "planner.query_decomposition"}:
        return "规划"
    if task_id.startswith(("search_", "recall_")):
        return "搜索／召回"
    if task_id in {"dedup", "relevance_gate"}:
        return "去重／筛选"
    if task_id in {"extract", "extractor"}:
        return "抽取（含全文获取）"
    if task_id == "adversarial_review":
        return "写作／审阅"
    if task_id.startswith(("evaluation.", "evidence_rewrite")):
        return task_id
    if ":synthesis:r" in task_id or ":reviewer:r" in task_id:
        return task_id
    return None


def stage_spans(events: list[dict]) -> tuple[dict, list[str]]:
    """Measure complete stage spans and retain individual repair attempts."""
    pending = {}
    grouped = defaultdict(list)
    missing = set()
    for event in events:
        name = event.get("event", "")
        if not name.startswith(("worker.", "subspan.", "fulltext.")):
            continue
        lifecycle, _, state = name.partition(".")
        data = event.get("data") or {}
        task = data.get("task_id") or ""
        group = _span_group(task, name)
        if group is None:
            continue
        key = (lifecycle, data.get("operation_id") or task)
        timestamp = _time(event)
        if state == "start":
            if key in pending:
                missing.add(group)
            pending[key] = (group, timestamp)
        elif state in {"complete", "end", "failed", "cancelled"}:
            if key not in pending:
                missing.add(group)
                continue
            _, start = pending.pop(key)
            if start is None or timestamp is None or timestamp < start:
                missing.add(group)
                continue
            grouped[group].append((start, timestamp))
    missing.update(group for group, _ in pending.values())
    spans = {}
    for group, intervals in grouped.items():
        if group in missing:
            continue
        if group.startswith(("evaluation.", "evidence_rewrite", "adversarial_review:")):
            spans[group] = [
                round((end - start).total_seconds() * 1000) for start, end in intervals
            ]
        else:
            spans[group] = round(
                (
                    max(end for _, end in intervals)
                    - min(start for start, _ in intervals)
                ).total_seconds()
                * 1000
            )
    if "evidence_rewrite" not in spans:
        windows = _rewrite_evaluation_windows(events)
        if windows:
            spans["修订（含准备）"] = windows
    return spans, sorted(missing)


def _rewrite_evaluation_windows(events: list[dict]) -> list[int]:
    """Measure a repair bounded by two evaluations when no repair span is emitted."""
    previous_end = None
    saw_rewrite = False
    windows = []
    for event in events:
        name = event.get("event")
        task = (event.get("data") or {}).get("task_id")
        if name == "subspan.end" and task in {
            "evaluation.initial",
            "evaluation.post_rewrite",
        }:
            previous_end = _time(event)
            saw_rewrite = False
        elif (
            name == "llm.start"
            and task == "evidence_rewrite"
            and previous_end is not None
        ):
            saw_rewrite = True
        elif (
            name == "subspan.start"
            and task == "evaluation.post_rewrite"
            and saw_rewrite
        ):
            timestamp = _time(event)
            if timestamp is not None and timestamp >= previous_end:
                windows.append(round((timestamp - previous_end).total_seconds() * 1000))
            saw_rewrite = False
    return windows


def write_bytes(path: Path, data: bytes) -> None:
    """Publish a complete local report or snapshot atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, value) -> None:
    """Publish human-readable JSON without interpolating its contents."""
    write_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode() + b"\n")


def llm_outcomes(events: list[dict]) -> dict[str, int]:
    """Count terminal model failures and reported provider retries independently."""
    counts = Counter(event.get("event") for event in events)
    rejected_responses = sum(
        type(code) is int and code >= 400
        for event in events
        if event.get("event") == "llm.complete"
        for code in [(event.get("data") or {}).get("provider_http_status")]
    )
    return {
        "llm_failures": counts["llm.failed"] + rejected_responses,
        "llm_retries": counts["llm.retry"],
    }


def save_observation(
    slot,
    case,
    source_path,
    root,
    judged,
    browser=None,
    snapshot=None,
    search_tasks=None,
):
    """Project metrics from the exact recorded bytes and retain source versions."""
    raw = source_path.read_bytes()
    artifact = json.loads(raw)
    path = root / "snapshots" / slot.key / "run.json"
    write_bytes(path, raw)
    events = artifact.get("events") or []
    report = artifact.get("report") or {}
    evaluation = report.get("evaluation") or {}
    quality = report.get("quality") or {}
    delivery = report.get("delivery") or {}
    reasons = [
        str(reason)
        for reason in (
            artifact.get("error"),
            *(delivery.get("reason_codes") or ()),
            *(quality.get("strict_acceptance", {}).get("failure_codes") or ()),
        )
        if reason
    ]
    archived_reasons = archive_failure_reasons(artifact)
    if delivery.get("status") != "ready":
        reasons = list(dict.fromkeys([*reasons, *archived_reasons]))
    metrics = evaluate_survey_artifact(
        case=case, artifact=artifact, judged_paper_ids=judged
    )
    stages, incomplete_stages = stage_spans(events)
    elapsed = artifact.get("elapsed_ms")
    downstream = (
        _controlled_downstream_ms(artifact) if slot.source == "replay" else None
    )
    summary_checkpoint = any(
        event.get("event") == "benchmark.summary_checkpoint" for event in events
    )
    if summary_checkpoint:
        downstream = summary_checkpoint_duration_ms(events)
    replay_points = [
        e.get("data", {}).get("fingerprint")
        for e in events
        if e.get("event") == "external_inputs_ready"
    ]
    replay_complete = replay_points == [snapshot]
    if search_tasks is not None:
        expected = {("planner", None, snapshot)}
        expected.update(("search", task, snapshot) for task in search_tasks)
        receipts = [
            (
                data.get("component"),
                data.get("task_id"),
                data.get("snapshot_fingerprint"),
            )
            for event in events
            if event.get("event") == "benchmark.external_inputs.replayed"
            for data in [event.get("data") or {}]
        ]
        replay_complete = (
            replay_complete
            and len(receipts) == len(expected)
            and set(receipts) == expected
        )
    if slot.source == "replay" and not replay_complete:
        downstream = None
        reasons.append("external_input_fingerprint_mismatch")
    versions = []
    for index, event in enumerate(events):
        data = event.get("data") or {}
        task = data.get("task_id") or ""
        first_writer = task.endswith(":synthesis:r1")
        if not (
            (
                event.get("event") == "subspan.end"
                and (first_writer or task.startswith("evaluation."))
            )
            or (task == "evidence_rewrite" and event.get("event") == "subspan.end")
        ):
            continue
        if data.get("output") is None:
            continue
        version = path.parent / f"{index:04d}.json"
        write_json(version, event)
        versions.append({"task": task, "path": str(version.relative_to(root))})
    run_id = artifact["run_id"]
    if Path(run_id).name == run_id:
        journal = root / "repairs" / f"{run_id}.json.writer-v2.json"
        if journal.is_file():
            version = path.parent / "writer-revisions.json"
            write_bytes(version, journal.read_bytes())
            versions.append(
                {"task": "写作与修订记录", "path": str(version.relative_to(root))}
            )
    snapshot_sha = hashlib.sha256(raw).hexdigest()
    terminal = artifact.get("status", "unknown")
    if terminal == "failed":
        if any("timeout" in reason.lower() for reason in reasons):
            terminal = "timeout"
        elif any("cancel" in reason.lower() for reason in reasons):
            terminal = "cancelled"
    delivery_status = delivery.get("status")
    group = delivery_status if terminal == "completed" and delivery_status else terminal
    if report.get("partial") is True and any(
        reason.endswith("_request_timeout") for reason in archived_reasons
    ):
        group = "timeout"
    basis = next(
        (
            e.get("data") or {}
            for e in reversed(events)
            if e.get("event") == "benchmark.accounting"
        ),
        artifact.get("budget_state") or {},
    )
    uncertain = basis.get("usage_uncertain")
    usage = {**metrics.cost.model_dump(), **llm_outcomes(events)}
    usage["billed_tokens"] = basis.get("known_tokens") if uncertain is False else None
    if uncertain is True:
        usage["budget_accounted_tokens"] = basis.get("known_tokens")
    usage["paid_requests"] = basis.get("requests")
    usage["usage_uncertain"] = uncertain if type(uncertain) is bool else True
    citations = metrics.citations.model_dump() if report.get("survey") else {}
    for key in (
        "missing_required_paper_ids",
        "unknown_evidence_ids",
        "malformed_evidence_refs",
    ):
        if key in citations:
            citations[f"{key}_count"] = len(citations[key])
    report_sha = (
        hashlib.sha256(report["survey"].encode()).hexdigest()
        if isinstance(report.get("survey"), str)
        else None
    )
    faithfulness = (evaluation.get("faithfulness") or {}).get("details") or {}
    eval_sha = faithfulness.get("source_report_sha256")
    policy = faithfulness.get(BENCHMARK_POLICY_FIELD)
    if eval_sha is not None and eval_sha != report_sha:
        reasons.append("evaluation_report_identity_mismatch")
        evaluation = {}
    elif eval_sha is not None and policy != BENCHMARK_FAITHFULNESS_POLICY:
        reasons.append("benchmark_evaluation_policy_mismatch")
        evaluation = {}
    effective = (artifact.get("config") or {}).get("effective") or {}
    return {
        "run_id": run_id,
        "started_at": artifact.get("started_at"),
        "completed_at": artifact.get("completed_at"),
        "status": terminal,
        "delivery": delivery_status,
        "group": group,
        "delivery_accurate": metrics.delivery_accurate if delivery_status else None,
        "policy": policy or (effective.get("eval") or {}).get(BENCHMARK_POLICY_FIELD),
        "report_sha256": report_sha,
        "config_fingerprint": artifact.get("config_fingerprint"),
        "snapshot": str(path.relative_to(root)),
        "snapshot_sha256": snapshot_sha,
        "external_inputs": snapshot,
        "trace_id": (artifact.get("trace") or {}).get("trace_id"),
        "trace_url": (artifact.get("trace") or {}).get("trace_url"),
        "versions": versions,
        "evaluation": {
            name: {
                "score": (
                    item.get("score")
                    if item.get("status") in {"passed", "failed"}
                    and not item.get("skipped")
                    else None
                ),
                "status": item.get("status", "unverified"),
                **{
                    key: item.get("details", {}).get(key)
                    for key in (
                        "threshold",
                        "factual_claim_count",
                        "supported_count",
                        "unsupported_count",
                        "unverified_count",
                        "conflict_count",
                    )
                    if key in item.get("details", {})
                },
            }
            for name in PRODUCT_METRICS
            for item in [evaluation.get(name) or {}]
        },
        "coverage": metrics.paper_flow.model_dump(),
        "citations": citations,
        "judge": "未运行",
        "elapsed_ms": elapsed,
        "downstream_ms": downstream,
        **({"timing_scope": SUMMARY_CHECKPOINT_SCOPE} if summary_checkpoint else {}),
        "stages_ms": stages,
        "browser_ms": {name: (browser or {}).get(name) for name in WORKBENCH_MARKERS},
        "usage": usage,
        "missing": {
            "stages": incomplete_stages,
            "downstream": (
                "workbench_live_inputs"
                if slot.source == "workbench"
                else (
                    "missing_or_invalid_replay_boundary" if downstream is None else None
                )
            ),
            "browser": (
                "replay"
                if slot.source == "replay"
                else ",".join(
                    name
                    for name in WORKBENCH_MARKERS
                    if (browser or {}).get(name) is None
                )
                or None
            ),
        },
        "reasons": reasons,
    }


def summary_rows(rows: list[dict]) -> list[dict]:
    """Summarize individual runs, never percentiles of per-case means."""
    buckets = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        if row.get("timing_scope") == SUMMARY_CHECKPOINT_SCOPE:
            continue
        group = row.get("group") or row["status"]
        buckets[(row["source"], row["profile_id"], group)].append(row)
        if (
            row["source"] == "replay"
            and row.get("external_inputs")
            and row.get("downstream_ms") is not None
        ):
            paired[(row["case_id"], row["repetition"], row["external_inputs"])][
                row["profile_id"]
            ] = row
    summaries = []
    for (source, profile, group), samples in sorted(buckets.items()):
        stats = {}
        series = {
            "total": [row.get("elapsed_ms") for row in samples],
            "downstream": [row.get("downstream_ms") for row in samples],
        }
        for key in WORKBENCH_MARKERS:
            series[key] = [(row.get("browser_ms") or {}).get(key) for row in samples]
        for key, durations in series.items():
            valid = [value for value in durations if value is not None]
            if valid:
                stats[key] = {
                    "n": len(valid),
                    "p50": _percentile_optional(valid, 0.5),
                    "max": max(valid),
                    **(
                        {"p95": _percentile_optional(valid, 0.95)}
                        if len(valid) >= 40
                        else {}
                    ),
                }
        summaries.append(
            {
                "source": source,
                "profile": profile,
                "group": group,
                "n": len(samples),
                "stats": stats,
            }
        )
    comparisons = defaultdict(list)
    for (case, repetition, _), runs in paired.items():
        for left_index, left in enumerate(PROFILES):
            for right in PROFILES[left_index + 1 :]:
                if left in runs and right in runs:
                    a, b = runs[left], runs[right]
                    comparisons[(left, right, a["group"], b["group"])].append(
                        (
                            f"{case}/r{repetition}",
                            b["downstream_ms"] - a["downstream_ms"],
                        )
                    )
    for (left, right, left_outcome, right_outcome), pairs in sorted(
        comparisons.items()
    ):
        differences = [difference for _, difference in pairs]
        summaries.append(
            {
                "source": "paired",
                "profile": f"{right} − {left}",
                "group": f"{left_outcome} → {right_outcome}",
                "n": len(differences),
                "stats": {
                    "delta_ms": {
                        "n": len(differences),
                        "p50": _percentile_optional(differences, 0.5),
                        "max": max(differences),
                        "values": differences,
                        "by_case": dict(pairs),
                    }
                },
            }
        )
    return summaries


def archive_failure_reasons(artifact):
    """Read stable failure codes from a real run without exposing error payloads."""
    writer = ((artifact.get("report") or {}).get("metadata") or {}).get(
        "writer_failure"
    ) or {}
    reasons = []
    if isinstance(writer.get("code"), str):
        reasons.append(writer["code"])
    for event in artifact.get("events") or ():
        data = event.get("data") or {}
        if event.get("event") == "worker.failed" and (
            data.get("error_type") == "LLMRequestTimeout" and data.get("task_id")
        ):
            reasons.append(f"{data['task_id']}_request_timeout")
        if (
            event.get("event") in {"llm.failed", "llm.complete"}
            and data.get("provider_http_status") == 403
            and data.get("provider_error_code") == "AccountOverdueError"
        ):
            reasons.append("AccountOverdueError")
    return list(dict.fromkeys(reasons))


def _workbench_url(row):
    """Link only a run that has an archived artifact for the viewer."""
    if not row.get("run_id") or not row.get("snapshot"):
        return None
    return "/flow-demo?" + urlencode(
        {"run": f"archive:{row['run_id']}", "view": "execution"}
    )


def _format_score(value):
    """Use the same percentage display for all recorded quality measures."""
    return "—" if value is None else f"{value:.2%}"


def _format_seconds(value):
    """Keep the latency boundary explicit and render missing data separately."""
    return "—" if value is None else f"{value / 1000:,.1f}"


def _evaluation_score(metric):
    """Hide placeholders from runs that did not complete the metric."""
    if metric.get("status") in {"failed", "passed"}:
        return metric.get("score")
    return None


def _delivery_view(row, revalidated_ids):
    """Expose original delivery and any independently verified correction."""
    delivery = row.get("delivery")
    if delivery == "ready":
        return "通过", None, "ready"
    if row.get("run_id") in revalidated_ids:
        return "离线复核通过", "原交付因账本绑定被拒", "reviewed"
    reasons = row.get("reasons") or []
    reason = next(
        (
            label
            for code, label in CAUSE_LABELS.items()
            if any(code in r for r in reasons)
        ),
        reasons[0] if reasons else None,
    )
    status = row["status"]
    if status in {"未运行", "failed", "cancelled", "timeout", "timed_out"}:
        return (
            {"failed": "执行失败", "timed_out": "超时"}.get(
                status, STATUS_LABELS.get(status, status)
            ),
            reason,
            "pending",
        )
    return STATUS_LABELS.get(delivery, delivery or "—"), reason, "blocked"


def _row_view(row, revalidated_ids):
    """Project only the metrics that appear in the compact benchmark table."""
    source = "retry" if row.get("retry_of") else row["source"]
    if row.get("timing_scope") == SUMMARY_CHECKPOINT_SCOPE:
        source = SUMMARY_CHECKPOINT_SCOPE
    evaluation = row.get("evaluation") or {}
    faithfulness = evaluation.get("faithfulness") or {}
    support = faithfulness.get("supported_count")
    population = faithfulness.get("factual_claim_count")
    faith = _format_score(_evaluation_score(faithfulness))
    if faith != "—" and support is not None and population:
        faith += f" ({support}/{population})"
    usage = row.get("usage") or {}
    tokens = usage.get("total_tokens")
    tokens_label = "—" if tokens is None else f"{tokens:,}"
    if usage.get("usage_uncertain") and tokens is not None:
        tokens_label += "（部分）"
    if (reconciled := usage.get("reconciled_billed_tokens")) is not None:
        tokens_label = f"{reconciled:,}（已对账）"
    browser = row.get("browser_ms") or {}
    duration = row.get("downstream_ms")
    if source not in {"replay", "retry", SUMMARY_CHECKPOINT_SCOPE}:
        duration = browser.get("final_report")
        if duration is None:
            duration = browser.get("terminal")
    return {
        "case": row["case_id"],
        "configuration": (
            f"{COHORT_LABELS.get(source, source)} · "
            f"{row['profile_id']} · r{row['repetition']}"
        ),
        "delivery": _delivery_view(row, revalidated_ids),
        "metrics": (
            faith,
            _format_score(_evaluation_score(evaluation.get("citation_accuracy") or {})),
            _format_score(
                _evaluation_score(evaluation.get("internal_consistency") or {})
            ),
            _format_seconds(duration),
            tokens_label,
        ),
        "workbench": _workbench_url(row),
    }


def _presentation_state(rows, status, revalidated):
    """Keep recorded, offline and actual retry delivery counts distinct."""
    matrix = [row for row in rows if row["source"] in {"workbench", "replay"}]
    retries = [row for row in rows if row["source"] == "retry"]
    replay_retries = [
        row for row in retries if row.get("timing_scope") != SUMMARY_CHECKPOINT_SCOPE
    ]
    checkpoint_retries = [
        row for row in retries if row.get("timing_scope") == SUMMARY_CHECKPOINT_SCOPE
    ]
    completed = sum(row.get("status") != "未运行" for row in matrix)
    ready = sum(row.get("delivery") == "ready" for row in matrix)
    revalidated_ids = {
        row["run_id"]
        for row in (revalidated or {}).get("rows", [])
        if row["recorded_status"] != "ready" and row["revalidated_status"] == "ready"
    }
    delivery_label = (
        "本轮交付" if any(row.get("retry_of") for row in matrix) else "原始交付"
    )
    counts = [
        f"执行 {completed}/{len(matrix)}",
        f"{delivery_label} {ready}/{len(matrix)}",
    ]
    if status == "dry-run":
        counts.insert(0, "仅预检")
    if revalidated:
        counts.append(
            f"同批归档离线复核 {revalidated['revalidated_ready']}/{len(matrix)}"
        )
    blocked = {row.get("run_id") for row in matrix if row.get("delivery") == "blocked"}
    if replay_retries:
        replay_recovered = {
            row["retry_of"]["run_id"]
            for row in replay_retries
            if row.get("delivery") == "ready"
        }
        improved = len(blocked & replay_recovered)
        counts.append(f"计入真实复跑交付 {ready + improved}/{len(matrix)}")
        counts.append(f"修复复跑 {len(replay_retries)} 次")
    if checkpoint_retries:
        recovered = {
            row["retry_of"]["run_id"]
            for row in retries
            if row.get("delivery") == "ready"
        }
        counts.append(
            f"计入摘要阶段续跑交付 {ready + len(blocked & recovered)}/{len(matrix)}"
        )
        counts.append(f"摘要阶段续跑 {len(checkpoint_retries)} 次")
    diagnostics = len(rows) - len(matrix) - len(retries)
    if diagnostics:
        counts.append(f"诊断复验 {diagnostics} 次")
    return " · ".join(counts), revalidated_ids


def _view_note(rows):
    """Explain the distinct clock only when the table includes a continuation."""
    if any(row.get("timing_scope") == SUMMARY_CHECKPOINT_SCOPE for row in rows):
        return VIEW_NOTE + CHECKPOINT_NOTE
    return VIEW_NOTE


def _summary_cells(summaries):
    """Keep per-cohort latency and paired comparisons available on demand."""
    for row in summaries:
        source = row["source"]
        if source in {"workbench-diagnostic", "retry"}:
            continue
        key = {
            "workbench": "final_report",
            "replay": "downstream",
            "paired": "delta_ms",
        }.get(source)
        stats = row["stats"].get(key, {})
        label = COHORT_LABELS.get(source, "同题配对")
        group = " → ".join(STATUS_LABELS.get(p, p) for p in row["group"].split(" → "))
        by_case = stats.get("by_case") or {}
        yield (
            f"{label} · {row['profile']}",
            group,
            str(stats.get("n", 0)),
            _format_seconds(stats.get("p50")),
            _format_seconds(stats.get("max")),
            _format_seconds(stats.get("p95")),
            "、".join(f"{k} {_format_seconds(v)}" for k, v in by_case.items()) or "—",
        )


SUMMARY_HEADERS = ("配置", "原始交付", "有效样本", "p50", "最大值", "p95", "同题差值")


def _md_row(cells):
    """Render Markdown-safe cells without exposing storage paths."""
    safe = (
        c.replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("|", r"\|")
        .replace("\r", " ")
        .replace("\n", " ")
        for c in cells
    )
    return "| " + " | ".join(safe) + " |"


def render_table(rows, summaries, status, *, revalidated=None):
    """Render one readable Markdown table with a workbench link for every run."""
    counts, corrected = _presentation_state(rows, status, revalidated)
    lines = [
        "# Survey Benchmark",
        "",
        counts + " · Faithfulness ≥ 0.95",
        "",
        _md_row(VIEW_HEADERS),
        _md_row(["---"] * len(VIEW_HEADERS)),
    ]
    for row in rows:
        view = _row_view(row, corrected)
        delivery, reason, _ = view["delivery"]
        lines.append(
            _md_row(
                (
                    f"{view['case']} · {view['configuration']}",
                    (
                        f"[打开]({VIEWER_URL}{view['workbench']})"
                        if view["workbench"]
                        else "—"
                    ),
                    f"{delivery}（{reason}）" if reason else delivery,
                    *view["metrics"],
                )
            )
        )
    lines.extend(["", _view_note(rows), ""])
    if groups := list(_summary_cells(summaries)):
        lines.extend(
            [
                "<details>",
                "<summary>延迟汇总与同题配对差值（秒）</summary>",
                "",
                _md_row(SUMMARY_HEADERS),
                _md_row(["---"] * len(SUMMARY_HEADERS)),
                *(_md_row(cells) for cells in groups),
                "",
                "</details>",
                "",
            ]
        )
    return "\n".join(lines)


def render_benchmark_page(rows, *, status, revalidated=None):
    """Render the local browser overview and link to the normal Workbench."""
    counts, corrected = _presentation_state(rows, status, revalidated)
    output_rows = []
    for row in rows:
        view = _row_view(row, corrected)
        delivery, reason, level = view["delivery"]
        link = view["workbench"]
        subtext = f"<small>{escape(reason)}</small>" if reason else ""
        action = (
            f'<a href="{escape(link)}" aria-label="打开 {escape(row["run_id"])} 工作台">打开 <span aria-hidden="true">↗</span></a>'
            if link
            else "—"
        )
        output_rows.append(
            "<tr>"
            f'<th scope="row">{escape(view["case"])}<small>{escape(view["configuration"])}</small></th>'
            f"<td>{action}</td>"
            f'<td><span class="pill {level}">{escape(delivery)}</span>{subtext}</td>'
            + "".join(f"<td>{escape(metric)}</td>" for metric in view["metrics"])
            + "</tr>"
        )
    cohort_rows = [
        row
        for row in rows
        if row["status"] != "未运行" and row["source"] in {"workbench", "replay"}
    ]
    summary = list(_summary_cells(summary_rows(cohort_rows)))
    summary_html = ""
    if summary:
        heads = "".join(
            f'<th scope="col">{heading}</th>' for heading in SUMMARY_HEADERS
        )
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in cells) + "</tr>"
            for cells in summary
        )
        summary_html = (
            "<details><summary>延迟汇总与同题配对差值（秒）</summary>"
            f'<div class="table-scroll"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div></details>'
        )
    headings = "".join(f'<th scope="col">{label}</th>' for label in VIEW_HEADERS)
    return BENCHMARK_PAGE.format(
        counts=escape(counts),
        headings=headings,
        rows="".join(output_rows),
        notes=escape(_view_note(rows)),
        summary=summary_html,
    )


BENCHMARK_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Survey Benchmark · LitAgent</title>
<style>
:root {{ color-scheme: light; --ink:#17253a; --muted:#566478; --line:#dce3ec; --paper:#fff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:#f4f7fa; color:var(--ink); font:14px/1.55 system-ui,sans-serif; }}
main {{ width:min(1440px,100%); margin:auto; padding:44px clamp(16px,4vw,56px); }}
h1 {{ margin:0; font-size:clamp(28px,4vw,42px); letter-spacing:-.04em; line-height:1.1; }}
header p {{ margin:11px 0 22px; color:var(--muted); }}
.threshold {{ display:inline-block; margin-left:.6rem; color:#164b8c; font-weight:650; white-space:nowrap; }}
.table-scroll {{ overflow:auto; background:var(--paper); border:1px solid var(--line); border-radius:12px; }}
.table-scroll:focus-visible, a:focus-visible, summary:focus-visible {{ outline:2px solid #296cca; outline-offset:2px; }}
table {{ width:100%; min-width:920px; border-collapse:collapse; text-align:left; font-variant-numeric:tabular-nums; }}
caption {{ position:absolute; width:1px; height:1px; overflow:hidden; clip-path:inset(50%); }}
th,td {{ padding:12px 13px; border-bottom:1px solid var(--line); vertical-align:top; }}
thead th {{ background:#eaf0f7; color:#334863; font-size:12px; white-space:nowrap; }}
tbody tr:nth-child(even) {{ background:#f9fbfd; }}
tbody tr:last-child>* {{ border-bottom:0; }}
tbody th {{ font-weight:650; white-space:nowrap; }}
tbody td:nth-child(2), tbody td:nth-child(n+4) {{ white-space:nowrap; }}
small {{ display:block; margin-top:2px; color:var(--muted); font-size:11px; font-weight:400; }}
.pill {{ display:inline-block; padding:1px 7px; border-radius:4px; font-size:12px; font-weight:650; white-space:nowrap; }}
.ready {{ color:#15553b; background:#e3f4e9; }} .reviewed {{ color:#174d8d; background:#e1edfd; }}
.blocked {{ color:#8b3025; background:#faebe7; }} .pending {{ color:#525f72; background:#edf0f4; }}
a {{ color:#1758a2; font-weight:650; text-decoration:none; }} a:hover {{ text-decoration:underline; }}
p.note {{ color:var(--muted); margin:13px 0 19px; font-size:12px; }}
summary {{ cursor:pointer; font-weight:650; width:fit-content; margin:10px 0; }}
</style>
</head>
<body><main>
<header><h1>Survey Benchmark</h1><p>{counts}<span class="threshold">Faithfulness ≥ 0.95</span></p></header>
<div class="table-scroll" role="region" aria-label="按运行列出主要指标" tabindex="0">
<table><caption>每次 Survey 的主要指标</caption><thead><tr>{headings}</tr></thead><tbody>{rows}</tbody></table>
</div>
<p class="note">{notes}</p>
{summary}
</main></body></html>
"""


def publish(
    root: Path, slots: list[RunSlot], observations: dict, status: str, reason: str
):
    """Update the complete table after every observation, including unrun rows."""
    rows = [
        {
            **asdict(slot),
            **observations.get(slot.key, {"status": "未运行", "reasons": [reason]}),
        }
        for slot in slots
    ]
    summaries = summary_rows([row for row in rows if row["status"] != "未运行"])
    write_json(
        root / "benchmark.json",
        {"contract": CONTRACT, "status": status, "rows": rows, "summaries": summaries},
    )
    write_bytes(root / "benchmark.md", render_table(rows, summaries, status).encode())
