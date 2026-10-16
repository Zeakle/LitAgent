"""Run the fixed automatic quality matrix with durable no-replay accounting."""

from __future__ import annotations

import asyncio
import copy
import csv
import hashlib
import html
import io
import json
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.survey_runner import (
    SurveyBenchmarkRunner,
    runtime_prompt_source_fingerprint,
)
from litagent.contracts import build_config_summary
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import ReportAuditService, atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.workbench import RunOptions, verification_enabled

PILOT_IDS = ("few-shot-metric", "vit-data-efficient", "nerf-foundations")
LIMITS = {"survey": 500000, "judge": 100000, "jev": 200000}
REQUEST_LIMITS = {"survey": 64, "judge": 2, "jev": 20}


def _technical_failure(failures: list[str]) -> str | None:
    """Stop invalid transport and resource failures while retaining semantic failures."""
    protocol_errors = {
        "strict_native_repair_path_invalid",
        "strict_native_repair_contract_invalid",
        "strict_native_repair_diagnostic_invalid",
        "strict_native_repair_dependency_invalid",
        "strict_native_repair_source_identity_invalid",
        "strict_native_repair_binding_invalid",
        "strict_native_repair_response_invalid",
        "strict_native_repair_patch_invalid",
        "strict_native_repair_usage_incomplete",
        "strict_targeted_repair_state_invalid",
        "strict_targeted_repair_checkpoint_conflict",
        "strict_targeted_repair_restart_no_replay",
        "strict_targeted_repair_limit_exceeded",
        "strict_writing_passage_source_invalid",
        "strict_writing_passage_fields_invalid",
        "strict_writing_passage_binding_invalid",
        "strict_writing_passage_range_invalid",
        "strict_writing_passage_empty",
        "strict_writing_summary_binding_invalid",
        "strict_repair_passage_fields_invalid",
        "strict_repair_passage_binding_invalid",
        "strict_repair_summary_passage_not_in_body",
        "strict_repair_summary_body_unavailable",
        "strict_repair_body_binding_invalid",
        "evaluation_decision_consistency_invalid",
        "strict_writing_duplicate_object_key",
        "evaluation_inventory_shape_invalid",
        "evaluation_inventory_duplicate_key",
        "evaluation_inventory_unit_coverage_invalid",
        "evaluation_inventory_fragment_coverage_invalid",
        "evaluation_inventory_host_partition_invalid",
        "evaluation_inventory_claim_empty",
        "evaluation_source_fragment_invalid",
        "evaluation_citation_ownership_invalid",
        "evaluation_duplicate_assertion",
        "evaluation_unit_exclusion_unexplained",
        "evaluation_unit_exclusion_conflict",
        "evaluation_assessment_unit_coverage_invalid",
        "evaluation_decision_coverage_invalid",
        "evaluation_global_unit_coverage_invalid",
        "evaluation_global_issue_identity_invalid",
        "evaluation_core_coverage_invalid",
        "evaluation_analysis_coverage_invalid",
        "evaluation_conflict_identity_invalid",
        "evaluation_batch_conflicts_not_allowed",
        "evaluation_decision_status_invalid",
        "evaluation_host_methodology_invalid",
        "evaluation_usage_incomplete",
        "evaluation_assertion_capacity_exceeded",
        "evaluation_batch_capacity_exceeded",
    }
    legacy_terms = (
        "extraction_verification_incomplete",
        "extraction_receipt_invalid",
        "extraction_interrupted_no_replay",
        "extraction_decision_invalid",
        "extraction_usage_incomplete",
        "extraction_fields_invalid",
        "budget",
        "usage_unknown",
        "identity",
        "model_mismatch",
        "authentication",
        "provider",
        "contract_invalid",
        "audit_error",
        "missing_key",
        "automatic_error:",
        "automatic_source_fidelity_invalid",
        "automatic_inventory_invalid",
        "automatic_original_support_missing",
        "jev_http_",
        "jev_response_invalid",
        "jev_timeout",
        "jev_connection_error",
        "jev_transport_error",
        "jev_request_error",
        "automatic_judge_usage_unknown",
        "evaluation_contract_invalid",
        "evaluation_timeout",
        "evaluation_provider_error",
    )
    # Valid semantic judgments can include "invalid"; do not classify by suffix.
    return next(
        (
            reason
            for reason in failures
            if reason in protocol_errors or any(term in reason for term in legacy_terms)
        ),
        None,
    )


def _candidate_evaluation_failures(metadata: dict) -> tuple[list[str], list[dict]]:
    """Keep rejected candidate failures visible when the original report is retained."""
    candidates = [("metadata.evidence_rewrite", metadata.get("evidence_rewrite"))]
    attempts = metadata.get("evidence_rewrite_attempts")
    if isinstance(attempts, list):
        candidates.extend(
            (f"metadata.evidence_rewrite_attempts.{index}", item)
            for index, item in enumerate(attempts)
        )
    codes, locations = [], []
    for path, candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        body = candidate.get("candidate")
        sha = (
            hashlib.sha256(body.encode()).hexdigest() if isinstance(body, str) else None
        )

        def record(reason, location):
            """Retain every private location while deduplicating public reason codes."""
            if isinstance(reason, str) and reason:
                codes.append(reason)
                locations.append(
                    {
                        "path": location,
                        "reason_code": reason,
                        "candidate_sha256": sha,
                    }
                )

        reason = candidate.get("reason_code")
        if isinstance(reason, str) and _technical_failure([reason]):
            record(reason, f"{path}.reason_code")
        validation = candidate.get("validation_reason_codes")
        if isinstance(validation, list):
            for index, reason in enumerate(validation):
                record(reason, f"{path}.validation_reason_codes.{index}")
        evaluation = candidate.get("candidate_evaluation")
        if not isinstance(evaluation, dict):
            continue
        for metric, result in evaluation.items():
            if not isinstance(result, dict) or not isinstance(
                result.get("details"), dict
            ):
                continue
            for field in ("skipped_reason", "unverified_reason"):
                record(
                    result["details"].get(field),
                    f"{path}.candidate_evaluation.{metric}.details.{field}",
                )
    return list(dict.fromkeys(codes)), locations


@contextmanager
def suite_lock(path):
    """Lock one suite across local processes on Unix and Windows."""
    with path.open("a+b") as stream:
        try:
            import fcntl
        except ImportError:
            import msvcrt

            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            yield


class QualitySuiteResult(BaseModel):
    """Keep automatic evaluation distinct from historical human-calibrated baselines."""

    model_config = ConfigDict(extra="forbid")
    benchmark_type: Literal["survey_quality"] = "survey_quality"
    schema_version: Literal[1] = 1
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,100}$")
    stage: Literal["quality"] = "quality"
    status: str
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    judgment_status: str
    self_judged: bool = True
    human_calibration: Literal["unverified"] = "unverified"
    formal_eligible: bool = False
    annotation_source: str
    binding: str
    planned: list[dict]
    observations: list[dict] = Field(default_factory=list)
    counts: dict = Field(default_factory=dict)
    budgets: dict = Field(default_factory=dict)
    stop_reasons: list[str] = Field(default_factory=list)
    prior_suite_id: str | None = None
    prior_suite_sha256: str | None = None
    prior_rounds: int = Field(default=0, ge=0, le=18)
    max_new_observations: int | None = Field(default=None, ge=1, le=18)
    resume_unrun: bool = False
    inherited_observations: int = Field(default=0, ge=0, le=18)
    scientific_accuracy: None = None
    false_acceptance_rate: None = None
    false_rejection_rate: None = None


def summarize(result: dict) -> dict:
    """Count every planned formal case, retaining science failures and unrun cases."""
    formal = [o for o in result["observations"] if o["phase"] == "formal"]
    pilot = [o for o in result["observations"] if o["phase"] == "pilot"]

    def counts(items, planned):
        passed = sum(o.get("acceptance", {}).get("publishable") is True for o in items)
        complete = sum(o.get("complete_verification") is True for o in items)
        pending = sum(
            o.get("status") in {"started_no_replay", "queued", "running"} for o in items
        )
        terminal = len(items) - pending
        return {
            "planned": planned,
            "run": len(items),
            "complete_verification": complete,
            "automatic_passed": passed,
            "pending": pending,
            "terminal": terminal,
            "blocked": terminal - passed,
            "not_run": planned - len(items),
            "pass_rate_planned": passed / planned if planned else None,
            "pass_rate_completed": passed / terminal if terminal else None,
        }

    return {
        "pilot": counts(
            pilot,
            (
                sum(p["phase"] == "pilot" for p in result["planned"])
                if "planned" in result
                else 3
            ),
        ),
        "formal": counts(
            formal,
            (
                sum(p["phase"] == "formal" for p in result["planned"])
                if "planned" in result
                else 15
            ),
        ),
        "goal_met": len(formal) == 15
        and all(o.get("acceptance", {}).get("publishable") is True for o in formal),
    }


def public_projection(result: dict) -> dict:
    """Exclude query text, body, private paths, raw responses and runtime records."""
    keys = (
        "benchmark_type",
        "planned",
        "schema_version",
        "run_id",
        "stage",
        "status",
        "dataset_id",
        "dataset_version",
        "judgment_status",
        "self_judged",
        "human_calibration",
        "annotation_source",
        "counts",
        "budgets",
        "stop_reasons",
        "prior_suite_id",
        "prior_rounds",
        "max_new_observations",
        "resume_unrun",
        "inherited_observations",
        "scientific_accuracy",
        "false_acceptance_rate",
        "false_rejection_rate",
    )
    output = {key: result.get(key) for key in keys}
    output["observations"] = [
        {
            key: row.get(key)
            for key in (
                "phase",
                "case_id",
                "domain",
                "status",
                "gates",
                "counts",
                "costs",
                "elapsed_ms",
                "complete_verification",
                "failure_codes",
                "report_sha256",
            )
        }
        for row in result["observations"]
    ]
    for row in output["observations"]:
        gates = dict(row.get("gates") or {})
        for service in ("judge", "jev"):
            if (row.get("costs") or {}).get(service, {}).get("requests") == 0:
                disabled = gates.get(service) == "disabled" or (
                    service == "jev" and gates.get("extraction_jev") == "disabled"
                )
                gates[service] = "disabled" if disabled else "not_run"
        row["gates"] = gates
        row["jev_stage"] = "extraction" if "extraction_jev" in gates else "report"
    output["service_usage"] = {
        service: {
            "accounted_tokens": (
                sum(
                    row.get("costs", {}).get(service, {}).get("tokens", 0)
                    for row in result["observations"]
                )
                if all(
                    row.get("costs", {}).get(service, {}).get("usage_complete") is True
                    and type(row.get("costs", {}).get(service, {}).get("tokens")) is int
                    for row in result["observations"]
                )
                else None
            ),
            "known_requests": sum(
                (row.get("costs", {}).get(service, {}).get("requests") or 0)
                for row in result["observations"]
            ),
            "usage_complete": all(
                row.get("costs", {}).get(service, {}).get("usage_complete") is True
                for row in result["observations"]
            ),
        }
        for service in LIMITS
    }
    return output


def export_bundle(root: Path, result: dict) -> None:
    """Render four presentation formats from one sanitized projection."""
    data = public_projection(result)
    atomic_json(root / "benchmark.json", data)
    columns = [
        "phase",
        "case_id",
        "domain",
        "jev_stage",
        "status",
        "evaluation",
        "judge",
        "jev",
        "report_sha256",
        "elapsed_ms",
        "failure_codes",
        *[
            f"{service}_{field}"
            for service in LIMITS
            for field in ("tokens", "requests", "usage_complete")
        ],
    ]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    lines = [
        "# 严格自动验收 Benchmark",
        "",
        "自动评测 · self_judged · 人工校准未验证。",
        "",
        f"正式套件：{data['counts']['formal']['automatic_passed']}/15 自动交付；实际运行 {data['counts']['formal']['run']}/15。",
        f"试点：{data['counts']['pilot']['automatic_passed']}/{data['counts']['pilot']['planned']} 自动交付；实际运行 {data['counts']['pilot']['run']}/{data['counts']['pilot']['planned']}。",
        f"停止原因：{', '.join(data['stop_reasons']) or '无'}。",
        "用量不完整时，tokens 是包含未知请求预留的保守计账，不是精确实际消耗或费用。",
        "",
        "| 阶段 | 题目 | Evaluation | Judge | Jev | 交付 |",
        "|---|---|---|---|---|---|",
    ]
    rows = []
    for planned in result["planned"]:
        item = next(
            (
                o
                for o in data["observations"]
                if o["phase"] == planned["phase"] and o["case_id"] == planned["case_id"]
            ),
            {**planned, "status": "not_run"},
        )
        gates = item.get("gates") or {}
        if "extraction_jev" in gates:
            gates = {**gates, "jev": gates["extraction_jev"]}
        csv_row = {**item, **gates}
        csv_row["failure_codes"] = ";".join(item.get("failure_codes") or [])
        for service in LIMITS:
            for field in ("tokens", "requests", "usage_complete"):
                csv_row[f"{service}_{field}"] = (
                    (item.get("costs") or {}).get(service, {}).get(field)
                )
        writer.writerow({key: csv_row.get(key) for key in columns})
        cells = [
            item["phase"],
            item["case_id"],
            *(gates.get(k, "unverified") for k in ("evaluation", "judge", "jev")),
            item["status"],
        ]
        lines.append(
            "| " + " | ".join(str(c).replace("|", "\\|") for c in cells) + " |"
        )
        rows.append(
            "<tr>"
            + "".join("<td>" + html.escape(str(c)) + "</td>" for c in cells)
            + "</tr>"
        )
    (root / "benchmark.csv").write_text(stream.getvalue(), encoding="utf-8")
    lines.extend(
        [
            "",
            "Jev stage=extraction 时只核验提取事实；正文由 Evaluation / Judge 核验。历史 stage=report 表示终稿复核。",
            "",
            "## 服务用量（保守计账）",
            "",
            "| 服务 | tokens | 已知请求数 | 用量完整 |",
            "|---|---:|---:|---|",
            *[
                f"| {service} | {usage['accounted_tokens'] if usage['accounted_tokens'] is not None else '不可用 / unavailable'} | {usage['known_requests']} | {usage['usage_complete']} |"
                for service, usage in data["service_usage"].items()
            ],
        ]
    )
    (root / "benchmark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    encoded = json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
    formal = data["counts"]["formal"]
    pilot = data["counts"]["pilot"]
    stop_summary = html.escape(" · ".join(data["stop_reasons"]) or "无 / None")
    document = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Strict Automatic Benchmark</title><style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem;background:#f7f8fa;color:#192535}}h1{{font-size:1.8rem}}section{{background:white;border:1px solid #ccd4de;border-radius:12px;padding:1rem;margin:1rem 0}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.7rem;text-align:left;border-bottom:1px solid #ddd}}svg{{width:100%;max-width:600px;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}button{{padding:.6rem}}@media(max-width:420px){{body{{margin:0;padding:.6rem}}h1{{font-size:1.4rem}}}}</style><button id="locale">English / 中文</button><h1 id="title">严格自动验收 Benchmark</h1><p id="note">自动评测 · self_judged · 人工校准未验证；科学准确率、误放行率和误拒绝率不可用。</p><section><h2>运行状态 / Run status</h2><p id="pilot-summary">试点 / Pilot: {pilot['run']}/{pilot['planned']} 已运行 / run · {pilot['automatic_passed']}/{pilot['planned']} 自动交付 / passed · {pilot['not_run']} 未运行 / not run</p><p id="stop-summary">{stop_summary}</p></section><section><h2 id="overview">正式套件</h2><p id="formal-summary">{formal['automatic_passed']}/15 自动交付 · {formal['run']}/15 已运行 · {formal['not_run']} 未运行</p><svg viewBox="0 0 600 90" role="img" aria-label="Passed {formal['automatic_passed']} of 15 planned cases"><rect x="0" y="10" width="600" height="20" fill="#dce3ec"/><rect x="0" y="10" width="{formal['automatic_passed'] * 40}" height="20" fill="#287a60"/><text x="0" y="65">{formal['automatic_passed']} passed / 15 planned</text></svg></section><section class="scroll"><table><thead><tr><th>阶段 / Phase</th><th>题目 / Case</th><th>Evaluation</th><th>Judge</th><th>Jev</th><th>交付 / Delivery</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section><section><h2>统计与服务预算 / Statistics and budgets</h2><p>用量不完整时，tokens 包含未知请求的保守预留；预算 charged 不是精确费用。 / Incomplete usage includes conservative reservations; charged budget is not exact cost.</p><pre id="stats"></pre></section><script id="data" type="application/json">{encoded}</script><script>const d=JSON.parse(document.getElementById('data').textContent);document.getElementById('stats').textContent=JSON.stringify({{counts:d.counts,service_usage:d.service_usage,budgets:d.budgets,stop_reasons:d.stop_reasons}},null,2);let en=false;document.getElementById('locale').onclick=()=>{{en=!en;document.documentElement.lang=en?'en':'zh-CN';document.getElementById('title').textContent=en?'Strict Automatic Benchmark':'严格自动验收 Benchmark';document.getElementById('overview').textContent=en?'Formal suite':'正式套件';const f=d.counts.formal;document.getElementById('formal-summary').textContent=en?(f.automatic_passed+'/15 automatically delivered · '+f.run+'/15 run · '+f.not_run+' not run'):(f.automatic_passed+'/15 自动交付 · '+f.run+'/15 已运行 · '+f.not_run+' 未运行');document.getElementById('note').textContent=en?'Automatic evaluation · self_judged · no human calibration. Accuracy, false acceptance and false rejection rates unavailable.':'自动评测 · self_judged · 人工校准未验证；科学准确率、误放行率和误拒绝率不可用。';}};</script></html>"""
    (root / "benchmark.html").write_text(document, encoding="utf-8")


class QualitySuiteRunner:
    """Own the fixed 3+15 sequence, cumulative reservations and terminal exports."""

    def __init__(
        self,
        config,
        dataset,
        profile,
        *,
        root: Path,
        run_archive: ArchiveRepository,
        suite_id: str,
        executor=None,
        prior_suite_id: str | None = None,
        resume_unrun: bool = False,
        quality_policy: str = "strict_auto_v2",
        run_options: RunOptions | None = None,
    ):
        """Allow transport-free execution fixtures without relaxing production gates."""
        if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", suite_id) is None:
            raise ValueError("quality_suite_id_invalid")
        if prior_suite_id is not None and (
            re.fullmatch(r"[A-Za-z0-9_-]{1,100}", prior_suite_id) is None
            or prior_suite_id == suite_id
        ):
            raise ValueError("quality_continuation_invalid")
        if profile.profile_id != "external-fulltext" or len(dataset.cases) != 15:
            raise ValueError("quality_fixed_matrix_invalid")
        if type(resume_unrun) is not bool or (resume_unrun and not prior_suite_id):
            raise ValueError("quality_resume_parent_required")
        self.config, self.dataset, self.profile = config, dataset, profile
        if quality_policy not in {"strict_auto_v1", "strict_auto_v2"}:
            raise ValueError("quality_policy_invalid")
        self.quality_policy = quality_policy
        if run_options is not None and run_options.quality_policy != quality_policy:
            raise ValueError("quality_options_policy_mismatch")
        self.run_options = run_options
        self.round_limit = 18
        self.service_limits = dict(LIMITS)
        if run_options is not None:
            selected = run_options.model_dump()
            for service in ("judge", "jev"):
                if not verification_enabled(selected, service):
                    self.service_limits[service] = 0
        self.root, self.archive, self.suite_id = root, run_archive, suite_id
        self.executor = executor
        self.prior_suite_id = prior_suite_id
        self.resume_unrun = resume_unrun
        self.parent = None
        self.parent_sha = None
        if prior_suite_id:
            raw = (root / f"{prior_suite_id}.json").read_bytes()
            self.parent = QualitySuiteResult.model_validate_json(raw).model_dump(
                mode="json"
            )
            self.parent_sha = hashlib.sha256(raw).hexdigest()
            if (
                self.parent["run_id"] != prior_suite_id
                or self.parent["status"] not in {"completed", "stopped"}
                or self.parent["dataset_fingerprint"] != dataset.fingerprint
            ):
                raise ValueError("quality_parent_not_terminal_or_incompatible")
        self.service = ReportAuditService(
            config, root=root / "quality-private" / suite_id / "audits"
        )
        self.state_path = root / "checkpoints" / f"{suite_id}.json"
        identity = {
            "dataset": dataset.fingerprint,
            "profile": profile.fingerprint,
            "config": build_config_summary(config, quality_policy),
            "prompts": runtime_prompt_source_fingerprint(),
            "policy": quality_policy,
            "prior_suite_id": prior_suite_id,
            "prior_suite_sha256": self.parent_sha,
            "automatic_sources": {
                name: hashlib.sha256(
                    (Path(__file__).parents[1] / name).read_bytes()
                ).hexdigest()
                for name in (
                    "quality.py",
                    "eval/automatic.py",
                    "eval/batched.py",
                    "safety/budget.py",
                    "benchmark/citation_audit.py",
                    "benchmark/quality_suite.py",
                )
            },
        }
        if run_options is not None:
            identity["run_options"] = run_options.model_dump(mode="json")
        self.verification_contract = digest(
            {
                key: value
                for key, value in identity.items()
                if not key.startswith("prior_")
            }
        )
        if quality_policy == "strict_auto_v2":
            identity["extraction_verification_source"] = hashlib.sha256(
                (Path(__file__).parents[1] / "extraction_verification.py").read_bytes()
            ).hexdigest()
            self.verification_contract = digest(
                {
                    key: value
                    for key, value in identity.items()
                    if not key.startswith("prior_")
                }
            )
        if resume_unrun:
            identity["resume_unrun"] = True
            self._validate_resume_parent()
        self.binding = digest(identity)

    def _validate_resume_parent(self):
        """Require known terminal slots and an intact, non-reset campaign ledger."""
        parent = self.parent
        by_id = {c.case_id: c for c in self.dataset.cases}
        expected = [
            {"phase": phase, "case_id": c.case_id, "domain": c.domain}
            for phase, cases in (
                ("pilot", [by_id[cid] for cid in PILOT_IDS]),
                ("formal", self.dataset.cases),
            )
            for c in cases
        ]
        rows = parent["observations"]
        if parent["planned"] != expected or parent["prior_rounds"] + len(rows) > 18:
            raise ValueError("quality_resume_plan_invalid")
        for index, row in enumerate(rows):
            if {key: row.get(key) for key in expected[index]} != expected[
                index
            ] or row.get("status") == "started_no_replay":
                raise ValueError("quality_resume_slot_invalid")
            for service, limit in LIMITS.items():
                cost = row.get("costs", {}).get(service, {})
                if (
                    cost.get("usage_complete") is not True
                    or type(cost.get("tokens")) is not int
                    or not 0 <= cost["tokens"] <= limit
                    or type(cost.get("requests")) is not int
                    or not 0 <= cost["requests"] <= REQUEST_LIMITS[service]
                ):
                    raise ValueError("quality_resume_usage_unknown_or_invalid")
        for service, limit in LIMITS.items():
            budget = parent["budgets"].get(service, {})
            if (
                budget.get("per_round") != limit
                or budget.get("aggregate") != limit * 18
                or type(budget.get("charged")) is not int
                or not sum(r["costs"][service]["tokens"] for r in rows)
                <= budget["charged"]
                <= limit * 18
                or type(budget.get("requests")) is not int
                or not sum(r["costs"][service]["requests"] for r in rows)
                <= budget["requests"]
                <= REQUEST_LIMITS[service] * 18
            ):
                raise ValueError("quality_resume_budget_invalid")

    def initial(self):
        """Include unrun formal cases before the first paid request."""
        by_id = {c.case_id: c for c in self.dataset.cases}
        planned = [
            {"phase": phase, "case_id": c.case_id, "domain": c.domain}
            for phase, cases in (
                ("pilot", [by_id[cid] for cid in PILOT_IDS]),
                ("formal", self.dataset.cases),
            )
            for c in cases
        ]
        return QualitySuiteResult(
            run_id=self.suite_id,
            status="planned",
            dataset_id=self.dataset.dataset_id,
            dataset_version=self.dataset.dataset_version,
            dataset_fingerprint=self.dataset.fingerprint,
            judgment_status=self.dataset.judgment_status,
            annotation_source=self.dataset.judgment_status,
            binding=self.binding,
            prior_suite_id=self.prior_suite_id,
            prior_suite_sha256=self.parent_sha,
            prior_rounds=(
                self.parent.get("prior_rounds", 0)
                + (0 if self.resume_unrun else len(self.parent["observations"]))
                if self.parent
                else 0
            ),
            planned=planned,
            resume_unrun=self.resume_unrun,
            inherited_observations=(
                len(self.parent["observations"]) if self.resume_unrun else 0
            ),
            observations=(
                copy.deepcopy(self.parent["observations"]) if self.resume_unrun else []
            ),
            budgets={
                service: {
                    "per_round": limit,
                    "aggregate": limit * self.round_limit,
                    "charged": (
                        self.parent["budgets"][service]["charged"] if self.parent else 0
                    ),
                    "requests": (
                        max(
                            self.parent["budgets"][service]["requests"],
                            sum(
                                (o.get("costs") or {}).get(service, {}).get("requests")
                                or 0
                                for o in self.parent["observations"]
                            ),
                        )
                        if self.parent
                        else 0
                    ),
                }
                for service, limit in self.service_limits.items()
            },
        ).model_dump(mode="json")

    def save(self, state):
        """Commit checkpoint and presentation together after each terminal observation."""
        state["counts"] = summarize(state)
        atomic_json(self.state_path, state)
        atomic_json(self.root / f"{self.suite_id}.json", state)
        bundle = self.root / "presentation" / self.suite_id
        bundle.mkdir(parents=True, exist_ok=True)
        export_bundle(bundle, state)

    async def run(self, *, dry_run=False, max_new_observations: int | None = None):
        """Stop technical failures, continue science failures and never retry a slot."""
        if max_new_observations is not None and (
            type(max_new_observations) is not int or not 1 <= max_new_observations <= 18
        ):
            raise ValueError("quality_observation_limit_invalid")
        self.root.mkdir(parents=True, exist_ok=True)
        with suite_lock(self.root / f".{self.suite_id}.lock"):
            state = (
                json.loads(self.state_path.read_text())
                if self.state_path.exists()
                else self.initial()
            )
            if state["binding"] != self.binding:
                if (
                    state["status"] == "planned"
                    and not state["observations"]
                    and all(b["charged"] == 0 for b in state["budgets"].values())
                ):
                    state = self.initial()
                else:
                    raise ValueError("quality_suite_identity_conflict")
            persisted_limit = state.get("max_new_observations")
            if (
                persisted_limit is not None
                and max_new_observations is not None
                and persisted_limit != max_new_observations
            ):
                raise ValueError("quality_observation_limit_conflict")
            observation_limit = persisted_limit or max_new_observations
            if observation_limit is not None:
                state["max_new_observations"] = observation_limit
            if dry_run:
                return {**state, "counts": summarize(state)}
            if state["status"] in {"completed", "stopped"}:
                return state
            if self.parent:
                # A terminal parent has one explicit child. Further continuations
                # must reference that child, preserving every charge in the chain.
                owners = self.root / "continuations"
                owners.mkdir(exist_ok=True)
                with suite_lock(owners / f".{self.prior_suite_id}.lock"):
                    current_sha = hashlib.sha256(
                        (self.root / f"{self.prior_suite_id}.json").read_bytes()
                    ).hexdigest()
                    if current_sha != self.parent_sha:
                        raise ValueError("quality_parent_identity_changed")
                    path = owners / f"{self.prior_suite_id}.json"
                    expected = {"child": self.suite_id, "parent_sha": self.parent_sha}
                    if path.exists() and json.loads(path.read_text()) != expected:
                        raise ValueError("quality_continuation_already_owned")
                    atomic_json(path, expected)
            if any(
                o.get("status") == "started_no_replay" for o in state["observations"]
            ):
                state.update(
                    status="stopped",
                    stop_reasons=["quality_interrupted_no_replay_usage_unknown"],
                )
                self.save(state)
                return state
            if self.executor is None:
                caps = self.service.capabilities()
                selected = (
                    self.run_options.model_dump()
                    if self.run_options is not None
                    else {}
                )
                if (
                    verification_enabled(selected, "judge")
                    and not caps["judge_available"]
                ) or (
                    verification_enabled(selected, "jev") and not caps["jev_available"]
                ):
                    state.update(
                        status="stopped",
                        stop_reasons=[caps["judge_reason"] or caps["jev_reason"]],
                    )
                    self.save(state)
                    return state
                preparer = SurveyBenchmarkRunner(
                    base_config=self.config,
                    judge_config=self.service.judge_config(),
                    artifacts=BenchmarkArtifactRepository(self.root),
                    run_archive=self.archive,
                )
                try:
                    await preparer._prepare_corpora(self.dataset, [self.profile])
                except Exception as exc:
                    state.update(
                        status="stopped",
                        stop_reasons=[
                            f"quality_corpus_preparation_failed:{type(exc).__name__}"
                        ],
                    )
                    self.save(state)
                    return state
                effective = preparer._config_for(
                    self.profile, self.dataset.dataset_version
                )
            else:
                effective = self.config
            await self.service.initialize()
            try:
                for index, planned in enumerate(state["planned"]):
                    if index < len(state["observations"]):
                        continue
                    if (
                        observation_limit is not None
                        and len(state["observations"])
                        - state.get("inherited_observations", 0)
                        >= observation_limit
                    ):
                        state.update(
                            status="stopped",
                            stop_reasons=["quality_requested_observation_limit"],
                        )
                        break
                    if (
                        state["prior_rounds"] + len(state["observations"])
                        >= self.round_limit
                    ):
                        state.update(
                            status="stopped",
                            stop_reasons=["quality_campaign_round_limit"],
                        )
                        break
                    if (
                        index == 3
                        and not state.get("formal_eligible")
                        and not any(
                            o.get("complete_verification")
                            and (
                                not self.resume_unrun
                                or o.get("verification_contract")
                                == self.verification_contract
                            )
                            for o in state["observations"]
                        )
                    ):
                        state.update(
                            status="stopped",
                            stop_reasons=["pilot_no_completed_judge_jev_contract"],
                        )
                        break
                    if any(
                        state["budgets"][service]["charged"] + limit
                        > limit * self.round_limit
                        for service, limit in self.service_limits.items()
                    ):
                        state.update(
                            status="stopped",
                            stop_reasons=["quality_aggregate_budget_exceeded"],
                        )
                        break
                    for service, limit in self.service_limits.items():
                        state["budgets"][service]["charged"] += limit
                    run_id = f"{self.suite_id}-{planned['phase']}-{planned['case_id']}"
                    entry = {
                        **planned,
                        "run_id": run_id,
                        "status": "started_no_replay",
                        "verification_contract": self.verification_contract,
                    }
                    state["observations"].append(entry)
                    state["status"] = "running"
                    self.save(state)
                    case = next(
                        c for c in self.dataset.cases if c.case_id == planned["case_id"]
                    )
                    try:
                        result = await (
                            self.executor(case, effective, run_id)
                            if self.executor
                            else self._live(case, effective, run_id)
                        )
                        entry.update(result)
                        technical = result.get("technical_failure")
                        for service, limit in self.service_limits.items():
                            cost = result.get("costs", {}).get(service, {})
                            actual = cost.get("tokens")
                            requests = cost.get("requests")
                            # Known transport counts remain meaningful even when token
                            # accounting is incomplete; only token refunds need usage.
                            if type(requests) is int and requests >= 0:
                                state["budgets"][service]["requests"] += requests
                            if (
                                actual is None
                                or requests is None
                                or cost.get("usage_complete") is not True
                            ):
                                technical = technical or "quality_usage_unknown"
                            elif actual > limit or requests > (
                                REQUEST_LIMITS[service] if limit else 0
                            ):
                                technical = technical or "quality_budget_exceeded"
                            else:
                                state["budgets"][service]["charged"] += actual - limit
                        if technical:
                            state.update(status="stopped", stop_reasons=[technical])
                            break
                    except asyncio.CancelledError:
                        state.update(
                            status="stopped",
                            stop_reasons=["quality_cancelled_no_replay"],
                        )
                        raise
                    except Exception as exc:
                        entry.update(
                            status="technical_failed",
                            failure_codes=[
                                f"quality_system_error:{type(exc).__name__}"
                            ],
                        )
                        state.update(
                            status="stopped", stop_reasons=entry["failure_codes"]
                        )
                        break
                    finally:
                        self.save(state)
                else:
                    state["status"] = "completed"
                self.save(state)
                return state
            finally:
                await self.service.shutdown()

    async def _live(self, case, config, run_id):
        """Pass only the research question into generation; labels are post-hoc only."""
        recorder = RunRecorder(run_id, case.query, self.archive)
        recorder.set_citation_style("ieee")
        recorder.set_config_summary(build_config_summary(config, self.quality_policy))
        started = time.monotonic()
        tracer = (
            LangFuseTracer(
                config.observability.langfuse_host,
                os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=run_id,
            )
            if config.observability.enabled
            else None
        )
        hook = CompositeTraceHook(
            recorder,
            (
                RedactingTraceHook(tracer, payload_mode="metadata_only")
                if tracer
                else None
            ),
        )
        try:
            async with LitAgent(
                config,
                trace_hook=hook,
                run_policy=RunPolicy.isolated_benchmark(),
                audit_service=self.service,
            ) as agent:
                agent._session_id = run_id
                report = await agent.run(
                    case.query,
                    options=self.run_options
                    or RunOptions(
                        quality_policy=self.quality_policy,
                        report_language="zh-CN",
                        review_structure="academic_v2",
                    ),
                )
                survey_budget = agent._cost_budget.snapshot()
        except BaseException as exc:
            if tracer:
                recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
                tracer.flush()
            recorder.finalize(
                error=f"quality_survey_failed:{type(exc).__name__}",
                terminal_status="failed",
            )
            raise
        if tracer:
            recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            tracer.flush()
        recorder.finalize(report=report, terminal_status="completed")
        path = self.archive.path_for(run_id)
        acceptance = report.get("metadata", {}).get("automatic_acceptance", {})
        audit = (
            await self.service.wait(acceptance["audit_id"])
            if acceptance.get("audit_id")
            else {}
        )
        judge_usage = audit.get("judge_usage") or {}
        extraction = report.get("metadata", {}).get("extraction_verification") or {}
        jev_result = (
            extraction.get("jev", {})
            if self.quality_policy == "strict_auto_v2"
            else audit.get("jev", {})
        )
        jev_usage = jev_result.get("usage") or {}
        details = (
            report.get("evaluation", {}).get("faithfulness", {}).get("details", {})
        )
        costs = {
            "survey": {
                "tokens": survey_budget["used_tokens"],
                "requests": survey_budget.get(
                    "requests_sent", survey_budget["call_count"]
                ),
                "usage_complete": not survey_budget.get("usage_uncertain", False),
            },
            "judge": {
                "tokens": judge_usage.get("total_tokens", 0),
                "requests": judge_usage.get("call_count", 0),
                "usage_complete": judge_usage.get("usage_complete", not audit),
            },
            "jev": {
                "tokens": jev_usage.get("input_tokens", 0),
                "requests": jev_usage.get("call_count", 0),
                "usage_complete": jev_usage.get(
                    "usage_complete",
                    (
                        not extraction
                        if self.quality_policy == "strict_auto_v2"
                        else jev_result.get("status")
                        in {None, "blocked", "not_requested"}
                    ),
                ),
            },
        }
        evaluation_reason = details.get("skipped_reason") or details.get(
            "unverified_reason"
        )
        failures = (
            ([evaluation_reason] if evaluation_reason else [])
            + acceptance.get("failure_codes", [])
            + audit.get("reason_codes", [])
            + jev_result.get("reason_codes", [])
            + ([extraction["reason_code"]] if extraction.get("reason_code") else [])
        )
        # A candidate can fail before commitment. Its protocol/budget failure
        # still stops the campaign even though the retained original is readable.
        candidate_failures, diagnostic_failures = _candidate_evaluation_failures(
            report.get("metadata", {})
        )
        failures += candidate_failures
        technical = _technical_failure(failures)
        if survey_budget.get("usage_uncertain"):
            technical = "quality_usage_unknown"
        elif survey_budget.get("stop_reason"):
            technical = survey_budget["stop_reason"]
        elif report.get("partial"):
            technical = technical or "quality_worker_failed"
        return {
            "status": "delivered" if acceptance.get("publishable") else "blocked",
            "acceptance": acceptance,
            "gates": acceptance.get("gates", {}),
            "counts": {
                key: details.get(key)
                for key in ("supported_count", "unsupported_count", "unverified_count")
            },
            "costs": costs,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
            "complete_verification": (
                (
                    details.get("coverage", {}).get("complete") is True
                    and details.get("stages", {}).get("global_check") == "completed"
                    and (
                        not verification_enabled(self.run_options.model_dump(), "judge")
                        or audit.get("judge", {}).get("status") == "completed"
                    )
                    and (
                        not verification_enabled(self.run_options.model_dump(), "jev")
                        or jev_result.get("status") == "completed"
                    )
                )
                if self.run_options is not None
                else audit.get("judge", {}).get("status") == "completed"
                and jev_result.get("status") == "completed"
            ),
            "failure_codes": failures,
            "diagnostic_failures": diagnostic_failures,
            "technical_failure": technical,
            "report_sha256": hashlib.sha256(
                report.get("survey", "").encode()
            ).hexdigest(),
            "artifact_path": str(path),
            "artifact_sha256": "sha256:"
            + hashlib.sha256(path.read_bytes()).hexdigest(),
            "judgment": {
                "source": self.dataset.judgment_status,
                "reference_papers": case.relevant_paper_ids,
            },
            "trace_local": str(path),
            "audit_id": acceptance.get("audit_id"),
        }
