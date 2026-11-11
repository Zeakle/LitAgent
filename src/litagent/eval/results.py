"""Project independent evaluation metrics and gates from one frozen execution."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

CONTRACT = "independent-evaluation-v1"
FIELDS = ("evaluation_result_contract", "evaluation_result_identity")
IMPLEMENTATION = "litagent-actual-citation-support-v1"
HISTORICAL_RESULT_IDENTITIES = frozenset(
    {"a4273caa2f6f848615de3ea2a7dcff7ce6244f6e31408ceab84f82d4a1c65b2f"}
)


def historical_checks(details, report_sha256):
    """Read stored checks for a known old execution without recomputing its verdicts."""
    identity = details.get("evaluation_result_identity")
    if type(identity) is not str or identity not in HISTORICAL_RESULT_IDENTITIES:
        return None
    if (
        details.get("evaluation_result_contract") != CONTRACT
        or details.get("source_report_sha256") != report_sha256
    ):
        raise ValueError("evaluation_result_report_identity_invalid")
    checks = details.get("evaluation_checks")
    if checks is None:
        return {}
    if not isinstance(checks, dict) or any(
        not isinstance(item, dict)
        or item.get("status")
        not in {"passed", "failed", "unverified", "error", "not_applicable"}
        for item in checks.values()
    ):
        raise ValueError("evaluation_result_historical_checks_invalid")
    return copy.deepcopy(checks)


@lru_cache(maxsize=1)
def contract_identity() -> str:
    """Bind responsibilities, state semantics and formulas without changing old IDs."""
    from litagent.body_quality import (
        bound_repair_diagnostics,
        report_repair_diagnostics,
    )
    from litagent.eval.responsibilities import (
        FORBIDDEN_CODES,
        GLOBAL_INSTRUCTION,
        parse_global_response,
        project_global_schema,
    )

    value = [
        CONTRACT,
        IMPLEMENTATION,
        GLOBAL_INSTRUCTION,
        sorted(FORBIDDEN_CODES),
        inspect.getsource(report_repair_diagnostics),
        inspect.getsource(bound_repair_diagnostics),
        inspect.getsource(project_global_schema),
        inspect.getsource(parse_global_response),
        inspect.getsource(result),
        inspect.getsource(check),
        inspect.getsource(_legacy_execute_evaluation),
        inspect.getsource(derive_quality),
        inspect.getsource(EvaluationRunResult),
        inspect.getsource(acceptance),
    ]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def selected(envelope) -> bool:
    """Authenticate only explicitly selected new envelopes."""
    if envelope is None:
        return False
    if not isinstance(envelope, dict):
        raise ValueError("evaluation_result_contract_invalid")
    if not any(field in envelope for field in FIELDS):
        return False
    if (
        envelope.get(FIELDS[0]) != CONTRACT
        or envelope.get(FIELDS[1]) != contract_identity()
        or envelope.get("global_findings_contract") != "anchored-global-findings-v2"
        or envelope.get("support_validation_revision") != "host-support-receipts-v2"
        or envelope.get("contract") != "flat-writer-envelope-v1"
    ):
        raise ValueError("evaluation_result_contract_invalid")
    return True


def result(status, *, score=None, reasons=(), **details) -> dict:
    """Use null for unavailable scores and never publish a skipped pass."""
    return {
        "status": status,
        "score": score,
        "passed": status == "passed",
        "skipped": status in {"unverified", "error", "not_applicable"},
        "reason_codes": list(dict.fromkeys(reasons)),
        "details": details,
    }


def check(status, *reasons, **scope) -> dict:
    """Keep checks distinct from numeric metrics."""
    return {"status": status, "reason_codes": list(reasons), **scope}


@dataclass(frozen=True)
class EvaluationRunResult:
    """Own one execution independently of any metric's pass/fail result."""

    report_sha256: str
    details: dict[str, Any]
    execution_error: str | None = None

    @classmethod
    def from_details(cls, details, report_sha256, *, error=None):
        """Copy execution facts and reject cross-report result reuse."""
        if any(field in details for field in FIELDS) and (
            details.get(FIELDS[0]) != CONTRACT
            or details.get(FIELDS[1]) != contract_identity()
        ):
            raise ValueError("evaluation_result_contract_identity_invalid")
        actual = details.get("source_report_sha256")
        if actual is not None and actual != report_sha256:
            raise ValueError("evaluation_result_report_identity_invalid")
        value = copy.deepcopy(details)
        component_errors = {
            name: item.get("reason_codes", [])
            for name, item in value.get("stages", {}).get("global_sections", {}).items()
            if item.get("status") == "error"
        }
        if component_errors:
            value["component_errors"] = component_errors
            value["error"] = "evaluation_contract_invalid"
            value["coverage"] = {**value.get("coverage", {}), "complete": False}
        error = error or value.get("execution_error")
        if error is not None:
            value["execution_error"] = error
            value.setdefault("skipped_reason", error)
        value.update(
            evaluation_result_contract=CONTRACT,
            evaluation_result_identity=contract_identity(),
            source_report_sha256=report_sha256,
        )
        return cls(report_sha256, value, error)

    @property
    def assertions(self):
        """Return canonical host-validated assertions, never raw batch decisions."""
        return self.details.get("assertions") or []

    def execution_complete(self):
        """Require exact, unique claim coverage and every support batch settled."""
        stages = self.details.get("stages") or {}
        rows = stages.get("batches") or []
        ids = [c.get("claim_id") for c in self.assertions]
        actual = [cid for row in rows for cid in row.get("claim_ids", [])]
        return bool(
            not self.execution_error
            and rows
            and ids
            and len(ids) == len(set(ids))
            and stages.get("expected_assertions") == len(ids)
            and stages.get("processed_assertions") == len(ids)
            and stages.get("expected_batches") == len(rows)
            and all(row.get("status") == "completed" for row in rows)
            and len(actual) == len(set(actual))
            and set(actual) == set(ids)
        )

    def inventory_valid(self):
        """Keep per-claim extraction checks and global omission coverage mandatory."""
        d = self.details
        coverage = d.get("coverage") or {}
        section = (d.get("stages", {}).get("global_sections") or {}).get(
            "inventory", {}
        )
        return bool(
            self.execution_complete()
            and d.get("inventory_check", {}).get("status") == "passed"
            and section.get("status", "passed") == "passed"
            and d.get("stages", {}).get("global_check") == "completed"
            and coverage.get("source_units") is not None
            and coverage.get("source_units") == coverage.get("processed_units")
            and all(
                c.get("atomic") is True
                and c.get("classification_valid") is True
                and c.get("source_faithful") is True
                for c in self.assertions
            )
        )

    def metrics(self):
        """Calculate support and consistency with independent necessary dependencies."""
        science = [c for c in self.assertions if c.get("kind") == "scientific"]
        counts = {
            name: sum(c.get("status") == name for c in science)
            for name in ("supported", "unsupported", "unverifiable")
        }
        inventory = self.inventory_valid()
        reasons = []
        if not inventory:
            reasons.append(
                "inventory_invalid"
                if self.execution_complete()
                else "incomplete_verification"
            )
        if not science:
            reasons.append("no_scientific_assertions")
        if any(c.get("status") not in {"supported", "unsupported"} for c in science):
            reasons.append("scientific_assertions_unverified")
        score = counts["supported"] / len(science) if not reasons else None
        status = (
            ("error" if self.execution_error else "unverified")
            if reasons
            else ("passed" if score >= self.details.get("threshold", 0.8) else "failed")
        )
        support = result(
            status,
            score=score,
            reasons=reasons,
            implementation=IMPLEMENTATION,
            display_name="LitAgent 实际引用支持率",
            threshold=self.details.get("threshold", 0.8),
            factual_claim_count=len(science),
            supported_count=counts["supported"],
            unsupported_count=counts["unsupported"],
            unverified_count=counts["unverifiable"],
        )
        sections = self.details.get("stages", {}).get("global_sections", {})
        conflict_state = sections.get("conflicts", {"status": "passed"})
        conflicts = self.details.get("conflicts") or []
        if not science:
            consistency = result("unverified", reasons=["no_scientific_assertions"])
        elif not inventory:
            consistency = result(
                "error" if self.execution_error else "unverified",
                reasons=["inventory_invalid"],
            )
        elif conflict_state.get("status") != "passed":
            consistency = result(
                conflict_state.get("status", "unverified"),
                reasons=conflict_state.get("reason_codes", []),
            )
        else:
            value = max(0.0, 1 - 0.2 * len(conflicts))
            consistency = result(
                "passed" if value >= 0.8 else "failed",
                score=value,
                threshold=0.8,
                penalty=0.2,
                conflict_count=len(conflicts),
                conflicts=conflicts,
            )
        consistency["details"].setdefault("threshold", 0.8)
        support["details"] = {**self.details, **support["details"]}
        support["details"]["evaluation_checks"] = self.checks()
        return {"faithfulness": support, "internal_consistency": consistency}

    def checks(self):
        """Expose each gate without using another metric as its execution state."""
        d, claims = self.details, self.assertions
        stages = d.get("stages") or {}
        global_sections = stages.get("global_sections") or {}
        supported = {
            c.get("claim_id")
            for c in claims
            if c.get("kind") == "scientific" and c.get("status") == "supported"
        }
        requirements, rows = (
            d.get("coverage_requirements") or [],
            d.get("core_coverage") or [],
        )
        coverage_valid = bool(
            requirements
            and len(rows) == len(requirements)
            and len({r.get("requirement_id") for r in rows}) == len(rows)
            and {r.get("requirement_id") for r in rows}
            == {r.get("requirement_id") for r in requirements}
        )
        coverage_passed = coverage_valid and all(
            r.get("answered") is True
            and r.get("claim_ids")
            and set(r["claim_ids"]).issubset(supported)
            for r in rows
        )
        out = {
            "execution": check(
                (
                    "passed"
                    if self.execution_complete() and not d.get("component_errors")
                    else (
                        "error"
                        if self.execution_error or d.get("component_errors")
                        else "unverified"
                    )
                ),
                *(
                    [self.execution_error]
                    if self.execution_error
                    else (
                        ["evaluation_contract_invalid"]
                        if d.get("component_errors")
                        else []
                    )
                ),
                processed_assertions=stages.get("processed_assertions"),
                expected_assertions=stages.get("expected_assertions"),
            ),
            "inventory": check(
                (
                    "passed"
                    if self.inventory_valid()
                    else (
                        "failed"
                        if self.execution_complete()
                        and d.get("inventory_check", {}).get("status") == "failed"
                        else "unverified"
                    )
                ),
                *([] if self.inventory_valid() else ["inventory_invalid"]),
            ),
            "core_coverage": check(
                (
                    "passed"
                    if coverage_passed
                    else "failed" if coverage_valid else "unverified"
                ),
                *([] if coverage_passed else ["core_coverage_unverified"]),
                answered=sum(
                    r.get("answered") is True
                    and bool(r.get("claim_ids"))
                    and set(r["claim_ids"]).issubset(supported)
                    for r in rows
                ),
                expected=len(requirements),
            ),
        }
        for name, field in (
            ("structure", "structure_checks"),
            ("method", "method_checks"),
            ("language", "language_checks"),
            ("analysis", "analysis_check"),
        ):
            value = copy.deepcopy(d.get(field) or {"status": "unverified"})
            out[name] = value
        for kind, name in (
            ("runtime_method", "method_statements"),
            ("report_structure", "organization"),
        ):
            items = [c for c in claims if c.get("kind") == kind]
            state = (
                "not_applicable"
                if not items and self.inventory_valid()
                else (
                    "unverified"
                    if not self.inventory_valid()
                    or any(c.get("status") == "unverifiable" for c in items)
                    else (
                        "failed"
                        if any(c.get("status") != "supported" for c in items)
                        else "passed"
                    )
                )
            )
            out[name] = check(state, total=len(items))
        issues = d.get("global_issues") or []
        out["global_report"] = check(
            (
                "failed"
                if any(i.get("owner") == "report" for i in issues)
                else "passed" if self.inventory_valid() else "unverified"
            ),
            findings=issues,
        )
        limitations = copy.deepcopy(d.get("limitations_check") or {})
        out["limitations"] = check(
            (
                "passed"
                if limitations.get("adequate") is True
                else "failed" if "adequate" in limitations else "unverified"
            ),
            **limitations,
        )
        for key, target in (
            ("coverage", "core_coverage"),
            ("analyses", "analysis"),
            ("limitations", "limitations"),
            ("inventory", "inventory"),
        ):
            if global_sections.get(key, {}).get("status") == "error":
                out[target] = copy.deepcopy(global_sections[key])
        return out


async def execute_evaluation(*, llm, survey, context, config):
    """Run the scientific pipeline once and preserve failures independently."""
    from litagent.eval.ragas_eval import _checkpoint_failure
    from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped

    report_sha = hashlib.sha256(survey.encode()).hexdigest()
    try:
        details = await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=config
        )
        details.pop("score", None)
        return EvaluationRunResult.from_details(details, report_sha)
    except ScopedEvaluationError as exc:
        _checkpoint_failure(exc, survey, context)
        return EvaluationRunResult.from_details(exc.details, report_sha, error=str(exc))


def acceptance(details):
    """Calculate strict delivery solely from canonical checks and metric facts."""
    run = EvaluationRunResult.from_details(details, details["source_report_sha256"])
    checks, failures = run.checks(), []
    metrics = run.metrics()
    support, consistency = metrics["faithfulness"], metrics["internal_consistency"]
    if (
        support["status"] not in {"passed", "failed"}
        or checks["execution"]["status"] != "passed"
    ):
        failures.append("incomplete_verification")
    if support["score"] != 1.0:
        failures.append("science_not_fully_supported")
    if checks["inventory"]["status"] != "passed":
        failures.append("inventory_invalid")
    if consistency["status"] not in {"passed", "failed"} or consistency["details"].get(
        "conflict_count"
    ):
        failures.append("global_inconsistency")
    names = {
        "core_coverage": "core_coverage_unverified",
        "analysis": "body_analysis_incomplete",
        "structure": "structure_checks",
        "method": "method_checks",
        "language": "language_checks",
        "method_statements": "method_assertion_errors",
        "organization": "organization_errors",
        "global_report": "global_inconsistency",
        "limitations": "scientific_limitations_missing",
    }
    for name, reason in names.items():
        if checks[name].get("status") not in {"passed", "not_applicable"}:
            failures.append(reason)
    if not details.get("fact_card_map"):
        failures.append("fact_cards_unavailable")
    return {
        "policy": details.get("quality_policy"),
        "status": "failed" if failures else "passed",
        "science_required_ratio": 1.0,
        "failure_codes": list(dict.fromkeys(failures)),
        "human_approval_required": details.get("quality_policy")
        not in {"strict_auto_v1", "strict_auto_v2"},
    }


def derive_quality(evaluation):
    """Keep completed failures and unavailable prerequisites separately visible."""
    details = evaluation["faithfulness"]["details"]
    gate = acceptance(details)
    failed, unverified = [], []
    for name in ("citation_accuracy", "faithfulness", "internal_consistency"):
        state = evaluation.get(name, {}).get("status", "unverified")
        if state == "failed":
            failed.append(name)
        elif state != "passed":
            unverified.append(name)
    checks = EvaluationRunResult.from_details(
        details, details["source_report_sha256"]
    ).checks()
    for name, item in checks.items():
        if item.get("status") == "failed":
            failed.append(name)
        elif item.get("status") not in {"passed", "not_applicable"}:
            unverified.append(name)
    if gate["status"] != "passed" and not failed and not unverified:
        failed.extend(gate["failure_codes"])
    details["strict_acceptance"] = gate
    return {
        "status": "failed" if failed else "unverified" if unverified else "passed",
        "failed_metrics": failed,
        "unverified_metrics": unverified,
        "strict_acceptance": gate,
    }


def public_summary(report):
    """Expose only statuses, scores and counts; omit prose, private IDs and raw IO."""
    evaluation = report.get("evaluation") or {}
    details = evaluation.get("faithfulness", {}).get("details", {})
    if details.get("evaluation_result_contract") != CONTRACT:
        return None
    sha = hashlib.sha256(report.get("survey", "").encode()).hexdigest()
    checks = historical_checks(details, sha)
    if checks is None:
        checks = EvaluationRunResult.from_details(details, sha).checks()
    metrics = {
        name: {
            "status": item.get("status", "unverified"),
            "score": item.get("score"),
            "threshold": item.get("details", {}).get("threshold"),
            "reason_codes": item.get("reason_codes", []),
        }
        for name, item in evaluation.items()
        if name in {"citation_accuracy", "faithfulness", "internal_consistency"}
    }
    history = (report.get("metadata") or {}).get("evaluation_history") or {}
    from litagent.eval.ragas_repair import sanitize_recovery

    recovery = (
        sanitize_recovery(history.get("recovery"))
        if history.get("source_report_sha256") == sha
        else None
    )
    return {
        "contract": CONTRACT,
        "report_sha256": sha,
        **({"recovery": recovery} if recovery else {}),
        "metrics": metrics,
        "checks": {
            name: {
                "status": item.get("status", "unverified"),
                "reason_codes": item.get("reason_codes", []),
            }
            for name, item in checks.items()
        },
        "science": {
            key: details.get(key)
            for key in (
                "factual_claim_count",
                "supported_count",
                "unsupported_count",
                "unverified_count",
            )
        },
        "scope": {
            key: details.get("coverage", {}).get(key)
            for key in ("source_units", "processed_units")
        },
    }


def sanitize_summary(value):
    """Whitelist a stored summary before adding it to a public benchmark bundle."""
    import math
    import re

    if not isinstance(value, dict) or value.get("contract") != CONTRACT:
        return None
    sha = value.get("report_sha256")
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-f]{64}", sha) is None:
        return None
    states = {"passed", "failed", "unverified", "error", "not_applicable"}
    metrics = {"citation_accuracy", "faithfulness", "internal_consistency"}
    checks = {
        "execution",
        "inventory",
        "core_coverage",
        "structure",
        "method",
        "language",
        "analysis",
        "method_statements",
        "organization",
        "global_report",
        "limitations",
    }
    out = {"contract": CONTRACT, "report_sha256": sha}
    for group, allowed in (("metrics", metrics), ("checks", checks)):
        source = value.get(group)
        out[group] = {}
        if not isinstance(source, dict):
            continue
        for name, item in source.items():
            if name not in allowed or not isinstance(item, dict):
                continue
            row = {
                "status": (
                    item.get("status") if item.get("status") in states else "unverified"
                ),
                "reason_codes": [
                    code
                    for code in (item.get("reason_codes") or [])
                    if isinstance(code, str)
                    and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code)
                ],
            }
            if group == "metrics":
                for field in ("score", "threshold"):
                    number = item.get(field)
                    row[field] = (
                        number
                        if type(number) in (int, float)
                        and math.isfinite(number)
                        and 0 <= number <= 1
                        else None
                    )
            out[group][name] = row
    for group, fields in (
        (
            "science",
            (
                "factual_claim_count",
                "supported_count",
                "unsupported_count",
                "unverified_count",
            ),
        ),
        ("scope", ("source_units", "processed_units")),
    ):
        source = value.get(group) if isinstance(value.get(group), dict) else {}
        out[group] = {
            key: (
                source.get(key)
                if type(source.get(key)) is int and source[key] >= 0
                else None
            )
            for key in fields
        }
    from litagent.eval.ragas_repair import sanitize_recovery

    if recovery := sanitize_recovery(value.get("recovery")):
        out["recovery"] = recovery
    return out


def summary_tables(rows):
    """Render the same sanitized result into bilingual Markdown and HTML tables."""
    import html

    entries = []
    for row in rows:
        summary = row.get("evaluation_summary")
        if not isinstance(summary, dict) or summary.get("contract") != CONTRACT:
            continue
        if recovery := summary.get("recovery"):
            remaining = recovery.get("remaining_by_category")
            balance = (
                " / ".join(
                    f"{kind} {remaining[kind]}"
                    for kind in ("capacity", "inventory", "prose")
                )
                if remaining
                else str(recovery.get("remaining_attempts", "—"))
            )
            entries.append(
                [
                    row.get("case_id", ""),
                    "修复 / Recovery · " + recovery["action"],
                    recovery["reassessment"],
                    balance + " remaining / 剩余",
                    recovery["outcome_reason"] or recovery["decision_reason"] or "—",
                ]
            )
        for group in ("metrics", "checks"):
            for name, item in summary.get(group, {}).items():
                score = item.get("score")
                entries.append(
                    [
                        row.get("case_id", ""),
                        name,
                        item.get("status", "unverified"),
                        "—" if score is None else str(score),
                        ", ".join(item.get("reason_codes") or []) or "—",
                    ]
                )
    if not entries:
        return "", ""
    title = "独立指标与检查 / Independent metrics and checks"
    header = [
        "Case",
        "指标/检查 Metric/check",
        "状态 Status",
        "分数 Score",
        "原因 Reason",
    ]
    markdown = (
        "\n## " + title + "\n\n|" + "|".join(header) + "|\n|---|---|---|---|---|\n"
    )
    markdown += "".join(
        "|"
        + "|".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row)
        + "|\n"
        for row in entries
    )
    document = '<section class="scroll"><h2>' + title + "</h2><table><thead><tr>"
    document += "".join(
        '<th scope="col">' + html.escape(value) + "</th>" for value in header
    )
    document += "</tr></thead><tbody>" + "".join(
        "<tr>"
        + "".join("<td>" + html.escape(str(value)) + "</td>" for value in row)
        + "</tr>"
        for row in entries
    )
    return markdown, document + "</tbody></table></section>"


# Keep the historical execution source as an identity input for old envelopes.
_legacy_execute_evaluation = execute_evaluation


async def execute_evaluation(*, llm, survey, context, config):
    """Select native RAGAS only from the server-frozen budget owner."""
    from litagent.eval.engine_policy import validate_engine
    from litagent.safety.planned_budget import locked_writer_envelope

    envelope = locked_writer_envelope(getattr(llm, "_cost_budget", None))
    engine = (envelope or {}).get("semantic_engine")
    if engine is None:
        return await _legacy_execute_evaluation(
            llm=llm, survey=survey, context=context, config=config
        )
    validate_engine(engine)
    from litagent.eval.ragas_native import execute_native

    return await execute_native(llm=llm, survey=survey, context=context, config=config)
