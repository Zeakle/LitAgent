"""Project independent evaluation metrics and gates from one frozen execution."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

CONTRACT = "independent-evaluation-v1"
FIELDS = ("evaluation_result_contract", "evaluation_result_identity")
IMPLEMENTATION = "litagent-actual-citation-support-v1"
FULL_REVIEW_IDENTITY = (
    "567a9318619118471476bca87c005714784ae56670c176d071041145980108f8"
)
HISTORICAL_RESULT_IDENTITIES = frozenset(
    {"a4273caa2f6f848615de3ea2a7dcff7ce6244f6e31408ceab84f82d4a1c65b2f"}
)


def evaluation_run_identity(
    run_id, survey, phase, *, engine_identity=None, inventory_recovery=None
):
    """Bind request IDs to a report, evaluation phase and optional recovery."""
    from litagent.quality import digest

    identity = digest([run_id, survey, phase])
    if inventory_recovery is not None:
        identity = digest([identity, inventory_recovery])
    if engine_identity is not None:
        identity = digest([identity, engine_identity])
    return identity


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


@lru_cache(maxsize=3)
def contract_identity(global_scope=None) -> str:
    """Bind responsibilities, state semantics and formulas without changing old IDs."""
    if global_scope is None:
        # The full-review contract remains versioned at its published identity.
        return FULL_REVIEW_IDENTITY
    if global_scope == "question-coverage-v1":
        return "938db468a626ef977d3a2274f8c50a486236147b037b9cfb349e5771fb335783"
    if global_scope != "report-review-no-omissions-v1":
        raise ValueError("evaluation_global_scope_invalid")
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
        global_scope,
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
        inspect.getsource(question_only),
        inspect.getsource(required_metrics),
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
        or envelope.get(FIELDS[1])
        != contract_identity(
            (envelope.get("semantic_engine") or {}).get("global_scope")
        )
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


def scope_identity(details):
    """Use the same scope for result projection and the frozen budget owner."""
    return contract_identity(details.get("stages", {}).get("global_scope"))


def question_only(details):
    """Read the scope recorded by the native engine for this evaluation."""
    return (
        details.get("evaluation_version") == "survey-evaluation-ragas-v1"
        and details.get("stages", {}).get("global_scope") == "question-coverage-v1"
    )


def required_metrics(details):
    """Return the scored metrics required for this report version."""
    if question_only(details):
        return ("citation_accuracy", "faithfulness")
    return ("citation_accuracy", "faithfulness", "internal_consistency")


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
            or details.get(FIELDS[1]) != scope_identity(details)
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
            evaluation_result_identity=scope_identity(details),
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
        """Check complete accounting and the inventory verdict for the frozen scope."""
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
        measured_conflicts_required = (
            self.details.get("stages", {}).get("global_scope")
            == "report-review-no-omissions-v1"
        )
        conflict_state = sections.get(
            "conflicts",
            {"status": "unverified" if measured_conflicts_required else "passed"},
        )
        conflicts = self.details.get("conflicts") or []
        if question_only(self.details):
            consistency = result(
                "not_applicable", reasons=["outside_global_review_scope"]
            )
        elif not science:
            consistency = result("unverified", reasons=["no_scientific_assertions"])
        elif not inventory:
            consistency = result(
                "error" if self.execution_error else "unverified",
                reasons=["inventory_invalid"],
            )
        elif measured_conflicts_required and not isinstance(
            self.details.get("conflicts"), list
        ):
            consistency = result("unverified", reasons=["conflict_results_unavailable"])
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
        if not question_only(self.details):
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
        if d.get("inventory_check", {}).get("scope") == "processed_assertions":
            out["inventory"]["scope"] = "processed_assertions"
        if question_only(d):
            for name in ("global_report", "analysis", "limitations"):
                out[name] = check("not_applicable", "outside_global_review_scope")
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
    if not question_only(details) and (
        consistency["status"] not in {"passed", "failed"}
        or consistency["details"].get("conflict_count")
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
    for name in required_metrics(details):
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


def sanitize_request_preflight(value):
    """Keep only numeric, non-sensitive facts about a global preflight rejection."""
    if (
        not isinstance(value, dict)
        or value.get("code") != "global_request_preflight_exceeded"
    ):
        return None
    fields = ("reservation", "context_limit", "request_cap")
    if any(type(value.get(key)) is not int or value[key] < 1 for key in fields):
        return None
    if value["reservation"] <= min(value["context_limit"], value["request_cap"]):
        return None
    if value.get("stage") != "global" or value.get("unexecuted_stages") != [
        "assessment",
        "global",
    ]:
        return None
    return {
        "code": "global_request_preflight_exceeded",
        "stage": "global",
        **{key: value[key] for key in fields},
        "unexecuted_stages": ["assessment", "global"],
    }


# Only report-local, machine-issued identifiers may enter public exports.
_GATE_REASONS = frozenset(
    """incomplete_verification science_not_fully_supported inventory_invalid
    global_inconsistency core_coverage_unverified body_analysis_incomplete
    structure_checks method_checks language_checks method_assertion_errors
    organization_errors scientific_limitations_missing fact_cards_unavailable""".split()
)


def failure_explanation(report):
    """Explain dependent chapter failures without changing the archived gate."""
    from litagent.body_quality import ROLES
    from litagent.review_structure import structure_sections

    survey = report.get("survey")
    evaluation = report.get("evaluation")
    metric = evaluation.get("faithfulness") if isinstance(evaluation, dict) else None
    details = metric.get("details") if isinstance(metric, dict) else None
    gate = details.get("strict_acceptance") if isinstance(details, dict) else None
    if not isinstance(gate, dict):
        return None
    codes = gate.get("failure_codes")
    if (
        gate.get("status") != "failed"
        or details.get("evaluation_result_contract") != CONTRACT
        or details.get("evaluation_result_identity") != scope_identity(details)
        or not isinstance(survey, str)
        or hashlib.sha256(survey.encode()).hexdigest()
        != details.get("source_report_sha256")
        or not isinstance(codes, list)
        or not codes
        or any(not isinstance(code, str) or code not in _GATE_REASONS for code in codes)
    ):
        return None
    value = {"primary": list(codes), "analysis_dependencies": {}}
    checks = details.get("evaluation_checks") or {}
    if not isinstance(checks, dict):
        return value
    required = [checks.get(k) for k in ("execution", "inventory")]
    if any(not isinstance(check, dict) for check in required):
        return value
    if metric.get("score") is None and any(
        check.get("status") != "passed" for check in required
    ):
        value["primary"] = [
            "science_support_unverified" if c == "science_not_fully_supported" else c
            for c in codes
        ]
        return value
    analysis = checks.get("analysis")
    if (
        not {"science_not_fully_supported", "body_analysis_incomplete"}.issubset(codes)
        or any(check.get("status") != "passed" for check in required)
        or not isinstance(analysis, dict)
        or analysis.get("status") != "failed"
        or metric.get("score") is None
    ):
        return value

    claims = details.get("assertions") or []
    rows = analysis.get("sections") or []
    if (
        not isinstance(claims, list)
        or any(
            not isinstance(c, dict) or not isinstance(c.get("claim_id"), str)
            for c in claims
        )
        or not isinstance(rows, list)
        or any(
            not isinstance(r, dict) or not isinstance(r.get("section_key"), str)
            for r in rows
        )
    ):
        return value
    by_id = {c["claim_id"]: c for c in claims}
    if (
        len(by_id) != len(claims)
        or len(rows) != len(ROLES)
        or {r.get("section_key") for r in rows} != set(ROLES)
    ):
        return value
    titles = [dict(structure_sections("academic_v2", lang)) for lang in ("en", "zh-CN")]
    dependent = {}
    for row in rows:
        if row.get("adequate") is True:
            continue
        section = row["section_key"]
        host = row.get("host_validation") or {}
        ids = row.get("claim_ids") or []
        if (
            not isinstance(host, dict)
            or not isinstance(host.get("allowed_claim_ids"), list)
            or row.get("adequate") is not False
            or row.get("model_adequate") is not True
            or host.get("status") != "failed"
            or host.get("failure_codes") != ["analysis_claim_not_supported"]
            or not isinstance(ids, list)
            or not ids
            or any(not isinstance(cid, str) for cid in ids)
            or len(ids) != len(set(ids))
        ):
            continue
        unsupported = []
        for cid in ids:
            c = by_id.get(cid, {})
            if (
                cid not in host.get("allowed_claim_ids", ())
                or c.get("kind") != "scientific"
                or c.get("section") not in {t[section] for t in titles}
                or c.get("status") not in {"supported", "unsupported"}
            ):
                break
            if c["status"] == "unsupported":
                start, end = c.get("source_start"), c.get("source_end")
                if (
                    not re.fullmatch(r"claim_[0-9]{1,12}", cid)
                    or type(start) is not int
                    or type(end) is not int
                    or not 0 <= start < end <= len(survey)
                    or survey[start:end] != c.get("source_quote")
                ):
                    break
                unsupported.append(cid)
        else:
            if unsupported:
                dependent[section] = unsupported
    value["analysis_dependencies"] = dependent
    if dependent and all(
        row.get("adequate") is True or row["section_key"] in dependent for row in rows
    ):
        value["primary"].remove("body_analysis_incomplete")
    return value


def _public_failure_explanation(value):
    """Whitelist a stored explanation without admitting prose or source IDs."""
    from litagent.body_quality import ROLES

    if not isinstance(value, dict):
        return None
    primary, deps = value.get("primary"), value.get("analysis_dependencies")
    if (
        not isinstance(primary, list)
        or not primary
        or any(
            not isinstance(code, str)
            or (code not in _GATE_REASONS and code != "science_support_unverified")
            for code in primary
        )
        or not isinstance(deps, dict)
        or (deps and "science_not_fully_supported" not in primary)
        or any(
            key not in ROLES
            or not isinstance(ids, list)
            or not ids
            or any(
                not isinstance(cid, str) or not re.fullmatch(r"claim_[0-9]{1,12}", cid)
                for cid in ids
            )
            for key, ids in deps.items()
        )
    ):
        return None
    return {
        "primary": list(primary),
        "analysis_dependencies": {key: list(ids) for key, ids in deps.items()},
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
    from litagent.review_cycles import public_projection

    workflow = public_projection(
        (report.get("metadata") or {}).get("review_workflow"), sha
    )
    preflight = sanitize_request_preflight(
        details.get("stages", {}).get("capacity_admission", {}).get("rejection")
    )
    explanation = failure_explanation(report)
    return {
        "contract": CONTRACT,
        "report_sha256": sha,
        **({"failure_explanation": explanation} if explanation else {}),
        **({"request_preflight": preflight} if preflight else {}),
        **({"review_workflow": workflow} if workflow else {}),
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
    if explanation := _public_failure_explanation(value.get("failure_explanation")):
        out["failure_explanation"] = explanation
    if preflight := sanitize_request_preflight(value.get("request_preflight")):
        out["request_preflight"] = preflight
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
    from litagent.review_cycles import public_projection

    if workflow := public_projection(value.get("review_workflow"), sha):
        out["review_workflow"] = workflow
    return out


def summary_tables(rows):
    """Render the same sanitized result into bilingual Markdown and HTML tables."""
    import html

    entries = []
    for row in rows:
        summary = row.get("evaluation_summary")
        if not isinstance(summary, dict) or summary.get("contract") != CONTRACT:
            continue
        if explanation := _public_failure_explanation(
            summary.get("failure_explanation")
        ):
            for code in explanation["primary"]:
                affected = ""
                if code == "science_not_fully_supported":
                    affected = ", ".join(
                        f"{section}: {', '.join(ids)}"
                        for section, ids in explanation["analysis_dependencies"].items()
                    )
                entries.append(
                    [
                        row.get("case_id", ""),
                        "主要原因 / Primary reason",
                        code,
                        "—",
                        (
                            f"受影响章节 / Affected sections: {affected}"
                            if affected
                            else "—"
                        ),
                    ]
                )
        if preflight := sanitize_request_preflight(summary.get("request_preflight")):
            entries.append(
                [
                    row.get("case_id", ""),
                    "全局请求预检超限 / Global request preflight exceeded",
                    "未核验 / Unverified",
                    preflight["reservation"],
                    f"上限 / limit {min(preflight['context_limit'], preflight['request_cap'])}; "
                    "支持核验及全局检查未执行 / NLI and global review not sent; "
                    "预留非费用 / reservation, not billed usage",
                ]
            )
        if workflow := summary.get("review_workflow"):
            for kind, quota in workflow["allowances"].items():
                entries.append(
                    [
                        row.get("case_id", ""),
                        kind,
                        "已用 / used",
                        quota["used"],
                        f'{quota["remaining"]} remaining / 剩余',
                    ]
                )
            for review in workflow["reviews"]:
                entries.append(
                    [
                        row.get("case_id", ""),
                        f'写作审阅 / Review {review["round"] + 1}',
                        review["status"],
                        "—",
                        review["source_report_sha256"],
                    ]
                )
            for evaluation in workflow["evaluations"]:
                label = {
                    "repair_preflight": "修复预检／未发送 / Repair preflight · not sent",
                    "repair_incomplete": "修复终止／未复评 / Repair ended · not reevaluated",
                }.get(
                    evaluation.get("kind"),
                    f'完整评估 / Evaluation {evaluation["round"]}',
                )
                entries.append(
                    [
                        row.get("case_id", ""),
                        label,
                        evaluation["status"],
                        "—",
                        evaluation["reason"] or "—",
                    ]
                )
            for attempt in [*workflow["reviews"], *workflow["evaluations"]]:
                from litagent.review_cycles import (
                    patch_rejection_projection,
                    summary_rejection_projection,
                )
                from litagent.stage_repair import public_rejection

                stage_rejection = public_rejection(
                    attempt.get("patch_rejection")
                ) or public_rejection(attempt.get("summary_rejection"))
                if stage_rejection:
                    action = {
                        "repair_summary": "仅修复摘要 / Repair summary only",
                        "repair_response": "重做授权响应 / Repair authorized response",
                        "stop": "停止续修 / Stop repair",
                    }[stage_rejection["action"]]
                    entries.append(
                        [
                            row.get("case_id", ""),
                            "候选校验失败 / Candidate validation failed",
                            action,
                            "—",
                            f'{stage_rejection["reason_code"]}; {stage_rejection["remaining"]} remaining; '
                            f'{stage_rejection["actual_tokens"]} settled tokens; {stage_rejection["retained_report_sha256"]}',
                        ]
                    )
                    continue
                patch = patch_rejection_projection(attempt.get("patch_rejection"))
                if patch is not None:
                    action = {
                        "evaluate_retained": "保留稿继续核验 / Evaluate retained draft",
                        "retry_evaluation_revision": "继续验收修订 / Continue bounded repair",
                        "stop": "次数耗尽，无法续行 / No remaining attempts",
                    }[patch["action"]]
                    entries.append(
                        [
                            row.get("case_id", ""),
                            "补丁候选被拒 / Patch candidate rejected",
                            action,
                            "—",
                            f'{patch["retained_report_sha256"]}; '
                            f'{patch["remaining"]} remaining / 剩余; '
                            f'{patch["actual_tokens"]} settled tokens / 已结算 tokens',
                        ]
                    )
                    for issue in patch["diagnostics"]:
                        entries.append(
                            [
                                row.get("case_id", ""),
                                issue["target_key"],
                                "candidate_rejected",
                                "—",
                                f'{issue["target_id"]} · {issue["action"]} · {issue["reason"]}',
                            ]
                        )
                rejection = summary_rejection_projection(
                    attempt.get("summary_rejection")
                )
                if not rejection:
                    continue
                action = {
                    "evaluate_retained": "保留稿继续核验 / Evaluate retained draft",
                    "retry_evaluation_revision": "继续验收修订 / Continue bounded repair",
                    "stop": "次数耗尽，无法续行 / No remaining attempts",
                }[rejection["action"]]
                entries.append(
                    [
                        row.get("case_id", ""),
                        "摘要候选被拒 / Summary candidate rejected",
                        action,
                        "—",
                        f'{rejection["retained_report_sha256"]}; '
                        f'{rejection["remaining"]} remaining / 剩余; '
                        f'{rejection["actual_tokens"]} settled tokens / 已结算 tokens',
                    ]
                )
                for issue in rejection["diagnostics"]:
                    reason = {
                        "unknown_frozen_card": "卡号不在冻结证据池 / Unknown frozen card",
                        "outside_body_dependencies": "超出正文依赖 / Outside body dependencies",
                    }.get(issue.get("reason"), issue["code"])
                    entries.append(
                        [
                            row.get("case_id", ""),
                            "/".join(map(str, issue["path"])),
                            "candidate_rejected",
                            "—",
                            f'{issue.get("reference", "—")} · {reason} · offset {issue.get("offset", "—")}',
                        ]
                    )
            entries.append(
                [
                    row.get("case_id", ""),
                    "最终交付 / Delivery",
                    "passed" if workflow["publishable"] else "blocked",
                    "—",
                    workflow["report_sha256"],
                ]
            )
        if recovery := summary.get("recovery"):
            remaining = recovery.get("remaining_by_category")
            balance = (
                " / ".join(f"{kind} {remaining[kind]}" for kind in remaining)
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
