"""Run exactly three frozen-evidence writing cases, with durable no-replay accounting."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.frozen_writing import CASES, frozen_options, read_bundle, run_frozen
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.writing_v2 import CONTRACT, classify_failure
from scripts.run_stage_model_quality import check_models

VERSION = "frozen-writing-three-domain-v1"
LIMITS = {
    "cases": 3,
    "per_case_tokens": 500000,
    "per_case_requests": 64,
    "total_tokens": 1500000,
    "total_requests": 192,
    "repairs": 2,
    "retries": 0,
    "formal_cases": 0,
}
DISABLED_PROFILE = "disabled_v1"
THINKING_PROFILE = "writer_reviewer_thinking_v1"
LOW_THINKING_PROFILE = "writer_reviewer_low_thinking_v1"
LEGACY_RESPONSES_PROFILE = "writer_reviewer_low_responses_v1"
RESPONSES_PROFILE = "writer_reviewer_low_responses_v2"
CONSERVATIVE_ACCOUNTING = "carry_terminal_rejection_v1"


def profile_mode(profile):
    """Resolve only server-defined campaign modes, never arbitrary provider kwargs."""
    if profile == DISABLED_PROFILE:
        return "disabled"
    if profile in (
        THINKING_PROFILE,
        LOW_THINKING_PROFILE,
        RESPONSES_PROFILE,
        LEGACY_RESPONSES_PROFILE,
    ):
        return "enabled"
    raise ValueError("frozen_writing_profile_invalid")


def profile_parameters(profile):
    """Bind the single supported native effort without adding legacy config fields."""
    profile_mode(profile)
    if profile in (RESPONSES_PROFILE, LEGACY_RESPONSES_PROFILE):
        return {"native_reasoning_effort": "low", "native_provider_api": "responses"}
    return {"native_reasoning_effort": "low"} if profile == LOW_THINKING_PROFILE else {}


def profile_config(config, profile):
    """Create an explicit experiment override without mutating deployment defaults."""
    mode = profile_mode(profile)
    if profile == DISABLED_PROFILE:
        return config
    result = config.model_copy(deep=True)
    result.adversarial.thinking_mode = mode
    return result


def check_profile(config, profile):
    """Reuse the unchanged baseline gate while checking the explicit mode separately."""
    if profile == LEGACY_RESPONSES_PROFILE:
        raise ValueError("frozen_writing_profile_read_only")
    if config.adversarial.thinking_mode != profile_mode(profile):
        raise ValueError("frozen_writing_profile_mode_conflict")
    if profile == DISABLED_PROFILE:
        check_models(config)
        return
    if config.adversarial.review_max_tokens != 4096 or config.eval.max_tokens != 16384:
        raise ValueError("frozen_writing_profile_output_conflict")
    baseline = config.model_copy(deep=True)
    baseline.adversarial.thinking_mode = "disabled"
    check_models(baseline)


def profile_description(profile, summary):
    """Publish only mode and output metadata that belongs to the frozen config."""
    mode = profile_mode(profile)
    summary = summary.get("effective", {})
    llm = summary.get("llm", {})
    routing = llm.get("model_routing", {})
    result = {
        "id": profile,
        "generation_thinking_mode": mode,
        "reviewer_thinking_mode": mode,
        "evaluation_thinking_mode": "disabled",
        "output_limit_parameter": (
            "max_tokens" if mode == "disabled" else "max_completion_tokens"
        ),
        "generation_output_tokens": llm.get("max_tokens"),
        "reviewer_output_tokens": summary.get("adversarial", {}).get(
            "review_max_tokens"
        ),
        "evaluation_output_tokens": summary.get("eval", {}).get("max_tokens"),
        "configured_models": {
            role: routing.get(role, llm.get("model"))
            for role in ("extraction", "generation", "evaluation")
        },
    }
    if profile in (LOW_THINKING_PROFILE, RESPONSES_PROFILE, LEGACY_RESPONSES_PROFILE):
        result["reasoning_effort"] = "low"
    if profile in (RESPONSES_PROFILE, LEGACY_RESPONSES_PROFILE):
        result["provider_api"] = "responses"
        result["output_limit_parameter"] = "max_output_tokens"
    if profile == RESPONSES_PROFILE:
        from litagent.llm.responses import RESPONSES_CONTRACT

        result.update(
            mapping_contract=RESPONSES_CONTRACT,
            storage=False,
            stream=False,
            host_input="complete",
            server_truncation_control="unavailable",
        )
    return result


def runtime_identity():
    """Bind core transport, schema and workflow dependencies before execution."""
    return {
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in ("openai", "httpx", "pydantic", "langgraph", "PyMuPDF")
        },
    }


def source_paths():
    """Freeze the complete implementation and test inventory, including new files."""
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    paths = [
        Path(p)
        for p in tracked
        if p.startswith(("src/", "tests/", "scripts/", "config/"))
        or p in {"pyproject.toml", "requirements-demo.txt"}
    ]
    paths += (
        list(Path("src/litagent").rglob("*.py"))
        + list(Path("scripts").glob("*.py"))
        + list(Path("tests").rglob("*.py"))
        + list(Path("tests/fixtures").rglob("*.json"))
    )
    return sorted({p for p in paths if p.is_file() and "__pycache__" not in str(p)})


def bound_native_receipt(start, end, journal):
    """Bind exactly one private HTTP receipt to the frozen host request identity."""
    from litagent.llm.responses import ARK_NATIVE_FIELDS, decode_private_receipt
    from litagent.quality import digest as writer_digest
    from litagent.safety.budget import strict_request_reservation

    matches = []
    for call in journal.get("calls", {}).values():
        receipt = call.get("provider_receipt")
        if not isinstance(receipt, dict):
            continue
        request, response = decode_private_receipt(receipt)
        rf = start.get("response_format")
        cap = start.get("max_tokens")
        mode, effort = call.get("thinking_mode"), call.get("reasoning_effort")
        extra = None if mode == "provider_default" else {"thinking": {"type": mode}}
        identity = [start.get("messages"), rf, {"max_output_tokens": cap}, extra]
        if effort is not None:
            identity.append({"reasoning_effort": effort})
        identity.append(["responses", receipt["contract"]])
        if call.get("input_sha256") != writer_digest(identity):
            continue
        if (
            not isinstance(rf, dict)
            or rf.get("type") != "json_schema"
            or type(cap) is not int
            or not 0 < cap <= 20000
            or set(request)
            - (
                ARK_NATIVE_FIELDS
                | (
                    {"background", "truncation"}
                    if receipt["contract"].endswith("-v1")
                    else set()
                )
            )
            or (
                receipt["contract"].endswith("-v1")
                and (
                    request.get("background") is not False
                    or request.get("truncation") != "disabled"
                )
            )
            or request.get("input") != start.get("messages")
            or request.get("text")
            != {"format": {"type": "json_schema", **rf["json_schema"]}}
            or request.get("model") != start.get("model")
            or request.get("max_output_tokens") != cap
            or request.get("thinking") != (extra or {}).get("thinking")
            or request.get("reasoning") != ({"effort": effort} if effort else None)
            or request.get("store") is not False
            or request.get("stream") is not False
            or call.get("provider_api") != "responses"
            or start.get("provider_api") != "responses"
            or start.get("output_limit_parameter") != "max_output_tokens"
            or end.get("model") != start.get("model")
            or end.get("messages") != start.get("messages")
            or call.get("provider_schema_sha256")
            != writer_digest(rf["json_schema"]["schema"])
            or call.get("reservation")
            != strict_request_reservation(start["messages"], cap, rf)
            or len(base64.b64decode(receipt["request_body_base64"], validate=True))
            + cap
            > call.get("reservation", 0)
            or start.get("reasoning_effort") != effort
            or call.get("max_tokens") != cap
            or call.get("output_limit_parameter") != "max_output_tokens"
        ):
            raise ValueError("frozen_accounting_receipt_binding_invalid")
        matches.append((call, receipt, request, response))
    if len(matches) != 1:
        raise ValueError("frozen_accounting_receipt_binding_missing_or_ambiguous")
    return matches[0]


def conservative_rejection_receipt(root, manifest, state, row, archive, starts, ends):
    """Carry a complete uncertain charge; never fabricate a received usage record."""
    from litagent.observability.context import reset_task_id, set_task_id
    from litagent.safety import budget as budget_module
    from litagent.safety.budget import CostBudget, strict_request_reservation

    budget_path = "src/litagent/safety/budget.py"
    budget_sha = hashlib.sha256(Path(budget_module.__file__).read_bytes()).hexdigest()
    saved_budget = root / "source" / budget_path
    if (
        state.get("status") != "stopped"
        or row.get("publishable") is not False
        or manifest.get("version") != VERSION
        or manifest.get("limits") != LIMITS
        or archive.get("config") != manifest.get("config")
        or archive.get("config_fingerprint")
        != manifest.get("config", {}).get("fingerprint")
        or manifest.get("source_hashes", {}).get(budget_path) != budget_sha
        or not saved_budget.exists()
        or hashlib.sha256(saved_budget.read_bytes()).hexdigest() != budget_sha
        or any(e["event"] == "llm.retry" for e in archive["events"])
    ):
        raise ValueError("frozen_accounting_unknown_proof_missing")
    path = Path("artifacts/targeted-repairs", row["run_id"] + ".json.writer-v2.json")
    journal = json.loads(path.read_text())
    reconstructed = CostBudget(max_tokens=LIMITS["per_case_tokens"])
    reconstructed.configure_strict(halt_on_unknown_usage=True)
    total_unknown, rejection_calls = 0, []
    indexed = {e["operation_id"]: e for e in ends}
    expected_model = manifest.get("config", {}).get("effective", {}).get("llm", {})
    routes = expected_model.get("model_routing", {})
    for start in starts:
        end = indexed[start["operation_id"]]
        cap = start.get("max_tokens")
        if type(cap) is not int or not 0 < cap <= 20000:
            raise ValueError("frozen_accounting_cap_invalid")
        reservation = strict_request_reservation(
            start.get("messages"), cap, start.get("response_format")
        )
        if not isinstance(start.get("task_id"), str) or not start["task_id"]:
            raise ValueError("frozen_accounting_task_invalid")
        context = set_task_id(start["task_id"])
        reserved = False
        try:
            reconstructed.reserve(reservation)
            reserved = True
            reconstructed.begin_request()
            if end.get("usage_available") is True:
                reconstructed.record(end)
            else:
                call, receipt, request, response = bound_native_receipt(
                    start, end, journal
                )
                code = receipt["status_code"]
                route = (
                    "evaluation" if call.get("stage") == "reviewer" else "generation"
                )
                if (
                    not 400 <= code < 500
                    or not isinstance(response.get("error"), dict)
                    or "usage" in response
                    or response.get("output")
                    or call.get("status") != "failed_no_replay"
                    or call.get("usage", {}).get("usage_available") is not False
                    or end.get("provider_error") != "responses_http_" + str(code)
                    or request.get("model")
                    != routes.get(route, expected_model.get("model"))
                    or receipt["endpoint"] != "/api/v3/responses"
                    or (
                        receipt.get("origin") is not None
                        and receipt["origin"] != expected_model.get("origin")
                    )
                    or cap
                    != (
                        manifest["config"]["effective"]
                        .get("adversarial", {})
                        .get("review_max_tokens")
                        if route == "evaluation"
                        else expected_model.get("max_tokens")
                    )
                    or call.get("thinking_mode") != "enabled"
                    or call.get("reasoning_effort") != "low"
                ):
                    raise ValueError("frozen_accounting_unknown_not_terminal_rejection")
                reconstructed.charge_unknown_usage(reservation)
                total_unknown += reservation
                rejection_calls.append(
                    {
                        "operation_id": start["operation_id"],
                        "reservation": reservation,
                        "request_sha256": receipt["request_sha256"],
                        "response_sha256": receipt["response_sha256"],
                        "http_status": code,
                        "origin_evidence": (
                            "recorded_http_origin"
                            if receipt.get("origin")
                            else "frozen_config_origin_only"
                        ),
                    }
                )
        finally:
            reset_task_id(context)
            # Every successful reserve is released even when proof validation fails.
            if reserved:
                reconstructed.release(reservation)
    snapshot = reconstructed.snapshot()
    if (
        type(row.get("unknown_hold")) is not int
        or row["unknown_hold"] != total_unknown
        or total_unknown <= 0
        or snapshot["used_tokens"] != row["known_tokens"] + total_unknown
        or snapshot["requests_sent"] != row["requests"]
        or row.get("usage_complete") is not False
    ):
        raise ValueError("frozen_accounting_unknown_amount_invalid")
    actual = row.get("budget_snapshot")
    if actual is not None:
        restored = CostBudget.from_snapshot(actual)
        for key in (
            "max_tokens",
            "used_tokens",
            "requests_sent",
            "phase_usage",
            "usage_uncertain",
            "call_count",
        ):
            if restored.snapshot().get(key) != snapshot.get(key):
                raise ValueError("frozen_accounting_budget_snapshot_invalid")
    return {
        "version": CONSERVATIVE_ACCOUNTING,
        "unknown_hold": total_unknown,
        "journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "budget_evidence": (
            "persisted_snapshot"
            if actual is not None
            else "reconstructed_from_frozen_records"
        ),
        "budget_source_sha256": budget_sha,
        "conservative_snapshot": snapshot,
        "rejections": rejection_calls,
    }


def accounting_history(history, parents, *, unknown_policy="reject"):
    """Carry a verified terminal prefix forward without refunding or duplicating calls."""
    if unknown_policy not in ("reject", CONSERVATIVE_ACCOUNTING):
        raise ValueError("frozen_accounting_policy_invalid")
    history, receipts, seen, seen_archives = dict(history), [], set(), set()
    for parent in parents:
        root = Path(parent)
        if root.resolve() in seen:
            raise ValueError("frozen_accounting_parent_duplicate")
        seen.add(root.resolve())
        mp, sp = root / "manifest.json", root / "state.json"
        manifest, state = json.loads(mp.read_text()), json.loads(sp.read_text())
        terminal_probe_failure = (
            state.get("status") == "blocked"
            and manifest.get("version")
            in {
                "writer-v2-protocol-observation-v2",
                "writer-v2-protocol-observation-v3",
            }
            and state.get("failure", {}).get("family") == "protocol"
            and state.get("requests") == 1
        )
        if (
            manifest.get("history") != history
            or manifest.get("identity")
            != digest({k: v for k, v in manifest.items() if k != "identity"})
            or state.get("identity") != manifest["identity"]
            or (
                state.get("status")
                not in {"stopped", "completed", "protocol_completed"}
                and not terminal_probe_failure
            )
        ):
            raise ValueError("frozen_accounting_prefix_invalid")
        rows = state.get("cases", [state])
        archives = {}
        requests, tokens = 0, 0
        conservative = []
        for row in rows:
            uncertain = row.get("usage_complete") is not True or row.get(
                "unknown_hold", 0
            )
            if uncertain and unknown_policy == "reject":
                raise ValueError("frozen_accounting_usage_unknown")
            run_id = row.get("run_id", root.name)
            if Path(run_id).name != run_id:
                raise ValueError("frozen_accounting_run_invalid")
            path = Path("artifacts/runs", run_id + ".json")
            archive = json.loads(path.read_text())
            if archive["run_id"] != run_id or path.resolve() in seen_archives:
                raise ValueError("frozen_accounting_archive_duplicate_or_invalid")
            seen_archives.add(path.resolve())
            events = archive["events"]
            starts = [e["data"] for e in events if e["event"] == "llm.start"]
            ends = [e["data"] for e in events if e["event"] == "llm.complete"]
            if (
                type(row.get("requests")) is not int
                or type(row.get("known_tokens")) is not int
                or row["requests"] < 0
                or row["known_tokens"] < 0
                or len(starts) != len(ends)
                or len(starts) != row["requests"]
                or len({e["operation_id"] for e in starts}) != len(starts)
                or {e["operation_id"] for e in starts}
                != {e["operation_id"] for e in ends}
                or any(
                    e.get("usage_available") is not True and not uncertain for e in ends
                )
                or any(
                    any(
                        type(e.get(k)) is not int or e[k] < 0
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    )
                    or e["total_tokens"] != e["prompt_tokens"] + e["completion_tokens"]
                    for e in ends
                    if e.get("usage_available") is True
                )
                or sum(
                    e["total_tokens"] for e in ends if e.get("usage_available") is True
                )
                != row["known_tokens"]
            ):
                raise ValueError("frozen_accounting_archive_usage_invalid")
            if uncertain:
                conservative.append(
                    conservative_rejection_receipt(
                        root, manifest, state, row, archive, starts, ends
                    )
                )
            requests += row["requests"]
            tokens += row["known_tokens"]
            archives[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipts.append(
            {
                "root": str(root),
                "manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                "state_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
                "archives": archives,
                "requests": requests,
                "known_tokens": tokens,
                "history_before": dict(history),
                **(
                    {
                        "conservative_rejections": conservative,
                        "unknown_hold": sum(r["unknown_hold"] for r in conservative),
                    }
                    if conservative
                    else {}
                ),
            }
        )
        history["model_requests"] += requests
        history["model_known_tokens"] += tokens
        if conservative:
            history["model_unknown_conservative_tokens"] = history.get(
                "model_unknown_conservative_tokens", 0
            ) + sum(r["unknown_hold"] for r in conservative)
    return history, receipts


def prepare(root, config, *, accounting_parents=(), writing_profile=DISABLED_PROFILE):
    """Freeze all three materials and the same source before any paid request."""
    check_profile(config, writing_profile)
    root = Path(root)
    if root.exists():
        raise ValueError("frozen_campaign_already_exists")
    rows = []
    for case, run in CASES:
        bundle = read_bundle(Path("artifacts/runs", run + ".json"))
        rows.append({"case_id": case, "bundle": bundle})
    paths = source_paths()
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
        if p.is_file() and "__pycache__" not in str(p)
    }
    history = json.loads(
        Path(
            "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
        ).read_text()
    )["cumulative"]
    policy = (
        CONSERVATIVE_ACCOUNTING if writing_profile == RESPONSES_PROFILE else "reject"
    )
    history, receipts = accounting_history(
        history, accounting_parents, unknown_policy=policy
    )
    manifest = {
        "version": VERSION,
        "suite_id": root.name,
        "contract": CONTRACT,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "options": frozen_options().model_dump(mode="json"),
        "config": build_config_summary(config, "strict_auto_v2"),
        "source_hashes": hashes,
        "cases": rows,
        "history": history,
        "limits": {
            "cases": 3,
            "per_case_tokens": 500000,
            "per_case_requests": 64,
            "total_tokens": 1500000,
            "total_requests": 192,
            "repairs": 2,
            "retries": 0,
            "formal_cases": 0,
        },
    }
    manifest["limits"] = dict(LIMITS)
    if writing_profile != DISABLED_PROFILE:
        manifest["writing_profile"] = writing_profile
    if profile_parameters(writing_profile):
        manifest["writing_parameters"] = profile_parameters(writing_profile)
    if policy != "reject":
        manifest["accounting_policy"] = policy
    if receipts:
        manifest["accounting_receipts"] = receipts
    manifest["runtime"] = runtime_identity()
    manifest["identity"] = digest(manifest)
    root.mkdir(parents=True)
    atomic_json(root / "manifest.json", manifest)
    for name, h in hashes.items():
        p = root / "source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(Path(name).read_bytes())
    return {
        "identity": manifest["identity"],
        "cases": [
            {
                "id": r["case_id"],
                "source": r["bundle"]["source_run_id"],
                "cards": len(r["bundle"]["selection"]["selected_items"]),
            }
            for r in rows
        ],
    }


def validate(root, config):
    """Reject changed sources, materials, runtime config or campaign scope."""
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("version") != VERSION
        or m.get("limits") != LIMITS
        or m.get("runtime") != runtime_identity()
        or m.get("contract") != CONTRACT
        or set(m.get("source_hashes", {})) != {str(p) for p in source_paths()}
        or digest({k: v for k, v in m.items() if k != "identity"}) != m["identity"]
        or m["config"] != build_config_summary(config, "strict_auto_v2")
        or m["options"] != frozen_options().model_dump(mode="json")
        or [(r["case_id"], r["bundle"]["source_run_id"]) for r in m["cases"]]
        != list(CASES)
    ):
        raise ValueError("frozen_campaign_identity_conflict")
    check_profile(config, m.get("writing_profile", DISABLED_PROFILE))
    if m.get("writing_parameters", {}) != profile_parameters(
        m.get("writing_profile", DISABLED_PROFILE)
    ):
        raise ValueError("frozen_writing_parameters_conflict")
    policy = (
        CONSERVATIVE_ACCOUNTING
        if m.get("writing_profile") == RESPONSES_PROFILE
        else "reject"
    )
    if m.get("accounting_policy", "reject") != policy:
        raise ValueError("frozen_accounting_policy_conflict")
    if m.get("accounting_receipts"):
        receipts = m["accounting_receipts"]
        history, verified = accounting_history(
            receipts[0]["history_before"],
            [r["root"] for r in receipts],
            unknown_policy=policy,
        )
        if history != m["history"] or verified != receipts:
            raise ValueError("frozen_campaign_accounting_changed")
    for name, h in m["source_hashes"].items():
        if (
            hashlib.sha256(Path(name).read_bytes()).hexdigest() != h
            or hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() != h
        ):
            raise ValueError("frozen_campaign_source_changed")
    for row in m["cases"]:
        path = Path("artifacts/runs", row["bundle"]["source_run_id"] + ".json")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != row["bundle"]["archive_sha256"]
        ):
            raise ValueError("frozen_campaign_archive_changed")
    return m


def safe_reason(reason):
    """Keep exception prose and model content out of public machine-code fields."""
    return (
        reason
        if isinstance(reason, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason)
        else "evaluation_not_completed"
    )


def evaluation_diagnostic(report):
    """Return safe scalar diagnostics without publishing evaluator request payloads."""
    details = report.get("evaluation", {}).get("faithfulness", {}).get("details", {})
    reason = next(
        (details[k] for k in ("skipped_reason", "error", "reason") if details.get(k)),
        None,
    )
    if not reason:
        return None
    stages = details.get("stages", {})

    def count(value):
        """Unknown historical counts remain unavailable, not scientific zero."""
        return value if type(value) is int and value >= 0 else None

    return {
        "reason_code": safe_reason(reason),
        "expected_assertions": count(stages.get("expected_assertions")),
        "processed_assertions": count(stages.get("processed_assertions")),
        "scientific_unverified": count(details.get("unverified_count")),
    }


def transport_diagnostic(run_id):
    """Read only a receipt bound to the exact archive, never free error prose."""
    from litagent.llm.responses import receipt_diagnostic

    if not isinstance(run_id, str) or Path(run_id).name != run_id:
        return None
    archive_path = Path("artifacts/runs", run_id + ".json")
    journal_path = Path("artifacts/targeted-repairs", run_id + ".json.writer-v2.json")
    if not archive_path.exists() or not journal_path.exists():
        return None
    try:
        archive = json.loads(archive_path.read_text())
        journal = json.loads(journal_path.read_text())
        if archive.get("run_id") != run_id:
            return None
        ends = {
            e["data"]["operation_id"]: e["data"]
            for e in archive["events"]
            if e["event"] == "llm.complete"
        }
        for event in reversed(archive["events"]):
            if (
                event["event"] != "llm.start"
                or event["data"].get("provider_api") != "responses"
            ):
                continue
            start = event["data"]
            end = ends.get(start["operation_id"])
            if not end or not end.get("provider_error"):
                continue
            _, receipt, _, _ = bound_native_receipt(start, end, journal)
            return {
                **receipt_diagnostic(receipt),
                "usage_available": end.get("usage_available") is True,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
            }
    except (KeyError, ValueError, TypeError, OSError):
        return None
    return None


def export(root, state, manifest):
    """Publish one sanitized data source with separate execution and quality outcomes."""
    executed = state["cases"]
    writing_profile = profile_description(
        manifest.get("writing_profile", DISABLED_PROFILE), manifest.get("config", {})
    )
    rows = [
        next(
            (dict(r) for r in executed if r["case_id"] == case_id),
            {
                "case_id": case_id,
                "run_id": None,
                "status": "not_run",
                "generated": False,
                "reviewer_completed": False,
                "evaluation": "not_run",
                "publishable": False,
                "supported": None,
                "unsupported": None,
                "unverified": None,
                "requests": 0,
                "known_tokens": 0,
                "unknown_hold": 0,
                "failure_code": (
                    "not_started_after_stop"
                    if state["status"] in {"stopped", "interrupted"}
                    else "not_started"
                ),
            },
        )
        for case_id, _ in CASES
    ]
    for row in rows:
        if not row.get("run_id") or Path(row["run_id"]).name != row["run_id"]:
            continue
        archive_path = Path("artifacts/runs", row["run_id"] + ".json")
        if not archive_path.exists() or not row.get("report_sha256"):
            continue
        archive = json.loads(archive_path.read_text())
        report = archive.get("report") or {}
        survey = report.get("survey")
        if (
            archive.get("run_id") != row["run_id"]
            or not isinstance(survey, str)
            or hashlib.sha256(survey.encode()).hexdigest() != row["report_sha256"]
        ):
            continue
        diagnostic = evaluation_diagnostic(report)
        if diagnostic:
            row["evaluation_diagnostic"] = diagnostic
        attempts = report.get("metadata", {}).get("evidence_rewrite_attempts", [])
        not_sent = {
            f"repair_{i+1}": a
            for i, a in enumerate(attempts)
            if a.get("attempted") is False
        }
        if not_sent:
            row["evaluation_history"] = [
                h
                for h in row.get("evaluation_history", [])
                if h.get("stage") not in not_sent
            ]
            row["repair_preflight_failures"] = [
                {"attempted": False, "reason_code": safe_reason(a.get("reason_code"))}
                for a in not_sent.values()
            ]
    for row in rows:
        diagnostic = transport_diagnostic(row.get("run_id"))
        if diagnostic:
            row["transport_diagnostic"] = diagnostic
        for key in ("http_status", "provider_error_code", "rejected_parameter"):
            row[key] = (diagnostic or {}).get(key)
    for row in rows:
        if row.get("failure_code") != "writer_json_invalid" or not row.get("run_id"):
            continue
        journal = Path(
            "artifacts/targeted-repairs", row["run_id"] + ".json.writer-v2.json"
        )
        if not journal.exists():
            continue
        calls = json.loads(journal.read_text()).get("calls", {}).values()
        call = next(
            (
                c
                for c in calls
                if c.get("failure", {}).get("code") == "writer_json_invalid"
            ),
            None,
        )
        if call and isinstance(call.get("raw"), str):
            raw = call["raw"]
            offset = call["failure"]["diagnostics"][0]["offset"]
            row["protocol_diagnostic"] = {
                "stage": call["stage"],
                "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "offset": offset,
                "line": raw.count("\n", 0, offset) + 1,
                "column": offset - raw.rfind("\n", 0, offset),
                "finish_reason": call.get("finish_reason"),
                "scientific_assessment": "not_run",
            }

    def display(value):
        """Render absent statistics as unavailable rather than scientific zero."""
        return (
            "—"
            if value is None
            else "Yes" if value is True else "No" if value is False else str(value)
        )

    def reason(row):
        """Display the bound evaluator cause while preserving the original run state."""
        cause = row.get("evaluation_diagnostic", {}).get("reason_code") or row.get(
            "failure_code", ""
        )
        if row.get("http_status"):
            cause += " / HTTP " + str(row["http_status"])
            cause += " / " + (row.get("provider_error_code") or "unavailable")
            if row.get("rejected_parameter"):
                cause += " / " + row["rejected_parameter"]
        return cause

    for row in rows:
        row["writing_profile"] = writing_profile["id"]
        row["provider_api"] = writing_profile.get("provider_api", "chat")
        row["reasoning_effort"] = writing_profile.get("reasoning_effort")
        for key in (
            "generation_thinking_mode",
            "reviewer_thinking_mode",
            "evaluation_thinking_mode",
            "output_limit_parameter",
            "generation_output_tokens",
            "reviewer_output_tokens",
            "evaluation_output_tokens",
        ):
            row[key] = writing_profile[key]
        row["evaluation_reason"] = reason(row)
        for key in ("expected_assertions", "processed_assertions"):
            row[key] = row.get("evaluation_diagnostic", {}).get(key)

    public = {
        "version": VERSION,
        "view_version": "frozen-writing-view-v7",
        "suite_id": root.name,
        "identity": manifest["identity"],
        "source_commit": manifest["source_commit"],
        "contract": CONTRACT,
        "writing_profile": writing_profile,
        "state": state["status"],
        "stop_reason": state.get("stop_reason"),
        "planned": 3,
        "run": len(executed),
        "generated": sum(bool(r.get("generated")) for r in rows),
        "evaluation_completed": sum(r.get("evaluation") == "completed" for r in rows),
        "passed": sum(r.get("publishable", False) for r in rows),
        "not_run": 3 - len(executed),
        "judge": "disabled",
        "jev": "disabled",
        "human_calibration": "not_validated",
        "cases": rows,
        "historical_usage": manifest["history"],
        "new_usage": {
            "requests": sum(r.get("requests", 0) for r in rows),
            "known_tokens": sum(r.get("known_tokens", 0) for r in rows),
            "unknown_hold": sum(r.get("unknown_hold", 0) for r in rows),
            "request_hold": sum(r.get("request_hold", 0) for r in rows),
        },
        "limits": manifest["limits"],
    }
    out = Path("benchmarks/survey/results", root.name)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "benchmark.json", public)
    keys = [
        "case_id",
        "writing_profile",
        "provider_api",
        "reasoning_effort",
        "reasoning_tokens",
        "reasoning_usage_complete",
        "reasoning_usage_available_requests",
        "generation_thinking_mode",
        "reviewer_thinking_mode",
        "evaluation_thinking_mode",
        "output_limit_parameter",
        "generation_output_tokens",
        "reviewer_output_tokens",
        "evaluation_output_tokens",
        "run_id",
        "status",
        "generated",
        "evaluation",
        "publishable",
        "supported",
        "unsupported",
        "unverified",
        "requests",
        "known_tokens",
        "unknown_hold",
        "elapsed_ms",
        "report_sha256",
        "failure_family",
        "failure_code",
        "http_status",
        "provider_error_code",
        "rejected_parameter",
        "evaluation_reason",
        "expected_assertions",
        "processed_assertions",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    (out / "benchmark.csv").write_text(buf.getvalue())
    md = f"# 三域冻结正文验收\n\n计划3题，运行{public['run']}题，生成{public['generated']}题，Evaluation完成{public['evaluation_completed']}题，严格交付{public['passed']}/3。状态：{state['status']}。\n\n源码 `{manifest['source_commit']}`，协议 `{CONTRACT}`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。\n\n|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|\n|---|---|---|---|---|---|\n"
    md += "".join(
        f"|{r['case_id']}|{display(r.get('generated'))}|{r.get('evaluation')}|{display(r.get('publishable'))}|{display(r.get('supported'))}/{display(r.get('unsupported'))}/{display(r.get('unverified'))}|{reason(r)}|\n"
        for r in rows
    )
    md += (
        "\n冻结模式 / Frozen profile: `"
        + writing_profile["id"]
        + "`。\n\n```json\n"
        + json.dumps(writing_profile, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    md += f"\n新增已知用量{public['new_usage']['known_tokens']} tokens，未知保守占用{public['new_usage']['unknown_hold']}，{public['new_usage']['requests']}请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。\n\n[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)\n"
    if any("reasoning_usage_complete" in r for r in rows):
        md += "\n思考明细是 completion 的子集，不再次累计；缺失时不推算。\n\n|Case|Reasoning tokens|Details complete|Available requests / all requests|\n|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{display(r.get('reasoning_tokens'))}|{display(r.get('reasoning_usage_complete'))}|{display(r.get('reasoning_usage_available_requests'))}/{r.get('requests', 0)}|\n"
            for r in rows
        )
    (out / "benchmark.md").write_text(md)
    (out / "README.md").write_text(md)
    table = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(display(r.get(k))) + "</td>"
            for k in [
                "case_id",
                "generated",
                "evaluation",
                "publishable",
                "supported",
                "unsupported",
                "unverified",
            ]
        )
        + (
            '<td><a href="http://127.0.0.1:8000/flow-demo?run=archive%3A'
            + r["run_id"]
            + '&view=quality">'
            + (
                "质量与原文 / Report"
                if r.get("report_sha256")
                else "运行诊断 / Diagnostics"
            )
            + "</a></td></tr>"
            if r.get("run_id")
            else "<td>未运行 / Not run</td></tr>"
        )
        for r in rows
    )
    details = "".join(
        "<details><summary>"
        + html.escape(r["case_id"])
        + " · "
        + html.escape(str(reason(r) or r["status"]))
        + "</summary><pre>"
        + html.escape(json.dumps(r, ensure_ascii=False, indent=2))
        + "</pre></details>"
        for r in rows
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LitAgent · Frozen writing validation</title><style>*{box-sizing:border-box}body{font:16px/1.6 system-ui;background:#f4f6fa;color:#172c43;max-width:1100px;padding:20px;margin:auto}section,header{background:white;padding:20px;margin:16px 0;border:1px solid #dbe2ee;border-radius:12px}h1{font-size:1.7rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.stat{background:#edf2f9;padding:14px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85em}button,a,summary{cursor:pointer}button{font:inherit;padding:8px}a{color:#135a9e}*:focus-visible{outline:3px solid #1679d2}summary{padding:12px 0}p{overflow-wrap:anywhere}@media(max-width:430px){body{padding:10px}section,header{padding:14px}h1{font-size:1.3rem}}</style><header><button id="locale">English / 中文</button><h1 id="title">三域冻结正文验收</h1><p id="scope">固定证据 → 正文 → Reviewer → Evaluation → 自动交付或失败归档；不重新检索。Judge/Jev关闭。</p><p id="profile"></p><p id="transport"></p></header><section><div id="stats" class="grid"></div><p id="note">自动评测；人工校准未验证。未核验不等于通过，协议完成不等于科学通过。失败及未运行题均保留在分母。</p><p id="status"></p></section><section class="scroll"><table><thead><tr><th>Case</th><th>Generated</th><th>Evaluation</th><th>Delivered</th><th>Supported</th><th>Unsupported</th><th>Unverified</th><th>Detail</th></tr></thead><tbody>__ROWS__</tbody></table></section><section><h2>诊断 / Diagnostics</h2>__DETAILS__</section><section><a href="benchmark.json">JSON</a> · <a href="benchmark.csv">CSV</a> · <a href="benchmark.md">Markdown</a><pre id="usage"></pre></section><script id="data" type="application/json">__DATA__</script><script>const d=JSON.parse(document.getElementById('data').textContent);let en=false;function draw(){document.documentElement.lang=en?'en':'zh-CN';document.getElementById('title').textContent=en?'Three-domain frozen writing validation':'三域冻结正文验收';document.getElementById('scope').textContent=en?'Frozen evidence → writing → Reviewer → Evaluation → automatic delivery or archived failure. No retrieval. Judge/Jev disabled.':'固定证据 → 正文 → Reviewer → Evaluation → 自动交付或失败归档；不重新检索。Judge/Jev关闭。';document.getElementById('note').textContent=en?'Automatic evaluation; no human calibration. Unverified is not passed. Protocol completion is not scientific acceptance. Failures and unrun cases remain in the denominator.':'自动评测；人工校准未验证。未核验不等于通过，协议完成不等于科学通过。失败及未运行题均保留在分母。';const t=d.cases.filter(r=>r.transport_diagnostic).map(r=>r.case_id+' · HTTP '+r.http_status+' · '+(r.provider_error_code||'—')+' · '+(r.rejected_parameter||'—')).join('; ');document.getElementById('transport').textContent=t?((en?'Provider diagnosis (usage unknown is not zero cost): ':'供应商诊断（未知用量不等于零费用）：')+t):'';const p=d.writing_profile;document.getElementById('profile').textContent=(en?'Frozen thinking profile: ':'冻结思考配置：')+p.id+' · '+(p.provider_api||'chat')+' · '+(en?'Writing':'写作')+' '+p.generation_thinking_mode+' / Reviewer '+p.reviewer_thinking_mode+' / Evaluation '+p.evaluation_thinking_mode+' · '+(p.reasoning_effort?('effort='+p.reasoning_effort+' · '):'')+p.output_limit_parameter+' '+(p.generation_output_tokens??'—')+' / '+(p.reviewer_output_tokens??'—');document.getElementById('status').textContent=d.state+' · '+(d.stop_reason||'');let s=document.getElementById('stats');s.replaceChildren();for(const [k,v]of Object.entries({Planned:3,Run:d.run,Generated:d.generated,Evaluated:d.evaluation_completed,Delivered:d.passed})){let e=document.createElement('div');e.className='stat';e.textContent=k+' '+v+' / 3';s.append(e)}document.getElementById('usage').textContent=JSON.stringify({new_usage:d.new_usage,historical_usage:d.historical_usage,writing_profile:d.writing_profile,source_commit:d.source_commit,contract:d.contract},null,2)}document.getElementById('locale').onclick=()=>{en=!en;draw()};draw();</script></html>"""
    page = (
        page.replace("__ROWS__", table)
        .replace("__DETAILS__", details)
        .replace(
            "__DATA__", json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
        )
    )
    (out / "benchmark.html").write_text(page)
    return public


async def execute(root, config):
    """Reject concurrent submissions before touching the durable suite state."""
    with suite_lock(root / "execution.lock"):
        return await _execute(root, config)


async def _execute(root, config):
    """Never replay a started observation or expand the three-case manifest."""
    m = validate(root, config)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != m["identity"]:
            raise ValueError("frozen_campaign_state_conflict")
        if state["status"] != "prepared":
            if state["status"] == "running":
                state.update(
                    status="interrupted", stop_reason="process_interrupted_no_replay"
                )
                for entry in state["cases"]:
                    if entry["status"] == "started_no_replay":
                        entry.update(
                            status="interrupted_no_replay",
                            failure_family="interrupted",
                            failure_code="process_interrupted_no_replay",
                            unknown_hold=500000,
                            request_hold=64,
                            accounting_scope="unsettled_case_cap",
                            usage_complete=False,
                        )
                atomic_json(state_path, state)
            export(root, state, m)
            return state
    else:
        state = {"identity": m["identity"], "status": "prepared", "cases": []}
    state["status"] = "running"
    atomic_json(state_path, state)
    for row in m["cases"]:
        run_id = root.name + "-" + row["case_id"]
        entry = {
            "case_id": row["case_id"],
            "run_id": run_id,
            "status": "started_no_replay",
            "generated": False,
            "reviewer_completed": False,
            "evaluation": "not_run",
            "publishable": False,
            "supported": None,
            "unsupported": None,
            "unverified": None,
        }
        state["cases"].append(entry)
        atomic_json(state_path, state)
        recorder = RunRecorder(run_id, row["bundle"]["query"], ArchiveRepository())
        recorder.set_config_summary(build_config_summary(config, "strict_auto_v2"))
        recorder.set_citation_style("ieee")
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
        agent = LitAgent(
            config, trace_hook=hook, run_policy=RunPolicy.isolated_benchmark()
        )
        start = time.monotonic()
        failure = None
        report = None
        try:
            report = await run_frozen(
                agent,
                row["bundle"],
                run_id,
                **profile_parameters(m.get("writing_profile", DISABLED_PROFILE)),
            )
        except BaseException as exc:
            failure = classify_failure(exc)
        finally:
            budget = agent._cost_budget.snapshot() if agent._cost_budget else {}
            await agent.cleanup()
        if tracer:
            recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            tracer.flush()
        artifact = recorder.finalize(
            report=report, error=(failure["code"] if failure else None)
        )
        events = artifact["events"]
        ends = [
            e["data"]
            for e in events
            if e["event"] == "llm.complete" and e["data"].get("usage_available")
        ]
        known = sum(e["total_tokens"] for e in ends)
        reasoning_available = [
            e for e in ends if e.get("reasoning_usage_available") is True
        ]
        reasoning_complete = bool(ends) and len(reasoning_available) == budget.get(
            "requests_sent", 0
        )
        entry.update(
            requests=budget.get("requests_sent", 0),
            known_tokens=known,
            unknown_hold=max(0, budget.get("used_tokens", 0) - known),
            usage_complete=not budget.get("usage_uncertain", False),
            budget_snapshot=budget,
            elapsed_ms=round((time.monotonic() - start) * 1000),
            trace_url=tracer.trace_url if tracer else None,
            reasoning_tokens=(
                sum(e["reasoning_tokens"] for e in reasoning_available)
                if reasoning_complete
                else None
            ),
            reasoning_usage_complete=reasoning_complete,
            reasoning_usage_available_requests=len(reasoning_available),
        )
        if report:
            meta = report.get("metadata", {})
            faith = report.get("evaluation", {}).get("faithfulness", {})
            details = faith.get("details", {})
            entry.update(
                generated=bool(meta.get("writer_document")),
                evaluation=(
                    "completed"
                    if not faith.get("skipped", True)
                    and details.get("stages", {}).get("global_check") == "completed"
                    else "not_complete"
                ),
                publishable=report.get("delivery", {}).get("publishable", False),
                report_sha256=(
                    hashlib.sha256(report["survey"].encode()).hexdigest()
                    if meta.get("writer_document")
                    else None
                ),
                supported=details.get("supported_count"),
                unsupported=details.get("unsupported_count"),
                unverified=details.get("unverified_count"),
                repairs=len(meta.get("targeted_repair_state", {}).get("attempts", [])),
                core_coverage=details.get("core_coverage", []),
                global_issues=details.get("global_issues", []),
                evaluation_history=[
                    {
                        "stage": label,
                        "supported": value.get("faithfulness", {})
                        .get("details", {})
                        .get("supported_count"),
                        "unsupported": value.get("faithfulness", {})
                        .get("details", {})
                        .get("unsupported_count"),
                        "unverified": value.get("faithfulness", {})
                        .get("details", {})
                        .get("unverified_count"),
                    }
                    for label, value in [
                        (
                            "initial",
                            meta.get("evaluation_history", {}).get(
                                "initial_evaluation", {}
                            ),
                        ),
                        *[
                            (f"repair_{i+1}", a.get("candidate_evaluation") or {})
                            for i, a in enumerate(
                                meta.get("evidence_rewrite_attempts", [])
                            )
                        ],
                    ]
                ],
            )
            failure = meta.get("writer_failure") or failure
            journal = Path(
                "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
            )
            if not failure and not entry["publishable"] and journal.exists():
                j = json.loads(journal.read_text())
                failure = j.get("failure") or (
                    None
                    if entry["generated"]
                    else next(
                        (
                            v.get("failure")
                            for v in reversed(list(j["calls"].values()))
                            if v.get("failure")
                        ),
                        None,
                    )
                )
            if not failure and entry["evaluation"] != "completed":
                failure = {
                    "family": "protocol",
                    "code": (evaluation_diagnostic(report) or {}).get("reason_code")
                    or "evaluation_not_completed",
                }
        journal_path = Path(
            "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
        )
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            entry["generated"] = (
                entry["generated"] or journal.get("status") == "generated"
            )
            reviewed = journal.get("calls", {}).get("reviewer", {})
            entry["reviewer_completed"] = reviewed.get(
                "status"
            ) == "response_received" and not reviewed.get("failure")
            entry["compiled_report_sha256"] = journal.get("draft_sha256")
        if entry["unknown_hold"] or not entry["usage_complete"]:
            failure = {"family": "interrupted", "code": "usage_unknown"}
        failure = failure or (
            None
            if entry["publishable"]
            else {"family": "quality", "code": "strict_quality_failed"}
        )
        entry.update(
            status="passed" if entry["publishable"] else "blocked",
            failure_family=failure["family"] if failure else None,
            failure_code=failure["code"] if failure else None,
        )
        atomic_json(state_path, state)
        export(root, state, m)
        if failure and failure["family"] not in {"quality", "evidence"}:
            state.update(status="stopped", stop_reason=failure["code"])
            break
    else:
        state["status"] = "completed"
    atomic_json(state_path, state)
    export(root, state, m)
    return state


def main():
    """Require explicit preparation or bounded live execution of the frozen suite."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--accounting-parent", type=Path, action="append", default=[])
    p.add_argument(
        "--writing-profile",
        choices=(
            DISABLED_PROFILE,
            THINKING_PROFILE,
            LOW_THINKING_PROFILE,
            RESPONSES_PROFILE,
        ),
        default=None,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--live", action="store_true")
    args = p.parse_args()
    if args.live and args.accounting_parent:
        p.error("--accounting-parent is frozen during --prepare only")
    if args.live and args.writing_profile is not None:
        p.error("--writing-profile is frozen during --prepare only")
    cfg = load_config()
    profile = (
        (args.writing_profile or DISABLED_PROFILE)
        if args.prepare
        else json.loads((args.root / "manifest.json").read_text()).get(
            "writing_profile", DISABLED_PROFILE
        )
    )
    cfg = profile_config(cfg, profile)
    result = (
        prepare(
            args.root,
            cfg,
            accounting_parents=args.accounting_parent,
            writing_profile=profile,
        )
        if args.prepare
        else asyncio.run(execute(args.root, cfg))
    )
    print(
        json.dumps(
            {
                "suite": args.root.name,
                **{
                    k: v
                    for k, v in result.items()
                    if k in {"identity", "status", "stop_reason", "cases"}
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
