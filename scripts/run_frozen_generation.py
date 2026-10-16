"""Validate one new writing contract against immutable public survey inputs."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
import sys
from importlib.metadata import version
from pathlib import Path

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.quality_suite import (
    _candidate_evaluation_failures,
    _technical_failure,
    suite_lock,
)
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.evidence import collect_ledger
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.task_graph import SubTask, TaskGraph, TaskStatus
from litagent.quality import build_fact_cards, validate_fact_snapshot
from litagent.report_audit import ReportAuditService, atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions, verification_enabled

ORIGIN = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-0-mini-260428"
LIMITS = {"survey": (500000, 64), "judge": (100000, 2), "jev": (200000, 20)}
VERSION = "frozen-generation-validation-v1"
ACCOUNTING_CORRECTION_VERSION = "frozen-generation-accounting-correction-v1"


def file_sha(path: Path) -> str:
    """Hash exact bytes without rewriting their historical representation."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_accounting_correction(parent: Path) -> dict:
    """Reconstruct received Jev billing without changing its failed verdict or hold."""
    from litagent.benchmark.citation_audit import (
        JevAuditConfig,
        _question_specs,
        _request_payload,
        _response_usage,
        build_citation_audit_inputs,
    )
    from litagent.extraction_verification import VERSION as extraction_version
    from litagent.extraction_verification import (
        build_extraction_inventory,
    )
    from litagent.quality import digest as evidence_digest

    parent = parent.resolve()
    pp, cp, ip = (
        parent / n
        for n in ("execution-plan.json", "checkpoint.json", "input-snapshot.json")
    )
    plan, state, inputs = (json.loads(p.read_text()) for p in (pp, cp, ip))
    archive_path = Path("artifacts/runs") / f"{plan['run_id']}.json"
    archive = json.loads(archive_path.read_text())
    if (
        plan.get("version") != VERSION
        or plan.get("binding")
        != digest({k: v for k, v in plan.items() if k != "binding"})
        or state.get("binding") != plan["binding"]
        or plan.get("quality_policy") != "strict_auto_v2"
        or state.get("status") != "failed"
        or state.get("report") is not None
        or archive.get("report") is not None
        or archive.get("run_id") != plan["run_id"]
        or archive.get("query") != inputs["question"]
        or archive.get("status") != "failed"
        or plan["hashes"].get(str(ip)) != file_sha(ip)
        or any(
            state["costs"][s] != {"tokens": 0, "requests": 0, "usage_complete": True}
            for s in ("survey", "judge")
        )
        or state["costs"]["jev"].get("usage_complete") is not False
        or settle(plan, state["costs"]) != state["cumulative_after"]
        or state.get("total_rounds") != plan["rounds_before"] + 1
    ):
        raise ValueError("frozen_accounting_source_invalid")
    er = state["extraction_receipt"]
    ep = Path(er["path"])
    if file_sha(ep) != er["sha256"]:
        raise ValueError("frozen_accounting_source_changed")
    receipt = json.loads(ep.read_text())
    selection = inputs["evidence_selection"]
    ledger = selection["selected_items"]
    cfg = JevAuditConfig(enabled=True)
    if (
        receipt.get("version") != extraction_version
        or receipt.get("stage") != "extraction"
        or receipt.get("status") != "failed_no_replay"
        or receipt.get("source_run_id") != plan["run_id"]
        or receipt.get("selection") != selection
        or receipt.get("candidate_ledger") != ledger
        or receipt.get("candidate_sha256") != evidence_digest(ledger)
        or receipt.get("binding")
        != evidence_digest(
            [extraction_version, plan["run_id"], selection, cfg.fingerprint]
        )
        or artifact_extraction(archive) != receipt
    ):
        raise ValueError("frozen_accounting_extraction_binding_invalid")
    result = receipt["jev"]
    inventory = build_extraction_inventory(ledger)
    expected = build_citation_audit_inputs(
        inventory, ledger, artifact={"quality_policy": "strict_auto_v2"}
    )
    if (
        result.get("source_run_id") != plan["run_id"]
        or result.get("config_fingerprint") != cfg.fingerprint
        or result.get("inventory_sha256") != inventory.inventory_sha256
        or result.get("ledger_sha256") != inventory.ledger_sha256
        or result.get("inputs") != [i.model_dump(mode="json") for i in expected]
        or result.get("status") not in {"partial", "failed", "cancelled"}
    ):
        raise ValueError("frozen_accounting_input_invalid")
    specs = {
        s.question_id: s
        for s in _question_specs(expected, verification_stage="extraction")
    }
    starts = {}
    ends = {}
    for event in archive["events"]:
        data = event.get("data", {})
        op = data.get("task_id")
        if (
            event["event"] == "subspan.start"
            and data.get("name") == "citation_audit.batch"
        ):
            if not op or op in starts:
                raise ValueError("frozen_accounting_duplicate_operation")
            starts[op] = data
        elif (
            event["event"] == "subspan.end"
            and isinstance(data.get("output"), dict)
            and "batch_id" in data["output"]
        ):
            if not op or op in ends:
                raise ValueError("frozen_accounting_duplicate_operation")
            ends[op] = data["output"]
    operations = []
    seen = set()
    questions = set()
    input_tokens = output_tokens = 0
    for batch in result["batches"]:
        op = batch.get("operation_id")
        qids = batch["question_ids"]
        response = batch.get("response")
        if (
            not op
            or op in seen
            or op not in starts
            or op not in ends
            or batch["status"] not in {"completed", "failed", "cancelled"}
            or not isinstance(response, dict)
            or response.get("model") != cfg.model
            or not isinstance(qids, list)
            or not 1 <= len(qids) <= 16
            or len(set(qids)) != len(qids)
            or questions.intersection(qids)
            or any(q not in specs for q in qids)
            or batch["request"] != _request_payload([specs[q] for q in qids], cfg.model)
            or starts[op].get("input") != batch["request"]
            or any(
                ends[op].get(k) != batch.get(k)
                for k in (
                    "operation_id",
                    "batch_id",
                    "question_ids",
                    "request",
                    "response",
                    "status",
                )
            )
        ):
            raise ValueError("frozen_accounting_response_binding_invalid")
        usage = _response_usage(response)
        if not usage.usage_complete:
            raise ValueError("frozen_accounting_usage_unknown")
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        seen.add(op)
        questions.update(qids)
        operations.append(
            {
                "operation_id": op,
                "batch_id": batch["batch_id"],
                "request_sha256": digest(batch["request"]),
                "response_sha256": digest(response),
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }
        )
    if (
        not operations
        or seen != set(starts)
        or seen != set(ends)
        or len(operations) != state["costs"]["jev"]["requests"]
        or len(operations) > 20
        or input_tokens > 200000
    ):
        raise ValueError("frozen_accounting_operation_coverage_invalid")
    charge = (
        state["cumulative_after"]["jev"]["charged"]
        - plan["cumulative_before"]["jev"]["charged"]
    )
    if charge != 200000 or input_tokens > charge:
        raise ValueError("frozen_accounting_hold_invalid")
    record = {
        "version": ACCOUNTING_CORRECTION_VERSION,
        "actor": "system",
        "purpose": "accounting_only",
        "source_root": str(parent),
        "source_run_id": plan["run_id"],
        "source_binding": plan["binding"],
        "source_hashes": {
            str(p.resolve()): file_sha(p) for p in (pp, cp, ip, ep, archive_path)
        },
        "source_status": state["status"],
        "scientific_status": "unchanged_failed",
        "actual_usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "call_count": len(operations),
        },
        "retained_budget_charge": {"tokens": charge, "requests": len(operations)},
        "operations": operations,
    }
    record["binding"] = digest(record)
    return record


def validate_accounting_correction(path: Path, parent: Path) -> dict:
    """Verify an appended correction against unchanged original billing evidence."""
    record = json.loads(path.read_text())
    if record != build_accounting_correction(parent):
        raise ValueError("frozen_accounting_correction_invalid")
    return record


class FrozenSelection:
    """Return the frozen selector output without ranking or external requests."""

    def __init__(self, inputs: dict):
        """Bind both the question and the complete selected evidence."""
        self.inputs = inputs

    async def select(self, query, ledger, **kwargs):
        """Refuse changed evidence instead of silently rebuilding the pool."""
        selection = EvidenceSelection.from_dict(self.inputs["evidence_selection"])
        if query != self.inputs["question"] or any(
            digest(ledger.get(eid)) != digest(item)
            for eid, item in selection.selected_items.items()
        ):
            raise ValueError("frozen_generation_selection_changed")
        return selection


def _survey_usage(artifact: dict, operations: set[str]) -> dict:
    """Rebuild known transport usage without accepting duplicated operations."""
    opened, completed = {}, set()
    tokens = 0
    for event in artifact["events"]:
        data, kind = event["data"], event["event"]
        if kind == "llm.start":
            op, task = data.get("operation_id"), data.get("task_id")
            if (
                not isinstance(op, str)
                or not op
                or op in opened
                or op in operations
                or not isinstance(task, str)
            ):
                raise ValueError("frozen_generation_history_lifecycle_invalid")
            opened[op] = task
        elif kind == "llm.failed":
            raise ValueError("frozen_generation_history_usage_unknown")
        elif kind == "llm.complete":
            op = data.get("operation_id")
            if op not in opened or op in completed or data.get("task_id") != opened[op]:
                raise ValueError("frozen_generation_history_lifecycle_invalid")
            if data.get("usage_available") is not True or any(
                type(data.get(k)) is not int or data[k] < 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                raise ValueError("frozen_generation_history_usage_unknown")
            if (
                data["total_tokens"]
                != data["prompt_tokens"] + data["completion_tokens"]
            ):
                raise ValueError("frozen_generation_history_usage_invalid")
            tokens += data["total_tokens"]
            completed.add(op)
    if (
        set(opened) != completed
        or type(artifact["usage"].get("total_tokens")) is not int
        or artifact["usage"]["total_tokens"] != tokens
    ):
        raise ValueError("frozen_generation_history_usage_unknown")
    operations.update(completed)
    return {"tokens": tokens, "requests": len(opened), "usage_complete": True}


def matrix_stop_reason(state: dict) -> str | None:
    """Preserve scientific failures while stopping incomplete or invalid execution."""
    if state.get("status") not in {"completed", "blocked", "failed"}:
        return "frozen_matrix_parent_not_terminal"
    if any(
        state.get("costs", {}).get(s, {}).get("usage_complete") is not True
        for s in LIMITS
    ):
        return "frozen_matrix_usage_unknown"
    if state.get("budget", {}).get("stop_reason"):
        return str(state["budget"]["stop_reason"])
    reasons = state.get("reason", [])
    reasons = [reasons] if isinstance(reasons, str) else list(reasons)
    report = state.get("report") or {}
    metadata = report.get("metadata") or {}
    reasons.extend(metadata.get("automatic_acceptance", {}).get("failure_codes", []))
    reasons.extend(_candidate_evaluation_failures(metadata)[0])
    for value in (report.get("evaluation") or {}).values():
        details = value.get("details", {})
        reasons.extend(
            details.get(k)
            for k in ("skipped_reason", "unverified_reason")
            if details.get(k)
        )
    if any(r.startswith("frozen_") for r in reasons):
        return next(r for r in reasons if r.startswith("frozen_"))
    technical = _technical_failure(reasons)
    if technical:
        return technical
    if state.get("status") == "failed":
        semantic = {
            "strict_writing_uncited_science",
            "strict_writing_uncited_scientific_gap",
            "strict_writing_scientific_section_missing",
            "strict_writing_summary_content_duplicate",
        }
        if not reasons or any(r.split(":", 1)[0] not in semantic for r in reasons):
            return "frozen_matrix_unclassified_execution_failure"
    return None


def validate_matrix(path: Path, root: Path, run_id: str, parent: Path, config):
    """Validate the complete ordered public-input matrix before permitting an edge."""
    try:
        matrix = json.loads(path.read_text())
        cases = matrix["cases"]
        if (
            matrix["version"] != "frozen-three-domain-v1"
            or [c["domain"] for c in cases] != ["vit", "few-shot", "nerf"]
            or len({c["run_id"] for c in cases}) != 3
            or len({str(Path(c["root"]).resolve()) for c in cases}) != 3
            or matrix["config"] != build_config_summary(config, "strict_auto_v2")
            or type(matrix["rounds_before"]) is not int
            or not 0 <= matrix["rounds_before"] <= 15
        ):
            raise ValueError("frozen_matrix_contract_invalid")
        code = {
            str(p.resolve()): file_sha(p)
            for p in sorted(Path("src/litagent").rglob("*.py"))
        }
        code[str(Path(__file__).resolve())] = file_sha(Path(__file__))
        if matrix["code_hashes"] != code:
            raise ValueError("frozen_matrix_code_changed")
        hashes = {str(path.resolve()): file_sha(path)}
        inputs = []
        for case in cases:
            ip = Path(case["root"]) / "input-snapshot.json"
            if file_sha(ip) != case["input_sha256"]:
                raise ValueError("frozen_matrix_input_changed")
            value = json.loads(ip.read_text())
            if any(
                value.get(k) != v
                for k, v in {
                    "quality_policy": "strict_auto_v1",
                    "review_structure": "academic_v2",
                    "report_language": "zh-CN",
                    "benchmark_labels_in_generation": False,
                }.items()
            ):
                raise ValueError("frozen_matrix_input_policy_invalid")
            selection = EvidenceSelection.from_dict(value["evidence_selection"])
            validate_fact_snapshot(
                value["fact_cards"],
                selection.selected_items,
                value["fact_cards_sha256"],
            )
            if digest(build_fact_cards(selection.selected_items)) != digest(
                value["fact_cards"]
            ):
                raise ValueError("frozen_matrix_cards_changed")
            sp = Path(value["source_artifact"])
            source = json.loads(sp.read_text())
            if (
                file_sha(sp) != value["source_artifact_sha256"]
                or source["query"] != value["question"]
                or source["run_id"] != value["source_run_id"]
            ):
                raise ValueError("frozen_matrix_source_changed")
            hashes.update(
                {str(ip.resolve()): file_sha(ip), str(sp.resolve()): file_sha(sp)}
            )
            inputs.append(value)
        index = next(
            i
            for i, c in enumerate(cases)
            if c["run_id"] == run_id and Path(c["root"]).resolve() == root.resolve()
        )
        anchor = Path(matrix["parent_root"]).resolve()
        ap = json.loads((anchor / "execution-plan.json").read_text())
        if (
            ap["binding"] != matrix["parent_binding"]
            or ap["rounds_before"] + 1 != matrix["rounds_before"]
        ):
            raise ValueError("frozen_matrix_anchor_changed")
        expected_parent = (
            anchor if index == 0 else Path(cases[index - 1]["root"]).resolve()
        )
        if parent is None or parent.resolve() != expected_parent:
            raise ValueError("frozen_matrix_order_invalid")
        ref = {"path": str(path.resolve()), "sha256": file_sha(path), "index": index}
        for i in range(index):
            prior = Path(cases[i]["root"])
            pp = json.loads((prior / "execution-plan.json").read_text())
            state = json.loads((prior / "checkpoint.json").read_text())
            if (
                pp.get("matrix") != {**ref, "index": i}
                or pp["run_id"] != cases[i]["run_id"]
                or pp["rounds_before"] != matrix["rounds_before"] + i
                or pp.get("quality_policy") != "strict_auto_v2"
                or Path(pp["parent_trial"]["root"]).resolve()
                != (anchor if i == 0 else Path(cases[i - 1]["root"]).resolve())
            ):
                raise ValueError("frozen_matrix_prior_identity_invalid")
            reason = matrix_stop_reason(state)
            if reason:
                raise ValueError("frozen_matrix_stopped:" + reason)
        old = json.loads((anchor / "input-snapshot.json").read_text())
        sequence = [old] + inputs
        edges = {
            (digest(sequence[i + 1]), digest(sequence[i])) for i in range(index + 1)
        }
        return ref, edges, hashes, matrix["rounds_before"] + index
    except (KeyError, TypeError, StopIteration, OSError, json.JSONDecodeError) as exc:
        raise ValueError("frozen_matrix_contract_invalid") from exc


def _parent_history(
    parent: Path,
    history: Path,
    inputs: dict,
    before: dict,
    rounds: int,
    operations: set[str],
    run_id: str,
    seen=None,
    corrections=None,
    allowed_transitions=None,
):
    """Validate immutable terminal receipts before adding a new contract attempt."""
    parent = parent.resolve()
    seen = set() if seen is None else seen
    if parent in seen:
        raise ValueError("frozen_generation_parent_cycle")
    seen.add(parent)
    plan_path, state_path, input_path = (
        parent / name
        for name in ("execution-plan.json", "checkpoint.json", "input-snapshot.json")
    )
    plan, state, prior_input = (
        json.loads(p.read_text()) for p in (plan_path, state_path, input_path)
    )
    refs = dict(plan.get("accounting_corrections", {}))
    for target, ref in (corrections or {}).items():
        if target in refs and refs[target] != ref:
            raise ValueError("frozen_accounting_reference_conflict")
        refs[target] = ref
    own_ref = refs.get(str(parent))
    correction = None
    applied = {}
    if own_ref:
        path = Path(own_ref["path"])
        if file_sha(path) != own_ref["sha256"]:
            raise ValueError("frozen_accounting_source_changed")
        correction = validate_accounting_correction(path, parent)
    if (
        plan.get("version") != VERSION
        or plan.get("binding")
        != digest({k: v for k, v in plan.items() if k != "binding"})
        or state.get("binding") != plan["binding"]
        or state.get("status") not in {"failed", "blocked", "completed"}
        or plan.get("run_id") == run_id
        or (
            digest(prior_input) != digest(inputs)
            and (digest(inputs), digest(prior_input))
            not in (allowed_transitions or set())
        )
        or plan.get("hashes", {}).get(str(history.resolve())) != file_sha(history)
        or plan.get("hashes", {}).get(str(input_path)) != file_sha(input_path)
        or plan.get("max_new_rounds") != 1
        or type(plan.get("max_new_rounds")) is not int
        or plan.get("max_total_rounds") != 18
        or plan.get("limits")
        != {s: {"tokens": t, "requests": r} for s, (t, r) in LIMITS.items()}
    ):
        raise ValueError("frozen_generation_parent_binding_invalid")
    if plan.get("parent_trial"):
        previous = plan["parent_trial"]
        before, rounds, receipt = _parent_history(
            Path(previous["root"]),
            history,
            prior_input,
            before,
            rounds,
            operations,
            run_id,
            seen,
            refs,
            allowed_transitions,
        )
        applied = dict(receipt.get("accounting_corrections", {}))
        if digest(receipt) != digest(previous):
            raise ValueError("frozen_generation_parent_receipt_changed")
    if (
        digest(before) != digest(plan["cumulative_before"])
        or type(plan.get("rounds_before")) is not int
        or plan["rounds_before"] != rounds
    ):
        raise ValueError("frozen_generation_parent_budget_reset")
    archive_path = Path("artifacts/runs") / f"{plan['run_id']}.json"
    archive = json.loads(archive_path.read_text())
    if (
        archive.get("run_id") != plan["run_id"]
        or archive.get("query") != prior_input["question"]
        or archive.get("status") not in {"completed", "failed"}
        or digest(archive.get("report")) != digest(state.get("report"))
    ):
        raise ValueError("frozen_generation_parent_archive_invalid")
    actual = _survey_usage(archive, operations)
    budget = CostBudget.from_snapshot(state["budget"])
    snapshot = budget.snapshot()
    if (
        actual != state["costs"]["survey"]
        or snapshot.get("usage_uncertain")
        or snapshot["used_tokens"] != actual["tokens"]
        or snapshot["requests_sent"] != actual["requests"]
        or snapshot["call_count"] != actual["requests"]
    ):
        raise ValueError("frozen_generation_parent_usage_invalid")
    audit_receipt = state.get("audit_receipt")
    extraction_receipt = state.get("extraction_receipt")
    extraction = {}
    if plan.get("quality_policy") == "strict_auto_v2" and extraction_receipt:
        ep = Path(extraction_receipt["path"])
        if file_sha(ep) != extraction_receipt["sha256"]:
            raise ValueError("frozen_generation_parent_extraction_changed")
        extraction = json.loads(ep.read_text())
        from litagent.extraction_verification import validate_extraction_receipt

        candidates = prior_input["evidence_selection"]["selected_items"]
        verified = {
            eid: candidates[eid]
            for eid in extraction.get("accepted_evidence_ids", [])
            if eid in candidates
        }
        if (
            extraction.get("source_run_id") != plan["run_id"]
            or extraction.get("selection") != prior_input["evidence_selection"]
            or (
                correction is None
                and not validate_extraction_receipt(extraction, verified)
            )
            or artifact_extraction(archive) != extraction
        ):
            raise ValueError("frozen_generation_parent_extraction_binding_invalid")
    audit = {}
    if audit_receipt:
        ap = Path(audit_receipt["path"])
        if file_sha(ap) != audit_receipt["sha256"]:
            raise ValueError("frozen_generation_parent_audit_changed")
        audit = json.loads(ap.read_text())
        body = (state.get("report") or {}).get("survey")
        if (
            not isinstance(body, str)
            or audit.get("source_run_id") != plan["run_id"]
            or type(audit.get("source_report_revision")) is not int
            or audit["source_report_revision"] != 1
            or audit.get("source_report_sha256")
            != hashlib.sha256(body.encode()).hexdigest()
        ):
            raise ValueError("frozen_generation_parent_audit_binding_invalid")
    for service, usage, token_key in (
        ("judge", audit.get("judge_usage", {}), "total_tokens"),
        (
            "jev",
            (
                extraction.get("jev", {}).get("usage", {})
                if plan.get("quality_policy") == "strict_auto_v2"
                else audit.get("jev", {}).get("usage", {})
            ),
            "input_tokens",
        ),
    ):
        cost = state["costs"][service]
        if service == "jev" and correction is not None:
            continue
        if cost.get("usage_complete") is not True:
            raise ValueError("frozen_generation_history_usage_unknown")
        if cost["requests"] or cost["tokens"] or audit_receipt:
            if usage.get("usage_complete") is not True or cost != {
                "tokens": usage.get(token_key),
                "requests": usage.get("call_count"),
                "usage_complete": usage.get("usage_complete"),
            }:
                raise ValueError("frozen_generation_parent_audit_usage_invalid")
    after = settle(plan, state["costs"])
    if (
        digest(after) != digest(state["cumulative_after"])
        or type(state.get("total_rounds")) is not int
        or state["total_rounds"] != rounds + 1
    ):
        raise ValueError("frozen_generation_parent_budget_reset")
    receipt_paths = [plan_path, state_path, input_path, archive_path]
    repair_state_path = parent / "targeted-repair-state.json"
    if repair_state_path.exists():
        receipt_paths.append(repair_state_path)
    if audit_receipt:
        receipt_paths.append(Path(audit_receipt["path"]))
    if extraction_receipt:
        receipt_paths.append(Path(extraction_receipt["path"]))
    if correction is not None:
        # The old checkpoint and conservative charge stay unchanged. Only the
        # newly inherited view knows its original usage has now been verified.
        after["jev"]["usage_complete"] = True
        applied[str(parent)] = own_ref
        receipt_paths.append(Path(own_ref["path"]))
    return (
        after,
        rounds + 1,
        {
            "root": str(parent),
            "binding": plan["binding"],
            "hashes": {str(p.resolve()): file_sha(p) for p in receipt_paths},
            **({"accounting_corrections": applied} if applied else {}),
        },
    )


def artifact_extraction(artifact: dict) -> dict:
    """Read frozen pre-writing results without requiring a generated body."""
    return (
        artifact.get("extraction_verification")
        or (artifact.get("report") or {})
        .get("metadata", {})
        .get("extraction_verification")
        or {}
    )


def prepare(
    root: Path,
    config,
    run_id: str,
    history: Path,
    grant: Path,
    *,
    parent_root: Path | None = None,
    quality_policy: str = "strict_auto_v1",
    accounting_correction: Path | None = None,
    matrix_path: Path | None = None,
) -> dict:
    """Bind a single increment to intact inputs, code and cumulative accounting."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id):
        raise ValueError("frozen_generation_id_invalid")
    if quality_policy not in {"strict_auto_v1", "strict_auto_v2"}:
        raise ValueError("frozen_generation_policy_invalid")
    if (
        config.llm.base_url.rstrip("/") != ORIGIN
        or config.llm.model != MODEL
        or config.safety.max_cost_tokens != 500000
        or config.eval.strict_max_claims != 128
        or config.eval.strict_max_batches != 8
    ):
        raise ValueError("frozen_generation_configuration_invalid")
    inputs = json.loads((root / "input-snapshot.json").read_text())
    if any(
        inputs.get(k) != v
        for k, v in {
            "quality_policy": "strict_auto_v1",
            "review_structure": "academic_v2",
            "report_language": "zh-CN",
            "benchmark_labels_in_generation": False,
        }.items()
    ):
        raise ValueError("frozen_generation_input_policy_invalid")
    selection = EvidenceSelection.from_dict(inputs["evidence_selection"])
    cards = inputs["fact_cards"]
    validate_fact_snapshot(cards, selection.selected_items, inputs["fact_cards_sha256"])
    if digest(build_fact_cards(selection.selected_items)) != digest(cards):
        raise ValueError("frozen_generation_fact_cards_changed")
    source = Path(inputs["source_artifact"])
    if file_sha(source) != inputs["source_artifact_sha256"]:
        raise ValueError("frozen_generation_source_changed")
    source_run = json.loads(source.read_text())
    if source_run["run_id"] == run_id or source_run["query"] != inputs["question"]:
        raise ValueError("frozen_generation_requires_new_identity")
    previous = json.loads(history.read_text())
    authorization = json.loads(grant.read_text())
    if (
        authorization.get("actor") != "user"
        or "后续不要我审批" not in authorization.get("text", "")
        or previous.get("status") != "stopped"
    ):
        raise ValueError("frozen_generation_scope_unavailable")
    observations = previous["observations"]
    before = copy.deepcopy(previous["budgets"])
    operations = set()
    for observation in observations:
        path = Path(observation["artifact_path"])
        if observation.get("artifact_sha256") != "sha256:" + file_sha(path):
            raise ValueError("frozen_generation_history_archive_changed")
        artifact = json.loads(path.read_text())
        if (
            artifact.get("run_id") != observation.get("run_id")
            or _survey_usage(artifact, operations) != observation["costs"]["survey"]
        ):
            raise ValueError("frozen_generation_history_usage_invalid")
    for service, (tokens, requests) in LIMITS.items():
        costs = [o["costs"][service] for o in observations]
        if (
            any(
                type(before[service].get(k)) is not int or before[service][k] < 0
                for k in ("charged", "requests")
            )
            or not all(c.get("usage_complete") is True for c in costs)
            or any(
                type(c.get(k)) is not int or c[k] < 0
                for c in costs
                for k in ("tokens", "requests")
            )
            or sum(c["tokens"] for c in costs) != before[service]["charged"]
            or sum(c["requests"] for c in costs) != before[service]["requests"]
            or before[service]["charged"] + tokens > tokens * 18
            or before[service]["requests"] + requests > requests * 18
        ):
            raise ValueError("frozen_generation_history_usage_invalid")
    rounds, parent_receipt = len(observations), None
    corrections = {}
    if accounting_correction:
        if parent_root is None:
            raise ValueError("frozen_accounting_parent_required")
        record = validate_accounting_correction(accounting_correction, parent_root)
        corrections[record["source_root"]] = {
            "path": str(accounting_correction.resolve()),
            "sha256": file_sha(accounting_correction),
        }
    matrix_ref, edges, matrix_hashes, matrix_round = None, None, {}, None
    if matrix_path:
        if quality_policy != "strict_auto_v2":
            raise ValueError("frozen_matrix_policy_invalid")
        matrix_ref, edges, matrix_hashes, matrix_round = validate_matrix(
            matrix_path, root, run_id, parent_root, config
        )
    if parent_root:
        before, rounds, parent_receipt = _parent_history(
            parent_root,
            history,
            inputs,
            before,
            rounds,
            operations,
            run_id,
            corrections=corrections,
            allowed_transitions=edges,
        )
    if matrix_ref and rounds != matrix_round:
        raise ValueError("frozen_matrix_rounds_changed")
    if rounds >= 18 or any(
        before[s]["charged"] + t > t * 18 or before[s]["requests"] + count > count * 18
        for s, (t, count) in LIMITS.items()
    ):
        raise ValueError("frozen_generation_round_limit")
    hashes = {
        str(p.resolve()): file_sha(p)
        for p in sorted(Path("src/litagent").rglob("*.py"))
    }
    hashes[str(Path(__file__).resolve())] = file_sha(Path(__file__))
    wrapper = root / "runtime-wrapper.py"
    if wrapper.exists():
        hashes[str(wrapper.resolve())] = file_sha(wrapper)
    for path in (source, history, grant, root / "input-snapshot.json"):
        hashes[str(path.resolve())] = file_sha(path)
    protected = json.loads((root / "protected-before.json").read_text())["hashes"]
    if matrix_ref and parent_receipt:
        protected = {**protected, **parent_receipt["hashes"]}
    for path, sha in protected.items():
        if file_sha(Path(path)) != sha:
            raise ValueError("frozen_generation_protected_history_changed")
    latest = []
    for path in protected:
        candidate = Path(path)
        pp = candidate.parent / "execution-plan.json"
        if candidate.name != "checkpoint.json" or not pp.exists():
            continue
        prior_plan = json.loads(pp.read_text())
        if prior_plan.get("version") != VERSION or prior_plan.get("hashes", {}).get(
            str(history.resolve())
        ) != file_sha(history):
            continue
        latest.append((prior_plan["rounds_before"] + 1, candidate.parent.resolve()))
    if latest and (
        parent_root is None
        or max(latest, key=lambda item: item[0])[1] != parent_root.resolve()
    ):
        raise ValueError("frozen_generation_latest_parent_required")
    hashes.update(protected)
    hashes.update(matrix_hashes)
    if parent_receipt:
        hashes.update(parent_receipt["hashes"])
    plan = {
        "version": VERSION,
        "run_id": run_id,
        "scope": "one_new_contract_frozen_generation_validation",
        "endpoints": {
            "survey_judge": ORIGIN,
            "jev": "https://api.typesafe.ai/v1/systemone",
        },
        "models": {"survey_judge": MODEL, "jev": "jev-1.13.0"},
        "formal_benchmark": False,
        "search_requests": 0,
        "limits": {s: {"tokens": t, "requests": r} for s, (t, r) in LIMITS.items()},
        "cumulative_before": before,
        "rounds_before": rounds,
        "max_new_rounds": 1,
        "max_total_rounds": 18,
        "remaining_rounds_after_reservation": 17 - rounds,
        "history_reference": {
            "path": str(history.resolve()),
            "sha256": file_sha(history),
        },
        "parent_trial": parent_receipt,
        **({"matrix": matrix_ref} if matrix_ref else {}),
        **(
            {"accounting_corrections": parent_receipt["accounting_corrections"]}
            if parent_receipt and parent_receipt.get("accounting_corrections")
            else {}
        ),
        "authorization_reference": {"path": str(grant), "sha256": file_sha(grant)},
        "manual_approval_required": False,
        "retries": 0,
        "config": build_config_summary(config, quality_policy),
        **(
            {"quality_policy": quality_policy}
            if quality_policy == "strict_auto_v2"
            else {}
        ),
        "hashes": hashes,
        "runtime": {
            "python": sys.version,
            "packages": {
                name: version(name) for name in ("openai", "pydantic", "langfuse")
            },
        },
    }
    plan["binding"] = digest(plan)
    return plan


def validate_frozen_worker_evidence(
    result: dict, inputs: dict, *, expected_run_id=None, require_verification=True
) -> dict:
    """Validate the compact producer against its full verified, frozen receipt."""
    from litagent.extraction_verification import validate_extraction_receipt
    from litagent.quality import digest as evidence_digest

    try:
        original = EvidenceSelection.from_dict(inputs["evidence_selection"]).to_dict()
        if require_verification:
            receipt = result["extraction_verification"]
            if (
                receipt["selection"] != original
                or receipt["candidate_ledger"] != original["selected_items"]
            ):
                raise ValueError("foreign extraction input")
            if (
                expected_run_id is not None
                and receipt.get("source_run_id") != expected_run_id
            ):
                raise ValueError("foreign extraction run")
            accepted = receipt["accepted_evidence_ids"]
            if (
                not isinstance(accepted, list)
                or any(not isinstance(eid, str) for eid in accepted)
                or len(accepted) != len(set(accepted))
            ):
                raise ValueError("invalid accepted set")
            filtered = {eid: original["selected_items"][eid] for eid in accepted}
            if not validate_extraction_receipt(receipt, filtered):
                raise ValueError("invalid extraction receipt")
        else:
            filtered = copy.deepcopy(original["selected_items"])
        expected = copy.deepcopy(original)
        expected.update(
            selected_items=filtered,
            section_evidence_ids={
                key: [eid for eid in ids if eid in filtered]
                for key, ids in original["section_evidence_ids"].items()
            },
            omitted_count=original["candidate_count"] - len(filtered),
        )
        # Synthesis adds shared summary scopes after verification, without new evidence.
        if inputs.get("review_structure", "legacy") != "legacy":
            for key in ("abstract", "conclusion"):
                expected["section_evidence_ids"][key] = list(filtered)
                expected["method_by_section"][key] = "selected_union"
        expected = EvidenceSelection.from_dict(expected).to_dict()
        summary = {
            "valid": True,
            "candidate_count": expected["candidate_count"],
            "selected_count": len(filtered),
            "section_evidence_ids": expected["section_evidence_ids"],
            "method_by_section": expected["method_by_section"],
            "omitted_count": expected["omitted_count"],
            "estimated_tokens": expected["estimated_tokens"],
        }
        actual = result["evidence_selection"]
        if not isinstance(actual, dict) or any(
            type(actual.get(k)) is not int
            for k in ("candidate_count", "omitted_count", "estimated_tokens")
        ):
            raise ValueError("invalid selection")
        if "selected_items" in actual:
            if (
                set(actual) - (set(expected) | {"valid", "selected_count"})
                or ("valid" in actual and actual["valid"] is not True)
                or (
                    "selected_count" in actual
                    and (
                        type(actual["selected_count"]) is not int
                        or actual["selected_count"] != len(filtered)
                    )
                )
                or EvidenceSelection.from_dict(actual).to_dict() != expected
            ):
                raise ValueError("changed full selection")
        elif (
            set(actual) != set(summary)
            or type(actual["selected_count"]) is not int
            or actual["valid"] is not True
            or actual != summary
        ):
            raise ValueError("changed selection summary")
        cards = result["fact_cards"]
        expected_cards = build_fact_cards(filtered)
        if cards != expected_cards or any(
            card != inputs["fact_cards"].get(cid) for cid, card in cards.items()
        ):
            raise ValueError("changed frozen cards")
        # Evidence hashes and plan hashes use different historical JSON encodings.
        validate_fact_snapshot(cards, filtered, evidence_digest(cards))
        if result.get("evidence_snapshot_sha256") not in (None, evidence_digest(cards)):
            raise ValueError("changed evidence hash")
        return copy.deepcopy(filtered)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("frozen_generation_result_evidence_changed") from exc


def settle(plan: dict, costs: dict) -> dict:
    """Keep the full increment reserved whenever service usage is uncertain."""
    result = copy.deepcopy(plan["cumulative_before"])
    for service, limit in plan["limits"].items():
        cost = costs[service]
        known = cost.get("usage_complete") is True
        requests = cost.get("requests")
        if requests is not None and (type(requests) is not int or requests < 0):
            raise ValueError("frozen_generation_usage_invalid")
        if known and any(
            type(cost.get(k)) is not int or cost[k] < 0 for k in ("tokens", "requests")
        ):
            raise ValueError("frozen_generation_usage_invalid")
        if known and (cost["tokens"] > limit["tokens"] or requests > limit["requests"]):
            raise ValueError("frozen_generation_budget_exceeded")
        result[service]["charged"] += cost["tokens"] if known else limit["tokens"]
        result[service]["requests"] += (
            requests if type(requests) is int else limit["requests"]
        )
        result[service]["usage_complete"] = known
    return result


async def run_trial(
    root: Path,
    config,
    plan: dict,
    *,
    live=False,
    client_factory=None,
    worker_factory=None,
    service_factory=None,
    trace_factory=None,
    extraction_auditor_factory=None,
) -> dict:
    """Use real workers and finalization once; a persisted start can never replay."""
    if not live:
        return {"status": "planned", "binding": plan["binding"], "paid_calls": 0}
    policy = plan.get("quality_policy", "strict_auto_v1")
    optional = plan.get("verification") is not None
    selected_verification = {"verification": plan.get("verification")}
    if digest(build_config_summary(config, policy)) != digest(plan["config"]):
        raise ValueError("frozen_generation_configuration_changed")
    for path, sha in plan["hashes"].items():
        if file_sha(Path(path)) != sha:
            raise ValueError("frozen_generation_identity_changed")
    if digest({k: v for k, v in plan.items() if k != "binding"}) != plan["binding"]:
        raise ValueError("frozen_generation_plan_invalid")
    if sys.version != plan["runtime"]["python"] or any(
        version(n) != pin for n, pin in plan["runtime"]["packages"].items()
    ):
        raise ValueError("frozen_generation_runtime_changed")
    checkpoint = root / "checkpoint.json"
    with suite_lock(root / ".trial.lock"):
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state["binding"] != plan["binding"]:
                raise ValueError("frozen_generation_identity_conflict")
            if state["status"] == "started_no_replay":
                state.update(
                    status="interrupted",
                    reason="frozen_generation_interrupted_no_replay",
                )
                atomic_json(checkpoint, state)
            return state
        if (Path("artifacts/runs") / f"{plan['run_id']}.json").exists():
            raise ValueError("frozen_generation_output_collision")
        budget = CostBudget()
        budget.configure_strict(halt_on_unknown_usage=True)
        state = {
            "binding": plan["binding"],
            "status": "started_no_replay",
            "cumulative_reserved": settle(plan, {s: {} for s in LIMITS}),
        }
        atomic_json(checkpoint, state)
        inputs = json.loads((root / "input-snapshot.json").read_text())
        source = json.loads(Path(inputs["source_artifact"]).read_text())
        run_id = plan["run_id"]
        recorder = RunRecorder(
            run_id, inputs["question"], ArchiveRepository(Path("artifacts/runs"))
        )
        recorder.set_config_summary(plan["config"])
        recorder.set_citation_style("ieee")
        tracer = client = service = report = hook = graph = None
        audit = {}
        try:
            tracer = (
                trace_factory(run_id)
                if trace_factory
                else (
                    LangFuseTracer(
                        config.observability.langfuse_host,
                        os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                        os.getenv("LANGFUSE_SECRET_KEY", ""),
                        trace_seed=run_id,
                    )
                    if config.observability.enabled and not client_factory
                    else None
                )
            )

            def journal(event, data):
                """Persist raw requests/responses locally before losing process state."""
                with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(
                            {"event": event, "data": data},
                            ensure_ascii=False,
                            default=str,
                        )
                        + "\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())

            hook = CompositeTraceHook(
                recorder,
                journal,
                (
                    RedactingTraceHook(tracer, payload_mode="metadata_only")
                    if tracer
                    else None
                ),
            )
            hook(
                "survey.start",
                {
                    "session_id": run_id,
                    "query": inputs["question"],
                    "quality_policy": policy,
                },
            )
            client = (client_factory or OpenAICompatibleClient)(
                base_url=ORIGIN,
                model=MODEL,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                cost_budget=budget,
                trace_hook=hook,
                transport_max_attempts=1,
            )
            if not optional or any(
                verification_enabled(selected_verification, k) for k in ("judge", "jev")
            ):
                service = (service_factory or ReportAuditService)(
                    config, root=root / "audits"
                )
                capabilities = service.capabilities()
                judge = service.judge_config()
                if (
                    verification_enabled(selected_verification, "judge")
                    and (
                        not capabilities.get("judge_available")
                        or judge.base_url.rstrip("/") != ORIGIN
                        or judge.model != MODEL
                    )
                ) or (
                    verification_enabled(selected_verification, "jev")
                    and not capabilities.get("jev_available")
                ):
                    raise ValueError("frozen_generation_audit_configuration_invalid")
            agent = LitAgent(
                config,
                trace_hook=hook,
                audit_service=service,
                run_policy=RunPolicy.isolated_benchmark(),
            )
            agent._session_id, agent._llm, agent._cost_budget = run_id, client, budget
            agent._targeted_repair_state_path = root / "targeted-repair-state.json"
            # Direct frozen execution skips run(), which normally binds this policy.
            agent._config_summary = copy.deepcopy(plan["config"])
            from litagent.extraction_verification import ExtractionVerifier

            verifier = ExtractionVerifier(
                root / "extraction",
                auditor=(
                    extraction_auditor_factory(hook)
                    if extraction_auditor_factory
                    else None
                ),
                trace_hook=hook,
            )
            agent._synthesis = SynthesisWorker(
                client,
                context_config=config.context,
                evidence_selector=FrozenSelection(inputs),
                trace_hook=hook,
                thinking_mode=config.adversarial.thinking_mode,
                extraction_verifier=verifier,
                run_id_provider=lambda: run_id,
                repair_state_path_provider=agent._targeted_repair_path,
            )
            reviewer = ReviewerWorker(
                client,
                config=config.adversarial,
                context_config=config.context,
                trace_hook=hook,
            )
            worker = (worker_factory or AdversarialReviewWorker)(
                client,
                agent._synthesis,
                reviewer,
                max_rounds=config.adversarial.max_rounds,
                pass_threshold=config.adversarial.pass_threshold,
                trace_hook=hook,
                thinking_mode=config.adversarial.thinking_mode,
            )
            results = {
                n["task_id"]: copy.deepcopy(n["output"])
                for n in source["nodes"].values()
                if n["kind"] == "worker" and n["task_id"] != "adversarial_review"
            }
            # Prior reports/evaluations/labels are not passed to the writing worker.
            upstream = {
                "extract": LitAgent._collect_extractions(results),
                "graph_analysis": results.get("graph_analysis", {}),
            }
            await FrozenSelection(inputs).select(
                inputs["question"], collect_ledger(upstream["extract"])
            )
            task = SubTask(
                "adversarial_review",
                "New writing on frozen public evidence",
                "adversarial_review",
                input_data={
                    "query": inputs["question"],
                    "upstream_results": upstream,
                    "quality_policy": policy,
                    **(
                        {
                            "verification": plan["verification"],
                            "body_quality": plan.get("body_quality", "legacy"),
                        }
                        if optional
                        else {}
                    ),
                    "report_language": "zh-CN",
                    "review_structure": "academic_v2",
                    "coverage_requirements": inputs["coverage_requirements"],
                    "review_methodology": inputs["review_methodology"],
                },
                max_retries=0,
            )
            graph = TaskGraph()
            for tid, value in upstream.items():
                graph.add_task(
                    SubTask(
                        tid,
                        "Frozen prior evidence output",
                        tid,
                        status=TaskStatus.DONE,
                        result=value,
                        max_retries=0,
                    )
                )
            graph.add_task(task, depends_on=list(upstream))
            hook.capture_graph(graph)
            reused = dict(upstream)
            for tid in ("dedup", "relevance_gate"):
                value = results.get(tid)
                if isinstance(value, list):
                    # Import source metadata only; benchmark labels never enter writing.
                    reused[tid] = [
                        {
                            k: p[k]
                            for k in (
                                "paper_id",
                                "title",
                                "authors",
                                "year",
                                "author_details",
                                "doi",
                                "url",
                                "journal",
                                "venue",
                                "volume",
                                "issue",
                                "pages",
                            )
                            if k in p
                        }
                        for p in value
                        if isinstance(p, dict)
                    ]
            for tid, value in reused.items():
                hook(
                    "worker.complete",
                    {
                        "task_id": tid,
                        "agent_type": tid,
                        "name": "Frozen prior evidence output",
                        "output": value,
                        "elapsed_ms": 0,
                        "execution": "reused_frozen_input",
                        "source_run_id": inputs["source_run_id"],
                        "source_artifact_sha256": inputs["source_artifact_sha256"],
                    },
                )
            hook(
                "worker.start", {"task_id": task.task_id, "agent_type": task.agent_type}
            )
            results[task.task_id] = await worker.execute(task)
            graph.mark_done(task.task_id, results[task.task_id])
            hook(
                "worker.complete",
                {"task_id": task.task_id, "output": results[task.task_id]},
            )
            result_cards = results[task.task_id].get("fact_cards", {})
            if policy == "strict_auto_v2":
                validate_frozen_worker_evidence(
                    results[task.task_id],
                    inputs,
                    expected_run_id=run_id,
                    require_verification=verification_enabled(
                        selected_verification, "jev"
                    ),
                )
            elif digest(result_cards) != digest(inputs["fact_cards"]):
                raise ValueError("frozen_generation_result_evidence_changed")
            if plan.get("evaluation_body_quality"):
                results[task.task_id]["body_quality"] = plan["evaluation_body_quality"]
            agent._evaluators = [
                CitationEvaluator(client),
                ConsistencyEvaluator(client),
                RagasFaithfulnessEvaluator(config, llm=client),
            ]
            report, degradation = await agent._prepare_report(
                query=inputs["question"],
                graph=graph,
                results=results,
                options=RunOptions(
                    quality_policy=policy,
                    **(
                        {
                            "verification": plan["verification"],
                            "body_quality": plan.get(
                                "evaluation_body_quality",
                                plan.get("body_quality", "legacy"),
                            ),
                        }
                        if optional
                        else {}
                    ),
                    report_language="zh-CN",
                    review_structure="academic_v2",
                ),
                search_outcomes=[],
                degradation_codes=[],
                cancellation=None,
            )
            report["metadata"]["frozen_generation_provenance"] = {
                "source_run_id": inputs["source_run_id"],
                "source_sha256": inputs["source_artifact_sha256"],
                "input_sha256": file_sha(root / "input-snapshot.json"),
                "binding": plan["binding"],
                "scope": plan["scope"],
                "generation_body_quality": plan.get("body_quality", "legacy"),
                "common_evaluation_body_quality": plan.get(
                    "evaluation_body_quality", "legacy"
                ),
                "search_performed_in_this_trial": False,
            }
            await agent._automatic_review(report, inputs["question"], results, None)
            acceptance = report["metadata"]["automatic_acceptance"]
            if acceptance.get("audit_id"):
                audit = await service.wait(acceptance["audit_id"]) or {}
            report = await agent._finalize_report(
                query=inputs["question"],
                report_data=report,
                results=results,
                degradation_codes=degradation,
                cancellation=None,
            )
            state.update(
                status="completed" if report["delivery"]["publishable"] else "blocked",
                report_sha256=hashlib.sha256(report["survey"].encode()).hexdigest(),
                reason=report["delivery"].get("reason_codes", []),
            )
        except BaseException as exc:
            state.update(
                status=(
                    "interrupted"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed"
                ),
                reason=(
                    str(exc)
                    if str(exc).startswith(
                        ("frozen_", "strict_", "evaluation_", "extraction_")
                    )
                    else f"frozen_generation_error:{type(exc).__name__}"
                ),
            )
            if report:
                report["delivery"] = {"status": "blocked", "publishable": False}
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
        finally:
            snapshot = budget.snapshot()
            extraction_path = root / "extraction" / run_id / "receipt.json"
            extraction = {}
            if policy == "strict_auto_v2" and extraction_path.exists():
                extraction = json.loads(extraction_path.read_text())
                state["extraction_receipt"] = {
                    "path": str(extraction_path.resolve()),
                    "sha256": file_sha(extraction_path),
                }
                state["extraction_verification"] = extraction
            jev_usage = (
                extraction.get("jev", {}).get("usage", {})
                if policy == "strict_auto_v2"
                else audit.get("jev", {}).get("usage", {})
            )
            if audit:
                audit_path = root / "audit-result.json"
                atomic_json(audit_path, audit)
                state["audit_receipt"] = {
                    "path": str(audit_path.resolve()),
                    "sha256": file_sha(audit_path),
                }
            acceptance = (
                (report or {}).get("metadata", {}).get("automatic_acceptance", {})
            )
            audit_unknown = bool(acceptance.get("audit_id")) and not audit
            costs = {
                "survey": {
                    "tokens": snapshot["used_tokens"],
                    "requests": snapshot["requests_sent"],
                    "usage_complete": not snapshot["usage_uncertain"],
                },
                "judge": {
                    "tokens": audit.get("judge_usage", {}).get("total_tokens", 0),
                    "requests": audit.get("judge_usage", {}).get("call_count", 0),
                    "usage_complete": audit.get("judge_usage", {}).get(
                        "usage_complete", not audit and not audit_unknown
                    ),
                },
                "jev": {
                    "tokens": jev_usage.get("input_tokens", 0),
                    "requests": jev_usage.get("call_count", 0),
                    "usage_complete": jev_usage.get(
                        "usage_complete",
                        (
                            not extraction
                            if policy == "strict_auto_v2"
                            else not audit and not audit_unknown
                        ),
                    ),
                },
            }
            state.update(
                budget=snapshot,
                costs=costs,
                cumulative_after=settle(plan, costs),
                total_rounds=plan["rounds_before"] + 1,
                report=report,
            )
            if extraction:
                recorder.set_extraction_verification(
                    extraction,
                    {
                        "quality_policy": policy,
                        "status": state["status"],
                        "reason": state.get("reason"),
                        "costs": costs,
                        "stages": {
                            "extraction_jev": extraction.get("status"),
                            "writing": (
                                "completed"
                                if report
                                else (
                                    "failed" if snapshot["requests_sent"] else "not_run"
                                )
                            ),
                            "evaluation": (report or {})
                            .get("quality", {})
                            .get("status", "not_run"),
                            "judge": audit.get("judge", {}).get("status", "not_run"),
                        },
                    },
                )
            if hook:
                if graph:
                    hook.capture_graph_state(graph)
                hook(
                    (
                        "survey.error"
                        if state["status"] in {"failed", "interrupted"}
                        else "trial.complete"
                    ),
                    {
                        "session_id": run_id,
                        "status": state["status"],
                        "reason_code": str(state.get("reason")),
                        "error": (
                            state.get("reason")
                            if state["status"] in {"failed", "interrupted"}
                            else None
                        ),
                    },
                )
            if tracer:
                recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
                tracer.flush()
            recorder.finalize(
                report=report,
                terminal_status=(
                    "failed"
                    if state["status"] in {"failed", "interrupted"}
                    else "completed"
                ),
                error=(
                    str(state.get("reason"))
                    if state["status"] in {"failed", "interrupted"}
                    else None
                ),
            )
            atomic_json(checkpoint, state)
            if service:
                await service.shutdown()
            if client:
                await client.close()
        return state


def main():
    """Prepare or execute exactly the persisted bounded plan."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--parent-trial", type=Path)
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--accounting-correction", type=Path)
    parser.add_argument(
        "--quality-policy",
        choices=["strict_auto_v1", "strict_auto_v2"],
        default="strict_auto_v1",
    )
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    plan = prepare(
        args.root,
        config,
        args.run_id,
        args.history,
        args.grant,
        parent_root=args.parent_trial,
        quality_policy=args.quality_policy,
        accounting_correction=args.accounting_correction,
        matrix_path=args.matrix,
    )
    path = args.root / "execution-plan.json"
    if path.exists() and digest(json.loads(path.read_text())) != digest(plan):
        raise ValueError("frozen_generation_plan_changed")
    if not path.exists():
        atomic_json(path, plan)
    result = asyncio.run(run_trial(args.root, config, plan, live=args.live))
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "binding", "reason", "costs", "total_rounds")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
