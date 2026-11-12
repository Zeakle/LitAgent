"""Prepare a single native repair round from settled historical evidence."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from litagent.body_quality import validate_analysis_decisions
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.eval.base import CTX_EVIDENCE
from litagent.eval.batched import limitations_check
from litagent.eval.engine_policy import validate_engine
from litagent.eval.ragas_global_wire import unpack_global_input
from litagent.eval.ragas_native import (
    NativeEvaluationRunResult,
    parse_native_global_response,
)
from litagent.eval.ragas_repair import _followup_budget, recovery_decision
from litagent.extraction_verification import (
    admitted_writing_ledger,
    require_admission_policy,
)
from litagent.quality import digest, targeted_repair_state, validate_fact_snapshot
from litagent.report_audit import atomic_json
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget
from litagent.workbench import extraction_admission_enabled, verification_enabled
from litagent.writing_v2 import WriterSession, compile_document, repair_report


def read_sealed_source(root: Path, proof: Path, frozen_input: Path, *, workspace: Path):
    """Read only the settled parent files covered by the successful regression seal."""
    receipt = json.loads((proof / "default-exit.json").read_bytes())
    manifest_bytes = (proof / "default-source.json").read_bytes()
    log_hash = hashlib.sha256((proof / "default-pytest.log").read_bytes()).hexdigest()
    if (
        receipt["actual_exit_code"] != 0
        or receipt["source_verified"] is not True
        or receipt["new_paid_requests"] != 0
        or receipt["log_sha256"] != log_hash
        or receipt["source_manifest_sha256"]
        != hashlib.sha256(manifest_bytes).hexdigest()
    ):
        raise ValueError("native_trial_regression_proof_invalid")
    manifest = json.loads(manifest_bytes)
    archive = {}
    for path in (
        root / "report.json",
        root / "progress-archive.json",
        root / "final-budget.json",
        root / "repair.json.writer-v2.json",
        root / "launch-exit.json",
        frozen_input,
    ):
        data = path.read_bytes()
        name = str(path.resolve().relative_to(workspace.resolve()))
        if manifest.get(name) != hashlib.sha256(data).hexdigest():
            raise ValueError("native_trial_sealed_source_changed")
        archive[path] = data
    terminal = json.loads(archive[root / "launch-exit.json"])
    if (
        terminal["actual_exit_code"] != 0
        or terminal["all_protected_files_unchanged"] is not True
    ):
        raise ValueError("native_trial_source_not_terminal")
    return archive


class RepairTrialBudget(PlannedBudget):
    """Keep historical accounting and reserve at most one complete new round."""

    def __init__(self, history, plan, *, path: Path | None = None):
        """Rebind equivalent resource geometry to the active native engine."""
        old = history["plan"]
        validate_accounting_snapshot(history, expected_plan=old)
        validate_engine(plan["envelope"]["semantic_engine"])
        if (
            history["usage_uncertain"]
            or history["stop_reason"]
            or any(
                e["status"] != "settled" or not e["sent"]
                for e in history["entries"].values()
            )
        ):
            raise ValueError("native_trial_history_unsettled")
        unchanged = set(old) - {"identity", "envelope"}
        envelope_fields = set(old["envelope"]) - {
            "identity",
            "builder_identity",
            "semantic_engine",
        }
        if (
            set(old) != set(plan)
            or set(old["envelope"]) != set(plan["envelope"])
            or any(plan[k] != old[k] for k in unchanged)
            or any(plan["envelope"][k] != old["envelope"][k] for k in envelope_fields)
        ):
            raise ValueError("native_trial_plan_drift")
        self._trial_limits = _followup_budget(history, edit=True, summary=True)
        self._historical_ids = frozenset(history["entries"])
        self.source_envelope_identity = old["envelope"]["identity"]
        super().__init__(plan)
        self._restore_planned({**history, "plan": plan})
        if path is not None:
            # A trial owns a fresh ledger. A previous start, even before a
            # response was written, is never a reason to replay the trial.
            self.path = Path(path)
            with self.path.open("x", encoding="utf-8") as file:
                json.dump(self.snapshot(), file, ensure_ascii=False)

    @property
    def trial_limits(self):
        """Return the extra allowance without giving callers a mutable owner view."""
        return copy.deepcopy(self._trial_limits)

    def check_stage_requests(self, stage, amounts):
        """Check pending and settled new reservations before any stage dispatch."""
        amounts = tuple(amounts)
        limit = self._trial_limits.get(stage)
        if limit is None:
            raise ValueError("native_trial_stage_not_authorized")
        new = [
            e
            for key, e in self.entries.items()
            if key not in self._historical_ids and e["stage"] == stage
        ]
        super().check_stage_requests(stage, amounts)
        if (
            len(new) + len(amounts) > limit["requests"]
            or sum(e["reserved_tokens"] for e in new) + sum(amounts) > limit["tokens"]
        ):
            raise ValueError("native_trial_allowance_exceeded")

    def reserve(self, tokens):
        """Apply the same one-round limit at the durable send boundary."""
        stage, _ = self._owner()
        self.check_stage_requests(stage, [tokens])
        super().reserve(tokens)

    def restore(self, data):
        """Require a fresh trial rather than reopen any execution checkpoint."""
        raise ValueError("native_trial_no_replay")


def replay_source(report, events, budget, context):
    """Recover the settled global verdict in memory and admit only its exact edits."""
    working = copy.deepcopy(report)
    details = working["evaluation"]["faithfulness"]["details"]
    stages = details["stages"]
    if (
        working["delivery"]["publishable"] is not False
        or details.get("execution_error")
        != "evaluation_global_finding_identity_invalid"
        or stages.get("global_check") != "started_no_replay"
    ):
        raise ValueError("native_trial_source_not_eligible")
    starts = [
        e["data"]
        for e in events
        if e["event"] == "llm.start"
        and (e["data"].get("response_format") or {}).get("json_schema", {}).get("name")
        == "ragas_report_review"
    ]
    if len(starts) != 1:
        raise ValueError("native_trial_global_request_not_unique")
    start = starts[0]
    responses = [
        e["data"]
        for e in events
        if e["event"] == "llm.complete"
        and e["data"]["operation_id"] == start["operation_id"]
    ]
    if len(responses) != 1 or responses[0].get("usage_available") is not True:
        raise ValueError("native_trial_global_usage_unavailable")
    response = responses[0]
    request_hash = hashlib.sha256(
        json.dumps(
            [start["messages"], start["response_format"]], ensure_ascii=False
        ).encode()
    ).hexdigest()
    payload = unpack_global_input(json.loads(start["messages"][1]["content"]))
    parsed = parse_native_global_response(response["content"], payload, stages)
    supported = {
        a["claim_id"] for a in details["assertions"] if a["status"] == "supported"
    }
    omissions = {x.source_unit_id for x in parsed.issues if x.owner == "inventory"}
    if not omissions:
        raise ValueError("native_trial_inventory_conflicts_missing")
    stages.update(
        global_check="completed",
        global_request_id=stages["evaluation_identity"] + ":" + request_hash,
        global_usage={
            k: response[k]
            for k in ("prompt_tokens", "completion_tokens", "usage_available")
        },
        global_findings_contract=payload["global_findings_contract"],
    )
    stages.pop("execution_error")
    for key in (
        "execution_error",
        "unverified_reason",
        "partial_samples",
        "skipped_reason",
    ):
        details.pop(key, None)
    details["coverage"]["complete"] = True
    details.update(
        conflicts=[x.model_dump() for x in parsed.conflicts],
        core_coverage=[x.model_dump() for x in parsed.coverage],
        global_issues=[{**x.model_dump(), "owner": x.owner} for x in parsed.issues],
        inventory_check={
            "status": "failed",
            "conflicting_unit_ids": sorted(
                omissions & {x.source_unit_id for x in parsed.units if x.complete}
            ),
        },
        analysis_check=validate_analysis_decisions(
            parsed.analyses,
            details["assertions"],
            supported,
            payload["report_structure"],
        ),
        limitations_check=limitations_check(
            parsed,
            details["assertions"],
            supported,
            payload["report_structure"],
            independent=True,
        ),
    )
    # Restore the evidence inputs for the repair decision only. The archived
    # delivery and its original failed evaluation remain the external record.
    replayed = NativeEvaluationRunResult.from_details(
        details, details["source_report_sha256"]
    )
    working["evaluation"].update(replayed.metrics())
    decision = recovery_decision(working, context, budget)
    if (
        decision["action"] != "combined"
        or decision["budget"].get("stages") != budget.trial_limits
    ):
        raise ValueError("native_trial_repair_not_admitted:" + decision["reason_code"])
    return working, decision


def prepare_candidate(runner, report, events, results):
    """Bind the same archived text to an independent session and current plan."""
    budget = runner._cost_budget
    if not isinstance(budget, RepairTrialBudget):
        raise ValueError("native_trial_budget_required")
    old_state = report["metadata"]["targeted_repair_state"]
    if runner._session_id == old_state["run_id"] or old_state["attempts"]:
        raise ValueError("native_trial_requires_new_identity")
    query = report["metadata"]["query"]
    context = runner._build_eval_context(
        query=query, survey=report["survey"], results=results
    )
    parent = report["metadata"]["automatic_acceptance"]
    source_id = old_state["run_id"]
    if (
        parent["source_run_id"] != source_id
        or parent["report_sha256"]
        != hashlib.sha256(report["survey"].encode()).hexdigest()
        or type(parent["revision"]) is not int
        or parent["revision"] < 1
        or parent["publishable"] is not False
    ):
        raise ValueError("native_trial_parent_acceptance_invalid")
    validate_source_admission(report["metadata"], context[CTX_EVIDENCE], source_id)
    working, original_decision = replay_source(report, events, budget, context)
    metadata = working["metadata"]
    document = metadata["writer_document"]
    if document["budget_envelope"]["identity"] != budget.source_envelope_identity:
        raise ValueError("native_trial_source_envelope_mismatch")
    survey, bindings = compile_document(
        document["body"],
        document["summaries"],
        metadata["fact_cards"],
        document["aliases"],
        metadata["report_language"],
        metadata["review_methodology"],
        writing_plan=document["generation_writing_plan"],
        envelope=budget.effective_envelope,
    )
    expected = {
        **metadata["writing_bindings"],
        "writer_envelope_identity": budget.effective_envelope["identity"],
    }
    if survey != working["survey"] or bindings != expected:
        raise ValueError("native_trial_source_binding_changed")
    document["budget_envelope"] = budget.effective_envelope
    metadata.update(
        writing_bindings=bindings,
        native_repair_trial={
            "parent_run_id": source_id,
            "parent_report_sha256": parent["report_sha256"],
            "revision": parent["revision"] + 1,
            "attempt_id": runner._session_id,
        },
        targeted_repair_state=targeted_repair_state(
            None, runner._session_id, limit=budget.plan["repair_limit"]
        ),
    )
    working["evaluation"]["faithfulness"]["details"]["writer_bindings"] = bindings
    prepared_results = copy.deepcopy(results)
    prepared_results["adversarial_review"].update(metadata)
    context = runner._build_eval_context(
        query=query, survey=survey, results=prepared_results
    )
    decision = recovery_decision(working, context, budget)
    if any(
        decision[k] != original_decision[k]
        for k in ("action", "targets", "budget", "inventory_findings")
    ):
        raise ValueError("native_trial_edit_scope_changed")
    return working, prepared_results, decision


def validate_source_admission(metadata, available, source_id):
    """Verify settled Jev admission under its immutable logical source ID."""
    if (
        metadata["quality_policy"] != "strict_auto_v2"
        or verification_enabled(metadata, "judge")
        or not extraction_admission_enabled(metadata)
    ):
        raise ValueError("native_trial_unexpected_verification_policy")
    receipt = metadata["extraction_verification"]
    cards = metadata["fact_cards"]
    require_admission_policy(receipt, metadata["verification"])
    admitted = admitted_writing_ledger(receipt, available, cards, run_id=source_id)
    validate_fact_snapshot(cards, admitted, metadata["evidence_snapshot_sha256"])


def record_trial_delivery(runner, report, results, outcome):
    """Run the standard automatic gate on the exact, fully reevaluated version."""
    metadata = report["metadata"]
    trial = metadata["native_repair_trial"]
    body_sha = hashlib.sha256(report["survey"].encode()).hexdigest()
    snapshot = {
        "source_run_id": trial["parent_run_id"],
        "source_report_revision": trial["revision"],
        "source_report_sha256": body_sha,
        "report": report,
        "context": {"review_methodology": metadata["review_methodology"]},
    }
    failure = None if outcome.committed else "native_trial_candidate_not_committed"
    try:
        context = runner._build_eval_context(
            query=metadata["query"], survey=report["survey"], results=results
        )
        validate_source_admission(
            metadata, context[CTX_EVIDENCE], trial["parent_run_id"]
        )
        snapshot["ledger"] = context[CTX_EVIDENCE]
        if outcome.committed:
            details = report["evaluation"]["faithfulness"]["details"]
            if (
                outcome.candidate != report["survey"]
                or outcome.candidate_evaluation != report["evaluation"]
                or details["source_report_sha256"] != body_sha
                or body_sha == trial["parent_report_sha256"]
                or trial["attempt_id"] != runner._session_id
                or report["quality"]["status"] != "passed"
                or report.get("partial")
            ):
                raise ValueError("native_trial_candidate_identity_invalid")
            snapshot = freeze_automatic_inputs(
                report,
                context[CTX_EVIDENCE],
                run_id=trial["parent_run_id"],
                revision=trial["revision"],
                budget=runner._cost_budget,
            )
    except (KeyError, TypeError, ValueError) as exc:
        failure = (
            str(exc)
            if isinstance(exc, ValueError)
            else "native_trial_candidate_contract_invalid"
        )
    acceptance = automatic_acceptance(snapshot, report["evaluation"])
    if failure:
        acceptance.update(status="failed", publishable=False)
        acceptance["failure_codes"].append(failure)
    metadata["automatic_acceptance"] = acceptance
    report["delivery"] = {
        "status": "ready" if acceptance["publishable"] else "blocked",
        "publishable": acceptance["publishable"],
        "reason_codes": acceptance["failure_codes"],
    }
    return acceptance


def reference_session(runner, report, historical_writer):
    """Import only the settled source references into a fresh writer journal."""
    metadata = report["metadata"]
    document = metadata["writer_document"]
    budget = runner._cost_budget
    if (
        historical_writer["draft_sha256"]
        != hashlib.sha256(report["survey"].encode()).hexdigest()
        or historical_writer["body"] != document["body"]
        or historical_writer["summaries"] != document["summaries"]
    ):
        raise ValueError("native_trial_source_reference_mismatch")
    session = WriterSession(
        runner._stage_llm("generation"),
        runner._targeted_repair_path(),
        runner._session_id,
        document["packet"],
        document["aliases"],
        trace_hook=runner._trace_hook,
        thinking_mode=document["generation_thinking_mode"],
        reasoning_effort=document["generation_reasoning_effort"],
        provider_api=document["generation_provider_api"],
        syntax_policy=document["generation_syntax_policy"],
        writing_plan=document["generation_writing_plan"],
    )
    documents = (document["body"], document["summaries"])
    records = {
        digest(doc): historical_writer["reference_inputs"][digest(doc)]
        for doc in documents
    }
    calls = {}
    for name in {
        origin["call"] for row in records.values() for origin in row["origins"]
    }:
        call = historical_writer["calls"][name]
        usage = {
            key: call["usage"][key] for key in ("prompt_tokens", "completion_tokens")
        }
        ledger_id = historical_writer["identity"] + ":" + name
        entry = budget.entries[ledger_id]
        if (
            ledger_id not in budget._historical_ids
            or call["status"] != "response_received"
            or call["stage"] not in {"synthesis", "summary"}
            or entry["stage"] not in {"body_initial", "body", "summary"}
            or call["usage"].get("usage_available") is not True
            or entry["usage_sha256"] != digest(usage)
            or entry["actual_tokens"] != sum(usage.values())
            or entry["reserved_tokens"] != call["reservation"]
        ):
            raise ValueError("native_trial_source_reference_unsettled")
        calls[name] = call
    session.journal.update(
        calls=copy.deepcopy(calls), reference_inputs=copy.deepcopy(records)
    )
    for doc in documents:
        session.reference_input(doc)
    return session


async def repair_once(
    runner, report, results, decision, *, lock: Path, historical_writer
):
    """Invoke the production repair and full evaluator under an exclusive trial lock."""
    budget = runner._cost_budget
    if not isinstance(budget, RepairTrialBudget) or any(
        runner._stage_llm(stage)._cost_budget is not budget
        for stage in ("generation", "evaluation")
    ):
        raise ValueError("native_trial_budget_required")
    lock = Path(lock).resolve()
    repair_path = runner._targeted_repair_path().resolve()
    journal_path = repair_path.with_name(repair_path.name + ".writer-v2.json")
    delivery_path = lock.with_suffix(".delivery.json")
    paths = (
        lock,
        repair_path,
        journal_path,
        delivery_path,
        budget.path,
        runner._evaluation_progress_root,
    )
    if any(path is None for path in paths):
        raise ValueError("native_trial_storage_not_isolated")
    resolved = {Path(path).resolve() for path in paths}
    if len(resolved) != len(paths) or any(
        lock.parent not in path.parents for path in resolved
    ):
        raise ValueError("native_trial_storage_not_isolated")
    metadata = report["metadata"]
    state = metadata["targeted_repair_state"]
    trial = metadata["native_repair_trial"]
    if (
        state["run_id"] != runner._session_id
        or trial["attempt_id"] != runner._session_id
        or trial["parent_run_id"] == runner._session_id
        or trial["parent_report_sha256"]
        != hashlib.sha256(report["survey"].encode()).hexdigest()
        or state["attempts"]
    ):
        raise ValueError("native_trial_requires_new_identity")
    if any(path.exists() for path in (lock, repair_path, journal_path, delivery_path)):
        raise ValueError("native_trial_no_replay")
    query = metadata["query"]
    context = runner._build_eval_context(
        query=query, survey=report["survey"], results=results
    )
    validate_source_admission(metadata, context[CTX_EVIDENCE], trial["parent_run_id"])
    if (
        recovery_decision(report, context, budget) != decision
        or decision["action"] != "combined"
    ):
        raise ValueError("native_trial_repair_decision_changed")
    session = reference_session(runner, report, historical_writer)
    with lock.open("x", encoding="utf-8") as file:
        json.dump(
            {"session_id": runner._session_id, "decision": decision},
            file,
            ensure_ascii=False,
        )
    session.save()
    outcome = await repair_report(
        runner,
        query=query,
        report_data=report,
        results=results,
        initial_evaluation=report["evaluation"],
        initial_quality=report["quality"],
        report_language=metadata["report_language"],
        recovery=decision,
    )
    acceptance = record_trial_delivery(runner, report, results, outcome)
    atomic_json(delivery_path, acceptance)
    return outcome
