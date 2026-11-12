"""Replay the locally retained 074 report with the current recovery consumers."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.eval.engine_policy import configured_engine, phase_batches
from litagent.eval.ragas_adapter import planned_batches
from litagent.eval.ragas_native import global_review_request, source_samples
from litagent.eval.ragas_repair import _settled_evaluation, recovery_decision
from litagent.eval.results import CONTRACT as RESULT_CONTRACT
from litagent.evidence import collect_ledger
from litagent.frozen_writing import plan_frozen_budget
from litagent.output_capacity import configured_output_tokens
from litagent.quality import digest
from litagent.run_limits import configured_run_limits
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget
from litagent.writing_v2 import (
    current_repair_guidance_context,
    prepare_planned_input,
    provider_response_format,
    repair_request_input,
    repair_targets,
    request_messages,
)

ARCHIVE = Path("artifacts/acceptance/ragas-native-single-20261109-074")


@pytest.mark.integration
def test_local_074_all_assertions_can_use_last_shared_round():
    """Keep 074 immutable while replaying its bindings and real settled usage."""
    paths = [
        ARCHIVE / name
        for name in ("report.json", "final-budget.json", "execution.json")
    ]
    if not all(path.is_file() for path in paths):
        pytest.skip("Requires the local immutable 074 run")
    contents = [path.read_bytes() for path in paths]
    report, snapshot, execution = map(json.loads, contents)
    old_plan = snapshot["plan"]
    validate_accounting_snapshot(snapshot, expected_plan=old_plan)
    meta = report["metadata"]
    details = report["evaluation"]["faithfulness"]["details"]
    assert report["evaluation"]["faithfulness"]["score"] is None
    assert len(details["assertions"]) == 410
    assert len(details["unsupported_claims"]) == 5
    _settled_evaluation(details["stages"], snapshot["entries"], None)
    negative = {row["claim_id"]: row for row in details["unsupported_claims"]}
    assert {row["source_unit_id"] for row in details["global_issues"]} == {
        negative[key]["source_unit_id"]
        for key in ("claim_380", "claim_381", "claim_383")
    }
    # Adapt historical keys in memory to the new round namespace. The real
    # payloads, source binding and paid usage remain the fixture inputs.
    evaluation_identity = digest(["074-offline", details["source_report_sha256"]])
    stages = details["stages"]
    stages["evaluation_identity"] = evaluation_identity
    for row in stages["requests"]:
        old = row["request_id"]
        row["request_id"] = evaluation_identity + ":" + old
        snapshot["entries"][row["request_id"]] = snapshot["entries"].pop(old)
    old = stages["global_request_id"]
    stages["global_request_id"] = evaluation_identity + ":" + old
    snapshot["entries"][stages["global_request_id"]] = snapshot["entries"].pop(old)
    details["evidence_snapshot_sha256"] = meta["evidence_snapshot_sha256"]
    cfg = load_config("config/operational-plus20.yaml")
    plan = plan_frozen_budget(
        execution["bundle"],
        budget_policy="planned-budget-v7",
        capacity=cfg.capacity,
        output_tokens=configured_output_tokens(cfg),
        preparation_budget_snapshot=execution["preparation_snapshot"],
        semantic_engine=configured_engine(cfg),
        run_limits=configured_run_limits(cfg),
        **profile_flags(PROFILE),
        evaluation_result_contract=RESULT_CONTRACT,
    )
    envelope = plan["envelope"]
    assert plan["planned_tokens"] == old_plan["planned_tokens"]
    assert plan["planned_requests"] == old_plan["planned_requests"]
    snapshot["plan"] = plan
    owner = PlannedBudget(plan)
    owner._restore_planned(snapshot)
    context = {
        "query": meta["query"],
        "review_methodology": meta["review_methodology"],
        "writing_bindings": meta["writing_bindings"],
        "fact_cards": meta["fact_cards"],
        "evidence_snapshot_sha256": meta["evidence_snapshot_sha256"],
        "evidence": collect_ledger(execution["bundle"]["extractions"]),
        "report_language": meta["report_language"],
        "coverage_requirements": meta["coverage_requirements"],
    }
    choice = recovery_decision(report, context, owner)
    assert choice["action"] == "combined", choice
    assert choice["remaining_attempts"] == 1
    assert len(choice["targets"]) == 2
    assert choice["budget"]["status"] == "admitted"
    assert (
        choice["source_report_sha256"]
        == hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    doc = {**meta["writer_document"], "budget_envelope": envelope}
    bindings = {
        **meta["writing_bindings"],
        "writer_envelope_identity": envelope["identity"],
    }
    diagnostics = details["unsupported_claims"]
    targets = repair_targets(
        meta["writing_bindings"], diagnostics, doc["body"], doc["summaries"], merge=True
    )
    assert list(targets) == choice["targets"]
    guidance = current_repair_guidance_context(
        report["survey"],
        bindings,
        doc,
        targets,
        diagnostics,
        diagnostic_categories=["evaluation"] * len(diagnostics),
    )
    system, payload, schema = repair_request_input(
        doc["packet"],
        doc["body"],
        targets,
        doc["aliases"],
        cause="evaluation",
        previous_failure=None,
        writing_plan=doc["generation_writing_plan"],
        envelope=envelope,
        wire_contract=doc["writer_wire_contract"],
        guidance_context={"current": guidance, "prior_candidate": None},
    )
    system, payload = prepare_planned_input(
        system,
        payload,
        doc["packet"],
        doc["aliases"],
        doc["generation_writing_plan"],
        envelope,
    )
    fmt = provider_response_format(
        "repair", schema, wire_contract=doc["writer_wire_contract"]
    )
    messages = request_messages(
        system, payload, schema, wire_contract=doc["writer_wire_contract"]
    )
    spec = plan["stages"]["repair"]
    amount = strict_request_reservation(messages, spec["output_tokens"], fmt)
    owner.check_stage_requests("repair", [amount])
    assert amount <= choice["budget"]["stages"]["repair"]["tokens"]
    # Rebuild the actual 410-claim request shapes under the latest roster and
    # compare both complete stage groups to the still-locked last-round bounds.
    cfg = cfg.eval
    _, units, _, samples = source_samples(report["survey"], context)
    by_fragment = {sample["source_fragment_id"]: sample for sample in samples}
    for sample in samples:
        sample["statements"] = []
    for claim in details["assertions"]:
        by_fragment[claim["source_fragment_id"]]["statements"].append(
            claim["claim_text"]
        )
    for phase in ("inventory", "assessment"):
        batches = planned_batches(
            phase,
            samples,
            cfg,
            envelope["semantic_engine"]["batch_size"],
            phase_batches(envelope),
        )
        amounts = [
            strict_request_reservation(messages, cfg.max_tokens, fmt)
            for _, messages, fmt, _ in batches
        ]
        owner.check_stage_requests(phase, amounts)
        assert sum(amounts) <= choice["budget"]["stages"][phase]["tokens"]
    _, messages, fmt = global_review_request(
        report["survey"], context, cfg, units, details["assertions"], force_compact=True
    )
    owner.check_stage_requests(
        "global", [strict_request_reservation(messages, cfg.max_tokens, fmt)]
    )
    interrupted = copy.deepcopy(report)
    interrupted["evaluation"]["faithfulness"]["details"][
        "execution_error"
    ] = "ragas_usage_unknown"
    stop = recovery_decision(interrupted, context, owner)
    assert stop["action"] == "stop" and stop["reason_code"] == "ragas_usage_unknown"
    wrong_source = copy.deepcopy(report)
    wrong_source["evaluation"]["faithfulness"]["details"]["assertions"][0][
        "source_start"
    ] += 1
    stop = recovery_decision(wrong_source, context, owner)
    assert (
        stop["action"] == "stop"
        and stop["reason_code"] == "ragas_recovery_source_invalid"
    )
    owner.closed_stages.add("global")
    stop = recovery_decision(report, context, owner)
    assert stop["action"] == "stop"
    assert stop["reason_code"] == "ragas_recovery_followup_budget_insufficient"
    assert stop["budget"]["status"] == "rejected"
    owner.closed_stages.clear()
    assert owner.snapshot() == snapshot
    assert [path.read_bytes() for path in paths] == contents
