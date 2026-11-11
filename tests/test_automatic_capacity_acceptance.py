"""Exercise funded adaptive inventories at the real automatic delivery consumer."""

import copy
import hashlib

import pytest
from model_config_helpers import load_previous_default_config as load_config
from test_adaptive_capacity import plan_for, settle, settle_initial
from test_optional_verification import optional_report
from test_round_capacity import make_owner, send, wires
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_evaluation import adaptive_eval_config, preflight_round
from litagent.capacity_rounds import choose_refinement, plan_round
from litagent.config import EvalConfig
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.eval.base import CTX_EVIDENCE
from litagent.eval.batched import (
    batch_prompt,
    build_global_payload,
    global_prompt,
    plan_batches,
)
from litagent.eval.scoped import build_evaluation_request, build_source_fragments
from litagent.quality import build_fact_cards, digest, strict_acceptance
from litagent.report_render import build_report_source_units
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget, request_scope


def complete_report(count, *, repeat_quote=False, long_prefix=0):
    """Create distinct original-bound claims within the existing 66-unit capacity."""
    report, ledger = optional_report()
    sentences = [
        f"Method M{index:03d} "
        + (
            "under a separately stated experimental condition " * 5
            if index < long_prefix
            else ""
        )
        + "learns a comparison function. [E:a]"
        for index in range(count)
    ]
    if repeat_quote:
        sentences[1] = sentences[0]
    survey = "\n\n".join(
        " ".join(sentences[index : index + 3]) for index in range(0, count, 3)
    )
    ledger["a"]["supporting_text"] = " ".join(sentences)
    report["survey"] = survey
    cards = build_fact_cards(ledger)
    report["metadata"].update(
        capacity_policy="adaptive-capacity-v1",
        budget_policy="planned-budget-v7",
        generation_strategy="body_summary_v3",
        fact_cards=cards,
        evidence_snapshot_sha256=digest(cards),
    )
    assertions = []
    units = build_report_source_units(survey)
    assert len(units) <= 66
    for unit in units:
        for fragment in build_source_fragments(unit):
            assertions.append(
                {
                    "claim_id": f"c{len(assertions) + 1}",
                    "claim_text": fragment["source_quote"].removesuffix(" [E:a]"),
                    "kind": "scientific",
                    "status": "supported",
                    "local_error": None,
                    "source_unit_id": unit.unit_id,
                    "source_fragment_id": fragment["source_fragment_id"],
                    "source_quote": fragment["source_quote"],
                    "source_start": unit.start + fragment["start"],
                    "source_end": unit.start + fragment["end"],
                    "evidence_ids": ["a"],
                }
            )
    assert len(assertions) == count
    details = report["evaluation"]["faithfulness"]["details"]
    details.update(
        assertions=assertions,
        source_report_sha256=hashlib.sha256(survey.encode()).hexdigest(),
    )
    batches = [
        {
            "status": "completed",
            "claim_ids": [
                claim["claim_id"] for claim in assertions[index : index + 16]
            ],
        }
        for index in range(0, count, 16)
    ]
    details["stages"].update(
        processed_assertions=count,
        expected_assertions=count,
        expected_batches=len(batches),
        batches=batches,
    )
    details["core_coverage"][0]["claim_ids"] = [
        claim["claim_id"] for claim in assertions
    ]
    assert strict_acceptance(details)["status"] == "passed"
    return report, ledger


def fund_round(owner, *, count, report_sha, prefix):
    """Allocate, check and settle every actual assessment/global slot without transport."""
    current = wires((count + 15) // 16, text=prefix)
    allocation = plan_round(
        owner,
        assertions=count,
        batches=len(current) - 1,
        requests=[value[2] for value in current],
        report_sha256=report_sha,
    )
    for index, (stage, messages, proof) in enumerate(current):
        send(owner, stage, messages, proof["reserved_tokens"], f"{prefix}:{index}")
    validate_accounting_snapshot(owner.snapshot(), expected_plan=owner.plan)
    return allocation


def funded_report(case, count):
    """Bind a complete evaluated report to a real settled v7 allocation."""
    report, ledger = complete_report(count)
    owner = make_owner(case)
    settle_initial(owner)
    details = report["evaluation"]["faithfulness"]["details"]
    allocation = fund_round(
        owner,
        count=count,
        report_sha=details["source_report_sha256"],
        prefix="original",
    )
    # Evaluation records the inventory envelope before the funded expansion.
    details["stages"]["writer_envelope"] = copy.deepcopy(owner.plan["envelope"])
    details["stages"]["capacity"] = {
        "policy": "adaptive-capacity-v1",
        "budget_policy": "planned-budget-v7",
        "status": "allocated",
        "reason": None,
        "source_report_sha256": details["source_report_sha256"],
        "initial_assertions": owner.plan["envelope"]["model_assertions"],
        "effective_assertions": allocation["envelope"]["model_assertions"],
        "actual_assertions": count,
        "actual_batches": (count + 15) // 16,
        "allocation_identity": allocation["identity"],
        "round": copy.deepcopy(allocation["round"]),
    }
    return report, ledger, owner


@pytest.mark.parametrize("count", [129, 144, 176])
def test_funded_complete_adaptive_inventory_reaches_automatic_acceptance(case, count):
    report, ledger, owner = funded_report(case, count)
    snapshot = freeze_automatic_inputs(
        report, ledger, run_id="capacity-auto", budget=owner
    )
    assert len(snapshot["frozen_inventory"]["assertions"]) == count
    assert len(snapshot["frozen_inventory"]["inventory"]["claims"]) == count
    accepted = automatic_acceptance(snapshot, report["evaluation"])
    assert accepted["publishable"], accepted
    assert accepted["gates"] == {
        "evaluation": "passed",
        "judge": "disabled",
        "extraction_jev": "disabled",
    }


@pytest.mark.asyncio
async def test_runner_forwards_actual_budget_to_automatic_freeze(
    case, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    report, ledger, owner = funded_report(case, 144)
    agent = LitAgent(load_config())
    agent._session_id = "capacity-auto"
    agent._cost_budget = owner
    agent._build_eval_context = lambda **kwargs: {CTX_EVIDENCE: ledger}
    await agent._automatic_review(report, "comparison methods", {}, None)
    assert report["delivery"]["publishable"], report["delivery"]


def test_retained_report_can_use_its_prior_completed_round(case):
    report, ledger, owner = funded_report(case, 144)
    selected = copy.deepcopy(owner.allocation)
    for stage in ("repair", "summary", "inventory"):
        settle(
            owner,
            stage,
            f"candidate:{stage}",
            owner.plan["stages"][stage]["request_cap"],
        )
    later = fund_round(owner, count=144, report_sha="f" * 64, prefix="failed-candidate")
    assert (
        later["round"]["report_sha256"]
        != hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    before = owner.snapshot()
    snapshot = freeze_automatic_inputs(
        report, ledger, run_id="retained-original", budget=owner
    )
    assert len(snapshot["frozen_inventory"]["assertions"]) == 144
    receipt = snapshot["frozen_inventory"]["capacity_authorization"]
    assert receipt["allocation_identity"] == selected["identity"]
    assert receipt["round_number"] == selected["round"]["number"]
    assert receipt["source_report_sha256"] == selected["round"]["report_sha256"]
    assert receipt["budget_plan_identity"] == owner.plan["identity"]
    assert receipt["budget_snapshot_sha256"] == digest(before)
    assert owner.snapshot() == before


@pytest.mark.parametrize("count", [1, 128])
def test_legacy_inventory_limit_preserves_complete_old_inputs(count):
    report, ledger = complete_report(count)
    report["metadata"].pop("capacity_policy")
    snapshot = freeze_automatic_inputs(report, ledger, run_id="legacy-auto")
    assert len(snapshot["frozen_inventory"]["assertions"]) == count


def test_capacity_diagnostic_without_trusted_budget_does_not_authorize_growth(case):
    report, ledger, _ = funded_report(case, 129)
    with pytest.raises(ValueError, match="automatic_"):
        freeze_automatic_inputs(report, ledger, run_id="untrusted-capacity")


@pytest.mark.parametrize("fault", ["unknown", "inflight", "stopped", "allocation_hash"])
def test_unsafe_budget_cannot_authorize_automatic_growth(case, fault):
    report, ledger, owner = funded_report(case, 144)
    if fault in {"unknown", "inflight"}:
        with request_scope("repair", "unsafe-next"):
            owner.reserve(20000)
            owner.begin_request()
            if fault == "unknown":
                owner.charge_unknown_usage(20000)
    elif fault == "stopped":
        owner.stop_after_worker_failure()
    else:
        owner.revisions[0]["allocation"]["identity"] = "0" * 64
    with pytest.raises(ValueError):
        freeze_automatic_inputs(report, ledger, run_id="unsafe-capacity", budget=owner)


@pytest.mark.parametrize(
    "fault", ["allocation", "report", "count", "batches", "status", "policy"]
)
def test_unbound_capacity_diagnostic_cannot_authorize_growth(case, fault):
    report, ledger, owner = funded_report(case, 144)
    capacity = report["evaluation"]["faithfulness"]["details"]["stages"]["capacity"]
    if fault == "allocation":
        capacity["allocation_identity"] = "0" * 64
    elif fault == "report":
        capacity["source_report_sha256"] = "0" * 64
    elif fault == "count":
        capacity["actual_assertions"] -= 1
    elif fault == "batches":
        capacity["actual_batches"] -= 1
    elif fault == "status":
        capacity["status"] = "needs_refinement"
    else:
        report["metadata"]["capacity_policy"] = "unknown"
    with pytest.raises(ValueError):
        freeze_automatic_inputs(report, ledger, run_id="unbound-capacity", budget=owner)


@pytest.mark.parametrize(
    "fault", ["skipped", "unsupported", "batch", "global", "duplicate", "source"]
)
def test_funded_capacity_does_not_bypass_complete_scientific_validation(case, fault):
    report, ledger, owner = funded_report(case, 144)
    details = report["evaluation"]["faithfulness"]["details"]
    if fault == "skipped":
        report["evaluation"]["faithfulness"]["skipped"] = True
    elif fault == "unsupported":
        details["assertions"][0]["status"] = "unsupported"
    elif fault == "batch":
        details["stages"]["batches"][0]["status"] = "started"
    elif fault == "global":
        details["stages"]["global_check"] = "started"
    elif fault == "duplicate":
        details["assertions"][1]["claim_id"] = details["assertions"][0]["claim_id"]
    else:
        details["assertions"][0]["source_end"] -= 1
    with pytest.raises(ValueError):
        freeze_automatic_inputs(report, ledger, run_id="bad-science", budget=owner)


def production_round_report(
    case,
    *,
    batch_size=16,
    repeat_quote=False,
    long_prefix=0,
    omit_stage=None,
    wrapped=False,
):
    """Fund real batch/request builders and retain their exact settled wire receipts."""
    report, ledger = complete_report(
        144, repeat_quote=repeat_quote, long_prefix=long_prefix
    )
    owner = make_owner(case)
    budget = owner
    if wrapped:
        budget = CostBudget()
        budget.activate_plan(owner.plan)
        owner = budget._planned_backend
    settle_initial(owner)
    details = report["evaluation"]["faithfulness"]["details"]
    units = build_report_source_units(report["survey"])
    payload = {
        "evaluation_input_contract": "flat-evaluation-input-v1",
        "writer_envelope": copy.deepcopy(owner.effective_envelope),
        "source_units": [
            {
                "source_unit_id": unit.unit_id,
                "text": unit.text,
                "citation_aliases": ["e1"],
                "source_fragments": [
                    {
                        "source_fragment_id": fragment["source_fragment_id"],
                        "text": fragment["source_quote"],
                        "citation_aliases": ["e1"],
                    }
                    for fragment in build_source_fragments(unit)
                ],
            }
            for unit in units
        ],
        "report_structure": [
            {"source_unit_id": unit.unit_id, "section": "methods"} for unit in units
        ],
        "unit_inventory": [],
        "assertions": [
            {
                **{
                    key: claim[key]
                    for key in (
                        "claim_id",
                        "kind",
                        "claim_text",
                        "source_unit_id",
                        "source_fragment_id",
                    )
                },
                "evidence": [{"evidence_alias": "e1", "material_ref": "e1"}],
                "required_support_aspects": ["predicate"],
            }
            for claim in details["assertions"]
        ],
        "cited_materials": {"e1": {"original_support_ref": "s1"}},
        "original_supports": {
            "s1": {"paper_id": "evit", "text": ledger["a"]["supporting_text"]}
        },
        "support_passages": {
            "p1": ["e1", "s1", 0, len(ledger["a"]["supporting_text"])]
        },
        "coverage_requirements": details["coverage_requirements"],
        "host_runtime_record": {},
    }
    config = adaptive_eval_config(EvalConfig(), owner).model_copy(
        update={"strict_batch_size": batch_size}
    )
    batches = plan_batches(payload, config)
    stages = details["stages"]
    stages["batches"] = [
        {
            "status": "completed",
            "claim_ids": [claim["claim_id"] for claim in batch["assertions"]],
        }
        for batch in batches
    ]
    stages["expected_batches"] = len(batches)
    stages["writer_envelope"] = copy.deepcopy(owner.plan["envelope"])
    global_payload = build_global_payload(payload, stages)
    global_messages, global_schema = build_evaluation_request(
        global_prompt(global_payload), global_payload
    )
    from litagent.safety.planned_budget import wire_reservation

    preflight_round(
        owner,
        batches,
        payload,
        config,
        stages,
        wire_reservation(global_messages, config.max_tokens, global_schema),
        details["source_report_sha256"],
    )
    requests = [
        ("assessment", *build_evaluation_request(batch_prompt(batch), batch))
        for batch in batches
    ] + [("global", global_messages, global_schema)]
    omit_index = next(
        (
            index
            for index in reversed(range(len(requests)))
            if requests[index][0] == omit_stage
        ),
        None,
    )
    for index, (stage, messages, schema) in enumerate(requests):
        if index == omit_index:
            continue
        with request_scope(stage, f"production:{index}"):
            amount = owner.check_wire(messages, config.max_tokens, schema)
            owner.reserve(amount)
            owner.begin_request()
            owner.record(
                {
                    "prompt_tokens": amount - config.max_tokens,
                    "completion_tokens": config.max_tokens,
                }
            )
    validate_accounting_snapshot(owner.snapshot(), expected_plan=owner.plan)
    assert strict_acceptance(details)["status"] == "passed"
    return report, ledger, budget


def test_real_flat_balanced_batches_preserve_complete_inventory(case):
    report, ledger, owner = production_round_report(case, long_prefix=16)
    details = report["evaluation"]["faithfulness"]["details"]
    flattened = [
        cid for batch in details["stages"]["batches"] for cid in batch["claim_ids"]
    ]
    assert flattened != [claim["claim_id"] for claim in details["assertions"]]
    assert len(flattened) == len(set(flattened)) == 144
    snapshot = freeze_automatic_inputs(report, ledger, run_id="balanced", budget=owner)
    assert len(snapshot["frozen_inventory"]["assertions"]) == 144


def test_real_report_without_optional_policy_metadata_uses_host_authority(case):
    report, ledger, owner = production_round_report(case)
    for key in ("capacity_policy", "budget_policy", "generation_strategy"):
        report["metadata"].pop(key)
    before = owner.snapshot()
    snapshot = freeze_automatic_inputs(
        report, ledger, run_id="host-policy", budget=owner
    )
    assert automatic_acceptance(snapshot, report["evaluation"])["publishable"]
    assert owner.snapshot() == before


def test_actual_batch_count_can_exceed_ceil_of_initial_batch_size(case):
    report, ledger, owner = production_round_report(case, batch_size=15)
    stages = report["evaluation"]["faithfulness"]["details"]["stages"]
    assert len(stages["batches"]) == 10 > (144 + 15) // 16
    assert stages["capacity"]["round"]["batches"] == 10
    snapshot = freeze_automatic_inputs(
        report, ledger, run_id="extra-batch", budget=owner
    )
    assert (
        snapshot["frozen_inventory"]["capacity_authorization"]["actual_batches"] == 10
    )
    assert len(snapshot["frozen_inventory"]["assertions"]) == 144


@pytest.mark.asyncio
async def test_actual_cost_budget_backend_is_accepted_without_mutation(
    case, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    report, ledger, budget = production_round_report(case, wrapped=True)
    assert type(budget) is CostBudget
    assert isinstance(budget._planned_backend, PlannedBudget)
    before = budget.snapshot()
    snapshot = freeze_automatic_inputs(report, ledger, run_id="wrapped", budget=budget)
    receipt = snapshot["frozen_inventory"]["capacity_authorization"]
    assert receipt["budget_snapshot_sha256"] == digest(before)
    assert (
        receipt["allocation_identity"] == budget._planned_backend.allocation["identity"]
    )
    assert budget.snapshot() == before
    agent = LitAgent(load_config())
    agent._session_id = "wrapped"
    agent._cost_budget = budget
    agent._build_eval_context = lambda **kwargs: {CTX_EVIDENCE: ledger}
    await agent._automatic_review(report, "comparison methods", {}, None)
    assert report["delivery"]["publishable"], report["delivery"]
    assert budget.snapshot() == before


@pytest.mark.parametrize("stage", ["assessment", "global"])
def test_consistent_budget_with_unconsumed_round_slot_still_blocks(case, stage):
    report, ledger, owner = production_round_report(case, omit_stage=stage)
    before = owner.snapshot()
    # This is a valid partial ledger, so rejection must come from final-round completeness.
    validate_accounting_snapshot(before, expected_plan=owner.plan)
    with pytest.raises(ValueError, match="automatic_capacity_binding_invalid"):
        freeze_automatic_inputs(report, ledger, run_id="missing-slot", budget=owner)
    assert owner.snapshot() == before


@pytest.mark.parametrize("policy", ["refine", "non_v7"])
def test_unverified_or_non_v7_allocation_does_not_authorize_growth(case, policy):
    report, ledger, _ = funded_report(case, 144)
    details = report["evaluation"]["faithfulness"]["details"]
    if policy == "non_v7":
        owner = PlannedBudget(plan_for(case))
        settle_initial(owner)
        owner.expand_capacity(144, 9)
    else:
        owner = make_owner(case)
        settle_initial(owner)
        allocation = choose_refinement(
            owner,
            assertions=144,
            batches=9,
            report_sha256=details["source_report_sha256"],
        )
        capacity = details["stages"]["capacity"]
        capacity["allocation_identity"] = allocation["identity"]
        capacity["round"] = copy.deepcopy(allocation["round"])
        capacity["effective_assertions"] = allocation["envelope"]["model_assertions"]
        assert allocation["round"]["mode"] == "refine"
    before = owner.snapshot()
    validate_accounting_snapshot(before, expected_plan=owner.plan)
    with pytest.raises(ValueError, match="automatic_"):
        freeze_automatic_inputs(report, ledger, run_id="wrong-policy", budget=owner)
    assert owner.snapshot() == before


@pytest.mark.parametrize(
    "field",
    [
        "round_offset",
        "initial",
        "effective",
        "expected_assertions",
        "processed_assertions",
        "expected_batches",
        "generation_strategy",
    ],
)
def test_capacity_binding_rejects_coerced_counts_and_unknown_generation(case, field):
    report, ledger, owner = funded_report(case, 144)
    stages = report["evaluation"]["faithfulness"]["details"]["stages"]
    capacity = stages["capacity"]
    if field == "round_offset":
        assert capacity["round"]["offsets"]["assessment"] == 0
        capacity["round"]["offsets"]["assessment"] = False
    elif field in {"initial", "effective"}:
        key = field + "_assertions"
        capacity[key] = float(capacity[key])
    elif field == "generation_strategy":
        report["metadata"][field] = "unknown"
    else:
        stages[field] = float(stages[field])
    before = owner.snapshot()
    with pytest.raises(ValueError, match="automatic_capacity_binding_invalid"):
        freeze_automatic_inputs(report, ledger, run_id="typed-capacity", budget=owner)
    assert owner.snapshot() == before


def test_equal_quotes_at_different_positions_remain_separate_claims(case):
    report, ledger, owner = production_round_report(case, repeat_quote=True)
    assertions = report["evaluation"]["faithfulness"]["details"]["assertions"]
    first, second = assertions[:2]
    assert first["source_quote"] == second["source_quote"]
    assert first["source_start"] != second["source_start"]
    assert first["source_fragment_id"] != second["source_fragment_id"]
    snapshot = freeze_automatic_inputs(
        report, ledger, run_id="equal-quotes", budget=owner
    )
    assert snapshot["frozen_inventory"]["assertions"] == assertions
    assert len(snapshot["frozen_inventory"]["inventory"]["claims"]) == 144


@pytest.mark.parametrize("field", ["source_start", "source_fragment_id"])
def test_equal_quote_does_not_bypass_original_position_fidelity(case, field):
    report, ledger, owner = production_round_report(case, repeat_quote=True)
    assertions = report["evaluation"]["faithfulness"]["details"]["assertions"]
    assertions[1][field] = assertions[0][field]
    with pytest.raises(ValueError, match="automatic_source_fidelity_invalid"):
        freeze_automatic_inputs(report, ledger, run_id="crossed-quotes", budget=owner)
