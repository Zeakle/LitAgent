"""Read maximum legal expanded ledgers using their frozen context allowance."""

import copy

import pytest
import test_joint_round_escrow as helpers
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.eval.results import CONTRACT
from litagent.frozen_writing import plan_frozen_budget
from litagent.output_capacity import configured_output_tokens
from litagent.safety.budget import CostBudget
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget, request_scope


def wire(plan, stage, amount, marker):
    """Use the real stage output reservation instead of a legacy fixed value."""
    output = plan["stages"][stage]["output_tokens"]
    messages = [{"role": "user", "content": marker}]
    overhead = helpers.strict_request_reservation(messages, output, None)
    assert amount >= overhead
    messages[0]["content"] += "x" * (amount - overhead)
    proof = helpers.request_proof(stage, messages, output, None)
    assert proof["reserved_tokens"] == amount
    return stage, messages, None, proof


def send(owner, request, label):
    """Settle one synthetic accounting wire with real guard and identity checks."""
    stage, messages, native, proof = request
    key = helpers.round_request_identity(owner, stage, label)
    output = owner.plan["stages"][stage]["output_tokens"]
    with request_scope(stage, key):
        assert owner.check_wire(messages, output, native) == proof["reserved_tokens"]
        owner.reserve(proof["reserved_tokens"])
        owner.begin_request()
        usage = {
            "prompt_tokens": proof["reserved_tokens"] - output,
            "completion_tokens": output,
        }
        owner.record(usage)
        before = owner.snapshot()
        owner.record(usage)
        assert owner.snapshot() == before
    return key


@pytest.fixture
def maximum_ledger(frozen_bundle, tmp_path):
    """Exhaust three complete rounds and two repairs without external calls."""
    cfg = load_config("config/expanded-capacity.yaml")
    plan = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        capacity=cfg.capacity,
        output_tokens=configured_output_tokens(cfg),
        evaluation_result_contract=CONTRACT,
        **profile_flags(PROFILE),
    )
    assert plan["status"] == "admitted"
    assert plan["context_reservation_limit"] == 176947
    assert plan["stages"]["global"]["request_cap"] == 120000
    assert plan["stages"]["repair"]["request_cap"] == 122540
    owner = PlannedBudget(plan, path=tmp_path / "ledger.json")
    domain = helpers._domain(plan)
    amounts = helpers._amounts(domain["reserved_tokens"], domain["max_requests"])
    helpers._settle_initial_max(owner)
    for index in range(3):
        requests = [
            wire(plan, "assessment", amount, f"round:{index}:batch:{slot}")
            for slot, amount in enumerate(amounts)
        ]
        requests.append(
            wire(
                plan,
                "global",
                plan["stages"]["global"]["request_cap"],
                f"global:{index}",
            )
        )
        helpers._allocate(owner, requests, report_sha=str(index + 1) * 64)
        for slot, request in enumerate(requests):
            send(owner, request, f"slot:{index}:{slot}")
        snapshot = owner.snapshot()
        assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
        assert validate_accounting_snapshot(snapshot, expected_plan=plan) == snapshot
        if index < 2:
            for stage in ("repair", "summary", "inventory"):
                helpers._settle_cap(owner, stage, f"candidate:{index}:{stage}")
    snapshot = owner.snapshot()
    assert snapshot["used_tokens"] == plan["required_tokens"]
    assert snapshot["requests_sent"] == plan["planned_requests"] <= 64
    return plan, snapshot


def test_maximum_expanded_global_and_repairs_roundtrip_without_losing_scope(
    maximum_ledger,
):
    plan, snapshot = maximum_ledger
    assert plan["repair_limit"] == 2
    assert sum(e["stage"] == "global" for e in snapshot["entries"].values()) == 3
    repairs = [e for e in snapshot["entries"].values() if e["stage"] == "repair"]
    assert len(repairs) == 2 and all(e["actual_tokens"] == 122540 for e in repairs)
    assert validate_accounting_snapshot(snapshot, expected_plan=plan) == snapshot


@pytest.mark.parametrize("fault", ["stage_cap", "context_cap", "forged_plan_context"])
def test_expanded_context_does_not_bypass_frozen_stage_or_plan_limits(
    maximum_ledger, fault
):
    original_plan, original_snapshot = maximum_ledger
    plan, snapshot = copy.deepcopy(original_plan), copy.deepcopy(original_snapshot)
    if fault == "forged_plan_context":
        plan["context_reservation_limit"] += 1
        snapshot["plan"] = copy.deepcopy(plan)
    else:
        entry = next(e for e in snapshot["entries"].values() if e["stage"] == "global")
        entry["reserved_tokens"] = (
            plan["stages"]["global"]["request_cap"] + 1
            if fault == "stage_cap"
            else plan["context_reservation_limit"] + 1
        )
    with pytest.raises(ValueError, match="planned_accounting_"):
        validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert original_snapshot == maximum_ledger[1]


def test_legacy_frozen_context_does_not_inherit_expanded_limits(case):
    _, plan = helpers._joint_plan(case)
    assert plan["context_reservation_limit"] == 117964
    owner = PlannedBudget(plan)
    helpers._settle_initial_max(owner)
    amounts = helpers._amounts(helpers._domain(plan)["reserved_tokens"], 15)
    requests = helpers._round_wires(owner, amounts, "legacy")
    helpers._allocate(owner, requests)
    for slot, request in enumerate(requests):
        helpers._send_max(owner, request, f"legacy-slot:{slot}")
    snapshot = owner.snapshot()
    assert validate_accounting_snapshot(snapshot, expected_plan=plan) == snapshot
    broken = copy.deepcopy(snapshot)
    entry = next(e for e in broken["entries"].values() if e["stage"] == "global")
    entry["reserved_tokens"] = 120000
    with pytest.raises(ValueError, match="planned_accounting_entry_invalid"):
        validate_accounting_snapshot(broken, expected_plan=plan)
