"""Validate native RAGAS stage escrows, accounting and restart behavior."""

import copy

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.eval.engine_policy import engine_descriptor, native_writer_budget_plan
from litagent.run_limits import configured_run_limits
from litagent.safety.budget import CostBudget
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget, request_scope


@pytest.fixture
def native_plan(case):
    """Build the actual server preset rather than a handwritten budget."""
    config = load_config("config/expanded-capacity.yaml")
    plan = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        semantic_engine=engine_descriptor(),
        capacity=config.capacity,
        methodology={},
        run_limits=configured_run_limits(config),
    )
    assert plan["status"] == "admitted", plan
    return plan


def test_complete_stage_maxima_preserve_two_full_reevaluations(native_plan):
    """Spend every protected slot at its ceiling, including both repair branches."""
    owner = PlannedBudget(native_plan)
    for stage, spec in native_plan["stages"].items():
        for index in range(spec["count"]):
            with request_scope(stage, f"{stage}:{index}"):
                owner.reserve(spec["request_cap"])
                owner.bind_wire({"stage": stage, "index": index})
                owner.begin_request()
                owner.record(
                    {
                        "prompt_tokens": spec["request_cap"] - spec["output_tokens"],
                        "completion_tokens": spec["output_tokens"],
                    }
                )
    snapshot = owner.snapshot()
    validate_accounting_snapshot(snapshot, expected_plan=native_plan)
    assert snapshot["usage_uncertain"] is False
    assert snapshot["stop_reason"] is None
    assert snapshot["used_tokens"] == native_plan["required_tokens"]
    assert snapshot["requests_sent"] <= 64
    assert native_plan["repair_limit"] == 2
    restored = CostBudget.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    first = next(iter(native_plan["stages"]))
    with request_scope(first, f"{first}:0"), pytest.raises(ValueError):
        restored.reserve(1)


def test_unknown_native_request_is_not_replayed_after_restart(native_plan):
    """Unknown in-flight usage is held instead of being refunded on restart."""
    owner = PlannedBudget(native_plan)
    stage = "assessment"
    with request_scope(stage, "unknown"):
        owner.reserve(native_plan["stages"][stage]["request_cap"])
        owner.bind_wire({"fixture": "unknown"})
        owner.begin_request()
    restored = CostBudget.from_snapshot(owner.snapshot())
    assert restored.snapshot()["usage_uncertain"] is True
    assert (
        restored.snapshot()["entries"]["unknown"]["status"] == "interrupted_no_replay"
    )
    with request_scope(stage, "unknown"), pytest.raises(ValueError):
        restored.reserve(1)


def test_native_accounting_rejects_unfunded_stage_and_bad_identity(native_plan):
    """The new transport cannot borrow legacy round slots or change engine IDs."""
    owner = PlannedBudget(native_plan)
    snapshot = owner.snapshot()
    forged = copy.deepcopy(native_plan)
    forged["envelope"]["semantic_engine"]["identity"] = "0" * 64
    with pytest.raises(ValueError):
        PlannedBudget(forged)
    bad = copy.deepcopy(snapshot)
    bad["capacity_allocation"] = {"fake": True}
    with pytest.raises(ValueError):
        validate_accounting_snapshot(bad, expected_plan=native_plan)
