"""Prove locked v5 capacity independently of actual model usage averages."""

import copy

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import (
    PlannedBudget,
    build_budget_plan,
    request_scope,
)
from litagent.writing_v2 import writer_budget_plan


def test_maximum_legal_consumption_including_two_reevaluations(case, tmp_path):
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )
    budget = PlannedBudget(plan, path=tmp_path / "ledger.json")
    for stage, spec in plan["stages"].items():
        for index in range(spec["count"]):
            amount = spec["request_cap"]
            with request_scope(stage, f"{stage}:{index}"):
                assert budget.remaining() >= amount
                budget.reserve(amount)
                budget.begin_request()
                budget.record(
                    {
                        "prompt_tokens": amount - spec["output_tokens"],
                        "completion_tokens": spec["output_tokens"],
                    }
                )
    snapshot = budget.snapshot()
    assert snapshot["used_tokens"] == plan["required_tokens"]
    assert snapshot["requests_sent"] == 41
    assert snapshot["used_tokens"] <= plan["planned_tokens"] <= 4500000
    assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot


@pytest.mark.parametrize("fault", ["tokens", "context", "requests"])
def test_v5_rejects_outside_complete_plan_before_transport(case, fault):
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )
    stages = {
        k: {f: v[f] for f in ("count", "request_cap", "output_tokens", "bound")}
        for k, v in plan["stages"].items()
    }
    scope = copy.deepcopy(plan["envelope"])
    kwargs = dict(
        identity=plan["input_identity"],
        contract=plan["contract"],
        envelope=scope,
        repairs=2,
    )
    if fault == "tokens":
        kwargs["spent_tokens"] = 4500000
    elif fault == "requests":
        kwargs["spent_requests"] = 64
    else:
        stages["body_initial"]["request_cap"] = 131072
    result = build_budget_plan(stages, **kwargs)
    assert result["status"] == "preflight_rejected"


def test_v5_rejects_zero_repairs_and_legacy_limit_is_unchanged(case):
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )
    stages = {
        k: {f: v[f] for f in ("count", "request_cap", "output_tokens", "bound")}
        for k, v in plan["stages"].items()
    }
    with pytest.raises(ValueError):
        build_budget_plan(
            stages,
            identity=plan["input_identity"],
            contract=plan["contract"],
            envelope=plan["envelope"],
            repairs=0,
        )
    old = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v4"
    )
    assert old["maximum_tokens"] == 1000000


def test_flat_campaign_cannot_start_without_source_bound_scope_proof(tmp_path):
    import json

    from scripts.run_frozen_writing import FLAT_PROFILE, offline_gates_verified

    manifest = {
        "writing_profile": FLAT_PROFILE,
        "identity": "manifest",
        "source_hashes": {"writer.py": "frozen"},
        "budget_plans": {"case": {"identity": "budget"}},
    }
    assert not offline_gates_verified(tmp_path, manifest)
    gates = {
        "contract": "flat-writer-offline-gates-v1",
        "manifest_identity": "manifest",
        "source_hashes": manifest["source_hashes"],
        "pytest_exit_code": 0,
        "future_scope_verified": True,
        "cases": {"case": {"budget_identity": "budget", "scope_verified": True}},
    }
    path = tmp_path / "offline-gates.json"
    path.write_text(json.dumps(gates))
    assert offline_gates_verified(tmp_path, manifest)
    gates["future_scope_verified"] = False
    path.write_text(json.dumps(gates))
    assert not offline_gates_verified(tmp_path, manifest)
    gates["future_scope_verified"] = True
    gates["source_hashes"] = {"writer.py": "changed"}
    path.write_text(json.dumps(gates))
    assert not offline_gates_verified(tmp_path, manifest)
