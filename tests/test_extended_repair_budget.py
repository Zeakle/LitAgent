"""Check independent repair escrows, durable balances and archive compatibility."""

import json
from pathlib import Path

import pytest

from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.eval.engine_policy import configured_engine
from litagent.eval.results import CONTRACT as RESULT_CONTRACT
from litagent.frozen_writing import plan_frozen_budget
from litagent.output_capacity import configured_output_tokens
from litagent.quality import (
    begin_targeted_repair,
    digest,
    load_targeted_repairs,
    remaining_repair_allowances,
    save_targeted_repairs,
    targeted_repair_state,
)
from litagent.run_limits import configured_run_limits, extended_repair_stages
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget


def test_three_budgets_are_independent_and_survive_restart(tmp_path):
    state = targeted_repair_state(None, "offline", limit=11)
    path = tmp_path / "repair.json"
    for kind, count in (("capacity", 3), ("inventory", 3), ("prose", 5)):
        stage = "native_initial" if kind == "capacity" else "evaluation"
        for index in range(count):
            entry = begin_targeted_repair(
                state, stage, digest([kind, index]), category=kind
            )
            if kind == "inventory":
                entry["purpose"] = "ragas-recovery-v1"
            save_targeted_repairs(path, state)
            with pytest.raises(ValueError, match="restart_no_replay"):
                load_targeted_repairs(path, "offline")
            entry["status"] = "failed" if index == 0 else "completed"
            save_targeted_repairs(path, state)
            assert load_targeted_repairs(path, "offline") == state
        assert remaining_repair_allowances(state)[kind] == 0
        with pytest.raises(
            ValueError,
            match="strict_targeted_repair_.*exhausted|strict_targeted_repair_limit_exceeded",
        ):
            begin_targeted_repair(state, stage, digest(kind), category=kind)
    assert remaining_repair_allowances(state) == dict(capacity=0, inventory=0, prose=0)


@pytest.mark.integration
def test_074_extended_worst_path_is_admitted_without_upgrading_its_archive():
    path = Path(
        "artifacts/acceptance/ragas-native-single-20261109-074/final-budget.json"
    )
    if not path.is_file():
        pytest.skip("Requires the archived 074 budget")
    raw = path.read_bytes()
    historical = json.loads(raw)
    previous = historical["plan"]
    validate_accounting_snapshot(historical, expected_plan=previous)
    execution = json.loads(path.with_name("execution.json").read_bytes())
    cfg = load_config("config/extended-repair.yaml")
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
    counts = extended_repair_stages(plan["envelope"], 11)
    assert counts == {
        "summary": 9,
        "reviewer": 1,
        "repair": 8,
        "inventory": 45,
        "assessment": 45,
        "global": 9,
    }
    assert plan["status"] == "admitted"
    assert plan["planned_requests"] == 130
    assert plan["planned_tokens"] <= 32_000_000
    owner = PlannedBudget(plan)
    validate_accounting_snapshot(owner.snapshot(), expected_plan=plan)
    assert path.read_bytes() == raw
