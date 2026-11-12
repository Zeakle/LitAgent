"""Prove all independent loop and format escrows with the production planner."""

import copy
import json
from pathlib import Path

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent import review_policy
from litagent.config import load_config
from litagent.eval.engine_policy import configured_engine, native_writer_budget_plan
from litagent.output_capacity import configured_output_tokens
from litagent.run_limits import configured_run_limits
from litagent.safety.budget import CostBudget
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget, request_scope


@pytest.fixture(scope="module")
def limits_config():
    """Use the exact opt-in server configuration."""
    return load_config("config/separate-review-evaluation.yaml")


@pytest.fixture
def plan(case, limits_config):
    """Measure the native producer, including its actual schemas and wrappers."""
    cfg = limits_config
    result = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        methodology={},
        semantic_engine=configured_engine(cfg),
        capacity=cfg.capacity,
        run_limits=configured_run_limits(cfg),
        output_tokens=configured_output_tokens(cfg),
    )
    assert result["status"] == "admitted", result
    return result


def test_all_independent_maxima_and_format_fallback_fit(plan):
    owner = PlannedBudget(plan)
    for stage, spec in plan["stages"].items():
        for index in range(spec["count"]):
            with request_scope(stage, f"{stage}:{index}"):
                owner.reserve(spec["request_cap"])
                owner.bind_wire({"stage": stage, "index": index})
                owner.begin_request()
                usage = {
                    "prompt_tokens": spec["request_cap"] - spec["output_tokens"],
                    "completion_tokens": spec["output_tokens"],
                }
                owner.record(usage)
                owner.record(usage)
    snapshot = owner.snapshot()
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert (
        snapshot["used_tokens"]
        == plan["required_tokens"]
        <= plan["planned_tokens"]
        <= 47420000
    )
    assert snapshot["requests_sent"] == plan["planned_requests"] <= 226
    assert snapshot["stop_reason"] is None and not snapshot["usage_uncertain"]
    assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
    assert plan["stages"]["reviewer"]["count"] == 4
    for category in ("review_revision", "evaluation_revision"):
        assert plan["stages"][category + "/repair"]["count"] == 3
        with request_scope(category + "/repair", "another"), pytest.raises(ValueError):
            owner.reserve(1)


def test_review_savings_cannot_buy_fourth_edit_or_consume_evaluation(plan):
    owner = PlannedBudget(plan)
    with review_policy.partition("review_revision"):
        for index in range(3):
            with request_scope("repair", f"review:{index}"):
                owner.reserve(10)
                owner.bind_wire({"review": index})
                owner.begin_request()
                owner.record({"prompt_tokens": 1, "completion_tokens": 1})
        with request_scope("repair", "fourth"), pytest.raises(ValueError):
            owner.reserve(1)
    with (
        review_policy.partition("evaluation_revision"),
        request_scope("repair", "independent"),
    ):
        assert (
            owner.remaining()
            == plan["stages"]["evaluation_revision/repair"]["request_cap"]
        )
        owner.reserve(10)
        owner.bind_wire({"evaluation": 1})
        owner.begin_request()
    restored = CostBudget.from_snapshot(owner.snapshot())
    assert (
        restored.snapshot()["entries"]["independent"]["status"]
        == "interrupted_no_replay"
    )
    with (
        request_scope("evaluation_revision/repair", "independent"),
        pytest.raises(ValueError),
    ):
        restored.reserve(10)


def test_075_inputs_read_only_replan_under_independent_cap(limits_config):
    from litagent.closeout import PROFILE, profile_flags
    from litagent.eval.results import CONTRACT as RESULT_CONTRACT
    from litagent.frozen_writing import plan_frozen_budget

    p = Path("artifacts/acceptance/ragas-native-single-20261109-075/execution.json")
    if not p.exists():
        pytest.skip("Private historical input is not in the distribution")
    original = p.read_bytes()
    data = json.loads(original)
    cfg = limits_config
    plan = plan_frozen_budget(
        data["bundle"],
        budget_policy="planned-budget-v7",
        capacity=cfg.capacity,
        output_tokens=configured_output_tokens(cfg),
        preparation_budget_snapshot=data["preparation_snapshot"],
        semantic_engine=configured_engine(cfg),
        run_limits=configured_run_limits(cfg),
        **profile_flags(PROFILE),
        evaluation_result_contract=RESULT_CONTRACT,
    )
    assert plan["status"] == "admitted"
    assert plan["planned_tokens"] <= 47420000 and plan["planned_requests"] <= 226
    validate_accounting_snapshot(PlannedBudget(plan).snapshot(), expected_plan=plan)
    assert p.read_bytes() == original
