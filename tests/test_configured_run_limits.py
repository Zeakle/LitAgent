"""Keep server budget expansion aligned with actual requests and durable accounting."""

import copy

import pytest
import test_joint_round_escrow as helpers
from test_expanded_accounting_context import send, wire
from test_writer_v2 import case  # noqa: F401

from litagent.config import SafetyConfig, load_config
from litagent.output_capacity import configured_output_tokens
from litagent.run_limits import (
    CONTRACT,
    REQUEST_CONTRACT,
    PlannedRunLimits,
    configured_run_limits,
    envelope_run_limits,
    validate_run_limits,
)
from litagent.safety.budget import CostBudget
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import (
    PlannedBudget,
    request_scope,
    validate_writer_envelope,
)
from litagent.writing_v2 import writer_budget_plan


def expanded_plan(case, **kwargs):
    """Build a real writer plan; external transmission is never involved."""
    cfg = load_config("config/expanded-capacity.yaml")
    return helpers._joint_plan(
        case,
        output_tokens=configured_output_tokens(cfg),
        run_limits=configured_run_limits(cfg),
        **kwargs,
    )


@pytest.mark.parametrize("value", [True, 0, -1, 499999, 10_000_000.0, "10000000"])
def test_invalid_server_limits_are_not_coerced(value):
    with pytest.raises(ValueError):
        PlannedRunLimits(max_tokens=value)
    with pytest.raises(ValueError):
        validate_run_limits({"contract": CONTRACT, "max_tokens": value})


def test_limits_are_server_owned_and_absent_settings_preserve_old_serialization():
    old = SafetyConfig()
    assert "planned_run_limits" not in old.model_dump()
    for path in ("config/default.yaml", "config/expanded-capacity.yaml"):
        cfg = load_config(path)
        requests = {"max_requests": 200} if path == "config/default.yaml" else {}
        assert configured_run_limits(cfg) == {
            "contract": REQUEST_CONTRACT if requests else CONTRACT,
            "max_tokens": 10_000_000,
            **requests,
        }
        assert configured_run_limits(cfg.model_dump()) == configured_run_limits(cfg)
    with pytest.raises(ValueError):
        PlannedRunLimits(max_tokens=10_000_000, endpoint="https://example.invalid")
    from litagent.api import SurveyRequest

    with pytest.raises(ValueError):
        SurveyRequest(query="test", run_limits={"max_tokens": 10_000_000})


def test_request_contract_cannot_be_substituted_for_legacy():
    """Keep old request counts implicit and authenticate new counts explicitly."""
    old = {"contract": CONTRACT, "max_tokens": 10_000_000}
    assert validate_run_limits(old) == old
    assert envelope_run_limits({"run_limits": old})["max_requests"] == 64
    with pytest.raises(ValueError, match="run_limits_contract_invalid"):
        validate_run_limits({**old, "max_requests": 200})
    new = {**old, "contract": REQUEST_CONTRACT}
    for value in ({"max_requests": 64}, {"max_requests": True}, {}):
        with pytest.raises(ValueError):
            validate_run_limits({**new, **value})


def test_default_plan_counts_preparation_against_the_new_request_ceiling(case):
    """Accept 200 total requests, reject 201, and restore the frozen ceiling."""
    config = load_config("config/default.yaml")
    kwargs = {
        "output_tokens": configured_output_tokens(config),
        "run_limits": configured_run_limits(config),
    }
    baseline = helpers._base_plan(case.packet, case.aliases, **kwargs)
    writing_requests = baseline["planned_requests"]
    plan = helpers._base_plan(
        case.packet, case.aliases, spent_requests=200 - writing_requests, **kwargs
    )
    assert plan["status"] == "admitted"
    assert plan["maximum_requests"] == plan["planned_requests"] == 200
    assert plan["maximum_tokens"] == 10_000_000
    assert plan["repair_limit"] == 2
    owner = PlannedBudget(plan)
    restored = CostBudget.from_snapshot(owner.snapshot())
    assert restored.snapshot() == owner.snapshot()
    assert validate_accounting_snapshot(owner.snapshot(), expected_plan=plan)

    rejected = helpers._base_plan(
        case.packet, case.aliases, spent_requests=201 - writing_requests, **kwargs
    )
    assert rejected["planned_requests"] == 201
    assert rejected["reason"] == "budget_plan_requests_exceeded"


def test_new_plan_uses_real_context_ceiling_and_preserves_two_full_repairs(case):
    base, plan = expanded_plan(case)
    assert base["status"] == plan["status"] == "admitted"
    assert plan["maximum_tokens"] == 10_000_000
    assert 4_500_000 < plan["planned_tokens"] <= 10_000_000
    assert plan["maximum_requests"] == 64
    assert plan["repair_limit"] == 2
    assert plan["envelope"]["body_max_units"] == 66
    assert plan["envelope"]["model_assertions"] == 128
    assert set(plan["envelope"]["request_caps"].values()) == {176947}
    assert plan["stages"]["assessment"]["output_tokens"] == 32768
    assert plan["stages"]["repair"]["output_tokens"] == 24576
    assert plan["envelope"]["source_identity"] == base["envelope"]["source_identity"]
    _, old = helpers._joint_plan(case)
    assert old["maximum_tokens"] == 4_500_000
    assert "run_limits" not in old["envelope"]


@pytest.fixture
def completed_new_ledger(case, tmp_path):
    """Simulate every permitted request at full reservation, including both repairs."""
    _, plan = expanded_plan(case)
    owner = PlannedBudget(plan, path=tmp_path / "ledger.json")
    helpers._settle_initial_max(owner)
    domain = helpers._domain(plan)
    amounts = helpers._amounts(domain["reserved_tokens"], domain["max_requests"])
    for index in range(3):
        requests = [
            wire(plan, "assessment", amount, f"round:{index}:batch:{slot}")
            for slot, amount in enumerate(amounts)
        ]
        requests.append(wire(plan, "global", 176947, f"global:{index}"))
        helpers._allocate(owner, requests, report_sha=str(index + 1) * 64)
        for slot, request in enumerate(requests):
            send(owner, request, f"round:{index}:slot:{slot}")
        snapshot = owner.snapshot()
        assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
        assert validate_accounting_snapshot(snapshot, expected_plan=plan) == snapshot
        if index < 2:
            for stage in ("repair", "summary", "inventory"):
                helpers._settle_cap(owner, stage, f"candidate:{index}:{stage}")
    assert owner.snapshot()["used_tokens"] == plan["required_tokens"]
    assert owner.snapshot()["requests_sent"] == plan["planned_requests"] <= 64
    return plan, owner.snapshot()


def test_complete_maximum_path_is_affordable_and_accounting_is_idempotent(
    completed_new_ledger,
):
    plan, snapshot = completed_new_ledger
    assert plan["required_tokens"] <= snapshot["max_tokens"] <= 10_000_000
    assert sum(e["stage"] == "repair" for e in snapshot["entries"].values()) == 2
    assert validate_accounting_snapshot(snapshot, expected_plan=plan) == snapshot


@pytest.mark.parametrize("fault", ["limits", "stage", "context"])
def test_new_limits_do_not_allow_forged_usage_or_context(completed_new_ledger, fault):
    plan, original = completed_new_ledger
    snapshot = copy.deepcopy(original)
    if fault == "limits":
        snapshot["plan"]["maximum_tokens"] *= 2
    else:
        item = next(iter(snapshot["entries"].values()))
        item["reserved_tokens"] = plan["context_reservation_limit"] + 1
        if fault == "stage":
            item["actual_tokens"] = item["reserved_tokens"]
    with pytest.raises(ValueError):
        validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert original == completed_new_ledger[1]


def test_unknown_usage_remains_occupied_and_never_replayed(case):
    _, plan = expanded_plan(case)
    owner = PlannedBudget(plan)
    amount = plan["stages"]["body_initial"]["request_cap"]
    with request_scope("body_initial", "unknown"):
        owner.reserve(amount)
        owner.begin_request()
        owner.charge_unknown_usage(amount)
    before = owner.snapshot()
    restored = CostBudget.from_snapshot(before)
    assert restored.snapshot() == before
    assert restored.stop_reason == "strict_usage_unknown"
    with request_scope("body_initial", "unknown"):
        with pytest.raises(ValueError):
            restored.reserve(amount)
    assert restored.snapshot() == before


def test_limits_cannot_be_attached_to_older_budget_contracts(case):
    with pytest.raises(ValueError, match="run_limits_require_v7"):
        writer_budget_plan(
            case.packet,
            case.aliases,
            budget_policy="planned-budget-v6",
            run_limits={"contract": CONTRACT, "max_tokens": 10_000_000},
        )
    _, plan = expanded_plan(case)
    envelope = copy.deepcopy(plan["envelope"])
    envelope["run_limits"]["max_tokens"] += 1
    with pytest.raises(ValueError):
        validate_writer_envelope(envelope)


def test_configured_lower_limit_is_a_real_guard_and_not_only_a_display_value(case):
    cfg = load_config("config/expanded-capacity.yaml")
    plan = helpers._base_plan(
        case.packet,
        case.aliases,
        output_tokens=configured_output_tokens(cfg),
        run_limits={"contract": CONTRACT, "max_tokens": 4_500_000},
    )
    assert plan["status"] == "preflight_rejected"
    assert plan["reason"] == "budget_plan_total_exceeded"
    assert plan["maximum_tokens"] == 4_500_000
    with pytest.raises(ValueError, match="budget_preflight_rejected"):
        PlannedBudget(plan)


@pytest.mark.asyncio
@pytest.mark.parametrize("failed_candidates", [0, 2])
async def test_real_writer_review_evaluation_and_repairs_use_frozen_limits(
    case, tmp_path, monkeypatch, failed_candidates
):
    """Replace HTTP only; raising resource limits never grants scientific support."""
    from test_writer_source_view_pipeline import run_scientific_case

    import litagent.frozen_writing as frozen

    cfg = load_config("config/expanded-capacity.yaml")
    actual_plan = frozen.plan_frozen_budget

    def plan(*args, **kwargs):
        kwargs.setdefault("run_limits", configured_run_limits(cfg))
        return actual_plan(*args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", plan)
    result = await run_scientific_case(
        case, tmp_path, monkeypatch, failures=failed_candidates, config=cfg
    )
    saved = result.ledger["plan"]
    assert saved["maximum_tokens"] == 10_000_000
    assert saved["envelope"]["run_limits"] == configured_run_limits(cfg)
    assert result.report["delivery"]["publishable"] is (failed_candidates == 0)
    assert len(result.report["metadata"]["targeted_repair_state"]["attempts"]) == (
        1 if failed_candidates == 0 else 2
    )
    assert result.ledger["usage_uncertain"] is False
    validate_accounting_snapshot(result.ledger, expected_plan=saved)


@pytest.mark.asyncio
async def test_frozen_run_rejects_changed_server_limits_before_sending(
    case, tmp_path, monkeypatch
):
    """A changed allowance requires a new plan, never mutation of an old ledger."""
    from test_writer_source_view_pipeline import run_scientific_case

    import litagent.frozen_writing as frozen
    from litagent.writing_v2 import WriterError

    cfg = load_config("config/expanded-capacity.yaml")
    actual_plan, actual_run = frozen.plan_frozen_budget, frozen.run_frozen

    def plan(*args, **kwargs):
        kwargs.setdefault("run_limits", configured_run_limits(cfg))
        return actual_plan(*args, **kwargs)

    async def run(agent, *args, **kwargs):
        agent._config.safety.planned_run_limits.max_tokens += 1
        return await actual_run(agent, *args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", plan)
    monkeypatch.setattr(frozen, "run_frozen", run)
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        await run_scientific_case(case, tmp_path, monkeypatch, config=cfg)


def test_unselected_legacy_joint_domain_cannot_inherit_larger_stage_caps(case):
    """Only an explicitly selected server policy may widen the aggregate domain."""
    from litagent.safety.planned_budget import (
        _digest,
        build_budget_plan,
        with_joint_round_escrow,
    )

    old = helpers._base_plan(case.packet, case.aliases)
    envelope = copy.deepcopy(old["envelope"])
    envelope["request_caps"]["assessment"] = 100001
    envelope["identity"] = _digest(
        {k: v for k, v in envelope.items() if k != "identity"}
    )
    stages = {
        key: {k: v for k, v in stage.items() if k != "reserved_tokens"}
        for key, stage in old["stages"].items()
    }
    stages["assessment"]["request_cap"] = 100001
    changed = build_budget_plan(
        stages,
        identity=old["input_identity"],
        contract=old["contract"],
        envelope=envelope,
        repairs=2,
    )
    assert changed["status"] == "admitted"
    with pytest.raises(ValueError, match="joint_round_escrow_domain_invalid"):
        with_joint_round_escrow(changed)
