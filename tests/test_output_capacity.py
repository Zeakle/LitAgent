"""Prove expanded output limits reach real requests, escrows and recovery."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_structured_repair_pipeline as structured
import test_writer_source_view_pipeline as writer_view
from test_adaptive_capacity import settle_initial
from test_closeout_pipeline import RecheckGateway, select
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_rounds import plan_round, request_proof
from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.frozen_writing import (
    plan_frozen_budget,
    run_frozen,
    validate_frozen_writer_plan,
)
from litagent.output_capacity import (
    LEGACY_OUTPUT_TOKENS,
    configured_output_tokens,
    envelope_output_tokens,
    validate_output_tokens,
)
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.writing_v2 import RANGE_WRITING_PLAN, WriterError, writer_budget_plan
from scripts import run_frozen_writing as campaign

PROFILE_PATH = Path(__file__).parents[1] / "config/expanded-capacity.yaml"


@pytest.fixture
def expanded():
    """Load the shipped server profile rather than reconstruct a test-only policy."""
    return load_config(str(PROFILE_PATH))


def test_expanded_config_is_explicit_and_legacy_fingerprints_stay_unchanged(expanded):
    original = load_config()
    before = original.model_dump()
    assert configured_output_tokens(original) is None
    assert original.model_dump() == before
    assert original.adversarial.review_max_tokens == 4096
    assert expanded.adversarial.review_max_tokens == 16384
    assert expanded.extractor.max_tokens == 8192
    assert expanded.planner.max_tokens == 16384
    outputs = configured_output_tokens(expanded)
    assert outputs["body"] == outputs["summary"] == outputs["repair"] == 24576
    assert outputs["inventory"] == outputs["assessment"] == outputs["global"] == 32768
    campaign.check_profile(
        expanded, campaign.ROUND_PROFILE, writing_plan=RANGE_WRITING_PLAN
    )
    with pytest.raises(ValueError, match="output_conflict"):
        campaign.check_profile(
            campaign.profile_config(expanded, campaign.THINKING_PROFILE),
            campaign.THINKING_PROFILE,
        )


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "bool", "small", "large", "split"]
)
def test_bad_server_output_limits_never_enter_a_budget(fault):
    value = dict(LEGACY_OUTPUT_TOKENS)
    if fault == "missing":
        value.pop("reviewer")
    elif fault == "extra":
        value["unknown_stage"] = 32768
    elif fault == "bool":
        value["reviewer"] = True
    elif fault == "small":
        value["inventory"] = 8192
    elif fault == "large":
        value["reviewer"] = 32769
    else:
        value["summary"] = 32768
    with pytest.raises(ValueError, match="stage_output_capacity_invalid"):
        validate_output_tokens(value)


def test_optional_output_field_changes_identity_without_rewriting_old_plan(
    frozen_bundle, expanded
):
    flags = profile_flags(PROFILE)
    old = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v7", **flags)
    before = copy.deepcopy(old)
    new = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        output_tokens=configured_output_tokens(expanded),
        **flags,
    )
    assert "output_tokens" not in old["envelope"]
    assert envelope_output_tokens(old["envelope"]) == LEGACY_OUTPUT_TOKENS
    assert new["status"] == "admitted"
    assert old["identity"] != new["identity"]
    assert old == before
    for stage, spec in new["stages"].items():
        key = "body" if stage == "body_initial" else stage
        assert spec["output_tokens"] == new["envelope"]["output_tokens"][key]
    assert new["repair_limit"] == 2
    assert new["planned_tokens"] <= 4500000 and new["planned_requests"] <= 64
    validate_accounting_snapshot(PlannedBudget(new).snapshot(), expected_plan=new)
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(old, new)


@pytest.mark.asyncio
async def test_old_plan_cannot_hide_expanded_server_settings_before_client_setup(
    frozen_bundle, expanded
):
    flags = profile_flags(PROFILE)
    old = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v7", **flags)
    agent = SimpleNamespace(_wired=False, _config=expanded)
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        await run_frozen(
            agent,
            frozen_bundle,
            "not-started-output-drift",
            budget_policy="planned-budget-v7",
            expected_budget_plan=old,
            **flags,
        )
    assert not agent._wired and not hasattr(agent, "_llm")


def test_two_full_repairs_and_final_assessment_remain_funded(case, expanded):
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        output_tokens=configured_output_tokens(expanded),
    )
    owner = PlannedBudget(plan)
    settle_initial(owner, maximum=True)
    for round_index in range(3):
        requests = []
        for index in range(9):
            stage = "global" if index == 8 else "assessment"
            amount = plan["stages"][stage]["request_cap"]
            messages = [
                {"role": "user", "content": f"round {round_index}, slot {index}"}
            ]
            overhead = strict_request_reservation(messages, 32768)
            messages[0]["content"] += "x" * (amount - overhead)
            proof = request_proof(stage, messages, 32768, None)
            assert proof["reserved_tokens"] == amount
            requests.append((stage, messages, proof))
        plan_round(
            owner,
            assertions=128,
            batches=8,
            requests=[item[2] for item in requests],
            report_sha256=str(round_index) * 64,
        )
        for index, (stage, messages, proof) in enumerate(requests):
            with request_scope(stage, f"round:{round_index}:{index}"):
                amount = owner.check_wire(messages, 32768)
                assert amount == proof["reserved_tokens"]
                owner.reserve(amount)
                owner.begin_request()
                owner.record(
                    {"prompt_tokens": amount - 32768, "completion_tokens": 32768}
                )
        if round_index < 2:
            for stage in ("repair", "summary", "inventory"):
                spec = plan["stages"][stage]
                amount, output = spec["request_cap"], spec["output_tokens"]
                with request_scope(stage, f"candidate:{round_index}:{stage}"):
                    owner.reserve(amount)
                    owner.begin_request()
                    owner.record(
                        {"prompt_tokens": amount - output, "completion_tokens": output}
                    )
    result = owner.snapshot()
    assert result["used_tokens"] == plan["required_tokens"] <= plan["planned_tokens"]
    assert len(owner._stage_entries("repair")) == 2
    assert len(owner._stage_entries("global")) == 3
    assert CostBudget.from_snapshot(result).snapshot() == result


def test_expanded_batches_authenticate_output_limit_during_restore(case, expanded):
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        output_tokens=configured_output_tokens(expanded),
    )
    owner = PlannedBudget(plan)
    settle_initial(owner)
    requests = []
    for index in range(10):
        stage = "global" if index == 9 else "assessment"
        messages = [{"role": "user", "content": f"complete batch {index}"}]
        requests.append((stage, messages, request_proof(stage, messages, 32768, None)))
    plan_round(
        owner,
        assertions=144,
        batches=9,
        requests=[item[2] for item in requests],
        report_sha256="a" * 64,
    )
    for index, (stage, messages, proof) in enumerate(requests):
        with request_scope(stage, f"expanded:{index}"):
            amount = owner.check_wire(messages, 32768)
            assert amount == proof["reserved_tokens"]
            owner.reserve(amount)
            owner.begin_request()
            owner.record({"prompt_tokens": amount - 32768, "completion_tokens": 32768})
    snapshot = owner.snapshot()
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot


def test_expanded_repair_keeps_full_material_beyond_the_old_context_slot(
    case, expanded
):
    """More output must not push a previously complete repair over its old slot."""
    outputs = configured_output_tokens(expanded)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        output_tokens=outputs,
    )
    assert plan["context_tokens"] == 196608
    owner = PlannedBudget(plan)
    maximum = plan["stages"]["repair"]["request_cap"]
    assert maximum > 117964
    messages = [{"role": "user", "content": "complete original and exact diagnostics"}]
    overhead = strict_request_reservation(messages, outputs["repair"])
    messages[0]["content"] += "x" * (maximum - overhead)
    with request_scope("repair", "full-new-repair"):
        assert owner.check_wire(messages, outputs["repair"]) == maximum
        with pytest.raises(ValueError, match="planned_request_envelope_exceeded"):
            owner.check_wire(messages, 20000)
    assert owner.snapshot()["requests_sent"] == 0
    expanded.eval.context_tokens = 131072
    with pytest.raises(ValueError, match="stage_output_context_capacity_invalid"):
        configured_output_tokens(expanded)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 2])
async def test_expanded_actual_sdk_writer_reviewer_repair_and_complete_evaluation(
    source_case, tmp_path, monkeypatch, expanded, failing_candidates
):
    select(monkeypatch)
    result = await writer_view.run_scientific_case(
        source_case,
        tmp_path,
        monkeypatch,
        failures=failing_candidates,
        config=expanded,
    )
    assert result.report["delivery"]["publishable"] is (failing_candidates == 0)
    assert len(result.report["metadata"]["targeted_repair_state"]["attempts"]) == (
        1 if failing_candidates == 0 else 2
    )
    assert result.report["evaluation"]["faithfulness"]["details"]["coverage"][
        "complete"
    ]
    assert result.journal["calls"]["reviewer"]["max_tokens"] == 16384
    assert any(name.startswith("repair_") for name in result.journal["calls"])
    reviewer_seen = evaluation_seen = 0
    for entry in result.gateway.sent:
        request = entry["request"]
        output = request.get("max_output_tokens", request.get("max_tokens"))
        if "input" in request:
            assert output in (16384, 24576)
            reviewer_seen += output == 16384
        else:
            evaluation_seen += 1
            assert output == 32768
    assert reviewer_seen == 1 and evaluation_seen >= 6
    assert not result.ledger["usage_uncertain"]
    validate_accounting_snapshot(result.ledger, expected_plan=result.ledger["plan"])


@pytest.mark.asyncio
async def test_expanded_inventory_recheck_keeps_one_prose_repair(
    source_case, tmp_path, monkeypatch, expanded
):
    select(monkeypatch)
    gateway = RecheckGateway(source_case)
    monkeypatch.setattr(structured, "LongScientificGateway", lambda *_a, **_kw: gateway)
    result = await writer_view.run_scientific_case(
        source_case, tmp_path, monkeypatch, config=expanded
    )
    attempts = result.report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2
    assert attempts[0]["purpose"] == "bounded-inventory-recovery-v2"
    assert attempts[0]["inventory_review"]["status"] == "resolved"
    assert result.report["delivery"]["publishable"]
    rechecks = [row for row in gateway.sent if "inventory_recovery" in row["payload"]]
    assert len(rechecks) == 1 and rechecks[0]["request"]["max_tokens"] == 32768
    assert CostBudget.from_snapshot(result.ledger).snapshot() == result.ledger
