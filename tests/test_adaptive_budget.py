"""Verify fixed-ceiling adaptive repairs without weakening complete evaluation."""

import copy
import json

import httpx
import pytest
from test_native_responses import envelope, owner_for
from test_planned_budget import stages
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import (
    ADAPTIVE_CONTRACT,
    PlannedBudget,
    adaptive_writer_plan,
    build_budget_plan,
    complete_writer_plan,
    request_scope,
)
from litagent.workbench import RunOptions, VerificationOptions
from litagent.writing_v2 import WriterError, WriterSession
from scripts import run_frozen_writing as campaign


def certified_bounds(assessment=18000):
    """Use controlled complete-wire ceilings, never historical-average estimates."""
    values = {
        k: {field: spec[field] for field in ("request_cap", "bound")}
        for k, spec in stages().items()
    }
    values["assessment"]["request_cap"] = assessment
    values["repair"]["request_cap"] = 40000
    return values


@pytest.mark.parametrize("cap,expected", [(18000, 2), (20000, 1), (35000, 0)])
def test_largest_complete_path_is_locked_before_writing(cap, expected):
    result = adaptive_writer_plan(
        identity="controlled-complete-path", stage_bounds=certified_bounds(cap)
    )
    assert result["contract"] == ADAPTIVE_CONTRACT
    assert result["status"] == "admitted"
    assert (
        result["repair_limit"] == result["scope_bounds"]["shared_repairs"] == expected
    )
    assert result["planned_tokens"] <= 1000000
    assert result["repair_allocation"] == "locked"
    assert result["initial_evaluation_reserved"]
    assert result["planned_requests"] == 17 + expected * 12
    assert result["stages"]["inventory"]["count"] == expected + 1
    assert result["stages"]["assessment"]["count"] == 8 * (expected + 1)
    assert result["stages"]["global"]["count"] == expected + 1
    assert result["stages"]["summary"]["count"] == expected + 1
    assert ("repair" in result["stages"]) is bool(expected)
    budget = PlannedBudget(result)
    for stage, spec in result["stages"].items():
        for index in range(spec["count"]):
            with request_scope(stage, f"{stage}:{index}"):
                budget.reserve(spec["request_cap"])
                budget.begin_request()
                budget.record(
                    {"prompt_tokens": spec["request_cap"], "completion_tokens": 0}
                )
    assert budget.used == result["required_tokens"]
    assert CostBudget.from_snapshot(budget.snapshot()).snapshot() == budget.snapshot()


def test_first_complete_evaluation_is_not_replaced_by_zero_repair_admission():
    result = adaptive_writer_plan(identity="unbounded-future-input")
    assert result["status"] == "preflight_rejected"
    assert result["repair_limit"] == 0
    assert result["repair_allocation"] == "unallocated"
    assert result["initial_evaluation_included"]
    assert not result["initial_evaluation_reserved"]
    assert result["stages"]["assessment"]["count"] == 8
    assert result["stages"]["global"]["count"] == 1
    with pytest.raises(ValueError, match="preflight_rejected"):
        PlannedBudget(result)


@pytest.mark.parametrize("missing", ["inventory", "assessment", "global", "reviewer"])
def test_missing_mandatory_stage_never_admits(missing):
    plan = adaptive_writer_plan(
        identity="controlled-complete-path", stage_bounds=certified_bounds()
    )
    invalid = copy.deepcopy(plan["stages"])
    invalid.pop(missing)
    for spec in invalid.values():
        spec.pop("reserved_tokens")
    with pytest.raises(ValueError, match="complete_path_invalid"):
        build_budget_plan(
            invalid, identity="missing", contract=ADAPTIVE_CONTRACT, repairs=2
        )


def test_first_chapter_slot_cannot_replace_later_chapter_slots():
    """The actual consumer uses body_initial once and body for later chapters."""
    plan = adaptive_writer_plan(
        identity="controlled", stage_bounds=certified_bounds(35000)
    )
    invalid = copy.deepcopy(plan["stages"])
    invalid["body_initial"] = invalid.pop("body")
    for spec in invalid.values():
        spec.pop("reserved_tokens")
    with pytest.raises(ValueError, match="complete_path_invalid"):
        build_budget_plan(
            invalid, identity="bad-topology", contract=ADAPTIVE_CONTRACT, repairs=0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cap,limit", [(35000, 0), (20000, 1)])
async def test_real_shared_repair_consumer_cannot_exceed_locked_limit(
    case, tmp_path, cap, limit
):
    owner = Provider(case)
    owner._cost_budget = PlannedBudget(
        adaptive_writer_plan(
            identity="actual-consumer", stage_bounds=certified_bounds(cap)
        )
    )
    session = WriterSession(
        owner, tmp_path / "repair.json", "v3", case.packet, case.aliases
    )
    targets = {
        "target": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
            "explanation": "Controlled original-bound diagnostic.",
        }
    }
    if limit:
        await session.patch(case.body, targets, cause="evaluation")
    calls = len(owner.calls)
    with pytest.raises(WriterError, match="repair_limit"):
        await session.patch(case.body, targets, cause="evaluation")
    assert len(owner.calls) == calls == limit
    assert len(session.state["attempts"]) == limit


def test_policy_identity_and_explicit_profile_preserve_v2_semantics():
    common = dict(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v2",
        verification=VerificationOptions(judge=False, jev=False),
    )
    v2 = RunOptions(**common, budget_policy="planned-budget-v2")
    v3 = RunOptions(**common, budget_policy="planned-budget-v3")
    assert v2.fingerprint() != v3.fingerprint()
    assert (
        complete_writer_plan(identity="unchanged")["scope_bounds"]["shared_repairs"]
        == 2
    )
    assert "repair_limit" not in complete_writer_plan(identity="unchanged")
    a = campaign.profile_parameters(
        campaign.PLANNED_PROFILE, writing_plan=campaign.RANGE_WRITING_PLAN
    )
    b = campaign.profile_parameters(
        campaign.ADAPTIVE_PROFILE, writing_plan=campaign.RANGE_WRITING_PLAN
    )
    assert a["budget_policy"] == "planned-budget-v2"
    assert b == {**a, "budget_policy": "planned-budget-v3"}
    assert json.loads(v3.model_dump_json())["budget_policy"] == "planned-budget-v3"


@pytest.mark.asyncio
async def test_sdk_records_one_locked_repair_and_never_sends_a_second(case, tmp_path):
    """Retain actual request encoding, accounting and consumers around mock HTTP."""
    requests = []

    async def transport(request):
        payload = json.loads(request.content)
        requests.append(payload)
        data = json.loads(payload["input"][1]["content"])
        result = {
            "outcome": "patched",
            "explanation": "Controlled original-bound repair.",
            "patches": [
                {
                    "target_id": key,
                    "sentences": [
                        {
                            "text": "Relation networks learn a comparison function.",
                            "fact_card_refs": [case.by_eid["b"]],
                        }
                    ],
                }
                for key in data["targets"]
            ],
        }
        return httpx.Response(200, json=envelope(json.dumps(result)))

    owner, _, _ = await owner_for(transport)
    owner._cost_budget = PlannedBudget(
        adaptive_writer_plan(
            identity="sdk-repair", stage_bounds=certified_bounds(20000)
        )
    )
    session = WriterSession(
        owner,
        tmp_path / "sdk-repair.json",
        "v3-sdk",
        case.packet,
        case.aliases,
        provider_api="responses",
    )
    targets = {
        "target": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
            "explanation": "A controlled current diagnostic.",
        }
    }
    try:
        await session.patch(case.body, targets, cause="evaluation")
        assert owner._cost_budget.used == 120
        assert owner._cost_budget.snapshot()["requests_sent"] == 1
        with pytest.raises(WriterError, match="repair_limit"):
            await session.patch(case.body, targets, cause="evaluation")
        assert len(requests) == 1
        assert requests[0]["thinking"] == {"type": "disabled"}
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_three_actual_frozen_plans_reject_before_creating_clients(
    manifest, case, monkeypatch
):
    """Exercise profile, frozen identity and admission without mocking the planner."""
    from litagent.quality import digest

    root, config = manifest
    read = campaign.read_bundle

    def bundle(path):
        return {
            **read(path),
            "selection": case.selection.to_dict(),
            "fact_cards_sha256": digest(case.cards),
            "requirements": case.packet["requirements"],
        }

    def forbid(*args, **kwargs):
        raise AssertionError("Rejected plans must not construct a client")

    monkeypatch.setattr(campaign, "read_bundle", bundle)
    monkeypatch.setattr(campaign, "LitAgent", forbid)
    target = root.with_name("adaptive-rejected")
    config = campaign.profile_config(config, campaign.ADAPTIVE_PROFILE)
    campaign.prepare(
        target,
        config,
        writing_profile=campaign.ADAPTIVE_PROFILE,
        writing_plan=campaign.RANGE_WRITING_PLAN,
    )
    frozen = campaign.validate(target, config)
    assert frozen["options"]["budget_policy"] == "planned-budget-v3"
    assert all(p["repair_limit"] == 0 for p in frozen["budget_plans"].values())
    assert frozen["limits"]["total_tokens"] == 3000000
    result = await campaign.execute(target, config)
    assert result["status"] == "preflight_rejected" and result["cases"] == []
