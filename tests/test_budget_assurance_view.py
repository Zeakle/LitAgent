"""Display adaptive reservations without claiming an unproved future guarantee."""

import copy

import pytest
from test_run_budget_view import RENDER, node_result, payload, recorded_budget


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("status", ["admitted", "preflight_rejected"])
@pytest.mark.parametrize("policy", ["planned-budget-v6", "planned-budget-v7"])
def test_adaptive_budget_view_separates_initial_capacity_from_future_guarantee(
    locale, status, policy
):
    """Run the real adapter and renderer while preserving stored decisions."""
    plan = {
        "contract": policy,
        "status": status,
        "identity": "a" * 64,
        "repair_limit": 2,
        "required_tokens": 4064945,
        "planned_tokens": 4480000,
        "maximum_tokens": 4500000,
        "planned_requests": 41,
        "maximum_requests": 64,
        "stages": {},
        "envelope": {
            "body_max_units": 66,
            "summary_max_units_per_section": 2,
            "model_assertions": 128,
            "assessment_batches": 8,
        },
    }
    raw = payload(recorded_budget(), budget_plan=plan)
    original = copy.deepcopy(raw)
    rendered = node_result(RENDER, {"raw": raw}, locale=locale)
    en = locale == "en"
    assert (
        "The initial batch count is not a guaranteed upper bound."
        if en
        else "初始批数不是保证上界。"
    ) in rendered["text"]
    assert (
        "future candidates require another resource check"
        if en
        else "未来候选仍须重新检查资源"
    ) in rendered["text"]
    assert (
        "Unknown future input uses an enforced upper bound."
        if en
        else "未知后续输入使用可执行的上界。"
    ) not in rendered["text"]
    assert "4064945 / 4480000 / 4500000" in rendered["text"]
    assert rendered["scores"] == ["unverified"]
    assert rendered["delivery"] == {"status": "blocked", "publishable": False}
    assert raw == original


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_historical_nonadaptive_budget_description_keeps_its_contract(locale):
    """Do not reinterpret a historical nonadaptive planning scope."""
    plan = {
        "contract": "planned-budget-v5",
        "status": "admitted",
        "repair_limit": 2,
        "stages": {},
    }
    rendered = node_result(
        RENDER, {"raw": payload(recorded_budget(), budget_plan=plan)}, locale=locale
    )
    assert (
        "Unknown future input uses an enforced upper bound."
        if locale == "en"
        else "未知后续输入使用可执行的上界。"
    ) in rendered["text"]
    assert (
        "The initial batch count is not a guaranteed upper bound."
        if locale == "en"
        else "初始批数不是保证上界。"
    ) not in rendered["text"]


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_joint_escrow_view_keeps_count_cap_and_total_independent(locale):
    """An aggregate escrow is not displayed as a false multiplication."""
    plan = {
        "contract": "planned-budget-v7",
        "status": "admitted",
        "repair_limit": 2,
        "stages": {
            "assessment": {
                "count": 45,
                "request_cap": 100000,
                "reserved_tokens": 2407782,
                "bound": "complete requests",
            }
        },
        "joint_round_escrow": {
            "contract": "joint-round-escrow-v1",
            "assessment": {
                "max_requests": 15,
                "request_cap": 100000,
                "reserved_tokens": 802594,
            },
        },
    }
    raw = payload(recorded_budget(), budget_plan=plan)
    original = copy.deepcopy(raw)
    rendered = node_result(RENDER, {"raw": raw}, locale=locale)
    assert "15 / 100000 / 802594" in rendered["text"]
    assert "45 × 100000 = 2407782" not in rendered["text"]
    assert (
        "Total reservation: 2407782" if locale == "en" else "总预留: 2407782"
    ) in rendered["text"]
    assert rendered["scores"] == ["unverified"]
    assert rendered["delivery"]["publishable"] is False
    assert raw == original
