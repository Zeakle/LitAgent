"""Render release repair counts without granting quality or mutating source data."""

import pytest
from test_quality_view import render_repair_state


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_closeout_ui_distinguishes_inventory_recheck_from_prose_repair(locale):
    value = {
        "evaluation": {"faithfulness": {"skipped": True, "score": None}},
        "options": {
            "budgetPlan": {
                "contract": "planned-budget-v7",
                "status": "admitted",
                "repair_limit": 2,
                "envelope": {"closeout_profile": "engineering-closeout-v1"},
            },
            "targetedRepairState": {
                "version": "shared-targeted-repair-limit-v1",
                "limit": 2,
                "attempts": [
                    {"purpose": "bounded-inventory-recovery-v2", "status": "completed"},
                    {"stage": "evaluation", "status": "failed"},
                ],
            },
        },
    }
    rendered = render_repair_state(value, locale)
    expected = (
        "Inventory rechecks: 1 / 1 · Prose repairs: 1 / 2"
        if locale == "en"
        else "清单独立复核: 1 / 1 · 正文修复: 1 / 2"
    )
    assert expected in rendered["repairTexts"]
    assert rendered["metricStatuses"] == ["unverified"]
