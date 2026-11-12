"""Show original model verdicts beside conservative host effects without promotion."""

import copy
import json

import pytest
from test_quality_view import render_repair_state
from test_writer_v2 import case  # noqa: F401

POLICY = "host-unverified-continue-v1"
EFFECT = "host-assessment-conflict-effect-v1"
CODES = (
    "positive_status_with_negative_required_aspect",
    "negative_status_with_complete_positive_diagnostics",
)


def projection(code, *, partial=False):
    """Describe the new typed display shape without granting scientific validity."""
    positive = code == CODES[0]
    claim = {
        "claim_id": "claim_48",
        "claim_text": "A source-bound assertion needing independent review.",
        "status": "unverifiable",
        "reason_code": "unverifiable",
        "local_error": "evaluation_decision_consistency_invalid",
        "source_quote": "The unchanged original text.",
        "assessment_conflict_effect": {
            "contract": EFFECT,
            "policy": POLICY,
            "policy_identity": "a" * 64,
            "code": code,
            "original_status": "supported" if positive else "unsupported",
            "original_reason_code": "supported" if positive else "insufficient_support",
            "decision_sha256": "b" * 64,
            "binding_sha256": "c" * 64,
        },
    }
    details = {
        "evaluation_version": "survey-evaluation-v3",
        "source_report_sha256": "current-report",
        "factual_claim_count": 1,
        "supported_count": 0,
        "unsupported_count": 0,
        "unverified_count": 1,
        "coverage": {"processed_units": 1, "source_units": 1, "complete": False},
        "unsupported_claims": [] if partial else [claim],
    }
    if partial:
        details["partial_assessment"] = {
            "stage": "host_validated",
            "supported_count": 0,
            "unsupported_count": 0,
            "unverified_count": 1,
            "processed_assertions": 1,
            "expected_assertions": 2,
            "assertions": [claim],
        }
    return {
        "evaluation": {
            "faithfulness": {"score": None, "skipped": True, "details": details}
        },
        "quality": {"status": "unverified"},
        "delivery": {"status": "blocked", "publishable": False},
        "options": {"qualityPolicy": "strict_auto_v2", "sha": "current-report"},
    }


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("code", CODES)
@pytest.mark.parametrize("partial", [False, True])
def test_raw_verdict_and_host_unverified_remain_separate(locale, code, partial):
    """Render two distinct meanings without deriving a positive score or mutating data."""
    value = projection(code, partial=partial)
    before = json.dumps(value, sort_keys=True)
    actual = render_repair_state(value, locale)
    en = locale == "en"
    assert ("Original model decision" if en else "模型原判") in actual["text"]
    assert ("Host effect" if en else "宿主处理结果") in actual["text"]
    assert code in actual["text"]
    assert ("adds no supported assertion" if en else "不能贡献支持数量") in actual[
        "text"
    ]
    assert actual["metricStatuses"] == ["unverified"]
    assert "1.000" not in actual["text"]
    if partial:
        assert ("not a final score" if en else "非最终分数") in actual["text"]
    assert json.dumps(value, sort_keys=True) == before


@pytest.mark.parametrize(
    "change", [{"contract": "unknown"}, {"policy": "unknown"}, {"code": "other"}]
)
def test_unknown_effect_shape_does_not_borrow_selected_policy_labels(change):
    """Unrelated legacy or malformed diagnostics retain their existing display."""
    value = projection(CODES[0])
    effect = value["evaluation"]["faithfulness"]["details"]["unsupported_claims"][0][
        "assessment_conflict_effect"
    ]
    effect.update(change)
    actual = render_repair_state(value, "en")
    assert "Original model decision" not in actual["text"]
    assert "Host effect" not in actual["text"]
    assert actual["metricStatuses"] == ["unverified"]


def test_old_diagnostic_without_effect_retains_its_original_presentation():
    """Absence of the opt-in effect never invents a new interpretation."""
    value = projection(CODES[1])
    details = value["evaluation"]["faithfulness"]["details"]
    details["unsupported_claims"][0].pop("assessment_conflict_effect")
    before = copy.deepcopy(value)
    actual = render_repair_state(value, "en")
    assert "Original model decision" not in actual["text"]
    assert "Host effect" not in actual["text"]
    assert "The unchanged original text." in actual["text"]
    assert value == before


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_effect_from_another_report_sha_is_not_presented_as_current(locale):
    """Keep a previous report's conservative effect outside the selected version."""
    value = projection(CODES[1])
    value["evaluation"]["faithfulness"]["details"][
        "source_report_sha256"
    ] = "prior-report"
    actual = render_repair_state(value, locale)
    assert ("Original model decision" if locale == "en" else "模型原判") not in actual[
        "text"
    ]
    assert ("Host effect" if locale == "en" else "宿主处理结果") not in actual["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["zh-CN", "en"])
async def test_real_frozen_initial_conflict_reaches_the_unchanged_quality_outlet(
    case, tmp_path, monkeypatch, locale
):
    """Follow actual provider raw, host effect and report metadata into the DOM."""
    from test_semantic_conflict_pipeline import run_conflict_case

    report, _, _ = await run_conflict_case(case, tmp_path, monkeypatch)
    initial = report["metadata"]["evaluation_history"]["initial_evaluation"]
    details = initial["faithfulness"]["details"]
    value = {
        "evaluation": initial,
        "delivery": {"status": "blocked", "publishable": False},
        "quality": {"status": "unverified"},
        "options": {"sha": details["source_report_sha256"]},
    }
    before = copy.deepcopy(value)
    actual = render_repair_state(value, locale)
    assert ("Original model decision" if locale == "en" else "模型原判") in actual[
        "text"
    ]
    assert ("Host effect" if locale == "en" else "宿主处理结果") in actual["text"]
    assert CODES[1] in actual["text"]
    assert "unverified" in actual["metricStatuses"]
    assert value == before
