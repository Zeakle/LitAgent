"""Do not turn a bounded named result into an average or a guarantee."""

import copy

import pytest
from test_quantitative_attribution import evaluate_claim
from test_result_binding import check

from litagent.eval.scoped import _explicit_quantity_roles

SOURCE = "UE4-NeRF renders at 4K resolution with a frame rate of up to 43 FPS."


@pytest.mark.parametrize(
    "claim,source",
    [
        ("UE4-NeRF achieves an average frame rate of 43 FPS.", SOURCE),
        ("UE4-NeRF guarantees 43 FPS at 4K resolution.", SOURCE),
        ("UE4-NeRF achieves at least 43 FPS.", SOURCE),
        ("UE4-NeRF的平均帧率为43 FPS。", "UE4-NeRF的最高帧率为43 FPS。"),
        ("UE4-NeRF保证43 FPS。", "UE4-NeRF最多达到43 FPS。"),
        ("UE4-NeRF至少达到43 FPS。", "UE4-NeRF最高达到43 FPS。"),
        (
            "Model-A achieves average accuracy of 90%.",
            "Model-A achieves accuracy of up to 90%.",
        ),
        (
            "UE4-NeRF achieves an average frame rate of 43 FPS.",
            "UE4-NeRF achieves an average frame rate of 20 FPS and a peak frame rate of 43 FPS.",
        ),
        (
            "UE4-NeRF achieves an average frame rate of 43 FPS.",
            "UE4-NeRF achieves up to 43 FPS. Model-B achieves an average frame rate of 43 FPS.",
        ),
    ],
)
def test_explicit_qualifier_strengthening_is_unverified(claim, source):
    result = check(claim, source)
    assert result["status"] == "unverified"
    assert result["issue"] == "support_quantity_role_unverified"
    assert result["spans"]


@pytest.mark.parametrize(
    "claim,source",
    [
        ("UE4-NeRF achieves up to 43 FPS.", SOURCE),
        ("UE4-NeRF的最高帧率为43 FPS。", "UE4-NeRF最高达到43 FPS。"),
        (
            "UE4-NeRF achieves an average frame rate of 43 FPS.",
            "UE4-NeRF achieves up to 43 FPS. UE4-NeRF has an average frame rate of 43 FPS.",
        ),
        (
            "UE4-NeRF achieves an average frame rate of 20 FPS.",
            "UE4-NeRF achieves an average frame rate of 20 FPS and a peak frame rate of 43 FPS.",
        ),
        (
            "UE4-NeRF achieves an average frame rate of 43 FPS.",
            "UE4-NeRF achieves 43 FPS on average.",
        ),
        (
            "Model-A guarantees 90% accuracy.",
            "Model-A guarantees accuracy of 90%.",
        ),
    ],
)
def test_matching_explicit_roles_keep_local_support(claim, source):
    assert check(claim, source)["status"] == "passed"


@pytest.mark.parametrize(
    "claim,source",
    [
        (
            "UE4-NeRF achieves an average frame rate of 43 FPS.",
            "UE4-NeRF reaches 43 FPS.",
        ),
        ("UE4-NeRF reaches 43 FPS.", SOURCE),
        (
            "UE4-NeRF achieves an average frame rate of 43 FPS.",
            "UE4-NeRF is evaluated. It achieves up to 43 FPS.",
        ),
        (
            "UE4-NeRF achieves an average frame rate of 43 FPS.",
            "UE4-NeRF and Model-B achieve up to 43 FPS.",
        ),
        (
            "Model-A has an average latency of 43 milliseconds.",
            "Model-A has a maximum latency of 43 milliseconds.",
        ),
    ],
)
def test_unknown_roles_pronouns_multiple_owners_and_units_are_not_guessed(
    claim, source
):
    assert check(claim, source)["status"] == "passed"


def test_separate_result_cannot_lend_its_role_to_the_next_number():
    rows = _explicit_quantity_roles(
        "UE4-NeRF achieves average accuracy of 90% with a frame rate of up to 43 FPS."
    )
    assert [(r["value"], r["unit"], r["role"]) for r in rows] == [
        ("9E+1", "%", "average"),
        ("43", "fps", "upper"),
    ]


@pytest.mark.parametrize(
    "claim",
    [
        "UE4-NeRF achieves an average frame rate of 43 FPS.",
        "UE4-NeRF guarantees 43 FPS.",
    ],
)
def test_legacy_consumer_does_not_acquire_the_new_guard(claim):
    assert check(claim, SOURCE, strict=False)["status"] == "passed"


@pytest.mark.asyncio
async def test_actual_evaluation_rejects_positive_model_with_changed_upper_bound():
    claim = "UE4-NeRF achieves an average frame rate of 43 FPS."
    body, llm, result = await evaluate_claim(claim, SOURCE)
    assertion = result["assertions"][0]
    assert len(llm.calls) == 3 and result["score"] != 1.0
    assert assertion["status"] == "unverifiable"
    assert (
        assertion["support_validation"]["issue"] == "support_quantity_role_unverified"
    )
    assert (
        body[assertion["source_start"] : assertion["source_end"]]
        == assertion["source_quote"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim,expected",
    [
        ("UE4-NeRF achieves an average frame rate of 43 FPS.", 0.0),
        ("UE4-NeRF achieves up to 43 FPS.", 1.0),
    ],
)
async def test_frozen_judge_applies_same_guard_to_a_positive_model(claim, expected):
    from test_batched_evaluation import BatchLLM
    from test_strict_automatic import frozen

    from litagent.audit_context import SurveyAuditContext
    from litagent.benchmark.survey_judge import SurveyBenchmarkJudge, SurveyJudgeConfig
    from litagent.eval.automatic import freeze_automatic_inputs
    from litagent.quality import build_fact_cards, digest
    from litagent.report_render import build_report_source_units

    report = copy.deepcopy(frozen()["report"])
    report["survey"] = claim + " [E:a]"
    ledger = {
        "a": {
            "paper_id": "public-nerf",
            "text": "unverified",
            "supporting_text": SOURCE,
        }
    }
    cards = build_fact_cards(ledger)
    report["metadata"].update(fact_cards=cards, evidence_snapshot_sha256=digest(cards))
    unit = build_report_source_units(report["survey"])[0]
    assertion = report["evaluation"]["faithfulness"]["details"]["assertions"][0]
    assertion.update(
        claim_text=claim,
        source_quote=report["survey"],
        source_unit_id=unit.unit_id,
        source_fragment_id=f"{unit.unit_id}:fragment:0",
        source_start=0,
        source_end=len(report["survey"]),
    )
    snapshot = freeze_automatic_inputs(report, ledger, run_id="quantity-role-test")
    llm = BatchLLM()
    result, usage = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline")
    ).evaluate(
        case=SurveyAuditContext.model_validate(snapshot["context"]),
        survey=report["survey"],
        ledger=ledger,
        frozen_inventory=snapshot["frozen_inventory"],
    )
    assert result.status == "completed" and result.faithfulness == expected
    assert usage.call_count == 2
