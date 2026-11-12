"""Bind each explicit comparative to its own result without changing old callers."""

import copy
import json
from pathlib import Path

import pytest
from test_batched_evaluation import BatchLLM, materials
from test_result_binding import check
from test_support_passages import selected

from litagent.config import EvalConfig
from litagent.eval.scoped import (
    CHINESE_COMPARATIVE_CONTRACT,
    LOCAL_NUMERIC_CONTRACT,
    _bound_numeric_facts,
    evaluate_scoped,
    validate_support_basis,
)
from litagent.quality import build_fact_cards, digest

SOURCE = (
    "On the Pet dataset, Hint-Aug achieves a 2.22% higher accuracy "
    "with 50% less training data over SOTA data augmentation methods."
)
CLAIM = "Hint-Aug在Pet数据集上，使用减少50%的训练数据时，准确率提升2.22%。"
POSTPOSED_CLAIM = (
    "Hint-Aug方法在Pet数据集上，使用50%更少的训练数据时，"
    "准确率比当前最优数据增强方法高出2.22%，验证了其在数据高效场景下的有效性"
)


def validate(claim, source=SOURCE, *, chinese_comparatives=False):
    """Use actual cited originals and an intentionally permissive model decision."""
    item = {"claim_id": "c", "claim_text": claim, "evidence_ids": ["a"]}
    ledger = {"a": {"text": claim, "supporting_text": source}}
    decision, passages = selected(item, ledger, {"a": "e1"})
    return validate_support_basis(
        item,
        decision,
        ledger,
        {"a": "e1"},
        passages=passages,
        original_only=True,
        local_numeric_bindings=True,
        chinese_comparatives=chinese_comparatives,
    )


def test_real_postposed_chinese_comparison_has_its_own_explicit_target():
    fixture = json.loads(
        Path("tests/fixtures/hint-aug-local-comparatives.json").read_text()
    )
    assert validate(POSTPOSED_CLAIM, fixture["source"])["issue"] == (
        "support_metric_unverified"
    )
    result = validate(POSTPOSED_CLAIM, fixture["source"], chinese_comparatives=True)
    assert result["status"] == "passed"
    assert _bound_numeric_facts(POSTPOSED_CLAIM, chinese_comparatives=True) == [
        ("5E+1", True, "training_data", "decrease"),
        ("2.22", True, "accuracy", None),
    ]


@pytest.mark.parametrize(
    "claim",
    [
        POSTPOSED_CLAIM.replace("更少", "更多"),
        POSTPOSED_CLAIM.replace("50%", "40%"),
        POSTPOSED_CLAIM.replace("训练数据", "FLOPs"),
        POSTPOSED_CLAIM.replace("50%更少的训练数据", "减少到50%的训练数据"),
        POSTPOSED_CLAIM.replace("2.22%", "2.22个百分点"),
        POSTPOSED_CLAIM.replace("Pet", "CIFAR"),
    ],
)
def test_chinese_comparative_fix_does_not_license_other_facts(claim):
    assert validate(claim, chinese_comparatives=True)["status"] == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "analysis,wrong", [(False, False), (True, False), (True, True)]
)
async def test_real_evaluation_limits_new_comparatives_to_versioned_analysis(
    analysis, wrong
):
    from test_evaluation_wire_contract import AnalysisTransport

    _, context = materials(1)
    context["quality_policy"] = "strict_auto_v2"
    if analysis:
        context["body_quality"] = "analysis_v1"
    context["evidence"]["a0"].update(text=SOURCE, supporting_text=SOURCE)
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    claim = POSTPOSED_CLAIM.replace("更少", "更多") if wrong else POSTPOSED_CLAIM
    result = await evaluate_scoped(
        llm=AnalysisTransport(),
        survey=claim + " [E:a0]",
        context=context,
        config=EvalConfig(),
    )
    assert result["assertions"][0]["status"] == (
        "supported" if analysis and not wrong else "unverifiable"
    )
    assert result.get("local_comparative_contract") == (
        CHINESE_COMPARATIVE_CONTRACT if analysis else None
    )
    assert result["score"] == (1.0 if analysis and not wrong else None)


def test_adjacent_opposite_results_have_separate_targets_and_directions():
    assert _bound_numeric_facts(SOURCE) == [
        ("2.22", True, "accuracy", "increase"),
        ("5E+1", True, "training_data", "decrease"),
    ]
    assert validate(CLAIM)["status"] == "passed"
    # Historical algorithms and consumers retain their original behavior.
    assert check(CLAIM, SOURCE)["issue"] == "support_direction_unverified"


def test_real_full_abstract_and_reversed_chinese_suffix_are_checked_separately():
    fixture = json.loads(
        (
            Path(__file__).parent / "fixtures/hint-aug-local-comparatives.json"
        ).read_text()
    )
    assert len(fixture["source"]) > 1800
    assert validate(fixture["claim"], fixture["source"])["status"] == "passed"
    assert (
        validate(
            fixture["claim"].replace("准确率提升", "准确率下降"), fixture["source"]
        )["status"]
        == "unverified"
    )


def test_another_result_without_a_target_remains_unknown():
    assert _bound_numeric_facts("Model-A achieves 2% higher accuracy with 50%.")[
        -1
    ] == ("5E+1", True, None, None)


@pytest.mark.asyncio
async def test_real_v2_freeze_binds_numeric_contract(tmp_path, monkeypatch):
    from test_verified_extraction_freeze import ready

    from litagent.eval.automatic import freeze_automatic_inputs

    report, ledger, _, _ = await ready(tmp_path, monkeypatch)
    snapshot = freeze_automatic_inputs(report, ledger, run_id="test-auto")
    assert (
        snapshot["frozen_inventory"]["local_numeric_contract"] == LOCAL_NUMERIC_CONTRACT
    )


@pytest.mark.parametrize(
    "claim",
    [
        "Hint-Aug在Pet数据集上，训练数据增加50%时，准确率提升2.22%。",
        "Hint-Aug在Pet数据集上，训练数据减少2.22%时，准确率提升50%。",
        "Hint-Aug在Pet数据集上，训练数据减少50%时，准确率下降2.22%。",
        "Hint-Aug在Pet数据集上，训练数据减少到50%时，准确率提升2.22%。",
        "Hint-Aug在Pet数据集上，训练数据减少50%时，准确率提升2.22个百分点。",
    ],
)
def test_wrong_result_direction_amount_basis_or_unit_still_blocks(claim):
    assert validate(claim)["status"] == "unverified"


@pytest.mark.parametrize(
    "claim,source",
    [
        (CLAIM, SOURCE),
        (SOURCE, SOURCE),
        (
            "Hint-Aug reduces training data by 50% and increases accuracy by 2.22%.",
            SOURCE,
        ),
        (
            "Model-A increases accuracy by 2% with 20% fewer training samples.",
            "Model-A achieves 2% higher accuracy with 20% fewer training samples.",
        ),
    ],
)
def test_explicit_same_results_remain_locally_verifiable(claim, source):
    assert validate(claim, source)["status"] == "passed"


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong", [False, True])
async def test_actual_evaluation_uses_local_binding_in_v2_only(wrong):
    _, context = materials(1)
    context["quality_policy"] = "strict_auto_v2"
    context["evidence"]["a0"].update(text=SOURCE, supporting_text=SOURCE)
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    claim = CLAIM.replace("减少50%", "增加50%") if wrong else CLAIM

    class CaptureLLM(BatchLLM):
        """Retain actual prompts alongside controlled semantic responses."""

        def __init__(self):
            super().__init__()
            self.prompts = []

        async def chat(self, messages, **kwargs):
            self.prompts.append(messages[0]["content"])
            return await super().chat(messages, **kwargs)

    llm = CaptureLLM()
    result = await evaluate_scoped(
        llm=llm,
        survey=claim + " [E:a0]",
        context=context,
        config=EvalConfig(),
    )
    assert result["score"] is None if wrong else result["score"] == 1.0
    assert result["local_numeric_contract"] == LOCAL_NUMERIC_CONTRACT
    assert "source cited only once is valid" in llm.prompts[-1]
    assert result["assertions"][0]["status"] == (
        "unverifiable" if wrong else "supported"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("wrong", [False, True])
@pytest.mark.parametrize("chinese_comparatives", [False, True])
async def test_frozen_judge_consumes_the_bound_contract(wrong, chinese_comparatives):
    from test_strict_automatic import frozen

    from litagent.audit_context import SurveyAuditContext
    from litagent.benchmark.survey_judge import SurveyBenchmarkJudge, SurveyJudgeConfig
    from litagent.eval.automatic import freeze_automatic_inputs
    from litagent.report_render import build_report_source_units

    report = copy.deepcopy(frozen()["report"])
    correct = POSTPOSED_CLAIM if chinese_comparatives else CLAIM
    claim = (
        correct.replace("更少", "更多").replace("减少50%", "增加50%")
        if wrong
        else correct
    )
    report["survey"] = claim + " [E:a]"
    ledger = {"a": {"paper_id": "public", "text": CLAIM, "supporting_text": SOURCE}}
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
    if chinese_comparatives:
        report["evaluation"]["faithfulness"]["details"][
            "local_comparative_contract"
        ] = CHINESE_COMPARATIVE_CONTRACT
    snapshot = freeze_automatic_inputs(report, ledger, run_id="local-quantity")
    snapshot["frozen_inventory"]["local_numeric_contract"] = LOCAL_NUMERIC_CONTRACT
    assert snapshot["frozen_inventory"].get("local_comparative_contract") == (
        CHINESE_COMPARATIVE_CONTRACT if chinese_comparatives else None
    )
    result, usage = await SurveyBenchmarkJudge(
        BatchLLM(),
        SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline"),
    ).evaluate(
        case=SurveyAuditContext.model_validate(snapshot["context"]),
        survey=report["survey"],
        ledger=ledger,
        frozen_inventory=snapshot["frozen_inventory"],
    )
    assert result.status == "completed" and result.faithfulness == (
        0.0 if wrong else 1.0
    )
    assert usage.call_count == 2
