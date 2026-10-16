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


def validate(claim, source=SOURCE):
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
    )


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
async def test_frozen_judge_consumes_the_bound_contract(wrong):
    from test_strict_automatic import frozen

    from litagent.audit_context import SurveyAuditContext
    from litagent.benchmark.survey_judge import SurveyBenchmarkJudge, SurveyJudgeConfig
    from litagent.eval.automatic import freeze_automatic_inputs
    from litagent.report_render import build_report_source_units

    report = copy.deepcopy(frozen()["report"])
    claim = CLAIM.replace("减少50%", "增加50%") if wrong else CLAIM
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
    snapshot = freeze_automatic_inputs(report, ledger, run_id="local-quantity")
    snapshot["frozen_inventory"]["local_numeric_contract"] = LOCAL_NUMERIC_CONTRACT
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
