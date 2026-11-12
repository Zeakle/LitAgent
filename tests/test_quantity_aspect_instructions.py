"""Keep dimensions explicit without accepting missing or misattributed support."""

import copy
import json
from pathlib import Path

import pytest
from test_batched_evaluation import BatchLLM, materials
from test_strict_automatic import frozen
from test_support_passages import selected

from litagent.audit_context import SurveyAuditContext
from litagent.benchmark.survey_judge import SurveyBenchmarkJudge, SurveyJudgeConfig
from litagent.config import EvalConfig
from litagent.eval.automatic import frozen_judge_fingerprint
from litagent.eval.scoped import (
    ASSESSMENT_PROMPT,
    STRICT_ASSESSMENT_PROMPT,
    STRICT_QUANTITY_CONTRACT,
    PassageDecision,
    build_support_passages,
    evaluate_scoped,
    required_support_aspects,
    validate_support_basis,
)
from litagent.quality import build_fact_cards, digest

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/nerf-quantity-aspects.json").read_text()
)


@pytest.mark.parametrize(
    "text",
    [
        "NeRF represents a continuous 5D radiance field.",
        "NeRF表示连续5D神经辐射场。",
        "UE4-NeRF renders at 4K resolution with up to 43 FPS.",
        "UE4-NeRF在4K分辨率下最高达到43 FPS。",
    ],
)
def test_dimensions_and_resolution_are_required_quantities(text):
    assert "quantities" in required_support_aspects(text)


def test_actual_old_no_quantity_decision_remains_unverified():
    case = FIXTURE["cases"][0]
    before = copy.deepcopy(case)
    decision = PassageDecision.model_validate(case["old_decision"])
    result = validate_support_basis(
        case["claim"],
        decision,
        case["ledger"],
        case["aliases"],
        passages=build_support_passages(
            case["ledger"], case["aliases"], original_only=True
        ),
        original_only=True,
    )
    assert result["issue"] == "support_aspect_unverified"
    assert result["aspect"] == "quantities"
    assert case == before


def test_explicit_dimension_aspect_passes_local_guard_but_wrong_dimension_fails():
    case = FIXTURE["cases"][0]
    claim = case["claim"]
    decision, passages = selected(claim, case["ledger"], case["aliases"])
    good = validate_support_basis(
        claim,
        decision,
        case["ledger"],
        case["aliases"],
        passages=passages,
        original_only=True,
    )
    assert good["status"] == "passed"
    wrong = {**claim, "claim_text": claim["claim_text"].replace("5D", "4D")}
    bad = validate_support_basis(
        wrong,
        decision,
        case["ledger"],
        case["aliases"],
        passages=passages,
        original_only=True,
    )
    assert bad["status"] == "unverified"
    assert bad["issue"] == "support_number_unverified"


def test_actual_wrong_ue4_citations_still_fail_with_affirmative_quantities():
    case = FIXTURE["cases"][1]
    decision, passages = selected(case["claim"], case["ledger"], case["aliases"])
    result = validate_support_basis(
        case["claim"],
        decision,
        case["ledger"],
        case["aliases"],
        passages=passages,
        original_only=True,
    )
    assert result["issue"] == "support_entity_unverified"
    assert result["entity"] == "UE4-NeRF"


@pytest.mark.parametrize("frame_rate,passed", [("43", True), ("430", False)])
def test_correct_source_frame_rate_does_not_license_a_different_number(
    frame_rate, passed
):
    source = "UE4-NeRF renders at 4K resolution with up to 43 FPS."
    ledger = {"a": {"text": "unverified", "supporting_text": source}}
    claim = {
        "claim_id": "c",
        "claim_text": f"UE4-NeRF renders at 4K resolution with up to {frame_rate} FPS.",
        "evidence_ids": ["a"],
    }
    decision, passages = selected(claim, ledger, {"a": "e1"})
    result = validate_support_basis(
        claim, decision, ledger, {"a": "e1"}, passages=passages, original_only=True
    )
    assert (result["status"] == "passed") is passed


class RecordingBatch(BatchLLM):
    """Record actual consumer prompts while preserving the scripted protocol."""

    def __init__(self):
        super().__init__()
        self.prompts = []

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        return await super().chat(messages, **kwargs)


@pytest.mark.asyncio
async def test_actual_evaluation_batch_receives_dimension_contract():
    survey, context = materials(1)
    source = "NeRF represents a continuous 5D radiance field."
    survey = survey.replace("ViT uses image patches.", source)
    context["evidence"]["a0"].update(text=source, supporting_text=source)
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    llm = RecordingBatch()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    assert len(llm.calls) == 3
    assert STRICT_QUANTITY_CONTRACT in llm.prompts[1]
    assert "quantities" in llm.calls[1]["assertions"][0]["required_support_aspects"]
    assert result["strict_quantity_contract"] == STRICT_QUANTITY_CONTRACT


@pytest.mark.asyncio
async def test_frozen_judge_receives_same_dimension_contract_in_two_calls():
    snapshot = frozen()
    llm = RecordingBatch()
    result, usage = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline")
    ).evaluate(
        case=SurveyAuditContext.model_validate(snapshot["context"]),
        survey=snapshot["report"]["survey"],
        ledger=snapshot["ledger"],
        frozen_inventory=snapshot["frozen_inventory"],
    )
    assert result.status == "completed" and usage.call_count == 2
    assert STRICT_QUANTITY_CONTRACT in llm.prompts[0]


def test_legacy_prompt_unchanged_and_frozen_identity_binds_quantity_contract(
    monkeypatch,
):
    import litagent.eval.scoped as scoped

    assert STRICT_QUANTITY_CONTRACT not in ASSESSMENT_PROMPT
    assert STRICT_QUANTITY_CONTRACT in STRICT_ASSESSMENT_PROMPT
    current = frozen_judge_fingerprint()
    monkeypatch.setattr(scoped, "STRICT_ASSESSMENT_PROMPT", ASSESSMENT_PROMPT)
    assert frozen_judge_fingerprint() != current
