"""Reproduce semantic evaluation defects without a provider or paid calls."""

import json
from decimal import Decimal

import pytest

from litagent.config import EvalConfig
from litagent.eval.scoped import (
    Decision,
    _numeric_facts,
    evaluate_scoped,
    required_support_aspects,
    validate_support_basis,
)
from litagent.llm.client import LLMResponse


def support_check(claim_text, selected_quote, *, original_support=None):
    """Check one owned passage while retaining the full source for explicit aliases."""
    ledger = {
        "paper-a": {
            "text": selected_quote,
            "supporting_text": original_support or selected_quote,
        }
    }
    decision = Decision(
        claim_id="claim_1",
        correct_kind="scientific",
        atomic=True,
        classification_valid=True,
        source_faithful=True,
        status="supported",
        reason_code="supported",
        explanation="Controlled assessment claims that this owned passage supports it.",
        support_basis=[
            {
                "evidence_alias": "e1",
                "source": "evidence_text",
                "quote": selected_quote,
            }
        ],
    )
    return validate_support_basis(
        {"claim_text": claim_text, "evidence_ids": ["paper-a"]},
        decision,
        ledger,
        {"paper-a": "e1"},
    )


@pytest.mark.parametrize(
    "text",
    [
        "DeiT-Tiny is trimmed to 50% of the original FLOPs.",
        "DeiT-Tiny FLOPs are reduced to 50% of the original cost.",
        "DeiT-Tiny FLOPs are reduced by 50%.",
    ],
)
def test_compute_reduction_keeps_quantity_metric_and_direction(text):
    facts = _numeric_facts(text)
    assert len(facts) == 1
    value, percent, metric, direction = facts[0]
    assert Decimal(value) == Decimal("50")
    assert percent is True
    assert metric == "compute"
    assert direction == "decrease"


def test_live_trimmed_to_passage_supports_chinese_remaining_flops_claim():
    """The live failure had an explicit remaining-compute result, not a speedup."""
    result = support_check(
        "DeiT-Tiny的FLOPs降低至原始的50%。",
        "DeiT-Tiny is trimmed to 50% of the original FLOPs.",
    )
    assert result["status"] == "passed"


@pytest.mark.parametrize("quantity", ["40", "50"])
@pytest.mark.parametrize("claim_form,source_form", [("to", "by"), ("by", "to")])
def test_reduction_to_and_by_are_not_interchangeable(quantity, claim_form, source_form):
    """Reject a changed meaning even where 50% makes the final arithmetic equal."""
    result = support_check(
        f"DeiT-Tiny FLOPs are reduced {claim_form} {quantity}%.",
        f"DeiT-Tiny FLOPs are reduced {source_form} {quantity}%.",
    )
    assert result["status"] == "unverified"


@pytest.mark.parametrize(
    "claim,quote",
    [
        (
            "DeiT-Tiny inference speed increases by 50%.",
            "DeiT-Tiny FLOPs are reduced by 50%.",
        ),
        (
            "DeiT-Tiny FLOPs increase by 50%.",
            "DeiT-Tiny FLOPs are reduced by 50%.",
        ),
        (
            "DeiT-Tiny FLOPs are reduced by 40%.",
            "DeiT-Tiny FLOPs are reduced by 50%.",
        ),
    ],
)
def test_reduction_fix_preserves_metric_direction_and_value_guards(claim, quote):
    assert support_check(claim, quote)["status"] == "unverified"


def test_dataset_identifier_does_not_create_a_result_quantity():
    text = "ViT uses JFT-300M pretraining."
    assert _numeric_facts(text) == []
    assert "quantities" not in required_support_aspects(text)


def test_dataset_identifier_does_not_hide_an_actual_result_quantity():
    text = "ViT pretrained on JFT-300M achieves 88.5% accuracy."
    facts = _numeric_facts(text)
    assert len(facts) == 1
    assert Decimal(facts[0][0]) == Decimal("88.5")
    assert facts[0][1:3] == (True, "accuracy")
    assert "quantities" in required_support_aspects(text)


def test_source_defined_acronym_supports_long_form_entity_in_selected_passage():
    """Alias identity may use this paper's definition; the result stays passage-owned."""
    quote = "LSA improves Tiny-ImageNet accuracy by 2.96%."
    source = "Locality Self-Attention (LSA) is proposed. " + quote
    result = support_check(
        "Locality Self-Attention improves Tiny-ImageNet accuracy by 2.96%.",
        quote,
        original_support=source,
    )
    assert result["status"] == "passed"


def test_unknown_long_form_cannot_borrow_source_acronym_definition():
    quote = "LSA improves Tiny-ImageNet accuracy by 2.96%."
    source = "Locality Self-Attention (LSA) is proposed. " + quote
    result = support_check(
        "Global Self-Attention improves Tiny-ImageNet accuracy by 2.96%.",
        quote,
        original_support=source,
    )
    assert result["status"] == "unverified"


def test_full_source_cannot_lend_result_numbers_to_an_uninformative_selected_passage():
    quote = "MPTQ-ViT is a quantization method."
    source = quote + " MPTQ-ViT improves accuracy by 0.90%."
    assert (
        support_check(
            "MPTQ-ViT improves accuracy by 0.90%.",
            quote,
            original_support=source,
        )["status"]
        == "unverified"
    )


class SemanticApprovalFake:
    """Approve adversarial classifications and rewrites at the model boundary."""

    def __init__(self, *, kind="scientific", rewrite=None):
        self.kind = kind
        self.rewrite = rewrite
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        payload = json.loads(messages[1]["content"])
        if len(self.calls) == 1:
            output = {
                "units": [
                    {
                        "source_unit_id": unit["source_unit_id"],
                        "assertions": [
                            {
                                "kind": self.kind,
                                "claim_text": (
                                    self.rewrite(fragment["text"])
                                    if callable(self.rewrite)
                                    else self.rewrite or fragment["text"]
                                ),
                                "source_fragment_id": fragment["source_fragment_id"],
                                "evidence_aliases": fragment["citation_aliases"],
                            }
                            for fragment in unit["source_fragments"]
                        ],
                        "exclusion_reason": "",
                    }
                    for unit in payload["source_units"]
                ]
            }
        else:
            output = {
                "units": [
                    {
                        "source_unit_id": unit["source_unit_id"],
                        "complete": True,
                        "explanation": "Controlled fake approves full coverage.",
                    }
                    for unit in payload["source_units"]
                ],
                "decisions": [
                    {
                        "claim_id": claim["claim_id"],
                        "correct_kind": self.kind,
                        "atomic": True,
                        "classification_valid": True,
                        "source_faithful": True,
                        "status": "supported",
                        "reason_code": "supported",
                        "explanation": "Controlled fake approves semantics and kind.",
                        "support_basis": [],
                        "aspect_checks": [],
                    }
                    for claim in payload["assertions"]
                ],
                "conflicts": [],
            }
        return LLMResponse(
            content=json.dumps(output, ensure_ascii=False),
            model="offline-semantic-approval",
            usage={"prompt_tokens": 100, "completion_tokens": 100},
        )


async def assert_host_rejects_semantic_approval(survey, fake, expected_error):
    """Allow explicit protocol rejection or a located, unverified host diagnostic."""
    try:
        result = await evaluate_scoped(
            llm=fake, survey=survey, context={"evidence": {}}, config=EvalConfig()
        )
    except ValueError as exc:
        assert expected_error in str(exc) or "classification_or_source" in str(exc)
    else:
        assert result["score"] is None
        assert any(
            expected_error in claim.get("local_error", "")
            or "classification" in claim.get("local_error", "")
            for claim in result["assertions"]
        )
    assert len(fake.calls) <= 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source,rewritten",
    [
        (
            "现有证据未覆盖以下方向的系统研究：\n- 三类架构在小样本图像分类上的直接对比；",
            "三类架构在小样本图像分类上的系统研究不足。",
        ),
        (
            "The selected evidence does not cover these directions:\n- Direct architecture comparisons on few-shot classification.",
            "There is no research comparing architectures on few-shot classification.",
        ),
    ],
)
async def test_missing_evidence_scope_cannot_be_upgraded_to_field_wide_absence(
    source, rewritten
):
    survey = "# Review\n\n## Open questions\n\n" + source + "\n"

    def rewrite_bullet(text):
        """Keep the scope preamble intact while the model upgrades its subordinate item."""
        if "直接对比" in text or "Direct architecture comparisons" in text:
            return rewritten
        return text

    fake = SemanticApprovalFake(rewrite=rewrite_bullet)
    await assert_host_rejects_semantic_approval(survey, fake, "source")


@pytest.mark.asyncio
async def test_pure_report_organization_rejects_scientific_denominator_even_if_approved():
    survey = (
        "# Review\n\n## Conclusion\n\n"
        "本综述系统梳理了ViT、DeiT和Swin Transformer在图像分类中的方法、实验表现及局限。\n"
    )
    await assert_host_rejects_semantic_approval(
        survey, SemanticApprovalFake(kind="scientific"), "classification"
    )


@pytest.mark.asyncio
async def test_correct_report_organization_remains_non_scientific():
    survey = (
        "# Review\n\n## Conclusion\n\n本综述比较ViT与CNN，并介绍选定方法的实验表现。\n"
    )
    result = await evaluate_scoped(
        llm=SemanticApprovalFake(kind="report_structure"),
        survey=survey,
        context={"evidence": {}},
        config=EvalConfig(),
    )
    assert result["factual_claim_count"] == 0
    assert result["assertions"][0]["kind"] == "report_structure"


@pytest.mark.asyncio
async def test_mixed_organization_sentence_keeps_scientific_comparison_checked():
    survey = "# Review\n\n## Conclusion\n\n本综述比较ViT与CNN，并发现ViT准确率更高。\n"
    await assert_host_rejects_semantic_approval(
        survey, SemanticApprovalFake(kind="report_structure"), "classification"
    )
