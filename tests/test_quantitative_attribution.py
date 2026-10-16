"""Reject explicit cross-result borrowing without changing legacy evaluation."""

import json
from pathlib import Path

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped
from litagent.quality import build_fact_cards, digest


async def evaluate_claim(claim, source, *, evidence_id="a0", paper_id="paper0"):
    """Exercise the complete strict evaluator with a deliberately permissive model."""
    _, context = materials(1)
    evidence = context["evidence"].pop("a0")
    evidence.update(
        text=source,
        supporting_text=source,
        evidence_id=evidence_id,
        paper_id=paper_id,
    )
    context["evidence"][evidence_id] = evidence
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    survey = "# Report\n\n## Results\n\n" + claim + f" [E:{evidence_id}]"
    llm = BatchLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    return survey, llm, result


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", [" ", "\n", "; ", "；", ""])
@pytest.mark.parametrize(
    "claim,first,second",
    [
        (
            "ResNet-18 achieves accuracy of 98.93% on TBX11K.",
            "ResNet-18 achieves accuracy of 92.4% on TBX11K.",
            "ResNet-12 achieves accuracy of 98.93% on TBX11K.",
        ),
        (
            "ResNet-18在TBX11K数据集上的准确率达到98.93%。",
            "ResNet-18在TBX11K数据集上的准确率达到92.4%。",
            "ResNet-12在TBX11K数据集上的准确率达到98.93%。",
        ),
        (
            "ResNet-18 achieves accuracy of 98.93% on TBX11K.",
            "ResNet-18 achieves accuracy of 92.4% on TBX11K.",
            "ResNet-18 achieves accuracy of 98.93% on CIFAR-10.",
        ),
    ],
)
async def test_number_in_another_explicit_result_cannot_be_borrowed(
    separator, claim, first, second
):
    survey, llm, result = await evaluate_claim(claim, first + separator + second)
    assert result["score"] != 1.0
    assertion = result["assertions"][0]
    assert assertion["status"] == "unverifiable"
    assert (
        assertion["support_validation"]["issue"]
        == "support_numeric_attribution_unverified"
    )
    assert (
        survey[assertion["source_start"] : assertion["source_end"]]
        == assertion["source_quote"]
    )
    assert len(llm.calls) == 3
    assert result["strict_binding_contract"] == "quantitative-attribution-v3"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "claim,source",
    [
        (
            "ResNet-18在TBX11K数据集上的准确率达到92.4%。",
            "ResNet-18 achieves accuracy of 92.4% on TBX11K. ResNet-12 achieves accuracy of 98.93% on TBX11K.",
        ),
        (
            "ResNet-12 achieves accuracy of 98.93% on TBX11K.",
            "ResNet-18 achieves accuracy of 92.4% on TBX11K. ResNet-12 achieves accuracy of 98.93% on TBX11K.",
        ),
        (
            "ResNet-18 achieves accuracy of 98.93% on CIFAR-10.",
            "ResNet-18 achieves accuracy of 92.4% on TBX11K. ResNet-18 achieves accuracy of 98.93% on CIFAR-10.",
        ),
        (
            "ResNet-18 achieves accuracy of 98.93% on TBX11K.",
            "ResNet-18 is evaluated on TBX11K. Its accuracy is 98.93%.",
        ),
        (
            "ResNet-18 achieves accuracy of 98.93% on TBX11K.",
            "ResNet-18 achieves accuracy of 92.4% on TBX11K with one-shot support. With five-shot support, its accuracy is 98.93% on TBX11K.",
        ),
    ],
)
async def test_explicit_correct_results_and_shared_context_are_preserved(claim, source):
    _, _, result = await evaluate_claim(claim, source)
    assert result["score"] == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("separator", [", ", "，", "; ", "\n"])
async def test_distinct_result_clauses_cannot_share_entity_presence(separator):
    source = (
        "ResNet-18 achieves accuracy of 92.4% on TBX11K"
        + separator
        + "ResNet-12 achieves accuracy of 98.93% on TBX11K."
    )
    _, _, result = await evaluate_claim(
        "ResNet-18 achieves accuracy of 98.93% on TBX11K.", source
    )
    assert (
        result["assertions"][0]["support_validation"]["issue"]
        == "support_numeric_attribution_unverified"
    )


@pytest.mark.parametrize("original_only", [False, True])
def test_new_binding_guard_is_strict_only_and_uses_explicit_source_aliases(
    original_only,
):
    from litagent.eval.scoped import Decision, validate_support_basis

    source = "Residual Network (RN) achieves accuracy of 92.4%. DeiT achieves accuracy of 98.93%."
    ledger = {"a": {"text": source, "supporting_text": source}}
    decision = Decision(
        claim_id="c1",
        correct_kind="scientific",
        atomic=True,
        classification_valid=True,
        source_faithful=True,
        status="supported",
        reason_code="supported",
        explanation="Deliberately permissive assessment.",
        support_basis=[
            {"evidence_alias": "e1", "source": "original_support", "quote": source}
        ],
    )
    result = validate_support_basis(
        {
            "claim_text": "Residual Network achieves accuracy of 98.93%.",
            "evidence_ids": ["a"],
        },
        decision,
        ledger,
        {"a": "e1"},
        original_only=original_only,
    )
    assert result["status"] == ("unverified" if original_only else "passed")
    if original_only:
        assert result["issue"] == "support_numeric_attribution_unverified"


def test_explicit_cross_result_guard_checks_separate_owned_passages():
    from litagent.eval.scoped import _quantitative_binding_issue

    quotes = [
        "ResNet-18 achieves accuracy of 92.4%.",
        "ResNet-12 achieves accuracy of 98.93%.",
    ]
    result = _quantitative_binding_issue(
        "ResNet-18 achieves accuracy of 98.93%.", quotes, quotes
    )
    assert result["issue"] == "support_numeric_attribution_unverified"


def test_strict_identity_binds_numeric_contract_without_changing_legacy():
    from litagent.config import load_config
    from litagent.contracts import build_config_summary

    config = load_config()
    before = build_config_summary(config)
    for policy in ("strict_v1", "strict_auto_v1"):
        result = build_config_summary(config, policy)
        contracts = result["effective"]["quality_policy"]["contracts"]
        assert "quantitative-attribution-v3" in contracts
        assert result["fingerprint"] != before["fingerprint"]
    assert build_config_summary(config, "legacy") == before


@pytest.mark.asyncio
async def test_frozen_control16_original_cannot_support_unreported_experiment():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/control16-attribution.json").read_text()
    )
    survey, llm, result = await evaluate_claim(
        fixture["claim_text"],
        fixture["source_quote"],
        evidence_id=fixture["evidence_id"],
        paper_id=fixture["paper_id"],
    )
    assert result["score"] != 1.0
    assertion = result["assertions"][0]
    assert assertion["status"] == "unverifiable"
    assert assertion["support_validation"]["issue"] == "support_entity_unverified"
    assert result["factual_claim_count"] == 1 and result["supported_count"] == 0
    assert assertion["evidence_ids"] == [fixture["evidence_id"]]
    assert (
        survey[assertion["source_start"] : assertion["source_end"]]
        == assertion["source_quote"]
    )
    assert (
        next(iter(llm.calls[1]["original_supports"].values()))["text"]
        == fixture["source_quote"]
    )
    assert (
        next(iter(llm.calls[1]["original_supports"].values()))["paper_id"]
        == fixture["paper_id"]
    )
    assert not fixture["human_gold_standard"]


@pytest.mark.asyncio
async def test_same_frozen_source_preserves_its_actual_supported_finding():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/control16-attribution.json").read_text()
    )
    _, _, result = await evaluate_claim(
        "LWFM-SPM achieves competitive performance across the four evaluated few-shot classification benchmarks.",
        fixture["source_quote"],
        evidence_id=fixture["evidence_id"],
        paper_id=fixture["paper_id"],
    )
    assert result["score"] == 1.0
