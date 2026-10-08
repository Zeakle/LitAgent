"""Reject real false-positive support without consulting services or old artifacts."""

import json
from pathlib import Path

import pytest
from test_scoped_evaluation import ScriptedLLM, ctx

from litagent.config import EvalConfig, load_config
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.eval.scoped import (
    CLASSIFICATION_CONTRACT,
    SUPPORT_CONTRACT,
    Decision,
    evaluate_scoped,
    has_scientific_predicate,
    validate_support_basis,
)
from litagent.report_render import build_report_source_units
from litagent.runner import LitAgent

FIXTURE = json.loads(
    Path(__file__)
    .with_name("fixtures")
    .joinpath("run003-support-cases.json")
    .read_text()
)


def decision(basis, kind="scientific", **extra):
    """Build a complete second-stage decision with independently declared kind."""
    return Decision(
        claim_id="claim_0",
        correct_kind=kind,
        atomic=True,
        classification_valid=True,
        source_faithful=True,
        status="supported",
        reason_code="supported",
        explanation="Support selected.",
        support_basis=basis,
        **extra,
    )


def basis(eid="e1", source="original_support", quote="EViT speeds up DeiT-S by 50%."):
    """Select exact original text in the test's actual citation."""
    return {"evidence_alias": eid, "source": source, "quote": quote}


@pytest.mark.parametrize(
    "case",
    [i for i in FIXTURE["issues"] if i["kind"] == "scientific"],
    ids=lambda i: i["issue_type"],
)
def test_actual_run003_false_positives_do_not_pass_host_support(case):
    """Use exact bound source evidence, including the fully supported compound part."""
    ledger = {
        m["alias"]: {
            "text": m["evidence_text"],
            "supporting_text": m["original_support"]["text"],
        }
        for m in case["materials"]
    }
    refs = list(ledger)
    proof = [basis(e, quote=m["supporting_text"]) for e, m in ledger.items()]
    value = validate_support_basis(
        {"claim_text": case["claim"], "evidence_ids": refs},
        decision(proof),
        ledger,
        {e: e for e in refs},
    )
    assert value["status"] == "unverified"
    assert value["issue"] in {
        "support_entity_unverified",
        "support_metric_unverified",
        "support_number_unverified",
    }
    assert value["spans"]


@pytest.mark.parametrize(
    "quote,source,alias,expected",
    [
        ("EViT ... 50%.", "original_support", "e1", "support_basis_quote_invalid"),
        (
            "EViT speeds up DeiT-S by 50%.",
            "original_support",
            "e2",
            "support_basis_ownership_invalid",
        ),
        (
            "EViT speeds up DeiT-S by 50%.",
            "evidence_text",
            "e1",
            "support_basis_quote_invalid",
        ),
        ("invented", "original_support", "e1", "support_basis_quote_invalid"),
    ],
)
def test_basis_must_be_continuous_and_owned(quote, source, alias, expected):
    ledger = {
        "a": {
            "text": "A compressed summary.",
            "supporting_text": "EViT speeds up DeiT-S by 50%.",
        },
        "b": {"text": "EViT speeds up DeiT-S by 50%."},
    }
    result = validate_support_basis(
        {"claim_text": "EViT speeds up DeiT-S by 50%.", "evidence_ids": ["a"]},
        decision([basis(alias, source, quote)]),
        ledger,
        {"a": "e1", "b": "e2"},
    )
    assert result["issue"] == expected


def test_complete_original_supports_numeric_fact_missing_from_compression():
    text = "SPT and LSA improve accuracy on Tiny-ImageNet by an average of 2.96%."
    result = validate_support_basis(
        {"claim_text": text, "evidence_ids": ["a"]},
        decision([basis(quote=text)]),
        {"a": {"text": "SPT and LSA improve accuracy.", "supporting_text": text}},
        {"a": "e1"},
    )
    assert result["status"] == "passed"
    assert result["spans"][0]["source_start"] == 0
    assert result["spans"][0]["source_end"] == len(text)


@pytest.mark.parametrize(
    "claim,proof,expected",
    [
        ("MCTF reduces FLOPs by 44%.", "MCTF reduces FLOPs by 44%.", "passed"),
        (
            "MCTF achieves a speedup of 44%.",
            "MCTF reduces FLOPs by 44%. MCTF achieves a speedup of 31%.",
            "unverified",
        ),
        ("Accuracy improves by +0.5%.", "Accuracy improves by 0.50%.", "passed"),
        ("Accuracy improves by 0%.", "Accuracy improves by 0%.", "passed"),
        ("Accuracy improves by -0.5%.", "Accuracy improves by +0.5%.", "unverified"),
        (
            "EViT speeds up DeiT-S by 50%.",
            "EViT speeds up DeiT-S by 150%.",
            "unverified",
        ),
        (
            "EViT increases inference speed by 50% with 0.3% accuracy loss.",
            "EViT increases inference speed by 50% with 0.3% accuracy loss.",
            "passed",
        ),
        ("Accuracy improves by 0.3%.", "Accuracy decreases by 0.3%.", "unverified"),
        ("Latency decreases by 20%.", "Throughput increases by 20%.", "unverified"),
        ("MCTF减少44%的FLOPs。", "MCTF减少44%的FLOPs。", "passed"),
    ],
)
def test_numerical_metric_bindings_are_not_bag_of_numbers(claim, proof, expected):
    result = validate_support_basis(
        {"claim_text": claim, "evidence_ids": ["a"]},
        decision([basis(quote=proof)]),
        {"a": {"text": "Summary", "supporting_text": proof}},
        {"a": "e1"},
    )
    assert result["status"] == expected


@pytest.mark.parametrize(
    "text",
    [
        FIXTURE["issues"][-2]["claim"],
        FIXTURE["issues"][-1]["claim"],
        "本综述发现ViT优于CNN。",
        "未来研究旨在提升ViT在小数据上的性能。",
    ],
)
def test_mixed_scientific_organization_is_not_exempted(text):
    assert has_scientific_predicate(text)


@pytest.mark.parametrize(
    "text",
    [
        "This review compares ViT and CNN.",
        "This review covers data efficiency improvements and open challenges.",
        "本综述比较ViT、CNN与数据效率。",
        "本综述将选定方法划为三类。",
    ],
)
def test_pure_organization_remains_organization(text):
    assert not has_scientific_predicate(text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["missing_basis", "wrong_entity", "wrong_kind", "compound", "org_science"],
)
async def test_supported_model_decisions_cannot_override_host_gates(failure):
    text = "EViT accelerates DeiT-S by 50%."
    if failure == "wrong_entity":
        text = "Evo-ViT accelerates DeiT-S by 50%."
    if failure == "org_science":
        text = "This review finds that EViT accelerates DeiT-S by 50%."
    survey = f"# Review\n\n{text} [E:a]\n"
    u = build_report_source_units(survey)[0]
    from litagent.eval.scoped import build_source_fragments

    inventory = [
        {
            "source_unit_id": u.unit_id,
            "exclusion_reason": "",
            "assertions": [
                {
                    "kind": (
                        "report_structure" if failure == "org_science" else "scientific"
                    ),
                    "claim_text": text,
                    "source_fragment_id": build_source_fragments(u)[0][
                        "source_fragment_id"
                    ],
                    "evidence_aliases": ["e1"],
                }
            ],
        }
    ]

    def mutate(stage, output, payload):
        if stage == 2:
            if failure == "missing_basis":
                output["decisions"][0]["support_basis"] = []
            elif failure == "wrong_kind":
                output["decisions"][0]["correct_kind"] = "report_structure"
            elif failure == "compound":
                output["decisions"][0]["atomic"] = False

    llm = ScriptedLLM(inventory=inventory, mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, ctx()
    )
    assert result.skipped and result.details["coverage"]["complete"] is False
    assert result.details["assertions"][0]["status"] == "unverifiable"
    assert result.details["support_contract"] == SUPPORT_CONTRACT
    assert result.details["classification_contract"] == CLASSIFICATION_CONTRACT
    assert len(llm.calls) == 2
    assert LitAgent._derive_quality(
        {
            "citation_accuracy": {"passed": True},
            "faithfulness": {"skipped": True, "details": result.details},
        }
    )["status"] == ("failed" if failure == "org_science" else "unverified")


def test_actual_spt_lsa_original_support_retains_numeric_positive_control():
    positive = FIXTURE["positive_controls"][0]
    material = positive["actual_cited_materials"][0]
    claim = positive["frozen_assertion"]["claim_text"]
    result = validate_support_basis(
        {"claim_text": claim, "evidence_ids": ["a"]},
        decision([basis(quote=material["original_support"]["text"])]),
        {
            "a": {
                "text": material["evidence_text"],
                "supporting_text": material["original_support"]["text"],
            }
        },
        {"a": "e1"},
    )
    assert result["status"] == "passed"
    assert "2.96%" not in material["evidence_text"]
    assert "2.96%" in result["spans"][0]["quote"]


@pytest.mark.asyncio
async def test_missing_correct_kind_or_atomic_never_reuses_old_decision_contract():
    survey = "# Review\n\nEViT accelerates DeiT-S by 50%. [E:a]\n"

    def mutate(stage, output, payload):
        if stage == 2:
            del output["decisions"][0]["correct_kind"]

    llm = ScriptedLLM(mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, ctx()
    )
    assert result.skipped and result.details["coverage"]["complete"] is False
    assert result.details["support_contract"] == SUPPORT_CONTRACT
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_invalid_provider_output_retains_host_language_failure():
    from litagent.review_structure import check_report_language

    survey = "# 中文报告\n\nThe proposed method improves the accuracy of the model while reducing the number of tokens required for inference.\n"

    def mutate(stage, output, payload):
        if stage == 1:
            output["units"] = []

    language = check_report_language(survey, "zh-CN")
    result = await RagasFaithfulnessEvaluator(
        load_config(), llm=ScriptedLLM(mutate=mutate)
    ).evaluate(survey, ctx(language_checks=language))
    assert result.skipped and result.details["language_checks"]["status"] == "failed"
    assert (
        LitAgent._derive_quality(
            {
                "citation_accuracy": {"passed": True},
                "faithfulness": {"skipped": True, "details": result.details},
            }
        )["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "claim,expected",
    [
        ("DeiT-T improves performance by 0.5%.", "passed"),
        ("DeiT-S improves performance by 0.3%.", "passed"),
        ("DeiT-S improves performance by 0.5%.", "unverified"),
        ("DeiT-T improves performance by 0.3%.", "unverified"),
        (
            "DeiT-T and DeiT-S improve performance by 0.5% and 0.3% respectively.",
            "passed",
        ),
        (
            "DeiT-T and DeiT-S improve performance by 0.3% and 0.5% respectively.",
            "unverified",
        ),
        ("The method improves performance by 0.5%.", "unverified"),
    ],
)
def test_ordered_results_cannot_swap_experimental_models(claim, expected):
    source = "DeiT-T and DeiT-S improve performance by 0.5% and 0.3% respectively."
    result = validate_support_basis(
        {"claim_text": claim, "evidence_ids": ["a"]},
        decision([basis(quote=source)]),
        {"a": {"text": "Summary", "supporting_text": source}},
        {"a": "e1"},
    )
    assert result["status"] == expected


def test_separate_support_spans_cannot_lend_a_metric_to_an_unqualified_number():
    ledger = {
        "a": {"text": "The inference speed increases."},
        "b": {"text": "FLOPs decrease by 44%."},
    }
    proof = [
        basis("e1", "evidence_text", "The inference speed increases."),
        basis("e2", "evidence_text", "44%"),
    ]
    result = validate_support_basis(
        {
            "claim_text": "The inference speed increases by 44%.",
            "evidence_ids": ["a", "b"],
        },
        decision(proof),
        ledger,
        {"a": "e1", "b": "e2"},
    )
    assert result["status"] == "unverified"
    assert result["issue"] == "support_metric_unverified"


@pytest.mark.asyncio
async def test_exclusion_cannot_hide_a_clear_scientific_predicate():
    survey = "# Review\n\nEViT accelerates DeiT-S by 50%. [E:a]\n\nEViT achieves 90% accuracy.\n"

    def mutate(stage, output, payload):
        if stage == 1:
            output["units"][1]["assertions"] = []
            output["units"][1]["exclusion_reason"] = "Organization only."

    llm = ScriptedLLM(mutate=mutate)
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, ctx()
    )
    assert result.skipped
    assert result.details["skipped_reason"] == "evaluation_extraction_incomplete"
    assert len(llm.calls) == 2
