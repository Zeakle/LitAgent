"""Derive selected body prose without granting support to scientific premises."""

import copy

import pytest
from test_summary_body_reference import candidate, render
from test_writing_passages import facts

from litagent.quality import GroundedWritingError


def test_summary_copies_complete_body_text_and_rejects_a_second_free_text_field():
    cards, payload = gap_candidate()
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    assert set(summary) == {"body_reference"}
    body_text = payload["sections"]["experiments"]["first_block"]["first_sentence"][
        "text"
    ]
    assert render(payload, cards).count(body_text) == 2
    summary["text"] = "A second paper proves a universal optimum."
    with pytest.raises(GroundedWritingError):
        render(payload, cards)


def test_same_body_unit_cannot_inflate_findings_by_duplicate_summary_selection():
    cards, payload = gap_candidate()
    block = payload["sections"]["abstract"]["first_block"]
    block["additional_sentences"] = [copy.deepcopy(block["first_sentence"])]
    with pytest.raises(GroundedWritingError):
        render(payload, cards)


def gap_candidate(kind="gap"):
    """Keep a real uncited body unit and an exactly classified summary target."""
    cards = facts()
    payload = candidate(cards)
    body = payload["sections"]["experiments"]["first_block"]["first_sentence"]
    body.update(
        kind=kind,
        fact_card_ids=[],
        support_passage_ids=[],
        text="The selected materials do not record comparable experimental settings.",
    )
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    summary["body_reference"] = {
        "section_key": "experiments",
        "block_index": 0,
        "sentence_index": 0,
    }
    return cards, payload


@pytest.mark.parametrize("kind", ["gap", "organization"])
def test_same_kind_uncited_body_reference_produces_no_scientific_citation(kind):
    cards, payload = gap_candidate(kind)
    report = render(payload, cards)
    assert (
        report.count(
            "The selected materials do not record comparable experimental settings."
        )
        == 2
    )
    assert report.count("[E:a]") == 5


@pytest.mark.parametrize(
    "mutation",
    ["override_kind", "scientific_without_cards", "body_cards", "override_text"],
)
def test_structural_reference_cannot_borrow_scientific_material_or_another_kind(
    mutation,
):
    cards, payload = gap_candidate()
    body = payload["sections"]["experiments"]["first_block"]["first_sentence"]
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    if mutation == "override_kind":
        body["kind"] = "organization"
        summary["kind"] = "gap"
    elif mutation == "scientific_without_cards":
        body["kind"] = "scientific"
    elif mutation == "body_cards":
        body.update(fact_card_ids=list(cards), support_passage_ids=["r1.full"])
    else:
        summary["text"] = "Invented scientific finding."
    with pytest.raises(GroundedWritingError):
        render(payload, cards)


@pytest.mark.asyncio
async def test_scientific_premise_in_linked_gap_is_independently_evaluated():
    from test_batched_evaluation import BatchLLM

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import digest

    cards, payload = gap_candidate()
    text = "EViT achieves 80% accuracy on ImageNet."
    payload["sections"]["experiments"]["first_block"]["first_sentence"]["text"] = text
    report = render(payload, cards)
    source = next(iter(cards.values()))["source_quote"]
    context = {
        "query": "EViT speed",
        "quality_policy": "strict_auto_v2",
        "evaluation_version": "survey-evaluation-v3",
        "fact_cards": cards,
        "evidence_snapshot_sha256": digest(cards),
        "evidence": {
            "a": {
                "evidence_id": "a",
                "paper_id": "evit",
                "text": source,
                "supporting_text": source,
                "chunk_key": "abstract",
            }
        },
        "coverage_requirements": [
            {"requirement_id": "question", "kind": "question", "text": "EViT speed"}
        ],
        "structure_checks": {"status": "not_applicable"},
        "method_checks": {"status": "not_applicable"},
    }
    result = await evaluate_scoped(
        llm=BatchLLM(), survey=report, context=context, config=EvalConfig()
    )
    claims = [c for c in result["assertions"] if "80%" in c["claim_text"]]
    assert len(claims) == 2 and all(c["status"] != "supported" for c in claims)
    assert result["score"] != 1.0
    assert result["coverage"]["processed_units"] == result["coverage"]["source_units"]
