"""Do not hide field-wide scientific gaps or duplicate derived summary findings."""

import copy

import pytest
from test_host_full_source_binding import full_document, render_full
from test_writing_passages import facts

from litagent.quality import GroundedWritingError, grounded_messages


@pytest.mark.parametrize(
    "text",
    [
        "现有研究对医学数据上的性能仍缺乏系统的对比分析。",
        "这些方法的长期泛化能力尚未得到充分验证。",
        "Current research lacks systematic comparisons on medical data.",
        "Long-term generalization has not been sufficiently validated.",
    ],
)
@pytest.mark.parametrize("kind", ["gap", "organization"])
def test_uncited_field_wide_science_is_rejected_with_actual_location(text, kind):
    cards = facts()
    payload = full_document(cards)
    unit = payload["sections"]["open_problems"]["first_block"]["first_sentence"]
    unit.update(text=text, kind=kind, fact_card_ids=[])
    with pytest.raises(GroundedWritingError) as rejected:
        render_full(payload, cards)
    assert any(
        d["location"] == "open_problems.blocks.0.sentences.0"
        and d["reason_code"] == "strict_writing_uncited_scientific_gap"
        for d in rejected.value.diagnostics
    )


def test_bounded_material_gap_and_actual_cited_science_are_not_deleted():
    cards = facts()
    payload = full_document(cards)
    unit = payload["sections"]["open_problems"]["first_block"]["first_sentence"]
    unit.update(
        text="The supplied material does not answer this review question.",
        kind="gap",
        fact_card_ids=[],
    )
    assert unit["text"].rstrip(".") in render_full(payload, cards)
    unit.update(
        text="Current research lacks systematic comparisons.",
        kind="scientific",
        fact_card_ids=list(cards),
    )
    # Compilation only preserves a located claim; Evaluation must still prove it.
    assert unit["text"].rstrip(".") in render_full(payload, cards)


def test_distinct_body_locations_cannot_duplicate_a_summary_sentence():
    cards = facts()
    payload = full_document(cards)
    body = payload["sections"]["methods"]["first_block"]
    body["additional_sentences"] = [copy.deepcopy(body["first_sentence"])]
    summary = payload["sections"]["conclusion"]["first_block"]
    duplicate = copy.deepcopy(summary["first_sentence"])
    duplicate["body_reference"]["sentence_index"] = 1
    summary["additional_sentences"] = [duplicate]
    with pytest.raises(GroundedWritingError) as rejected:
        render_full(payload, cards)
    diagnostic = next(
        d
        for d in rejected.value.diagnostics
        if d["location"] == "conclusion.blocks.0.sentences.1"
    )
    assert diagnostic["reason_code"] == "strict_writing_summary_content_duplicate"
    assert diagnostic["actual_body_reference"]["sentence_index"] == 1
    body["additional_sentences"][0]["text"] = "EViT improves speed on DeiT-S."
    assert "EViT improves speed on DeiT-S [E:a]" in render_full(payload, cards)


def test_cross_summary_reuse_remains_valid_and_prompt_forbids_gap_exemption():
    cards = facts()
    payload = full_document(cards)
    assert render_full(payload, cards).count("[E:a]") == 7
    prompt = grounded_messages(
        cards, "academic_v2", "en", "EViT", use_full_sources=True
    )[0]["content"]
    assert "field-wide" in prompt and "same text" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["empty", "organization"])
async def test_evaluation_cannot_exclude_uncited_field_wide_gap(mode):
    from test_batched_evaluation import materials
    from test_inventory_exclusion import OmittingLLM

    from litagent.config import EvalConfig
    from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped

    _, context = materials(1)
    context["quality_policy"] = "strict_auto_v2"
    survey = "ViT uses patches. [E:a0] Current research lacks systematic comparisons."
    llm = OmittingLLM(mode)
    with pytest.raises(ScopedEvaluationError, match="evaluation_extraction_incomplete"):
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert len(llm.calls) == 1 and llm.batches == 0


def test_global_scope_clarification_is_versioned_and_budgeted_as_sent():
    from litagent.eval.batched import (
        GLOBAL_CONSISTENCY_CONTRACT,
        GLOBAL_PROMPT,
        global_prompt,
        request_size,
    )

    assert global_prompt({}) == GLOBAL_PROMPT
    payload = {"global_consistency_contract": GLOBAL_CONSISTENCY_CONTRACT}
    prompt = global_prompt(payload)
    assert "source cited only once is valid" in prompt
    assert "twice" in prompt and "concrete problems" in prompt
    assert request_size(prompt, payload, 16384) > request_size(
        GLOBAL_PROMPT, payload, 16384
    )
