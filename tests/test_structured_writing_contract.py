"""Keep summaries concise and reject wrong-language structured prose offline."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_strict_quality import cards, draft, to_native

from litagent.agents.synthesis import SynthesisWorker
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import GroundedReport, grounded_messages, render_grounded_report
from litagent.review_structure import structure_sections


def summary(report, key):
    """Access one actual template section without depending on its index."""
    return next(s for s in report["sections"] if s["section_key"] == key)


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
def test_summary_contract_keeps_complete_sources_and_core_requirements(
    language, structure
):
    facts = cards()
    requirements = [
        {
            "requirement_id": "question",
            "kind": "question",
            "text": "Compare EViT and Evo-ViT under their actual conditions.",
        }
    ]
    messages = grounded_messages(
        facts, structure, language, requirements[0]["text"], coverage=requirements
    )
    packet = json.loads(messages[1]["content"])
    assert packet["report_language"] == language
    assert packet["writing_contract"] == "grounded-writing-v14"
    assert packet["summary_rules"]["max_sentence_units"] == 6
    assert packet["summary_rules"]["length"][language]["max"] > 0
    assert packet["coverage_requirements"] == requirements
    assert list(packet["cards"]) == list(facts)
    assert {s["source_quote"] for s in packet["sources"].values()} == {
        c["source_quote"] for c in facts.values()
    }
    body = render_grounded_report(
        json.dumps(draft(facts, language)), facts, structure, language
    )
    assert body.count("[E:a]") == 7
    assert [s["section_key"] for s in packet["section_contract"]] == [
        k for k, _ in structure_sections(structure, language) if k != "run_methodology"
    ]


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("key", ["abstract", "conclusion"])
@pytest.mark.parametrize("count", [6, 7, 30, 46])
def test_summary_capacity_is_aggregate_and_never_clips_input(language, key, count):
    facts = cards()
    report = draft(facts, language)
    block = summary(report, key)["blocks"][0]
    sentence = block["sentences"][0]
    # Each block respects the old 30-unit limit; splitting must not evade summary limits.
    summary(report, key)["blocks"] = [
        {"sentences": [copy.deepcopy(sentence)] * n}
        for n in ([3, 3] if count == 6 else [count] if count <= 30 else [23, 23])
    ]
    before = copy.deepcopy(report)
    if count == 6:
        body = render_grounded_report(
            json.dumps(report), facts, "academic_v2", language
        )
        assert body.count("[E:a]") == 12
    else:
        with pytest.raises(ValueError, match=f"summary_capacity_exceeded:{key}"):
            render_grounded_report(json.dumps(report), facts, "academic_v2", language)
    assert report == before
    assert (
        GroundedReport.model_json_schema()["$defs"]["GroundedBlock"]["properties"][
            "sentences"
        ]["maxItems"]
        == 30
    )


@pytest.mark.parametrize("key", ["abstract", "conclusion"])
@pytest.mark.parametrize("mode", ["list", "heading", "three_blocks"])
def test_summary_is_prose_in_at_most_two_paragraphs(key, mode):
    facts = cards()
    report = draft(facts)
    block = summary(report, key)["blocks"][0]
    if mode == "list":
        block["type"] = "list"
    elif mode == "heading":
        block["heading"] = "Paper inventory"
    else:
        summary(report, key)["blocks"] = [copy.deepcopy(block) for _ in range(3)]
    with pytest.raises(ValueError, match=f"summary_layout_invalid:{key}"):
        render_grounded_report(json.dumps(report), facts, "academic_v2", "en")


@pytest.mark.parametrize("language,max_length", [("en", 250), ("zh-CN", 1000)])
@pytest.mark.parametrize("key", ["abstract", "conclusion"])
@pytest.mark.parametrize("extra", [0, 1])
def test_summary_prose_length_boundary_does_not_expand_sentence_schema(
    language, max_length, key, extra
):
    facts = cards()
    report = draft(facts, language)
    sentence = summary(report, key)["blocks"][0]["sentences"][0]
    sentence["text"] = (
        " ".join(["word"] * (max_length + extra))
        if language == "en"
        else "文" * (max_length + extra)
    )
    if extra:
        with pytest.raises(ValueError, match=f"summary_length_exceeded:{key}"):
            render_grounded_report(json.dumps(report), facts, "academic_v2", language)
    else:
        assert render_grounded_report(
            json.dumps(report), facts, "academic_v2", language
        )


@pytest.mark.parametrize("key", ["abstract", "conclusion"])
def test_summary_scientific_card_must_be_used_in_scientific_body(key):
    facts = cards()
    report = draft(facts)
    other = "conclusion" if key == "abstract" else "abstract"
    summary(report, other)["blocks"][0]["sentences"][0].update(
        kind="organization",
        text="This review organizes the available evidence.",
        fact_card_ids=[],
    )
    for section in report["sections"]:
        if section["section_key"] not in {"abstract", "conclusion"}:
            sentence = section["blocks"][0]["sentences"][0]
            sentence.update(
                kind="organization",
                text="This review organizes the available evidence.",
            )
    with pytest.raises(ValueError, match=f"summary_card_not_in_body:{key}"):
        render_grounded_report(json.dumps(report), facts, "academic_v2", "en")
    summary(report, "methods")["blocks"][0]["sentences"][0].update(
        kind="scientific", text="EViT improves speed by 50% on DeiT-S."
    )
    assert render_grounded_report(json.dumps(report), facts, "academic_v2", "en")


def test_complete_body_still_respects_128_units_even_with_short_summaries():
    facts = cards()
    report = draft(facts)
    block = summary(report, "methods")["blocks"][0]
    block["sentences"] *= 30
    summary(report, "methods")["blocks"] = [copy.deepcopy(block) for _ in range(4)]
    summary(report, "introduction")["blocks"][0]["sentences"] *= 4
    with pytest.raises(ValueError, match="strict_writing_assertion_capacity_exceeded"):
        render_grounded_report(json.dumps(report), facts, "academic_v2", "en")


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_new_complete_packet_keeps_full_output_reservation_and_context_failure(
    language,
):
    from types import SimpleNamespace

    from litagent.eval.scoped import check_request_budget

    messages = grounded_messages(cards(), "academic_v2", language, "EViT")
    reservation = check_request_budget(
        SimpleNamespace(), messages, 20000, int(131072 * 0.9)
    )
    assert reservation > 20000
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_request_budget(SimpleNamespace(), messages, 20000, reservation - 1)


@pytest.mark.parametrize(
    "field", ["title", "heading", "scientific", "organization", "gap"]
)
@pytest.mark.parametrize(
    "text",
    [
        "EViT improves speed.",
        "文 The model has lower cost for training.",
        "The proposed model is evaluated on the same dataset and has lower cost for training.",
        "文 The proposed model is evaluated on the same dataset and has lower cost for training.",
        "文 `The proposed model is evaluated on the same dataset and has lower cost for training.`",
    ],
)
def test_chinese_heading_or_source_ids_cannot_mask_an_english_unit(field, text):
    facts = cards()
    report = draft(facts, "zh-CN")
    block = summary(report, "methods")["blocks"][0]
    if field == "title":
        report["title"] = text
    elif field == "heading":
        block["heading"] = text
    else:
        block["sentences"][0].update(kind=field, text=text)
    with pytest.raises(ValueError, match="strict_writing_language_mismatch:zh-CN"):
        render_grounded_report(json.dumps(report), facts, "academic_v2", "zh-CN")


def test_chinese_technical_names_numbers_and_conditions_remain_unchanged():
    facts = cards()
    report = draft(facts, "zh-CN")
    text = "EViT 在 DeiT-S 的给定实验条件下速度提高50%；该数字不是 FLOPs 降幅或精度百分点。"
    summary(report, "methods")["blocks"][0].update(
        heading="EViT 与 DeiT-S 条件分析",
        sentences=[{"text": text, "kind": "scientific", "fact_card_ids": list(facts)}],
    )
    body = render_grounded_report(json.dumps(report), facts, "academic_v2", "zh-CN")
    assert text + " [E:a]" in body
    assert "EViT 与 DeiT-S 条件分析" in body


@pytest.mark.parametrize("language", ["invalid", "zh", ""])
def test_unsupported_language_is_rejected_by_producer_and_consumer(language):
    facts = cards()
    with pytest.raises(ValueError, match="unsupported_report_language"):
        grounded_messages(facts, "academic_v2", language, "question")
    with pytest.raises(ValueError, match="unsupported_report_language"):
        render_grounded_report(json.dumps(draft(facts)), facts, "academic_v2", language)


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
@pytest.mark.parametrize("mode", ["valid", "language", "summary"])
async def test_actual_synthesis_rejects_invalid_body_after_one_bounded_request(
    policy, mode
):
    facts = cards()
    item = {
        "evidence_id": "a",
        "paper_id": "evit",
        "text": "EViT improves speed.",
        "supporting_text": next(iter(facts.values()))["source_quote"],
        "chunk_key": "abstract",
    }
    selection = EvidenceSelection(
        1,
        {"a": item},
        {k: ["a"] for k, _ in structure_sections("academic_v2")},
        {k: "offline" for k, _ in structure_sections("academic_v2")},
        0,
        1,
    )
    report = draft(facts, "zh-CN")
    if mode == "language":
        summary(report, "methods")["blocks"][0]["sentences"][0][
            "text"
        ] = "EViT improves speed by 50% on DeiT-S."
    elif mode == "summary":
        summary(report, "abstract")["blocks"][0]["sentences"] *= 7
    selector = AsyncMock()
    selector.select.return_value = selection
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(to_native(report, facts)), model="offline", usage={}
    )
    worker = SynthesisWorker(llm, evidence_selector=selector)
    task = SubTask(
        "synthesis",
        "offline",
        "synthesis",
        input_data={
            "query": "Compare EViT",
            "quality_policy": policy,
            "review_structure": "academic_v2",
            "report_language": "zh-CN",
        },
    )
    if mode == "valid":
        assert (await worker.execute(task))["draft"].count("[E:a]") == 7
    else:
        with pytest.raises(ValueError, match="strict_writing_"):
            await worker.execute(task)
    llm.chat.assert_awaited_once()
    assert llm.chat.await_args.kwargs["max_tokens"] == 20000
    assert llm.chat.await_args.kwargs["transport_max_attempts"] == 1
    packet = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert packet["report_language"] == "zh-CN"
    assert packet["summary_rules"]["max_sentence_units"] == 6
