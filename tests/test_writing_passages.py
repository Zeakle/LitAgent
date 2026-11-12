"""Verify host passage selection independently of scientific entailment."""

import copy
import json

import pytest
from test_strict_quality import draft, to_native

from litagent.quality import (
    GroundedWritingError,
    apply_fragment_repairs,
    build_fact_cards,
    grounded_messages,
    prepare_fragment_repair_request,
    prepare_grounded_request,
    render_grounded_report,
    render_grounded_report_traced,
)
from litagent.writing_passages import (
    FULL_VERSION,
    NOTICE_VERSION,
    build_passage_catalog,
    compile_passage_report,
    select_passage,
)


def facts(text="EViT improves speed by 50% on DeiT-S."):
    return build_fact_cards(
        {
            "a": {
                "evidence_id": "a",
                "paper_id": "evit",
                "text": text,
                "supporting_text": text,
                "chunk_key": "abstract",
            }
        }
    )


def document(cards, language="en"):
    result = to_native(draft(cards, language), cards)
    for key, section in result["sections"].items():
        for block in [section["first_block"], *section["additional_blocks"]]:
            for sentence in [block["first_sentence"], *block["additional_sentences"]]:
                sentence.pop("support_quotes")
                if key in {"abstract", "conclusion"}:
                    sentence.pop("fact_card_ids")
                    sentence.clear()
                    sentence["body_reference"] = {
                        "section_key": "methods",
                        "block_index": 0,
                        "sentence_index": 0,
                    }
                else:
                    sentence["support_passage_ids"] = ["r1.full"]
    return result


def test_complete_original_keeps_unicode_hyphens_whitespace_and_repeated_positions():
    text = "  A visual trans-former improves 2.22% accuracy.\nMethod A. Method A.  "
    cards = facts(text)
    catalog = build_passage_catalog(cards)
    source = catalog["sources"]["r1"]
    assert "".join(p["text"] for p in source["parts"]) == text
    cid = next(iter(cards))
    assert select_passage(cid, "r1.full", catalog)["source_quote"] == text
    part = select_passage(cid, "r1.p3", catalog)
    assert part["source_quote"] == " Method A."
    assert part["quote_start"] > text.index("Method A.")


@pytest.mark.parametrize(
    "pid",
    ["r2.p1", "r1.p0", "r1.p999", "r1.p2-r1.p1", "r1.p1-r2.p2", "r1.p1;DROP", "fake"],
)
def test_unknown_cross_source_and_reverse_ranges_fail(pid):
    cards = facts()
    catalog = build_passage_catalog(cards)
    with pytest.raises(ValueError):
        select_passage(next(iter(cards)), pid, catalog)


def test_native_compiler_derives_summary_cards_and_locates_duplicate_original():
    cards = facts("EViT improves speed. EViT improves speed.")
    payload = document(cards)
    for key, section in payload["sections"].items():
        if key not in {"abstract", "conclusion"}:
            section["first_block"]["first_sentence"]["support_passage_ids"] = ["r1.p2"]
    native = json.dumps(payload)
    compiled, spans = compile_passage_report(native, cards)
    normalized = json.loads(compiled)
    assert normalized["sections"]["abstract"]["first_block"]["first_sentence"][
        "fact_card_ids"
    ] == list(cards)
    assert (
        spans["abstract.blocks.0.sentences.0"] == spans["methods.blocks.0.sentences.0"]
    )
    report = render_grounded_report(
        native,
        cards,
        "academic_v2",
        "en",
        require_native=True,
        require_passage_ids=True,
    )
    assert report.count("[E:a]") == 7
    assert "EViT improves speed" in report


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "organization",
        "summary",
        "new_quote",
        "new_card",
        "duplicate_card",
        "fake_offset",
        "wrong_source",
    ],
)
def test_model_cannot_supply_new_summary_material_or_host_positions(mutation):
    cards = facts()
    payload = document(cards)
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    body = payload["sections"]["methods"]["first_block"]["first_sentence"]
    if mutation == "missing":
        summary["body_reference"]["block_index"] = 9
    elif mutation == "organization":
        body["kind"] = "organization"
    elif mutation == "summary":
        summary["body_reference"]["section_key"] = "conclusion"
    elif mutation == "new_quote":
        summary["support_quotes"] = ["invented"]
    elif mutation == "new_card":
        summary["fact_card_ids"] = list(cards)
    elif mutation == "duplicate_card":
        body["fact_card_ids"] *= 2
        body["support_passage_ids"] *= 2
    elif mutation == "fake_offset":
        body["quote_start"] = 0
    elif mutation == "wrong_source":
        body["support_passage_ids"] = ["r2.full"]
    with pytest.raises(GroundedWritingError):
        render_grounded_report(
            json.dumps(payload),
            cards,
            "academic_v2",
            "en",
            require_native=True,
            require_passage_ids=True,
        )


def test_request_schema_matches_body_and_summary_without_repeating_source_text():
    cards = facts()
    messages = grounded_messages(
        cards, "academic_v2", "en", "EViT", use_passage_ids=True
    )
    prepared, native = prepare_grounded_request(messages)
    packet = json.loads(prepared[1]["content"])
    parts = packet["sources"]["r1"]["parts"]
    assert "".join(parts) == next(iter(cards.values()))["source_quote"]
    assert len(parts) == 1
    assert "source_quote" not in packet["sources"]["r1"]
    assert packet["writing_contract"] == "grounded-writing-v18"
    assert (
        "support_passage_ids"
        in native["json_schema"]["schema"]["$defs"]["s0"]["properties"]
    )
    assert (
        "body_reference" in native["json_schema"]["schema"]["$defs"]["s5"]["properties"]
    )
    assert "support_quotes:" not in prepared[0]["content"]
    assert (
        "Every sentence has text, kind and fact_card_ids." not in prepared[0]["content"]
    )


def test_old_quote_output_cannot_silently_fall_back_in_new_mode():
    cards = facts()
    old = json.dumps(to_native(draft(cards), cards))
    with pytest.raises(GroundedWritingError):
        render_grounded_report(
            old,
            cards,
            "academic_v2",
            "en",
            require_native=True,
            require_passage_ids=True,
        )
    assert render_grounded_report(
        old, cards, "academic_v2", "en", require_native=True, require_source_quotes=True
    )


def test_fragment_repair_uses_catalog_and_preserves_body_binding_for_summary():
    from test_reference_occurrence_policy import bound

    from litagent.runner import LitAgent

    cards = facts()
    proof = {}
    original = render_grounded_report_traced(
        json.dumps(document(cards)),
        cards,
        "academic_v2",
        "en",
        require_native=True,
        require_passage_ids=True,
        bindings=proof,
    )
    evaluation = bound(original)
    targets = LitAgent._strict_repair_targets(evaluation, original, "academic_v2", "en")
    assert targets
    response = {
        "patches": {
            t["source_fragment_id"]: {
                "first_sentence": {
                    "text": "EViT improves speed by 50% on DeiT-S.",
                    "fact_card_ids": list(cards),
                    "support_passage_ids": ["r1.full"],
                },
                "additional_sentences": [],
            }
            for t in targets
        }
    }
    updated = {}
    candidate = apply_fragment_repairs(
        json.dumps(response),
        original,
        targets,
        cards,
        "academic_v2",
        "en",
        require_passage_ids=True,
        body_bindings=proof,
        bindings=updated,
    )
    assert candidate == original
    assert updated["report_sha256"] == proof["report_sha256"]
    messages = grounded_messages(
        cards, "academic_v2", "en", "EViT", draft=original, use_passage_ids=True
    )
    prepared, schema = prepare_fragment_repair_request(
        messages, targets, body_bindings=proof
    )
    assert "support_passage_ids" in str(schema) and "support_quotes" not in str(schema)


def test_v2_all_revision_messages_keep_full_source_contract():
    from litagent.agents.synthesis import SynthesisWorker

    cards = facts()
    worker = SynthesisWorker(None)
    for method, args in [
        (worker.revise, ("Draft", "Feedback", {})),
        (worker.rewrite_with_evidence, ("Draft", "[]", "[]")),
    ]:
        messages = method(
            *args,
            quality_policy="strict_auto_v2",
            fact_cards=cards,
            review_structure="academic_v2",
        )
        assert json.loads(messages[1]["content"])["writing_contract"] == NOTICE_VERSION


def test_malformed_card_identity_is_a_bound_protocol_error():
    cards = facts()
    payload = document(cards)
    sentence = payload["sections"]["methods"]["first_block"]["first_sentence"]
    sentence["fact_card_ids"] = [{"invented": "card"}]
    with pytest.raises(GroundedWritingError) as error:
        render_grounded_report(
            json.dumps(payload), cards, "academic_v2", "en", require_passage_ids=True
        )
    assert (
        error.value.diagnostics[0]["reason_code"]
        == "strict_writing_passage_binding_invalid"
    )
    assert error.value.diagnostics[0]["raw_json_path"].startswith("sections.methods.")


@pytest.mark.asyncio
async def test_valid_passage_location_does_not_approve_wrong_average_or_summary():
    from test_batched_evaluation import BatchLLM

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import digest

    source = (
        "EViT achieves an average speed improvement of 25% and a peak speed "
        "improvement of 50%."
    )
    cards = facts(source)
    payload = document(cards)
    for key, section in payload["sections"].items():
        if key in {"abstract", "conclusion"}:
            continue
        section["first_block"]["first_sentence"][
            "text"
        ] = "EViT achieves an average speed improvement of 50%."
    proof = {}
    report = render_grounded_report_traced(
        json.dumps(payload),
        cards,
        "academic_v2",
        "en",
        require_native=True,
        require_passage_ids=True,
        bindings=proof,
    )
    assert all(
        row["verification"] == "source_location_only" for row in proof["locations"]
    )
    context = {
        "query": "EViT average speed improvement",
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
            {
                "requirement_id": "question",
                "kind": "question",
                "text": "EViT average speed improvement",
            }
        ],
        "structure_checks": {"status": "not_applicable"},
        "method_checks": {"status": "not_applicable"},
    }
    result = await evaluate_scoped(
        llm=BatchLLM(), survey=report, context=context, config=EvalConfig()
    )
    assert result["score"] is None
    assert result["factual_claim_count"] == 7
    assert not result["coverage"]["complete"]
    assert all(row["status"] != "supported" for row in result["assertions"])
