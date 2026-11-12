"""Check explicit summary selectors against actual scientific body units."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_writing_passages import document, facts

from litagent.quality import (
    GroundedWritingError,
    grounded_messages,
    prepare_grounded_request,
    render_grounded_report,
)


def candidate(cards):
    """Keep native body content while replacing the superseded string selector."""
    payload = document(cards)
    for key in ("abstract", "conclusion"):
        for block in [
            payload["sections"][key]["first_block"],
            *payload["sections"][key]["additional_blocks"],
        ]:
            for sentence in [block["first_sentence"], *block["additional_sentences"]]:
                sentence.pop("body_sentence_id", None)
                sentence["body_reference"] = {
                    "section_key": "methods",
                    "block_index": 0,
                    "sentence_index": 0,
                }
    return payload


def render(payload, cards):
    """Use the production compiler and all ordinary report contracts."""
    return render_grounded_report(
        json.dumps(payload),
        cards,
        "academic_v2",
        "en",
        require_native=True,
        require_passage_ids=True,
    )


@pytest.mark.parametrize("block_index,sentence_index", [(0, 0), (0, 1), (1, 0)])
def test_summary_resolves_actual_first_and_additional_body_units(
    block_index, sentence_index
):
    cards = facts()
    payload = candidate(cards)
    body = payload["sections"]["methods"]
    body["first_block"]["additional_sentences"] = [
        copy.deepcopy(body["first_block"]["first_sentence"])
    ]
    body["additional_blocks"] = [copy.deepcopy(body["first_block"])]
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    summary["body_reference"].update(
        block_index=block_index, sentence_index=sentence_index
    )
    assert render(payload, cards).count("[E:a]") == 10


@pytest.mark.parametrize(
    "reference",
    [
        None,
        "methods.first_sentence",
        {},
        {"section_key": "abstract", "block_index": 0, "sentence_index": 0},
        {"section_key": "fake", "block_index": 0, "sentence_index": 0},
        {"section_key": "methods", "block_index": -1, "sentence_index": 0},
        {"section_key": "methods", "block_index": False, "sentence_index": 0},
        {"section_key": "methods", "block_index": 0, "sentence_index": True},
        {"section_key": "methods", "block_index": 0, "sentence_index": 0.0},
        {"section_key": "methods", "block_index": 0, "sentence_index": "0"},
        {"section_key": "methods", "block_index": 9, "sentence_index": 0},
        {"section_key": "methods", "block_index": 0, "sentence_index": 9},
        {
            "section_key": "methods",
            "block_index": 0,
            "sentence_index": 0,
            "source_quote": "forged",
        },
    ],
)
def test_bad_summary_selection_is_bound_to_raw_candidate_without_fallback(reference):
    cards = facts()
    payload = candidate(cards)
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    summary["body_reference"] = reference
    raw = json.dumps(payload)
    with pytest.raises(GroundedWritingError) as error:
        render_grounded_report(
            raw, cards, "academic_v2", "en", require_passage_ids=True
        )
    assert error.value.candidate_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    diagnostic = error.value.diagnostics[0]
    assert diagnostic["actual_body_reference"] == reference
    assert diagnostic["location"] == "abstract.blocks.0.sentences.0"


@pytest.mark.parametrize("kind", ["organization", "gap"])
def test_model_cannot_force_scientific_kind_for_an_uncited_body_unit(kind):
    cards = facts()
    payload = candidate(cards)
    body = payload["sections"]["methods"]["first_block"]["first_sentence"]
    body.update(
        kind=kind,
        fact_card_ids=[],
        support_passage_ids=[],
        text="Evidence is unavailable.",
    )
    payload["sections"]["abstract"]["first_block"]["first_sentence"][
        "kind"
    ] = "scientific"
    with pytest.raises(GroundedWritingError):
        render(payload, cards)


@pytest.mark.parametrize("kind", ["organization", "gap"])
def test_selector_derives_body_kind_and_does_not_accept_an_empty_target(kind):
    cards = facts()
    payload = candidate(cards)
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    summary["kind"] = kind
    with pytest.raises(GroundedWritingError):
        render(payload, cards)
    del summary["kind"]
    body = payload["sections"]["methods"]["first_block"]["first_sentence"]
    body.update(
        kind=kind,
        fact_card_ids=[],
        support_passage_ids=[],
        text="Evidence is unavailable.",
    )
    assert render(payload, cards).count("[E:a]") == 4
    summary["body_reference"] = {
        "section_key": "",
        "block_index": 0,
        "sentence_index": 0,
    }
    with pytest.raises(GroundedWritingError):
        render(payload, cards)


def test_provider_schema_exposes_body_sections_and_integer_indices():
    messages, response = prepare_grounded_request(
        grounded_messages(facts(), "academic_v2", "en", "EViT", use_passage_ids=True)
    )
    schema = response["json_schema"]["schema"]["$defs"]["s5"]["properties"]
    ref = schema["body_reference"]
    assert set(ref["required"]) == {"section_key", "block_index", "sentence_index"}
    assert set(ref["properties"]["section_key"]["enum"]) == {
        "introduction",
        "taxonomy",
        "methods",
        "experiments",
        "open_problems",
    }
    assert ref["properties"]["block_index"]["type"] == "integer"
    assert ref["properties"]["sentence_index"]["type"] == "integer"
    assert "body_sentence_id" not in str(response)
    assert "body_reference" in messages[0]["content"]


def test_old_free_string_output_is_not_reinterpreted_as_a_corrected_body_reference():
    cards = facts()
    payload = candidate(cards)
    summary = payload["sections"]["abstract"]["first_block"]["first_sentence"]
    summary.pop("body_reference")
    summary["body_sentence_id"] = "methods.first_sentence"
    with pytest.raises(GroundedWritingError):
        render(payload, cards)


def test_all_seven_real_v15_failures_keep_original_selectors_and_diagnostics():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/summary-v15-real-failure.json").read_text()
    )
    cards = facts()
    payload = candidate(cards)
    for key in ("abstract", "conclusion"):
        units = [row["unit"] for row in fixture["units"] if row["section_key"] == key]
        payload["sections"][key]["first_block"].update(
            first_sentence=units[0], additional_sentences=units[1:]
        )
    with pytest.raises(GroundedWritingError) as error:
        render(payload, cards)
    diagnostics = error.value.diagnostics
    assert len(diagnostics) == 7
    assert [d["actual_body_sentence_id"] for d in diagnostics] == [
        row["unit"]["body_sentence_id"] for row in fixture["units"]
    ]
    assert all(d["actual_body_reference"] is None for d in diagnostics)


@pytest.mark.asyncio
async def test_valid_summary_reference_cannot_borrow_a_second_methods_result():
    from test_batched_evaluation import BatchLLM

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import digest

    cards = facts()
    source = next(iter(cards.values()))["source_quote"]
    payload = candidate(cards)
    payload["sections"]["methods"]["first_block"]["first_sentence"][
        "text"
    ] = "EViT improves speed by 50% on DeiT-S, while Evo-ViT improves speed by 50%."
    report = render(payload, cards)
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
    summary = [row for row in result["assertions"] if "Evo-ViT" in row["claim_text"]]
    assert summary and all(row["status"] != "supported" for row in summary)
    assert result["score"] != 1.0
