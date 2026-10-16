"""Derive full original support without confusing location with entailment."""

import copy
import hashlib
import json

import pytest
from test_writing_passages import document, facts

from litagent.quality import (
    GroundedWritingError,
    grounded_messages,
    prepare_grounded_request,
    render_grounded_report,
)
from litagent.writing_passages import FULL_VERSION, compile_passage_report


def full_document(cards, language="en"):
    """Represent the new producer without model-written source identifiers."""
    result = document(cards, language)
    for key, section in result["sections"].items():
        if key in {"abstract", "conclusion"}:
            continue
        for block in [section["first_block"], *section["additional_blocks"]]:
            for sentence in [block["first_sentence"], *block["additional_sentences"]]:
                sentence.pop("support_passage_ids")
    return result


def render_full(payload, cards, **kwargs):
    return render_grounded_report(
        json.dumps(payload),
        cards,
        "academic_v2",
        "en",
        require_native=True,
        require_full_sources=True,
        **kwargs,
    )


def test_host_binds_complete_original_and_copies_summary_without_new_prose():
    text = "  EViT improves speed by 50% on DeiT-S.\nA trans-former. A trans-former.  "
    cards = facts(text)
    payload = full_document(cards)
    compiled, spans = compile_passage_report(
        json.dumps(payload), cards, binding_mode="full_sources"
    )
    proof = spans["methods.blocks.0.sentences.0"][0]
    assert proof["source_quote"] == text
    assert (proof["quote_start"], proof["quote_end"]) == (0, len(text))
    assert proof["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert proof["verification"] == "source_location_only"
    normalized = json.loads(compiled)
    assert (
        normalized["sections"]["abstract"]["first_block"]["first_sentence"]
        == normalized["sections"]["methods"]["first_block"]["first_sentence"]
    )
    assert render_full(payload, cards).count("[E:a]") == 7


@pytest.mark.parametrize(
    "field", ["support_passage_ids", "support_quotes", "source_ref", "source_start"]
)
def test_model_cannot_override_host_source_identity_or_location(field):
    cards = facts()
    payload = full_document(cards)
    payload["sections"]["methods"]["first_block"]["first_sentence"][field] = "invented"
    with pytest.raises(GroundedWritingError):
        render_full(payload, cards)


@pytest.mark.parametrize(
    "mutation", ["unknown", "duplicate", "empty_science", "gap_with_card"]
)
def test_bad_card_sets_do_not_gain_support_by_full_source_binding(mutation):
    cards = facts()
    payload = full_document(cards)
    unit = payload["sections"]["methods"]["first_block"]["first_sentence"]
    if mutation == "unknown":
        unit["fact_card_ids"] = ["f:unknown"]
    elif mutation == "duplicate":
        unit["fact_card_ids"] *= 2
    elif mutation == "empty_science":
        unit["fact_card_ids"] = []
    else:
        unit["kind"] = "gap"
    with pytest.raises((GroundedWritingError, ValueError)):
        render_full(payload, cards)


def test_new_provider_schema_requires_only_prose_kind_and_scoped_card_ids():
    cards = facts()
    messages = grounded_messages(
        cards, "academic_v2", "en", "EViT", use_full_sources=True
    )
    prepared, schema = prepare_grounded_request(messages)
    packet = json.loads(prepared[1]["content"])
    assert packet["writing_contract"] == FULL_VERSION
    assert (
        next(iter(packet["sources"].values()))["source_quote"]
        == next(iter(cards.values()))["source_quote"]
    )
    defs = schema["json_schema"]["schema"]["$defs"]
    bodies = [v for v in defs.values() if "kind" in v.get("properties", {})]
    assert bodies and all(
        set(v["properties"]) == {"kind", "fact_card_ids", "text"} for v in bodies
    )
    assert "support_passage_ids" not in str(schema)
    assert "support_passage_ids" not in prepared[0]["content"]
    assert schema["json_schema"]["name"] == FULL_VERSION.replace(
        "grounded-writing-", "grounded_report_"
    )


def test_modes_are_explicit_and_never_silently_downgrade():
    cards = facts()
    with pytest.raises(ValueError):
        grounded_messages(
            cards,
            "academic_v2",
            "en",
            "EViT",
            use_full_sources=True,
            use_passage_ids=True,
        )
    with pytest.raises(ValueError):
        render_full(full_document(cards), cards, require_passage_ids=True)
    with pytest.raises(GroundedWritingError):
        render_full(document(cards), cards)


@pytest.mark.asyncio
async def test_wrong_scientific_value_is_still_rejected_by_complete_evaluation():
    from test_batched_evaluation import BatchLLM

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import digest

    source = "EViT reduces average FLOPs by 25%; peak reduction reaches 50%."
    cards = facts(source)
    payload = full_document(cards)
    for key, section in payload["sections"].items():
        if key not in {"abstract", "conclusion"}:
            section["first_block"]["first_sentence"][
                "text"
            ] = "EViT reduces average FLOPs by 50%."
    report = render_full(payload, cards)
    result = await evaluate_scoped(
        llm=BatchLLM(),
        survey=report,
        config=EvalConfig(),
        context={
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
        },
    )
    assert result["score"] != 1.0
    assert result["coverage"]["processed_units"] == result["coverage"]["source_units"]
    assert all(c["status"] != "supported" for c in result["assertions"])


def test_multi_card_units_bind_each_actual_original_once():
    from litagent.quality import build_fact_cards

    cards = build_fact_cards(
        {
            "a": {
                "evidence_id": "a",
                "paper_id": "evit",
                "text": "EViT improves speed by 50% on DeiT-S.",
                "supporting_text": "EViT improves speed by 50% on DeiT-S.",
                "chunk_key": "abstract",
            },
            "b": {
                "evidence_id": "b",
                "paper_id": "evo",
                "text": "Evo-ViT reduces FLOPs by 25%.",
                "supporting_text": "Evo-ViT reduces FLOPs by 25%.",
                "chunk_key": "abstract",
            },
        }
    )
    payload = full_document(cards)
    unit = payload["sections"]["methods"]["first_block"]["first_sentence"]
    unit["fact_card_ids"] = list(cards)
    _, spans = compile_passage_report(
        json.dumps(payload), cards, binding_mode="full_sources"
    )
    located = spans["methods.blocks.0.sentences.0"]
    assert len(located) == 2
    for row in located:
        assert row["source_quote"] == cards[row["fact_card_id"]]["source_quote"]
    assert located[0]["source_sha256"] != located[1]["source_sha256"]


def test_fragment_repair_and_its_schema_use_the_same_host_full_sources():
    from test_fragment_repair import bound

    from litagent.quality import (
        apply_fragment_repairs,
        prepare_fragment_repair_request,
        render_grounded_report_traced,
    )
    from litagent.runner import LitAgent

    cards = facts()
    proof = {}
    original = render_grounded_report_traced(
        json.dumps(full_document(cards)),
        cards,
        "academic_v2",
        "en",
        require_full_sources=True,
        bindings=proof,
    )
    targets = LitAgent._strict_repair_targets(
        bound(original), original, "academic_v2", "en"
    )
    assert targets
    patches = {
        "patches": {
            t["source_fragment_id"]: {
                "first_sentence": {
                    "text": "EViT improves speed by 50% on DeiT-S.",
                    "fact_card_ids": list(cards),
                },
                "additional_sentences": [],
            }
            for t in targets
        }
    }
    updated = {}
    candidate = apply_fragment_repairs(
        json.dumps(patches),
        original,
        targets,
        cards,
        "academic_v2",
        "en",
        require_full_sources=True,
        body_bindings=proof,
        bindings=updated,
    )
    assert candidate == original and updated["report_sha256"] == proof["report_sha256"]
    messages = grounded_messages(
        cards, "academic_v2", "en", "EViT", draft=original, use_full_sources=True
    )
    prepared, schema = prepare_fragment_repair_request(
        messages, targets, body_bindings=proof
    )
    assert "support_passage_ids" not in str(schema) and "support_quotes" not in str(
        schema
    )
    assert schema["json_schema"]["name"] == "fragment_repair_v3"
    assert "Choose fact_card_ids only" in prepared[0]["content"]
    changed = copy.deepcopy(patches)
    changed["patches"][targets[0]["source_fragment_id"]]["first_sentence"][
        "source_quote"
    ] = "another paper"
    with pytest.raises(ValueError):
        apply_fragment_repairs(
            json.dumps(changed),
            original,
            targets,
            cards,
            "academic_v2",
            "en",
            require_full_sources=True,
            body_bindings=proof,
        )
