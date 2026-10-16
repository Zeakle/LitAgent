"""Repair only host-located native units and retain the original failed candidate."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_host_full_source_binding import full_document
from test_writing_passages import facts

from litagent.quality import (
    GroundedWritingError,
    apply_native_repairs,
    grounded_messages,
    prepare_grounded_request,
    prepare_native_repair_request,
    render_grounded_report,
)


def fixture():
    """Load only public frozen material; this never reads credentials or live state."""
    return json.loads(
        (Path(__file__).parent / "fixtures/native-gap-repair-real.json").read_text()
    )


def prepare(value):
    return prepare_native_repair_request(
        value["messages"],
        value["candidate"],
        value["diagnostics"],
        cards=value["cards"],
        structure="academic_v2",
        language="zh-CN",
        evidence_selection=value["selection"],
    )


def replacement(value, binding):
    """Choose an actual allowed source without asserting that compilation proves support."""
    allowed = binding["targets"]["u0"]["allowed_fact_card_ids"]
    cid = next(
        c for c in allowed if value["cards"][c]["paper_id"] == "arxiv:2408.05952"
    )
    return {
        "can_repair": True,
        "patches": {
            "u0": {
                "kind": "scientific",
                "text": "视觉Transformer模型规模较大，依赖大量训练数据。",
                "fact_card_ids": [cid],
            }
        },
    }


def test_real_failed_candidate_changes_only_one_unit_and_resolves_summary_dependency():
    value = fixture()
    messages, schema, binding = prepare(value)
    assert (
        hashlib.sha256(value["candidate"].encode()).hexdigest()
        == value["candidate_sha256"]
    )
    packet = json.loads(messages[1]["content"])
    original_packet = json.loads(value["messages"][1]["content"])
    assert packet["sources"] == original_packet["sources"]
    assert packet["cards"] == original_packet["cards"]
    assert schema["json_schema"]["schema"]["additionalProperties"] is False
    patched, proof = apply_native_repairs(
        value["candidate"],
        json.dumps(replacement(value, binding)),
        binding,
        cards=value["cards"],
    )
    before, after = json.loads(value["candidate"]), json.loads(patched)
    expected = copy.deepcopy(before)
    expected["sections"]["open_problems"]["first_block"]["first_sentence"] = (
        replacement(value, binding)["patches"]["u0"]
    )
    assert after == expected
    assert after["sections"]["conclusion"] == before["sections"]["conclusion"]
    assert proof["modified_paths"] == [
        "sections.open_problems.first_block.first_sentence"
    ]
    rendered = render_grounded_report(
        patched,
        value["cards"],
        "academic_v2",
        "zh-CN",
        require_full_sources=True,
        evidence_selection=value["selection"],
    )
    assert "研究对视觉Transformer数据高效训练方法的泛化性评估仍存在不足" not in rendered
    assert rendered.count("视觉Transformer模型规模较大，依赖大量训练数据") == 2


@pytest.mark.parametrize(
    "mutation",
    [
        "empty_cards",
        "unknown_card",
        "wrong_scope",
        "duplicate_card",
        "gap_kind",
        "empty_text",
        "extra_unit",
        "extra_field",
        "false",
        "binding",
        "source",
    ],
)
def test_bad_repair_never_modifies_or_promotes_the_original(mutation):
    value = fixture()
    _, _, binding = prepare(value)
    response = replacement(value, binding)
    original = copy.deepcopy(value)
    unit = response["patches"]["u0"]
    if mutation == "empty_cards":
        unit["fact_card_ids"] = []
    elif mutation == "unknown_card":
        unit["fact_card_ids"] = ["f:unknown"]
    elif mutation == "wrong_scope":
        unit["fact_card_ids"] = [
            next(
                cid
                for cid in value["cards"]
                if cid not in binding["targets"]["u0"]["allowed_fact_card_ids"]
            )
        ]
    elif mutation == "duplicate_card":
        unit["fact_card_ids"] *= 2
    elif mutation == "gap_kind":
        unit["kind"] = "gap"
    elif mutation == "empty_text":
        unit["text"] = ""
    elif mutation == "extra_unit":
        response["patches"]["u1"] = copy.deepcopy(unit)
    elif mutation == "extra_field":
        unit["body_reference"] = {}
    elif mutation == "false":
        response["can_repair"] = False
    elif mutation == "binding":
        binding["candidate_sha256"] = "0" * 64
    else:
        value["cards"][unit["fact_card_ids"][0]]["source_quote"] += " changed"
    with pytest.raises(ValueError):
        apply_native_repairs(
            value["candidate"], json.dumps(response), binding, cards=value["cards"]
        )
    assert value["candidate"] == original["candidate"]


@pytest.mark.parametrize(
    "mutation",
    [
        "position",
        "text",
        "code",
        "independent_summary",
        "source_packet",
        "duplicate_json",
    ],
)
def test_forged_diagnostics_or_unrelated_errors_are_not_repairable(mutation):
    value = fixture()
    if mutation == "position":
        value["diagnostics"][0][
            "raw_json_path"
        ] = "sections.introduction.first_block.first_sentence"
    elif mutation == "text":
        value["diagnostics"][0]["candidate_text"] = "forged"
    elif mutation == "code":
        value["diagnostics"][0][
            "reason_code"
        ] = "strict_writing_passage_binding_invalid"
    elif mutation == "independent_summary":
        raw = json.loads(value["candidate"])
        raw["sections"]["conclusion"]["first_block"]["first_sentence"][
            "body_reference"
        ]["section_key"] = "missing"
        value["candidate"] = json.dumps(raw)
    elif mutation == "source_packet":
        packet = json.loads(value["messages"][1]["content"])
        next(iter(packet["sources"].values()))["source_quote"] = "invented"
        value["messages"][1]["content"] = json.dumps(packet)
    else:
        value["candidate"] = value["candidate"].replace(
            '"title":', '"title":"forged","title":', 1
        )
    with pytest.raises(ValueError):
        prepare(value)


@pytest.mark.asyncio
async def test_a_valid_patch_cannot_turn_wrong_science_into_evaluation_success():
    from test_batched_evaluation import BatchLLM, materials

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import build_fact_cards, digest

    cards = facts()
    raw = full_document(cards)
    raw["sections"]["open_problems"]["first_block"]["first_sentence"].update(
        text="Current research lacks comparisons.", kind="gap", fact_card_ids=[]
    )
    candidate = json.dumps(raw)
    with pytest.raises(GroundedWritingError) as failure:
        render_grounded_report(
            candidate, cards, "academic_v2", "en", require_full_sources=True
        )
    messages, _ = prepare_grounded_request(
        grounded_messages(cards, "academic_v2", "en", "EViT", use_full_sources=True)
    )
    _, _, binding = prepare_native_repair_request(
        messages,
        candidate,
        failure.value.diagnostics,
        cards=cards,
        structure="academic_v2",
        language="en",
    )
    response = {
        "can_repair": True,
        "patches": {
            "u0": {
                "kind": "scientific",
                "text": "EViT improves speed by 99% on DeiT-S.",
                "fact_card_ids": list(cards),
            }
        },
    }
    patched, _ = apply_native_repairs(
        candidate, json.dumps(response), binding, cards=cards
    )
    body = render_grounded_report(
        patched, cards, "academic_v2", "en", require_full_sources=True
    )
    _, context = materials(1)
    context["evidence"] = {
        "a": {
            "paper_id": "evit",
            "text": "unverified",
            "supporting_text": next(iter(cards.values()))["source_quote"],
        }
    }
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    context["quality_policy"] = "strict_auto_v2"
    result = await evaluate_scoped(
        llm=BatchLLM(), survey=body, context=context, config=EvalConfig()
    )
    assert result["score"] != 1.0
    assert any(
        c["status"] != "supported" and "99%" in c["claim_text"]
        for c in result["assertions"]
    )
