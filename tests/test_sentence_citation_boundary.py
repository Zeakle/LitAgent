"""Keep native scientific units and their citations independently locatable."""

import json
from unittest.mock import patch

import pytest
from test_batched_evaluation import BatchLLM, materials
from test_strict_quality import draft, to_native

from litagent.config import EvalConfig, load_config
from litagent.contracts import build_config_summary
from litagent.eval.scoped import build_source_fragments, evaluate_scoped
from litagent.quality import build_fact_cards, digest, render_grounded_report
from litagent.report_render import build_report_source_units


def render_pair(first, second, language="en", *, native=True, combined=False):
    """Render two model units through the complete template validation path."""
    _, context = materials(2)
    facts = context["fact_cards"]
    ids = list(facts)
    report = draft(facts, language)
    report["sections"][1]["blocks"][0]["sentences"] = [
        {
            "kind": "scientific",
            "text": first,
            "fact_card_ids": ids if combined else ids[:1],
        },
        {"kind": "scientific", "text": second, "fact_card_ids": ids[1:]},
    ]
    body = render_grounded_report(
        json.dumps(to_native(report) if native else report),
        facts,
        "academic_v2",
        language,
        require_native=native,
    )
    unit = next(
        u
        for u in build_report_source_units(body)
        if first.rstrip(".!?。！？") in u.text
    )
    return body, unit, build_source_fragments(unit)


@pytest.mark.parametrize(
    "language,first,second,terminator",
    [
        ("en", "ViT uses image patches", "DeiT uses distillation", "."),
        ("zh-CN", "ViT使用图像块", "DeiT使用蒸馏", "。"),
    ],
)
@pytest.mark.parametrize("ending", ["", ".", "?", "!", "。", "？", "！"])
def test_native_sentence_terminators_preserve_distinct_actual_references(
    language, first, second, terminator, ending
):
    body, unit, fragments = render_pair(first + ending, second, language)
    assert [f["citation_aliases"] for f in fragments] == [["a0"], ["a1"]]
    assert (
        unit.text.rstrip()
        == f"{first} [E:a0]{ending or terminator} {second} [E:a1]{terminator}"
    )
    for fragment in fragments:
        assert (
            unit.text[fragment["start"] : fragment["end"]] == fragment["source_quote"]
        )
        assert (
            body[unit.start + fragment["start"] : unit.start + fragment["end"]]
            == fragment["source_quote"]
        )


@pytest.mark.parametrize(
    "first",
    [
        "ResNet-18 accuracy is 98.93% (five-shot)",
        'ViT uses the term "image patches."',
        "ViT uses image patches (as described in Section 2.1.)",
        "ViT uses image patches [with a fixed resolution]",
    ],
)
def test_inner_decimal_quotes_and_conditions_remain_unchanged(first):
    _, unit, fragments = render_pair(first, "DeiT uses distillation")
    assert unit.text.startswith(first + " [E:a0].")
    assert [f["citation_aliases"] for f in fragments] == [["a0"], ["a1"]]


def test_multiple_citations_remain_with_their_own_scientific_unit():
    _, _, fragments = render_pair(
        "ViT and DeiT are compared under stated conditions",
        "DeiT uses distillation",
        combined=True,
    )
    assert [f["citation_aliases"] for f in fragments] == [["a0", "a1"], ["a1"]]


@pytest.mark.parametrize("kind", ["paragraph", "list"])
def test_same_paper_repeated_units_keep_their_distinct_evidence_locations(kind):
    _, context = materials(2)
    for item in context["evidence"].values():
        item["paper_id"] = "same-paper"
    facts = build_fact_cards(context["evidence"])
    report = draft(facts, "zh-CN")
    block = report["sections"][1]["blocks"][0]
    block["type"] = kind
    block["sentences"] = [
        {"kind": "scientific", "text": text, "fact_card_ids": [cid]}
        for cid, text in zip(
            facts, ["ViT使用“图像块。”", "ViT在固定条件（分辨率224）下运行"]
        )
    ]
    body = render_grounded_report(
        json.dumps(to_native(report)),
        facts,
        "academic_v2",
        "zh-CN",
        require_native=True,
    )
    fragments = [
        f
        for u in build_report_source_units(body)
        if any(text in u.text for text in ("图像块", "分辨率224"))
        for f in build_source_fragments(u)
    ]
    assert [f["citation_aliases"] for f in fragments if f["citation_aliases"]] == [
        ["a0"],
        ["a1"],
    ]
    # Existing Markdown numbering remains a separate uncited organization fragment.
    assert [f["source_quote"] for f in fragments if not f["citation_aliases"]] == (
        ["1.", "2."] if kind == "list" else []
    )


@pytest.mark.parametrize("kind", ["organization", "gap"])
def test_uncited_non_scientific_unit_does_not_inherit_next_citation(kind):
    _, context = materials(1)
    report = draft(context["fact_cards"])
    sentences = report["sections"][1]["blocks"][0]["sentences"]
    sentences.insert(
        0,
        {
            "kind": kind,
            "text": "This review describes the available scope",
            "fact_card_ids": [],
        },
    )
    body = render_grounded_report(
        json.dumps(to_native(report)),
        context["fact_cards"],
        "academic_v2",
        "en",
        require_native=True,
    )
    unit = next(
        u for u in build_report_source_units(body) if "available scope" in u.text
    )
    assert [f["citation_aliases"] for f in build_source_fragments(unit)] == [[], ["a0"]]


def test_old_non_native_sentence_rendering_is_preserved():
    _, unit, _ = render_pair(
        "ViT uses image patches", "DeiT uses distillation", native=False
    )
    assert (
        unit.text.rstrip()
        == "ViT uses image patches [E:a0] DeiT uses distillation [E:a1]"
    )


def test_boundary_version_changes_strict_identity_only():
    config = load_config()
    for policy in ("legacy", "strict_v1", "strict_auto_v1"):
        current = build_config_summary(config, policy)
        with patch("litagent.contracts.WRITING_VERSION", "grounded-writing-v10"):
            previous = build_config_summary(config, policy)
        assert (current == previous) is (policy == "legacy")


@pytest.mark.asyncio
async def test_wrong_citation_cannot_borrow_correct_material_from_next_sentence():
    _, context = materials(2)
    context["evidence"]["a0"].update(
        text="Evo-ViT uses token evolution.",
        supporting_text="Evo-ViT uses token evolution.",
    )
    context["evidence"]["a1"].update(
        text="EViT improves speed by 50% on DeiT-S.",
        supporting_text="EViT improves speed by 50% on DeiT-S.",
    )
    facts = context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(facts)
    report = draft(facts)
    ids = list(facts)
    first = report["sections"][1]["blocks"][0]["sentences"][0]
    first.update(text="EViT improves speed by 50% on DeiT-S", fact_card_ids=ids[:1])
    report["sections"][1]["blocks"][0]["sentences"].append(
        {**first, "fact_card_ids": ids[1:]}
    )
    body = render_grounded_report(
        json.dumps(to_native(report)), facts, "academic_v2", "en", require_native=True
    )
    llm = BatchLLM()
    result = await evaluate_scoped(
        llm=llm, survey=body, context=context, config=EvalConfig()
    )
    wrong = next(a for a in result["assertions"] if a["evidence_ids"] == ["a0"])
    correct = next(a for a in result["assertions"] if a["evidence_ids"] == ["a1"])
    assert wrong["status"] == "unverifiable" and correct["status"] == "supported"
    assert result["score"] != 1.0
    assert body[wrong["source_start"] : wrong["source_end"]] == wrong["source_quote"]
    for request in llm.calls:
        for claim in request.get("assertions", []):
            if claim["claim_id"] == wrong["claim_id"]:
                assert len(claim["evidence"]) == 1
