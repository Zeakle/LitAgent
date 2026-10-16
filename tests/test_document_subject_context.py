"""Use an explicit same-paper name without treating its title as scientific proof."""

import copy
import json
from pathlib import Path

import pytest
import yaml
from test_support_passages import selected

from litagent.eval.scoped import _scientific_materials, validate_support_basis
from litagent.quality import (
    build_fact_cards,
    declared_document_subject,
    digest,
    grounded_messages,
    validate_fact_snapshot,
)


def original():
    manifest = yaml.safe_load(
        Path("benchmarks/survey/vit_nerf_manifest.yaml").read_text()
    )
    row = next(r for r in manifest["papers"] if r["paper_id"] == "arxiv:2003.08934")
    return {
        "paper_id": row["paper_id"],
        "paper_title": row["title"],
        "supporting_text": row["abstract"],
        "text": "unverified extraction",
    }


def assess(text, item, quote=None, strict=True):
    claim = {"claim_id": "c", "claim_text": text, "evidence_ids": ["a"]}
    ledger = {"a": item}
    decision, passages = selected(claim, ledger, {"a": "e1"}, quote=quote)
    return validate_support_basis(
        claim, decision, ledger, {"a": "e1"}, passages=passages, original_only=strict
    )


def test_source_declares_named_subject_but_semantic_method_stays_missing():
    item = original()
    before = copy.deepcopy(item)
    assert "NeRF" not in item["supporting_text"]
    assert declared_document_subject(item) == "NeRF"
    cards = build_fact_cards({"a": item})
    card = next(iter(cards.values()))
    assert card["document_subject"] == "NeRF" and card["method"] is None
    assert (
        "method" in card["missing_fields"] and card["semantic_status"] == "not_verified"
    )
    validate_fact_snapshot(cards, {"a": item}, digest(cards))
    packet = json.loads(
        grounded_messages(cards, "academic_v2", "en", "NeRF")[1]["content"]
    )
    source = next(iter(packet["sources"].values()))
    assert (
        source["source_quote"] == item["supporting_text"]
        and source["document_subject"] == "NeRF"
    )
    assert item == before


@pytest.mark.parametrize(
    "text",
    [
        "NeRF represents a scene using a fully-connected deep network.",
        "NeRF uses classic volume rendering techniques.",
    ],
)
def test_actual_named_paper_self_reference_is_not_a_missing_entity(text):
    assert assess(text, original())["status"] == "passed"


@pytest.mark.parametrize(
    "text,issue",
    [
        ("EViT uses classic volume rendering techniques.", "support_entity_unverified"),
        ("NeRF achieves accuracy of 99%.", "support_number_unverified"),
        ("NeRF uses edge deployment.", "support_relation_unverified"),
    ],
)
def test_title_context_cannot_lend_another_method_number_or_condition(text, issue):
    assert assess(text, original())["issue"] == issue


@pytest.mark.parametrize(
    "change",
    [
        {"paper_id": ""},
        {"paper_title": "An analysis of NeRF"},
        {"paper_title": "Mip-NeRF: Rendering"},
        {"paper_title": "NeRF methods: Rendering"},
        {"supporting_text": "A background finding only."},
    ],
)
def test_missing_ambiguous_or_other_document_does_not_define_claim_subject(change):
    item = original()
    item.update(change)
    if change.get("paper_title", "").startswith("Mip-NeRF:"):
        assert declared_document_subject(item) == "Mip-NeRF"
        assert (
            assess("NeRF uses classic volume rendering techniques.", item)["issue"]
            == "support_entity_unverified"
        )
    else:
        assert declared_document_subject(item) is None


def test_local_intro_window_cannot_borrow_document_context_or_later_findings():
    item = original()
    intro = item["supporting_text"].split(". ")[0] + "."
    assert (
        assess("NeRF uses classic volume rendering techniques.", item, quote=intro)[
            "issue"
        ]
        == "support_entity_unverified"
    )


def test_forged_card_context_fails_source_lineage_and_legacy_does_not_change():
    item = original()
    cards = build_fact_cards({"a": item})
    next(iter(cards.values()))["document_subject"] = "EViT"
    with pytest.raises(ValueError, match="evaluation_fact_source_invalid"):
        validate_fact_snapshot(cards, {"a": item}, digest(cards))
    assert (
        assess("NeRF uses classic volume rendering techniques.", item, strict=False)[
            "issue"
        ]
        == "support_entity_unverified"
    )
    materials, _ = _scientific_materials({"a": item}, ["a"])
    assert "document_subject" not in materials["a"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text,item,expected",
    [
        ("NeRF uses classic volume rendering techniques.", original(), 1.0),
        (
            "EViT achieves an average speed improvement of 50%.",
            {
                "paper_id": "evit",
                "supporting_text": "EViT achieves an average speed improvement of 25% and a peak speed improvement of 50%.",
            },
            0.0,
        ),
    ],
)
async def test_frozen_judge_uses_strict_original_context_and_relation_guard(
    text, item, expected
):
    from test_batched_evaluation import BatchLLM
    from test_strict_automatic import frozen

    from litagent.audit_context import SurveyAuditContext
    from litagent.benchmark.survey_judge import SurveyBenchmarkJudge
    from litagent.benchmark.survey_models import SurveyJudgeConfig
    from litagent.eval.automatic import freeze_automatic_inputs
    from litagent.report_render import build_report_source_units

    report = copy.deepcopy(frozen()["report"])
    report["survey"] = text + " [E:a]"
    ledger = {"a": {**item, "text": "Compressed scientific text is not proof."}}
    cards = build_fact_cards(ledger)
    report["metadata"].update(fact_cards=cards, evidence_snapshot_sha256=digest(cards))
    unit = build_report_source_units(report["survey"])[0]
    assertion = report["evaluation"]["faithfulness"]["details"]["assertions"][0]
    assertion.update(
        claim_text=text,
        source_quote=report["survey"],
        source_unit_id=unit.unit_id,
        source_fragment_id=f"{unit.unit_id}:fragment:0",
        source_start=0,
        source_end=len(report["survey"]),
    )
    snapshot = freeze_automatic_inputs(report, ledger, run_id="context-guard")
    llm = BatchLLM()
    judge, usage = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline")
    ).evaluate(
        case=SurveyAuditContext.model_validate(snapshot["context"]),
        survey=report["survey"],
        ledger=ledger,
        frozen_inventory=snapshot["frozen_inventory"],
    )
    assert judge.status == "completed" and judge.faithfulness == expected
    assert usage.call_count == 2
    assert next(iter(llm.calls[0]["support_passages"])) == "e1:full_original"


def test_jev_transmits_subject_separately_from_unchanged_full_original():
    from test_jev_citation_audit import _stable_test_sha as stable_digest
    from test_strict_automatic import frozen

    from litagent.benchmark.citation_audit import (
        _input_identity,
        _question_specs,
        _request_payload,
        build_citation_audit_inputs,
    )
    from litagent.benchmark.survey_models import JudgeInventorySnapshot

    item = original()
    ledger = {"a": item}
    inventory = JudgeInventorySnapshot.model_validate(
        frozen()["frozen_inventory"]["inventory"]
    )
    inventory = inventory.model_copy(update={"ledger_sha256": stable_digest(ledger)})
    inputs = build_citation_audit_inputs(
        inventory, ledger, artifact={"quality_policy": "strict_auto_v1"}
    )
    payload = _request_payload(_question_specs(inputs), "jev-1.13.0")
    for state in payload["state"]["audit_items"].values():
        evidence = state["cited_evidence"]
        if isinstance(evidence, list):
            evidence = evidence[0]
        assert evidence["document_subject"] == "NeRF"
        assert evidence["text"] == item["supporting_text"]
    legacy = build_citation_audit_inputs(inventory, ledger, artifact={})
    old = legacy[0].model_dump(mode="json")
    for evidence in old["evidence"]:
        evidence.pop("document_subject")
    assert _input_identity(legacy[0]) == old
