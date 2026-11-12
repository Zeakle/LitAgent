"""Keep source-grounded controls for the actual 027 support failures."""

import hashlib
import json
from pathlib import Path

import pytest
from test_support_passages import selected
from test_writer_v2 import case  # noqa: F401

from litagent.eval.batched import GLOBAL_FINDINGS_CONTRACT, batch_prompt
from litagent.eval.scoped import (
    LOCATED_INVENTORY_INSTRUCTION,
    LOCATED_SUPPORT_INSTRUCTION,
    _bound_numeric_facts,
    _protocol_numeric_facts,
    located_evaluation_identity,
    validate_assessment_decisions,
    validate_support_basis,
)
from litagent.workbench import RunOptions
from litagent.writing_v2 import writer_budget_plan

FIXTURE = json.loads(
    Path(__file__)
    .with_name("fixtures")
    .joinpath("run027-support-cases.json")
    .read_text()
)


def checked(text, source, *, enabled=True):
    """Use production passage indexing and full-original support validation."""
    ledger = {"a": {"text": source, "supporting_text": source}}
    claim = {"claim_text": text, "evidence_ids": ["a"]}
    decision, passages = selected(claim, ledger, {"a": "e1"})
    result = validate_support_basis(
        claim,
        decision,
        ledger,
        {"a": "e1"},
        passages=passages,
        original_only=True,
        local_numeric_bindings=True,
        chinese_comparatives=True,
        protocol_numeric_scope=enabled,
    )
    return result, claim, decision, ledger, passages


def test_real_originals_and_diagnostic_provenance_are_self_contained():
    assert len(FIXTURE["cases"]) == 25
    for item in FIXTURE["sources"].values():
        assert (
            hashlib.sha256(item["supporting_text"].encode()).hexdigest()
            == item["sha256"]
        )
    assert FIXTURE["source_report_sha256"] == (
        "6ca325ac65fbe2a85caf45330755a94aef63f2acfd4afec888cb8871d7eca5db"
    )


@pytest.mark.parametrize(
    "text",
    [
        "增加整合成分后，在5-way 1-shot和5-shot条件下评估。",
        "After increasing integration components, evaluation uses 5-way 1-shot tasks.",
    ],
)
def test_protocol_counts_do_not_inherit_unrelated_increase(text):
    source = "Evaluation uses 5-way 1-shot and 5-shot tasks."
    old, *_ = checked(text, source, enabled=False)
    new, *_ = checked(text, source)
    assert old["issue"] == "support_direction_unverified"
    assert new["status"] == "passed"
    assert new["spans"] == old["spans"]
    # This is a local numeric check, not a verdict on the compound sentence.
    assert all(f[3] is None for f in _protocol_numeric_facts(text))


@pytest.mark.parametrize(
    "claim,source",
    [
        ("Evaluation uses 5-way tasks.", "Evaluation uses 5-shot tasks."),
        ("Evaluation uses 5-shot tasks.", "Evaluation uses 5-way tasks."),
        ("Evaluation uses 5-way 1-shot tasks.", "Evaluation uses 5-way 5-shot tasks."),
        ("Evaluation uses 10-way tasks.", "Evaluation uses 5-way tasks."),
        ("Accuracy improves by 5%.", "Evaluation uses 5-way tasks."),
        ("Accuracy improves by 2 percentage points.", "Accuracy improves by 2%."),
        ("Latency decreases by 20%.", "Throughput increases by 20%."),
        ("EViT accelerates DeiT-S by 50%.", "Evo-ViT accelerates DeiT-S by 50%."),
        ("DeiT reduces FLOPs by 30%.", "ConViT reduces FLOPs by 30%."),
    ],
)
def test_protocol_rule_never_licenses_wrong_counts_types_or_science(claim, source):
    result, *_ = checked(claim, source)
    assert result["status"] == "unverified"


@pytest.mark.parametrize(
    "text",
    [
        "EViT reduces FLOPs by 44% with 0.3% accuracy loss.",
        "模型准确率提升2%，延迟降低20%。",
        "Average speed is 43 FPS and peak speed is 50 FPS.",
        "No numerical performance is reported.",
    ],
)
def test_non_protocol_numeric_facts_keep_the_existing_strict_interpretation(text):
    assert _protocol_numeric_facts(text) == _bound_numeric_facts(text)


@pytest.mark.parametrize("ident", ["claim_2", "claim_5", "claim_16", "claim_82"])
def test_real_source_spelling_and_undeclared_abbreviations_are_not_guessed(ident):
    row = next(c for c in FIXTURE["cases"] if c["claim_id"] == ident)
    # Match the actually selected paper, without lending a second paper's aliases.
    selected_alias = row["support_basis"][0]["support_id"].split(":")[0]
    index = {"e3": "arxiv:2305.09062", "e6": "arxiv:1802.04376"}[selected_alias]
    source = next(
        i["supporting_text"]
        for i in FIXTURE["sources"].values()
        if i["paper_id"] == index
    )
    result, *_ = checked(row["claim_text"], source)
    assert result["issue"] == "support_entity_unverified"
    if ident != "claim_5":
        assert "miniImagenNet" in source and "miniImageNet" not in source


def test_wrong_passage_owner_stays_invalid_and_reports_exact_mapping():
    text = "A comparison function is learned."
    result, claim, decision, ledger, _ = checked(text, text)
    assert result["status"] == "passed"
    ledger["other"] = {"text": text, "supporting_text": text}
    from litagent.eval.scoped import build_support_passages

    aliases = {"a": "e1", "other": "e2"}
    passages = build_support_passages(ledger, aliases, original_only=True)
    decision.support_basis[0].support_id = "e2:full_original"
    invalid = validate_support_basis(
        claim, decision, ledger, aliases, passages=passages, protocol_numeric_scope=True
    )
    assert invalid["issue"] == "support_basis_ownership_invalid"
    assert invalid["actual_evidence_id"] == "other"
    assert invalid["expected_evidence_ids"] == ["a"]
    assert invalid["support_id"] == "e2:full_original"
    assert not invalid["spans"]


def test_a_local_numeric_positive_does_not_approve_a_non_atomic_inventory():
    text = "增加整合成分后，在5-way 1-shot条件下评估。"
    check, claim, decision, ledger, passages = checked(
        text, "Evaluation uses 5-way 1-shot tasks."
    )
    assert check["status"] == "passed"
    claim.update(claim_id="claim_0", kind="scientific", local_error=None)
    decision.atomic = False
    values = validate_assessment_decisions(
        [claim],
        [decision],
        ledger=ledger,
        aliases={"a": "e1"},
        support_passages=passages,
        context={
            "_evaluation_progress": {
                "global_findings_contract": GLOBAL_FINDINGS_CONTRACT
            }
        },
        strict=True,
    )
    assert values[0]["status"] == "unverifiable"
    assert values[0]["local_error"] == "evaluation_atomic_classification_unverified"


def test_only_explicit_located_requests_receive_the_shared_instruction_and_identity(
    case, monkeypatch
):
    old_prompt = batch_prompt({})
    new_prompt = batch_prompt({"global_findings_contract": GLOBAL_FINDINGS_CONTRACT})
    assert new_prompt == old_prompt + LOCATED_SUPPORT_INSTRUCTION
    assert "under the SAME host fragment" in LOCATED_INVENTORY_INSTRUCTION
    assert (
        "Do not solve an extraction defect by changing report prose"
        in LOCATED_INVENTORY_INSTRUCTION
    )
    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v7",
        capacity_policy="adaptive-capacity-v1",
        verification={"judge": False, "jev": False},
    )
    original = options.fingerprint()
    v6 = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v6"
    )
    v7 = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )
    from litagent.eval import scoped

    monkeypatch.setattr(
        scoped, "LOCATED_SUPPORT_CHECK_CONTRACT", "controlled-other-check"
    )
    assert options.fingerprint() != original
    assert (
        writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v6"
        )
        == v6
    )
    assert (
        writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
        )["envelope"]["builder_identity"]
        != v7["envelope"]["builder_identity"]
    )
    assert len(located_evaluation_identity()) == 64
