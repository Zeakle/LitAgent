"""Preserve recorded failures and separate atomicity from paper support.

Recorded responses are historical observations, not scientific gold. Controlled
decisions exercise host consumers; they do not measure an LLM's semantic accuracy.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_evaluation_definition_contract import prompt_for, request_payload
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_support_passages import selected

from litagent.eval import scoped

FIXTURE_PATH = (
    Path(__file__).with_name("fixtures") / "evaluation-051-atomic-support.json"
)
FIXTURE_BYTES = FIXTURE_PATH.read_bytes()
FIXTURE = json.loads(FIXTURE_BYTES)
FIXTURE_SHA256 = "c22b6eb930842d681763b37b3be9584354a0c519b4c529a4066abd2c6a3ecdff"


def test_recorded_051_raw_keeps_original_native_shape_without_current_authentication():
    assert hashlib.sha256(FIXTURE_BYTES).hexdigest() == FIXTURE_SHA256
    assert FIXTURE["scientific_gold_verified"] is False
    assert FIXTURE["recorded_support_identity"] != scoped.support_validation_identity(
        scoped.SUPPORT_VALIDATION_REVISION_V2
    )
    for record in FIXTURE["raw_batches"]:
        raw = record["raw_response"]
        assert hashlib.sha256(raw.encode()).hexdigest() == record["raw_sha256"]
        value = json.loads(raw)
        Draft202012Validator(record["native_format"]["json_schema"]["schema"]).validate(
            value
        )
        assert all("classification_valid" not in d for d in value["decisions"])
    # No current envelope, checkpoint or funding identity is supplied here.
    assert FIXTURE_BYTES == FIXTURE_PATH.read_bytes()


def test_current_identity_binds_actual_grouped_kind_instruction(monkeypatch):
    from litagent.eval import batched

    old = scoped.support_validation_identity()
    current = scoped.support_validation_identity(scoped.SUPPORT_VALIDATION_REVISION_V2)
    monkeypatch.setattr(
        batched,
        "GLOBAL_GROUPED_KIND_INPUT_INSTRUCTION",
        batched.GLOBAL_GROUPED_KIND_INPUT_INSTRUCTION + " Changed grouped input rule.",
    )
    assert scoped.support_validation_identity() == old
    assert (
        scoped.support_validation_identity(scoped.SUPPORT_VALIDATION_REVISION_V2)
        != current
    )
    with pytest.raises(
        ValueError, match="evaluation_support_validation_identity_invalid"
    ):
        scoped._support_revision(
            {
                "support_validation_revision": scoped.SUPPORT_VALIDATION_REVISION_V2,
                "support_validation_identity": current,
            }
        )


@pytest.mark.parametrize("index", range(4))
def test_recorded_atomic_false_never_becomes_supported_from_positive_aspects(index):
    record = FIXTURE["atomic_false_records"][index]
    before = copy.deepcopy(record)
    claim = record["claim"]
    decision = scoped.PassageDecision.model_validate(record["canonical_decision"])
    assert decision.atomic is False
    if decision.claim_id == "claim_19":
        assert decision.status == "supported"
        assert all(a.supported for a in decision.aspect_checks)
    [result] = scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger={},
        aliases={},
        support_passages={},
        context={"quality_policy": "strict_auto_v2"},
        strict=True,
    )
    assert result["kind"] == "scientific"
    assert result["status"] == "unverifiable"
    assert result["local_error"] == "evaluation_atomic_classification_unverified"
    assert record == before


def test_recorded_global_misstatement_does_not_replace_the_actual_host_kind():
    record = FIXTURE["global_misstatement"]
    before = copy.deepcopy(record)
    row = dict(zip(record["assertion_columns"], record["wire_row"], strict=True))
    assert row["claim_id"] == "a102" and row["kind"] is None
    assert (
        record["assertion_kind_default"] == record["host_claim"]["kind"] == "scientific"
    )
    raw = record["raw_response"]
    assert hashlib.sha256(raw.encode()).hexdigest() == record["raw_sha256"]
    assert any("a102" in i["anchor_ids"] for i in json.loads(raw)["issues"])
    assert record == before


@pytest.mark.parametrize("stage", ["inventory", "assessment", "global"])
def test_actual_native_tasks_receive_one_report_only_atomicity_definition(
    canonical, stage
):
    payload = request_payload(canonical, stage)
    before = copy.deepcopy(payload)
    messages, native = scoped.build_evaluation_request(
        prompt_for(stage, payload), payload
    )
    description = native["json_schema"]["schema"]["description"]
    definition = scoped.EVALUATION_DEFINITIONS_V2["atomic"]
    assert description.count("[atomic] " + definition) == 1
    assert "A and B each improve X" in definition
    assert "single reported A-vs-B comparison" in definition
    assert "owning REPORT only" in definition
    assert "support problem, not non-atomic" in definition
    assert "Return ONLY" in messages[0]["content"] or stage != "inventory"
    assert payload == before


@pytest.mark.parametrize("citation_count", [1, 2])
def test_controlled_single_comparison_keeps_support_and_atomicity_separate(
    citation_count,
):
    text = (
        "ProtoNet has higher accuracy than MatchingNet under the same "
        "MiniImageNet 5-way 1-shot evaluation."
    )
    ledger = {
        f"evidence{i}": {"text": text, "supporting_text": text}
        for i in range(citation_count)
    }
    aliases = {eid: f"e{i+1}" for i, eid in enumerate(ledger)}
    claim = {
        "claim_id": "claim_0",
        "kind": "scientific",
        "claim_text": text,
        "evidence_ids": list(ledger),
        "local_error": None,
    }
    decision, passages = selected(claim, ledger, aliases)
    before = copy.deepcopy((claim, decision.model_dump(), ledger, passages))
    [result] = scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger=ledger,
        aliases=aliases,
        support_passages=passages,
        context={"quality_policy": "strict_auto_v2", "body_quality": "analysis_v1"},
        strict=True,
    )
    assert result["atomic"] is True and result["status"] == "supported"
    # An independent extraction defect still blocks the same positive paper basis.
    decision.atomic = False
    [invalid] = scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger=ledger,
        aliases=aliases,
        support_passages=passages,
        context={"quality_policy": "strict_auto_v2"},
        strict=True,
    )
    assert invalid["status"] == "unverifiable"
    decision.atomic = True
    assert (claim, decision.model_dump(), ledger, passages) == before


def test_controlled_compound_mechanisms_stay_unverified_even_with_one_complete_source():
    text = "ProtoNet uses class means; MatchingNet uses support-set attention."
    ledger = {"a": {"text": text, "supporting_text": text}}
    claim = {
        "claim_id": "claim_0",
        "kind": "scientific",
        "claim_text": text,
        "evidence_ids": ["a"],
        "local_error": None,
    }
    decision, passages = selected(claim, ledger, {"a": "e1"})
    decision.atomic = False
    [result] = scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger=ledger,
        aliases={"a": "e1"},
        support_passages=passages,
        context={"quality_policy": "strict_auto_v2"},
        strict=True,
    )
    assert result["status"] == "unverifiable"
    assert result["local_error"] == "evaluation_atomic_classification_unverified"


def test_atomic_faithful_uncited_science_does_not_gain_support():
    claim = {
        "claim_id": "claim_0",
        "kind": "scientific",
        "claim_text": "This review finds ProtoNet more accurate than MatchingNet.",
        "evidence_ids": [],
        "local_error": "uncited_claim",
    }
    decision = scoped.PassageDecision(
        claim_id="claim_0",
        correct_kind="scientific",
        atomic=True,
        classification_valid=True,
        source_faithful=True,
        status="unsupported",
        reason_code="uncited_claim",
        explanation="Controlled missing-citation case.",
        support_basis=[],
        aspect_checks=[],
    )
    [result] = scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger={},
        aliases={},
        support_passages={},
        context={"quality_policy": "strict_auto_v2"},
        strict=True,
    )
    assert result["atomic"] is True and result["source_faithful"] is True
    assert result["kind"] == "scientific"
    assert (
        result["status"] == "unsupported" and result["reason_code"] == "uncited_claim"
    )
