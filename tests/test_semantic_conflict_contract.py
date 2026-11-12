"""Keep conservative conflict effects separate from scientific verification."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_semantic_conflict_identity import scope

from litagent.eval.batched import batch_conflict_effects
from litagent.eval.scoped import Assessment, PassageDecision, ScopedEvaluationError
from litagent.eval.semantic_conflict import (
    CONFLICT_CODES,
    _digest,
    make_effect,
    selected_policy,
    semantic_conflict_fields,
    semantic_conflict_identity,
    validate_conflict_inputs,
    validate_effect,
)


@pytest.fixture
def conflict(canonical):
    """Use a local shape receipt only; live ownership is tested through the SDK."""
    payload = copy.deepcopy(canonical)
    payload["writer_envelope"] = scope()
    payload["assertions"] = payload["assertions"][:1]
    payload["cited_materials"]["e1"]["paper_id"] = "paper"
    claim = payload["assertions"][0]
    decision = PassageDecision.model_validate(
        {
            "claim_id": claim["claim_id"],
            "correct_kind": "scientific",
            "classification_valid": True,
            "source_faithful": True,
            "atomic": True,
            "status": "unsupported",
            "reason_code": "insufficient_support",
            "explanation": "Controlled conflicting judgments, not a gold label.",
            "support_basis": [{"support_id": "p1"}],
            "aspect_checks": [
                {
                    "aspect": name,
                    "supported": True,
                    "support_ids": ["p1"],
                    "explanation": "Controlled positive subcheck.",
                }
                for name in claim["required_support_aspects"]
            ],
        }
    )
    receipt = {
        "contract": "host-settled-assessment-receipt-v1",
        "stage": "assessment",
        **semantic_conflict_fields(payload["writer_envelope"]),
    }
    receipt["identity"] = _digest(receipt)
    settlement = {
        "receipt": receipt,
        "report_sha256": "1" * 64,
        "response_sha256": hashlib.sha256(
            decision.model_dump_json().encode()
        ).hexdigest(),
        "input_sha256": "2" * 64,
        "batch_payload_sha256": _digest(payload),
    }
    return payload, decision, settlement


def result(decision):
    """Return the canonical batch container used by production validation."""
    return Assessment.model_validate(
        {"decisions": [decision.model_dump()], "units": [], "conflicts": []}
    )


@pytest.mark.parametrize("code", sorted(CONFLICT_CODES))
def test_only_two_authenticated_semantic_shapes_become_conservative_effects(
    conflict, code
):
    payload, decision, settlement = conflict
    if code.startswith("positive"):
        decision.status = decision.reason_code = "supported"
        decision.aspect_checks[0].supported = False
        decision.aspect_checks[0].support_ids = []
    before = copy.deepcopy((payload, decision.model_dump(), settlement))
    [effect] = batch_conflict_effects(payload, result(decision), settlement=settlement)
    assert effect["code"] == code
    assert effect["original_status"] == decision.status
    assert effect["decision_sha256"] == _digest(decision.model_dump())
    assert (payload, decision.model_dump(), settlement) == before
    assert "status" not in effect
    assert "support_validation" not in effect


@pytest.mark.parametrize(
    "fault",
    [
        "paper",
        "source",
        "range",
        "fragment",
        "citation",
        "duplicate_aspect",
        "missing_aspect",
        "empty_positive",
        "unknown_basis",
        "atomic",
        "fidelity",
        "classification",
    ],
)
def test_malformed_source_or_classification_cannot_receive_conflict_effect(
    conflict, fault
):
    payload, decision, settlement = conflict
    if fault == "paper":
        payload["cited_materials"]["e1"]["paper_id"] = "other-paper"
    elif fault == "source":
        payload["cited_materials"]["e1"]["original_support_ref"] = "other-source"
    elif fault == "range":
        payload["support_passages"]["p1"][3] = 999999
    elif fault == "fragment":
        payload["assertions"][0]["source_fragment_id"] = "unknown"
    elif fault == "citation":
        payload["source_units"][0]["source_fragments"][0]["citation_aliases"] = []
    elif fault == "duplicate_aspect":
        decision.aspect_checks.append(copy.deepcopy(decision.aspect_checks[0]))
    elif fault == "missing_aspect":
        decision.aspect_checks.pop()
    elif fault == "empty_positive":
        decision.aspect_checks[0].support_ids = []
    elif fault == "unknown_basis":
        decision.support_basis[0].support_id = "missing"
    elif fault == "atomic":
        decision.atomic = False
    elif fault == "fidelity":
        decision.source_faithful = False
    else:
        decision.correct_kind = "report_structure"
    with pytest.raises(ValueError, match="evaluation_semantic_conflict_input_invalid"):
        validate_conflict_inputs(payload["assertions"][0], decision, payload)


@pytest.mark.parametrize(
    "field", ["claim_id", "claim_text", "source_unit_id", "source_fragment_id"]
)
def test_same_words_at_another_source_location_cannot_borrow_an_effect(conflict, field):
    payload, decision, settlement = conflict
    claim = payload["assertions"][0]
    code = "negative_status_with_complete_positive_diagnostics"
    effect = make_effect(claim, decision, code=code, settlement=settlement)
    changed = {**claim, field: claim[field] + "different"}
    with pytest.raises(ValueError, match="effect_invalid"):
        validate_effect(effect, changed, decision, settlement=settlement)


@pytest.mark.parametrize(
    "field",
    ["report_sha256", "response_sha256", "input_sha256", "batch_payload_sha256"],
)
def test_original_request_and_report_binding_cannot_be_resealed_as_another_effect(
    conflict, field
):
    payload, decision, settlement = conflict
    claim = payload["assertions"][0]
    effect = make_effect(
        claim,
        decision,
        code="negative_status_with_complete_positive_diagnostics",
        settlement=settlement,
    )
    with pytest.raises(ValueError, match="effect_invalid"):
        validate_effect(
            effect, claim, decision, settlement={**settlement, field: "9" * 64}
        )


def test_unselected_guard_still_rejects_original_semantic_conflict(conflict):
    from litagent.eval.batched import _validate_decision_consistency

    payload, decision, _ = conflict
    payload.pop("writer_envelope")
    assert selected_policy(payload.get("writer_envelope")) is False
    with pytest.raises(ScopedEvaluationError, match="decision_consistency_invalid"):
        _validate_decision_consistency(payload["assertions"][0], decision, payload)


def test_policy_identity_is_distinct_and_does_not_mutate_legacy_support_revision():
    from litagent.eval.scoped import support_validation_identity

    original = support_validation_identity("host-support-receipts-v2")
    identity = semantic_conflict_identity()
    assert len(identity) == 64 and identity != original
    assert semantic_conflict_fields(None) == {}
    assert support_validation_identity("host-support-receipts-v2") == original


def long_original(conflict):
    """Build the existing long-original window table through its real producer."""
    from litagent.eval.scoped import build_support_passages, support_passage_table

    payload, decision, settlement = conflict
    source = "A clearly stated scientific condition is present. " * 150
    ledger = {
        "a": {"paper_id": "paper", "text": "candidate", "supporting_text": source}
    }
    passages = build_support_passages(ledger, {"a": "e1"}, original_only=True)
    payload["support_passages"] = support_passage_table(
        passages, {"a": "e1"}, {"a": {"original_support_ref": "s1"}}
    )
    payload["original_supports"]["s1"]["text"] = source
    pid = next(iter(payload["support_passages"]))
    decision.support_basis[0].support_id = pid
    for aspect in decision.aspect_checks:
        aspect.support_ids = [pid]
    settlement["batch_payload_sha256"] = _digest(payload)
    return payload, decision, settlement, source, pid


def test_long_original_window_preview_is_not_a_second_evidence_limit(conflict):
    payload, decision, settlement, source, pid = long_original(conflict)
    row = payload["support_passages"][pid]
    assert len(source) > 6000 and len(row) == 5
    [effect] = batch_conflict_effects(payload, result(decision), settlement=settlement)
    assert effect["original_status"] == "unsupported"
    assert payload["original_supports"]["s1"]["text"] == source
    assert row[4]["display_preview"] == source[row[2] : min(row[3], row[2] + 80)]


@pytest.mark.parametrize(
    "fault", ["preview", "extra", "range", "paper", "unknown_source"]
)
def test_long_original_display_hint_cannot_replace_or_rebind_support(conflict, fault):
    payload, decision, _, source, pid = long_original(conflict)
    row = payload["support_passages"][pid]
    if fault == "preview":
        row[4]["display_preview"] = "A borrowed statement."
    elif fault == "extra":
        row[4]["evidence"] = "must not be accepted"
    elif fault == "range":
        row[3] = len(source) + 1
    elif fault == "paper":
        payload["original_supports"]["s1"]["paper_id"] = "other-paper"
    else:
        row[1] = "unknown-source"
    with pytest.raises(ValueError, match="evaluation_semantic_conflict_input_invalid"):
        validate_conflict_inputs(payload["assertions"][0], decision, payload)


def test_recorded_052_decision_remains_rejected_without_a_new_funded_owner():
    """Replay original diagnostic algebra, without granting a receipt or restart."""
    from litagent.eval.batched import _validate_decision_consistency
    from litagent.eval.scoped import required_support_aspects

    path = Path(__file__).with_name("fixtures") / "run052-semantic-conflict.json"
    original_bytes = path.read_bytes()
    assert hashlib.sha256(original_bytes).hexdigest() == (
        "05d7a0e81d24674b0ae287505d5c86113f2c24d0f4c4f618be14e6a60cba109a"
    )
    record = json.loads(original_bytes)
    assert record["scientific_gold_verified"] is False
    assert record["raw_response_sha256"] == (
        "cc4caa0f39a340ed3b2dfa626b7749fafbfc3ea0c50e7cba8ddf30063a44a7b5"
    )
    decision = PassageDecision.model_validate(record["issue"]["original_decision"])
    aliases = [entry.support_id.split(":", 1)[0] for entry in decision.support_basis]
    claim = {
        **record["claim"],
        "required_support_aspects": required_support_aspects(
            record["claim"]["claim_text"],
            applicability=True,
            support_validation_revision="host-support-receipts-v2",
        ),
        "evidence": [{"evidence_alias": alias} for alias in aliases],
    }
    # These local shape references exercise only the historical consistency
    # predicate. They are not paper evidence, settlement or live authorization.
    payload = {
        "support_passages": {
            entry.support_id: [alias, "shape-only", 0, 1]
            for entry, alias in zip(decision.support_basis, aliases, strict=True)
        },
        "original_supports": {"shape-only": {}},
    }
    with pytest.raises(
        ScopedEvaluationError, match="decision_consistency_invalid"
    ) as error:
        _validate_decision_consistency(claim, decision, payload)
    assert (
        error.value.details["decision_inconsistencies"][0]["subreason"]
        == record["issue"]["subreason"]
    )
    assert decision.status == "unsupported"
    assert not selected_policy(payload.get("writer_envelope"))
    assert path.read_bytes() == original_bytes
