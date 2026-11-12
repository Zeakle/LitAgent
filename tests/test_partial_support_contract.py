"""Clarify selected partial support without relabeling real provider errors."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_support_receipts import envelope

import litagent.eval.scoped as scoped
from litagent.eval.batched import _validate_decision_consistency, batch_prompt
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import validate_writer_envelope

FIXTURE = Path(__file__).parent / "fixtures/run043-decision-consistency.json"
FIXTURE_SHA256 = "4184813c4ec15df0e1a408701de3594d3326633c21a911c14903d6ea5f3fcba9"
REVISION = "host-support-receipts-v1"
OLD_EMPTY_RULE = "Other kinds and unsupported decisions return an empty support_basis."
LEGACY_REQUESTS = {
    "no_envelope": "e05fc00c282d73b2a7fc2ab44b5cf9589d0cd3116722db1aa4995e2acf2dee29",
    "unselected": "14cef23aa789ed2e0470116f32ce0e8f0f03b3c356bb7ef1a5d50ebd98e11b97",
}


@pytest.fixture(scope="module")
def real043():
    """Keep complete public originals and original non-gold decision fields."""
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    return json.loads(raw)


def request(canonical, selection):
    """Use the actual sender builder on a source-owned literal citation suffix."""
    payload = copy.deepcopy(canonical)
    payload["source_units"][0]["source_fragments"][0]["text"] += " [E:e1]"
    if selection != "no_envelope":
        payload["writer_envelope"] = envelope(active=selection == "selected")
    payload["evaluation_input_contract"] = "flat-evaluation-input-v1"
    return payload, scoped.build_evaluation_request(batch_prompt(payload), payload)


def request_digest(messages, schema):
    """Bind the complete native wire rather than a prompt-only approximation."""
    encoded = json.dumps(
        [messages, schema], ensure_ascii=False, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def original_packet(fixture, index):
    """Preserve the actual p1/p3 permissions for the original consistency guard."""
    case = fixture["cases"][index]
    claim = {
        **copy.deepcopy(case["wire_claim"]),
        "source_unit_id": case["claim"]["source_unit_id"],
        "source_fragment_id": case["claim"]["source_fragment_id"],
        "evidence": [
            {"evidence_alias": alias, "material_ref": alias}
            for alias in fixture["fragment"]["citation_aliases"]
        ],
    }
    decision = scoped.PassageDecision.model_validate(case["raw_decision"])
    return (
        claim,
        decision,
        {
            "support_passages": copy.deepcopy(fixture["support_passages"]),
            "original_supports": copy.deepcopy(fixture["original_supports"]),
        },
    )


def host_packet(fixture, index):
    """Resolve original wire IDs through complete source identity, not spelling."""
    case = fixture["cases"][index]
    aliases = dict(
        zip(
            case["claim"]["evidence_ids"],
            fixture["fragment"]["citation_aliases"],
            strict=True,
        )
    )
    ledger = {}
    for eid, alias in aliases.items():
        material = fixture["cited_materials"][alias]
        original = fixture["original_supports"][material["original_support_ref"]]
        ledger[eid] = {
            "paper_id": original["paper_id"],
            "paper_title": material["paper_title"],
            "text": original["text"],
            "supporting_text": original["text"],
        }
    passages = scoped.build_support_passages(ledger, aliases, original_only=True)
    mapping = {}
    for short, (alias, ref, start, end) in fixture["support_passages"].items():
        candidates = [
            pid
            for pid, passage in passages.items()
            if passage["evidence_alias"] == alias
            and passage["source_start"] == start
            and passage["source_end"] == end
            and ledger[passage["evidence_id"]]["supporting_text"]
            == fixture["original_supports"][ref]["text"]
        ]
        assert len(candidates) == 1
        mapping[short] = candidates[0]
    raw = copy.deepcopy(case["raw_decision"])
    for basis in raw["support_basis"]:
        basis["support_id"] = mapping[basis["support_id"]]
    for aspect in raw["aspect_checks"]:
        aspect["support_ids"] = [mapping[pid] for pid in aspect["support_ids"]]
    return (
        copy.deepcopy(case["claim"]),
        scoped.PassageDecision.model_validate(raw),
        ledger,
        aliases,
        passages,
    )


def host_checked(claim, decision, ledger, aliases, passages):
    """Use final and partial host validation with the same bound source inputs."""
    identity = {
        "support_validation_revision": REVISION,
        "support_validation_identity": scoped.support_validation_identity(),
    }
    context = {
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "_independent_classification": True,
        "_support_validation_revision": REVISION,
        "_host_validation_inputs": {
            **identity,
            "claims": [claim],
            "ledger": ledger,
            "aliases": aliases,
            "support_passages": passages,
        },
    }
    final = scoped.validate_assessment_decisions(
        [claim],
        [decision],
        ledger=ledger,
        aliases=aliases,
        support_passages=passages,
        context=context,
        strict=True,
    )
    partial = scoped.partial_assessment(
        {
            **identity,
            "batches": [
                {
                    "status": "completed",
                    "claim_ids": [decision.claim_id],
                    "decisions": [decision.model_dump()],
                }
            ],
        },
        context,
    )
    assert partial["assertions"] == final
    assert partial["overall_valid"] is False
    return final[0], partial


def test_fixture_preserves_full_originals_and_unmodified_negative_fields(real043):
    assert real043["science_gold_standard"] is False
    assert real043["raw_provider_judgments_unvalidated"] is True
    assert [c["claim"]["claim_id"] for c in real043["cases"]] == [
        "claim_60",
        "claim_61",
    ]
    assert len(real043["original_supports"]["s3"]["text"]) == 1220
    assert len(real043["original_supports"]["s7"]["text"]) == 1053
    fragment = real043["fragment"]
    for case in real043["cases"]:
        assert case["claim"]["source_fragment_id"] == (
            fragment["native_source_fragment_id"]
        )
        assert case["wire_claim"]["source_fragment_id"] == "f5"
        assert case["wire_claim"]["allowed_support_ids"] == ["p1", "p3"]
        assert case["raw_decision"]["status"] == "unsupported"
        assert case["raw_decision"]["reason_code"] == "attribution_mismatch"
        assert all(a["supported"] for a in case["raw_decision"]["aspect_checks"])
        assert hashlib.sha256(case["claim"]["source_quote"].encode()).hexdigest() == (
            real043["provenance"]["native_fragment_sha256"]
        )
    assert "全局向量原型" in real043["cases"][1]["claim"]["claim_text"]
    assert "global vector representation" in real043["original_supports"]["s7"]["text"]


@pytest.mark.parametrize("index", (0, 1))
def test_original043_negative_with_all_positive_aspects_remains_rejected(
    real043, index
):
    original = copy.deepcopy(real043)
    claim, decision, payload = original_packet(real043, index)
    with pytest.raises(
        scoped.ScopedEvaluationError, match="^evaluation_decision_consistency_invalid$"
    ) as caught:
        _validate_decision_consistency(claim, decision, payload)
    diagnostic = caught.value.details["decision_inconsistencies"][0]
    assert diagnostic["original_decision"] == real043["cases"][index]["raw_decision"]
    assert (
        diagnostic["subreason"] == "negative_status_with_complete_positive_diagnostics"
    )
    assert real043 == original


@pytest.mark.parametrize("selection", LEGACY_REQUESTS)
def test_unselected_native_requests_keep_original_complete_wire_bytes(
    canonical, selection
):
    _, (messages, schema) = request(canonical, selection)
    assert request_digest(messages, schema) == LEGACY_REQUESTS[selection]
    assert OLD_EMPTY_RULE in messages[0]["content"]
    assert scoped.SUPPORT_VALIDATION_INSTRUCTION not in messages[0]["content"]


def test_selected_actual_request_resolves_partial_basis_conflict_without_growing(
    canonical,
):
    original = copy.deepcopy(canonical)
    payload, (messages, schema) = request(canonical, "selected")
    prompt = messages[0]["content"]
    assert OLD_EMPTY_RULE not in prompt
    assert "Non-scientific or baseless decisions use an empty support_basis." in prompt
    assert prompt.count(scoped.SUPPORT_VALIDATION_INSTRUCTION) == 1
    assert prompt.count("support_basis may use a sufficient subset") == 1
    assert (
        "subject/method ownership, relation, object/comparator and polarity" in prompt
    )
    assert "task, dataset and setup, including applicability limits" in prompt
    assert "Partial negatives may retain basis for established aspects" in prompt
    assert "mark missing aspects false" in prompt
    assert "Unused citations alone aren't attribution_mismatch" in prompt
    for original_rule, replacement in scoped._SUPPORT_PROMPT_PROJECTION:
        assert batch_prompt(payload).count(original_rule) == 1
        assert original_rule not in prompt
        assert prompt.count(replacement) == 1
    assert strict_request_reservation(messages, 16384, schema) <= 39008
    wire = json.loads(messages[1]["content"])
    assert wire["original_supports"] == payload["original_supports"]
    _, (_, previous_schema) = request(canonical, "unselected")
    clean_schema = copy.deepcopy(schema)
    properties = clean_schema["json_schema"]["schema"]["$defs"]["PassageDecision"][
        "properties"
    ]
    properties["status"].pop("description")
    properties["reason_code"].pop("description")
    assert clean_schema == previous_schema
    assert canonical == original


def test_projection_rules_are_bound_to_identity_and_do_not_change_none(
    canonical, monkeypatch
):
    before = scoped.support_validation_identity()
    selected = envelope()
    _, (old_messages, old_schema) = request(canonical, "unselected")
    rules = scoped._SUPPORT_PROMPT_PROJECTION
    monkeypatch.setattr(
        scoped,
        "_SUPPORT_PROMPT_PROJECTION",
        (*rules[:-1], (rules[-1][0], rules[-1][1] + " Clarified.")),
    )
    assert scoped.support_validation_identity() != before
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(selected)
    _, (messages, schema) = request(canonical, "unselected")
    assert messages == old_messages and schema == old_schema
    _, (messages, _) = request(canonical, "selected")
    assert rules[-1][1] + " Clarified." in messages[0]["content"]


@pytest.mark.parametrize("index", (0, 1))
@pytest.mark.parametrize("negative_aspect", ("predicate", "conditions"))
def test_partial_negative_can_keep_established_basis_without_promotion(
    real043, index, negative_aspect
):
    original = copy.deepcopy(real043)
    claim, raw, payload = original_packet(real043, index)
    next(a for a in raw.aspect_checks if a.aspect == negative_aspect).supported = False
    _validate_decision_consistency(claim, raw, payload)
    host_claim, decision, ledger, aliases, passages = host_packet(real043, index)
    next(a for a in decision.aspect_checks if a.aspect == negative_aspect).supported = (
        False
    )
    final, partial = host_checked(host_claim, decision, ledger, aliases, passages)
    assert final["status"] == "unsupported"
    assert final["support_basis"] and partial["supported_count"] == 0
    assert partial["unsupported_count"] == 1
    decision.status, decision.reason_code = "supported", "supported"
    with pytest.raises(
        scoped.ScopedEvaluationError, match="decision_consistency_invalid"
    ):
        _validate_decision_consistency(claim, decision, payload)
    assert real043 == original


@pytest.mark.parametrize("uncited", (False, True))
def test_no_support_keeps_empty_basis_and_cannot_become_a_passing_score(
    real043, uncited
):
    claim, decision, payload = original_packet(real043, 0)
    decision.reason_code = "uncited_claim" if uncited else "insufficient_support"
    decision.support_basis = []
    for aspect in decision.aspect_checks:
        aspect.supported, aspect.support_ids = False, []
    if uncited:
        claim["evidence"] = []
    _validate_decision_consistency(claim, decision, payload)
    host_claim, _, ledger, aliases, passages = host_packet(real043, 0)
    if uncited:
        host_claim["evidence_ids"] = []
    final, partial = host_checked(host_claim, decision, ledger, aliases, passages)
    assert final["status"] == "unsupported"
    assert final["support_basis"] == []
    assert partial["supported_count"] == 0 and partial["unsupported_count"] == 1


@pytest.mark.parametrize("fault", ("method", "dataset", "quantity"))
def test_sufficient_subset_never_licenses_wrong_entity_condition_or_number(
    real043, fault
):
    claim, decision, ledger, aliases, passages = host_packet(real043, 0)
    # These are separate host-guard controls, never replacements for raw labels.
    replacements = {
        "method": ("SRL", "MSENet"),
        "dataset": ("当每个类别有 K 张图像时", "在 CIFAR-100 上每个类别有 K 张图像时"),
        "quantity": ("两种模板子空间", "3 种模板子空间"),
    }
    old, new = replacements[fault]
    assert old in claim["claim_text"]
    claim["claim_text"] = claim["claim_text"].replace(old, new)
    decision.status, decision.reason_code = "supported", "supported"
    aspects = scoped.required_support_aspects(
        claim["claim_text"], applicability=True, support_validation_revision=REVISION
    )
    basis = decision.aspect_checks[0]
    decision.aspect_checks = [basis.model_copy(update={"aspect": a}) for a in aspects]
    final, partial = host_checked(claim, decision, ledger, aliases, passages)
    assert final["status"] == "unverifiable"
    assert final["support_validation"]["status"] == "unverified"
    assert partial["supported_count"] == 0
