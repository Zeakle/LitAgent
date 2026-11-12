"""Align selected task instructions without accepting old scientific failures."""

import copy
import json
from pathlib import Path

import pytest
from test_evaluation_definition_contract import (
    OLD_V1_IDENTITY,
    V1,
    V2,
    prompt_for,
    request_payload,
    without_descriptions,
)
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_support_passages import selected

from litagent.eval import scoped
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_accounting import _envelope
from litagent.safety.planned_budget import _digest, validate_writer_envelope

SUPPORT_IDENTITY_049 = (
    "832c62a700de010d1ba7536139a8c6872071aa3cfb9036538cb35ed428025ffc"
)


@pytest.mark.parametrize("stage", ["inventory", "assessment", "global"])
def test_actual_native_request_keeps_shared_relation_and_scope_definitions(
    canonical, stage
):
    """Inspect actual native schema and retain each stage's material boundary."""
    payload = request_payload(canonical, stage)
    before = copy.deepcopy(payload)
    messages, native = scoped.build_evaluation_request(
        prompt_for(stage, payload), payload
    )
    schema = native["json_schema"]["schema"]
    definitions = schema["description"]
    assert "A single reported A-vs-B comparison" in definitions
    assert "never invent separate experiments" in definitions
    assert "'cannot infer P' is not 'infer not-P'" in definitions
    assert "never inferred from prose" in definitions
    assert "their scientific premises remain scientific" in definitions
    assert payload == before
    assert strict_request_reservation(messages, 16384, native) > 16384
    described = scoped.evaluation_definition_schema(
        payload["schema"], stage=stage, host_kind_agreement=stage == "assessment"
    )
    assert without_descriptions(described) == without_descriptions(payload["schema"])
    if stage == "assessment":
        assert "emit no extra N/A true/false rows" in definitions
        assert "never omit them" in definitions
        assert "Lack of grounds to infer P is not evidence for not-P" in definitions
        wire = json.loads(messages[1]["content"])
        assert wire["original_supports"] == before["original_supports"]
    if stage == "global":
        system = " ".join(messages[0]["content"].split())
        assert "infer separate experiments from a joint result" in system
        assert "global input has no paper originals" in system
        assert "only assessment checks them against paper originals" in system
        assert "identify two independent relations in the anchored assertion" in system
        assert "original_supports" not in json.loads(messages[1]["content"])


@pytest.mark.parametrize("revision", [None, V1])
def test_selected_task_changes_do_not_reach_historical_requests(canonical, revision):
    """Keep absent and explicitly old selections out of the changed task."""
    payload = request_payload(canonical, revision=revision)
    messages, native = scoped.build_evaluation_request(
        prompt_for("assessment", payload), payload
    )
    text = json.dumps([messages, native], ensure_ascii=False)
    assert "emit no extra N/A true/false rows" not in text
    assert "'cannot infer P' is not 'infer not-P'" not in text
    assert scoped.support_validation_identity() == OLD_V1_IDENTITY
    assert scoped.support_validation_identity(V1) == OLD_V1_IDENTITY


def test_old_v2_pin_remains_fee_data_without_granting_current_execution(canonical):
    """Separate archived accounting identity from current execution authority."""
    current = request_payload(canonical)["writer_envelope"]
    assert scoped.support_validation_identity(V2) != SUPPORT_IDENTITY_049
    old = {**current, "support_validation_identity": SUPPORT_IDENTITY_049}
    old["identity"] = _digest(
        {key: value for key, value in old.items() if key != "identity"}
    )
    before = copy.deepcopy(old)
    assert _envelope(old).max_body_units == 66
    assert old == before
    with pytest.raises(ValueError, match="writer_envelope_support_validation_invalid"):
        validate_writer_envelope(old)


def support_case():
    """Use real passage construction; scripted labels are not a semantic gold."""
    text = "A method has a source-bound limitation."
    claim = {"claim_text": text, "evidence_ids": ["a"]}
    ledger = {"a": {"text": text, "supporting_text": text}}
    aliases = {"a": "e1"}
    decision, passages = selected(claim, ledger, aliases)
    required = scoped.required_support_aspects(
        text, applicability=True, support_validation_revision=V2
    )
    decision.aspect_checks = [
        decision.aspect_checks[0].model_copy(update={"aspect": aspect})
        for aspect in required
    ]
    return claim, decision, ledger, aliases, passages


def check_support(claim, decision, ledger, aliases, passages):
    """Exercise the unchanged full host validator, not a mocked acceptance."""
    return scoped.validate_support_basis(
        claim,
        decision,
        ledger,
        aliases,
        passages=passages,
        original_only=True,
        applicability=True,
        local_numeric_bindings=True,
        protocol_numeric_scope=True,
        support_validation_revision=V2,
    )


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("extra", ["quantities", "causal_relation"])
def test_extra_positive_or_negative_aspect_still_rejects_without_mutating_raw(
    supported, extra
):
    """Reject extra rows regardless of their sign and preserve the decision."""
    claim, decision, ledger, aliases, passages = support_case()
    decision.aspect_checks.append(
        decision.aspect_checks[0].model_copy(
            update={"aspect": extra, "supported": supported}
        )
    )
    before = decision.model_dump()
    result = check_support(claim, decision, ledger, aliases, passages)
    assert result["status"] == "unverified"
    assert result["issue"] == "support_aspect_coverage_invalid"
    assert result["actual_aspects"][-1] == extra
    assert decision.model_dump() == before


@pytest.mark.parametrize("fault", ["duplicate", "missing", "negative", "ownership"])
def test_exact_aspect_guidance_does_not_relax_other_support_guards(fault):
    """Retain coverage, negative-aspect and actual-citation guards."""
    claim, decision, ledger, aliases, passages = support_case()
    assert (
        check_support(claim, decision, ledger, aliases, passages)["status"] == "passed"
    )
    if fault == "duplicate":
        decision.aspect_checks.append(decision.aspect_checks[0].model_copy())
    elif fault == "missing":
        decision.aspect_checks.pop()
    elif fault == "negative":
        decision.aspect_checks[0].supported = False
    else:
        claim["evidence_ids"] = ["another-paper"]
    assert (
        check_support(claim, decision, ledger, aliases, passages)["status"] != "passed"
    )


@pytest.mark.parametrize("case_id", ["extra_positive", "extra_negative"])
def test_real_recorded_extra_aspects_remain_protocol_negative_controls(case_id):
    """Preserve both signs of the actual malformed reply, not a scientific gold."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/evaluation-049-aspects.json").read_text()
    )
    assert fixture["scientific_gold_verified"] is False
    record = next(row for row in fixture["cases"] if row["case_id"] == case_id)
    before = copy.deepcopy(record)
    assert record["later_host_checks_verified"] is False
    decision = scoped.PassageDecision.model_validate(record["raw_provider_decision"])
    assert decision.status == "supported"
    assert any(not aspect.supported for aspect in decision.aspect_checks) is (
        case_id == "extra_negative"
    )
    result = check_support(
        record["claim"],
        decision,
        record["ledger"],
        record["aliases"],
        record["passages"],
    )
    assert result["status"] == record["expected_existing_host_result"]["status"]
    assert result["issue"] == record["expected_existing_host_result"]["issue"]
    assert result["required_aspects"] == record["required_support_aspects"]
    assert result["actual_aspects"] == record["actual_aspects"]
    assert record == before
