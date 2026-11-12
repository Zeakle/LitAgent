"""Validate mixed concerns without treating evaluator diagnoses as evidence."""

import copy

import pytest
from test_inventory_recovery import diagnostic_fixture

from litagent.eval.inventory_recovery import (
    CONTRACT,
    MIXED_CONTRACT,
    inventory_identity,
    inventory_recovery_feedback,
    recovery_instruction,
    validate_recovered_inventory,
)


def mixed_fixture():
    """Keep exact, distinct host locations and a settled transport boundary."""
    survey, details = diagnostic_fixture()
    for claim in details["assertions"]:
        claim["correct_kind"] = "scientific"
    details["assertions"][0]["classification_valid"] = False
    details["assertions"][1]["source_faithful"] = False
    details["stages"].update(
        requests=[{"status": "response_received", "usage": {"usage_available": True}}],
        fragment_inventory=[
            {
                "source_unit_id": c["source_unit_id"],
                "source_fragment_id": c["source_fragment_id"],
                "assertion_count": 1,
                "exclusion_reason": "",
            }
            for c in details["assertions"]
        ],
    )
    details["global_issues"] = [
        {
            "owner": "inventory",
            "code": "missing_assertion",
            "source_unit_id": details["assertions"][0]["source_unit_id"],
            "claim_ids": [],
            "explanation": "All assertions were extracted; no omission was found.",
        },
        {
            "owner": "inventory",
            "code": "source_fidelity",
            "source_unit_id": details["assertions"][1]["source_unit_id"],
            "claim_ids": [details["assertions"][1]["claim_id"]],
            "explanation": "The extraction matches the original with no drift.",
        },
    ]
    return survey, details


def test_explicit_mixed_feedback_retains_contradictory_concerns_and_all_flags():
    survey, details = mixed_fixture()
    before = copy.deepcopy(details)
    assert inventory_recovery_feedback(survey, details) is None
    feedback = inventory_recovery_feedback(survey, details, contract=MIXED_CONTRACT)
    assert feedback["contract"] == MIXED_CONTRACT
    assert details == before
    assert len(feedback["findings"]) == 6
    assert [f["explanation"] for f in feedback["findings"][:2]] == [
        i["explanation"] for i in details["global_issues"]
    ]
    assert {f["origin"] for f in feedback["findings"]} == {"batch", "global"}
    assert feedback["findings"][0]["claim_ids"] == []
    assert len(feedback["findings"][0]["source_fragment_ids"]) == 1
    assert [c["evidence_ids"] for c in feedback["concern_claims"]] == [["a"], ["b"]]
    assert feedback["concern_claims"][0]["classification_valid"] is False
    assert feedback["concern_claims"][1]["source_faithful"] is False
    assert feedback["previous_inventory_sha256"] == inventory_identity(
        details["assertions"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "sha",
        "duplicate",
        "source_start",
        "source_unit_id",
        "source_fragment_id",
        "source_quote",
        "evidence_ids",
        "kind",
        "correct_kind",
        "atomic",
        "classification_valid",
        "source_faithful",
        "explanation",
        "unit_missing",
        "unit_duplicate",
        "count",
        "unknown_code",
        "unknown_claim",
        "foreign_unit",
        "usage",
        "pending",
        "requests_missing",
        "claims_container",
        "fragment_container",
    ],
)
def test_forged_or_unsettled_basis_cannot_authorize_recovery(fault):
    survey, d = mixed_fixture()
    if fault == "sha":
        d["source_report_sha256"] = "0" * 64
    elif fault == "duplicate":
        d["assertions"][1]["claim_id"] = d["assertions"][0]["claim_id"]
    elif fault in {
        "source_start",
        "source_unit_id",
        "source_fragment_id",
        "source_quote",
        "evidence_ids",
        "kind",
        "correct_kind",
        "atomic",
        "classification_valid",
        "source_faithful",
        "explanation",
    }:
        d["assertions"][0][fault] = None
    elif fault == "unit_missing":
        d["stages"]["fragment_inventory"].pop()
    elif fault == "unit_duplicate":
        d["stages"]["fragment_inventory"].append(d["stages"]["fragment_inventory"][0])
    elif fault == "count":
        d["stages"]["fragment_inventory"][0]["assertion_count"] = 0
    elif fault == "unknown_code":
        d["global_issues"][0]["code"] = "guess"
    elif fault == "unknown_claim":
        d["global_issues"][1]["claim_ids"] = ["claim_missing"]
    elif fault == "foreign_unit":
        d["global_issues"][1]["source_unit_id"] = d["assertions"][0]["source_unit_id"]
    elif fault == "usage":
        d["stages"]["requests"][0]["usage"]["usage_available"] = False
    elif fault == "pending":
        d["stages"]["requests"][0]["status"] = "started_no_replay"
    elif fault == "requests_missing":
        del d["stages"]["requests"]
    elif fault == "claims_container":
        d["assertions"] = None
    else:
        d["stages"]["fragment_inventory"] = [None]
    with pytest.raises(ValueError, match="identity_invalid"):
        inventory_recovery_feedback(survey, d, contract=MIXED_CONTRACT)


def test_only_unresolved_inventory_is_eligible_and_unknown_revision_is_rejected():
    survey, d = mixed_fixture()
    for key, value in [
        ("assessment_stage", "started"),
        ("inventory_check", {"status": "passed"}),
    ]:
        changed = {**d, key: value}
        assert (
            inventory_recovery_feedback(survey, changed, contract=MIXED_CONTRACT)
            is None
        )
    with pytest.raises(ValueError, match="policy_invalid"):
        inventory_recovery_feedback(survey, d, contract="future")
    assert "non-atomic assertions" in recovery_instruction(CONTRACT)
    assert "UNRESOLVED" in recovery_instruction(MIXED_CONTRACT)


def test_identity_only_renaming_or_cycling_is_not_progress():
    _, details = mixed_fixture()
    claims = details["assertions"]
    feedback = {
        "contract": MIXED_CONTRACT,
        "previous_inventory_sha256": inventory_identity(claims),
    }
    renamed = [{**c, "claim_id": "renamed_" + c["claim_id"]} for c in claims]
    with pytest.raises(ValueError, match="no_progress"):
        validate_recovered_inventory(renamed, feedback)
    changed = [
        {**c, "claim_text": c["claim_text"] + " Different condition."} for c in claims
    ]
    validate_recovered_inventory(changed, feedback)
    cycle = {
        **feedback,
        "previous_inventory_sha256": inventory_identity(changed),
        "seen_inventory_sha256": [inventory_identity(claims)],
    }
    with pytest.raises(ValueError, match="no_progress"):
        validate_recovered_inventory(claims, cycle)
