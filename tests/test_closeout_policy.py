"""Keep the release policy bounded without changing historical evaluation rules."""

import copy

import pytest
from test_evaluation_dimensions_identity import dimension_envelope
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.closeout import (
    PROFILE,
    profile_fields,
    profile_flags,
    recheck_prompt,
    recovery_allowed,
    result_summary,
)
from litagent.eval.inventory_recovery import (
    DEFINITION_RECOVERY_INSTRUCTION,
    MIXED_CONTRACT,
    inventory_identity,
    validate_recovered_inventory,
)
from litagent.frozen_writing import plan_frozen_budget
from litagent.safety.planned_accounting import _plan


def scope():
    """Construct the complete real envelope, including both independent parents."""
    return dimension_envelope(
        writer_source_view="source-only-writer-v1",
        semantic_conflict_policy="host-unverified-continue-v1",
        closeout_profile=PROFILE,
    )


@pytest.mark.parametrize("bad", ["", False, 1, {}, "unknown"])
def test_unknown_closeout_profile_cannot_select_an_execution_policy(bad):
    with pytest.raises(ValueError, match="closeout_profile_invalid"):
        profile_flags(bad)


def test_closeout_plan_uses_complete_existing_stack_and_distinct_identity(
    frozen_bundle,
):
    flags = profile_flags(PROFILE)
    plan = plan_frozen_budget(frozen_bundle, budget_policy="planned-budget-v7", **flags)
    old = plan_frozen_budget(
        frozen_bundle,
        budget_policy="planned-budget-v7",
        **{k: v for k, v in flags.items() if k != "closeout_profile"},
    )
    assert plan["status"] == "admitted"
    assert plan["identity"] != old["identity"]
    assert profile_fields(plan["envelope"])["closeout_profile"] == PROFILE
    assert profile_fields(old["envelope"]) == {}
    assert plan["maximum_tokens"] == 4500000
    assert plan["maximum_requests"] == 64
    assert plan["repair_limit"] == 2
    _plan(plan)
    with pytest.raises(ValueError, match="closeout_joint_budget_required"):
        plan_frozen_budget(
            frozen_bundle,
            budget_policy="planned-budget-v7",
            **{**flags, "round_resource_contract": None},
        )


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "started"])
def test_sent_inventory_attempt_is_not_released_by_failure_or_new_report(status):
    envelope = scope()
    state = {"attempts": []}
    assert recovery_allowed(envelope, state)
    state["attempts"].append(
        {"purpose": MIXED_CONTRACT, "status": status, "source_report_sha256": "old"}
    )
    assert not recovery_allowed(envelope, state)
    state["attempts"][0]["source_report_sha256"] = "new"
    assert not recovery_allowed(envelope, state)


def test_two_prose_attempts_do_not_count_as_inventory_recovery():
    assert recovery_allowed(scope(), {"attempts": [{}, {}]})
    assert recovery_allowed(None, {})


@pytest.mark.parametrize("key", ["closeout_profile", "closeout_profile_identity"])
def test_incomplete_profile_pair_is_rejected(key):
    envelope = scope()
    envelope.pop(key)
    with pytest.raises(ValueError, match="closeout_profile_invalid"):
        profile_fields(envelope)


def test_unchanged_inventory_can_be_rechecked_only_on_explicit_new_path():
    claims = [
        dict(
            source_fragment_id="f",
            kind="scientific",
            claim_text="A relates to B.",
            evidence_ids=["e"],
        )
    ]
    feedback = {
        "contract": MIXED_CONTRACT,
        "previous_inventory_sha256": inventory_identity(claims),
        "seen_inventory_sha256": [],
    }
    with pytest.raises(ValueError, match="inventory_recovery_no_progress"):
        validate_recovered_inventory(claims, feedback)
    before = copy.deepcopy(claims)
    validate_recovered_inventory(claims, feedback, allow_unchanged=True)
    assert claims == before


def test_new_recheck_prompt_does_not_change_the_historical_prompt():
    assert (
        recheck_prompt(DEFINITION_RECOVERY_INSTRUCTION, None)
        == DEFINITION_RECOVERY_INSTRUCTION
    )
    current = recheck_prompt(DEFINITION_RECOVERY_INSTRUCTION, scope())
    assert "cycling inventories is no progress" not in current
    assert "single inventory-recovery slot" in current
    assert "fresh independent assessment" in current


@pytest.mark.parametrize(
    "invalid", ["skipped", "partial", "inventory", "global", "overall"]
)
def test_incomplete_evaluation_is_never_presented_as_completed(invalid):
    metric = {
        "skipped": False,
        "score": 0.5,
        "details": {
            "coverage": {"complete": True},
            "inventory_check": {"status": "passed"},
            "stages": {"global_check": "completed"},
        },
    }
    assert result_summary({"evaluation": {"faithfulness": metric}})["evaluation_valid"]
    if invalid == "skipped":
        metric["skipped"] = True
    elif invalid == "partial":
        metric["details"]["coverage"]["complete"] = False
    elif invalid == "inventory":
        metric["details"]["inventory_check"]["status"] = "failed"
    elif invalid == "global":
        metric["details"]["stages"]["global_check"] = "invalid_response"
    else:
        metric["details"]["overall_valid"] = False
    summary = result_summary({"evaluation": {"faithfulness": metric}})
    assert summary["evaluation"] == "not_complete"
    assert not summary["evaluation_valid"]
