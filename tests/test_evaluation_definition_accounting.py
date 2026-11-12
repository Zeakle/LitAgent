"""Keep definition selection explicit and historical fees independent of code."""

import copy

import pytest
from pydantic import ValidationError
from test_host_lossless_identity import selected_envelope as parent_envelope
from test_host_lossless_identity import selected_options as parent_options
from test_structured_repair_identity import guidance_envelope, rehash

from litagent.safety.planned_accounting import _envelope

V1 = "host-support-receipts-v1"
V2 = "host-support-receipts-v2"
OLD_NONE_FINGERPRINT = (
    "cba961a2fef5a7e5baf85203cb0ec70efa5381d5a67096d8603a341233417c5c"
)
OLD_V1_FINGERPRINT = "d02c1e2dedce250e1f0ebdf57af1a407f938b000f730d4986081bd5745a270d2"
OLD_V1_IDENTITY = "1cafb459f74f09ce5276aacebabc85bb8dbac3ee092ace9de01aa3039313f607"


def options(revision, **changes):
    """Select a revision using all established parents and the actual model."""
    return parent_options(**{"support_validation_revision": revision, **changes})


def pinned_scope(**changes):
    """Model pinned metadata rather than granting runtime execution authority."""
    return rehash(
        {
            **guidance_envelope(),
            "support_validation_revision": V2,
            "support_validation_identity": "e" * 64,
            **changes,
        }
    )


def test_existing_workbench_fingerprints_equal_pre_edit_values():
    """The new allowed value changes neither absent nor explicitly old inputs."""
    from litagent.eval.scoped import support_validation_identity

    assert parent_options().fingerprint() == OLD_NONE_FINGERPRINT
    assert options(None).fingerprint() == OLD_NONE_FINGERPRINT
    assert options(V1).fingerprint() == OLD_V1_FINGERPRINT
    assert support_validation_identity() == OLD_V1_IDENTITY
    assert "support_validation_revision" not in options(None).model_dump(mode="json")


def test_actual_revision_binds_options_without_promoting_the_old_default(monkeypatch):
    """Use the old zero-argument API for v1 and the actual v2 for its identity."""
    import litagent.eval.scoped as scoped

    original = scoped.support_validation_identity
    calls = []

    def observed(*args):
        calls.append(args)
        return original(*args)

    monkeypatch.setattr(scoped, "support_validation_identity", observed)
    assert parent_options().fingerprint() == OLD_NONE_FINGERPRINT
    assert not calls
    assert options(V1).fingerprint() == OLD_V1_FINGERPRINT
    assert calls[-1] == ()
    boundary = len(calls)
    assert options(V2).fingerprint() not in {OLD_NONE_FINGERPRINT, OLD_V1_FINGERPRINT}
    assert calls[boundary] == (V2,)
    assert options(V2).model_dump(mode="json")["support_validation_revision"] == V2


@pytest.mark.parametrize("wrong", [True, False, 0, 1, "", "future", [], {}])
def test_options_reject_unknown_or_coerced_definition_values(wrong):
    with pytest.raises(ValidationError):
        options(wrong)


@pytest.mark.parametrize(
    "change",
    [
        {"budget_policy": "planned-budget-v6"},
        {"generation_strategy": "body_summary_v2"},
        {"classification_review": False},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"transport_revision": None},
    ],
)
def test_new_definition_retains_every_existing_options_parent(change):
    with pytest.raises(ValueError, match="support_validation_requires_explicit_v7"):
        options(V2, **change)


def test_definition_revision_does_not_add_a_public_survey_field():
    from litagent.api import SurveyRequest

    assert "support_validation_revision" not in SurveyRequest.model_fields
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SurveyRequest(
            query="Compare the public methods.", support_validation_revision=V2
        )


def test_fee_reader_keeps_none_and_both_pinned_revisions_without_mutation():
    """Plausible pinned shape is data, not a current semantic approval."""
    for value in (parent_envelope(), guidance_envelope(), pinned_scope()):
        before = copy.deepcopy(value)
        assert _envelope(value) is not None
        assert value == before


def test_pinned_v2_fees_do_not_execute_current_definitions_or_restore_owners(
    monkeypatch,
):
    import litagent.eval.scoped as scoped
    import litagent.safety.budget as budget
    import litagent.safety.planned_budget as planned

    value = pinned_scope()
    before = copy.deepcopy(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("pinned_cost_must_not_execute_semantics_or_owner")

    for name in (
        "support_validation_identity",
        "_check_support_revision",
        "_support_revision",
        "evaluation_definition_prompt",
        "evaluation_definition_schema",
    ):
        monkeypatch.setattr(scoped, name, forbidden, raising=False)
    monkeypatch.setattr(planned, "validate_writer_envelope", forbidden)
    monkeypatch.setattr(planned.PlannedBudget, "__init__", forbidden)
    monkeypatch.setattr(planned.PlannedBudget, "from_snapshot", forbidden)
    monkeypatch.setattr(budget.CostBudget, "__init__", forbidden)
    monkeypatch.setattr(budget.CostBudget, "from_snapshot", forbidden)
    assert _envelope(value) is not None
    assert value == before


@pytest.mark.parametrize(
    "change",
    [
        {"support_validation_revision": None},
        {"support_validation_revision": False},
        {"support_validation_revision": 1},
        {"support_validation_revision": ""},
        {"support_validation_revision": "future"},
        {"support_validation_revision": []},
        {"support_validation_revision": {}},
        {"support_validation_identity": None},
        {"support_validation_identity": True},
        {"support_validation_identity": "E" * 64},
        {"support_validation_identity": "e" * 63},
    ],
)
def test_fee_reader_rejects_illegal_pairs_without_current_code(change):
    value = pinned_scope(**change)
    before = copy.deepcopy(value)
    with pytest.raises(ValueError, match="planned_accounting_"):
        _envelope(value)
    assert value == before


@pytest.mark.parametrize(
    "missing",
    [
        "support_validation_revision",
        "support_validation_identity",
        "capacity",
        "classification_review_contract",
        "inventory_recovery_contract",
        "repair_feedback_contract",
        "reference_contract",
        "transport_revision",
        "transport_identity",
    ],
)
def test_pinned_v2_cannot_bypass_complete_parent_or_pair_checks(missing):
    value = pinned_scope()
    value.pop(missing)
    rehash(value)
    with pytest.raises(ValueError, match="planned_accounting_"):
        _envelope(value)
