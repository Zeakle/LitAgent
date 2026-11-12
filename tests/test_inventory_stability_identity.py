"""Isolate stable recovery identities from historical mixed-inventory choices."""

import pytest
from test_host_lossless_identity import selected_options as legacy_options
from test_support_receipt_identity import FLAGS, selected_options
from test_writer_v2 import case  # noqa: F401

from litagent import writing_flat
from litagent.eval import inventory_recovery
from litagent.frozen_writing import validate_frozen_writer_plan
from litagent.writing_v2 import WriterError, writer_budget_plan


def plan(case, *, stable):
    """Use the real planner and the existing server-owned policy selector."""
    return writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **FLAGS,
        **(
            {"support_validation_revision": "host-support-receipts-v1"}
            if stable
            else {}
        ),
    )


@pytest.mark.parametrize(
    ("owner", "field"),
    [
        (inventory_recovery, "MIXED_STABILITY_INSTRUCTION"),
        (writing_flat, "INVENTORY_STABILITY_TRANSPORT_INSTRUCTION"),
    ],
    ids=["recovery_scope", "lossless_transport"],
)
def test_stability_rules_bind_only_selected_run_and_budget_identities(
    case, monkeypatch, owner, field
):
    """A new recovery rule cannot silently upgrade old None run identities."""
    old_options = legacy_options()
    new_options = selected_options()
    old_fp, new_fp = old_options.fingerprint(), new_options.fingerprint()
    old_plan, new_plan = plan(case, stable=False), plan(case, stable=True)
    old_identity = inventory_recovery.mixed_inventory_recovery_identity()
    selected_identity = inventory_recovery.mixed_inventory_recovery_identity(
        stable=True
    )
    assert selected_identity != old_identity
    monkeypatch.setattr(
        owner, field, getattr(owner, field) + "\nChanged selected rule."
    )
    assert inventory_recovery.mixed_inventory_recovery_identity() == old_identity
    assert old_options.fingerprint() == old_fp
    assert plan(case, stable=False) == old_plan
    assert new_options.fingerprint() != new_fp
    candidate = plan(case, stable=True)
    assert candidate["identity"] != new_plan["identity"]
    assert candidate["input_identity"] == new_plan["input_identity"]
    with pytest.raises(WriterError):
        validate_frozen_writer_plan(new_plan, candidate)


def test_stable_recovery_does_not_add_public_option_fields():
    """The internal scope contract is selected through the locked server profile."""
    from litagent.api import SurveyRequest

    assert "stable" not in SurveyRequest.model_fields
    assert "preserved_fragments" not in SurveyRequest.model_fields
    old, new = legacy_options(), selected_options()
    assert old.support_validation_revision is None
    assert new.support_validation_revision == "host-support-receipts-v1"
    assert "preserved_fragments" not in new.model_dump(mode="json")
