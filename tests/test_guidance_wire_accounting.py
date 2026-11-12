"""Check historical wire tags as pinned data without executable codec restore."""

import copy

import pytest
from test_structured_repair_identity import guidance_envelope, rehash

from litagent.safety.planned_accounting import _envelope

REVISION = "host-unit-ref-guidance-v1"
FIELDS = {"repair_guidance_wire_revision", "repair_guidance_wire_identity"}


def pinned_scope(**changes):
    """Use an authentic parent envelope and an explicitly pinned identity pair."""
    return rehash(
        {
            **guidance_envelope(),
            "repair_guidance_wire_revision": REVISION,
            "repair_guidance_wire_identity": "a" * 64,
            **changes,
        }
    )


def test_old_guidance_envelope_remains_unchanged_and_has_no_wire_markers():
    """Missing selection retains the prior shape rather than implicit promotion."""
    value = guidance_envelope()
    before = copy.deepcopy(value)
    assert not FIELDS.intersection(value)
    assert _envelope(value) is not None
    assert value == before


def test_pinned_fee_shape_does_not_execute_current_codec_or_budget_owner(monkeypatch):
    """An old receipt remains readable after today's implementation changes."""
    import litagent.safety.planned_budget as planned
    import litagent.writing_flat as flat

    value = pinned_scope()
    before = copy.deepcopy(value)

    def forbidden(*args, **kwargs):
        raise AssertionError("cost_data_must_not_execute_runtime_codec_or_owner")

    monkeypatch.setattr(flat, "repair_guidance_wire_identity", forbidden, raising=False)
    monkeypatch.setattr(flat, "encode_repair_guidance_wire", forbidden, raising=False)
    monkeypatch.setattr(flat, "restore_repair_guidance_wire", forbidden, raising=False)
    monkeypatch.setattr(planned, "validate_writer_envelope", forbidden)
    monkeypatch.setattr(planned.PlannedBudget, "__init__", forbidden)
    monkeypatch.setattr(planned.PlannedBudget, "from_snapshot", forbidden)
    assert _envelope(value) is not None
    assert value == before


@pytest.mark.parametrize("missing", sorted(FIELDS))
def test_fee_reader_rejects_half_pair_even_with_recomputed_outer_hash(missing):
    value = pinned_scope()
    value.pop(missing)
    rehash(value)
    with pytest.raises(ValueError, match="envelope_repair_guidance_wire_invalid"):
        _envelope(value)


@pytest.mark.parametrize(
    "change",
    [
        {"repair_guidance_wire_revision": None},
        {"repair_guidance_wire_revision": False},
        {"repair_guidance_wire_revision": 1},
        {"repair_guidance_wire_revision": "future"},
        {"repair_guidance_wire_identity": None},
        {"repair_guidance_wire_identity": True},
        {"repair_guidance_wire_identity": "F" * 64},
        {"repair_guidance_wire_identity": "a" * 63},
        {"repair_guidance_wire_identity": "not-a-hash"},
    ],
)
def test_fee_reader_rejects_illegal_wire_pair_values(change):
    value = pinned_scope(**change)
    before = copy.deepcopy(value)
    with pytest.raises(ValueError, match="envelope_repair_guidance_wire_invalid"):
        _envelope(value)
    assert value == before


@pytest.mark.parametrize(
    "missing",
    [
        "repair_guidance_contract",
        "repair_guidance_identity",
        "repair_guidance_json_bytes",
        "repair_mapping_contract",
        "source_map_identity",
        "capacity",
        "support_validation_revision",
        "classification_review_contract",
    ],
)
def test_wire_pair_does_not_bypass_original_guidance_parent_checks(missing):
    value = pinned_scope()
    value.pop(missing)
    rehash(value)
    with pytest.raises(ValueError):
        _envelope(value)


def test_wire_tag_cannot_be_added_to_an_unselected_old_envelope():
    value = guidance_envelope()
    for name in (
        "repair_guidance_contract",
        "repair_guidance_identity",
        "repair_guidance_json_bytes",
    ):
        value.pop(name)
    value.update(
        repair_guidance_wire_revision=REVISION,
        repair_guidance_wire_identity="a" * 64,
    )
    rehash(value)
    with pytest.raises(ValueError, match="envelope_repair_guidance_wire_invalid"):
        _envelope(value)
