"""Keep one reviewed historical cost drift separate from current task recovery."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from litagent.quality import digest as usage_digest
from litagent.safety.planned_accounting import validate_accounting_snapshot
from scripts import run_frozen_writing as campaign

FIXTURE = Path(__file__).parent / "fixtures/run020-historical-cost-pin.json"
FIXTURE_SHA256 = "47d0dd4bd3a988c4ee55f82b3a43ad37885b5c78fa400cd01a4fca56c83d179f"


@pytest.fixture
def reviewed_cost():
    """Load only original cost data; hashes attest separate private file checks."""
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FIXTURE_SHA256
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def binding_for(value):
    """Supply the scalar attestations checked by the real I/O wrapper."""
    metadata = value["provenance"]
    fields = (
        "root",
        "run_id",
        "case_id",
        "profile",
        "anchor_manifest_sha256",
        "anchor_receipt_sha256",
        "manifest_sha256",
        "state_sha256",
        "ledger_sha256",
        "archive_sha256",
    )
    return {
        **{key: metadata[key] for key in fields},
        **value["costs"],
        **{
            key: True
            for key in (
                "caller_manifest_matches",
                "caller_state_matches",
                "caller_row_matches",
                "ledger_equals_snapshot",
                "receipt_history_before_matches",
                "anchor_receipt_matches",
            )
        },
    }


def select(value, binding=None):
    """Exercise the production pin helper without mocking its cost validation."""
    return campaign._historical_cost_plan_pin(
        value["snapshot"],
        value["prepared_plan"],
        binding_for(value) if binding is None else binding,
    )


def test_reviewed_original_costs_are_three_received_usages_not_a_scientific_pass(
    reviewed_cost,
):
    value = reviewed_cost
    assert value["science_gold_standard"] is False
    assert value["provenance"]["cost_metadata_only"] is True
    assert value["prepared_plan"] != value["snapshot"]["plan"]
    entries = list(value["snapshot"]["entries"].values())
    assert len(entries) == len(value["usage"]) == 3
    for entry, usage in zip(entries, value["usage"]):
        assert entry["status"] == "settled" and entry["sent"] is True
        assert entry["usage_sha256"] == usage_digest(
            {key: usage[key] for key in ("prompt_tokens", "completion_tokens")}
        )
        assert entry["actual_tokens"] == usage["total_tokens"]
    assert sum(entry["actual_tokens"] for entry in entries) == 32720
    with pytest.raises(ValueError, match="pinned_plan_conflict"):
        validate_accounting_snapshot(
            value["snapshot"], expected_plan=value["prepared_plan"]
        )


def test_only_the_reviewed_pin_selects_an_immutable_data_plan(reviewed_cost):
    before = copy.deepcopy(reviewed_cost)
    result = select(reviewed_cost)
    assert reviewed_cost == before
    assert result == reviewed_cost["snapshot"]["plan"]
    assert result is not reviewed_cost["snapshot"]["plan"]
    assert result["envelope"] is not reviewed_cost["snapshot"]["plan"]["envelope"]
    assert (
        validate_accounting_snapshot(reviewed_cost["snapshot"], expected_plan=result)
        == reviewed_cost["snapshot"]
    )
    assert not callable(getattr(result, "reserve", None))


def test_pin_helper_does_not_restore_a_budget_or_read_original_files(
    reviewed_cost, monkeypatch
):
    from litagent.safety.budget import CostBudget
    from litagent.safety.planned_budget import PlannedBudget

    def forbidden(*args, **kwargs):
        pytest.fail("a historical cost pin grants no runtime or I/O authority")

    monkeypatch.setattr(CostBudget, "__init__", forbidden)
    monkeypatch.setattr(PlannedBudget, "__init__", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    assert select(reviewed_cost) == reviewed_cost["snapshot"]["plan"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("root", "artifacts/acceptance/a-copy-of-the-original"),
        (
            "root",
            "/Users/ke/Coding/LitAgent/artifacts/acceptance/frozen-writer-three-domain-20261027-020",
        ),
        ("run_id", "another-run"),
        ("case_id", "vit-data-efficient"),
        ("profile", campaign.ROUND_PROFILE),
        *[
            (key, "f" * 64)
            for key in (
                "anchor_manifest_sha256",
                "anchor_receipt_sha256",
                "manifest_sha256",
                "state_sha256",
                "ledger_sha256",
                "archive_sha256",
            )
        ],
        *[
            (key, False)
            for key in (
                "caller_manifest_matches",
                "caller_state_matches",
                "caller_row_matches",
                "ledger_equals_snapshot",
                "receipt_history_before_matches",
                "anchor_receipt_matches",
            )
        ],
        *[
            (key, 1)
            for key in (
                "caller_manifest_matches",
                "caller_state_matches",
                "caller_row_matches",
                "ledger_equals_snapshot",
                "receipt_history_before_matches",
                "anchor_receipt_matches",
            )
        ],
        ("requests", 4),
        ("requests", True),
        ("known_tokens", 32721),
        ("known_tokens", "32720"),
        ("unknown_hold", False),
        ("unknown_hold", 1),
        ("usage_complete", 1),
        ("usage_complete", False),
    ],
)
def test_a_different_file_scope_or_cost_never_uses_the_reviewed_pin(
    reviewed_cost, field, value
):
    binding = binding_for(reviewed_cost)
    binding[field] = value
    before = copy.deepcopy(reviewed_cost)
    with pytest.raises(ValueError, match="frozen_accounting"):
        select(reviewed_cost, binding)
    assert reviewed_cost == before


@pytest.mark.parametrize(
    "fault", ["missing", "extra", "self_pin", "plan", "usage", "entry"]
)
def test_rehashed_or_incomplete_data_is_not_a_new_legacy_exception(
    reviewed_cost, fault
):
    value = copy.deepcopy(reviewed_cost)
    binding = binding_for(value)
    if fault == "missing":
        del binding["anchor_manifest_sha256"]
    elif fault == "extra":
        binding["skip_identity"] = True
    elif fault == "self_pin":
        value["prepared_plan"] = copy.deepcopy(value["snapshot"]["plan"])
    elif fault == "plan":
        value["prepared_plan"]["input_identity"] = "forged-original"
        value["prepared_plan"]["identity"] = usage_digest(
            {k: v for k, v in value["prepared_plan"].items() if k != "identity"}
        )
    elif fault == "usage":
        value["snapshot"]["used_tokens"] += 1
    else:
        entry = copy.deepcopy(next(iter(value["snapshot"]["entries"].values())))
        value["snapshot"]["entries"]["forged-received-request"] = entry
    with pytest.raises(ValueError, match="frozen_accounting"):
        select(value, binding)
