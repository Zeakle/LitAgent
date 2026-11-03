"""Expose broad future-input contracts without presenting fixtures as science."""

import copy
import json

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.safety.planned_budget import PlannedBudget
from scripts import review_future_budget_scope as audit


@pytest.fixture
def bundle(case):
    """Reuse the real source constructor and frozen selection identity."""
    return {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "requirements": case.packet["requirements"],
        "methodology": {},
        "archive_sha256": "controlled-archive-identity",
    }


def test_local_validity_does_not_bound_dependent_request_size(bundle):
    original = copy.deepcopy(bundle)
    result = audit.scope_witness(bundle)
    assert result["local_body_schema_and_plan_valid"]
    assert result["local_report_and_binding_valid"]
    assert result["body_units"] == 66
    assert result["characters_per_body_unit"] == 2000
    assert result["actual_card_refs_per_body_unit"] == 1
    assert result["report_native_units"] == 68
    assert result["last_chapter_context_exceeded"]
    assert result["reviewer_necessary_part_context_exceeded"]
    assert not result["inventory_claim_text_schema_has_max_length"]
    assert result["ten_thousand_character_claim_passes_current_size_check"]
    assert not result["can_admit"] and not result["scientific_run"]
    assert result["new_paid_requests"] == 0
    assert bundle == original
    with pytest.raises(ValueError):
        PlannedBudget(result)


def test_scalar_audit_has_no_source_text_or_fabricated_body(bundle):
    encoded = json.dumps(audit.scope_witness(bundle))
    assert "Relation networks learn a comparison function." not in encoded
    assert "Non-scientific scope witness." not in encoded
    assert "source_quote" not in encoded and "xxx" not in encoded


def test_invalid_snapshot_is_rejected_before_scope_construction(bundle):
    bundle["fact_cards_sha256"] = "bad"
    with pytest.raises(ValueError, match="fact_card_identity_conflict"):
        audit.scope_witness(bundle)


def test_fixed_review_binds_actual_bundle_and_relative_source_hashes(
    bundle, monkeypatch
):
    monkeypatch.setattr(audit, "read_bundle", lambda path: copy.deepcopy(bundle))
    result = audit.build_review()
    assert len(result["cases"]) == 3
    assert all(
        row["archive_sha256"] == bundle["archive_sha256"] for row in result["cases"]
    )
    assert all(not p.startswith("/") for p in result["source_hashes"])
    assert result["maximum_tokens"] == 1000000
    assert result["new_paid_requests"] == 0 and not result["can_admit"]
    assert result["identity"] == digest(
        {k: v for k, v in result.items() if k != "identity"}
    )
