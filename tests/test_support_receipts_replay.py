"""Replay actual 038 counterexamples without turning them into a gold standard."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from litagent.eval.scoped import (
    PassageDecision,
    required_support_aspects,
    validate_assessment_decisions,
    validate_support_basis,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures/run038-support-receipts.json"
FIXTURE_SHA256 = "335e2eed91939af54d944e690d5ab89b0f9e4735438b59badd1d82e19a8b5b65"
REVISION = "host-support-receipts-v1"


@pytest.fixture(scope="module")
def real038():
    """Use full selected originals and actual original locations, not summaries."""
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == FIXTURE_SHA256
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def check_case(fixture, row, *, revision=None, assertion=None):
    """Replay the host guard while preserving the original model decision."""
    return validate_support_basis(
        assertion or row["assertion"],
        PassageDecision.model_validate(row["raw_decision"]),
        fixture["ledger"],
        fixture["aliases"],
        passages=fixture["support_passages"],
        original_only=True,
        local_numeric_bindings=True,
        chinese_comparatives=True,
        applicability=True,
        metric_inflections=True,
        protocol_numeric_scope=True,
        **({"support_validation_revision": revision} if revision else {}),
    )


def test_real038_all48_original_locations_and_all20_sources_are_preserved(real038):
    assert real038["science_gold_standard"] is False
    assert len(real038["ledger"]) == len(real038["aliases"]) == 20
    assert len(real038["cases"]) == 48
    assert Counter(row["archived_status"] for row in real038["cases"]) == {
        "unverifiable": 18,
        "unsupported": 30,
    }
    for row in real038["cases"]:
        claim = row["assertion"]
        assert (
            real038["original_survey"][claim["source_start"] : claim["source_end"]]
            == claim["source_quote"]
        )
        for eid in claim["evidence_ids"]:
            assert real038["ledger"][eid]["supporting_text"]
    assert hashlib.sha256(real038["original_survey"].encode()).hexdigest() == (
        real038["source_report_sha256"]
    )


def test_real038_default_consumer_keeps_original18_and30_verdicts(real038):
    before = copy.deepcopy(real038)
    claims = [row["assertion"] for row in real038["cases"]]
    decisions = [
        PassageDecision.model_validate(row["raw_decision"]) for row in real038["cases"]
    ]
    actual = validate_assessment_decisions(
        claims,
        decisions,
        ledger=real038["ledger"],
        aliases=real038["aliases"],
        support_passages=real038["support_passages"],
        context={
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "review_methodology": True,
            "_independent_classification": True,
            "_evaluation_progress": {
                "global_findings_contract": "anchored-global-findings-v2"
            },
        },
        strict=True,
    )
    assert len(actual) == 48
    for checked, row in zip(actual, real038["cases"], strict=True):
        assert checked["claim_id"] == row["assertion"]["claim_id"]
        assert checked["status"] == row["archived_status"]
        assert checked["reason_code"] == row["archived_reason_code"]
        assert checked["local_error"] == row["archived_local_error"]
    assert real038 == before


def test_real038_all18_old_host_errors_remain_exactly_reproducible(real038):
    errors = Counter()
    for row in real038["cases"]:
        if row["archived_status"] != "unverifiable":
            continue
        checked = check_case(real038, row)
        assert checked == row["archived_support_validation"]
        errors[checked["issue"]] += 1
    assert errors == {
        "support_entity_unverified": 7,
        "support_basis_ownership_invalid": 7,
        "support_number_unverified": 3,
        "support_aspect_coverage_invalid": 1,
    }


def test_real038_new_receipts_never_invent_entity_aliases_or_citation_rights(real038):
    for row in real038["cases"]:
        if row["archived_local_error"] not in {
            "support_entity_unverified",
            "support_basis_ownership_invalid",
        }:
            continue
        checked = check_case(real038, row, revision=REVISION)
        assert checked["issue"] == row["archived_local_error"]
        assert checked["status"] != "passed"


@pytest.mark.parametrize("claim_id", ["claim_106", "claim_111"])
def test_real038_explicit_absent_shot_results_are_not_positive_result_counts(
    real038, claim_id
):
    row = next(r for r in real038["cases"] if r["assertion"]["claim_id"] == claim_id)
    assert check_case(real038, row)["issue"] == "support_number_unverified"
    checked = check_case(real038, row, revision=REVISION)
    assert checked.get("issue") != "support_number_unverified"
    # A local role fix is not a new model judgment or complete scientific score.
    assert row["archived_status"] == "unverifiable"


def test_real038_causal_connector_uses_one_shared_producer_consumer_plan(real038):
    row = next(r for r in real038["cases"] if r["assertion"]["claim_id"] == "claim_109")
    text = row["assertion"]["claim_text"]
    assert "causal_relation" not in required_support_aspects(text, applicability=True)
    assert "causal_relation" in required_support_aspects(
        text, applicability=True, support_validation_revision=REVISION
    )
    assert check_case(real038, row)["issue"] == "support_aspect_coverage_invalid"
    assert check_case(real038, row, revision=REVISION).get("issue") != (
        "support_aspect_coverage_invalid"
    )


@pytest.mark.parametrize("claim_id", ["claim_4", "claim_61", "claim_105", "claim_144"])
def test_real038_fixing_first_entity_guard_cannot_hide_percentage_point_risk(
    real038, claim_id
):
    row = next(r for r in real038["cases"] if r["assertion"]["claim_id"] == claim_id)
    assert row["archived_local_error"] == "support_entity_unverified"
    changed = copy.deepcopy(row["assertion"])
    changed["claim_text"] = changed["claim_text"].replace(
        "miniImageNet", "miniImagenNet"
    )
    assert changed["claim_text"] != row["assertion"]["claim_text"]
    checked = check_case(real038, row, revision=REVISION, assertion=changed)
    assert checked["status"] != "passed"


def test_real038_actual_percentage_points_and_wrong_paper_stay_negative(real038):
    point = next(
        r for r in real038["cases"] if r["assertion"]["claim_id"] == "claim_130"
    )
    assert check_case(real038, point, revision=REVISION)["status"] != "passed"
    wrong = next(
        r for r in real038["cases"] if r["assertion"]["claim_id"] == "claim_69"
    )
    assert wrong["raw_decision"]["status"] == "unsupported"
    assert wrong["raw_decision"]["reason_code"] == "attribution_mismatch"
    assert not wrong["raw_decision"]["support_basis"]
    assert wrong["assertion"]["evidence_ids"]
