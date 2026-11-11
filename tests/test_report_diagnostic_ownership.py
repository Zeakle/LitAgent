"""Keep global finding owners and authorized report edits distinct from science."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_independent_evaluation import (  # noqa: F401
    global_regions,
    independent_details,
)

from litagent.body_quality import bound_repair_diagnostics, report_repair_diagnostics
from litagent.eval.responsibilities import (
    FORBIDDEN_CODES,
    parse_global_response,
    project_global_schema,
)
from litagent.eval.results import CONTRACT, contract_identity
from litagent.runner import LitAgent


@pytest.mark.parametrize(
    "code",
    [
        "core_coverage",
        "analysis",
        "contradiction",
        "non_atomic",
        "source_fidelity",
        "classification",
    ],
)
def test_duplicate_global_responsibility_is_rejected_by_schema_and_consumer(
    global_regions, code
):
    from litagent.eval.batched import global_review_type

    payload, reply = global_regions
    reply["issues"] = [
        {
            "code": code,
            "anchor_ids": ["claim:claim-one"],
            "explanation": "Positive or duplicated conclusion is not a defect.",
        }
    ]
    schema = project_global_schema(global_review_type(payload).model_json_schema())
    assert not Draft202012Validator(schema).is_valid(reply)
    with pytest.raises(ValueError, match="responsibility_invalid"):
        parse_global_response(json.dumps(reply), payload, {})


def details_for_report(independent_details):
    details = copy.deepcopy(independent_details)
    claim = details["assertions"][0]
    claim.update(
        claim_text="A has the stated property.",
        source_fragment_id="fragment-one",
        source_quote="A has the stated property [E:a].",
        source_start=0,
        source_end=35,
        evidence_ids=["a"],
        explanation="Scientific evidence supports the proposition.",
        reason_code="supported",
    )
    details["stages"]["global_sections"].update(
        issues={"status": "passed"},
        coverage={"status": "passed"},
        limitations={"status": "passed"},
    )
    details["unsupported_claims"] = []
    return details, claim


@pytest.mark.parametrize("code", ["category_count", "repetition"])
def test_real_report_problem_can_target_a_supported_fact(independent_details, code):
    details, claim = details_for_report(independent_details)
    issue = dict(
        owner="report",
        source_unit_id=claim["source_unit_id"],
        claim_ids=[claim["claim_id"]],
        code=code,
        explanation="This same-section list repeats the same category.",
    )
    diag = dict(
        claim,
        reason_code="evaluation_report_" + code,
        explanation=issue["explanation"],
        diagnostic_owner="report",
        diagnostic_source="survey-evaluation-v3",
        alignment_status="exact",
        citation_scope="exact",
    )
    details["global_issues"] = [issue]
    details["unsupported_claims"] = [diag]
    before = copy.deepcopy(details)
    assert report_repair_diagnostics(details) == [diag]
    assert bound_repair_diagnostics(details) == [diag]
    assert details == before and claim["status"] == "supported"


@pytest.mark.parametrize(
    "fault",
    [
        "unbound_issue",
        "wrong_reason",
        "wrong_location",
        "wrong_refs",
        "unknown_claim",
        "region_error",
        "inventory_failed",
        "old_identity",
    ],
)
def test_untrusted_report_diagnostic_does_not_authorize_supported_changes(
    independent_details, fault
):
    details, claim = details_for_report(independent_details)
    issue = dict(
        owner="report",
        source_unit_id="unit-one",
        claim_ids=["claim-one"],
        code="repetition",
        explanation="Repeated fact.",
    )
    diag = dict(
        claim,
        reason_code="evaluation_report_repetition",
        explanation="Repeated fact.",
        diagnostic_owner="report",
        diagnostic_source="survey-evaluation-v3",
        alignment_status="exact",
        citation_scope="exact",
    )
    details.update(global_issues=[issue], unsupported_claims=[diag])
    if fault == "unbound_issue":
        details["global_issues"] = []
    elif fault == "wrong_reason":
        diag["reason_code"] = "evaluation_report_core_coverage"
    elif fault == "wrong_location":
        diag["source_start"] = 1
    elif fault == "wrong_refs":
        diag["evidence_ids"] = ["other-paper"]
    elif fault == "unknown_claim":
        diag["claim_id"] = "unknown"
    elif fault == "region_error":
        details["stages"]["global_sections"]["issues"]["status"] = "error"
    elif fault == "inventory_failed":
        details["inventory_check"]["status"] = "failed"
    elif fault == "old_identity":
        details["evaluation_result_identity"] = "old"
    assert not report_repair_diagnostics(details)
    # Raw diagnostics remain visible; denial must not silently erase observations.
    assert diag in bound_repair_diagnostics(details)


@pytest.mark.parametrize("answered,expected", [(True, False), (False, True)])
def test_only_negative_coverage_owns_coverage_repair(
    independent_details, answered, expected
):
    details, claim = details_for_report(independent_details)
    details["core_coverage"][0].update(
        answered=answered,
        explanation="The requested comparison dimension needs expansion.",
    )
    rows = report_repair_diagnostics(details)
    assert bool(rows) == expected
    if expected:
        assert (
            rows[0]["claim_id"] == claim["claim_id"]
            and rows[0]["preserve_supported_claim"]
        )
    # A negative host dependency after a positive model conclusion is not missing prose.
    claim["status"] = "unverifiable"
    assert not report_repair_diagnostics(details)


@pytest.mark.parametrize("model_adequate,expected", [(True, False), (False, True)])
def test_specific_limitations_need_a_real_negative_model_finding(
    independent_details, model_adequate, expected
):
    details, claim = details_for_report(independent_details)
    details["limitations_check"].update(
        adequate=False,
        model_adequate=model_adequate,
        explanation="The concrete trade-off needs explanation.",
    )
    rows = report_repair_diagnostics(details)
    assert bool(rows) == expected
    if rows:
        assert rows[0]["reason_code"] == "evaluation_report_limitations"


def test_old_067_is_not_regraded_or_silently_fixed():
    path = Path(
        "artifacts/acceptance/frozen-writer-single-20261105-067/writer-stage/result.json"
    )
    if not path.exists():
        pytest.skip("Private real-run artifact is not distributed")
    raw = path.read_bytes()
    value = json.loads(raw)
    archive = Path(value["artifact"])
    archive_raw = archive.read_bytes()
    report = json.loads(archive_raw)["report"]
    assert report["delivery"]["publishable"] is False
    assert report["evaluation"]["faithfulness"]["score"] is None
    assert not LitAgent._completed_bound_repair(
        report["evaluation"], report["survey"], allow_reordered_batches=True
    )
    assert path.read_bytes() == raw and archive.read_bytes() == archive_raw


def test_conflicts_keep_their_own_bound_repair_targets(independent_details):
    details, claim = details_for_report(independent_details)
    other = {**claim, "claim_id": "claim-two", "source_unit_id": "unit-two"}
    details["assertions"].append(other)
    details["conflicts"] = [
        {
            "claim_a": claim["claim_id"],
            "claim_b": other["claim_id"],
            "reason": "The same metric and setting have incompatible conclusions.",
        }
    ]
    rows = report_repair_diagnostics(details)
    assert {d["claim_id"] for d in rows} == {"claim-one", "claim-two"}
    assert all(
        d["reason_code"] == "evaluation_report_contradiction"
        and d["statement_id"] == d["claim_id"]
        for d in rows
    )
    details["stages"]["global_sections"]["conflicts"]["status"] = "error"
    assert not report_repair_diagnostics(details)


def test_same_supported_anchor_keeps_report_and_analysis_defects(independent_details):
    details, claim = details_for_report(independent_details)
    claim["section"] = "Introduction"
    details["body_analysis_contract"] = "body-analysis-v1"
    details["analysis_check"] = {
        "status": "failed",
        "sections": [
            {
                "section_key": "introduction",
                "adequate": False,
                "model_adequate": False,
                "claim_ids": [claim["claim_id"]],
                "explanation": "The introduction needs a specific scoped comparison.",
            }
        ],
    }
    issue = {
        "owner": "report",
        "source_unit_id": claim["source_unit_id"],
        "claim_ids": [claim["claim_id"]],
        "code": "repetition",
        "explanation": "The finding is repeated within this chapter.",
    }
    diag = {
        **claim,
        "reason_code": "evaluation_report_repetition",
        "explanation": issue["explanation"],
        "diagnostic_owner": "report",
        "diagnostic_source": "survey-evaluation-v3",
        "alignment_status": "exact",
        "citation_scope": "exact",
    }
    details.update(global_issues=[issue], unsupported_claims=[diag])
    result = bound_repair_diagnostics(details)
    assert {d["reason_code"] for d in result} == {
        "evaluation_report_repetition",
        "body_analysis_incomplete",
    }
    assert len(result) == 2
