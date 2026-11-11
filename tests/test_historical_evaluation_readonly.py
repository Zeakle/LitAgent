"""Read historical scores through production projections without creating new gates."""

import copy

import pytest
from test_independent_evaluation import independent_details  # noqa: F401

from litagent.benchmark.quality_suite import evaluation_execution
from litagent.contracts import normalize_survey_result
from litagent.eval.results import (
    HISTORICAL_RESULT_IDENTITIES,
    EvaluationRunResult,
    historical_checks,
    public_summary,
)


def archived_report(details):
    """Make the stored score deliberately differ from recalculation to detect regrading."""
    details = copy.deepcopy(details)
    details["evaluation_result_identity"] = next(iter(HISTORICAL_RESULT_IDENTITIES))
    details["evaluation_checks"] = {
        "execution": {"status": "passed", "reason_codes": []},
        "core_coverage": {
            "status": "failed",
            "reason_codes": ["historical_requirement_gap"],
        },
    }
    return {
        "survey": "report",
        "metadata": {},
        "evaluation": {
            "citation_accuracy": {
                "score": 1.0,
                "status": "passed",
                "passed": True,
                "skipped": False,
            },
            "faithfulness": {
                "score": 0.42,
                "status": "failed",
                "passed": False,
                "skipped": False,
                "details": details,
            },
            "internal_consistency": {
                "score": 1.0,
                "status": "passed",
                "passed": True,
                "skipped": False,
            },
        },
        "evaluation_checks": copy.deepcopy(details["evaluation_checks"]),
        "delivery": {"publishable": False, "status": "blocked"},
    }


def test_normalization_public_summary_and_benchmark_read_stored_values(
    independent_details,
):
    report = archived_report(independent_details)
    before = copy.deepcopy(report)
    normalized = normalize_survey_result(report, config_fingerprint="historical")
    assert normalized["evaluation"] == report["evaluation"]
    assert normalized["evaluation_checks"] == report["evaluation_checks"]
    summary = public_summary(report)
    assert summary["metrics"]["faithfulness"]["score"] == 0.42
    assert summary["metrics"]["faithfulness"]["status"] == "failed"
    assert summary["checks"]["core_coverage"]["status"] == "failed"
    assert evaluation_execution(report["evaluation"]["faithfulness"]) == "completed"
    assert report == before
    details = report["evaluation"]["faithfulness"]["details"]
    with pytest.raises(ValueError, match="contract_identity_invalid"):
        EvaluationRunResult.from_details(details, details["source_report_sha256"])


@pytest.mark.parametrize(
    "fault", ["report", "checks", "unknown_identity", "identity_object"]
)
def test_old_view_does_not_accept_changed_report_or_invalid_identity(
    independent_details, fault
):
    report = archived_report(independent_details)
    if fault == "report":
        report["survey"] = "changed"
    elif fault == "checks":
        report["evaluation"]["faithfulness"]["details"]["evaluation_checks"][
            "execution"
        ] = False
    elif fault == "identity_object":
        report["evaluation"]["faithfulness"]["details"][
            "evaluation_result_identity"
        ] = {}
    else:
        report["evaluation"]["faithfulness"]["details"][
            "evaluation_result_identity"
        ] = "unknown"
    with pytest.raises(ValueError):
        public_summary(report)
    with pytest.raises(ValueError):
        normalize_survey_result(report, config_fingerprint="historical")


def test_historical_checks_do_not_fabricate_missing_data(independent_details):
    report = archived_report(independent_details)
    details = report["evaluation"]["faithfulness"]["details"]
    details.pop("evaluation_checks")
    assert historical_checks(details, details["source_report_sha256"]) == {}
    assert evaluation_execution(report["evaluation"]["faithfulness"]) == "incomplete"
