"""Validate independent metric states and a single owner for extraction checks."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from litagent.eval.responsibilities import project_global_schema
from litagent.eval.results import (
    CONTRACT,
    EvaluationRunResult,
    acceptance,
    contract_identity,
    selected,
)
from litagent.quality import strict_acceptance
from litagent.runner import LitAgent


@pytest.fixture
def independent_details():
    """Provide one complete host-validated science decision and its check provenance."""
    claim = {
        "claim_id": "claim-one",
        "source_unit_id": "unit-one",
        "kind": "scientific",
        "atomic": True,
        "classification_valid": True,
        "source_faithful": True,
        "status": "supported",
    }
    return {
        "source_report_sha256": hashlib.sha256(b"report").hexdigest(),
        "evaluation_result_contract": CONTRACT,
        "evaluation_result_identity": contract_identity(),
        "evaluation_version": "survey-evaluation-v3",
        "quality_policy": "strict_auto_v2",
        "threshold": 0.8,
        "assertions": [claim],
        "fact_card_map": {"card-one": "evidence-one"},
        "coverage": {"source_units": 1, "processed_units": 1, "complete": True},
        "inventory_check": {"status": "passed"},
        "conflicts": [],
        "global_issues": [],
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
        "language_checks": {"status": "passed"},
        "analysis_check": {"status": "passed"},
        "limitations_check": {"adequate": True, "claim_ids": ["claim-one"]},
        "coverage_requirements": [{"requirement_id": "question-one"}],
        "core_coverage": [
            {
                "requirement_id": "question-one",
                "answered": True,
                "claim_ids": ["claim-one"],
            }
        ],
        "stages": {
            "global_check": "completed",
            "processed_assertions": 1,
            "expected_assertions": 1,
            "expected_batches": 1,
            "batches": [{"status": "completed", "claim_ids": ["claim-one"]}],
            "global_sections": {
                "conflicts": {"status": "passed"},
                "inventory": {"status": "passed"},
            },
        },
    }


def run_for(details):
    """Build the same immutable execution view used by the runner."""
    return EvaluationRunResult.from_details(details, details["source_report_sha256"])


def test_support_failure_does_not_invalidate_consistency(independent_details):
    independent_details["assertions"][0]["status"] = "unsupported"
    metrics = run_for(independent_details).metrics()
    assert metrics["faithfulness"]["status"] == "failed"
    assert metrics["faithfulness"]["score"] == 0.0
    assert metrics["internal_consistency"]["status"] == "passed"
    assert metrics["internal_consistency"]["score"] == 1.0
    assert strict_acceptance(metrics["faithfulness"]["details"])["status"] == "failed"


@pytest.mark.parametrize(
    "key", ["analysis_check", "method_checks", "structure_checks", "language_checks"]
)
def test_report_check_failure_preserves_science_score(independent_details, key):
    independent_details[key] = {"status": "failed"}
    metrics = run_for(independent_details).metrics()
    assert metrics["faithfulness"]["score"] == 1.0
    assert acceptance(metrics["faithfulness"]["details"])["status"] == "failed"


def test_unknown_method_record_does_not_erase_science(independent_details):
    claim = {
        **independent_details["assertions"][0],
        "claim_id": "method-one",
        "kind": "runtime_method",
        "status": "unverifiable",
    }
    independent_details["assertions"].append(claim)
    stage = independent_details["stages"]
    stage.update(expected_assertions=2, processed_assertions=2)
    stage["batches"][0]["claim_ids"].append("method-one")
    metrics = run_for(independent_details).metrics()
    assert metrics["faithfulness"]["score"] == 1.0
    assert acceptance(metrics["faithfulness"]["details"])["status"] == "failed"


@pytest.mark.parametrize(
    "change",
    ["atomic", "classification_valid", "source_faithful", "omission", "partial"],
)
def test_invalid_inventory_never_publishes_score(independent_details, change):
    if change == "omission":
        independent_details["inventory_check"]["status"] = "failed"
    elif change == "partial":
        independent_details["stages"]["batches"][0]["status"] = "interrupted_no_replay"
    else:
        independent_details["assertions"][0][change] = False
    for metric in run_for(independent_details).metrics().values():
        assert metric["score"] is None
        assert metric["passed"] is False
        assert metric["skipped"] is True
    assert acceptance(independent_details)["status"] == "failed"


def test_scientific_unverified_preserves_complete_consistency(independent_details):
    independent_details["assertions"][0]["status"] = "unverifiable"
    metrics = run_for(independent_details).metrics()
    assert metrics["faithfulness"]["score"] is None
    assert metrics["internal_consistency"]["score"] == 1.0


def test_conflict_region_error_does_not_erase_support(independent_details):
    independent_details["stages"]["global_sections"]["conflicts"] = {
        "status": "error",
        "reason_codes": ["evaluation_global_conflicts_invalid"],
    }
    metrics = run_for(independent_details).metrics()
    assert metrics["faithfulness"]["score"] == 1.0
    assert metrics["internal_consistency"]["status"] == "error"
    assert metrics["internal_consistency"]["score"] is None
    assert acceptance(independent_details)["status"] == "failed"


def test_strict_gate_and_quality_consume_same_result(independent_details):
    metrics = {
        "citation_accuracy": {
            "status": "passed",
            "score": 1.0,
            "passed": True,
            "skipped": False,
        },
        **run_for(independent_details).metrics(),
    }
    assert acceptance(independent_details)["status"] == "passed"
    assert LitAgent._derive_quality(metrics)["status"] == "passed"
    independent_details["core_coverage"][0]["answered"] = False
    metrics.update(run_for(independent_details).metrics())
    assert acceptance(independent_details)["status"] == "failed"
    assert LitAgent._derive_quality(metrics)["status"] == "failed"


def test_reject_cross_report_reuse(independent_details):
    with pytest.raises(ValueError, match="report_identity_invalid"):
        EvaluationRunResult.from_details(independent_details, "0" * 64)


def test_schema_removes_duplicate_responsibilities_only():
    schema = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "enum": [
                    "non_atomic",
                    "classification",
                    "source_fidelity",
                    "missing_assertion",
                    "contradiction",
                ],
            }
        },
    }
    updated = project_global_schema(schema)
    assert updated["properties"]["code"]["enum"] == [
        "missing_assertion",
        "contradiction",
    ]
    assert schema["properties"]["code"]["enum"][0] == "non_atomic"
    with pytest.raises(Exception):
        Draft202012Validator(updated).validate({"code": "non_atomic"})


def test_057_remains_historical_invalid():
    path = Path("artifacts/runs/frozen-writer-single-20261104-057-few-shot-metric.json")
    if not path.exists():
        pytest.skip("Private real-run artifact is not distributed")
    raw = path.read_bytes()
    data = json.loads(raw)

    # Locate the stored normalized report without assuming archive wrapper layout.
    def walk(value):
        if isinstance(value, dict):
            if (
                isinstance(value.get("evaluation"), dict)
                and "faithfulness" in value["evaluation"]
            ):
                yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    reports = list(walk(data))
    assert reports
    final = reports[0]["evaluation"]["faithfulness"]
    assert final["skipped"] is True
    assert final["details"]["inventory_check"]["status"] == "failed"
    assert "evaluation_result_contract" not in final["details"]
    assert path.read_bytes() == raw


@pytest.fixture
def global_regions():
    """Provide canonical host IDs to test region isolation without transport mocks."""
    from litagent.body_quality import ROLES

    payload = {
        "body_analysis_contract": "body-analysis-v1",
        "global_findings_contract": "anchored-global-findings-v2",
        "source_units": [{"source_unit_id": "unit-one"}],
        "assertions": [
            {
                "claim_id": "claim-one",
                "source_unit_id": "unit-one",
                "kind": "scientific",
            }
        ],
        "coverage_requirements": [{"requirement_id": "question-one"}],
    }
    value = {
        "units": [
            {
                "source_unit_id": "unit-one",
                "complete": True,
                "explanation": "All facts are listed.",
            }
        ],
        "conflicts": [],
        "issues": [],
        "coverage": [
            {
                "requirement_id": "question-one",
                "answered": True,
                "claim_ids": ["claim-one"],
                "explanation": "Answered.",
            }
        ],
        "analyses": [
            {
                "section_key": name,
                "adequate": False,
                "claim_ids": [],
                "explanation": "Limited analysis.",
            }
            for name in ROLES
        ],
        "limitations": {
            "adequate": False,
            "claim_ids": [],
            "explanation": "No supported limitation.",
        },
    }
    return payload, value


@pytest.mark.parametrize("field", ["coverage", "analyses", "limitations", "conflicts"])
def test_valid_global_regions_survive_an_unrelated_malformed_region(
    global_regions, field
):
    from litagent.eval.responsibilities import parse_global_response

    payload, value = global_regions
    value[field] = "malformed region"
    progress = {}
    parsed = parse_global_response(json.dumps(value), payload, progress)
    assert progress["global_sections"][field]["status"] == "error"
    assert progress["global_sections"]["inventory"]["status"] == "passed"
    assert parsed.units[0].complete is True
    if field != "conflicts":
        assert progress["global_sections"]["conflicts"]["status"] == "passed"


@pytest.mark.parametrize("code", ["non_atomic", "classification", "source_fidelity"])
def test_global_cannot_reintroduce_duplicate_checks(global_regions, code):
    from litagent.eval.responsibilities import parse_global_response

    payload, value = global_regions
    value["issues"] = [
        {
            "code": code,
            "anchor_ids": ["claim:claim-one"],
            "explanation": "Duplicate judgment.",
        }
    ]
    with pytest.raises(ValueError, match="responsibility_invalid"):
        parse_global_response(json.dumps(value), payload, {})


@pytest.mark.parametrize(
    "fault", ["invalid_json", "duplicate_key", "unknown_unit", "unknown_claim"]
)
def test_global_json_or_identity_failure_is_not_salvaged(global_regions, fault):
    from litagent.eval.responsibilities import parse_global_response

    payload, value = global_regions
    if fault == "unknown_unit":
        value["units"][0]["source_unit_id"] = "unknown"
    elif fault == "unknown_claim":
        value["conflicts"] = [
            {"claim_a": "claim-one", "claim_b": "unknown", "reason": "Contradiction."}
        ]
    raw = json.dumps(value)
    if fault == "invalid_json":
        raw = raw[:-2]
    elif fault == "duplicate_key":
        raw = '{"units": [],' + raw[1:]
    with pytest.raises(ValueError):
        parse_global_response(raw, payload, {})


def test_new_contract_is_explicit_and_identity_bound():
    from litagent.eval.results import selected

    assert selected(None) is False
    assert selected({}) is False
    envelope = {
        "contract": "flat-writer-envelope-v1",
        "global_findings_contract": "anchored-global-findings-v2",
        "support_validation_revision": "host-support-receipts-v2",
        "evaluation_result_contract": CONTRACT,
        "evaluation_result_identity": contract_identity(),
    }
    assert selected(envelope)
    envelope["evaluation_result_identity"] = "0" * 64
    with pytest.raises(ValueError, match="contract_invalid"):
        selected(envelope)


def test_frozen_result_identity_is_not_silently_rewritten(independent_details):
    independent_details["evaluation_result_identity"] = "0" * 64
    with pytest.raises(ValueError, match="contract_identity_invalid"):
        run_for(independent_details)


def test_report_response_projects_checks_and_null_scores(independent_details):
    from litagent.contracts import normalize_survey_result

    independent_details["inventory_check"]["status"] = "failed"
    metrics = run_for(independent_details).metrics()
    report = normalize_survey_result(
        {"survey": "report", "evaluation": metrics}, config_fingerprint="test"
    )
    assert report["evaluation_checks"]["inventory"]["status"] == "failed"
    assert report["evaluation"]["faithfulness"]["score"] is None
    assert report["evaluation"]["faithfulness"]["passed"] is False
    legacy = normalize_survey_result(
        {"survey": "old", "evaluation": {}}, config_fingerprint="old"
    )
    assert "evaluation_checks" not in legacy


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_quality_renderer_separates_execution_errors_and_invalid_scores(
    independent_details, locale
):
    from test_quality_view import render_repair_state

    independent_details["stages"]["global_sections"]["conflicts"] = {
        "status": "error",
        "reason_codes": ["evaluation_global_conflicts_invalid"],
    }
    metrics = run_for(independent_details).metrics()
    before = copy.deepcopy(metrics)
    view = render_repair_state({"evaluation": metrics}, locale)
    assert view["metricStatuses"] == ["passed", "error"]
    assert ("Independent checks" if locale == "en" else "独立检查与交付门禁") in view[
        "text"
    ]
    assert ("Execution error" if locale == "en" else "执行错误") in view["text"]
    assert "litagent" in view["text"].lower()
    assert metrics == before


def test_public_summary_excludes_original_text_and_shares_four_format_values(
    independent_details,
):
    import csv
    import io

    from litagent.eval.results import public_summary, summary_tables

    independent_details["assertions"][0]["source_quote"] = "PRIVATE_SOURCE_SENTINEL"
    independent_details["stages"]["requests"] = [
        {"response": "PRIVATE_RESPONSE_SENTINEL"}
    ]
    metrics = run_for(independent_details).metrics()
    summary = public_summary({"survey": "report", "evaluation": metrics})
    rows = [{"case_id": "offline-contract", "evaluation_summary": summary}]
    markdown, html = summary_tables(rows)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=["case_id", "evaluation_summary"])
    writer.writeheader()
    writer.writerow(
        {"case_id": rows[0]["case_id"], "evaluation_summary": json.dumps(summary)}
    )
    restored = json.loads(
        next(csv.DictReader(io.StringIO(stream.getvalue())))["evaluation_summary"]
    )
    assert restored == summary
    for name, item in summary["metrics"].items():
        assert name in markdown and name in html
        assert item["status"] in markdown and item["status"] in html
    for text in (json.dumps(summary), stream.getvalue(), markdown, html):
        assert "PRIVATE_SOURCE_SENTINEL" not in text
        assert "PRIVATE_RESPONSE_SENTINEL" not in text


def test_stored_public_summary_is_whitelisted_again(independent_details):
    from litagent.eval.results import public_summary, sanitize_summary

    summary = public_summary(
        {"survey": "report", "evaluation": run_for(independent_details).metrics()}
    )
    summary["raw_response"] = "SECRET_SENTINEL"
    summary["metrics"]["faithfulness"]["private_path"] = "/Users/private"
    summary["metrics"]["faithfulness"]["reason_codes"] = [
        "safe_reason",
        "/Users/secret",
        "secret key",
    ]
    public = sanitize_summary(summary)
    assert "SECRET_SENTINEL" not in json.dumps(public)
    assert "/Users/" not in json.dumps(public)
    assert public["metrics"]["faithfulness"]["reason_codes"] == ["safe_reason"]


def test_execution_error_survives_serialization_and_cannot_become_passed(
    independent_details,
):
    run = EvaluationRunResult.from_details(
        independent_details,
        independent_details["source_report_sha256"],
        error="evaluation_contract_invalid",
    )
    first = run.metrics()["faithfulness"]
    assert first["status"] == "error"
    restored = EvaluationRunResult.from_details(
        json.loads(json.dumps(first["details"])),
        independent_details["source_report_sha256"],
    )
    assert restored.metrics()["faithfulness"]["status"] == "error"
    assert restored.checks()["execution"]["status"] == "error"
    assert acceptance(restored.details)["status"] == "failed"


def test_missing_historical_labels_do_not_render_nan_or_undefined(independent_details):
    from test_quality_view import render_repair_state

    metrics = run_for(independent_details).metrics()
    view = render_repair_state({"evaluation": metrics}, "zh-CN")
    assert "NaN" not in view["text"]
    assert "undefined" not in view["text"]
    assert "question-one" in view["text"]
    assert metrics["internal_consistency"]["details"]["threshold"] == 0.8


def test_component_error_preserves_metrics_but_never_claims_execution_complete(
    independent_details,
):
    from litagent.benchmark.quality_suite import (
        _technical_failure,
        evaluation_execution,
    )
    from scripts.run_frozen_writing import evaluation_diagnostic

    independent_details["stages"]["global_sections"]["analyses"] = {
        "status": "error",
        "reason_codes": ["evaluation_global_analyses_invalid"],
    }
    metrics = run_for(independent_details).metrics()
    assert metrics["faithfulness"]["score"] == 1.0
    assert metrics["internal_consistency"]["score"] == 1.0
    assert (
        metrics["faithfulness"]["details"]["evaluation_checks"]["execution"]["status"]
        == "error"
    )
    assert evaluation_execution(metrics["faithfulness"]) == "incomplete"
    diagnostic = evaluation_diagnostic({"evaluation": metrics})
    assert _technical_failure([diagnostic["reason_code"]]) is not None
    assert acceptance(metrics["faithfulness"]["details"])["status"] == "failed"


def test_candidate_component_error_is_not_hidden_by_retaining_original(
    independent_details,
):
    from litagent.benchmark.quality_suite import _candidate_evaluation_failures

    independent_details["stages"]["global_sections"]["analyses"] = {
        "status": "error",
        "reason_codes": ["evaluation_global_analyses_invalid"],
    }
    metrics = run_for(independent_details).metrics()
    metadata = {
        "evidence_rewrite_attempts": [
            {
                "attempted": True,
                "candidate_evaluation": metrics,
                "candidate_sha256": independent_details["source_report_sha256"],
            }
        ]
    }
    reasons, _ = _candidate_evaluation_failures(metadata)
    assert "evaluation_contract_invalid" in reasons


@pytest.mark.parametrize("details", [[1], "invalid", True])
def test_historical_malformed_details_stay_unrecorded(details):
    from litagent.benchmark.quality_suite import evaluation_execution

    assert evaluation_execution({"details": details}) == "unrecorded"
