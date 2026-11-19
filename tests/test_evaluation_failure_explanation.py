"""Explain strict failures while preserving scores and exact evidence ownership."""

import copy
import csv
import json
from pathlib import Path

import pytest
from test_independent_evaluation import independent_details, run_for  # noqa: F401
from test_quality_view import render_repair_state

from litagent.body_quality import ROLES, validate_analysis_decisions
from litagent.demo import project_flow_demo_view
from litagent.eval.results import (
    acceptance,
    failure_explanation,
    public_summary,
    sanitize_summary,
    summary_tables,
)
from litagent.review_structure import structure_sections


@pytest.fixture
def chapter_report(independent_details):
    """Give each section its own assessment and one exact unsupported conclusion."""
    d = independent_details
    titles = dict(structure_sections("academic_v2", "en"))
    prototype = d["assertions"][0]
    claims, decisions = [], []
    for i, section in enumerate(ROLES):
        cid = f"claim_{i}"
        claims.append(
            {
                **prototype,
                "claim_id": cid,
                "source_unit_id": f"unit:{i}",
                "source_start": 0,
                "source_end": 6,
                "source_quote": "report",
                "section": titles[section],
                "status": "unsupported" if section == "conclusion" else "supported",
            }
        )
        decisions.append(
            {
                "section_key": section,
                "adequate": True,
                "claim_ids": [cid],
                "explanation": "Complete.",
            }
        )
    d["assertions"] = claims
    supported = {c["claim_id"] for c in claims if c["status"] == "supported"}
    d["analysis_check"] = validate_analysis_decisions(decisions, claims, supported, [])
    d["stages"].update(
        expected_assertions=len(claims), processed_assertions=len(claims)
    )
    d["stages"]["batches"][0]["claim_ids"] = [c["claim_id"] for c in claims]
    d["core_coverage"][0]["claim_ids"] = ["claim_0"]
    d["limitations_check"]["claim_ids"] = ["claim_0"]
    metrics = run_for(d).metrics()
    details = metrics["faithfulness"]["details"]
    details["strict_acceptance"] = acceptance(details)
    return {"survey": "report", "evaluation": metrics, "quality": {"status": "failed"}}


def test_one_bound_failure_shared_by_view_and_exports(chapter_report, tmp_path):
    report = chapter_report
    original = copy.deepcopy(report)
    expected = {
        "primary": ["science_not_fully_supported"],
        "analysis_dependencies": {"conclusion": ["claim_6"]},
    }
    view = project_flow_demo_view({"report": report}, source="archive")
    assert view["quality"]["failure_explanation"] == expected
    summary = sanitize_summary(public_summary(report))
    assert summary["failure_explanation"] == expected
    for output in summary_tables(
        [{"case_id": "example", "evaluation_summary": summary}]
    ):
        assert output.count("science_not_fully_supported") == 1
        assert "conclusion: claim_6" in output
    from litagent.benchmark.quality_suite import export_bundle, summarize

    case = {"phase": "pilot", "case_id": "example", "domain": "vit"}
    bundle = {
        "planned": [case],
        "observations": [
            {
                **case,
                "evaluation_summary": summary,
                "costs": {},
                "gates": {},
                "status": "blocked",
            }
        ],
        "stop_reasons": [],
        "budgets": {},
    }
    bundle["counts"] = summarize(bundle)
    export_bundle(tmp_path, bundle)
    observed = json.loads((tmp_path / "benchmark.json").read_text())["observations"][0]
    assert observed["evaluation_summary"]["failure_explanation"] == expected
    with (tmp_path / "benchmark.csv").open() as stream:
        restored = json.loads(next(csv.DictReader(stream))["evaluation_summary"])
    assert restored == observed["evaluation_summary"]
    for name in ("benchmark.md", "benchmark.html"):
        assert "conclusion: claim_6" in (tmp_path / name).read_text()
    for locale, label in (
        ("zh-CN", "以下断言缺少充分依据"),
        ("en", "Selected claims lack sufficient support"),
    ):
        ui = render_repair_state(
            {"evaluation": view["evaluation"], "quality": view["quality"]}, locale
        )
        assert label in ui["text"] and "claim_6" in ui["text"]
    assert report == original
    assert (
        acceptance(report["evaluation"]["faithfulness"]["details"])["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "fault",
    [
        "semantic",
        "wrong_section",
        "unknown",
        "unverified",
        "missing_coverage",
        "stale_span",
        "other_failed_chapter",
    ],
)
def test_independent_errors_are_not_merged(chapter_report, fault):
    details = chapter_report["evaluation"]["faithfulness"]["details"]
    analysis = details["evaluation_checks"]["analysis"]["sections"]
    row = next(r for r in analysis if r["section_key"] == "conclusion")
    c = details["assertions"][-1]
    if fault == "semantic":
        row["model_adequate"] = False
    elif fault == "wrong_section":
        c["section"] = "Introduction"
    elif fault == "unknown":
        row["claim_ids"] = ["claim_999"]
    elif fault == "unverified":
        c["status"] = "unverifiable"
    elif fault == "missing_coverage":
        chapter_report["evaluation"]["faithfulness"]["score"] = None
        details["evaluation_checks"]["inventory"]["status"] = "failed"
    elif fault == "stale_span":
        c["source_end"] -= 1
    else:
        analysis[0].update(adequate=False, model_adequate=False)
    explanation = failure_explanation(chapter_report)
    assert "body_analysis_incomplete" in explanation["primary"]
    assert bool(explanation["analysis_dependencies"]) == (
        fault == "other_failed_chapter"
    )


def test_full_science_and_insufficient_analysis_remain_blocked(independent_details):
    d = independent_details
    d["analysis_check"] = {"status": "failed", "sections": []}
    metrics = run_for(d).metrics()
    details = metrics["faithfulness"]["details"]
    details["strict_acceptance"] = acceptance(details)
    assert metrics["faithfulness"]["score"] == 1.0
    assert failure_explanation({"survey": "report", "evaluation": metrics}) == {
        "primary": ["body_analysis_incomplete"],
        "analysis_dependencies": {},
    }


def test_incomplete_inventory_and_cross_report_never_claim_a_shared_failure(
    independent_details,
):
    d = independent_details
    d["inventory_check"]["status"] = "failed"
    metrics = run_for(d).metrics()
    details = metrics["faithfulness"]["details"]
    details["strict_acceptance"] = acceptance(details)
    report = {"survey": "report", "evaluation": metrics}
    assert metrics["faithfulness"]["score"] is None
    primary = failure_explanation(report)["primary"]
    assert {"inventory_invalid", "science_support_unverified"}.issubset(primary)
    assert "science_not_fully_supported" not in primary
    assert (
        "science_not_fully_supported" in details["strict_acceptance"]["failure_codes"]
    )
    assert failure_explanation({**report, "survey": "different"}) is None


def test_frozen_vit_503_assertions_preserve_all_negative_diagnostics():
    from litagent.eval.ragas_repair import native_repair_diagnostics

    root = Path(__file__).resolve().parents[1] / (
        "artifacts/acceptance/vit14-organization-guard-20261118/profile-coverage"
    )
    result = root / "real-experimental-evaluation-20261118" / "result.json"
    draft = root / "vit14-experimental-metrics-draft.json"
    if not result.exists() or not draft.exists():
        pytest.skip("Local 34-card ViT evaluation archive is unavailable")
    archived = json.loads(result.read_bytes())
    manuscript = json.loads(draft.read_bytes())
    report = {"survey": manuscript["survey"], "evaluation": archived["evaluation"]}
    details = report["evaluation"]["faithfulness"]["details"]
    problems = native_repair_diagnostics(details)
    assert acceptance(details) == archived["strict_acceptance"]
    assert archived["strict_acceptance"]["status"] == "failed"
    assert (details["supported_count"], details["factual_claim_count"]) == (449, 503)
    assert len(problems) == len(details["unsupported_claims"]) == 54
    assert public_summary(report)["failure_explanation"] == {
        "primary": ["science_not_fully_supported"],
        "analysis_dependencies": {"conclusion": ["claim_502"]},
    }


def test_public_explanation_does_not_export_arbitrary_identifiers(chapter_report):
    summary = public_summary(chapter_report)
    summary["failure_explanation"]["analysis_dependencies"]["conclusion"] = [
        "/private/file"
    ]
    assert "failure_explanation" not in sanitize_summary(summary)
