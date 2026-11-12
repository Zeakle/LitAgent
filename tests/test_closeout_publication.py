"""Verify public closeout data binding, failure denominators and sanitization."""

import hashlib
import json
import re

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.report_audit import digest as manifest_digest
from litagent.review_structure import prepare_writer_methodology
from litagent.writing_v2 import (
    RANGE_WRITING_PLAN,
    compile_document,
    writer_budget_plan,
)
from scripts.publish_closeout import CASES, publication, write_publication


def records():
    """Keep one failed report and two unrun planned cases in a complete matrix."""
    text = "# Report\n\nAn unresolved scientific claim. [E:a]\n"
    sha = hashlib.sha256(text.encode()).hexdigest()
    rows = [
        {
            "case_id": case,
            "generated": index == 0,
            "reviewer_completed": index == 0,
            "evaluation_valid": False,
            "publishable": False,
            "run_id": "suite-" + case if index == 0 else None,
            "report_sha256": sha if index == 0 else None,
            "raw_request": {"api_key": "PRIVATE_MUST_NOT_LEAK"},
            "failure_code": (
                "evaluation_inventory_invalid"
                if index == 0
                else "not_started_after_stop"
            ),
        }
        for index, case in enumerate(CASES)
    ]
    benchmark = {
        "suite_id": "suite",
        "identity": "binding",
        "planned": 3,
        "run": 1,
        "cases": rows,
        "state": "stopped",
    }
    manifest = {
        "suite_id": "suite",
        "identity": "binding",
        "closeout_profile": "engineering-closeout-v1",
    }
    archives = {CASES[0]: {"run_id": rows[0]["run_id"], "report": {"survey": text}}}
    manifest["identity"] = manifest_digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    benchmark["identity"] = manifest["identity"]
    return benchmark, manifest, archives


def test_public_bundle_keeps_failures_unrun_cases_and_four_format_consistency(tmp_path):
    value = publication(*records())
    assert value["planned"] == 3 and value["run"] == 1
    assert value["generated"] == 1 and value["delivered"] == 0
    assert len(value["cases"]) == 3 and value["human_accuracy"] is None
    assert "PRIVATE_MUST_NOT_LEAK" not in json.dumps(value)
    outputs = write_publication(value, tmp_path)
    assert set(outputs) == {"json", "csv", "md", "html"}
    assert json.loads((tmp_path / "closeout.json").read_text()) == value
    assert b"\r" not in (tmp_path / "closeout.csv").read_bytes()
    for case in CASES:
        assert all(
            case in (tmp_path / ("closeout." + kind)).read_text() for kind in outputs
        )
    with pytest.raises(ValueError, match="closeout_publication_exists"):
        write_publication(value, tmp_path)


@pytest.mark.parametrize("fault", ["sha", "run", "manifest", "missing", "sensitive"])
def test_publication_rejects_wrong_identity_missing_report_and_private_text(fault):
    benchmark, manifest, archives = records()
    if fault == "sha":
        benchmark["cases"][0]["report_sha256"] = "changed"
    elif fault == "run":
        archives[CASES[0]]["run_id"] = "other-run"
    elif fault == "manifest":
        manifest["identity"] = "other-suite"
    elif fault == "missing":
        archives.clear()
    else:
        text = "/Users/private-key-file"
        archives[CASES[0]]["report"]["survey"] = text
        benchmark["cases"][0]["report_sha256"] = hashlib.sha256(
            text.encode()
        ).hexdigest()
    with pytest.raises(ValueError, match="closeout_publication_"):
        publication(benchmark, manifest, archives)


@pytest.fixture
def draft_records(case):
    """Use the real compiler for a Reviewer-interrupted, unevaluated draft."""
    benchmark, manifest, archives = records()
    bundle = {
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "methodology": {},
        "extractions": [],
    }
    methodology = prepare_writer_methodology(
        {},
        selection=case.selection,
        cards=case.cards,
        extractions=[],
        upstream_results={"extract": []},
    )
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology=methodology,
        budget_policy="planned-budget-v5",
    )
    text, bindings = compile_document(
        case.body,
        case.summaries,
        case.cards,
        case.aliases,
        "en",
        methodology,
        writing_plan=RANGE_WRITING_PLAN,
        envelope=plan["envelope"],
    )
    manifest.update(
        options={"report_language": "en"},
        cases=[{"case_id": CASES[0], "bundle": bundle}],
        budget_plans={CASES[0]: plan},
        writing_plan=RANGE_WRITING_PLAN,
    )
    manifest["identity"] = manifest_digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    benchmark["identity"] = manifest["identity"]
    benchmark["cases"][0].update(
        report_sha256=None,
        compiled_report_sha256=bindings["report_sha256"],
        reviewer_completed=False,
        failure_code="responses_incomplete",
    )
    archives[CASES[0]]["report"]["survey"] = "Interrupted placeholder."
    journal = {
        "status": "generated",
        "body": case.body,
        "summaries": case.summaries,
        "draft_sha256": bindings["report_sha256"],
        "calls": {
            "reviewer": {
                "stage": "reviewer",
                "max_tokens": 4096,
                "usage": {"completion_tokens": 4096, "reasoning_tokens": 0},
                "failure": {"code": "responses_incomplete"},
                "raw": "PRIVATE_MUST_NOT_LEAK",
            }
        },
    }
    return benchmark, manifest, archives, {CASES[0]: journal}, text


def test_interrupted_draft_is_exact_readable_and_never_a_scientific_pass(
    draft_records, tmp_path
):
    benchmark, manifest, archives, journals, text = draft_records
    before = json.dumps([benchmark, manifest, archives, journals], sort_keys=True)
    value = publication(benchmark, manifest, archives, journals)
    row = value["cases"][0]
    assert row["draft"] == text and "report" not in row
    assert row["draft_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert row["material_status"] == "unreviewed_draft"
    assert value["generated"] == 1 and value["reviewer_completed"] == 0
    assert value["evaluation_completed"] == value["delivered"] == 0
    assert row["supported"] is row["unsupported"] is row["unverified"] is None
    assert "PRIVATE_MUST_NOT_LEAK" not in json.dumps(value)
    assert "source_quote" not in json.dumps(row["sources"])
    assert (
        json.dumps([benchmark, manifest, archives, journals], sort_keys=True) == before
    )
    write_publication(value, tmp_path)
    page = (tmp_path / "closeout.html").read_text()
    assert '<article class="report">' in page
    assert "Unreviewed and unevaluated draft" in page
    assert "有效完整核验 / Valid Evaluation</dt><dd>未核验 / Incomplete" in page
    assert "严格交付 / Delivered</dt><dd>阻止交付 / Blocked" in page
    assert '<details class="appendix">' in page
    assert '<details class="appendix" open' not in page
    assert "#few-shot-metric-source-" in page
    assert text in (tmp_path / "closeout.md").read_text()


@pytest.mark.parametrize("fault", ["missing", "sha", "changed_text", "sources"])
def test_draft_publication_cannot_guess_missing_or_changed_material(
    draft_records, fault
):
    benchmark, manifest, archives, journals, _ = draft_records
    if fault == "missing":
        journals.clear()
    elif fault == "sha":
        journals[CASES[0]]["draft_sha256"] = "wrong"
    elif fault == "changed_text":
        journals[CASES[0]]["body"]["methods"][0]["sentences"][0]["text"] += " Changed."
    else:
        manifest["cases"][0]["bundle"]["fact_cards_sha256"] = "wrong"
        manifest["identity"] = manifest_digest(
            {k: v for k, v in manifest.items() if k != "identity"}
        )
        benchmark["identity"] = manifest["identity"]
    with pytest.raises(ValueError, match="closeout_publication_draft_"):
        publication(benchmark, manifest, archives, journals)


def test_html_embedded_data_roundtrips_angle_brackets_without_executable_html(tmp_path):
    benchmark, manifest, archives = records()
    text = "# Report\n\nA < B. </script><script>alert('untrusted')</script>\n"
    archives[CASES[0]]["report"]["survey"] = text
    benchmark["cases"][0]["report_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    value = publication(benchmark, manifest, archives)
    write_publication(value, tmp_path)
    page = (tmp_path / "closeout.html").read_text()
    embedded = re.search(
        r'<script type="application/json" id="publication">(.*?)</script>', page, re.S
    )[1]
    assert json.loads(embedded) == value
    assert "A &lt; B" in page
    assert "<script>alert" not in page


def test_publication_uses_the_actual_campaign_manifest_digest():
    benchmark, manifest, archives = records()
    unsigned = {key: value for key, value in manifest.items() if key != "identity"}
    assert manifest["identity"] == manifest_digest(unsigned)
    assert digest(unsigned) != manifest["identity"]
    assert (
        publication(benchmark, manifest, archives)["manifest_identity"]
        == manifest["identity"]
    )
    manifest["identity"] = digest(unsigned)
    benchmark["identity"] = manifest["identity"]
    with pytest.raises(ValueError, match="closeout_publication_identity_invalid"):
        publication(benchmark, manifest, archives)
