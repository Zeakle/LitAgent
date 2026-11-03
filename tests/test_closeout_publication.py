"""Verify public closeout data binding, failure denominators and sanitization."""

import hashlib
import json

import pytest

from litagent.quality import digest
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
    manifest["identity"] = digest(
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
