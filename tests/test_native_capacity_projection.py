"""Keep recoverable capacity failures distinct from scientific decisions."""

import copy
import csv
import hashlib
import json

import pytest
from test_benchmark_evaluation_execution import result_fixture
from test_quality_view import render_repair_state

from litagent.benchmark.quality_suite import export_bundle
from litagent.config import load_config
from litagent.eval.engine_policy import configured_engine
from litagent.eval.results import (
    CONTRACT,
    contract_identity,
    public_summary,
    sanitize_summary,
)


def report_fixture():
    """Build a public state fixture without pretending it is a model run."""
    survey = "Offline capacity display fixture."
    sha = hashlib.sha256(survey.encode()).hexdigest()
    recovery = {
        "action": "capacity",
        "decision_reason": "evaluation_native_capacity_refinement_required",
        "outcome_reason": "writer_patch_declined",
        "attempted": True,
        "committed": False,
        "reassessment": "unverified",
        "budget_status": "admitted",
        "remaining_attempts": 11,
        "remaining_by_category": {
            "capacity": 2,
            "inventory": 3,
            "review_revision": 3,
            "evaluation_revision": 3,
        },
    }
    details = {
        "evaluation_version": "survey-evaluation-ragas-v1",
        "implementation": "ragas-cited-faithfulness-v1",
        "ragas_version": "0.4.3",
        "evaluation_result_contract": CONTRACT,
        "evaluation_result_identity": contract_identity(),
        "source_report_sha256": sha,
        "execution_error": "evaluation_native_capacity_refinement_required",
        "factual_claim_count": 505,
        "supported_count": 0,
        "unsupported_count": 101,
        "unverified_count": 404,
        "coverage": {"source_units": 126, "processed_units": 82},
        "stages": {
            "capacity_admission": {
                "rejection": {
                    "code": "global_request_preflight_exceeded",
                    "stage": "global",
                    "reservation": 214091,
                    "context_limit": 212337,
                    "request_cap": 212337,
                    "unexecuted_stages": ["assessment", "global"],
                }
            }
        },
    }
    return {
        "survey": survey,
        "evaluation": {
            "faithfulness": {
                "status": "unverified",
                "score": None,
                "passed": False,
                "skipped": True,
                "details": details,
            }
        },
        "metadata": {
            "evaluation_history": {
                "version": "version-bound-evaluation-history-v1",
                "source_report_sha256": sha,
                "recovery": recovery,
            }
        },
        "quality": {"status": "unverified"},
        "delivery": {"status": "blocked", "publishable": False},
    }


def test_capacity_choice_requires_the_native_inline_independent_contract():
    config = load_config("config/separate-review-evaluation.yaml")
    config.eval.native_capacity_recovery = True
    with pytest.raises(ValueError, match="native_capacity_contract_invalid"):
        configured_engine(config)
    config.safety.planned_run_limits.citation_contract = "flat-writer-v2"
    assert configured_engine(config)["capacity_recovery_contract"] == (
        "native-capacity-recovery-v1"
    )
    config.eval.semantic_engine = None
    with pytest.raises(ValueError, match="native_capacity_contract_invalid"):
        configured_engine(config)


def test_capacity_recovery_is_identical_in_four_public_formats(tmp_path):
    report = report_fixture()
    original = copy.deepcopy(report)
    summary = sanitize_summary(public_summary(report))
    assert summary["metrics"]["faithfulness"]["score"] is None
    assert summary["science"]["unverified_count"] == 404
    assert summary["recovery"]["action"] == "capacity"
    assert not summary["recovery"]["committed"]
    fixture = result_fixture()
    fixture["observations"][0]["evaluation_summary"] = summary
    export_bundle(tmp_path, fixture)
    assert (
        json.loads((tmp_path / "benchmark.json").read_text())["observations"][0][
            "evaluation_summary"
        ]
        == summary
    )
    with (tmp_path / "benchmark.csv").open() as stream:
        row = next(csv.DictReader(stream))
    assert json.loads(row["evaluation_summary"]) == summary
    for suffix in ("md", "html"):
        text = (tmp_path / f"benchmark.{suffix}").read_text()
        assert "Recovery · capacity" in text
        assert "writer_patch_declined" in text
        assert "214091" in text and "212337" in text
    assert report == original


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_capacity_ui_preserves_unknown_science_and_actual_remaining_attempts(locale):
    value = report_fixture()
    history = value["metadata"]["evaluation_history"]
    value["options"] = {
        "evaluationHistory": history,
        "sha": history["source_report_sha256"],
    }
    rendered = render_repair_state(value, locale)
    assert rendered["metricStatuses"] == ["unverified"]
    assert "1.000" not in rendered["text"]
    assert "214091 / 212337" in rendered["text"]
    assert ("未核验" if locale == "zh-CN" else "unverified") in rendered["text"]
    assert "writer_patch_declined" in rendered["text"]
    assert (
        "保留草稿，尚未交付" if locale == "zh-CN" else "Retained draft; not delivered"
    ) in rendered["text"]
    assert "Committed report" not in rendered["text"]
    assert "已提交正文" not in rendered["text"].replace("不代表已提交正文", "")
