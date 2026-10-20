"""Replay the received notice patches against old and complete dependency bindings."""

import copy
import json
from pathlib import Path

import pytest
from test_complete_native_repair import real_request

from litagent.demo import project_flow_demo_view
from litagent.quality import (
    GroundedWritingError,
    apply_native_repairs,
    render_grounded_report_traced,
)


def test_summary_dependencies_require_scientific_patches_before_dispatch():
    f, _, _, _, schema, binding = real_request()
    assert binding["version"] == "complete-native-unit-repair-v5"
    required = [
        t for t in binding["targets"].values() if t.get("summary_dependency_paths")
    ]
    assert len(required) >= 2
    assert any(t["original"]["kind"] != "scientific" for t in required)
    for key, target in binding["targets"].items():
        if target.get("summary_dependency_paths"):
            fields = schema["json_schema"]["schema"]["properties"]["patches"][
                "properties"
            ][key]["properties"]
            assert target["must_remain_scientific"]
            assert fields["kind"]["enum"] == ["scientific"]
            assert fields["scope_notice_id"]["enum"] == ["none"]
    before = copy.deepcopy(f)
    old = json.loads(
        (
            Path(__file__).parent / "fixtures/native-summary-dependency-real.json"
        ).read_text()
    )
    candidate, _ = apply_native_repairs(
        f["candidate"], old["response"], old["binding"], cards=f["cards"]
    )
    with pytest.raises(GroundedWritingError) as failure:
        render_grounded_report_traced(
            candidate,
            f["cards"],
            "academic_v2",
            "zh-CN",
            evidence_selection=f["selection"],
            require_native=True,
            require_full_sources=True,
            require_scope_notices=True,
            analysis_mode=True,
        )
    assert {d["reason_code"] for d in failure.value.diagnostics} == {
        "strict_writing_scope_notice_repeated",
        "strict_writing_summary_binding_invalid",
    }
    with pytest.raises(ValueError, match="strict_native_repair_patch_invalid"):
        apply_native_repairs(f["candidate"], old["response"], binding, cards=f["cards"])
    assert f == before


def test_generation_failure_projection_does_not_invent_evaluation_or_mutate_archive():
    old = json.loads(
        (
            Path(__file__).parent / "fixtures/native-summary-dependency-real.json"
        ).read_text()
    )
    artifact = {
        "run_id": "current",
        "query": "public",
        "status": "completed",
        "events": [
            {
                "event": "writing.validation.failed",
                "data": {
                    "phase": "native_initial_repair",
                    "candidate_sha256": "a" * 64,
                    "diagnostics": old["diagnostics"],
                },
            }
        ],
        "report": {
            "survey": "No report generated",
            "metadata": {},
            "evaluation": {
                "faithfulness": {
                    "skipped": True,
                    "details": {"skipped_reason": "final_report_unavailable"},
                }
            },
        },
    }
    before = copy.deepcopy(artifact)
    view = project_flow_demo_view(artifact, source="archive")
    pipeline = view["pipeline_verification"]
    assert pipeline["assessment_scope"] == "generation_only"
    assert pipeline["stages"] == {"writing": "failed", "evaluation": "not_run"}
    assert len(pipeline["writing_diagnostics"]) == len(old["diagnostics"])
    assert all(
        d["candidate_sha256"] == "a" * 64 and d["phase"] == "native_initial_repair"
        for d in pipeline["writing_diagnostics"]
    )
    assert view["evaluation"] == artifact["report"]["evaluation"] and artifact == before
    previous = {
        "run_id": "old",
        "events": [],
        "report": {"survey": "old report", "evaluation": {}},
    }
    assert not project_flow_demo_view(previous, source="archive")[
        "pipeline_verification"
    ]
