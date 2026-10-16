"""Route uncommitted candidate failures through the actual suite consumer."""

import copy
import hashlib

import pytest
from test_quality_protocol_stop import live_runner, report

from litagent.benchmark import quality_suite
from litagent.benchmark.quality_suite import public_projection


def failed_candidate(reason):
    """Keep a rejected candidate separate from the retained original evaluation."""
    return {
        "attempted": True,
        "committed": False,
        "reason_code": "metric_regressed",
        "validation_reason_codes": [],
        "candidate": "# Candidate\n\nA claim. [E:a]",
        "candidate_evaluation": {
            "faithfulness": {
                "details": {
                    "skipped_reason": reason,
                    "unverified_reason": reason,
                }
            }
        },
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("location", ["single", "attempts", "both"])
@pytest.mark.parametrize(
    "reason",
    [
        "evaluation_inventory_unit_coverage_invalid",
        "evaluation_usage_incomplete",
        "evaluation_context_budget_exceeded",
    ],
)
async def test_uncommitted_protocol_failure_stops_before_the_next_paid_case(
    tmp_path, monkeypatch, location, reason
):
    value = report("evaluation_assertions_unverifiable")
    candidate = failed_candidate(reason)
    if location in {"single", "both"}:
        value["metadata"]["evidence_rewrite"] = candidate
    if location in {"attempts", "both"}:
        value["metadata"]["evidence_rewrite_attempts"] = [candidate]
    unchanged = copy.deepcopy(value)
    runner, calls = live_runner(
        tmp_path, monkeypatch, [value, report(None, verified=True)]
    )
    result = await runner.run()
    assert len(calls) == 1
    assert result["status"] == "stopped" and result["stop_reasons"] == [reason]
    entry = result["observations"][0]
    assert entry["technical_failure"] == reason
    assert entry["failure_codes"].count(reason) == 1
    assert entry["acceptance"]["publishable"] is False
    assert entry["diagnostic_failures"]
    for issue in entry["diagnostic_failures"]:
        assert issue["reason_code"] == reason
        assert (
            issue["candidate_sha256"]
            == hashlib.sha256(candidate["candidate"].encode()).hexdigest()
        )
        assert issue["path"].startswith("metadata.evidence_rewrite")
    assert result["budgets"]["survey"]["charged"] == 15
    assert result["budgets"]["survey"]["requests"] == 1
    assert result["counts"]["formal"]["planned"] == 15
    assert result["counts"]["formal"]["run"] == 0
    runner.service.wait.assert_not_awaited()
    assert await runner.run() == result and len(calls) == 1
    assert value == unchanged
    public = public_projection(result)
    assert reason in public["observations"][0]["failure_codes"]
    assert "diagnostic_failures" not in public["observations"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "evaluation_classification_or_source_invalid",
        "evaluation_global_structure_invalid",
        "evaluation_assertions_unverifiable",
    ],
)
async def test_valid_candidate_semantic_failure_does_not_become_a_protocol_error(
    tmp_path, monkeypatch, reason
):
    value = report("evaluation_assertions_unverifiable")
    value["metadata"]["evidence_rewrite_attempts"] = [failed_candidate(reason)]
    runner, calls = live_runner(
        tmp_path, monkeypatch, [value, report(None, verified=True), value]
    )
    result = await runner.run()
    assert len(calls) == 18 and result["status"] == "completed"
    assert result["observations"][0]["technical_failure"] is None
    assert result["observations"][0]["status"] == "blocked"
    assert result["counts"]["formal"]["run"] == 15
    assert result["counts"]["formal"]["automatic_passed"] == 0
    assert not result["counts"]["goal_met"]


@pytest.mark.asyncio
async def test_earlier_attempt_failure_is_not_hidden_by_the_latest_alias(
    tmp_path, monkeypatch
):
    reason = "evaluation_inventory_fragment_coverage_invalid"
    value = report("evaluation_assertions_unverifiable")
    earlier = failed_candidate(reason)
    latest = failed_candidate("evaluation_assertions_unverifiable")
    value["metadata"].update(
        evidence_rewrite=latest, evidence_rewrite_attempts=[earlier, latest]
    )
    runner, calls = live_runner(tmp_path, monkeypatch, [value])
    result = await runner.run()
    assert len(calls) == 1 and result["stop_reasons"] == [reason]
    paths = [i["path"] for i in result["observations"][0]["diagnostic_failures"]]
    assert any("evidence_rewrite_attempts.0" in p for p in paths)


@pytest.mark.asyncio
async def test_formal_candidate_protocol_failure_keeps_spent_and_unrun_slots(
    tmp_path, monkeypatch
):
    reason = "evaluation_inventory_unit_coverage_invalid"
    value = report("evaluation_assertions_unverifiable")
    value["metadata"]["evidence_rewrite"] = failed_candidate(reason)
    runner, calls = live_runner(
        tmp_path, monkeypatch, [report(None, verified=True)] * 3 + [value]
    )
    result = await runner.run()
    assert len(calls) == 4 and result["stop_reasons"] == [reason]
    assert result["counts"]["formal"]["run"] == 1
    assert result["counts"]["formal"]["not_run"] == 14
    assert result["budgets"]["survey"]["charged"] == 60
    assert result["budgets"]["survey"]["requests"] == 4
    assert await runner.run() == result and len(calls) == 4


def test_candidate_local_validation_paths_and_missing_history_are_preserved():
    reason = "evaluation_citation_ownership_invalid"
    candidate = failed_candidate(None)
    candidate["validation_reason_codes"] = [reason]
    metadata = {"evidence_rewrite": candidate}
    unchanged = copy.deepcopy(metadata)
    codes, locations = quality_suite._candidate_evaluation_failures(metadata)
    assert codes == [reason]
    assert locations[0]["path"] == "metadata.evidence_rewrite.validation_reason_codes.0"
    assert metadata == unchanged
    assert quality_suite._candidate_evaluation_failures({}) == ([], [])
    assert quality_suite._candidate_evaluation_failures(
        {"evidence_rewrite": {"attempted": False}, "evidence_rewrite_attempts": None}
    ) == ([], [])
