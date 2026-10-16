"""Exercise live-result classification and scheduling without external transports."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_strict_automatic import suite

from litagent.benchmark import quality_suite
from litagent.benchmark.quality_suite import public_projection
from litagent.safety.budget import CostBudget


def report(reason, *, verified=False):
    """Use synthetic terminal reports solely to verify stop and accounting rules."""
    acceptance = {
        "publishable": verified,
        "failure_codes": [reason] if reason else [],
        "gates": {
            "evaluation": "passed" if verified else "failed",
            "judge": "passed" if verified else "not_run",
            "jev": "passed" if verified else "not_run",
        },
    }
    if verified:
        acceptance["audit_id"] = "offline-test-audit"
    return {
        "survey": "# Report\n\nA uses patches. [E:a]",
        "partial": False,
        "metadata": {"automatic_acceptance": acceptance},
        "evaluation": {
            "faithfulness": {
                "details": {
                    "skipped_reason": reason,
                    "supported_count": 1 if verified else None,
                    "unsupported_count": 0 if verified else None,
                    "unverified_count": 0 if verified else None,
                }
            }
        },
        "quality": {"status": "passed" if verified else "unverified"},
        "delivery": {"publishable": verified},
    }


def live_runner(tmp_path, monkeypatch, reports):
    """Replace only the generation transport and audit response, not result routing."""
    calls = []

    class OfflineAgent:
        def __init__(self, *args, **kwargs):
            self._cost_budget = CostBudget()
            self._cost_budget.configure_strict()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def run(self, query, *, options):
            assert options.quality_policy == "strict_auto_v1"
            calls.append(query)
            self._cost_budget.begin_request()
            self._cost_budget.record({"prompt_tokens": 10, "completion_tokens": 5})
            return copy.deepcopy(reports[min(len(calls) - 1, len(reports) - 1)])

    monkeypatch.setattr(quality_suite, "LitAgent", OfflineAgent)
    runner = suite(tmp_path, executor=None)
    runner.config.observability.enabled = False
    # Corpus preparation is unrelated to the producer/consumer stop contract.
    runner.executor = runner._live
    runner.service.wait = AsyncMock(
        return_value={
            "judge": {"status": "completed"},
            "judge_usage": {
                "total_tokens": 10,
                "call_count": 2,
                "usage_complete": True,
            },
            "jev": {
                "status": "completed",
                "usage": {
                    "input_tokens": 10,
                    "call_count": 1,
                    "usage_complete": True,
                },
            },
        }
    )
    return runner, calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "evaluation_inventory_shape_invalid",
        "strict_writing_passage_fields_invalid",
        "strict_writing_passage_binding_invalid",
        "strict_writing_summary_binding_invalid",
        "strict_repair_summary_passage_not_in_body",
        "strict_repair_body_binding_invalid",
        "evaluation_inventory_duplicate_key",
        "evaluation_inventory_unit_coverage_invalid",
        "evaluation_inventory_fragment_coverage_invalid",
        "evaluation_inventory_host_partition_invalid",
        "evaluation_inventory_claim_empty",
        "evaluation_source_fragment_invalid",
        "evaluation_citation_ownership_invalid",
        "evaluation_duplicate_assertion",
        "evaluation_unit_exclusion_unexplained",
        "evaluation_unit_exclusion_conflict",
        "evaluation_assessment_unit_coverage_invalid",
        "evaluation_decision_coverage_invalid",
        "evaluation_global_unit_coverage_invalid",
        "evaluation_global_issue_identity_invalid",
        "evaluation_core_coverage_invalid",
        "evaluation_conflict_identity_invalid",
        "evaluation_batch_conflicts_not_allowed",
        "evaluation_host_methodology_invalid",
        "evaluation_decision_status_invalid",
        "evaluation_usage_incomplete",
        "evaluation_assertion_capacity_exceeded",
        "evaluation_batch_capacity_exceeded",
    ],
)
async def test_transport_contract_error_stops_live_suite_without_replaying(
    tmp_path, monkeypatch, reason
):
    runner, calls = live_runner(tmp_path, monkeypatch, [report(reason)])
    result = await runner.run()
    assert len(calls) == 1
    assert result["status"] == "stopped" and result["stop_reasons"] == [reason]
    assert result["observations"][0]["technical_failure"] == reason
    assert result["counts"]["pilot"]["run"] == 1
    assert result["counts"]["formal"]["planned"] == 15
    assert result["counts"]["formal"]["run"] == 0
    assert result["budgets"]["survey"]["charged"] == 15
    assert result["budgets"]["survey"]["requests"] == 1
    assert all(result["budgets"][s]["charged"] == 0 for s in ("judge", "jev"))
    assert all(result["budgets"][s]["requests"] == 0 for s in ("judge", "jev"))
    runner.service.wait.assert_not_awaited()
    assert await runner.run() == result
    assert len(calls) == 1
    public = public_projection(result)
    assert public["counts"]["goal_met"] is False
    path = runner.root / "presentation" / runner.suite_id
    for fmt in ("json", "csv", "md", "html"):
        assert (path / f"benchmark.{fmt}").exists()
    assert json.loads((path / "benchmark.json").read_text())["stop_reasons"] == [reason]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "evaluation_classification_or_source_invalid",
        "evaluation_global_structure_invalid",
        "evaluation_extraction_incomplete",
        "evaluation_assertions_unverifiable",
    ],
)
async def test_semantic_failures_remain_visible_and_continue_the_fixed_matrix(
    tmp_path, monkeypatch, reason
):
    runner, calls = live_runner(
        tmp_path,
        monkeypatch,
        [report(reason), report(None, verified=True), report(reason)],
    )
    result = await runner.run()
    assert len(calls) == 18 and result["status"] == "completed"
    assert result["counts"]["pilot"]["automatic_passed"] == 1
    assert result["counts"]["formal"]["planned"] == 15
    assert result["counts"]["formal"]["run"] == 15
    assert result["counts"]["formal"]["blocked"] == 15
    assert result["counts"]["formal"]["automatic_passed"] == 0
    assert not result["counts"]["goal_met"]
    assert result["observations"][0]["failure_codes"] == [reason, reason]
    assert result["observations"][0]["technical_failure"] is None
    runner.service.wait.assert_awaited_once()
    assert await runner.run() == result
    assert len(calls) == 18


@pytest.mark.asyncio
async def test_protocol_failure_in_formal_stage_preserves_spent_and_unrun_cases(
    tmp_path, monkeypatch
):
    reason = "evaluation_inventory_fragment_coverage_invalid"
    runner, calls = live_runner(
        tmp_path,
        monkeypatch,
        [report(None, verified=True)] * 3 + [report(reason)],
    )
    result = await runner.run()
    assert len(calls) == 4 and result["stop_reasons"] == [reason]
    assert result["counts"]["formal"]["run"] == 1
    assert result["counts"]["formal"]["blocked"] == 1
    assert result["counts"]["formal"]["not_run"] == 14
    assert result["counts"]["formal"]["pass_rate_planned"] == 0.0
    assert result["budgets"]["survey"]["charged"] == 60
    assert result["budgets"]["survey"]["requests"] == 4
    assert result["budgets"]["judge"]["requests"] == 6
    assert result["budgets"]["jev"]["requests"] == 3
    assert await runner.run() == result
    assert len(calls) == 4


@pytest.mark.asyncio
async def test_first_technical_reason_is_preserved_with_existing_provider_failures(
    tmp_path, monkeypatch
):
    value = report("evaluation_inventory_shape_invalid")
    value["metadata"]["automatic_acceptance"]["failure_codes"] = [
        "provider_authentication_failed"
    ]
    runner, calls = live_runner(tmp_path, monkeypatch, [value])
    result = await runner.run()
    assert len(calls) == 1
    assert result["stop_reasons"] == ["evaluation_inventory_shape_invalid"]


def test_export_unknown_usage_stays_unavailable_and_stop_reason_is_visible(tmp_path):
    """Never display an unknown transport total as a measured zero."""
    from litagent.benchmark.quality_suite import export_bundle, summarize

    result = {
        "planned": [
            {"phase": "pilot", "case_id": name, "domain": name}
            for name in ("vit", "few_shot", "nerf")
        ],
        "observations": [
            {
                "phase": "pilot",
                "case_id": "vit",
                "status": "failed",
                "costs": {
                    s: {
                        "tokens": None if s == "jev" else 0,
                        "requests": 2 if s == "jev" else 0,
                        "usage_complete": s != "jev",
                    }
                    for s in ("survey", "judge", "jev")
                },
            }
        ],
        "stop_reasons": ["jev_connection_error <private>"],
        "budgets": {},
    }
    result["counts"] = summarize(result)
    public = public_projection(result)
    assert public["service_usage"]["jev"]["accounted_tokens"] is None
    assert public["service_usage"]["survey"]["accounted_tokens"] == 0
    export_bundle(tmp_path, result)
    document = (tmp_path / "benchmark.html").read_text()
    assert 'id="stop-summary">jev_connection_error &lt;private&gt;' in document
    assert 'id="pilot-summary">试点 / Pilot: 1/3' in document
    assert "不可用 / unavailable" in (tmp_path / "benchmark.md").read_text()
