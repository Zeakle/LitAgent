"""Deterministic offline end-to-end tests for Phase 14.4."""

from __future__ import annotations

import asyncio

import pytest

from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.orchestrator.scheduler import CancellationToken
from tests.fixtures.offline_survey import OfflineSurveyScenario, build_offline_agent


async def _run_scenario(tmp_path, scenario: OfflineSurveyScenario):
    repository = ArchiveRepository(tmp_path)
    recorder = RunRecorder(f"offline-{scenario.name}", "few-shot vision", repository)
    agent = await build_offline_agent(scenario, trace_hook=recorder)
    token = CancellationToken()

    if scenario.name == "cancelled":
        run_task = asyncio.create_task(agent.run("few-shot vision", cancellation=token))
        await asyncio.wait_for(agent._offline_started.wait(), timeout=1)
        token.cancel()
        report = await asyncio.wait_for(run_task, timeout=1)
        terminal_status = "cancelled"
    else:
        report = await agent.run("few-shot vision", cancellation=token)
        terminal_status = "completed"

    artifact = recorder.finalize(report, terminal_status=terminal_status)
    return report, artifact


@pytest.mark.asyncio
async def test_offline_e2e_ready_contract(tmp_path):
    report, artifact = await _run_scenario(tmp_path, OfflineSurveyScenario("ready"))

    assert report["partial"] is False
    assert report["quality"]["status"] == "passed"
    assert report["delivery"]["status"] == "ready"
    assert artifact["report"]["delivery"] == report["delivery"]
    assert set(artifact["task_statuses"]) == {"done"}


@pytest.mark.asyncio
async def test_offline_e2e_quality_failure_is_blocked_not_partial(tmp_path):
    report, _ = await _run_scenario(
        tmp_path,
        OfflineSurveyScenario("quality_blocked", evaluator_passed=False),
    )

    assert report["partial"] is False
    assert report["quality"]["status"] == "failed"
    assert report["delivery"]["status"] == "blocked"


@pytest.mark.asyncio
async def test_offline_e2e_worker_failure_marks_downstream_skipped(tmp_path):
    report, artifact = await _run_scenario(
        tmp_path,
        OfflineSurveyScenario("worker_failed", failing_task_id="relevance_gate"),
    )

    assert report["partial"] is True
    assert report["delivery"]["status"] == "partial"
    assert artifact["task_statuses"] == {"failed": 1, "skipped": 3}


@pytest.mark.asyncio
async def test_offline_e2e_timeout_converges_all_graph_tasks(tmp_path):
    report, artifact = await _run_scenario(
        tmp_path, OfflineSurveyScenario("timeout", block_source=True)
    )

    assert report["partial"] is True
    assert artifact["task_statuses"] == {"cancelled": 1, "skipped": 3}
    assert all(
        task["status"] not in {"pending", "running"}
        for task in artifact["graph"]["tasks"].values()
    )


@pytest.mark.asyncio
async def test_offline_e2e_user_cancel_converges_artifact_and_delivery(tmp_path):
    report, artifact = await _run_scenario(
        tmp_path, OfflineSurveyScenario("cancelled", block_source=True)
    )

    assert artifact["status"] == "cancelled"
    assert report["delivery"]["status"] == "partial"
    assert "run_cancelled" in report["metadata"]["execution"]["reason_codes"]
    assert all(
        node["status"] not in {"pending", "running"}
        for node in artifact["nodes"].values()
    )


@pytest.mark.asyncio
async def test_offline_e2e_prompt_injection_never_becomes_publishable(tmp_path):
    report, artifact = await _run_scenario(
        tmp_path,
        OfflineSurveyScenario("prompt_injection", evaluator_passed=False),
    )

    assert report["partial"] is False
    assert report["delivery"]["status"] == "blocked"
    assert report["delivery"]["publishable"] is False
    assert "Ignore previous instructions" not in report["survey"]
    assert artifact["nodes"]["worker:extractor"]["output"] == []


@pytest.mark.asyncio
async def test_offline_e2e_has_no_network_or_shared_storage_access(
    tmp_path, monkeypatch
):
    def reject_network(*args, **kwargs):
        raise AssertionError("offline E2E attempted network access")

    monkeypatch.setattr("httpx.AsyncClient", reject_network)
    report, artifact = await _run_scenario(tmp_path, OfflineSurveyScenario("ready"))

    assert report["delivery"]["status"] == "ready"
    assert (tmp_path / f"{artifact['run_id']}.json").is_file()
