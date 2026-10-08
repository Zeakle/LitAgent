"""Verify automatic audit consent and immutable report persistence boundaries."""

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from pydantic import ValidationError

from litagent import api
from litagent.config import AgentConfig, AppConfig, LoggingConfig, ObservabilityConfig
from litagent.interaction import ExecutionCheckpoint, RunAdvanceResult
from litagent.interaction_store import SqliteReviewRepository
from litagent.interactive_service import InteractiveRunService
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.report_audit import ReportAuditService
from litagent.runner import CancellationToken
from litagent.workbench import ResumeIdentity, ReviewRequest, RunOptions


def config():
    """Disable remote telemetry for all fixtures."""
    return AppConfig(
        agent=AgentConfig(),
        logging=LoggingConfig(),
        observability=ObservabilityConfig(enabled=False),
    )


def options(**updates):
    """Return a persisted, explicitly authorized automatic policy."""
    values = {
        "human_review": True,
        "auto_audit_mode": "judge",
        "auto_audit_max_versions": 1,
    }
    values.update(updates)
    return RunOptions(**values)


def paused_result(run_id, policy, *, stage="final_report", revision=1):
    """Construct actual checkpoint and review contracts at a safe boundary."""
    final = stage == "final_report"
    report = (
        {
            "survey": "# Final\n\nClaim [E:ev-1].",
            "quality": {"status": "passed"},
            "partial": False,
            "delivery": {"status": "needs_review", "publishable": False},
            "metadata": {
                "query": "A question",
                "report_language": "en",
                "report_revision": revision,
                "human_review": {"status": "pending"},
                "automatic_delivery": {"status": "ready", "publishable": True},
            },
        }
        if final
        else None
    )
    checkpoint = ExecutionCheckpoint(
        identity=ResumeIdentity(
            run_id=run_id,
            source_fingerprint="a" * 64,
            config_fingerprint="b" * 64,
            options=policy,
        ),
        query="A question",
        options=policy,
        boundary=stage,
        graph={"schema_version": 1, "tasks": [], "dependencies": {}},
        results={},
        budget={"schema_version": 1},
        report_data=report,
        report_revision=revision if final else 0,
        review_request_id=f"request-{revision}",
    )
    review = ReviewRequest(
        request_id=checkpoint.review_request_id,
        run_id=run_id,
        stage=stage,
        revision=revision,
        report_revision=revision if final else None,
        report_sha256=(
            hashlib.sha256(report["survey"].encode()).hexdigest() if final else None
        ),
        candidate_paper_ids=() if final else ("paper-1",),
    )
    return RunAdvanceResult(
        status="awaiting_human", checkpoint=checkpoint, review=review
    )


@pytest.mark.parametrize("structure", ["legacy", "academic_v1"])
def test_off_defaults_preserve_existing_fingerprint_and_checkpoint(structure):
    """Defaults deserialize old options without granting paid authorization."""
    previous = {"report_language": "zh-CN", "human_review": True}
    if structure != "legacy":
        previous["review_structure"] = structure
    policy = RunOptions.model_validate(previous)
    expected = hashlib.sha256(json.dumps(previous, sort_keys=True).encode()).hexdigest()
    assert policy.auto_audit_mode == "off"
    assert policy.auto_audit_max_versions == 1
    assert policy.fingerprint() == expected
    old = ResumeIdentity(
        run_id="old",
        source_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        options=previous,
    )
    restored = ResumeIdentity.model_validate_json(
        json.dumps(
            {
                "run_id": "old",
                "source_fingerprint": "a" * 64,
                "config_fingerprint": "b" * 64,
                "options": previous,
            }
        )
    )
    old.require_compatible(restored)
    enabled = policy.model_copy(update={"auto_audit_mode": "judge"})
    assert enabled.fingerprint() != expected
    with pytest.raises(ValueError, match="incompatible_checkpoint"):
        old.require_compatible(restored.model_copy(update={"options": enabled}))
    increased = enabled.model_copy(update={"auto_audit_max_versions": 2})
    assert increased.fingerprint() != enabled.fingerprint()
    active = restored.model_copy(update={"options": enabled})
    with pytest.raises(ValueError, match="incompatible_checkpoint"):
        active.require_compatible(active.model_copy(update={"options": increased}))


@pytest.mark.parametrize("model", [RunOptions, api.SurveyRequest])
@pytest.mark.parametrize(
    "extra",
    [
        {"auto_audit_mode": "automatic"},
        {"auto_audit_max_versions": 0},
        {"auto_audit_max_versions": 4},
        {"auto_audit_model": "untrusted-model"},
    ],
)
def test_automatic_policy_rejects_unbounded_or_untrusted_request_fields(model, extra):
    """API and persistence share the same bounded authorization vocabulary."""
    body = {"query": "question"} if model is api.SurveyRequest else {}
    with pytest.raises(ValidationError):
        model.model_validate({**body, **extra})


@pytest.fixture
def audit_api(monkeypatch, tmp_path):
    """Stub submission ownership while keeping actual HTTP authorization checks."""
    capabilities = {
        "enabled": True,
        "judge_available": True,
        "jev_available": True,
        "judge_model": "offline-fixture",
        "jev_model": "jev-1.13.0",
        "judge_reason": None,
        "jev_reason": None,
        "limits": {"judge_calls": 2},
    }
    service = MagicMock()
    service.capabilities.side_effect = lambda: dict(capabilities)
    monkeypatch.setattr(api.app.state, "report_audit_service", service)
    monkeypatch.setattr(api.app.state, "config", config())
    repository = ArchiveRepository(tmp_path / "archive")
    recorder = MagicMock()
    entry = api.SurveyTaskEntry(
        task_id="offline-survey",
        query="question",
        status="queued",
        progress="queued",
        cancellation=CancellationToken(),
        recorder=recorder,
    )
    create = MagicMock(return_value=entry)
    monkeypatch.setattr(api, "_new_entry", create)
    monkeypatch.setattr(api.app.state, "flow_repository", repository)
    return capabilities, create, entry


@pytest.mark.asyncio
async def test_automatic_survey_requires_same_origin_token_without_human_review(
    audit_api,
):
    """Automatic billing consent remains protected for unattended Surveys."""
    _, create, _ = audit_api
    body = {"query": "question", "auto_audit_mode": "judge", "human_review": False}
    headers = {"X-LitAgent-Operation-Token": api.app.state.operation_token}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        assert (await client.post("/survey", json=body)).status_code == 403
        create.assert_not_called()
        denied = await client.post(
            "/survey", json=body, headers={**headers, "Origin": "http://evil.test"}
        )
        assert denied.status_code == 403
        create.assert_not_called()
        accepted = await client.post("/survey", json=body, headers=headers)
        assert accepted.status_code == 202
        policy = create.call_args.kwargs["options"]
        assert policy.auto_audit_mode == "judge" and policy.auto_audit_max_versions == 1


@pytest.mark.asyncio
async def test_old_survey_request_remains_off_without_new_token_requirement(audit_api):
    """Historic unattended requests cannot acquire new automatic costs."""
    _, create, _ = audit_api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post("/survey", json={"query": "question"})
    assert response.status_code == 202
    assert create.call_args.kwargs["options"].auto_audit_mode == "off"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,disabled,reason",
    [
        ("judge", "judge_available", "missing_key:LLM_API_KEY"),
        ("judge_jev", "jev_available", "missing_key:TYPESAFE_API_KEY"),
    ],
)
async def test_unavailable_auto_capability_rejects_before_survey_creation(
    audit_api, mode, disabled, reason
):
    """Missing configuration cannot consume generation cost first."""
    capabilities, create, _ = audit_api
    capabilities[disabled] = False
    capabilities["judge_reason" if disabled == "judge_available" else "jev_reason"] = (
        reason
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/survey",
            json={"query": "question", "auto_audit_mode": mode},
            headers={"X-LitAgent-Operation-Token": api.app.state.operation_token},
        )
    assert response.status_code in {409, 422, 503}
    assert reason in response.text
    create.assert_not_called()


@pytest.mark.asyncio
async def test_workbench_context_exposes_capability_without_scheduling_audit(audit_api):
    """Reading capability never submits or resumes an audit."""
    capabilities, _, _ = audit_api
    service = api.app.state.report_audit_service
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        result = await client.get("/workbench/context")
    assert result.status_code == 200
    automatic = result.json()["auto_audit_capabilities"]
    assert automatic["judge_available"] == capabilities["judge_available"]
    assert automatic["jev_available"] == capabilities["jev_available"]
    service.submit.assert_not_called()
    service.submit_automatic.assert_not_called()


@pytest.mark.asyncio
async def test_final_report_hook_runs_only_after_version_and_pause_are_durable(
    tmp_path,
):
    """A hook sees the same immutable version the human can subsequently review."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    policy = options()
    await repo.create_run("report-run", "A question", policy)
    result = paused_result("report-run", policy)
    calls = []

    async def submit(run_id, revision, stored_options):
        """Read committed state at the callback boundary."""
        row = await repo.get_run(run_id)
        report = await repo.get_report_version(run_id, revision)
        checkpoint = await repo.load_checkpoint(run_id)
        assert row["status"] == "awaiting_human"
        assert report == result.checkpoint.report_data
        assert checkpoint["report_revision"] == revision
        assert stored_options == RunOptions.model_validate_json(row["options_json"])
        calls.append((run_id, revision, stored_options))

    service = InteractiveRunService(config(), repo, auto_audit_submitter=submit)
    before = result.checkpoint.model_dump_json()
    await service._persist_result("report-run", result)
    assert calls == [("report-run", 1, policy)]
    assert result.checkpoint.model_dump_json() == before


@pytest.mark.asyncio
async def test_paper_selection_and_default_off_never_invoke_auto_callback(tmp_path):
    """Intermediate selection and old policies cannot authorize audit spending."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    callback = AsyncMock()
    service = InteractiveRunService(config(), repo, auto_audit_submitter=callback)
    for run_id, policy, stage in (
        ("selection", options(), "paper_selection"),
        ("legacy", RunOptions(human_review=True), "final_report"),
    ):
        await repo.create_run(run_id, "A question", policy)
        await service._persist_result(
            run_id, paused_result(run_id, policy, stage=stage)
        )
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_auto_hook_error_does_not_destroy_persisted_human_review(tmp_path):
    """An independent transport failure leaves review and report quality intact."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    policy = options()
    await repo.create_run("report-run", "A question", policy)
    callback = AsyncMock(side_effect=RuntimeError("offline audit failure"))
    service = InteractiveRunService(config(), repo, auto_audit_submitter=callback)
    result = paused_result("report-run", policy)
    await service._persist_result("report-run", result)
    callback.assert_awaited_once()
    row = await repo.get_run("report-run")
    assert row["status"] == "awaiting_human" and row["error"] is None
    assert (
        await repo.get_report_version("report-run", 1) == result.checkpoint.report_data
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["save_report_version", "pause"])
async def test_persistence_error_prevents_automatic_callback(
    tmp_path, monkeypatch, failure_point
):
    """A partially written review cannot be submitted as a frozen final version."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    policy = options()
    await repo.create_run("report-run", "A question", policy)
    callback = AsyncMock()
    service = InteractiveRunService(config(), repo, auto_audit_submitter=callback)
    monkeypatch.setattr(
        repo, failure_point, AsyncMock(side_effect=RuntimeError("storage unavailable"))
    )
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await service._persist_result("report-run", paused_result("report-run", policy))
    callback.assert_not_awaited()


@pytest.mark.asyncio
async def test_initialize_with_historical_final_report_does_not_replay_audit(tmp_path):
    """Startup reads persisted reviews without retroactive automatic submission."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    policy = options()
    await repo.create_run("historical", "A question", policy)
    old = paused_result("historical", policy)
    await repo.save_report_version("historical", 1, old.checkpoint.report_data)
    await repo.pause("historical", old.checkpoint.model_dump(mode="json"), old.review)
    callback = AsyncMock()
    service = InteractiveRunService(config(), repo, auto_audit_submitter=callback)
    await service.initialize()
    callback.assert_not_awaited()
    assert not service._tasks
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "human_status,terminal_status",
    [
        ("approved", "completed"),
        ("rejected", "rejected"),
    ],
)
async def test_human_terminal_decision_preserves_frozen_version_and_audit(
    tmp_path, human_status, terminal_status
):
    """Approval state belongs to the run and cannot rewrite the audit snapshot."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    policy = options()
    await repo.create_run("report-run", "A question", policy)
    callback = AsyncMock()
    service = InteractiveRunService(config(), repo, auto_audit_submitter=callback)
    paused = paused_result("report-run", policy)
    await service._persist_result("report-run", paused)
    frozen = await repo.get_report_version("report-run", 1)
    frozen_bytes = json.dumps(frozen, sort_keys=True)
    approved = json.loads(json.dumps(frozen))
    approved["metadata"]["human_review"]["status"] = human_status
    approved["delivery"] = {
        "status": "ready" if human_status == "approved" else "blocked",
        "publishable": human_status == "approved",
    }
    await service._persist_result(
        "report-run", RunAdvanceResult(status=terminal_status, report=approved)
    )
    row = await repo.get_run("report-run")
    assert row["status"] == terminal_status
    assert (
        json.loads(row["result_json"])["metadata"]["human_review"]["status"]
        == human_status
    )
    assert (
        json.dumps(await repo.get_report_version("report-run", 1), sort_keys=True)
        == frozen_bytes
    )
    callback.assert_awaited_once_with("report-run", 1, policy)


@pytest.mark.asyncio
async def test_terminal_same_revision_with_different_body_still_rejects(tmp_path):
    """Approval handling must not weaken immutable report content checks."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    policy = options()
    await repo.create_run("report-run", "A question", policy)
    callback = AsyncMock()
    service = InteractiveRunService(config(), repo, auto_audit_submitter=callback)
    paused = paused_result("report-run", policy)
    await service._persist_result("report-run", paused)
    revised = json.loads(json.dumps(paused.checkpoint.report_data))
    revised["survey"] += "\nChanged without a new revision."
    with pytest.raises(ValueError, match="report_revision_conflict"):
        await service._persist_result(
            "report-run", RunAdvanceResult(status="completed", report=revised)
        )
    callback.assert_awaited_once()
    assert (await repo.get_report_version("report-run", 1))["survey"] != revised[
        "survey"
    ]


@pytest.mark.asyncio
async def test_plain_survey_freezes_its_own_artifact_without_lookup_or_mutation(
    monkeypatch,
):
    """Ordinary API runs need no flow-demo index to freeze their own evidence."""
    report = paused_result("plain-api", options()).checkpoint.report_data
    artifact = {
        "run_id": "plain-api",
        "query": "A question",
        "status": "completed",
        "report": report,
        "nodes": {"worker-node": {"output": {"items": []}}},
        "config": {"llm": {"model": "offline-generation", "origin": "http://fixture"}},
    }
    artifact_before = json.dumps(artifact, sort_keys=True)
    evidence = [{"evidence_id": "ev-1", "paper_id": "paper-1", "text": "Evidence."}]
    service = MagicMock()
    service.submit_automatic = AsyncMock()
    monkeypatch.setattr(api.app.state, "report_audit_service", service)
    lookup = MagicMock(side_effect=AssertionError("No archive lookup is permitted"))
    monkeypatch.setattr(api, "_flow_artifact", lookup)
    monkeypatch.setattr(api, "extract_report_evidence", lambda _: evidence)
    await api._submit_automatic_artifact(artifact, options())
    frozen = service.submit_automatic.call_args.args[0]
    assert frozen["source_run_id"] == "plain-api"
    assert frozen["source_report_revision"] == 1
    assert (
        frozen["source_report_sha256"]
        == hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    assert frozen["ledger"] == {"ev-1": evidence[0]}
    assert frozen["generation_provider"]["model"] == "offline-generation"
    assert service.submit_automatic.call_args.kwargs == {
        "mode": "judge",
        "max_versions": 1,
    }
    frozen["report"]["metadata"]["human_review"]["status"] = "unexpected mutation"
    frozen["ledger"]["ev-1"]["text"] = "changed evidence"
    assert json.dumps(artifact, sort_keys=True) == artifact_before
    assert evidence[0]["text"] == "Evidence."
    lookup.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("submission_failure", ["none", "error", "cancelled"])
async def test_noninteractive_hook_follows_terminal_archive_and_isolates_errors(
    tmp_path, monkeypatch, submission_failure
):
    """Audit submission observes a committed report and cannot reverse success."""
    repository = ArchiveRepository(tmp_path / "archive")
    report = paused_result("plain-api", options()).checkpoint.report_data
    report["metadata"]["human_review"]["status"] = "not_requested"
    report["delivery"] = {"status": "ready", "publishable": True}
    recorder = RunRecorder("plain-api", "A question", repository)
    policy = options(human_review=False)
    entry = api.SurveyTaskEntry(
        task_id="plain-api",
        query="A question",
        status="queued",
        progress="queued",
        cancellation=CancellationToken(),
        recorder=recorder,
        options=policy,
    )

    class FakeAgent:
        """Return one local completed report without external resources."""

        def __init__(self, *_args, **_kwargs):
            """Accept the same composition-root arguments as the real agent."""

        async def __aenter__(self):
            """Keep execution entirely within this fixture."""
            return self

        async def __aexit__(self, *_args):
            """Own no external clients or resources."""

        async def run(self, _query, **_kwargs):
            """Return the final local report after the workflow boundary."""
            return report

    async def submit(artifact, stored_options):
        """Check atomic archive and API status before simulating a failure."""
        persisted = repository.get("plain-api")
        assert repository.path_for("plain-api").is_file()
        assert persisted["status"] == "completed" and persisted["report"] == report
        assert artifact == persisted
        assert entry.status == "completed" and entry.result == report
        assert stored_options == policy
        if submission_failure == "error":
            raise RuntimeError("offline automatic submission error")
        if submission_failure == "cancelled":
            raise asyncio.CancelledError

    callback = AsyncMock(side_effect=submit)
    monkeypatch.setattr(api, "LitAgent", FakeAgent)
    monkeypatch.setattr(api, "_submit_automatic_artifact", callback)
    monkeypatch.setattr(api.app.state, "config", config())
    monkeypatch.setattr(api.app.state, "survey_semaphore", asyncio.Semaphore(1))
    if submission_failure == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await api._run_survey(entry)
    else:
        await api._run_survey(entry)
    callback.assert_awaited_once()
    assert entry.status == "completed" and entry.error is None
    assert repository.get("plain-api")["report"] == report
    assert repository.get("plain-api")["status"] == "completed"


@pytest.mark.asyncio
async def test_interactive_auto_helper_uses_exact_version_without_mutating_options(
    monkeypatch,
):
    """An explicit persisted revision is passed unchanged to the audit owner."""
    policy = options(auto_audit_mode="judge_jev", auto_audit_max_versions=2)
    source = {
        "source_run_id": "interactive",
        "source_report_revision": 2,
        "report": {"survey": "r2"},
    }
    resolver = AsyncMock(return_value=source)
    service = MagicMock()
    service.submit_automatic = AsyncMock()
    monkeypatch.setattr(api, "_report_audit_snapshot", resolver)
    monkeypatch.setattr(api.app.state, "report_audit_service", service)
    before = policy.model_dump_json()
    await api._submit_automatic_audit("interactive", 2, policy)
    resolver.assert_awaited_once_with("interactive", 2)
    service.submit_automatic.assert_awaited_once_with(
        source, mode="judge_jev", max_versions=2
    )
    assert policy.model_dump_json() == before


@pytest.mark.asyncio
async def test_cancelled_audit_submission_keeps_committed_pending_review(tmp_path):
    """Audit cancellation cannot interrupt an already committed review boundary."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    await repo.initialize()
    policy = options()
    await repo.create_run("report-run", "A question", policy)
    callback = AsyncMock(side_effect=asyncio.CancelledError)
    service = InteractiveRunService(config(), repo, auto_audit_submitter=callback)
    paused = paused_result("report-run", policy)
    await service._persist_result("report-run", paused)
    assert (await repo.get_run("report-run"))["status"] == "awaiting_human"
    assert (
        await repo.get_report_version("report-run", 1) == paused.checkpoint.report_data
    )
    callback.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("restart", [False, True])
async def test_pending_report_keeps_host_ledger_and_manual_dedup_after_restart(
    tmp_path, monkeypatch, restart
):
    """Local event replay preserves evidence and cannot silently create paid attempts."""
    repo = SqliteReviewRepository(tmp_path / "reviews")
    archive = ArchiveRepository(tmp_path / "archive")
    await repo.initialize()
    policy = options()
    await repo.create_run("pending-run", "A question", policy)
    callback = AsyncMock()
    service = InteractiveRunService(
        config(), repo, archive_repository=archive, auto_audit_submitter=callback
    )
    hook = await service._trace_hook("pending-run", "A question")
    evidence = {
        "evidence_id": "ev-1",
        "paper_id": "paper-1",
        "paper_title": "Paper One",
        "text": "Evidence supports Claim.",
        "content_scope": "abstract",
        "source_locator": "abstract",
    }
    hook(
        "evidence.retrieve.start",
        {
            "operation_id": "selected",
            "task_id": "synthesis",
            "input": {"section": "methods"},
        },
    )
    hook(
        "evidence.retrieve.complete",
        {
            "operation_id": "selected",
            "task_id": "synthesis",
            "output": {"selected_items": {"ev-1": evidence}},
        },
    )
    paused = paused_result("pending-run", policy)
    await service._persist_result("pending-run", paused)
    monkeypatch.setattr(api.app.state, "interactive_service", service)
    monkeypatch.setattr(
        api,
        "_flow_artifact",
        MagicMock(
            side_effect=AssertionError("Pending evidence must come from its owner")
        ),
    )
    first = await api._report_audit_snapshot("pending-run", 1)
    assert first["ledger"]["ev-1"]["text"] == evidence["text"]
    assert first["report"] == paused.checkpoint.report_data
    assert not archive.path_for("pending-run").exists()
    if restart:
        service = InteractiveRunService(
            config(), repo, archive_repository=archive, auto_audit_submitter=AsyncMock()
        )
        await service.initialize()
        service._auto_audit_submitter.assert_not_awaited()
        monkeypatch.setattr(api.app.state, "interactive_service", service)
    second = await api._report_audit_snapshot("pending-run", 1)
    third = await api._report_audit_snapshot("pending-run", 1)
    assert second["ledger"] == third["ledger"] == first["ledger"]
    assert second["source_report_sha256"] == first["source_report_sha256"]
    assert not archive.path_for("pending-run").exists()
    monkeypatch.setenv("LLM_API_KEY", "offline-fixture-key")
    monkeypatch.setenv("LITAGENT_DEMO_MODE", "live")
    auditor = ReportAuditService(config(), root=tmp_path / "audits")
    execute = AsyncMock()
    monkeypatch.setattr(auditor, "_execute", execute)
    try:
        attempt = await auditor.submit(first)
        repeated = await auditor.submit(second)
        again = await auditor.submit(third)
        assert repeated["audit_id"] == again["audit_id"] == attempt["audit_id"]
        assert repeated["reused"] and again["reused"]
        assert len((await auditor.list("pending-run", 1))["audits"]) == 1
    finally:
        await auditor.shutdown()
        await service.shutdown()
