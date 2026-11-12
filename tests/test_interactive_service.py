"""Verify the durable interactive service across reconstructed agent segments."""

import asyncio

import pytest

from litagent.config import AgentConfig, AppConfig, LoggingConfig, ObservabilityConfig
from litagent.interaction import ExecutionCheckpoint, RunAdvanceResult
from litagent.interaction_store import SqliteReviewRepository
from litagent.interactive_service import InteractiveRunService
from litagent.observability.recorder import ArchiveRepository
from litagent.workbench import (
    HumanDecision,
    ResumeIdentity,
    ReviewRequest,
    RunOptions,
)


async def wait_for_status(service, run_id, expected):
    for _ in range(100):
        state = await service.get(run_id)
        if state["status"] == expected:
            return state
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not reach {expected}: {state}")


@pytest.mark.asyncio
async def test_service_reconstructs_agent_and_resumes_without_restarting_product(
    tmp_path,
):
    repository = SqliteReviewRepository(tmp_path)
    calls = []

    class FakeAgent:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def advance(self, query, **kwargs):
            calls.append({"query": query, **kwargs})
            options = kwargs["options"]
            if kwargs["checkpoint"] is None:
                run_id = kwargs["run_id"]
                checkpoint = ExecutionCheckpoint(
                    identity=ResumeIdentity(
                        run_id=run_id,
                        source_fingerprint="a" * 64,
                        config_fingerprint="b" * 64,
                        options=options,
                    ),
                    query=query,
                    options=options,
                    boundary="paper_selection",
                    graph={"schema_version": 1, "tasks": [], "dependencies": {}},
                    results={},
                    budget={"schema_version": 1},
                    review_request_id="request-1",
                )
                review = ReviewRequest(
                    request_id="request-1",
                    run_id=run_id,
                    stage="paper_selection",
                    revision=1,
                    candidate_paper_ids=("p1",),
                )
                return RunAdvanceResult(
                    status="awaiting_human",
                    checkpoint=checkpoint,
                    review=review,
                )
            return RunAdvanceResult(status="completed", report={"survey": "# Final"})

    service = InteractiveRunService(
        AppConfig(
            agent=AgentConfig(),
            logging=LoggingConfig(),
            observability=ObservabilityConfig(enabled=False),
        ),
        repository,
        agent_factory=lambda _config, _hook: FakeAgent(),
        archive_repository=ArchiveRepository(tmp_path / "artifacts"),
    )
    await service.initialize()
    run_id = await service.start("query", RunOptions(human_review=True))
    await wait_for_status(service, run_id, "awaiting_human")

    await service.submit_decision(
        run_id,
        HumanDecision(
            request_id="request-1",
            revision=1,
            action="confirm",
            selected_paper_ids=("p1",),
        ),
    )
    state = await wait_for_status(service, run_id, "completed")
    assert state["result"] == {"survey": "# Final"}
    assert len(calls) == 2
    assert calls[1]["checkpoint"] is not None
    assert calls[1]["decision"].selected_paper_ids == ("p1",)
    assert (tmp_path / "artifacts" / f"{run_id}.json").is_file()
    await service.shutdown()


@pytest.mark.asyncio
async def test_service_events_have_stable_sequence(tmp_path):
    repository = SqliteReviewRepository(tmp_path)
    await repository.initialize()
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    hook = repository.event_hook("run-1")
    hook("worker.start", {"task_id": "a"})
    hook("worker.complete", {"task_id": "a"})
    assert [event["seq"] for event in await repository.list_events("run-1")] == [
        1,
        2,
    ]
    assert [
        event["seq"] for event in await repository.list_events("run-1", after=1)
    ] == [2]


@pytest.mark.asyncio
async def test_service_persists_and_exports_each_report_revision(tmp_path):
    repository = SqliteReviewRepository(tmp_path)
    service = InteractiveRunService(
        AppConfig(
            agent=AgentConfig(),
            logging=LoggingConfig(),
            observability=ObservabilityConfig(enabled=False),
        ),
        repository,
        archive_repository=ArchiveRepository(tmp_path / "artifacts"),
    )
    await repository.initialize()
    await repository.create_run(
        "run-1", "query", RunOptions(report_language="en", human_review=True)
    )
    await repository.save_report_version(
        "run-1",
        1,
        {
            "survey": "# First",
            "quality": {"status": "passed"},
            "delivery": {"status": "needs_review", "publishable": False},
            "metadata": {
                "report_revision": 1,
                "report_language": "en",
                "human_review": {"status": "pending"},
            },
        },
    )
    await repository.save_report_version(
        "run-1",
        2,
        {
            "survey": "# Second",
            "quality": {"status": "passed"},
            "delivery": {"status": "ready", "publishable": True},
            "metadata": {
                "report_revision": 2,
                "report_language": "en",
                "human_review": {"status": "approved"},
            },
        },
    )

    versions = await service.versions("run-1")
    assert [item["revision"] for item in versions] == [2, 1]
    exported = await service.export("run-1", 1, "md")
    assert b"# First" in exported.content
    assert b"# Second" not in exported.content


@pytest.mark.asyncio
async def test_service_view_uses_checkpoint_graph_and_persisted_events(tmp_path):
    repository = SqliteReviewRepository(tmp_path)
    service = InteractiveRunService(
        AppConfig(
            agent=AgentConfig(),
            logging=LoggingConfig(),
            observability=ObservabilityConfig(enabled=False),
        ),
        repository,
        archive_repository=ArchiveRepository(tmp_path / "artifacts"),
    )
    await repository.initialize()
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    checkpoint = ExecutionCheckpoint(
        identity=ResumeIdentity(
            run_id="run-1",
            source_fingerprint="a" * 64,
            config_fingerprint="b" * 64,
            options=RunOptions(human_review=True),
        ),
        query="query",
        options=RunOptions(human_review=True),
        boundary="paper_selection",
        graph={
            "schema_version": 1,
            "tasks": [
                {
                    "task_id": "search",
                    "description": "search",
                    "agent_type": "search",
                    "input_data": {},
                    "output_schema": {},
                    "status": "done",
                    "priority": 0,
                    "timeout_ms": 100,
                    "max_retries": 0,
                    "result": [{"paper_id": "p1"}],
                    "error": None,
                },
                {
                    "task_id": "dedup",
                    "description": "dedup",
                    "agent_type": "dedup",
                    "input_data": {},
                    "output_schema": {},
                    "status": "done",
                    "priority": 0,
                    "timeout_ms": 100,
                    "max_retries": 0,
                    "result": [{"paper_id": "p1", "title": "Paper"}],
                    "error": None,
                },
            ],
            "dependencies": {"search": [], "dedup": ["search"]},
        },
        results={},
        budget={"schema_version": 1},
        review_request_id="request-1",
    )
    await repository.pause(
        "run-1",
        checkpoint.model_dump(mode="json"),
        ReviewRequest(
            request_id="request-1",
            run_id="run-1",
            stage="paper_selection",
            revision=1,
            candidate_paper_ids=("p1",),
        ),
    )
    repository.event_hook("run-1")("worker.complete", {"task_id": "search"})

    view = await service.get_view("run-1")

    assert set(view["graph"]["tasks"]) == {"search", "dedup"}
    assert view["graph"]["dependencies"] == {"search": [], "dedup": ["search"]}
    assert view["timeline"][0]["event"] == "worker.complete"
    assert view["review"]["stage"] == "paper_selection"
    assert view["capabilities"]["can_review"] is True


@pytest.mark.asyncio
async def test_service_version_preview_binds_body_and_sha(tmp_path):
    repository = SqliteReviewRepository(tmp_path)
    service = InteractiveRunService(
        AppConfig(
            agent=AgentConfig(),
            logging=LoggingConfig(),
            observability=ObservabilityConfig(enabled=False),
        ),
        repository,
    )
    await repository.initialize()
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    await repository.save_report_version(
        "run-1",
        2,
        {
            "survey": "# Version two",
            "evaluation": {"faithfulness": {"score": 0.9, "passed": True}},
            "quality": {"status": "passed"},
            "delivery": {"status": "ready", "publishable": True},
            "metadata": {"report_revision": 2, "report_language": "en"},
        },
    )

    version = await service.get_version("run-1", 2)

    assert version["revision"] == 2
    assert version["report"]["survey"] == "# Version two"
    assert version["content_sha256"] == version["report_sha256"]
    assert len(version["content_sha256"]) == 64
    assert '<h1 id="version-two">Version two</h1>' in version["rendered_report"]["html"]
