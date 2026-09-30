"""Verify durable review decisions and checkpoint integrity."""

import asyncio

import pytest
import pytest_asyncio

from litagent.interaction_store import SqliteReviewRepository
from litagent.workbench import HumanDecision, ReviewRequest, RunOptions


@pytest_asyncio.fixture
async def repository(tmp_path):
    repo = SqliteReviewRepository(tmp_path)
    await repo.initialize()
    return repo


def pending_request():
    return ReviewRequest(
        request_id="req-1",
        run_id="run-1",
        stage="paper_selection",
        revision=1,
        candidate_paper_ids=("p1", "p2"),
    )


@pytest.mark.asyncio
async def test_pause_roundtrip_and_strict_sha(repository):
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    await repository.pause(
        "run-1", {"schema_version": 1, "graph": {"tasks": []}}, pending_request()
    )
    assert (await repository.get_pending("run-1")).request_id == "req-1"
    assert (await repository.load_checkpoint("run-1"))["graph"] == {"tasks": []}

    path = repository.checkpoint_path("run-1")
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint_sha_mismatch"):
        await repository.load_checkpoint("run-1")


@pytest.mark.asyncio
async def test_decision_is_idempotent_but_conflicts_on_changed_payload(repository):
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    await repository.pause("run-1", {"schema_version": 1}, pending_request())
    decision = HumanDecision(
        request_id="req-1",
        revision=1,
        action="confirm",
        selected_paper_ids=("p1",),
    )
    first = await repository.decide("run-1", decision)
    second = await repository.decide("run-1", decision)
    assert first == second
    assert first.execution_status == "queued"

    changed = decision.model_copy(update={"selected_paper_ids": ("p2",)})
    with pytest.raises(ValueError, match="decision_conflict"):
        await repository.decide("run-1", changed)


@pytest.mark.asyncio
async def test_only_one_executor_can_claim_queued_run(repository):
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    claimed = await asyncio.gather(
        repository.claim_execution("run-1"),
        repository.claim_execution("run-1"),
    )
    assert sorted(claimed) == [False, True]


@pytest.mark.asyncio
async def test_running_rows_become_interrupted_after_restart(repository):
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    assert await repository.claim_execution("run-1")
    assert await repository.recover_startup() == ["run-1"]
    assert (await repository.get_run("run-1"))["status"] == "interrupted"


@pytest.mark.asyncio
async def test_unknown_run_is_not_a_cache_miss(repository):
    with pytest.raises(KeyError):
        await repository.get_run("missing")
    with pytest.raises(KeyError):
        await repository.load_checkpoint("missing")


@pytest.mark.asyncio
async def test_report_versions_are_immutable_and_ordered(repository):
    await repository.create_run("run-1", "query", RunOptions(human_review=True))
    first = {
        "survey": "# Draft one",
        "metadata": {"report_revision": 1, "report_language": "en"},
    }
    second = {
        "survey": "# Draft two",
        "metadata": {"report_revision": 2, "report_language": "en"},
    }

    await repository.save_report_version("run-1", 1, first)
    await repository.save_report_version("run-1", 1, first)
    await repository.save_report_version("run-1", 2, second)

    versions = await repository.list_report_versions("run-1")
    assert [item["revision"] for item in versions] == [2, 1]
    assert (await repository.get_report_version("run-1", 1))["survey"] == (
        "# Draft one"
    )

    with pytest.raises(ValueError, match="report_revision_conflict"):
        await repository.save_report_version(
            "run-1", 1, {**first, "survey": "# Replaced"}
        )
    with pytest.raises(KeyError):
        await repository.get_report_version("run-1", 99)
