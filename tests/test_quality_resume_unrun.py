"""Continue a fixed campaign without replaying slots or resetting its ledger."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_strict_automatic import observation, suite

from litagent.benchmark.quality_suite import QualitySuiteRunner


def resume(parent, executor):
    return QualitySuiteRunner(
        parent.config,
        parent.dataset,
        parent.profile,
        root=parent.root,
        run_archive=parent.archive,
        suite_id="resume-unrun",
        prior_suite_id=parent.suite_id,
        resume_unrun=True,
        executor=executor,
    )


@pytest.mark.asyncio
async def test_remaining_fixed_slots_and_accounting_are_not_replayed(tmp_path):
    parent = suite(tmp_path, AsyncMock(return_value=observation(False, complete=False)))
    prior = await parent.run(max_new_observations=2)
    raw = (tmp_path / f"{parent.suite_id}.json").read_bytes()
    executor = AsyncMock(return_value=observation())
    child = resume(parent, executor)
    preview = await child.run(dry_run=True, max_new_observations=16)
    assert preview["observations"] == prior["observations"]
    assert preview["prior_rounds"] == 0 and preview["budgets"] == prior["budgets"]
    result = await child.run(max_new_observations=16)
    assert executor.await_count == 16
    assert executor.await_args_list[0].args[0].case_id == "nerf-foundations"
    assert len(result["observations"]) == 18
    assert result["counts"]["pilot"]["blocked"] == 2
    assert result["counts"]["formal"]["run"] == 15
    assert result["budgets"]["survey"]["charged"] == 180
    assert result["budgets"]["survey"]["requests"] == 18
    assert result["observations"][:2] == prior["observations"]
    assert (tmp_path / f"{parent.suite_id}.json").read_bytes() == raw
    assert await child.run(max_new_observations=16) == result
    assert executor.await_count == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unknown_usage", "interrupted", "order", "budget"])
async def test_resume_rejects_unknown_or_inconsistent_history(tmp_path, fault):
    parent = suite(tmp_path, AsyncMock(return_value=observation(False, complete=False)))
    prior = await parent.run(max_new_observations=2)
    if fault == "unknown_usage":
        prior["observations"][0]["costs"]["survey"]["usage_complete"] = False
    elif fault == "interrupted":
        prior["observations"][0]["status"] = "started_no_replay"
    elif fault == "order":
        prior["observations"].reverse()
    else:
        prior["budgets"]["survey"]["charged"] = 0
    parent.save(prior)
    executor = AsyncMock()
    with pytest.raises(ValueError, match="quality_resume"):
        resume(parent, executor)
    executor.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_completed_contract_does_not_open_new_formal_gate(tmp_path):
    parent = suite(tmp_path, AsyncMock(return_value=observation()))
    await parent.run(max_new_observations=2)
    executor = AsyncMock(return_value=observation(False, complete=False))
    runner = resume(parent, executor)
    for row in runner.parent["observations"]:
        row["verification_contract"] = "old-rules"
    result = await runner.run(max_new_observations=16)
    assert executor.await_count == 1
    assert result["stop_reasons"] == ["pilot_no_completed_judge_jev_contract"]
    assert len(result["observations"]) == 3
    assert result["counts"]["formal"]["run"] == 0


@pytest.mark.asyncio
async def test_resume_limit_counts_only_new_slots_and_persists(tmp_path):
    parent = suite(tmp_path, AsyncMock(return_value=observation(False, complete=False)))
    await parent.run(max_new_observations=2)
    executor = AsyncMock(return_value=observation())
    runner = resume(parent, executor)
    result = await runner.run(max_new_observations=1)
    assert executor.await_count == 1 and len(result["observations"]) == 3
    assert result["inherited_observations"] == 2
    with pytest.raises(ValueError, match="quality_observation_limit_conflict"):
        await runner.run(max_new_observations=16)
    executor.assert_awaited_once()
