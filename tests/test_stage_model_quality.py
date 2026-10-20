"""Verify the mixed-model campaign through real identity, scheduling, and exports."""

import copy
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from litagent.benchmark.quality_suite import public_projection
from litagent.config import load_config
from scripts import run_stage_model_quality as stage


@pytest.fixture
def campaign(tmp_path):
    cfg = load_config()
    cfg.rag.raw_root = str(tmp_path / "raw")
    history = tmp_path / "history.json"
    history.write_text(
        json.dumps(
            {
                "cumulative": {
                    "initial_generation_rounds": 19,
                    "model_requests": 241,
                    "model_known_tokens": 1450497,
                    "model_unknown_conservative_tokens": 3467,
                    "jev_requests": 67,
                    "jev_conservative_input_tokens": 756769,
                    "jev_usage_complete": False,
                    "judge_requests": 0,
                }
            }
        )
    )
    root = tmp_path / "mixed-model-001"
    stage.prepare(root, cfg, history)
    return root, cfg, history


def observation(passed=True, technical=None, complete=True):
    return {
        "status": "delivered" if passed else "blocked",
        "acceptance": {"publishable": passed},
        "complete_verification": complete,
        "costs": {
            s: {
                "tokens": 100 if s == "survey" else 0,
                "requests": 1 if s == "survey" else 0,
                "usage_complete": True,
            }
            for s in ("survey", "judge", "jev")
        },
        "technical_failure": technical,
        "failure_codes": [] if passed else ["evaluation_not_passed"],
    }


def runner(campaign, tmp_path, executor):
    root, cfg, _ = campaign
    return stage.StageModelQualityRunner(
        root, cfg, executor=executor, output_root=tmp_path / "results"
    )


def test_frozen_identity_and_history_preserve_unknown_holds(campaign):
    root, cfg, history = campaign
    before = history.read_bytes()
    m = stage.validate_manifest(root, cfg)
    assert m["historical_accounting"]["model_unknown_conservative_tokens"] == 3467
    assert m["limits"]["rounds"] == 18
    assert len(m["formal_cases"]) == 15
    assert m["options"]["verification"] == {
        "version": "optional-verification-v1",
        "judge": False,
        "jev": False,
    }
    assert history.read_bytes() == before
    with pytest.raises(FileExistsError):
        stage.prepare(root, cfg, history)


def test_changed_input_or_role_rejected_before_transport(campaign):
    root, cfg, history = campaign
    changed = cfg.model_copy(deep=True)
    changed.llm.generation_model = changed.llm.extraction_model
    with pytest.raises(ValueError, match="model_configuration"):
        stage.validate_manifest(root, changed)
    history.write_text(history.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen_input_changed"):
        stage.validate_manifest(root, cfg)


@pytest.mark.asyncio
async def test_three_pilots_pass_then_all_fifteen_once(campaign, tmp_path):
    execute = AsyncMock(return_value=observation())
    r = runner(campaign, tmp_path, execute)
    state = await r.run()
    assert execute.await_count == 18
    assert state["counts"]["formal"]["automatic_passed"] == 15
    assert state["formal_eligible"] is True
    assert state["budgets"]["survey"]["charged"] == 1800
    assert (
        state["budgets"]["judge"]["charged"] == state["budgets"]["jev"]["charged"] == 0
    )
    assert await r.run() == state
    assert execute.await_count == 18
    public = public_projection(state)
    assert public["execution_context"]["models"] == stage.MODELS
    assert public["execution_context"]["historical_accounting"]["model_requests"] == 241
    assert str(campaign[2]) not in json.dumps(public)
    bundle = tmp_path / "results/presentation" / r.suite_id
    html = (bundle / "benchmark.html").read_text()
    assert "Judge disabled" in html and "1450497" in html
    assert "self_judged ·" not in html
    assert json.loads((bundle / "benchmark.json").read_text()) == public


@pytest.mark.asyncio
@pytest.mark.parametrize("index", [0, 1, 2])
async def test_scientific_failure_is_kept_and_formal_still_runs_all_cases(
    campaign, tmp_path, index
):
    items = [observation() for _ in range(18)]
    items[index] = observation(False)
    items[3] = observation(False)
    execute = AsyncMock(side_effect=items)
    state = await runner(campaign, tmp_path, execute).run()
    assert execute.await_count == 18
    assert state["counts"]["pilot"]["blocked"] == 1
    assert state["counts"]["formal"]["run"] == 15
    assert state["counts"]["formal"]["automatic_passed"] == 14
    assert state["counts"]["goal_met"] is False
    assert state["formal_eligible"] is True
    assert not state["stop_reasons"]


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_technical_and_unknown_usage_stop_without_replay(
    campaign, tmp_path, unknown
):
    item = observation(False, technical="quality_provider_error")
    if unknown:
        item["technical_failure"] = None
        item["costs"]["survey"].update(tokens=None, usage_complete=False)
    execute = AsyncMock(return_value=item)
    r = runner(campaign, tmp_path, execute)
    state = await r.run()
    assert state["status"] == "stopped" and execute.await_count == 1
    if unknown:
        assert state["budgets"]["survey"]["charged"] == 500000
    assert await r.run() == state
    assert execute.await_count == 1


def test_projection_drops_private_nested_execution_fields(campaign, tmp_path):
    r = runner(campaign, tmp_path, AsyncMock())
    state = r.initial()
    state["execution_context"]["private_path"] = "/private/report"
    state["execution_context"]["models"]["api_key"] = "secret-key"
    state["execution_context"]["historical_accounting"][
        "private_path"
    ] = "/private/usage"
    encoded = json.dumps(public_projection(state))
    assert "secret-key" not in encoded and "/private/" not in encoded


@pytest.mark.asyncio
async def test_incomplete_pilot_cannot_open_formal_even_with_positive_flag(
    campaign, tmp_path
):
    execute = AsyncMock(
        side_effect=[observation(), observation(complete=False), observation()]
    )
    state = await runner(campaign, tmp_path, execute).run()
    assert execute.await_count == 3
    assert state["counts"]["formal"]["run"] == 0
    assert state["stop_reasons"] == ["stage_quality_pilot_contract_inconsistent"]


@pytest.mark.asyncio
async def test_all_scientific_failures_are_visible_in_the_full_fixed_matrix(
    campaign, tmp_path
):
    execute = AsyncMock(return_value=observation(False))
    r = runner(campaign, tmp_path, execute)
    state = await r.run()
    assert execute.await_count == 18
    assert state["counts"]["pilot"]["blocked"] == 3
    assert state["counts"]["formal"]["run"] == 15
    assert state["counts"]["formal"]["blocked"] == 15
    assert state["counts"]["formal"]["automatic_passed"] == 0
    assert not state["counts"]["goal_met"]
    assert state["budgets"]["survey"]["requests"] == 18
    assert (await r.run()) == state and execute.await_count == 18


@pytest.mark.asyncio
async def test_no_complete_evaluation_keeps_all_formal_cases_unrun(campaign, tmp_path):
    execute = AsyncMock(return_value=observation(False, complete=False))
    state = await runner(campaign, tmp_path, execute).run()
    assert execute.await_count == 3 and state["counts"]["formal"]["run"] == 0
    assert state["stop_reasons"] == ["stage_quality_no_complete_evaluation"]


@pytest.mark.asyncio
async def test_one_complete_evaluation_allows_other_explained_quality_failures(
    campaign, tmp_path
):
    execute = AsyncMock(
        side_effect=[
            observation(False, complete=False),
            observation(False),
            observation(False, complete=False),
        ]
        + [observation() for _ in range(15)]
    )
    state = await runner(campaign, tmp_path, execute).run()
    assert execute.await_count == 18
    assert state["counts"]["pilot"]["blocked"] == 3
    assert state["counts"]["pilot"]["complete_verification"] == 1
    assert state["counts"]["formal"]["automatic_passed"] == 15


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing_reason", "missing_complete", "inconsistent_status"]
)
async def test_unexplained_or_inconsistent_pilot_contract_stops_dispatch(
    campaign, tmp_path, fault
):
    invalid = observation(False)
    if fault == "missing_reason":
        invalid.pop("failure_codes")
    elif fault == "missing_complete":
        invalid.pop("complete_verification")
    else:
        invalid["status"] = "delivered"
    execute = AsyncMock(side_effect=[observation(), invalid, observation()])
    state = await runner(campaign, tmp_path, execute).run()
    assert execute.await_count == 3 and state["counts"]["formal"]["run"] == 0
    assert state["stop_reasons"] == [
        (
            "stage_quality_pilot_contract_inconsistent"
            if fault == "inconsistent_status"
            else "stage_quality_pilot_contract_incomplete"
        )
    ]


@pytest.mark.asyncio
async def test_formal_technical_failure_still_stops_after_the_first_affected_case(
    campaign, tmp_path
):
    execute = AsyncMock(
        side_effect=[
            observation(False),
            observation(),
            observation(False),
            observation(False, technical="quality_provider_error"),
        ]
    )
    state = await runner(campaign, tmp_path, execute).run()
    assert execute.await_count == 4
    assert (
        state["counts"]["formal"]["run"] == 1
        and state["counts"]["formal"]["not_run"] == 14
    )
    assert state["stop_reasons"] == ["quality_provider_error"]


def test_public_gate_is_fixed_protocol_metadata_without_private_nested_values(
    campaign, tmp_path
):
    r = runner(campaign, tmp_path, AsyncMock())
    state = r.initial()
    assert (
        public_projection(state)["execution_context"]["formal_gate"]
        == stage.PILOT_EVALUATION_GATE_CONTRACT
    )
    state["execution_context"]["formal_gate"] = {
        "api_key": "do-not-export",
        "private_path": "/private/file",
    }
    encoded = json.dumps(public_projection(state))
    assert "do-not-export" not in encoded and "/private/file" not in encoded
    assert "formal_gate" not in public_projection(state)["execution_context"]
    state["execution_context"].pop("formal_gate")
    assert "formal_gate" not in public_projection(state)["execution_context"]
