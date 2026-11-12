"""Freeze separate writer/reviewer thinking without changing old profiles or caps."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_writer_thinking_profile import configured, manifest  # noqa: F401

from litagent.frozen_writing import run_frozen
from litagent.report_audit import atomic_json, digest
from scripts import run_frozen_writing as campaign

PROFILE = "writer_low_reviewer_disabled_responses_reemit_v2"
PARAMETERS = {
    "native_reasoning_effort": "low",
    "native_provider_api": "responses",
    "native_reemission_contract": "intact-unit-reemission-v2",
    "native_reviewer_thinking_mode": "disabled",
}


def test_fixed_role_profile_preserves_config_and_legacy_metadata(configured):
    """Only the new frozen parameter changes the Reviewer, never deployment defaults."""
    before = configured.model_dump()
    config = campaign.profile_config(configured, PROFILE)
    campaign.check_profile(config, PROFILE)
    assert configured.model_dump() == before
    assert config.adversarial.thinking_mode == "enabled"
    assert campaign.profile_parameters(PROFILE) == PARAMETERS
    old = campaign.profile_description(campaign.REEMISSION_PROFILE, {})
    new = campaign.profile_description(PROFILE, {})
    assert old["reviewer_thinking_mode"] == "enabled"
    assert "reviewer_reasoning_effort" not in old
    assert new["reviewer_thinking_mode"] == "disabled"
    assert new["reviewer_reasoning_effort"] is None
    assert new["generation_thinking_mode"] == "enabled"
    assert new["reasoning_effort"] == "low"
    assert new["reemission_contract"] == "intact-unit-reemission-v2"
    assert campaign.profile_parameters(campaign.REEMISSION_PROFILE) == {
        k: v for k, v in PARAMETERS.items() if k != "native_reviewer_thinking_mode"
    }


@pytest.mark.parametrize("tamper", [None, "enabled", "provider_default"])
def test_role_profile_identity_rejects_rehashed_parameter_drift(manifest, tamper):
    """A recomputed digest cannot silently inherit writer thinking for the Reviewer."""
    root, config = manifest
    target = root.with_name("role-identity")
    config = campaign.profile_config(config, PROFILE)
    campaign.prepare(target, config, writing_profile=PROFILE)
    value = campaign.validate(target, config)
    assert value["writing_parameters"] == PARAMETERS
    assert value["accounting_policy"] == campaign.CONSERVATIVE_ACCOUNTING
    if tamper is None:
        value["writing_parameters"].pop("native_reviewer_thinking_mode")
    else:
        value["writing_parameters"]["native_reviewer_thinking_mode"] = tamper
    value["identity"] = digest({k: v for k, v in value.items() if k != "identity"})
    atomic_json(target / "manifest.json", value)
    with pytest.raises(ValueError, match="parameters_conflict"):
        campaign.validate(target, config)


def test_role_profile_all_exports_use_actual_frozen_roles(manifest):
    """The same payload drives all formats without relabeling historical profiles."""
    root, config = manifest
    target = root.with_name("role-export")
    config = campaign.profile_config(config, PROFILE)
    campaign.prepare(target, config, writing_profile=PROFILE)
    value = campaign.validate(target, config)
    state = {"identity": value["identity"], "status": "offline_fixture", "cases": []}
    public = campaign.export(target, state, value)
    out = Path("benchmarks/survey/results") / target.name
    rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert all(r["reviewer_thinking_mode"] == "disabled" for r in rows)
    assert all(r["generation_thinking_mode"] == "enabled" for r in rows)
    html = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            html.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )
    assert public["writing_profile"]["reviewer_reasoning_effort"] is None
    markdown = (out / "benchmark.md").read_text()
    assert (
        json.loads(markdown.split("```json\n")[1].split("\n```")[0])
        == public["writing_profile"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["enabled", "provider_default", True, {}, 1])
async def test_invalid_role_override_rejected_before_runtime_mutation(mode):
    """Only the explicitly supported disabled override may enter the frozen runtime."""
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="reviewer_thinking_configuration_invalid"):
        await run_frozen(agent, {}, "offline", native_reviewer_thinking_mode=mode)
    assert vars(agent) == {"_wired": False}


@pytest.mark.asyncio
async def test_driver_consumes_frozen_roles_and_retains_protocol_stop(
    manifest, monkeypatch
):
    """The real campaign dispatches only its bound parameters and never replays a stop."""
    from litagent.safety.budget import CostBudget
    from litagent.writing_v2 import WriterError

    root, config = manifest
    target = root.with_name("role-driver")
    config = campaign.profile_config(config, PROFILE)
    config.observability.enabled = False
    campaign.prepare(target, config, writing_profile=PROFILE)
    calls = []

    async def external_run(agent, bundle, run_id, **parameters):
        calls.append(parameters)
        assert parameters == PARAMETERS
        budget = CostBudget(500000)
        budget.configure_strict(halt_on_unknown_usage=True)
        agent._cost_budget = budget
        budget.begin_request()
        usage = {"prompt_tokens": 20, "completion_tokens": 100, "usage_available": True}
        budget.record(usage)
        data = {
            "operation_id": "one",
            "model": "pro",
            "messages": [],
            "max_tokens": 20000,
        }
        agent._trace_hook("llm.start", data)
        agent._trace_hook(
            "llm.complete",
            {
                **data,
                **usage,
                "total_tokens": 120,
                "content": "",
                "finish_reason": "stop",
            },
        )
        raise WriterError(
            "writer_json_invalid", "protocol", [{"reason": "writer_repair_limit"}]
        )

    monkeypatch.setattr(campaign, "run_frozen", external_run)
    state = await campaign.execute(target, config)
    assert len(calls) == 1
    assert (
        state["status"] == "stopped" and state["stop_reason"] == "writer_json_invalid"
    )
    assert len(state["cases"]) == 1
    row = state["cases"][0]
    assert row["failure_family"] == "protocol" and row["known_tokens"] == 120
    assert row["requests"] == 1 and row["unknown_hold"] == 0
    assert row["publishable"] is False
    before = (target / "state.json").read_bytes()
    assert await campaign.execute(target, config) == state
    assert len(calls) == 1 and (target / "state.json").read_bytes() == before
