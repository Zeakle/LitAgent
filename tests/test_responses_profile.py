"""Bind the optional provider mapping to frozen scope, parameters and exports."""

import csv
import json
from pathlib import Path

import pytest
from test_writer_thinking_profile import configured, manifest  # noqa: F401

from litagent.report_audit import atomic_json
from litagent.safety.budget import CostBudget
from litagent.writing_v2 import WriterError
from scripts import run_frozen_writing as campaign


@pytest.mark.parametrize("tamper", [None, "api", "effort", "mode", "extra"])
def test_fixed_responses_profile_preserves_config_and_rejects_parameter_drift(
    manifest, tamper
):
    """Rehashing a manifest cannot grant new provider parameters or a fallback."""
    root, config = manifest
    before = config.model_dump()
    target = root.with_name("responses-suite")
    campaign.prepare(target, config, writing_profile=campaign.RESPONSES_PROFILE)
    value = campaign.validate(target, config)
    assert config.model_dump() == before
    assert value["writing_parameters"] == {
        "native_reasoning_effort": "low",
        "native_provider_api": "responses",
    }
    assert value["limits"] == campaign.LIMITS
    assert not (target / "state.json").exists()
    if tamper is None:
        return
    if tamper == "api":
        value["writing_parameters"]["native_provider_api"] = "chat"
    if tamper == "effort":
        value["writing_parameters"]["native_reasoning_effort"] = "high"
    if tamper == "mode":
        value["writing_profile"] = campaign.LOW_THINKING_PROFILE
    if tamper == "extra":
        value["writing_parameters"]["fallback"] = True
    value["identity"] = campaign.digest(
        {k: v for k, v in value.items() if k != "identity"}
    )
    atomic_json(target / "manifest.json", value)
    with pytest.raises(ValueError, match="parameters"):
        campaign.validate(target, config)


@pytest.mark.asyncio
async def test_actual_campaign_consumes_api_and_exports_failure_without_resetting_fees(
    manifest, monkeypatch
):
    """Exercise driver, recorder and all export consumers; mock only the run boundary."""
    root, config = manifest
    config.observability.enabled = False
    target = root.with_name("responses-driver")
    campaign.prepare(target, config, writing_profile=campaign.RESPONSES_PROFILE)

    async def external_run(
        agent, bundle, run_id, *, native_reasoning_effort, native_provider_api
    ):
        assert native_reasoning_effort == "low" and native_provider_api == "responses"
        agent._cost_budget = CostBudget()
        agent._cost_budget.configure_strict(halt_on_unknown_usage=True)
        agent._cost_budget.begin_request()
        agent._cost_budget.record({"prompt_tokens": 20, "completion_tokens": 100})
        data = {
            "operation_id": "one",
            "model": "pro",
            "provider_api": "responses",
            "max_tokens": 20000,
            "output_limit_parameter": "max_output_tokens",
            "reasoning_effort": "low",
        }
        agent._trace_hook("llm.start", data)
        agent._trace_hook(
            "llm.complete",
            {
                **data,
                "usage_available": True,
                "prompt_tokens": 20,
                "completion_tokens": 100,
                "total_tokens": 120,
                "reasoning_tokens": 70,
                "reasoning_usage_available": True,
                "finish_reason": "stop",
                "content": "{",
            },
        )
        raise WriterError("writer_json_invalid")

    monkeypatch.setattr(campaign, "run_frozen", external_run)
    state = await campaign.execute(target, config)
    assert state["status"] == "stopped" and len(state["cases"]) == 1
    public_root = Path("benchmarks/survey/results") / target.name
    data = json.loads((public_root / "benchmark.json").read_text())
    assert data["passed"] == 0 and data["planned"] == 3 and data["not_run"] == 2
    assert data["new_usage"]["known_tokens"] == 120
    assert data["historical_usage"]["model_requests"] == 749
    assert data["historical_usage"]["model_unknown_conservative_tokens"] == 1030414
    assert data["writing_profile"]["provider_api"] == "responses"
    assert data["writing_profile"]["output_limit_parameter"] == "max_output_tokens"
    assert data["judge"] == data["jev"] == "disabled"
    page = (public_root / "benchmark.html").read_text()
    embedded = page.split('<script id="data" type="application/json">')[1].split(
        "</script>"
    )[0]
    assert json.loads(embedded) == data
    assert "p.provider_api||'chat'" in page
    rows = list(csv.DictReader((public_root / "benchmark.csv").open()))
    assert len(rows) == 3 and rows[0]["output_limit_parameter"] == "max_output_tokens"
    assert rows[0]["provider_api"] == "responses"
    assert "max_output_tokens" in (public_root / "benchmark.md").read_text()
    assert await campaign.execute(target, config) == state


def test_legacy_responses_profile_can_be_described_but_not_prepared(manifest):
    """Historical v1 remains readable without silently reinterpreting its wire format."""
    root, config = manifest
    from litagent.contracts import build_config_summary

    description = campaign.profile_description(
        campaign.LEGACY_RESPONSES_PROFILE,
        build_config_summary(config, "strict_auto_v2"),
    )
    assert description["provider_api"] == "responses"
    assert "mapping_contract" not in description
    target = root.with_name("forbidden-legacy")
    with pytest.raises(ValueError, match="profile_read_only"):
        campaign.prepare(
            target, config, writing_profile=campaign.LEGACY_RESPONSES_PROFILE
        )
    assert not target.exists()


def test_new_profile_identity_binds_provider_and_accounting_contract(manifest):
    """Omitting the conservative policy cannot turn a new manifest into an old one."""
    root, config = manifest
    target = root.with_name("new-contract")
    campaign.prepare(target, config, writing_profile=campaign.RESPONSES_PROFILE)
    value = campaign.validate(target, config)
    assert value["accounting_policy"] == campaign.CONSERVATIVE_ACCOUNTING
    description = campaign.profile_description(
        campaign.RESPONSES_PROFILE, value["config"]
    )
    assert description["mapping_contract"] == "ark-responses-mapping-v2"
    assert description["server_truncation_control"] == "unavailable"
    value.pop("accounting_policy")
    value["identity"] = campaign.digest(
        {k: v for k, v in value.items() if k != "identity"}
    )
    atomic_json(target / "manifest.json", value)
    with pytest.raises(ValueError, match="policy_conflict"):
        campaign.validate(target, config)


def test_untracked_vendor_fixture_is_in_the_frozen_inventory():
    """The independently checked provider directory must not disappear before commit."""
    assert (
        Path("tests/fixtures/ark-responses-fields-20261025.json")
        in campaign.source_paths()
    )
