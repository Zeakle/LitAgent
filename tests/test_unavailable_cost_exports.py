"""Keep unavailable presentation costs distinct from authenticated zero usage."""

import copy
import csv
import json
import re
from pathlib import Path

import pytest
from test_planned_mixed_receipts import frozen_bundle
from test_writer_v2 import case  # noqa: F401

from litagent import frozen_writing as frozen
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    envelope_writing_spec,
    writer_budget_plan,
)
from scripts import run_frozen_writing as campaign


@pytest.fixture
def display_case(case, tmp_path, monkeypatch):
    """Use an actual plan and empty producer ledger, without executing a model."""
    monkeypatch.chdir(tmp_path)
    config = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    config.observability.enabled = False
    config.mcp_servers = {}
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )
    manifest = {
        "writing_profile": campaign.ROUND_PROFILE,
        "writing_plan": RANGE_WRITING_PLAN,
        "writer_wire_contract": "flat-writer-v1",
        "config": build_config_summary(config, "strict_auto_v2"),
        "source_commit": "offline-display-fixture",
        "source_hashes": {},
        "history": {},
        "limits": campaign.LIMITS,
        "budget_plans": {cid: copy.deepcopy(plan) for cid, _ in campaign.CASES},
    }
    manifest["identity"] = campaign.digest(manifest)
    state = {
        "identity": manifest["identity"],
        "status": "stopped",
        "stop_reason": "original_scientific_failure",
        "cases": [
            {
                "case_id": campaign.CASES[0][0],
                "run_id": "offline-display-costs",
                "status": "passed",
                "generated": True,
                "reviewer_completed": True,
                "evaluation": "not_complete",
                "publishable": True,
                "supported": 14,
                "unsupported": 1,
                "unverified": 2,
                "requests": 0,
                "known_tokens": 0,
                "unknown_hold": 0,
                "usage_complete": True,
                "failure_code": "original_scientific_failure",
                "failure_family": "quality",
                "budget_snapshot": PlannedBudget(plan).snapshot(),
            }
        ],
    }
    return tmp_path / "display-suite", manifest, state, plan


def exported(root, state, manifest):
    """Read all four real exports and prove their input state remains immutable."""
    before_state, before_manifest = copy.deepcopy(state), copy.deepcopy(manifest)
    root.mkdir(exist_ok=True)
    campaign.export(root, state, manifest)
    assert state == before_state and manifest == before_manifest
    out = Path("benchmarks/survey/results", root.name)
    public = json.loads((out / "benchmark.json").read_text())
    html = (out / "benchmark.html").read_text()
    embedded = re.search(
        r'<script id="data" type="application/json">(.*?)</script>', html, re.S
    )
    assert embedded is not None
    assert json.loads(embedded.group(1)) == public
    with (out / "benchmark.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == public["planned"] == 3
    assert public["run"] == 1 and public["not_run"] == 2
    assert all(row["requests"] == 0 for row in public["cases"][1:])
    return public, rows, (out / "benchmark.md").read_text(), html


def authenticate_rejected(root, state, manifest):
    """The strict historical consumer must still reject unavailable evidence."""
    pinned = copy.deepcopy(manifest)
    pinned["identity"] = campaign.digest(
        {k: v for k, v in pinned.items() if k != "identity"}
    )
    stored = copy.deepcopy(state)
    stored["identity"] = pinned["identity"]
    root.mkdir(exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(pinned))
    (root / "state.json").write_text(json.dumps(stored))
    with pytest.raises(ValueError, match="frozen_accounting|planned_accounting"):
        campaign.accounting_history(
            pinned["history"],
            [root],
            unknown_policy=campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING,
        )


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("missing", "budget_snapshot_missing"),
        ("null", "budget_snapshot_missing"),
        ("empty", "budget_snapshot_missing"),
        ("string", "budget_snapshot_invalid"),
        ("schema1", "budget_snapshot_invalid"),
        ("stripped", "budget_snapshot_invalid"),
        ("wrong_plan", "budget_plan_conflict"),
        ("missing_plan", "budget_plan_missing"),
    ],
)
def test_unavailable_costs_do_not_crash_exports_or_gain_accounting_authority(
    display_case, fault, reason
):
    root, manifest, state, _ = display_case
    row = state["cases"][0]
    if fault == "missing":
        row.pop("budget_snapshot")
    elif fault == "null":
        row["budget_snapshot"] = None
    elif fault == "empty":
        row["budget_snapshot"] = {}
    elif fault == "string":
        row["budget_snapshot"] = "not-a-ledger"
    elif fault == "schema1":
        row["budget_snapshot"]["schema_version"] = 1
    elif fault == "stripped":
        row["budget_snapshot"].pop("entries")
    elif fault == "wrong_plan":
        row["budget_snapshot"]["plan"]["input_identity"] = "0" * 64
    else:
        manifest["budget_plans"].pop(row["case_id"])
    public, csv_rows, markdown, html = exported(root, state, manifest)
    actual = public["cases"][0]
    diagnostic = actual["cost_diagnostic"]
    assert diagnostic == {
        "status": "unavailable",
        "reason_code": reason,
        "reported_requests": 0,
        "reported_known_tokens": 0,
        "reported_unknown_hold": None,
    }
    assert all(
        actual[key] is None
        for key in ("requests", "known_tokens", "unknown_hold", "request_hold")
    )
    assert all(value is None for value in public["new_usage"].values())
    assert actual["publishable"] is False and actual["usage_complete"] is False
    assert actual["status"] == "blocked" and public["passed"] == 0
    assert actual["failure_code"] == "original_scientific_failure"
    assert actual["failure_family"] == "quality"
    assert (actual["supported"], actual["unsupported"], actual["unverified"]) == (
        14,
        1,
        2,
    )
    assert actual["evaluation"] == "not_complete"
    assert actual["evaluation_reason"] == "original_scientific_failure"
    assert "cost unavailable: " + reason in actual["terminal_reason"]
    assert json.loads(csv_rows[0]["cost_diagnostic"]) == diagnostic
    assert csv_rows[0]["requests"] == csv_rows[0]["known_tokens"] == ""
    assert "—" in markdown and "Cost evidence unavailable" in html
    authenticate_rejected(root / "parent", state, manifest)


@pytest.mark.parametrize("reported", [True, -1, "17", None])
def test_unavailable_reported_values_never_treat_bool_or_negative_as_usage(
    display_case, reported
):
    root, manifest, state, _ = display_case
    row = state["cases"][0]
    row.pop("budget_snapshot")
    row.update(requests=reported, known_tokens=reported, unknown_hold=reported)
    public, *_ = exported(root, state, manifest)
    diagnostic = public["cases"][0]["cost_diagnostic"]
    assert diagnostic["reported_requests"] is None
    assert diagnostic["reported_known_tokens"] is None
    assert diagnostic["reported_unknown_hold"] is None


def test_real_unknown_occupancy_is_retained_as_reported_when_snapshot_missing(
    display_case,
):
    root, manifest, state, plan = display_case
    owner = PlannedBudget(plan)
    with request_scope("body_initial", "display-unknown-request"):
        owner.reserve(25000)
        owner.begin_request()
        owner.charge_unknown_usage(25000)
    snapshot = owner.snapshot()
    assert (
        campaign.historical_unknown_token_hold(snapshot, 0, expected_plan=plan) == 25000
    )
    row = state["cases"][0]
    row.update(
        requests=snapshot["requests_sent"],
        known_tokens=snapshot["used_tokens"],
        unknown_hold=25000,
    )
    row.pop("budget_snapshot")
    public, *_ = exported(root, state, manifest)
    actual = public["cases"][0]
    assert actual["cost_diagnostic"]["reported_unknown_hold"] == 25000
    assert actual["cost_diagnostic"]["reported_requests"] == 1
    assert actual["unknown_hold"] is None and actual["known_tokens"] is None
    assert public["new_usage"]["unknown_hold"] is None


def test_valid_empty_ledger_and_current_metadata_keep_original_projection(display_case):
    root, manifest, state, plan = display_case
    public, *_ = exported(root, state, manifest)
    assert "cost_diagnostic" not in public["cases"][0]
    assert public["new_usage"]["requests"] == public["new_usage"]["known_tokens"] == 0
    assert public["new_usage"]["unknown_hold"] == 0
    assert public["writing_profile"]["writing_plan"] == envelope_writing_spec(
        RANGE_WRITING_PLAN, plan["envelope"]
    )


def test_valid_unknown_ledger_keeps_original_positive_hold_and_usage_status(
    display_case,
):
    root, manifest, state, plan = display_case
    owner = PlannedBudget(plan)
    with request_scope("body_initial", "valid-display-unknown-request"):
        owner.reserve(25000)
        owner.begin_request()
        owner.charge_unknown_usage(25000)
    snapshot = owner.snapshot()
    state["cases"][0].update(
        budget_snapshot=snapshot,
        requests=snapshot["requests_sent"],
        known_tokens=snapshot["used_tokens"],
        unknown_hold=25000,
        usage_complete=False,
        publishable=False,
        status="blocked",
    )
    public, csv_rows, *_ = exported(root, state, manifest)
    actual = public["cases"][0]
    assert "cost_diagnostic" not in actual
    assert (actual["requests"], actual["known_tokens"], actual["unknown_hold"]) == (
        1,
        0,
        25000,
    )
    assert actual["usage_complete"] is False and actual["publishable"] is False
    assert actual["failure_code"] == "usage_unknown"
    assert actual["failure_family"] == "interrupted"
    assert (actual["supported"], actual["unsupported"], actual["unverified"]) == (
        14,
        1,
        2,
    )
    assert public["new_usage"]["unknown_hold"] == 25000
    assert csv_rows[0]["unknown_hold"] == "25000"


def test_original040_metadata_remains_readable_without_restoring_old_runtime(
    display_case,
):
    root, manifest, state, _ = display_case
    fixture = Path(__file__).parent / "fixtures/run040-cost-snapshot.json"
    snapshot = json.loads(fixture.read_text())["snapshot"]
    plan = snapshot["plan"]
    row = state["cases"][0]
    manifest["budget_plans"] = {cid: copy.deepcopy(plan) for cid, _ in campaign.CASES}
    row.update(
        budget_snapshot=snapshot,
        requests=snapshot["requests_sent"],
        known_tokens=snapshot["used_tokens"],
        unknown_hold=0,
        publishable=False,
        status="blocked",
    )
    with pytest.raises(ValueError):
        envelope_writing_spec(RANGE_WRITING_PLAN, plan["envelope"])
    public, *_ = exported(root, state, manifest)
    actual = public["cases"][0]
    assert "cost_diagnostic" not in actual
    assert (actual["requests"], actual["known_tokens"], actual["unknown_hold"]) == (
        22,
        344347,
        0,
    )
    spec = public["writing_profile"]["writing_plan"]
    assert spec["body_max_units"] == plan["envelope"]["body_max_units"]
    assert spec["atomic_claim_capacity"] == plan["envelope"]["model_assertions"]
    assert actual["publishable"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["preowner", "sdk_factory"])
async def test_real_frozen_early_failures_export_unavailable_not_measured_zero(
    case, tmp_path, monkeypatch, phase
):
    """Execute the real local run entry and replace only initial SDK construction."""
    monkeypatch.chdir(tmp_path)
    config = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    config.observability.enabled = False
    config.mcp_servers = {}
    events, factory_calls = [], []
    agent = LitAgent(
        config,
        run_policy=RunPolicy.isolated_benchmark(),
        trace_hook=lambda event, data: events.append((event, data)),
    )
    bundle = frozen_bundle(case)
    plan = frozen.plan_frozen_budget(
        bundle, budget_policy="planned-budget-v7", capacity=config.capacity
    )

    def failed_factory(*args, **kwargs):
        factory_calls.append((args, kwargs))
        raise RuntimeError("offline_initial_sdk_failure")

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", failed_factory)
    if phase == "preowner":
        bundle["fact_cards_sha256"] = "0" * 64
    expected_error = (
        "frozen_writing_fact_card_identity_conflict"
        if phase == "preowner"
        else "offline_initial_sdk_failure"
    )
    try:
        with pytest.raises((RuntimeError, ValueError), match=expected_error):
            await frozen.run_frozen(
                agent,
                bundle,
                "offline-early-" + phase,
                native_provider_api="responses",
                native_reviewer_thinking_mode="disabled",
                native_syntax_policy=EOF_CONTAINER_POLICY,
                native_writing_plan=RANGE_WRITING_PLAN,
                budget_policy="planned-budget-v7",
                expected_budget_plan=plan,
            )
        snapshot = (
            agent._cost_budget.snapshot() if agent._cost_budget is not None else None
        )
        assert len(factory_calls) == (0 if phase == "preowner" else 1)
        assert not any(event == "llm.start" for event, _ in events)
        if phase == "preowner":
            assert snapshot is None
        else:
            assert snapshot["schema_version"] == 1
            assert snapshot["requests_sent"] == 0
    finally:
        await agent.cleanup()
    manifest = {
        "writing_profile": campaign.ROUND_PROFILE,
        "writing_plan": RANGE_WRITING_PLAN,
        "writer_wire_contract": "flat-writer-v1",
        "config": build_config_summary(config, "strict_auto_v2"),
        "source_commit": "real-local-entry-fixture",
        "source_hashes": {},
        "history": {},
        "limits": campaign.LIMITS,
        "budget_plans": {cid: copy.deepcopy(plan) for cid, _ in campaign.CASES},
    }
    manifest["identity"] = campaign.digest(manifest)
    state = {
        "identity": manifest["identity"],
        "status": "stopped",
        "cases": [
            {
                "case_id": campaign.CASES[0][0],
                "run_id": "offline-early-" + phase,
                "status": "failed",
                "failure_code": expected_error,
                "failure_family": "technical",
                "generated": False,
                "reviewer_completed": False,
                "evaluation": "not_run",
                "publishable": False,
                "budget_snapshot": snapshot,
                "requests": None if snapshot is None else snapshot["requests_sent"],
                "known_tokens": None if snapshot is None else snapshot["used_tokens"],
                "unknown_hold": None,
                "usage_complete": False,
            }
        ],
    }
    root = tmp_path / ("early-" + phase)
    public, *_ = exported(root, state, manifest)
    row = public["cases"][0]
    assert (
        row["failure_code"] == expected_error and row["failure_family"] == "technical"
    )
    assert row["generated"] is False and row["evaluation"] == "not_run"
    assert row["requests"] is row["known_tokens"] is row["unknown_hold"] is None
    assert row["usage_complete"] is row["publishable"] is False
    assert row["cost_diagnostic"]["reason_code"] == (
        "budget_snapshot_missing" if phase == "preowner" else "budget_snapshot_invalid"
    )
    authenticate_rejected(root / "parent", state, manifest)
