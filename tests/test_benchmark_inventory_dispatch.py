"""Consume real SDK-produced evaluations through the frozen benchmark dispatcher."""

import copy
import csv
import hashlib
import json

import pytest
import test_located_evaluation as located
from model_config_helpers import load_previous_default_config as load_config
from test_writer_v2 import case  # noqa: F401

from litagent.contracts import build_config_summary
from litagent.eval.batched import ANCHORED_FINDINGS_CONTRACT, GLOBAL_FINDINGS_CONTRACT
from litagent.safety.planned_budget import PlannedBudget
from litagent.writing_v2 import RANGE_WRITING_PLAN, writer_budget_plan
from scripts import run_frozen_writing as campaign


class HostFixtureAgent:
    """Isolate CLI host bookkeeping; model calls already ran through the real SDK."""

    def __init__(self, *_args, **_kwargs):
        """Skip model ownership in this host-only consumer fixture."""
        self._cost_budget = None

    async def cleanup(self):
        """Own no external resources in the isolated dispatcher fixture."""


async def dispatch_report(
    report, case, tmp_path, monkeypatch, *, inventory_recovery=False, expected_passed=0
):
    """Run real scheduling, state persistence and exports using a validated fixture."""
    config = load_config()
    config.observability.enabled = False
    config.mcp_servers = {}
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=inventory_recovery,
    )
    original = copy.deepcopy(report)
    calls = []
    manifest = {
        "identity": "offline-dispatch-fixture",
        "source_commit": "offline-fixture",
        "source_hashes": {},
        "config": build_config_summary(config, "strict_auto_v2"),
        "writing_profile": campaign.ROUND_PROFILE,
        "writing_plan": RANGE_WRITING_PLAN,
        "limits": {**campaign.LIMITS, "per_case_tokens": 4500000},
        "cases": [
            {"case_id": cid, "bundle": {"query": case.packet["query"]}}
            for cid, _ in campaign.CASES
        ],
        "budget_plans": {cid: copy.deepcopy(plan) for cid, _ in campaign.CASES},
        "history": {
            "model_known_tokens": 0,
            "model_unknown_conservative_tokens": 0,
            "model_requests": 0,
            "jev_requests": 0,
            "jev_conservative_input_tokens": 0,
            "judge_requests": 0,
            "initial_generation_rounds": 0,
        },
    }
    root = tmp_path / "suite"
    if inventory_recovery:
        manifest["inventory_recovery"] = True
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps(manifest))

    async def cached_producer(_agent, _bundle, run_id, **_kwargs):
        # The producer was exercised before entering this isolated host fixture.
        # Only this cached host dispatch has zero calls. Its real empty plan
        # snapshot must not pretend to account for the upstream SDK fixture.
        _agent._cost_budget = PlannedBudget(_kwargs["expected_budget_plan"])
        assert _agent._cost_budget.snapshot()["requests_sent"] == 0
        calls.append(run_id)
        assert _kwargs.get("inventory_recovery", False) is inventory_recovery
        return copy.deepcopy(report)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(campaign, "validate", lambda *_: manifest)
    monkeypatch.setattr(campaign, "offline_gates_verified", lambda *_: True)
    monkeypatch.setattr(campaign, "LitAgent", HostFixtureAgent)
    monkeypatch.setattr(campaign, "run_frozen", cached_producer)
    state = await campaign.execute(root, config)
    assert report == original
    assert all(
        row["report_sha256"] == hashlib.sha256(report["survey"].encode()).hexdigest()
        for row in state["cases"]
    )
    public_root = tmp_path / "benchmarks/survey/results/suite"
    public = json.loads((public_root / "benchmark.json").read_text())
    assert public["planned"] == 3 and public["passed"] == expected_passed
    assert sum(row["publishable"] for row in state["cases"]) == expected_passed
    return state, calls, public_root


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "contract", [GLOBAL_FINDINGS_CONTRACT, ANCHORED_FINDINGS_CONTRACT]
)
async def test_valid_inventory_quality_failure_continues_and_exports_both_contracts(
    case, tmp_path, monkeypatch, contract
):
    """Keep failed science blocked while allowing the remaining planned cases."""
    monkeypatch.setattr("litagent.eval.batched.CURRENT_FINDINGS_CONTRACT", contract)
    report, gateway = await located.run_case(
        case, tmp_path / "producer", monkeypatch, "inventory"
    )
    assert gateway.sent
    assert (
        report["evaluation"]["faithfulness"]["details"]["inventory_check"]["status"]
        == "failed"
    )
    state, calls, public_root = await dispatch_report(
        report, case, tmp_path, monkeypatch
    )
    assert state["status"] == "completed" and len(calls) == 3
    assert all(row["failure_family"] == "quality" for row in state["cases"])
    assert all(row["inventory_status"] == "failed" for row in state["cases"])
    assert all(row["global_findings_contract"] == contract for row in state["cases"])
    rows = list(csv.DictReader((public_root / "benchmark.csv").open()))
    assert len(rows) == 3 and all(row["inventory_status"] == "failed" for row in rows)
    html = (public_root / "benchmark.html").read_text()
    assert "Partial diagnostics are not final scores" in html


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["global_json", "forged", "unknown_marker"])
async def test_invalid_json_or_identity_still_stops_later_cases(
    case, tmp_path, monkeypatch, fault
):
    """Keep protocol and corrupt host metadata out of the quality continuation path."""
    report, _ = await located.run_case(
        case,
        tmp_path / "producer",
        monkeypatch,
        "inventory" if fault == "unknown_marker" else fault,
    )
    if fault == "unknown_marker":
        report["evaluation"]["faithfulness"]["details"][
            "global_findings_contract"
        ] = "unknown-contract"
    state, calls, public_root = await dispatch_report(
        report, case, tmp_path, monkeypatch
    )
    assert state["status"] == "stopped" and len(calls) == 1
    assert state["cases"][0]["failure_family"] == "protocol"
    public = json.loads((public_root / "benchmark.json").read_text())
    assert len(public["cases"]) == 3
    assert sum(row["status"] == "not_run" for row in public["cases"]) == 2


@pytest.mark.asyncio
async def test_cli_dispatches_explicit_recovery_and_exports_same_sha_history(
    case, tmp_path, monkeypatch
):
    """Check the CLI consumer using a real SDK result, not synthetic pass flags."""
    from test_inventory_recovery import execute

    report, _ = await execute(case, tmp_path / "producer", monkeypatch)
    assert report["delivery"]["publishable"]
    state, calls, public_root = await dispatch_report(
        report, case, tmp_path, monkeypatch, inventory_recovery=True, expected_passed=3
    )
    assert state["status"] == "completed" and len(calls) == 3
    assert all(row["repairs"] == 1 for row in state["cases"])
    assert all(
        row["evaluation_history"][1]["stage"] == "inventory_recovery_1"
        for row in state["cases"]
    )
    csv_text = (public_root / "benchmark.csv").read_text()
    assert "inventory_recovery_1" in csv_text
    assert "inventory_recovery_1" in (public_root / "benchmark.md").read_text()
    assert "inventory_recovery_1" in (public_root / "benchmark.html").read_text()


@pytest.mark.asyncio
async def test_terminal_mapping_failure_and_upstream_capacity_reason_survive_all_exports(
    case, tmp_path, monkeypatch
):
    """Use the SDK producer and actual dispatcher rather than handcrafted scores."""
    from test_capacity_map_feedback import MapFeedbackGateway
    from test_capacity_repair import run_capacity_case

    report, _ = await run_capacity_case(
        case,
        tmp_path / "producer",
        monkeypatch,
        gateway=MapFeedbackGateway(case, fault="repeat"),
    )
    failure = report["metadata"]["writer_failure"]["code"]
    assert failure == "writer_capacity_new_unit_unaccounted"
    original = copy.deepcopy(report)
    state, calls, root = await dispatch_report(report, case, tmp_path, monkeypatch)
    assert state["status"] == "completed" and len(calls) == 3
    public = json.loads((root / "benchmark.json").read_text())
    rows = list(csv.DictReader((root / "benchmark.csv").open()))
    for row, exported in zip(public["cases"], rows, strict=True):
        assert row["terminal_reason"] == exported["terminal_reason"] == failure
        assert (
            row["evaluation_reason"]
            == exported["evaluation_reason"]
            == "evaluation_capacity_refinement_required"
        )
        assert not row["publishable"] and row["evaluation"] == "not_complete"
    for name in ("benchmark.md", "benchmark.html"):
        text = (root / name).read_text()
        assert failure in text and "evaluation_capacity_refinement_required" in text
    assert report == original
