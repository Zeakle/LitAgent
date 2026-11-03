"""Keep planned unknown charges visible without using legacy rejection proofs."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_native_responses import envelope, owner_for
from test_planned_budget import plan
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.llm.routing import ModelBoundClient
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.writing_v2 import WriterSession, schema_for
from scripts import run_frozen_writing as campaign


async def planned_observation(case, tmp_path, mode):
    """Mock only HTTP while retaining real SDK accounting and durable journals."""
    sent = []

    async def handler(request):
        sent.append(request)
        if len(sent) == 1 or mode == "known":
            return httpx.Response(200, json=envelope(json.dumps(case.body)))
        if mode == "cancelled":
            raise asyncio.CancelledError()
        if mode == "rejected":
            return httpx.Response(429, json={"error": {"code": "RateLimit"}})
        response = envelope(json.dumps(case.body))
        response.pop("usage")
        return httpx.Response(200, json=response)

    owner, budget, events = await owner_for(handler)
    ledger_path = tmp_path / "ledger.json"
    budget.activate_plan(plan(), path=ledger_path)
    session = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "run.json",
        "planned-parent",
        case.packet,
        case.aliases,
        thinking_mode="disabled",
        provider_api="responses",
    )
    try:
        await session.call(
            "body:introduction",
            "Use complete sources.",
            case.packet,
            schema_for("body", case.aliases),
        )
        if mode != "known":
            error = asyncio.CancelledError if mode == "cancelled" else ValueError
            with pytest.raises(error):
                await session.call(
                    "body:taxonomy",
                    "Use complete sources.",
                    case.packet,
                    schema_for("body", case.aliases),
                )
        snapshot = budget.snapshot()
        known = sum(
            data["total_tokens"]
            for event, data in events
            if event == "llm.complete" and data.get("usage_available") is True
        )
        return SimpleNamespace(
            snapshot=snapshot,
            known=known,
            sent=sent,
            ledger_path=ledger_path,
            journal_path=session.journal_path,
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["known", "cancelled", "rejected", "missing_usage"])
async def test_real_planned_usage_keeps_unknown_escrows_separate(case, tmp_path, mode):
    observation = await planned_observation(case, tmp_path, mode)
    snapshot = observation.snapshot
    original = copy.deepcopy(snapshot)
    persisted = observation.ledger_path.read_bytes()
    hold = campaign.unknown_token_hold(snapshot, observation.known)
    assert snapshot["used_tokens"] == observation.known == 120
    assert snapshot == original and observation.ledger_path.read_bytes() == persisted
    if mode == "known":
        assert hold == 0 and snapshot["usage_uncertain"] is False
        assert len(observation.sent) == 1
        return
    entry = next(e for e in snapshot["entries"].values() if e["status"] != "settled")
    assert entry["sent"] is True and hold == entry["reserved_tokens"] > 20000
    assert len(observation.sent) == 2 and snapshot["usage_uncertain"] is True
    restored = PlannedBudget(plan(), path=observation.ledger_path)
    assert campaign.unknown_token_hold(restored.snapshot(), observation.known) == hold
    with request_scope("body", "new-attempt"):
        with pytest.raises(ValueError, match="unknown"):
            restored.reserve(1)
    assert len(observation.sent) == 2


@pytest.mark.asyncio
async def test_planned_export_reads_escrows_without_mutating_terminal_history(
    case, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    observation = await planned_observation(case, tmp_path, "cancelled")
    root = tmp_path / "planned-parent"
    root.mkdir()
    row = {
        "case_id": campaign.CASES[0][0],
        "run_id": root.name,
        "status": "blocked",
        "publishable": False,
        "requests": 2,
        "known_tokens": observation.known,
        "unknown_hold": 0,
        "usage_complete": False,
        "budget_snapshot": observation.snapshot,
    }
    state = {"status": "stopped", "cases": [row]}
    summary = build_config_summary(load_config(), "strict_auto_v2")
    manifest = {
        "identity": "offline-terminal",
        "source_commit": "offline",
        "source_hashes": {},
        "config": summary,
        "history": {},
        "limits": campaign.LIMITS,
    }
    original = copy.deepcopy(state)
    campaign.export(root, state, manifest)
    unavailable = json.loads(
        Path("benchmarks/survey/results/planned-parent/benchmark.json").read_text()
    )
    assert unavailable["new_usage"]["known_tokens"] is None
    assert unavailable["new_usage"]["unknown_hold"] is None
    assert unavailable["cases"][0]["cost_diagnostic"]["reason_code"] == (
        "budget_plan_missing"
    )
    assert unavailable["cases"][0]["usage_complete"] is False
    assert state == original
    manifest["budget_plans"] = {
        case_id: observation.snapshot["plan"] for case_id, _ in campaign.CASES
    }
    campaign.export(root, state, manifest)
    public = json.loads(
        Path("benchmarks/survey/results/planned-parent/benchmark.json").read_text()
    )
    hold = campaign.unknown_token_hold(observation.snapshot, observation.known)
    assert public["new_usage"]["unknown_hold"] == hold > 0
    assert public["new_usage"]["known_tokens"] == observation.known
    assert public["cases"][0]["usage_complete"] is False
    assert public["cases"][0]["publishable"] is False
    assert state == original


@pytest.mark.asyncio
async def test_unknown_planned_parent_explicitly_refuses_legacy_proof(
    case, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    observation = await planned_observation(case, tmp_path, "rejected")
    root = tmp_path / "planned-parent"
    root.mkdir()
    history = {"model_requests": 10, "model_known_tokens": 500, "preserved": "yes"}
    manifest = {"history": history}
    manifest["identity"] = digest(manifest)
    # Even a stale zero hold or incorrect caller flag cannot waive uncertainty.
    row = {
        "budget_snapshot": observation.snapshot,
        "known_tokens": observation.known,
        "unknown_hold": 0,
        "usage_complete": True,
    }
    state = {"identity": manifest["identity"], "status": "stopped", "cases": [row]}
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "state.json", state)
    before = {p: p.read_bytes() for p in (root / "manifest.json", root / "state.json")}
    monkeypatch.setattr(
        campaign,
        "conservative_rejection_receipt",
        lambda *args: pytest.fail("planned escrow cannot use a legacy proof"),
    )
    with pytest.raises(ValueError, match="frozen_accounting_usage_unknown"):
        campaign.accounting_history(history, [root])
    with pytest.raises(
        ValueError, match="frozen_accounting_planned_unknown_receipt_unsupported"
    ):
        campaign.accounting_history(
            history, [root], unknown_policy=campaign.CONSERVATIVE_ACCOUNTING
        )
    assert all(path.read_bytes() == value for path, value in before.items())
    assert len(observation.sent) == 2


@pytest.mark.parametrize("mutation", ["known", "reserved", "sent", "plan"])
@pytest.mark.asyncio
async def test_forged_planned_snapshot_cannot_hide_unknown_charge(
    case, tmp_path, mutation
):
    observation = await planned_observation(case, tmp_path, "cancelled")
    snapshot = copy.deepcopy(observation.snapshot)
    entry = next(e for e in snapshot["entries"].values() if e["status"] != "settled")
    if mutation == "known":
        snapshot["used_tokens"] -= 1
    elif mutation == "reserved":
        entry["reserved_tokens"] = 0
    elif mutation == "sent":
        entry["sent"] = False
    else:
        snapshot["plan"]["planned_tokens"] += 10000
    with pytest.raises(ValueError):
        campaign.unknown_token_hold(snapshot, observation.known)
    assert len(observation.sent) == 2


def test_untransmitted_planned_reservation_is_not_invented_as_provider_spend():
    budget = PlannedBudget(plan())
    with request_scope("body", "never-sent"):
        budget.reserve(25000)
    snapshot = budget.snapshot()
    assert campaign.unknown_token_hold(snapshot, 0) == 0
    restored = CostBudget.from_snapshot(snapshot)
    assert restored.snapshot()["usage_uncertain"] is True
    with request_scope("body", "retry"):
        with pytest.raises(ValueError, match="unknown"):
            restored.reserve(1)


def test_legacy_unknown_charge_keeps_existing_semantics():
    assert (
        campaign.unknown_token_hold({"schema_version": 1, "used_tokens": 123}, 20)
        == 103
    )
