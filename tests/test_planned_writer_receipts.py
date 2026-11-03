"""Carry authenticated writer escrows without refunding or replaying old calls."""

import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_native_responses import envelope, owner_for
from test_planned_budget import plan
from test_writer_thinking_profile import configured  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.llm.responses import RESPONSES_CONTRACT
from litagent.llm.routing import ModelBoundClient
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.quality import digest as writer_digest
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import request_scope
from litagent.writing_v2 import WriterSession, schema_for
from scripts import run_frozen_writing as campaign


async def parent_fixture(
    case, tmp_path, monkeypatch, mode="cancelled", same_input=False
):
    """Persist real SDK, recorder and writer output with only HTTP replaced."""
    summary = build_config_summary(load_config(), "strict_auto_v2")
    summary["effective"]["llm"]["model"] = "pro"
    summary["effective"]["llm"]["model_routing"] = {
        "generation": "pro",
        "evaluation": "pro",
    }
    monkeypatch.chdir(tmp_path)
    root, run_id = tmp_path / "parent", "offline-parent"
    root.mkdir()
    paths = Path("artifacts/targeted-repairs")
    paths.mkdir(parents=True)
    sent = []

    async def handler(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(200, json=envelope(json.dumps(case.body)))
        if mode == "cancelled":
            raise asyncio.CancelledError()
        if mode == "rejected":
            return httpx.Response(429, json={"error": {"code": "RateLimit"}})
        body = envelope(json.dumps(case.body))
        body.pop("usage")
        return httpx.Response(200, json=body)

    owner, budget, events = await owner_for(handler)
    repository = ArchiveRepository("artifacts/runs")
    recorder = RunRecorder(run_id, "offline", repository)
    original = owner._trace_hook

    def hook(event, data):
        original(event, data)
        recorder(event, data)

    owner._trace_hook = hook
    ledger_path = paths / (run_id + ".json.budget-ledger.json")
    budget.activate_plan(plan(), path=ledger_path)
    session = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        paths / (run_id + ".json"),
        run_id,
        case.packet,
        case.aliases,
        thinking_mode="disabled",
        provider_api="responses",
    )
    try:
        await session.call(
            "body:introduction",
            "Use sources for introduction.",
            case.packet,
            schema_for("body", case.aliases),
        )
        error = asyncio.CancelledError if mode == "cancelled" else ValueError
        with pytest.raises(error):
            await session.call(
                "body:taxonomy",
                (
                    "Use sources for introduction."
                    if same_input
                    else "Use sources for taxonomy."
                ),
                case.packet,
                schema_for("body", case.aliases),
            )
    finally:
        await owner.close()
    snapshot = budget.snapshot()
    known = sum(
        d["total_tokens"]
        for e, d in events
        if e == "llm.complete" and d.get("usage_available")
    )
    archive_path = Path("artifacts/runs", run_id + ".json")
    atomic_json(archive_path, recorder.snapshot())
    history = {
        "model_requests": 7,
        "model_known_tokens": 500,
        "model_unknown_conservative_tokens": 30,
    }
    manifest = {
        "version": campaign.VERSION,
        "contract": session.journal["contract"],
        "writer_prompt_contract": session.journal["prompt_contract"],
        "history": history,
        "config": summary,
        "budget_plans": {"offline": snapshot["plan"]},
    }
    manifest["identity"] = digest(manifest)
    row = {
        "case_id": "offline",
        "run_id": run_id,
        "status": "blocked",
        "publishable": False,
        "requests": 2,
        "known_tokens": known,
        "unknown_hold": campaign.unknown_token_hold(snapshot, known),
        "usage_complete": False,
        "budget_snapshot": snapshot,
    }
    state = {
        "identity": manifest["identity"],
        "status": "stopped",
        "stop_reason": "usage_unknown",
        "cases": [row],
    }
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "state.json", state)
    return SimpleNamespace(
        root=root,
        manifest=manifest,
        state=state,
        row=row,
        history=history,
        sent=sent,
        archive_path=archive_path,
        ledger_path=ledger_path,
        journal_path=session.journal_path,
    )


def carry(parent):
    """Exercise the actual parent-chain consumer with its explicit new policy."""
    return campaign.accounting_history(
        parent.history,
        [parent.root],
        unknown_policy=campaign.PLANNED_UNKNOWN_ACCOUNTING,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["cancelled", "rejected", "missing_usage"])
async def test_explicit_writer_receipt_carries_known_and_unknown_without_replay(
    case, tmp_path, monkeypatch, mode
):
    parent = await parent_fixture(case, tmp_path, monkeypatch, mode)
    files = list(tmp_path.rglob("*.json"))
    before = {p: p.read_bytes() for p in files}
    history, receipts = carry(parent)
    assert history == {
        "model_requests": 9,
        "model_known_tokens": 620,
        "model_unknown_conservative_tokens": 30 + parent.row["unknown_hold"],
    }
    assert len(receipts) == 1 and "conservative_rejections" not in receipts[0]
    receipt = receipts[0]["planned_unknown_receipts"][0]
    assert receipt["version"] == campaign.PLANNED_UNKNOWN_ACCOUNTING
    assert receipt["unknown_hold"] == parent.row["unknown_hold"] > 0
    assert len(receipt["requests"]) == 2
    assert {x["status"] for x in receipt["requests"]} == {
        "settled",
        "interrupted_no_replay",
    }
    assert all(p.read_bytes() == raw for p, raw in before.items())
    restored = CostBudget.from_snapshot(parent.row["budget_snapshot"])
    with request_scope("body", "new-attempt"):
        with pytest.raises(ValueError, match="unknown"):
            restored.reserve(1)
    assert len(parent.sent) == 2
    with pytest.raises(ValueError, match="planned_unknown_receipt_unsupported"):
        campaign.accounting_history(
            parent.history,
            [parent.root],
            unknown_policy=campaign.CONSERVATIVE_ACCOUNTING,
        )
    with pytest.raises(ValueError, match="parent_duplicate"):
        campaign.accounting_history(
            parent.history,
            [parent.root, parent.root],
            unknown_policy=campaign.PLANNED_UNKNOWN_ACCOUNTING,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "known",
        "hold",
        "flag",
        "plan",
        "ledger",
        "wire",
        "reservation",
        "input",
        "schema",
        "messages",
        "model",
        "duplicate",
        "missing_terminal",
        "usage",
        "unsent",
        "journal_missing",
        "non_writer",
        "requests_bool",
        "output_plan",
    ],
)
async def test_planned_receipt_refuses_forged_or_incomplete_records(
    case, tmp_path, monkeypatch, fault
):
    parent = await parent_fixture(case, tmp_path, monkeypatch)
    archive = json.loads(parent.archive_path.read_text())
    journal = json.loads(parent.journal_path.read_text())
    ledger = copy.deepcopy(parent.row["budget_snapshot"])
    call = journal["calls"]["body:taxonomy"]
    entry = next(e for e in ledger["entries"].values() if e["status"] != "settled")
    starts = [e for e in archive["events"] if e["event"] == "llm.start"]
    if fault == "known":
        parent.row["known_tokens"] += 1
    elif fault == "hold":
        parent.row["unknown_hold"] = 0
    elif fault == "flag":
        parent.row["usage_complete"] = True
    elif fault == "plan":
        parent.manifest["budget_plans"]["offline"]["planned_tokens"] += 10000
    elif fault == "ledger":
        ledger["used_tokens"] += 1
    elif fault == "wire":
        entry["wire_sha256"] = "0" * 64
        parent.row["budget_snapshot"] = ledger
    elif fault == "reservation":
        entry["reserved_tokens"] += 1
        parent.row["budget_snapshot"] = ledger
        parent.row["unknown_hold"] += 1
    elif fault == "input":
        call["input_sha256"] = "0" * 64
    elif fault == "schema":
        call["provider_schema_sha256"] = "0" * 64
    elif fault == "messages":
        starts[-1]["data"]["messages"][0]["content"] += " changed"
    elif fault == "model":
        starts[-1]["data"]["model"] = "unapproved-model"
    elif fault == "duplicate":
        archive["events"].append(copy.deepcopy(starts[-1]))
    elif fault == "missing_terminal":
        archive["events"] = [e for e in archive["events"] if e["event"] != "llm.failed"]
    elif fault == "usage":
        end = next(e["data"] for e in archive["events"] if e["event"] == "llm.complete")
        end["prompt_tokens"] += 1
        end["total_tokens"] += 1
    elif fault == "unsent":
        entry["sent"] = False
        parent.row["budget_snapshot"] = ledger
    elif fault == "journal_missing":
        parent.journal_path.unlink()
    elif fault == "non_writer":
        journal["calls"].pop("body:taxonomy")
    elif fault == "output_plan":
        start = starts[-1]["data"]
        call["max_tokens"] -= 1
        start["max_tokens"] = cap = call["max_tokens"]
        call["reservation"] -= 1
        entry["reserved_tokens"] -= 1
        parent.row["unknown_hold"] -= 1
        extra = {"thinking": {"type": "disabled"}}
        rf = start["response_format"]
        call["input_sha256"] = writer_digest(
            [
                start["messages"],
                rf,
                {"max_output_tokens": cap},
                extra,
                ["responses", RESPONSES_CONTRACT],
            ]
        )
        entry["wire_sha256"] = writer_digest(
            [
                start["messages"],
                {
                    "response_format": rf,
                    "max_output_tokens": cap,
                    "extra_body": extra,
                    "provider_api": "responses",
                    "model": start["model"],
                },
                "responses",
            ]
        )
        parent.row["budget_snapshot"] = ledger
    else:
        parent.row["requests"] = True
    parent.manifest["identity"] = digest(
        {k: v for k, v in parent.manifest.items() if k != "identity"}
    )
    parent.state["identity"] = parent.manifest["identity"]
    atomic_json(parent.root / "manifest.json", parent.manifest)
    atomic_json(parent.root / "state.json", parent.state)
    atomic_json(parent.archive_path, archive)
    atomic_json(parent.ledger_path, ledger)
    if fault != "journal_missing":
        atomic_json(parent.journal_path, journal)
    reason = (
        "output_plan_conflict"
        if fault == "output_plan"
        else "frozen_accounting|planned_|budget_plan_"
    )
    with pytest.raises(ValueError, match=reason):
        carry(parent)
    assert len(parent.sent) == 2


@pytest.mark.asyncio
async def test_ambiguous_identical_writer_requests_are_not_matched_by_text_or_order(
    case, tmp_path, monkeypatch
):
    parent = await parent_fixture(case, tmp_path, monkeypatch, same_input=True)
    with pytest.raises(ValueError, match="writer_binding_ambiguous_or_missing"):
        carry(parent)
    assert len(parent.sent) == 2


@pytest.mark.parametrize("value", [True, "yes", 1])
def test_prepare_does_not_enable_new_accounting_on_legacy_profile(
    tmp_path, configured, value
):
    with pytest.raises(ValueError, match="planned_unknown_configuration_invalid"):
        campaign.prepare(tmp_path / "new", configured, carry_planned_unknown=value)
    assert not (tmp_path / "new").exists()


def test_cli_cannot_change_accounting_at_live_execution(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["frozen", "--root", str(tmp_path), "--live", "--carry-planned-unknown"],
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("classification_review", [False, True])
async def test_prepare_and_validate_bind_the_explicit_receipt_and_history(
    case, tmp_path, monkeypatch, configured, classification_review
):
    parent = await parent_fixture(case, tmp_path, monkeypatch)
    source = Path("source.py")
    source.write_text("# Offline receipt fixture\n")
    monkeypatch.setattr(campaign, "source_paths", lambda: [source])
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: "commit\n")
    for _, run in campaign.CASES:
        atomic_json(Path("artifacts/runs", run + ".json"), {"run": run})

    def bundle(path):
        return {
            "query": "Explain mechanisms and limitations.",
            "source_run_id": path.stem,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "fact_cards_sha256": writer_digest(case.cards),
            "selection": case.selection.to_dict(),
            "requirements": [
                {
                    "requirement_id": "q1",
                    "kind": "question",
                    "text": "Explain mechanisms and limitations.",
                }
            ],
            "methodology": {},
            "extractions": [],
        }

    monkeypatch.setattr(campaign, "read_bundle", bundle)
    atomic_json(
        Path(
            "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
        ),
        {"cumulative": parent.history},
    )
    root = tmp_path / "new-suite"
    campaign.prepare(
        root,
        configured,
        accounting_parents=[parent.root],
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=campaign.RANGE_WRITING_PLAN,
        carry_planned_unknown=True,
        classification_review=classification_review,
    )
    manifest = campaign.validate(root, configured)
    assert ("classification_review" in manifest["options"]) is classification_review
    assert all(
        ("classification_review_contract" in p["envelope"]) is classification_review
        for p in manifest["budget_plans"].values()
    )
    assert manifest["accounting_policy"] == campaign.PLANNED_UNKNOWN_ACCOUNTING
    assert manifest["history"]["model_requests"] == 9
    assert (
        manifest["history"]["model_unknown_conservative_tokens"]
        == 30 + parent.row["unknown_hold"]
    )
    assert not (root / "state.json").exists() and len(parent.sent) == 2
    ledger = json.loads(parent.ledger_path.read_text())
    ledger["used_tokens"] += 1
    atomic_json(parent.ledger_path, ledger)
    with pytest.raises(ValueError, match="persisted_snapshot_conflict"):
        campaign.validate(root, configured)
    assert len(parent.sent) == 2
