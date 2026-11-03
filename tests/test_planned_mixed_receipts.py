"""Authenticate mixed planned escrows using real SDK and checkpoint producers."""

import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_flat_writer_pipeline import FlatGateway
from test_mixed_inventory_pipeline import MixedInventoryGateway
from test_native_responses import owner_for
from test_planned_writer_receipts import parent_fixture as writer_parent
from test_scientific_repair_pipeline import ScientificRepairGateway
from test_writer_thinking_profile import configured  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent import frozen_writing as frozen
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.quality import digest
from litagent.report_audit import atomic_json
from litagent.review_structure import build_methodology
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.planned_budget import request_scope
from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN
from scripts import run_frozen_writing as campaign


def frozen_bundle(case):
    """Provide real discovery and evidence lineage for a portable two-paper run."""
    records = {
        "__search_queries__": {"search_fixture": "relation networks"},
        **{
            key: list(case.ledger.values())
            for key in ("search_fixture", "dedup", "relevance_gate")
        },
    }
    return {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {
                "paper_id": row["paper_id"],
                "claims": [row["text"]],
                "evidence_items": [row],
            }
            for row in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"],
            records,
            source_outcomes=[
                {
                    "task_id": "search_fixture",
                    "source": "offline-public-fixture",
                    "query": "relation networks",
                    "status": "completed",
                    "result_count": len(case.ledger),
                }
            ],
            started_at=1700000000,
        ),
        "fact_cards_sha256": digest(case.cards),
    }


async def mixed_parent(
    case,
    tmp_path,
    monkeypatch,
    *,
    phase="assessment",
    mode="missing_usage",
    candidate=0,
    recovery=False,
    unsent=False,
    selected=False,
):
    """Persist a genuine frozen run, replacing only its external HTTP replies."""
    monkeypatch.chdir(tmp_path)
    run_id = "offline-mixed-parent"
    root = tmp_path / "parent"
    root.mkdir()
    gateway = (
        MixedInventoryGateway(case)
        if recovery
        else ScientificRepairGateway(case, 2) if candidate else FlatGateway(case)
    )
    http_calls, faulted = [], []

    async def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body.get("input", body.get("messages"))[1]["content"])
        current = (
            "inventory"
            if "allowed_citation_aliases" in payload
            else (
                "assessment"
                if "support_passages" in payload
                else "global" if "global_output_contract" in payload else "writer"
            )
        )
        http_calls.append(current)
        eligible = mode != "known" and current == phase and not faulted
        if candidate:
            eligible = eligible and gateway.inventories >= candidate + 1
        if recovery:
            eligible = eligible and "inventory_recovery" in payload
        if eligible:
            faulted.append(current)
            if unsent:
                # Exercise the durable producer without beginning a request.
                # This reservation must never become a paid/unknown request.
                with request_scope("repair", "offline-reserved-never-sent"):
                    owner._cost_budget.reserve(1)
            if mode == "cancelled":
                raise asyncio.CancelledError()
            if mode == "rejected":
                return httpx.Response(403, json={"error": {"code": "OfflineDenied"}})
            response = await gateway(request)
            value = response.json()
            value.pop("usage", None)
            return httpx.Response(200, json=value)
        return await gateway(request)

    owner, _, _ = await owner_for(handler)
    config = load_config()
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens = 20000
    config.adversarial.review_max_tokens = 4096
    recorder = RunRecorder(
        run_id, case.packet["query"], ArchiveRepository("artifacts/runs")
    )
    summary = build_config_summary(config, "strict_auto_v2")
    recorder.set_config_summary(summary)
    agent = LitAgent(
        config, run_policy=RunPolicy.isolated_benchmark(), trace_hook=recorder
    )
    repair_path = Path("artifacts/targeted-repairs", run_id + ".json")
    agent._targeted_repair_state_path = repair_path

    def factory(**kwargs):
        owner._model = kwargs["model"]
        owner._cost_budget = kwargs["cost_budget"]
        owner._trace_hook = kwargs["trace_hook"]
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    original_options = frozen.frozen_options
    monkeypatch.setattr(
        frozen,
        "frozen_options",
        lambda: original_options().model_copy(update={"report_language": "en"}),
    )
    bundle = frozen_bundle(case)
    flags = dict(
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        classification_review=True,
    )
    if selected:
        flags.update(
            repair_feedback=True,
            reference_contract="ordered-reference-set-v1",
            transport_revision="host-lossless-transport-v1",
            support_validation_revision="host-support-receipts-v1",
        )
    plan = frozen.plan_frozen_budget(bundle, budget_policy="planned-budget-v7", **flags)
    report = None
    cancelled = False
    try:
        report = await frozen.run_frozen(
            agent,
            bundle,
            run_id,
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            **flags,
        )
    except asyncio.CancelledError:
        cancelled = True
    finally:
        await agent.cleanup()
    assert faulted == ([] if mode == "known" else [phase]), (http_calls, report)
    assert cancelled is (mode == "cancelled")
    recorder.finalize(report=report, error="offline_cancelled" if cancelled else None)
    archive_path = Path("artifacts/runs", run_id + ".json")
    ledger_path = Path(str(repair_path) + ".budget-ledger.json")
    journal_path = Path(str(repair_path) + ".writer-v2.json")
    snapshot = json.loads(ledger_path.read_text())
    archive = json.loads(archive_path.read_text())
    journal = json.loads(journal_path.read_text())
    known = sum(
        event["data"]["total_tokens"]
        for event in archive["events"]
        if event["event"] == "llm.complete"
        and event["data"].get("usage_available") is True
    )
    history = {
        "model_requests": 0,
        "model_known_tokens": 0,
        "model_unknown_conservative_tokens": 0,
    }
    manifest = {
        "version": campaign.VERSION,
        "contract": journal["contract"],
        "writer_prompt_contract": journal["prompt_contract"],
        "history": history,
        "config": summary,
        "budget_plans": {"offline": snapshot["plan"]},
        "cases": [{"case_id": "offline", "bundle": bundle}],
        "writing_profile": campaign.ROUND_PROFILE,
        "accounting_policy": campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING,
        **(flags if selected else {}),
    }
    if selected:
        import litagent.eval.scoped as scoped

        # This test-only parent freezes the actual semantic producer bytes.
        # Historical accounting authenticates them without importing old code.
        name = "src/litagent/eval/scoped.py"
        raw = Path(scoped.__file__).read_bytes()
        path = root / "source" / name
        path.parent.mkdir(parents=True)
        path.write_bytes(raw)
        manifest["source_hashes"] = {name: hashlib.sha256(raw).hexdigest()}
    manifest["identity"] = campaign.digest(manifest)
    row = {
        "case_id": "offline",
        "run_id": run_id,
        "status": "blocked",
        "publishable": False,
        "requests": snapshot["requests_sent"],
        "known_tokens": known,
        "unknown_hold": campaign.unknown_token_hold(snapshot, known),
        "usage_complete": not snapshot["usage_uncertain"],
        "budget_snapshot": snapshot,
    }
    if mode == "known":
        row["publishable"] = (
            (report or {}).get("delivery", {}).get("publishable", False)
        )
        row["status"] = "passed" if row["publishable"] else "blocked"
    state = {
        "identity": manifest["identity"],
        "status": "completed" if mode == "known" else "stopped",
        "stop_reason": "usage_unknown" if snapshot["usage_uncertain"] else None,
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
        archive_path=archive_path,
        ledger_path=ledger_path,
        journal_path=journal_path,
        report=report,
        gateway=gateway,
        http_calls=http_calls,
        config=config,
        bundle=bundle,
        flags=flags,
    )


def carry(parent, policy=None):
    """Read the actual parent-chain consumer without replaying any provider call."""
    return campaign.accounting_history(
        parent.history,
        [parent.root],
        unknown_policy=(policy or campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING),
    )


def persist_parent(parent):
    """Update only this test's parent identity after an intentional mutation."""
    parent.manifest["identity"] = campaign.digest(
        {key: value for key, value in parent.manifest.items() if key != "identity"}
    )
    parent.state["identity"] = parent.manifest["identity"]
    atomic_json(parent.root / "manifest.json", parent.manifest)
    atomic_json(parent.root / "state.json", parent.state)


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["inventory", "assessment", "global"])
@pytest.mark.parametrize("mode", ["missing_usage", "rejected", "cancelled"])
async def test_actual_mixed_stages_keep_unknown_escrows(
    case, tmp_path, monkeypatch, phase, mode
):
    parent = await mixed_parent(case, tmp_path, monkeypatch, phase=phase, mode=mode)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
    with pytest.raises(ValueError, match="planned_writer_binding"):
        carry(parent, campaign.PLANNED_UNKNOWN_ACCOUNTING)
    history, receipts = carry(parent)
    receipt = receipts[0]["planned_unknown_receipts"][0]
    assert receipt["version"] == campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING
    assert history == {
        "model_requests": len(parent.http_calls),
        "model_known_tokens": parent.row["known_tokens"],
        "model_unknown_conservative_tokens": parent.row["unknown_hold"],
    }
    assert history["model_unknown_conservative_tokens"] > 0
    assert len(receipt["requests"]) == len(parent.http_calls)
    unknown = [r for r in receipt["requests"] if r["actual_tokens"] is None]
    assert len(unknown) == 1 and unknown[0]["phase"] == phase
    assert unknown[0]["status"] == "interrupted_no_replay"
    assert unknown[0]["reservation"] == parent.row["unknown_hold"]
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.asyncio
async def test_writer_only_v2_dispatch_is_exactly_unchanged_v1(
    case, tmp_path, monkeypatch
):
    parent = await writer_parent(case, tmp_path, monkeypatch)
    assert carry(parent) == carry(parent, campaign.PLANNED_UNKNOWN_ACCOUNTING)


@pytest.mark.asyncio
async def test_unsent_reservation_and_future_global_slot_are_not_charged(
    case, tmp_path, monkeypatch
):
    parent = await mixed_parent(case, tmp_path, monkeypatch, unsent=True)
    ledger = parent.row["budget_snapshot"]
    unsent = ledger["entries"]["offline-reserved-never-sent"]
    assert unsent["status"] == "reserved_not_sent" and unsent["sent"] is False
    allocations = [r["allocation"] for r in ledger["capacity_revisions"]]
    assert any(
        slot["stage"] == "global"
        for allocation in allocations
        for slot in allocation["round"]["requests"]
    )
    assert not any(
        e["sent"] and e["stage"] == "global" for e in ledger["entries"].values()
    )
    history, receipts = carry(parent)
    requests = receipts[0]["planned_unknown_receipts"][0]["requests"]
    assert "offline-reserved-never-sent" not in {r["request_id"] for r in requests}
    assert history["model_requests"] == len(parent.http_calls)
    assert history["model_unknown_conservative_tokens"] == sum(
        e["reserved_tokens"]
        for e in ledger["entries"].values()
        if e["sent"] and e["actual_tokens"] is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing_usage", "cancelled"])
async def test_recovery_inventory_is_authenticated_as_repair(
    case, tmp_path, monkeypatch, mode
):
    parent = await mixed_parent(
        case, tmp_path, monkeypatch, phase="inventory", recovery=True, mode=mode
    )
    _, receipts = carry(parent)
    receipt = receipts[0]["planned_unknown_receipts"][0]
    recovery = [
        r
        for r in receipt["requests"]
        if r["phase"] == "inventory" and r["stage"] == "repair"
    ]
    assert len(recovery) == 1 and recovery[0]["actual_tokens"] is None
    entry = parent.row["budget_snapshot"]["entries"][recovery[0]["request_id"]]
    assert entry["inventory_recovery"]["source_report_sha256"]
    assert len(receipt["evaluation_progress"]) == 2


@pytest.mark.asyncio
async def test_uncommitted_second_candidate_cost_binding_is_not_fake_report(
    case, tmp_path, monkeypatch
):
    parent = await mixed_parent(
        case, tmp_path, monkeypatch, candidate=2, mode="cancelled"
    )
    journal = json.loads(parent.journal_path.read_text())
    assert parent.gateway.repair_count == 2
    assert len(journal["candidates"]) == 1
    _, receipts = carry(parent)
    progress = receipts[0]["planned_unknown_receipts"][0]["evaluation_progress"]
    assert len(progress) == 3 and len({p["identity"] for p in progress}) == 3
    assert any(
        p["normal_identity_preimage"] == "not_available_cost_binding_only"
        for p in progress
    )


def change_progress_owners(parent, owner, *, legacy=False):
    """Mutate test-local durable owner evidence without inventing survey text."""
    archive = json.loads(parent.archive_path.read_text())
    records = campaign.evaluation_progress_records(parent.row["run_id"], archive)
    for record in records:
        checkpoint = record["checkpoint"]
        if owner is None:
            checkpoint.pop("session_id", None)
        else:
            checkpoint["session_id"] = owner
        atomic_json(Path(record["path"]), checkpoint)
    for event in archive["events"]:
        if event["event"] == "evaluation.batch":
            if owner is None:
                event["data"].pop("session_id", None)
            else:
                event["data"]["session_id"] = owner
    if legacy:
        parent.manifest["accounting_policy"] = campaign.PLANNED_UNKNOWN_ACCOUNTING
    atomic_json(parent.archive_path, archive)
    persist_parent(parent)


@pytest.mark.asyncio
@pytest.mark.parametrize("owner", [None, "other-run"])
async def test_new_policy_requires_exact_local_owner_even_with_survey(
    case, tmp_path, monkeypatch, owner
):
    parent = await mixed_parent(case, tmp_path, monkeypatch)
    carry(parent)
    change_progress_owners(parent, owner)
    with pytest.raises(ValueError):
        carry(parent)


@pytest.mark.asyncio
async def test_historical_ownerless_initial_has_exact_preimage_but_candidate_does_not(
    case, tmp_path, monkeypatch
):
    parent = await mixed_parent(case, tmp_path, monkeypatch)
    change_progress_owners(parent, None, legacy=True)
    _, receipts = carry(parent)
    progress = receipts[0]["planned_unknown_receipts"][0]["evaluation_progress"]
    assert all(p["normal_identity_preimage"] == "verified" for p in progress)


@pytest.mark.asyncio
async def test_historical_ownerless_uncommitted_candidate_is_not_authenticated(
    case, tmp_path, monkeypatch
):
    parent = await mixed_parent(
        case, tmp_path, monkeypatch, candidate=2, mode="cancelled"
    )
    change_progress_owners(parent, None, legacy=True)
    with pytest.raises(ValueError):
        carry(parent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "messages",
        "schema",
        "temperature",
        "model",
        "output_cap",
        "operation_duplicate",
        "missing_terminal",
        "known_usage",
        "unknown_usage_flag",
        "progress_input",
        "progress_report",
        "progress_config",
        "progress_identity",
        "progress_phase",
        "progress_orphan",
        "progress_missing",
        "cross_run",
        "plan",
        "round_wire",
        "stage",
        "reservation",
        "ledger_missing",
        "checkpoint_missing",
        "wire_sha",
        "archive_config",
        "archive_fingerprint",
    ],
)
async def test_mixed_receipts_reject_incomplete_or_forged_evidence(
    case, tmp_path, monkeypatch, fault
):
    parent = await mixed_parent(case, tmp_path, monkeypatch)
    # Every negative starts with an actual accepted producer output. A rejection
    # cannot pass merely because the fixture was already invalid.
    carry(parent)
    archive = json.loads(parent.archive_path.read_text())
    record = campaign.evaluation_progress_records(parent.row["run_id"], archive)[0]
    checkpoint_path = Path(record["path"])
    checkpoint = record["checkpoint"]
    ledger = copy.deepcopy(parent.row["budget_snapshot"])
    start_events = [e for e in archive["events"] if e["event"] == "llm.start"]
    start = next(
        e["data"]
        for e in start_events
        if "support_passages" in json.loads(e["data"]["messages"][1]["content"])
    )
    terminal = next(
        e["data"]
        for e in archive["events"]
        if e["event"] in {"llm.complete", "llm.failed"}
        and e["data"]["operation_id"] == start["operation_id"]
    )
    entry = next(v for v in ledger["entries"].values() if v["status"] != "settled")
    request = next(
        v for v in checkpoint["progress"]["requests"] if v["phase"] == "assessment"
    )
    if fault == "messages":
        start["messages"][0]["content"] += " forged"
    elif fault == "schema":
        start["response_format"]["json_schema"]["schema"]["description"] = "forged"
    elif fault == "temperature":
        start["temperature"] = 1
    elif fault == "model":
        start["model"] = "different-model"
    elif fault == "output_cap":
        start["max_tokens"] -= 1
    elif fault == "operation_duplicate":
        start["operation_id"] = start_events[0]["data"]["operation_id"]
    elif fault == "missing_terminal":
        archive["events"] = [e for e in archive["events"] if e["data"] is not terminal]
    elif fault == "known_usage":
        known = next(
            e["data"]
            for e in archive["events"]
            if e["event"] == "llm.complete" and e["data"].get("usage_available") is True
        )
        known["prompt_tokens"] += 1
    elif fault == "unknown_usage_flag":
        terminal["usage_available"] = True
    elif fault == "progress_input":
        request["input_sha256"] = "f" * 64
    elif fault == "progress_report":
        checkpoint["report_sha256"] = "f" * 64
    elif fault == "progress_config":
        checkpoint["config_sha256"] = "f" * 64
    elif fault == "progress_identity":
        checkpoint["identity"] = "f" * 64
    elif fault == "progress_phase":
        checkpoint["phase"] = "post_rewrite"
    elif fault == "progress_orphan":
        checkpoint["progress"]["requests"].append(
            {
                **request,
                "input_sha256": "f" * 64,
            }
        )
    elif fault == "progress_missing":
        checkpoint["progress"]["requests"].remove(request)
    elif fault == "cross_run":
        parent.row["run_id"] = archive["run_id"] = "other-run"
        for suffix in (".json.budget-ledger.json", ".json.writer-v2.json", ".json"):
            source = Path("artifacts/targeted-repairs", "offline-mixed-parent" + suffix)
            if source.exists():
                source.with_name("other-run" + suffix).write_bytes(source.read_bytes())
        parent.archive_path = Path("artifacts/runs/other-run.json")
    elif fault == "plan":
        parent.manifest["budget_plans"]["offline"]["planned_tokens"] += 1
    elif fault == "round_wire":
        entry["round_wire_sha256"] = "f" * 64
    elif fault == "stage":
        entry["stage"] = "global"
    elif fault == "reservation":
        entry["reserved_tokens"] += 1
    elif fault == "ledger_missing":
        parent.ledger_path.unlink()
    elif fault == "checkpoint_missing":
        checkpoint_path.unlink()
    elif fault == "wire_sha":
        entry["wire_sha256"] = "f" * 64
    elif fault == "archive_config":
        archive["config"]["effective"]["llm"]["model"] = "different-model"
    elif fault == "archive_fingerprint":
        archive["config_fingerprint"] = "f" * 64
    if fault not in {"ledger_missing", "cross_run"}:
        atomic_json(parent.ledger_path, ledger)
        parent.row["budget_snapshot"] = ledger
    if fault != "checkpoint_missing":
        atomic_json(checkpoint_path, checkpoint)
    atomic_json(parent.archive_path, archive)
    persist_parent(parent)
    with pytest.raises(ValueError):
        carry(parent)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["source", "parent", "feedback", "attempt", "wire_metadata"]
)
async def test_recovery_bindings_require_actual_attempt_and_wire(
    case, tmp_path, monkeypatch, fault
):
    parent = await mixed_parent(
        case, tmp_path, monkeypatch, phase="inventory", recovery=True
    )
    carry(parent)
    ledger = copy.deepcopy(parent.row["budget_snapshot"])
    entry = next(e for e in ledger["entries"].values() if "inventory_recovery" in e)
    if fault in {"source", "parent", "feedback"}:
        key = {
            "source": "source_report_sha256",
            "parent": "parent_allocation",
            "feedback": "feedback_sha256",
        }[fault]
        entry["inventory_recovery"][key] = "f" * 64
        parent.row["budget_snapshot"] = ledger
        atomic_json(parent.ledger_path, ledger)
    elif fault == "attempt":
        path = Path("artifacts/targeted-repairs", parent.row["run_id"] + ".json")
        value = json.loads(path.read_text())
        value["state"]["attempts"][-1]["input_sha256"] = "f" * 64
        atomic_json(path, value)
    else:
        archive = json.loads(parent.archive_path.read_text())
        record = campaign.evaluation_progress_records(parent.row["run_id"], archive)[-1]
        value = record["checkpoint"]
        value["progress"]["inventory_recovery"]["bindings_sha256"] = "f" * 64
        atomic_json(Path(record["path"]), value)
    persist_parent(parent)
    with pytest.raises(ValueError):
        carry(parent)


@pytest.mark.asyncio
async def test_prepare_validate_freezes_mixed_policy_and_rejects_downgrade(
    case, tmp_path, monkeypatch, configured
):
    parent = await mixed_parent(case, tmp_path, monkeypatch)
    carry(parent)
    monkeypatch.setattr(campaign, "source_paths", lambda: [])
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **k: "commit\n")
    for _, run_id in campaign.CASES:
        atomic_json(Path("artifacts/runs", run_id + ".json"), {"run": run_id})

    def read_bundle(path):
        return {
            **frozen_bundle(case),
            "source_run_id": path.stem,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(campaign, "read_bundle", read_bundle)
    atomic_json(
        Path(
            "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
        ),
        {"cumulative": parent.history},
    )
    root = tmp_path / "new-offline-suite"
    campaign.prepare(
        root,
        configured,
        accounting_parents=[parent.root],
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        accounting_policy=campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING,
    )
    manifest = campaign.validate(root, configured)
    assert manifest["accounting_policy"] == campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING
    assert manifest["history"] == carry(parent)[0]
    assert manifest["accounting_receipts"] == carry(parent)[1]
    # Cost policy neither changes generation options nor becomes budget input.
    for value in [
        manifest["options"],
        manifest["writing_parameters"],
        *manifest["budget_plans"].values(),
    ]:
        assert "accounting_policy" not in value
    complete = copy.deepcopy(manifest)
    for policy in [
        None,
        campaign.PLANNED_UNKNOWN_ACCOUNTING,
        campaign.CONSERVATIVE_ACCOUNTING,
    ]:
        manifest = copy.deepcopy(complete)
        if policy is None:
            manifest.pop("accounting_policy")
        else:
            manifest["accounting_policy"] = policy
        manifest["identity"] = campaign.digest(
            {k: v for k, v in manifest.items() if k != "identity"}
        )
        atomic_json(root / "manifest.json", manifest)
        with pytest.raises(ValueError):
            campaign.validate(root, configured)
    assert not (root / "state.json").exists() and not (root / "suite.lock").exists()


@pytest.mark.parametrize(
    "policy,profile,carry_old",
    [
        ("bogus", campaign.ROUND_PROFILE, False),
        (campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING, campaign.FLAT_PROFILE, False),
        (campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING, campaign.ROUND_PROFILE, True),
    ],
)
def test_invalid_prepare_accounting_dependencies_have_no_side_effect(
    tmp_path, policy, profile, carry_old
):
    root = tmp_path / "never-created"
    with pytest.raises(ValueError, match="accounting_policy_invalid"):
        campaign.prepare(
            root,
            None,
            writing_profile=profile,
            accounting_policy=policy,
            carry_planned_unknown=carry_old,
        )
    assert not root.exists()


@pytest.mark.parametrize(
    "extra",
    [
        ["--live"],
        ["--prepare"],
        [
            "--prepare",
            "--writing-profile",
            campaign.ROUND_PROFILE,
            "--carry-planned-unknown",
        ],
    ],
)
def test_cli_accounting_rejects_invalid_dependency_before_config(
    tmp_path, monkeypatch, extra
):
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("configuration loaded")
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_writing",
            "--root",
            str(tmp_path / "never"),
            "--accounting-policy",
            campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING,
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as error:
        campaign.main()
    assert error.value.code == 2 and not (tmp_path / "never").exists()
