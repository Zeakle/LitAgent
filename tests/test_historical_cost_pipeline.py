"""Read historical costs across semantic changes without resuming old work."""

import copy
import json
import subprocess
from pathlib import Path

import pytest
from test_native_responses import owner_for
from test_planned_mixed_receipts import (
    carry,
    frozen_bundle,
    mixed_parent,
    persist_parent,
)
from test_writer_thinking_profile import configured  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent import frozen_writing as frozen
from litagent.eval import scoped
from litagent.llm.routing import ModelBoundClient
from litagent.report_audit import atomic_json
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import validate_writer_envelope
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
)
from scripts import run_frozen_writing as campaign


def change_current_semantics(monkeypatch):
    """Change the actual instruction preimage, never mock an identity function."""
    before = scoped.support_validation_identity()
    monkeypatch.setattr(
        scoped,
        "SUPPORT_VALIDATION_INSTRUCTION",
        scoped.SUPPORT_VALIDATION_INSTRUCTION + " New independent test semantics.",
    )
    assert scoped.support_validation_identity() != before
    return before


def durable_bytes(root):
    """Preserve all test-local historical JSON while using the real readers."""
    return {p: p.read_bytes() for p in root.rglob("*.json")}


def assert_costs(parent, result):
    """Separate settled provider use from only sent unknown reservations."""
    history, receipts = result
    ledger = parent.row["budget_snapshot"]
    hold = sum(
        e["reserved_tokens"]
        for e in ledger["entries"].values()
        if e["sent"] and e["actual_tokens"] is None
    )
    assert history == {
        "model_requests": len(parent.http_calls),
        "model_known_tokens": parent.row["known_tokens"],
        "model_unknown_conservative_tokens": hold,
    }
    assert receipts[0]["requests"] == len(parent.http_calls)
    assert receipts[0]["known_tokens"] == parent.row["known_tokens"]
    if hold:
        requests = receipts[0]["planned_unknown_receipts"][0]["requests"]
        unknown = [r for r in requests if r["actual_tokens"] is None]
        assert len(unknown) == 1 and unknown[0]["reservation"] == hold
        assert receipts[0]["unknown_hold"] == hold
    else:
        assert "planned_unknown_receipts" not in receipts[0]
        assert "unknown_hold" not in receipts[0]
    return hold


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "phase,mode,recovery",
    [
        ("assessment", "known", False),
        ("inventory", "missing_usage", False),
        ("assessment", "cancelled", False),
        ("global", "missing_usage", False),
        ("inventory", "missing_usage", True),
        ("assessment", "known", True),
    ],
)
async def test_real_sdk_history_survives_changed_instruction_without_live_restore(
    case, tmp_path, monkeypatch, phase, mode, recovery
):
    parent = await mixed_parent(
        case,
        tmp_path,
        monkeypatch,
        selected=True,
        phase=phase,
        mode=mode,
        recovery=recovery,
        unsent=mode != "known",
    )
    original = carry(parent)
    hold = assert_costs(parent, original)
    before = durable_bytes(tmp_path)
    requests = len(parent.http_calls)
    old = change_current_semantics(monkeypatch)
    plan = parent.row["budget_snapshot"]["plan"]
    assert plan["envelope"]["support_validation_identity"] == old
    assert carry(parent) == original
    assert_costs(parent, carry(parent))
    assert (
        campaign.historical_unknown_token_hold(
            parent.row["budget_snapshot"],
            parent.row["known_tokens"],
            expected_plan=plan,
        )
        == hold
    )
    # A cost-only reader grants no reservation owner or old scientific restore.
    with pytest.raises(ValueError, match="support_validation"):
        validate_writer_envelope(plan["envelope"])
    with pytest.raises(ValueError, match="support_validation"):
        CostBudget.from_snapshot(parent.row["budget_snapshot"])
    archive = json.loads(parent.archive_path.read_text())
    records = campaign.evaluation_progress_records(parent.row["run_id"], archive)
    assert records
    for record in records:
        with pytest.raises(ValueError, match="support_validation"):
            CostBudget.from_snapshot(record["checkpoint"]["budget"])
    assert len(parent.http_calls) == requests
    assert all(p.read_bytes() == raw for p, raw in before.items())


@pytest.mark.asyncio
async def test_current_owner_cannot_reuse_historical_selected_journal(
    case, tmp_path, monkeypatch
):
    parent = await mixed_parent(
        case, tmp_path, monkeypatch, mode="known", selected=True
    )
    original = carry(parent)
    before = durable_bytes(tmp_path)
    change_current_semantics(monkeypatch)
    current = frozen.plan_frozen_budget(
        parent.bundle, budget_policy="planned-budget-v7", **parent.flags
    )
    assert current["identity"] != parent.row["budget_snapshot"]["plan"]["identity"]

    async def no_request(_request):
        pytest.fail("journal identity rejection must precede external transport")

    owner, budget, _ = await owner_for(no_request)
    try:
        budget.activate_plan(current)
        with pytest.raises(WriterError, match="writer_checkpoint_identity_conflict"):
            WriterSession(
                ModelBoundClient(owner, "pro", "generation"),
                Path("artifacts/targeted-repairs", parent.row["run_id"] + ".json"),
                parent.row["run_id"],
                case.packet,
                case.aliases,
                provider_api="responses",
                syntax_policy=EOF_CONTAINER_POLICY,
                writing_plan=RANGE_WRITING_PLAN,
            )
        assert carry(parent) == original
        assert all(p.read_bytes() == raw for p, raw in before.items())
    finally:
        await owner.close()


def local_prepare_materials(case, parent, tmp_path):
    """Supply valid local input archives to actual prepare/read_bundle/validate."""
    # Git is only source metadata for this isolated test directory. No repo data
    # or remote is touched, and the actual preparation functions are not mocked.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Offline Test",
            "-c",
            "user.email=offline@example.invalid",
            "-c",
            "core.hooksPath=/dev/null",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "fixture",
        ],
        check=True,
        cwd=tmp_path,
    )
    source = Path("src/litagent/eval/scoped.py")
    source.parent.mkdir(parents=True)
    source.write_bytes(Path(scoped.__file__).read_bytes())
    fixture = frozen_bundle(case)
    for _, run_id in campaign.CASES:
        archive = {
            "run_id": run_id,
            "query": case.packet["query"],
            "events": [],
            "nodes": {
                "selection": {
                    "kind": "evidence.retrieve",
                    "output": fixture["selection"],
                },
                "extractor": {
                    "agent_type": "extractor",
                    "output": fixture["extractions"],
                },
            },
            "report": {"metadata": {"review_methodology": fixture["methodology"]}},
        }
        atomic_json(Path("artifacts/runs", run_id + ".json"), archive)
    atomic_json(
        Path(
            "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
        ),
        {"cumulative": parent.history},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["known", "missing_usage"])
async def test_actual_prepare_validate_recomputes_old_costs_with_new_semantics(
    case, tmp_path, monkeypatch, configured, mode
):
    parent = await mixed_parent(case, tmp_path, monkeypatch, selected=True, mode=mode)
    original = carry(parent)
    local_prepare_materials(case, parent, tmp_path)
    before = durable_bytes(parent.root)
    old_identity = change_current_semantics(monkeypatch)
    root = tmp_path / "new-test-only-suite"
    campaign.prepare(
        root,
        configured,
        accounting_parents=[parent.root],
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        accounting_policy=campaign.PLANNED_MIXED_UNKNOWN_ACCOUNTING,
        **parent.flags,
    )
    manifest = campaign.validate(root, configured)
    assert "src/litagent/eval/scoped.py" in manifest["source_hashes"]
    assert manifest["history"] == original[0]
    assert manifest["accounting_receipts"] == original[1]
    assert all(
        plan["envelope"]["support_validation_identity"]
        == scoped.support_validation_identity()
        != old_identity
        for plan in manifest["budget_plans"].values()
    )
    assert carry(parent) == original
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert not any(
        (root / name).exists()
        for name in ("state.json", "execution.lock", "offline-gates.json")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "journal_top_missing",
        "journal_top_current",
        "journal_call_missing",
        "journal_call_current",
        "progress_tag_missing",
        "progress_tag_current",
        "progress_envelope_current",
        "progress_envelope_capacity",
        "journal_transport",
        "frozen_source",
        "snapshot_missing",
        "snapshot_none",
        "snapshot_legacy",
        "owner",
        "wire",
        "usage",
    ],
)
async def test_known_history_authenticates_semantic_tags_owner_and_wire(
    case, tmp_path, monkeypatch, fault
):
    parent = await mixed_parent(
        case,
        tmp_path,
        monkeypatch,
        mode="known",
        selected=True,
    )
    original = carry(parent)
    assert_costs(parent, original)
    archive = json.loads(parent.archive_path.read_text())
    journal = json.loads(parent.journal_path.read_text())
    record = campaign.evaluation_progress_records(parent.row["run_id"], archive)[0]
    checkpoint = record["checkpoint"]
    change_current_semantics(monkeypatch)
    assert carry(parent) == original
    current = scoped.support_validation_identity()
    field = "support_validation_identity"
    if fault.startswith("journal_top"):
        if fault.endswith("missing"):
            journal.pop(field)
        else:
            journal[field] = current
    elif fault.startswith("journal_call"):
        call = next(iter(journal["calls"].values()))
        if fault.endswith("missing"):
            call.pop(field)
        else:
            call[field] = current
    elif fault.startswith("progress_tag"):
        if fault.endswith("missing"):
            checkpoint["progress"].pop(field)
        else:
            checkpoint["progress"][field] = current
    elif fault == "progress_envelope_current":
        checkpoint["progress"]["writer_envelope"][field] = current
    elif fault == "progress_envelope_capacity":
        envelope = checkpoint["progress"]["writer_envelope"]
        envelope["body_max_units"] -= 1
        envelope["identity"] = frozen.digest(
            {k: v for k, v in envelope.items() if k != "identity"}
        )
    elif fault == "journal_transport":
        next(iter(journal["calls"].values()))["transport_identity"] = "f" * 64
    elif fault == "frozen_source":
        path = parent.root / "source/src/litagent/eval/scoped.py"
        path.write_bytes(path.read_bytes() + b"\n# changed frozen source\n")
    elif fault == "snapshot_missing":
        parent.row.pop("budget_snapshot")
    elif fault == "snapshot_none":
        parent.row["budget_snapshot"] = None
    elif fault == "snapshot_legacy":
        parent.row["budget_snapshot"] = {
            **parent.row["budget_snapshot"],
            "schema_version": 1,
        }
    elif fault == "owner":
        checkpoint["session_id"] = "different-run"
    elif fault == "wire":
        start = next(e["data"] for e in archive["events"] if e["event"] == "llm.start")
        start["messages"][0][
            "content"
        ] += " Forged old input with unchanged event totals."
    elif fault == "usage":
        end = next(e["data"] for e in archive["events"] if e["event"] == "llm.complete")
        end["prompt_tokens"] += 1
        end["completion_tokens"] -= 1
    atomic_json(parent.journal_path, journal)
    atomic_json(Path(record["path"]), checkpoint)
    atomic_json(parent.archive_path, archive)
    persist_parent(parent)
    with pytest.raises(ValueError):
        carry(parent)


def test_real040_inventory_scope_uses_request_position_not_later_allocation():
    """Keep real128/167/168 cost scopes distinct without importing report text."""
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/run040-cost-snapshot.json").read_text()
    )
    assert fixture["science_gold_standard"] is False
    ledger = fixture["snapshot"]
    scopes = [ledger["plan"]["envelope"]] + [
        r["allocation"]["envelope"] for r in ledger["capacity_revisions"]
    ]
    assert [e["model_assertions"] for e in scopes] == [128, 167, 168]
    inventories = [
        (index, request_id)
        for index, (request_id, entry) in enumerate(ledger["entries"].items())
        if entry["stage"] == "inventory" or "inventory_recovery" in entry
    ]
    assert [i for i, _ in inventories] == [7, 20]
    original = copy.deepcopy(ledger)
    for expected_index, (entry_index, request_id) in enumerate(inventories):
        record = {
            "checkpoint": {
                "progress": {"writer_envelope": copy.deepcopy(scopes[expected_index])}
            }
        }
        assert ledger["entries"][request_id]["sent"] is True
        campaign._historical_progress_envelope(record, ledger, request_id)
        for wrong_index, wrong_scope in enumerate(scopes):
            if wrong_index == expected_index:
                continue
            record["checkpoint"]["progress"]["writer_envelope"] = copy.deepcopy(
                wrong_scope
            )
            with pytest.raises(ValueError, match="progress_envelope_invalid"):
                campaign._historical_progress_envelope(record, ledger, request_id)
        altered = copy.deepcopy(scopes[expected_index])
        altered["transport_revision"] = "different-transport"
        record["checkpoint"]["progress"]["writer_envelope"] = altered
        with pytest.raises(ValueError, match="progress_envelope_invalid"):
            campaign._historical_progress_envelope(record, ledger, request_id)
    assert ledger == original
