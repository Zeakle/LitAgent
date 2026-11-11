"""Exercise complete preparation escrows and their safe writer transition."""

import asyncio
import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_extraction_paper_batches import strategy
from test_question_focused_preparation import attach, chunk, paper
from test_writer_source_view_pipeline import source_flags
from test_writer_v2 import case  # noqa: F401

from litagent.frozen_writing import _checked_preparation_budget, plan_frozen_budget
from litagent.llm.client import OpenAICompatibleClient
from litagent.quality import digest
from litagent.review_structure import build_methodology
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import (
    CONTEXT_LIMIT,
    PREPARATION_BUDGET_CONTRACT,
    planned_backend,
    preparation_budget_plan,
    request_scope,
    settled_preparation_budget,
)
from litagent.workbench import RunOptions
from litagent.writing_v2 import writer_budget_plan


def requests(amounts):
    """Use unique, host-owned positions for controlled accounting fixtures."""
    return [
        {
            "stage": f"preparation_{i}",
            "request_id": hashlib.sha256(str(i).encode()).hexdigest(),
            "reservation": amount,
            "output_tokens": 8192,
        }
        for i, amount in enumerate(amounts)
    ]


def budget_plan(rows, **kwargs):
    """Bind the whole ordered queue, never only its first request."""
    return preparation_budget_plan(rows, identity=digest(rows), **kwargs)


def owner(plan, path):
    """Retain the same owner shared by actual model clients."""
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.activate_plan(plan, path=path)
    return budget


def settle(budget, row, *, amount=None):
    """Use real reservation, binding, send and settlement consumers."""
    usage = {
        "prompt_tokens": row["reservation"] if amount is None else amount,
        "completion_tokens": 0,
    }
    with request_scope(row["stage"], row["request_id"]):
        budget.reserve(row["reservation"])
        planned_backend(budget).bind_wire([row])
        budget.begin_request()
        budget.record(usage)
        before = budget.used
        budget.record(usage)
        assert budget.used == before
        with pytest.raises(ValueError, match="usage_identity_conflict"):
            budget.record({**usage, "completion_tokens": 1})
        budget.release(row["reservation"])


@pytest.mark.parametrize(
    "amounts",
    [
        [48217, 49073, 37475, 49100, 36768, 26531, 26289, 27302],
        [37261, 49294, 48539, 37855, 26735, 26722, 26381],
        [49691, 49939, 49068, 49030, 27016, 25925, 36486, 26525, 27088, 15381],
    ],
)
def test_complete_selected_paper_queues_fit_without_old_phase_borrowing(amounts):
    "Admit the entire measured queue under its explicit preparation allowance."
    value = budget_plan(requests(amounts))
    assert sum(amounts) > 200000
    assert value["status"] == "admitted"
    assert value["contract"] == PREPARATION_BUDGET_CONTRACT
    assert value["required_tokens"] == sum(amounts)
    assert value["planned_tokens"] == value["maximum_tokens"] == 500000
    assert value["scope_bounds"] == {
        "purpose": "source_preparation",
        "model_requests": len(amounts),
        "shared_repairs": 0,
    }


@pytest.mark.parametrize(
    "amounts,reason",
    [
        ([60000] * 8, "total_exceeded"),
        ([8500] * 65, "requests_exceeded"),
        ([CONTEXT_LIMIT + 1], "context_exceeded"),
    ],
)
def test_complete_preparation_overflow_is_rejected_before_activation(amounts, reason):
    "Reject all bounded-resource overflows without creating a paid request."
    plan = budget_plan(requests(amounts))
    assert plan["status"] == "preflight_rejected" and reason in plan["reason"]
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    with pytest.raises(ValueError, match="spend_conflict|preflight_rejected"):
        budget.activate_plan(plan)
    assert budget.used == budget.call_count == 0
    assert budget.snapshot()["requests_sent"] == 0


@pytest.mark.parametrize("fault", ["extra", "stage", "duplicate", "boolean", "id"])
def test_bad_preparation_request_definitions_are_not_coerced(fault):
    "Reject ambiguous identities, unauthorized stages and coerced quantities."
    rows = requests([30000, 30000])
    if fault == "extra":
        rows[0]["endpoint"] = "https://unexpected.invalid"
    elif fault == "stage":
        rows[1]["stage"] = "body"
    elif fault == "duplicate":
        rows[1]["request_id"] = rows[0]["request_id"]
    elif fault == "boolean":
        rows[0]["reservation"] = True
    else:
        rows[0]["request_id"] = "not-a-sha"
    with pytest.raises(ValueError):
        budget_plan(rows)


def test_full_upper_bound_settles_once_and_sorted_snapshot_preserves_all_requests(
    tmp_path,
):
    "Preserve full-cost settlement and replay guards across canonical JSON."
    rows = requests([30000] * 11)
    budget = owner(budget_plan(rows), tmp_path / "ledger.json")
    for row in rows:
        settle(budget, row)
    assert budget.used == 330000
    snapshot = json.loads(json.dumps(budget.snapshot(), sort_keys=True))
    restored = CostBudget.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    assert settled_preparation_budget(restored) is restored
    assert _checked_preparation_budget(snapshot).used == 330000
    assert len(restored.entries) == 11
    with request_scope(rows[0]["stage"], rows[0]["request_id"]):
        with pytest.raises(ValueError, match="no_replay|envelope_exceeded"):
            restored.reserve(30000)
    assert json.loads((tmp_path / "ledger.json").read_text()) == snapshot


@pytest.mark.parametrize("failure", ["not_sent", "inflight", "unknown"])
def test_incomplete_or_unknown_preparation_cannot_be_handed_to_writing(
    tmp_path, failure
):
    "Keep partial and uncertain requests out of the writer handoff."
    rows = requests([30000, 30000])
    value = budget_plan(rows)
    budget = owner(value, tmp_path / "ledger.json")
    with request_scope(rows[0]["stage"], rows[0]["request_id"]):
        budget.reserve(30000)
        if failure != "not_sent":
            planned_backend(budget).bind_wire([rows[0]])
            budget.begin_request()
        if failure == "unknown":
            budget.charge_unknown_usage(30000)
        budget.release(30000)
    snapshot = budget.snapshot()
    with pytest.raises(ValueError):
        _checked_preparation_budget(snapshot)
    restored = CostBudget.from_snapshot(snapshot)
    assert restored.used == budget.used
    if failure == "unknown":
        assert restored.used == 0 and restored.stop_reason == "strict_usage_unknown"
        assert (
            sum(
                e["reserved_tokens"]
                for e in restored.entries.values()
                if e["status"] == "interrupted_no_replay"
            )
            == 30000
        )
    fresh = CostBudget()
    fresh.configure_strict(halt_on_unknown_usage=True)
    with pytest.raises(ValueError, match="restart_no_replay"):
        fresh.activate_plan(value, path=tmp_path / "ledger.json")


def test_two_owners_cannot_recreate_one_preparation_ledger(tmp_path):
    "Keep one durable owner when independent activations race."
    value = budget_plan(requests([30000]))

    def create():
        try:
            owner(value, tmp_path / "shared.json")
        except ValueError as error:
            assert "restart_no_replay" in str(error)
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: create(), range(2))) == [False, True]


@pytest.mark.parametrize("actual_per_request", [55000, 60000])
def test_only_settled_spend_transitions_to_complete_writer_plan(
    case, tmp_path, actual_per_request
):
    "Carry known fees forward only when the complete writer plan fits."
    rows = requests([80000] * 4)
    budget = owner(budget_plan(rows), tmp_path / "preparation.json")
    for row in rows:
        settle(budget, row, amount=actual_per_request)
    before = budget.snapshot()
    bundle = {
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "selection": case.selection.to_dict(),
        "extractions": [],
        "fact_cards_sha256": digest(case.cards),
        "methodology": build_methodology(case.packet["query"], {}),
    }
    plan = plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        preparation_budget_snapshot=before,
        **source_flags(),
    )
    if actual_per_request == 60000:
        assert plan["status"] == "preflight_rejected"
        assert plan["reason"] == "budget_plan_total_exceeded"
        with pytest.raises(ValueError, match="preflight_rejected"):
            budget.activate_plan(plan, path=tmp_path / "writer.json")
        assert budget.snapshot() == before
        return
    assert plan["status"] == "admitted" and plan["scope_bounds"]["shared_repairs"] == 2
    assert plan["spent_tokens"] == 220000 and plan["spent_requests"] == 4
    invalid = copy.deepcopy(plan)
    invalid["spent_tokens"] -= 1
    with pytest.raises(ValueError, match="spend_conflict"):
        budget.activate_plan(invalid, path=tmp_path / "wrong.json")
    assert budget.snapshot() == before
    budget.activate_plan(plan, path=tmp_path / "writer.json")
    assert budget.used == 220000 and budget.snapshot()["requests_sent"] == 4
    assert budget.snapshot()["plan"] == plan
    assert json.loads((tmp_path / "preparation.json").read_text()) == before


def test_legacy_phase_and_public_policy_stay_unchanged():
    "Keep the historical phase ceiling and public writer policy contract."
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    assert budget.remaining(advance=False) == 200000
    assert budget.snapshot()["schema_version"] == 1
    with pytest.raises(ValueError):
        RunOptions(budget_policy=PREPARATION_BUDGET_CONTRACT)


def test_preparation_contract_cannot_be_used_as_a_writer_policy(case):
    "Reject an internal preparation plan at the actual writer boundary."
    with pytest.raises(ValueError, match="budget_policy_invalid"):
        writer_budget_plan(
            case.packet, case.aliases, budget_policy=PREPARATION_BUDGET_CONTRACT
        )


def test_preparation_plan_preserves_an_already_settled_prefix(tmp_path):
    "Include earlier paid preparation instead of granting a fresh allowance."
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.begin_request()
    budget.record({"prompt_tokens": 1234, "completion_tokens": 0})
    rows = requests([30000, 30000])
    value = budget_plan(rows, spent_tokens=1234, spent_requests=1)
    budget.activate_plan(value, path=tmp_path / "prefix.json")
    for row in rows:
        settle(budget, row, amount=1000)
    assert budget.used == 3234 and budget.snapshot()["requests_sent"] == 3
    assert _checked_preparation_budget(budget.snapshot()).used == 3234


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["wire", "plan", "stage", "missing_path"])
async def test_changed_or_unpersisted_preparation_never_reaches_sdk(
    tmp_path, monkeypatch, fault
):
    "Reject modified requests and missing persistence before transport."
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    create = AsyncMock()
    monkeypatch.setattr(client._client.chat.completions, "create", create)
    selected, groups = strategy(client), [[attach(paper())]]
    try:
        selected.preflight_batches(
            groups,
            budget_path=None if fault == "missing_path" else tmp_path / "proof.json",
        )
        source = groups[0][0]
        if fault == "wire":
            source["_evidence_preparation"]["question"] = source[
                "_evidence_preparation"
            ]["question"].swapcase()
        elif fault == "plan":
            source["_preparation_request_scope"]["plan_identity"] = "0" * 64
        elif fault == "stage":
            source["_preparation_request_scope"]["stage"] = "preparation_1"
        with pytest.raises(ValueError, match="budget_identity_invalid"):
            await selected.extract_batch(groups[0])
        create.assert_not_called()
        assert budget.used == 0 and budget.snapshot()["requests_sent"] == 0
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "cancel"])
async def test_sdk_unknown_response_keeps_occupancy_and_cannot_restart(
    tmp_path, monkeypatch, failure
):
    "Retain uncertain occupancy after timeouts and task cancellation."
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    started = asyncio.Event()

    async def create(**kwargs):
        started.set()
        if failure == "timeout":
            raise TimeoutError("controlled preparation timeout")
        await asyncio.Event().wait()

    transport = AsyncMock(side_effect=create)
    monkeypatch.setattr(client._client.chat.completions, "create", transport)
    selected, groups = strategy(client), [[attach(paper())]]
    try:
        bounds = selected.preflight_batches(
            groups, budget_path=tmp_path / "unknown.json"
        )
        if failure == "timeout":
            with pytest.raises(TimeoutError):
                await selected.extract_batch(groups[0])
        else:
            pending = asyncio.create_task(selected.extract_batch(groups[0]))
            await asyncio.wait_for(started.wait(), 2)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        snapshot = budget.snapshot()
        assert snapshot["usage_uncertain"] and snapshot["requests_sent"] == 1
        assert (
            sum(
                e["reserved_tokens"]
                for e in snapshot["entries"].values()
                if e["status"] == "interrupted_no_replay"
            )
            == bounds[0]
        )
        with pytest.raises(ValueError):
            _checked_preparation_budget(snapshot)
        with pytest.raises(ValueError, match="no_replay"):
            await selected.extract_batch(groups[0])
        transport.assert_awaited_once()
        assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_actual_sdk_complete_queue_is_durable_and_does_not_replay(
    tmp_path, monkeypatch
):
    "Run concurrent SDK consumers within the single durable queue plan."
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    groups = []
    for i in range(12):
        p = attach(paper())
        p["paper_id"] = f"arxiv:2601.{12345 + i}"
        p["chunks"] = [
            chunk(
                "mechanism",
                "Volume rendering uses the given calibrated training views. " * 100,
                "4_volume_rendering",
                page=5,
            )
        ]
        p["chunks"][0]["paper_id"] = p["paper_id"]
        groups.append([p])
    selected = strategy(client)
    ledger_path = tmp_path / "sdk.json"
    calls = []

    async def create(**kwargs):
        calls.append(copy.deepcopy(kwargs))
        stored = json.loads(ledger_path.read_text())
        assert any(
            e["status"] == "started_no_replay" and e.get("wire_sha256")
            for e in stored["entries"].values()
        )
        await asyncio.sleep(0)
        data = json.loads(kwargs["messages"][1]["content"])
        return SimpleNamespace(
            model="offline",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "papers": {
                                    key: {"claim_records": []} for key in data["papers"]
                                }
                            }
                        ),
                        tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

    monkeypatch.setattr(
        client._client.chat.completions, "create", AsyncMock(side_effect=create)
    )
    try:
        bounds = selected.preflight_batches(groups, budget_path=ledger_path)
        assert 200000 < sum(bounds) < 450000
        assert (
            budget.snapshot()["plan"]["scope_bounds"]["purpose"] == "source_preparation"
        )
        await asyncio.gather(*(selected.extract_batch(group) for group in groups))
        assert budget.used == 240 and budget.snapshot()["requests_sent"] == 12
        assert settled_preparation_budget(planned_backend(budget)) is planned_backend(
            budget
        )
        with pytest.raises(ValueError, match="no_replay|envelope_exceeded"):
            await selected.extract_batch(groups[0])
        assert len(calls) == 12
    finally:
        await client.close()
