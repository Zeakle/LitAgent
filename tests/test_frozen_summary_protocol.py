"""Exercise lossless input tables and one fully accounted summary-only call."""

import asyncio
import copy
import json
from pathlib import Path

import pytest
import pytest_asyncio
from test_body_analysis import analysis_document
from test_received_staged_body import received_prepared, reply, sdk
from test_staged_native_generation import provider_repair, provider_summaries
from test_staged_protocol_probe import accounted, prepared
from test_summary_reference_wire import parts

from litagent import staged_generation as staged
from litagent.report_audit import atomic_json, digest
from scripts import probe_frozen_summary as frozen
from scripts import probe_received_body as received
from scripts import probe_staged_generation as probe


def packet_for(parts):
    """Prepare complete controlled materials with the actual v7 producer."""
    body, _, messages, rf, binding = parts
    request, _ = staged.prepare_summary_request(
        body,
        binding["body_unit_count"],
        messages,
        rf,
        short_refs=True,
        compact_body=True,
    )
    return json.loads(request[1]["content"])


def test_compact_table_restores_every_field_and_original_source_record(parts):
    body, _, messages, rf, binding = parts
    old, _ = staged.prepare_summary_request(
        body, binding["body_unit_count"], messages, rf, short_refs=True
    )
    original = json.loads(old[1]["content"])
    # Reverse only explicitly encoded card identity slots for equality.
    original["frozen_body"] = staged._short_reference_fields(
        original["frozen_body"], binding["reading_cards"]
    )
    original["cards"] = {
        cid: original["cards"][ref] for ref, cid in binding["reading_cards"].items()
    }
    packet = packet_for(parts)
    assert "body_unit_index" not in packet
    assert staged.restore_summary_input(packet, binding, body=body) == original
    assert packet["sources"] == original["sources"]
    assert packet["frozen_body"]["columns"] == staged.SUMMARY_COLUMNS


@pytest.mark.parametrize(
    "fault",
    [
        "columns",
        "width",
        "position",
        "boolean",
        "reference",
        "duplicate",
        "text",
        "missing",
        "kind",
    ],
)
def test_compact_table_cannot_hide_drift_or_guess_locations(parts, fault):
    body, _, _, _, binding = parts
    packet = packet_for(parts)
    table = packet["frozen_body"]
    rows = table["rows"]["methods"]
    row = rows[0]
    if fault == "columns":
        table["columns"].reverse()
    elif fault == "width":
        row.pop()
    elif fault == "position":
        row[2] += 1
    elif fault == "boolean":
        row[1] = False
    elif fault == "reference":
        row[0] = "b999"
    elif fault == "duplicate":
        rows.append(copy.deepcopy(row))
    elif fault == "text":
        row[7] += " altered"
    elif fault == "missing":
        table["rows"]["methods"] = []
    else:
        row[5] = "organization"
    with pytest.raises(ValueError):
        staged.restore_summary_input(packet, binding, body=body)


@pytest_asyncio.fixture
async def summary_prepared(received_prepared, monkeypatch):
    """Produce the historical v5 failed summary through the real SDK consumers."""
    tip, m, _, _, summaries = received_prepared
    original = analysis_document(m["input"]["cards"])
    patch = {
        **original["sections"]["methods"]["first_block"]["first_sentence"],
        "additional_units": [],
    }
    sent = []

    async def send(request):
        p = json.loads(request.content)
        sent.append(p)
        return reply(
            provider_repair(
                {"can_repair": True, "patches": {"u0": patch}}, m["input"]["cards"]
            )
            if len(sent) == 1
            else "{"
        )

    sdk(monkeypatch, send)
    with monkeypatch.context() as old:
        old.setattr(probe, "CONTRACT", "native-body-summary-v5")
        old.setattr(received, "CONTRACT", "native-body-summary-v5")
        old.setattr(received, "EXPECTED_CONTRACT", "native-body-summary-v5")
        old.setattr(staged, "CONTRACT", "native-body-summary-v5")
        m["contract"] = "native-body-summary-v5"
        m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
        atomic_json(tip / "manifest.json", m)
        result = await received.run(tip, live=True)
        assert result["new_requests"] == 2 and result["status"] == "failed_no_replay"
    # This fixture verifies the historical v7 executor, whose grant stays closed
    # to the new output contract until its own accounting migration is implemented.
    monkeypatch.setattr(staged, "CONTRACT", "native-body-summary-v7")
    monkeypatch.setattr(probe, "CONTRACT", "native-body-summary-v7")
    monkeypatch.setattr(frozen, "CONTRACT", "native-body-summary-v7")
    monkeypatch.setattr(frozen, "load_config", probe.load_config)
    root = tip.parent / "frozen-summary"
    manifest = frozen.prepare(root, Path(m["parent"]), tip)
    return root, manifest, tip, sent, summaries


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "json", "usage", "model", "cancel"])
async def test_single_sdk_summary_no_body_no_repair_and_no_replay(
    summary_prepared, monkeypatch, fault
):
    root, m, tip, _, summaries = summary_prepared
    protect = {p: probe.sha(Path(p)) for p in m["input"]["inputs"]}
    sent = []

    async def send(request):
        p = json.loads(request.content)
        sent.append(p)
        assert (
            p["response_format"]["json_schema"]["name"] == "grounded_summaries_v2_refs"
        )
        assert (
            json.loads(p["messages"][1]["content"])["summary_input_contract"]
            == staged.SUMMARY_TABLE_CONTRACT
        )
        if fault == "cancel":
            raise asyncio.CancelledError()
        return reply(
            "{" if fault == "json" else provider_summaries(summaries, p),
            usage=fault != "usage",
            model="wrong" if fault == "model" else None,
        )

    sdk(monkeypatch, send)
    plan = await frozen.run(root)
    assert plan["status"] == "planned" and not sent
    before = m["input"]["parent_budget"]
    assert before["used_tokens"] == 450 and before["requests_sent"] == 8
    state = await frozen.run(root, live=True)
    assert state["status"] == (
        "protocol_completed" if fault is None else "failed_no_replay"
    ), state
    assert state["new_requests"] == 1 and len(sent) == 1
    assert state["scientific_verdict"] == "not_evaluated"
    assert state["cumulative"]["initial_generation_rounds"] == 1
    assert state["cumulative"]["model_requests"] == 9
    assert (state["unknown_hold"] > 0) == (fault in {"usage", "cancel"})
    journal = json.loads(
        next(root.glob("generation.json.generation-*.json")).read_text()
    )
    assert list(journal["steps"]) == ["summaries"] and "body" not in journal["steps"]
    assert "raw_response" not in journal["frozen_body"]
    repairs = json.loads((root / "generation.json").read_text())["state"]
    assert repairs == m["repair_state"] and len(repairs["attempts"]) == 1
    assert await frozen.run(root, live=True) == state and len(sent) == 1
    assert all(probe.sha(Path(p)) == h for p, h in protect.items())
    other = root.parent / "duplicate-summary"
    frozen.prepare(other, Path(m["parent"]), tip)
    with pytest.raises(FileExistsError):
        await frozen.run(other, live=True)
    assert len(sent) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["limit", "body", "receipt", "checkpoint", "proof", "frozen_source"]
)
async def test_summary_only_rejects_changed_inputs_before_paid_transport(
    summary_prepared, fault
):
    root, m, tip, sent, _ = summary_prepared
    if fault == "receipt":
        path = tip / "acceptance-summary.json"
        v = json.loads(path.read_text())
        v["new_known_tokens"] = 0
        atomic_json(path, v)
    elif fault in {"checkpoint", "proof"}:
        path = tip / "generation.json"
        v = json.loads(path.read_text())
        if fault == "checkpoint":
            v["state"]["attempts"] = []
        else:
            v["staged_body_attempt_0"]["applied_candidate"] = "{}"
        atomic_json(path, v)
    elif fault == "frozen_source":
        (root / next(iter(m["frozen_code"]))).write_text("changed")
    else:
        if fault == "limit":
            m["requests"] = 2
        else:
            m["frozen_body"]["body"]["title"] += " changed"
        m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
        atomic_json(root / "manifest.json", m)
    with pytest.raises(ValueError):
        await frozen.run(root, live=True)
    assert len(sent) == 2 and not (root / "state.json").exists()


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_table_preserves_literal_alias_words_in_prose_and_all_positions(language):
    from test_native_array_integration import request
    from test_staged_native_generation import split
    from test_writing_passages import facts

    cards = facts()
    body, _ = split(analysis_document(cards, language))
    body["sections"]["methods"]["blocks"][0]["sentences"][0][
        "text"
    ] += " c1 b1 fact_card_ids body_ref"
    binding = staged.summary_reference_binding(body)
    messages, rf = request(cards, language, array_wire=True)
    prepared, _ = staged.prepare_summary_request(
        body,
        binding["body_unit_count"],
        messages,
        rf,
        short_refs=True,
        compact_body=True,
    )
    restored = staged.restore_summary_input(
        json.loads(prepared[1]["content"]), binding, body=body
    )
    assert restored["frozen_body"] == body


@pytest.mark.asyncio
async def test_frozen_summary_interruption_keeps_unknown_hold_and_never_resends(
    summary_prepared,
):
    root, m, _, sent, _ = summary_prepared
    atomic_json(
        root / "state.json",
        {
            "binding": m["binding"],
            "status": "started_no_replay",
            "scientific_verdict": "not_evaluated",
        },
    )
    state = await frozen.run(root, live=True)
    assert state["status"] == "interrupted_no_replay"
    assert state["new_requests"] == 1 and state["unknown_hold"] == 120000
    assert await frozen.run(root, live=True) == state
    assert len(sent) == 2


def test_summary_full_reservation_boundary_never_borrows_evaluation(parts):
    from litagent.observability.context import reset_task_id, set_task_id
    from litagent.safety.budget import CostBudget, strict_request_reservation

    body, _, messages, rf, binding = parts
    prepared, schema = staged.prepare_summary_request(
        body,
        binding["body_unit_count"],
        messages,
        rf,
        short_refs=True,
        compact_body=True,
    )
    amount = strict_request_reservation(prepared, 20000, schema)
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    snapshot = budget.snapshot()
    used = 300000 - amount
    snapshot.update(
        used_tokens=used,
        strict_phase=1,
        phase_usage=[98267, used - 98267, 0],
        requests_sent=32,
        call_count=32,
    )
    budget = CostBudget.from_snapshot(snapshot)
    token = set_task_id("synthesis.summary")
    try:
        budget.reserve(amount)
        budget.release(amount)
        assert budget.snapshot() == snapshot
        with pytest.raises(ValueError, match="budget_exceeded"):
            budget.reserve(amount + 1)
        assert budget.snapshot() == snapshot
    finally:
        reset_task_id(token)
