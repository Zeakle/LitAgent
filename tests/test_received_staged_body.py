"""Reuse already charged body bytes under complete lineage and bounded new calls."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_staged_body_diagnostics import repair_inputs
from test_staged_native_generation import (
    provider_body,
    provider_repair,
    provider_summaries,
    split,
)
from test_staged_protocol_probe import accounted, prepared

from litagent import staged_generation as staged
from litagent.llm import client as client_module
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.quality import save_targeted_repairs, targeted_repair_state
from litagent.report_audit import atomic_json, digest
from scripts import probe_received_body as received
from scripts import probe_staged_generation as probe


def envelope(raw, cards, scopes):
    """Represent an immutable, previously reconciled host input."""
    result = {
        "contract": staged.RECEIVED_BODY_CONTRACT,
        "raw_response": raw,
        "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "source_binding": "a" * 64,
        "source_journal_sha256": "b" * 64,
        "reference_binding_sha256": digest(
            staged.card_reference_binding(cards, scopes)
        ),
    }
    result["sha256"] = digest(result)
    return result


def sdk(monkeypatch, transport):
    """Replace only the HTTP boundary; retain actual reservations and consumers."""
    monkeypatch.setattr(
        client_module,
        "AsyncOpenAI",
        lambda **kw: AsyncOpenAI(
            **kw,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        ),
    )


def reply(value, *, usage=True, model=None):
    """Build a complete provider reply with controlled transport faults."""
    result = {
        "id": "offline-received",
        "object": "chat.completion",
        "created": 0,
        "model": model or probe.MODELS["generation"],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": value if isinstance(value, str) else json.dumps(value),
                },
            }
        ],
    }
    if usage:
        result["usage"] = {
            "prompt_tokens": 30,
            "completion_tokens": 20,
            "total_tokens": 50,
        }
    return httpx.Response(200, json=result)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", [None, "schema", "usage", "model", "cancel", "second_repair"]
)
async def test_received_sdk_skips_body_preserves_attempts_and_never_replays(
    repair_inputs, tmp_path, monkeypatch, fault
):
    x = repair_inputs
    # The immutable received-v1 envelope predates the block directory.
    raw = json.dumps(staged.restore_body_layout(x["wire"]))
    scopes = staged._writing_scopes(x["cards"], "academic_v2", "en", x["selected"])
    receipt = envelope(raw, x["cards"], scopes)
    path = tmp_path / "repair.json"
    state = targeted_repair_state(None, "original-parent")
    save_targeted_repairs(path, state)
    budget = probe.ProbeBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.bind(2)
    sent = []

    async def send(request):
        packet = json.loads(request.content)
        sent.append(packet)
        name = packet["response_format"]["json_schema"]["name"]
        assert name in {"native_body_repair_v7_refs", "grounded_summaries_v3_layout"}
        if fault == "cancel":
            raise asyncio.CancelledError()
        value = (
            provider_repair(x["response"], x["cards"])
            if name == "native_body_repair_v7_refs"
            else provider_summaries(x["summaries"], packet)
        )
        if fault == "second_repair" and len(sent) == 1:
            value = copy.deepcopy(value)
            value["patches"]["u0"]["text"] = (
                "On DeiT-S, EViT improves speed by 50%; "
                "EViT improves speed by 20% on DeiT-B."
            )
            value["patches"]["u0"]["additional_units"] = []
        return reply(
            "{" if fault == "schema" else value,
            usage=fault != "usage",
            model="wrong" if fault == "model" else None,
        )

    sdk(monkeypatch, send)
    owner = OpenAICompatibleClient(
        probe.ORIGIN,
        probe.MODELS["generation"],
        api_key="offline",
        cost_budget=budget,
        transport_max_attempts=1,
    )
    args = dict(
        cards=x["cards"],
        structure="academic_v2",
        language="en",
        selection=x["selected"],
        checkpoint_path=path,
        repair_state=state,
        received_body=receipt,
    )
    token = set_task_id("synthesis")
    try:
        if fault is None:
            candidate, proof = await staged.generate_staged_report(
                owner, x["messages"], x["rf"], **args
            )
            assert "50%" in candidate and "20%" in candidate
            assert (
                proof["received_body"]["raw_response_sha256"]
                == receipt["raw_response_sha256"]
            )
            assert len(sent) == 2 and state["attempts"][0]["status"] == "completed"
        else:
            error = {
                "schema": json.JSONDecodeError,
                "usage": ValueError,
                "model": ValueError,
                "cancel": asyncio.CancelledError,
                "second_repair": ValueError,
            }[fault]
            with pytest.raises(error):
                await staged.generate_staged_report(
                    owner, x["messages"], x["rf"], **args
                )
            assert len(sent) == (2 if fault == "second_repair" else 1)
        with pytest.raises(ValueError, match="restart_no_replay"):
            await staged.generate_staged_report(owner, x["messages"], x["rf"], **args)
        journal = json.loads(next(tmp_path.glob("*.generation-*.json")).read_text())
        assert (
            "body" not in journal["steps"] and "usage" not in journal["received_body"]
        )
        assert journal["received_body"]["raw_response"] == raw
        assert budget.snapshot()["requests_sent"] == len(sent)
        assert len(state["attempts"]) == (2 if fault == "second_repair" else 1)
        if fault == "second_repair":
            assert [a["status"] for a in state["attempts"]] == ["failed", "completed"]
            assert [p["response_format"]["json_schema"]["name"] for p in sent] == [
                "native_body_repair_v7_refs",
                "native_body_repair_v7_refs",
            ]
            assert journal["error_code"] == "staged_probe_request_limit"
    finally:
        reset_task_id(token)
        await owner.close()


@pytest.mark.parametrize("fault", ["raw", "source", "mapping", "sha", "extra"])
def test_received_envelope_rejects_tampering_before_journal(repair_inputs, fault):
    x = repair_inputs
    scopes = staged._writing_scopes(x["cards"], "academic_v2", "en", x["selected"])
    value = envelope(json.dumps(x["wire"]), x["cards"], scopes)
    if fault == "raw":
        value["raw_response"] += " "
    elif fault == "source":
        value["source_binding"] = "forged"
    elif fault == "mapping":
        value["reference_binding_sha256"] = "0" * 64
    elif fault == "extra":
        value["already_valid"] = True
    else:
        value["sha256"] = "0" * 64
    if fault != "sha":
        value["sha256"] = digest({k: v for k, v in value.items() if k != "sha256"})
    with pytest.raises(ValueError, match="received_body_binding_invalid"):
        staged.validate_received_body(
            value, staged.card_reference_binding(x["cards"], scopes)
        )


@pytest_asyncio.fixture
async def received_prepared(accounted, monkeypatch):
    """Create all three paid predecessors through actual SDK accounting."""
    prior, m, _, calls = accounted
    with monkeypatch.context() as old:
        old.setattr(probe, "CONTRACT", "native-body-summary-v3")
        old.setattr(probe, "ACCOUNTED_CONTRACT", "native-body-summary-v3")
        old.setattr(staged, "CONTRACT", "native-body-summary-v3")
        m["contract"] = probe.CONTRACT
        m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
        atomic_json(prior / "manifest.json", m)
        assert (await probe.run(prior, live=True))["status"] == "protocol_completed"
    monkeypatch.setattr(probe, "ACCOUNTED_CONTRACT", "native-body-summary-v3")
    parent = prior.parent / "parent"
    tip = prior.parent / "typed-received"
    original_generator = staged.generate_staged_report
    cards = m["input"]["cards"]
    body, summaries = split(analysis_document(cards))
    wire = provider_body(body, cards, block_layout=False)
    wire["methods"][0]["fact_card_refs"] = []
    sent = []

    async def send(request):
        sent.append(json.loads(request.content))
        return reply(wire)

    sdk(monkeypatch, send)

    async def original_unrepaired(*args, **kwargs):
        kwargs["repair_state"] = None
        return await original_generator(*args, **kwargs)

    with monkeypatch.context() as old:
        old.setattr(probe, "CONTRACT", "native-body-summary-v4")
        old.setattr(staged, "CONTRACT", "native-body-summary-v4")
        old.setattr(probe, "generate_staged_report", original_unrepaired)
        probe.prepare(tip, parent, prior)
        assert (await probe.run(tip, live=True))["status"] == "failed_no_replay"
    monkeypatch.setattr(received, "load_config", probe.load_config)
    # Only this MockTransport harness opts into the new offline contract.
    # The production v5 received-body executor remains closed to v6.
    monkeypatch.setattr(received, "EXPECTED_CONTRACT", received.CONTRACT)
    root = prior.parent / "received"
    manifest = received.prepare(root, parent, tip)
    return root, manifest, tip, sent, summaries


@pytest.mark.asyncio
async def test_received_three_hop_executor_counts_only_new_repair_summary(
    received_prepared, monkeypatch
):
    root, m, tip, old_calls, summaries = received_prepared
    protected = {p: probe.sha(Path(p)) for p in m["input"]["inputs"]}
    assert m["input"]["parent_budget"]["used_tokens"] == 350
    assert m["input"]["parent_budget"]["requests_sent"] == 6
    assert m["input"]["historical"]["model_unknown_conservative_tokens"] == 73
    original = analysis_document(m["input"]["cards"])
    patch = {
        **original["sections"]["methods"]["first_block"]["first_sentence"],
        "additional_units": [],
    }
    sent = []

    async def send(request):
        packet = json.loads(request.content)
        sent.append(packet)
        name = packet["response_format"]["json_schema"]["name"]
        assert name in {"native_body_repair_v7_refs", "grounded_summaries_v3_layout"}
        value = (
            provider_repair(
                {"can_repair": True, "patches": {"u0": patch}}, m["input"]["cards"]
            )
            if len(sent) == 1
            else provider_summaries(summaries, packet)
        )
        return reply(value)

    sdk(monkeypatch, send)
    assert (await received.run(root))["status"] == "planned" and not sent
    state = await received.run(root, live=True)
    assert state["status"] == "protocol_completed", state
    assert state["scientific_verdict"] == "not_evaluated"
    assert state["new_known_tokens"] == 100 and state["new_requests"] == 2
    assert state["budget"]["used_tokens"] == 450
    assert state["cumulative"]["model_requests"] == 8
    assert state["cumulative"]["initial_generation_rounds"] == 1
    assert await received.run(root, live=True) == state and len(sent) == 2
    assert all(probe.sha(Path(p)) == h for p, h in protected.items())
    other = root.parent / "duplicate"
    received.prepare(other, Path(m["parent"]), tip)
    with pytest.raises(FileExistsError):
        await received.run(other, live=True)
    assert len(sent) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["receipt", "checkpoint", "received", "limit", "source", "frozen"]
)
async def test_received_preflight_rejects_material_or_accounting_drift(
    received_prepared, fault
):
    root, m, tip, sent, _ = received_prepared
    if fault == "receipt":
        path = tip / "acceptance-summary.json"
        value = json.loads(path.read_text())
        value["new_known_tokens"] = 0
        atomic_json(path, value)
    elif fault == "checkpoint":
        atomic_json(
            tip / "generation.json", {"state": targeted_repair_state(None, "different")}
        )
    elif fault == "frozen":
        (root / next(iter(m["frozen_code"]))).write_text("changed")
    else:
        if fault == "received":
            m["received_body"]["raw_response"] += " "
        elif fault == "source":
            m["input"]["cards"][next(iter(m["input"]["cards"]))][
                "source_quote"
            ] = "forged"
        else:
            m["requests"] = 3
        m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
        atomic_json(root / "manifest.json", m)
    with pytest.raises(ValueError):
        await received.run(root, live=True)
    assert len(sent) == 1 and not (root / "state.json").exists()


@pytest.mark.asyncio
async def test_received_interrupted_execution_keeps_holds_and_never_resends(
    received_prepared,
):
    root, m, _, sent, _ = received_prepared
    atomic_json(
        root / "state.json",
        {
            "binding": m["binding"],
            "status": "started_no_replay",
            "scientific_verdict": "not_evaluated",
            "new_known_tokens": 50,
        },
    )
    state = await received.run(root, live=True)
    assert state["status"] == "interrupted_no_replay"
    assert state["unknown_hold"] == 119950 and state["new_requests"] == 2
    assert state["cumulative"]["model_unknown_conservative_tokens"] == 120023
    assert await received.run(root, live=True) == state
    assert len(sent) == 1
