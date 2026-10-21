"""Exercise frozen evidence, inherited charges and two real SDK boundaries."""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_complete_native_repair import source_selection
from test_staged_native_generation import provider_body, provider_repair, split
from test_writing_passages import facts

from litagent import staged_generation
from litagent.config import load_config
from litagent.llm import client as client_module
from litagent.quality import (
    build_fact_cards,
    grounded_messages,
    prepare_grounded_request,
)
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget
from scripts import probe_staged_generation as probe


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Construct a complete failed parent using actual evidence producers."""
    config = load_config()
    monkeypatch.setattr(probe, "load_config", lambda: config)
    monkeypatch.setattr(probe, "ARTIFACTS", tmp_path / "artifacts")
    monkeypatch.setattr(probe, "REGISTRY", tmp_path / "grants")
    monkeypatch.setenv("LLM_API_KEY", "offline-secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    selection = source_selection(facts())
    for key in ("abstract", "conclusion"):
        selection.section_evidence_ids[key] = list(selection.selected_items)
        selection.method_by_section[key] = "selected_union"
    cards = build_fact_cards(selection.selected_items)
    messages, rf = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            "en",
            "Explain EViT on DeiT-S",
            evidence_selection=selection,
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
            require_unit_citation_scope=True,
        ),
        array_wire=True,
    )
    body_messages = [dict(message) for message in messages]
    body_rf = rf
    packet = json.loads(body_messages[1]["content"])
    packet.update(
        generation_contract="native-body-summary-v2",
        generation_stage="body",
        generation_section_order=[
            "introduction",
            "taxonomy",
            "methods",
            "experiments",
            "open_problems",
        ],
    )
    body_messages[1]["content"] = json.dumps(packet)
    parent = tmp_path / "parent"
    parent.mkdir()
    archive = probe.ARTIFACTS / "runs" / "fixture-pilot-few-shot-metric.json"
    archive.parent.mkdir(parents=True)
    atomic_json(
        archive,
        {
            "run_id": "fixture-pilot-few-shot-metric",
            "events": [
                {
                    "event": "llm.start",
                    "data": {
                        "operation_id": "old",
                        "messages": body_messages,
                        "response_format": body_rf,
                    },
                },
                {
                    "event": "llm.complete",
                    "data": {
                        "operation_id": "old",
                        "usage_available": True,
                        "total_tokens": 100,
                    },
                },
            ],
            "nodes": {
                "evidence": {"kind": "evidence.retrieve", "output": selection.to_dict()}
            },
        },
    )
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    snapshot = budget.snapshot()
    snapshot.update(
        used_tokens=100,
        requests_sent=1,
        call_count=1,
        strict_phase=1,
        phase_usage=[0, 100, 0],
    )
    journal = (
        probe.ARTIFACTS
        / "targeted-repairs"
        / "fixture-pilot-few-shot-metric.json.generation-old.json"
    )
    journal.parent.mkdir()
    atomic_json(journal, {"status": "failed_no_replay", "budget": snapshot})
    history = {
        "initial_generation_rounds": 1,
        "model_requests": 1,
        "model_known_tokens": 100,
        "model_unknown_conservative_tokens": 73,
        "jev_requests": 0,
        "jev_conservative_input_tokens": 0,
        "judge_requests": 0,
        "jev_usage_complete": True,
    }
    atomic_json(
        parent / "acceptance-summary.json",
        {
            "suite_id": "fixture",
            "status": "stopped",
            "new_unknown_usage": False,
            "source_archive_sha256": probe.sha(archive),
            "new_requests_started": 1,
            "new_known_tokens": 100,
            "cumulative": history,
        },
    )
    root = tmp_path / "probe"
    manifest = probe.prepare(root, parent)
    return root, manifest, cards


@pytest.mark.parametrize(
    "fault", ["limit", "requests", "output", "root", "hashes", "source", "packet"]
)
def test_manifest_rejects_expanded_or_changed_inputs_before_transport(prepared, fault):
    root, m, _ = prepared
    if fault in {"limit", "requests", "output"}:
        m[fault] += 1
    elif fault == "root":
        m["root"] += "-copy"
    elif fault == "hashes":
        m["hashes"].pop(next(iter(m["hashes"])))
    elif fault == "source":
        m["input"]["cards"][next(iter(m["input"]["cards"]))]["source_quote"] = "changed"
    else:
        m["input"]["messages"][1]["content"] = "{}"
    m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
    atomic_json(root / "manifest.json", m)
    with pytest.raises(ValueError, match="staged_probe_identity_invalid"):
        probe.validate(root)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "body", "summary", "usage", "cancel", "model"])
async def test_sdk_calls_budget_no_replay_and_parent_preservation(
    prepared, monkeypatch, fault
):
    root, m, cards = prepared
    # Only the fake-transport harness opts into this future contract. CLI cannot.
    monkeypatch.setattr(probe, "LIVE_CONTRACT", probe.CONTRACT)
    original = {p: probe.sha(Path(p)) for p in m["input"]["inputs"]}
    body, summaries = split(analysis_document(cards))
    calls = []

    async def transport(request):
        packet = json.loads(request.content)
        calls.append(packet)
        assert packet["model"] == probe.MODELS["generation"]
        assert packet["max_tokens"] == 20000 and packet["thinking"] == {
            "type": "disabled"
        }
        assert (
            json.loads((root / "state.json").read_text())["status"]
            == "started_no_replay"
        )
        if fault == "cancel":
            raise asyncio.CancelledError()
        content = provider_body(body, cards) if len(calls) == 1 else summaries
        value = {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "wrong" if fault == "model" else probe.MODELS["generation"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "{"
                            if fault == "body" or fault == "summary" and len(calls) == 2
                            else json.dumps(content)
                        ),
                    },
                }
            ],
        }
        if fault != "usage":
            value["usage"] = {
                "prompt_tokens": 30,
                "completion_tokens": 20,
                "total_tokens": 50,
            }
        return httpx.Response(200, json=value)

    monkeypatch.setattr(
        client_module,
        "AsyncOpenAI",
        lambda **kw: AsyncOpenAI(
            **kw,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        ),
    )
    assert (await probe.run(root))["status"] == "planned" and not calls
    state = await probe.run(root, live=True)
    assert state["status"] == (
        "protocol_completed" if fault is None else "failed_no_replay"
    )
    assert len(calls) == (2 if fault in {None, "summary"} else 1)
    assert state["scientific_verdict"] == "not_evaluated"
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    assert events[-1]["event"] == "survey.complete"
    assert events[-1]["data"]["accepted"] is False
    assert events[-1]["data"]["quality_status"] == "unverified"
    assert events[-1]["data"]["total_tokens"] == state["new_known_tokens"]
    if fault is None:
        candidate = json.loads((root / "candidate.json").read_text())
        assert (
            state["local_report_sha256"]
            == hashlib.sha256(candidate["rendered"].encode()).hexdigest()
        )
    assert state["cumulative"]["initial_generation_rounds"] == 1
    assert state["cumulative"]["model_requests"] == 1 + len(calls)
    assert state["cumulative"]["model_unknown_conservative_tokens"] >= 73
    assert (state["unknown_hold"] > 0) is (fault in {"usage", "cancel"})
    assert (await probe.run(root, live=True)) == state
    assert all(probe.sha(Path(p)) == h for p, h in original.items())
    other = root.parent / "another"
    probe.prepare(other, root.parent / "parent")
    with pytest.raises(FileExistsError):
        await probe.run(other, live=True)
    assert len(calls) == state["new_requests"]


def test_local_grant_retains_inherited_budget_and_limits():
    b = probe.ProbeBudget()
    b.configure_strict(halt_on_unknown_usage=True)
    b.bind()
    with pytest.raises(ValueError, match="staged_probe_token_limit"):
        b.reserve(probe.LIMIT + 1)
    b.begin_request()
    b.begin_request()
    with pytest.raises(ValueError, match="staged_probe_request_limit"):
        b.begin_request()


@pytest.mark.asyncio
async def test_interrupted_state_is_conservative_and_never_resent(
    prepared, monkeypatch
):
    root, m, _ = prepared
    monkeypatch.setattr(probe, "LIVE_CONTRACT", probe.CONTRACT)
    atomic_json(
        root / "state.json",
        {
            "binding": m["binding"],
            "status": "started_no_replay",
            "new_known_tokens": 50,
        },
    )
    state = await probe.run(root, live=True)
    assert state["status"] == "interrupted_no_replay"
    assert state["unknown_hold"] == probe.LIMIT - 50 and state["new_requests"] == 2
    assert not (root / "events.jsonl").exists()
    clone = root.parent / "clone"
    shutil.copytree(root, clone)
    with pytest.raises(ValueError, match="identity_invalid"):
        await probe.run(clone, live=True)


@pytest.mark.asyncio
async def test_new_contract_cannot_reuse_retired_paid_budget(tmp_path):
    assert probe.CONTRACT != probe.LIVE_CONTRACT
    with pytest.raises(ValueError, match="requires_accounting_migration"):
        await probe.run(tmp_path / "not-prepared", live=True)
    assert not (tmp_path / "not-prepared").exists()


@pytest_asyncio.fixture
async def accounted(prepared, monkeypatch):
    """Record real SDK accounting, then advance that completed predecessor."""
    prior, manifest, cards = prepared
    body, summaries = split(analysis_document(cards))
    calls = []

    async def transport(request):
        packet = json.loads(request.content)
        calls.append(packet)
        content = (
            summaries
            if packet["response_format"]["json_schema"]["name"]
            == "grounded_summaries_v1"
            else provider_body(body, cards)
        )
        return httpx.Response(
            200,
            json={
                "id": "offline-lineage",
                "object": "chat.completion",
                "created": 0,
                "model": probe.MODELS["generation"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(content),
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 30,
                    "completion_tokens": 20,
                    "total_tokens": 50,
                },
            },
        )

    monkeypatch.setattr(
        client_module,
        "AsyncOpenAI",
        lambda **kw: AsyncOpenAI(
            **kw,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        ),
    )
    # Stamp the historical version only around its fake SDK execution. Its input
    # and accounting consumers are real; the production v3 guard stays enabled.
    with monkeypatch.context() as historical:
        historical.setattr(probe, "CONTRACT", probe.LIVE_CONTRACT)
        historical.setattr(staged_generation, "CONTRACT", probe.LIVE_CONTRACT)
        manifest["contract"] = probe.LIVE_CONTRACT
        manifest["binding"] = digest(
            {k: v for k, v in manifest.items() if k != "binding"}
        )
        atomic_json(prior / "manifest.json", manifest)
        state = await probe.run(prior, live=True)
        assert state["status"] == "protocol_completed" and len(calls) == 2
    root = prior.parent / "accounted"
    monkeypatch.setattr(probe, "ACCOUNTED_CONTRACT", probe.CONTRACT)
    current = probe.prepare(root, prior.parent / "parent", prior)
    return root, current, prior, calls


@pytest.mark.asyncio
async def test_accounted_sdk_uses_latest_phase_and_cumulative_without_refund(accounted):
    root, manifest, prior, calls = accounted
    assert manifest["version"] == probe.ACCOUNTED_VERSION
    assert manifest["input"]["parent_budget"]["used_tokens"] == 200
    assert manifest["input"]["parent_budget"]["requests_sent"] == 3
    assert manifest["input"]["parent_budget"]["phase_usage"] == [0, 200, 0]
    assert manifest["input"]["historical"]["model_known_tokens"] == 200
    protected = {p: probe.sha(Path(p)) for p in manifest["input"]["inputs"]}
    assert (await probe.run(root))["status"] == "planned" and len(calls) == 2
    state = await probe.run(root, live=True)
    assert state["status"] == "protocol_completed" and len(calls) == 4
    assert state["budget"]["used_tokens"] == 300
    assert state["budget"]["phase_usage"] == [0, 300, 0]
    assert state["budget"]["requests_sent"] == 5
    assert state["cumulative"]["initial_generation_rounds"] == 1
    assert state["cumulative"]["model_requests"] == 5
    assert state["cumulative"]["model_known_tokens"] == 300
    assert state["cumulative"]["model_unknown_conservative_tokens"] == 73
    assert state["scientific_verdict"] == "not_evaluated"
    assert await probe.run(root, live=True) == state
    other = root.parent / "duplicate-accounted"
    probe.prepare(other, prior.parent / "parent", prior)
    with pytest.raises(FileExistsError):
        await probe.run(other, live=True)
    assert len(calls) == 4
    assert all(probe.sha(Path(p)) == h for p, h in protected.items())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "binding",
        "input",
        "grant",
        "state",
        "unknown",
        "started",
        "requests",
        "tokens",
        "budget",
        "phase",
        "history",
        "missing_end",
        "duplicate_end",
        "wrong_id",
        "usage",
        "model",
        "negative_usage",
        "sum",
        "raw",
        "journal",
        "step_input",
        "step_budget",
    ],
)
async def test_accounting_rejects_forgery_or_incomplete_predecessor(accounted, fault):
    root, _, prior, calls = accounted
    m = json.loads((prior / "manifest.json").read_text())
    state = json.loads((prior / "state.json").read_text())
    events = [
        json.loads(line) for line in (prior / "events.jsonl").read_text().splitlines()
    ]
    if fault in {"binding", "input"}:
        if fault == "binding":
            m["binding"] = "changed"
        else:
            m["input"]["parent_budget"]["used_tokens"] -= 1
            m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
        atomic_json(prior / "manifest.json", m)
    elif fault == "grant":
        grant = probe.REGISTRY / (
            digest([m["input"]["parent_archive_sha256"], probe.LIVE_CONTRACT]) + ".json"
        )
        atomic_json(grant, {"binding": m["binding"], "root": "different"})
    elif fault == "state":
        state["new_known_tokens"] += 1
        atomic_json(prior / "state.json", state)
    elif fault in {
        "unknown",
        "started",
        "requests",
        "tokens",
        "budget",
        "phase",
        "history",
    }:
        if fault == "unknown":
            state["unknown_hold"] = 1
        elif fault == "started":
            state["status"] = "started_no_replay"
        elif fault == "requests":
            state["new_requests"] -= 1
        elif fault == "tokens":
            state["new_known_tokens"] -= 1
        elif fault == "budget":
            state["budget"]["used_tokens"] -= 1
        elif fault == "phase":
            state["budget"]["phase_usage"] = [0, 0, 200]
        else:
            state["cumulative"]["model_unknown_conservative_tokens"] = 0
        for name in ("state.json", "acceptance-summary.json"):
            atomic_json(prior / name, state)
    elif fault in {"raw", "journal", "step_input", "step_budget"}:
        path = next(prior.glob("generation.json.generation-*.json"))
        value = json.loads(path.read_text())
        if fault == "raw":
            value["steps"]["body"]["raw_response"] = "changed"
        elif fault == "step_input":
            value["steps"]["body"]["input_sha256"] = "foreign"
        elif fault == "step_budget":
            value["steps"]["body"]["budget"]["used_tokens"] -= 1
        else:
            value["steps"]["summaries"]["budget"]["used_tokens"] -= 1
        atomic_json(path, value)
    else:
        end = next(e for e in events if e["event"] == "llm.complete")
        if fault == "missing_end":
            events.remove(end)
        elif fault == "duplicate_end":
            events.append(end)
        elif fault == "wrong_id":
            end["data"]["operation_id"] = "foreign"
        elif fault == "usage":
            end["data"]["usage_available"] = False
        elif fault == "model":
            end["data"]["actual_model"] = "foreign"
        elif fault == "negative_usage":
            end["data"]["prompt_tokens"] = -1
        else:
            end["data"]["total_tokens"] += 1
        (prior / "events.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in events)
        )
    with pytest.raises(ValueError):
        await probe.run(root, live=True)
    with pytest.raises(ValueError):
        probe.prepare(root.parent / "forged", prior.parent / "parent", prior)
    assert len(calls) == 2 and not (root / "state.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing", "changed", "mapping"])
async def test_accounted_requires_readable_exact_frozen_source(accounted, fault):
    root, m, _, calls = accounted
    path = root / next(iter(m["frozen_code"]))
    if fault == "missing":
        path.unlink()
    elif fault == "changed":
        path.write_text("changed")
    else:
        m["frozen_code"].pop(next(iter(m["frozen_code"])))
        m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
        atomic_json(root / "manifest.json", m)
    with pytest.raises(ValueError, match="identity_invalid"):
        await probe.run(root, live=True)
    assert len(calls) == 2 and not (root / "state.json").exists()


@pytest.mark.asyncio
async def test_accounted_interruption_and_retired_parent_never_resend(accounted):
    root, m, prior, calls = accounted
    atomic_json(
        root / "state.json",
        {
            "binding": m["binding"],
            "status": "started_no_replay",
            "new_known_tokens": 50,
        },
    )
    result = await probe.run(root, live=True)
    assert result["status"] == "interrupted_no_replay"
    assert result["cumulative"]["model_requests"] == 5
    assert (
        result["cumulative"]["model_unknown_conservative_tokens"]
        == 73 + probe.LIMIT - 50
    )
    assert await probe.run(root, live=True) == result
    retired = root.parent / "retired"
    probe.prepare(retired, prior.parent / "parent")
    with pytest.raises(ValueError, match="requires_accounting_migration"):
        await probe.run(retired, live=True)
    assert len(calls) == 2


@pytest_asyncio.fixture
async def typed_accounted(accounted, monkeypatch):
    """Advance both historical protocol generations through the actual SDK."""
    previous, manifest, _, calls = accounted
    with monkeypatch.context() as historical:
        historical.setattr(probe, "CONTRACT", "native-body-summary-v3")
        historical.setattr(probe, "ACCOUNTED_CONTRACT", "native-body-summary-v3")
        historical.setattr(staged_generation, "CONTRACT", "native-body-summary-v3")
        manifest["contract"] = probe.CONTRACT
        manifest["binding"] = digest(
            {k: v for k, v in manifest.items() if k != "binding"}
        )
        atomic_json(previous / "manifest.json", manifest)
        assert (await probe.run(previous, live=True))["status"] == "protocol_completed"
    monkeypatch.setattr(probe, "ACCOUNTED_CONTRACT", "native-body-summary-v3")
    # Explicit MockTransport harness only; production v5 remains closed.
    monkeypatch.setattr(probe, "TYPED_CONTRACT", probe.CONTRACT)
    root = previous.parent / "typed"
    current = probe.prepare(root, previous.parent / "parent", previous)
    assert len(calls) == 4
    return root, current, previous, calls


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "repair", "usage", "json", "cancel"])
async def test_typed_grant_inherits_two_hops_and_runs_bounded_sdk(
    typed_accounted, monkeypatch, fault
):
    root, manifest, _, calls = typed_accounted
    assert manifest["version"] == probe.TYPED_VERSION
    assert manifest["requests"] == 4 and manifest["limit"] == 120000
    assert manifest["input"]["parent_budget"]["used_tokens"] == 300
    assert manifest["input"]["parent_budget"]["requests_sent"] == 5
    assert manifest["input"]["historical"]["model_unknown_conservative_tokens"] == 73
    protected = {p: probe.sha(Path(p)) for p in manifest["input"]["inputs"]}
    original = analysis_document(manifest["input"]["cards"])
    body, summaries = split(original)
    wire = provider_body(body, manifest["input"]["cards"])
    patch = {
        **original["sections"]["methods"]["first_block"]["first_sentence"],
        "additional_units": [],
    }
    if fault == "repair":
        wire["methods"][0]["fact_card_refs"] = []
    sent = []

    async def transport(request):
        packet = json.loads(request.content)
        sent.append(packet)
        name = packet["response_format"]["json_schema"]["name"]
        if fault == "cancel":
            raise asyncio.CancelledError()
        value = (
            summaries
            if name == "grounded_summaries_v1"
            else (
                {"can_repair": True, "patches": {"u0": patch}}
                if name == "native_body_repair_v7_refs"
                else wire
            )
        )
        if name == "native_body_repair_v7_refs":
            state = json.loads((root / "generation.json").read_text())["state"]
            assert len(state["attempts"]) == 1
            assert state["attempts"][0]["status"] == "started_no_replay"
            repair_packet = json.loads(packet["messages"][1]["content"])
            assert (
                repair_packet["sources"]
                == json.loads(manifest["input"]["messages"][1]["content"])["sources"]
            )
        if name == "native_body_repair_v7_refs":
            value = provider_repair(value, manifest["input"]["cards"])
        result = {
            "id": "typed-offline",
            "object": "chat.completion",
            "created": 0,
            "model": probe.MODELS["generation"],
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "{" if fault == "json" else json.dumps(value),
                    },
                }
            ],
        }
        if fault != "usage":
            result["usage"] = {
                "prompt_tokens": 30,
                "completion_tokens": 20,
                "total_tokens": 50,
            }
        return httpx.Response(200, json=result)

    monkeypatch.setattr(
        client_module,
        "AsyncOpenAI",
        lambda **kw: AsyncOpenAI(
            **kw,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        ),
    )
    assert (await probe.run(root))["body_reservation"] > 20000 and not sent
    result = await probe.run(root, live=True)
    assert result["status"] == (
        "protocol_completed" if fault in {None, "repair"} else "failed_no_replay"
    )
    assert len(sent) == (3 if fault == "repair" else 2 if fault is None else 1)
    assert result["new_requests"] == len(sent)
    assert result["cumulative"]["model_requests"] == 5 + len(sent)
    assert result["scientific_verdict"] == "not_evaluated"
    assert (result["unknown_hold"] > 0) == (fault in {"usage", "cancel"})
    assert await probe.run(root, live=True) == result
    assert all(probe.sha(Path(p)) == h for p, h in protected.items())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["old_tip", "broken_chain", "receipt", "repair_reset", "grant"]
)
async def test_typed_preflight_rejects_old_or_changed_accounting(
    typed_accounted, fault
):
    root, m, previous, calls = typed_accounted
    if fault == "old_tip":
        m["accounting_parent"] = str(previous.parent / "probe")
    elif fault == "broken_chain":
        value = json.loads((previous / "manifest.json").read_text())
        value["accounting_parent"] = str(previous)
        value["binding"] = digest({k: v for k, v in value.items() if k != "binding"})
        atomic_json(previous / "manifest.json", value)
    elif fault == "receipt":
        value = json.loads((previous / "acceptance-summary.json").read_text())
        value["budget"]["used_tokens"] -= 1
        atomic_json(previous / "acceptance-summary.json", value)
    elif fault == "repair_reset":
        m["repair_state"]["run_id"] = "other"
    else:
        path = probe.REGISTRY / (
            digest([m["input"]["parent_archive_sha256"], probe.ACCOUNTED_CONTRACT])
            + ".json"
        )
        atomic_json(path, {"binding": "foreign", "root": str(previous)})
    m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
    atomic_json(root / "manifest.json", m)
    with pytest.raises(ValueError):
        await probe.run(root, live=True)
    assert len(calls) == 4 and not (root / "state.json").exists()


def test_typed_four_calls_still_use_parent_phase_and_token_limits():
    b = probe.ProbeBudget()
    b.configure_strict(halt_on_unknown_usage=True)
    b.bind(4)
    for _ in range(4):
        b.begin_request()
    with pytest.raises(ValueError, match="request_limit"):
        b.begin_request()
    with pytest.raises(ValueError, match="token_limit"):
        b.reserve(120001)


@pytest.mark.asyncio
async def test_typed_interruption_holds_four_calls_and_never_replays(typed_accounted):
    root, m, _, calls = typed_accounted
    atomic_json(
        root / "state.json",
        {
            "binding": m["binding"],
            "status": "started_no_replay",
            "new_known_tokens": 50,
        },
    )
    result = await probe.run(root, live=True)
    assert result["status"] == "interrupted_no_replay"
    assert result["new_requests"] == 4
    assert result["unknown_hold"] == 119950
    assert result["cumulative"]["model_requests"] == 9
    assert result["cumulative"]["model_unknown_conservative_tokens"] == 73 + 119950
    assert await probe.run(root, live=True) == result
    assert len(calls) == 4 and not (root / "events.jsonl").exists()


def test_typed_repair_cannot_borrow_evaluation_reserve():
    b = probe.ProbeBudget()
    b.configure_strict(halt_on_unknown_usage=True)
    snapshot = b.snapshot()
    snapshot.update(
        used_tokens=176723,
        call_count=29,
        requests_sent=29,
        strict_phase=1,
        phase_usage=[98267, 78456, 0],
    )
    b = probe.ProbeBudget.from_snapshot(snapshot)
    b.bind(4)
    token = probe.set_task_id("synthesis.protocol_conformance")
    try:
        b.reserve(91443)
        b.release(91443)
        # Charge known usage before reserving the complete repair input.
        b.record(
            {"prompt_tokens": 18682, "completion_tokens": 3788, "usage_available": True}
        )
        with pytest.raises(ValueError, match="token_limit"):
            b.reserve(113867)
        assert b.snapshot()["phase_usage"][2] == 0
    finally:
        probe.reset_task_id(token)
