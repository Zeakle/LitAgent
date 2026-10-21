"""Exercise frozen evidence, inherited charges and two real SDK boundaries."""

import asyncio
import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_complete_native_repair import source_selection
from test_staged_native_generation import split
from test_writing_passages import facts

from litagent.config import load_config
from litagent.llm import client as client_module
from litagent.quality import (
    build_fact_cards,
    grounded_messages,
    prepare_grounded_request,
)
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget
from litagent.staged_generation import flatten_body, prepare_body_request
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
    body_messages, body_rf = prepare_body_request(messages, rf)
    packet = json.loads(body_messages[1]["content"])
    packet["generation_contract"] = "native-body-summary-v1"
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
        content = flatten_body(body) if len(calls) == 1 else summaries
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
async def test_interrupted_state_is_conservative_and_never_resent(prepared):
    root, m, _ = prepared
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
