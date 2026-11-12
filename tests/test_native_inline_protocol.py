"""Validate one actual-input protocol grant through real SDK and source consumers."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_native_array_integration import request
from test_native_array_protocol import prepared
from test_writing_passages import facts

from litagent.llm import client as client_module
from litagent.report_audit import atomic_json, digest
from scripts import probe_native_array_wire as probe


@pytest.fixture
def actual(prepared):
    """Keep complete real producer fields while replacing only external transport."""
    old_root, old = prepared
    cards = facts()
    messages, rf = request(cards, array_wire=True)
    path = Path(old["archive"])
    archive = json.loads(path.read_text())
    archive["events"][0]["data"].update(messages=messages, response_format=rf)
    card = next(iter(cards.values()))
    archive["nodes"] = {
        "evidence.retrieve:x": {
            "kind": "evidence.retrieve",
            "output": {
                "selected_items": {
                    "a": {
                        "evidence_id": "a",
                        "paper_id": "evit",
                        "text": card["source_quote"],
                        "supporting_text": card["source_quote"],
                        "chunk_key": "abstract",
                    }
                }
            },
        }
    }
    atomic_json(path, archive)
    receipt = Path(old["parent"])
    parent = json.loads(receipt.read_text())
    parent["source_archive_sha256"] = probe.sha(path)
    atomic_json(receipt, parent)
    root = old_root.parent / "actual"
    manifest = probe.prepare_protocol(
        root, receipt.parent, scenario=probe.REAL_CONTRACT
    )
    return root, manifest, cards, messages


def test_actual_input_preserves_sources_permissions_and_parent_allowance(actual):
    root, m, cards, messages = actual
    assert m["max_tokens"] == 120000 and m["frame"]["max_tokens"] == 20000
    assert m["frame"]["reservation"] <= min(120000, 117964)
    before = json.loads(messages[1]["content"])
    after = json.loads(m["frame"]["messages"][1]["content"])
    for key in (
        "cards",
        "sources",
        "research_question",
        "coverage_requirements",
        "section_source_index",
    ):
        assert after[key] == before[key]
    assert probe.validate_manifest(root) == m
    assert set(probe._real_cards(m)) == set(cards)
    assert m["parent_tokens"] + m["max_tokens"] <= 500000
    assert m["parent_requests"] + 1 <= 64


@pytest.mark.parametrize("fault", ["cap", "output", "input", "model", "root"])
def test_rebound_manifest_does_not_expand_grant(actual, fault):
    root, m, _, _ = actual
    if fault == "cap":
        m["max_tokens"] += 1
    elif fault == "output":
        m["frame"]["max_tokens"] -= 1
    elif fault == "input":
        m["frame"]["messages"][0]["content"] = "remove original conditions"
        m["frame"]["reservation"] = probe.strict_request_reservation(
            m["frame"]["messages"], 20000, m["frame"]["response_format"]
        )
    elif fault == "model":
        m["model"] = "another-model"
    else:
        m["root"] = str(root.parent)
    m["binding"] = digest({k: v for k, v in m.items() if k != "binding"})
    atomic_json(root / "probe-manifest.json", m)
    with pytest.raises(ValueError):
        probe.validate_manifest(root)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", [None, "json", "schema", "reference", "model", "usage", "cancel"]
)
async def test_actual_sdk_full_compile_one_request_and_no_replay(
    actual, monkeypatch, fault
):
    root, m, cards, _ = actual
    monkeypatch.setenv("LLM_API_KEY", "offline-secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    calls = []

    async def transport(req):
        body = json.loads(req.content)
        calls.append(body)
        assert body["messages"] == m["frame"]["messages"]
        assert (
            json.loads((root / "probe-state.json").read_text())["unknown_hold"]
            == 120000
        )
        if fault == "cancel":
            raise asyncio.CancelledError()
        doc = probe.array_from_native(analysis_document(cards))
        if fault == "schema":
            doc["sections"].pop("methods")
        elif fault == "reference":
            doc["sections"]["abstract"]["blocks"][0]["sentences"][0]["body_references"][
                0
            ]["sentence_index"] = 999
        value = {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "wrong" if fault == "model" else probe.MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "{" if fault == "json" else json.dumps(doc),
                    },
                    "finish_reason": "stop",
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
    result = await probe.run_protocol(root, live=True)
    assert len(calls) == 1
    assert result["status"] == (
        "completed"
        if fault is None
        else (
            "failed_no_replay"
            if fault == "usage"
            else "interrupted_no_replay" if fault == "cancel" else "protocol_failed"
        )
    )
    receipt = json.loads((root / "acceptance-summary.json").read_text())
    assert receipt["new_survey_rounds"] == 0
    assert receipt["scientific_verdict"] == "not_applicable"
    if fault in {"usage", "cancel"}:
        assert receipt["new_unknown_hold"] == 120000
    assert (await probe.run_protocol(root, live=True))["status"] == result["status"]
    assert len(calls) == 1
    other = root.parent / "other"
    probe.prepare_protocol(
        other, Path(m["parent"]).parent, scenario=probe.REAL_CONTRACT
    )
    with pytest.raises(ValueError, match="already_owned"):
        await probe.run_protocol(other, live=True)
    assert len(calls) == 1
