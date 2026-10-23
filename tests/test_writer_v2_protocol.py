"""Keep a one-call protocol observation immutable, private and separate from science."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import strict_request_reservation
from litagent.writing_v2 import (
    INSTRUCTION,
    provider_response_format,
    request_messages,
    schema_for,
)
from scripts import probe_writer_v2 as probe


@pytest.fixture
def observation(case, tmp_path, monkeypatch):
    """Freeze controlled inputs; external HTTP alone is replaced in execute tests."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config()
    cfg.observability.enabled = False
    local_schema = schema_for("body", case.aliases)
    messages = request_messages(INSTRUCTION, case.packet, local_schema)
    rf = provider_response_format("synthesis", local_schema)
    manifest = {
        "packet": case.packet,
        "aliases": case.aliases,
        "messages": messages,
        "response_format": rf,
        "local_schema": local_schema,
        "source_hashes": {},
        "limits": dict(probe.LIMITS),
        "reservation": strict_request_reservation(messages, 20000, rf),
        "config": {},
        "source_commit": "fixture",
        "runtime": {"sdk": "fixture"},
    }
    manifest["identity"] = digest(manifest)
    monkeypatch.setattr(probe, "build_manifest", lambda _c: copy.deepcopy(manifest))
    root = tmp_path / "probe"
    probe.prepare(root, cfg)
    return SimpleNamespace(root=root, cfg=cfg, manifest=manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["valid", "json", "usage", "cancel", "identity", "host_bounds"]
)
async def test_single_sdk_observation_never_becomes_scientific_acceptance(
    observation, case, monkeypatch, outcome
):
    """All real SDK exits consume at most one request, including repeat submission."""
    original = probe.OpenAICompatibleClient
    sent = []
    unused_clients = []
    traced = []
    observation.cfg.observability.enabled = True

    class Trace:
        """Expose a link only after the standalone run has created its parent."""

        trace_id = None
        trace_url = None

        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, event, data):
            traced.append((event, data))
            if event == "survey.start":
                self.trace_id = "offline-trace"
                self.trace_url = "http://offline.invalid/trace"

        def flush(self):
            pass

    monkeypatch.setattr(probe, "LangFuseTracer", Trace)

    def transport(request):
        body = json.loads(request.content)
        sent.append(body)
        assert body["messages"] == observation.manifest["messages"]
        assert body["response_format"] == observation.manifest["response_format"]
        assert body["max_tokens"] == 20000 and body["thinking"]["type"] == "disabled"
        if outcome == "cancel":
            raise asyncio.CancelledError()
        raw = json.dumps(case.body)
        if outcome == "json":
            raw = raw[:-1]
        if outcome == "host_bounds":
            raw = json.dumps({**case.body, "title": "x" * 301})
        return httpx.Response(
            200,
            headers={"x-request-id": "offline-observation"},
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": (
                    "wrong" if outcome == "identity" else probe.MODELS["generation"]
                ),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": raw},
                        "finish_reason": "stop",
                    }
                ],
                "usage": (
                    None
                    if outcome == "usage"
                    else {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    }
                ),
            },
        )

    def factory(*args, **kwargs):
        owner = original(*args, **{**kwargs, "api_key": "offline"})
        unused_clients.append(owner._client)
        owner._client = AsyncOpenAI(
            base_url="https://offline.invalid/v1",
            api_key="offline",
            max_retries=0,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        )
        return owner

    monkeypatch.setattr(probe, "OpenAICompatibleClient", factory)
    try:
        result = await probe.execute(observation.root, observation.cfg)
    finally:
        for unused in unused_clients:
            await unused.close()
    assert len(sent) == result["requests"] == 1
    assert result["status"] == (
        "protocol_completed" if outcome == "valid" else "blocked"
    )
    assert result["evaluation"] == "not_run" and not result["publishable"]
    assert result["scientific_score"] is None
    assert result["trace_url"] == "http://offline.invalid/trace"
    assert traced[0][0] == "survey.start" and traced[-1][0] == "survey.complete"
    assert traced[-1][1]["quality_status"] == "unverified"
    assert traced[-1][1]["accepted"] is False
    assert case.packet["query"] not in json.dumps(traced)
    if outcome in {"usage", "cancel"}:
        assert not result["usage_complete"] and result["unknown_hold"] > 20000
    else:
        assert result["usage_complete"] and result["known_tokens"] == 30
    if outcome != "cancel":
        wire = json.loads((observation.root / "http-response.body").read_bytes())
        private = json.loads(
            (observation.root / "request.json.writer-v2.json").read_text()
        )
        if outcome in {"valid", "json", "host_bounds"}:
            assert (
                wire["choices"][0]["message"]["content"]
                == private["calls"]["body"]["raw"]
            )
        assert (
            json.loads((observation.root / "http-request.body").read_bytes()) == sent[0]
        )
        assert (
            "Authorization" not in (observation.root / "http-metadata.json").read_text()
        )
    assert await probe.execute(observation.root, observation.cfg) == result
    assert len(sent) == 1
    public = (observation.root / "result-public.json").read_text()
    assert case.body["title"] not in public
    assert case.packet["query"] not in public
    assert "offline-observation" not in public
    if outcome == "host_bounds":
        assert result["failure"]["code"] == "writer_schema_invalid"


@pytest.mark.asyncio
async def test_started_observation_cannot_create_a_new_client(observation, monkeypatch):
    """An unknown process boundary retains the one-request reservation and blocks replay."""
    atomic_json(
        observation.root / "state.json",
        {
            "identity": observation.manifest["identity"],
            "status": "started_no_replay",
            "known_tokens": 0,
            "unknown_hold": 0,
            "requests": 0,
        },
    )

    def forbidden(*_a, **_kw):
        raise AssertionError("No new client is permitted")

    monkeypatch.setattr(probe, "OpenAICompatibleClient", forbidden)
    result = await probe.execute(observation.root, observation.cfg)
    assert result["status"] == "interrupted_no_replay"
    assert result["unknown_hold"] == observation.manifest["reservation"]
    assert result["requests"] == 1 and not result["usage_complete"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["limits", "source_commit", "runtime", "messages", "response_format"]
)
async def test_drift_is_rejected_before_any_send(observation, monkeypatch, field):
    """Rehashing a modified local file cannot widen the accepted observation."""
    path = observation.root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest[field] = {"changed": 1}
    manifest["identity"] = digest(
        {k: v for k, v in manifest.items() if k != "identity"}
    )
    atomic_json(path, manifest)
    with pytest.raises(ValueError, match="identity_conflict"):
        await probe.execute(observation.root, observation.cfg)
    assert not (observation.root / "state.json").exists()


def test_public_error_drops_echoed_material(observation):
    """A provider error may contain request data, which must stay in private state."""
    result = probe.publish(
        observation.root,
        {
            "status": "blocked",
            "failure": {
                "family": "protocol",
                "code": "HTTP 400: private full source and secret credential",
                "diagnostics": [{"raw": "private"}],
            },
        },
    )
    assert result["failure"] == {
        "family": "protocol",
        "code": "provider_or_transport_error",
    }
    assert "private" not in (observation.root / "result-public.json").read_text()


def test_prepare_does_not_overwrite_existing_observation(observation):
    """Re-preparation cannot reset the allowance or identity."""
    with pytest.raises(ValueError, match="already_exists"):
        probe.prepare(observation.root, observation.cfg)
