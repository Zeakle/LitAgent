"""Verify paid extraction accounting and UI projection through the real frozen runner."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from test_frozen_generation import prepared
from test_frozen_generation_parent import child
from test_jev_citation_audit import _valid_response
from test_scope_notice_contract import notice_document as document
from test_strict_quality import draft, to_native

from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.demo import project_flow_demo_view
from litagent.llm.client import LLMResponse
from litagent.quality import digest
from scripts.run_frozen_generation import prepare, run_trial


async def trial(
    tmp_path, monkeypatch, fault=None, *, valid_writer=False, source_metadata=None
):
    """Use the real Jev protocol and Synthesis validation, with all transports offline."""
    root, cfg, old_plan, inputs, history, grant = prepared(
        tmp_path, source_metadata=source_metadata
    )
    plan = prepare(
        root, cfg, "new-jev", history, grant, quality_policy="strict_auto_v2"
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    payload = document(inputs["fact_cards"], "zh-CN")
    if not valid_writer:
        payload["sections"]["taxonomy"]["first_block"]["first_sentence"][
            "support_passage_ids"
        ] = ["r999.full"]
    client = SimpleNamespace(_cost_budget=None, _model=cfg.llm.model, close=AsyncMock())

    async def chat(*args, **kwargs):
        client._cost_budget.begin_request()
        client.hook(
            "llm.start",
            {
                "operation_id": f"new-writing-{client.chat.await_count}",
                "task_id": "adversarial_review.synthesis.r1",
                "messages": args[0],
            },
        )
        client._cost_budget.record({"prompt_tokens": 9, "completion_tokens": 1})
        client.hook(
            "llm.complete",
            {
                "operation_id": f"new-writing-{client.chat.await_count}",
                "task_id": "adversarial_review.synthesis.r1",
                "content": json.dumps(payload),
                "prompt_tokens": 9,
                "completion_tokens": 1,
                "total_tokens": 10,
                "usage_available": True,
            },
        )
        return LLMResponse(
            json.dumps(payload), usage={"prompt_tokens": 9, "completion_tokens": 1}
        )

    client.chat = AsyncMock(side_effect=chat)

    def factory(**kwargs):
        client._cost_budget = kwargs["cost_budget"]
        client.hook = kwargs["trace_hook"]
        return client

    service = SimpleNamespace(
        capabilities=lambda: {"judge_available": True, "jev_available": True},
        judge_config=lambda: SimpleNamespace(
            base_url=cfg.llm.base_url, model=cfg.llm.model
        ),
        submit=AsyncMock(),
        shutdown=AsyncMock(),
    )
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        response = _valid_response(request)
        if fault == "usage":
            response.pop("usage")
        elif fault == "probability":
            answer = next(iter(response["answers"].values()))
            answer["probabilities"] = {key: 0.33 for key in answer["probabilities"]}
        elif fault == "negative":
            for answer in response["answers"].values():
                options = list(answer["probabilities"])
                answer["choice"] = options[-1]
                answer["probabilities"] = {o: float(o == options[-1]) for o in options}
        return httpx.Response(200, json=response)

    (root / "execution-plan.json").write_text(json.dumps(plan))
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
        state = await run_trial(
            root,
            cfg,
            plan,
            live=True,
            client_factory=factory,
            service_factory=lambda *a, **kw: service,
            extraction_auditor_factory=lambda hook: JevCitationAuditor(
                client=transport, trace_hook=hook
            ),
        )
    artifact = json.loads(Path("artifacts/runs/new-jev.json").read_text())
    return root, cfg, history, grant, state, artifact, client, service, calls


@pytest.mark.asyncio
async def test_real_front_jev_usage_survives_initial_writing_failure_and_ui(
    tmp_path, monkeypatch
):
    root, cfg, history, grant, state, artifact, client, service, calls = await trial(
        tmp_path, monkeypatch
    )
    assert state["status"] == "failed" and state["report"] is None
    assert state["costs"]["survey"] == {
        "tokens": 10,
        "requests": 1,
        "usage_complete": True,
    }
    assert state["costs"]["jev"] == {
        "tokens": 100,
        "requests": 1,
        "usage_complete": True,
    }
    assert state["cumulative_after"]["jev"]["charged"] == 100
    assert len(calls) == 1 and not service.submit.called
    view = project_flow_demo_view(artifact, source="archive")
    assert view["report"]["survey"] == "" and not view["delivery"]["publishable"]
    assert len(view["pipeline_verification"]["writing_diagnostics"]) == 1
    assert view["pipeline_verification"]["stages"] == {
        "extraction_jev": "completed",
        "writing": "failed",
        "evaluation": "not_run",
        "judge": "not_run",
    }
    assert (
        view["report"]["metadata"]["extraction_verification"]
        == state["extraction_verification"]
    )
    next_root = child(root, "next")
    plan = prepare(
        next_root,
        cfg,
        "next-v2",
        history,
        grant,
        parent_root=root,
        quality_policy="strict_auto_v2",
    )
    assert plan["rounds_before"] == 2
    assert plan["cumulative_before"]["jev"] == {
        "charged": 100,
        "requests": 1,
        "usage_complete": True,
    }
    assert plan["cumulative_before"]["survey"] == {
        "charged": 110,
        "requests": 2,
        "usage_complete": True,
    }
    again = await run_trial(
        root, cfg, json.loads((root / "execution-plan.json").read_text()), live=True
    )
    assert again["costs"] == state["costs"] and len(calls) == 1


@pytest.mark.asyncio
async def test_front_negative_blocks_writer_without_erasing_paid_usage(
    tmp_path, monkeypatch
):
    *_, state, artifact, client, service, calls = await trial(
        tmp_path, monkeypatch, "negative"
    )
    assert state["reason"] == "extraction_no_supported_candidates"
    assert state["costs"]["survey"]["requests"] == 0
    assert state["costs"]["jev"]["tokens"] == 100
    client.chat.assert_not_called()
    service.submit.assert_not_called()
    view = project_flow_demo_view(artifact, source="archive")
    assert view["pipeline_verification"]["stages"]["writing"] == "not_run"
    assert view["report"]["metadata"]["extraction_verification"]["rejected"]


@pytest.mark.asyncio
async def test_unknown_front_usage_keeps_full_reservation_and_blocks_parent(
    tmp_path, monkeypatch
):
    root, cfg, history, grant, state, artifact, client, service, calls = await trial(
        tmp_path, monkeypatch, "usage"
    )
    assert state["costs"]["jev"]["usage_complete"] is False
    assert state["cumulative_after"]["jev"]["charged"] == 200000
    assert state["costs"]["survey"]["requests"] == 0
    client.chat.assert_not_called()
    with pytest.raises(ValueError):
        prepare(
            child(root, "next"),
            cfg,
            "next-unknown",
            history,
            grant,
            parent_root=root,
            quality_policy="strict_auto_v2",
        )


@pytest.mark.asyncio
async def test_changed_front_receipt_cannot_refund_parent(tmp_path, monkeypatch):
    root, cfg, history, grant, state, *_ = await trial(tmp_path, monkeypatch)
    rp = Path(state["extraction_receipt"]["path"])
    receipt = json.loads(rp.read_text())
    receipt["jev"]["usage"]["input_tokens"] = 0
    rp.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="extraction_changed"):
        prepare(
            child(root, "next"),
            cfg,
            "next-refund",
            history,
            grant,
            parent_root=root,
            quality_policy="strict_auto_v2",
        )
