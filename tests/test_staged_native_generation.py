"""Validate two real SDK calls, immutable partitions and durable failure boundaries."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import ValidationError as SchemaError
from openai import AsyncOpenAI
from test_body_analysis import analysis_document, render_analysis
from test_native_array_integration import request
from test_writing_passages import facts

from litagent.api import SurveyRequest
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.native_wire import array_from_native
from litagent.observability.context import reset_task_id, set_task_id
from litagent.safety.budget import CostBudget
from litagent.staged_generation import (
    generate_staged_report,
    prepare_body_request,
    prepare_summary_request,
    validate_body,
)
from litagent.workbench import RunOptions


def split(document):
    """Act only as the provider transport; production code owns all validation."""
    wire = array_from_native(document)
    body = copy.deepcopy(wire)
    summaries = {
        "sections": {k: body["sections"].pop(k) for k in ("abstract", "conclusion")}
    }
    return body, summaries


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_body_and_summary_retain_exact_sources_and_frozen_prose(language):
    cards = facts()
    messages, rf = request(cards, language, array_wire=True)
    original = json.loads(messages[1]["content"])
    body, summaries = split(analysis_document(cards, language))
    body_messages, body_format = prepare_body_request(messages, rf)
    packet = json.loads(body_messages[1]["content"])
    for key in (
        "sources",
        "cards",
        "research_question",
        "coverage_requirements",
        "analysis_plan",
        "section_source_index",
    ):
        assert packet[key] == original[key]
    assert set(
        body_format["json_schema"]["schema"]["properties"]["sections"]["properties"]
    ) == set(body["sections"])
    checked, count = validate_body(
        json.dumps(body), body_format, cards, "academic_v2", language, None
    )
    assert checked == body
    summary_messages, summary_format = prepare_summary_request(
        checked, count, messages, rf
    )
    packet = json.loads(summary_messages[1]["content"])
    assert packet["frozen_body"] == body
    for ref, source in packet["sources"].items():
        assert source == original["sources"][ref]
    assert packet["remaining_unit_capacity"] == 128 - count
    assert set(
        summary_format["json_schema"]["schema"]["properties"]["sections"]["properties"]
    ) == set(summaries["sections"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "body_json",
        "body_empty",
        "summary_json",
        "body_mutation",
        "unknown_usage",
        "cancel",
        "model",
        "budget",
        "summary_budget",
    ],
)
async def test_real_sdk_partitions_and_failures_never_replay(tmp_path, fault):
    cards = facts()
    messages, rf = request(cards, array_wire=True)
    document = analysis_document(cards)
    body, summaries = split(document)
    calls = []
    budget = CostBudget(max_tokens=1000 if fault == "budget" else 500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    if fault == "summary_budget":
        snapshot = budget.snapshot()
        snapshot.update(
            used_tokens=250000,
            call_count=1,
            strict_phase=1,
            requests_sent=1,
            phase_usage=[0, 250000, 0],
        )
        budget = CostBudget.from_snapshot(snapshot)

    async def transport(req):
        value = json.loads(req.content)
        calls.append(value)
        assert value["model"] == "pro-offline" and value["max_tokens"] == 20000
        name = value["response_format"]["json_schema"]["name"]
        assert name == (
            "grounded_body_v1" if len(calls) == 1 else "grounded_summaries_v1"
        )
        journal = next(tmp_path.glob("*.generation-*.json"))
        persisted = json.loads(journal.read_text())
        assert persisted["status"] == "started_no_replay"
        if fault == "cancel":
            raise asyncio.CancelledError()
        content = copy.deepcopy(body if len(calls) == 1 else summaries)
        if fault == "body_empty":
            content["sections"]["methods"]["blocks"] = []
        if fault == "body_mutation" and len(calls) == 2:
            content["sections"]["methods"] = body["sections"]["methods"]
        malformed = fault == "body_json" or (
            fault == "summary_json" and len(calls) == 2
        )
        result = {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "wrong" if fault == "model" else "pro-offline",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "{" if malformed else json.dumps(content),
                    },
                    "finish_reason": "stop",
                }
            ],
        }
        if fault != "unknown_usage":
            result["usage"] = {
                "prompt_tokens": 10000 if fault == "summary_budget" else 30,
                "completion_tokens": 20000 if fault == "summary_budget" else 20,
                "total_tokens": 30000 if fault == "summary_budget" else 50,
            }
        return httpx.Response(200, json=result)

    base = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "lite-offline",
        api_key="offline",
        cost_budget=budget,
        transport_max_attempts=1,
    )
    await base._client.close()
    base._client = AsyncOpenAI(
        api_key="offline",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    client = ModelBoundClient(base, "pro-offline", "generation")
    token = set_task_id("synthesis:staged")
    kwargs = dict(
        cards=cards,
        structure="academic_v2",
        language="en",
        selection=None,
        checkpoint_path=tmp_path / "run.json",
    )
    try:
        if fault:
            expected = {
                "body_json": (json.JSONDecodeError, None),
                "body_empty": (ValueError, "array_wire_shape_invalid"),
                "summary_json": (json.JSONDecodeError, None),
                "body_mutation": (SchemaError, "Additional properties"),
                "unknown_usage": (ValueError, "strict_usage_incomplete"),
                "cancel": (asyncio.CancelledError, None),
                "model": (ValueError, "stage_model_identity_mismatch"),
                "budget": (ValueError, "budget"),
                "summary_budget": (ValueError, "budget"),
            }[fault]
            with pytest.raises(expected[0], match=expected[1]):
                await generate_staged_report(client, messages, rf, **kwargs)
        else:
            candidate, proof = await generate_staged_report(
                client, messages, rf, **kwargs
            )
            assert json.loads(candidate) == document
            assert render_analysis(json.loads(candidate), cards) == render_analysis(
                document, cards
            )
            assert proof["transport"] == "host_assembled_stages"
            assert proof["scientific_verdict"] == "not_evaluated"
            assert set(proof["stage_response_sha256"]) == {"body", "summaries"}
        count = len(calls)
        assert count == (
            0
            if fault == "budget"
            else 2 if fault in {None, "summary_json", "body_mutation"} else 1
        )
        with pytest.raises(ValueError, match="staged_generation_restart_no_replay"):
            await generate_staged_report(client, messages, rf, **kwargs)
        assert len(calls) == count
        if fault in {"unknown_usage", "cancel"}:
            assert (
                budget.snapshot()["usage_uncertain"]
                and budget.snapshot()["used_tokens"] > 0
            )
        if fault is None:
            assert budget.snapshot()["requests_sent"] == 2 and budget.snapshot()[
                "phase_usage"
            ] == [0, 100, 0]
    finally:
        reset_task_id(token)
        await base.close()


def test_strategy_is_explicit_and_legacy_identity_is_unchanged(monkeypatch):
    import litagent.staged_generation as staged

    old = RunOptions()
    previous = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    fresh = previous.model_copy(update={"generation_strategy": "body_summary_v1"})
    a, b, c = old.fingerprint(), previous.fingerprint(), fresh.fingerprint()
    assert b != c
    monkeypatch.setattr(staged, "CONTRACT", "future-stage")
    assert (
        old.fingerprint() == a
        and previous.fingerprint() == b
        and fresh.fingerprint() != c
    )
    assert SurveyRequest(query="test").generation_strategy == "single_v1"
    for model, fields in [(RunOptions, {}), (SurveyRequest, {"query": "test"})]:
        with pytest.raises(
            ValueError, match="staged_generation_requires_body_analysis"
        ):
            model(**fields, generation_strategy="body_summary_v1")


@pytest.mark.asyncio
async def test_actual_human_revision_uses_two_parts_and_never_commits_body_mutation(
    tmp_path,
):
    from unittest.mock import AsyncMock

    from test_complete_native_repair import source_selection

    from litagent.agents.synthesis import SynthesisWorker
    from litagent.config import load_config
    from litagent.eval.base import CTX_EVIDENCE, CTX_SELECTED_EVIDENCE
    from litagent.runner import LitAgent

    cards = facts()
    selection = source_selection(cards)
    selection.section_evidence_ids.update(
        abstract=list(selection.selected_items),
        conclusion=list(selection.selected_items),
    )
    body, summaries = split(analysis_document(cards))
    calls = []

    def transport(req):
        packet = json.loads(req.content)
        calls.append(packet)
        assert packet["model"] == "pro-offline"
        output = copy.deepcopy(body if len(calls) == 1 else summaries)
        if len(calls) == 2:
            output["sections"]["methods"] = body["sections"]["methods"]
        return httpx.Response(
            200,
            json={
                "id": "human",
                "object": "chat.completion",
                "created": 0,
                "model": "pro-offline",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json.dumps(output)},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 10,
                    "total_tokens": 20,
                },
            },
        )

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    base = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "lite",
        api_key="offline",
        cost_budget=budget,
        transport_max_attempts=1,
    )
    await base._client.close()
    base._client = AsyncOpenAI(
        api_key="offline",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    pro = ModelBoundClient(base, "pro-offline", "generation")
    agent = LitAgent(load_config())
    agent._llm = base
    agent._stage_llms = {"generation": pro}
    agent._synthesis = SynthesisWorker(pro)
    agent._targeted_repair_state_path = tmp_path / "human.json"
    agent._evaluate = AsyncMock()
    agent._build_eval_context = lambda **kwargs: {
        CTX_EVIDENCE: selection.selected_items,
        CTX_SELECTED_EVIDENCE: list(selection.selected_items.values()),
    }
    report = {
        "survey": render_analysis(analysis_document(cards), cards),
        "metadata": {
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "generation_strategy": "body_summary_v1",
            "fact_cards": cards,
            "review_structure": "academic_v2",
            "review_methodology": {},
        },
        "quality": {"status": "failed"},
    }
    before = copy.deepcopy(report)
    try:
        returned, outcome = await agent._attempt_human_revision(
            query="Explain EViT on DeiT-S",
            report_data=report,
            results={"adversarial_review": {"evidence_selection": selection.to_dict()}},
            feedback="Clarify without changing conditions.",
            report_language="en",
        )
        assert len(calls) == 2
        assert [p["response_format"]["json_schema"]["name"] for p in calls] == [
            "grounded_body_v1",
            "grounded_summaries_v1",
        ]
        assert not outcome["committed"] and outcome["error_type"] == "ValidationError"
        assert returned == report == before
        agent._evaluate.assert_not_awaited()
        assert budget.snapshot()["phase_usage"] == [0, 0, 40]
    finally:
        await base.close()
