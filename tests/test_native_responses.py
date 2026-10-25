"""Exercise the optional same-schema transport through SDK and durable consumers."""

import asyncio
import base64
import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from openai import APITimeoutError, AsyncOpenAI
from test_writer_v2 import BODY, EvaluatorTransport, Provider, case  # noqa: F401

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import AdversarialConfig
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.responses import normalize_responses
from litagent.llm.routing import ModelBoundClient
from litagent.orchestrator.task_graph import SubTask
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.writing_v2 import WriterError, WriterSession, schema_for


def envelope(text="{}", **changes):
    """Construct a provider envelope without asserting scientific correctness."""
    return {
        "id": "resp_offline",
        "object": "response",
        "created_at": 0,
        "model": "pro",
        "status": "completed",
        "output": [
            {
                "id": "reason",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "PRIVATE_THOUGHT"}],
            },
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            },
        ],
        "usage": {
            "input_tokens": 20,
            "output_tokens": 100,
            "total_tokens": 120,
            "output_tokens_details": {"reasoning_tokens": 70},
        },
        **changes,
    }


async def owner_for(handler, *, allowance=500000):
    """Replace only HTTP; retain SDK serialization, trace and budget ownership."""
    budget = CostBudget(allowance)
    budget.configure_strict(halt_on_unknown_usage=True)
    events = []
    owner = OpenAICompatibleClient(
        "https://offline.invalid/api/v3",
        "pro",
        api_key="offline",
        max_tokens=20000,
        cost_budget=budget,
        trace_hook=lambda e, d: events.append((e, d)),
    )
    await owner.close()

    async def checked_transport(request):
        if request.url.path.endswith("/responses"):
            catalog = json.loads(
                (
                    Path(__file__).parent
                    / "fixtures/ark-responses-fields-20261025.json"
                ).read_text()
            )
            body = json.loads(request.content)
            unknown = set(body) - set(catalog["request_fields"])
            if unknown:
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "code": "InvalidParameter",
                            "message": 'json: unknown field "'
                            + sorted(unknown)[0]
                            + '"',
                        }
                    },
                )
        value = handler(request)
        return await value if inspect.isawaitable(value) else value

    owner._client = AsyncOpenAI(
        base_url="https://offline.invalid/api/v3",
        api_key="offline",
        max_retries=0,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(checked_transport), trust_env=False
        ),
    )
    return owner, budget, events


def session_for(owner, case, tmp_path, **kwargs):
    """Use the real journal and stage-model binding for native calls."""
    return WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode="enabled",
        reasoning_effort="low",
        provider_api="responses",
        **kwargs,
    )


@pytest.mark.asyncio
async def test_sdk_preserves_complete_input_schema_cap_and_private_raw(case, tmp_path):
    """A completed wrapper still passes through the original compiler schema."""
    sent, received = [], []

    def handler(request):
        body = json.loads(request.content)
        sent.append((body, request.content))
        assert request.url.path == "/api/v3/responses"
        assert budget._reserved_tokens == strict_request_reservation(
            body["input"],
            20000,
            {
                "type": "json_schema",
                "json_schema": {
                    k: v for k, v in body["text"]["format"].items() if k != "type"
                },
            },
        )
        assert len(request.content) + 20000 <= budget._reserved_tokens
        response = httpx.Response(
            200,
            json=envelope(json.dumps(case.body, ensure_ascii=False)),
            headers={"x-request-id": "local-id", "set-cookie": "PRIVATE_COOKIE"},
        )
        received.append(response.content)
        return response

    owner, budget, events = await owner_for(handler)
    session = session_for(owner, case, tmp_path)
    schema = schema_for("body", case.aliases)
    try:
        result = await session.call(
            "body", "Preserve 原文 exactly.", case.packet, schema
        )
        assert result == case.body
        body = sent[0][0]
        assert body["input"][0]["content"].startswith("Preserve 原文 exactly.")
        assert json.loads(body["input"][1]["content"]) == case.packet
        assert body["text"]["format"]["strict"] is True
        assert (
            body["text"]["format"]["schema"]["properties"].keys()
            == schema["properties"].keys()
        )
        assert body["max_output_tokens"] == 20000
        assert (
            not {"max_tokens", "max_completion_tokens", "messages", "response_format"}
            & body.keys()
        )
        assert body["reasoning"] == {"effort": "low"} and body["thinking"] == {
            "type": "enabled"
        }
        assert body["store"] is body["stream"] is False
        assert not {"background", "truncation"} & body.keys()
        entry = session.journal["calls"]["body"]
        receipt = entry["provider_receipt"]
        assert base64.b64decode(receipt["request_body_base64"]) == sent[0][1]
        assert base64.b64decode(receipt["response_body_base64"]) == received[0]
        assert receipt["response_sha256"] == hashlib.sha256(received[0]).hexdigest()
        assert receipt["headers"].get("x-request-id") == "local-id"
        assert (
            "cookie" not in str(receipt).lower()
            and "authorization" not in str(receipt).lower()
        )
        assert entry["usage"]["reasoning_tokens"] == 70
        assert entry["returned_reasoning_characters"] is None
        assert budget.used == 120 and budget.snapshot()["requests_sent"] == 1
        assert budget._reserved_tokens == 0
        assert "PRIVATE_THOUGHT" not in str(events)
        assert "request_body_base64" not in str(events)
        with pytest.raises(WriterError, match="no_replay"):
            await session.call("body", "unused", {}, schema)
        assert len(sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "json",
        "length",
        "incomplete",
        "failed",
        "cancelled",
        "refusal",
        "unknown_output",
        "bad_role",
        "empty_output",
        "missing_model",
        "wrong_model",
        "output_over_cap",
        "http_error",
        "http_unauthorized",
        "http_rate_limit",
        "http_server_error",
        "mixed_output",
    ],
)
async def test_received_failures_keep_known_bill_once_and_never_replay(
    case, tmp_path, fault
):
    """Output validity never erases a received bill or turns it into another charge."""
    value = envelope(json.dumps(case.body))
    expected = "responses_"
    status = 200
    if fault == "json":
        value["output"][-1]["content"][0]["text"] = "{"
        expected = "writer_json_invalid"
    elif fault == "length":
        value.update(
            status="incomplete", incomplete_details={"reason": "max_output_tokens"}
        )
        expected = "writer_output_truncated"
    elif fault in {"incomplete", "failed", "cancelled", "queued"}:
        value["status"] = fault
    elif fault in {"refusal", "mixed_output"}:
        value["output"][-1]["content"].append({"type": "refusal", "refusal": "No"})
    elif fault == "unknown_output":
        value["output"].append({"type": "function_call", "name": "forbidden"})
    elif fault == "bad_role":
        value["output"][-1]["role"] = "user"
    elif fault == "empty_output":
        value["output"] = []
    elif fault == "missing_model":
        value.pop("model")
        expected = "stage_model_identity_mismatch"
    elif fault == "wrong_model":
        value["model"] = "wrong"
        expected = "stage_model_identity_mismatch"
    elif fault == "output_over_cap":
        value["usage"].update(output_tokens=20001, total_tokens=20021)
    elif fault in {
        "http_error",
        "http_unauthorized",
        "http_rate_limit",
        "http_server_error",
    }:
        status = {
            "http_error": 400,
            "http_unauthorized": 401,
            "http_rate_limit": 429,
            "http_server_error": 500,
        }[fault]
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(status, json=value)

    owner, budget, _ = await owner_for(handler)
    session = session_for(owner, case, tmp_path)
    schema = schema_for("body", case.aliases)
    try:
        with pytest.raises(ValueError, match=expected):
            await session.call("body", "untrusted input", case.packet, schema)
        assert len(sent) == budget.snapshot()["requests_sent"] == 1
        assert budget.used == value["usage"]["total_tokens"]
        assert not budget.snapshot()["usage_uncertain"] and budget._reserved_tokens == 0
        entry = session.journal["calls"]["body"]
        assert entry["status"] == "failed_no_replay" and entry["provider_receipt"]
        assert entry["usage"]["usage_available"] is True
        restarted = session_for(owner, case, tmp_path)
        with pytest.raises(WriterError, match="no_replay"):
            await restarted.call("body", "untrusted input", case.packet, schema)
        assert len(sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "missing_usage",
        "bad_primary",
        "timeout",
        "cancel",
        "malformed_envelope",
        "queued",
        "in_progress",
        "unknown_status",
    ],
)
async def test_unknown_primary_retains_full_hold_and_stops_without_retry(
    case, tmp_path, fault
):
    """An uncertain request cannot be silently refunded on cancellation or restart."""
    sent = []

    def handler(request):
        sent.append(request)
        if fault == "timeout":
            raise httpx.ReadTimeout("offline", request=request)
        if fault == "cancel":
            raise asyncio.CancelledError()
        if fault == "malformed_envelope":
            return httpx.Response(200, content=b"{")
        value = envelope(json.dumps(case.body))
        if fault in {"queued", "in_progress", "unknown_status"}:
            value["status"] = fault
            return httpx.Response(200, json=value)
        value["usage"] = (
            None
            if fault == "missing_usage"
            else {
                "input_tokens": True,
                "output_tokens": 100,
                "total_tokens": 101,
            }
        )
        return httpx.Response(200, json=value)

    owner, budget, _ = await owner_for(handler)
    session = session_for(owner, case, tmp_path)
    try:
        expected = (
            asyncio.CancelledError
            if fault == "cancel"
            else APITimeoutError if fault == "timeout" else ValueError
        )
        with pytest.raises(expected):
            await session.call(
                "body", "keep source", case.packet, schema_for("body", case.aliases)
            )
        entry = session.journal["calls"]["body"]
        assert budget.used == entry["reservation"] and budget._reserved_tokens == 0
        assert budget.stop_reason == "strict_usage_unknown"
        if fault not in {"timeout", "cancel"}:
            assert entry["provider_receipt"] and not entry["usage"]["usage_available"]
        with pytest.raises(WriterError, match="no_replay"):
            await session_for(owner, case, tmp_path).call(
                "body", "keep source", case.packet, schema_for("body", case.aliases)
            )
        assert len(sent) == budget.snapshot()["requests_sent"] == 1
    finally:
        await owner.close()


@pytest.mark.parametrize("value", [True, -1, 101, "70", 70, 0, None])
def test_raw_reasoning_types_do_not_use_sdk_coercion_or_duplicate_billing(value):
    """Optional detail is independent from complete primary billing."""
    body = envelope()
    body["usage"]["output_tokens_details"]["reasoning_tokens"] = value
    result = normalize_responses(json.dumps(body).encode())
    assert result["usage"]["usage_available"]
    assert (
        result["usage"]["prompt_tokens"] + result["usage"]["completion_tokens"] == 120
    )
    valid = type(value) is int and 0 <= value <= 100
    assert result["usage"].get("reasoning_tokens") == (value if valid else None)
    if value is not None:
        assert result["usage"]["reasoning_usage_available"] is valid


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_output_tokens": 0},
        {"max_output_tokens": True},
        {"max_output_tokens": 20000, "max_tokens": 20000},
        {"max_output_tokens": 20000, "max_completion_tokens": 20000},
        {"provider_api": "unknown"},
        {"tools": [{"type": "function"}]},
        {"extra_body": {"thinking": {"type": "enabled"}, "store": True}},
        {"response_format": {"type": "json_object"}},
        {"reasoning_effort": "high"},
        {"transport_max_attempts": 2},
        {"transport_max_attempts": True},
        {"model": "x" * 2048},
    ],
)
async def test_invalid_mapping_is_local_and_does_not_charge(case, tmp_path, kwargs):
    """No silent dropping, fallback, envelope overflow or partial output allowance."""
    sent = []
    owner, budget, events = await owner_for(lambda req: sent.append(req))
    session = session_for(owner, case, tmp_path)
    from litagent.writing_v2 import provider_response_format, request_messages

    schema = schema_for("body", case.aliases)
    messages = request_messages("source", case.packet, schema)
    options = {
        "provider_api": "responses",
        "max_output_tokens": 20000,
        "extra_body": {"thinking": {"type": "enabled"}},
        "reasoning_effort": "low",
        "response_format": provider_response_format("synthesis", schema),
        **kwargs,
    }
    before = budget.snapshot()
    try:
        with pytest.raises(ValueError):
            await owner.chat(messages, **options)
        assert budget.snapshot() == before and not sent and not events
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_chat_identity_unchanged_and_responses_cannot_resume_it(case, tmp_path):
    """The optional mapping adds identity only to the explicitly selected path."""
    from litagent.body_quality import (
        NATIVE_REPORT_INPUT_CONTRACT,
        WRITER_CONTENT_CONTRACT,
    )
    from litagent.quality import digest
    from litagent.writing_v2 import (
        CONTRACT,
        EXECUTION_CONTRACT,
        PROMPT_CONTRACT,
        PROVIDER_SCHEMA_CONTRACT,
    )

    old = WriterSession(
        Provider(case), tmp_path / "r.json", "r", case.packet, case.aliases
    )
    assert old.identity == digest(
        [
            CONTRACT,
            EXECUTION_CONTRACT,
            PROMPT_CONTRACT,
            PROVIDER_SCHEMA_CONTRACT,
            NATIVE_REPORT_INPUT_CONTRACT,
            WRITER_CONTENT_CONTRACT,
            "disabled",
            None,
            case.packet,
            case.aliases,
        ]
    )
    old.save()
    with pytest.raises(WriterError, match="identity_conflict"):
        WriterSession(
            Provider(case),
            tmp_path / "r.json",
            "r",
            case.packet,
            case.aliases,
            provider_api="responses",
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsupported,missing_records", [(False, False), (True, False), (False, True)]
)
async def test_actual_frozen_runtime_sdk_evaluation_and_shared_repair(
    case, tmp_path, monkeypatch, unsupported, missing_records
):
    """Only HTTP is replaced across writing, review, assembly, v3 and acceptance."""
    from litagent.config import load_config
    from litagent.frozen_writing import frozen_options, run_frozen
    from litagent.review_structure import build_methodology
    from litagent.runner import LitAgent, RunPolicy

    provider = Provider(case, decline=unsupported)
    evaluator = EvaluatorTransport(unsupported=unsupported)
    sent = []

    async def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        native = "input" in body
        messages = body["input"] if native else body["messages"]
        rf = (
            {
                "type": "json_schema",
                "json_schema": {
                    k: v for k, v in body["text"]["format"].items() if k != "type"
                },
            }
            if native
            else body["response_format"]
        )
        payload = json.loads(messages[1]["content"])
        is_eval = any(
            k in payload
            for k in ("allowed_citation_aliases", "cited_materials", "report_structure")
        )
        controlled = await (evaluator if is_eval else provider).chat(
            messages, response_format=rf
        )
        if native:
            assert not is_eval and body["reasoning"] == {"effort": "low"}
            response = envelope(controlled.content, model=body["model"])
            return httpx.Response(200, json=response)
        assert is_eval and request.url.path.endswith("/chat/completions")
        return httpx.Response(
            200,
            json={
                "id": "eval",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": controlled.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

    owner, _, _ = await owner_for(handler)
    config = load_config()
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens = 20000
    config.adversarial.thinking_mode = "enabled"
    config.adversarial.review_max_tokens = 4096
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def client_factory(**kwargs):
        owner._model = kwargs["model"]
        owner._cost_budget = kwargs["cost_budget"]
        owner._trace_hook = kwargs["trace_hook"]
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", client_factory)
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {"paper_id": i["paper_id"], "claims": [i["text"]], "evidence_items": [i]}
            for i in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"],
            (
                {}
                if missing_records
                else {
                    "__search_queries__": {"search_0": "relation networks"},
                    "search_0": [{"paper_id": "relation"}],
                    "dedup": [{"paper_id": "relation"}],
                    "relevance_gate": [{"paper_id": "relation"}],
                }
            ),
            source_outcomes=(
                []
                if missing_records
                else [
                    {
                        "task_id": "search_0",
                        "query": "relation networks",
                        "source": "arxiv",
                        "status": "completed",
                        "result_count": 1,
                    }
                ]
            ),
            started_at=1700000000,
        ),
    }
    try:
        result = await run_frozen(
            agent,
            bundle,
            "offline-responses",
            native_reasoning_effort="low",
            native_provider_api="responses",
        )
        assert result["metadata"]["generation_strategy"] == "body_summary_v2"
        details = result["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert result["evaluation"]["faithfulness"]["score"] == (
            0.0 if unsupported else 1.0
        )
        assert result["delivery"]["publishable"] is (
            not unsupported and not missing_records
        ), json.dumps(
            {
                "quality": result["quality"],
                "delivery": result["delivery"],
                "strict": details.get("strict_acceptance"),
                "automatic": result["metadata"].get("automatic_acceptance"),
            }
        )
        if missing_records:
            assert "method_checks" in details["strict_acceptance"]["failure_codes"]
        assert result["metadata"]["automatic_acceptance"]["actor"] == "system"
        assert [b["model"] for b in sent[:7]] == ["pro"] * 6 + ["lite"]
        assert all(b["max_output_tokens"] == 20000 for b in sent[:6])
        assert sent[6]["max_output_tokens"] == 4096
        assert agent._cost_budget.used == 120 * len(sent)
        journal = json.loads(
            Path(str(agent._targeted_repair_state_path) + ".writer-v2.json").read_text()
        )
        assert journal["generation_provider_api"] == "responses"
        if unsupported:
            assert len(journal["candidates"]) <= 2
            repair_calls = [
                e for k, e in journal["calls"].items() if k.startswith("repair_")
            ]
            assert repair_calls and all(
                e["provider_api"] == "responses" for e in repair_calls
            )
        assert all(
            e["provider_receipt"]
            for e in journal["calls"].values()
            if e["status"] == "response_received"
        )
    finally:
        await agent.cleanup()
        await owner.close()


def test_responses_metadata_redacted_but_safe_status_reaches_langfuse():
    """Private receipt, reasoning and raw payloads never enter the external trace."""
    from litagent.observability.recorder import RedactingTraceHook
    from litagent.observability.tracing import LangFuseTracer

    tracer = LangFuseTracer(host="", public_key="", secret_key="")
    tracer._client, tracer._root = object(), MagicMock()
    hook = RedactingTraceHook(tracer, payload_mode="metadata_only")
    data = {
        "operation_id": "offline",
        "model": "pro",
        "provider_api": "responses",
        "provider_status": "completed",
        "provider_error": None,
        "output_limit_parameter": "max_output_tokens",
        "max_tokens": 20000,
        "messages": ["PRIVATE"],
        "content": "PRIVATE",
        "provider_receipt": {"body": "PRIVATE"},
        "reasoning_content": "PRIVATE",
        "usage_available": True,
        "prompt_tokens": 20,
        "completion_tokens": 100,
        "total_tokens": 120,
    }
    hook("llm.start", data)
    hook("llm.complete", data)
    generation = tracer._root.start_observation.return_value
    metadata = generation.update.call_args.kwargs["metadata"]
    assert metadata["provider_api"] == "responses"
    assert metadata["provider_status"] == "completed"
    assert metadata["output_limit_parameter"] == "max_output_tokens"
    assert "PRIVATE" not in str(tracer._root.mock_calls)


@pytest.mark.parametrize(
    "changes",
    [
        {"object": "chat.completion"},
        {"id": ""},
        {"id": None},
        {"store": True},
        {"store": 0},
        {"background": True},
        {"truncation": "auto"},
        {"error": {"code": "failure"}},
        {"incomplete_details": {"reason": "limit"}},
    ],
)
def test_conflicting_envelope_keeps_bill_but_cannot_be_completed(changes):
    """Echoed settings and incompatible wrappers cannot masquerade as completion."""
    result = normalize_responses(json.dumps(envelope(**changes)).encode())
    assert result["provider_error"] and result["finish_reason"] != "stop"
    assert (
        result["usage"]["usage_available"]
        and result["usage"]["completion_tokens"] == 100
    )


@pytest.mark.asyncio
async def test_reviewer_can_choose_chat_without_reinterpreting_generation(
    case, tmp_path
):
    """Reviewer API and thinking mode are explicit, while the journal keeps Pro's identity."""
    provider, sent = Provider(case), []

    async def handler(request):
        body = json.loads(request.content)
        sent.append(body)
        native = "input" in body
        rf = (
            {
                "type": "json_schema",
                "json_schema": {
                    k: v for k, v in body["text"]["format"].items() if k != "type"
                },
            }
            if native
            else body["response_format"]
        )
        value = await provider.chat(
            body["input"] if native else body["messages"], response_format=rf
        )
        if native:
            return httpx.Response(
                200, json=envelope(value.content, model=body["model"])
            )
        return httpx.Response(
            200,
            json={
                "id": "review",
                "object": "chat.completion",
                "created": 0,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": value.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

    owner, budget, _ = await owner_for(handler)
    pro = ModelBoundClient(owner, "pro", "generation")
    lite = ModelBoundClient(owner, "lite", "evaluation")
    synthesis = SynthesisWorker(
        pro,
        evidence_selector=SimpleNamespace(
            select=AsyncMock(return_value=case.selection)
        ),
        thinking_mode="enabled",
        native_reasoning_effort="low",
        native_provider_api="responses",
        run_id_provider=lambda: "r",
        repair_state_path_provider=lambda: tmp_path / "r.json",
    )
    reviewer = ReviewerWorker(lite, config=AdversarialConfig(thinking_mode="disabled"))
    task = SubTask(
        "adversarial_review",
        "Offline mixed mapping",
        "adversarial_review",
        input_data={
            "query": case.packet["query"],
            "report_language": "en",
            "review_structure": "academic_v2",
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "generation_strategy": "body_summary_v2",
            "verification": {"judge": False, "jev": False},
            "upstream_results": {},
        },
    )
    try:
        result = await AdversarialReviewWorker(
            pro, synthesis, reviewer, thinking_mode="enabled"
        ).execute(task)
        assert len(sent) == 7 and all("input" in b for b in sent[:6])
        assert "messages" in sent[-1] and sent[-1]["model"] == "lite"
        assert sent[-1]["max_tokens"] == 4096
        assert "reasoning_effort" not in sent[-1]
        assert sent[-1]["thinking"] == {"type": "disabled"}
        assert result["writer_document"]["generation_provider_api"] == "responses"
        journal = json.loads((tmp_path / "r.json.writer-v2.json").read_text())
        assert journal["calls"]["reviewer"].get("provider_api", "chat") == "chat"
        assert journal["calls"]["reviewer"]["output_limit_parameter"] == "max_tokens"
        assert budget.used == 840
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_response_mapping_version_rejects_an_old_checkpoint(
    case, tmp_path, monkeypatch
):
    """The business schema stays fixed while v1 and v2 transport identities differ."""
    from litagent.llm import responses as mapping

    monkeypatch.setattr(mapping, "RESPONSES_CONTRACT", "ark-responses-mapping-v1")
    old = WriterSession(
        Provider(case),
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        provider_api="responses",
    )
    old.save()
    before = old.journal_path.read_bytes()
    monkeypatch.setattr(mapping, "RESPONSES_CONTRACT", "ark-responses-mapping-v2")
    with pytest.raises(WriterError, match="identity_conflict"):
        WriterSession(
            Provider(case),
            tmp_path / "r.json",
            "r",
            case.packet,
            case.aliases,
            provider_api="responses",
        )
    assert old.journal_path.read_bytes() == before
