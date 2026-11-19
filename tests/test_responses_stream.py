"""Exercise opt-in SSE transport without changing the deployed writer contract."""

import asyncio
import base64
import copy
import hashlib
import json

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError
from test_native_responses import envelope, owner_for
from test_request_timeout_diagnostics import arguments
from test_writer_v2 import case  # noqa: F401

from litagent.llm.client import (
    LLMRequestTimeout,
    ProviderResponseError,
    chat_with_timeout,
)
from litagent.llm.responses import (
    RESPONSES_CONTRACT,
    STREAM_CONTRACT,
    ResponsesStreamDecoder,
    build_responses_request,
    decode_private_receipt,
    normalize_private_receipt,
    normalize_responses,
)
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.safety.planned_budget import request_scope
from litagent.writing_v2 import WriterError, parse_response


def frame(kind, sequence=0, *, newline="\n", **values):
    """Encode wire events, leaving field types and text unchanged."""
    event = json.dumps(
        {"type": kind, "sequence_number": sequence, **values}, ensure_ascii=False
    )
    return f"event: {kind}{newline}data: {event}{newline}{newline}".encode()


def frames(body=None, *, newline="\n"):
    """Include provisional usage and delta text which cannot become final output."""
    body = envelope('{"text":"原文[C:c1]。"}') if body is None else body
    return [
        frame(
            "response.created",
            0,
            newline=newline,
            response=envelope(
                status="in_progress",
                output=[],
                usage={"input_tokens": 999, "output_tokens": 999, "total_tokens": 1998},
            ),
        ),
        frame(
            "response.output_text.delta",
            1,
            newline=newline,
            response_id=body["id"],
            delta="PRIVATE_PARTIAL_NOT_FINAL",
        ),
        frame("response." + body["status"], 2, newline=newline, response=body),
    ]


class WireStream(httpx.AsyncByteStream):
    """Yield exact chunks and expose cancellation, early close and transport loss."""

    def __init__(self, chunks, *, fail=False, stall=False):
        self.chunks = chunks
        self.fail = fail
        self.stall = stall
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        for chunk in self.chunks:
            yield chunk
        if self.fail:
            raise httpx.RemoteProtocolError("PRIVATE_STREAM_DETAIL")
        if self.stall:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def request_args():
    return {**arguments("responses"), "responses_transport": STREAM_CONTRACT}


@pytest.mark.asyncio
@pytest.mark.parametrize("chunking", ["whole", "bytes", "events"])
@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
async def test_sdk_stream_keeps_schema_bytes_and_only_terminal_usage(chunking, newline):
    chunks = frames(newline=newline)
    received = b"".join(chunks)
    if chunking == "whole":
        chunks = [received]
    elif chunking == "bytes":
        chunks = [bytes([byte]) for byte in received]
    stream = WireStream(chunks)
    sent = []

    def transport(request):
        sent.append(request)
        return httpx.Response(
            200,
            stream=stream,
            headers={
                "content-type": "text/event-stream; charset=utf-8",
                "x-request-id": "stream-id",
                "set-cookie": "PRIVATE_COOKIE",
            },
        )

    owner, budget, events = await owner_for(transport)
    try:
        response = await owner.chat(
            [{"role": "user", "content": "原文"}], **request_args()
        )
        wire = json.loads(sent[0].content)
        assert wire["stream"] is True and wire["store"] is False
        assert (
            wire["text"]["format"]["schema"]
            == request_args()["response_format"]["json_schema"]["schema"]
        )
        assert "responses_transport" not in wire and wire["max_output_tokens"] == 100
        assert response.content == '{"text":"原文[C:c1]。"}'
        assert response.usage["prompt_tokens"] == 20
        assert response.usage["completion_tokens"] == 100
        assert response.finish_reason == "stop" and response.provider_error is None
        receipt = response.provider_receipt
        assert receipt["contract"] == STREAM_CONTRACT
        assert base64.b64decode(receipt["response_body_base64"]) == received
        assert receipt["response_sha256"] == hashlib.sha256(received).hexdigest()
        assert decode_private_receipt(receipt)[0] == wire
        assert normalize_private_receipt(receipt)["content"] == response.content
        assert "PRIVATE_COOKIE" not in repr(receipt)
        assert "PRIVATE_PARTIAL" not in repr(events) and "PRIVATE_THOUGHT" not in repr(
            events
        )
        assert len(sent) == 1 and stream.closed
        assert budget.used == 120 and not budget.snapshot()["usage_uncertain"]
        assert budget._reserved_tokens == 0
        for _ in range(2):
            normalize_private_receipt(receipt)
        assert budget.used == 120 and len(sent) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_terminal_closes_without_waiting_for_connection_eof():
    stream = WireStream(frames(), stall=True)
    owner, budget, _ = await owner_for(
        lambda _: httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )
    )
    try:
        result = await asyncio.wait_for(
            owner.chat([{"role": "user", "content": "input"}], **request_args()), 2
        )
        assert result.finish_reason == "stop" and stream.closed
        assert budget.used == 120
    finally:
        await owner.close()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_terminal",
        "bad_json",
        "duplicate_key",
        "identity",
        "model",
        "sequence",
        "bool_sequence",
        "wrong_event",
        "status",
        "duplicate_terminal",
        "truncated_frame",
        "done_without_terminal",
    ],
)
def test_stream_protocol_errors_cannot_produce_a_valid_terminal(fault):
    chunks = frames()
    if fault == "missing_terminal":
        chunks.pop()
    elif fault in {"bad_json", "duplicate_key"}:
        chunks[1] = (
            b"data: "
            + (
                b'{"type":"response.x","type":"response.y"}'
                if fault == "duplicate_key"
                else b"{"
            )
            + b"\n\n"
        )
    elif fault in {"identity", "model", "status"}:
        body = envelope(
            **{{"identity": "id", "model": "model", "status": "status"}[fault]: "wrong"}
        )
        chunks[-1] = frame("response.completed", 2, response=body)
    elif fault in {"sequence", "bool_sequence"}:
        chunks[-1] = frame(
            "response.completed",
            True if fault == "bool_sequence" else 0,
            response=envelope(),
        )
    elif fault == "wrong_event":
        chunks[-1] = chunks[-1].replace(
            b"event: response.completed", b"event: response.failed"
        )
    elif fault == "duplicate_terminal":
        chunks.append(chunks[-1])
    elif fault == "truncated_frame":
        chunks[-1] = chunks[-1][:-1]
    else:
        chunks = [b"data: [DONE]\n\n"]
    parser = ResponsesStreamDecoder()
    with pytest.raises(ValueError):
        parser.feed(b"".join(chunks))
        parser.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["eof", "disconnect", "cancel", "deadline"])
async def test_partial_stream_keeps_unknown_usage_and_never_retries(fault):
    stream = WireStream(
        frames()[:-1], fail=fault == "disconnect", stall=fault in {"cancel", "deadline"}
    )
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )

    owner, budget, events = await owner_for(transport)
    owner._request_timeouts = {"default": 0.1 if fault == "deadline" else 5}
    try:
        task = asyncio.create_task(
            chat_with_timeout(
                owner,
                [{"role": "user", "content": "original"}],
                stage="repair",
                **request_args(),
            )
        )
        if fault == "cancel":
            await stream.started.wait()
            task.cancel()
        expected = {
            "eof": ProviderResponseError,
            "disconnect": APIConnectionError,
            "cancel": asyncio.CancelledError,
            "deadline": LLMRequestTimeout,
        }[fault]
        with pytest.raises(expected) as error:
            await task
        partial = error.value.response
        assert not partial.usage["usage_available"] and partial.content == ""
        assert base64.b64decode(
            partial.provider_receipt["response_body_base64"]
        ) == b"".join(frames()[:-1])
        assert stream.closed and len(calls) == 1
        snapshot = budget.snapshot()
        assert snapshot["requests_sent"] == 1 and snapshot["usage_uncertain"]
        assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
        with pytest.raises(ValueError):
            await owner.chat(
                [{"role": "user", "content": "original"}], **request_args()
            )
        assert len(calls) == 1
        assert "PRIVATE_STREAM_DETAIL" not in repr(events)
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled"])
async def test_terminal_failure_preserves_usage_but_never_yields_valid_text(status):
    body = envelope(
        status=status,
        incomplete_details=(
            {"reason": "max_output_tokens"} if status == "incomplete" else None
        ),
    )
    stream = WireStream(frames(body))
    owner, budget, _ = await owner_for(
        lambda _: httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )
    )
    try:
        result = await owner.chat(
            [{"role": "user", "content": "input"}], **request_args()
        )
        assert result.finish_reason != "stop" and result.provider_error
        assert result.usage["usage_available"] and budget.used == 120
        assert not budget.snapshot()["usage_uncertain"]
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_stream_receipt_tampering_and_schema_failure_still_reject():
    body = envelope('{"unexpected":"field"}')
    stream = WireStream(frames(body))
    owner, _, _ = await owner_for(
        lambda _: httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )
    )
    try:
        result = await owner.chat(
            [{"role": "user", "content": "input"}], **request_args()
        )
        with pytest.raises(WriterError):
            parse_response(
                result.content,
                request_args()["response_format"]["json_schema"]["schema"],
            )
        for field, value in [
            ("response_sha256", "bad"),
            ("request_sha256", "bad"),
            ("contract", RESPONSES_CONTRACT),
        ]:
            changed = copy.deepcopy(result.provider_receipt)
            changed[field] = value
            with pytest.raises(ValueError, match="receipt_invalid"):
                decode_private_receipt(changed)
        changed = copy.deepcopy(result.provider_receipt)
        changed["headers"]["content-type"] = "application/json"
        with pytest.raises(ValueError):
            decode_private_receipt(changed)
    finally:
        await owner.close()


def test_request_mapping_preserves_old_wire_and_conservative_reservation():
    messages = [{"role": "user", "content": "完整原文"}]
    args = dict(
        model="pro",
        output_limit=20000,
        temperature=0.1,
        response_format=request_args()["response_format"],
        extra_body={"thinking": {"type": "disabled"}},
    )
    legacy = build_responses_request(messages, **args)
    stream = build_responses_request(messages, **args, transport=STREAM_CONTRACT)
    assert legacy["stream"] is False
    assert {**legacy, "stream": True} == stream
    assert len(
        json.dumps(stream, ensure_ascii=False).encode()
    ) + 20000 <= strict_request_reservation(messages, 20000, args["response_format"])
    with pytest.raises(ValueError, match="transport_invalid"):
        build_responses_request(messages, **args, transport=True)
    assert (
        normalize_responses(json.dumps(envelope()).encode())["finish_reason"] == "stop"
    )


@pytest.mark.asyncio
async def test_http_failure_remains_json_and_does_not_trigger_stream_retry():
    calls = []

    def transport(request):
        calls.append(request)
        return httpx.Response(403, json={"error": {"code": "AccountOverdueError"}})

    owner, budget, _ = await owner_for(transport)
    try:
        with pytest.raises(ProviderResponseError) as error:
            await owner.chat([{"role": "user", "content": "input"}], **request_args())
        receipt = error.value.response.provider_receipt
        assert receipt["status_code"] == 403 and len(calls) == 1
        assert (
            decode_private_receipt(receipt)[1]["error"]["code"] == "AccountOverdueError"
        )
        assert budget.snapshot()["usage_uncertain"]
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("complete", [False, True])
async def test_planned_stream_settles_once_and_restart_never_resends(
    case, tmp_path, complete
):
    from litagent.closeout import PRESET
    from litagent.config import load_config
    from litagent.eval.engine_policy import configured_engine
    from litagent.output_capacity import configured_output_tokens
    from litagent.run_limits import configured_run_limits
    from litagent.writing_v2 import writer_budget_plan

    config = load_config("config/llm-stage-repair-stream.yaml")
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        methodology={},
        capacity=config.capacity,
        run_limits=configured_run_limits(config),
        semantic_engine=configured_engine(config),
        output_tokens=configured_output_tokens(config),
        **PRESET,
    )
    stream = WireStream(frames() if complete else frames()[:-1])
    requests = []

    def transport(request):
        requests.append(request)
        return httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )

    owner, budget, _ = await owner_for(transport)
    budget.activate_plan(plan, path=tmp_path / "ledger.json")
    output_limit = plan["stages"]["body"]["output_tokens"]
    args = {**request_args(), "max_output_tokens": output_limit}
    messages = [{"role": "user", "content": "complete original material"}]
    try:
        before = budget.snapshot()
        missing_transport = {
            k: v for k, v in args.items() if k != "responses_transport"
        }
        for stage, wrong in (("body", missing_transport), ("inventory", args)):
            with request_scope(stage, "wrong-transport"):
                with pytest.raises(
                    ValueError, match="planned_responses_transport_conflict"
                ):
                    await owner.chat(messages, **wrong)
            assert budget.snapshot() == before and not requests
        with request_scope("body", "one-stream-request"):
            if complete:
                response = await owner.chat(messages, **args)
                budget.record(response.usage)
            else:
                with pytest.raises(ProviderResponseError):
                    await owner.chat(messages, **args)
        snapshot = budget.snapshot()
        item = snapshot["entries"]["one-stream-request"]
        assert item["wire_sha256"] and item["sent"]
        assert item["status"] == ("settled" if complete else "interrupted_no_replay")
        assert item["actual_tokens"] == (120 if complete else None)
        if not complete:
            assert item["reserved_tokens"] == strict_request_reservation(
                messages, output_limit, args["response_format"]
            )
        restored = CostBudget.from_snapshot(snapshot)
        assert restored.snapshot() == snapshot
        owner._cost_budget = restored
        with request_scope("body", "one-stream-request"):
            with pytest.raises(ValueError):
                await owner.chat(messages, **args)
        assert len(requests) == 1 and restored.snapshot() == snapshot
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_stream_transport_read_timeout_keeps_its_original_category():
    class TimedOut(WireStream):
        async def __aiter__(self):
            yield frames()[0]
            raise httpx.ReadTimeout("PRIVATE_TIMEOUT")

    stream = TimedOut([])
    owner, budget, events = await owner_for(
        lambda _: httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )
    )
    try:
        with pytest.raises(APITimeoutError) as caught:
            await owner.chat([{"role": "user", "content": "input"}], **request_args())
        assert caught.value.response.provider_receipt["contract"] == STREAM_CONTRACT
        failure = next(data for event, data in events if event == "llm.failed")
        assert failure["reason_code"] == "provider_transport_timeout"
        assert budget.snapshot()["usage_uncertain"] and stream.closed
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["chat", "unknown_contract"])
async def test_transport_selection_is_validated_before_reservation_or_send(fault):
    calls = []
    owner, budget, _ = await owner_for(lambda request: calls.append(request))
    args = request_args()
    if fault == "chat":
        args.update(arguments("chat"))
    else:
        args["responses_transport"] = "unknown"
    try:
        before = budget.snapshot()
        with pytest.raises(ValueError, match="responses_transport"):
            await owner.chat([{"role": "user", "content": "input"}], **args)
        assert not calls and budget.snapshot() == before
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    ["missing_usage", "string_usage", "wrong_content_type", "over_output_limit"],
)
async def test_stream_final_billing_and_output_constraints_remain_strict(fault):
    body = envelope()
    if fault == "missing_usage":
        body.pop("usage")
    elif fault == "string_usage":
        body["usage"]["input_tokens"] = "20"
    elif fault == "over_output_limit":
        body["usage"].update(output_tokens=101, total_tokens=121)
    content_type = (
        "application/json" if fault == "wrong_content_type" else "text/event-stream"
    )
    stream = WireStream(frames(body))
    owner, budget, _ = await owner_for(
        lambda _: httpx.Response(
            200, stream=stream, headers={"content-type": content_type}
        )
    )
    try:
        if fault == "over_output_limit":
            result = await owner.chat(
                [{"role": "user", "content": "input"}], **request_args()
            )
            assert result.provider_error == "responses_output_limit_exceeded"
            assert result.finish_reason == "invalid" and budget.used == 121
        else:
            with pytest.raises(ProviderResponseError) as caught:
                await owner.chat(
                    [{"role": "user", "content": "input"}], **request_args()
                )
            assert not caught.value.response.usage["usage_available"]
            assert budget.snapshot()["usage_uncertain"]
        assert budget.snapshot()["requests_sent"] == 1 and stream.closed
    finally:
        await owner.close()


def test_sse_comments_multiline_json_and_done_preserve_terminal_values():
    body = envelope('{"text":"a\\nb\\"c"}')
    data = json.dumps(
        {"type": "response.completed", "response": body}, ensure_ascii=False, indent=2
    )
    raw = (
        ": heartbeat\nid: event-id\nretry: 1000\n"
        + "\n".join("data: " + line for line in data.splitlines())
        + "\n\ndata: [DONE]\n\n"
    ).encode()
    decoder = ResponsesStreamDecoder()
    assert decoder.feed(raw)
    assert decoder.finish() == body
