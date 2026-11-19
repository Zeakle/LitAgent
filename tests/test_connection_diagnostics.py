"""Keep transport causes visible without retrying unknown provider requests."""

import json

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_observability import _tracer_with_mock
from test_request_timeout_diagnostics import arguments
from test_step_timeouts import client_with_transport
from test_writer_v2 import case  # noqa: F401

from litagent.llm.client import _failure_timeout_metadata, chat_with_timeout
from litagent.observability.recorder import (
    ArchiveRepository,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.safety.budget import CostBudget
from litagent.writing_v2 import WriterError, writer_budget_plan


@pytest.mark.asyncio
@pytest.mark.parametrize("api", ["responses", "chat"])
@pytest.mark.parametrize(
    "error_class", [httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError]
)
async def test_real_sdk_connection_failure_keeps_safe_cause_and_unknown_usage(
    tmp_path, api, error_class
):
    requests = []

    async def transport(request):
        requests.append(request)
        raise error_class("PRIVATE_PROXY_SECRET / PRIVATE_PAPER", request=request)

    client, budget, events = await client_with_transport(transport)
    tracer, trace = _tracer_with_mock()
    recorder = RunRecorder("connection", "offline", ArchiveRepository(tmp_path))
    redacted = RedactingTraceHook(tracer, payload_mode="metadata_only")
    original_hook = client._trace_hook

    def hook(event, data):
        original_hook(event, data)
        recorder(event, data)
        redacted(event, data)

    client._trace_hook = hook
    try:
        with pytest.raises(APIConnectionError) as caught:
            await chat_with_timeout(
                client,
                [{"role": "user", "content": "PRIVATE_PAPER"}],
                stage="repair",
                **arguments(api),
            )
        assert isinstance(caught.value.__cause__, error_class)
        assert len(requests) == 1
        failure = next(data for event, data in events if event == "llm.failed")
        assert failure["reason_code"] == "transport_connection_error"
        assert failure["transport_error_type"] == error_class.__name__
        assert failure["request_stage"] == "repair"
        assert failure["request_timeout_seconds"] == 180
        assert "timeout_scope" not in failure
        wire = json.loads(requests[0].content)
        assert "transport_error_type" not in wire
        assert "reason_code" not in wire
        node = recorder.snapshot()["nodes"]["llm:" + failure["operation_id"]]
        assert node["metadata"]["transport_error_type"] == error_class.__name__
        assert node["usage"] is None
        update = [data for action, data in trace if action == "update"][-1]
        assert update["metadata"]["transport_error_type"] == error_class.__name__
        assert update["metadata"]["usage_complete"] is False
        assert "usage_details" not in update
        assert "PRIVATE_" not in repr(trace)
        assert not tracer._operations
        snapshot = budget.snapshot()
        assert snapshot["requests_sent"] == 1 and snapshot["usage_uncertain"]
    finally:
        await client.close()


@pytest.mark.parametrize("cause_kind", ["unknown", "cycle", "nested", "too_deep"])
def test_cause_traversal_is_bounded_and_does_not_export_exception_text(cause_kind):
    request = httpx.Request("POST", "https://offline.invalid/responses")
    error = APIConnectionError(request=request)
    inner = ValueError("PRIVATE_SECRET")
    if cause_kind == "cycle":
        inner.__cause__ = inner
    elif cause_kind == "nested":
        inner.__cause__ = httpx.RemoteProtocolError("PRIVATE_SECRET", request=request)
    elif cause_kind == "too_deep":
        current = inner
        for _ in range(12):
            current.__cause__ = ValueError("PRIVATE_SECRET")
            current = current.__cause__
        current.__cause__ = httpx.ReadError("PRIVATE_SECRET", request=request)
    error.__cause__ = inner
    metadata = _failure_timeout_metadata(error)
    assert metadata["reason_code"] == "transport_connection_error"
    assert ("transport_error_type" in metadata) == (cause_kind == "nested")
    assert "PRIVATE_" not in repr(metadata)
    assert "timeout_scope" not in metadata


def test_sdk_timeout_is_not_reclassified_as_a_connection_failure():
    request = httpx.Request("POST", "https://offline.invalid/responses")
    error = APITimeoutError(request=request)
    error.__cause__ = httpx.ReadTimeout("PRIVATE_SECRET", request=request)
    metadata = _failure_timeout_metadata(error)
    assert metadata["reason_code"] == "provider_transport_timeout"
    assert metadata["timeout_scope"] == "transport"
    assert "transport_error_type" not in metadata


@pytest.mark.asyncio
async def test_writer_disconnect_preserves_planned_hold_and_restart_never_resends(
    case, tmp_path
):
    requests = []

    async def transport(request):
        requests.append(request)
        raise httpx.RemoteProtocolError("Server disconnected", request=request)

    owner, budget, events = await owner_for(transport)
    plan = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )
    budget.activate_plan(plan, path=tmp_path / "ledger.json")
    session = make_session(owner, case, tmp_path)
    try:
        with pytest.raises(APIConnectionError):
            await session.initial_body()
        snapshot = budget.snapshot()
        assert snapshot["requests_sent"] == 1 and snapshot["usage_uncertain"]
        entry = next(iter(snapshot["entries"].values()))
        assert entry["status"] == "interrupted_no_replay"
        assert entry["actual_tokens"] is None and entry["reserved_tokens"] > 0
        assert CostBudget.from_snapshot(snapshot).snapshot() == snapshot
        restarted = make_session(owner, case, tmp_path)
        with pytest.raises((WriterError, ValueError)):
            await restarted.initial_body()
        assert len(requests) == 1 and budget.snapshot() == snapshot
        failure = next(data for event, data in events if event == "llm.failed")
        assert failure["transport_error_type"] == "RemoteProtocolError"
    finally:
        await owner.close()
