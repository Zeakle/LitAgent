"""Preserve known billing when Jev rejects a response contract."""

import json

import httpx
import pytest
from test_jev_citation_audit import _ledger, _multi_snapshot, _snapshot, _valid_response

from litagent.benchmark.citation_audit import JevAuditConfig, JevCitationAuditor


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["probability", "model", "missing_answer"])
async def test_invalid_answer_keeps_known_usage_without_decisions(monkeypatch, failure):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    calls = []

    async def handler(request):
        calls.append(request)
        payload = _valid_response(request)
        qid = next(iter(payload["answers"]))
        if failure == "probability":
            payload["answers"][qid]["probabilities"] = {
                key: 0.33 for key in payload["answers"][qid]["probabilities"]
            }
        elif failure == "model":
            payload["model"] = "unexpected-model"
        else:
            del payload["answers"][qid]
        payload["usage"] = {"input_tokens": 6396, "output_tokens": 485}
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JevCitationAuditor(client=client).evaluate(
            _snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True),
            artifact={"run_id": "bad"},
        )
    assert len(calls) == 1 and result.status == "failed" and result.decisions == []
    assert result.usage.usage_complete
    assert result.usage.input_tokens == 6396 and result.usage.output_tokens == 485
    if failure == "probability":
        assert "jev_probability_invalid" in result.reason_codes
        diagnostic = result.batches[0].validation_diagnostics[0]
        assert diagnostic["question_id"] == next(
            iter(json.loads(calls[0].content)["questions"])
        )
        assert diagnostic["probability_sum"] == pytest.approx(0.99)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"input_tokens": -1, "output_tokens": 5},
        {"input_tokens": True, "output_tokens": 5},
        {"input_tokens": "5", "output_tokens": 5},
    ],
)
async def test_bad_usage_stays_unknown_on_invalid_response(monkeypatch, usage):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")

    async def handler(request):
        payload = _valid_response(request)
        payload["model"] = "wrong"
        payload["usage"] = usage
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JevCitationAuditor(client=client).evaluate(
            _snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True),
            artifact={"run_id": "bad-usage"},
        )
    assert not result.usage.usage_complete
    assert result.usage.input_tokens is None and result.usage.call_count == 1
    assert result.decisions == []


@pytest.mark.asyncio
async def test_failed_wave_preserves_both_returned_usages_and_stops_dispatch(
    monkeypatch,
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    calls = []

    async def handler(request):
        calls.append(request)
        index = len(calls)
        payload = _valid_response(request)
        if index == 1:
            answer = next(iter(payload["answers"].values()))
            answer["probabilities"] = {key: 0.33 for key in answer["probabilities"]}
            payload["usage"] = {"input_tokens": 6396, "output_tokens": 485}
        else:
            payload["usage"] = {"input_tokens": 5903, "output_tokens": 419}
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JevCitationAuditor(client=client).evaluate(
            _multi_snapshot(9),
            _ledger(),
            JevAuditConfig(enabled=True, max_questions_per_batch=2, concurrency=2),
            artifact={"run_id": "partial"},
        )
    assert len(calls) == 2 and result.status == "partial"
    assert result.usage.input_tokens == 12299 and result.usage.output_tokens == 904
    assert result.usage.call_count == 2 and result.usage.usage_complete
    assert len(result.decisions) == 2


@pytest.mark.asyncio
async def test_valid_scientific_negative_does_not_stop_other_batches(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    calls = []

    async def handler(request):
        calls.append(request)
        payload = _valid_response(request)
        for answer in payload["answers"].values():
            choice = next(reversed(answer["probabilities"]))
            answer["choice"] = choice
            answer["probabilities"] = {
                key: 1.0 if key == choice else 0.0 for key in answer["probabilities"]
            }
        return httpx.Response(200, json=payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JevCitationAuditor(client=client).evaluate(
            _multi_snapshot(9),
            _ledger(),
            JevAuditConfig(enabled=True, max_questions_per_batch=2, concurrency=2),
            artifact={"run_id": "scientific-negative"},
        )
    assert len(calls) == 9 and result.status == "completed"
    assert len(result.decisions) == 18
    assert all(row.choice in {"insufficient", "unrelated"} for row in result.decisions)
    assert result.usage.call_count == 9 and result.usage.usage_complete


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_type,reason",
    [
        (httpx.ConnectError, "jev_connection_error"),
        (httpx.ReadError, "jev_transport_error"),
        (httpx.RemoteProtocolError, "jev_transport_error"),
        (httpx.ProxyError, "jev_transport_error"),
        (httpx.RequestError, "jev_request_error"),
        (httpx.ConnectTimeout, "jev_timeout"),
    ],
)
async def test_transport_failure_stops_wave_without_retry_or_response_verdict(
    monkeypatch, error_type, reason
):
    """Keep received sibling usage and never pretend a missing response was parsed."""
    from litagent.benchmark.quality_suite import _technical_failure

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-key")
    calls = []

    async def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise error_type("private proxy detail must not escape", request=request)
        response = _valid_response(request)
        response["usage"] = {"input_tokens": 5903, "output_tokens": 419}
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await JevCitationAuditor(client=client).evaluate(
            _multi_snapshot(9),
            _ledger(),
            JevAuditConfig(enabled=True, max_questions_per_batch=2, concurrency=2),
            artifact={"run_id": "connection-failure"},
        )
    assert len(calls) == 2 and result.status == "partial"
    failed, completed = result.batches
    assert failed.response is None and failed.reason_codes == [reason]
    assert failed.error_type == error_type.__name__
    assert failed.usage.input_tokens is None and not failed.usage.usage_complete
    assert completed.usage.input_tokens == 5903 and completed.usage.usage_complete
    assert not result.usage.usage_complete and result.usage.call_count == 2
    assert "jev_response_invalid" not in result.reason_codes
    assert "private proxy detail" not in result.model_dump_json()
    assert _technical_failure(result.reason_codes) == reason
