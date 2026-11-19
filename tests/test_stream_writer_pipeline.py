"""Keep streamed writes, format recovery and candidate replay on one contract."""

import copy
import inspect
import json
from pathlib import Path

import httpx
import pytest
from test_inline_citations import InlineGateway
from test_responses_stream import WireStream, frame
from test_separate_review_pipeline import execute
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.format_repair import validate_recovery_record
from litagent.llm.responses import (
    RESPONSES_CONTRACT,
    STREAM_CONTRACT,
    decode_private_receipt,
    normalize_private_receipt,
    selected_transport,
    transport_identity,
)
from litagent.run_limits import (
    PlannedRunLimits,
    configured_run_limits,
    validate_run_limits,
)
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.writing_v2 import WriterError


def stream_config():
    return load_config("config/llm-stage-repair-stream.yaml")


class StreamingGateway(InlineGateway):
    """Replace only provider HTTP with raw SSE; all application consumers run."""

    def __init__(self, *args, regenerate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.regenerate = regenerate
        self.stream_requests = []
        self.nonstream_requests = []

    async def __call__(self, request):
        response = await super().__call__(request)
        if not request.url.path.endswith("/responses"):
            return response
        request_body = json.loads(request.content)
        if request_body["stream"] is False:
            self.nonstream_requests.append(request_body)
            return response
        self.stream_requests.append(request_body)
        assert request_body["stream"] is True
        assert "responses_transport" not in request_body
        wire = response.json()
        payload = json.loads(request_body["input"][1]["content"])
        if self.regenerate and "anchors" in payload:
            wire["output"][-1]["content"][0]["text"] = json.dumps(
                {"outcome": "cannot_repair", "edits": []}
            )
        created = {**wire, "status": "in_progress", "output": [], "usage": None}
        return httpx.Response(
            response.status_code,
            stream=WireStream(
                [
                    frame("response.created", response=created),
                    frame("response.completed", 1, response=wire),
                ]
            ),
            headers={**response.headers, "content-type": "text/event-stream"},
        )


def journals(tmp_path):
    return [json.loads(p.read_text()) for p in tmp_path.rglob("*.writer-v2.json")]


def assert_stream_receipts(states):
    count = 0
    for state in states:
        assert state["generation_responses_transport"] == STREAM_CONTRACT
        for call in state["calls"].values():
            receipt = call.get("provider_receipt")
            if not receipt or call.get("provider_api") != "responses":
                continue
            count += 1
            assert receipt["contract"] == call["responses_transport"] == STREAM_CONTRACT
            request, _ = decode_private_receipt(receipt)
            assert request["stream"] is True
            normalized = normalize_private_receipt(receipt)
            assert normalized["content"] == call["raw"]
            assert normalized["usage"] == call["usage"]
    assert count > 0


def test_server_transport_is_optional_and_enters_frozen_identity():
    old = load_config("config/llm-stage-repair.yaml")
    current = stream_config()
    old_limits = configured_run_limits(old)
    new_limits = configured_run_limits(current)
    assert "responses_transport" not in old_limits
    assert "responses_transport_identity" not in old_limits
    assert {
        k: v for k, v in new_limits.items() if not k.startswith("responses_transport")
    } == old_limits
    assert new_limits["responses_transport_identity"] == transport_identity()
    assert validate_run_limits(new_limits) == new_limits
    assert selected_transport({"run_limits": new_limits}) == STREAM_CONTRACT
    assert PlannedRunLimits().model_dump() == {"max_tokens": 10000000}
    bad = copy.deepcopy(new_limits)
    bad["responses_transport_identity"] = "0" * 64
    with pytest.raises(ValueError, match="transport_identity"):
        validate_run_limits(bad)
    values = current.safety.planned_run_limits.model_dump()
    values.pop("candidate_recovery")
    with pytest.raises(ValueError, match="requires_stage_repair"):
        PlannedRunLimits.model_validate(values)


@pytest.mark.asyncio
@pytest.mark.parametrize("regenerate", [False, True])
async def test_workbench_streams_writing_format_and_both_revision_loops(
    case, tmp_path, monkeypatch, regenerate
):
    gateway = StreamingGateway(
        case, reviews=1, format_fault="extra_brace", regenerate=regenerate
    )
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stream_config())
    assert gateway.global_calls >= 2, report["metadata"].get("writer_failure")
    assert report["delivery"]["publishable"], report["metadata"].get("writer_failure")
    assert gateway.editorial_edits == (2 if regenerate else 1)
    assert gateway.repair_count == 1
    allowances = report["metadata"]["review_workflow"]["allowances"]
    assert allowances["review_revision"]["used"] == 1
    assert allowances["evaluation_revision"]["used"] == 1
    assert gateway.format_calls == (2 if regenerate else 1)
    assert gateway.nonstream_requests
    states = journals(tmp_path)
    assert_stream_receipts(states)
    recoveries = [
        call["format_recovery"]
        for s in states
        for call in s["calls"].values()
        if "format_recovery" in call
    ]
    assert recoveries
    assert recoveries[0]["outcome"] == ("regenerated" if regenerate else "recovered")
    for state in recoveries:
        for attempt in state["attempts"]:
            if attempt.get("sent"):
                assert attempt["provider_receipt"]["contract"] == STREAM_CONTRACT
                assert attempt["responses_transport"] == STREAM_CONTRACT
    for state in states:
        for call in state["calls"].values():
            if "format_recovery" not in call:
                continue
            validate_recovery_record(call)
            for location in ("original", "attempt", "entry"):
                changed = copy.deepcopy(call)
                saved = (
                    changed["format_recovery"]["original"]
                    if location == "original"
                    else (
                        changed["format_recovery"]["attempts"][0]
                        if location == "attempt"
                        else changed
                    )
                )
                saved["provider_receipt"]["response_sha256"] = "0" * 64
                with pytest.raises(WriterError, match="writer_format_receipt_conflict"):
                    validate_recovery_record(changed)
            changed = copy.deepcopy(call)
            changed["format_recovery"]["original"]["provider_receipt"][
                "contract"
            ] = RESPONSES_CONTRACT
            with pytest.raises(WriterError, match="writer_format_receipt_conflict"):
                validate_recovery_record(changed)


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["review", "evaluation"])
async def test_streamed_summary_error_is_authenticated_then_repaired_in_its_stage(
    case, tmp_path, monkeypatch, category
):
    class Gateway(StreamingGateway):
        injected = False

        def answer(self, payload, schema):
            result = super().answer(payload, schema)
            ready = self.editorial_edits if category == "review" else self.repair_count
            if (
                "abstract" in schema.get("properties", {})
                and ready
                and not self.injected
            ):
                self.injected = True
                result["abstract"][0]["text"] += "[C:c99998]"
                result["conclusion"][0]["text"] += "Missing source[C:c99999]."
            return result

    gateway = Gateway(case, reviews=1 if category == "review" else 0)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stream_config())
    assert gateway.injected and gateway.global_calls >= 1, report["metadata"].get(
        "writer_failure"
    )
    assert (
        gateway.editorial_edits == 1
        if category == "review"
        else gateway.repair_count == 1
    )
    states = journals(tmp_path)
    assert_stream_receipts(states)
    rejected = [s for s in states if s.get("candidate_rejections")]
    assert rejected and all(not s.get("pending_candidate") for s in rejected)


def test_vit13_accounting_is_readable_but_cannot_adopt_stream_transport():
    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/vit-235d5bf1dba2-inline-reassessment-13/budget-ledger.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT13 accounting")
    original = path.read_bytes()
    snapshot = json.loads(original)
    assert (
        validate_accounting_snapshot(snapshot, expected_plan=snapshot["plan"])
        == snapshot
    )
    limits = snapshot["plan"]["envelope"]["run_limits"]
    with pytest.raises(ValueError, match="stage_repair_identity"):
        validate_run_limits(limits)
    changed = {
        **limits,
        "responses_transport": STREAM_CONTRACT,
        "responses_transport_identity": transport_identity(),
    }
    with pytest.raises(ValueError, match="stage_repair_identity"):
        validate_run_limits(changed, archived=True)
    assert path.read_bytes() == original


@pytest.mark.asyncio
async def test_streamed_capacity_candidate_uses_authenticated_bounded_repair(
    tmp_path, monkeypatch
):
    import test_capacity_stage_recovery as capacity_test

    config = stream_config()
    config.llm.base_url = "https://offline.invalid/api/v3"
    config.observability.enabled = False
    monkeypatch.setattr(capacity_test, "configuration", lambda: config)
    original_transport = httpx.MockTransport
    streamed = []

    def transport(handler):
        async def receive(request):
            result = handler(request)
            response = await result if inspect.isawaitable(result) else result
            sent = json.loads(request.content)
            if not sent.get("stream"):
                return response
            streamed.append(request)
            wire = response.json()
            return httpx.Response(
                response.status_code,
                stream=WireStream([frame("response.completed", response=wire)]),
                headers={**response.headers, "content-type": "text/event-stream"},
            )

        return original_transport(receive)

    monkeypatch.setattr(httpx, "MockTransport", transport)
    await capacity_test.test_capacity_candidate_repair_is_bounded_and_settled(
        tmp_path, monkeypatch, "recovered"
    )
    assert streamed
    assert_stream_receipts(journals(tmp_path))
