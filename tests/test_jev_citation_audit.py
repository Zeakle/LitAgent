"""Validate the Jev citation sidecar without external network access."""

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.citation_audit import (
    JevAuditConfig,
    JevCitationAuditor,
    build_citation_audit_inputs,
)
from litagent.benchmark.survey_models import (
    AnchoredJudgeClaim,
    JudgeInventorySnapshot,
)


def _snapshot(evidence_ids=("ev-1", "ev-2")):
    claim = AnchoredJudgeClaim(
        claim_id="claim_0",
        claim_text="Method A improves accuracy.",
        source_unit_id="unit:0:0",
        source_quote="Method A improves accuracy [E:ev-1] [E:ev-2].",
        source_start=0,
        source_end=55,
        evidence_ids=evidence_ids,
    )
    return JudgeInventorySnapshot(
        survey_sha256="sha256:" + "1" * 64,
        source_units_sha256="sha256:" + "2" * 64,
        ledger_sha256=_stable_test_sha(_ledger()),
        inventory_sha256=_stable_test_sha([claim.model_dump(mode="json")]),
        claims=(claim,),
    )


def _ledger():
    return {
        "ev-1": {"paper_id": "arxiv:1", "text": "Method A improves accuracy."},
        "ev-2": {"paper_id": "arxiv:2", "text": "Method A uses adaptation."},
    }


def _stable_test_sha(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _multi_snapshot(count=9):
    claims = tuple(
        AnchoredJudgeClaim(
            claim_id=f"claim_{index}",
            claim_text=f"Claim {index}.",
            source_unit_id=f"unit:{index}:0",
            source_quote=f"Claim {index} [E:ev-1].",
            source_start=index * 20,
            source_end=index * 20 + 19,
            evidence_ids=("ev-1",),
        )
        for index in range(count)
    )
    return JudgeInventorySnapshot(
        survey_sha256="sha256:" + "1" * 64,
        source_units_sha256="sha256:" + "2" * 64,
        ledger_sha256=_stable_test_sha(_ledger()),
        inventory_sha256=_stable_test_sha(
            [claim.model_dump(mode="json") for claim in claims]
        ),
        claims=claims,
    )


def _valid_response(request):
    payload = json.loads(request.content)
    answers = {}
    for question_id, question in payload["questions"].items():
        options = list(question["criteria"])
        answers[question_id] = {
            "type": "choice",
            "choice": options[0],
            "probabilities": {
                option: 1.0 if option == options[0] else 0.0 for option in options
            },
            "confidence": 0.97,
        }
    return {
        "model": "jev-1.13.0",
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 5},
    }


def test_build_inputs_keeps_claim_specific_ledger_evidence_and_local_errors():
    inputs = build_citation_audit_inputs(
        _snapshot(),
        _ledger(),
        artifact={
            "nodes": {
                "worker:evidence": {
                    "output": {"items": [{"evidence_id": "ev-1", "text": "raw"}]}
                }
            }
        },
    )
    assert len(inputs) == 1
    assert inputs[0].status == "ready"
    assert [item.evidence_id for item in inputs[0].evidence] == ["ev-1", "ev-2"]
    assert all(item.input_scope == "ledger" for item in inputs[0].evidence)
    assert inputs[0].evidence[0].raw_chunk == "raw"

    unavailable = build_citation_audit_inputs(
        _snapshot(("missing",)), _ledger(), artifact={}
    )
    assert unavailable[0].status == "local_error"
    assert unavailable[0].reason_codes == ["citation_evidence_missing"]
    assert unavailable[0].unavailable_evidence_ids == ("missing",)

    tampered = dict(_ledger())
    tampered["ev-1"] = {**tampered["ev-1"], "text": "Changed after freeze."}
    with pytest.raises(ValueError, match="jev_ledger_hash_mismatch"):
        build_citation_audit_inputs(_snapshot(), tampered, artifact={})


@pytest.mark.asyncio
async def test_jev_auditor_maps_group_and_member_choices(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    requests = []

    async def handler(request):
        requests.append(json.loads(request.content))
        answers = {}
        for question_id, question in requests[-1]["questions"].items():
            options = list(question["criteria"])
            choice = options[0]
            answers[question_id] = {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    option: 1.0 if option == choice else 0.0 for option in options
                },
                "confidence": 0.97,
            }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 321, "output_tokens": 12},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await JevCitationAuditor(client=client).evaluate(
            _snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True),
            artifact={"run_id": "product-1"},
            reference_judge={
                "inventory_snapshot": _snapshot().model_dump(mode="json"),
                "claim_assessments": [
                    {"claim_id": "claim_0", "supported": True, "source_faithful": True}
                ],
            },
        )
    finally:
        await client.aclose()

    assert result.status == "completed"
    assert [item.choice for item in result.decisions] == [
        "supported",
        "contributes",
        "contributes",
    ]
    assert result.summary["protocol_completion_rate"] == 1.0
    assert result.summary["agreement"] == 1.0
    assert result.summary["accuracy"] is None
    assert result.usage.input_tokens == 321
    assert result.usage.usage_complete is True
    assert requests[0]["model"] == "jev-1.13.0"
    assert len(requests[0]["questions"]) == 3


@pytest.mark.asyncio
async def test_jev_accepts_documented_v1_base_url_without_duplicating_path(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    urls = []

    async def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_valid_response(request))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        await JevCitationAuditor(client=client).evaluate(
            _snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True, base_url="https://api.typesafe.ai/v1"),
            artifact={"run_id": "product-v1"},
        )
    finally:
        await client.aclose()

    assert urls == ["https://api.typesafe.ai/v1/systemone"]


@pytest.mark.asyncio
async def test_jev_auditor_rejects_invalid_probability_distribution(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")

    async def handler(request):
        payload = json.loads(request.content)
        answers = {
            question_id: {
                "type": "choice",
                "choice": next(iter(question["criteria"])),
                "probabilities": {option: 0.2 for option in question["criteria"]},
                "confidence": 0.9,
            }
            for question_id, question in payload["questions"].items()
        }
        return httpx.Response(
            200,
            json={
                "model": "jev-1.13.0",
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 1},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    try:
        result = await JevCitationAuditor(client=client).evaluate(
            _snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True),
            artifact={"run_id": "product-1"},
        )
    finally:
        await client.aclose()

    assert result.status == "failed"
    assert result.decisions == []
    assert "jev_response_invalid" in result.reason_codes


def test_jev_config_is_opt_in_and_forbids_automatic_retries():
    assert JevAuditConfig().enabled is False
    with pytest.raises(ValueError):
        JevAuditConfig(max_retries=1)


def test_citation_audit_artifact_has_a_dedicated_markdown_projection(tmp_path):
    from litagent.benchmark.citation_audit import CitationAuditResult

    result = CitationAuditResult(
        run_id="citation-audit-test",
        status="failed",
        source_run_id="product-1",
        source_artifact_sha256="sha256:" + "a" * 64,
        inventory_sha256="sha256:" + "b" * 64,
        evidence_sha256="sha256:" + "c" * 64,
        requested_model="jev-1.13.0",
        reason_codes=["inventory_unavailable"],
        started_at="2026-09-22T00:00:00Z",
        completed_at="2026-09-22T00:00:01Z",
    )
    paths = BenchmarkArtifactRepository(tmp_path).write(result)
    markdown = Path(paths.markdown_path).read_text(encoding="utf-8")
    assert "Citation Audit" in markdown
    assert "inventory_unavailable" in markdown


@pytest.mark.asyncio
async def test_checkpoint_resume_does_not_repeat_terminal_paid_batches(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    repository = BenchmarkArtifactRepository(tmp_path)
    first_calls = 0

    async def first_handler(request):
        nonlocal first_calls
        first_calls += 1
        if first_calls == 2:
            return httpx.Response(401, json={"detail": "unauthorized"})
        return httpx.Response(200, json=_valid_response(request))

    first_client = httpx.AsyncClient(transport=httpx.MockTransport(first_handler))
    try:
        first = await JevCitationAuditor(
            client=first_client, repository=repository
        ).evaluate(
            _multi_snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True, concurrency=1),
            artifact={"run_id": "product-resume"},
        )
    finally:
        await first_client.aclose()

    resumed_calls = 0

    async def resumed_handler(request):
        nonlocal resumed_calls
        resumed_calls += 1
        return httpx.Response(200, json=_valid_response(request))

    monkeypatch.delenv("TYPESAFE_API_KEY")
    resumed_client = httpx.AsyncClient(transport=httpx.MockTransport(resumed_handler))
    try:
        resumed = await JevCitationAuditor(
            client=resumed_client, repository=repository
        ).evaluate(
            _multi_snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True, concurrency=1),
            artifact={"run_id": "product-resume"},
        )
    finally:
        await resumed_client.aclose()

    assert first.status == "partial"
    assert first_calls == 2
    assert resumed_calls == 0
    assert resumed.model_dump(mode="json") == first.model_dump(mode="json")
    assert resumed.usage.usage_complete is False


@pytest.mark.asyncio
async def test_cancellation_persists_cancelled_sidecar(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    started = asyncio.Event()

    async def handler(request):
        started.set()
        await asyncio.Event().wait()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    repository = BenchmarkArtifactRepository(tmp_path)
    task = asyncio.create_task(
        JevCitationAuditor(client=client, repository=repository).evaluate(
            _snapshot(),
            _ledger(),
            JevAuditConfig(enabled=True),
            artifact={"run_id": "product-cancel"},
        )
    )
    await started.wait()
    task.cancel()
    try:
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        await client.aclose()

    artifacts = [
        path
        for path in tmp_path.glob("citation-audit-*.json")
        if "checkpoints" not in path.parts
    ]
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["status"] == "cancelled"
    assert "jev_cancelled" in payload["reason_codes"]
    assert payload["usage"]["usage_complete"] is False
