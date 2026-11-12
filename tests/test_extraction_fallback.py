"""Exercise safe partial Jev admission through real HTTP and receipt consumers."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from test_extraction_verification import selection
from test_jev_citation_audit import _valid_response
from test_optional_verification import optional_report
from test_quality_view import render_repair_state

from litagent.agents.extractor import ExtractorWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.evidence import SOURCE_FIELDS_IDENTITY, collect_ledger
from litagent.extraction_verification import (
    ADMISSION_VERSION,
    FALLBACK_POLICY,
    ExtractionVerifier,
    admission_identity,
    admitted_writing_ledger,
    validate_extraction_receipt,
)
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import build_fact_cards, digest
from litagent.workbench import (
    RunOptions,
    VerificationOptions,
    default_verification_options,
)


def candidates(count=20, *, source_size=0):
    """Keep distinct source identities even for equal claim text."""
    result = {}
    for i in range(count):
        item = copy.deepcopy(selection().selected_items["a"])
        item["evidence_id"] = "a" if i == 0 else f"c{i}"
        if source_size:
            item["supporting_text"] += "a" * source_size
        result[item["evidence_id"]] = item
    return result


def failing_transport(
    captured, *, status=503, usage=True, all_failed=False, fault=None
):
    """Replace external transport, preserving real packing, accounting and decoding."""

    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        should_fail = all_failed or len(captured) == 2
        if not should_fail:
            response = _valid_response(request)
            for answer in response["answers"].values():
                answer["choice"] = "ok"
                answer["probabilities"] = {"ok": 1.0, "not_ok": 0.0, "uncertain": 0.0}
            if fault == "model":
                response["model"] = "wrong"
            if fault == "omitted":
                response["answers"].pop(next(iter(response["answers"])))
            if fault == "low":
                for answer in response["answers"].values():
                    answer["confidence"] = 0.89
            return httpx.Response(200, json=response)
        if fault == "timeout":
            raise httpx.ReadTimeout("offline timeout", request=request)
        body = {"error": "offline unavailable"}
        if usage:
            body["usage"] = {"input_tokens": 7, "output_tokens": 0}
        return httpx.Response(status, json=body)

    return httpx.MockTransport(respond)


async def run_gate(
    tmp_path,
    monkeypatch,
    *,
    ledger=None,
    policy=FALLBACK_POLICY,
    confidence_threshold=None,
    resource_policy=None,
    **fault,
):
    """Run the real auditor with no network and return its authenticated receipt."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline")
    captured = []
    client = httpx.AsyncClient(transport=failing_transport(captured, **fault))
    gate = ExtractionVerifier(
        tmp_path, protocol=ADMISSION_VERSION, auditor=JevCitationAuditor(client=client)
    )
    try:
        admitted, receipt = await gate.verify_candidates(
            ledger or candidates(),
            run_id="partial",
            failure_policy=policy,
            confidence_threshold=confidence_threshold,
            resource_policy=resource_policy,
        )
        return admitted, receipt, captured, gate
    finally:
        await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
async def test_transient_known_usage_uses_only_verified_pairs(
    tmp_path, monkeypatch, status
):
    admitted, receipt, captured, gate = await run_gate(
        tmp_path, monkeypatch, status=status
    )
    assert len(captured) == 2  # Stop later waves; never retry the failed batch.
    assert 0 < len(admitted) < 20
    assert receipt["status"] == "degraded"
    assert receipt["jev"]["status"] == "partial"
    assert receipt["fallback"]["active"]
    assert receipt["fallback"]["verified_count"] == len(admitted)
    assert set(receipt["fallback"]["unverified_evidence_ids"]) == set(
        candidates()
    ) - set(admitted)
    assert receipt["jev"]["usage"]["usage_complete"]
    assert receipt["jev"]["usage"]["call_count"] == 2
    assert receipt["jev"]["batches"][1]["usage"]["input_tokens"] == 7
    assert validate_extraction_receipt(receipt, admitted)
    assert await gate.verify_candidates(
        candidates(), run_id="partial", failure_policy=FALLBACK_POLICY
    ) == (admitted, receipt)
    assert len(captured) == 2
    with pytest.raises(ValueError, match="identity_conflict"):
        await gate.verify_candidates(candidates(), run_id="partial")
    assert (
        admitted_writing_ledger(
            receipt, admitted, build_fact_cards(admitted), run_id="partial"
        )
        == admitted
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        {"usage": False},
        {"fault": "timeout"},
        {"status": 401},
        {"status": 403},
        {"status": 400},
        {"fault": "model"},
        {"fault": "omitted"},
    ],
)
async def test_unsafe_failures_cannot_degrade_or_replay(tmp_path, monkeypatch, fault):
    with pytest.raises(ValueError, match="fallback_ineligible|verification_incomplete"):
        await run_gate(tmp_path, monkeypatch, **fault)
    receipt = json.loads((tmp_path / "partial" / "receipt.json").read_text())
    assert receipt["status"] == "failed_no_replay"
    assert not validate_extraction_receipt(receipt, {})
    auditor = AsyncMock()
    gate = ExtractionVerifier(tmp_path, protocol=ADMISSION_VERSION, auditor=auditor)
    with pytest.raises(ValueError, match="no_replay"):
        await gate.verify_candidates(
            candidates(), run_id="partial", failure_policy=FALLBACK_POLICY
        )
    auditor.evaluate.assert_not_awaited()


@pytest.mark.asyncio
async def test_old_policy_still_rejects_partial_result(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="verification_incomplete"):
        await run_gate(tmp_path, monkeypatch, policy=None)


@pytest.mark.asyncio
async def test_empty_or_low_confidence_subset_never_admits(tmp_path, monkeypatch):
    admitted, receipt, _, _ = await run_gate(tmp_path, monkeypatch, all_failed=True)
    assert not admitted and receipt["status"] == "degraded"
    assert validate_extraction_receipt(receipt, {})
    admitted, receipt, _, _ = await run_gate(tmp_path / "low", monkeypatch, fault="low")
    assert not admitted
    assert any("extraction_low_confidence" in r for r in receipt["rejected"].values())


@pytest.mark.asyncio
async def test_one_question_completed_is_not_a_passed_pair(tmp_path, monkeypatch):
    ledger = candidates(1, source_size=12500)
    admitted, receipt, captured, _ = await run_gate(
        tmp_path, monkeypatch, ledger=ledger
    )
    assert len(captured) == 2 and all(len(p["questions"]) == 1 for p in captured)
    assert not admitted
    assert receipt["rejected"]["a"] == ["extraction_verification_unavailable"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field", ["request", "decisions", "usage", "status", "policy", "fallback"]
)
async def test_tampering_cannot_turn_partial_results_into_admission(
    tmp_path, monkeypatch, field
):
    admitted, receipt, _, _ = await run_gate(tmp_path, monkeypatch)
    changed = copy.deepcopy(receipt)
    if field == "request":
        changed["jev"]["batches"][0]["request"]["model"] = "wrong"
    elif field == "decisions":
        changed["jev"]["decisions"].pop()
    elif field == "usage":
        changed["jev"]["usage"]["input_tokens"] -= 1
    elif field == "status":
        changed["status"] = "completed"
    elif field == "policy":
        changed.pop("failure_policy")
    else:
        changed["fallback"]["unverified_evidence_ids"] = []
    assert not validate_extraction_receipt(changed, admitted)


def test_defaults_and_legacy_serialization_identity():
    assert VerificationOptions().model_dump() == {
        "version": "optional-verification-v1",
        "judge": False,
        "jev": False,
    }
    old = VerificationOptions(version="optional-verification-v2", jev=True)
    assert "extraction_failure_policy" not in old.model_dump()
    assert default_verification_options().extraction_failure_policy == FALLBACK_POLICY
    assert (
        admission_identity()
        == "47d56c886ccc4d8c2080eb7cc44e88d8a92a761caaf4462fdc32c9a0bec69a2c"
    )
    assert (
        RunOptions(quality_policy="strict_auto_v2", verification=old).fingerprint()
        != RunOptions(
            quality_policy="strict_auto_v2", verification=default_verification_options()
        ).fingerprint()
    )
    with pytest.raises(ValueError, match="requires_admission_v2"):
        VerificationOptions(extraction_failure_policy=FALLBACK_POLICY)


@pytest.mark.asyncio
async def test_verified_subset_still_requires_evaluation_and_bound_policy(
    tmp_path, monkeypatch
):
    admitted, receipt, _, _ = await run_gate(
        tmp_path,
        monkeypatch,
        confidence_threshold=default_verification_options().extraction_confidence_threshold,
        resource_policy=default_verification_options().extraction_resource_policy,
    )
    cards = build_fact_cards(admitted)
    report, _ = optional_report()
    report["metadata"].update(
        verification=default_verification_options().model_dump(),
        extraction_verification=receipt,
        fact_cards=cards,
        evidence_snapshot_sha256=digest(cards),
    )
    snapshot = freeze_automatic_inputs(report, admitted, run_id="partial")
    assert automatic_acceptance(snapshot, report["evaluation"], None)["publishable"]
    failed = copy.deepcopy(report["evaluation"])
    failed["faithfulness"]["passed"] = False
    assert not automatic_acceptance(snapshot, failed, None)["publishable"]
    report["metadata"]["verification"].pop("extraction_failure_policy")
    with pytest.raises(ValueError, match="extraction"):
        freeze_automatic_inputs(report, admitted, run_id="partial")


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_quality_view_explains_degraded_admission(locale):
    receipt = {
        "version": ADMISSION_VERSION,
        "status": "degraded",
        "candidate_ledger": {"a": {}, "b": {}},
        "accepted_evidence_ids": ["a"],
        "rejected": {"b": ["extraction_verification_unavailable"]},
        "fallback": {
            "active": True,
            "unverified_evidence_ids": ["b"],
            "reason_codes": ["jev_http_503"],
        },
    }
    output = render_repair_state(
        {"options": {"extractionVerification": receipt}}, locale
    )["text"]
    assert ("Degraded" if locale == "en" else "已降级") in output
    assert "jev_http_503" in output


@pytest.mark.asyncio
@pytest.mark.parametrize("all_failed", [False, True])
async def test_extractor_to_synthesis_respects_degraded_subset(
    tmp_path, monkeypatch, all_failed
):
    """Exercise actual ledger filtering and synthesis, replacing only external services."""
    from test_scope_notice_contract import notice_document

    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.llm.client import LLMResponse

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline")
    records = [
        {
            "text": f"EViT finding {i}.",
            "source_chunk_key": f"chunk{i}",
            "fact_fields": {"method": "EViT"},
        }
        for i in range(20)
    ]
    strategy = AsyncMock()
    strategy.extract.return_value = {
        "claims": [r["text"] for r in records],
        "claim_records": records,
        "record_identity": SOURCE_FIELDS_IDENTITY,
    }
    loader = AsyncMock()
    loader.enrich.side_effect = lambda papers, _: papers
    captured = []
    options = default_verification_options().model_dump()
    async with httpx.AsyncClient(
        transport=failing_transport(captured, all_failed=all_failed)
    ) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        worker = ExtractorWorker(
            strategy,
            fulltext_loader=loader,
            extraction_verifier=gate,
            run_id_provider=lambda: "pipeline",
        )
        task = SubTask(
            "extract",
            "offline",
            "extractor",
            input_data={
                "quality_policy": "strict_auto_v2",
                "verification": options,
                "upstream_results": {
                    "dedup": [
                        {
                            "paper_id": "evit",
                            "title": "EViT",
                            "chunks": [
                                {"chunk_key": r["source_chunk_key"], "text": r["text"]}
                                for r in records
                            ],
                        }
                    ]
                },
            },
        )
        if all_failed:
            with pytest.raises(ValueError, match="extraction_no_supported_candidates"):
                await worker.execute(task)
            return
        output = await worker.execute(task)
        ledger = collect_ledger(output)
        receipt = output[0]["extraction_verification"]
        assert receipt["status"] == "degraded"
        assert 0 < len(ledger) < 20
        assert len(output[0]["candidate_claim_records"]) == 20
        assert len(output[0]["claim_records"]) == len(ledger)
        assert set(ledger) == set(receipt["accepted_evidence_ids"])
        assert all(r["text"] in output[0]["claims"] for r in output[0]["claim_records"])
        sections = (
            "introduction",
            "taxonomy",
            "methods",
            "experiments",
            "open_problems",
        )
        picked = EvidenceSelection(
            len(ledger), ledger, {s: list(ledger) for s in sections}, {}, 0, 100
        )
        selector, llm = AsyncMock(), AsyncMock()
        selector.select.return_value = picked
        llm.chat.return_value = LLMResponse(
            content=json.dumps(notice_document(build_fact_cards(ledger))),
            model="offline",
            usage={},
        )
        synthesis = SynthesisWorker(
            llm,
            evidence_selector=selector,
            extraction_verifier=gate,
            run_id_provider=lambda: "pipeline",
        )
        result = await synthesis.execute(
            SubTask(
                "synthesis",
                "offline",
                "synthesis",
                input_data={
                    "quality_policy": "strict_auto_v2",
                    "verification": options,
                    "review_structure": "academic_v2",
                    "query": "EViT",
                    "upstream_results": {"extract": output},
                },
            )
        )
        assert result["extraction_verification"] == receipt
        assert result["evidence_selection"]["selected_items"] == ledger
        assert len(captured) == 4
        assert receipt["resource_policy"] == "expanded-v1"
        assert result.get("quality", {}).get("status") != "passed"


@pytest.mark.asyncio
async def test_cancellation_never_degrades(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline")
    entered = asyncio.Event()
    blocked = asyncio.Event()

    async def respond(request):
        entered.set()
        await blocked.wait()
        return httpx.Response(200, json=_valid_response(request))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        task = asyncio.create_task(
            gate.verify_candidates(
                candidates(), run_id="cancel", failure_policy=FALLBACK_POLICY
            )
        )
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    receipt = json.loads((tmp_path / "cancel" / "receipt.json").read_text())
    assert receipt["status"] == "failed_no_replay"
    assert not validate_extraction_receipt(receipt, {})
