"""Exercise expanded admission resources without changing historical audit budgets."""

import asyncio
import copy
import json

import httpx
import pytest
from test_extraction_fallback import candidates, run_gate
from test_jev_citation_audit import _valid_response
from test_quality_view import render_repair_state

from litagent.benchmark.citation_audit import (
    JevAuditConfig,
    JevCitationAuditor,
    _pack_batches,
    _QuestionSpec,
    _request_payload,
)
from litagent.extraction_verification import (
    ADMISSION_VERSION,
    RESOURCE_POLICY,
    ExtractionVerifier,
    admission_timeout_ms,
    confidence_identity,
    extraction_config,
    require_admission_policy,
    resource_limits,
    validate_extraction_receipt,
)
from litagent.workbench import (
    RunOptions,
    VerificationOptions,
    default_verification_options,
)


def test_expanded_profile_and_historical_identity():
    old = VerificationOptions(
        version="optional-verification-v2",
        jev=True,
        extraction_failure_policy="verified_subset_v1",
        extraction_confidence_threshold=0.6,
    )
    assert "extraction_resource_policy" not in old.model_dump()
    cfg = extraction_config(0.6, resource_policy=RESOURCE_POLICY)
    assert (
        cfg.max_input_tokens,
        cfg.max_requests,
        cfg.max_questions_per_batch,
        cfg.concurrency,
    ) == (4_000_000, 200, 32, 4)
    assert (cfg.max_request_bytes, cfg.timeout_seconds, cfg.max_retries) == (
        24000,
        30,
        0,
    )
    assert admission_timeout_ms(RESOURCE_POLICY) == 1510000
    assert admission_timeout_ms() == 310000
    assert JevAuditConfig().max_input_tokens == 200000
    assert JevAuditConfig().max_requests == 20
    assert JevAuditConfig().max_questions_per_batch == 16
    assert JevAuditConfig().concurrency == 2
    assert (
        confidence_identity()
        == "9cd436df5355008cab3984b701468c8c3a3081d849028bb85c8e5969bb449f38"
    )
    assert (
        extraction_config(0.6).fingerprint
        == "sha256:f36f828c9e19e7dd4df857c63db9da14d6a79327d4e1745ba07cb841be8fc453"
    )
    assert (
        RunOptions(quality_policy="strict_auto_v2", verification=old).fingerprint()
        == "1b7ca74f3f47bb0bf86b3a85325fc969e6e7beaab5c98eff559be5b7645e38fd"
    )
    assert (
        RunOptions(
            quality_policy="strict_auto_v2", verification=default_verification_options()
        ).fingerprint()
        != RunOptions(quality_policy="strict_auto_v2", verification=old).fingerprint()
    )


@pytest.mark.parametrize("value", ["unbounded", True, {"max_input_tokens": 9999999}])
def test_only_named_server_policy_is_accepted(value):
    with pytest.raises(ValueError):
        VerificationOptions(
            version="optional-verification-v2", extraction_resource_policy=value
        )
    with pytest.raises(ValueError):
        extraction_config(resource_policy=value)


def test_legacy_protocol_cannot_enable_expanded_profile():
    with pytest.raises(ValueError):
        VerificationOptions(extraction_resource_policy=RESOURCE_POLICY)
    with pytest.raises(ValueError):
        extraction_config(
            protocol="jev-extraction-verification-v1", resource_policy=RESOURCE_POLICY
        )


def test_question_count_is_separate_from_full_request_bytes():
    specs = [
        _QuestionSpec(
            question_id=f"q{i}",
            claim_id=f"c{i}",
            kind="group",
            state={"cited_evidence": []},
            criteria={"ok": "supported"},
        )
        for i in range(65)
    ]
    expanded = extraction_config(resource_policy=RESOURCE_POLICY)
    batches, rejected = _pack_batches(specs, expanded)
    assert not rejected
    assert [len(b) for b in batches] == [32, 32, 1]
    old, rejected = _pack_batches(specs, JevAuditConfig())
    assert not rejected and max(map(len, old)) == 16
    for batch in batches:
        size = len(json.dumps(_request_payload(batch, expanded.model)).encode())
        assert size <= expanded.max_request_bytes
        # Every question receives the shared state; checking its pointer alone is insufficient.
        assert size + 64 < int(32000 * 0.9) < int(64000 * 0.9)


@pytest.mark.asyncio
async def test_real_auditor_dispatches_four_and_authenticates_profile(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline")
    active = peak = 0
    calls = []

    async def respond(request):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        calls.append(json.loads(request.content))
        try:
            await asyncio.sleep(0.01)
            response = _valid_response(request)
            response["usage"]["input_tokens"] = 12000
            return httpx.Response(200, json=response)
        finally:
            active -= 1

    ledger = candidates(100, source_size=900)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        kw = dict(
            run_id="expanded", confidence_threshold=0.6, resource_policy=RESOURCE_POLICY
        )
        admitted, receipt = await gate.verify_candidates(ledger, **kw)
        assert peak == 4 and len(calls) > 20
        assert admitted == ledger
        assert validate_extraction_receipt(receipt, admitted)
        require_admission_policy(
            receipt,
            {
                "extraction_confidence_threshold": 0.6,
                "extraction_resource_policy": RESOURCE_POLICY,
            },
        )
        assert receipt["resource_limits"] == resource_limits(
            extraction_config(resource_policy=RESOURCE_POLICY)
        )
        n = len(calls)
        assert await gate.verify_candidates(ledger, **kw) == (admitted, receipt)
        assert len(calls) == n
        with pytest.raises(ValueError, match="budget_exceeded"):
            await gate.verify_candidates(
                ledger, run_id="expanded", confidence_threshold=0.6
            )
        small = candidates(1)
        await gate.verify_candidates(
            small, run_id="small", resource_policy=RESOURCE_POLICY
        )
        with pytest.raises(ValueError, match="identity_conflict"):
            await gate.verify_candidates(small, run_id="small")
        for key in ("resource_policy", "resource_identity", "resource_limits"):
            changed = copy.deepcopy(receipt)
            changed.pop(key)
            assert not validate_extraction_receipt(changed, admitted)
        changed = copy.deepcopy(receipt)
        changed["resource_limits"]["max_requests"] = 201
        assert not validate_extraction_receipt(changed, admitted)
        with pytest.raises(ValueError, match="resource_policy_mismatch"):
            require_admission_policy(receipt, {"extraction_confidence_threshold": 0.6})


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [501, 3100])
async def test_full_pool_is_rejected_before_http_when_any_limit_exceeded(
    tmp_path, monkeypatch, count
):
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: calls.append(r))
    ) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        with pytest.raises(ValueError, match="budget_exceeded"):
            await gate.verify_candidates(
                candidates(count, source_size=6000),
                run_id="too-large",
                resource_policy=RESOURCE_POLICY,
            )
    assert not calls and not (tmp_path / "too-large" / "receipt.json").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [{"usage": False}, {"status": 401}, {"status": 429}])
async def test_expanded_dispatch_preserves_failure_policy(tmp_path, monkeypatch, fault):
    if fault == {"status": 429}:
        admitted, receipt, calls, _ = await run_gate(
            tmp_path,
            monkeypatch,
            ledger=candidates(40, source_size=900),
            confidence_threshold=0.6,
            resource_policy=RESOURCE_POLICY,
            **fault,
        )
        assert admitted and receipt["status"] == "degraded" and len(calls) == 4
        assert validate_extraction_receipt(receipt, admitted)
        return
    with pytest.raises(ValueError, match="fallback_ineligible"):
        await run_gate(
            tmp_path,
            monkeypatch,
            ledger=candidates(40, source_size=900),
            confidence_threshold=0.6,
            resource_policy=RESOURCE_POLICY,
            **fault,
        )
    receipt = json.loads((tmp_path / "partial" / "receipt.json").read_text())
    assert receipt["status"] == "failed_no_replay"
    assert len(receipt["jev"]["batches"]) == 4


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_ui_distinguishes_resource_limits_from_usage(locale):
    receipt = {
        "version": ADMISSION_VERSION,
        "status": "completed",
        "resource_limits": resource_limits(
            extraction_config(resource_policy=RESOURCE_POLICY)
        ),
    }
    text = render_repair_state(
        {"options": {"extractionVerification": receipt}}, locale
    )["text"]
    for value in ("4000000", "200", "32", "4"):
        assert value in text
    assert ("not actual usage" if locale == "en" else "不是实际用量") in text
