"""Exercise explicit admission thresholds through real decoding and consumers."""

import copy
import json

import httpx
import pytest
from test_extraction_fallback import run_gate
from test_extraction_verification import selection
from test_jev_citation_audit import _valid_response
from test_quality_view import render_repair_state

from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.config import ContextConfig, load_config
from litagent.extraction_verification import (
    ADMISSION_VERSION,
    FALLBACK_POLICY,
    ExtractionVerifier,
    require_admission_policy,
    validate_extraction_receipt,
)
from litagent.workbench import (
    RunOptions,
    VerificationOptions,
    default_verification_options,
)


async def verify(tmp_path, monkeypatch, *, confidence, threshold=0.6, choice="ok"):
    """Replace HTTP only, allowing the auditor to derive its own uncertain flag."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline")
    calls = []

    def respond(request):
        calls.append(json.loads(request.content))
        response = _valid_response(request)
        for answer in response["answers"].values():
            answer["choice"] = choice
            answer["probabilities"] = {
                key: float(key == choice) for key in answer["probabilities"]
            }
            answer["confidence"] = confidence
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        ledger = {"a": selection().selected_items["a"]}
        admitted, receipt = await gate.verify_candidates(
            ledger,
            run_id="threshold",
            failure_policy=FALLBACK_POLICY,
            confidence_threshold=threshold,
        )
        assert validate_extraction_receipt(receipt, admitted)
        assert await gate.verify_candidates(
            ledger,
            run_id="threshold",
            failure_policy=FALLBACK_POLICY,
            confidence_threshold=threshold,
        ) == (admitted, receipt)
        with pytest.raises(ValueError, match="identity_conflict"):
            await gate.verify_candidates(
                ledger,
                run_id="threshold",
                failure_policy=FALLBACK_POLICY,
                confidence_threshold=0.9 if threshold == 0.6 else 0.6,
            )
        assert len(calls) == 1
        return admitted, receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confidence,threshold,choice,accepted",
    [
        (0.6, 0.6, "ok", True),
        (0.599, 0.6, "ok", False),
        (0.89, None, "ok", False),
        (0.89, 0.6, "ok", True),
        (0.99, 0.6, "not_ok", False),
        (0.99, 0.6, "uncertain", False),
    ],
)
async def test_decoder_and_admission_share_effective_threshold(
    tmp_path, monkeypatch, confidence, threshold, choice, accepted
):
    admitted, receipt = await verify(
        tmp_path,
        monkeypatch,
        confidence=confidence,
        threshold=threshold,
        choice=choice,
    )
    assert bool(admitted) is accepted
    assert receipt.get("confidence_threshold") == threshold
    if choice == "ok":
        effective = threshold if threshold is not None else 0.9
        assert all(
            d["uncertain"] == (confidence < effective)
            for d in receipt["jev"]["decisions"]
        )


@pytest.mark.asyncio
async def test_threshold_is_authenticated_in_receipt_and_run_policy(
    tmp_path, monkeypatch
):
    admitted, receipt = await verify(tmp_path, monkeypatch, confidence=0.75)
    options = default_verification_options().model_dump()
    options.pop("extraction_resource_policy")
    require_admission_policy(receipt, options)
    for field in ("confidence_threshold", "confidence_identity"):
        changed = copy.deepcopy(receipt)
        changed.pop(field)
        assert not validate_extraction_receipt(changed, admitted)
    changed = copy.deepcopy(receipt)
    changed["confidence_threshold"] = 0.5
    assert not validate_extraction_receipt(changed, admitted)
    options.pop("extraction_confidence_threshold")
    with pytest.raises(ValueError, match="confidence_threshold_mismatch"):
        require_admission_policy(receipt, options)


@pytest.mark.asyncio
async def test_partial_admission_uses_threshold_without_retries(tmp_path, monkeypatch):
    admitted, receipt, calls, _ = await run_gate(
        tmp_path, monkeypatch, confidence_threshold=0.6, fault="low"
    )
    assert admitted and len(calls) == 2
    assert receipt["status"] == "degraded"
    assert receipt["confidence_threshold"] == 0.6
    assert validate_extraction_receipt(receipt, admitted)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [{"usage": False}, {"status": 401}])
async def test_lower_threshold_does_not_bypass_transport_safeguards(
    tmp_path, monkeypatch, fault
):
    with pytest.raises(ValueError, match="fallback_ineligible"):
        await run_gate(tmp_path, monkeypatch, confidence_threshold=0.6, **fault)
    receipt = json.loads((tmp_path / "partial" / "receipt.json").read_text())
    assert receipt["status"] == "failed_no_replay"
    assert not validate_extraction_receipt(receipt, {})


@pytest.mark.parametrize("value", [-0.1, 1.1, True, "0.6", float("nan")])
def test_invalid_confidence_policy_is_rejected(value):
    with pytest.raises(ValueError):
        VerificationOptions(
            version="optional-verification-v2", extraction_confidence_threshold=value
        )


def test_old_threshold_and_fingerprint_are_unchanged():
    old = VerificationOptions(
        version="optional-verification-v2",
        jev=True,
        extraction_failure_policy=FALLBACK_POLICY,
    )
    assert "extraction_confidence_threshold" not in old.model_dump()
    assert (
        RunOptions(quality_policy="strict_auto_v2", verification=old).fingerprint()
        == "cbe7f3e853952e26383cfe3674c3bc3eff2939913a863084c8a6c5dbc5ba738f"
    )
    current = default_verification_options()
    assert current.extraction_confidence_threshold == 0.6
    assert (
        RunOptions(quality_policy="strict_auto_v2", verification=current).fingerprint()
        != RunOptions(quality_policy="strict_auto_v2", verification=old).fingerprint()
    )
    with pytest.raises(ValueError, match="requires_admission_v2"):
        VerificationOptions(extraction_confidence_threshold=0.6)


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
@pytest.mark.parametrize("threshold", [None, 0.6])
def test_quality_view_reads_recorded_threshold(locale, threshold):
    receipt = {
        "version": ADMISSION_VERSION,
        "candidate_ledger": {"a": {}},
        "accepted_evidence_ids": ["a"] if threshold else [],
        "jev": {
            "decisions": [
                {
                    "kind": k,
                    "choice": "ok",
                    "confidence": 0.75,
                    "uncertain": not threshold,
                }
                for k in ("claim_validity", "claim_format")
            ]
        },
    }
    if threshold is not None:
        receipt["confidence_threshold"] = threshold
    output = render_repair_state(
        {"options": {"extractionVerification": receipt}}, locale
    )["text"]
    assert f"≥ {threshold if threshold is not None else 0.9}" in output
    assert ("1 / 1" if threshold else "0 / 1") in output


def test_configured_section_recommendations_are_twenty():
    assert ContextConfig().evidence_top_k_per_section == 20
    for path in ("config/default.yaml", "config/expanded-capacity.yaml"):
        context = load_config(path).context
        assert context.evidence_top_k_per_section == 20
        assert context.evidence_max_items == 60
        assert context.evidence_per_paper_cap == 4
        assert context.synthesis_evidence_max_tokens == 16000
