"""Verify optional extraction receipts reach real frozen Survey consumers."""

import copy

import httpx
import pytest
import test_writer_source_view_pipeline as writer_view
from test_extraction_admission import dual_transport
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.extraction_verification import ADMISSION_VERSION, ExtractionVerifier
from litagent.frozen_writing import frozen_options
from litagent.workbench import default_verification_options


def test_frozen_historical_defaults_do_not_enable_a_paid_gate():
    assert not frozen_options().verification.jev
    assert frozen_options().verification.version == "optional-verification-v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_receipt", [True, False])
async def test_frozen_gate_is_reused_before_real_writing_and_acceptance(
    source_case, tmp_path, monkeypatch, with_receipt
):
    import litagent.frozen_writing as frozen

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    real = frozen.run_frozen
    captured = []
    client = httpx.AsyncClient(transport=dual_transport(captured))
    gate = ExtractionVerifier(
        tmp_path / "jev",
        protocol=ADMISSION_VERSION,
        auditor=JevCitationAuditor(client=client),
    )
    options = default_verification_options()
    receipts = []

    async def run(agent, bundle, run_id, **kwargs):
        bundle = copy.deepcopy(bundle)
        if with_receipt:
            ledger = bundle["selection"]["selected_items"]
            admitted, receipt = await gate.verify_candidates(
                ledger,
                run_id=run_id,
                failure_policy=options.extraction_failure_policy,
                confidence_threshold=options.extraction_confidence_threshold,
                resource_policy=options.extraction_resource_policy,
            )
            assert admitted == ledger
            receipts.append(receipt)
            bundle["extractions"][0]["extraction_verification"] = receipt
        report = await real(
            agent, bundle, run_id, verification=options.model_dump(), **kwargs
        )
        if not with_receipt:
            assert not report["delivery"]["publishable"]
            assert agent._cost_budget.snapshot()["requests_sent"] == 0
            raise RuntimeError("offline_missing_receipt_rejected")
        return report

    monkeypatch.setattr(frozen, "run_frozen", run)
    try:
        if not with_receipt:
            with pytest.raises(RuntimeError, match="offline_missing_receipt_rejected"):
                await writer_view.run_scientific_case(
                    source_case, tmp_path / "run", monkeypatch
                )
            assert not captured
        else:
            outcome = await writer_view.run_scientific_case(
                source_case, tmp_path / "run", monkeypatch
            )
            assert outcome.report["metadata"]["verification"] == options.model_dump()
            assert outcome.report["metadata"]["extraction_verification"] == receipts[0]
            assert len(captured) == 1
            assert outcome.report["delivery"]["publishable"]
    finally:
        await client.aclose()
