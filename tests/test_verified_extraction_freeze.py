"""Freeze all accepted evidence while retaining independent failed body gates."""

import copy
from unittest.mock import AsyncMock

import pytest
from test_extraction_verification import selection, verify
from test_strict_automatic import frozen, passing_audit

from litagent.config import load_config
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.eval.base import CTX_EVIDENCE
from litagent.extraction_verification import verified_extraction_ledger
from litagent.quality import build_fact_cards, digest
from litagent.runner import LitAgent


async def ready(tmp_path, monkeypatch):
    selected, receipt, _ = await verify(tmp_path, monkeypatch, run_id="test-auto")
    report = copy.deepcopy(frozen()["report"])
    report["metadata"].update(
        quality_policy="strict_auto_v2",
        extraction_verification=receipt,
        fact_cards=build_fact_cards(selected.selected_items),
    )
    report["metadata"]["evidence_snapshot_sha256"] = digest(
        report["metadata"]["fact_cards"]
    )
    report["evaluation"]["faithfulness"]["details"]["quality_policy"] = "strict_auto_v2"
    available = selection().selected_items
    available["other"] = {
        "paper_id": "other",
        "evidence_id": "other",
        "text": "Other study",
        "supporting_text": "Other source",
        "chunk_key": "abstract",
    }
    return report, available, receipt, selected


@pytest.mark.asyncio
async def test_full_available_inventory_freezes_only_all_verified_accepted_sources(
    tmp_path, monkeypatch
):
    report, available, receipt, selected = await ready(tmp_path, monkeypatch)
    before = copy.deepcopy((report, available, receipt))
    assert (
        verified_extraction_ledger(receipt, available, run_id="test-auto")
        == selected.selected_items
    )
    snapshot = freeze_automatic_inputs(report, available, run_id="test-auto")
    assert snapshot["ledger"] == selected.selected_items
    audit = passing_audit(snapshot)
    audit["jev"] = {"status": "not_requested"}
    accepted = automatic_acceptance(snapshot, report["evaluation"], audit)
    assert accepted["publishable"] and accepted["gates"]["extraction_jev"] == "passed"
    assert (report, available, receipt) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["missing", "changed", "duplicate", "foreign_run", "receipt", "rejected_card"],
)
async def test_projection_does_not_make_bad_originals_or_receipts_verified(
    tmp_path, monkeypatch, mutation
):
    report, available, receipt, selected = await ready(tmp_path, monkeypatch)
    run_id = "test-auto"
    if mutation == "missing":
        available.pop("a")
    elif mutation == "changed":
        available["a"]["supporting_text"] = "Different source"
    elif mutation == "duplicate":
        receipt["accepted_evidence_ids"].append("a")
    elif mutation == "foreign_run":
        run_id = "foreign"
    elif mutation == "receipt":
        receipt["jev"]["decisions"][0]["confidence"] = 0.1
    else:
        report["metadata"]["fact_cards"] = build_fact_cards(selection().selected_items)
        report["metadata"]["evidence_snapshot_sha256"] = digest(
            report["metadata"]["fact_cards"]
        )
    with pytest.raises(ValueError):
        freeze_automatic_inputs(report, available, run_id=run_id)


@pytest.mark.asyncio
async def test_actual_failed_preflight_preserves_passed_extraction_without_dispatch(
    tmp_path, monkeypatch
):
    report, available, receipt, _ = await ready(tmp_path, monkeypatch)
    report["quality"]["status"] = "failed"
    report["evaluation"]["faithfulness"]["skipped"] = True
    service = AsyncMock()
    agent = LitAgent(load_config(), audit_service=service)
    agent._session_id = "test-auto"
    agent._build_eval_context = lambda **kwargs: {CTX_EVIDENCE: available}
    await agent._automatic_review(report, "EViT", {}, None)
    acceptance = report["metadata"]["automatic_acceptance"]
    assert acceptance["gates"] == {
        "evaluation": "failed",
        "judge": "blocked",
        "extraction_jev": "passed",
    }
    assert not acceptance["publishable"]
    assert acceptance["extraction_ledger_contract"] == "accepted-extraction-ledger-v1"
    assert "extraction_jev_not_passed" not in acceptance["failure_codes"]
    service.submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_unused_accepted_evidence_is_retained_in_the_frozen_snapshot(
    tmp_path, monkeypatch
):
    import httpx
    from test_jev_citation_audit import _valid_response

    from litagent.benchmark.citation_audit import JevCitationAuditor
    from litagent.extraction_verification import ExtractionVerifier

    selected_input = selection()
    selected_input.selected_items["b"]["text"] = selected_input.selected_items["b"][
        "supporting_text"
    ]
    captured = []
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")

    def respond(request):
        captured.append(request)
        return httpx.Response(200, json=_valid_response(request))

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        selected, receipt = await ExtractionVerifier(
            tmp_path, auditor=JevCitationAuditor(client=client)
        ).verify(selected_input, run_id="test-auto")
    assert set(selected.selected_items) == {"a", "b"}
    report = copy.deepcopy(frozen()["report"])
    report["metadata"].update(
        quality_policy="strict_auto_v2",
        extraction_verification=receipt,
        fact_cards=build_fact_cards(selected.selected_items),
    )
    report["metadata"]["evidence_snapshot_sha256"] = digest(
        report["metadata"]["fact_cards"]
    )
    report["evaluation"]["faithfulness"]["details"]["quality_policy"] = "strict_auto_v2"
    snapshot = freeze_automatic_inputs(
        report, selected_input.selected_items, run_id="test-auto"
    )
    assert set(snapshot["ledger"]) == {"a", "b"}
    assert all(
        "b" not in c["evidence_ids"] for c in snapshot["frozen_inventory"]["assertions"]
    )


@pytest.mark.asyncio
async def test_citing_rejected_material_cannot_be_saved_by_global_available_sources(
    tmp_path, monkeypatch
):
    report, available, _, _ = await ready(tmp_path, monkeypatch)
    report["survey"] = report["survey"].replace("[E:a]", "[E:b]")
    claim = report["evaluation"]["faithfulness"]["details"]["assertions"][0]
    claim["source_quote"] = report["survey"]
    claim["evidence_ids"] = ["b"]
    with pytest.raises(ValueError, match="automatic_original_support_missing"):
        freeze_automatic_inputs(report, available, run_id="test-auto")


@pytest.mark.asyncio
async def test_older_auto_policy_keeps_failed_preflight_before_context_build(
    tmp_path, monkeypatch
):
    from unittest.mock import Mock

    report, _, _, _ = await ready(tmp_path, monkeypatch)
    report["metadata"]["quality_policy"] = "strict_auto_v1"
    report["quality"]["status"] = "failed"
    service = AsyncMock()
    agent = LitAgent(load_config(), audit_service=service)
    agent._session_id = "test-auto"
    agent._build_eval_context = Mock(
        side_effect=AssertionError("old preflight should not build context")
    )
    await agent._automatic_review(report, "EViT", {}, None)
    agent._build_eval_context.assert_not_called()
    service.submit.assert_not_awaited()
    assert (
        "automatic_evaluation_not_passed"
        in report["metadata"]["automatic_acceptance"]["failure_codes"]
    )
