"""Keep unverified extraction candidates out of cross-run trusted memory."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.body_quality import EXTRACTION_CLAIM_CONTRACT
from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped
from litagent.quality import build_fact_cards, digest
from litagent.rag.claim_promotion import ClaimsPromoter


async def assessed(*, multiple=False, repeated=False):
    """Obtain final proof through the actual v3 evaluator, mocking transport only."""
    survey, context = materials(2 if multiple else 1)
    for item in context["evidence"].values():
        item.update(
            claim_text="Unverified extractor finding; do not promote.",
            evidence_version="v2",
            section="abstract",
            content_scope="abstract",
            content_hash="original-document-sha",
            source_locator="paper/abstract",
            confidence=0.99,
        )
    if multiple:
        survey = "# Report\n\nViT uses image patches. [E:a0] [E:a1]"
    if repeated:
        survey += "\n\nViT uses image patches. [E:a0]"
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    details = await evaluate_scoped(
        llm=BatchLLM(), survey=survey, context=context, config=EvalConfig()
    )
    assert details["score"] == 1.0
    report = {
        "survey": survey,
        "partial": False,
        "quality": {"status": "passed"},
        "delivery": {"status": "ready", "publishable": True},
        "metadata": {"quality_policy": "strict_auto_v2"},
        "evaluation": {"faithfulness": {"score": 1.0, "details": details}},
    }
    extractions = [
        {
            "claim_contract": EXTRACTION_CLAIM_CONTRACT,
            "evidence_items": list(context["evidence"].values()),
        }
    ]
    return report, extractions


async def promote(report, extractions):
    """Call the production promoter and capture actual trusted-index writes."""
    index = SimpleNamespace(
        upsert_trusted=AsyncMock(side_effect=lambda claims: len(claims))
    )
    summary = await ClaimsPromoter(index).promote(
        run_id="new-run", domain="vision", report_data=report, extractions=extractions
    )
    return summary, index


@pytest.mark.asyncio
async def test_final_supported_statement_promoted_not_unused_or_wrong_candidates():
    report, extractions = await assessed()
    unused = {
        **extractions[0]["evidence_items"][0],
        "evidence_id": "unused",
        "claim_text": "A false, unused finding.",
        "text": "A false, unused finding.",
    }
    extractions[0]["evidence_items"].append(unused)
    before = copy.deepcopy((report, extractions))
    summary, index = await promote(report, extractions)
    assert summary.status == "succeeded" and summary.promoted_count == 1
    claim = index.upsert_trusted.await_args.args[0][0]
    assert claim.text == "ViT uses image patches."
    assert claim.supporting_text == "ViT uses image patches."
    assert claim.confidence is None and claim.evidence_id == "a0"
    assert (report, extractions) == before


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage",
    [
        "report_sha",
        "source_sha",
        "card_id",
        "card_locator",
        "coverage",
        "processed_units",
        "global",
        "expected",
        "processed",
        "duplicate_claim",
        "unsupported",
        "unverifiable",
        "classification",
        "faithfulness",
        "atomic",
        "local_error",
        "position",
        "boolean_position",
        "actual_citations",
        "unknown_citation",
        "support_status",
        "support_quote",
        "support_source",
        "support_position",
        "missing_details",
        "malformed_details",
        "conflicting_evidence",
    ],
)
async def test_bad_or_incomplete_bound_proof_never_writes(damage):
    report, extractions = await assessed()
    d = report["evaluation"]["faithfulness"]["details"]
    c = d["assertions"][0]
    span = c["support_validation"]["spans"][0]
    if damage == "report_sha":
        report["survey"] += " changed"
    elif damage == "source_sha":
        extractions[0]["evidence_items"][0]["supporting_text"] += " changed"
    elif damage == "card_id":
        d["fact_card_map"]["invalid"] = d["fact_card_map"].pop(
            next(iter(d["fact_card_map"]))
        )
    elif damage == "card_locator":
        next(iter(d["fact_card_map"].values()))["source_locator"] = "forged"
    elif damage == "coverage":
        d["coverage"]["complete"] = False
    elif damage == "processed_units":
        d["coverage"]["processed_units"] = 0
    elif damage == "global":
        d["stages"]["global_check"] = "interrupted_no_replay"
    elif damage == "expected":
        d["stages"]["expected_assertions"] += 1
    elif damage == "processed":
        d["stages"]["processed_assertions"] = 0
    elif damage == "duplicate_claim":
        d["assertions"].append(copy.deepcopy(c))
        d["stages"].update(expected_assertions=2, processed_assertions=2)
    elif damage in {"unsupported", "unverifiable"}:
        c["status"] = damage
    elif damage == "classification":
        c["classification_valid"] = False
    elif damage == "faithfulness":
        c["source_faithful"] = False
    elif damage == "atomic":
        c["atomic"] = False
    elif damage == "local_error":
        c["local_error"] = "unknown_evidence_ref"
    elif damage == "position":
        c["source_start"] += 1
    elif damage == "boolean_position":
        c["source_start"] = True
    elif damage == "actual_citations":
        c["evidence_ids"] = []
    elif damage == "unknown_citation":
        c["evidence_ids"] = ["unknown"]
    elif damage == "support_status":
        c["support_validation"]["status"] = "unverified"
    elif damage == "support_quote":
        span["quote"] = "invented"
    elif damage == "support_source":
        span["source"] = "evidence_text"
    elif damage == "support_position":
        span["source_end"] += 1
    elif damage == "missing_details":
        report["evaluation"] = {}
    elif damage == "malformed_details":
        report["evaluation"] = 42
    elif damage == "conflicting_evidence":
        extractions[0]["evidence_items"].append(
            {**extractions[0]["evidence_items"][0], "claim_text": "conflict"}
        )
    summary, index = await promote(report, extractions)
    assert summary.status == "skipped"
    index.upsert_trusted.assert_not_awaited()


@pytest.mark.asyncio
async def test_supported_cross_source_comparison_not_squeezed_into_single_source():
    report, extractions = await assessed(multiple=True)
    summary, index = await promote(report, extractions)
    assert summary.status == "skipped"
    index.upsert_trusted.assert_not_awaited()


@pytest.mark.asyncio
async def test_repeated_supported_statement_is_written_once():
    report, extractions = await assessed(repeated=True)
    summary, index = await promote(report, extractions)
    assert summary.promoted_count == 1
    assert len(index.upsert_trusted.await_args.args[0]) == 1


@pytest.mark.asyncio
async def test_grounded_candidate_without_final_proof_cannot_use_legacy_path():
    report, extractions = await assessed()
    report.pop("metadata")
    report.pop("evaluation")
    summary, index = await promote(report, extractions)
    assert summary.status == "skipped"
    index.upsert_trusted.assert_not_awaited()


@pytest.mark.asyncio
async def test_supported_same_source_references_preserve_one_source_and_statement():
    report, extractions = await assessed(multiple=True)
    items = extractions[0]["evidence_items"]
    items[1]["paper_id"] = items[0]["paper_id"]
    # Re-evaluate the changed source identity; old proof must not be reused.
    from litagent.evidence import collect_ledger

    ledger = collect_ledger(extractions)
    cards = build_fact_cards(ledger)
    _, context = materials(2)
    context.update(
        evidence=ledger, fact_cards=cards, evidence_snapshot_sha256=digest(cards)
    )
    report["evaluation"]["faithfulness"]["details"] = await evaluate_scoped(
        llm=BatchLLM(), survey=report["survey"], context=context, config=EvalConfig()
    )
    summary, index = await promote(report, extractions)
    assert summary.promoted_count == 1
    assert index.upsert_trusted.await_args.args[0][0].paper_id == "paper0"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error", [RuntimeError("index unavailable"), asyncio.CancelledError()]
)
async def test_trusted_index_failure_preserves_cancellation_and_degraded_contract(
    error,
):
    report, extractions = await assessed()
    index = SimpleNamespace(upsert_trusted=AsyncMock(side_effect=error))
    promoter = ClaimsPromoter(index)
    if isinstance(error, asyncio.CancelledError):
        with pytest.raises(type(error)):
            await promoter.promote(
                run_id="r", domain="v", report_data=report, extractions=extractions
            )
    else:
        summary = await promoter.promote(
            run_id="r", domain="v", report_data=report, extractions=extractions
        )
        assert (
            summary.status == "degraded"
            and summary.reason_code == "claims_upsert_failed"
        )
