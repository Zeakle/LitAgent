"""Regression tests for paired Survey benchmark comparisons."""

from __future__ import annotations

import hashlib

import pytest
from test_phase14_5 import _artifact, _case

from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.survey_metrics import (
    aggregate_profiles,
    evaluate_survey_artifact,
)
from litagent.benchmark.survey_models import SurveyCaseObservation
from litagent.benchmark.survey_runner import _profile_comparisons
from litagent.context.evidence_selector import EvidenceSelection
from litagent.evidence import (
    build_citation_aliases,
    decode_citation_aliases,
    encode_citation_aliases,
)
from litagent.llm.client import MockLLMClient
from litagent.rag.manifest import ManifestPaper
from litagent.rag.models import (
    ContentScope,
    PaperCandidate,
    SourceKind,
    SourceRef,
    merge_paper_candidates,
)
from litagent.report_render import (
    align_claim_to_source,
    build_reader_report,
    build_report_source_units,
)
from litagent.workbench import ReportVersion


def _observation(case_id: str, profile: str, coverage: float, repetition: int = 1):
    metrics = evaluate_survey_artifact(case=_case(), artifact=_artifact())
    metrics.coverage.evidence = coverage
    return SurveyCaseObservation(
        run_id=f"{case_id}-{profile}-{repetition}",
        case_id=case_id,
        domain="few_shot",
        profile_id=profile,
        repetition=repetition,
        status="completed",
        metrics=metrics,
        external_input_fingerprint=f"source-{case_id}-{repetition}",
    )


def test_profile_delta_uses_only_matching_case_repetitions():
    observations = [
        _observation("a", "left", 1.0),
        _observation("b", "left", 0.0),
        _observation("c", "left", 0.0),
        _observation("a", "right", 0.0),
        _observation("b", "right", 0.0),
    ]
    _, summaries = aggregate_profiles(observations)

    comparison = _profile_comparisons(observations, summaries)[0]

    assert comparison.comparable is True
    assert comparison.shared_cohort_count == 2
    assert comparison.metric_deltas["coverage_evidence"] == pytest.approx(0.5)
    assert comparison.metric_deltas["latency_p95_ms"] is None


def test_duplicate_cohort_identity_is_rejected():
    observations = [
        _observation("a", "left", 1.0),
        _observation("a", "left", 0.0),
        _observation("a", "right", 1.0),
    ]
    _, summaries = aggregate_profiles(observations)

    with pytest.raises(ValueError, match="duplicate benchmark cohort"):
        _profile_comparisons(observations, summaries)


@pytest.mark.asyncio
async def test_original_query_reaches_generation_review_revision_and_rewrite():
    query = "Compare ViT with CNN on image classification."
    selection = EvidenceSelection(
        candidate_count=0,
        selected_items={},
        section_evidence_ids={},
        method_by_section={},
        omitted_count=0,
        estimated_tokens=0,
    )
    synthesis = SynthesisWorker(MockLLMClient())
    context, _ = await synthesis._build_synthesis_context(
        query=query, extractions=[], graph_data={}, selection=selection
    )
    review, _ = await ReviewerWorker(MockLLMClient())._build_review_context(
        "# Draft", selection, query=query
    )
    revision = synthesis.revise("# Draft", "Fix scope", selection, query=query)
    rewrite = synthesis.rewrite_with_evidence(
        "# Draft", "ledger", "diagnostics", query=query
    )

    for message in (context, review, revision[1]["content"], rewrite[1]["content"]):
        assert query in message


def test_verified_abstract_wins_over_later_unverified_summary():
    original = "This is the complete source abstract with methods and results."
    source = SourceRef(
        kind=SourceKind.ARXIV_METADATA,
        source_id="2010.11929",
        uri="https://arxiv.org/abs/2010.11929",
        sha256=hashlib.sha256(original.encode("utf-8")).hexdigest(),
    )
    verified = PaperCandidate(
        paper_id="arxiv:2010.11929",
        title="ViT",
        abstract=original,
        source="rag_index",
        content_scope=ContentScope.ABSTRACT,
        provenance=[source],
    )
    summary = PaperCandidate(
        paper_id="arxiv:2010.11929",
        title="ViT",
        abstract="Short search summary.",
        source="huggingface",
        content_scope=ContentScope.ABSTRACT,
    )

    merged = merge_paper_candidates([verified, summary])[0]

    assert merged.abstract == original


def test_manifest_rejects_false_verified_abstract_hash():
    with pytest.raises(ValueError, match="abstract_sha256"):
        ManifestPaper(
            paper_id="arxiv:2010.11929",
            arxiv_id="2010.11929",
            title="ViT",
            abstract="Summary",
            abstract_source="arxiv_api",
            abstract_source_version="arxiv_atom_v1",
            abstract_url="https://arxiv.org/abs/2010.11929",
            abstract_accessed_at="2026-09-19",
            abstract_sha256="0" * 64,
        )


def test_citation_aliases_round_trip_without_changing_prose_or_unknown_ids():
    canonical = "arxiv:2010.11929:abstract:claim:0123456789ab"
    aliases = build_citation_aliases([canonical])
    original = f"ViT result [E:{canonical}]. Unknown [E:bogus]."

    encoded = encode_citation_aliases(original, aliases)
    decoded = decode_citation_aliases(encoded.text, aliases)

    assert encoded.text == "ViT result [E:e1]. Unknown [E:bogus]."
    assert decoded.text == original
    assert [item.reason_code for item in decoded.rejected] == ["unknown_evidence_id"]


def test_claim_alignment_never_borrows_a_neighboring_sentence_citation():
    report = "# Survey\n\nClaim A [E:e1]. Claim B [E:e2]."
    units = build_report_source_units(report)

    first = align_claim_to_source("Claim A", units)
    second = align_claim_to_source("Claim B", units)

    assert first.citation_scope == "exact"
    assert first.evidence_ids == ("e1",)
    assert second.evidence_ids == ("e2",)
    assert align_claim_to_source("Invented claim", units).status == "unmapped"


def test_reader_projection_numbers_papers_but_preserves_source_version():
    report = ReportVersion(
        run_id="r1",
        revision=1,
        report_language="en",
        survey="# Survey\n\nClaim A [E:ev-a]. Claim B [E:ev-b].",
    )
    evidence = {
        "ev-a": {"paper_id": "arxiv:2010.11929", "paper_title": "ViT"},
        "ev-b": {"paper_id": "arxiv:2010.11929", "paper_title": "ViT"},
    }

    reader = build_reader_report(
        report,
        evidence_ledger=evidence,
        paper_catalog={"arxiv:2010.11929": {"title": "ViT", "year": 2020}},
    )

    assert "Claim A [1]. Claim B [1]." in reader.markdown
    assert reader.markdown.count("## References") == 1
    assert reader.citation_occurrences == ((1, "ev-a"), (1, "ev-b"))
    assert reader.source_sha256 == report.content_sha256
    assert report.survey.endswith("[E:ev-b].")


def test_controlled_downstream_requires_actual_start_and_end_events():
    from litagent.benchmark.survey_metrics import _controlled_downstream_ms

    artifact = {
        "events": [{"event": "survey.complete", "at": "2026-01-01T00:00:02+00:00"}]
    }
    assert _controlled_downstream_ms(artifact) is None
    artifact["events"].insert(
        0, {"event": "external_inputs_ready", "at": "2026-01-01T00:00:01+00:00"}
    )
    assert _controlled_downstream_ms(artifact) == 1000
    artifact["events"][1]["at"] = "2025-12-31T23:59:59+00:00"
    assert _controlled_downstream_ms(artifact) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("query", ["", "   ", "x" * 2001])
async def test_runner_rejects_invalid_research_question_before_wiring(query):
    from litagent.runner import LitAgent

    with pytest.raises(ValueError, match="invalid_research_question"):
        await object.__new__(LitAgent).run(query)
