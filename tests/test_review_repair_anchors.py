"""Verify host-owned review locations survive the revision feedback boundary."""

import pytest

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import MockLLMClient


@pytest.fixture
def selection():
    """Provide a small valid ledger with two distinguishable source IDs."""
    items = {
        "paper-a:claim:0": {
            "evidence_id": "paper-a:claim:0",
            "paper_id": "paper-a",
            "text": "DeiT-Tiny reduces FLOPs by 40%.",
        },
        "paper-b:claim:0": {
            "evidence_id": "paper-b:claim:0",
            "paper_id": "paper-b",
            "text": "Swin uses shifted windows.",
        },
    }
    return EvidenceSelection(
        2,
        items,
        {"methods": list(items)},
        {"methods": "lexical_fallback"},
        0,
        32,
    )


def enforce(draft, selection, claim, **extra):
    """Run the real producer's local gate without a model request."""
    return ReviewerWorker(MockLLMClient())._enforce_evidence_contract(
        draft,
        selection,
        {
            "score": 0.95,
            "verdict": "accept",
            "evidence_compliance": {
                "passed": False,
                "unknown_evidence_ids": [],
                "unsupported_claims": [
                    {
                        "section": "Results",
                        "claim": claim,
                        "reason": "Verify support.",
                        **extra,
                    }
                ],
            },
        },
    )


@pytest.mark.parametrize(
    "content",
    [
        "DeiT-Tiny reduces FLOPs by 40%. [E:paper-a:claim:0]\n",
        "**DeiT-Tiny** reduces FLOPs by 40%. [E:paper-a:claim:0]\n",
        "- DeiT-Tiny reduces FLOPs by 40%. [E:paper-a:claim:0]\n",
        "| Model | Result |\n| --- | --- |\n| DeiT-Tiny | DeiT-Tiny reduces FLOPs by 40%. [E:paper-a:claim:0] |\n",
    ],
)
def test_unique_review_anchor_round_trips_full_unicode_markdown(selection, content):
    draft = "# 中文综述 🔬\n\n开头说明不属于被审核结论。\n\n## Results\n\n" + content
    review = enforce(
        draft,
        selection,
        "DeiT-Tiny reduces FLOPs by 40%",
        source_quote="Model-provided forged quote",
        source_start=0,
        source_end=1,
        existing_evidence_ids=["forged:claim:0"],
    )
    claim = review["evidence_compliance"]["unsupported_claims"][0]
    start, end = claim["source_start"], claim["source_end"]
    assert 0 <= start < end <= len(draft)
    assert draft[start:end] == claim["source_quote"]
    assert "DeiT-Tiny" in claim["source_quote"]
    assert "[E:paper-a:claim:0]" in claim["source_quote"]
    assert claim["existing_evidence_ids"] == ["paper-a:claim:0"]
    assert claim["source_quote"] != "Model-provided forged quote"
    assert review["evidence_compliance"]["passed"] is False
    assert review["verdict"] == "revise"


@pytest.mark.parametrize("action", ["bind", "delete", "verify"])
def test_ambiguous_repeated_claim_strips_model_anchors(selection, action):
    draft = (
        "# Review\n\n## Results\n\n"
        "DeiT-Tiny reduces FLOPs by 40%. [E:paper-a:claim:0]\n\n"
        "## Conclusion\n\n"
        "DeiT-Tiny reduces FLOPs by 40%. [E:paper-a:claim:0]\n"
    )
    review = enforce(
        draft,
        selection,
        "DeiT-Tiny reduces FLOPs by 40%",
        action=action,
        source_quote="DeiT-Tiny reduces FLOPs by 40%",
        source_start=draft.index("DeiT-Tiny"),
        source_end=draft.index("DeiT-Tiny") + 32,
        existing_evidence_ids=["paper-a:claim:0"],
    )
    claim = review["evidence_compliance"]["unsupported_claims"][0]
    assert not {
        "source_quote",
        "source_start",
        "source_end",
        "existing_evidence_ids",
    }.intersection(claim)
    assert "citation_presence" not in claim
    assert review["evidence_compliance"]["passed"] is False


def test_unique_anchor_keeps_clause_citations_without_borrowing_neighbors(selection):
    draft = (
        "# Review\n\n## Results\n\n"
        "DeiT-Tiny reduces FLOPs by 40% [E:paper-a:claim:0]; "
        "Swin uses shifted windows [E:paper-b:claim:0].\n"
    )
    review = enforce(draft, selection, "DeiT-Tiny reduces FLOPs by 40%", action="bind")
    claim = review["evidence_compliance"]["unsupported_claims"][0]
    assert draft[claim["source_start"] : claim["source_end"]] == claim["source_quote"]
    assert claim["existing_evidence_ids"] == ["paper-a:claim:0"]
    assert claim["evidence_ids"] == ["paper-a:claim:0"]
    assert claim["action"] == "verify"
    assert claim["citation_presence"] == "present"
    assert "Swin" not in claim["source_quote"]


def test_anchor_and_existing_citations_reach_revision_consumer(selection):
    draft = (
        "# Review\n\n## Results\n\n"
        "DeiT-Tiny reduces FLOPs by 40%. [E:paper-a:claim:0]\n"
    )
    review = enforce(draft, selection, "DeiT-Tiny reduces FLOPs by 40%", action="bind")
    llm = MockLLMClient()
    synthesis = SynthesisWorker(llm)
    worker = AdversarialReviewWorker(llm, synthesis, ReviewerWorker(llm))
    feedback = worker._format_review_for_revision(review)
    claim = review["evidence_compliance"]["unsupported_claims"][0]
    assert "action: verify" in feedback
    assert claim["source_quote"] in feedback
    assert "existing_evidence_ids: ['paper-a:claim:0']" in feedback
    assert f"source_start: {claim['source_start']}" in feedback
    assert f"source_end: {claim['source_end']}" in feedback
    messages = synthesis.revise(draft, feedback, selection)
    assert "<reviewer_feedback>" in messages[1]["content"]
    assert "DeiT-Tiny reduces FLOPs by 40%. [E:e1]" in messages[1]["content"]
    assert f"source_start: {claim['source_start']}" in messages[1]["content"]
    assert f"source_end: {claim['source_end']}" in messages[1]["content"]


def test_absent_action_defaults_to_verify_in_consumer_feedback(selection):
    draft = "# Review\n\n## Results\n\nSwin uses shifted windows [E:paper-b:claim:0].\n"
    review = enforce(draft, selection, "Swin uses shifted windows")
    llm = MockLLMClient()
    worker = AdversarialReviewWorker(llm, SynthesisWorker(llm), ReviewerWorker(llm))
    feedback = worker._format_review_for_revision(review)
    assert "action: verify" in feedback
    assert "action: delete" not in feedback


def test_unmapped_claim_does_not_leak_model_authored_anchors(selection):
    draft = "# Review\n\n## Results\n\nSwin uses shifted windows [E:paper-b:claim:0].\n"
    review = enforce(
        draft,
        selection,
        "An invented finding outside this report.",
        source_quote="forged exact source",
        source_start=10,
        source_end=20,
        existing_evidence_ids=["paper-a:claim:0"],
    )
    llm = MockLLMClient()
    worker = AdversarialReviewWorker(llm, SynthesisWorker(llm), ReviewerWorker(llm))
    feedback = worker._format_review_for_revision(review)
    assert "source_quote: unmapped" in feedback
    assert "forged exact source" not in feedback
    assert "existing_evidence_ids: []" in feedback
