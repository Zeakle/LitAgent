"""Locate copied body findings without authorizing partial ambiguous edits."""

from litagent.eval.scoped import build_source_fragments
from litagent.quality import bind_review_feedback
from litagent.report_render import build_report_source_units


def test_v2_binds_every_exact_complete_occurrence_and_v1_remains_ambiguous():
    text = "EViT improves speed by 50% on DeiT-S. [E:a]"
    report = f"# Report\n\n## Abstract\n\n{text}\n\n## Methods\n\n{text}"
    fragment = build_source_fragments(build_report_source_units(report)[0])[0]
    quote = fragment["source_quote"]
    feedback = {
        "evidence_compliance": {
            "unsupported_claims": [
                {
                    "source_quote": quote,
                    "existing_evidence_ids": ["a"],
                    "issue": "Check actual source conditions.",
                    "source_start": 999,
                }
            ]
        }
    }
    assert not bind_review_feedback(report, feedback)["bound_feedback"]
    result = bind_review_feedback(report, feedback, allow_identical_occurrences=True)
    assert len(result["bound_feedback"]) == 2
    for item in result["bound_feedback"]:
        assert report[item["source_start"] : item["source_end"]] == quote
        assert item["source_start"] != 999
    editorial = {
        "issues": [
            {
                "source_quote": quote,
                "existing_evidence_ids": ["a"],
                "issue": "Shorten the abstract.",
            }
        ]
    }
    assert not bind_review_feedback(
        report, editorial, allow_identical_occurrences=True
    )["bound_feedback"]


def test_partial_repetition_or_wrong_citation_set_does_not_authorize_edits():
    report = "# Report\n\n## Abstract\n\nEViT improves speed. [E:a]\n\n## Methods\n\nEViT improves speed on DeiT-S. [E:a]"
    for quote, refs in [
        ("EViT improves speed", []),
        ("EViT improves speed. [E:a]", ["wrong"]),
    ]:
        result = bind_review_feedback(
            report,
            {"issues": [{"source_quote": quote, "existing_evidence_ids": refs}]},
            allow_identical_occurrences=True,
        )
        assert not result["bound_feedback"]
