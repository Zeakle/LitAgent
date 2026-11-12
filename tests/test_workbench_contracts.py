"""Verify the local workbench's fail-closed interaction contracts."""

import pytest
from pydantic import ValidationError

from litagent.workbench import (
    HumanDecision,
    ReportVersion,
    ResumeIdentity,
    ReviewRequest,
    RunOptions,
    can_publish,
    validate_decision,
)


def request(stage="paper_selection", **kwargs):
    """Build a server-owned pending review request."""
    return ReviewRequest(
        request_id="review-1",
        run_id="run-1",
        stage=stage,
        revision=1,
        candidate_paper_ids=("p1", "p2"),
        **kwargs,
    )


def decision(action="confirm", **kwargs):
    """Build a version-bound client decision."""
    return HumanDecision(request_id="review-1", revision=1, action=action, **kwargs)


def test_options_preserve_legacy_defaults():
    assert RunOptions().report_language == "en"
    assert not RunOptions().human_review
    assert (
        RunOptions().fingerprint() != RunOptions(report_language="zh-CN").fingerprint()
    )
    with pytest.raises(ValidationError):
        RunOptions(report_language="fr")


def test_selection_cannot_inject_unknown_papers():
    validate_decision(request(), decision(selected_paper_ids=("p2",)))
    with pytest.raises(ValueError, match="unknown_paper"):
        validate_decision(request(), decision(selected_paper_ids=("other",)))


@pytest.mark.parametrize("ids", [(), ("p1", "p1")])
def test_selection_must_be_nonempty_and_unique(ids):
    with pytest.raises(ValueError):
        validate_decision(request(), decision(selected_paper_ids=ids))


def test_selection_cap_is_server_owned():
    with pytest.raises(ValueError, match="paper_limit"):
        validate_decision(
            request(max_selected_papers=1),
            decision(selected_paper_ids=("p1", "p2")),
        )


def test_stale_or_resolved_request_rejects_decision():
    with pytest.raises(ValueError, match="stale_review"):
        validate_decision(request(), decision().model_copy(update={"revision": 2}))
    with pytest.raises(ValueError, match="review_not_pending"):
        validate_decision(request(status="resolved"), decision())


def test_feedback_is_bounded_and_requires_content():
    with pytest.raises(ValueError, match="feedback_required"):
        validate_decision(request("final_report"), decision("revise"))
    with pytest.raises(ValueError, match="revision_limit"):
        validate_decision(
            request("final_report", revision_attempts=1),
            decision("revise", feedback="Clarify the limitations."),
        )
    validate_decision(
        request("final_report"), decision("revise", feedback="Clarify limitations.")
    )


def test_actions_and_payloads_cannot_cross_stages():
    with pytest.raises(ValueError, match="action_not_allowed"):
        validate_decision(request(), decision("approve"))
    with pytest.raises(ValueError, match="unexpected_payload"):
        validate_decision(
            request("final_report"), decision("approve", selected_paper_ids=("p1",))
        )


def test_human_approval_never_overrides_automatic_gate():
    assert not can_publish(False, RunOptions(human_review=True), "approved")
    assert not can_publish(True, RunOptions(human_review=True), "pending")
    assert can_publish(True, RunOptions(human_review=True), "approved")
    assert can_publish(True, RunOptions(), "not_requested")


def test_contract_roundtrip_and_unknown_fields():
    original = request()
    assert ReviewRequest.model_validate_json(original.model_dump_json()) == original
    with pytest.raises(ValidationError):
        HumanDecision(request_id="x", revision=1, action="approve", publishable=True)


def test_report_hash_preserves_exact_content():
    report = ReportVersion(run_id="r", revision=1, report_language="en", survey="A")
    assert (
        report.content_sha256
        != report.model_copy(update={"survey": "A "}).content_sha256
    )


def test_resume_checks_language_and_source_identity():
    identity = ResumeIdentity(
        run_id="r",
        source_fingerprint="a" * 64,
        config_fingerprint="b" * 64,
        options=RunOptions(),
    )
    identity.require_compatible(
        ResumeIdentity.model_validate_json(identity.model_dump_json())
    )
    with pytest.raises(ValueError, match="incompatible_checkpoint"):
        identity.require_compatible(
            identity.model_copy(update={"options": RunOptions(report_language="zh-CN")})
        )


def test_resume_accepts_project_config_fingerprint_format():
    identity = ResumeIdentity(
        run_id="run-1",
        source_fingerprint="a" * 64,
        config_fingerprint=f"sha256:{'b' * 64}",
        options=RunOptions(),
    )
    assert identity.config_fingerprint.startswith("sha256:")
