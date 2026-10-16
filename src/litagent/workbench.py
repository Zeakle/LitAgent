"""Define local workbench contracts without enabling unwired API features."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewStructure = Literal[
    "legacy", "academic_v1", "method_transparent_v1", "academic_v2"
]
CitationStyle = Literal[
    "evidence",
    "numbered",
    "ieee",
    "apa7",
    "gbt7714_2025_numeric",
    "gbt7714_2025_author_date",
]

ReportLanguage = Literal["en", "zh-CN"]
AutoAuditMode = Literal["off", "judge", "judge_jev"]
QualityPolicy = Literal["legacy", "strict_v1", "strict_auto_v1", "strict_auto_v2"]
ReviewStage = Literal["paper_selection", "final_report"]
ReviewAction = Literal["confirm", "approve", "revise", "reject", "cancel"]
HumanReviewStatus = Literal["not_requested", "pending", "approved", "rejected"]


def is_strict_policy(policy: str | None) -> bool:
    """Share scientific safeguards without changing historical approval rules."""
    return policy in {"strict_v1", "strict_auto_v1", "strict_auto_v2"}


def language_instruction(language: ReportLanguage) -> str:
    """Return the shared generation and review contract for one language."""
    if language == "zh-CN":
        return (
            "Write all report prose and section headings in Simplified Chinese. "
            "Keep paper titles, quoted source text, DOI/URL values, and Evidence IDs "
            "in their source form; do not translate or alter those identifiers."
        )
    return (
        "Write all report prose and section headings in English. Keep paper titles, "
        "quoted source text, DOI/URL values, and Evidence IDs in their source form; "
        "do not translate or alter those identifiers."
    )


class Contract(BaseModel):
    """Reject undeclared fields and prohibit reassignment of contract values."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class VerificationOptions(Contract):
    """Bind optional verification choices to new runs without changing old defaults."""

    version: Literal["optional-verification-v1"] = "optional-verification-v1"
    judge: bool = Field(default=False, strict=True)
    jev: bool = Field(default=False, strict=True)


def verification_enabled(value: Mapping, service: Literal["judge", "jev"]) -> bool:
    """Preserve mandatory historical gates when the new contract is absent."""
    selected = value.get("verification")
    if selected is None:
        return True
    return getattr(VerificationOptions.model_validate(selected), service)


class RunOptions(Contract):
    """Keep legacy defaults separate from UI defaults and UI locale."""

    report_language: ReportLanguage = "en"
    human_review: bool = False
    review_structure: ReviewStructure = "legacy"
    auto_audit_mode: AutoAuditMode = "off"
    auto_audit_max_versions: int = Field(default=1, ge=1, le=3, strict=True)
    quality_policy: QualityPolicy = "legacy"
    verification: VerificationOptions | None = None
    body_quality: Literal["legacy", "analysis_v1"] = "legacy"

    @model_validator(mode="after")
    def enforce_strict_review(self):
        """Apply policy-owned defaults while honoring explicit optional reviewers."""
        if self.body_quality != "legacy" and self.quality_policy != "strict_auto_v2":
            raise ValueError("body_analysis_requires_strict_auto_v2")
        if self.verification is not None and self.quality_policy != "strict_auto_v2":
            raise ValueError("optional_verification_requires_strict_auto_v2")
        if self.quality_policy == "strict_v1":
            object.__setattr__(self, "human_review", True)
            if self.review_structure == "legacy":
                object.__setattr__(self, "review_structure", "academic_v2")
        elif self.quality_policy in {"strict_auto_v1", "strict_auto_v2"}:
            if "report_language" not in self.model_fields_set:
                object.__setattr__(self, "report_language", "zh-CN")
            object.__setattr__(self, "human_review", False)
            object.__setattr__(
                self,
                "auto_audit_mode",
                "judge" if self.quality_policy == "strict_auto_v2" else "judge_jev",
            )
            if self.verification is not None and not self.verification.judge:
                object.__setattr__(self, "auto_audit_mode", "off")
            object.__setattr__(self, "auto_audit_max_versions", 1)
            if self.review_structure == "legacy":
                object.__setattr__(self, "review_structure", "academic_v2")
        return self

    def fingerprint(self) -> str:
        """Bind generation and review choices to a stable run identity."""
        payload = self.model_dump(mode="json")
        if self.verification is None:
            payload.pop("verification")
        if self.body_quality == "legacy":
            payload.pop("body_quality")
        else:
            from litagent.body_quality import ANALYSIS_CONTRACT, ANALYSIS_VERSION

            payload["body_contracts"] = [
                ANALYSIS_CONTRACT,
                ANALYSIS_VERSION,
                "host-global-completeness-v1",
                "host-fragment-inventory-v2",
            ]
        if self.quality_policy == "legacy":
            payload.pop("quality_policy")
        else:
            payload["strict_contracts"] = [
                "fact-card-v1",
                "survey-evaluation-v3",
                "coverage-requirements-v1",
                "staged-budget-v1",
            ]
            if self.quality_policy in {"strict_auto_v1", "strict_auto_v2"}:
                payload["strict_contracts"].extend(
                    [
                        "automatic-acceptance-v1",
                        "frozen-original-audit-v1",
                        "automatic-stop-v1",
                    ]
                )
                payload["automatic_limits"] = [100000, 2, 200000, 20, 0.9]
                if self.quality_policy == "strict_auto_v2":
                    from litagent.quality import (
                        NATIVE_REPAIR_CONTRACT,
                        TARGETED_REPAIR_CONTRACT,
                    )

                    payload["strict_contracts"].append("jev-extraction-verification-v1")
                    from litagent.eval.batched import (
                        GLOBAL_CONSISTENCY_CONTRACT,
                        SOURCE_FIDELITY_CONTRACT,
                    )
                    from litagent.eval.scoped import LOCAL_NUMERIC_CONTRACT
                    from litagent.extraction_verification import LEDGER_CONTRACT
                    from litagent.writing_passages import (
                        FULL_VERSION,
                        NOTICE_VERSION,
                        SCOPE_NOTICE_CONTRACT,
                    )

                    payload["strict_contracts"].extend(
                        [
                            FULL_VERSION,
                            "host-writing-full-sources-v1",
                            SOURCE_FIDELITY_CONTRACT,
                            LEDGER_CONTRACT,
                            LOCAL_NUMERIC_CONTRACT,
                            GLOBAL_CONSISTENCY_CONTRACT,
                            NATIVE_REPAIR_CONTRACT,
                            TARGETED_REPAIR_CONTRACT,
                            NOTICE_VERSION,
                            SCOPE_NOTICE_CONTRACT,
                        ]
                    )
        if self.review_structure == "legacy":
            payload.pop("review_structure")
        if self.auto_audit_mode == "off":
            # Inactive policy fields must not invalidate legacy run identities.
            payload.pop("auto_audit_mode")
            payload.pop("auto_audit_max_versions")
        value = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ReviewRequest(Contract):
    """Describe a server-owned boundary awaiting a version-bound decision."""

    schema_version: Literal[1] = 1
    request_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    stage: ReviewStage
    revision: int = Field(ge=1)
    status: Literal["pending", "resolved", "cancelled"] = "pending"
    created_at: float = Field(default_factory=time.time, ge=0)
    candidate_paper_ids: tuple[str, ...] = ()
    automatic_selected_paper_ids: tuple[str, ...] = ()
    max_selected_papers: int = Field(default=50, ge=1, le=50)
    revision_attempts: int = Field(default=0, ge=0, le=1)
    report_revision: int | None = Field(default=None, ge=1)
    report_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @property
    def allowed_actions(self) -> tuple[ReviewAction, ...]:
        """Expose only actions valid for this stage and remaining revision budget."""
        if self.status != "pending":
            return ()
        if self.stage == "paper_selection":
            return ("confirm", "cancel")
        if self.revision_attempts:
            return ("approve", "reject", "cancel")
        return ("approve", "revise", "reject", "cancel")


class HumanDecision(Contract):
    """Accept intent only, never client-supplied quality or publication state."""

    request_id: str = Field(min_length=1, max_length=128)
    revision: int = Field(ge=1)
    action: ReviewAction
    selected_paper_ids: tuple[str, ...] = ()
    feedback: str = Field(default="", max_length=4000)


def validate_decision(request: ReviewRequest, decision: HumanDecision) -> None:
    """Reject stale decisions and stage-inappropriate or unauthorized payloads.

    The persistence layer must invoke this inside its compare-and-set transaction;
    validation alone neither consumes a request nor grants execution ownership.
    """
    if request.status != "pending":
        raise ValueError("review_not_pending")
    if (request.request_id, request.revision) != (
        decision.request_id,
        decision.revision,
    ):
        raise ValueError("stale_review")
    if decision.action == "revise" and request.revision_attempts:
        raise ValueError("revision_limit")
    if decision.action not in request.allowed_actions:
        raise ValueError("action_not_allowed")
    if decision.action == "confirm":
        selected = decision.selected_paper_ids
        if not selected:
            raise ValueError("empty_selection")
        if len(set(selected)) != len(selected):
            raise ValueError("duplicate_paper")
        if len(selected) > request.max_selected_papers:
            raise ValueError("paper_limit")
        if not set(selected).issubset(request.candidate_paper_ids):
            raise ValueError("unknown_paper")
        if decision.feedback:
            raise ValueError("unexpected_payload")
    elif decision.action == "revise":
        if decision.selected_paper_ids:
            raise ValueError("unexpected_payload")
        if not decision.feedback.strip():
            raise ValueError("feedback_required")
    elif decision.selected_paper_ids or decision.feedback:
        raise ValueError("unexpected_payload")


def can_publish(
    automatic_publishable: bool,
    options: RunOptions,
    human_status: HumanReviewStatus,
) -> bool:
    """Intersect automatic eligibility with human approval, never override quality."""
    return automatic_publishable is True and (
        human_status == "approved"
        if options.human_review
        else human_status == "not_requested"
    )


class ReportVersion(Contract):
    """Bind exports and approval to exact immutable report content."""

    run_id: str = Field(min_length=1)
    revision: int = Field(ge=1)
    report_language: ReportLanguage
    survey: str = Field(min_length=1)
    human_status: HumanReviewStatus = "not_requested"

    @property
    def content_sha256(self) -> str:
        """Hash exact report bytes without altering whitespace or citations."""
        return hashlib.sha256(self.survey.encode("utf-8")).hexdigest()


class ResumeIdentity(Contract):
    """Bind checkpoints to compatible code, configuration and run options."""

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    source_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    config_fingerprint: str = Field(pattern=r"^(?:sha256:)?[a-f0-9]{64}$")
    options: RunOptions

    def require_compatible(self, current: ResumeIdentity) -> None:
        """Reject incompatible recovery rather than silently rerun paid stages."""
        if self != current:
            raise ValueError("incompatible_checkpoint")


class DecisionReceipt(Contract):
    """Identify an accepted decision without promising that execution has resumed."""

    request_id: str
    run_id: str
    revision: int = Field(ge=1)
    action: ReviewAction
    execution_status: Literal["queued", "cancelled", "rejected", "completed"]


class ReviewRepository(Protocol):
    """Specify transactional review persistence; no implementation is wired yet."""

    async def get_pending(self, run_id: str) -> ReviewRequest | None:
        """Return the pending boundary, or None; storage failures must propagate."""
        ...

    async def decide(self, run_id: str, decision: HumanDecision) -> DecisionReceipt:
        """Validate and consume atomically, replay identical decisions idempotently.

        A different decision for an already consumed request is a conflict. This
        operation queues execution intent; it must not make model calls inside
        the database transaction. Unknown runs and stale revisions are errors.
        """
        ...


class ExportResult(Contract):
    """Return generated document bytes for one immutable report revision."""

    filename: str = Field(min_length=1)
    media_type: Literal[
        "text/markdown; charset=utf-8",
        "application/pdf",
        "application/x-bibtex; charset=utf-8",
        "application/x-research-info-systems; charset=utf-8",
        "application/vnd.citationstyles.csl+json; charset=utf-8",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]
    content: bytes
    report_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReportExporter(Protocol):
    """Specify deterministic exports; rendering implementations remain unwired."""

    async def export(
        self,
        report: ReportVersion,
        *,
        format: Literal["md", "pdf", "docx"],
        publishable: bool,
        citation_style: CitationStyle = "evidence",
    ) -> ExportResult:
        """Render exact content without LLM calls or remote asset retrieval.

        The service resolves publication eligibility from server-owned state.
        Nonpublishable content must carry a visible draft label. Rendering errors
        propagate rather than returning a corrupt or silently partial document.
        """
        ...
