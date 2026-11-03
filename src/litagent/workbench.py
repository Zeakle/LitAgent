"""Define local workbench contracts without enabling unwired API features."""

from __future__ import annotations

import hashlib
import json
import time
from typing import Literal, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

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
    budget_policy: Literal[
        "legacy",
        "planned-budget-v2",
        "planned-budget-v3",
        "planned-budget-v4",
        "planned-budget-v5",
        "planned-budget-v6",
        "planned-budget-v7",
    ] = "legacy"
    capacity_policy: Literal["legacy", "adaptive-capacity-v1"] = "legacy"
    classification_review: bool = Field(default=False, strict=True)
    repair_feedback: bool = Field(default=False, strict=True)
    mixed_inventory_recovery: bool = Field(default=False, strict=True)
    reference_contract: Literal["ordered-reference-set-v1"] | None = None
    transport_revision: Literal["host-lossless-transport-v1"] | None = None
    support_validation_revision: (
        Literal["host-support-receipts-v1", "host-support-receipts-v2"] | None
    ) = None
    generation_strategy: Literal[
        "single_v1", "body_summary_v1", "body_summary_v2", "body_summary_v3"
    ] = "single_v1"

    @model_serializer(mode="wrap")
    def serialize_budget_policy(self, handler):
        """Keep inactive defaults out of historical wire identities."""
        value = handler(self)
        if self.capacity_policy == "legacy":
            value.pop("capacity_policy", None)
        if self.budget_policy == "legacy":
            value.pop("budget_policy", None)
        if not self.classification_review:
            value.pop("classification_review", None)
        if not self.repair_feedback:
            value.pop("repair_feedback", None)
        if not self.mixed_inventory_recovery:
            value.pop("mixed_inventory_recovery", None)
        if self.reference_contract is None:
            value.pop("reference_contract", None)
        if self.transport_revision is None:
            value.pop("transport_revision", None)
        if self.support_validation_revision is None:
            value.pop("support_validation_revision", None)
        return value

    @model_validator(mode="after")
    def enforce_strict_review(self):
        """Apply policy-owned defaults while honoring explicit optional reviewers."""
        if self.support_validation_revision is not None and (
            self.budget_policy != "planned-budget-v7"
            or self.generation_strategy != "body_summary_v3"
            or self.classification_review is not True
            or self.mixed_inventory_recovery is not True
            or self.repair_feedback is not True
            or self.reference_contract != "ordered-reference-set-v1"
            or self.transport_revision != "host-lossless-transport-v1"
        ):
            raise ValueError("support_validation_requires_explicit_v7_contracts")
        if self.classification_review and self.budget_policy != "planned-budget-v7":
            raise ValueError("classification_review_requires_planned_budget_v7")
        if self.repair_feedback and self.budget_policy != "planned-budget-v7":
            raise ValueError("repair_feedback_requires_planned_budget_v7")
        if self.mixed_inventory_recovery and self.budget_policy != "planned-budget-v7":
            raise ValueError("mixed_inventory_recovery_requires_planned_budget_v7")
        if (
            self.reference_contract is not None
            and self.budget_policy != "planned-budget-v7"
        ):
            raise ValueError("reference_contract_requires_planned_budget_v7")
        if self.transport_revision is not None and (
            self.budget_policy != "planned-budget-v7"
            or self.generation_strategy != "body_summary_v3"
            or self.mixed_inventory_recovery is not True
            or self.repair_feedback is not True
            or self.reference_contract != "ordered-reference-set-v1"
        ):
            raise ValueError("transport_revision_requires_explicit_v7_contracts")
        if (self.budget_policy in ("planned-budget-v6", "planned-budget-v7")) != (
            self.capacity_policy == "adaptive-capacity-v1"
        ):
            raise ValueError("adaptive_capacity_requires_planned_budget_v6")
        if self.budget_policy != "legacy" and (
            self.generation_strategy
            != (
                "body_summary_v3"
                if self.budget_policy
                in ("planned-budget-v5", "planned-budget-v6", "planned-budget-v7")
                else "body_summary_v2"
            )
            or self.verification is None
            or self.verification.judge
            or self.verification.jev
        ):
            raise ValueError(
                "planned_budget_requires_writer_v2_without_optional_audits"
            )
        if (
            self.generation_strategy == "body_summary_v3"
            and self.budget_policy
            not in ("planned-budget-v5", "planned-budget-v6", "planned-budget-v7")
        ):
            raise ValueError("flat_writer_requires_planned_budget_v5")
        if self.body_quality != "legacy" and self.quality_policy != "strict_auto_v2":
            raise ValueError("body_analysis_requires_strict_auto_v2")
        if (
            self.generation_strategy != "single_v1"
            and self.body_quality != "analysis_v1"
        ):
            raise ValueError("staged_generation_requires_body_analysis")
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
        if (
            self.generation_strategy in ("body_summary_v2", "body_summary_v3")
            and self.review_structure != "academic_v2"
        ):
            raise ValueError("writer_v2_requires_academic_v2")
        return self

    def fingerprint(self) -> str:
        """Bind generation and review choices to a stable run identity."""
        payload = self.model_dump(mode="json")
        if self.budget_policy == "legacy":
            payload.pop("budget_policy", None)
        else:
            payload["budget_contract"] = {
                "maximum_tokens": (
                    4500000
                    if self.budget_policy
                    in ("planned-budget-v5", "planned-budget-v6", "planned-budget-v7")
                    else 1000000
                ),
                "requests": 64,
                "repairs": 2,
                "thinking": "disabled",
                "context_margin": 0.1,
            }
            if self.budget_policy in ("planned-budget-v3", "planned-budget-v4"):
                from litagent.body_quality import ADAPTIVE_ASSESSMENT_INPUT_CONTRACT

                payload["budget_contract"].update(
                    repairs="preplanned_0_to_2",
                    repair_selection="largest-admissible-complete-path-v1",
                    initial_evaluation_reserved=True,
                    assessment_input_contract=ADAPTIVE_ASSESSMENT_INPUT_CONTRACT,
                )
        if self.budget_policy == "planned-budget-v4":
            from litagent.writing_v2 import (
                BOUNDED_EOF_FOOTPRINT_CONTRACT,
                NATIVE_WIRE_CONTRACT,
            )

            payload["budget_contract"].update(
                envelope_contract="budgeted-writer-envelope-v1",
                capacity_selection="largest-content-then-repairs-v1",
                prepared_methodology_contract="prepared-writer-methodology-v1",
                native_capacity_contract="bounded-native-chapter-v1",
                request_builder_contract="budgeted-writer-request-builders-v2",
                bounded_eof_footprint_contract=BOUNDED_EOF_FOOTPRINT_CONTRACT,
                writer_wire_contract=NATIVE_WIRE_CONTRACT,
            )
        if self.budget_policy in (
            "planned-budget-v5",
            "planned-budget-v6",
            "planned-budget-v7",
        ):
            from litagent.writing_flat import REQUEST_BUILDERS

            payload["budget_contract"].update(
                request_builder_contract=REQUEST_BUILDERS,
                writer_wire_contract="flat-writer-v1",
                envelope_contract="flat-writer-envelope-v1",
                body_max_units=66,
                model_assertions=128,
                shared_repairs=2,
                evaluation_input_contract="flat-evaluation-input-v1",
                repair_input_contract="flat-repair-input-v1",
                claim_json_bytes=30000,
                claim_max_json_bytes=2000,
                diagnostic_json_bytes=12288,
            )
        if self.capacity_policy != "legacy":
            payload["budget_contract"]["capacity_policy"] = self.capacity_policy
        if self.budget_policy == "planned-budget-v7":
            from litagent.eval.batched import (
                CURRENT_FINDINGS_CONTRACT,
                global_findings_identity,
            )
            from litagent.eval.scoped import located_evaluation_identity
            from litagent.writing_v2 import SUMMARY_CONSTRAINTS_CONTRACT

            payload["budget_contract"][
                "global_findings_contract"
            ] = CURRENT_FINDINGS_CONTRACT
            payload["budget_contract"][
                "global_findings_identity"
            ] = global_findings_identity()
            payload["budget_contract"][
                "evaluation_instruction_identity"
            ] = located_evaluation_identity()
            payload["budget_contract"][
                "summary_constraints_contract"
            ] = SUMMARY_CONSTRAINTS_CONTRACT
            if self.reference_contract is not None:
                from litagent.writing_flat import reference_set_identity

                payload["budget_contract"][
                    "reference_contract"
                ] = self.reference_contract
                payload["budget_contract"][
                    "reference_set_identity"
                ] = reference_set_identity()
            if self.transport_revision is not None:
                from litagent.writing_flat import transport_identity

                payload["budget_contract"][
                    "transport_revision"
                ] = self.transport_revision
                payload["budget_contract"]["transport_identity"] = transport_identity()
            if self.support_validation_revision is not None:
                from litagent.eval.scoped import support_validation_identity

                payload["budget_contract"][
                    "support_validation_revision"
                ] = self.support_validation_revision
                payload["budget_contract"]["support_validation_identity"] = (
                    support_validation_identity()
                    if self.support_validation_revision == "host-support-receipts-v1"
                    else support_validation_identity(self.support_validation_revision)
                )
            if self.classification_review:
                from litagent.eval.scoped import independent_classification_identity

                payload["budget_contract"][
                    "classification_review_identity"
                ] = independent_classification_identity()
            if self.repair_feedback:
                from litagent.writing_flat import repair_feedback_identity

                payload["budget_contract"][
                    "repair_feedback_identity"
                ] = repair_feedback_identity()
            if self.mixed_inventory_recovery:
                from litagent.eval.inventory_recovery import (
                    mixed_inventory_recovery_identity,
                )

                payload["budget_contract"]["mixed_inventory_recovery_identity"] = (
                    mixed_inventory_recovery_identity(
                        **(
                            {"stable": True}
                            if self.support_validation_revision is not None
                            else {}
                        )
                    )
                )
        if self.generation_strategy == "single_v1":
            payload.pop("generation_strategy")
        elif self.generation_strategy in ("body_summary_v2", "body_summary_v3"):
            from litagent.body_quality import (
                NATIVE_REPORT_INPUT_CONTRACT,
                WRITER_CONTENT_CONTRACT,
            )
            from litagent.writing_v2 import (
                EXECUTION_CONTRACT,
                PROMPT_CONTRACT,
                PROVIDER_SCHEMA_CONTRACT,
            )

            payload["generation_contract"] = "grounded-writer-v1"
            payload["writer_execution_contract"] = EXECUTION_CONTRACT
            payload["writer_provider_schema_contract"] = PROVIDER_SCHEMA_CONTRACT
            payload["writer_prompt_contract"] = PROMPT_CONTRACT
            payload["writer_content_contract"] = WRITER_CONTENT_CONTRACT
            payload["writer_input_contract"] = NATIVE_REPORT_INPUT_CONTRACT
            payload["writing_scope_policy"] = "frozen_union_v1"
        else:
            from litagent.staged_generation import CONTRACT as generation_contract

            payload["generation_contract"] = generation_contract
        if self.verification is None:
            payload.pop("verification")
        if self.body_quality == "legacy":
            payload.pop("body_quality")
        else:
            from litagent.body_quality import (
                ANALYSIS_CHAPTER_CONTRACT,
                ANALYSIS_CONTRACT,
                ANALYSIS_VERSION,
                EVALUATION_ENCODING_CONTRACT,
                EXTRACTION_CLAIM_CONTRACT,
                EXTRACTION_FIELD_SPAN_CONTRACT,
                EXTRACTION_SOURCE_LAYOUT_CONTRACT,
                METRIC_INFLECTION_CONTRACT,
                NATIVE_ARRAY_WIRE_CONTRACT,
                NATIVE_PROMPT_COMPOSITION_CONTRACT,
                NATIVE_SCHEMA_ENCODING_CONTRACT,
                PAPER_BATCH_CONTRACT,
                SUMMARY_ARRAY_WIRE_CONTRACT,
                UNIT_CITATION_SCOPE_CONTRACT,
            )

            payload["body_contracts"] = [
                ANALYSIS_CONTRACT,
                ANALYSIS_VERSION,
                "host-global-completeness-v1",
                "host-fragment-inventory-v2",
                "host-batch-assessment-v1",
                "evaluation-wire-identity-v1",
                EVALUATION_ENCODING_CONTRACT,
                EXTRACTION_FIELD_SPAN_CONTRACT,
                EXTRACTION_CLAIM_CONTRACT,
                EXTRACTION_SOURCE_LAYOUT_CONTRACT,
                METRIC_INFLECTION_CONTRACT,
                SUMMARY_ARRAY_WIRE_CONTRACT,
                NATIVE_ARRAY_WIRE_CONTRACT,
                NATIVE_PROMPT_COMPOSITION_CONTRACT,
                NATIVE_SCHEMA_ENCODING_CONTRACT,
                "host-flat-fragment-inventory-v1",
                "local-chinese-comparatives-v1",
                "host-analysis-claim-ownership-v1",
                ANALYSIS_CHAPTER_CONTRACT,
                UNIT_CITATION_SCOPE_CONTRACT,
                "atomic-fragment-decomposition-v1",
                PAPER_BATCH_CONTRACT + ":size=2:concurrency=2",
                "complete-native-diagnostics-v1",
                "failed-worker-repair-ledger-v1",
                "chapter-source-menu-v1",
                "complete-native-unit-repair-v5",
                "source-applicability-v1",
                "analysis-diagnostic-cause-v1",
                "analysis-clean-repair-context-v1",
                "assessed-repair-repetition-v1",
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
