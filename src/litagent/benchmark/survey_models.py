"""Define strict contracts for end-to-end survey benchmarks."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from litagent.benchmark.models import RAGBenchmarkProfile, _fingerprint


class SurveyDomain(str, Enum):
    """Identify the three audited computer-vision benchmark domains."""

    FEW_SHOT = "few_shot"
    VISION_TRANSFORMER = "vision_transformer"
    NERF = "nerf"


class ExpectedTopic(BaseModel):
    """Describe one reviewable topic expected in a survey answer."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    source_paper_ids: list[str] = Field(min_length=1)
    required: bool = True


class EvidenceSource(BaseModel):
    """Record the source used by a human to audit one benchmark case."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    kind: Literal["paper", "survey", "official_index"]
    url: str = Field(pattern=r"^https://")
    accessed_at: date


class SurveyBenchmarkCase(BaseModel):
    """Bind one query to audited papers, topics, and delivery semantics."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    domain: SurveyDomain
    query: str = Field(min_length=1)
    required_paper_ids: list[str] = Field(min_length=1)
    relevant_paper_ids: list[str] = Field(min_length=1)
    expected_topics: list[ExpectedTopic] = Field(min_length=1)
    evidence_sources: list[EvidenceSource] = Field(min_length=1)
    knowledge_cutoff: date
    allowed_delivery_statuses: list[
        Literal["partial", "blocked", "needs_review", "ready"]
    ] = Field(min_length=1)
    annotation_version: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_case(self):
        """Reject ambiguous labels and references outside the reviewed set."""
        required = set(self.required_paper_ids)
        relevant = set(self.relevant_paper_ids)
        if len(required) != len(self.required_paper_ids):
            raise ValueError("required_paper_ids must be unique")
        if len(relevant) != len(self.relevant_paper_ids):
            raise ValueError("relevant_paper_ids must be unique")
        if not required.issubset(relevant):
            raise ValueError(
                "required_paper_ids must be a subset of relevant_paper_ids"
            )
        topic_ids = [topic.topic_id for topic in self.expected_topics]
        if len(set(topic_ids)) != len(topic_ids):
            raise ValueError("expected topic ids must be unique within a case")
        if not any(topic.required for topic in self.expected_topics):
            raise ValueError("each case requires at least one required expected topic")
        unknown_topic_papers = {
            paper_id
            for topic in self.expected_topics
            for paper_id in topic.source_paper_ids
            if paper_id not in relevant
        }
        if unknown_topic_papers:
            raise ValueError(
                "expected topics reference papers outside relevant_paper_ids: "
                f"{sorted(unknown_topic_papers)}"
            )
        source_ids = [source.source_id for source in self.evidence_sources]
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("evidence source ids must be unique within a case")
        if len(set(self.allowed_delivery_statuses)) != len(
            self.allowed_delivery_statuses
        ):
            raise ValueError("allowed_delivery_statuses must be unique")
        return self


class SurveyBenchmarkDataset(BaseModel):
    """Describe the versioned 15-case product benchmark."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    judgment_status: Literal[
        "candidate",
        "source_reviewed",
        "owner_approved_ai_assisted",
        "human_reviewed",
    ]
    review_notes: str = ""
    annotation_version: str = Field(min_length=1)
    manifest_paths: list[str] = Field(min_length=1)
    corpus_paper_ids: list[str] = Field(min_length=1)
    distractor_paper_ids: dict[SurveyDomain, list[str]]
    ablation_case_ids: list[str] = Field(min_length=1)
    cases: list[SurveyBenchmarkCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dataset(self):
        """Enforce balanced domains and complete corpus references."""
        corpus_ids = set(self.corpus_paper_ids)
        if len(corpus_ids) != len(self.corpus_paper_ids):
            raise ValueError("corpus_paper_ids must be unique")
        case_ids = [case.case_id for case in self.cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("survey case ids must be unique")
        unknown_ablation = set(self.ablation_case_ids) - set(case_ids)
        if unknown_ablation:
            raise ValueError(f"unknown ablation case ids: {sorted(unknown_ablation)}")
        for case in self.cases:
            unknown = set(case.relevant_paper_ids) - corpus_ids
            if unknown:
                raise ValueError(
                    f"case {case.case_id} references papers outside corpus: "
                    f"{sorted(unknown)}"
                )

        relevant_union = {
            paper_id for case in self.cases for paper_id in case.relevant_paper_ids
        }
        distractors: set[str] = set()
        if set(self.distractor_paper_ids) != set(SurveyDomain):
            raise ValueError("distractor_paper_ids must cover every survey domain")
        for domain, paper_ids in self.distractor_paper_ids.items():
            if len(paper_ids) != 2 or len(set(paper_ids)) != 2:
                raise ValueError(
                    f"domain {domain.value} requires exactly 2 distractors"
                )
            distractors.update(paper_ids)
        if distractors & relevant_union:
            raise ValueError("distractor papers must not appear in relevant judgments")
        if corpus_ids != relevant_union | distractors:
            raise ValueError(
                "corpus must equal the relevant-paper union plus declared distractors"
            )

        if len(self.cases) != 15:
            raise ValueError("formal survey dataset requires exactly 15 cases")
        for domain in SurveyDomain:
            count = sum(case.domain is domain for case in self.cases)
            if count != 5:
                raise ValueError(f"domain {domain.value} requires exactly 5 cases")
            ablation_count = sum(
                self.case_by_id(case_id).domain is domain
                for case_id in self.ablation_case_ids
            )
            if ablation_count != 2:
                raise ValueError(
                    f"domain {domain.value} requires exactly 2 ablation cases"
                )
        if len(self.ablation_case_ids) != 6:
            raise ValueError("ablation_case_ids must contain exactly 6 cases")
        return self

    def case_by_id(self, case_id: str) -> SurveyBenchmarkCase:
        """Return a case by stable identifier."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    @property
    def fingerprint(self) -> str:
        """Fingerprint all annotations and corpus references."""
        return _fingerprint(self.model_dump(mode="json"))


class SurveyBenchmarkProfile(BaseModel):
    """Select external-only or one explicit local RAG behavior."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1)
    rag_enabled: bool
    rag_profile: RAGBenchmarkProfile | None = None

    @model_validator(mode="after")
    def _validate_rag_mode(self):
        """Require a RAG profile exactly when local recall is enabled."""
        if self.rag_enabled != (self.rag_profile is not None):
            raise ValueError(
                "rag_profile must be present exactly when rag_enabled=true"
            )
        return self

    @property
    def fingerprint(self) -> str:
        """Fingerprint the complete product profile."""
        return _fingerprint(self.model_dump(mode="json", exclude={"profile_id"}))


class SurveyJudgeConfig(BaseModel):
    """Configure an independent, non-secret benchmark Judge."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(pattern=r"^https?://")
    model: str = Field(min_length=1)
    api_key_env: str = Field(default="BENCHMARK_JUDGE_API_KEY", min_length=1)
    max_tokens: int = Field(default=16384, ge=1024, le=65536)
    timeout_seconds: float = Field(default=180.0, gt=0, le=600)
    max_retries: int = Field(default=1, ge=0, le=2)

    @property
    def fingerprint(self) -> str:
        """Fingerprint Judge behavior without reading or storing credentials."""
        return _fingerprint(self.model_dump(mode="json"))


class CoverageMetrics(BaseModel):
    """Store required-paper coverage at each product stage."""

    model_config = ConfigDict(extra="forbid")

    dedup: float = Field(ge=0, le=1)
    relevance_gate: float = Field(ge=0, le=1)
    evidence: float = Field(ge=0, le=1)
    missing_by_stage: dict[str, list[str]] = Field(default_factory=dict)


class CitationMetrics(BaseModel):
    """Store deterministic paper-level citation observations."""

    model_config = ConfigDict(extra="forbid")

    recall: float = Field(ge=0, le=1)
    precision: float | None = Field(default=None, ge=0, le=1)
    cited_paper_ids: list[str] = Field(default_factory=list)
    missing_required_paper_ids: list[str] = Field(default_factory=list)
    unjudged_paper_ids: list[str] = Field(default_factory=list)
    unknown_evidence_ids: list[str] = Field(default_factory=list)
    unknown_reference_rate: float = Field(ge=0, le=1)


class RuntimeCostMetrics(BaseModel):
    """Store product-runtime cost without Judge usage."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    worker_calls: int = Field(ge=0)
    worker_retries: int = Field(ge=0)
    tool_retries: int = Field(ge=0)
    worker_failures: int = Field(ge=0)
    tool_failures: int = Field(ge=0)


class SurveyRunMetrics(BaseModel):
    """Store all deterministic metrics for one Survey run."""

    model_config = ConfigDict(extra="forbid")

    coverage: CoverageMetrics
    citations: CitationMetrics
    delivery_status: str
    delivery_accurate: bool
    partial: bool
    elapsed_ms: int = Field(ge=0)
    stage_latency_ms: dict[str, int] = Field(default_factory=dict)
    cost: RuntimeCostMetrics
    runtime_evaluation: dict[str, Any] = Field(default_factory=dict)


class TopicAssessment(BaseModel):
    """Store one independently judged topic decision."""

    model_config = ConfigDict(extra="forbid")

    topic_id: str
    covered: bool
    explanation: str = ""


class UnsupportedClaim(BaseModel):
    """Store one claim the Judge could not ground in cited evidence."""

    model_config = ConfigDict(extra="forbid")

    claim_text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(default_factory=list)
    explanation: str = ""


class ContradictionPair(BaseModel):
    """Store two report claims judged mutually contradictory."""

    model_config = ConfigDict(extra="forbid")

    claim_a: str = Field(min_length=1)
    claim_b: str = Field(min_length=1)
    explanation: str = ""


class SurveyJudgeMetrics(BaseModel):
    """Store independent semantic quality metrics with explicit availability."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["completed", "failed", "skipped"]
    topic_coverage: float | None = Field(default=None, ge=0, le=1)
    factual_claim_count: int | None = Field(default=None, ge=0)
    faithfulness: float | None = Field(default=None, ge=0, le=1)
    unsupported_claim_rate: float | None = Field(default=None, ge=0, le=1)
    contradiction_rate: float | None = Field(default=None, ge=0, le=1)
    topic_assessments: list[TopicAssessment] = Field(default_factory=list)
    unsupported_claims: list[UnsupportedClaim] = Field(default_factory=list)
    contradictions: list[ContradictionPair] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class JudgeUsage(BaseModel):
    """Store Judge cost independently from product runtime cost."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    elapsed_ms: int = Field(default=0, ge=0)


class SurveyCaseObservation(BaseModel):
    """Persist one execution of a profile/case pair."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    case_id: str
    domain: SurveyDomain
    profile_id: str
    repetition: int = Field(ge=1)
    status: Literal["completed", "failed"]
    metrics: SurveyRunMetrics | None = None
    judge: SurveyJudgeMetrics | None = None
    judge_usage: JudgeUsage = Field(default_factory=JudgeUsage)
    artifact_id: str | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    survey_trace_id: str | None = None
    survey_trace_url: str | None = None
    judge_trace_id: str | None = None
    judge_trace_url: str | None = None
    external_input_fingerprint: str | None = None
    generation_model: str | None = None
    judge_model: str | None = None
    git_sha: str | None = None
    git_dirty: bool | None = None
    config_fingerprint: str | None = None
    profile_fingerprint: str | None = None
    prompt_source_fingerprint: str | None = None
    dataset_fingerprint: str | None = None
    manifest_fingerprint: str | None = None
    reason_codes: list[str] = Field(default_factory=list)


class SurveyCaseAggregate(BaseModel):
    """Average repeated observations within one case before macro aggregation."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    domain: SurveyDomain
    profile_id: str
    observation_count: int = Field(ge=1)
    valid_observation_rate: float = Field(ge=0, le=1)
    coverage_evidence: float | None = Field(default=None, ge=0, le=1)
    citation_recall: float | None = Field(default=None, ge=0, le=1)
    citation_precision: float | None = Field(default=None, ge=0, le=1)
    delivery_accuracy: float | None = Field(default=None, ge=0, le=1)
    topic_coverage: float | None = Field(default=None, ge=0, le=1)
    faithfulness: float | None = Field(default=None, ge=0, le=1)
    unsupported_claim_rate: float | None = Field(default=None, ge=0, le=1)
    contradiction_rate: float | None = Field(default=None, ge=0, le=1)
    total_tokens: float | None = Field(default=None, ge=0)
    elapsed_ms: float | None = Field(default=None, ge=0)


class SurveyProfileSummary(BaseModel):
    """Macro-average case-level metrics without inventing one total score."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str
    case_count: int = Field(ge=0)
    observation_count: int = Field(ge=0)
    valid_observation_rate: float = Field(ge=0, le=1)
    coverage_evidence: float | None = Field(default=None, ge=0, le=1)
    citation_recall: float | None = Field(default=None, ge=0, le=1)
    citation_precision: float | None = Field(default=None, ge=0, le=1)
    delivery_accuracy: float | None = Field(default=None, ge=0, le=1)
    topic_coverage: float | None = Field(default=None, ge=0, le=1)
    faithfulness: float | None = Field(default=None, ge=0, le=1)
    unsupported_claim_rate: float | None = Field(default=None, ge=0, le=1)
    contradiction_rate: float | None = Field(default=None, ge=0, le=1)
    total_tokens: float | None = Field(default=None, ge=0)
    latency_p50_ms: float | None = Field(default=None, ge=0)
    latency_p95_ms: float | None = Field(default=None, ge=0)
    delivery_distribution: dict[str, int] = Field(default_factory=dict)
    stage_latency_p95_ms: dict[str, float] = Field(default_factory=dict)
    product_total_tokens: int = Field(default=0, ge=0)
    judge_total_tokens: int = Field(default=0, ge=0)
    llm_call_count: int = Field(default=0, ge=0)
    tool_call_count: int = Field(default=0, ge=0)
    worker_retry_count: int = Field(default=0, ge=0)
    tool_retry_count: int = Field(default=0, ge=0)
    worker_failure_rate: float | None = Field(default=None, ge=0, le=1)
    tool_failure_rate: float | None = Field(default=None, ge=0, le=1)


class SurveyProfileComparison(BaseModel):
    """Store one explicit pairwise profile comparison without a weighted score."""

    model_config = ConfigDict(extra="forbid")

    left_profile_id: str
    right_profile_id: str
    comparable: bool
    shared_cohort_count: int = Field(ge=0)
    metric_deltas: dict[str, float | None] = Field(default_factory=dict)
    reason_codes: list[str] = Field(default_factory=list)


class SurveyBenchmarkResult(BaseModel):
    """Store an ablation or final Survey benchmark batch."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    benchmark_type: Literal["survey"] = "survey"
    run_id: str
    status: Literal["succeeded", "partial", "failed"]
    stage: Literal["ablation", "baseline"]
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    judgment_status: Literal[
        "candidate",
        "source_reviewed",
        "owner_approved_ai_assisted",
        "human_reviewed",
    ]
    annotation_version: str
    manifest_fingerprint: str
    git_sha: str
    git_dirty: bool
    generation_model: str
    judge_model: str
    self_judged: bool
    formal_eligible: bool
    prompt_source_fingerprint: str
    judge_prompt_fingerprint: str
    config_fingerprint: str
    profile_fingerprints: dict[str, str]
    observations: list[SurveyCaseObservation]
    case_aggregates: list[SurveyCaseAggregate]
    profile_summaries: list[SurveyProfileSummary]
    domain_summaries: dict[str, dict[str, Any]] = Field(default_factory=dict)
    recommended_profile_id: str | None = None
    reused_ablation_run_id: str | None = None
    comparable_profile_pairs: list[list[str]] = Field(default_factory=list)
    profile_comparisons: list[SurveyProfileComparison] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)
