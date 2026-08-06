"""Define strict, fingerprinted benchmark contracts."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from litagent.config import (
    ChunkStrategy,
    EmbeddingBackend,
    RAGConfig,
    RetrievalMode,
)


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


class RAGBenchmarkProfile(BaseModel):
    """Describe one explicit, reproducible retrieval experiment."""

    model_config = ConfigDict(extra="forbid")

    profile_id: str = Field(min_length=1)
    content_mode: Literal[
        "abstract",
        "abstract_and_selected_fulltext",
    ]
    chunk_strategy: ChunkStrategy
    chunk_size: int = Field(ge=200, le=8000)
    chunk_overlap: int = Field(ge=0, le=2000)
    embedding_backend: EmbeddingBackend
    embedding_model: str = Field(min_length=1)
    embedding_document_adapter: str | None = None
    embedding_query_adapter: str | None = None
    retrieval_mode: RetrievalMode
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    candidate_k: int = Field(default=40, ge=1, le=500)
    top_k: int = Field(default=20, ge=1, le=100)
    max_representative_chunks: int = Field(default=4, ge=1, le=20)
    repetitions: int = Field(default=3, ge=1, le=20)

    @model_validator(mode="after")
    def _validate_profile(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.candidate_k < self.top_k:
            raise ValueError("candidate_k must be >= top_k")
        if self.embedding_backend is EmbeddingBackend.SPECTER2 and not (
            self.embedding_document_adapter and self.embedding_query_adapter
        ):
            raise ValueError("SPECTER2 profile requires both adapters")
        return self

    @property
    def fingerprint(self) -> str:
        """Fingerprint every storage and query behavior dimension."""
        return _fingerprint(self.model_dump(mode="json", exclude={"profile_id"}))

    def apply(self, base: RAGConfig) -> RAGConfig:
        """Return a validated effective config for this profile."""
        values = base.model_dump(mode="json")
        values.update(
            {
                "content_mode": self.content_mode,
                "chunk_strategy": self.chunk_strategy.value,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "embedding_backend": self.embedding_backend.value,
                "embedding_model": self.embedding_model,
                "embedding_document_adapter": self.embedding_document_adapter,
                "embedding_query_adapter": self.embedding_query_adapter,
                "retrieval_mode": self.retrieval_mode.value,
                "reranker_enabled": self.retrieval_mode is RetrievalMode.RRF_RERANK,
                "reranker_model": self.reranker_model,
                "candidate_k": self.candidate_k,
                "top_k": self.top_k,
                "max_representative_chunks": self.max_representative_chunks,
            }
        )
        return RAGConfig.model_validate(values)


class RetrievalJudgment(BaseModel):
    """Bind one stable query to human-reviewed relevant papers."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant_paper_ids: list[str] = Field(min_length=1)
    relevant_locators: dict[str, list[str]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_relevance(self):
        if len(set(self.relevant_paper_ids)) != len(self.relevant_paper_ids):
            raise ValueError("relevant_paper_ids must be unique")
        unknown = set(self.relevant_locators) - set(self.relevant_paper_ids)
        if unknown:
            raise ValueError("relevant locator references a non-relevant paper")
        return self


class RAGBenchmarkDataset(BaseModel):
    """Describe a versioned corpus and its retrieval judgments."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    judgment_status: Literal["candidate", "source_reviewed", "human_reviewed"]
    review_notes: str = ""
    manifest_path: str = "corpus/manifest.yaml"
    corpus_paper_ids: list[str] = Field(min_length=1)
    queries: list[RetrievalJudgment] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_dataset(self):
        corpus_ids = set(self.corpus_paper_ids)
        if len(corpus_ids) != len(self.corpus_paper_ids):
            raise ValueError("corpus_paper_ids must be unique")
        query_ids = [query.query_id for query in self.queries]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("query_id values must be unique")
        for query in self.queries:
            unknown = set(query.relevant_paper_ids) - corpus_ids
            if unknown:
                raise ValueError(
                    f"query {query.query_id} references unknown papers: "
                    f"{sorted(unknown)}"
                )
        return self

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.model_dump(mode="json"))


class RetrievalCaseMetrics(BaseModel):
    """Store hand-checkable paper-level ranking metrics."""

    model_config = ConfigDict(extra="forbid")

    recall_at_k: dict[int, float]
    mrr_at_10: float
    ndcg_at_10: float
    duplicate_paper_ratio: float


class RetrievalCaseResult(BaseModel):
    """Persist one query outcome before aggregation."""

    model_config = ConfigDict(extra="forbid")

    query_id: str
    query: str
    relevant_paper_ids: list[str]
    retrieved_paper_ids: list[str]
    metrics: RetrievalCaseMetrics
    elapsed_ms: int = Field(ge=0)
    latency_samples_ms: list[int] = Field(default_factory=list)


class RetrievalBenchmarkSummary(BaseModel):
    """Aggregate retrieval metrics without inventing one total score."""

    model_config = ConfigDict(extra="forbid")

    case_count: int = Field(ge=0)
    recall_at_k: dict[int, float]
    mrr_at_10: float
    ndcg_at_10: float
    duplicate_paper_ratio: float
    empty_result_rate: float
    latency_p50_ms: float
    latency_p95_ms: float


class RAGBenchmarkResult(BaseModel):
    """Store one terminal profile result, including failures."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    status: Literal["succeeded", "failed"]
    dataset_id: str
    dataset_version: str
    dataset_fingerprint: str
    judgment_status: Literal["candidate", "source_reviewed", "human_reviewed"]
    manifest_hash: str
    profile_id: str
    profile_fingerprint: str
    profile_config: dict[str, Any]
    collection_identity: str
    collection_config: dict[str, Any]
    embedding_input_strategy: str
    git_sha: str
    git_dirty: bool
    repetitions: int = Field(ge=1)
    cases: list[RetrievalCaseResult] = Field(default_factory=list)
    summary: RetrievalBenchmarkSummary | None = None
    index_elapsed_ms: int = Field(default=0, ge=0)
    index_points_count: int = Field(default=0, ge=0)
    index_footprint_bytes: int | None = Field(default=None, ge=0)
    reason_codes: list[str] = Field(default_factory=list)
    environment: dict[str, Any] = Field(default_factory=dict)


class IngestionFixtureCase(BaseModel):
    """Describe one deterministic dirty-data fixture."""

    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    fixture_type: str = Field(min_length=1)
    expected_outcome: Literal[
        "indexed",
        "metadata_only",
        "quarantined",
        "failed",
    ]
    expected_reason_codes: list[str] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class IngestionCaseObservation(BaseModel):
    """Separate expected behavior from one measured ingestion outcome."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_outcome: str
    actual_outcome: str
    expected_reason_codes: list[str] = Field(default_factory=list)
    actual_reason_codes: list[str] = Field(default_factory=list)
    metadata_correct: bool | None = None
    locator_preserved: bool | None = None
    duplicates_suppressed: bool | None = None
    incremental_update_correct: bool | None = None
    elapsed_ms: int = Field(ge=0)


class IngestionBenchmarkSummary(BaseModel):
    """Aggregate independent parser/quality behavior metrics."""

    model_config = ConfigDict(extra="forbid")

    case_count: int
    expected_outcome_accuracy: float
    reason_code_accuracy: float
    contract_accuracy: float
    indexable_parse_success: float
    quarantine_precision: float
    quarantine_recall: float
    metadata_accuracy: float
    locator_preservation: float
    duplicate_suppression: float
    incremental_update_accuracy: float
    metadata_case_count: int = Field(ge=0)
    locator_case_count: int = Field(ge=0)
    duplicate_case_count: int = Field(ge=0)
    incremental_case_count: int = Field(ge=0)
    latency_p50_ms: float
    latency_p95_ms: float


class IngestionBenchmarkResult(BaseModel):
    """Store all ingestion observations and their independent summary."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    run_id: str
    status: Literal["succeeded", "failed"]
    dataset_fingerprint: str
    cases: list[IngestionCaseObservation]
    summary: IngestionBenchmarkSummary
    reason_codes: list[str] = Field(default_factory=list)
