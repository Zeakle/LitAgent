"""Define canonical paper contracts across ingestion, retrieval, and the DAG."""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SourceKind(str, Enum):
    """Identify one trusted corpus source boundary."""

    MANIFEST = "manifest"
    ARXIV_METADATA = "arxiv_metadata"
    LOCAL_PDF = "local_pdf"
    ARXIV_PDF = "arxiv_pdf"


class ContentScope(str, Enum):
    """Describe the strongest content available for a paper."""

    METADATA_ONLY = "metadata_only"
    ABSTRACT = "abstract"
    SELECTED_FULLTEXT = "selected_fulltext"


_SCOPE_RANK = {
    ContentScope.METADATA_ONLY: 0,
    ContentScope.ABSTRACT: 1,
    ContentScope.SELECTED_FULLTEXT: 2,
}


def _digest(value: Any) -> str:
    """Return a stable content digest."""
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_paper_id(
    *,
    arxiv_id: str | None = None,
    doi: str | None = None,
    source: str | None = None,
    source_id: str | None = None,
) -> str:
    """Return a stable namespaced id without provider-version noise."""
    if doi:
        normalized = doi.strip().lower()
        normalized = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", normalized)
        if normalized:
            return f"doi:{normalized}"
    if arxiv_id:
        normalized = arxiv_id.strip().split("/")[-1]
        normalized = re.sub(r"v\d+$", "", normalized, flags=re.IGNORECASE)
        if normalized:
            return f"arxiv:{normalized}"
    if source and source_id:
        normalized = source_id.strip().lower()
        if normalized:
            return f"{source.strip().lower()}:{normalized}"
    raise ValueError("paper identity requires DOI, arXIV id, or source/source_id")


class SourceRef(BaseModel):
    """Record reviewable provenance without embedding local file paths."""

    model_config = ConfigDict(extra="forbid")

    kind: SourceKind
    source_id: str
    uri: str | None = None
    license: str | None = None
    sha256: str | None = None


class RawPaperAsset(BaseModel):
    """Carry validated local ingestion input before parsing."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    asset_hash: str
    sources: list[SourceRef] = Field(default_factory=list)
    pdf_path: Path | None = None
    pdf_url: str | None = None
    warnings: list[str] = Field(default_factory=list)


class ChunkSourceSpan(BaseModel):
    """Locate one chunk fragment in a cleaned PDF text block."""

    model_config = ConfigDict(extra="forbid")

    page: int = Field(ge=1)
    block_index: int = Field(ge=0)
    bbox: tuple[float, float, float, float]
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    raw_content_hash: str = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_span(self):
        """Validate source-span offsets and page bounds."""
        if self.start_char >= self.end_char:
            raise ValueError("source span start_char must be smaller than end_char")
        if not all(math.isfinite(value) for value in self.bbox):
            raise ValueError("source span bbox values must be finite")
        return self


class ContentChunk(BaseModel):
    """Represent one deterministic retrieval unit and its source lineage."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    chunk_key: str
    text: str = Field(min_length=1)
    section: str
    content_scope: ContentScope
    content_hash: str
    raw_content_hash: str | None = None
    transformations: list[str] = Field(default_factory=list)
    page: int | None = Field(default=None, ge=1)
    block_index: int | None = Field(default=None, ge=0)
    bbox: tuple[float, float, float, float] | None = None
    source_spans: list[ChunkSourceSpan] = Field(default_factory=list)

    @classmethod
    def from_text(
        cls,
        *,
        paper_id: str,
        chunk_key: str,
        text: str,
        section: str,
        content_scope: ContentScope,
        raw_text: str | None = None,
        raw_content_hash: str | None = None,
        transformations: Sequence[str] = (),
        page: int | None = None,
        block_index: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
        source_spans: Sequence[ChunkSourceSpan | dict[str, Any]] = (),
    ) -> "ContentChunk":
        """Normalize text while preserving explicit raw and locator lineage."""
        normalized = " ".join(text.split()).strip()
        if not normalized:
            raise ValueError("empty chunks are not indexable")

        spans = [ChunkSourceSpan.model_validate(span) for span in source_spans]
        if spans:
            first = spans[0]
            page = first.page if page is None else page
            block_index = first.block_index if block_index is None else block_index
            bbox = first.bbox if bbox is None else bbox

        if raw_content_hash is None:
            if raw_text is not None:
                raw_content_hash = hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
            elif spans:
                lineage = "|".join(span.raw_content_hash for span in spans)
                raw_content_hash = hashlib.sha256(lineage.encode("utf-8")).hexdigest()
            else:
                raw_content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return cls(
            paper_id=paper_id,
            chunk_key=chunk_key,
            text=normalized,
            section=section.strip().lower() or "unknown",
            content_scope=content_scope,
            content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            raw_content_hash=raw_content_hash,
            transformations=list(dict.fromkeys(transformations)),
            page=page,
            block_index=block_index,
            bbox=bbox,
            source_spans=spans,
        )

    @model_validator(mode="after")
    def _validate_locator(self):
        """Validate locator requirements for the chunk type."""
        locator_values = (self.page, self.block_index, self.bbox)
        if self.content_scope is ContentScope.ABSTRACT:
            if any(value is not None for value in locator_values) or self.source_spans:
                raise ValueError("abstract chunks cannot have PDF locators")
            return self

        if self.content_scope is ContentScope.SELECTED_FULLTEXT and any(
            value is None for value in locator_values
        ):
            raise ValueError("selected-fulltext chunks require page, block and bbox")
        if self.source_spans:
            first = self.source_spans[0]
            if (self.page, self.block_index, self.bbox) != (
                first.page,
                first.block_index,
                first.bbox,
            ):
                raise ValueError("legacy locator must match the first source span")
        return self

    @property
    def locator(self) -> dict[str, Any]:
        """Return the stable locator copied into EvidenceItem v2."""
        return {
            "paper_id": self.paper_id,
            "chunk_key": self.chunk_key,
            "section": self.section,
            "page": self.page,
            "block_index": self.block_index,
            "bbox": list(self.bbox) if self.bbox else None,
            "source_spans": [
                span.model_dump(mode="json") for span in self.source_spans
            ],
            "content_scope": self.content_scope.value,
            "content_hash": self.content_hash,
            "raw_content_hash": self.raw_content_hash,
        }


class PaperRecord(BaseModel):
    """Represent one normalized paper ready for corpus synchronization."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    content_scope: ContentScope
    chunks: list[ContentChunk] = Field(default_factory=list)
    asset_hash: str
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    ocr_required: bool = False

    @property
    def metadata_hash(self) -> str:
        """Hash metadata projected into every Qdrant chunk payload"""
        return _digest(
            {
                "paper_id": self.paper_id,
                "title": self.title,
                "abstract": self.abstract,
                "authors": self.authors,
                "year": self.year,
                "doi": self.doi,
                "arxiv_id": self.arxiv_id,
                "content_scope": self.content_scope.value,
                "sources": [source.model_dump(mode="json") for source in self.sources],
                "warnings": self.warnings,
                "ocr_required": self.ocr_required,
            }
        )

    @classmethod
    def from_abstract_asset(cls, asset: RawPaperAsset) -> "PaperRecord":
        """Normalize API/manifest abstract into standard chunk shape"""
        chunks: list[ContentChunk] = []
        if asset.abstract.strip():
            chunks.append(
                ContentChunk.from_text(
                    paper_id=asset.paper_id,
                    chunk_key="abstract",
                    text=asset.abstract,
                    section="abstract",
                    content_scope=ContentScope.ABSTRACT,
                )
            )

        return cls(
            paper_id=asset.paper_id,
            title=asset.title,
            abstract=asset.abstract.strip(),
            authors=asset.authors,
            year=asset.year,
            doi=asset.doi,
            arxiv_id=asset.arxiv_id,
            content_scope=(
                ContentScope.ABSTRACT if chunks else ContentScope.METADATA_ONLY
            ),
            chunks=chunks,
            asset_hash=asset.asset_hash,
            sources=asset.sources,
            warnings=asset.warnings,
        )


class ScoredChunkHit(BaseModel):
    """Return one versioned Qdrant chunk hit."""

    model_config = ConfigDict(extra="forbid")

    chunk: ContentChunk
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    score: float
    collection: str
    corpus_version: str
    schema_version: str
    parser_version: str
    chunking_version: str
    chunk_strategy: str = "page_block"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    embedding_backend: str = "sentence_transformer"
    embedding_model: str
    embedding_document_adapter: str | None = None


class ScoredPaperHit(BaseModel):
    """Aggregate multiple chunk hits into one query-specific paper result."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    sources: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    content_scope: ContentScope
    chunks: list[ContentChunk]
    score: float
    collection: str
    corpus_version: str
    schema_version: str
    parser_version: str
    chunking_version: str
    chunk_strategy: str = "page_block"
    chunk_size: int = 1200
    chunk_overlap: int = 150
    embedding_backend: str = "sentence_transformer"
    embedding_model: str
    embedding_document_adapter: str | None = None


class PaperCandidate(BaseModel):
    """Expose the sole paper contract consumed by the Survey DAG."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    citation_count: int = 0
    source: str
    score: float | None = None
    content_scope: ContentScope = ContentScope.METADATA_ONLY
    chunks: list[ContentChunk] = Field(default_factory=list)
    provenance: list[SourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    retrieval: dict[str, Any] | None = None

    @classmethod
    def from_external(cls, raw: dict[str, Any]) -> "PaperCandidate":
        """Validate and project an untrusted provider response."""
        source = str(raw.get("source") or "").strip().lower()
        provider_id = str(raw.get("paper_id") or "").strip()
        if not source or not provider_id:
            raise ValueError("external paper requires source and paper_id")
        paper_id = canonical_paper_id(
            arxiv_id=provider_id if source == "arxiv" else None,
            doi=str(raw.get("doi") or "") or None,
            source=source,
            source_id=provider_id,
        )
        abstract = str(raw.get("abstract") or "").strip()
        chunks = (
            [
                ContentChunk.from_text(
                    paper_id=paper_id,
                    chunk_key="abstract",
                    text=abstract,
                    section="abstract",
                    content_scope=ContentScope.ABSTRACT,
                )
            ]
            if abstract
            else []
        )

        return cls(
            paper_id=paper_id,
            title=str(raw.get("title") or "").strip(),
            abstract=abstract,
            authors=[
                str(author)
                for author in (raw.get("authors") or [])
                if str(author).strip()
            ],
            year=raw.get("year"),
            citation_count=max(int(raw.get("citation_count") or 0), 0),
            source=source,
            content_scope=(
                ContentScope.ABSTRACT if abstract else ContentScope.METADATA_ONLY
            ),
            chunks=chunks,
        )

    @classmethod
    def from_scored_hit(cls, hit: ScoredPaperHit) -> "PaperCandidate":
        """Project a typed retrieval hit into the DAG contract."""
        return cls(
            paper_id=hit.paper_id,
            title=hit.title,
            abstract=hit.abstract,
            authors=hit.authors,
            year=hit.year,
            source="rag_index",
            score=hit.score,
            content_scope=hit.content_scope,
            chunks=hit.chunks,
            provenance=hit.sources,
            warnings=hit.warnings,
            retrieval={
                "collection": hit.collection,
                "corpus_version": hit.corpus_version,
                "schema_version": hit.schema_version,
                "parser_version": hit.parser_version,
                "chunking_version": hit.chunking_version,
                "chunk_strategy": hit.chunk_strategy,
                "chunk_size": hit.chunk_size,
                "chunk_overlap": hit.chunk_overlap,
                "embedding_backend": hit.embedding_backend,
                "embedding_model": hit.embedding_model,
                "embedding_document_adapter": hit.embedding_document_adapter,
            },
        )

    def to_dag_dict(self) -> dict[str, Any]:
        """Serialize enums and nested chunks for existing dict consumers."""
        return self.model_dump(mode="json", exclude_none=True)


def merge_paper_records(records: list[PaperRecord]) -> list[PaperRecord]:
    """Merge abstract/PDF records by stable id without duplicating chunks."""
    grouped: dict[str, PaperRecord] = {}
    for record in records:
        current = grouped.get(record.paper_id)
        if current is None:
            grouped[record.paper_id] = record.model_copy(deep=True)
            continue
        chunks = _merge_chunks(current.chunks, record.chunks)
        sources = {
            (
                source.kind.value,
                source.source_id,
                source.sha256,
            ): source
            for source in [*current.sources, *record.sources]
        }
        strongest = max(
            (current.content_scope, record.content_scope),
            key=_SCOPE_RANK.__getitem__,
        )
        grouped[record.paper_id] = current.model_copy(
            update={
                "title": record.title or current.title,
                "abstract": record.abstract or current.abstract,
                "authors": record.authors or current.authors,
                "year": record.year or current.year,
                "content_scope": strongest,
                "chunks": chunks,
                "asset_hash": _digest(sorted({current.asset_hash, record.asset_hash})),
                "sources": list(sources.values()),
                "warnings": list(dict.fromkeys([*current.warnings, *record.warnings])),
                "ocr_required": current.ocr_required or record.ocr_required,
            }
        )
    return list(grouped.values())


def _merge_chunks(
    current: Sequence[ContentChunk],
    incoming: Sequence[ContentChunk],
) -> list[ContentChunk]:
    """Merge by key while preserving order and preferring richer content."""
    merged = list(current)
    positions = {chunk.chunk_key: index for index, chunk in enumerate(merged)}
    for chunk in incoming:
        position = positions.get(chunk.chunk_key)
        if position is None:
            positions[chunk.chunk_key] = len(merged)
            merged.append(chunk)
            continue
        if (
            _SCOPE_RANK[chunk.content_scope]
            >= _SCOPE_RANK[merged[position].content_scope]
        ):
            merged[position] = chunk
    return merged


def merge_paper_candidates(
    candidates: list[PaperCandidate],
) -> list[PaperCandidate]:
    """Merge Search/Recall candidates without discarding richer chunks."""
    grouped: dict[str, PaperCandidate] = {}
    title_keys: dict[str, str] = {}
    for candidate in candidates:
        fallback = re.sub(r"\W+", "", candidate.title.lower())
        key = (
            candidate.paper_id
            if candidate.paper_id in grouped
            else title_keys.get(fallback, candidate.paper_id)
        )
        current = grouped.get(key)
        if current is None:
            grouped[key] = candidate.model_copy(deep=True)
            if fallback:
                title_keys[fallback] = key
            continue

        merged_chunks = _merge_chunks(current.chunks, candidate.chunks)
        strongest = max(
            (current.content_scope, candidate.content_scope),
            key=_SCOPE_RANK.__getitem__,
        )
        external_source = next(
            (
                source
                for source in (current.source, candidate.source)
                if source != "rag_index"
            ),
            current.source,
        )
        retrieval = current.retrieval or candidate.retrieval
        grouped[key] = current.model_copy(
            update={
                "title": candidate.title or current.title,
                "abstract": candidate.abstract or current.abstract,
                "authors": candidate.authors or current.authors,
                "year": candidate.year or current.year,
                "citation_count": max(
                    current.citation_count,
                    candidate.citation_count,
                ),
                "source": external_source,
                "score": (
                    max(
                        value
                        for value in (current.score, candidate.score)
                        if value is not None
                    )
                    if current.score is not None or candidate.score is not None
                    else None
                ),
                "content_scope": strongest,
                "chunks": merged_chunks,
                "provenance": list(
                    {
                        (
                            source.kind.value,
                            source.source_id,
                            source.sha256,
                        ): source
                        for source in [
                            *current.provenance,
                            *candidate.provenance,
                        ]
                    }.values()
                ),
                "warnings": list(
                    dict.fromkeys([*current.warnings, *candidate.warnings])
                ),
                "retrieval": retrieval,
            }
        )
    return list(grouped.values())
