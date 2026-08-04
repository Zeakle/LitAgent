"""Define canonical paper contracts across ingestion, retrieval, and the DAG."""

from __future__ import annotations

import hashlib
import json
import re
from enum import Enum
from pathlib import Path
from typing import Any

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


class ContentChunk(BaseModel):
    """Represent one deterministic retrieval unit and its source locator."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    chunk_key: str
    text: str = Field(min_length=1)
    section: str
    content_scope: ContentScope
    content_hash: str
    page: int | None = Field(default=None, ge=1)
    block_index: int | None = Field(default=None, ge=0)
    bbox: tuple[float, float, float, float] | None = None

    @classmethod
    def from_text(
        cls,
        *,
        paper_id: str,
        chunk_key: str,
        text: str,
        section: str,
        content_scope: ContentScope,
        page: int | None = None,
        block_index: int | None = None,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> "ContentChunk":
        """Normalize text once and drive its stable content hash."""
        normalized = " ".join(text.split()).strip()
        if not normalized:
            raise ValueError("empty chunks are not indexable")

        return cls(
            paper_id=paper_id,
            chunk_key=chunk_key,
            text=normalized,
            section=section.strip().lower() or "unknown",
            content_scope=content_scope,
            content_hash=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
            page=page,
            block_index=block_index,
            bbox=bbox,
        )

    @model_validator(mode="after")
    def _validate_locator(self):
        if self.content_scope is ContentScope.ABSTRACT:
            if any(
                value is not None for value in (self.page, self.block_index, self.bbox)
            ):
                raise ValueError("abstract chunks cannot have PDF locators")
        return self


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
    """Return one versioned Qdrant chunk hit"""

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
    embedding_model: str


class ScoredPaperHit(BaseModel):
    """Aggregate multiple chunk hits into one query-specific paper result"""

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
    embedding_model: str


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
                "embedding_model": hit.embedding_model,
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
        chunks = {chunk.chunk_key: chunk for chunk in current.chunks}
        for chunk in record.chunks:
            existing = chunks.get(chunk.chunk_key)
            if (
                existing is None
                or _SCOPE_RANK[chunk.content_scope]
                >= _SCOPE_RANK[existing.content_scope]
            ):
                chunks[chunk.chunk_key] = chunk
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
                "chunks": sorted(chunks.values(), key=lambda item: item.chunk_key),
                "asset_hash": _digest(sorted({current.asset_hash, record.asset_hash})),
                "sources": list(sources.values()),
                "warnings": list(dict.fromkeys([*current.warnings, *record.warnings])),
                "ocr_required": current.ocr_required or record.ocr_required,
            }
        )
    return list(grouped.values())


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

        chunks = {chunk.chunk_key: chunk for chunk in current.chunks}
        chunks.update({chunk.chunk_key: chunk for chunk in candidate.chunks})
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
                "chunks": sorted(chunks.values(), key=lambda item: item.chunk_key),
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
