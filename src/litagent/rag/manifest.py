"""Load a versioned, reviewable paper-corpus manifest."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from litagent.rag.models import RawPaperAsset, SourceKind, SourceRef, canonical_paper_id


class ManifestValidationError(ValueError):
    """Carry a stable source-boundary code and a safe diagnostic message."""

    def __init__(self, code: str, message: str | None = None) -> None:
        """Initialize the manifest validation error."""
        super().__init__(message or code)
        self.code = code


class PDFSource(BaseModel):
    """Describe one local or allowlisted arXiv PDF."""

    model_config = ConfigDict(extra="forbid")

    kind: SourceKind
    path: str | None = None
    url: str | None = None
    sha256: str | None = None

    @model_validator(mode="after")
    def _validate_kind(self):
        """Validate source fields for the selected PDF kind."""
        if self.kind not in {SourceKind.LOCAL_PDF, SourceKind.ARXIV_PDF}:
            raise ValueError("pdf.kind must be local_pdf or arxiv_pdf")
        return self


class ManifestPaper(BaseModel):
    """Describe one canonical manifest paper."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    title: str
    abstract: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    license: str | None = None
    pdf: PDFSource | None = None


class CorpusManifest(BaseModel):
    """Hold a validated manifest and its trusted local root."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    corpus_version: str
    papers: list[ManifestPaper]
    raw_root: Path

    def to_assets(self) -> list[RawPaperAsset]:
        """Resolve all entries into the common RawPaperAsset boundary."""
        assets: list[RawPaperAsset] = []
        for paper in self.papers:
            sources = [
                SourceRef(
                    kind=SourceKind.MANIFEST,
                    source_id=paper.paper_id,
                    license=paper.license,
                )
            ]
            if paper.arxiv_id:
                sources.append(
                    SourceRef(
                        kind=SourceKind.ARXIV_METADATA,
                        source_id=paper.arxiv_id,
                        license=paper.license,
                    )
                )

            pdf_path = None
            pdf_url = None
            asset_hash = hashlib.sha256(
                f"{paper.title}\n{paper.abstract}".encode("utf-8")
            ).hexdigest()
            if paper.pdf and paper.pdf.kind is SourceKind.LOCAL_PDF:
                pdf_path = _resolve_local_pdf(
                    self.raw_root,
                    paper.pdf.path,
                )
                if not paper.pdf.sha256:
                    raise ManifestValidationError(
                        "asset_hash_mismatch",
                        "local_pdf requires expected sha256",
                    )
                sources.append(
                    SourceRef(
                        kind=SourceKind.LOCAL_PDF,
                        source_id=paper.paper_id,
                        license=paper.license,
                        sha256=paper.pdf.sha256,
                    )
                )
            elif paper.pdf and paper.pdf.kind is SourceKind.ARXIV_PDF:
                pdf_url = validate_arxiv_pdf_url(paper.pdf.url)
                sources.append(
                    SourceRef(
                        kind=SourceKind.ARXIV_PDF,
                        source_id=paper.arxiv_id or paper.paper_id,
                        uri=pdf_url,
                        license=paper.license,
                        sha256=paper.pdf.sha256,
                    )
                )

            expected_id = canonical_paper_id(
                doi=paper.doi,
                arxiv_id=paper.arxiv_id,
                source="manifest",
                source_id=paper.paper_id,
            )
            if paper.paper_id != expected_id:
                raise ManifestValidationError(
                    "metadata_id_mismatch",
                    f"paper_id mismatch: {paper.paper_id!r} != {expected_id!r}",
                )
            assets.append(
                RawPaperAsset(
                    paper_id=paper.paper_id,
                    title=paper.title.strip(),
                    abstract=paper.abstract.strip(),
                    authors=paper.authors,
                    year=paper.year,
                    doi=paper.doi,
                    arxiv_id=paper.arxiv_id,
                    asset_hash=asset_hash,
                    sources=sources,
                    pdf_path=pdf_path,
                    pdf_url=pdf_url,
                )
            )
        return assets


def validate_arxiv_pdf_url(value: str | None) -> str:
    """Allow only HTTPS PDF endpoints owned by arXiv."""
    if not value:
        raise ManifestValidationError("source_missing", "arxiv_pdf requires url")
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname not in {"arxiv.org", "export.arxiv.org"}
        or not parsed.path.startswith("/pdf/")
        or parsed.username
        or parsed.password
    ):
        raise ManifestValidationError(
            "source_not_allowlisted",
            "PDF URL is outside the arXiv allowlist",
        )
    return value


def _resolve_local_pdf(
    raw_root: Path,
    relative_path: str | None,
) -> Path:
    """Resolve a local source without allowing traversal outside raw_root."""
    if not relative_path:
        raise ManifestValidationError("source_missing", "local_pdf requires path")
    root = raw_root.resolve()
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root):
        raise ManifestValidationError(
            "source_not_allowlisted",
            "local PDF escapes raw_root",
        )
    return candidate


def validate_unique_assets(manifest: CorpusManifest) -> None:
    """Reject duplicate paper ids and duplicate declared PDF hashes."""
    paper_ids: set[str] = set()
    pdf_hashes: dict[str, str] = {}
    for paper in manifest.papers:
        if paper.paper_id in paper_ids:
            raise ManifestValidationError(
                "duplicate_paper_id",
                f"duplicate paper_id: {paper.paper_id!r}",
            )
        paper_ids.add(paper.paper_id)
        sha256 = paper.pdf.sha256.lower() if paper.pdf and paper.pdf.sha256 else ""
        owner = pdf_hashes.get(sha256) if sha256 else None
        if owner is not None and owner != paper.paper_id:
            raise ManifestValidationError(
                "duplicate_asset",
                f"PDF sha256 shared by {owner!r} and {paper.paper_id!r}",
            )
        if sha256:
            pdf_hashes[sha256] = paper.paper_id


def load_manifest(
    path: Path,
    *,
    raw_root: Path,
) -> CorpusManifest:
    """Load YAML, reject unknown fields, and validate every source eagerly."""
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        manifest = CorpusManifest.model_validate(
            {**payload, "raw_root": raw_root.resolve()}
        )

        if manifest.schema_version != 1:
            raise ManifestValidationError(
                "manifest_invalid",
                "unsupported manifest schema_version",
            )
    except ManifestValidationError:
        raise
    except Exception as exc:
        raise ManifestValidationError("manifest_invalid", str(exc)) from exc

    validate_unique_assets(manifest)
    materialize_manifest_assets(manifest)
    return manifest


def materialize_manifest_assets(manifest: CorpusManifest) -> list[RawPaperAsset]:
    """Convert entries while preserving coded errors and normalizing surprises."""
    try:
        return manifest.to_assets()
    except ManifestValidationError:
        raise
    except Exception as exc:
        raise ManifestValidationError("manifest_invalid", str(exc)) from exc
