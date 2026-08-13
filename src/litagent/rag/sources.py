"""Materialize allowlisted arXiv sources into RawPaperAsset objects."""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
from collections.abc import Collection
from pathlib import Path

import httpx

from litagent.rag.loader import ArxivLoader
from litagent.rag.manifest import ManifestValidationError, validate_arxiv_pdf_url
from litagent.rag.models import RawPaperAsset, SourceKind, SourceRef, canonical_paper_id


class ArxivMetadataAdapter:
    """Normalize the existing arXiv loader at the corpus source boundary."""

    def __init__(self, loader: ArxivLoader | None = None) -> None:
        """Initialize the arXiv metadata adapter."""
        self._loader = loader or ArxivLoader()

    async def load(self, arxiv_id: str) -> RawPaperAsset:
        """Fetch one id and reject empty or mismatched provider results."""
        documents = await self._loader.load(arxiv_id)
        if not documents:
            raise ManifestValidationError("source_missing", "arxiv_metadata_not_found")
        document = documents[0]
        returned_id = str(document.metadata.get("arxiv_id") or "")
        paper_id = canonical_paper_id(arxiv_id=returned_id)
        expected = canonical_paper_id(arxiv_id=arxiv_id)
        if paper_id != expected:
            raise ManifestValidationError(
                "metadata_id_mismatch",
                "arxiv_metadata_id_mismatch",
            )
        title = str(document.metadata.get("title") or "").strip()
        content = document.page_content.strip()
        abstract = (
            content[len(title) :].strip()
            if title and content.startswith(title)
            else content
        )
        asset_hash = hashlib.sha256(f"{title}\n{abstract}".encode("utf-8")).hexdigest()

        return RawPaperAsset(
            paper_id=paper_id,
            title=title,
            abstract=abstract,
            arxiv_id=returned_id,
            asset_hash=asset_hash,
            sources=[
                SourceRef(
                    kind=SourceKind.ARXIV_METADATA,
                    source_id=returned_id,
                    license="arxiv",
                )
            ],
        )


class LocalPDFAdapter:
    """Verify an allowlisted local PDF before parser access."""

    def __init__(self, *, max_pdf_bytes: int) -> None:
        """Initialize the local PDF adapter."""
        self._max_pdf_bytes = max_pdf_bytes

    async def materialize(self, asset: RawPaperAsset) -> RawPaperAsset:
        """Validate size, magic and manifest hash without changing paths"""
        if asset.pdf_path is None:
            raise ManifestValidationError("source_missing", "local PDF path missing")
        if not asset.pdf_path.is_file():
            raise ManifestValidationError("source_missing", "local PDF not found")
        if asset.pdf_path.stat().st_size == 0:
            raise ManifestValidationError("empty_asset", "local asset is zero bytes")
        if asset.pdf_path.stat().st_size > self._max_pdf_bytes:
            raise ManifestValidationError(
                "asset_too_large",
                "local PDF exceeds max_pdf_bytes",
            )

        content = await asyncio.to_thread(asset.pdf_path.read_bytes)
        if content[:5] != b"%PDF-":
            raise ManifestValidationError("not_pdf", "local asset is not a PDF")
        actual_hash = hashlib.sha256(content).hexdigest()
        expected_hash = next(
            (
                source.sha256
                for source in asset.sources
                if source.kind is SourceKind.LOCAL_PDF and source.sha256
            ),
            None,
        )

        if not expected_hash or actual_hash.lower() != expected_hash.lower():
            raise ManifestValidationError(
                "asset_hash_mismatch",
                "local PDF hash mismatch",
            )
        return asset.model_copy(update={"asset_hash": actual_hash})


class ArxivPDFAdapter:
    """Download a bounded arXiv PDF atomically into the ignored raw root."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        raw_root: Path,
        max_pdf_bytes: int,
        download_timeout_seconds: float = 60.0,
        allowed_content_types: Collection[str] = (
            "application/pdf",
            "application/octet-stream",
        ),
    ) -> None:
        """Initialize the arXiv PDF adapter."""
        if download_timeout_seconds <= 0:
            raise ValueError("download_timeout_seconds must be positive")
        self._client = client
        self._raw_root = raw_root.resolve()
        self._max_pdf_bytes = max_pdf_bytes
        self._timeout = httpx.Timeout(download_timeout_seconds)
        self._allowed_content_types = frozenset(
            content_type.split(";", 1)[0].strip().lower()
            for content_type in allowed_content_types
            if content_type.strip()
        )
        if not self._allowed_content_types:
            raise ValueError("allowed_content_types must not be empty")

    def _validate_content_type(self, response: httpx.Response) -> None:
        """Reject non-PDF response media types before consuming body bytes."""
        raw_value = response.headers.get("content-type")
        content_type = raw_value.split(";", 1)[0].strip().lower() if raw_value else ""
        if content_type not in self._allowed_content_types:
            raise ManifestValidationError(
                "invalid_content_type",
                "arxiv PDF response has an invalid content type",
            )

    def _declared_content_length(self, response: httpx.Response) -> int | None:
        """Parse a single consistent decimal Content-Length value."""
        raw_value = response.headers.get("content-length")
        if raw_value is None:
            return None
        values = [value.strip() for value in raw_value.split(",")]
        if (
            not values
            or any(not re.fullmatch(r"[0-9]+", value) for value in values)
            or len(set(values)) != 1
        ):
            raise ManifestValidationError(
                "invalid_content_length",
                "arxiv PDF response has an invalid content length",
            )
        return int(values[0])

    @staticmethod
    def _expected_hash(asset: RawPaperAsset) -> str | None:
        """Return the optional manifest checksum for the arXiv PDF source."""
        return next(
            (
                source.sha256
                for source in asset.sources
                if source.kind is SourceKind.ARXIV_PDF and source.sha256
            ),
            None,
        )

    async def _reuse_cached(
        self,
        asset: RawPaperAsset,
        target: Path,
    ) -> RawPaperAsset | None:
        """Reuse a complete cached PDF after validating its bounded content."""
        if not target.is_file():
            return None
        size = target.stat().st_size
        if size == 0 or size > self._max_pdf_bytes:
            return None
        content = await asyncio.to_thread(target.read_bytes)
        if content[:5] != b"%PDF-":
            return None
        actual_hash = hashlib.sha256(content).hexdigest()
        expected_hash = self._expected_hash(asset)
        if expected_hash and actual_hash.lower() != expected_hash.lower():
            return None
        return asset.model_copy(
            update={
                "pdf_path": target,
                "asset_hash": actual_hash,
            }
        )

    async def materialize(self, asset: RawPaperAsset) -> RawPaperAsset:
        """Return a copied asset with a verified local PDF path."""
        url = validate_arxiv_pdf_url(asset.pdf_url)
        self._raw_root.mkdir(parents=True, exist_ok=True)
        safe_name = asset.paper_id.replace(":", "_").replace("/", "_")
        target = (self._raw_root / f"{safe_name}.pdf").resolve()
        if not target.is_relative_to(self._raw_root):
            raise ManifestValidationError(
                "source_not_allowlisted",
                "download target escapes raw_root",
            )
        cached = await self._reuse_cached(asset, target)
        if cached is not None:
            return cached
        temporary = target.with_suffix(".pdf.tmp")
        digest = hashlib.sha256()
        total = 0
        prefix = b""
        try:
            async with self._client.stream(
                "GET",
                url,
                follow_redirects=True,
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                validate_arxiv_pdf_url(str(response.url))
                self._validate_content_type(response)
                declared = self._declared_content_length(response)
                if declared is not None and declared > self._max_pdf_bytes:
                    raise ManifestValidationError(
                        "asset_too_large",
                        "arxiv PDF exceeds max_pdf_bytes",
                    )
                with temporary.open("wb") as handle:
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > self._max_pdf_bytes:
                            raise ManifestValidationError(
                                "asset_too_large",
                                "arxiv PDF exceeds max_pdf_bytes",
                            )
                        if len(prefix) < 5:
                            prefix = (prefix + chunk)[:5]
                        digest.update(chunk)
                        handle.write(chunk)
            if prefix != b"%PDF-":
                raise ManifestValidationError(
                    "not_pdf", "downloaded asset is not a PDF"
                )
            actual_hash = digest.hexdigest()
            expected_hash = self._expected_hash(asset)
            if expected_hash and actual_hash.lower() != expected_hash.lower():
                raise ManifestValidationError(
                    "asset_hash_mismatch",
                    "arxiv PDF sha256 mismatch",
                )
            os.replace(temporary, target)
            return asset.model_copy(
                update={
                    "pdf_path": target,
                    "asset_hash": actual_hash,
                }
            )
        except httpx.TimeoutException as exc:
            raise ManifestValidationError(
                "download_timeout",
                "arxiv PDF download timed out",
            ) from exc
        except httpx.HTTPError as exc:
            raise ManifestValidationError(
                "source_download_failed",
                "arxiv PDF download failed",
            ) from exc
        finally:
            if temporary.exists():
                temporary.unlink()
