"""Parse trusted PDFs into deterministic selected-fulltext chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from litagent.rag.chunking import CorpusChunker, PageBlockChunker
from litagent.rag.models import ContentScope, PaperRecord, RawPaperAsset
from litagent.rag.quality import (
    CleanedDocument,
    CorpusTextQualityGate,
    DocumentQualityReport,
    QualityDecision,
    RawTextBlock,
)

_HEADING = re.compile(
    r"^(abstract|introduction|related work|background|methods?|approach|"
    r"experiments?|results?|discussion|conclusions?|references)$",
    re.IGNORECASE,
)


class CorpusParseError(RuntimeError):
    """Carry a stable paper reason code for quarantine and resume."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _default_open(path: Path):
    import pymupdf

    return pymupdf.open(path)


@dataclass(frozen=True)
class ParsedPaper:
    """Bundle an optional record with its local quality and audit data."""

    record: PaperRecord | None
    quality: DocumentQualityReport
    audit: CleanedDocument


def metadata_completeness(asset: RawPaperAsset) -> float:
    """Score deterministic metadata fields without using an LLM."""
    checks = (
        bool(asset.paper_id),
        bool(asset.title.strip()),
        bool(asset.abstract.strip()),
        bool(asset.authors),
        asset.year is not None,
    )
    return sum(checks) / len(checks)


class PyMuPDFParser:
    """Extract selected text blocks while retaining page/bbox provenance."""

    def __init__(
        self,
        *,
        opener: Callable[[Path], Any] | None = None,
        quality_gate: CorpusTextQualityGate | None = None,
        chunker: CorpusChunker | None = None,
    ) -> None:
        self._opener = opener or _default_open
        self._quality_gate = quality_gate or CorpusTextQualityGate()
        self._chunker = chunker or PageBlockChunker()

    def parse(self, asset: RawPaperAsset) -> PaperRecord:
        """Parse one local PDF and always release the underlying document."""
        parsed = self.parse_with_quality(asset)
        if parsed.record is None:
            raise CorpusParseError(
                "document_quarantined",
                "quality gate quarantined PDF",
            )
        return parsed.record

    def parse_with_quality(self, asset: RawPaperAsset) -> ParsedPaper:
        if asset.pdf_path is None:
            record = PaperRecord.from_abstract_asset(asset)
            quality = DocumentQualityReport(
                decision=QualityDecision.ACCEPTED,
                metrics={"metadata_completeness": metadata_completeness(asset)},
            )
            empty = CleanedDocument(
                source_blocks=[], blocks=[], excluded_blocks=[], report=quality
            )
            return ParsedPaper(record=record, quality=quality, audit=empty)

        document = None
        try:
            with asset.pdf_path.open("rb") as handle:
                if handle.read(5) != b"%PDF-":
                    raise CorpusParseError("not_pdf", "asset lacks PDF magic")
            document = self._opener(asset.pdf_path)
            if getattr(document, "needs_pass", False) or getattr(
                document,
                "is_encrypted",
                False,
            ):
                raise CorpusParseError("encrypted_pdf", "encrypted PDF unsupported")
            return self._parse_quality_document(asset, document)
        except CorpusParseError:
            raise
        except Exception as exc:
            raise CorpusParseError("parser_failed", str(exc)) from exc
        finally:
            if document is not None:
                document.close()

    def _parse_quality_document(self, asset: RawPaperAsset, document) -> ParsedPaper:
        """Clean PDF blocks, then delegate chunk construction once."""
        base = PaperRecord.from_abstract_asset(asset)
        raw_blocks: list[RawTextBlock] = []
        current_section = "unknown"
        image_only_pages = 0

        for page_number, page in enumerate(document, start=1):
            page_blocks = []
            for block in page.get_text("blocks") or []:
                if len(block) < 7 or int(block[6]) != 0 or not str(block[4]).strip():
                    continue
                text = str(block[4])
                normalized_heading = " ".join(text.split())
                if _HEADING.fullmatch(normalized_heading):
                    current_section = normalized_heading.lower().replace(" ", "_")
                    continue
                raw = RawTextBlock.from_text(
                    paper_id=asset.paper_id,
                    text=text,
                    page=page_number,
                    block_index=int(block[5]),
                    bbox=tuple(float(value) for value in block[:4]),
                    page_width=float(page.rect.width),
                    page_height=float(page.rect.height),
                    section=current_section,
                )
                page_blocks.append(raw)
                raw_blocks.append(raw)
            if not page_blocks and page.get_images(full=True):
                image_only_pages += 1

        cleaned = self._quality_gate.evaluate(
            raw_blocks,
            page_count=len(document),
            image_only_pages=image_only_pages,
            metadata_completeness=metadata_completeness(asset),
        )
        if cleaned.report.decision is QualityDecision.QUARANTINED:
            return ParsedPaper(record=None, quality=cleaned.report, audit=cleaned)

        try:
            fulltext_chunks = self._chunker.chunk(
                paper_id=asset.paper_id,
                blocks=cleaned.blocks,
            )
        except Exception as exc:
            raise CorpusParseError("chunking_failed", str(exc)) from exc

        canonical_abstract = " ".join(base.abstract.split())
        fulltext_chunks = [
            chunk
            for chunk in fulltext_chunks
            if not canonical_abstract or chunk.text != canonical_abstract
        ]
        chunks = [*base.chunks, *fulltext_chunks]
        has_fulltext = bool(fulltext_chunks)
        record = base.model_copy(
            update={
                "content_scope": (
                    ContentScope.SELECTED_FULLTEXT
                    if has_fulltext
                    else base.content_scope
                ),
                "chunks": chunks,
                "warnings": list(
                    dict.fromkeys(
                        [
                            *base.warnings,
                            *cleaned.report.reason_codes,
                            *cleaned.report.warnings,
                        ]
                    )
                ),
                "ocr_required": "ocr_required" in cleaned.report.reason_codes,
            }
        )
        return ParsedPaper(record=record, quality=cleaned.report, audit=cleaned)
