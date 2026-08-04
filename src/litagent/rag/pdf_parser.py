"""Parse trusted PDFs into deterministic selected-fulltext chunks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

from litagent.rag.models import ContentChunk, ContentScope, PaperRecord, RawPaperAsset

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


class PyMuPDFParser:
    """Extract selected text blocks while retaining page/bbox provenance."""

    def __init__(
        self,
        *,
        opener: Callable[[Path], Any] | None = None,
    ) -> None:
        self._opener = opener or _default_open

    def parse(self, asset: RawPaperAsset) -> PaperRecord:
        """Parse one local PDF and always release the underlying document."""
        if asset.pdf_path is None:
            return PaperRecord.from_abstract_asset(asset)
        document = None
        try:
            if asset.pdf_path.read_bytes()[:5] != b"%PDF-":
                raise CorpusParseError("not_pdf", "asset lacks PDF magic")
            document = self._opener(asset.pdf_path)
            if getattr(document, "needs_pass", False) or getattr(
                document,
                "is_encrypted",
                False,
            ):
                raise CorpusParseError("encrypted_pdf", "encrypted PDF unsupported")
            return self._parse_document(asset, document)
        except CorpusParseError:
            raise
        except Exception as exc:
            raise CorpusParseError("parser_failed", str(exc)) from exc
        finally:
            if document is not None:
                document.close()

    def _parse_document(self, asset: RawPaperAsset, document: Any) -> PaperRecord:
        """Convert text blocks while treating scanned pages as degradation."""
        base = PaperRecord.from_abstract_asset(asset)
        chunks = list(base.chunks)
        warnings = list(base.warnings)
        current_section = "unknown"
        image_only_pages = 0

        for page_number, page in enumerate(document, start=1):
            blocks = page.get_text("blocks") or []
            text_blocks = [
                block
                for block in blocks
                if len(block) >= 7 and int(block[6]) == 0 and str(block[4]).strip()
            ]
            if not text_blocks and page.get_images(full=True):
                image_only_pages += 1
                continue
            for block in text_blocks:
                text = " ".join(str(block[4]).split()).strip()
                if _HEADING.fullmatch(text):
                    current_section = text.lower().replace(" ", "_")
                    continue

                if base.abstract and text == " ".join(base.abstract.split()):
                    # Canonical abstract chunk already represents this text.
                    continue
                block_index = int(block[5])
                chunks.append(
                    ContentChunk.from_text(
                        paper_id=asset.paper_id,
                        chunk_key=f"page:{page_number}:block:{block_index}",
                        text=text,
                        section=current_section,
                        content_scope=ContentScope.SELECTED_FULLTEXT,
                        page=page_number,
                        block_index=block_index,
                        bbox=tuple(float(value) for value in block[:4]),
                    )
                )

        ocr_required = image_only_pages > 0 and not any(
            chunk.content_scope is ContentScope.SELECTED_FULLTEXT for chunk in chunks
        )
        if ocr_required:
            warnings.append("ocr_required:image_only_pdf")
        scope = (
            ContentScope.SELECTED_FULLTEXT
            if any(
                chunk.content_scope is ContentScope.SELECTED_FULLTEXT
                for chunk in chunks
            )
            else base.content_scope
        )
        return base.model_copy(
            update={
                "content_scope": scope,
                "chunks": chunks,
                "warnings": warnings,
                "ocr_required": ocr_required,
            }
        )
