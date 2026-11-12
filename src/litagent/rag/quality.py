"""Clean untrusted PDF blocks and produce explainable quality decision."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from litagent.safety.injection import InjectionDetector, InjectionRisk


class QualityDecision(str, Enum):
    """Describe whether cleaned document text may enter the corpus."""

    ACCEPTED = "accepted"
    DEGRADED = "degraded"
    QUARANTINED = "quarantined"


class IngestionOutcome(str, Enum):
    """Describe the paper-level content outcome independently of storage state."""

    INDEXED = "indexed"
    METADATA_ONLY = "metadata_only"
    QUARANTINED = "quarantined"
    FAILED = "failed"


class RawTextBlock(BaseModel):
    """Keep immutable parser output before trust decisions."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    raw_text: str
    raw_content_hash: str
    page: int = Field(ge=1)
    block_index: int = Field(ge=0)
    bbox: tuple[float, float, float, float]
    page_width: float = Field(gt=0)
    page_height: float = Field(gt=0)
    section: str = "unknown"

    @classmethod
    def from_text(cls, *, text: str, **kwargs) -> "RawTextBlock":
        """Build a raw text block with derived measurements."""
        return cls(
            raw_text=text,
            raw_content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            **kwargs,
        )


class CleanTextBlock(BaseModel):
    """Represent one trusted block ready to become a ContentChunk."""

    model_config = ConfigDict(extra="forbid")

    paper_id: str
    text: str
    raw_content_hash: str
    transformations: list[str] = Field(default_factory=list)
    page: int
    block_index: int
    bbox: tuple[float, float, float, float]
    section: str


class ExcludedTextBlock(BaseModel):
    """Retain local-only audit data for a block excluded from indexing."""

    model_config = ConfigDict(extra="forbid")

    raw_text: str
    raw_content_hash: str
    page: int
    block_index: int
    bbox: tuple[float, float, float, float]
    reason_code: str
    injection_risk: str = "none"


class DocumentQualityMetrics(BaseModel):
    """Store normalized parser and content-quality measurements."""

    page_count: int = 0
    text_page_count: int = 0
    text_page_ratio: float = 0.0
    metadata_completeness: float = Field(default=0.0, ge=0.0, le=1.0)
    duplicate_block_ratio: float = 0.0
    gibberish_ratio: float = 0.0
    section_count: int = 0
    locator_coverage: float = 0.0


class DocumentQualityReport(BaseModel):
    """Describe a document-quality decision and its stable reasons."""

    decision: QualityDecision
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: DocumentQualityMetrics = Field(default_factory=DocumentQualityMetrics)


class PaperIngestionReport(BaseModel):
    """Join quality outcome with, but never conflate it with, storage state."""

    paper_id: str
    outcome: IngestionOutcome
    storage_status: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    metrics: DocumentQualityMetrics = Field(default_factory=DocumentQualityMetrics)
    elapsed_ms: int = Field(default=0, ge=0)


class CleanedDocument(BaseModel):
    """Join cleaned blocks, excluded content, and the quality report."""

    source_blocks: list[RawTextBlock]
    blocks: list[CleanTextBlock]
    excluded_blocks: list[ExcludedTextBlock]
    report: DocumentQualityReport


_LIGATURES = str.maketrans({"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi"})
_PAGE_NUMBER = re.compile(r"^(?:page\s+)?\d+(?:\s*/\s*\d+)?$", re.IGNORECASE)


def _normalize_text(raw: str) -> tuple[str, list[str]]:
    """Normalize extracted text and return applied repair codes."""
    transformations: list[str] = []
    text = unicodedata.normalize("NFKC", raw).translate(_LIGATURES)
    if text != raw:
        transformations.append("ligature_normalized")
    dehyphenated = re.sub(r"-\s*\n\s*", "-", text)
    if dehyphenated != text:
        transformations.append("dehyphenated")
    normalized = " ".join(dehyphenated.split())
    if normalized != dehyphenated.strip():
        transformations.append("whitespace_normalized")
    return normalized, transformations


def _gibberish_ratio(text: str) -> float:
    """Estimate the fraction of non-readable characters."""
    if not text:
        return 1.0
    bad = sum(
        1
        for char in text
        if char == "�" or (unicodedata.category(char) == "Cc" and char not in "\n\t")
    )
    return bad / len(text)


def _is_margin(block: RawTextBlock, margin_ratio: float) -> bool:
    """Return whether a block lies in a page margin."""
    top = block.bbox[1] <= block.page_height * margin_ratio
    bottom = block.bbox[3] >= block.page_height * (1.0 - margin_ratio)
    return top or bottom


def _has_double_column(blocks: list[RawTextBlock]) -> bool:
    """Return whether blocks indicate a two-column layout."""
    by_page: dict[int, list[RawTextBlock]] = {}
    for block in blocks:
        by_page.setdefault(block.page, []).append(block)
    for page_blocks in by_page.values():
        left = [
            block for block in page_blocks if block.bbox[2] <= block.page_width * 0.55
        ]
        right = [
            block for block in page_blocks if block.bbox[0] >= block.page_width * 0.45
        ]
        if (
            left
            and right
            and any(
                a.block_index != b.block_index
                and a.bbox[2] <= b.bbox[0]
                and max(a.bbox[1], b.bbox[1]) < min(a.bbox[3], b.bbox[3])
                for a in left
                for b in right
            )
        ):
            return True
    return False


class CorpusTextQualityGate:
    """Apply deterministic cleaning before any vector or LLM boundary."""

    def __init__(
        self,
        *,
        detector: InjectionDetector | None = None,
        max_gibberish_ratio: float = 0.15,
        min_text_page_ratio: float = 0.20,
        repeated_margin_min_pages: int = 2,
        margin_ratio: float = 0.10,
    ) -> None:
        """Initialize the corpus text quality gate."""
        self._detector = detector or InjectionDetector()
        self._max_gibberish_ratio = max_gibberish_ratio
        self._min_text_page_ratio = min_text_page_ratio
        self._repeated_margin_min_pages = repeated_margin_min_pages
        self._margin_ratio = margin_ratio

    def evaluate(
        self,
        blocks: list[RawTextBlock],
        *,
        page_count: int,
        image_only_pages: int,
        metadata_completeness: float = 0.0,
    ) -> CleanedDocument:
        """Clean text blocks and produce a document-quality decision."""
        normalized = [(block, *_normalize_text(block.raw_text)) for block in blocks]
        margin_pages: dict[str, set[int]] = {}
        for block, text, _ in normalized:
            if text and _is_margin(block, self._margin_ratio):
                margin_pages.setdefault(text.casefold(), set()).add(block.page)
        repeated_margin = {
            text
            for text, pages in margin_pages.items()
            if len(pages) >= self._repeated_margin_min_pages
        }

        trusted: list[CleanTextBlock] = []
        excluded: list[ExcludedTextBlock] = []
        reasons: list[str] = []
        warnings: list[str] = []
        seen_body: set[str] = set()
        duplicate_count = 0
        gibberish_values: list[float] = []

        for block, text, transformations in normalized:
            if not text:
                continue
            if text.casefold() in repeated_margin:
                reasons.append("repeated_header_footer_removed")
                excluded.append(self._excluded(block, "repeated_margin"))
                continue
            if _is_margin(block, self._margin_ratio) and _PAGE_NUMBER.fullmatch(text):
                reasons.append("page_number_removed")
                excluded.append(self._excluded(block, "page_number"))
                continue

            detection = self._detector.scan(text)
            if detection.risk is InjectionRisk.HIGH:
                return self._quarantined(
                    blocks, block, page_count, "prompt_injection_high"
                )
            if detection.risk is InjectionRisk.SUSPICIOUS:
                reasons.append("suspicious_block_excluded")
                excluded.append(self._excluded(block, "prompt_injection", "suspicious"))
                continue

            gibberish = _gibberish_ratio(text)
            gibberish_values.append(gibberish)
            if gibberish > self._max_gibberish_ratio:
                reasons.append("gibberish_block_excluded")
                excluded.append(self._excluded(block, "gibberish"))
                continue

            body_key = hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()
            if body_key in seen_body:
                duplicate_count += 1
                reasons.append("duplicate_block_removed")
                excluded.append(self._excluded(block, "duplicate_block"))
                continue
            seen_body.add(body_key)
            trusted.append(
                CleanTextBlock(
                    paper_id=block.paper_id,
                    text=text,
                    raw_content_hash=block.raw_content_hash,
                    transformations=transformations,
                    page=block.page,
                    block_index=block.block_index,
                    bbox=block.bbox,
                    section=block.section,
                )
            )

        text_pages = {block.page for block in blocks if block.raw_text.strip()}
        text_page_ratio = len(text_pages) / page_count if page_count else 0.0
        if image_only_pages or text_page_ratio < self._min_text_page_ratio:
            reasons.append("ocr_required")
        if _has_double_column(blocks):
            warnings.append("double_column_order_uncertain")

        metrics = DocumentQualityMetrics(
            page_count=page_count,
            text_page_count=len(text_pages),
            text_page_ratio=text_page_ratio,
            metadata_completeness=metadata_completeness,
            duplicate_block_ratio=duplicate_count / len(blocks) if blocks else 0.0,
            gibberish_ratio=(
                sum(gibberish_values) / len(gibberish_values)
                if gibberish_values
                else 0.0
            ),
            section_count=len({block.section for block in trusted}),
            locator_coverage=(
                sum(bool(block.bbox) for block in trusted) / len(trusted)
                if trusted
                else 0.0
            ),
        )
        unique_reasons = list(dict.fromkeys(reasons))
        decision = (
            QualityDecision.DEGRADED
            if unique_reasons or warnings
            else QualityDecision.ACCEPTED
        )
        return CleanedDocument(
            source_blocks=blocks,
            blocks=trusted,
            excluded_blocks=excluded,
            report=DocumentQualityReport(
                decision=decision,
                reason_codes=unique_reasons,
                warnings=warnings,
                metrics=metrics,
            ),
        )

    @staticmethod
    def _excluded(
        block: RawTextBlock,
        reason: str,
        risk: str = "none",
    ) -> ExcludedTextBlock:
        """Build an excluded-block quality result."""
        return ExcludedTextBlock(
            raw_text=block.raw_text,
            raw_content_hash=block.raw_content_hash,
            page=block.page,
            block_index=block.block_index,
            bbox=block.bbox,
            reason_code=reason,
            injection_risk=risk,
        )

    def _quarantined(
        self,
        blocks: list[RawTextBlock],
        offending: RawTextBlock,
        page_count: int,
        reason: str,
    ) -> CleanedDocument:
        """Build a quarantined-document result."""
        return CleanedDocument(
            source_blocks=blocks,
            blocks=[],
            excluded_blocks=[self._excluded(offending, reason, "high")],
            report=DocumentQualityReport(
                decision=QualityDecision.QUARANTINED,
                reason_codes=[reason],
                metrics=DocumentQualityMetrics(page_count=page_count),
            ),
        )
