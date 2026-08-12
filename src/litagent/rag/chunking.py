"""Build deterministic corpus chunks from trusted PDF text blocks."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

from litagent.config import ChunkStrategy, RAGConfig
from litagent.rag.models import (
    ChunkSourceSpan,
    ContentChunk,
    ContentScope,
)
from litagent.rag.quality import CleanTextBlock

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


class CorpusChunker(Protocol):
    """Convert ordered trusted blocks into retrieval chunks."""

    def chunk(
        self,
        *,
        paper_id: str,
        blocks: Sequence[CleanTextBlock],
    ) -> list[ContentChunk]:
        """Return deterministic chunks with complete source lineage."""
        raise NotImplementedError


def _digest(text: str) -> str:
    """Return a stable content digest."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _section(value: str) -> str:
    """Normalize a source block section name."""
    normalized = re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
    return normalized or "unknown"


def _validate_blocks(
    paper_id: str,
    blocks: Sequence[CleanTextBlock],
) -> list[CleanTextBlock]:
    """Validate source-block ordering and paper ownership."""
    ordered = list(blocks)
    previous: tuple[int, int] | None = None
    for block in ordered:
        if block.paper_id != paper_id:
            raise ValueError("chunk input contains a different paper_id")
        locator = (block.page, block.block_index)
        if previous is not None and locator <= previous:
            raise ValueError("clean blocks must have unique increasing locators")
        previous = locator
    return ordered


def _window_offsets(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[int, int]]:
    """Split text into bounded, overlapping half-open intervals."""
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("invalid chunk size or overlap")
    if not text:
        return []

    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        hard_end = min(start + chunk_size, len(text))
        end = hard_end
        if hard_end < len(text):
            search_floor = start + max(1, chunk_size // 2)
            for separator in _SEPARATORS:
                position = text.rfind(separator, search_floor, hard_end)
                if position >= 0:
                    end = position + len(separator)
                    break

        visible_start = start
        visible_end = end
        while visible_start < visible_end and text[visible_start].isspace():
            visible_start += 1
        while visible_end > visible_start and text[visible_end - 1].isspace():
            visible_end -= 1
        if visible_start < visible_end:
            windows.append((visible_start, visible_end))
        if end >= len(text):
            break

        # Advance at least one character even when a separator is near start.
        start = max(end - chunk_overlap, start + 1)
    return windows


def _span(
    block: CleanTextBlock,
    start_char: int,
    end_char: int,
) -> ChunkSourceSpan:
    """Build a source span for one chunk slice."""
    return ChunkSourceSpan(
        page=block.page,
        block_index=block.block_index,
        bbox=block.bbox,
        start_char=start_char,
        end_char=end_char,
        raw_content_hash=block.raw_content_hash,
    )


def _build_chunk(
    *,
    paper_id: str,
    chunk_key: str,
    text: str,
    section: str,
    spans: Sequence[ChunkSourceSpan],
    transformations: Sequence[str],
) -> ContentChunk:
    """Build one content chunk with stable provenance metadata."""
    if not spans:
        raise ValueError("selected-fulltext chunks require source spans")
    first = spans[0]
    return ContentChunk.from_text(
        paper_id=paper_id,
        chunk_key=chunk_key,
        text=text,
        section=section,
        content_scope=ContentScope.SELECTED_FULLTEXT,
        transformations=transformations,
        page=first.page,
        block_index=first.block_index,
        bbox=first.bbox,
        source_spans=spans,
    )


class PageBlockChunker:
    """Preserve the Phase 14.1 one-block-per-chunk baseline."""

    def chunk(
        self,
        *,
        paper_id: str,
        blocks: Sequence[CleanTextBlock],
    ) -> list[ContentChunk]:
        """Split input text into content chunks."""
        chunks: list[ContentChunk] = []
        for block in _validate_blocks(paper_id, blocks):
            if not block.text.strip():
                continue
            chunks.append(
                _build_chunk(
                    paper_id=paper_id,
                    chunk_key=f"page:{block.page}:block:{block.block_index}",
                    text=block.text,
                    section=_section(block.section),
                    spans=[_span(block, 0, len(block.text))],
                    transformations=block.transformations,
                )
            )
        return chunks


class RecursiveChunker:
    """Split each clean block independently with deterministic overlap."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the recursive chunker."""
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(
        self,
        *,
        paper_id: str,
        blocks: Sequence[CleanTextBlock],
    ) -> list[ContentChunk]:
        """Split input text into content chunks."""
        chunks: list[ContentChunk] = []
        for block in _validate_blocks(paper_id, blocks):
            for ordinal, (start, end) in enumerate(
                _window_offsets(
                    block.text,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
            ):
                text = block.text[start:end]
                key = (
                    f"page:{block.page}:block:{block.block_index}:"
                    f"recursive:{ordinal}:{_digest(text)}"
                )
                chunks.append(
                    _build_chunk(
                        paper_id=paper_id,
                        chunk_key=key,
                        text=text,
                        section=_section(block.section),
                        spans=[_span(block, start, end)],
                        transformations=block.transformations,
                    )
                )
        return chunks


class SectionAwareChunker:
    """Window consecutive blocks without crossing section boundaries."""

    def __init__(self, *, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize the section-aware chunker."""
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @staticmethod
    def _groups(
        blocks: Sequence[CleanTextBlock],
    ) -> list[tuple[str, list[CleanTextBlock]]]:
        """Group adjacent blocks by normalized section."""
        groups: list[tuple[str, list[CleanTextBlock]]] = []
        for block in blocks:
            section = _section(block.section)
            if not groups or groups[-1][0] != section:
                groups.append((section, [block]))
            else:
                groups[-1][1].append(block)
        return groups

    def _chunk_group(
        self,
        *,
        paper_id: str,
        section: str,
        blocks: Sequence[CleanTextBlock],
    ) -> list[ContentChunk]:
        """Split one section group into bounded chunks."""
        parts: list[str] = []
        ranges: list[tuple[int, int, CleanTextBlock]] = []
        cursor = 0
        for block in blocks:
            if parts:
                parts.append("\n\n")
                cursor += 2
            start = cursor
            parts.append(block.text)
            cursor += len(block.text)
            ranges.append((start, cursor, block))
        combined = "".join(parts)

        chunks: list[ContentChunk] = []
        for ordinal, (start, end) in enumerate(
            _window_offsets(
                combined,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
        ):
            spans: list[ChunkSourceSpan] = []
            transformations: list[str] = []
            for block_start, block_end, block in ranges:
                intersection_start = max(start, block_start)
                intersection_end = min(end, block_end)
                if intersection_start >= intersection_end:
                    continue
                spans.append(
                    _span(
                        block,
                        intersection_start - block_start,
                        intersection_end - block_start,
                    )
                )
                transformations.extend(block.transformations)
            text = combined[start:end]
            key = f"section:{section}:{ordinal}:{_digest(text)}"
            chunks.append(
                _build_chunk(
                    paper_id=paper_id,
                    chunk_key=key,
                    text=text,
                    section=section,
                    spans=spans,
                    transformations=list(dict.fromkeys(transformations)),
                )
            )
        return chunks

    def chunk(
        self,
        *,
        paper_id: str,
        blocks: Sequence[CleanTextBlock],
    ) -> list[ContentChunk]:
        """Split input text into content chunks."""
        ordered = [
            block for block in _validate_blocks(paper_id, blocks) if block.text.strip()
        ]
        chunks: list[ContentChunk] = []
        for section, group in self._groups(ordered):
            chunks.extend(
                self._chunk_group(
                    paper_id=paper_id,
                    section=section,
                    blocks=group,
                )
            )
        return chunks


def build_corpus_chunker(config: RAGConfig) -> CorpusChunker:
    """Build exactly one configured Corpus chunk policy."""
    if config.chunk_strategy is ChunkStrategy.PAGE_BLOCK:
        return PageBlockChunker()
    if config.chunk_strategy is ChunkStrategy.RECURSIVE:
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    if config.chunk_strategy is ChunkStrategy.SECTION_AWARE:
        return SectionAwareChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
    raise ValueError(f"unsupported chunk strategy: {config.chunk_strategy}")
