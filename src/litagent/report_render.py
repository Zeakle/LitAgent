"""Render untrusted survey Markdown into a bounded browser-safe projection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

from litagent.evidence import extract_evidence_refs_ordered
from litagent.workbench import ReportVersion


@dataclass(frozen=True)
class ReportHeading:
    """Describe one rendered report heading for the client-side outline."""

    level: int
    text: str
    anchor: str


@dataclass(frozen=True)
class RenderedReport:
    """Return safe HTML and its stable outline from one Markdown document."""

    html: str
    headings: tuple[ReportHeading, ...]


@dataclass(frozen=True)
class ReportSourceUnit:
    """Identify one exact Markdown source fragment for claim diagnostics."""

    unit_id: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class ClaimAlignment:
    """Keep source location distinct from citation ownership."""

    status: str
    citation_scope: str
    source_unit_id: str | None
    source_quote: str | None
    start: int | None
    end: int | None
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReaderReport:
    """Keep numbered presentation separate from the reviewed source Markdown."""

    markdown: str
    bibliography: tuple[str, ...]
    citation_occurrences: tuple[tuple[int, str], ...]
    warnings: tuple[str, ...]
    source_sha256: str
    projection_sha256: str


def build_reader_report(
    report: ReportVersion,
    *,
    evidence_ledger: Mapping[str, Mapping[str, object]],
    paper_catalog: Mapping[str, Mapping[str, object]],
) -> ReaderReport:
    """Number known papers by first citation while retaining evidence identity."""
    numbers: dict[str, int] = {}
    titles: dict[str, str] = {}
    occurrences: list[tuple[int, str]] = []
    warnings: list[str] = []
    pieces: list[str] = []
    cursor = 0
    for match in re.finditer(r"\[E:([^\[\]\s;,]+)\]", report.survey):
        pieces.append(report.survey[cursor : match.start()])
        evidence_id = match.group(1)
        item = evidence_ledger.get(evidence_id)
        paper_id = str(item.get("paper_id") or "") if item else ""
        if not paper_id:
            pieces.append(match.group())
            warnings.append(f"unresolved_evidence:{evidence_id}")
        else:
            if paper_id not in numbers:
                numbers[paper_id] = len(numbers) + 1
                titles[paper_id] = str(item.get("paper_title") or "")
            number = numbers[paper_id]
            pieces.append(f"[{number}]")
            occurrences.append((number, evidence_id))
        cursor = match.end()
    pieces.append(report.survey[cursor:])
    bibliography: list[str] = []
    for paper_id, number in numbers.items():
        metadata = paper_catalog.get(paper_id) or {}
        raw_authors = metadata.get("authors")
        authors = (
            ", ".join(
                author for author in raw_authors if isinstance(author, str) and author
            )
            if isinstance(raw_authors, list)
            else ""
        )
        authors = authors or "Authors unavailable"
        title = str(metadata.get("title") or titles[paper_id] or "Title unavailable")
        year = metadata.get("year")
        year_text = str(year) if isinstance(year, int) else "Year unavailable"
        url = (
            f"https://arxiv.org/abs/{paper_id[6:]}"
            if re.fullmatch(r"arxiv:\d{4}\.\d{4,5}", paper_id)
            else ""
        )
        bibliography.append(
            f"[{number}] {authors}. {title} ({year_text}). {url}".strip()
        )
    markdown = "".join(pieces)
    if bibliography:
        markdown = (
            markdown.rstrip()
            + "\n\n## References\n\n"
            + "\n\n".join(bibliography)
            + "\n"
        )
    payload = {
        "source_sha256": report.content_sha256,
        "version": 1,
        "bibliography": bibliography,
        "citation_occurrences": occurrences,
        "markdown": markdown,
    }
    projection_sha = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return ReaderReport(
        markdown=markdown,
        bibliography=tuple(bibliography),
        citation_occurrences=tuple(occurrences),
        warnings=tuple(dict.fromkeys(warnings)),
        source_sha256=report.content_sha256,
        projection_sha256=projection_sha,
    )


def paper_catalog_from_nodes(
    nodes: Mapping[str, object]
) -> dict[str, dict[str, object]]:
    """Recover source paper metadata from a run's deterministic worker output."""
    catalog: dict[str, dict[str, object]] = {}
    for task_id in ("worker:dedup", "worker:relevance_gate"):
        node = nodes.get(task_id)
        if not isinstance(node, Mapping):
            continue
        output = node.get("output")
        if not isinstance(output, list):
            continue
        for paper in output:
            if not isinstance(paper, Mapping):
                continue
            paper_id = paper.get("paper_id")
            if isinstance(paper_id, str) and paper_id:
                catalog.setdefault(paper_id, {}).update(
                    {
                        key: paper[key]
                        for key in ("title", "authors", "year")
                        if paper.get(key)
                    }
                )
    return catalog


def reader_report_projection(reader: ReaderReport) -> dict[str, object]:
    """Render numbered report text and retain occurrence-level evidence links."""
    projection = render_report_projection(reader.markdown)
    projection.update(
        {
            "citation_occurrences": [
                {"number": number, "evidence_id": evidence_id}
                for number, evidence_id in reader.citation_occurrences
            ],
            "warnings": list(reader.warnings),
            "source_sha256": reader.source_sha256,
            "projection_sha256": reader.projection_sha256,
        }
    )
    return projection


def build_report_source_units(markdown: str) -> list[ReportSourceUnit]:
    """Locate exact paragraph and table-cell slices in the Markdown source."""
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(markdown)
    lines = markdown.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    spans: list[tuple[int, int]] = []
    table_tokens: dict[int, list[object]] = {}
    for index, token in enumerate(tokens):
        if token.type != "inline" or token.map is None or index == 0:
            continue
        parent_type = tokens[index - 1].type
        if parent_type not in {"paragraph_open", "td_open", "th_open"}:
            continue
        start_line, end_line = token.map
        if end_line > len(lines):
            raise ValueError("source_unit_mapping_failed")
        if parent_type in {"td_open", "th_open"}:
            if end_line != start_line + 1:
                raise ValueError("source_unit_mapping_failed")
            table_tokens.setdefault(start_line, []).append(token)
            continue
        start, end = offsets[start_line], offsets[end_line]
        spans.append((start, end))

    for line_number, inline_tokens in table_tokens.items():
        cell_spans = _table_cell_spans(lines[line_number], offsets[line_number])
        if len(cell_spans) != len(inline_tokens):
            raise ValueError("source_unit_mapping_failed")
        spans.extend(cell_spans)

    units: list[ReportSourceUnit] = []
    for ordinal, (start, end) in enumerate(sorted(set(spans))):
        text = markdown[start:end]
        if not text:
            continue
        units.append(
            ReportSourceUnit(
                unit_id=f"unit:{ordinal}:{start}",
                text=text,
                start=start,
                end=end,
            )
        )
    return units


def _table_cell_spans(line: str, line_start: int) -> list[tuple[int, int]]:
    """Return trimmed source spans while respecting escapes and code fences."""
    content_end = len(line.rstrip("\r\n"))
    boundaries: list[int] = []
    code_fence = 0
    index = 0
    while index < content_end:
        char = line[index]
        if char == "`":
            run_end = index + 1
            while run_end < content_end and line[run_end] == "`":
                run_end += 1
            run_length = run_end - index
            if code_fence == 0:
                code_fence = run_length
            elif code_fence == run_length:
                code_fence = 0
            index = run_end
            continue
        if char == "|" and code_fence == 0:
            slash_count = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                slash_count += 1
                cursor -= 1
            if slash_count % 2 == 0:
                boundaries.append(index)
        index += 1

    if not boundaries:
        raise ValueError("source_unit_mapping_failed")
    starts = [0, *(position + 1 for position in boundaries)]
    ends = [*boundaries, content_end]
    if boundaries[0] == 0:
        starts = starts[1:]
        ends = ends[1:]
    if boundaries[-1] == content_end - 1:
        starts = starts[:-1]
        ends = ends[:-1]

    spans: list[tuple[int, int]] = []
    for start, end in zip(starts, ends, strict=True):
        while start < end and line[start] in " \t":
            start += 1
        while end > start and line[end - 1] in " \t":
            end -= 1
        spans.append((line_start + start, line_start + end))
    return spans


def align_claim_to_source(
    claim_text: str, units: list[ReportSourceUnit]
) -> ClaimAlignment:
    """Map one verbatim claim only when it has a unique source occurrence."""
    claim = claim_text.strip()
    if not claim:
        return ClaimAlignment("unmapped", "unknown", None, None, None, None)
    matches = [
        (unit, position)
        for unit in units
        for position in [unit.text.find(claim)]
        if position >= 0 and unit.text.count(claim) == 1
    ]
    if len(matches) != 1:
        return ClaimAlignment("unmapped", "unknown", None, None, None, None)
    unit, position = matches[0]
    after = unit.text[position + len(claim) :]
    marker_match = re.match(r"\s*((?:\[E:[^\[\]]+\]\s*)+)", after)
    refs = (
        tuple(extract_evidence_refs_ordered(marker_match.group(1)))
        if marker_match
        else ()
    )
    scope = "exact" if refs else "ambiguous"
    start = unit.start + position
    end = start + len(claim) + (len(marker_match.group()) if marker_match else 0)
    return ClaimAlignment(
        "exact",
        scope,
        unit.unit_id,
        unit.text[position : position + len(claim)]
        + (marker_match.group() if marker_match else ""),
        start,
        end,
        refs,
    )


def _slug(value: str, fallback: str) -> str:
    """Build a readable Unicode anchor without treating text as markup."""
    normalized = re.sub(r"[^\w\-\u4e00-\u9fff]+", "-", value.lower()).strip("-")
    return normalized or fallback


def _safe_href(value: str) -> bool:
    """Allow local anchors and ordinary web links while rejecting active schemes."""
    if value.startswith(("#", "/", "./", "../")):
        return True
    scheme = urlsplit(value).scheme.lower()
    return scheme in {"http", "https", "mailto"}


def render_report_html(markdown: str) -> RenderedReport:
    """Render Markdown with raw HTML disabled and unsafe links removed."""
    parser = MarkdownIt(
        "commonmark",
        {"html": False, "linkify": False, "typographer": False},
    ).enable("table")
    tokens = parser.parse(markdown or "")
    headings: list[ReportHeading] = []
    anchor_counts: dict[str, int] = {}

    for index, token in enumerate(tokens):
        if token.type == "heading_open" and index + 1 < len(tokens):
            inline = tokens[index + 1]
            text = inline.content if inline.type == "inline" else ""
            base = _slug(text, f"section-{len(headings) + 1}")
            count = anchor_counts.get(base, 0) + 1
            anchor_counts[base] = count
            anchor = base if count == 1 else f"{base}-{count}"
            token.attrSet("id", anchor)
            headings.append(
                ReportHeading(level=int(token.tag[1:]), text=text, anchor=anchor)
            )
        if token.type == "inline" and token.children:
            for child in token.children:
                if child.type != "link_open":
                    continue
                href = child.attrGet("href")
                if href is not None and not _safe_href(href):
                    child.attrSet("href", "#unsafe-link")
                    child.attrSet("data-unsafe-link", "true")
                elif href and urlsplit(href).scheme in {"http", "https"}:
                    child.attrSet("rel", "noopener noreferrer")

    html = parser.renderer.render(tokens, parser.options, {})
    return RenderedReport(html=html, headings=tuple(headings))


def render_report_projection(markdown: str) -> dict[str, object]:
    """Serialize one rendered report into the stable browser API contract."""
    rendered = render_report_html(markdown)
    return {
        "html": rendered.html,
        "headings": [
            {
                "level": heading.level,
                "text": heading.text,
                "anchor": heading.anchor,
            }
            for heading in rendered.headings
        ],
    }
