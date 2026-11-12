"""Render untrusted survey Markdown into a bounded browser-safe projection."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from markdown_it import MarkdownIt

from litagent.citation_formats import (
    REF_PATTERN,
    STYLES,
    build_bibliographic_snapshot,
    csl_markdown,
    format_citations,
)
from litagent.evidence import extract_evidence_refs_ordered
from litagent.review_structure import (
    METHOD_BEGIN,
    METHOD_END,
    appendix_span,
)
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
    citation_style: str = "numbered"
    citation_links: tuple[dict, ...] = ()
    bibliography_span: tuple[int, int] | None = None


def _insert_bibliography(
    markdown: str, bibliography: tuple | list, language: str
) -> tuple[str, tuple[int, int] | None]:
    """Insert host references before the appendix without altering source blocks."""
    if not bibliography:
        return markdown, None
    span = appendix_span(markdown)
    at = span["start"] if span else len(markdown)
    prefix = markdown[:at].rstrip() + "\n\n"
    references = (
        ("## 参考文献\n\n" if language == "zh-CN" else "## References\n\n")
        + "\n\n".join(bibliography)
        + "\n\n"
    )
    return prefix + references + markdown[at:], (
        len(prefix),
        len(prefix) + len(references),
    )


def build_reader_report(
    report: ReportVersion,
    *,
    evidence_ledger: Mapping[str, Mapping[str, object]],
    paper_catalog: Mapping[str, Mapping[str, object]],
    citation_style: str = "numbered",
) -> ReaderReport:
    """Number known papers by first citation while retaining evidence identity."""
    if citation_style not in STYLES:
        raise ValueError("unsupported_citation_style")
    if citation_style not in {"numbered", "evidence"}:
        return _build_styled_report(
            report, evidence_ledger, paper_catalog, citation_style
        )
    if citation_style == "evidence":
        return ReaderReport(
            report.survey,
            (),
            (),
            (),
            report.content_sha256,
            report.content_sha256,
            "evidence",
        )
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
        chinese = report.report_language == "zh-CN"
        authors = authors or ("作者信息不可用" if chinese else "Authors unavailable")
        title = str(
            metadata.get("title")
            or titles[paper_id]
            or ("标题不可用" if chinese else "Title unavailable")
        )
        year = metadata.get("year")
        year_text = (
            str(year)
            if isinstance(year, int)
            else ("年份信息不可用" if chinese else "Year unavailable")
        )
        url = (
            f"https://arxiv.org/abs/{paper_id[6:]}"
            if re.fullmatch(r"arxiv:\d{4}\.\d{4,5}", paper_id)
            else ""
        )
        bibliography.append(
            f"[{number}] {authors}. {title} ({year_text}). {url}".strip()
        )
    markdown = "".join(pieces)
    markdown, bibliography_span = _insert_bibliography(
        markdown, bibliography, report.report_language
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
        bibliography_span=bibliography_span,
    )


def _build_styled_report(report, ledger, catalog, style) -> ReaderReport:
    """Render scholarly styles while preserving occurrence-level evidence links."""
    snapshot = build_bibliographic_snapshot(report, ledger, catalog)
    numbers = {item["id"]: i + 1 for i, item in enumerate(snapshot["items"])}
    group_pattern = re.compile(r"\[E:[^\[\]\s;,]+\](?:[ \t]*\[E:[^\[\]\s;,]+\])*")
    groups = []
    for group in group_pattern.finditer(report.survey):
        refs = list(REF_PATTERN.finditer(group.group()))
        # Unknown IDs remain in place and cannot become fabricated CSL items.
        if all(
            (ledger.get(ref.group(1)) or {}).get("paper_id") in numbers for ref in refs
        ):
            groups.append((group.start(), group.end(), [ref.group(1) for ref in refs]))
        else:
            for ref in refs:
                if (ledger.get(ref.group(1)) or {}).get("paper_id") in numbers:
                    groups.append(
                        (
                            group.start() + ref.start(),
                            group.start() + ref.end(),
                            [ref.group(1)],
                        )
                    )
    clusters = [
        list(dict.fromkeys(ledger[eid]["paper_id"] for eid in ids))
        for _, _, ids in groups
    ]
    formatted = format_citations(snapshot["items"], clusters, style)
    links = []
    occurrences = []
    pieces = []
    cursor = 0
    for (start, end, ids), raw_label in zip(
        groups, formatted["citations"], strict=True
    ):
        label = csl_markdown(raw_label)
        anchor = f"citation-{len(links) + 1}"
        links.append({"anchor": anchor, "label": label, "evidence_ids": ids})
        occurrences.extend((numbers[ledger[eid]["paper_id"]], eid) for eid in ids)
        pieces.extend([report.survey[cursor:start], f"[{label}](#{anchor})"])
        cursor = end
    pieces.append(report.survey[cursor:])
    markdown = "".join(pieces)
    bibliography = tuple(csl_markdown(x) for x in formatted["bibliography"])
    markdown, bibliography_span = _insert_bibliography(
        markdown, bibliography, report.report_language
    )
    payload = {
        "source": report.content_sha256,
        "style": style,
        "snapshot": snapshot["snapshot_sha256"],
        "markdown": markdown,
    }
    sha = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return ReaderReport(
        markdown,
        bibliography,
        tuple(occurrences),
        tuple(snapshot["warnings"]),
        report.content_sha256,
        sha,
        style,
        tuple(links),
        bibliography_span,
    )


def paper_catalog_from_nodes(
    nodes: Mapping[str, object],
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
                        for key in (
                            "title",
                            "authors",
                            "year",
                            "author_details",
                            "doi",
                            "url",
                            "journal",
                            "venue",
                            "volume",
                            "issue",
                            "pages",
                        )
                        if paper.get(key)
                    }
                )
    return catalog


def reader_report_projection(reader: ReaderReport) -> dict[str, object]:
    """Render numbered report text and retain occurrence-level evidence links."""
    projection = render_report_projection(
        reader.markdown, bibliography_span=reader.bibliography_span
    )
    if projection.get("methodology"):
        projection["methodology"]["source_sha256"] = reader.source_sha256
    projection.update(
        {
            "citation_occurrences": [
                {"number": number, "evidence_id": evidence_id}
                for number, evidence_id in reader.citation_occurrences
            ],
            "citation_style": reader.citation_style,
            "citation_links": list(reader.citation_links),
            "warnings": list(reader.warnings),
            "source_sha256": reader.source_sha256,
            "projection_sha256": reader.projection_sha256,
        }
    )
    return projection


def build_report_source_units(markdown: str) -> list[ReportSourceUnit]:
    """Locate exact paragraph and table-cell slices in the Markdown source."""
    parser = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = parser.parse(_without_managed_markers(markdown))
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


def _without_managed_markers(markdown: str) -> str:
    """Hide host block delimiters while preserving source line and byte positions."""
    for marker in (METHOD_BEGIN, METHOD_END):
        markdown = markdown.replace(marker, " " * len(marker))
    return markdown


def render_report_html(
    markdown: str, *, bibliography_span: tuple[int, int] | None = None
) -> RenderedReport:
    """Render Markdown with raw HTML disabled and unsafe links removed."""
    parser = MarkdownIt(
        "commonmark",
        {"html": False, "linkify": False, "typographer": False},
    ).enable("table")
    tokens = parser.parse(_without_managed_markers(markdown or ""))
    headings: list[ReportHeading] = []
    anchor_counts: dict[str, int] = {}
    source_ordinal = 0
    offsets = [0]
    for line in markdown.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))

    for index, token in enumerate(tokens):
        if token.type == "inline" and token.map is not None and index > 0:
            parent = tokens[index - 1]
            if parent.type in {"paragraph_open", "td_open", "th_open"}:
                # Citation projections preserve block order, even as citation lengths change.
                offset = offsets[token.map[0]]
                if (
                    bibliography_span is None
                    or not bibliography_span[0] <= offset < bibliography_span[1]
                ):
                    target = parent
                    # Tight-list paragraphs are hidden by MarkdownIt; bind their LI instead.
                    if (
                        parent.hidden
                        and index >= 2
                        and tokens[index - 2].type == "list_item_open"
                    ):
                        target = tokens[index - 2]
                    target.attrSet("data-source-unit-ordinal", str(source_ordinal))
                    source_ordinal += 1
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


def render_report_projection(
    markdown: str, *, bibliography_span: tuple[int, int] | None = None
) -> dict[str, object]:
    """Expose a safe, source-bound appendix without hiding it from evaluation."""
    rendered = render_report_html(markdown, bibliography_span=bibliography_span)
    appendix = appendix_span(markdown)
    anchor = next(
        (
            h.anchor
            for h in rendered.headings
            if appendix and h.text == appendix["section"] and h.level == 2
        ),
        None,
    )
    methodology = None
    # Only frozen host blocks are displayed; never infer records from global run state.
    if markdown.count(METHOD_BEGIN) == 1 and markdown.count(METHOD_END) == 1:
        start = markdown.find(METHOD_BEGIN)
        end = markdown.find(METHOD_END, start) + len(METHOD_END)
        if end > start:
            methodology = {
                "html": render_report_html(markdown[start:end]).html,
                "source_sha256": hashlib.sha256(markdown.encode()).hexdigest(),
            }
    return {
        "html": rendered.html,
        "headings": [
            {"level": h.level, "text": h.text, "anchor": h.anchor}
            for h in rendered.headings
        ],
        "appendix_anchor": anchor,
        "methodology": methodology,
    }
