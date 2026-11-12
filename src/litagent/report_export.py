"""Render immutable report versions to local Markdown, PDF, and Word files."""

from __future__ import annotations

import asyncio
import html
import io
import math
import re
import sys
from html.parser import HTMLParser
from typing import Literal

import fitz
from docx import Document
from docx.enum.text import WD_BREAK
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt
from markdown_it import MarkdownIt

from litagent.citation_formats import STYLES
from litagent.report_render import (
    ReaderReport,
    _without_managed_markers,
    render_report_html,
)
from litagent.workbench import ExportResult, ReportVersion

_DRAFT_LABEL = "DRAFT - NOT APPROVED FOR PUBLICATION"


class LocalReportExporter:
    """Render server-owned report text without model or network access."""

    async def export(
        self,
        report: ReportVersion,
        *,
        format: Literal["md", "pdf", "docx"],
        publishable: bool,
        citation_style: str = "evidence",
        reader_report: ReaderReport | None = None,
    ) -> ExportResult:
        """Render one exact revision and expose its source-content hash."""
        if format not in {"md", "pdf", "docx"}:
            raise ValueError("unsupported_export_format")
        if citation_style not in STYLES:
            raise ValueError("unsupported_citation_style")
        if citation_style != "evidence" and (
            reader_report is None
            or reader_report.source_sha256 != report.content_sha256
        ):
            raise ValueError("reader_projection_unavailable")
        markdown = (
            reader_report.markdown
            if citation_style != "evidence" and reader_report is not None
            else report.survey
        )
        if (
            format != "md"
            and reader_report is not None
            and citation_style != "evidence"
        ):
            markdown = _export_reader_markdown(reader_report)
        content = await asyncio.to_thread(
            self._render, report, markdown, format, publishable
        )
        media_type = {
            "md": "text/markdown; charset=utf-8",
            "pdf": "application/pdf",
            "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }[format]
        return ExportResult(
            filename=f"litagent-{report.run_id}-r{report.revision}.{format}",
            media_type=media_type,
            content=content,
            report_sha256=report.content_sha256,
        )

    def _render(
        self, report: ReportVersion, markdown: str, format: str, publishable: bool
    ) -> bytes:
        if format == "md":
            return self._markdown(report, markdown, publishable).encode("utf-8")
        markdown = _without_managed_markers(markdown)
        if format == "pdf":
            return self._pdf(report, markdown, publishable)
        return self._docx(report, markdown, publishable)

    @staticmethod
    def _metadata_lines(report: ReportVersion, publishable: bool) -> list[str]:
        lines = [] if publishable else [_DRAFT_LABEL]
        lines.extend(
            [
                f"Run: {report.run_id}",
                f"Revision {report.revision}",
                f"Language: {report.report_language}",
                f"Human review: {report.human_status}",
                f"Content SHA-256: {report.content_sha256}",
            ]
        )
        return lines

    def _markdown(self, report: ReportVersion, markdown: str, publishable: bool) -> str:
        metadata = "\n".join(
            f"> {line}" for line in self._metadata_lines(report, publishable)
        )
        return f"{metadata}\n\n{markdown}\n"

    def _pdf(self, report: ReportVersion, markdown: str, publishable: bool) -> bytes:
        # Story receives escaped HTML without images, so it never fetches
        # untrusted assets embedded by a report or follows links at render time.
        rendered = render_report_html(markdown).html
        rendered = re.sub(r"<img\b[^>]*>", "[image omitted]", rendered)
        metadata = "<br>".join(
            html.escape(line) for line in self._metadata_lines(report, publishable)
        )
        css = (
            "body{font-family:sans-serif;font-size:10pt;line-height:1.4}"
            "h1{font-size:19pt}h2{font-size:15pt}h3{font-size:12pt}"
            "li{line-height:1.25}li>p{margin:2pt 0}"
            "table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #888;padding:4pt}"
            "pre{white-space:pre-wrap}a{color:#075c83}"
        )
        page_rect = fitz.paper_rect("a4")
        document = _pdf_story_with_heading_leads(
            f"<p class='metadata'>{metadata}</p>{rendered}", css, page_rect
        )
        try:
            if not publishable:
                # Keep the warning visible after page one without dropping links.
                for page in document:
                    page.insert_text(
                        (48, page.rect.height - 30),
                        _DRAFT_LABEL,
                        fontsize=8,
                        color=(0.56, 0.25, 0.15),
                    )
            return document.tobytes()
        finally:
            document.close()

    def _docx(self, report: ReportVersion, markdown: str, publishable: bool) -> bytes:
        document = Document()
        # Explicit East Asian fonts prevent theme fallbacks from dropping CJK glyphs.
        east_asian_font = (
            "Arial Unicode MS" if sys.platform == "darwin" else "Microsoft YaHei"
        )
        for style in document.styles:
            fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
            fonts.set(qn("w:eastAsia"), east_asian_font)
            fonts.attrib.pop(qn("w:eastAsiaTheme"), None)
        document.sections[0].header.paragraphs[0].text = (
            _DRAFT_LABEL if not publishable else ""
        )
        for line in self._metadata_lines(report, publishable):
            document.add_paragraph(line, style="Subtitle")
        tokens = (
            MarkdownIt("commonmark", {"html": False}).enable("table").parse(markdown)
        )
        list_stack: list[str] = []
        paragraph = None
        table = None
        row = None
        cell = None
        column = 0
        for index, token in enumerate(tokens):
            kind = token.type
            if kind in {"bullet_list_open", "ordered_list_open"}:
                list_stack.append(kind)
            elif kind in {"bullet_list_close", "ordered_list_close"}:
                list_stack.pop()
            elif kind == "table_open":
                columns = 0
                for entry in tokens[index + 1 :]:
                    if entry.type == "tr_close":
                        break
                    columns += entry.type in {"th_open", "td_open"}
                table = document.add_table(rows=0, cols=max(columns, 1))
                table.style = "Table Grid"
            elif kind == "table_close":
                table = None
            elif kind == "tr_open" and table is not None:
                row = table.add_row()
                column = 0
            elif kind in {"th_open", "td_open"} and row is not None:
                cell = row.cells[column]
                column += 1
                paragraph = cell.paragraphs[0]
            elif kind in {"heading_open", "paragraph_open"}:
                if cell is not None:
                    continue
                if kind == "heading_open":
                    paragraph = document.add_heading(level=min(int(token.tag[1:]), 9))
                    paragraph.paragraph_format.keep_with_next = True
                    paragraph.paragraph_format.keep_together = True
                else:
                    style = None
                    if list_stack:
                        style = (
                            "List Bullet"
                            if list_stack[-1] == "bullet_list_open"
                            else "List Number"
                        )
                    paragraph = document.add_paragraph(style=style)
                    paragraph.paragraph_format.widow_control = True
                    if list_stack:
                        # Compact repeated records without deleting appendix content.
                        paragraph.paragraph_format.space_after = Pt(2)
                        paragraph.paragraph_format.line_spacing = 1.0
            elif kind == "inline" and paragraph is not None:
                _append_inline(paragraph, token.children or [])
            elif kind == "fence":
                document.add_paragraph(token.content, style="No Spacing")
            elif kind in {"th_close", "td_close"}:
                cell = None
                paragraph = None
            elif kind in {"heading_close", "paragraph_close"}:
                if cell is None:
                    paragraph = None
        buffer = io.BytesIO()
        document.save(buffer)
        return buffer.getvalue()


class _PDFHeadingBlocks(HTMLParser):
    """Locate balanced root blocks without rewriting report HTML or its text."""

    _VOID = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
    _HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}

    def __init__(self, rendered: str):
        super().__init__(convert_charrefs=True)
        self._rendered = rendered
        self._line_starts = [0]
        for match in re.finditer("\n", rendered):
            self._line_starts.append(match.end())
        self._stack: list[str] = []
        self._blocks: list[dict] = []
        self._nodes: list[dict] = []
        self._ids: set[str] = set()
        self._heading_events: dict[str, int] = {}
        self._event = 0
        self._last_body_event = -1
        self.feed(rendered)
        self.close()
        if self._stack:
            raise ValueError("pdf_html_structure_invalid")
        self.boundaries: dict[str, int] = {}
        self.leading_boundaries: dict[int, int] = {}
        chain_start = 0
        previous_heading = False
        for block in self._blocks:
            if not previous_heading:
                chain_start = block["start"]
            self.leading_boundaries[block["start"]] = chain_start
            for anchor in block["ids"]:
                self.boundaries[anchor] = (
                    chain_start if block["heading"] else block["start"]
                )
            previous_heading = block["heading"]
        self.has_body_after = {
            anchor: event < self._last_body_event
            for anchor, event in self._heading_events.items()
        }
        self.layout_nodes = {node["id"]: node for node in self._nodes if node["id"]}

    def with_layout_ids(self) -> str:
        """Give layout blocks unique IDs without changing text or link targets."""
        prefix = "litagent-pdf-layout"
        while any(anchor.startswith(prefix) for anchor in self._ids):
            prefix += "-next"
        rendered = self._rendered
        for index, node in reversed(list(enumerate(self._nodes))):
            if node["id"] and not node["duplicate"]:
                continue
            anchor = f"{prefix}-{index}"
            if node["duplicate"]:
                # Keep the first destination for an existing href. Slug
                # collisions in later headings need only a new layout ID.
                attrs = "".join(
                    (
                        f" {name}"
                        if value is None
                        else f' {name}="{html.escape(anchor if name == "id" else value, quote=True)}"'
                    )
                    for name, value in node["attrs"]
                )
                tag = f'<{node["tag"]}{attrs}>'
                start = node["start"]
                rendered = (
                    rendered[:start] + tag + rendered[start + node["tag_length"] :]
                )
                continue
            at = node["start"] + len(node["tag"]) + 1
            rendered = rendered[:at] + f' id="{anchor}"' + rendered[at:]
        return rendered

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Associate nested headings with their complete containing root block."""
        line, column = self.getpos()
        start = self._line_starts[line - 1] + column
        anchor = dict(attrs).get("id")
        duplicate = bool(anchor and anchor in self._ids)
        if anchor:
            if duplicate and tag not in self._HEADINGS:
                raise ValueError("pdf_duplicate_element_id")
            self._ids.add(anchor)
        if not self._stack:
            self._blocks.append(
                {"start": start, "heading": tag in self._HEADINGS, "ids": []}
            )
        if tag not in self._VOID and (
            not self._stack or tag in self._HEADINGS | {"p", "li", "pre", "tr"}
        ):
            self._nodes.append(
                {
                    "id": anchor,
                    "start": start,
                    "root": self._blocks[-1]["start"],
                    "tag": tag,
                    "depth": len(self._stack),
                    "duplicate": duplicate,
                    "attrs": attrs,
                    "tag_length": len(self.get_starttag_text()),
                }
            )
        if tag in self._HEADINGS:
            if anchor and not duplicate:
                self._blocks[-1]["ids"].append(anchor)
                self._heading_events[anchor] = self._event
                self._event += 1
        if tag not in self._VOID:
            self._stack.append(tag)

    def handle_endtag(self, tag: str) -> None:
        """Reject unbalanced fragments rather than splitting through a container."""
        if tag in self._VOID:
            return
        if not self._stack or self._stack.pop() != tag:
            raise ValueError("pdf_html_structure_invalid")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Accept explicit self-closing tags in host-rendered HTML."""
        self.handle_starttag(tag, attrs)
        if tag not in self._VOID:
            self.handle_endtag(tag)

    def handle_data(self, data: str) -> None:
        """Distinguish terminal headings from headings with later body content."""
        if data.strip() and not any(tag in self._HEADINGS for tag in self._stack):
            self._last_body_event = self._event
            self._event += 1


def _pdf_fill_text_bounds(page: fitz.Page) -> list[fitz.Rect] | None:
    """Prove fill-text ink with the original native font and glyph IDs."""
    native = getattr(fitz, "mupdf", None)
    if native is None or not hasattr(native, "FzDevice2"):
        return None

    class GlyphBoundsDevice(native.FzDevice2):
        """Collect conservative per-glyph bounds without honoring page clips."""

        def __init__(self):
            super().__init__()
            self.use_virtual_fill_text()
            self.bounds = []
            self.reliable = True
            self.calls = 0

        def fill_text(self, _ctx, text, ctm, _colorspace, _color, _alpha, _params):
            self.calls += 1
            try:
                span = text.head
                while span:
                    wrapped = native.FzTextSpan(span)
                    font = wrapped.font()
                    descriptor = font.m_internal
                    flags = descriptor.flags
                    if (
                        descriptor.use_glyph_bbox != 1
                        or not native.fz_font_ft_face(font)
                        or native.fz_font_t3_procs(font)
                        or any(
                            (
                                flags.ft_substitute,
                                flags.ft_stretch,
                                flags.fake_bold,
                                flags.fake_italic,
                                flags.invalid_bbox,
                            )
                        )
                    ):
                        self.reliable = False
                        return
                    original = wrapped.trm()
                    transform = native.FzMatrix(
                        ctm.a, ctm.b, ctm.c, ctm.d, ctm.e, ctm.f
                    )
                    for index in range(wrapped.m_internal.len):
                        item = wrapped.items(index)
                        if item.gid < 0:
                            continue
                        if item.gid >= descriptor.glyph_count:
                            self.reliable = False
                            return
                        outline = native.fz_outline_glyph(
                            font, item.gid, native.FzMatrix()
                        )
                        if not outline.m_internal:
                            self.reliable = False
                            return
                        if native.fz_path_is_empty(outline):
                            # Transforming an empty outline can create a tiny
                            # floating-point rectangle at a space's origin.
                            continue
                        matrix = native.fz_concat(
                            native.FzMatrix(
                                original.a,
                                original.b,
                                original.c,
                                original.d,
                                item.x,
                                item.y,
                            ),
                            transform,
                        )
                        if (
                            not all(
                                math.isfinite(v)
                                for v in (
                                    matrix.a,
                                    matrix.b,
                                    matrix.c,
                                    matrix.d,
                                    matrix.e,
                                    matrix.f,
                                )
                            )
                            or matrix.b != 0
                            or matrix.c != 0
                        ):
                            self.reliable = False
                            return
                        box = native.fz_bound_glyph(font, item.gid, matrix)
                        rect = fitz.Rect(box.x0, box.y0, box.x1, box.y1)
                        if (
                            rect.is_empty
                            or rect.is_infinite
                            or not all(math.isfinite(v) for v in rect)
                        ):
                            self.reliable = False
                            return
                        self.bounds.append(rect)
                    span = span.next
            except Exception:
                # Missing native capabilities must never excuse a clipped run.
                self.reliable = False

    device = None
    try:
        device = GlyphBoundsDevice()
        native.fz_run_page(page.this, device, native.FzMatrix(), native.FzCookie())
        return device.bounds if device.reliable and device.calls else None
    except Exception:
        return None
    finally:
        if device is not None:
            native.fz_close_device(device)


def _pdf_body_overflows(
    document: fitz.Document, body_rect: fitz.Rect
) -> list[tuple[int, fitz.Rect]]:
    """Find painted ink outside the body, before separate footers are inserted."""
    # MuPDF's paint bounds include one point of conservative ink padding.
    # Font ascender boxes alone can extend above a correctly painted first line.
    allowed = body_rect + (-1.01, -1.01, 1.01, 1.01)
    overflows = []
    for page in document:
        glyph_bounds = None
        glyphs_checked = False
        for kind, bounds in page.get_bboxlog():
            rect = fitz.Rect(bounds)
            if (
                not kind.startswith(("fill-", "stroke-"))
                or rect.is_empty
                or rect in allowed
            ):
                continue
            if kind == "fill-text":
                if not glyphs_checked:
                    glyph_bounds = _pdf_fill_text_bounds(page)
                    glyphs_checked = True
                # Paint-operation bounds include leading/trailing whitespace.
                # Only original-font ink proves it safe; never use clipped
                # pixels or normalized character boxes to excuse overflow.
                if glyph_bounds is not None and all(
                    box in body_rect + (-0.01, -0.01, 0.01, 0.01)
                    for box in glyph_bounds
                ):
                    continue
            overflows.append((page.number, rect))
    return overflows


def _pdf_story_with_heading_leads(
    rendered: str, css: str, page_rect: fitz.Rect
) -> fitz.Document:
    """Reflow heading chains without Story's lossy forced-break layout path."""
    blocks = _PDFHeadingBlocks(rendered)
    rendered = blocks.with_layout_ids()
    blocks = _PDFHeadingBlocks(rendered)
    body_rect = fitz.Rect(48, 48, page_rect.width - 48, page_rect.height - 48)
    if body_rect.is_empty:
        raise ValueError("pdf_body_frame_invalid")
    forced_breaks: set[int] = set()
    while True:
        positions = []
        page_count = 0
        buffer = io.BytesIO()
        writer = None

        def next_page(_page, _filled):
            nonlocal page_count
            page_count += 1
            # A natural page consumes content; do not let a malformed layout
            # produce unbounded empty pages inside Story.write's own loop.
            if page_count > len(rendered) + 1:
                raise ValueError("pdf_pagination_no_progress")
            return page_rect, body_rect, None

        try:
            try:
                writer = fitz.DocumentWriter(buffer)
                boundaries = [0, *sorted(forced_breaks), len(rendered)]
                for start, end in zip(boundaries, boundaries[1:]):
                    fragment = rendered[start:end]
                    if not fragment.strip() and page_count:
                        continue
                    offset = page_count

                    def position_fn(position):
                        position.page_num += offset
                        positions.append(position)

                    # Splitting only at balanced root blocks preserves inline
                    # markup and lets long paragraphs paginate naturally.
                    story = fitz.Story(html=fragment, user_css=css)
                    story.write(writer, next_page, positionfn=position_fn)
            finally:
                if writer is not None:
                    writer.close()
            data = buffer.getvalue()
        finally:
            buffer.close()
        document = fitz.open(stream=data, filetype="pdf")
        try:
            overflows = _pdf_body_overflows(document, body_rect)
            overflowing_blocks = set()
            for page_index in {page for page, _bounds in overflows}:
                for trace in document[page_index].get_texttrace():
                    for character in trace["chars"]:
                        if not chr(character[0]).strip() or fitz.Rect(character[3]) in (
                            body_rect + (-1.01, -1.01, 1.01, 1.01)
                        ):
                            continue
                        owners = [
                            blocks.layout_nodes[p.id]
                            for p in positions
                            if p.page_num == page_index + 1
                            and p.id in blocks.layout_nodes
                            and blocks.layout_nodes[p.id]["tag"] not in blocks._HEADINGS
                            and fitz.Point(character[2])
                            in fitz.Rect(p.rect) + (-1, -1, 1, 1)
                        ]
                        if owners:
                            owner = max(owners, key=lambda node: node["depth"])
                            overflowing_blocks.add(
                                blocks.leading_boundaries[owner["root"]]
                            )
            headings = [p for p in positions if p.heading and p.open_close & 1]
            orphans = set()
            for position in headings:
                if not position.id or not blocks.has_body_after.get(position.id):
                    continue
                page = document[position.page_num - 1]
                on_page = [p for p in headings if p.page_num == position.page_num]
                has_body = any(
                    chr(character[0]).strip()
                    and character[2][1] > position.rect[3] + 1
                    and not any(
                        fitz.Point(character[2]) in fitz.Rect(p.rect) for p in on_page
                    )
                    for trace in page.get_texttrace()
                    for character in trace["chars"]
                )
                if not has_body:
                    orphans.add(blocks.boundaries[position.id])
            if not orphans and not overflows:
                # Resolve links once, after all segments have global page
                # numbers. Per-segment resolution loses cross-segment anchors.
                return fitz.Story.add_pdf_links(document, positions)
            additional = (orphans | overflowing_blocks) - forced_breaks - {0}
            if not additional:
                reason = (
                    "pdf_body_reflow_no_progress"
                    if overflows
                    else "pdf_heading_reflow_no_progress"
                )
                raise ValueError(reason)
            forced_breaks.update(additional)
        except BaseException:
            document.close()
            raise
        document.close()


def _export_reader_markdown(reader: ReaderReport) -> str:
    """Link source URLs and remove UI-only evidence targets in binary exports."""
    markdown = reader.markdown
    if reader.bibliography_span:
        start, end = reader.bibliography_span

        def autolink(match: re.Match) -> str:
            value = match.group()
            url = value.rstrip(".,;。；，)")
            return f"<{url}>" + value[len(url) :]

        bibliography = re.sub(r"https?://[^\s<>]+", autolink, markdown[start:end])
        markdown = markdown[:start] + bibliography + markdown[end:]
    # These targets represent browser evidence buttons, not document bookmarks.
    for link in reader.citation_links:
        markdown = markdown.replace(
            f"[{link['label']}](#{link['anchor']})", link["label"]
        )
    return markdown


def _append_inline(paragraph, children: list) -> None:
    """Map Markdown inline tokens to native Word runs and safe hyperlinks."""
    bold = italic = False
    link: str | None = None
    for child in children:
        kind = child.type
        if kind == "strong_open":
            bold = True
        elif kind == "strong_close":
            bold = False
        elif kind == "em_open":
            italic = True
        elif kind == "em_close":
            italic = False
        elif kind == "link_open":
            link = child.attrGet("href")
        elif kind == "link_close":
            link = None
        elif kind in {"text", "code_inline"}:
            if link and re.match(r"^(https?://|mailto:)", link):
                _add_hyperlink(paragraph, child.content, link)
            else:
                run = paragraph.add_run(child.content)
                run.bold = bold
                run.italic = italic
                if kind == "code_inline":
                    run.font.name = "Consolas"
        elif kind in {"softbreak", "hardbreak"}:
            paragraph.add_run().add_break(WD_BREAK.LINE)
        elif kind == "image":
            paragraph.add_run("[image omitted]")


def _add_hyperlink(paragraph, label: str, url: str) -> None:
    """Add a clickable link without dereferencing its untrusted target."""
    relation = paragraph.part.relate_to(
        url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), relation)
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.text = label
    run.append(text)
    hyperlink.append(run)
    paragraph._p.append(hyperlink)
