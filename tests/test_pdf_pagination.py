"""Verify painted PDF bounds, natural pagination, links, and resource ownership."""

from __future__ import annotations

import json
import re
from pathlib import Path

import fitz
import pytest

from litagent import report_export

PAGE = fitz.Rect(0, 0, 360, 360)
BODY = fitz.Rect(48, 48, 312, 312)
CSS = "body{font-size:10pt;line-height:1.4}"
FONT_LIGATURES = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
    }
)


def normalized(text: str) -> str:
    """Ignore layout whitespace and reversible font ligatures, not characters."""
    return "".join(text.translate(FONT_LIGATURES).split())


def document_with_ink(side: str, overflow: float) -> fitz.Document:
    """Place a real painted glyph bound at a measured body-frame boundary."""
    with fitz.open() as reference:
        page = reference.new_page(width=360, height=360)
        page.insert_text((100, 150), "InkProbe123", fontsize=12)
        ink = fitz.Rect(page.get_bboxlog()[0][1])
    dx = dy = 0.0
    if side == "left":
        dx = BODY.x0 - overflow - ink.x0
    elif side == "right":
        dx = BODY.x1 + overflow - ink.x1
    elif side == "top":
        dy = BODY.y0 - overflow - ink.y0
    elif side == "bottom":
        dy = BODY.y1 + overflow - ink.y1
    else:
        raise AssertionError("unknown test boundary")
    document = fitz.open()
    page = document.new_page(width=360, height=360)
    page.insert_text((100 + dx, 150 + dy), "InkProbe123", fontsize=12)
    return document


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize("overflow", [1.25, 8.0])
def test_actual_ink_outside_each_body_edge_is_detected(side, overflow):
    with document_with_ink(side, overflow) as document:
        # A text layer can remain extractable even when the visible body clips it.
        assert "InkProbe123" in document[0].get_text()
        failures = report_export._pdf_body_overflows(document, BODY)
        assert len(failures) == 1
        assert failures[0][0] == 0
        assert failures[0][1] == fitz.Rect(document[0].get_bboxlog()[0][1])


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
@pytest.mark.parametrize("padding", [0.0, 1.0])
def test_paint_bounds_allow_only_the_mupdf_ink_padding(side, padding):
    with document_with_ink(side, padding) as document:
        assert report_export._pdf_body_overflows(document, BODY) == []


def test_font_ascender_box_does_not_create_a_false_ink_overflow():
    with document_with_ink("top", 0.0) as document:
        span = document[0].get_text("dict")["blocks"][0]["lines"][0]["spans"][0]
        assert span["bbox"][1] < BODY.y0
        assert report_export._pdf_body_overflows(document, BODY) == []


def test_blank_space_outside_body_does_not_count_as_painted_content():
    with fitz.open() as document:
        page = document.new_page(width=360, height=360)
        page.insert_text((12, 20), "   ", fontsize=12)
        image = page.get_pixmap(colorspace=fitz.csGRAY, alpha=False)
        assert min(image.samples) == 255
        assert report_export._pdf_body_overflows(document, BODY) == []


@pytest.mark.parametrize(
    ("point", "text"),
    [((36, 150), "    Ink"), ((36, 150), "    Ink    "), ((294, 150), "Ink    ")],
)
def test_leading_and_trailing_spaces_do_not_expand_actual_ink_bounds(point, text):
    with fitz.open() as document:
        page = document.new_page(width=360, height=360)
        page.insert_text(point, text, fontsize=12)
        assert fitz.Rect(page.get_bboxlog()[0][1]) not in BODY
        # Render the whole unclipped page: the body frame is never a raster clip.
        image = page.get_pixmap(colorspace=fitz.csGRAY, alpha=False)
        colored = [
            (index % image.width, index // image.width)
            for index, sample in enumerate(image.samples)
            if sample < 255
        ]
        assert colored
        assert all(fitz.Point(x, y) in BODY for x, y in colored)
        assert report_export._pdf_body_overflows(document, BODY) == []


@pytest.mark.parametrize("point", [(32, 150), (298, 150)])
def test_real_ink_outside_frame_is_not_exempted_by_spaces_in_same_run(point):
    text = "    Ink" if point[0] == 32 else "Ink    "
    with fitz.open() as document:
        page = document.new_page(width=360, height=360)
        page.insert_text(point, text, fontsize=12)
        image = page.get_pixmap(colorspace=fitz.csGRAY, alpha=False)
        colored = [
            (index % image.width, index // image.width)
            for index, sample in enumerate(image.samples)
            if sample < 255
        ]
        assert any(fitz.Point(x, y) not in BODY for x, y in colored)
        assert report_export._pdf_body_overflows(document, BODY)


def test_missing_native_glyph_proof_does_not_excuse_suspicious_text(monkeypatch):
    with fitz.open() as document:
        page = document.new_page(width=360, height=360)
        page.insert_text((36, 150), "    Ink", fontsize=12)
        monkeypatch.setattr(report_export, "_pdf_fill_text_bounds", lambda _page: None)
        assert report_export._pdf_body_overflows(document, BODY)


def test_unproven_cjk_fallback_retains_original_paint_bound_rejection():
    with fitz.open() as document:
        page = document.new_page(width=360, height=360)
        page.insert_text((36, 150), "    墨", fontsize=12, fontname="china-s")
        assert report_export._pdf_fill_text_bounds(page) is None
        assert report_export._pdf_body_overflows(document, BODY)


def test_painted_path_and_its_page_number_are_not_silently_ignored():
    with fitz.open() as document:
        document.new_page(width=360, height=360)
        page = document.new_page(width=360, height=360)
        page.draw_rect(fitz.Rect(40, 100, 80, 120), fill=(0, 0, 0), width=0)
        failures = report_export._pdf_body_overflows(document, BODY)
        assert len(failures) == 1
        assert failures[0][0] == 1
        assert failures[0][1].x0 < BODY.x0


def test_one_long_paragraph_naturally_spans_pages_without_losing_citations():
    plain = " ".join(f"C{i:04d} [{i + 1}] alpha beta gamma." for i in range(180))
    with report_export._pdf_story_with_heading_leads(
        f"<p>{plain}</p>", CSS, PAGE
    ) as document:
        assert len(document) >= 3
        assert report_export._pdf_body_overflows(document, BODY) == []
        text = "".join(page.get_text() for page in document)
        assert normalized(text) == normalized(plain)
        assert re.findall(r"\[(\d+)\]", text) == [str(i) for i in range(1, 181)]


def test_neutral_real_story_regression_retains_text_that_forced_break_loses():
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/pdf-pagination-overflow.json").read_text()
    )
    assert fixture["geometry_only"] is True
    assert fixture["scientific_gold_standard"] is False
    page_rect, body_rect = fitz.Rect(fixture["page"]), fitz.Rect(fixture["frame"])

    def page_frame(*_):
        return page_rect, body_rect, None

    with (
        fitz.Story(html=fixture["html"], user_css=fixture["css"]).write_with_links(
            page_frame
        ) as natural,
        fitz.Story(
            html=fixture["html"],
            user_css=fixture["css"] + fixture["old_forced_break_css"],
        ).write_with_links(page_frame) as old_forced,
        report_export._pdf_story_with_heading_leads(
            fixture["html"], fixture["css"], page_rect
        ) as repaired,
    ):
        baseline = normalized("".join(page.get_text() for page in natural))
        clipped = normalized("".join(page.get_text() for page in old_forced))
        actual = normalized("".join(page.get_text() for page in repaired))
        assert report_export._pdf_body_overflows(natural, body_rect) == []
        assert report_export._pdf_body_overflows(old_forced, body_rect)
        assert clipped != baseline and len(clipped) < len(baseline)
        assert report_export._pdf_body_overflows(repaired, body_rect) == []
        assert actual == baseline
        assert len(repaired) > len(old_forced)


def test_tables_lists_and_code_paginate_without_losing_or_reordering_items():
    rows = "".join(
        f"<tr><td>CELL{i:03d}</td><td>[{i + 1}] value</td></tr>" for i in range(60)
    )
    items = "".join(f"<li>ITEM{i:03d}</li>" for i in range(30))
    code = "\n".join(f"CODE{i:03d}" for i in range(30))
    rendered = (
        f"<table>{rows}</table><blockquote><p>Quoted body.</p></blockquote>"
        f"<ul>{items}</ul><pre>{code}</pre>"
    )
    css = (
        CSS + "table{border-collapse:collapse;width:100%}"
        "td{border:1px solid #888;padding:2pt}pre{white-space:pre-wrap}"
    )
    with report_export._pdf_story_with_heading_leads(rendered, css, PAGE) as document:
        assert len(document) >= 3
        assert report_export._pdf_body_overflows(document, BODY) == []
        text = "".join(page.get_text() for page in document)
        tokens = re.findall(r"CELL\d{3}|\[\d+\]|ITEM\d{3}|CODE\d{3}", text)
        expected = [
            token for i in range(60) for token in (f"CELL{i:03d}", f"[{i + 1}]")
        ]
        expected.extend(f"ITEM{i:03d}" for i in range(30))
        expected.extend(f"CODE{i:03d}" for i in range(30))
        assert tokens == expected
        assert "Quoted body." in text


def test_reflow_preserves_external_links_and_global_cross_segment_anchors():
    rendered = (
        '<p id="start"><a href="#detail">Jump detail</a> '
        '<a href="#lastdetail">Jump final</a> '
        '<a href="https://example.org/one">External one</a><br>'
        + "Filler line.<br>" * 13
        + '</p><h2 id="section">Section</h2><h3 id="detail">Detail</h3>'
        '<p>Following first. <a href="#start">Back start</a> '
        '<a href="https://example.org/two">External two</a></p><p>'
        + "Second filler.<br>" * 14
        + '</p><h2 id="last">Last section</h2>'
        '<h3 id="lastdetail">Last detail</h3>'
        '<p>Following final. <a href="#detail">Back detail</a></p>'
    )
    with report_export._pdf_story_with_heading_leads(rendered, CSS, PAGE) as document:
        assert report_export._pdf_body_overflows(document, BODY) == []
        encoded = document.tobytes()
    # Reopen the actual PDF: in-memory positions alone do not prove link targets.
    with fitz.open(stream=encoded, filetype="pdf") as document:
        texts = [page.get_text().translate(FONT_LIGATURES) for page in document]
        assert len(document) >= 3
        detail_page = next(i for i, text in enumerate(texts) if "Detail" in text)
        final_page = next(i for i, text in enumerate(texts) if "Last detail" in text)
        assert 0 < detail_page < final_page
        assert (
            "Section" in texts[detail_page] and "Following first" in texts[detail_page]
        )
        assert "Last section" in texts[final_page]
        assert "Following final" in texts[final_page]
        first_links = document[0].get_links()
        assert {
            link["page"] for link in first_links if link["kind"] == fitz.LINK_GOTO
        } == {
            detail_page,
            final_page,
        }
        assert any(
            link["kind"] == fitz.LINK_GOTO and link["page"] == 0
            for link in document[detail_page].get_links()
        )
        assert any(
            link["kind"] == fitz.LINK_GOTO and link["page"] == detail_page
            for link in document[final_page].get_links()
        )
        uris = [
            link["uri"]
            for page in document
            for link in page.get_links()
            if link["kind"] == fitz.LINK_URI
        ]
        assert list(dict.fromkeys(uris)) == [
            "https://example.org/one",
            "https://example.org/two",
        ]
        assert normalized("".join(texts)).count("Fillerline.") == 13
        assert normalized("".join(texts)).count("Secondfiller.") == 14


def test_terminal_heading_is_preserved_without_requiring_invented_following_text():
    rendered = (
        "<p>" + "Filler line.<br>" * 14 + '</p><h2 id="last">Terminal heading</h2>'
    )
    with report_export._pdf_story_with_heading_leads(rendered, CSS, PAGE) as document:
        text = "".join(page.get_text() for page in document)
        assert text.count("Terminal heading") == 1
        assert text.count("Filler line.") == 14
        assert report_export._pdf_body_overflows(document, BODY) == []


def test_duplicate_rendered_heading_ids_keep_first_anchor_and_all_content():
    markdown = (
        "# A\n\nFirst body.\n\n"
        + "Filler line.  \n" * 14
        + "\n# A-2\n\nSecond body.\n\n"
        + "Middle line.  \n" * 14
        + "\n# A\n\nThird body.\n\n[First repeated target](#a-2)\n"
    )
    rendered = report_export.render_report_html(markdown).html
    assert rendered.count('id="a-2"') == 2
    metadata = "Run: duplicate-headings / Revision 7 / SHA: 0123456789abcdef"
    with report_export._pdf_story_with_heading_leads(
        f"<p>{metadata}</p>{rendered}", CSS, PAGE
    ) as document:
        text = "".join(page.get_text() for page in document)
        assert normalized(metadata) in normalized(text)
        for body in ("First body.", "Second body.", "Third body."):
            assert text.count(body) == 1
        assert text.count("Filler line.") == 14
        assert text.count("Middle line.") == 14
        target_page, target = next(
            (page.number, rectangle)
            for page in document
            for rectangle in page.search_for("A-2")
        )
        final_page = next(page for page in document if "Third body." in page.get_text())
        assert target_page < final_page.number
        links = [
            link for link in final_page.get_links() if link["kind"] == fitz.LINK_GOTO
        ]
        assert links and all(link["page"] == target_page for link in links)
        assert all(link["to"] in target for link in links)
        assert report_export._pdf_body_overflows(document, BODY) == []


def test_nested_unsplittable_heading_fails_with_no_progress_instead_of_looping():
    rendered = (
        "<blockquote><p>"
        + "Filler line.<br>" * 14
        + '</p><h2 id="nested">Nested heading</h2>'
        "<p>Following body.</p></blockquote>"
    )
    with pytest.raises(ValueError, match="^pdf_heading_reflow_no_progress$"):
        report_export._pdf_story_with_heading_leads(rendered, CSS, PAGE)


def test_invalid_body_frame_fails_before_story_layout():
    with pytest.raises(ValueError, match="^pdf_body_frame_invalid$"):
        report_export._pdf_story_with_heading_leads(
            "<p>Text.</p>", CSS, fitz.Rect(0, 0, 90, 90)
        )


def test_story_failure_closes_the_real_document_writer(monkeypatch):
    closed = []
    original_close = fitz.DocumentWriter.close

    def observe_close(writer):
        closed.append(writer)
        return original_close(writer)

    def failed_write(*args, **kwargs):
        raise RuntimeError("injected_story_failure")

    monkeypatch.setattr(fitz.DocumentWriter, "close", observe_close)
    monkeypatch.setattr(fitz.Story, "write", failed_write)
    with pytest.raises(RuntimeError, match="^injected_story_failure$"):
        report_export._pdf_story_with_heading_leads("<p>Text.</p>", CSS, PAGE)
    assert len(closed) == 1


def test_writer_constructor_failure_closes_its_buffer(monkeypatch):
    buffers = []
    original_buffer = report_export.io.BytesIO

    class ObservedBuffer(original_buffer):
        def __init__(self):
            super().__init__()
            buffers.append(self)

    def failed_writer(*args, **kwargs):
        raise RuntimeError("injected_writer_constructor_failure")

    monkeypatch.setattr(report_export.io, "BytesIO", ObservedBuffer)
    monkeypatch.setattr(fitz, "DocumentWriter", failed_writer)
    with pytest.raises(RuntimeError, match="^injected_writer_constructor_failure$"):
        report_export._pdf_story_with_heading_leads("<p>Text.</p>", CSS, PAGE)
    assert len(buffers) == 1 and buffers[0].closed


def test_geometric_failure_closes_the_intermediate_pdf(monkeypatch):
    opened = []
    original_open = fitz.open

    def observe_open(*args, **kwargs):
        document = original_open(*args, **kwargs)
        opened.append(document)
        return document

    monkeypatch.setattr(fitz, "open", observe_open)
    monkeypatch.setattr(
        report_export, "_pdf_body_overflows", lambda *_: [(0, fitz.Rect(0, 0, 10, 10))]
    )
    with pytest.raises(ValueError, match="^pdf_body_reflow_no_progress$"):
        report_export._pdf_story_with_heading_leads("<p>Text.</p>", CSS, PAGE)
    assert opened and all(document.is_closed for document in opened)


def test_link_resolution_failure_closes_the_intermediate_pdf(monkeypatch):
    opened = []
    original_open = fitz.open

    def observe_open(*args, **kwargs):
        document = original_open(*args, **kwargs)
        opened.append(document)
        return document

    def failed_links(*args, **kwargs):
        raise RuntimeError("injected_link_failure")

    monkeypatch.setattr(fitz, "open", observe_open)
    monkeypatch.setattr(fitz.Story, "add_pdf_links", failed_links)
    with pytest.raises(RuntimeError, match="^injected_link_failure$"):
        report_export._pdf_story_with_heading_leads("<p>Text.</p>", CSS, PAGE)
    assert opened and all(document.is_closed for document in opened)


def test_pathological_empty_pages_are_bounded_and_close_writer(monkeypatch):
    closed, page_requests = [], []
    original_close = fitz.DocumentWriter.close
    rendered = '<p id="fixed">x</p>'

    def observe_close(writer):
        closed.append(writer)
        return original_close(writer)

    def empty_pages(_story, _writer, next_page, **kwargs):
        for index in range(len(rendered) + 3):
            page_requests.append(index)
            next_page(index, None)

    monkeypatch.setattr(fitz.DocumentWriter, "close", observe_close)
    monkeypatch.setattr(fitz.Story, "write", empty_pages)
    with pytest.raises(ValueError, match="^pdf_pagination_no_progress$"):
        report_export._pdf_story_with_heading_leads(rendered, CSS, PAGE)
    assert len(page_requests) == len(rendered) + 2
    assert len(closed) == 1
