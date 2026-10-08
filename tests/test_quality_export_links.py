"""Verify exported links remain usable without changing frozen report sources."""

import io
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.etree import ElementTree

import fitz
import pytest

from litagent.report_export import LocalReportExporter
from litagent.report_render import build_reader_report
from litagent.workbench import ReportVersion

PAPER_URL = "https://arxiv.org/abs/2010.11929"
NS = {
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships",
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
}


def version(markdown):
    """Freeze a minimal report without any model or retrieval dependency."""
    return ReportVersion(
        run_id="export-links",
        revision=1,
        report_language="en",
        survey=markdown,
        human_status="pending",
    )


def docx_external_links(content):
    """Inspect actual OOXML hyperlink relationships, not rendered text alone."""
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        relationships = ElementTree.fromstring(
            archive.read("word/_rels/document.xml.rels")
        )
        body = ElementTree.fromstring(archive.read("word/document.xml"))
        header_text = "".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.startswith("word/header") and name.endswith(".xml")
        )
    links = [
        element.attrib["Target"]
        for element in relationships.findall("rel:Relationship", NS)
        if element.attrib.get("Type", "").endswith("/hyperlink")
        and element.attrib.get("TargetMode") == "External"
    ]
    return links, body, header_text


@pytest.mark.asyncio
@pytest.mark.parametrize("publishable", [True, False])
async def test_pdf_keeps_explicit_external_markdown_hyperlink(publishable):
    report = version(f"# Survey\n\nRead [paper]({PAPER_URL}).\n")
    exported = await LocalReportExporter().export(
        report, format="pdf", publishable=publishable
    )
    with fitz.open(stream=exported.content, filetype="pdf") as document:
        uri_links = [
            link["uri"]
            for page in document
            for link in page.get_links()
            if link.get("kind") == fitz.LINK_URI
        ]
        assert PAPER_URL in uri_links
        assert "paper" in document[0].get_text()
    assert exported.report_sha256 == report.content_sha256


@pytest.mark.asyncio
async def test_multpage_draft_pdf_preserves_links_and_warning_on_each_page():
    report = version(
        "# Survey\n\n"
        + "Evidence and limitations remain subject to review.\n\n" * 150
        + f"Read [paper]({PAPER_URL}).\n"
    )
    exported = await LocalReportExporter().export(
        report, format="pdf", publishable=False
    )
    with fitz.open(stream=exported.content, filetype="pdf") as document:
        assert len(document) > 1
        assert all(
            "DRAFT - NOT APPROVED FOR PUBLICATION" in page.get_text()
            for page in document
        )
        assert PAPER_URL in [
            link.get("uri") for page in document for link in page.get_links()
        ]


@pytest.mark.asyncio
async def test_docx_explicit_external_link_is_already_a_native_relationship():
    report = version(f"# Survey\n\nRead [paper]({PAPER_URL}).\n")
    exported = await LocalReportExporter().export(
        report, format="docx", publishable=False
    )
    links, body, headers = docx_external_links(exported.content)
    assert PAPER_URL in links
    assert body.findall(".//w:hyperlink", NS)
    assert "DRAFT - NOT APPROVED FOR PUBLICATION" in headers


@pytest.mark.asyncio
@pytest.mark.parametrize("style", ["numbered", "ieee"])
@pytest.mark.parametrize("format", ["pdf", "docx"])
async def test_actual_reader_bibliography_links_survive_file_export(style, format):
    report = version("# Survey\n\nViT uses image patches [E:paper-a:claim:0].\n")
    reader = build_reader_report(
        report,
        evidence_ledger={
            "paper-a:claim:0": {
                "paper_id": "arxiv:2010.11929",
                "paper_title": "An Image is Worth 16x16 Words",
            }
        },
        paper_catalog={
            "arxiv:2010.11929": {
                "title": "An Image is Worth 16x16 Words",
                "authors": ["Alexey Dosovitskiy"],
                "year": 2020,
                "url": PAPER_URL,
            }
        },
        citation_style=style,
    )
    original_markdown = reader.markdown
    original_source = report.survey
    assert PAPER_URL in reader.markdown
    if style == "ieee":
        assert "#citation-1" in reader.markdown
    exported = await LocalReportExporter().export(
        report,
        format=format,
        publishable=False,
        citation_style=style,
        reader_report=reader,
    )
    if format == "pdf":
        with fitz.open(stream=exported.content, filetype="pdf") as document:
            assert PAPER_URL in [
                link.get("uri") for page in document for link in page.get_links()
            ]
            assert "References" in "".join(page.get_text() for page in document)
            assert all(
                "DRAFT - NOT APPROVED FOR PUBLICATION" in page.get_text()
                for page in document
            )
    else:
        links, body, headers = docx_external_links(exported.content)
        assert PAPER_URL in links
        assert "References" in "".join(body.itertext())
        assert "DRAFT - NOT APPROVED FOR PUBLICATION" in headers
    assert reader.markdown == original_markdown
    assert report.survey == original_source
    assert exported.report_sha256 == report.content_sha256


@pytest.mark.asyncio
async def test_link_enrichment_does_not_rewrite_markdown_export_or_frozen_source():
    report = version("# Survey\n\nViT uses image patches [E:paper-a:claim:0].\n")
    reader = build_reader_report(
        report,
        evidence_ledger={
            "paper-a:claim:0": {
                "paper_id": "arxiv:2010.11929",
                "paper_title": "An Image is Worth 16x16 Words",
            }
        },
        paper_catalog={"arxiv:2010.11929": {"title": "An Image is Worth 16x16 Words"}},
        citation_style="numbered",
    )
    assert PAPER_URL in reader.markdown
    assert f"]({PAPER_URL})" not in reader.markdown
    exported = await LocalReportExporter().export(
        report,
        format="md",
        publishable=False,
        citation_style="numbered",
        reader_report=reader,
    )
    assert reader.markdown in exported.content.decode("utf-8")
    assert f"]({PAPER_URL})" not in exported.content.decode("utf-8")
    assert "[E:paper-a:claim:0]" in report.survey
    assert exported.report_sha256 == report.content_sha256


@pytest.fixture
def image_server():
    """Observe resource requests on a controlled loopback endpoint only."""
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(204)
            self.end_headers()

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/untrusted-image.png", requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.asyncio
async def test_pdf_hyperlink_repair_does_not_fetch_embedded_images(image_server):
    url, requests = image_server
    report = version(
        f"# Survey\n\nRead [paper]({PAPER_URL}).\n\n![Untrusted image]({url})\n"
    )
    exported = await LocalReportExporter().export(
        report, format="pdf", publishable=False
    )
    assert requests == []
    with fitz.open(stream=exported.content, filetype="pdf") as document:
        assert all(page.get_images() == [] for page in document)
        assert "[image omitted]" in "".join(page.get_text() for page in document)
