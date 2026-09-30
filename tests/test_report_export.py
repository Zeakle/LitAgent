"""Verify deterministic report exports without model or network calls."""

import io
import zipfile

import fitz
import pytest

from litagent.report_export import LocalReportExporter
from litagent.workbench import ReportVersion


@pytest.fixture
def report():
    return ReportVersion(
        run_id="run-1",
        revision=2,
        report_language="en",
        survey="# Survey\n\n## Methods\n\nSupported claim [E:ev-1].",
        human_status="approved",
    )


@pytest.mark.asyncio
async def test_markdown_preserves_exact_report_and_marks_draft(report):
    exported = await LocalReportExporter().export(
        report, format="md", publishable=False
    )
    text = exported.content.decode("utf-8")
    assert "DRAFT - NOT APPROVED FOR PUBLICATION" in text
    assert report.survey in text
    assert exported.report_sha256 == report.content_sha256


@pytest.mark.asyncio
async def test_pdf_contains_report_and_revision_metadata(report):
    exported = await LocalReportExporter().export(
        report, format="pdf", publishable=True
    )
    document = fitz.open(stream=exported.content, filetype="pdf")
    text = "".join(page.get_text() for page in document)
    assert "Survey" in text
    assert "run-1" in text
    assert "Revision 2" in text


@pytest.mark.asyncio
async def test_draft_pdf_labels_every_page(report):
    long_report = report.model_copy(
        update={
            "report_language": "zh-CN",
            "survey": (
                "# 文献综述\n\n## 方法\n\n| 方法 | 结果 |\n"
                "| --- | --- |\n| 基线 | 通过 |\n\n" + ("研究证据与限制。\n\n" * 180)
            ),
        }
    )
    exported = await LocalReportExporter().export(
        long_report, format="pdf", publishable=False
    )
    document = fitz.open(stream=exported.content, filetype="pdf")
    assert len(document) > 1
    assert all(
        "DRAFT - NOT APPROVED FOR PUBLICATION" in page.get_text() for page in document
    )
    first_page = document[0].get_text()
    assert "文献综述" in first_page
    assert "基线" in first_page


@pytest.mark.asyncio
async def test_docx_contains_visible_report_text(report):
    exported = await LocalReportExporter().export(
        report, format="docx", publishable=True
    )
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        document_xml = archive.read("word/document.xml").decode("utf-8")
    assert "Survey" in document_xml
    assert "Supported claim [E:ev-1]." in document_xml
    assert "run-1" in document_xml


@pytest.mark.asyncio
async def test_unknown_export_format_fails(report):
    with pytest.raises(ValueError, match="unsupported_export_format"):
        await LocalReportExporter().export(report, format="html", publishable=True)


@pytest.mark.asyncio
async def test_numbered_export_preserves_source_and_semantic_structure(report):
    from litagent.report_render import build_reader_report

    version = report.model_copy(
        update={
            "survey": (
                "# Survey\n\n## Methods\n\n- First **item** [E:ev-1]\n"
                "- Second [E:ev-2]\n\n| Method | Result |\n| --- | --- |\n"
                "| [Example](https://example.org/paper) | Yes |"
            )
        }
    )
    reader = build_reader_report(
        version,
        evidence_ledger={
            "ev-1": {"paper_id": "arxiv:2401.12345", "paper_title": "Paper"},
            "ev-2": {"paper_id": "arxiv:2401.12345", "paper_title": "Paper"},
        },
        paper_catalog={"arxiv:2401.12345": {"title": "Paper"}},
    )
    exporter = LocalReportExporter()
    pdf = await exporter.export(
        version,
        format="pdf",
        publishable=False,
        citation_style="numbered",
        reader_report=reader,
    )
    pdf_doc = fitz.open(stream=pdf.content, filetype="pdf")
    pdf_text = "".join(page.get_text() for page in pdf_doc)
    assert "[1]" in pdf_text and "References" in pdf_text
    assert "**item**" not in pdf_text and "| Method |" not in pdf_text
    assert pdf.report_sha256 == version.content_sha256
    word = await exporter.export(
        version,
        format="docx",
        publishable=False,
        citation_style="numbered",
        reader_report=reader,
    )
    with zipfile.ZipFile(io.BytesIO(word.content)) as archive:
        xml = archive.read("word/document.xml").decode("utf-8")
    assert "<w:tbl>" in xml
    assert "<w:numPr>" in xml or "ListBullet" in xml
    assert "References" in xml and "[1]" in xml
    assert "**item**" not in xml
