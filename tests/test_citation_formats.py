"""Exercise real CSL rendering, interchange exports, and immutable report views."""

import copy
import io
import json
import shutil
from unittest.mock import AsyncMock

import fitz
import pytest
from docx import Document

from litagent import citation_formats as formats
from litagent.report_export import LocalReportExporter
from litagent.report_render import build_reader_report, reader_report_projection
from litagent.workbench import ReportVersion


@pytest.fixture
def bibliography():
    """Include repeats, unparsed names, publication metadata, and an unknown ID."""
    version = ReportVersion(
        run_id="format-test",
        revision=1,
        report_language="zh-CN",
        survey="# 综述\n\n## 方法\n\n方法甲 [E:a] 与方法乙 [E:b]。重复 [E:a2]。未知 [E:missing]。",
    )
    ledger = {
        "a": {"paper_id": "arxiv:2010.11929", "paper_title": "A Study"},
        "a2": {"paper_id": "arxiv:2010.11929"},
        "b": {"paper_id": "paper-b", "paper_title": "中文研究"},
    }
    catalog = {
        "arxiv:2010.11929": {
            "title": "A Study",
            "authors": ["Alice Chen", "Bob Li"],
            "year": 2020,
        },
        "paper-b": {
            "title": "中文研究",
            "author_details": [{"family": "张", "given": "三"}],
            "authors": ["张三"],
            "year": 2021,
            "journal": "计算机研究",
            "volume": "12",
            "issue": "3",
            "pages": "10-20",
            "doi": "10.1234/test",
        },
    }
    return version, ledger, catalog


@pytest.mark.parametrize("style", formats.ACADEMIC_STYLES)
def test_styles_use_real_processor_and_keep_all_occurrence_links(bibliography, style):
    version, ledger, catalog = bibliography
    before = copy.deepcopy((version.model_dump(), ledger, catalog))
    reader = build_reader_report(
        version, evidence_ledger=ledger, paper_catalog=catalog, citation_style=style
    )
    assert len(reader.bibliography) == 2
    assert [x[1] for x in reader.citation_occurrences] == ["a", "b", "a2"]
    assert reader.source_sha256 == version.content_sha256
    assert "[E:missing]" in reader.markdown
    assert "unresolved_evidence:missing" in reader.warnings
    assert any(x.startswith("unparsed_author_names:") for x in reader.warnings)
    assert "中文研究" in reader.markdown
    assert "计算机研究" in reader.markdown
    html = reader_report_projection(reader)["html"]
    assert 'href="#citation-1"' in html
    assert "<script>" not in html
    assert before == (version.model_dump(), ledger, catalog)


def test_same_author_same_year_is_disambiguated_and_sorted():
    items = [
        {
            "id": x,
            "type": "article-journal",
            "title": title,
            "author": [{"family": "Chen", "given": "Alice"}],
            "issued": {"date-parts": [[2020]]},
        }
        for x, title in [("b", "Zebra"), ("a", "Alpha")]
    ]
    result = formats.format_citations(items, [["b"], ["a"], ["b"]], "apa7")
    assert result["citations"] == ["(Chen, 2020b)", "(Chen, 2020a)", "(Chen, 2020b)"]
    assert result["item_ids"] == ["a", "b"]


@pytest.mark.parametrize("style", formats.ACADEMIC_STYLES)
def test_missing_fields_and_institutional_names_are_not_invented(style):
    items = [
        {"id": "a", "type": "article", "title": "Missing fields"},
        {
            "id": "b",
            "type": "report",
            "title": "组织报告",
            "author": [{"literal": "中国科学院"}],
        },
    ]
    result = formats.format_citations(items, [["a", "b"]], style)
    assert len(result["bibliography"]) == 2
    assert "中国科学院" in "".join(result["bibliography"])
    assert "2020" not in "".join(result["bibliography"])


def test_unavailable_processor_and_bad_style_fail_explicitly(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="citation_engine_unavailable"):
        formats.format_citations([], [], "ieee")
    with pytest.raises(ValueError, match="unsupported_citation_style"):
        formats.format_citations([], [], "https://evil.example/style.csl")
    assert formats.format_capabilities()["academic_engine_available"] is False


def test_style_changes_only_projection_sha(bibliography):
    version, ledger, catalog = bibliography
    readers = [
        build_reader_report(
            version, evidence_ledger=ledger, paper_catalog=catalog, citation_style=s
        )
        for s in ["numbered", "ieee", "apa7"]
    ]
    assert len({x.source_sha256 for x in readers}) == 1
    assert len({x.projection_sha256 for x in readers}) == 3


def test_csl_snapshot_retains_metadata_and_excludes_uncited(bibliography):
    version, ledger, catalog = bibliography
    catalog["uncited"] = {"title": "Never cited", "year": 2010}
    snapshot = formats.build_bibliographic_snapshot(version, ledger, catalog)
    parsed = json.loads(formats.export_bibliography(snapshot, "csl_json"))
    assert len(parsed) == 2
    assert parsed[0]["author"][0] == {"literal": "Alice Chen"}
    assert parsed[1]["author"][0] == {"family": "张", "given": "三"}
    assert parsed[1]["DOI"] == "10.1234/test"
    assert parsed[1]["page"] == "10-20"
    assert snapshot["report_sha256"] == version.content_sha256


def test_interchange_bibtex_and_ris_parse_and_preserve_fields(bibliography):
    import bibtexparser
    import rispy

    version, ledger, catalog = bibliography
    snapshot = formats.build_bibliographic_snapshot(version, ledger, catalog)
    bib = bibtexparser.loads(formats.export_bibliography(snapshot, "bibtex").decode())
    ris = rispy.loads(formats.export_bibliography(snapshot, "ris").decode())
    assert len(bib.entries) == len(ris) == 2
    assert bib.entries[1]["title"] == "中文研究"
    assert bib.entries[1]["doi"] == ris[1]["doi"] == "10.1234/test"
    assert ris[1]["start_page"] == "10" and ris[1]["end_page"] == "20"
    assert bib.entries[0]["eprint"] == "2010.11929"


@pytest.mark.asyncio
@pytest.mark.parametrize("style", formats.ACADEMIC_STYLES)
async def test_report_exports_share_bibliography_and_source_sha(bibliography, style):
    version, ledger, catalog = bibliography
    reader = build_reader_report(
        version, evidence_ledger=ledger, paper_catalog=catalog, citation_style=style
    )
    exported = [
        await LocalReportExporter().export(
            version,
            format=f,
            publishable=False,
            citation_style=style,
            reader_report=reader,
        )
        for f in ["md", "pdf", "docx"]
    ]
    assert all(x.report_sha256 == version.content_sha256 for x in exported)
    pdf = fitz.open(stream=exported[1].content, filetype="pdf")
    pdf_text = "".join(page.get_text() for page in pdf)
    doc = Document(io.BytesIO(exported[2].content))
    doc_text = "\n".join(p.text for p in doc.paragraphs)
    for text in [exported[0].content.decode(), pdf_text, doc_text]:
        assert "A Study" in text and "中文研究" in text
        assert "DRAFT" in text
    assert "[E:a]" not in exported[0].content.decode()


@pytest.mark.asyncio
async def test_archived_api_formats_are_read_only_and_version_bound(
    tmp_path, monkeypatch, bibliography
):
    from fastapi.testclient import TestClient

    from litagent import api
    from litagent.observability.recorder import ArchiveRepository

    version, ledger, catalog = bibliography
    artifact = {
        "run_id": "archive-format",
        "report": {
            "survey": version.survey,
            "metadata": {"report_language": "zh-CN"},
            "quality": {"status": "passed"},
            "delivery": {"publishable": False},
        },
        "nodes": {
            "worker:dedup": {
                "output": [{"paper_id": k, **v} for k, v in catalog.items()]
            },
            "worker:extract": {
                "output": [
                    {
                        "evidence_items": [
                            {"evidence_id": k, **v} for k, v in ledger.items()
                        ]
                    }
                ]
            },
        },
    }
    repo = ArchiveRepository(tmp_path)
    path = repo.save(artifact)
    before = path.read_bytes()
    service = AsyncMock()
    service.get_version.side_effect = KeyError("archived")
    service.export.side_effect = ValueError("report_revision_not_found")
    monkeypatch.setattr(api, "_interactive_service", AsyncMock(return_value=service))
    monkeypatch.setattr(api, "_flow_artifact", lambda run: repo.get(run))
    monkeypatch.setattr(api, "_flow_source", lambda run: "archive")
    client = TestClient(api.app)
    response = client.get("/flow-demo/runs/archive-format/view?citation_style=apa7")
    assert response.status_code == 200
    assert response.json()["rendered_report"]["citation_style"] == "apa7"
    for f in ["bibtex", "ris", "csl_json"]:
        response = client.get(
            f"/survey/archive-format/bibliography?revision=1&format={f}"
        )
        assert response.status_code == 200
        assert response.headers["X-Report-SHA256"] == version.content_sha256
    response = client.get(
        "/survey/archive-format/export?revision=1&format=md&citation_style=ieee"
    )
    assert response.status_code == 200
    assert "A Study" in response.text
    assert (
        client.get(
            "/survey/archive-format/bibliography?revision=2&format=ris"
        ).status_code
        == 422
    )
    assert (
        client.get(
            "/flow-demo/runs/archive-format/view?citation_style=unknown"
        ).status_code
        == 422
    )
    assert path.read_bytes() == before


@pytest.mark.parametrize("style", formats.ACADEMIC_STYLES)
def test_combined_citations_deduplicate_papers_and_keep_each_evidence(
    bibliography, style
):
    """A citation cluster keeps every evidence link and one item per paper."""
    version, ledger, catalog = bibliography
    version = version.model_copy(
        update={
            "survey": "# Cluster\n\nFinding [E:a] [E:b][E:a2]. Missing [E:missing] [E:a]."
        }
    )
    reader = build_reader_report(
        version, evidence_ledger=ledger, paper_catalog=catalog, citation_style=style
    )
    assert len(reader.citation_links) == 2
    assert reader.citation_links[0]["evidence_ids"] == ["a", "b", "a2"]
    assert reader.citation_links[1]["evidence_ids"] == ["a"]
    assert "[E:missing]" in reader.markdown
    assert len(reader.bibliography) == 2


def test_archive_sidecars_cannot_shadow_canonical_report(tmp_path):
    """A run acceptance JSON must not replace the report in the archive index."""
    from litagent.observability.recorder import ArchiveRepository

    repo = ArchiveRepository(tmp_path)
    original = {"run_id": "survey", "report": {"survey": "Actual report"}}
    repo.save(original)
    (tmp_path / "survey-acceptance.json").write_text(
        json.dumps({"run_id": "survey", "status": "passed"})
    )
    assert repo.list() == [original]
    assert repo.get("survey") == original


def test_missing_processor_assets_fail_explicitly(monkeypatch, tmp_path):
    """Unavailable vendored assets never silently replace the requested style."""
    formats._verify_assets.cache_clear()
    monkeypatch.setattr(formats, "ASSET_ROOT", tmp_path)
    with pytest.raises(RuntimeError, match="citation_assets_unavailable"):
        formats.format_citations([], [], "apa7")
    formats._verify_assets.cache_clear()
