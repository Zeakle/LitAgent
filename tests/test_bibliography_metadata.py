"""Verify source bibliography metadata survives the report and export chain."""

import io
from xml.sax.saxutils import escape

import fitz
import httpx
import pytest
from docx import Document

from litagent.agents.dedup import DedupWorker
from litagent.agents.relevance_gate import RelevanceGateWorker
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.models import PaperCandidate
from litagent.report_export import LocalReportExporter
from litagent.report_render import build_reader_report, paper_catalog_from_nodes
from litagent.tools.builtin import search
from litagent.workbench import ReportVersion


def arxiv_feed(published="2020-10-22T12:00:00Z", authors=("Alice Chen", "Bob Li")):
    """Build a source feed including an unrelated later update date."""
    names = "".join(f"<author><name>{escape(n)}</name></author>" for n in authors)
    date = f"<published>{escape(published)}</published>" if published else ""
    return (
        '<feed xmlns="http://www.w3.org/2005/Atom"><entry>'
        "<id>https://arxiv.org/abs/2010.11929v1</id><title>Vision Transformer</title>"
        "<summary>Vision transformer methods.</summary>"
        f"{names}{date}<updated>2025-01-01T00:00:00Z</updated></entry></feed>"
    )


def mock_provider(monkeypatch, payload):
    """Replace HTTP transport while retaining real response processing."""
    requests = []
    client = httpx.AsyncClient

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=payload)

    monkeypatch.setattr(
        search.httpx,
        "AsyncClient",
        lambda **kwargs: client(transport=httpx.MockTransport(handler), **kwargs),
    )
    return requests


@pytest.mark.parametrize(
    "published,expected",
    [
        ("2020-10-22T12:00:00Z", 2020),
        (None, None),
        ("not-a-date", None),
        ("2020-02-30T00:00:00Z", None),
    ],
)
def test_arxiv_authors_and_original_publication_year(published, expected):
    paper = search._parse_arxiv_xml(arxiv_feed(published))[0]
    assert paper["authors"] == ["Alice Chen", "Bob Li"]
    assert paper["year"] == expected


def test_arxiv_missing_and_blank_authors_are_not_invented():
    assert search._parse_arxiv_xml(arxiv_feed(authors=()))[0]["authors"] == []
    assert search._parse_arxiv_xml(arxiv_feed(authors=(" ", "Alice  Chen")))[0][
        "authors"
    ] == ["Alice Chen"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "year,expected",
    [(2020, 2020), (None, None), (True, None), ("bad", None), (0, None)],
)
async def test_semantic_scholar_requests_and_maps_authors(monkeypatch, year, expected):
    requests = mock_provider(
        monkeypatch,
        {
            "data": [
                {
                    "paperId": "s2-paper",
                    "title": "Paper",
                    "year": year,
                    "authors": [
                        {"authorId": "1", "name": "Alice Chen"},
                        {"name": "Bob Li"},
                        {"name": None},
                        {},
                    ],
                }
            ]
        },
    )
    paper = (await search.search_semantic_scholar("vision"))[0]
    assert "authors" in requests[0].url.params["fields"].split(",")
    assert paper["authors"] == ["Alice Chen", "Bob Li"]
    assert paper["year"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "date,expected",
    [
        ("2020-10-22T12:00:00.000Z", 2020),
        (None, None),
        ("bad", None),
        ("2020-02-30", None),
    ],
)
async def test_huggingface_maps_source_authors_and_publication_date(
    monkeypatch, date, expected
):
    mock_provider(
        monkeypatch,
        [
            {
                "publishedAt": "2025-01-01",
                "paper": {
                    "id": "2010.11929",
                    "title": "Vision Transformer",
                    "authors": [{"name": "Alice Chen"}, {"name": "Bob Li"}],
                    "publishedAt": date,
                },
            }
        ],
    )
    paper = (await search.search_huggingface("vision"))[0]
    assert paper["authors"] == ["Alice Chen", "Bob Li"]
    assert paper["year"] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,payload",
    [
        (
            search.search_semantic_scholar,
            {"data": [{"paperId": "s2-paper", "title": "Paper"}]},
        ),
        (
            search.search_huggingface,
            [{"paper": {"id": "2010.11929", "title": "Paper"}}],
        ),
    ],
)
async def test_provider_missing_metadata_is_explicit(monkeypatch, provider, payload):
    mock_provider(monkeypatch, payload)
    paper = (await provider("vision"))[0]
    assert paper["authors"] == [] and paper["year"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reverse", [False, True])
async def test_cross_source_merge_preserves_populated_authors(reverse):
    rich = PaperCandidate.from_external(
        search._parse_arxiv_xml(arxiv_feed())[0]
    ).to_dag_dict()
    sparse = PaperCandidate.from_external(
        {
            "paper_id": "2010.11929",
            "title": "Vision Transformer",
            "source": "huggingface",
        }
    ).to_dag_dict()
    inputs = [rich, sparse] if not reverse else [sparse, rich]
    merged = await DedupWorker().execute(
        SubTask("dedup", "merge", "dedup", {"upstream_results": {"search": inputs}})
    )
    assert len(merged) == 1
    assert merged[0]["authors"] == ["Alice Chen", "Bob Li"]
    assert merged[0]["year"] == 2020


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh-CN"])
async def test_source_to_numbered_bibliography_and_all_exports(language):
    raw = search._parse_arxiv_xml(arxiv_feed())[0]
    candidate = PaperCandidate.from_external(raw)
    merged = await DedupWorker().execute(
        SubTask(
            "dedup",
            "merge",
            "dedup",
            {"upstream_results": {"search": [candidate.to_dag_dict()]}},
        )
    )
    selected = await RelevanceGateWorker(None).execute(
        SubTask(
            "relevance_gate",
            "rank",
            "relevance_gate",
            {"query": "vision transformer", "upstream_results": {"dedup": merged}},
        )
    )
    catalog = paper_catalog_from_nodes(
        {
            "worker:dedup": {"output": merged},
            "worker:relevance_gate": {"output": selected},
        }
    )
    report = ReportVersion(
        run_id="bibliography",
        revision=1,
        report_language=language,
        survey="# Survey\n\nResult [E:ev-1].",
    )
    reader = build_reader_report(
        report,
        evidence_ledger={"ev-1": {"paper_id": candidate.paper_id}},
        paper_catalog=catalog,
    )
    expected = "Alice Chen, Bob Li. Vision Transformer (2020)."
    assert expected in reader.markdown
    assert "unavailable" not in reader.markdown
    source_sha = report.content_sha256
    for format in ("md", "pdf", "docx"):
        result = await LocalReportExporter().export(
            report,
            format=format,
            publishable=True,
            citation_style="numbered",
            reader_report=reader,
        )
        if format == "md":
            text = result.content.decode()
        elif format == "pdf":
            with fitz.open(stream=result.content, filetype="pdf") as pdf:
                text = " ".join(page.get_text() for page in pdf)
        else:
            text = " ".join(
                p.text for p in Document(io.BytesIO(result.content)).paragraphs
            )
        assert expected in " ".join(text.split())
        assert result.report_sha256 == source_sha
    assert report.content_sha256 == source_sha and "[E:ev-1]" in report.survey


@pytest.mark.parametrize(
    "language,authors,year,heading",
    [
        ("en", "Authors unavailable", "Year unavailable", "References"),
        ("zh-CN", "作者信息不可用", "年份信息不可用", "参考文献"),
    ],
)
def test_reader_missing_metadata_is_localized(language, authors, year, heading):
    report = ReportVersion(
        run_id="missing",
        revision=1,
        report_language=language,
        survey="Result [E:ev-1].",
    )
    reader = build_reader_report(
        report,
        evidence_ledger={
            "ev-1": {"paper_id": "arxiv:2010.11929", "paper_title": "Paper"}
        },
        paper_catalog={},
    )
    assert authors in reader.markdown and year in reader.markdown
    assert f"## {heading}" in reader.markdown
