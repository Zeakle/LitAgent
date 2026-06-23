"""Search tools — arxiv / Semantic Scholar / PapersWithCode."""

import httpx
import xml.etree.ElementTree as ET

from pytest import param

from litagent.tools.base import ToolDefinition, ToolCategory
from litagent.tools.registry import get_registry


async def search_arxiv(query: str = "", max_results: int = 20) -> list[dict]:
    url = "http://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
    return _parse_arxiv_xml(resp.text)


def _parse_arxiv_xml(xml_text: str) -> list[dict]:
    papers = []
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        abstract = summary_el.text.strip() if summary_el is not None and summary_el.text else ""
        arxiv_id = id_el.text.strip().split("/")[-1] if id_el is not None and id_el.text else ""
        papers.append({"paper_id": arxiv_id, "title": title, "abstract": abstract, "source": "arxiv"})
    return papers


async def search_semantic_scholar(query: str = "", max_results: int = 20) -> list[dcit]:
    params = {"query": query, "limit": max_results, "fields": "paperId,title,abstract,citationCount,year"}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
    data = resp.json().get('data', [])
    return [{"paper_id": item.get("paperId", ""), "title": item.get("title", ""),
            "abstract": item.get("abstract", "") or "", "citation_count": item.get("citationCount", 0),
            "year": item.get("year"), "source": "semantic_scholar"} for item in data]


async def search_paperswithcode(query: str = "", max_results: int = 20) -> list[dict]:
    url = "https://paperswithcode.com/api/v1/papers/"
    params = {"q": query, "page": 1, "items_per_page": max_results}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
    results = resp.json().get("results", [])
    return [{"paper_id": item.get("id", ""), "title": item.get("title", ""),
             "abstract": item.get("abstract", "") or "", "source": "paperswithcode",
             "url_pdf": item.get("url_pdf", "")} for item in results]

    
def register_search_tools():
    r = get_registry()
    r.register(ToolDefinition(name="search_arxiv", description="Search arxiv by keyword",
            category=ToolCategory.READ, timeout_ms=30000, max_retries=2), search_arxiv)
    r.register(ToolDefinition(name="search_semantic_scholar", description="Search Semantic Scholar",
              category=ToolCategory.READ, timeout_ms=30000, max_retries=2), search_semantic_scholar)
    r.register(ToolDefinition(name="search_paperswithcode", description="Search PapersWithCode",
              category=ToolCategory.READ, timeout_ms=30000, max_retries=2), search_paperswithcode)