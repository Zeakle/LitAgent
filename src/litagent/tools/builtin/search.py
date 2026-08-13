"""Search arXiv, Semantic Scholar, and Hugging Face papers."""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import httpx

from litagent.logging import get_logger
from litagent.tools.base import FallbackStep, ToolCategory, ToolDefinition
from litagent.tools.registry import ToolRegistry, get_registry

logger = get_logger("tools.search")


async def search_arxiv(query: str = "", max_results: int = 20) -> list[dict]:
    """Search arXiv and return normalized paper records."""
    url = "https://export.arxiv.org/api/query"
    params = {"search_query": f"all:{query}", "start": 0, "max_results": max_results}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
    return _parse_arxiv_xml(resp.text)


def _parse_arxiv_xml(xml_text: str) -> list[dict]:
    """Parse an arXiv Atom feed into normalized paper records."""
    papers = []
    root = ET.fromstring(xml_text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        id_el = entry.find("atom:id", ns)
        title = title_el.text.strip() if title_el is not None and title_el.text else ""
        abstract = (
            summary_el.text.strip()
            if summary_el is not None and summary_el.text
            else ""
        )
        arxiv_id = (
            id_el.text.strip().split("/")[-1]
            if id_el is not None and id_el.text
            else ""
        )
        papers.append(
            {
                "paper_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "source": "arxiv",
            }
        )
    return papers


async def search_semantic_scholar(query: str = "", max_results: int = 20) -> list[dict]:
    """Search Semantic Scholar, returning no results for a non-200 response."""
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "paperId,title,abstract,citationCount,year",
    }
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    headers = {"x-api-key": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params, headers=headers)
    if resp.status_code != 200:

        logger.warning(
            f"Semantic Scholar returned {resp.status_code}, skipping (degraded)"
        )
        return []
    data = resp.json().get("data", []) or []
    return [
        {
            "paper_id": item.get("paperId", ""),
            "title": item.get("title", ""),
            "abstract": item.get("abstract", "") or "",
            "citation_count": item.get("citationCount", 0),
            "year": item.get("year"),
            "source": "semantic_scholar",
        }
        for item in data
    ]


async def search_huggingface(query: str = "", max_results: int = 20) -> list[dict]:
    """Search Hugging Face papers, returning no results for a non-200 response."""
    url = "https://huggingface.co/api/papers/search"
    params = {"q": query}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params)
    if resp.status_code != 200:
        logger.warning(
            f"HuggingFace papers returned {resp.status_code}, skipping (degraded)"
        )
        return []
    items = resp.json() or []
    papers = []
    for entry in items[:max_results]:
        p = entry.get("paper", {}) if isinstance(entry, dict) else {}
        if not p:
            continue
        papers.append(
            {
                "paper_id": p.get("id", ""),
                "title": p.get("title", ""),
                "abstract": p.get("summary", "") or "",
                "citation_count": p.get("upvotes", 0),
                "source": "huggingface",
            }
        )
    return papers


def register_search_tools(registry: ToolRegistry | None = None) -> None:
    """Register paper-search tools in an injected or compatibility registry."""
    r = registry if registry is not None else get_registry()
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "minLength": 1},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    r.register(
        ToolDefinition(
            name="search_arxiv",
            description="Search arxiv by keyword",
            parameters=parameters,
            category=ToolCategory.READ,
            timeout_ms=30000,
            max_retries=2,
            fallback=[
                FallbackStep(
                    type="alternative_tool", alternative_tool="search_semantic_scholar"
                ),
                FallbackStep(
                    type="alternative_tool", alternative_tool="search_huggingface"
                ),
                FallbackStep(type="skip"),
            ],
        ),
        search_arxiv,
    )
    r.register(
        ToolDefinition(
            name="search_semantic_scholar",
            description="Search Semantic Scholar",
            parameters=parameters,
            category=ToolCategory.READ,
            timeout_ms=30000,
            max_retries=2,
        ),
        search_semantic_scholar,
    )
    r.register(
        ToolDefinition(
            name="search_huggingface",
            description="Search HuggingFace papers",
            parameters=parameters,
            category=ToolCategory.READ,
            timeout_ms=30000,
            max_retries=2,
        ),
        search_huggingface,
    )
