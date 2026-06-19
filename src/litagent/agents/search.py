"""Search Worker--调API搜索论文"""


from __future__ import annotations
from typing import Any

import httpx

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger

logger = get_logger("agents.search")


class SearchWorker(Worker):
    """搜索Worker--根据 input_data['source']路由到对应的数据源"""

    @property
    def agent_type(self) -> str:
        return 'search'

    
    async def execute(self, task: SubTask) -> Any:
        source = task.input_data.get('source', 'arxiv')
        query = task.input_data.get('query', '')

        try:
            if source == 'arxiv':
                return await self._search_arxiv(query)
            elif source == 'semantic_scholar':
                return await self._search_semantic_scholar(query)
            elif source == 'paperswithcode':
                return await self._search_paperswithcode(query)
            else:
                logger.warning(f"Unknown source '{source}', falling back to arxiv")
                return await self._search_arxiv(query)
        except Exception as e:
            logger.warning(f"Search '{source}' failed: {e}, returning empty")
            return []
    

    async def _search_arxiv(self, query: str, max_results: int = 20) -> list[dict]:
        """调 arxiv API 搜索论文。复用已有的 XML 解析逻辑。"""
        url = "http://export.arxiv.org/api/query"
        params = {'search_query': f'all:{query}', 'start': 0, 'max_results': max_results}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()

        return self._parse_arxiv_xml(resp.text)

    
    def _parse_arxiv_xml(self, xml_text: str) -> list[dict]:
        import xml.etree.ElementTree as ET
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

            papers.append({
                "paper_id": arxiv_id,
                "title": title,
                "abstract": abstract,
                "source": "arxiv",
            })
        logger.info(f"arxiv: found {len(papers)} papers")
        return papers

    
    async def _search_semantic_scholar(self, query: str, limit: int=20) -> list[dict]:
        """调 Semantic Scholar API"""
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": query,
            "limit": limit,
            "fields": "paperId,title,abstract,citationCount,year",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()

        data = resp.json().get('data', [])
        papers = []
        for item in data:
            papers.append({
                "paper_id": item.get("paperId", ""),
                "title": item.get("title", ""),
                "abstract": item.get("abstract", "") or "",
                "citation_count": item.get("citationCount", 0),
                "year": item.get("year"),
                "source": "semantic_scholar",
            })
        logger.info(f"semantic_scholar: found {len(papers)} papers")
        return papers


    async def _search_paperswithcode(self, query: str, limit: int = 20) -> list[dict]:
        """调 PapersWithCode API。"""
        url = "https://paperswithcode.com/api/v1/papers/"
        params = {"q": query, "page": 1, "items_per_page": limit}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()

        results = resp.json().get('results', [])
        papers = []
        for item in results:
            papers.append({
                "paper_id": item.get("id", ""),
                "title": item.get("title", ""),
                "abstract": item.get("abstract", "") or "",
                "source": "paperswithcode",
                "url_pdf": item.get("url_pdf", ""),
            })
        logger.info(f"paperswithcode: found {len(papers)} papers")
        return papers