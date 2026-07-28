"""Load paper metadata and abstracts from the arXiv Atom API."""

import httpx
from langchain_core.documents import Document

from litagent.rag.interfaces import DocumentLoader
from litagent.logging import get_logger

logger = get_logger("rag.loader")


class ArxivLoader(DocumentLoader):
    """Load arXiv searches or identifiers as retrieval documents."""

    BASE_URL = "http://export.arxiv.org/api/query"

    async def load(self, source: str) -> list[Document]:
        """Query arXiv by search expression or identifier."""
        if ":" in source:
            params = {"search_query": source, "start": 0, "max_results": 10}
        else:
            params = {"id_list": source, "start": 0, "max_results": 10}
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self.BASE_URL, params=params)
            resp.raise_for_status()

        return self._parse_feed(resp.text)

    def _parse_feed(self, xml_text: str) -> list[Document]:
        """Parse an arXiv Atom feed into retrieval documents."""
        import xml.etree.ElementTree as ET

        docs = []
        root = ET.fromstring(xml_text)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)
            summary = entry.find("atom:summary", ns)
            arxiv_id = entry.find("atom:id", ns)

            title_text = title.text.strip() if title is not None and title.text else ""
            summary_text = (
                summary.text.strip() if summary is not None and summary.text else ""
            )

            arxiv_id_text = (
                arxiv_id.text.strip().split("/")[-1]
                if arxiv_id is not None and arxiv_id.text
                else ""
            )

            page_content = f"{title_text}\n{summary_text}"
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "title": title_text,
                        "source": "arxiv",
                        "arxiv_id": arxiv_id_text,
                    },
                )
            )

        logger.info(f"Loaded {len(docs)} papers from arxiv")
        return docs
