"""DocumentLoader 实现——arxiv API 来源。"""


import httpx
from langchain_core.documents import Document
from litagent.rag.interfaces import DocumentLoader
from litagent.logging import get_logger


logger = get_logger('rag.loader')


class ArxivLoader(DocumentLoader):
    """调 arxiv API 加载论文 metadata。PDF 全文解析器 Phase 8 接入。"""

    BASE_URL = "http://export.arxiv.org/api/query"

    async def load(self, source: str) -> list[Document]:
        """source = arxiv ID 或 search query"""
        params = {
            "search_query": source if ":" in source else f'id_list={source}',
            'start': 0,
            'max_results': 10,
        }
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(self.BASE_URL, params=params)
            resp.raise_for_status()

        return self._parse_feed(resp.text)

    
    def _parse_feed(self, xml_text: str) -> list[Document]:
        """解析 arxiv Atom XML → Document 列表。

        arxiv API 返回的是 Atom XML 格式:
          <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
              <title>...</title>
              <summary>...</summary>
              <id>http://arxiv.org/abs/1703.05175v1</id>
            </entry>
          </feed>

        ns 是 Atom 命名空间声明——缺了会导致 findall 返回空列表。
        每个 <entry> 变成一个 Document(id=arxiv-1703.05175, content=title+summary)。
        arxiv_id 从 <id> URL 路径最后一段提取（如 '1703.05175v1'）。
        """
        import xml.etree.ElementTree as ET

        docs = []
        root = ET.fromstring(xml_text)                        # 字符串 → XML 树
        ns = {"atom": "http://www.w3.org/2005/Atom"}          # Atom 命名空间

        for entry in root.findall("atom:entry", ns):
            title = entry.find("atom:title", ns)              # 论文标题
            summary = entry.find("atom:summary", ns)          # 摘要
            arxiv_id = entry.find("atom:id", ns)              # arxiv URL (含版本号)

            # .text 可能为 None——取出空字符串作为 fallback
            title_text = title.text.strip() if title is not None and title.text else ""
            summary_text = summary.text.strip() if summary is not None and summary.text else ""
            # <id>http://arxiv.org/abs/1703.05175v1</id> → 取 URL 最后一段
            arxiv_id_text = arxiv_id.text.strip().split("/")[-1] if arxiv_id is not None and arxiv_id.text else ""

            page_content = f"{title_text}\n{summary_text}"
            docs.append(Document(
                page_content=page_content,
                metadata={"title": title_text, "source": "arxiv", "arxiv_id": arxiv_id_text}
            ))

        logger.info(f"Loaded {len(docs)} papers from arxiv")
        return docs