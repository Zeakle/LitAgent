"""Chunker 实现--Semantic + FixedWindow"""


import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from litagent.rag.interfaces import Chunker
from litagent.logging import get_logger

logger = get_logger('rag.chunker')


class RecursiveChunker(Chunker):

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '.', ' ', '']
        )

    def chunk(self, doc: Document) -> list[Document]:
        texts = self._splitter.split_text(doc.page_content)
        if not texts:
            return [doc]
        chunks = []
        for i, text in enumerate(texts):
            chunks.append(
                Document(
                    page_content=text,
                    metadata={**doc.metadata, "section": f"chunk_{i}"}
                )
            )
        return chunks


class SemanticChunker(Chunker):
    """按论文章节切分。

    识别 Abstract / Introduction / Methods / Results / Discussion 等 section header，
    每个 section 一个 chunk。标题保留在 chunk 开头作为 context。
    """

    SECTION_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:\d+\.?\s*)?(Abstract|Introduction|Related Work|'
        r'Method[s]?|Approach|Experiments?|Results?|Discussion|'
        r'Conclusion|References?)\s*\n',
        re.IGNORECASE
    )

    # fallback: 匹配任何类似 section header 的行（全大写行 / 编号开头 / 超过 3 个大写词）
    FALLBACK_PATTERN = re.compile(
        r'(?:^|\n)\s*(?:\d+\.?\s+)?([A-Z][A-Za-z\s]{2,50})\s*\n'
    )

    def __init__(self, max_section_chars: int = 2000, chunk_size: int = 512, chunk_overlap: int = 64):
        self._max_section_chars = max_section_chars
        self._sub_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=['\n\n', '\n', '. ', ' ', '']
        )

    def chunk(self, doc: Document) -> list[Document]:
        sections = self.SECTION_PATTERN.split(doc.page_content)
        if len(sections) <= 1:
            # pattern未匹配，用fall_back pattern
            sections = self.FALLBACK_PATTERN.split(doc.page_content)
        if len(sections) <= 1:
            return [
                Document(page_content=doc.page_content, metadata={**doc.metadata, 'section': 'full'})
            ]

        chunks = []
        header = None
        for part in sections:
            part = part.strip()
            if not part:
                continue
            if re.match(r'^\s*(?:\d+\.?\s*)?(?:Abstract|Introduction|Related|Method|Approach|Experiment|Result|Discussion|Conclusion|Reference)',
                       part, re.IGNORECASE):
                header = part
            elif header and len(part) > 10:
                section_name = header.lower().replace(" ", "_")  
                if len(part) > self._max_section_chars:
                    sub_texts = self._sub_splitter.split_text(part)
                    for i, sub in enumerate(sub_texts):
                        chunks.append(
                            Document(
                                page_content=sub,
                                metadata={**doc.metadata, "section": section_name}
                            )
                        )
                else:
                    chunks.append(Document(
                        page_content=f"{header}\n{part}",
                        metadata={**doc.metadata, "section": section_name}
                    ))
        
        logger.debug(f"SemanticChunker: → {len(chunks)} chunks")
        return chunks