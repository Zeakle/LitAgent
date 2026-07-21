"""RAG 可插拔接口定义。直接使用 LangChain Document——不自定义文档类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from langchain_core.documents import Document


@dataclass
class ScoredDoc:
    """LangChain Document + 相关性分数。LangChain 没有这个类，我们补。"""
    doc: Document
    score: float


class DocumentLoader(ABC):
    @abstractmethod
    async def load(self, source: str) -> list[Document]:
        ...


class Chunker(ABC):
    @abstractmethod
    def chunk(self, doc: Document) -> list[Document]:
        ...


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        ...


class VectorStore(ABC):
    @abstractmethod
    async def add(self, docs: list[Document], vectors: list[list[float]]) -> None:
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[ScoredDoc]:
        ...


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        ...
