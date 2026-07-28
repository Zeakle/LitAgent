"""Define the core interfaces and result types for the RAG pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass
class ScoredDoc:
    """Pair a retrieved document with its relevance score."""

    doc: Document
    score: float


class DocumentLoader(ABC):
    """Load source documents for ingestion."""

    @abstractmethod
    async def load(self, source: str) -> list[Document]:
        """Load documents identified by a source query or identifier."""
        ...


class Chunker(ABC):
    """Split documents into retrieval units."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Document]:
        """Split a document into retrieval-ready chunks."""
        ...


class Embedder(ABC):
    """Convert text into dense vector representations."""

    @abstractmethod
    def embed(self, texts: str | list[str]) -> list[float] | list[list[float]]:
        """Generate embeddings for the supplied texts."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int:
        """Return the embedding dimension."""
        ...


class VectorStore(ABC):
    """Persist embedded documents and perform vector retrieval."""

    @abstractmethod
    async def add(self, docs: list[Document], vectors: list[list[float]]) -> None:
        """Persist documents with their precomputed dense vectors."""
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int) -> list[ScoredDoc]:
        """Search for items matching the supplied query."""
        ...


class Reranker(ABC):
    """Reorder retrieved documents by query-document relevance."""

    @abstractmethod
    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        """Rerank documents for the supplied query."""
        ...
