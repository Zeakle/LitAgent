"""Define the core interfaces and result types for the RAG pipeline."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol, Sequence

from langchain_core.documents import Document

from litagent.rag.models import ScoredPaperHit


@dataclass(frozen=True)
class EmbeddingDocument:
    """Carry paper context required by scientific document encoders."""

    paper_id: str
    title: str
    text: str
    section: str
    content_scope: str


@dataclass
class ScoredDoc:
    """Pair a retrieved document with its relevance score."""

    doc: Document
    score: float


class RetrievalEmbedder(Protocol):
    """Expose asymmetric document and query embedding operations."""

    @property
    def dim(self) -> int:
        """Return the dense vector dimension."""
        raise NotImplementedError

    def embed_documents(
        self,
        documents: Sequence[EmbeddingDocument],
    ) -> list[list[float]]:
        """Embed indexable paper chunks."""
        raise NotImplementedError

    def embed_query(self, query: str) -> list[float]:
        """Embed a short retrieval query."""
        raise NotImplementedError


class DocumentLoader(ABC):
    """Load source documents for ingestion."""

    @abstractmethod
    async def load(self, source: str) -> list[Document]:
        """Load documents identified by a source query or identifier."""
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
    """Reorder document or parent-paper candidates."""

    @abstractmethod
    def rerank(self, query: str, docs: list[ScoredDoc]) -> list[ScoredDoc]:
        """Rerank legacy documents for the supplied query."""
        raise NotImplementedError

    def rerank_papers(
        self,
        query: str,
        papers: list[ScoredPaperHit],
    ) -> list[ScoredPaperHit]:
        """Rerank unique parent papers when the backend supports it."""
        raise NotImplementedError("paper reranking is not implemented")
