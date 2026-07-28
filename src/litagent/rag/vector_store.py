"""Implement Qdrant dense-plus-BM25 retrieval with rank fusion."""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    Modifier,
    PointStruct,
    Prefetch,
    FusionQuery,
    Fusion,
)
from qdrant_client.models import Document as QdrantDocument
from langchain_core.documents import Document

from litagent.rag.interfaces import VectorStore, ScoredDoc
from litagent.rag.embedder import get_embedder
from litagent.config import MemoryConfig
from litagent.logging import get_logger

logger = get_logger("rag.vector_store")

DENSE_KEY = "dense"
SPARSE_KEY = "bm25_sparse"


class QdrantVectorStore(VectorStore):
    """Store documents in named dense and sparse Qdrant indexes."""

    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self._client = client
        self._collection = collection_name

    @classmethod
    async def ensure_compatible(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        dim: int,
    ) -> "QdrantVectorStore":
        """Ensure the collection exposes the expected named dense vector.

        Empty collections without that vector can be recreated in place.
        A non-empty collection without it is never deleted automatically.
        """

        async def _create() -> None:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    DENSE_KEY: VectorParams(size=dim, distance=Distance.COSINE)
                },
                sparse_vectors_config={
                    SPARSE_KEY: SparseVectorParams(modifier=Modifier.IDF)
                },
            )
            logger.info(f"Created dual-index collection: {collection_name} (dim={dim})")

        try:
            info = await client.get_collection(collection_name)
        except Exception:
            await _create()
            return cls(client, collection_name)

        vectors = info.config.params.vectors
        if not (isinstance(vectors, dict) and DENSE_KEY in vectors):
            if info.points_count and info.points_count > 0:
                from litagent.exceptions import ConfigError

                raise ConfigError(
                    f"RAG schema migration required for '{collection_name}': "
                    f"{info.points_count} points exist with incompatible schema "
                    f"(no named '{DENSE_KEY}' vector). "
                    f"No automatic deletion of non-empty collection."
                )

            logger.warning(
                f"Collection '{collection_name}' has incompatible vector schema "
                f"(no named '{DENSE_KEY}' vector) — recreating for dual-index"
            )
            await client.delete_collection(collection_name)
            await _create()
        return cls(client, collection_name)

    @classmethod
    async def connect(
        cls, config: MemoryConfig, collection_name: str
    ) -> "QdrantVectorStore":
        """Connect to Qdrant and ensure the named dense vector is available."""
        client = AsyncQdrantClient(url=config.qdrant_url)
        return await cls.ensure_compatible(client, collection_name, get_embedder().dim)

    async def add(self, docs: list[Document], vectors: list[list[float]]) -> None:
        """Index documents with dense and BM25 sparse representations."""
        points = []
        for d, v in zip(docs, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_KEY: v,
                        SPARSE_KEY: QdrantDocument(
                            text=d.page_content, model="Qdrant/bm25"
                        ),
                    },
                    payload={"page_content": d.page_content, **d.metadata},
                )
            )
        await self._client.upsert(collection_name=self._collection, points=points)
        logger.debug(f"Added {len(points)} points to {self._collection}")

    async def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        """Fuse dense and BM25 candidates with reciprocal-rank fusion."""
        embedder = get_embedder()
        query_vec = embedder.embed(query)

        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=query_vec, using=DENSE_KEY, limit=top_k * 2),
                Prefetch(
                    query=QdrantDocument(text=query, model="Qdrant/bm25"),
                    using=SPARSE_KEY,
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )

        scored = []
        for r in response.points:
            if r.payload:
                metadata = {k: v for k, v in r.payload.items() if k != "page_content"}
                doc = Document(
                    page_content=r.payload.get("page_content", ""), metadata=metadata
                )
                scored.append(ScoredDoc(doc=doc, score=r.score if r.score else 0))
        return scored

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()
