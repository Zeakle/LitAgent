"""Implement Qdrant dense-plus-BM25 retrieval with rank fusion."""

import asyncio
import json
import uuid

from langchain_core.documents import Document
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
)
from qdrant_client.models import Document as QdrantDocument
from qdrant_client.models import (
    Fusion,
    FusionQuery,
    Modifier,
    PointIdsList,
    PointStruct,
    Prefetch,
    SparseVectorParams,
    VectorParams,
)

from litagent.config import MemoryConfig, RetrievalMode
from litagent.logging import get_logger
from litagent.rag.corpus import CorpusStats, CorpusStatsStatus
from litagent.rag.embedder import get_embedder
from litagent.rag.interfaces import ScoredDoc, VectorStore
from litagent.rag.models import ContentChunk, ScoredChunkHit

logger = get_logger("rag.vector_store")

DENSE_KEY = "dense"
SPARSE_KEY = "bm25_sparse"


class QdrantVectorStore(VectorStore):
    """Store documents in named dense and sparse Qdrant indexes."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str,
        *,
        identity=None,
        embedder=None,
    ):
        self._client = client
        self._collection = collection_name
        self._identity = identity
        self._embedder = embedder or get_embedder()

    @classmethod
    async def ensure_compatible(
        cls,
        client: AsyncQdrantClient,
        collection_name: str,
        dim: int,
        *,
        identity=None,
        embedder=None,
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
            return cls(
                client,
                collection_name,
                identity=identity,
                embedder=embedder,
            )

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
        return cls(
            client,
            collection_name,
            identity=identity,
            embedder=embedder,
        )

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

    async def upsert_chunks(self, writes) -> None:
        """Upsert deterministic dense/sparse points with safe payloads."""
        points = [
            PointStruct(
                id=write.point_id,
                vector={
                    DENSE_KEY: write.vector,
                    SPARSE_KEY: QdrantDocument(
                        text=write.chunk.text,
                        model="Qdrant/bm25",
                    ),
                },
                payload=write.payload,
            )
            for write in writes
        ]
        if points:
            await self._client.upsert(
                collection_name=self._collection,
                points=points,
                wait=True,
            )

    async def update_payloads(self, updates) -> None:
        """update paper metadata without replacing vectors."""
        for update in updates:
            await self._client.set_payload(
                collection_name=self._collection,
                payload=update.payload,
                points=[update.point_id],
                wait=True,
            )

    async def _query_vector(self, query: str) -> list[float]:
        """Embed one query and guard against dimension drift."""
        vector = await asyncio.to_thread(self._embedder.embed_query, query)
        expected = getattr(self._embedder, "dim", None)
        if expected is not None and len(vector) != expected:
            raise ValueError(f"embedder returned dimension {len(vector)} != {expected}")
        return vector

    async def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        """Fuse dense and BM25 candidates with reciprocal-rank fusion."""
        query_vec = await self._query_vector(query)

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

    async def search_chunks(
        self,
        query: str,
        top_k: int,
        *,
        mode: RetrievalMode = RetrievalMode.RRF,
    ) -> list[ScoredChunkHit]:
        """Execute one typed retrieval mode and restore chunk contracts."""
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        common = {
            "collection_name": self._collection,
            "limit": top_k,
            "with_payload": True,
        }
        if mode is RetrievalMode.BM25:
            response = await self._client.query_points(
                **common,
                query=QdrantDocument(text=query, model="Qdrant/bm25"),
                using=SPARSE_KEY,
            )
        elif mode is RetrievalMode.DENSE:
            query_vector = await self._query_vector(query)
            response = await self._client.query_points(
                **common,
                query=query_vector,
                using=DENSE_KEY,
            )
        elif mode in (RetrievalMode.RRF, RetrievalMode.RRF_RERANK):
            query_vector = await self._query_vector(query)
            response = await self._client.query_points(
                **common,
                prefetch=[
                    Prefetch(query=query_vector, using=DENSE_KEY, limit=top_k),
                    Prefetch(
                        query=QdrantDocument(text=query, model="Qdrant/bm25"),
                        using=SPARSE_KEY,
                        limit=top_k,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
            )
        else:
            raise ValueError(f"unsupported retrieval mode: {mode}")

        hits: list[ScoredChunkHit] = []
        for point in response.points:
            payload = point.payload or {}
            try:
                hits.append(
                    ScoredChunkHit(
                        chunk=ContentChunk.model_validate(payload["chunk"]),
                        title=payload["title"],
                        abstract=payload.get("abstract", ""),
                        authors=payload.get("authors", []),
                        year=payload.get("year"),
                        sources=payload.get("sources", []),
                        warnings=payload.get("warnings", []),
                        score=float(point.score or 0.0),
                        collection=payload["collection"],
                        corpus_version=payload["corpus_version"],
                        schema_version=payload["schema_version"],
                        parser_version=payload["parser_version"],
                        chunking_version=payload["chunking_version"],
                        chunk_strategy=payload["chunk_strategy"],
                        chunk_size=payload["chunk_size"],
                        chunk_overlap=payload["chunk_overlap"],
                        embedding_backend=payload["embedding_backend"],
                        embedding_model=payload["embedding_model"],
                        embedding_document_adapter=payload.get(
                            "embedding_document_adapter"
                        ),
                    )
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    "Skipping malformed paper payload point_id=%s",
                    point.id,
                )
        return hits

    async def stats(self) -> CorpusStats:
        """Distinguish a missing collection from empty and ready indexes."""
        exists = await self._client.collection_exists(self._collection)
        if not exists:
            return CorpusStats(
                status=CorpusStatsStatus.NOT_FOUND,
                collection_name=self._collection,
                points_count=0,
                identity_fingerprint=(
                    self._identity.fingerprint if self._identity else "unknown"
                ),
            )
        info = await self._client.get_collection(self._collection)
        count = int(info.points_count or 0)
        return CorpusStats(
            status=(CorpusStatsStatus.READY if count else CorpusStatsStatus.EMPTY),
            collection_name=self._collection,
            points_count=count,
            identity_fingerprint=(
                self._identity.fingerprint if self._identity else "unknown"
            ),
        )

    async def delete_points(self, point_ids) -> None:
        """Delete deterministic stale points."""
        if point_ids:
            await self._client.delete(
                collection_name=self._collection,
                points_selector=PointIdsList(points=list(point_ids)),
                wait=True,
            )

    async def estimate_logical_footprint_bytes(self) -> int:
        """Estimate reproducible payload and vector bytes for this collection."""
        total = 0
        offset = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=True,
            )
            for record in records:
                total += len(
                    json.dumps(
                        record.payload or {},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                vectors = record.vector or {}
                values = vectors.values() if isinstance(vectors, dict) else [vectors]
                for vector in values:
                    if isinstance(vector, list):
                        total += len(vector) * 4
                    elif hasattr(vector, "values") and hasattr(vector, "indices"):
                        total += len(vector.values) * 4 + len(vector.indices) * 4
            if offset is None:
                return total

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()
