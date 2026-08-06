"""Store and retrieve trusted, locator-backed claims in a Qdrant collection."""

from __future__ import annotations

import asyncio
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from litagent.config import MemoryConfig
from litagent.logging import get_logger
from litagent.observability.lifecycle import traced_io
from litagent.rag.embedder import get_embedder
from litagent.rag.models import ChunkSourceSpan

logger = get_logger("rag.claims_index")
COLLECTION_NAME = "claims"


class TrustedClaim(BaseModel):
    """Represent one promoted, locator-backed, cross-run claim."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    text: str = Field(min_length=1)
    supporting_text: str = Field(min_length=1)
    run_id: str
    domain: str
    paper_id: str
    evidence_id: str
    chunk_key: str
    section: str
    page: int | None = None
    block_index: int | None = None
    bbox: list[float] | None = None
    source_spans: list[ChunkSourceSpan] = Field(default_factory=list)
    content_scope: str
    content_hash: str
    evidence_version: str
    quality_status: str
    delivery_status: str
    trust_state: Literal["trusted"] = "trusted"
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class ClaimsIndex:
    """Index trusted claims for semantic retrieval with trace events."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        trace_hook=None,
        *,
        collection_name: str = COLLECTION_NAME,
        embedder=None,
    ) -> None:
        self._client = client
        self._trace_hook = trace_hook
        self._collection = collection_name
        self._embedder = embedder or get_embedder()

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event without allowing hook failures to escape."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    @staticmethod
    async def connect(
        config: MemoryConfig,
        *,
        collection_name: str = COLLECTION_NAME,
        trace_hook=None,
        embedder=None,
    ) -> "ClaimsIndex":
        """Keep the existing standalone factory while honoring collection identity."""
        client = AsyncQdrantClient(url=config.qdrant_url)
        resolved_embedder = embedder or get_embedder()
        try:
            try:
                await client.get_collection(collection_name)
            except Exception:
                await client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=resolved_embedder.dim,
                        distance=Distance.COSINE,
                    ),
                )
            return ClaimsIndex(
                client,
                trace_hook,
                collection_name=collection_name,
                embedder=resolved_embedder,
            )
        except BaseException:
            # Ownership has not transferred to the returned ClaimsIndex yet.
            await client.close()
            raise

    async def upsert_trusted(self, claims: Sequence[TrustedClaim]) -> int:
        """Embed and index trusted claims with deterministic IDs."""
        if not claims:
            return 0
        vectors = await asyncio.to_thread(
            self._embedder.embed, [claim.text for claim in claims]
        )
        points = [
            PointStruct(
                id=claim.claim_id,
                vector=vector,
                payload=claim.model_dump(mode="json"),
            )
            for claim, vector in zip(claims, vectors, strict=True)
        ]
        async with traced_io(
            self._emit,
            "claims.promote",
            {"count": len(points), "collection": self._collection},
        ) as outcome:
            await self._client.upsert(
                collection_name=self._collection, points=points, wait=True
            )
            outcome["count"] = len(points)
        return len(points)

    async def search(self, query: str, top_k: int = 10) -> list[TrustedClaim]:
        """Retrieve trusted claims nearest to the query embedding."""
        query_vector = await asyncio.to_thread(self._embedder.embed, query)
        trusted_filter = Filter(
            must=[
                FieldCondition(
                    key="trust_state",
                    match=MatchValue(value="trusted"),
                )
            ]
        )
        async with traced_io(
            self._emit,
            "claims.search",
            {"query": query, "top_k": top_k, "trusted_only": True},
        ) as outcome:
            result = await self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                query_filter=trusted_filter,
                limit=top_k,
                with_payload=True,
            )
            claims = []
            for point in result.points:
                try:
                    claims.append(TrustedClaim.model_validate(point.payload or {}))
                except (TypeError, ValueError):
                    # Legacy/untrusted rows are intentionally invisible.
                    continue
            outcome["count"] = len(claims)
        return claims

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()
