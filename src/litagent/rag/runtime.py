"""Own corpus-only infrastructure used by CLI ingestion."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

import asyncpg
from qdrant_client import AsyncQdrantClient

from litagent.config import AppConfig
from litagent.rag.corpus import CollectionIdentity, CorpusService
from litagent.rag.embedder import LocalEmbedder
from litagent.rag.state import CorpusStateRepository
from litagent.rag.vector_store import QdrantVectorStore


@dataclass
class CorpusRuntime:
    """Own corpus-only infrastructure used by CLI ingestion."""

    identity: CollectionIdentity
    state: CorpusStateRepository
    store: QdrantVectorStore
    service: CorpusService
    embedder: LocalEmbedder
    qdrant_client: Any
    pg_pool: Any
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    async def connect(
        cls,
        config: AppConfig,
        *,
        purpose: Literal["runtime", "benchmark"] = "runtime",
    ) -> "CorpusRuntime":
        identity = CollectionIdentity.from_config(
            config.rag,
            purpose=purpose,
        )
        embedder = LocalEmbedder(config.rag.embedding_model)
        dim = await asyncio.to_thread(lambda: embedder.dim)
        qdrant_client = AsyncQdrantClient(url=config.memory.qdrant_url)
        pg_pool = None
        try:
            pg_pool = await asyncpg.create_pool(config.memory.pg_url)
            state = CorpusStateRepository(pg_pool)
            await state.ensure_tables()
            store = await QdrantVectorStore.ensure_compatible(
                qdrant_client,
                identity.collection_name,
                dim,
                identity=identity,
                embedder=embedder,
            )
            return cls(
                identity=identity,
                state=state,
                store=store,
                service=CorpusService(
                    identity=identity,
                    state_repository=state,
                    paper_index=store,
                    embedder=embedder,
                ),
                embedder=embedder,
                qdrant_client=qdrant_client,
                pg_pool=pg_pool,
            )
        except BaseException:
            await qdrant_client.close()
            if pg_pool is not None:
                await pg_pool.close()
            raise

    async def close(self) -> None:
        """Close each owned backend once."""
        if self._closed:
            return
        self._closed = True
        await self.qdrant_client.close()
        await self.pg_pool.close()
