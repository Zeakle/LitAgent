"""Store and retrieve time-weighted session episodes in Qdrant."""

import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, PointStruct, VectorParams

from litagent.config import MemoryConfig
from litagent.logging import get_logger
from litagent.memory.models import Episode
from litagent.rag.embedder import get_embedder

logger = get_logger("memory.episodic")

COLLECTION_NAME = "episodes"


class EpisodicMemory:
    """Persist summarized research sessions as vector-searchable episodes."""

    def __init__(self, client: AsyncQdrantClient):
        """Initialize the episodic memory."""
        self._client = client

    @staticmethod
    async def connect(config: MemoryConfig) -> "EpisodicMemory":
        """Connect to Qdrant and ensure the episode collection exists."""
        client = AsyncQdrantClient(url=config.qdrant_url)

        await EpisodicMemory._ensure_collection(client)
        logger.info(f"Connected to Qdrant")
        return EpisodicMemory(client)

    @staticmethod
    async def _ensure_collection(client: AsyncQdrantClient) -> None:
        """Create the episode collection when it is absent."""
        dim = get_embedder().dim
        try:
            await client.get_collection(COLLECTION_NAME)
        except UnexpectedResponse:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")

    async def store(self, episode: Episode) -> str:
        """Embed and persist an episode, assigning its identity and timestamp."""
        if not episode.episode_id:
            episode.episode_id = str(uuid.uuid4())

        import time

        episode.created_at = time.time()

        vector = get_embedder().embed(episode.summary)
        point = PointStruct(
            id=episode.episode_id, vector=vector, payload=episode.to_dict()
        )
        await self._client.upsert(collection_name=COLLECTION_NAME, points=[point])
        return episode.episode_id

    async def search(self, query: str, top_k: int = 5) -> list[Episode]:
        """Retrieve episodes by similarity and time-decayed importance."""
        query_vec = get_embedder().embed(query)
        results = await self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k * 2,
            with_payload=True,
        )

        episodes = []
        for r in results.points:
            if not r.payload:
                continue
            ep = Episode.from_dict(r.payload)
            qdr_score = r.score if r.score else 0.0

            # Blend semantic similarity with decayed importance before ranking.
            ep._score = qdr_score * (1.0 + ep.decay_score())
            episodes.append(ep)

        episodes.sort(key=lambda e: getattr(e, "_score", 0), reverse=True)

        for ep in episodes:
            delattr(ep, "_score")
        return episodes[:top_k]

    async def delete(self, episode_id: str) -> None:
        """Delete an episode by identifier."""
        from qdrant_client.models import PointIdsList

        await self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=[episode_id]),
        )

    async def close(self) -> None:
        """Close the Qdrant client."""
        await self._client.close()
