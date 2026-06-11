"""Episodic Memory — Qdrant 后端。

跨会话温存储。每次综述调研为一个 Episode，支持语义检索。
"""


import uuid

from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
)
from qdrant_client.http.exceptions import UnexpectedResponse

from litagent.config import MemoryConfig
from litagent.memory.models import Episode
from litagent.logging import get_logger


logger = get_logger('memory.episodic')

COLLECTION_NAME = "episodes"
VECTOR_SIZE = 1536


class EpisodicMemory:
    """Qdrant 存储的 Episodic Memory。

    每个 Episode 存储为一个 Qdrant Point:
    - vector: summary 的 embedding (Phase 6 引入 SPECTER2 前用 dummy)
    - payload: Episode.to_dict() 的全部字段
    """

    def __init__(self, client: AsyncQdrantClient):
        self._client = client

    @staticmethod
    async def connect(config: MemoryConfig) -> "EpisodicMemory":
        client = AsyncQdrantClient(url=config.qdrant_url)

        await EpisodicMemory._ensure_collection(client)
        logger.info(f"Connected to Qdrant: {config.qdrant_url}")
        return EpisodicMemory(client)

    @staticmethod
    async def _ensure_collection(client: AsyncQdrantClient) -> None:
        """创建 collection（如果不存在）。"""
        try:
            await client.get_collection(COLLECTION_NAME)
        except UnexpectedResponse:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    
    async def store(self, episode: Episode) -> str:
        """存储 Episode。分配 episode_id，写入 Qdrant。"""
        if not episode.episode_id:
            episode.episode_id = str(uuid.uuid4())

        import time
        episode.created_at = time.time()

        dummy_vector = [0.0] * VECTOR_SIZE

        point = PointStruct(
            id=episode.episode_id,
            vector=dummy_vector,
            payload=episode.to_dict(),
        )

        await self._client.upsert(collection_name=COLLECTION_NAME, points=[point])
        logger.debug(f"Stored episode: {episode.episode_id}")
        return episode.episode_id

    async def search(self, query: str, top_k: int = 5) -> list[Episode]:
        """语义（Phase 4 dummy）+ 关键词混合检索。
        
        Phase 4: embedding 是零向量，语义搜索无意义。
        先用 Qdrant 的 scroll + 客户端关键词过滤作为过渡。
        Phase 6 接入 SPECTER2 后换为真正的向量搜索。
        """

        records, _ = await self._client.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            with_payload=True
        )

        query_lower = query.lower()
        scored = []
        for record in records:
            payload = record.payload or {}
            text = f"{payload.get('summary', '')} {payload.get('intent', '')}".lower()
            score = sum(1 for word in query_lower.split() if word in text)
            if score > 0:
                scored.append((score, payload))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [Episode.from_dict(p) for _, p in scored[:top_k]]

    
    async def delete(self, episode_id: str) -> None:
        from qdrant_client.models import PointIdsList
        await self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=[episode_id])
        )