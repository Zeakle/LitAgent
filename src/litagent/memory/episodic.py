"""Episodic Memory — Qdrant 后端。

跨会话温存储。每次综述调研为一个 Episode，支持语义检索。
"""


import uuid

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import UnexpectedResponse

from litagent.config import MemoryConfig
from litagent.memory.models import Episode
from litagent.rag.embedder import get_embedder
from litagent.logging import get_logger


logger = get_logger('memory.episodic')

COLLECTION_NAME = "episodes"


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
        logger.info(f"Connected to Qdrant")
        return EpisodicMemory(client)

    @staticmethod
    async def _ensure_collection(client: AsyncQdrantClient) -> None:
        """创建 collection（如果不存在）。"""
        dim = get_embedder().dim
        try:
            await client.get_collection(COLLECTION_NAME)
        except UnexpectedResponse:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {COLLECTION_NAME}")
    

    async def store(self, episode: Episode) -> str:
        """存储 Episode。分配 episode_id，写入 Qdrant。"""
        if not episode.episode_id:
            episode.episode_id = str(uuid.uuid4())

        import time
        episode.created_at = time.time()

        vector = get_embedder().embed(episode.summary)
        point = PointStruct(id=episode.episode_id, vector=vector, payload=episode.to_dict())
        await self._client.upsert(collection_name=COLLECTION_NAME, points=[point])
        return episode.episode_id


    async def search(self, query: str, top_k: int = 5) -> list[Episode]:
        """语义搜索 + 时间衰减重排。"""
        query_vec = get_embedder().embed(query)
        results = await self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k * 2,  # 多取一些，decay 重排后截断
            with_payload=True,
        )

        episodes = []
        for r in results.points:
            if not r.payload:
                continue
            ep = Episode.from_dict(r.payload)
            qdr_score = r.score if r.score else 0.0
            # _score 是临时字段（不持久化），用于重排
            ep._score = qdr_score * (1.0 + ep.decay_score())
            episodes.append(ep)

        # 按合并分数降序
        episodes.sort(key=lambda e: getattr(e, '_score', 0), reverse=True)

        # 清理临时字段
        for ep in episodes:
            delattr(ep, '_score')
        return episodes[:top_k]

    
    async def delete(self, episode_id: str) -> None:
        from qdrant_client.models import PointIdsList
        await self._client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=[episode_id])
        )

    
    async def close(self) -> None:
        """关闭Qdrant连接"""
        await self._client.close()