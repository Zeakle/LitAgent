"""Claims Index——声明级 RAG 索引。"""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from litagent.config import MemoryConfig
from litagent.observability.context import get_task_id
from litagent.rag.embedder import get_embedder
from litagent.logging import get_logger


logger = get_logger('rag.claims_index')
COLLECTION_NAME = 'claims'


@dataclass
class Claim:
    """一条声明--论文中提取的断言"""
    text: str                          # e.g. "ProtoNet achieves 93.2% on miniImageNet"
    claim_id: str = ""
    source_paper: str = ""             # 来源 arxiv ID / title
    entities: list[str] = field(default_factory=list)  # 实体标签
    confidence: float = 0.5            # 提取置信度


class ClaimsIndex:
    """声明级 dense 索引，供 Reviewer 交叉验证声明使用"""

    def __init__(self, client: AsyncQdrantClient, trace_hook=None):
        self._client = client
        self._trace_hook = trace_hook


    def _emit(self, event: str, data: dict) -> None:
        """触发 trace hook"""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    
    @staticmethod
    async def connect(config: MemoryConfig) -> "ClaimsIndex":
        client = AsyncQdrantClient(url=config.qdrant_url)
        dim = get_embedder().dim

        try:
            await client.get_collection(COLLECTION_NAME)
        except Exception:
            await client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            logger.info(f"Created claims index collection: {COLLECTION_NAME} (dim={dim})")
        return ClaimsIndex(client)


    async def add(self, claims: list[Claim]) -> list[str]:
        """批量写入claims到Qdrant。返回claim_id列表"""
        embedder = get_embedder()
        points = []
        for c in claims:
            if not c.claim_id:
                c.claim_id = str(uuid.uuid4())
            vec = embedder.embed(c.text)
            points.append(PointStruct(
                id=c.claim_id,
                vector=vec,
                payload={
                    'text': c.text,
                    'source_paper': c.source_paper,
                    'entities': c.entities,
                    'confidence': c.confidence,
                },
            ))
        await self._client.upsert(collection_name=COLLECTION_NAME, points=points)
        logger.info(f"Indexed {len(claims)} claims")
        self._emit("claims.op", {"task_id": get_task_id(), "op": "add", "count": len(claims)})
        return [c.claim_id for c in claims]


    async def search(self, query: str, top_k: int = 10) -> list[Claim]:
        """语义搜索claims.按文本相似度排序"""
        embedder = get_embedder()
        query_vec = embedder.embed(query)
        results = await self._client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vec,
            limit=top_k,
            with_payload=True
        )

        claims = []
        for r in results.points:
            if r.payload:
                claims.append(Claim(
                    claim_id=r.id,
                    text=r.payload.get("text", ""),
                    source_paper=r.payload.get("source_paper", ""),
                    entities=r.payload.get("entities", []),
                    confidence=r.payload.get("confidence", 0.5),
                ))
        self._emit("claims.op", {"task_id": get_task_id(), "op": "search",
                                "query": query[:200], "count": len(results.points)})
        return claims


    async def close(self) -> None:
        """关闭Qdrant连接"""
        await self._client.close()