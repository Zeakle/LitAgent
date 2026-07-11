"""VectorStore 实现——Qdrant dual-index（dense + sparse BM25 + 服务端 RRF）。"""

import uuid
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, Modifier,
    PointStruct, Prefetch, FusionQuery, Fusion, Document as QdrantDocument
)
from langchain_core.documents import Document

from litagent.rag.interfaces import VectorStore, ScoredDoc
from litagent.rag.embedder import get_embedder
from litagent.config import MemoryConfig
from litagent.logging import get_logger


logger = get_logger('rag.vector_store')

DENSE_KEY = 'dense'
SPARSE_KEY = 'bm25_sparse'


class QdrantVectorStore(VectorStore):
    """Qdrant dual-index store。

    每个 collection 同时存储：
    - dense vector (embedding → HNSW)
    - sparse vector (BM25 → IDF 服务端计算)
    检索时 Qdrant 服务端做 RRF fusion，客户端只收发结果。
    """
    
    def __init__(self, client: AsyncQdrantClient, collection_name: str):
        self._client = client
        self._collection = collection_name

    @staticmethod
    async def connect(config: MemoryConfig, collection_name: str) -> "QdrantVectorStore":
        client = AsyncQdrantClient(url=config.qdrant_url)
        dim = get_embedder().dim

        async def _create() -> None:
            await client.create_collection(
                collection_name=collection_name,
                vectors_config={DENSE_KEY: VectorParams(size=dim, distance=Distance.COSINE)},
                sparse_vectors_config={SPARSE_KEY: SparseVectorParams(modifier=Modifier.IDF)},
            )
            logger.info(f"Created dual-index collection: {collection_name} (dim={dim})")

        try:
            info = await client.get_collection(collection_name)
        except Exception:
            await _create()          # 不存在 → 建
            return QdrantVectorStore(client, collection_name)

        # 已存在 → 校验结构：必须是命名向量且含 DENSE_KEY，否则是旧/不兼容结构。
        # 幂等检查只看"在不在"不够——旧版本可能建了无名默认向量，查 using=dense 会 400。
        vectors = info.config.params.vectors
        if not (isinstance(vectors, dict) and DENSE_KEY in vectors):
            logger.warning(
                f"Collection '{collection_name}' has incompatible vector schema "
                f"(no named '{DENSE_KEY}' vector) — recreating for dual-index"
            )
            await client.delete_collection(collection_name)
            await _create()
        return QdrantVectorStore(client, collection_name)

    
    async def add(self, docs: list[Document], vectors: list[list[float]]) -> None:
        """批量写入--同时传dense vector + BM25文本"""
        points = []
        for d, v in zip(docs, vectors):
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        DENSE_KEY: v,
                        SPARSE_KEY: QdrantDocument(text=d.page_content, model='Qdrant/bm25'),
                    },
                    payload={'page_content': d.page_content, **d.metadata},
                )
            )
        await self._client.upsert(collection_name=self._collection, points=points)
        logger.debug(f"Added {len(points)} points to {self._collection}")


    async def search(self, query: str, top_k: int = 20) -> list[ScoredDoc]:
        """Hybrid Search -- dense + sparse -> RRF -> return

        Qdrant 内部：prefetch 两路各取 top_k*2 → RRF 融合 → 返回 top_k。
        客户端只收发一次请求。
        """
        embedder = get_embedder()
        query_vec = embedder.embed(query)

        response = await self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(query=query_vec, using=DENSE_KEY, limit=top_k * 2),
                Prefetch(query=QdrantDocument(text=query, model='Qdrant/bm25'), using=SPARSE_KEY, limit=top_k * 2),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True
        )

        scored = []
        for r in response.points:
            if r.payload:
                metadata = {k: v for k, v in r.payload.items() if k != 'page_content'}
                doc = Document(page_content=r.payload.get('page_content', ""), metadata=metadata)
                scored.append(ScoredDoc(doc=doc, score=r.score if r.score else 0))
        return scored

    async def close(self) -> None:
        """关闭 Qdrant 连接。"""
        await self._client.close()