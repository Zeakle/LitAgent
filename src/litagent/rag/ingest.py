"""Ingestion Pipeline——加载→切分→embed→写入 dual index。"""

from litagent.rag.interfaces import DocumentLoader, Chunker
from litagent.rag.vector_store import VectorStore
from litagent.rag.embedder import get_embedder
from litagent.logging import get_logger


logger = get_logger('rag.ingest')


async def ingest_papers(
    source: str,
    loader: DocumentLoader,
    chunker: Chunker,
    vector_store: VectorStore,
) -> int:
    """完整的 ingestion pipeline。

    Args:
        source: arxiv search query ("all:few-shot+learning")
        loader: DocumentLoader 实例
        chunker: Chunker 实例
        vector_store: VectorStore 实例（Qdrant dual-index）

    Returns:
        已索引的 chunk 数量
    """
    embedder = get_embedder()

    # 1. Load
    try:
        docs = await loader.load(source) 
    except Exception as e:
        logger.warning(f"Ingestion failed at load state: {e}")
        return 0
    logger.info(f"Loaded {len(docs)} papers from {source}")

    # 2. Chunk
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    logger.info(f"Chunked into {len(all_chunks)} chunks")

    # 3. Embed (dense only -- sparse BM25 tokenization Qdrant完成)
    texts = [c.page_content for c in all_chunks]
    vectors = embedder.embed(texts)

    # 4. Index (dense vector + BM25 text -> Qdrant 一次写入)
    await vector_store.add(all_chunks, vectors)

    logger.info(f"Ingestion complete: {len(all_chunks)} chunks indexed")
    return len(all_chunks)
