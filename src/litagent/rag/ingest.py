"""Orchestrate document loading, chunking, embedding, and indexing."""

from litagent.rag.interfaces import DocumentLoader, Chunker
from litagent.rag.vector_store import VectorStore
from litagent.rag.embedder import get_embedder
from litagent.logging import get_logger

logger = get_logger("rag.ingest")


async def ingest_papers(
    source: str,
    loader: DocumentLoader,
    chunker: Chunker,
    vector_store: VectorStore,
) -> int:
    """Index paper chunks, returning zero when document loading fails."""
    embedder = get_embedder()

    try:
        docs = await loader.load(source)
    except Exception as e:
        logger.warning(f"Ingestion failed at load state: {e}")
        return 0
    logger.info(f"Loaded {len(docs)} papers from {source}")

    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
    logger.info(f"Chunked into {len(all_chunks)} chunks")

    texts = [c.page_content for c in all_chunks]
    vectors = embedder.embed(texts)

    await vector_store.add(all_chunks, vectors)

    logger.info(f"Ingestion complete: {len(all_chunks)} chunks indexed")
    return len(all_chunks)
