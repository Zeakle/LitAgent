"""Regression coverage for the shared Qdrant papers collection lifecycle."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from litagent.exceptions import ConfigError
from litagent.rag.vector_store import QdrantVectorStore


def _collection_info(points_count: int, vectors):
    return SimpleNamespace(
        points_count=points_count,
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)),
    )


@pytest.mark.asyncio
async def test_recreates_empty_legacy_papers_collection():
    client = SimpleNamespace(
        get_collection=AsyncMock(return_value=_collection_info(0, object())),
        delete_collection=AsyncMock(),
        create_collection=AsyncMock(),
    )

    store = await QdrantVectorStore.ensure_compatible(client, "papers", 1024)

    assert store._client is client
    client.delete_collection.assert_awaited_once_with("papers")
    client.create_collection.assert_awaited_once()


@pytest.mark.asyncio
async def test_never_deletes_nonempty_legacy_papers_collection():
    client = SimpleNamespace(
        get_collection=AsyncMock(return_value=_collection_info(3, object())),
        delete_collection=AsyncMock(),
        create_collection=AsyncMock(),
    )

    with pytest.raises(ConfigError, match="migration required"):
        await QdrantVectorStore.ensure_compatible(client, "papers", 1024)

    client.delete_collection.assert_not_awaited()
    client.create_collection.assert_not_awaited()


@pytest.mark.asyncio
async def test_new_collection_preserves_identity_and_configured_embedder():
    client = SimpleNamespace(
        get_collection=AsyncMock(side_effect=RuntimeError("missing")),
        create_collection=AsyncMock(),
    )
    identity = SimpleNamespace(fingerprint="sha256:test")
    embedder = MagicMock()

    store = await QdrantVectorStore.ensure_compatible(
        client,
        "papers-versioned",
        384,
        identity=identity,
        embedder=embedder,
    )

    assert store._identity is identity
    assert store._embedder is embedder


@pytest.mark.asyncio
async def test_upsert_chunks_uses_named_dense_vector():
    from litagent.rag.corpus import ChunkWrite
    from litagent.rag.models import ContentChunk, ContentScope

    client = SimpleNamespace(upsert=AsyncMock())
    store = QdrantVectorStore(client, "papers")
    chunk = ContentChunk.from_text(
        paper_id="arxiv:2401.00001",
        chunk_key="abstract",
        text="Evidence text.",
        section="abstract",
        content_scope=ContentScope.ABSTRACT,
    )

    await store.upsert_chunks(
        [
            ChunkWrite(
                point_id="3ecddb40-96c6-5c61-8977-83167e4c0c24",
                chunk=chunk,
                vector=[0.1, 0.2],
                payload={"chunk": chunk.model_dump(mode="json")},
            )
        ]
    )

    point = client.upsert.await_args.kwargs["points"][0]
    assert point.vector["dense"] == [0.1, 0.2]
