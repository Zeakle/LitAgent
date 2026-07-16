"""Regression coverage for the shared Qdrant papers collection lifecycle."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

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
