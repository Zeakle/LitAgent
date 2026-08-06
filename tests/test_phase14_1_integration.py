"""Isolated PostgreSQL/Qdrant integration tests for Phase 14.1."""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio


class _IntegrationEmbedder:
    """Provide deterministic vectors while exercising real storage backends."""

    dim = 3

    def embed(self, texts):
        if isinstance(texts, str):
            return [float(len(texts)), 1.0, 0.5]
        return [[float(len(text)), 1.0, 0.5] for text in texts]

    def embed_documents(self, documents):
        return self.embed([document.text for document in documents])

    def embed_query(self, query):
        return self.embed(query)


def _record(*, texts=("Abstract evidence.", "Method evidence."), keys=None):
    from litagent.rag.models import ContentChunk, ContentScope, PaperRecord

    keys = keys or ("abstract", "page:1:block:0")
    chunks = []
    for index, (key, text) in enumerate(zip(keys, texts, strict=True)):
        is_abstract = key == "abstract"
        chunks.append(
            ContentChunk.from_text(
                paper_id="arxiv:2401.00001",
                chunk_key=key,
                text=text,
                section="abstract" if is_abstract else "method",
                content_scope=(
                    ContentScope.ABSTRACT
                    if is_abstract
                    else ContentScope.SELECTED_FULLTEXT
                ),
                page=None if is_abstract else 1,
                block_index=None if is_abstract else index,
                bbox=None if is_abstract else (0.0, 0.0, 100.0, 20.0),
            )
        )
    return PaperRecord(
        paper_id="arxiv:2401.00001",
        title="Few-shot Vision",
        abstract=texts[0],
        authors=["A. Author"],
        year=2024,
        content_scope=(
            ContentScope.ABSTRACT
            if len(chunks) == 1
            else ContentScope.SELECTED_FULLTEXT
        ),
        chunks=chunks,
        asset_hash="asset-v1",
    )


@pytest_asyncio.fixture
async def corpus_runtime(monkeypatch):
    qdrant_url = os.getenv("TEST_QDRANT_URL")
    pg_url = os.getenv("TEST_PG_URL")
    if not qdrant_url or not pg_url:
        pytest.skip("TEST_QDRANT_URL and TEST_PG_URL are required")

    from litagent.config import MemoryConfig, RAGConfig, load_config
    from litagent.rag import runtime as runtime_module
    from litagent.rag.runtime import CorpusRuntime

    monkeypatch.setattr(
        runtime_module,
        "build_retrieval_embedder",
        lambda _config: _IntegrationEmbedder(),
    )

    suffix = uuid.uuid4().hex[:12]
    config = load_config().model_copy(
        update={
            "memory": MemoryConfig(qdrant_url=qdrant_url, pg_url=pg_url),
            "rag": RAGConfig(
                paper_collection=f"test_papers_{suffix}",
                benchmark_collection=f"test_benchmark_{suffix}",
                corpus_version=f"test-{suffix}",
                content_mode="abstract_and_selected_fulltext",
                embedding_model="all-MiniLM-L6-v2",
            ),
        },
        deep=True,
    )
    runtime = await CorpusRuntime.connect(config)
    try:
        yield runtime
    finally:
        try:
            await runtime.qdrant_client.delete_collection(
                runtime.identity.collection_name
            )
        finally:
            await runtime.state.reset_collection(runtime.identity.collection_name)
            await runtime.close()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_repeated_sync_is_idempotent_and_text_update_is_incremental(
    corpus_runtime,
):
    first = await corpus_runtime.service.sync_record(_record(), batch_id="batch-1")
    first_info = await corpus_runtime.qdrant_client.get_collection(
        corpus_runtime.identity.collection_name
    )
    repeated = await corpus_runtime.service.sync_record(
        _record(),
        batch_id="batch-2",
    )
    second_info = await corpus_runtime.qdrant_client.get_collection(
        corpus_runtime.identity.collection_name
    )
    changed = await corpus_runtime.service.sync_record(
        _record(texts=("Changed abstract.", "Method evidence.")),
        batch_id="batch-3",
    )

    assert first.embedded_count == 2
    assert repeated.embedded_count == 0
    assert first_info.points_count == second_info.points_count == 2
    assert changed.embedded_count == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stale_chunk_is_deleted_and_committed_state_matches_index(
    corpus_runtime,
):
    await corpus_runtime.service.sync_record(_record(), batch_id="batch-1")
    result = await corpus_runtime.service.sync_record(
        _record(
            texts=("Abstract evidence.",),
            keys=("abstract",),
        ),
        batch_id="batch-2",
    )
    info = await corpus_runtime.qdrant_client.get_collection(
        corpus_runtime.identity.collection_name
    )
    state = await corpus_runtime.state.get_paper(
        corpus_runtime.identity.collection_name,
        "arxiv:2401.00001",
    )

    assert result.deleted_count == 1
    assert info.points_count == 1
    assert state.status.value == "succeeded"
    assert len(state.active_point_ids) == 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_interrupted_delete_keeps_previous_state_and_resume_converges(
    corpus_runtime,
    monkeypatch,
):
    await corpus_runtime.service.sync_record(_record(), batch_id="batch-1")
    original_delete = corpus_runtime.store.delete_points

    async def fail_delete(_point_ids):
        raise RuntimeError("injected delete failure")

    monkeypatch.setattr(corpus_runtime.store, "delete_points", fail_delete)
    failed = await corpus_runtime.service.sync_record(
        _record(
            texts=("Abstract evidence.",),
            keys=("abstract",),
        ),
        batch_id="batch-2",
    )
    failed_state = await corpus_runtime.state.get_paper(
        corpus_runtime.identity.collection_name,
        "arxiv:2401.00001",
    )

    monkeypatch.setattr(corpus_runtime.store, "delete_points", original_delete)
    resumed = await corpus_runtime.service.sync_record(
        _record(
            texts=("Abstract evidence.",),
            keys=("abstract",),
        ),
        batch_id="batch-3",
    )
    resumed_state = await corpus_runtime.state.get_paper(
        corpus_runtime.identity.collection_name,
        "arxiv:2401.00001",
    )

    assert failed.reason_code == "qdrant_delete_failed"
    assert len(failed_state.active_point_ids) == 2
    assert resumed.status.value == "succeeded"
    assert len(resumed_state.active_point_ids) == 1
