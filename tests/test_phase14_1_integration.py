"""Isolated PostgreSQL/Qdrant integration tests for Phase 14.1."""

from __future__ import annotations

import os
import uuid

import pytest
import pytest_asyncio

_TEST_COLLECTION_PREFIX = "test_papers_"


def _isolated_rag_config(tmp_path, suffix):
    """Build one fixture-owned corpus namespace and filesystem layout."""
    from litagent.config import RAGConfig

    return RAGConfig(
        paper_collection=f"{_TEST_COLLECTION_PREFIX}{suffix}",
        benchmark_collection=f"test_benchmark_{suffix}",
        corpus_version=f"test-{suffix}",
        content_mode="abstract_and_selected_fulltext",
        embedding_model="all-MiniLM-L6-v2",
        manifest_path=str(tmp_path / suffix / "manifest.yaml"),
        raw_root=str(tmp_path / suffix / "raw"),
        parsed_root=str(tmp_path / suffix / "parsed"),
        quarantine_root=str(tmp_path / suffix / "quarantine"),
    )


def _assert_fixture_owned_collection(collection_name):
    """Fail closed before an integration teardown can touch shared storage."""
    if not collection_name.startswith(_TEST_COLLECTION_PREFIX):
        raise AssertionError(
            f"refusing to clean non-test collection: {collection_name!r}"
        )


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
async def corpus_runtime(monkeypatch, tmp_path):
    qdrant_url = os.getenv("TEST_QDRANT_URL")
    pg_url = os.getenv("TEST_PG_URL")
    if not qdrant_url or not pg_url:
        pytest.skip("TEST_QDRANT_URL and TEST_PG_URL are required")

    from litagent.config import MemoryConfig, load_config
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
            "rag": _isolated_rag_config(tmp_path, suffix),
        },
        deep=True,
    )
    runtime = await CorpusRuntime.connect(config)
    try:
        yield runtime
    finally:
        collection_name = runtime.identity.collection_name
        _assert_fixture_owned_collection(collection_name)
        try:
            await runtime.qdrant_client.delete_collection(collection_name)
        finally:
            _assert_fixture_owned_collection(collection_name)
            await runtime.state.reset_collection(collection_name)
            await runtime.close()


def test_integration_fixture_uses_unique_storage_namespaces(tmp_path):
    from litagent.rag.corpus import CollectionIdentity

    first = _isolated_rag_config(tmp_path, "first")
    second = _isolated_rag_config(tmp_path, "second")
    first_identity = CollectionIdentity.from_config(first)
    second_identity = CollectionIdentity.from_config(second)

    assert first_identity.collection_name != second_identity.collection_name
    assert first.corpus_version != second.corpus_version
    assert first.raw_root != second.raw_root
    for configured_path in (
        first.manifest_path,
        first.raw_root,
        first.parsed_root,
        first.quarantine_root,
    ):
        assert str(tmp_path) in configured_path


@pytest.mark.parametrize("collection_name", ["papers", "claims", "papers-v1"])
def test_integration_cleanup_refuses_default_collection_or_corpus_paths(
    collection_name,
):
    with pytest.raises(AssertionError, match="refusing to clean non-test collection"):
        _assert_fixture_owned_collection(collection_name)


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
