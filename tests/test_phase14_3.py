"""Phase 14.3 chunking, retrieval benchmark, and model-selection contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml


def _clean_block(
    *,
    text: str,
    page: int = 1,
    block_index: int = 0,
    section: str = "methods",
):
    from litagent.rag.quality import CleanTextBlock

    return CleanTextBlock(
        paper_id="arxiv:2401.00001",
        text=text,
        raw_content_hash=f"raw-{page}-{block_index}",
        transformations=["whitespace_normalized"],
        page=page,
        block_index=block_index,
        bbox=(0.0, float(block_index * 20), 100.0, float(block_index * 20 + 15)),
        section=section,
    )


def _rag_config(**overrides):
    from litagent.config import RAGConfig

    return RAGConfig(**overrides)


def _paper_record(
    *,
    title: str = "Few-shot Vision",
    page: int = 1,
    raw_content_hash: str = "raw-v1",
):
    from litagent.rag.models import (
        ChunkSourceSpan,
        ContentChunk,
        ContentScope,
        PaperRecord,
    )

    chunk = ContentChunk.from_text(
        paper_id="arxiv:2401.00001",
        chunk_key="page:1:block:0",
        text="Stable method evidence.",
        section="methods",
        content_scope=ContentScope.SELECTED_FULLTEXT,
        source_spans=[
            ChunkSourceSpan(
                page=page,
                block_index=0,
                bbox=(0.0, 10.0, 100.0, 30.0),
                start_char=0,
                end_char=len("Stable method evidence."),
                raw_content_hash=raw_content_hash,
            )
        ],
    )
    return PaperRecord(
        paper_id=chunk.paper_id,
        title=title,
        abstract="Stable abstract.",
        content_scope=ContentScope.SELECTED_FULLTEXT,
        chunks=[chunk],
        asset_hash="asset-v1",
    )


def test_phase14_3_default_config_exposes_real_behavior_switches():
    config = _rag_config()

    assert config.chunk_strategy.value == "page_block"
    assert config.chunk_size == 1200
    assert config.chunk_overlap == 150
    assert config.retrieval_mode.value == "rrf"
    assert config.embedding_backend.value == "sentence_transformer"
    assert config.reranker_model


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        _rag_config(chunk_size=400, chunk_overlap=400)


def test_specter2_config_requires_document_and_query_adapters():
    with pytest.raises(ValueError, match="adapter"):
        _rag_config(
            embedding_backend="specter2",
            embedding_model="allenai/specter2_base",
            embedding_document_adapter=None,
            embedding_query_adapter=None,
        )


def test_collection_identity_changes_for_every_chunking_behavior_field():
    from litagent.rag.corpus import CollectionIdentity

    baseline = CollectionIdentity.from_config(_rag_config())
    strategy = CollectionIdentity.from_config(_rag_config(chunk_strategy="recursive"))
    size = CollectionIdentity.from_config(_rag_config(chunk_size=800))
    overlap = CollectionIdentity.from_config(_rag_config(chunk_overlap=80))

    assert (
        len(
            {
                baseline.fingerprint,
                strategy.fingerprint,
                size.fingerprint,
                overlap.fingerprint,
            }
        )
        == 4
    )
    assert baseline.chunk_strategy == "page_block"
    assert baseline.chunk_size == 1200
    assert baseline.chunk_overlap == 150


def test_config_summary_fingerprints_chunking_and_retrieval_profile():
    from litagent.config import AppConfig, load_config
    from litagent.contracts import build_config_summary

    config = load_config().model_copy(deep=True)
    baseline = build_config_summary(config)
    values = config.model_dump(mode="json")
    values["rag"]["chunk_strategy"] = "section_aware"
    values["rag"]["retrieval_mode"] = "dense"
    changed = build_config_summary(AppConfig.model_validate(values))

    assert baseline["fingerprint"] != changed["fingerprint"]
    rag = changed["effective"]["rag"]
    assert rag["chunk_strategy"] == "section_aware"
    assert rag["retrieval_mode"] == "dense"


def test_chunk_source_span_serializes_exact_clean_text_offsets():
    from litagent.rag.models import ChunkSourceSpan

    span = ChunkSourceSpan(
        page=2,
        block_index=4,
        bbox=(1.0, 2.0, 3.0, 4.0),
        start_char=10,
        end_char=42,
        raw_content_hash="raw-hash",
    )

    assert span.model_dump(mode="json") == {
        "page": 2,
        "block_index": 4,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "start_char": 10,
        "end_char": 42,
        "raw_content_hash": "raw-hash",
    }


def test_page_block_chunker_preserves_one_to_one_locator_contract():
    from litagent.rag.chunking import PageBlockChunker

    blocks = [
        _clean_block(text="First block", block_index=0),
        _clean_block(text="Second block", block_index=1),
    ]

    chunks = PageBlockChunker().chunk(
        paper_id="arxiv:2401.00001",
        blocks=blocks,
    )

    assert [chunk.chunk_key for chunk in chunks] == [
        "page:1:block:0",
        "page:1:block:1",
    ]
    assert all(len(chunk.source_spans) == 1 for chunk in chunks)
    assert chunks[1].source_spans[0].block_index == 1
    assert chunks[1].source_spans[0].end_char == len("Second block")


def test_recursive_chunker_is_deterministic_bounded_and_overlap_aware():
    from litagent.rag.chunking import RecursiveChunker

    block = _clean_block(
        text=(
            "Few-shot learning compares support and query examples. "
            "Prototype methods aggregate class representations. "
            "Evaluation reports confidence intervals and repeated trials."
        )
    )
    chunker = RecursiveChunker(chunk_size=80, chunk_overlap=20)

    first = chunker.chunk(paper_id=block.paper_id, blocks=[block])
    second = chunker.chunk(paper_id=block.paper_id, blocks=[block])

    assert 2 <= len(first)
    assert [chunk.chunk_key for chunk in first] == [chunk.chunk_key for chunk in second]
    assert all(len(chunk.text) <= 80 for chunk in first)
    assert all(chunk.source_spans[0].page == 1 for chunk in first)
    assert first[0].source_spans[0].end_char > first[1].source_spans[0].start_char


def test_section_aware_chunker_never_merges_different_sections():
    from litagent.rag.chunking import SectionAwareChunker

    blocks = [
        _clean_block(text="Method block one.", block_index=0, section="methods"),
        _clean_block(text="Method block two.", block_index=1, section="methods"),
        _clean_block(text="Result block.", block_index=2, section="results"),
    ]

    chunks = SectionAwareChunker(chunk_size=120, chunk_overlap=20).chunk(
        paper_id="arxiv:2401.00001",
        blocks=blocks,
    )

    assert [chunk.section for chunk in chunks] == ["methods", "results"]
    assert len(chunks[0].source_spans) == 2
    assert {span.block_index for span in chunks[0].source_spans} == {0, 1}
    assert {span.block_index for span in chunks[1].source_spans} == {2}


def test_chunker_factory_consumes_the_effective_rag_config():
    from litagent.rag.chunking import RecursiveChunker, build_corpus_chunker

    chunker = build_corpus_chunker(
        _rag_config(
            chunk_strategy="recursive",
            chunk_size=640,
            chunk_overlap=64,
        )
    )

    assert isinstance(chunker, RecursiveChunker)
    assert chunker.chunk_size == 640
    assert chunker.chunk_overlap == 64


def test_parser_delegates_clean_blocks_to_the_configured_chunker(tmp_path):
    from litagent.rag.models import ContentChunk, ContentScope, RawPaperAsset
    from litagent.rag.pdf_parser import PyMuPDFParser

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nfixture")
    asset = RawPaperAsset(
        paper_id="arxiv:2401.00001",
        title="Few-shot Vision",
        abstract="Abstract evidence.",
        asset_hash="asset",
        pdf_path=pdf_path,
    )

    class _Page:
        rect = SimpleNamespace(width=100.0, height=100.0)

        def get_text(self, mode):
            assert mode == "blocks"
            return [(0.0, 10.0, 100.0, 30.0, "Method evidence.", 0, 0)]

        def get_images(self, full=True):
            return []

    class _Document:
        needs_pass = False
        is_encrypted = False

        def __iter__(self):
            return iter([_Page()])

        def __len__(self):
            return 1

        def close(self):
            return None

    expected = ContentChunk.from_text(
        paper_id=asset.paper_id,
        chunk_key="custom:0",
        text="Method evidence.",
        section="methods",
        content_scope=ContentScope.SELECTED_FULLTEXT,
        page=1,
        block_index=0,
        bbox=(0.0, 10.0, 100.0, 30.0),
    )
    chunker = SimpleNamespace(chunk=MagicMock(return_value=[expected]))

    parsed = PyMuPDFParser(
        opener=lambda _path: _Document(),
        chunker=chunker,
    ).parse_with_quality(asset)

    assert parsed.record is not None
    assert parsed.record.chunks[-1].chunk_key == "custom:0"
    chunker.chunk.assert_called_once()
    assert chunker.chunk.call_args.kwargs["paper_id"] == asset.paper_id
    assert chunker.chunk.call_args.kwargs["blocks"][0].text == "Method evidence."


def test_parser_does_not_index_pdf_abstract_twice(tmp_path):
    from litagent.rag.models import RawPaperAsset
    from litagent.rag.pdf_parser import PyMuPDFParser

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nfixture")
    asset = RawPaperAsset(
        paper_id="arxiv:2401.00001",
        title="Few-shot Vision",
        abstract="Same abstract text.",
        asset_hash="asset",
        pdf_path=pdf_path,
    )

    class _Page:
        rect = SimpleNamespace(width=100.0, height=100.0)

        def get_text(self, mode):
            assert mode == "blocks"
            return [(0.0, 10.0, 100.0, 30.0, "Same abstract text.", 0, 0)]

        def get_images(self, full=True):
            return []

    class _Document:
        needs_pass = False
        is_encrypted = False

        def __iter__(self):
            return iter([_Page()])

        def __len__(self):
            return 1

        def close(self):
            return None

    parsed = PyMuPDFParser(opener=lambda _path: _Document()).parse(asset)

    assert [chunk.chunk_key for chunk in parsed.chunks] == ["abstract"]


def test_specter2_title_change_reembeds_unchanged_chunk_text():
    from litagent.rag.corpus import CollectionIdentity, build_sync_plan
    from litagent.rag.state import PaperCorpusState

    identity = CollectionIdentity.from_config(
        _rag_config(
            embedding_backend="specter2",
            embedding_model="allenai/specter2_base",
            embedding_document_adapter="allenai/specter2",
            embedding_query_adapter="allenai/specter2_adhoc_query",
        )
    )
    initial = _paper_record()
    initial_plan = build_sync_plan(identity, initial, None)
    previous = PaperCorpusState.from_success(
        identity=identity,
        record=initial,
        point_ids=initial_plan.active_point_ids,
        batch_id="batch-1",
    )

    changed = build_sync_plan(
        identity,
        _paper_record(title="Renamed Few-shot Vision"),
        previous,
    )

    assert [chunk.chunk_key for chunk in changed.embed_chunks] == ["page:1:block:0"]
    assert changed.payload_only_chunks == []


def test_locator_change_refreshes_payload_without_reembedding_text():
    from litagent.rag.corpus import CollectionIdentity, build_sync_plan
    from litagent.rag.state import PaperCorpusState

    identity = CollectionIdentity.from_config(_rag_config())
    initial = _paper_record(page=1, raw_content_hash="raw-v1")
    initial_plan = build_sync_plan(identity, initial, None)
    previous = PaperCorpusState.from_success(
        identity=identity,
        record=initial,
        point_ids=initial_plan.active_point_ids,
        batch_id="batch-1",
    )

    changed = build_sync_plan(
        identity,
        _paper_record(page=2, raw_content_hash="raw-v2"),
        previous,
    )

    assert changed.embed_chunks == []
    assert [chunk.chunk_key for chunk in changed.payload_only_chunks] == [
        "page:1:block:0"
    ]


@pytest.mark.asyncio
async def test_corpus_state_repository_persists_incremental_hash_maps():
    from litagent.rag.corpus import CollectionIdentity, build_sync_plan
    from litagent.rag.state import CorpusStateRepository, PaperCorpusState

    identity = CollectionIdentity.from_config(_rag_config())
    record = _paper_record()
    plan = build_sync_plan(identity, record, None)
    state = PaperCorpusState.from_success(
        identity=identity,
        record=record,
        point_ids=plan.active_point_ids,
        batch_id="batch-1",
    )
    pool = SimpleNamespace(execute=AsyncMock())

    await CorpusStateRepository(pool).mark_succeeded(state)

    sql, *arguments = pool.execute.await_args.args
    assert "$18::jsonb" in sql
    assert "$19::jsonb" in sql
    assert json.loads(arguments[17]) == state.embedding_input_hashes
    assert json.loads(arguments[18]) == state.payload_hashes


def test_evidence_v2_copies_all_source_spans_from_the_real_chunk():
    from litagent.evidence import build_evidence_items
    from litagent.rag.models import ChunkSourceSpan, ContentChunk, ContentScope

    spans = [
        ChunkSourceSpan(
            page=2,
            block_index=4,
            bbox=(1.0, 2.0, 3.0, 4.0),
            start_char=0,
            end_char=20,
            raw_content_hash="raw-1",
        ),
        ChunkSourceSpan(
            page=2,
            block_index=5,
            bbox=(1.0, 5.0, 3.0, 7.0),
            start_char=0,
            end_char=18,
            raw_content_hash="raw-2",
        ),
    ]
    chunk = ContentChunk.from_text(
        paper_id="arxiv:2401.00001",
        chunk_key="section:methods:0",
        text="Combined method evidence.",
        section="methods",
        content_scope=ContentScope.SELECTED_FULLTEXT,
        page=2,
        block_index=4,
        # Legacy locator points at the first span; source_spans carries the
        # complete composite provenance without inventing a synthetic bbox.
        bbox=(1.0, 2.0, 3.0, 4.0),
        source_spans=spans,
    )

    items = build_evidence_items(
        {
            "paper_id": chunk.paper_id,
            "title": "Paper",
            "claim_records": [{"text": "Claim", "source_chunk_key": chunk.chunk_key}],
            "chunks": [chunk.model_dump(mode="json")],
        }
    )

    assert items[0]["source_spans"] == [span.model_dump(mode="json") for span in spans]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expects_dense"),
    [("bm25", False), ("dense", True), ("rrf", True)],
)
async def test_typed_qdrant_search_executes_the_selected_retrieval_mode(
    mode,
    expects_dense,
):
    from litagent.config import RetrievalMode
    from litagent.rag.vector_store import QdrantVectorStore

    client = SimpleNamespace(
        query_points=AsyncMock(return_value=SimpleNamespace(points=[]))
    )
    embed = MagicMock(return_value=[0.1, 0.2])
    embedder = SimpleNamespace(embed_query=embed)
    store = QdrantVectorStore(client, "papers", embedder=embedder)

    await store.search_chunks(
        "few-shot vision",
        10,
        mode=RetrievalMode(mode),
    )

    assert embed.called is expects_dense
    kwargs = client.query_points.await_args.kwargs
    if mode == "bm25":
        assert kwargs["using"] == "bm25_sparse"
        assert "prefetch" not in kwargs
    elif mode == "dense":
        assert kwargs["using"] == "dense"
        assert "prefetch" not in kwargs
    else:
        assert len(kwargs["prefetch"]) == 2
        assert "using" not in kwargs


@pytest.mark.asyncio
async def test_strict_parent_search_uses_exact_dense_candidates():
    from litagent.config import RetrievalMode
    from litagent.rag.retriever import HybridRetriever

    store = SimpleNamespace(search_chunks=AsyncMock(return_value=[]))
    retriever = HybridRetriever(store)

    await retriever.search_papers(
        "few-shot vision",
        top_k=10,
        candidate_k=40,
        mode=RetrievalMode.RRF,
        strict=True,
    )

    assert store.search_chunks.await_args.kwargs["exact"] is True
    assert store.search_chunks.await_args.args[1] == 40


@pytest.mark.asyncio
async def test_parent_search_preserves_a_chunk_budget_for_unique_papers():
    from litagent.config import RetrievalMode
    from litagent.rag.retriever import HybridRetriever

    store = SimpleNamespace(search_chunks=AsyncMock(return_value=[]))
    retriever = HybridRetriever(store)

    await retriever.search_papers(
        "few-shot vision",
        top_k=20,
        candidate_k=40,
        max_chunks_per_paper=4,
        mode=RetrievalMode.RRF,
    )

    assert store.search_chunks.await_args.args[1] == 80


@pytest.mark.asyncio
async def test_parent_search_expands_candidates_until_top_k_is_reached():
    """Widen a full-text window when many chunks map to one parent paper."""
    from litagent.config import RetrievalMode
    from litagent.rag.models import ContentChunk, ContentScope, ScoredChunkHit
    from litagent.rag.retriever import HybridRetriever

    def hit(paper_id: str, chunk_index: int, score: float) -> ScoredChunkHit:
        """Build one scored chunk for a parent paper."""
        chunk = ContentChunk.from_text(
            paper_id=paper_id,
            chunk_key=f"chunk:{chunk_index}",
            text=f"Evidence {chunk_index} for {paper_id}",
            section="abstract",
            content_scope=ContentScope.ABSTRACT,
        )
        return ScoredChunkHit(
            chunk=chunk,
            title=paper_id,
            score=score,
            collection="papers-benchmark",
            corpus_version="v1",
            schema_version="paper-v1",
            parser_version="pymupdf-v1",
            chunking_version="page-block-v1",
            embedding_model="all-MiniLM-L6-v2",
        )

    first_window = [hit("paper-a", index, 0.9) for index in range(8)]
    expanded_window = [*first_window, hit("paper-b", 0, 0.8)]
    store = SimpleNamespace(
        search_chunks=AsyncMock(side_effect=[first_window, expanded_window])
    )
    retriever = HybridRetriever(store)

    papers = await retriever.search_papers(
        "query",
        top_k=2,
        candidate_k=8,
        max_chunks_per_paper=4,
        mode=RetrievalMode.RRF,
        strict=True,
    )

    assert [paper.paper_id for paper in papers] == ["paper-a", "paper-b"]
    assert [call.args[1] for call in store.search_chunks.await_args_list] == [8, 16]


@pytest.mark.asyncio
async def test_parent_search_stops_expanding_when_store_is_exhausted():
    """Avoid another query after a short result proves the index is exhausted."""
    from litagent.config import RetrievalMode
    from litagent.rag.models import ContentChunk, ContentScope, ScoredChunkHit
    from litagent.rag.retriever import HybridRetriever

    hits = []
    for index in range(3):
        chunk = ContentChunk.from_text(
            paper_id="paper-a",
            chunk_key=f"chunk:{index}",
            text=f"Evidence {index}",
            section="abstract",
            content_scope=ContentScope.ABSTRACT,
        )
        hits.append(
            ScoredChunkHit(
                chunk=chunk,
                title="paper-a",
                score=0.9,
                collection="papers-benchmark",
                corpus_version="v1",
                schema_version="paper-v1",
                parser_version="pymupdf-v1",
                chunking_version="page-block-v1",
                embedding_model="all-MiniLM-L6-v2",
            )
        )
    store = SimpleNamespace(search_chunks=AsyncMock(return_value=hits))
    retriever = HybridRetriever(store)

    papers = await retriever.search_papers(
        "query",
        top_k=2,
        candidate_k=8,
        max_chunks_per_paper=4,
        mode=RetrievalMode.RRF,
        strict=True,
    )

    assert [paper.paper_id for paper in papers] == ["paper-a"]
    store.search_chunks.assert_awaited_once()


def test_retrieval_metrics_match_a_hand_checked_example():
    from litagent.benchmark.metrics import evaluate_retrieval_case

    metrics = evaluate_retrieval_case(
        retrieved_paper_ids=["p2", "p1", "p1", "p4"],
        relevant_paper_ids={"p1", "p3"},
        recall_ks=(1, 5, 10, 20),
    )

    assert metrics.recall_at_k == {1: 0.0, 5: 0.5, 10: 0.5, 20: 0.5}
    assert metrics.mrr_at_10 == pytest.approx(0.5)
    expected_ndcg = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert metrics.ndcg_at_10 == pytest.approx(expected_ndcg)
    assert metrics.duplicate_paper_ratio == pytest.approx(0.25)


def test_parent_paper_aggregation_breaks_equal_score_ties_deterministically():
    from litagent.rag.models import ContentChunk, ContentScope, ScoredChunkHit
    from litagent.rag.retriever import aggregate_chunk_hits

    def hit(paper_id: str, score: float) -> ScoredChunkHit:
        chunk = ContentChunk.from_text(
            paper_id=paper_id,
            chunk_key="abstract",
            text=f"Evidence for {paper_id}",
            section="abstract",
            content_scope=ContentScope.ABSTRACT,
        )
        return ScoredChunkHit(
            chunk=chunk,
            title=paper_id,
            score=score,
            collection="papers-benchmark",
            corpus_version="v1",
            schema_version="paper-v1",
            parser_version="pymupdf-v1",
            chunking_version="page-block-v1",
            embedding_model="all-MiniLM-L6-v2",
        )

    forward = aggregate_chunk_hits(
        [hit("paper:b", 0.25), hit("paper:a", 0.25)],
        top_k=2,
    )
    reversed_input = aggregate_chunk_hits(
        [hit("paper:a", 0.25), hit("paper:b", 0.25)],
        top_k=2,
    )

    assert [paper.paper_id for paper in forward] == ["paper:a", "paper:b"]
    assert [paper.paper_id for paper in reversed_input] == [
        "paper:a",
        "paper:b",
    ]


def test_cross_encoder_reranker_breaks_equal_score_ties_deterministically():
    from litagent.rag.models import ContentChunk, ContentScope, ScoredPaperHit
    from litagent.rag.reranker import CrossEncoderReranker

    def paper(paper_id: str) -> ScoredPaperHit:
        chunk = ContentChunk.from_text(
            paper_id=paper_id,
            chunk_key="abstract",
            text=f"Evidence for {paper_id}",
            section="abstract",
            content_scope=ContentScope.ABSTRACT,
        )
        return ScoredPaperHit(
            paper_id=paper_id,
            title=paper_id,
            content_scope=ContentScope.ABSTRACT,
            chunks=[chunk],
            score=0.25,
            collection="papers-benchmark",
            corpus_version="v1",
            schema_version="paper-v1",
            parser_version="pymupdf-v1",
            chunking_version="page-block-v1",
            embedding_model="all-MiniLM-L6-v2",
        )

    model = SimpleNamespace(predict=lambda pairs: [0.5] * len(pairs))
    reranker = CrossEncoderReranker(model=model)

    forward = reranker.rerank_papers("few-shot", [paper("paper:b"), paper("paper:a")])
    reversed_input = reranker.rerank_papers(
        "few-shot", [paper("paper:a"), paper("paper:b")]
    )

    assert [item.paper_id for item in forward] == ["paper:a", "paper:b"]
    assert [item.paper_id for item in reversed_input] == ["paper:a", "paper:b"]


def test_retrieval_metrics_reject_empty_ground_truth():
    from litagent.benchmark.metrics import evaluate_retrieval_case

    with pytest.raises(ValueError, match="relevant_paper_ids"):
        evaluate_retrieval_case(
            retrieved_paper_ids=["p1"],
            relevant_paper_ids=set(),
        )


def test_ingestion_metrics_enforce_reason_codes_and_metric_applicability():
    from litagent.benchmark.metrics import aggregate_ingestion_cases
    from litagent.benchmark.models import IngestionCaseObservation

    cases = [
        IngestionCaseObservation(
            case_id="indexed",
            expected_outcome="indexed",
            actual_outcome="indexed",
            expected_reason_codes=["expected_warning"],
            actual_reason_codes=[],
            metadata_correct=True,
            locator_preserved=True,
            duplicates_suppressed=None,
            incremental_update_correct=None,
            elapsed_ms=10,
        ),
        IngestionCaseObservation(
            case_id="failed",
            expected_outcome="failed",
            actual_outcome="failed",
            expected_reason_codes=["parser_failed"],
            actual_reason_codes=["parser_failed"],
            metadata_correct=None,
            locator_preserved=None,
            duplicates_suppressed=None,
            incremental_update_correct=None,
            elapsed_ms=20,
        ),
    ]

    result = aggregate_ingestion_cases(
        cases,
        run_id="ingestion-test",
        dataset_fingerprint="sha256:test",
    )

    assert result.status == "failed"
    assert result.summary.expected_outcome_accuracy == 1.0
    assert result.summary.reason_code_accuracy == 0.5
    assert result.summary.contract_accuracy == 0.5
    assert result.summary.metadata_accuracy == 1.0
    assert result.summary.locator_preservation == 1.0
    assert result.summary.metadata_case_count == 1
    assert result.reason_codes == ["case_contract_mismatch:indexed"]


@pytest.mark.asyncio
async def test_generated_version_update_executes_incremental_sync():
    from litagent.benchmark.generated_ingestion import (
        GeneratedIngestionCaseExecutor,
    )
    from litagent.benchmark.models import IngestionFixtureCase

    observation = await GeneratedIngestionCaseExecutor(_rag_config())(
        IngestionFixtureCase(
            case_id="version-update",
            fixture_type="version_update_pdf",
            expected_outcome="indexed",
        )
    )

    assert observation.actual_outcome == "indexed"
    assert observation.incremental_update_correct is True


def test_benchmark_profile_fingerprint_changes_for_every_experiment_dimension():
    from litagent.benchmark.models import RAGBenchmarkProfile

    baseline = RAGBenchmarkProfile(
        profile_id="baseline",
        content_mode="abstract",
        chunk_strategy="page_block",
        chunk_size=1200,
        chunk_overlap=150,
        embedding_backend="sentence_transformer",
        embedding_model="all-MiniLM-L6-v2",
        retrieval_mode="rrf",
    )

    def changed_profile(**updates):
        values = baseline.model_dump(mode="json")
        values.update(updates)
        return RAGBenchmarkProfile.model_validate(values)

    changed = [
        changed_profile(content_mode="abstract_and_selected_fulltext"),
        changed_profile(chunk_strategy="recursive"),
        changed_profile(chunk_size=800),
        changed_profile(embedding_model="candidate"),
        changed_profile(retrieval_mode="dense"),
        changed_profile(repetitions=5),
    ]

    assert all(item.fingerprint != baseline.fingerprint for item in changed)


def test_benchmark_dataset_rejects_duplicate_queries_and_unknown_papers():
    from litagent.benchmark.models import RAGBenchmarkDataset

    with pytest.raises(ValueError):
        RAGBenchmarkDataset.model_validate(
            {
                "schema_version": 1,
                "dataset_id": "cv-rag",
                "dataset_version": "v1",
                "corpus_paper_ids": ["p1"],
                "queries": [
                    {
                        "query_id": "q1",
                        "query": "few-shot",
                        "relevant_paper_ids": ["p1"],
                    },
                    {
                        "query_id": "q1",
                        "query": "duplicate id",
                        "relevant_paper_ids": ["missing"],
                    },
                ],
            }
        )


def test_versioned_benchmark_inputs_cover_required_ablation_dimensions():
    from litagent.benchmark.datasets import load_retrieval_dataset
    from litagent.rag.manifest import load_manifest, materialize_manifest_assets

    ingestion_path = Path("benchmarks/rag/ingestion_cases.yaml")
    dataset_path = Path("benchmarks/rag/retrieval_dataset.yaml")
    profiles_path = Path("benchmarks/rag/profiles.yaml")

    ingestion = yaml.safe_load(ingestion_path.read_text(encoding="utf-8"))
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    profiles = yaml.safe_load(profiles_path.read_text(encoding="utf-8"))["profiles"]
    dataset_model = load_retrieval_dataset(dataset_path)
    manifest = load_manifest(
        Path(dataset_model.manifest_path),
        raw_root=Path("artifacts/corpus/raw"),
    )

    assert 15 <= len(ingestion["cases"]) <= 20
    assert len(dataset_model.corpus_paper_ids) >= 25
    assert len(dataset["queries"]) >= 15
    assert dataset_model.judgment_status == "source_reviewed"
    assert set(dataset_model.corpus_paper_ids) == {
        asset.paper_id for asset in materialize_manifest_assets(manifest)
    }
    assert max(len(query.relevant_paper_ids) for query in dataset_model.queries) <= 5
    assert len(dataset_model.corpus_paper_ids) > 2 * max(
        len(query.relevant_paper_ids) for query in dataset_model.queries
    )
    assert {profile["chunk_strategy"] for profile in profiles} >= {
        "page_block",
        "recursive",
        "section_aware",
    }
    assert {profile["retrieval_mode"] for profile in profiles} >= {
        "bm25",
        "dense",
        "rrf",
        "rrf_rerank",
    }
    assert {profile["embedding_backend"] for profile in profiles} >= {
        "sentence_transformer",
        "specter2",
    }


def test_benchmark_artifact_repository_writes_reloadable_json_and_markdown(tmp_path):
    from litagent.benchmark.artifacts import BenchmarkArtifactRepository

    repository = BenchmarkArtifactRepository(tmp_path)
    payload = {
        "schema_version": 1,
        "run_id": "rag-bench-1",
        "dataset_fingerprint": "sha256:dataset",
        "profile_fingerprint": "sha256:profile",
        "collection_identity": "sha256:collection",
        "git_sha": "abc1234",
        "summary": {
            "recall_at_k": {"5": 0.5},
            "mrr_at_10": 0.5,
            "ndcg_at_10": 0.4,
        },
    }

    paths = repository.write(payload)

    assert json.loads(paths.json_path.read_text(encoding="utf-8")) == payload
    markdown = paths.markdown_path.read_text(encoding="utf-8")
    assert "rag-bench-1" in markdown
    assert "Recall@5" in markdown
    assert not list(tmp_path.glob("*.tmp"))


def test_cli_exposes_separate_ingestion_and_retrieval_benchmark_commands():
    from litagent.cli import build_parser

    parser = build_parser()
    ingestion = parser.parse_args(
        [
            "benchmark",
            "ingestion",
            "--dataset",
            "benchmarks/rag/ingestion_cases.yaml",
        ]
    )
    retrieval = parser.parse_args(
        [
            "benchmark",
            "retrieval",
            "--dataset",
            "benchmarks/rag/retrieval_dataset.yaml",
            "--profiles",
            "benchmarks/rag/profiles.yaml",
        ]
    )

    assert ingestion.benchmark_command == "ingestion"
    assert retrieval.benchmark_command == "retrieval"


@pytest.mark.asyncio
async def test_rag_runner_persists_complete_failure_artifact(tmp_path, monkeypatch):
    from litagent.benchmark.artifacts import BenchmarkArtifactRepository
    from litagent.benchmark.datasets import load_profiles, load_retrieval_dataset
    from litagent.benchmark.rag_runner import RAGBenchmarkRunner
    from litagent.config import load_config
    from litagent.rag.runtime import CorpusRuntime

    monkeypatch.setattr(
        CorpusRuntime,
        "connect",
        AsyncMock(side_effect=RuntimeError("benchmark_service_unavailable")),
    )
    dataset = load_retrieval_dataset(Path("benchmarks/rag/retrieval_dataset.yaml"))
    profile = load_profiles(Path("benchmarks/rag/profiles.yaml"))[0]
    results = await RAGBenchmarkRunner(
        base_config=load_config(),
        artifacts=BenchmarkArtifactRepository(tmp_path),
    ).run(
        dataset=dataset,
        profiles=[profile],
        git_sha="abc1234",
        git_dirty=True,
    )

    assert len(results) == 1
    result = results[0]
    assert result.status == "failed"
    assert result.dataset_version == "v3"
    assert result.judgment_status == "source_reviewed"
    assert result.manifest_hash.startswith("sha256:")
    assert result.profile_config["repetitions"] == 3
    assert result.collection_config["purpose"] == "benchmark"
    assert result.embedding_input_strategy == "chunk_text"
    assert result.git_dirty is True
    persisted = json.loads((tmp_path / f"{result.run_id}.json").read_text("utf-8"))
    assert persisted == result.model_dump(mode="json")
