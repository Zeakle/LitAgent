"""Phase 14.1 paper-corpus contracts and incremental-indexing tests."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml


def _rag_config(**overrides):
    from litagent.config import RAGConfig

    return RAGConfig(**overrides)


def _asset(
    *,
    paper_id: str = "arxiv:2401.00001",
    title: str = "Few-shot Vision",
    abstract: str = "A reproducible abstract.",
    asset_hash: str = "asset-v1",
):
    from litagent.rag.models import RawPaperAsset, SourceRef

    return RawPaperAsset(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=["A. Author"],
        year=2024,
        asset_hash=asset_hash,
        sources=[
            SourceRef(
                kind="arxiv_metadata",
                source_id="2401.00001",
                license="arxiv",
            )
        ],
    )


def _record(
    *,
    title: str = "Few-shot Vision",
    texts: tuple[str, ...] = ("Abstract evidence.", "Method evidence."),
    keys: tuple[str, ...] = ("abstract", "page:1:block:0"),
):
    from litagent.rag.models import ContentChunk, ContentScope, PaperRecord

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
                page=None if is_abstract else index,
                block_index=None if is_abstract else 0,
                bbox=None if is_abstract else (0.0, 0.0, 100.0, 20.0),
            )
        )
    return PaperRecord(
        paper_id="arxiv:2401.00001",
        title=title,
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


class _FakePage:
    def __init__(self, blocks, images=None, width=800.0, height=1000.0):
        self._blocks = blocks
        self._images = images or []
        self.rect = SimpleNamespace(width=width, height=height)

    def get_text(self, mode):
        assert mode == "blocks"
        return self._blocks

    def get_images(self, full=True):
        assert full is True
        return self._images


class _FakePDF:
    needs_pass = False
    is_encrypted = False

    def __init__(self, pages):
        self._pages = pages
        self.closed = False

    def __iter__(self):
        return iter(self._pages)

    def __len__(self):
        return len(self._pages)

    def close(self):
        self.closed = True


class _StateRepositoryFake:
    def __init__(self, previous=None, events=None):
        self.previous = previous
        self.events = events if events is not None else []
        self.failed = None
        self.succeeded = None

    async def get_paper(self, collection_name, paper_id):
        return self.previous

    async def mark_running(self, collection_name, paper_id, batch_id):
        self.events.append("state.running")

    async def mark_succeeded(self, state):
        self.events.append("state.succeeded")
        self.previous = state
        self.succeeded = state

    async def mark_failed(self, collection_name, paper_id, batch_id, error_code):
        self.events.append("state.failed")
        self.failed = error_code


class _PaperIndexFake:
    def __init__(self, events=None, fail_delete=False):
        self.events = events if events is not None else []
        self.fail_delete = fail_delete
        self.writes = []
        self.payload_updates = []
        self.deletes = []

    async def upsert_chunks(self, writes):
        self.events.append("index.upsert")
        self.writes.extend(writes)

    async def update_payloads(self, updates):
        self.events.append("index.payload")
        self.payload_updates.extend(updates)

    async def delete_points(self, point_ids):
        self.events.append("index.delete")
        if self.fail_delete:
            raise RuntimeError("delete interrupted")
        self.deletes.extend(point_ids)


class _EmbedderFake:
    model_name = "test-embedding"

    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(index), 1.0] for index, _ in enumerate(texts)]

    @property
    def dim(self) -> int:
        return 2

    def embed_documents(self, documents):
        return self.embed([document.text for document in documents])

    def embed_query(self, query: str) -> list[float]:
        vector = self.embed([query])
        return vector[0]


def test_rag_config_defaults_are_safe_and_match_default_yaml():
    from litagent.config import load_config

    config = load_config()

    assert config.rag.enabled is True
    assert config.rag.writeback_enabled is False
    assert config.rag.content_mode == "abstract"
    assert config.rag.candidate_k >= config.rag.top_k
    assert config.rag.paper_collection != config.rag.benchmark_collection
    assert config.rag == _rag_config()


def test_collection_identity_changes_for_incompatible_versions():
    from litagent.rag.corpus import CollectionIdentity

    baseline = CollectionIdentity.from_config(_rag_config())
    changed_parser = CollectionIdentity.from_config(
        _rag_config(parser_version="pymupdf-v2")
    )
    benchmark = CollectionIdentity.from_config(_rag_config(), purpose="benchmark")

    assert baseline.collection_name != changed_parser.collection_name
    assert baseline.fingerprint != changed_parser.fingerprint
    assert baseline.collection_name != benchmark.collection_name
    assert baseline.embedding_model == "all-MiniLM-L6-v2"


def test_canonical_paper_id_removes_arxiv_version_and_normalizes_doi():
    from litagent.rag.models import canonical_paper_id

    assert canonical_paper_id(arxiv_id="2401.00001v3") == "arxiv:2401.00001"
    assert (
        canonical_paper_id(doi="https://doi.org/10.1000/ABC.Def")
        == "doi:10.1000/abc.def"
    )


def test_abstract_is_a_standard_chunk_without_pdf_locator():
    from litagent.rag.models import PaperRecord

    record = PaperRecord.from_abstract_asset(_asset())
    chunk = record.chunks[0]

    assert record.content_scope.value == "abstract"
    assert chunk.chunk_key == "abstract"
    assert chunk.section == "abstract"
    assert chunk.page is None
    assert chunk.block_index is None
    assert chunk.bbox is None
    assert chunk.content_hash == hashlib.sha256(chunk.text.encode()).hexdigest()


def test_api_and_pdf_records_merge_by_stable_paper_id():
    from litagent.rag.models import merge_paper_records

    abstract = _record(texts=("Abstract evidence.",), keys=("abstract",))
    fulltext = _record(
        texts=("Abstract evidence.", "Method evidence."),
        keys=("abstract", "page:1:block:0"),
    )

    merged = merge_paper_records([abstract, fulltext])

    assert len(merged) == 1
    assert merged[0].paper_id == "arxiv:2401.00001"
    assert [chunk.chunk_key for chunk in merged[0].chunks] == [
        "abstract",
        "page:1:block:0",
    ]
    assert merged[0].content_scope.value == "selected_fulltext"


def test_external_candidate_drops_provider_only_fields():
    from litagent.rag.models import PaperCandidate

    candidate = PaperCandidate.from_external(
        {
            "paper_id": "2401.00001v2",
            "title": "Few-shot Vision",
            "abstract": "Evidence.",
            "citation_count": 5,
            "source": "arxiv",
            "provider_debug_payload": {"secret": "must not cross boundary"},
        }
    )
    payload = candidate.to_dag_dict()

    assert payload["paper_id"] == "arxiv:2401.00001"
    assert payload["abstract"] == "Evidence."
    assert payload["content_scope"] == "abstract"
    assert "provider_debug_payload" not in payload


def test_manifest_loads_metadata_and_local_pdf_into_one_asset(tmp_path):
    from litagent.rag.manifest import load_manifest

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    pdf = raw_root / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.7\nfixture")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "corpus_version": "cv1",
                "papers": [
                    {
                        "paper_id": "arxiv:2401.00001",
                        "title": "Few-shot Vision",
                        "abstract": "Evidence.",
                        "arxiv_id": "2401.00001",
                        "license": "arxiv",
                        "pdf": {
                            "kind": "local_pdf",
                            "path": "paper.pdf",
                            "sha256": digest,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    manifest = load_manifest(manifest_path, raw_root=raw_root)
    assets = manifest.to_assets()

    assert manifest.corpus_version == "cv1"
    assert len(assets) == 1
    assert assets[0].paper_id == "arxiv:2401.00001"
    assert assets[0].pdf_path == pdf.resolve()
    assert {
        "arxiv_metadata",
        "local_pdf",
    } <= {source.kind.value for source in assets[0].sources}


@pytest.mark.parametrize(
    "pdf",
    [
        {"kind": "local_pdf", "path": "../outside.pdf"},
        {"kind": "arxiv_pdf", "url": "https://example.com/unsafe.pdf"},
    ],
)
def test_manifest_rejects_path_escape_and_non_arxiv_pdf_urls(tmp_path, pdf):
    from litagent.rag.manifest import ManifestValidationError, load_manifest

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "corpus_version": "cv1",
                "papers": [
                    {
                        "paper_id": "arxiv:2401.00001",
                        "title": "Unsafe",
                        "arxiv_id": "2401.00001",
                        "license": "arxiv",
                        "pdf": pdf,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ManifestValidationError):
        load_manifest(manifest_path, raw_root=raw_root)


@pytest.mark.asyncio
async def test_abstract_mode_does_not_read_download_or_parse_pdf(tmp_path):
    from litagent.rag.corpus import PaperSyncResult
    from litagent.rag.ingest import CorpusIngestor
    from litagent.rag.state import IngestionStatus

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "corpus_version": "cv1",
                "papers": [
                    {
                        "paper_id": "arxiv:2401.00001",
                        "title": "Few-shot Vision",
                        "abstract": "Evidence.",
                        "arxiv_id": "2401.00001",
                        "license": "arxiv",
                        "pdf": {
                            "kind": "local_pdf",
                            "path": "missing-but-unused.pdf",
                            "sha256": "0" * 64,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    service = SimpleNamespace(
        sync_record=AsyncMock(
            return_value=PaperSyncResult(
                paper_id="arxiv:2401.00001",
                status=IngestionStatus.SUCCEEDED,
                embedded_count=1,
            )
        ),
        prune_missing=AsyncMock(return_value=[]),
    )
    local_pdf_adapter = SimpleNamespace(materialize=AsyncMock())
    arxiv_pdf_adapter = SimpleNamespace(materialize=AsyncMock())
    parser = MagicMock()
    ingestor = CorpusIngestor(
        config=SimpleNamespace(
            corpus_version="cv1",
            raw_root=str(raw_root),
            content_mode="abstract",
        ),
        service=service,
        parser=parser,
        local_pdf_adapter=local_pdf_adapter,
        pdf_adapter=arxiv_pdf_adapter,
        quarantine=SimpleNamespace(record=AsyncMock()),
    )

    summary = await ingestor.ingest_manifest(manifest_path)

    assert summary.status == "succeeded"
    local_pdf_adapter.materialize.assert_not_awaited()
    arxiv_pdf_adapter.materialize.assert_not_awaited()
    parser.parse.assert_not_called()
    service.sync_record.assert_awaited_once()


def test_pymupdf_parser_emits_page_block_locators_and_closes_document(tmp_path):
    from litagent.rag.pdf_parser import PyMuPDFParser

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nfixture")
    asset = _asset()
    asset.pdf_path = pdf_path
    document = _FakePDF(
        [
            _FakePage(
                [
                    (0.0, 0.0, 100.0, 12.0, "Abstract", 0, 0),
                    (0.0, 20.0, 100.0, 60.0, "Evidence from the paper.", 1, 0),
                ]
            )
        ]
    )

    record = PyMuPDFParser(opener=lambda _: document).parse(asset)

    assert document.closed is True
    assert record.content_scope.value == "selected_fulltext"
    assert record.ocr_required is False
    assert any(
        chunk.page == 1
        and chunk.block_index == 1
        and chunk.bbox == (0.0, 20.0, 100.0, 60.0)
        and chunk.text == "Evidence from the paper."
        for chunk in record.chunks
    )


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (b"not a PDF", "not_pdf"),
        (b"%PDF-1.7\nfixture", "encrypted_pdf"),
    ],
)
def test_pymupdf_parser_preserves_stable_validation_reason_codes(
    tmp_path,
    payload,
    expected_code,
):
    from litagent.rag.pdf_parser import CorpusParseError, PyMuPDFParser

    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(payload)
    asset = _asset()
    asset.pdf_path = pdf_path
    document = _FakePDF([])
    document.is_encrypted = True

    with pytest.raises(CorpusParseError) as exc_info:
        PyMuPDFParser(opener=lambda _: document).parse(asset)

    assert exc_info.value.code == expected_code
    if expected_code == "not_pdf":
        assert document.closed is False
    else:
        assert document.closed is True


@pytest.mark.asyncio
async def test_arxiv_metadata_adapter_normalizes_loader_document():
    from langchain_core.documents import Document

    from litagent.rag.sources import ArxivMetadataAdapter

    loader = SimpleNamespace(
        load=AsyncMock(
            return_value=[
                Document(
                    page_content="Few-shot Vision\nEvidence abstract.",
                    metadata={
                        "title": "Few-shot Vision",
                        "arxiv_id": "2401.00001v2",
                    },
                )
            ]
        )
    )

    asset = await ArxivMetadataAdapter(loader).load("2401.00001")

    assert asset.paper_id == "arxiv:2401.00001"
    assert asset.abstract == "Evidence abstract."
    assert asset.arxiv_id == "2401.00001v2"


@pytest.mark.asyncio
async def test_arxiv_pdf_adapter_reuses_a_valid_local_cache(tmp_path):
    """Avoid a second network request for a verified cached arXiv PDF."""
    from litagent.rag.models import SourceRef
    from litagent.rag.sources import ArxivPDFAdapter

    payload = b"%PDF-1.7\ncached paper"
    target = tmp_path / "arxiv_2401.00001.pdf"
    target.write_bytes(payload)
    asset = _asset().model_copy(
        update={
            "pdf_url": "https://arxiv.org/pdf/2401.00001",
            "sources": [
                SourceRef(
                    kind="arxiv_pdf",
                    source_id="2401.00001",
                    uri="https://arxiv.org/pdf/2401.00001",
                    sha256=hashlib.sha256(payload).hexdigest(),
                )
            ],
        }
    )
    client = SimpleNamespace(stream=MagicMock(side_effect=AssertionError("network")))

    materialized = await ArxivPDFAdapter(
        client,
        raw_root=tmp_path,
        max_pdf_bytes=1024,
    ).materialize(asset)

    assert materialized.pdf_path == target.resolve()
    assert materialized.asset_hash == hashlib.sha256(payload).hexdigest()
    client.stream.assert_not_called()


@pytest.mark.asyncio
async def test_arxiv_pdf_adapter_does_not_reuse_checksum_mismatch(tmp_path):
    """Reject a cached PDF whose bytes disagree with the manifest checksum."""
    from litagent.rag.models import SourceRef
    from litagent.rag.sources import ArxivPDFAdapter

    target = tmp_path / "arxiv_2401.00001.pdf"
    target.write_bytes(b"%PDF-1.7\nstale paper")
    asset = _asset().model_copy(
        update={
            "pdf_url": "https://arxiv.org/pdf/2401.00001",
            "sources": [
                SourceRef(
                    kind="arxiv_pdf",
                    source_id="2401.00001",
                    uri="https://arxiv.org/pdf/2401.00001",
                    sha256=hashlib.sha256(b"%PDF-1.7\nexpected").hexdigest(),
                )
            ],
        }
    )
    client = SimpleNamespace(stream=MagicMock(side_effect=RuntimeError("download")))

    with pytest.raises(RuntimeError, match="download"):
        await ArxivPDFAdapter(
            client,
            raw_root=tmp_path,
            max_pdf_bytes=1024,
        ).materialize(asset)

    client.stream.assert_called_once()


def test_scanned_pdf_degrades_without_creating_empty_chunks(tmp_path):
    from litagent.rag.pdf_parser import PyMuPDFParser

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\nfixture")
    asset = _asset()
    asset.pdf_path = pdf_path
    document = _FakePDF([_FakePage([], images=[("image",)])])

    record = PyMuPDFParser(opener=lambda _: document).parse(asset)

    assert record.ocr_required is True
    assert all(chunk.text.strip() for chunk in record.chunks)
    assert any("ocr_required" in warning for warning in record.warnings)


def test_deterministic_point_id_is_stable_and_version_bound():
    from litagent.rag.corpus import CollectionIdentity, deterministic_point_id

    chunk = _record().chunks[0]
    first_identity = CollectionIdentity.from_config(_rag_config())
    second_identity = CollectionIdentity.from_config(
        _rag_config(embedding_model="candidate-model")
    )

    first = deterministic_point_id(first_identity, chunk)
    again = deterministic_point_id(first_identity, chunk)
    changed = deterministic_point_id(second_identity, chunk)

    assert first == again
    assert first != changed


def test_sync_plan_distinguishes_unchanged_metadata_text_and_stale_chunks():
    from litagent.rag.corpus import CollectionIdentity, build_sync_plan
    from litagent.rag.state import PaperCorpusState

    identity = CollectionIdentity.from_config(_rag_config())
    initial = _record()
    added = build_sync_plan(identity, initial, previous=None)
    previous = PaperCorpusState.from_success(
        identity=identity,
        record=initial,
        point_ids=added.active_point_ids,
        batch_id="batch-1",
    )

    unchanged = build_sync_plan(identity, initial, previous)
    metadata_only = build_sync_plan(
        identity,
        _record(title="Renamed Few-shot Vision"),
        previous,
    )
    text_changed = build_sync_plan(
        identity,
        _record(texts=("Changed abstract.", "Method evidence.")),
        previous,
    )
    removed = build_sync_plan(
        identity,
        _record(texts=("Abstract evidence.",), keys=("abstract",)),
        previous,
    )

    assert len(added.embed_chunks) == 2
    assert len(unchanged.unchanged_point_ids) == 2
    assert unchanged.embed_chunks == []
    assert len(metadata_only.payload_only_chunks) == 2
    assert metadata_only.embed_chunks == []
    assert [chunk.chunk_key for chunk in text_changed.embed_chunks] == ["abstract"]
    assert len(removed.stale_point_ids) == 1


def test_manifest_prune_plan_removes_only_absent_papers():
    from litagent.rag.corpus import build_manifest_prune_plan

    states = [
        SimpleNamespace(
            paper_id="arxiv:keep",
            active_point_ids=["keep-1"],
        ),
        SimpleNamespace(
            paper_id="arxiv:remove",
            active_point_ids=["remove-1", "remove-2"],
        ),
    ]

    plan = build_manifest_prune_plan(states, {"arxiv:keep"})

    assert plan == {"arxiv:remove": ["remove-1", "remove-2"]}


@pytest.mark.asyncio
async def test_corpus_service_commits_state_only_after_qdrant_operations():
    from litagent.rag.corpus import CollectionIdentity, CorpusService

    events = []
    state = _StateRepositoryFake(events=events)
    index = _PaperIndexFake(events=events)
    embedder = _EmbedderFake()
    service = CorpusService(
        identity=CollectionIdentity.from_config(_rag_config()),
        state_repository=state,
        paper_index=index,
        embedder=embedder,
    )

    result = await service.sync_record(_record(), batch_id="batch-1")

    assert result.status.value == "succeeded"
    assert events == ["state.running", "index.upsert", "state.succeeded"]
    assert len(index.writes) == 2
    assert embedder.calls == [["Abstract evidence.", "Method evidence."]]


@pytest.mark.asyncio
async def test_failed_stale_delete_preserves_previous_state_and_resume_converges():
    from litagent.rag.corpus import CollectionIdentity, CorpusService, build_sync_plan
    from litagent.rag.state import PaperCorpusState

    identity = CollectionIdentity.from_config(_rag_config())
    initial = _record()
    initial_plan = build_sync_plan(identity, initial, previous=None)
    previous = PaperCorpusState.from_success(
        identity=identity,
        record=initial,
        point_ids=initial_plan.active_point_ids,
        batch_id="batch-1",
    )
    state = _StateRepositoryFake(previous=previous)
    failing_index = _PaperIndexFake(fail_delete=True)
    embedder = _EmbedderFake()
    changed = _record(texts=("Abstract evidence.",), keys=("abstract",))

    failed = await CorpusService(
        identity=identity,
        state_repository=state,
        paper_index=failing_index,
        embedder=embedder,
    ).sync_record(changed, batch_id="batch-2")

    assert failed.status.value == "failed"
    assert state.previous == previous
    assert state.failed == "qdrant_delete_failed"

    resumed_index = _PaperIndexFake()
    resumed = await CorpusService(
        identity=identity,
        state_repository=state,
        paper_index=resumed_index,
        embedder=embedder,
    ).sync_record(changed, batch_id="batch-3")

    assert resumed.status.value == "succeeded"
    assert len(state.previous.active_point_ids) == 1
    assert len(resumed_index.deletes) == 1
    assert resumed_index.deletes[0] in previous.active_point_ids


@pytest.mark.asyncio
async def test_metadata_only_change_does_not_call_embedder():
    from litagent.rag.corpus import CollectionIdentity, CorpusService, build_sync_plan
    from litagent.rag.state import PaperCorpusState

    identity = CollectionIdentity.from_config(_rag_config())
    initial = _record()
    plan = build_sync_plan(identity, initial, previous=None)
    previous = PaperCorpusState.from_success(
        identity=identity,
        record=initial,
        point_ids=plan.active_point_ids,
        batch_id="batch-1",
    )
    state = _StateRepositoryFake(previous=previous)
    index = _PaperIndexFake()
    embedder = _EmbedderFake()

    result = await CorpusService(
        identity=identity,
        state_repository=state,
        paper_index=index,
        embedder=embedder,
    ).sync_record(_record(title="Updated title"), batch_id="batch-2")

    assert result.status.value == "succeeded"
    assert embedder.calls == []
    assert len(index.payload_updates) == 2


def test_postgres_schema_has_resumable_corpus_state_tables():
    schema = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "litagent"
        / "rag"
        / "corpus_schema.sql"
    ).read_text(encoding="utf-8")

    assert "corpus_ingestion_batches" in schema
    assert "corpus_paper_state" in schema
    for status in ("pending", "running", "succeeded", "failed"):
        assert status in schema
    assert "active_point_ids" in schema
    assert "chunk_hashes" in schema


@pytest.mark.asyncio
async def test_chunk_hits_aggregate_to_one_scored_paper():
    from litagent.rag.models import ScoredChunkHit
    from litagent.rag.retriever import aggregate_chunk_hits

    record = _record()
    hits = [
        ScoredChunkHit(
            chunk=record.chunks[0],
            title=record.title,
            authors=record.authors,
            year=record.year,
            score=0.9,
            collection="papers-cv1-test",
            corpus_version="cv1",
            schema_version="paper-v1",
            parser_version="pymupdf-v1",
            chunking_version="blocks-v1",
            embedding_model="all-MiniLM-L6-v2",
        ),
        ScoredChunkHit(
            chunk=record.chunks[1],
            title=record.title,
            authors=record.authors,
            year=record.year,
            score=0.8,
            collection="papers-cv1-test",
            corpus_version="cv1",
            schema_version="paper-v1",
            parser_version="pymupdf-v1",
            chunking_version="blocks-v1",
            embedding_model="all-MiniLM-L6-v2",
        ),
    ]

    papers = aggregate_chunk_hits(hits, top_k=5)

    assert len(papers) == 1
    assert papers[0].paper_id == record.paper_id
    assert papers[0].score == 0.9
    assert papers[0].abstract == "Abstract evidence."
    assert len(papers[0].chunks) == 2
    assert papers[0].content_scope.value == "selected_fulltext"


@pytest.mark.asyncio
async def test_recall_emits_versioned_paper_candidates_without_local_paths():
    from litagent.agents.recall import RecallWorker
    from litagent.orchestrator.task_graph import SubTask
    from litagent.rag.models import ContentScope, ScoredPaperHit

    record = _record()
    hit = ScoredPaperHit(
        paper_id=record.paper_id,
        title=record.title,
        abstract="Abstract evidence.",
        authors=record.authors,
        year=record.year,
        content_scope=ContentScope.SELECTED_FULLTEXT,
        chunks=record.chunks,
        score=0.9,
        collection="papers-cv1-test",
        corpus_version="cv1",
        schema_version="paper-v1",
        parser_version="pymupdf-v1",
        chunking_version="blocks-v1",
        embedding_model="all-MiniLM-L6-v2",
    )
    retriever = MagicMock()
    retriever.search_papers = AsyncMock(return_value=[hit])
    events = []
    worker = RecallWorker(
        retriever=retriever,
        trace_hook=lambda event, data: events.append((event, data)),
    )

    result = await worker.execute(
        SubTask(
            task_id="recall_q0",
            description="recall",
            agent_type="recall",
            input_data={"query": "few-shot vision", "top_k": 5},
        )
    )

    assert result[0]["paper_id"] == "arxiv:2401.00001"
    assert result[0]["content_scope"] == "selected_fulltext"
    assert result[0]["chunks"][1]["page"] == 1
    assert "pdf_path" not in str(result)
    complete = [data for event, data in events if event == "rag.recall.complete"][0]
    assert complete["collection"] == "papers-cv1-test"
    assert complete["corpus_version"] == "cv1"
    assert complete["result_count"] == 1


@pytest.mark.asyncio
async def test_recall_cancellation_closes_trace_and_propagates():
    from litagent.agents.recall import RecallWorker
    from litagent.orchestrator.task_graph import SubTask

    retriever = MagicMock()
    retriever.search_papers = AsyncMock(side_effect=asyncio.CancelledError())
    events = []
    worker = RecallWorker(
        retriever=retriever,
        trace_hook=lambda event, data: events.append((event, data)),
    )

    with pytest.raises(asyncio.CancelledError):
        await worker.execute(
            SubTask(
                task_id="recall_q0",
                description="recall",
                agent_type="recall",
                input_data={"query": "few-shot vision", "top_k": 5},
            )
        )

    assert [event for event, _ in events] == [
        "rag.recall.start",
        "rag.recall.failed",
    ]
    assert events[-1][1]["reason_code"] == "recall_cancelled"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exists", "points_count", "expected_status"),
    [
        (False, None, "not_found"),
        (True, 0, "empty"),
        (True, 3, "ready"),
    ],
)
async def test_corpus_stats_distinguishes_missing_empty_and_ready(
    exists,
    points_count,
    expected_status,
):
    from litagent.rag.corpus import CollectionIdentity
    from litagent.rag.vector_store import QdrantVectorStore

    identity = CollectionIdentity.from_config(_rag_config())
    client = MagicMock()
    client.collection_exists = AsyncMock(return_value=exists)
    client.get_collection = AsyncMock(
        return_value=SimpleNamespace(points_count=points_count)
    )
    store = QdrantVectorStore(
        client,
        identity.collection_name,
        identity=identity,
        embedder=MagicMock(),
    )

    stats = await store.stats()

    assert stats.status.value == expected_status
    assert stats.points_count == (points_count or 0)
    if not exists:
        client.get_collection.assert_not_awaited()


def test_corpus_cli_exposes_validate_ingest_resume_stats_rebuild_and_quarantine():
    from litagent.cli import build_parser

    parser = build_parser()

    validate = parser.parse_args(
        ["corpus", "validate", "--manifest", "corpus/manifest.yaml"]
    )
    ingest = parser.parse_args(
        ["corpus", "ingest", "--manifest", "corpus/manifest.yaml", "--resume"]
    )
    stats = parser.parse_args(["corpus", "stats"])
    rebuild = parser.parse_args(
        ["corpus", "rebuild", "--manifest", "corpus/manifest.yaml", "--yes"]
    )
    quarantine = parser.parse_args(["corpus", "quarantine", "list"])

    assert (validate.command, validate.corpus_command) == ("corpus", "validate")
    assert ingest.resume is True
    assert stats.corpus_command == "stats"
    assert rebuild.yes is True
    assert quarantine.quarantine_command == "list"


@pytest.mark.asyncio
async def test_writeback_is_best_effort_and_rejects_invalid_candidates():
    from litagent.rag.writeback import writeback_candidates

    service = SimpleNamespace(
        sync_candidates=AsyncMock(side_effect=RuntimeError("down"))
    )
    candidates = [
        {
            "paper_id": "2401.00001",
            "title": "Valid",
            "abstract": "Evidence.",
            "source": "arxiv",
        },
        {
            "paper_id": "",
            "title": "",
            "abstract": "",
            "source": "arxiv",
        },
    ]

    summary = await writeback_candidates(service, candidates)

    assert summary.status == "degraded"
    assert summary.accepted_count == 1
    assert summary.rejected_count == 1
    assert summary.reason_code == "writeback_failed"


@pytest.mark.asyncio
async def test_rebuild_recreates_collection_and_reingests_manifest(tmp_path):
    from litagent.cli import _cmd_corpus
    from litagent.config import load_config

    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")
    config = load_config().model_copy(deep=True)
    config.rag.raw_root = str(tmp_path / "raw")
    config.rag.quarantine_root = str(tmp_path / "quarantine")
    config.rag.parsed_root = str(tmp_path / "parsed")
    config.rag.quality.max_gibberish_ratio = 0.01

    class _AsyncClientContext:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, *_args):
            return False

    qdrant = SimpleNamespace(
        collection_exists=AsyncMock(return_value=True),
        delete_collection=AsyncMock(),
    )
    runtime = SimpleNamespace(
        identity=SimpleNamespace(collection_name="papers-versioned"),
        qdrant_client=qdrant,
        embedder=SimpleNamespace(dim=384),
        state=SimpleNamespace(reset_collection=AsyncMock()),
        service=MagicMock(),
        close=AsyncMock(),
    )
    ingestor = SimpleNamespace(
        ingest_manifest=AsyncMock(
            return_value=SimpleNamespace(
                status="succeeded",
                paper_count=1,
                succeeded_count=1,
                failed_count=0,
                embedded_count=1,
                payload_updated_count=0,
                deleted_count=0,
                unchanged_count=0,
                reason_codes=[],
                reports=[],
            )
        )
    )

    with (
        patch("litagent.config.load_config", return_value=config),
        patch(
            "litagent.rag.runtime.CorpusRuntime.connect",
            new=AsyncMock(return_value=runtime),
        ),
        patch(
            "litagent.rag.ingest.CorpusIngestor",
            return_value=ingestor,
        ) as ingestor_factory,
        patch("litagent.rag.quality.CorpusTextQualityGate") as quality_gate_factory,
        patch("litagent.rag.pdf_parser.PyMuPDFParser") as parser_factory,
        patch("litagent.rag.ingest.ParsedAuditRepository") as audit_factory,
        patch(
            "litagent.rag.vector_store.QdrantVectorStore.ensure_compatible",
            new=AsyncMock(),
        ) as ensure_compatible,
        patch("httpx.AsyncClient", return_value=_AsyncClientContext()),
    ):
        await _cmd_corpus(
            SimpleNamespace(
                corpus_command="rebuild",
                manifest=str(manifest_path),
                yes=True,
            )
        )

    qdrant.delete_collection.assert_awaited_once_with("papers-versioned")
    runtime.state.reset_collection.assert_awaited_once_with("papers-versioned")
    ensure_compatible.assert_awaited_once()
    ingestor.ingest_manifest.assert_awaited_once_with(
        manifest_path,
        resume=False,
    )
    quality_gate_factory.assert_called_once_with(**config.rag.quality.model_dump())
    parser_factory.assert_called_once()
    parser_kwargs = parser_factory.call_args.kwargs
    assert parser_kwargs["quality_gate"] is quality_gate_factory.return_value
    from litagent.rag.chunking import PageBlockChunker

    assert isinstance(parser_kwargs["chunker"], PageBlockChunker)
    audit_factory.assert_called_once_with(Path(config.rag.parsed_root))
    assert (
        ingestor_factory.call_args.kwargs["audit_repository"]
        is audit_factory.return_value
    )
    runtime.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_runner_reads_the_same_versioned_collection_written_by_corpus():
    from litagent.config import load_config
    from litagent.rag.corpus import CollectionIdentity
    from litagent.runner import LitAgent

    config = load_config()
    expected = CollectionIdentity.from_config(config.rag)
    agent = LitAgent(config)
    qdrant = MagicMock()
    qdrant.get_collection = AsyncMock(return_value=SimpleNamespace())
    qdrant.close = AsyncMock()
    store = MagicMock()
    paper_embedder = SimpleNamespace(dim=384)

    with (
        patch(
            "litagent.memory.working.WorkingMemory.connect",
            new=AsyncMock(side_effect=RuntimeError("redis unavailable")),
        ),
        patch("qdrant_client.AsyncQdrantClient", return_value=qdrant),
        patch.object(
            agent, "_get_embedding_dim_async", new=AsyncMock(return_value=384)
        ),
        patch.object(agent, "_create_reranker_async", new=AsyncMock(return_value=None)),
        patch("litagent.runner.build_retrieval_embedder", return_value=paper_embedder),
        patch(
            "litagent.runner.QdrantVectorStore.ensure_compatible",
            new=AsyncMock(return_value=store),
        ) as ensure_compatible,
        patch(
            "asyncpg.create_pool",
            new=AsyncMock(side_effect=RuntimeError("postgres unavailable")),
        ),
    ):
        infra = await agent._connect_infra(config)

    assert ensure_compatible.await_args.args[1] == expected.collection_name
    assert ensure_compatible.await_args.kwargs["identity"] == expected
    assert ensure_compatible.await_args.kwargs["embedder"] is paper_embedder
    assert infra.retriever._store is store
    await infra.close()


@pytest.mark.asyncio
async def test_corpus_runtime_closes_qdrant_when_postgres_connect_fails():
    from litagent.config import load_config
    from litagent.rag.runtime import CorpusRuntime

    config = load_config()
    qdrant = SimpleNamespace(close=AsyncMock())
    embedder = SimpleNamespace(dim=384)

    with (
        patch("litagent.rag.runtime.build_retrieval_embedder", return_value=embedder),
        patch("litagent.rag.runtime.AsyncQdrantClient", return_value=qdrant),
        patch(
            "litagent.rag.runtime.asyncpg.create_pool",
            new=AsyncMock(side_effect=RuntimeError("postgres unavailable")),
        ),
    ):
        with pytest.raises(RuntimeError, match="postgres unavailable"):
            await CorpusRuntime.connect(config)

    qdrant.close.assert_awaited_once()
