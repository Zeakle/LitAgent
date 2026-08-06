"""Run strict live RAG profiles against benchmark collections."""

from __future__ import annotations

import asyncio
import hashlib
import platform
import statistics
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

import httpx

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.metrics import (
    aggregate_retrieval_cases,
    evaluate_retrieval_case,
)
from litagent.benchmark.models import (
    RAGBenchmarkDataset,
    RAGBenchmarkProfile,
    RAGBenchmarkResult,
    RetrievalCaseResult,
)
from litagent.config import AppConfig, RetrievalMode
from litagent.rag.chunking import build_corpus_chunker
from litagent.rag.corpus import CollectionIdentity
from litagent.rag.embedder import build_retrieval_embedder
from litagent.rag.ingest import (
    CorpusIngestor,
    ParsedAuditRepository,
    QuarantineRepository,
)
from litagent.rag.manifest import load_manifest, materialize_manifest_assets
from litagent.rag.pdf_parser import PyMuPDFParser
from litagent.rag.quality import CorpusTextQualityGate
from litagent.rag.reranker import CrossEncoderReranker
from litagent.rag.retriever import HybridRetriever
from litagent.rag.runtime import CorpusRuntime
from litagent.rag.sources import ArxivPDFAdapter, LocalPDFAdapter
from litagent.rag.vector_store import QdrantVectorStore


def _environment() -> dict[str, object]:
    packages = {}
    for distribution in (
        "qdrant-client",
        "sentence-transformers",
        "transformers",
        "torch",
        "adapters",
    ):
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = "not-installed"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "packages": packages,
    }


def _embedding_input_strategy(profile: RAGBenchmarkProfile) -> str:
    if profile.embedding_backend.value == "specter2":
        return "title_sep_chunk_text"
    return "chunk_text"


class RAGBenchmarkRunner:
    """Build compatible indexes and evaluate explicit profiles strictly."""

    def __init__(
        self,
        *,
        base_config: AppConfig,
        artifacts: BenchmarkArtifactRepository,
    ) -> None:
        self._base_config = base_config
        self._artifacts = artifacts

    def _config_for(self, profile: RAGBenchmarkProfile) -> AppConfig:
        rag = profile.apply(self._base_config.rag)
        return self._base_config.model_copy(update={"rag": rag}, deep=True)

    async def _ingest(
        self,
        *,
        config: AppConfig,
        runtime: CorpusRuntime,
        manifest_path: Path,
    ) -> None:
        async with httpx.AsyncClient() as client:
            parser = PyMuPDFParser(
                quality_gate=CorpusTextQualityGate(**config.rag.quality.model_dump()),
                chunker=build_corpus_chunker(config.rag),
            )
            ingestor = CorpusIngestor(
                config=config.rag,
                service=runtime.service,
                parser=parser,
                local_pdf_adapter=LocalPDFAdapter(
                    max_pdf_bytes=config.rag.max_pdf_bytes
                ),
                pdf_adapter=ArxivPDFAdapter(
                    client,
                    raw_root=Path(config.rag.raw_root),
                    max_pdf_bytes=config.rag.max_pdf_bytes,
                ),
                quarantine=QuarantineRepository(Path(config.rag.quarantine_root)),
                audit_repository=ParsedAuditRepository(Path(config.rag.parsed_root)),
            )
            summary = await ingestor.ingest_manifest(manifest_path, resume=True)
            if summary.status != "succeeded":
                raise RuntimeError("benchmark_ingestion_failed")

    @staticmethod
    def _failed_result(
        *,
        dataset: RAGBenchmarkDataset,
        profile: RAGBenchmarkProfile,
        identity: CollectionIdentity,
        git_sha: str,
        git_dirty: bool,
        manifest_hash: str,
        reason_code: str,
        run_id: str,
    ) -> RAGBenchmarkResult:
        return RAGBenchmarkResult(
            run_id=run_id,
            status="failed",
            dataset_id=dataset.dataset_id,
            dataset_version=dataset.dataset_version,
            dataset_fingerprint=dataset.fingerprint,
            judgment_status=dataset.judgment_status,
            manifest_hash=manifest_hash,
            profile_id=profile.profile_id,
            profile_fingerprint=profile.fingerprint,
            profile_config=profile.model_dump(mode="json"),
            collection_identity=identity.fingerprint,
            collection_config=asdict(identity),
            embedding_input_strategy=_embedding_input_strategy(profile),
            git_sha=git_sha,
            git_dirty=git_dirty,
            repetitions=profile.repetitions,
            reason_codes=[reason_code],
            environment=_environment(),
        )

    async def run(
        self,
        *,
        dataset: RAGBenchmarkDataset,
        profiles: Sequence[RAGBenchmarkProfile],
        git_sha: str,
        git_dirty: bool = False,
    ) -> list[RAGBenchmarkResult]:
        if not profiles:
            raise ValueError("retrieval benchmark requires profiles")
        profile_ids = [profile.profile_id for profile in profiles]
        fingerprints = [profile.fingerprint for profile in profiles]
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("profile_id values must be unique")
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("profiles contain duplicate behavior")

        manifest_path = Path(dataset.manifest_path)
        manifest_hash = (
            "sha256:"
            + hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        )
        manifest = load_manifest(
            manifest_path,
            raw_root=Path(self._base_config.rag.raw_root),
        )
        manifest_ids = {
            asset.paper_id for asset in materialize_manifest_assets(manifest)
        }
        if set(dataset.corpus_paper_ids) != manifest_ids:
            raise ValueError("dataset corpus_paper_ids do not match manifest")

        grouped: dict[str, list[RAGBenchmarkProfile]] = defaultdict(list)
        identities: dict[str, CollectionIdentity] = {}
        for profile in profiles:
            config = self._config_for(profile)
            identity = CollectionIdentity.from_config(
                config.rag,
                purpose="benchmark",
            )
            grouped[identity.fingerprint].append(profile)
            identities[identity.fingerprint] = identity

        results: list[RAGBenchmarkResult] = []
        for identity_key, group in grouped.items():
            identity = identities[identity_key]
            ingestion_config = self._config_for(group[0])
            runtime: CorpusRuntime | None = None
            index_started = time.perf_counter()
            try:
                runtime = await CorpusRuntime.connect(
                    ingestion_config,
                    purpose="benchmark",
                )
                if runtime.identity.fingerprint != identity.fingerprint:
                    raise RuntimeError("collection_identity_mismatch")
                await self._ingest(
                    config=ingestion_config,
                    runtime=runtime,
                    manifest_path=manifest_path,
                )
                index_elapsed_ms = int((time.perf_counter() - index_started) * 1000)
                stats = await runtime.store.stats()
                if stats.points_count <= 0:
                    raise RuntimeError("empty_benchmark_index")
                footprint_bytes = await runtime.store.estimate_logical_footprint_bytes()

                for profile in group:
                    run_id = f"rag-{profile.profile_id}-{uuid.uuid4().hex}"
                    try:
                        config = self._config_for(profile)
                        if profile.retrieval_mode is RetrievalMode.BM25:
                            # BM25 never creates or invokes a dense query encoder.
                            embedder = runtime.embedder
                        else:
                            embedder = build_retrieval_embedder(config.rag)
                            # Model loading is setup cost, not per-query latency.
                            await asyncio.to_thread(lambda: embedder.dim)
                        store = QdrantVectorStore(
                            runtime.qdrant_client,
                            identity.collection_name,
                            identity=identity,
                            embedder=embedder,
                        )
                        reranker = (
                            await asyncio.to_thread(
                                CrossEncoderReranker,
                                profile.reranker_model,
                            )
                            if profile.retrieval_mode is RetrievalMode.RRF_RERANK
                            else None
                        )
                        retriever = HybridRetriever(store, reranker)
                        cases: list[RetrievalCaseResult] = []
                        for judgment in dataset.queries:
                            latency_samples: list[int] = []
                            paper_ids: list[str] | None = None
                            for _ in range(profile.repetitions):
                                started = time.perf_counter()
                                papers = await retriever.search_papers(
                                    judgment.query,
                                    top_k=profile.top_k,
                                    candidate_k=profile.candidate_k,
                                    max_chunks_per_paper=(
                                        profile.max_representative_chunks
                                    ),
                                    mode=profile.retrieval_mode,
                                    strict=True,
                                )
                                latency_samples.append(
                                    int((time.perf_counter() - started) * 1000)
                                )
                                current_ids = [paper.paper_id for paper in papers]
                                if paper_ids is None:
                                    paper_ids = current_ids
                                elif current_ids != paper_ids:
                                    raise RuntimeError(
                                        "non_deterministic_retrieval_ranking"
                                    )
                            assert paper_ids is not None
                            cases.append(
                                RetrievalCaseResult(
                                    query_id=judgment.query_id,
                                    query=judgment.query,
                                    relevant_paper_ids=(judgment.relevant_paper_ids),
                                    retrieved_paper_ids=paper_ids,
                                    metrics=evaluate_retrieval_case(
                                        retrieved_paper_ids=paper_ids,
                                        relevant_paper_ids=set(
                                            judgment.relevant_paper_ids
                                        ),
                                    ),
                                    elapsed_ms=int(statistics.median(latency_samples)),
                                    latency_samples_ms=latency_samples,
                                )
                            )
                        result = RAGBenchmarkResult(
                            run_id=run_id,
                            status="succeeded",
                            dataset_id=dataset.dataset_id,
                            dataset_version=dataset.dataset_version,
                            dataset_fingerprint=dataset.fingerprint,
                            judgment_status=dataset.judgment_status,
                            manifest_hash=manifest_hash,
                            profile_id=profile.profile_id,
                            profile_fingerprint=profile.fingerprint,
                            profile_config=profile.model_dump(mode="json"),
                            collection_identity=identity.fingerprint,
                            collection_config=asdict(identity),
                            embedding_input_strategy=(
                                _embedding_input_strategy(profile)
                            ),
                            git_sha=git_sha,
                            git_dirty=git_dirty,
                            repetitions=profile.repetitions,
                            cases=cases,
                            summary=aggregate_retrieval_cases(cases),
                            index_elapsed_ms=index_elapsed_ms,
                            index_points_count=stats.points_count,
                            index_footprint_bytes=footprint_bytes,
                            environment={
                                **_environment(),
                                "index_footprint_kind": "logical_estimate",
                            },
                        )
                    except Exception as exc:
                        reason = str(exc) or type(exc).__name__
                        result = self._failed_result(
                            dataset=dataset,
                            profile=profile,
                            identity=identity,
                            git_sha=git_sha,
                            git_dirty=git_dirty,
                            manifest_hash=manifest_hash,
                            reason_code=reason[:128],
                            run_id=run_id,
                        )
                    try:
                        self._artifacts.write(result)
                    except Exception as write_exc:
                        # Disk-full or unsafe profile names must not escalate
                        # one write failure into a whole-round crash.
                        result = result.model_copy(
                            update={
                                "status": "failed",
                                "reason_codes": [
                                    *result.reason_codes,
                                    f"artifact_write_failed:"
                                    f"{type(write_exc).__name__}",
                                ],
                            }
                        )
                    results.append(result)
            except Exception as exc:
                reason = str(exc) or type(exc).__name__
                for profile in group:
                    run_id = f"rag-{profile.profile_id}-{uuid.uuid4().hex}"
                    result = self._failed_result(
                        dataset=dataset,
                        profile=profile,
                        identity=identity,
                        git_sha=git_sha,
                        git_dirty=git_dirty,
                        manifest_hash=manifest_hash,
                        reason_code=reason[:128],
                        run_id=run_id,
                    )
                    self._artifacts.write(result)
                    results.append(result)
            finally:
                if runtime is not None:
                    await runtime.close()
        return results
