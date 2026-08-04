"""Plan and execute versioned paper-corpus synchronization."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Protocol, Sequence

from litagent.config import RAGConfig
from litagent.rag.models import ContentChunk, PaperCandidate, PaperRecord
from litagent.rag.state import IngestionStatus, PaperCorpusState

_POINT_NAMESPACE = uuid.UUID("3ecddb40-96c6-5c61-8977-83167e4c0c24")


@dataclass(frozen=True)
class CollectionIdentity:
    """Bind one Qdrant collection to all vector-compatible versions."""

    collection_name: str
    fingerprint: str
    purpose: Literal["runtime", "benchmark"]
    corpus_version: str
    schema_version: str
    parser_version: str
    chunking_version: str
    embedding_model: str
    content_mode: str

    @classmethod
    def from_config(
        cls, config: RAGConfig, *, purpose: Literal["runtime", "benchmark"] = "runtime"
    ) -> "CollectionIdentity":
        """Derive a stable collection name from vector-compatible settings."""
        values = {
            "corpus_version": config.corpus_version,
            "schema_version": config.schema_version,
            "parser_version": config.parser_version,
            "chunking_version": config.chunking_version,
            "embedding_model": config.embedding_model,
            "content_mode": config.content_mode,
        }
        canonical = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fingerprint = hashlib.sha256(canonical).hexdigest()
        base = (
            config.paper_collection
            if purpose == "runtime"
            else config.benchmark_collection
        )
        safe_base = re.sub(r"[^A-Za-z0-9_-]+", "-", base).strip("-_")
        safe_corpus = re.sub(
            r"[^A-Za-z0-9_-]+",
            "-",
            config.corpus_version,
        ).strip("-_")

        return cls(
            collection_name=f"{safe_base}-{safe_corpus}-{fingerprint[:12]}",
            fingerprint=f"sha256:{fingerprint}",
            purpose=purpose,
            **values,
        )


def deterministic_point_id(identity: CollectionIdentity, chunk: ContentChunk) -> str:
    """Return a Qdrant-compatible UUID bound to version/paper/chunk identity."""
    value = f"{identity.fingerprint}|{chunk.paper_id}|{chunk.chunk_key}"
    return str(uuid.uuid5(_POINT_NAMESPACE, value))


@dataclass(frozen=True)
class PaperSyncPlan:
    """Describe side effects needed to converge one paper"""

    active_point_ids: list[str]
    embed_chunks: list[ContentChunk] = field(default_factory=list)
    payload_only_chunks: list[ContentChunk] = field(default_factory=list)
    stale_point_ids: list[str] = field(default_factory=list)
    unchanged_point_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkWrite:
    """Pair one deterministic point with content, payload, and dense vector."""

    point_id: str
    chunk: ContentChunk
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class PayloadUpdate:
    """Update metadata without recomputing dense or sparse content."""

    point_id: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class PaperSyncResult:
    """Report one paper-level ingestion outcome."""

    paper_id: str
    status: IngestionStatus
    embedded_count: int = 0
    payload_updated_count: int = 0
    deleted_count: int = 0
    unchanged_count: int = 0
    reason_code: str | None = None


class CorpusStatsStatus(str, Enum):
    """Distinguish missing storage from an existing empty corpus."""

    NOT_FOUND = "not_found"
    EMPTY = "empty"
    READY = "ready"


@dataclass(frozen=True)
class CorpusStats:
    """Report collection identity and current indexed point count."""

    status: CorpusStatsStatus
    collection_name: str
    points_count: int
    identity_fingerprint: str


def build_sync_plan(
    identity: CollectionIdentity,
    record: PaperRecord,
    previous: PaperCorpusState | None,
) -> PaperSyncPlan:
    """Diff deterministic ids and hashes without performing I/O."""
    current = {
        deterministic_point_id(identity, chunk): chunk for chunk in record.chunks
    }
    if previous is None or previous.collection_name != identity.collection_name:
        return PaperSyncPlan(
            active_point_ids=list(current),
            embed_chunks=list(current.values()),
        )

    embed_chunks: list[ContentChunk] = []
    payload_only: list[ContentChunk] = []
    unchanged: list[str] = []
    metadata_changed = previous.metadata_hash != record.metadata_hash
    for point_id, chunk in current.items():
        old_hash = previous.chunk_hashes.get(point_id)
        if old_hash != chunk.content_hash:
            embed_chunks.append(chunk)
        elif metadata_changed:
            payload_only.append(chunk)
        else:
            unchanged.append(point_id)

    return PaperSyncPlan(
        active_point_ids=list(current),
        embed_chunks=embed_chunks,
        payload_only_chunks=payload_only,
        stale_point_ids=sorted(set(previous.active_point_ids) - set(current)),
        unchanged_point_ids=unchanged,
    )


def build_manifest_prune_plan(
    states: Sequence[PaperCorpusState],
    manifest_paper_ids: set[str],
) -> dict[str, list[str]]:
    """Return point ids owned by papers removed from the manifest."""
    return {
        state.paper_id: list(state.active_point_ids)
        for state in states
        if state.paper_id not in manifest_paper_ids
    }


class PaperIndex(Protocol):
    """Define side effects required from the Qdrant adapter."""

    async def upsert_chunks(self, writes: Sequence[ChunkWrite]) -> None: ...

    async def update_payloads(
        self,
        updates: Sequence[PayloadUpdate],
    ) -> None: ...

    async def delete_points(self, point_ids: Sequence[str]) -> None: ...


class CorpusService:
    """Converge paper records while keeping PostgreSQL state resumable."""

    def __init__(
        self,
        *,
        identity: CollectionIdentity,
        state_repository,
        paper_index: PaperIndex,
        embedder,
    ) -> None:
        self.identity = identity
        self._state = state_repository
        self._index = paper_index
        self._embedder = embedder

    async def _mark_failed(
        self,
        paper_id: str,
        batch_id: str,
        reason_code: str,
    ) -> None:
        """Best-effort failure bookkeeping without masking the root error."""
        try:
            await self._state.mark_failed(
                self.identity.collection_name,
                paper_id,
                batch_id,
                reason_code,
            )
        except Exception:
            # PostgreSQL may be the failing dependency. The caller still needs
            # the original stage classification, not a second bookkeeping error.
            pass

    def _payload(
        self,
        record: PaperRecord,
        chunk: ContentChunk,
    ) -> dict[str, Any]:
        """Project only retrieval-safe metadata into Qdrant."""
        return {
            "paper_id": record.paper_id,
            "title": record.title,
            "abstract": record.abstract,
            "authors": record.authors,
            "year": record.year,
            "content_scope": record.content_scope.value,
            "chunk": chunk.model_dump(mode="json"),
            "sources": [source.model_dump(mode="json") for source in record.sources],
            "warnings": record.warnings,
            "ocr_required": record.ocr_required,
            "collection": self.identity.collection_name,
            "corpus_version": self.identity.corpus_version,
            "schema_version": self.identity.schema_version,
            "parser_version": self.identity.parser_version,
            "chunking_version": self.identity.chunking_version,
            "embedding_model": self.identity.embedding_model,
        }

    async def sync_record(
        self,
        record: PaperRecord,
        *,
        batch_id: str,
    ) -> PaperSyncResult:
        """Apply one idempotent paper diff and commit state last."""
        collection = self.identity.collection_name
        previous = await self._state.get_paper(collection, record.paper_id)
        await self._state.mark_running(collection, record.paper_id, batch_id)
        plan = build_sync_plan(self.identity, record, previous)
        stage = "planning"
        try:
            writes: list[ChunkWrite] = []
            if plan.embed_chunks:
                stage = "embedding"
                vectors = await asyncio.to_thread(
                    self._embedder.embed,
                    [chunk.text for chunk in plan.embed_chunks],
                )
                if len(vectors) != len(plan.embed_chunks) or any(
                    not vector for vector in vectors
                ):
                    raise ValueError("embedder returned empty/misaligned vectors")
                writes = [
                    ChunkWrite(
                        point_id=deterministic_point_id(self.identity, chunk),
                        chunk=chunk,
                        vector=vector,
                        payload=self._payload(record, chunk),
                    )
                    for chunk, vector in zip(
                        plan.embed_chunks,
                        vectors,
                        strict=True,
                    )
                ]
                stage = "qdrant_upsert"
                await self._index.upsert_chunks(writes)

            if plan.payload_only_chunks:
                stage = "qdrant_payload"
                await self._index.update_payloads(
                    [
                        PayloadUpdate(
                            point_id=deterministic_point_id(
                                self.identity,
                                chunk,
                            ),
                            payload=self._payload(record, chunk),
                        )
                        for chunk in plan.payload_only_chunks
                    ]
                )

            if plan.stale_point_ids:
                stage = "qdrant_delete"
                await self._index.delete_points(plan.stale_point_ids)

            stage = "state_commit"
            await self._state.mark_succeeded(
                PaperCorpusState.from_success(
                    identity=self.identity,
                    record=record,
                    point_ids=plan.active_point_ids,
                    batch_id=batch_id,
                )
            )
            return PaperSyncResult(
                paper_id=record.paper_id,
                status=IngestionStatus.SUCCEEDED,
                embedded_count=len(plan.embed_chunks),
                payload_updated_count=len(plan.payload_only_chunks),
                deleted_count=len(plan.stale_point_ids),
                unchanged_count=len(plan.unchanged_point_ids),
            )
        except asyncio.CancelledError:
            await self._mark_failed(
                record.paper_id,
                batch_id,
                "ingestion_cancelled",
            )
            raise
        except Exception:
            reason = {
                "embedding": "embedding_failed",
                "qdrant_upsert": "qdrant_upsert_failed",
                "qdrant_payload": "qdrant_payload_failed",
                "qdrant_delete": "qdrant_delete_failed",
                "state_commit": "state_commit_failed",
            }.get(stage, "ingestion_failed")
            await self._mark_failed(record.paper_id, batch_id, reason)
            return PaperSyncResult(
                paper_id=record.paper_id,
                status=IngestionStatus.FAILED,
                reason_code=reason,
            )

    async def sync_candidates(
        self,
        candidates: Sequence[PaperCandidate],
        *,
        batch_id: str | None = None,
    ) -> list[PaperSyncResult]:
        """Convert validated abstract candidates into idempotent records."""
        effective_batch = batch_id or f"writeback-{uuid.uuid4().hex}"
        results = []
        for candidate in candidates:
            asset_hash = hashlib.sha256(
                f"{candidate.title}\n{candidate.abstract}".encode("utf-8")
            ).hexdigest()
            record = PaperRecord(
                paper_id=candidate.paper_id,
                title=candidate.title,
                abstract=candidate.abstract,
                authors=candidate.authors,
                year=candidate.year,
                content_scope=candidate.content_scope,
                chunks=candidate.chunks,
                asset_hash=asset_hash,
            )
            results.append(await self.sync_record(record, batch_id=effective_batch))
        return results

    async def prune_missing(
        self,
        manifest_paper_ids: set[str],
        *,
        batch_id: str,
    ) -> list[PaperSyncResult]:
        """Delete points/state for papers removed from the source manifest."""
        states = await self._state.list_papers(self.identity.collection_name)
        plan = build_manifest_prune_plan(states, manifest_paper_ids)
        results = []
        for state in states:
            if state.paper_id not in plan:
                continue
            await self._state.mark_running(
                self.identity.collection_name,
                state.paper_id,
                batch_id,
            )
            try:
                await self._index.delete_points(plan[state.paper_id])
                await self._state.delete_paper(
                    self.identity.collection_name,
                    state.paper_id,
                )
                results.append(
                    PaperSyncResult(
                        paper_id=state.paper_id,
                        status=IngestionStatus.SUCCEEDED,
                        deleted_count=len(plan[state.paper_id]),
                    )
                )
            except asyncio.CancelledError:
                await self._mark_failed(
                    state.paper_id,
                    batch_id,
                    "ingestion_cancelled",
                )
                raise
            except Exception:
                # The last committed point ids remain available for resume.
                await self._mark_failed(
                    state.paper_id,
                    batch_id,
                    "qdrant_delete_failed",
                )
                results.append(
                    PaperSyncResult(
                        paper_id=state.paper_id,
                        status=IngestionStatus.FAILED,
                        reason_code="qdrant_delete_failed",
                    )
                )
        return results
