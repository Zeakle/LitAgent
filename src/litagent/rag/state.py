"""Persist resumable paper-ingestion state in PostgreSQL."""

from __future__ import annotations

import json
from enum import Enum
from importlib.resources import files
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from litagent.rag.models import PaperRecord


class IngestionStatus(str, Enum):
    """Describe one terminal or resumable ingestion state."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class PaperCorpusState(BaseModel):
    """Keep the last committed chunk set plus index identity."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    collection_name: str
    paper_id: str
    status: IngestionStatus
    batch_id: str | None = None
    asset_hash: str | None = None
    metadata_hash: str | None = None
    corpus_version: str | None = None
    schema_version: str | None = None
    parser_version: str | None = None
    chunking_version: str | None = None
    chunk_strategy: str | None = None
    chunk_size: int | None = None
    chunk_overlap: int | None = None
    embedding_backend: str | None = None
    embedding_model: str | None = None
    embedding_document_adapter: str | None = None
    active_point_ids: list[str] = Field(default_factory=list)
    chunk_hashes: dict[str, str] = Field(default_factory=dict)
    embedding_input_hashes: dict[str, str] = Field(default_factory=dict)
    payload_hashes: dict[str, str] = Field(default_factory=dict)
    error_code: str | None = None

    @classmethod
    def from_success(
        cls,
        *,
        identity,
        record: PaperRecord,
        point_ids: list[str],
        batch_id: str,
    ) -> "PaperCorpusState":
        """Build state only after all Qdrant side effects succeed."""
        return cls(
            collection_name=identity.collection_name,
            paper_id=record.paper_id,
            status=IngestionStatus.SUCCEEDED,
            batch_id=batch_id,
            asset_hash=record.asset_hash,
            metadata_hash=record.metadata_hash,
            corpus_version=identity.corpus_version,
            schema_version=identity.schema_version,
            parser_version=identity.parser_version,
            chunking_version=identity.chunking_version,
            chunk_strategy=identity.chunk_strategy,
            chunk_size=identity.chunk_size,
            chunk_overlap=identity.chunk_overlap,
            embedding_backend=identity.embedding_backend,
            embedding_model=identity.embedding_model,
            embedding_document_adapter=identity.embedding_document_adapter,
            active_point_ids=point_ids,
            chunk_hashes={
                point_id: chunk.content_hash
                for point_id, chunk in zip(
                    point_ids,
                    record.chunks,
                    strict=True,
                )
            },
            embedding_input_hashes={
                point_id: identity.embedding_input_hash(record, chunk)
                for point_id, chunk in zip(
                    point_ids,
                    record.chunks,
                    strict=True,
                )
            },
            payload_hashes={
                point_id: identity.payload_hash(record, chunk)
                for point_id, chunk in zip(
                    point_ids,
                    record.chunks,
                    strict=True,
                )
            },
        )


class CorpusStateRepository:
    """Use PostgreSQL as the authoritative incremental-state backend."""

    def __init__(self, pool) -> None:
        self._pool = pool

    @staticmethod
    def _deserialize(row) -> PaperCorpusState:
        """Normalize asyncpg JSON and omit database-only audit columns."""
        payload = dict(row)
        payload.pop("updated_at", None)
        for field_name in (
            "chunk_hashes",
            "embedding_input_hashes",
            "payload_hashes",
        ):
            value = payload.get(field_name)
            if isinstance(value, str):
                payload[field_name] = json.loads(value)
        return PaperCorpusState.model_validate(payload)

    async def ensure_tables(self) -> None:
        """Install the packaged idempotent CorpusState schema."""
        schema = (
            files("litagent.rag")
            .joinpath("corpus_schema.sql")
            .read_text(encoding="utf-8")
        )
        await self._pool.execute(schema)

    async def get_paper(
        self, collection_name: str, paper_id: str
    ) -> PaperCorpusState | None:
        """Read the latest state for one paper."""
        row = await self._pool.fetchrow(
            """
            SELECT * FROM corpus_paper_state
            WHERE collection_name = $1 AND paper_id = $2
            """,
            collection_name,
            paper_id,
        )
        return self._deserialize(row) if row else None

    async def mark_running(
        self,
        collection_name: str,
        paper_id: str,
        batch_id: str,
    ) -> None:
        """Start/restart work while preserving the last successful hashes."""
        await self._pool.execute(
            """
            INSERT INTO corpus_paper_state
                (collection_name, paper_id, status, batch_id)
            VALUES ($1, $2, 'running', $3)
            ON CONFLICT (collection_name, paper_id) DO UPDATE SET
                status = 'running',
                batch_id = EXCLUDED.batch_id,
                error_code = NULL,
                updated_at = now()
            """,
            collection_name,
            paper_id,
            batch_id,
        )

    async def mark_succeeded(self, state: PaperCorpusState) -> None:
        """Atomically replace hashes, point ownership, and index identity."""
        await self._pool.execute(
            """
            INSERT INTO corpus_paper_state (
                collection_name, paper_id, status, batch_id,
                asset_hash, metadata_hash, corpus_version, schema_version,
                parser_version, chunking_version, chunk_strategy,
                chunk_size, chunk_overlap, embedding_backend, embedding_model,
                embedding_document_adapter, active_point_ids, chunk_hashes,
                embedding_input_hashes, payload_hashes,
                error_code
            )
            VALUES (
                $1, $2, 'succeeded', $3, $4, $5, $6, $7,
                $8, $9, $10, $11, $12, $13, $14, $15, $16, $17::jsonb,
                $18::jsonb, $19::jsonb, NULL
            )
            ON CONFLICT (collection_name, paper_id) DO UPDATE SET
                status = 'succeeded',
                batch_id = EXCLUDED.batch_id,
                asset_hash = EXCLUDED.asset_hash,
                metadata_hash = EXCLUDED.metadata_hash,
                corpus_version = EXCLUDED.corpus_version,
                schema_version = EXCLUDED.schema_version,
                parser_version = EXCLUDED.parser_version,
                chunking_version = EXCLUDED.chunking_version,
                chunk_strategy = EXCLUDED.chunk_strategy,
                chunk_size = EXCLUDED.chunk_size,
                chunk_overlap = EXCLUDED.chunk_overlap,
                embedding_backend = EXCLUDED.embedding_backend,
                embedding_model = EXCLUDED.embedding_model,
                embedding_document_adapter = EXCLUDED.embedding_document_adapter,
                active_point_ids = EXCLUDED.active_point_ids,
                chunk_hashes = EXCLUDED.chunk_hashes,
                embedding_input_hashes = EXCLUDED.embedding_input_hashes,
                payload_hashes = EXCLUDED.payload_hashes,
                error_code = NULL,
                updated_at = now()
            """,
            state.collection_name,
            state.paper_id,
            state.batch_id,
            state.asset_hash,
            state.metadata_hash,
            state.corpus_version,
            state.schema_version,
            state.parser_version,
            state.chunking_version,
            state.chunk_strategy,
            state.chunk_size,
            state.chunk_overlap,
            state.embedding_backend,
            state.embedding_model,
            state.embedding_document_adapter,
            state.active_point_ids,
            json.dumps(state.chunk_hashes, sort_keys=True),
            json.dumps(state.embedding_input_hashes, sort_keys=True),
            json.dumps(state.payload_hashes, sort_keys=True),
        )

    async def mark_failed(
        self,
        collection_name: str,
        paper_id: str,
        batch_id: str,
        error_code: str,
    ) -> None:
        """Record failure without erasing the previous committed snapshot."""
        await self._pool.execute(
            """
            UPDATE corpus_paper_state
            SET status = 'failed',
                batch_id = $3,
                error_code = $4,
                updated_at = now()
            WHERE collection_name = $1 AND paper_id = $2
            """,
            collection_name,
            paper_id,
            batch_id,
            error_code,
        )

    async def list_papers(
        self,
        collection_name: str,
    ) -> list[PaperCorpusState]:
        """List committed/resumable rows used for manifest-level pruning."""
        rows = await self._pool.fetch(
            """
            SELECT * FROM corpus_paper_state
            WHERE collection_name = $1
            ORDER BY paper_id
            """,
            collection_name,
        )

        return [self._deserialize(row) for row in rows]

    async def delete_paper(
        self,
        collection_name: str,
        paper_id: str,
    ) -> None:
        """Delete state only after all owned Qdrant points are gone."""
        await self._pool.execute(
            """
            DELETE FROM corpus_paper_state
            WHERE collection_name = $1 AND paper_id = $2
            """,
            collection_name,
            paper_id,
        )

    async def reset_collection(self, collection_name: str) -> None:
        """Clear stale hash state before an explicitly confirmed rebuild."""
        await self._pool.execute(
            "DELETE FROM corpus_paper_state WHERE collection_name = $1",
            collection_name,
        )
