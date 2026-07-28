"""Persist source-aware semantic facts and embeddings in PostgreSQL."""

import json

import asyncpg

from litagent.config import MemoryConfig
from litagent.logging import get_logger

logger = get_logger("memory.semantic")


class SemanticMemory:
    """Store and retrieve semantic facts with provenance-based ranking."""

    def __init__(self, pool: asyncpg.Pool):
        self._pool = pool

    @staticmethod
    async def connect(config: MemoryConfig) -> "SemanticMemory":
        """Connect to the PostgreSQL semantic-memory store."""
        pool = await asyncpg.create_pool(config.pg_url, min_size=2, max_size=10)
        logger.info("Connected to PostgreSQL (Semantic Memory)")
        return SemanticMemory(pool)

    async def upsert(
        self,
        key: str,
        value: dict,
        entry_type: str = "domain_knowledge",
        source: str = "inferred",
        confidence: float = 0.5,
        episode_id: str = "",
    ) -> None:
        """Upsert a semantic fact after ranking competing provenance metadata."""
        existing = await self._get_by_key(key)
        if existing:
            winner = self._resolve_conflict(
                key,
                [
                    existing,
                    {"value": value, "confidence": confidence, "source": source},
                ],
            )
            confidence = winner["confidence"]
            source = winner["source"]

        from litagent.rag.embedder import get_embedder

        vec = get_embedder().embed(f"{key} {entry_type}")
        vec_str = _vector_to_pg(vec)

        await self._pool.execute(
            """INSERT INTO semantic_entries (entry_type, key, value, confidence, source,
               source_episode_ids, embedding)
               VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
               ON CONFLICT (key) DO UPDATE SET
               value = $3, confidence = $4, source = $5,
               source_episode_ids = semantic_entries.source_episode_ids || $6,
               updated_at = now()""",
            entry_type,
            key,
            json.dumps(value),
            confidence,
            source,
            [episode_id] if episode_id else [],
            vec_str,
        )

    async def get(self, key: str) -> dict | None:
        """Return one semantic entry and decode its stored JSON value."""
        row = await self._get_by_key(key)
        if row is None:
            return None
        value = row["value"]
        if isinstance(value, str):
            value = json.loads(value)
        return {
            "value": value,
            "confidence": row["confidence"],
            "source": row["source"],
            "entry_type": row["entry_type"],
        }

    async def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve semantic entries nearest to the query embedding."""
        from litagent.rag.embedder import get_embedder

        embed = get_embedder()
        vec_str = _vector_to_pg(embed.embed(query))
        rows = await self._pool.fetch(
            """SELECT *, embedding <=> $1::vector AS distance
               FROM semantic_entries ORDER BY distance LIMIT $2""",
            vec_str,
            top_k,
        )
        return [dict(r) for r in rows]

    async def _get_by_key(self, key: str) -> dict | None:
        row = await self._pool.fetchrow(
            "SELECT * FROM semantic_entries WHERE key = $1", key
        )
        return dict(row) if row else None

    def _resolve_conflict(self, key: str, candidates: list[dict]) -> dict:
        """Select a candidate by source authority, confidence, and support."""
        if not candidates:
            return {"confidence": 0.5, "source": "inferred"}

        # Explicit user facts receive a higher source-authority weight.
        source_weight = {"explicit_user": 3, "observed": 2, "inferred": 1}

        best = None
        best_score = -1
        for c in candidates:
            sw = source_weight.get(c.get("source", "inferred"), 0)
            score = sw * 0.4 + c.get("confidence", 0.5) * 0.3
            if "episode_count" in c:
                score += c["episode_count"] * 0.1
            if score > best_score:
                best_score = score
                best = c

        logger.debug(f"Conflict resolved for '{key}': winner source={best['source']}")
        return best

    async def close(self) -> None:
        """Close the PostgreSQL connection pool."""
        await self._pool.close()


def _vector_to_pg(vec: list[float]) -> str:
    return "[" + ",".join(str(v) for v in vec) + "]"
