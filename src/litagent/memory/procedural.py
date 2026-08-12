"""Store rolling execution profiles used for source ranking."""

import asyncpg

from litagent.config import MemoryConfig
from litagent.logging import get_logger

logger = get_logger("memory.procedural")


class ProceduralMemory:
    """Persist aggregate tool-execution profiles in PostgreSQL."""

    def __init__(self, pool: asyncpg.Pool):
        """Initialize the procedural memory."""
        self._pool = pool

    async def ensure_tables(self) -> None:
        """Create or migrate the procedural-memory schema."""
        await self._ensure_profile_table()

    @staticmethod
    async def connect(config: MemoryConfig) -> "ProceduralMemory":
        """Connect to PostgreSQL and initialize the profile schema."""
        pool = await asyncpg.create_pool(config.pg_url)
        logger.info("Connected to PostgresSQL (Procedural Memory)")
        mem = ProceduralMemory(pool)
        await mem.ensure_tables()
        return mem

    async def _ensure_profile_table(self) -> None:
        """Create the profile table and migrate legacy duration precision."""
        await self._pool.execute(
            """
            CREATE TABLE IF NOT EXISTS procedural_profiles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                profile_type VARCHAR(64) NOT NULL,
                profile_key VARCHAR(255) NOT NULL,
                subject VARCHAR(255) NOT NULL,
                scope VARCHAR(64) NOT NULL DEFAULT 'global',
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                empty_result_count INTEGER NOT NULL DEFAULT 0,
                rate_limit_count INTEGER NOT NULL DEFAULT 0,
                timeout_count INTEGER NOT NULL DEFAULT 0,
                execution_count INTEGER NOT NULL DEFAULT 0,
                avg_duration_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0,
                avg_result_count REAL NOT NULL DEFAULT 0.0,
                last_executed_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                UNIQUE(profile_type, profile_key, scope)
            )
        """
        )

        await self._pool.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_profiles_type_scope
            ON procedural_profiles(profile_type, scope)
        """
        )

        # Preserve fractional rolling averages when upgrading legacy INTEGER schemas.
        await self._pool.execute(
            """
            ALTER TABLE procedural_profiles
            ALTER COLUMN avg_duration_ms TYPE DOUBLE PRECISION
            USING avg_duration_ms::DOUBLE PRECISION
        """
        )

    async def upsert_profile(
        self,
        profile_type: str,
        profile_key: str,
        subject: str,
        scope: str = "global",
        *,
        success: bool = True,
        empty_result: bool = False,
        error_type: str | None = None,
        duration_ms: int = 0,
        result_count: int = 0,
    ) -> None:
        """Merge one execution outcome into a rolling procedural profile."""
        is_failure = not success
        is_empty = success and empty_result

        await self._pool.execute(
            """INSERT INTO procedural_profiles
                  (profile_type, profile_key, subject, scope,
                   success_count, failure_count, empty_result_count,
                   rate_limit_count, timeout_count, execution_count,
                   avg_duration_ms, avg_result_count, last_executed_at)
               VALUES ($1,$2,$3,$4, $5,$6,$7,$8,$9, 1, $10,$11, now())
               ON CONFLICT (profile_type, profile_key, scope) DO UPDATE SET
                   success_count = procedural_profiles.success_count + $5,
                   failure_count = procedural_profiles.failure_count + $6,
                   empty_result_count = procedural_profiles.empty_result_count + $7,
                   rate_limit_count = procedural_profiles.rate_limit_count + $8,
                   timeout_count = procedural_profiles.timeout_count + $9,
                   execution_count = procedural_profiles.execution_count + 1,
                   avg_duration_ms = CASE
                       WHEN procedural_profiles.execution_count > 0
                       THEN (procedural_profiles.avg_duration_ms
                             * procedural_profiles.execution_count + $10)
                            / (procedural_profiles.execution_count + 1)
                       ELSE $10
                   END,
                   avg_result_count = CASE
                       WHEN procedural_profiles.execution_count > 0
                       THEN (procedural_profiles.avg_result_count
                             * procedural_profiles.execution_count + $11)
                            / (procedural_profiles.execution_count + 1)
                       ELSE $11
                   END,
                   last_executed_at = now(),
                   updated_at = now()""",
            # Map this observation to the SQL statement's counter increments.
            profile_type,
            profile_key,
            subject,
            scope,
            0 if is_failure or is_empty else 1,
            1 if is_failure else 0,
            1 if is_empty else 0,
            1 if error_type == "rate_limit" else 0,
            1 if error_type == "timeout" else 0,
            duration_ms,
            float(result_count),
        )

    async def get_profiles(
        self,
        profile_type: str = "search_source",
        scope: str = "global",
    ) -> list[dict]:
        """Return profiles for a type and scope, ordered by subject."""
        rows = await self._pool.fetch(
            """SELECT * FROM procedural_profiles
            WHERE profile_type = $1 AND scope = $2
            ORDER BY subject""",
            profile_type,
            scope,
        )
        return [dict(r) for r in rows]

    async def close(self) -> None:
        """Close the PostgreSQL connection pool."""
        await self._pool.close()
