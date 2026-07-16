import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient
from langchain_core.messages import HumanMessage, AIMessage

from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory, COLLECTION_NAME
from litagent.memory.models import Episode
from litagent.memory.manager import MemoryManager
from litagent.config import MemoryConfig


# ── Working Memory Tests ──

@pytest_asyncio.fixture
async def working():
    config = MemoryConfig(redis_url="redis://localhost:6379", working_ttl_seconds=60)
    wm = await WorkingMemory.connect(config)
    yield wm
    await wm.delete("test_session")


class TestWorkingMemory:
    @pytest.mark.asyncio
    async def test_set_and_get(self, working):
        state = {"messages": [], "loop_count": 3, "final_answer": "done"}
        await working.set("test_session", state)
        result = await working.get("test_session")
        assert result is not None
        assert result["loop_count"] == 3

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, working):
        assert await working.get("no_such_session") is None

    @pytest.mark.asyncio
    async def test_delete(self, working):
        await working.set("test_session", {"data": 1})
        await working.delete("test_session")
        assert await working.get("test_session") is None

    @pytest.mark.asyncio
    async def test_exists(self, working):
        await working.set("test_session", {"data": 1})
        assert await working.exists("test_session") is True


# ── Episodic Memory Tests ──

@pytest_asyncio.fixture
async def episodic():
    client = AsyncQdrantClient(url="http://localhost:6333")
    em = await EpisodicMemory.connect(MemoryConfig(qdrant_url="http://localhost:6333"))
    yield em
    # cleanup: 清空整个 collection
    from qdrant_client.models import PointIdsList
    records, _ = await client.scroll(collection_name=COLLECTION_NAME, limit=100)
    if records:
        await client.delete(collection_name=COLLECTION_NAME,
                           points_selector=PointIdsList(points=[r.id for r in records]))


class TestEpisodicMemory:
    @pytest.mark.asyncio
    async def test_store_and_search(self, episodic):
        ep = Episode(summary="survey on few-shot learning", intent="literature_review", session_id="s1")
        eid = await episodic.store(ep)
        assert eid
        results = await episodic.search("few-shot learning")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_delete(self, episodic):
        ep = Episode(summary="test delete", intent="general", session_id="s1")
        eid = await episodic.store(ep)
        await episodic.delete(eid)
        results = await episodic.search("delete")
        assert len(results) == 0


# ── MemoryManager + Consolidate Tests ──

@pytest_asyncio.fixture
async def mm(working, episodic, semantic):
    return MemoryManager(working, episodic, semantic)


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_save_get_consolidate_recall(self, mm):
        messages = [HumanMessage(content="survey few-shot learning"), AIMessage(content="ok")]
        await mm.save_state("session_1", {"messages": messages, "loop_count": 1})
        ep = await mm.consolidate("session_1")
        assert ep is not None
        assert "few-shot" in ep.summary.lower()
        results = await mm.recall("few-shot")
        assert len(results["episodes"]) >= 1

    @pytest.mark.asyncio
    async def test_consolidate_empty_session(self, mm):
        await mm.save_state("empty_session", {"messages": [], "loop_count": 0})
        ep = await mm.consolidate("empty_session")
        assert ep is None


# ── Phase 5: Semantic Memory Tests ──

@pytest_asyncio.fixture
async def semantic():
    import asyncpg
    pool = await asyncpg.create_pool("postgresql://litagent:litagent@localhost:5432/litagent")
    from litagent.memory.semantic import SemanticMemory
    sm = SemanticMemory(pool)
    yield sm
    await pool.execute("DELETE FROM semantic_entries")
    await pool.close()


class TestSemanticMemory:
    @pytest.mark.asyncio
    async def test_upsert_and_get(self, semantic):
        await semantic.upsert("test_key", {"answer": 42})
        result = await semantic.get("test_key")
        assert result is not None
        assert result["value"]["answer"] == 42

    @pytest.mark.asyncio
    async def test_search_fallback(self, semantic):
        await semantic.upsert("few_shot_benchmarks", {"list": ["miniImageNet"]})
        results = await semantic.search("few_shot")
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_upsert_overwrites_same_key(self, semantic):
        await semantic.upsert("dup_key", {"v": 1})
        await semantic.upsert("dup_key", {"v": 2})
        result = await semantic.get("dup_key")
        assert result["value"]["v"] == 2


# ── 13.7.1 Procedural Memory Profile Tests ──

import os as _os
from urllib.parse import urlparse
from uuid import uuid4


def _test_pg_url() -> str:
    """Return the explicitly isolated PostgreSQL URL used by integration tests."""
    pg_url = _os.environ.get("TEST_PG_URL")
    if not pg_url:
        pytest.skip("TEST_PG_URL not set")
    if urlparse(pg_url).path.rstrip("/") != "/litagent_test":
        pytest.fail("TEST_PG_URL must target the isolated litagent_test database")
    return pg_url


def _test_subject(prefix: str) -> str:
    return f"{prefix}{uuid4().hex}"

@pytest_asyncio.fixture
async def procedural():
    """需要 TEST_PG_URL 环境变量；未设置时 skip。"""
    pg_url = _test_pg_url()
    import asyncpg
    pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=1)
    from litagent.memory.procedural import ProceduralMemory
    pm = ProceduralMemory(pool)
    await pm.ensure_tables()
    yield pm
    await pool.close()


@pytest.mark.integration
class TestProceduralProfiles:
    """13.7.1：procedural_profiles 表 + upsert_profile + get_profiles。"""

    _PREFIX = "test_13_7_1_"

    async def _cleanup(self, procedural, subject: str):
        await procedural._pool.execute(
            "DELETE FROM procedural_profiles WHERE subject = $1", subject)

    @pytest.mark.asyncio
    async def test_first_write_creates_row(self, procedural):
        subject = _test_subject(self._PREFIX)
        try:
            await procedural.upsert_profile(
                "search_source", f"search_source:{subject}", subject,
                success=True, duration_ms=100, result_count=5,
            )
            profiles = await procedural.get_profiles()
            assert any(p["subject"] == subject for p in profiles)
        finally:
            await self._cleanup(procedural, subject)

    @pytest.mark.asyncio
    async def test_updates_aggregate_fields(self, procedural):
        """两次写入 → 计数器累加，平均值为滚动平均。"""
        subject = _test_subject(self._PREFIX)
        try:
            await procedural.upsert_profile(
                "search_source", f"search_source:{subject}", subject,
                success=True, duration_ms=100, result_count=5,
            )
            await procedural.upsert_profile(
                "search_source", f"search_source:{subject}", subject,
                success=True, duration_ms=300, result_count=3,
            )
            profiles = await procedural.get_profiles()
            p = [p for p in profiles if p["subject"] == subject][0]
            assert p["success_count"] == 2
            assert p["execution_count"] == 2
            assert 190 < p["avg_duration_ms"] < 210
            assert 3.9 < p["avg_result_count"] < 4.1
        finally:
            await self._cleanup(procedural, subject)

    @pytest.mark.asyncio
    async def test_mutual_exclusion(self, procedural):
        """success / empty / failure 三者互斥。"""
        subject = _test_subject(self._PREFIX)
        try:
            await procedural.upsert_profile(
                "search_source", f"search_source:{subject}", subject,
                success=True, empty_result=True,
            )
            profiles = await procedural.get_profiles()
            p = [p for p in profiles if p["subject"] == subject][0]
            assert p["success_count"] == 0
            assert p["empty_result_count"] == 1
            assert p["failure_count"] == 0
            assert p["execution_count"] == 1
        finally:
            await self._cleanup(procedural, subject)

    @pytest.mark.asyncio
    async def test_error_subtype_stacking(self, procedural):
        """error_type='rate_limit' → failure 和 rate_limit 同时 +1。"""
        subject = _test_subject(self._PREFIX)
        try:
            await procedural.upsert_profile(
                "search_source", f"search_source:{subject}", subject,
                success=False, error_type="rate_limit",
            )
            profiles = await procedural.get_profiles()
            p = [p for p in profiles if p["subject"] == subject][0]
            assert p["failure_count"] == 1
            assert p["rate_limit_count"] == 1
            assert p["timeout_count"] == 0
        finally:
            await self._cleanup(procedural, subject)


@pytest.mark.integration
class TestProceduralPersistence:
    """13.7.1-C：跨连接画像持久化。"""

    _PREFIX = "test_persist_"

    @pytest.mark.asyncio
    async def test_profiles_persist_across_connections(self):
        pg_url = _test_pg_url()
        import asyncpg
        from unittest.mock import MagicMock
        from litagent.memory.manager import MemoryManager
        from litagent.memory.procedural import ProceduralMemory

        subject = _test_subject(self._PREFIX)
        pool1 = await asyncpg.create_pool(pg_url, min_size=1, max_size=1)
        try:
            pm1 = ProceduralMemory(pool1)
            await pm1.ensure_tables()
            for _ in range(5):
                await pm1.upsert_profile(
                    "search_source", f"search_source:{subject}", subject,
                    success=True, duration_ms=200, result_count=10)
        finally:
            await pool1.close()

        pool2 = await asyncpg.create_pool(pg_url, min_size=1, max_size=1)
        try:
            pm2 = ProceduralMemory(pool2)
            manager = MemoryManager(
                working=MagicMock(), episodic=MagicMock(), procedural=pm2,
            )
            ranked = await manager.rank_search_sources(
                ["unknown_source", subject], min_samples=3,
            )
            profiles = await pm2.get_profiles()
            p = [p for p in profiles if p["subject"] == subject]
            assert len(p) == 1
            assert p[0]["execution_count"] == 5
            assert p[0]["success_count"] == 5
            assert ranked == [subject, "unknown_source"]
        finally:
            await pm2._pool.execute(
                "DELETE FROM procedural_profiles WHERE subject = $1", subject)
            await pool2.close()

    @pytest.mark.asyncio
    async def test_ensure_tables_idempotent(self):
        pg_url = _test_pg_url()
        import asyncpg
        from litagent.memory.procedural import ProceduralMemory

        pool = await asyncpg.create_pool(pg_url, min_size=1, max_size=1)
        try:
            pm = ProceduralMemory(pool)
            await pm.ensure_tables()
            await pm.ensure_tables()  # 幂等
        finally:
            await pool.close()


@pytest.mark.integration
class TestConcurrentUpsert:
    """13.7.1-C：并发 UPSERT 聚合正确性。"""

    _PREFIX = "test_concurrent_"

    @pytest.mark.asyncio
    async def test_concurrent_writes_aggregate_correctly(self):
        pg_url = _test_pg_url()
        import asyncio as _asyncio
        import asyncpg
        from litagent.memory.procedural import ProceduralMemory

        subject = _test_subject(self._PREFIX)
        N = 20
        durations = list(range(100, 100 + N))

        async def write_one(pool, dur):
            pm = ProceduralMemory(pool)
            await pm.upsert_profile(
                "search_source", f"search_source:{subject}", subject,
                success=True, duration_ms=dur, result_count=5)

        pool = await asyncpg.create_pool(pg_url, min_size=5, max_size=5)
        try:
            pm = ProceduralMemory(pool)
            await pm.ensure_tables()
            await _asyncio.gather(*(write_one(pool, d) for d in durations))
            profiles = await pm.get_profiles()
            p = [p for p in profiles if p["subject"] == subject]
            assert len(p) == 1
            assert p[0]["execution_count"] == N
            expected_avg = sum(durations) / N
            assert abs(p[0]["avg_duration_ms"] - expected_avg) < 2
        finally:
            await pm._pool.execute(
                "DELETE FROM procedural_profiles WHERE subject = $1", subject)
            await pool.close()


# ── Phase 5: MemoryManager with Semantic ──

@pytest_asyncio.fixture
async def mm_with_semantic(working, episodic, semantic):
    from litagent.memory.manager import MemoryManager
    return MemoryManager(working, episodic, semantic)


class TestMemoryManagerWithSemantic:
    @pytest.mark.asyncio
    async def test_recall_returns_dict(self, mm_with_semantic):
        await mm_with_semantic.semantic.upsert("test_fact", {"x": 1})
        result = await mm_with_semantic.recall("test_fact")
        assert isinstance(result, dict)
        assert "episodes" in result
        assert "facts" in result

    @pytest.mark.asyncio
    async def test_recall_semantic(self, mm_with_semantic):
        await mm_with_semantic.semantic.upsert("test_key", {"data": "hello"})
        facts = await mm_with_semantic.recall_semantic("test_key")
        assert len(facts) >= 1


# ── Episode Model Tests ──

class TestEpisode:
    def test_create_and_serialize(self):
        ep = Episode(summary="test summary", intent="general", session_id="s1")
        d = ep.to_dict()
        assert d["summary"] == "test summary"
        ep2 = Episode.from_dict(d)
        assert ep2.intent == "general"
