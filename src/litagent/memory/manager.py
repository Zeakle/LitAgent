""" MemoryManager - 四层Memory统一入口 """

from litagent.llm.client import BaseLLMClient
from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.models import Episode
from litagent.memory.consolidate import consolidate_session
from litagent.logging import get_logger
from litagent.observability.context import get_task_id


logger = get_logger('memory.manager')


class MemoryManager:
    """四层Memory 统一入口"""

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory | None = None,
        procedural: ProceduralMemory | None = None,
        trace_hook=None,
    ):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural
        self._trace_hook = trace_hook


    def _emit(self, event: str, data: dict) -> None:
        """触发 trace hook"""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")


    # -- Working Memory --

    async def get_state(self, session_id: str) -> dict | None:
        return await self.working.get(session_id)
    
    async def save_state(self, session_id: str, state: dict) -> None:
        await self.working.set(session_id, state)

    async def delete_session(self, session_id: str) -> None:
        await self.working.delete(session_id)

    # -- Episodic Memory --

    async def remember_episode(self, episode: Episode) -> str:
        return await self.episodic.store(episode)

    async def recall_episode(self, query: str, top_k: int = 5) -> list[Episode]:
        """搜索 Episodic Memory。"""
        return await self.episodic.search(query, top_k)

    # -- Semantic Memory --

    async def recall_semantic(self, query: str, top_k: int = 5) -> list[dict]:
        """搜索 Semantic Memory"""
        return await self.semantic.search(query, top_k) if self.semantic else []

    # -- Recall (跨层) --

    async def recall(self, query: str, top_k: int = 5) -> dict:
        """跨层召回。Phase 4 只查 Episodic，Phase 5 合并 Semantic。"""
        episodes = await self.recall_episode(query, top_k)
        facts = await self.recall_semantic(query, top_k)
        self._emit("memory.recall", {
            "task_id": get_task_id(),
            "episodes": len(episodes),
            "facts": len(facts),
        })
        return {
            'episodes': episodes,
            'facts': facts
        }

    # -- Consolidate --

    async def consolidate(self, session_id: str, llm: BaseLLMClient | None = None) -> Episode | None:
        """Working → Episodic 提升 + Semantic 知识沉淀"""

        state = await self.working.get(session_id)
        if state is None:
            logger.warning(f"Session '{session_id}' not found for consolidate")
            return None

        episode = await consolidate_session(state, session_id, llm=llm)
        if episode is None:
            return None
        
        eid = None
        try:
            eid = await self.episodic.store(episode)
            episode.episode_id = eid
            logger.info(f"Consolidated session '{session_id}' → episode '{eid}'")
            self._emit("memory.write", {
                "task_id": get_task_id(),
                "layer": "episodic",
                "success": True,
                "episode_id": eid,
            })
        except Exception as e:
            logger.warning(f"Episodic store failed during consolidate: {e}")
            self._emit("memory.write", {
                "task_id": get_task_id(),
                "layer": "episodic",
                "success": False,
                "error": str(e),
            })

        # ── 写 Semantic（eid 可能为 None——episodic 失败时 fact 无 episode 关联）──
        written = 0
        for fact in episode.extracted_facts:
            if not isinstance(fact, dict) or 'key' not in fact:
                continue
            try:
                await self.semantic.upsert(
                    key=fact['key'],
                    value=fact.get('value', {}),
                    entry_type=fact.get('type', 'domain_knowledge'),
                    source='extracted',
                    confidence=fact.get('confidence', 0.5),
                    episode_id=eid,
                )
                written += 1
            except Exception as e:
             logger.warning(f"Semantic upsert failed for key '{fact.get('key')}': {e}")

        if episode.extracted_facts:
            logger.info(f"Extracted {written}/{len(episode.extracted_facts)} facts → Semantic Memory")
        self._emit("memory.write", {
            "task_id": get_task_id(),
            "layer": "semantic",
            "success": True,
            "facts": written,
        })

        return episode

    # -- Procedural Memory --


    async def record_search_source_execution(
        self, subject: str, success: bool, empty_result: bool = False,
        error_type: str | None = None, duration_ms: int = 0, result_count: int = 0,
    ) -> None:
        """SearchWorker 写入入口。procedural 不可用或写失败时 no-op。"""
        if not self.procedural:
            return

        try:
            await self.procedural.upsert_profile(
                profile_type='search_source',
                profile_key=f'search_source:{subject}',
                subject=subject,
                success=success,
                empty_result=empty_result,
                error_type=error_type,
                duration_ms=duration_ms,
                result_count=result_count
            )

            self._emit('memory.write', {
                'task_id': get_task_id(),
                'layer': 'procedural',
                'success': True,
                'source': subject,
                'duration_ms': duration_ms
            })
        except Exception as e:
            logger.warning(f"Procedural profile write failed for '{subject}': {e}")
            self._emit("memory.write", {
                "task_id": get_task_id(),
                "layer": "procedural",
                "success": False,
                "source": subject,
                "error": str(e),
            })


    async def rank_search_sources(
        self, sources: list[str], min_samples: int = 3,
    ) -> list[str]:
        """按 reliability 降序排列搜索源。procedural 不可用时返回原列表。"""
        if not self.procedural:
            return list(sources)

        try:
            profiles = await self.procedural.get_profiles('search_source', 'global')
            stats: dict[str, dict] = {p['subject']: p for p in profiles}

            def _reliability(subject: str) -> float:
                p = stats.get(subject)
                if p is None or p['execution_count'] < min_samples:
                    return 0.5

                ec = p['execution_count']

                raw = (
                    (p["success_count"] + p["empty_result_count"]) / ec
                    - (p["empty_result_count"] / ec) * 0.3
                    - (p["rate_limit_count"] / ec) * 0.5
                    - (p["timeout_count"] / ec) * 0.7
                )
                return max(0.0, min(1.0, raw))

            ranked = sorted(sources, key=lambda s: -_reliability(s))
            self._emit("memory.recall", {
                "task_id": get_task_id(),
                "layer": "procedural",
                "sources": sources,
                "ranked_sources": ranked,
            })
            return ranked
        except Exception as e:
            logger.warning(f"Source ranking failed: {e}")
            return list(sources)
