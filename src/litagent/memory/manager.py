""" MemoryManager - 四层Memory统一入口 """

from litagent.llm.client import BaseLLMClient
from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.models import Episode
from litagent.memory.consolidate import consolidate_session
from litagent.logging import get_logger


logger = get_logger('memory.manager')


class MemoryManager:
    """四层Memory 统一入口"""

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        procedural: ProceduralMemory | None = None,
    ):
        self.working = working
        self.episodic = episodic
        self.semantic = semantic
        self.procedural = procedural

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
        
        eid = await self.episodic.store(episode)
        episode.episode_id = eid
        logger.info(f"Consolidated session '{session_id}' → episode '{eid}'")

        for fact in episode.extracted_facts:
            if not isinstance(fact, dict) or 'key' not in fact:
                continue
            await self.semantic.upsert(
                key=fact['key'],
                value=fact.get('value', {}),
                entry_type=fact.get('type', 'domain_knowledge'),
                source='extracted',
                confidence=fact.get('confidence', 0.5),
                episode_id=eid,
            )

        if episode.extracted_facts:
            logger.info(f"Extracted {len(episode.extracted_facts)} facts → Semantic Memory")

        return episode

    # -- Procedural Memory --

    async def match_procedure(self, user_input: str) -> list[dict]:
        """匹配触发词 → 返回匹配的 Procedure 模板"""
        return await self.procedural.match(user_input) if self.procedural else []

    async def record_procedure_execution(self, procedure_id: str, success: bool, duration_ms: int) -> None:
        """记录 Procedure 执行结果"""
        if self.procedural:
            await self.procedural.record_execution(procedure_id, success, duration_ms)

