""" MemoryManager - 四层Memory统一入口 """

from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.models import Episode
from litagent.memory.consolidate import consolidate_session
from litagent.logging import get_logger


logger = get_logger('memory.manager')


class MemoryManager:
    """四层Memory 统一入口

    Usage: 
        working = await WorkingMemory.connect(memory_config)
        episodic = await EpisodicMemory.connect(memory_config)
        mm = MemoryManager(working, episodic)
    """

    def __init__(
        self,
        working: WorkingMemory,
        episodic: EpisodicMemory,
    ):
        self.working = working
        self.episodic = episodic

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

    # -- Recall (跨层) --

    async def recall(self, query: str, top_k: int = 5) -> list[Episode]:
        """跨层召回。Phase 4 只查 Episodic，Phase 5 合并 Semantic。"""
        return await self.recall_episode(query, top_k)

    # -- Consolidate --

    async def consolidate(self, session_id: str) -> Episode | None:
        """Working → Episodic 提升。

        读 Working Memory → 提取结构化摘要 → 写入 Qdrant。
        Phase 4 用规则提取；Phase 5 升级为 LLM 驱动。
        """
        state = await self.working.get(session_id)
        if state is None:
            logger.warning(f"Session '{session_id}' not found for consolidate")
            return None

        episode = await consolidate_session(state, session_id)
        if episode is None:
            return None
        
        eid = await self.episodic.store(episode)
        episode.episode_id = eid
        logger.info(f"Consolidated session '{session_id}' → episode '{eid}'")
        return episode

