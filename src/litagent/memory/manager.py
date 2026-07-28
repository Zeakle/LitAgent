"""Coordinate working, episodic, semantic, and procedural memory backends."""

from litagent.llm.client import BaseLLMClient
from litagent.memory.working import WorkingMemory
from litagent.memory.episodic import EpisodicMemory
from litagent.memory.semantic import SemanticMemory
from litagent.memory.procedural import ProceduralMemory
from litagent.memory.models import Episode
from litagent.memory.consolidate import consolidate_session
from litagent.observability.context import get_task_id
from litagent.observability.lifecycle import traced_io
from litagent.logging import get_logger

logger = get_logger("memory.manager")

LAYER_EPISODIC = "episodic"
LAYER_SEMANTIC = "semantic"
LAYER_PROCEDURAL = "procedural"


class MemoryManager:
    """Expose one interface over the configured memory backends."""

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
        """Emit a trace event without allowing hook failures to escape."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    async def get_state(self, session_id: str) -> dict | None:
        """Return the stored session state."""
        return await self.working.get(session_id)

    async def save_state(self, session_id: str, state: dict) -> None:
        """Persist the session state."""
        await self.working.set(session_id, state)

    async def delete_session(self, session_id: str) -> None:
        """Delete the stored session state."""
        await self.working.delete(session_id)

    async def remember_episode(self, episode: Episode) -> str:
        """Persist an episode and return its identifier."""
        return await self.episodic.store(episode)

    async def recall_episode(self, query: str, top_k: int = 5) -> list[Episode]:
        """Retrieve episodic memories relevant to a query."""
        return await self.episodic.search(query, top_k)

    async def recall_semantic(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve semantic facts when that backend is available."""
        return await self.semantic.search(query, top_k) if self.semantic else []

    async def recall(self, query: str, top_k: int = 5) -> dict:
        """Retrieve episodic and semantic memories with trace metadata."""
        async with traced_io(
            self._emit, "memory.recall", {"query": query[:200], "top_k": top_k}
        ) as outcome:
            episodes = await self.recall_episode(query, top_k)
            facts = await self.recall_semantic(query, top_k)
            outcome["episodes"] = len(episodes)
            outcome["facts"] = len(facts)

        return {"episodes": episodes, "facts": facts}

    async def consolidate(
        self, session_id: str, llm: BaseLLMClient | None = None
    ) -> Episode | None:
        """Promote a working-memory session into persistent memory layers."""

        state = await self.working.get(session_id)
        if state is None:
            logger.warning(f"Session '{session_id}' not found for consolidate")
            return None

        episode = await consolidate_session(state, session_id, llm=llm)
        if episode is None:
            return None

        eid = None
        # Keep consolidation best-effort when one persistent layer is unavailable.
        try:
            async with traced_io(
                self._emit, "memory.write", {"layer": LAYER_EPISODIC}
            ) as outcome:
                eid = await self.episodic.store(episode)
                outcome["episode_id"] = eid
            episode.episode_id = eid
            logger.info(f"Consolidated session '{session_id}' → episode '{eid}'")
        except Exception as e:
            logger.warning(f"Episodic store failed during consolidate: {e}")

        written = 0
        if episode.extracted_facts and self.semantic:
            async with traced_io(
                self._emit, "memory.write", {"layer": LAYER_SEMANTIC}
            ) as outcome:
                # Continue promoting valid facts when an individual write fails.
                for fact in episode.extracted_facts:
                    if not isinstance(fact, dict) or "key" not in fact:
                        continue
                    try:
                        await self.semantic.upsert(
                            key=fact["key"],
                            value=fact.get("value", {}),
                            entry_type=fact.get("type", "domain_knowledge"),
                            source="extracted",
                            confidence=fact.get("confidence", 0.5),
                            episode_id=eid,
                        )
                        written += 1
                    except Exception as e:
                        logger.warning(
                            f"Semantic upsert failed for key '{fact.get('key')}': {e}"
                        )

                outcome["facts"] = written
            logger.info(
                f"Extracted {written}/{len(episode.extracted_facts)} "
                "facts -> Semantic Memory"
            )

        return episode

    async def record_search_source_execution(
        self,
        subject: str,
        success: bool,
        empty_result: bool = False,
        error_type: str | None = None,
        duration_ms: int = 0,
        result_count: int = 0,
    ) -> None:
        """Record one search-source outcome in procedural memory."""
        if not self.procedural:
            return

        try:
            async with traced_io(
                self._emit,
                "memory.write",
                {"layer": LAYER_PROCEDURAL, "source": subject},
            ) as outcome:
                await self.procedural.upsert_profile(
                    profile_type="search_source",
                    profile_key=f"search_source:{subject}",
                    subject=subject,
                    success=success,
                    empty_result=empty_result,
                    error_type=error_type,
                    duration_ms=duration_ms,
                    result_count=result_count,
                )
                outcome["source"] = subject
                outcome["duration_ms"] = duration_ms
        except Exception as e:
            logger.warning(f"Procedural profile write failed for '{subject}': {e}")

    async def rank_search_sources(
        self,
        sources: list[str],
        min_samples: int = 3,
    ) -> list[str]:
        """Order search sources by learned reliability when samples suffice."""
        if not self.procedural:
            return list(sources)

        try:
            async with traced_io(
                self._emit, "memory.recall", {"layer": LAYER_PROCEDURAL}
            ) as outcome:
                profiles = await self.procedural.get_profiles("search_source", "global")
                stats: dict[str, dict] = {p["subject"]: p for p in profiles}

                def _reliability(subject: str) -> float:
                    p = stats.get(subject)
                    if p is None or p["execution_count"] < min_samples:
                        return 0.5

                    ec = p["execution_count"]

                    raw = (
                        (p["success_count"] + p["empty_result_count"]) / ec
                        - (p["empty_result_count"] / ec) * 0.3
                        - (p["rate_limit_count"] / ec) * 0.5
                        - (p["timeout_count"] / ec) * 0.7
                    )
                    return max(0.0, min(1.0, raw))

                ranked = sorted(sources, key=lambda s: -_reliability(s))
                return ranked
        except Exception as e:
            logger.warning(f"Source ranking failed: {e}")
            return list(sources)
