"""Classify survey queries and build the worker task graph."""

from __future__ import annotations

import json
import os
import re
from enum import Enum
from typing import Callable

from litagent.config import PlannerConfig, RAGConfig
from litagent.context.templates import wrap_xml
from litagent.exceptions import SafetyError
from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger
from litagent.observability.context import reset_task_id, set_task_id
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.safety.injection import InjectionDetector, InjectionRisk

logger = get_logger("agents.planner")


class QueryIntent(str, Enum):
    """Identify the supported survey-query input forms."""

    TOPIC = "topic"
    ARXIV_ID = "arxiv_id"
    DOI = "doi"
    URL = "url"


_ARXIV_NEW_RE = re.compile(r"^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$", re.IGNORECASE)
_ARXIV_LEGACY_RE = re.compile(
    r"^(arxiv:)?[a-z-]+(\.[a-z]{2})?/\d{7}(v\d+)?$", re.IGNORECASE
)
_DOI_RE = re.compile(r"^(doi:)?10\.\d{4,9}/\S+$", re.IGNORECASE)


def classify_query_intent(query: str) -> QueryIntent:
    """Classify a query as a topic, arXiv ID, DOI, or URL."""
    q = (query or "").strip()
    if q.lower().startswith(("http://", "https://")):
        return QueryIntent.URL
    if _ARXIV_NEW_RE.match(q) or _ARXIV_LEGACY_RE.match(q):
        return QueryIntent.ARXIV_ID
    if _DOI_RE.match(q):
        return QueryIntent.DOI
    return QueryIntent.TOPIC


_DECOMPOSE_SYSTEM = """You are a query planning assistant for an academic \
literature survey system.
Given a research survey topic, expand it into 2-3 focused sub-queries that \
together broaden literature coverage (different angles, subtopics, or \
methodological facets of the SAME topic).
Do NOT drift to unrelated topics. Keep each sub-query concise \
(a search-engine query, not a sentence).

Respond ONLY with JSON: {"sub_queries": ["...", "..."]}"""


class SurveyPlanner:
    """Build survey task graphs with optional topic decomposition."""

    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        config: PlannerConfig | None = None,
        trace_hook: Callable[[str, dict], None] | None = None,
        memory_manager: "MemoryManager | None" = None,
        rag_config: RAGConfig | None = None,
    ) -> None:
        """Initialize the survey planner."""
        self._llm = llm
        self._config = config or PlannerConfig()
        self._rag_config = rag_config or RAGConfig()
        self._trace_hook = trace_hook
        self._memory = memory_manager

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for {event} : {e}")

    async def plan(self, query: str) -> TaskGraph:
        """Create a task graph for the survey query."""
        detector = InjectionDetector()
        result = detector.scan(query)
        if result.risk == InjectionRisk.HIGH:
            raise SafetyError(f"Query rejected: injection detected {result.matched}")

        graph = TaskGraph()

        intent = classify_query_intent(query)
        if intent == QueryIntent.TOPIC and self._config.decompose_enabled and self._llm:
            sub_queries = await self._decompose_traced(query)
        else:
            sub_queries = [query]

        normalized: list[str] = []
        seen: set[str] = set()
        for q in sub_queries:
            qs = q.strip()
            k = qs.lower()
            if qs and k not in seen:
                seen.add(k)
                normalized.append(qs)
        sub_queries = normalized or [query]

        sources = ["arxiv", "huggingface"]
        if os.environ.get("SEMANTIC_SCHOLAR_API_KEY"):
            sources.append("semantic_scholar")

        if self._config.use_procedural_profiles and self._memory:
            try:
                sources = await self._memory.rank_search_sources(
                    sources,
                    min_samples=self._config.procedural_min_samples,
                )
            except Exception as e:
                logger.warning(f"Procedural ranking failed, keeping default order: {e}")

        search_ids: list[str] = []
        for rank_idx, source in enumerate(sources):
            for i, sub_query in enumerate(sub_queries):
                tid = f"search_{source}_q{i}"
                graph.add_task(
                    SubTask(
                        task_id=tid,
                        description=f"Search {source} for: {sub_query}",
                        agent_type="search",
                        priority=rank_idx,
                        input_data={
                            "mode": "external",
                            "source": source,
                            "query": sub_query,
                        },
                    )
                )
                search_ids.append(tid)

        recall_ids: list[str] = []
        for index, sub_query in enumerate(sub_queries):
            task_id = f"recall_q{index}"
            graph.add_task(
                SubTask(
                    task_id=task_id,
                    description=f"RAG recall for: {sub_query}",
                    agent_type="recall",
                    input_data={
                        "query": sub_query,
                        "query_index": index,
                        "top_k": self._rag_config.top_k,
                        "candidate_k": self._rag_config.candidate_k,
                        "max_representative_chunks": (
                            self._rag_config.max_representative_chunks
                        ),
                        "retrieval_mode": self._rag_config.retrieval_mode.value,
                    },
                )
            )
            recall_ids.append(task_id)

        graph.add_task(
            SubTask(
                task_id="dedup",
                description="Deduplicate search results",
                agent_type="dedup",
                input_data={"query": query},
            ),
            depends_on=search_ids + recall_ids,
        )

        graph.add_task(
            SubTask(
                task_id="relevance_gate",
                description="Rank papers by relevance to query",
                agent_type="relevance_gate",
                input_data={"query": query},
            ),
            depends_on=["dedup"],
        )

        graph.add_task(
            SubTask(
                task_id="extract",
                description="Extract structure info from papers",
                agent_type="extractor",
                input_data={"query": query},
                timeout_ms=300000,
                max_retries=0,
            ),
            depends_on=["relevance_gate"],
        )

        graph.add_task(
            SubTask(
                task_id="graph_analysis",
                description="Analyze citation network",
                agent_type="graph",
                input_data={"query": query},
            ),
            depends_on=["relevance_gate"],
        )

        # The worker owns its revision loop, so scheduler retries would duplicate work.
        graph.add_task(
            SubTask(
                task_id="adversarial_review",
                description="Adversarial synthesis + review loop",
                agent_type="adversarial_review",
                input_data={"query": query},
                timeout_ms=300000,
                max_retries=0,
            ),
            depends_on=["extract", "graph_analysis"],
        )
        logger.info(f"Planned {len(graph.tasks)} tasks for query {query}")
        return graph

    async def _decompose(self, query: str) -> list[str]:
        """Decompose a survey query into focused subqueries."""
        fallback = [query]
        if not self._llm or not self._config.decompose_enabled:
            return fallback

        try:
            resp = await self._llm.chat(
                [
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user", "content": wrap_xml("topic", query)},
                ],
                response_format={"type": "json_object"},
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            content = (resp.content or "").strip()

            if not content:
                logger.warning("Decompose: empty content, degrade to single query")
                return fallback

            raw = json.loads(content).get("sub_queries")
            if not isinstance(raw, list):
                logger.warning("Decompose: 'sub_queries' not a list, degrade")
                return fallback

            subs = [s.strip() for s in raw if isinstance(s, str) and s.strip()]

            merged, seen = [], set()
            for q in [query, *subs]:
                k = q.lower()
                if k not in seen:
                    seen.add(k)
                    merged.append(q)

            clamped = merged[: self._config.max_sub_queries]
            logger.info(f"Decompose: {query!r} -> {len(clamped)} sub-queries")
            return clamped
        except Exception as e:
            logger.warning(f"Decompose failed ({e}), degrade to single query")
            return fallback

    async def _decompose_traced(self, query: str) -> list[str]:
        """Decompose a query while emitting LLM lifecycle events."""
        self._emit(
            "subspan.start",
            {
                "task_id": "query_decomposition",
                "parent_task_id": "",
                "name": "planner.query_decomposition",
                "round": 0,
            },
        )
        token = set_task_id("query_decomposition")
        sub_queries = [query]
        try:
            sub_queries = await self._decompose(query)
            return sub_queries
        finally:
            reset_task_id(token)
            self._emit(
                "subspan.end",
                {
                    "task_id": "query_decomposition",
                    "output": {"sub_queries": sub_queries},
                },
            )
