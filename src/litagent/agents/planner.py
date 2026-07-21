"""Planner Agent——查询分解 + TaskGraph 生成。"""

from __future__ import annotations
import os
import re
import json
from enum import Enum
from typing import Callable

from litagent.config import PlannerConfig
from litagent.context.templates import wrap_xml
from litagent.exceptions import SafetyError
from litagent.observability.context import reset_task_id, set_task_id
from litagent.llm.client import BaseLLMClient
from litagent.orchestrator.task_graph import TaskGraph, SubTask
from litagent.logging import get_logger
from litagent.safety.injection import InjectionDetector, InjectionRisk

logger = get_logger("agents.planner")


class QueryIntent(str, Enum):
    TOPIC = 'topic'
    ARXIV_ID = 'arxiv_id'
    DOI = 'doi'
    URL = 'url'


_ARXIV_NEW_RE = re.compile(r'^(arxiv:)?\d{4}\.\d{4,5}(v\d+)?$', re.IGNORECASE)
_ARXIV_LEGACY_RE = re.compile(r'^(arxiv:)?[a-z-]+(\.[a-z]{2})?/\d{7}(v\d+)?$', re.IGNORECASE)
_DOI_RE = re.compile(r'^(doi:)?10\.\d{4,9}/\S+$', re.IGNORECASE)

_RECALL_TOP_K = 20


def classify_query_intent(query: str) -> QueryIntent:
    """纯规则判定 query 意图——无网络、无 LLM"""
    q = (query or "").strip()
    if q.lower().startswith(('http://', 'https://')):
        return QueryIntent.URL
    if _ARXIV_NEW_RE.match(q) or _ARXIV_LEGACY_RE.match(q):
        return QueryIntent.ARXIV_ID
    if _DOI_RE.match(q):
        return QueryIntent.DOI
    return QueryIntent.TOPIC


_DECOMPOSE_SYSTEM = """You are a query planning assistant for an academic literature survey system.
Given a research survey topic, expand it into 2-3 focused sub-queries that together broaden \
literature coverage (different angles, subtopics, or methodological facets of the SAME topic).
Do NOT drift to unrelated topics. Keep each sub-query concise (a search-engine query, not a sentence).

Respond ONLY with JSON: {"sub_queries": ["...", "..."]}"""


class SurveyPlanner:
    """标准综述流程的规则 Planner。"""
    def __init__(
        self,
        llm: BaseLLMClient | None = None,
        config: PlannerConfig | None = None,
        trace_hook: Callable[[str, dict], None] | None = None,
        memory_manager: "MemoryManager | None" = None,
    ):
        self._llm = llm
        self._config = config or PlannerConfig()
        self._trace_hook = trace_hook
        self._memory = memory_manager


    def _emit(self, event: str, data: dict) -> None:
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f'Trace hook failed for {event} : {e}')


    async def plan(self, query: str) -> TaskGraph:
        # prompt injection检测
        detector = InjectionDetector()
        result = detector.scan(query)
        if result.risk == InjectionRisk.HIGH:
            raise SafetyError(f"Query rejected: injection detected {result.matched}")

        graph = TaskGraph()

        # layer0: query扩展
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

        # layer1: 搜索
        search_ids: list[str] = []
        for rank_idx, source in enumerate(sources):
            for i, sub_query in enumerate(sub_queries):
                tid = f'search_{source}_q{i}'
                graph.add_task(SubTask(
                    task_id=tid,
                    description=f"Search {source} for: {sub_query}",
                    agent_type='search',
                    priority=rank_idx,
                    input_data={'mode': 'external', "source": source, "query": sub_query}
                ))
                search_ids.append(tid)

        # RAG call
        recall_ids: list[str] = []
        for i, sub_query in enumerate(sub_queries):
            tid = f'recall_q{i}'
            graph.add_task(SubTask(
                task_id=tid,
                description=f'RAG recall for: {sub_query}',
                agent_type='recall',
                input_data={'query': sub_query, 'top_k': _RECALL_TOP_K, 'query_index': i},
            ))
            recall_ids.append(tid)

        # layer2: 搜索结果去重
        graph.add_task(SubTask(
            task_id='dedup',
            description='Deduplicate search results',
            agent_type='dedup',
            input_data={"query": query},
        ), depends_on=search_ids + recall_ids)

        # layer3: 提取 + 引用分析(并行，只等dedup)
        graph.add_task(SubTask(
            task_id='extract',
            description='Extract structure info from papers',
            agent_type='extractor',
            input_data={"query": query},
        ), depends_on=['dedup'])

        graph.add_task(SubTask(
            task_id='graph_analysis',
            description='Analyze citation network',
            agent_type='graph',
            input_data={"query": query},
        ), depends_on=['dedup'])

        # layer4-5: 综合 -> 审稿(串行)
        graph.add_task(SubTask(
            task_id='adversarial_review',
            description='Adversarial synthesis + review loop',
            agent_type='adversarial_review',
            input_data={"query": query},
            timeout_ms=300000,
            max_retries=0,   # 内部已有对抗循环 + 异常兜底，外层重试只会重跑同样的失败、产生重复 span
        ), depends_on=['extract', 'graph_analysis'])
        logger.info(f"Planned {len(graph.tasks)} tasks for query {query}")
        return graph


    async def _decompose(self, query: str) -> list[str]:
        fallback = [query]
        if not self._llm or not self._config.decompose_enabled:
            return fallback

        try:
            resp = await self._llm.chat(
                [
                    {'role': 'system', 'content': _DECOMPOSE_SYSTEM},
                    {'role': 'user', 'content': wrap_xml('topic', query)},
                ],
                response_format={'type': 'json_object'},
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
            )

            content = (resp.content or "").strip()

            if not content:
                logger.warning("Decompose: empty content, degrade to single query")
                return fallback

            raw = json.loads(content).get('sub_queries')
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
            logger.info(f'Decompose: {query!r} -> {len(clamped)} sub-queries')
            return clamped
        except Exception as e:
            logger.warning(f"Decompose failed ({e}), degrade to single query")
            return fallback


    async def _decompose_traced(self, query: str) -> list[str]:
        self._emit("subspan.start", {
            "task_id": "query_decomposition",
            "parent_task_id": "",            # 兜底到 root survey span
            "name": "planner.query_decomposition",
            "round": 0,
        })
        token = set_task_id("query_decomposition")
        sub_queries = [query]                # 预置兜底，异常也有值
        try:
            sub_queries = await self._decompose(query)
            return sub_queries
        finally:
            reset_task_id(token)
            self._emit("subspan.end", {
                "task_id": "query_decomposition",
                "output": {"sub_queries": sub_queries},
            })
