"""Planner Agent——查询分解 + TaskGraph 生成。"""

from __future__ import annotations
from abc import ABC, abstractmethod

from litagent.orchestrator.task_graph import TaskGraph, SubTask
from litagent.logging import get_logger

logger = get_logger("agents.planner")


class BasePlanner(ABC):
    """Planner 接口 """

    @abstractmethod
    async def plan(self, query: str) -> TaskGraph:
        ...


class SurveyPlanner(BasePlanner):
    """标准综述流程的规则 Planner。

    生成固定 DAG 结构（对应 Plan.md 中的综述流程）：
        search_arxiv  ──┐
        search_semsch ──┼── dedup → extract + graph (并行) → synthesis → review → report
        search_pwc    ──┘

    Phase 8b+ 可替换为 LLM-based Planner，根据 query 动态生成不同的 DAG。
    """

    async def plan(self, query: str) -> TaskGraph:
        graph = TaskGraph()

        # layer1: 三个并行搜索（无依赖)
        graph.add_task(SubTask(
            task_id='search_arxiv',
            description=f"Search arxiv for: {query}",
            agent_type='search',
            input_data={"source": "arxiv", "query": query}
        ))

        graph.add_task(SubTask(
            task_id='search_semscholar',
            description=f'Search Semantic Scholar for: {query}',
            agent_type='search',
            input_data={'source': 'semantic_scholar', 'query': query},
        ))

        graph.add_task(SubTask(
            task_id='search_pwc',
            description=f'Search PaperWithCode for: {query}',
            agent_type='search',
            input_data={'source': 'paperswithcode', 'query': query},
        ))

        # layer2: 搜索结果去重
        graph.add_task(SubTask(
            task_id='dedup',
            description='Deduplicate search results',
            agent_type='dedup',
            input_data={},
        ), depends_on=['search_arxiv', 'search_semscholar', 'search_pwc'])

        # layer3: 提取 + 引用分析(并行，只等dedup)
        graph.add_task(SubTask(
            task_id='extract',
            description='Extract structure info from papers',
            agent_type='extractor',
            input_data={},
        ), depends_on=['dedup'])

        graph.add_task(SubTask(
            task_id='graph_analysis',
            description='Analyze citation network',
            agent_type='graph',
            input_data={},
        ), depends_on=['dedup'])

        # layer4-6: 综合 -> 审稿 -> 报告(串行)
        graph.add_task(SubTask(
            task_id='adversarial_review',
            description='Adversarial synthesis + review loop',
            agent_type='adversarial_review',
            input_data={},
            timeout_ms=300000
        ), depends_on=['extract', 'graph_analysis'])


        graph.add_task(SubTask(
            task_id='report',
            description='Generate final report',
            agent_type='report',
            input_data={},
        ), depends_on=['adversarial_review'])

        logger.info(f"Planned {len(graph.tasks)} tasks for query {query}")
        return graph