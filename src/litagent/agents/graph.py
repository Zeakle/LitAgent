"""Graph Worker--引用网络分析"""

from __future__ import annotations
from typing import Any

import httpx

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger


logger = get_logger("agents.graph")


class GraphWorker(Worker):
    """引用分析 Worker——调 citation API + 计算论文重要性 tier。

    纯算法，无 LLM。
    1. 从上游获取论文列表
    2. 查每篇论文的 citation count（如果上游没有的话）
    3. 按 citation count 分 tier（基石/高引/一般）
    4. 返回带 tier 标注的论文列表
    """

    TIER1_THRESHOLD = 500
    TIER2_THRESHOLD = 50

    @property
    def agent_type(self) -> str:
        return 'graph'

    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        papers = self._get_papers_from_upstream(upstream)

        tiered = self._assign_tiers(papers)

        tier_counts = {'tier1': 0, 'tier2': 0, 'tier3': 0}
        for p in tiered:
            tier_counts[f"tier{p['tier']}"] += 1
        logger.info(f'Graph: {len(tiered)} papers, tiers: {tier_counts}')

        return {
            'papers': tiered,
            'tier_counts': tier_counts,
            'seminal_papers': [p for p in tiered if p['tier'] == 1],
        }


    def _get_papers_from_upstream(self, upstream: dict) -> list[dict]:
        for task_id, result in upstream.items():
            if isinstance(result, list) and result:
                return result
        return []

    
    def _assign_tiers(self, papers: list[dict]) -> list[dict]:
        """按 citation_count 分 tier。不修改原始数据。

        Tier 1 (基石): citation >= 500
        Tier 2 (高引): citation >= 50
        Tier 3 (一般): 其余
        """
        result = []
        for p in papers:
            cc = p.get('citation_count', 0) or 0
            if cc >= self.TIER1_THRESHOLD:
                tier = 1
            elif cc >= self.TIER2_THRESHOLD:
                tier = 2
            else:
                tier = 3
            result.append({**p, 'tier': tier})
        return result