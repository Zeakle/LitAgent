"""Dedup Worker---搜索结果去重"""

from __future__ import annotations
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger


logger = get_logger('agent.dedup')


class DedupWorker(Worker):
    """去重Worker--合并3个数据源的搜索结果"""

    @property
    def agent_type(self) -> str:
        return 'dedup'

    
    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})

        all_papers: list[dict] = []
        for task_id, papers in upstream.items():
            if isinstance(papers, list):
                all_papers.extend(papers)
        
        deduped = self._dedup_by_title(all_papers)
        logger.info(f"Dedup: {len(all_papers)} -> {len(deduped)} papers")
        return deduped

        
    def _dedup_by_title(self, papers: list[dict]) -> list[dict]:
        """按标题去重。标题转小写后去空格作为 key。

        同一篇论文在不同数据源的标题可能有微小差异（大小写、空格），
        归一化后匹配。保留 citation_count 最高的那条。
        """
        seen: dict[str, dict] = {}
        for p in papers:
            key = p.get('title', '').lower().strip()
            if not key:
                continue
            if key in seen:
                if p.get('citation_count', 0) > seen[key].get('citation_count', 0):
                    seen[key] = p
            else:
                seen[key] = p
        
        return list(seen.values())