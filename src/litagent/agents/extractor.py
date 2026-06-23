"""Extractor Worker——结构化提取。"""


from __future__ import annotations
import re
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger
from litagent.rag.claims_index import ClaimsIndex, Claim
from litagent.tools.executor import ToolExecutor


logger = get_logger('agents.extractor')


class ExtractorWorker(Worker):
    """提取 Worker——从论文列表中提取结构化信息。
    """
    def __init__(self, executor: ToolExecutor, claims_index: ClaimsIndex | None = None):
        self._executor = executor
        self._claims_index = claims_index

    @property
    def agent_type(self) -> str:
        return 'extractor'

    
    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        papers = self._get_papers_from_upstream(upstream)

        extractions = []
        for paper in papers:
            title = paper.get('abstract', '')
            abstract = paper.get('abstract', '')
            text = f'{title} {abstract}'.lower()
            claims_r = await self._executor.execute("extract_claims", {"abstract": abstract})
            metrics_r = await self._executor.execute("extract_metrics", {"abstract": abstract})
            methods_r = await self._executor.execute("extract_methods", {"text": text})
            datasets_r = await self._executor.execute("extract_datasets", {"text": text})
            extractions.append({
                "paper_id": paper.get("paper_id", ""), "title": title, "abstract": abstract,
                "claims": claims_r.output if not claims_r.error else [],
                "metrics": metrics_r.output if not metrics_r.error else {},
                "methods": methods_r.output if not methods_r.error else [],
                "datasets": datasets_r.output if not datasets_r.error else [],
                "citation_count": paper.get("citation_count", 0),
                "source": paper.get("source", ""),
            })

        # 写入claim index
        if self._claims_index:
            try:
                claims_objs = []
                for ext in extractions:
                    for c in ext.get("claims", []):
                        claims_objs.append(Claim(
                            text=c, source_paper=ext.get('paper_id', ""),
                            confidence=0.5))
                if claims_objs:
                    await self._claims_index.add(claims_objs)
            except Exception as e:
                logger.warning(f"Claimsindex write failed: {e}")

        return extractions


    def _get_papers_from_upstream(self, upstream: dict) -> list[dict]:
        """从上游结果中获取论文列表(dedup的输出)"""
        for task_id, result in upstream.items():
            if isinstance(result, list) and result:
                return result
        return []