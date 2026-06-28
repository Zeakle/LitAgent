"""Extractor Worker——LLM 异步提取，regex 降级。"""

from __future__ import annotations
import asyncio
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.agents.extraction_strategy import ExtractionStrategy
from litagent.rag.claims_index import ClaimsIndex, Claim
from litagent.logging import get_logger
from litagent.safety.injection import InjectionDetector, InjectionRisk

logger = get_logger("agents.extractor")


class ExtractorWorker(Worker):
    """提取 Worker——从论文列表中提取结构化信息。
    """
    def __init__(self, strategy: ExtractionStrategy, claims_index: ClaimsIndex | None = None, max_concurrent: int = 5, detector: InjectionDetector = None):
        self._strategy = strategy
        self._claims_index = claims_index
        self._sem = asyncio.Semaphore(max_concurrent)
        self._detector = detector

    @property
    def agent_type(self) -> str:
        return 'extractor'

    
    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        papers = self._get_papers_from_upstream(upstream)

        if self._detector:
            safe = []
            for p in papers:
                res = self._detector.scan(f"{p.get('title', '')} {p.get('abstract', '')}")
                if res.risk == InjectionRisk.HIGH:
                    logger.warning(f"Skip paper {p.get('paper_id', '?')}: injection in content")
                    continue
                safe.append(p)
            papers = safe

        results = await asyncio.gather(*[self._extract_one(p) for p in papers],
                                        return_exceptions=True)

        extractions = []
        for paper, r in zip(papers, results):
            if isinstance(r, BaseException):
                logger.warning(f"Extraction fully failed for {paper.get('paper_id','?')}: {r}")
                continue
            # 补齐权威元信息（策略只产出提取字段，不产出 paper_id/title 等）
            r.update({
                'paper_id': paper.get('paper_id', ""),
                'title': paper.get('title', ''),
                'abstract': paper.get('abstract', ''),
                'citation_count': paper.get('citation_count', 0),
                'source': paper.get('source', ''),
            })
            extractions.append(r)

        if self._claims_index:
            try:
                claims_objs = [
                    Claim(text=c, source_paper=ext.get('paper_id', ""), confidence=0.5)
                    for ext in extractions for c in ext.get('claims', [])
                ]
                if claims_objs:
                    await self._claims_index.add(claims_objs)
            except Exception as e:
                logger.warning(f"ClaimsIndex write failed: {e}")

        return extractions


    def _get_papers_from_upstream(self, upstream: dict) -> list[dict]:
        """从上游结果中获取论文列表(dedup的输出)"""
        for task_id, result in upstream.items():
            if isinstance(result, list) and result:
                return result
        return []


    async def _extract_one(self, paper: dict) -> dict:
        async with self._sem:
            return await self._strategy.extract(paper)