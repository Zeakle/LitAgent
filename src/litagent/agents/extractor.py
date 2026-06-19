"""Extractor Worker——结构化提取。"""


from __future__ import annotations
import re
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger


logger = get_logger('agents.extractor')


class ExtractorWorker(Worker):
    """提取 Worker——从论文列表中提取结构化信息。
    """

    @property
    def agent_type(self) -> str:
        return 'extractor'

    
    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        papers = self._get_papers_from_upstream(upstream)

        extractions = []
        for paper in papers:
            extraction = self._extract_from_metadata(paper)
            extractions.append(extraction)

        logger.info(f'Extracted {len(extractions)} papers')
        return extractions
    

    def _get_papers_from_upstream(self, upstream: dict) -> list[dict]:
        """从上游结果中获取论文列表(dedup的输出)"""
        for task_id, result in upstream.items():
            if isinstance(result, list) and result:
                return result
        return []

    
    def _extract_from_metadata(self, paper: dict) -> dict:
        """从 title + abstract提取结构化信息(规则版)"""
        title = paper.get('title', "")
        abstract = paper.get('abstract', "")
        text = f"{title} {abstract}".lower()

        return {
            "paper_id": paper.get("paper_id", ""),
            "title": title,
            "abstract": abstract,
            "claims": self._extract_claims(abstract),
            "metrics": self._extract_metrics(abstract),
            "methods": self._extract_methods(text),
            "datasets": self._extract_datasets(text),
            "citation_count": paper.get("citation_count", 0),
            "source": paper.get("source", ""),
        }

    
    def _extract_claims(self, abstract: str) -> list[str]:
        """提取声明——包含 achieve/outperform/surpass/state-of-the-art 的句子。"""
        if not abstract:
            return []
        sentences = re.split(r'(?<=[.!?])\s+', abstract)
        claim_keywords = ["achieve", "outperform", "surpass", "state-of-the-art",
                          "sota", "best", "novel", "first", "superior"]
        claims = []
        for s in sentences:
            if any(kw in s.lower() for kw in claim_keywords):
                claims.append(s.strip())
        return claims[:5]

    
    def _extract_metrics(self, abstract: str) -> dict[str, str]:
        """提取数值指标--匹配xx.x%模式"""
        metrics = {}
        for m in re.finditer(r'(\w+)\s*(?:of|=|:)\s*(\d+\.?\d*)\s*%', abstract):
            metrics[m.group(1).lower()] = f"{m.group(2)}%"
        return metrics

    
    def _extract_methods(self, text: str) -> list[str]:
        """提取方法名——常见的模型/方法关键词。"""
        method_patterns = [
            r'(?:propose|introduce|present)\s+(?:a\s+)?(\w+(?:\s+\w+){0,2})',
        ]
        methods = []
        for pattern in method_patterns:
            for m in re.finditer(pattern, text):
                methods.append(m.group(1).strip())
        return methods[:3]

    
    def _extract_datasets(self, text: str) -> list[str]:
        """提取数据集名。"""
        known_datasets = [
            "imagenet", "miniImageNet", "tieredImageNet", "cifar", "cub-200",
            "coco", "voc", "meta-dataset", "omniglot",
        ]
        found = [d for d in known_datasets if d.lower() in text]
        return found