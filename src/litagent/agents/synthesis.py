"""Synthesis Worker——综述初稿生成。"""

from __future__ import annotations
from typing import Any

from litagent.memory.manager import MemoryManager
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.llm.client import BaseLLMClient
from litagent.context.compressor import TierCompressor, PaperInfo
from litagent.context.templates import build_system_prompt, wrap_xml
from litagent.logging import get_logger


logger = get_logger('agents.synthesis')

SYNTHESIS_ROLE = """You are an expert academic survey writer.
Your task is to write a comprehensive, well-structured literature review
based on the provided paper extractions and citation analysis."""

SYNTHESIS_INSTRUCTIONS = """Write a structured survey covering:
1. Introduction and motivation
2. Taxonomy of approaches
3. Detailed analysis of key methods
4. Experimental comparison
5. Open problems and future directions

Use specific paper citations. Be objective and comprehensive."""

REVISE_INSTRUCTIONS = """Revise the survey draft based on the reviewer's feedback.
Address each criticism specifically. Keep existing good parts."""


class SynthesisWorker(Worker):
    """综述生成 Worker。

    输入：论文 extraction 列表 + citation graph 分析
    输出：结构化综述初稿
    """

    def __init__(self, llm: BaseLLMClient, memory: MemoryManager | None = None):
        self._llm = llm
        self._compressor = TierCompressor()
        self._memory = memory

    
    @property
    def agent_type(self) -> str:
        return 'synthesis'


    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        extractions = self._get_extractions(upstream)
        graph_data = self._get_graph_data(upstream)
        query = task.input_data.get('query', "")

        papers_context = self._build_papers_context(extractions, graph_data)

        # Memory Recall
        memory_texdt = ""
        if self._memory:
            recalled = await self._memory.recall(query, top_k=5)
            memory_text = _format_recall(recalled)

        system = build_system_prompt(
            role=SYNTHESIS_ROLE,
            instructions=SYNTHESIS_INSTRUCTIONS,
        )
        user_parts = [wrap_xml('papers', papers_context)]
        if memory_text:
            user_parts.append(wrap_xml('memory', memory_text))
        user_msg = '\n\n'.join(user_parts)

        resp = await self._llm.chat([
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_msg}
        ])

        logger.info(f"Synthesis draft generated: {len(resp.content)} chars")
        return {
            'draft': resp.content,
            'usage': resp.usage,
        }

    
    def revise(self, draft: str, review_comments: str) -> list[dict]:
        """构建修订请求的 messages（供 AdversarialLoop 调用）。"""
        system = build_system_prompt(
            role=SYNTHESIS_ROLE,
            instructions = REVISE_INSTRUCTIONS,
        )

        return [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': '\n\n'.join([wrap_xml('current_draft', draft), wrap_xml('reviewer_feedback', review_comments)])}
        ]


    def _build_papers_context(self, extractions: list[dict], graph_data: dict) -> str:
        """用TierCompressor压缩论文数据为context"""
        papers = []
        tier_map = {}
        if graph_data and 'papers' in graph_data:
            for p in graph_data['papers']:
                tier_map[p.get('paper_id', "")] = p.get('tier', 3)

        for ext in extractions:
            pid = ext.get('paper_id', '')
            papers.append(PaperInfo(
                paper_id=pid,
                title=ext.get("title", ""),
                tier=tier_map.get(pid, 3),
                claims=ext.get("claims", []),
                metrics=ext.get("metrics", {}),
                summary=ext.get("abstract", "")[:200],
                tags=[],
                extraction={
                    k: str(v) for k, v in ext.items()
                    if k not in ('paper_id', 'title', 'abstract', 'claims', 'metrics')
                }
            ))

        return self._compressor.compress(papers)


    def _get_extractions(self, upstream: dict) -> list[dict]:
        for tid, result in upstream.items():
            if isinstance(result, list) and result and 'paper_id' in result[0]:
                return result
        return []

    
    def _get_graph_data(self, upstream: dict) -> dict:
        for tid, result in upstream.items():
            if isinstance(result, dict) and 'tier_counts' in result:
                return result
        return {}


    def _format_recall(recalled: dict) -> str:
        lines = []
        for ep in recalled.get('episodes', []):
            findings = "; ".join(ep.key_findings[:3]) if ep.key_findings else "none"
            lines.append(f"Previous session: {ep.summary} (findings: {findings})")
        for f in recalled.get("facts", []):
            lines.append(f"Known fact [{f.get('key')}]: {f.get('value')}")
        return '\n'.join(lines)