"""Synthesis Worker——综述初稿生成。"""

from __future__ import annotations
from typing import Any
import json

from litagent.agent.react import ReActRunner
from litagent.config import AgentConfig
from litagent.context.pipeline import ContextPipeline, ContextLayer
from litagent.context.budget import BudgetManager
from litagent.memory.manager import MemoryManager
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.llm.client import BaseLLMClient
from litagent.context.compressor import TierCompressor, PaperInfo
from litagent.context.templates import build_system_prompt, wrap_xml
from litagent.logging import get_logger
from litagent.skills.manager import SkillManager
from litagent.tools.worker_tools import make_load_skill_tool, make_recall_memory_tool
from litagent.evidence import collect_ledger, format_ledger


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

Use specific paper citations. Be objective and comprehensive.

CRITICAL EVIDENCE BOUNDARIES:
- Only reference papers explicitly listed in <papers> with their exact titles.
- If evidence is insufficient (<3 papers): write a SCOPED EVIDENCE SUMMARY stating
  the number of papers found, what conclusions they support, and what gaps remain.
  Do NOT use "comprehensive survey" language. Do NOT fabricate authors, years, or
  titles. For claims not supported by provided papers, write "evidence not provided".
- If no papers are provided, output a single paragraph explaining that the search
  returned no results, and suggest broader search terms.
  
EVIDENCE CITATION RULES:
- An <evidence_ledger> block lists every available evidence item as "[E:<id>] (<title>) <text>" lines.
- Every factual/empirical sentence MUST cite its supporting evidence inline with the
  exact marker, e.g. "ProtoNet reaches 93% on miniImageNet [E:1703.05175:claim:0]".
- Only use [E:...] ids that appear in the ledger — NEVER invent ids.
- Section titles, transitions, and hedged/general statements need no marker.
- Content with no supporting ledger item must be phrased as a known gap, not as fact."""


REVISE_INSTRUCTIONS = """Revise the survey draft based on the reviewer's feedback.
Address each criticism specifically. Keep existing good parts.
Never add new papers, titles, authors, or evidence ids to address missing coverage —
state it as a known gap instead. Keep all existing [E:...] markers accurate."""


REWRITE_INSTRUCTIONS = """Rewrite the survey to remove unsupported claims.
For each item in <unsupported_claims> you may ONLY do one of:
1. delete the claim, or
2. downgrade it to an explicit known gap ("evidence not provided"), or
3. bind it to an EXISTING ledger item by appending its exact [E:<id>] marker.
NEVER add new paper titles, authors, years, or evidence ids not in <evidence_ledger>.
Keep supported content and overall structure intact.
Return ONLY the rewritten survey text (no JSON, no commentary)."""


class SynthesisWorker(Worker):
    """综述生成 Worker。

    输入：论文 extraction 列表 + citation graph 分析
    输出：结构化综述初稿
    """

    def __init__(self, llm: BaseLLMClient, memory: MemoryManager | None = None,
                 budget: BudgetManager | None = None, skill_manager: SkillManager | None = None,
                 agent_config: AgentConfig | None = None):
        self._llm = llm
        self._compressor = TierCompressor()
        self._memory = memory
        self._budget = budget or BudgetManager(max_tokens=16000)
        self._tools = [make_recall_memory_tool(memory)] if memory else []
        self._skill_manager = skill_manager
        self._agent_config = agent_config or AgentConfig()
        if skill_manager:
            self._tools.append(make_load_skill_tool(skill_manager))

    
    @property
    def agent_type(self) -> str:
        return 'synthesis'


    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        extractions = self._get_extractions(upstream)

        graph_data = self._get_graph_data(upstream)
        query = task.input_data.get('query', "")

        # ContextPipeline 分层组装（按预算截断；papers 优先级高、memory 低）
        pipeline = ContextPipeline(self._budget)
        pipeline.add_layer(ContextLayer("papers", priority=0, max_tokens=12000,
                                        builder=self._papers_layer))
        pipeline.add_layer(ContextLayer('evidence_ledger', priority=1, max_tokens=5000,
                                        builder=self._ledger_layer))
        pipeline.add_layer(ContextLayer("memory", priority=2, max_tokens=2000,
                                        builder=self._memory_layer))

        user_msg, used = await pipeline.build({
            "extractions": extractions, "graph_data": graph_data, "query": query,
        })

        if len(extractions) < 3:
            user_msg = (
                f"<instruction>Only {len(extractions)} papers found. "
                f"Write a scoped evidence summary, not a comprehensive survey.</instruction>\n"
                + user_msg
            )

        skills_text = (self._skill_manager.to_metadata_text_for("writing a literature survey", top_k=2)
                       if self._skill_manager else "")

        system = build_system_prompt(
            role=SYNTHESIS_ROLE,
            instructions=SYNTHESIS_INSTRUCTIONS,
            skills=skills_text
        )

        runner = ReActRunner(self._llm, tools=self._tools, config=self._agent_config)
        result = await runner.run(system_prompt=system, user_message=user_msg)
        
        logger.info(f'Synthesis draft: {len(result)} chars, ctx {used} tokens')

        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return {'draft': result or '', 'error': 'JSON parse failed'}
    

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

    
    def rewrite_with_evidence(self, draft: str, ledger_text: str,
                              diagnostics_text: str) -> list[dict]:
        """构建 bounded rewrite 的 messages（供 runner 一次性调用）。"""
        system = build_system_prompt(
            role=SYNTHESIS_ROLE,
            instructions=REWRITE_INSTRUCTIONS
        )

        return [
            {'role': 'system', 'content': system},
            {
                'role': 'user', 'content': '\n\n'.join([
                    wrap_xml('current_draft', draft),
                    wrap_xml('evidence_ledger', ledger_text),
                    wrap_xml('unsupported_claims', diagnostics_text),
                ])
            }
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


    def _format_recall(self, recalled: dict) -> str:
        lines = []
        for ep in recalled.get('episodes', []):
            findings = "; ".join(ep.key_findings[:3]) if ep.key_findings else "none"
            lines.append(f"Previous session: {ep.summary} (findings: {findings})")
        for f in recalled.get("facts", []):
            lines.append(f"Known fact [{f.get('key')}]: {f.get('value')}")
        return '\n'.join(lines)


    async def _papers_layer(self, state: dict) -> str:
        ctx = self._build_papers_context(state["extractions"], state["graph_data"])
        return wrap_xml('papers', ctx) if ctx else ""

    
    async def _ledger_layer(self, state: dict) -> str:
        text = format_ledger(collect_ledger(state['extractions']))
        return wrap_xml('evidence_ledger', text) if text else ''


    async def _memory_layer(self, state: dict) -> str:
        if not self._memory:
            return ""
        recalled = await self._memory.recall(state["query"], top_k=5)
        text = self._format_recall(recalled)
        return wrap_xml('memory', text) if text else ""
