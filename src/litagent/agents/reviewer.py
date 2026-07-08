"""Reviewer Worker--对抗审稿"""

from __future__ import annotations
import json
import re
from typing import Any

from litagent.tools.worker_tools import make_lookup_claims_tool
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.llm.client import BaseLLMClient
from litagent.context.templates import build_system_prompt, wrap_xml
from litagent.context.pipeline import ContextPipeline, ContextLayer
from litagent.context.budget import BudgetManager
from litagent.logging import get_logger
from litagent.rag.claims_index import ClaimsIndex
from litagent.agent.react import ReActRunner
from litagent.config import AgentConfig


logger = get_logger('agents.reviewer')

REVIEWER_ROLE = """You are a senior reviewer for top AI conferences (NeurIPS, ICML, ICLR).
You review academic surveys with rigorous standards."""

REVIEWER_INSTRUCTIONS = """Review the survey draft and provide:
1. Overall score (0.0-1.0, where 0.8+ means acceptable)
2. Strengths (what is done well)
3. Weaknesses (what needs improvement)
4. Specific issues (list each with section reference)
5. Missing coverage (important papers or topics not addressed)

Respond in JSON format:
{
    "score": 0.0-1.0,
    "strengths": ["..."],
    "weaknesses": ["..."],
    "issues": [{"section": "...", "issue": "...", "severity": "major|minor"}],
    "missing_coverage": ["..."],
    "verdict": "accept|revise|reject"
}"""


class ReviewerWorker(Worker):
    """审稿Worker

    输入: 综述初稿
    输出：审稿意见(JSON)
    """

    def __init__(self, llm: BaseLLMClient, claims_index: ClaimsIndex | None = None,
                 budget: BudgetManager | None = None, trace_hook=None):
        self._llm = llm
        self._claims_index = claims_index
        self._budget = budget or BudgetManager(max_tokens=16000)
        self._tools = [make_lookup_claims_tool(claims_index)] if claims_index else []
        self._trace_hook = trace_hook


    @property
    def agent_type(self) -> str:
        return 'reviewer'


    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        draft = self._get_draft(upstream)

        # ContextPipeline 分层：draft 优先级高（必保），claims 低（超预算先截）
        pipeline = ContextPipeline(self._budget)
        pipeline.add_layer(ContextLayer("draft", priority=0, max_tokens=12000,
                                        builder=self._draft_layer))
        pipeline.add_layer(ContextLayer("claims", priority=1, max_tokens=3000,
                                        builder=self._claims_layer))
        user_msg, used = await pipeline.build({"draft": draft})

        system = build_system_prompt(
            role=REVIEWER_ROLE,
            instructions=REVIEWER_INSTRUCTIONS + "\nCross-reference related claims from other papers against the draft for completeness.")
            
        runner = ReActRunner(self._llm, tools=self._tools,
                             config=AgentConfig(max_loops=10), trace_hook=self._trace_hook, task_id=task.task_id)
        result = await runner.run(system_prompt=system, user_message=user_msg)

        review = self._parse_review(result)
        logger.info(f"Review score: {review.get('score', 'N/A')}, verdict: {review.get('verdict', 'N/A')}")
        return review


    async def _draft_layer(self, state: dict) -> str:
        return wrap_xml('survey_draft', state["draft"])


    async def _claims_layer(self, state: dict) -> str:
        draft = state["draft"]
        if not (self._claims_index and draft):
            return ""
        try:
            related = []
            for phrase in self._extract_key_phrases(draft):
                for c in await self._claims_index.search(phrase, top_k=5):
                    if c.text.lower() not in draft.lower():
                        related.append(c.text)
            if related:
                return wrap_xml('related_claims', '\n'.join(f"- {c}" for c in related[:10]))
        except Exception as e:
            logger.warning(f"ClaimsIndex search failed: {e}")
        return ""

    
    async def review_revision(self, revised_draft: str, previous_review: dict) -> dict:
        """审查修订稿（供 AdversarialLoop 调用）。"""
        system = build_system_prompt(
            role=REVIEWER_ROLE,
            instructions=REVIEWER_INSTRUCTIONS + "\n\nThis is a REVISION. "
                         "Check if previous issues were addressed. "
                         "Be fair: if issues are fixed, raise the score.",
        )

        user_msg = (
            '\n\n'.join(
                [
                    wrap_xml("revised_draft", revised_draft),
                    wrap_xml("previous_review", json.dumps(previous_review, ensure_ascii=False))
                ]
            )
        )

        resp = await self._llm.chat(
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_msg},
            ],
            response_format={'type': 'json_object'}
        )

        return self._parse_review(resp.content)


    def _parse_review(self, content: str) -> dict:
        """解析 JSON 审稿意见。response_format 保证输出是合法JSON"""
        try:
            parsed = json.loads(content)
            parsed['score'] = float(parsed.get('score', 0))
            return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse review JSON, using defaults")
        return {
            "score": 0.3,
            "strengths": [],
            "weaknesses": ["Review parsing failed"],
            "issues": [],
            "missing_coverage": [],
            "verdict": "revise",
        }


    def _get_draft(self ,upstream: dict) -> str:
        for tid, result in upstream.items():
            if isinstance(result, dict) and 'draft' in result:
                return result['draft']
        return ""


    def _extract_key_phrases(self, draft: str) -> list[str]:
        """提取包含指标/声明的关键句作为搜索短语。"""
        sentences = re.split(r'(?<=[.!?])\s+', draft)
        return [s.strip() for s in sentences
                if any(kw in s.lower() for kw in ["achieve", "%", "outperform", "state-of-the-art", "sota"])][:10]