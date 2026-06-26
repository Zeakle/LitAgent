"""Reviewer Worker--对抗审稿"""

from __future__ import annotations
import json
import re
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.llm.client import BaseLLMClient
from litagent.context.templates import build_system_prompt, wrap_xml
from litagent.logging import get_logger
from litagent.rag.claims_index import ClaimsIndex


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

    def __init__(self, llm: BaseLLMClient, claims_index: ClaimsIndex | None = None):
        self._llm = llm
        self._claims_index = claims_index


    @property
    def agent_type(self) -> str:
        return 'reviewer'


    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        draft = self._get_draft(upstream)

        # 交叉验证: 搜索Claims Index
        related_claims_text = ""
        if self._claims_index and draft:
            try:
                phrases = self._extract_key_phrases(draft)
                related = []
                for phrase in phrases:
                    similar = await self._claims_index.search(phrase, top_k=5)
                    for c in similar:
                        if c.text.lower() not in draft.lower():
                            related.append(c.text)

                if related:
                    related_claims_text = '\n'.join(f"- {c}" for c in related[:10])
            except Exception as e:
                logger.warning(f"ClaimsIndex search failed: {e}")

        system = build_system_prompt(
            role=REVIEWER_ROLE,
            instructions=REVIEWER_INSTRUCTIONS + "\nCross-reference related claims from other papers against the draft for completeness.")
        user_parts = [wrap_xml('survey_draft', draft)]

        if related_claims_text:
            user_parts.append(wrap_xml('related_claims', related_claims_text))
        user_msg = '\n\n'.join(user_parts)

        resp = await self._llm.chat(
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': user_msg}
            ],
            response_format={'type': 'json_object'}
        )

        review = self._parse_review(resp.content)
        logger.info(f"Review score: {review.get('score', 'N/A')}, verdict: {review.get('verdict', 'N/A')}")
        return review

    
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