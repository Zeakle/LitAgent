"""AdversarialReviewWorker——封装 Synthesis↔Reviewer 对抗循环。"""

from __future__ import annotations
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.llm.client import BaseLLMClient
from litagent.agents.synthesis import SynthesisWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.logging import get_logger

logger = get_logger("agents.adversarial")


class AdversarialReviewWorker(Worker):
    """对抗审稿 Worker——管理 Synthesis↔Reviewer 的多轮循环。

    流程：
    1. Synthesis 生成初稿
    2. Reviewer 审稿
    3. 如果 score < threshold → Synthesis 修订 → Reviewer 再审（最多 max_rounds 轮）
    4. 返回终稿
    """

    def __init__(self, llm: BaseLLMClient, synthesis: SynthesisWorker, reviewer: ReviewerWorker, max_rounds: int = 3, pass_threshold: float = 0.8):
        self._synthesis = synthesis
        self._reviewer = reviewer
        self._llm = llm
        self._max_rounds = max_rounds 
        self._pass_threshold = pass_threshold

    
    @property
    def agent_type(self) -> str:
        return 'adversarial_review'

    
    async def execute(self, task: SubTask) -> Any:
        # Round 1: Synthesis生成初稿
        synthesis_result = await self._synthesis.execute(task)
        draft = synthesis_result['draft']

        rounds: list[dict] = []

        for round_num in range(1, self._max_rounds + 1):
            # Reviewer 审稿
            if round_num == 1:
                review_task = SubTask(
                    task_id=f'review_r{round_num}',
                    description=f'Review round {round_num}',
                    agent_type='reviewer',
                    input_data={'upstream_results': {'synthesis': {'draft': draft}}},
                )

                review = await self._reviewer.execute(review_task)
            else:
                review = await self._reviewer.review_revision(draft, rounds[-1]['review'])

            rounds.append({
                'round': round_num,
                'review': review,
                'draft_length': len(draft),
            })

            score = review.get('score', 0)
            verdict = review.get('verdict', 'revise')
            logger.info(f"Round {round_num}: score={score}, verdict={verdict}")

            # 通过或达到最大轮次
            if score >= self._pass_threshold or verdict == 'accept':
                logger.info(f"Accepted at round {round_num}")
                break

            if round_num < self._max_rounds:
                # Synthesis修订(llm失败时保留当前draft,循环自然退出)
                review_text = self._format_review_for_revision(review)
                messages = self._synthesis.revise(draft, review_text)
                try:
                    resp = await self._llm.chat(messages)
                    draft = resp.content
                except Exception as e:
                    logger.warning(f"Revision LLM call failed: {e}, keeping current draft")
                    break
                logger.info(f"Revision {round_num}: {len(draft)} chars")
        
        return {
            'final_draft': draft,
            'rounds': rounds,
            'total_rounds': len(rounds),
            "final_score": rounds[-1]["review"].get("score", 0),
            "accepted": (rounds[-1]["review"].get("verdict") == "accept"
                        or rounds[-1]["review"].get("score", 0) >= self._pass_threshold),
        }

    
    def _format_review_for_revision(self, review: dict) -> str:
        """将结构化 review 转为 Synthesis 可读的文本。"""
        parts = []
        if review.get("weaknesses"):
            parts.append("Weaknesses:\n" + "\n".join(f"- {w}" for w in review["weaknesses"]))
        if review.get("issues"):
            parts.append("Issues:\n" + "\n".join(
                f"- [{i.get('severity', 'minor')}] {i.get('section', '')}: {i.get('issue', '')}"
                for i in review["issues"]
            ))
        if review.get("missing_coverage"):
            parts.append("Missing:\n" + "\n".join(f"- {m}" for m in review["missing_coverage"]))
        return "\n\n".join(parts) if parts else "No specific feedback."