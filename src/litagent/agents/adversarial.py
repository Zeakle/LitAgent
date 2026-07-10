"""AdversarialReviewWorker——封装 Synthesis↔Reviewer 对抗循环。"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.llm.client import BaseLLMClient
from litagent.agents.synthesis import SynthesisWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.observability.context import set_task_id, reset_task_id
from litagent.logging import get_logger

logger = get_logger("agents.adversarial")


class AdversarialReviewWorker(Worker):
    """对抗审稿 Worker——管理 Synthesis↔Reviewer 的多轮循环。

    流程：
    1. Synthesis 生成初稿
    2. Reviewer 审稿
    3. 如果 score < threshold → Synthesis 修订 → Reviewer 再审（最多 max_rounds 轮）
    4. 返回终稿

    追踪：synthesis/reviewer 是内部直接调用（不经 Scheduler），默认它们的 llm.call
    会全挂到 adversarial span 下（扁平）。用 _sub_span 包裹调用，为每轮的
    synthesis/reviewer 建独立子 span，让 llm.call 归到各自节点（可读性）。
    """

    def __init__(self, llm: BaseLLMClient, synthesis: SynthesisWorker, reviewer: ReviewerWorker,
                 max_rounds: int = 3, pass_threshold: float = 0.8, trace_hook=None):
        self._synthesis = synthesis
        self._reviewer = reviewer
        self._llm = llm
        self._max_rounds = max_rounds
        self._pass_threshold = pass_threshold
        self._trace_hook = trace_hook


    @property
    def agent_type(self) -> str:
        return 'adversarial_review'


    def _emit(self, event: str, data: dict) -> None:
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")


    @asynccontextmanager
    async def _sub_span(self, parent_task_id: str, component: str, round_num: int):
        """为内部 synthesis/reviewer 调用建独立子 span。

        emit subspan.start（挂 parent span 下）→ 临时切 contextvar 到子 task_id
        （子调用内 llm.call 用它 → 归到本子 span）→ 结束 emit subspan.end + reset。
        用 context manager 保证异常路径也 reset/关闭（配对逻辑不漏）。
        """
        sub_tid = f"{parent_task_id}:{component}:r{round_num}"
        self._emit('subspan.start', {
            'task_id': sub_tid, 'parent_task_id': parent_task_id,
            'name': f"{component}.r{round_num}", 'round': round_num,
        })
        token = set_task_id(sub_tid)
        error = None
        try:
            yield
        except Exception as e:
            error = str(e)
            raise
        finally:
            reset_task_id(token)
            self._emit('subspan.end', {'task_id': sub_tid, 'error': error})


    async def execute(self, task: SubTask) -> Any:
        # Round 1: Synthesis生成初稿。
        # 初稿不包在保底 try 内——若连初稿都生不出，确实无稿可返，让异常抛出（task 真失败）。
        async with self._sub_span(task.task_id, 'synthesis', 1):
            synthesis_result = await self._synthesis.execute(task)
        draft = synthesis_result['draft']          # 保底稿：后续任何步骤失败都返回它

        rounds: list[dict] = []

        # 对抗循环（reviewer + revision）整体包 try：任何一步失败只是"没优化成"，
        # 不该丢弃已生成的初稿。异常 → 记录 → 退出循环 → 返回当前最好的 draft。
        try:
            for round_num in range(1, self._max_rounds + 1):
                # Reviewer 审稿
                async with self._sub_span(task.task_id, 'reviewer', round_num):
                    if round_num == 1:
                        review_task = SubTask(
                            task_id=task.task_id,
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
                        async with self._sub_span(task.task_id, 'synthesis', round_num + 1):
                            resp = await self._llm.chat(messages)
                        draft = resp.content
                    except Exception as e:
                        logger.warning(f"Revision LLM call failed: {e}, keeping current draft")
                        break
                    logger.info(f"Revision {round_num}: {len(draft)} chars")
        except Exception as e:
            # 对抗循环中断（如 reviewer ReAct 因 reasoning_content 400）——保底返回初稿。
            logger.warning(f"Adversarial loop aborted ({e}), returning current draft as fallback")

        # rounds 可能为空（reviewer 首轮就崩）→ 防 rounds[-1] 越界
        last_review = rounds[-1]["review"] if rounds else {}
        return {
            'final_draft': draft,
            'rounds': rounds,
            'total_rounds': len(rounds),
            "final_score": last_review.get("score", 0),
            "accepted": (last_review.get("verdict") == "accept"
                        or last_review.get("score", 0) >= self._pass_threshold),
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