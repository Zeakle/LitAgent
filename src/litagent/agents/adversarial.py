"""AdversarialReviewWorker——封装 Synthesis↔Reviewer 对抗循环。"""

from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Any
from collections.abc import Mapping

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.llm.client import BaseLLMClient
from litagent.agents.synthesis import SynthesisWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.observability.context import set_task_id, reset_task_id
from litagent.context.evidence_selector import EvidenceSelection
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
        if not isinstance(draft, str) or not draft.strip():
            raise ValueError("synthesis returned an empty draft")
        draft = draft.strip()

        selection_raw = synthesis_result.get('evidence_selection')
        evidence_selection: EvidenceSelection | None = None
        evidence_selection_valid = True

        if isinstance(selection_raw, Mapping):
            try:
                evidence_selection = EvidenceSelection.from_dict(selection_raw)
            except (ValueError, TypeError) as e:
                logger.warning(f"Malformed evidence_selection from synthesis: {e}")
                evidence_selection_valid = False
        else:
            evidence_selection_valid = False

        if evidence_selection is None:
            evidence_selection = EvidenceSelection(
                candidate_count=0, selected_items={},
                section_evidence_ids={}, method_by_section={},
                omitted_count=0, estimated_tokens=0,
            )

        logger.info(
            "Adversarial start: draft=%d chars selection_items=%d/%d",
            len(draft), len(evidence_selection.selected_items),
            evidence_selection.candidate_count,
        )

        rounds: list[dict] = []

        try:
            for round_num in range(1, self._max_rounds + 1):
                async with self._sub_span(task.task_id, "reviewer", round_num):
                    if round_num == 1:
                        review_task = SubTask(
                            task_id=task.task_id,
                            description=f"Review round {round_num}",
                            agent_type="reviewer",
                            input_data={"upstream_results": {
                                "synthesis": {
                                    "draft": draft,
                                    "evidence_selection": evidence_selection.to_dict(),
                                },
                            }},
                        )
                        review = await self._reviewer.execute(review_task)
                    else:
                        review = await self._reviewer.review_revision(
                            draft, rounds[-1]["review"], evidence_selection,
                        )

                rounds.append({
                    'round': round_num, 'review': review, 'draft_length': len(draft)
                })

                score = review.get('score', 0)
                verdict = review.get('verdict', 'revise')

                ev_passed = (
                    review.get('evidence_compliance', {}).get('passed', False)
                    if isinstance(review.get('evidence_compliance'), dict)
                    else False
                )
                logger.info(
                    f"Round {round_num}: score={score}, verdict={verdict}, "
                    f"evidence_passed={ev_passed}"
                )

                if self._review_passes(review):
                    logger.info(f"Accepted at round {round_num}")
                    break

                if round_num < self._max_rounds:
                    review_text = self._format_review_for_revision(review)
                    try:
                        messages = self._synthesis.revise(
                            draft, review_text, evidence_selection
                        )
                    except (ValueError, TypeError) as e:
                        logger.warning(f"revise rejected malformed selection: {e}")
                        break

                    try:
                        async with self._sub_span(task.task_id, "synthesis", round_num + 1):
                            resp = await self._llm.chat(messages)

                        new_draft = (resp.content or '').strip()
                        if not new_draft:
                            logger.warning(
                                "Revision LLM returned empty content, keeping current draft"
                            )
                            break
                        draft = new_draft
                    except Exception as e:
                        logger.warning(f"Revision LLM call failed: {e}, keeping current draft")
                        break
                    logger.info(f"Revision {round_num}: {len(draft)} chars")
        except Exception as e:
            logger.warning(f"Adversarial loop aborted ({e}), returning current draft as fallback")

        last_review = rounds[-1]['review'] if rounds else {}

        #selection_summary
        sel_summary = {
            "valid": evidence_selection_valid,
            "candidate_count": evidence_selection.candidate_count,
            "selected_count": len(evidence_selection.selected_items),
            "section_evidence_ids": evidence_selection.section_evidence_ids,
            "method_by_section": evidence_selection.method_by_section,
            "omitted_count": evidence_selection.omitted_count,
            "estimated_tokens": evidence_selection.estimated_tokens,
        }

        return {
            "final_draft": draft,
            "rounds": rounds,
            "total_rounds": len(rounds),
            "final_score": last_review.get("score", 0),
            "accepted": (
                self._review_passes(last_review)
                if last_review else False
            ),
            "evidence_selection": sel_summary,
        }

    def _review_passes(self, review: dict[str, Any]) -> bool:
        """Single accept-gate helper — AND semantics: score + verdict + evidence.

        Used by both the loop break and final return to prevent condition drift.
        """
        compliance = review.get('evidence_compliance')
        return (
            review.get('score', 0) >= self._pass_threshold
            and review.get('verdict') == 'accept'
            and isinstance(compliance, dict)
            and compliance.get('passed') is True
        )

    def _format_review_for_revision(self, review: dict) -> str:
        """Convert structured review to revision-readable text.

        Includes evidence_compliance unsupported_claims and unknown_evidence_ids
        so Synthesis revision knows exactly what to delete/downgrade/bind.
        """
        parts: list[str] = []
        if review.get("weaknesses"):
            parts.append("Weaknesses:\n" + "\n".join(
                f"- {w}" for w in review["weaknesses"]
            ))
        if review.get("issues"):
            parts.append("Issues:\n" + "\n".join(
                f"- [{i.get('severity', 'minor')}] {i.get('section', '')}: {i.get('issue', '')}"
                for i in review["issues"]
            ))
        if review.get("missing_coverage"):
            parts.append("Missing:\n" + "\n".join(
                f"- {m}" for m in review["missing_coverage"]
            ))

        # Evidence compliance — tell synthesis exactly what to fix
        compliance = review.get("evidence_compliance")
        if isinstance(compliance, dict):
            unknown = compliance.get("unknown_evidence_ids", [])
            if isinstance(unknown, list) and unknown:
                parts.append(
                    "Unknown evidence IDs (remove or rebind these):\n"
                    + "\n".join(f"- [E:{eid}]" for eid in unknown)
                )
            unsupported = compliance.get("unsupported_claims", [])
            if isinstance(unsupported, list) and unsupported:
                safe_claims = [c for c in unsupported if isinstance(c, Mapping)]
                if safe_claims:
                    parts.append("Unsupported claims:\n" + "\n".join(
                        f"- [{c.get('section', '?')}] {c.get('claim', '')}"
                        f"  reason: {c.get('reason', '')}"
                        f"  action: {c.get('action', 'delete')}"
                        for c in safe_claims
                    ))

        return "\n\n".join(parts) if parts else "No specific feedback."
