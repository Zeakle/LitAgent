"""Report Worker--生成最后结构化报告"""


from __future__ import annotations
import time
from typing import Any

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger


logger = get_logger('agents.report')


class ReportWorker(Worker):
    """报告生成 Worker——汇总上游结果，输出结构化报告。

    纯模板拼接，无 LLM。
    """

    @property
    def agent_type(self) -> str:
        return 'report'

    async def execute(self, task: SubTask) -> Any:
        upstream = task.input_data.get('upstream_results', {})
        review_data = self._get_review_data(upstream)

        report = {
            'survey': review_data.get('final_draft', ""),
            'metadata': {
                'generated_at': time.time(),
                'total_rounds': review_data.get('total_rounds', 0),
                'final_score': review_data.get('final_score', 0),
                'accepted': review_data.get('accepted', False),
            },
            'review_history': self._format_review_history(review_data.get('rounds', [])),
        }

        logger.info(
            f"Report generated: {len(report['survey'])} chars, "
            f"{report['metadata']['total_rounds']} review rounds, "
            f"accepted={report['metadata']['accepted']}"
        )
        return report


    def _get_review_data(self, upstream: dict) -> dict:
        """从上游获取AdversarialReviewWorker的输出"""
        for tid, result in upstream.items():
            if isinstance(result, dict) and 'final_draft' in result:
                return result
        return {}


    def _format_review_history(self, rounds: list[dict]) -> list[dict]:
        """格式化审稿历史--每一轮保存score + verdict + 主要问题"""
        history = []
        for r in rounds:
            review = r.get('review', {})
            history.append({
                'round': r.get('round', 0),
                'score': review.get('score', 0),
                'verdict': review.get('verdict', 'unknown'),
                'weaknesses': review.get('weaknesses', []),
                'issue_count': len(review.get('issues', [])),
            })
        return history