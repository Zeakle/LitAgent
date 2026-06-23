"""Search Worker--调API搜索论文"""


from __future__ import annotations
from typing import Any

import httpx

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger
from litagent.tools.executor import ToolExecutor

logger = get_logger("agents.search")


class SearchWorker(Worker):
    """搜索Worker--根据 input_data['source']路由到对应的数据源"""

    def __init__(self, executor: ToolExecutor):
        self._executor = executor

    @property
    def agent_type(self) -> str:
        return 'search'

    
    async def execute(self, task: SubTask) -> Any:
        source = task.input_data.get('source', 'arxiv')
        query = task.input_data.get('query', '')

        tool_map = {
            'arxiv': 'search_arxiv',
            'semantic_scholar': 'search_semantic_scholar',
            'paperswithcode': 'search_paperswithcode'
        }
        tool_name = tool_map.get(source, 'search_arxiv')  # default arxiv

        try:
            result = await self._executor.execute(tool_name, {'query': query, 'max_results': 20})
            if result.error:
                logger.warning(f"Search tool '{tool_name}' error: {result.error}, returning empty")
                return []
            return result.output
        except Exception:
            logger.warning(f'Search failed for {source}, returning empty')
            return []