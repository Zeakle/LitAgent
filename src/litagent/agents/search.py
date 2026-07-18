"""Search Worker--调API搜索论文"""


from __future__ import annotations

from __future__ import annotations
import asyncio
from typing import Any
import time as _time

from litagent.memory.manager import MemoryManager
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger
from litagent.tools.executor import ToolExecutor

logger = get_logger("agents.search")


class SearchWorker(Worker):
    """搜索Worker--根据 input_data['source']路由到对应的数据源"""

    def __init__(self, executor: ToolExecutor, memory_manager: MemoryManager | None = None):
        self._executor = executor
        self._memory = memory_manager


    @property
    def agent_type(self) -> str:
        return 'search'


    async def execute(self, task: SubTask) -> Any:
        source = task.input_data.get('source', 'arxiv')
        query = task.input_data.get('query', '')

        # soruce -> tool路由
        tool_map = {
            'arxiv': 'search_arxiv',
            'semantic_scholar': 'search_semantic_scholar',
            'huggingface': 'search_huggingface'
        }
        tool_name = tool_map.get(source, 'search_arxiv')  # default arxiv

        t0 = _time.perf_counter()

        api_result = await self._executor.execute(
            tool_name, {'query': query, 'max_results': 20}
        )
        elapsed_ms = int((_time.perf_counter() - t0) * 1000)
        api_papers = (api_result.output or []) if not api_result.error else []

        profile_source = source if source in tool_map else 'arxiv'
        if self._memory:
            try:
                has_error = bool(api_result.error)
                is_empty = (not has_error) and len(api_papers) == 0
                err_type: str | None = None
                if has_error:
                    err_str = str(api_result.error)
                    if "429" in err_str or "rate limit" in err_str.lower():
                        err_type = "rate_limit"
                    elif "timeout" in err_str.lower():
                        err_type = "timeout"
                    else:
                        err_type = "tool_error"

                await self._memory.record_search_source_execution(
                    subject=profile_source,
                    success=not has_error,
                    empty_result=is_empty,
                    error_type=err_type,
                    duration_ms=elapsed_ms,
                    result_count=len(api_papers),
                )
            except Exception as e:
                logger.warning(f"Profile record failed for {profile_source} : {e}")
            
        return api_papers
