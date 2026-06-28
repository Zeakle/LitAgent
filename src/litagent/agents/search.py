"""Search Worker--调API搜索论文"""


from __future__ import annotations

from litagent.tools.base import FallbackStep
import asyncio
from typing import Any

import httpx

from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.logging import get_logger
from litagent.rag.retriever import HybridRetriever
from litagent.tools.executor import ToolExecutor

logger = get_logger("agents.search")


class SearchWorker(Worker):
    """搜索Worker--根据 input_data['source']路由到对应的数据源"""

    def __init__(self, executor: ToolExecutor, retriever: HybridRetriever | None = None):
        self._executor = executor
        self._retriever = retriever

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
            'paperswithcode': 'search_paperswithcode'
        }
        tool_name = tool_map.get(source, 'search_arxiv')  # default arxiv

        # hybrid search
        api_task = asyncio.create_task(
            self._executor.execute(tool_name, {'query': query, 'max_results': 20})
        )
        rag_task = asyncio.create_task(
            self._retriever.search(query, top_k=20)
        ) if self._retriever else None

        api_result = await api_task
        api_papers = api_result.output if not api_result.error else []

        semantic_results = []
        if rag_task:
            try:
                scored_docs = await rag_task
                semantic_results = [
                    {"paper_id": sd.doc.metadata.get("arxiv_id", ""),
                     "title": sd.doc.metadata.get("title", ""),
                     "abstract": sd.doc.page_content[:500],
                     "source": "rag_index", "score": sd.score}
                    for sd in scored_docs
                ]
            except Exception as e:
                logger.warning(f"RAG search failed: {e}")
        
        return api_papers + semantic_results