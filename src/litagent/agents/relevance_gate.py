"""CrossEncoder relevance ranking with deterministic lexical fallback."""


from __future__ import annotations

import asyncio
import re
from typing import Any

from langchain_core.documents import Document

from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.interfaces import Reranker, ScoredDoc


logger = get_logger('agents.relevance_gate')
_WORD_RE = re.compile(r'[a-z0-9]+')
_SOURCE_INDEX = '_relevance_source_index'


class RelevanceGateWorker(Worker):
    """Rank deduplicated papers and cap the downstream candidate set."""

    def __init__(self, reranker: Reranker | None, max_papers: int = 50) -> None:
        if max_papers <= 0:
            raise ValueError('max_papers must be positive')
        self._reranker = reranker
        self._max_papers = max_papers

    @property
    def agent_type(self) -> str:
        return 'relevance_gate'

    async def execute(self, task: SubTask) -> list[dict[str, Any]]:
        query = str(task.input_data.get('query') or "").strip()
        upstream = task.input_data.get('upstream_results', {})

        if not isinstance(upstream, dict):
            return []

        candidates = upstream.get('dedup', [])

        if not isinstance(candidates, list):
            return []

        papers = [
            paper
            for paper in candidates
            if isinstance(paper, dict)
            and (paper.get('title') or paper.get('abstract'))
        ]

        if not papers:
            return []

        if self._reranker is not None and query:
            try:
                return await self._cross_encoder_rank(query, papers)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Relevance reranker degraded reason=reranker_error error_type=%s",
                    type(exc).__name__,
                )
            
        return self._lexical_rank(query, papers)

    async def _cross_encoder_rank(
        self,
        query: str,
        papers: list[dict],
    ) -> list[dict[str, Any]]:
        docs = [
            ScoredDoc(
                doc=Document(
                    page_content=(
                        f"{paper.get('title') or ''}\n"
                        f"{(paper.get('abstract') or '')[:1000]}"
                    ),
                    metadata={_SOURCE_INDEX: index}
                ),
                score=0.0,
            )
            for index, paper in enumerate(papers)
        ] 

        ranked = await asyncio.to_thread(self._reranker.rerank, query, docs)

        output: list[dict[str, Any]] = []

        for rank, scored_doc in enumerate(ranked[:self._max_papers]):
            source_index = scored_doc.doc.metadata.get(_SOURCE_INDEX)
            
            if not isinstance(source_index, int) or not 0 <= source_index < len(papers):
                raise ValueError("reranker returned a document without a valid source index")

            item = dict(papers[source_index])
            item['relevance_score'] = float(scored_doc.score)
            item['relevance_rank'] = rank
            item['relevance_method'] = 'cross_encoder'

            output.append(item)

        return output

    
    def _lexical_rank(self, query: str, papers: list[dict]) -> list[dict[str, Any]]:
        query_tokens = set(_WORD_RE.findall(query.lower()))
        scored: list[tuple[float, int, dict]] = []

        for index, paper in enumerate(papers):
            title_tokens = set(_WORD_RE.findall(str(paper.get("title") or "").lower()))
            abstract_tokens = set(
                _WORD_RE.findall(str(paper.get("abstract") or "").lower())
            )

            if query_tokens:
                title_coverage = len(query_tokens & title_tokens) / len(query_tokens)
                abstract_coverage = len(query_tokens & abstract_tokens) / len(query_tokens)
                score = 2.0 * title_coverage + abstract_coverage
            else:
                score = 0.0
            
            scored.append((score, index, paper))

        scored.sort(key=lambda item: (-item[0], item[1]))
        output: list[dict[str, Any]] = []

        for rank, (score, _, paper) in enumerate(scored[:self._max_papers]):
            item = dict(paper)
            item['relevance_score'] = float(score)
            item['relevance_rank'] = rank
            item['relevance_method'] = 'lexical_fallback'
            output.append(item)

        return output