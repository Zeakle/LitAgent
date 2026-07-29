"""CrossEncoder relevance ranking with deterministic lexical fallback."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.documents import Document

from litagent.logging import get_logger
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.interfaces import Reranker, ScoredDoc

logger = get_logger("agents.relevance_gate")
_WORD_RE = re.compile(r"[a-z0-9]+")
_SOURCE_INDEX = "_relevance_source_index"


@dataclass(frozen=True)
class RelevanceSelection:
    """Summarize a ranked selection and its threshold-refill counts."""

    selected: tuple[ScoredDoc, ...]
    threshold_qualified_count: int
    refilled_count: int
    below_threshold_count: int
    filtered_count: int
    input_count: int
    output_count: int
    score_min: float | None
    score_max: float | None
    mode: Literal["threshold", "rank_cap_only", "lexical_fallback"]


def _select_ranked(
    scored: Sequence[ScoredDoc],
    *,
    min_score: float | None,
    min_papers: int,
    max_papers: int,
    mode: Literal["threshold", "rank_cap_only", "lexical_fallback"],
) -> RelevanceSelection:
    if max_papers <= 0 or min_papers < 0 or min_papers > max_papers:
        raise ValueError("invalid relevance min/max policy")

    indexed = list(enumerate(scored))
    ranked = [
        item
        for _, item in sorted(
            indexed, key=lambda pair: (-float(pair[1].score), pair[0])
        )
    ]
    scores = [float(item.score) for item in ranked]

    qualified = [
        item for item in ranked if min_score is None or float(item.score) >= min_score
    ]

    below = [
        item
        for item in ranked
        if min_score is not None and float(item.score) < min_score
    ]

    # Refill from the best lower-scored papers to meet the minimum output size.
    refill_count = min(max(0, min_papers - len(qualified)), len(below))

    selected = (qualified + below[:refill_count])[:max_papers]

    return RelevanceSelection(
        selected=tuple(selected),
        threshold_qualified_count=len(qualified),
        refilled_count=refill_count,
        below_threshold_count=len(below),
        filtered_count=len(ranked) - len(selected),
        input_count=len(ranked),
        output_count=len(selected),
        score_min=min(scores) if scores else None,
        score_max=max(scores) if scores else None,
        mode=mode,
    )


class RelevanceGateWorker(Worker):
    """Rank deduplicated papers and cap the downstream candidate set."""

    def __init__(
        self,
        reranker: Reranker | None,
        *,
        max_papers: int = 50,
        min_papers: int = 10,
        cross_encoder_min_score: float | None = None,
        lexical_min_score: float = 0.0,
        trace_hook=None,
    ) -> None:
        if max_papers <= 0 or min_papers < 0 or min_papers > max_papers:
            raise ValueError("invalid relevance min/max policy")
        if not 0.0 <= lexical_min_score <= 3.0:
            raise ValueError("lexical_min_score must be in [0, 3]")
        self._reranker = reranker
        self._max_papers = max_papers
        self._min_papers = min_papers
        self._cross_encoder_min_score = cross_encoder_min_score
        self._lexical_min_score = lexical_min_score
        self._trace_hook = trace_hook

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception:
                logger.debug("Relevance trace hook failed", exc_info=True)

    def _materialize(
        self,
        papers: list[dict[str, Any]],
        selection: RelevanceSelection,
        method: str,
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        seen_indices: set[int] = set()

        for rank, scored_doc in enumerate(selection.selected):
            if not isinstance(scored_doc, ScoredDoc):
                raise ValueError("reranker returned non-ScoredDoc")

            source_index = scored_doc.doc.metadata.get(_SOURCE_INDEX)

            if (
                not isinstance(source_index, int)
                or source_index < 0
                or source_index >= len(papers)
                or source_index in seen_indices
            ):
                raise ValueError("invalid or duplicate relevance source index")

            seen_indices.add(source_index)

            item = dict(papers[source_index])
            item["relevance_score"] = float(scored_doc.score)
            item["relevance_rank"] = rank
            item["relevance_method"] = method
            output.append(item)

        self._emit(
            "relevance.gate.complete",
            {
                "input_count": selection.input_count,
                "output_count": selection.output_count,
                "threshold_qualified_count": selection.threshold_qualified_count,
                "refilled_count": selection.refilled_count,
                "below_threshold_count": selection.below_threshold_count,
                "filtered_count": selection.filtered_count,
                "score_min": selection.score_min,
                "score_max": selection.score_max,
                "mode": selection.mode,
            },
        )
        return output

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "relevance_gate"

    async def execute(self, task: SubTask) -> list[dict[str, Any]]:
        """Rank deduplicated papers with CrossEncoder or lexical fallback."""
        query = str(task.input_data.get("query") or "").strip()
        upstream = task.input_data.get("upstream_results", {})

        if not isinstance(upstream, dict):
            return []

        candidates = upstream.get("dedup", [])

        if not isinstance(candidates, list):
            return []

        papers = [
            paper
            for paper in candidates
            if isinstance(paper, dict) and (paper.get("title") or paper.get("abstract"))
        ]

        if not papers:
            return []

        if self._reranker is not None and query:
            try:
                return await self._cross_encoder_rank(query, papers)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._emit(
                    "relevance.gate.degraded",
                    {
                        "reason_code": "reranker_error",
                        "error_type": type(exc).__name__,
                    },
                )
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
                    metadata={_SOURCE_INDEX: index},
                ),
                score=0.0,
            )
            for index, paper in enumerate(papers)
        ]

        ranked = await asyncio.to_thread(self._reranker.rerank, query, docs)
        if not isinstance(ranked, list):
            raise ValueError("reranker result must be a list")

        mode = (
            "threshold"
            if self._cross_encoder_min_score is not None
            else "rank_cap_only"
        )

        selection = _select_ranked(
            ranked,
            min_score=self._cross_encoder_min_score,
            min_papers=self._min_papers,
            max_papers=self._max_papers,
            mode=mode,
        )
        return self._materialize(papers, selection, "cross_encoder")

    def _lexical_rank(
        self, query: str, papers: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        query_tokens = set(_WORD_RE.findall(query.lower()))
        scored: list[ScoredDoc] = []

        for index, paper in enumerate(papers):
            title_tokens = set(_WORD_RE.findall(str(paper.get("title") or "").lower()))
            abstract_tokens = set(
                _WORD_RE.findall(str(paper.get("abstract") or "").lower())
            )
            if query_tokens:
                title_coverage = len(query_tokens & title_tokens) / len(query_tokens)
                abstract_coverage = len(query_tokens & abstract_tokens) / len(
                    query_tokens
                )
                score = 2.0 * title_coverage + abstract_coverage
            else:
                score = 0.0

            scored.append(
                ScoredDoc(
                    doc=Document(
                        page_content=(
                            f"{paper.get('title') or ''}\n"
                            f"{(paper.get('abstract') or '')[:1000]}"
                        ),
                        metadata={_SOURCE_INDEX: index},
                    ),
                    score=score,
                )
            )

        selection = _select_ranked(
            scored,
            min_score=self._lexical_min_score,
            min_papers=self._min_papers,
            max_papers=self._max_papers,
            mode="lexical_fallback",
        )
        return self._materialize(papers, selection, "lexical_fallback")
