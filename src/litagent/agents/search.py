"""External paper search with stable degradation outcomes."""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from litagent.logging import get_logger
from litagent.memory.manager import MemoryManager
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.rag.models import PaperCandidate
from litagent.tools.executor import ToolExecutor, ToolResult

logger = get_logger("agents.search")


_TOOL_BY_SOURCE = {
    "arxiv": "search_arxiv",
    "semantic_scholar": "search_semantic_scholar",
    "huggingface": "search_huggingface",
}


class SearchSourceStatus(str, Enum):
    """Describe the normalized outcome of one provider call."""

    SUCCESS = "success"
    EMPTY = "empty"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    FAILED = "failed"


@dataclass(frozen=True)
class SearchSourceOutcome:
    """Capture the normalized result of one search-provider task."""

    task_id: str
    source: str
    status: SearchSourceStatus
    result_count: int
    elapsed_ms: float
    reason_code: str | None = None
    error_type: str | None = None
    from_fallback: bool = False


class SearchWorker(Worker):
    """Run provider searches and record stable degradation outcomes."""

    def __init__(
        self,
        executor: ToolExecutor,
        memory_manager: MemoryManager | None = None,
        trace_hook=None,
    ) -> None:
        self._executor = executor
        self._memory = memory_manager
        self._trace_hook = trace_hook
        self._outcomes: dict[str, SearchSourceOutcome] = {}

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "search"

    def reset_source_outcomes(self) -> None:
        """Clear outcomes collected for prior runs."""
        self._outcomes.clear()

    def get_source_outcomes(self) -> tuple[SearchSourceOutcome, ...]:
        """Return the recorded source outcomes."""
        return tuple(self._outcomes.values())

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception:
                logger.debug("Search trace hook failed", exc_info=True)

    @staticmethod
    def _classify(
        task_id: str, source: str, result: ToolResult, elapsed_ms: float
    ) -> tuple[list[dict[str, Any]], SearchSourceOutcome]:
        if result.error is not None:
            if result.error_code in {"tool_rate_limited", "http_rate_limited"}:
                status = SearchSourceStatus.RATE_LIMITED
                reason = "provider_rate_limited"
            elif result.error_code == "tool_timeout":
                status = SearchSourceStatus.TIMEOUT
                reason = "provider_timeout"
            else:
                status = SearchSourceStatus.FAILED
                reason = "provider_error"

            return [], SearchSourceOutcome(
                task_id=task_id,
                source=source,
                status=status,
                result_count=0,
                elapsed_ms=elapsed_ms,
                reason_code=reason,
                error_type=result.error_type,
                from_fallback=result.from_fallback,
            )

        if not isinstance(result.output, list):
            return [], SearchSourceOutcome(
                task_id=task_id,
                source=source,
                status=SearchSourceStatus.FAILED,
                result_count=0,
                elapsed_ms=elapsed_ms,
                reason_code="invalid_provider_response",
                error_type=type(result.output).__name__,
                from_fallback=result.from_fallback,
            )

        normalized = []
        malformed_count = 0
        for raw in result.output:
            if not isinstance(raw, dict):
                malformed_count += 1
                continue
            try:
                normalized.append(PaperCandidate.from_external(raw).to_dag_dict())
            except (TypeError, ValueError):
                malformed_count += 1
        papers = normalized

        if malformed_count and not papers:
            return [], SearchSourceOutcome(
                task_id=task_id,
                source=source,
                status=SearchSourceStatus.FAILED,
                result_count=0,
                elapsed_ms=elapsed_ms,
                reason_code="invalid_provider_response",
                error_type="MalformedPaperList",
                from_fallback=result.from_fallback,
            )

        status = SearchSourceStatus.SUCCESS if papers else SearchSourceStatus.EMPTY
        if malformed_count:
            reason = "invalid_provider_items_dropped"
        elif result.from_fallback:
            reason = "provider_fallback_used"
        elif not papers:
            reason = "provider_empty"
        else:
            reason = None
        return papers, SearchSourceOutcome(
            task_id=task_id,
            source=source,
            status=status,
            result_count=len(papers),
            elapsed_ms=elapsed_ms,
            reason_code=reason,
            error_type=result.error_type,
            from_fallback=result.from_fallback,
        )

    async def _record_outcome(self, outcome: SearchSourceOutcome) -> None:
        self._outcomes[outcome.task_id] = outcome
        self._emit("search.source.complete", asdict(outcome))

        if not self._memory:
            return

        try:
            profile_error_type = {
                SearchSourceStatus.RATE_LIMITED: "rate_limit",
                SearchSourceStatus.TIMEOUT: "timeout",
                SearchSourceStatus.FAILED: "tool_error",
            }.get(outcome.status)
            await self._memory.record_search_source_execution(
                subject=outcome.source,
                success=outcome.status
                in {
                    SearchSourceStatus.SUCCESS,
                    SearchSourceStatus.EMPTY,
                },
                empty_result=outcome.status is SearchSourceStatus.EMPTY,
                error_type=profile_error_type,
                duration_ms=int(outcome.elapsed_ms),
                result_count=outcome.result_count,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Profile record failed for %s",
                outcome.source,
                exc_info=True,
            )

    async def execute(self, task: SubTask) -> list[dict[str, Any]]:
        """Run one provider search and record its normalized outcome."""
        source = str(task.input_data.get("source") or "")
        query = str(task.input_data.get("query") or "")
        tool_name = _TOOL_BY_SOURCE.get(source)

        if tool_name is None:
            outcome = SearchSourceOutcome(
                task_id=task.task_id,
                source=source,
                status=SearchSourceStatus.FAILED,
                result_count=0,
                elapsed_ms=0,
                reason_code="invalid_search_source",
                error_type="ValueError",
            )
            await self._record_outcome(outcome)
            return []

        started = time.perf_counter()
        result = await self._executor.execute(
            tool_name,
            {"query": query, "max_results": 20},
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        papers, outcome = self._classify(task.task_id, source, result, elapsed_ms)
        await self._record_outcome(outcome)
        return papers
