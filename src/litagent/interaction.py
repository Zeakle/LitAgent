"""Define resumable interactive-run payloads and deterministic review transforms."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from litagent.orchestrator.task_graph import TaskGraph
from litagent.workbench import (
    Contract,
    HumanDecision,
    ReportVersion,
    ResumeIdentity,
    ReviewRequest,
    RunOptions,
    validate_decision,
)


def interaction_source_fingerprint() -> str:
    """Hash behavior-bearing files that determine safe interactive recovery."""
    root = Path(__file__).resolve().parent
    paths = [
        root / "interaction.py",
        root / "runner.py",
        root / "workbench.py",
        root / "orchestrator" / "scheduler.py",
        root / "orchestrator" / "task_graph.py",
        root / "agents" / "synthesis.py",
        root / "agents" / "reviewer.py",
        root / "agents" / "adversarial.py",
    ]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class ExecutionCheckpoint(Contract):
    """Persist one quiescent boundary without serializing runtime objects."""

    schema_version: Literal[1] = 1
    identity: ResumeIdentity
    query: str = Field(min_length=1, max_length=2000)
    options: RunOptions
    boundary: Literal["paper_selection", "final_report"]
    graph: dict[str, Any]
    results: dict[str, Any]
    budget: dict[str, Any]
    search_outcomes: tuple[dict[str, Any], ...] = ()
    degradation_codes: tuple[str, ...] = ()
    report_data: dict[str, Any] | None = None
    report_revision: int = Field(default=0, ge=0)
    revision_attempts: int = Field(default=0, ge=0, le=1)
    review_request_id: str = Field(min_length=1)


class RunAdvanceResult(Contract):
    """Return a terminal report or one explicit persisted review boundary."""

    status: Literal[
        "awaiting_human", "completed", "cancelled", "rejected", "interrupted"
    ]
    checkpoint: ExecutionCheckpoint | None = None
    review: ReviewRequest | None = None
    report: dict[str, Any] | None = None


def paper_id(record: Mapping[str, Any]) -> str | None:
    """Return the existing canonical paper identity without inventing aliases."""
    value = record.get("paper_id") or record.get("id")
    return str(value) if isinstance(value, str) and value else None


def apply_paper_decision(
    checkpoint: ExecutionCheckpoint, decision: HumanDecision
) -> ExecutionCheckpoint:
    """Replace the relevance result with an ordered subset of dedup candidates."""
    if checkpoint.boundary != "paper_selection":
        raise ValueError("wrong_review_boundary")
    graph = TaskGraph.from_snapshot(checkpoint.graph)
    relevance = graph.get_task("relevance_gate")
    dedup = graph.get_task("dedup")
    if relevance is None or dedup is None or not isinstance(dedup.result, list):
        raise ValueError("paper_selection_checkpoint_invalid")
    request = ReviewRequest(
        request_id=checkpoint.review_request_id,
        run_id=checkpoint.identity.run_id,
        stage="paper_selection",
        revision=decision.revision,
        candidate_paper_ids=tuple(
            item_id
            for item in dedup.result
            if isinstance(item, Mapping) and (item_id := paper_id(item))
        ),
        automatic_selected_paper_ids=tuple(
            item_id
            for item in (relevance.result or [])
            if isinstance(item, Mapping) and (item_id := paper_id(item))
        ),
    )
    validate_decision(request, decision)
    by_id = {
        item_id: dict(item)
        for item in dedup.result
        if isinstance(item, Mapping) and (item_id := paper_id(item))
    }
    relevance.result = [by_id[item_id] for item_id in decision.selected_paper_ids]
    return checkpoint.model_copy(
        update={"graph": graph.to_snapshot(), "results": graph.get_results()}
    )


def report_version_from_checkpoint(checkpoint: ExecutionCheckpoint) -> ReportVersion:
    """Build the exact report identity used by approval and export."""
    if checkpoint.report_data is None:
        raise ValueError("report_checkpoint_missing")
    return ReportVersion(
        run_id=checkpoint.identity.run_id,
        revision=checkpoint.report_revision,
        report_language=checkpoint.options.report_language,
        survey=str(checkpoint.report_data.get("survey") or ""),
        human_status="pending",
    )
