"""Expose the LitAgent REST API and persistent flow-demo endpoints."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from litagent.config import AppConfig, load_config
from litagent.contracts import SurveyResult, build_config_summary
from litagent.logging import get_logger
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.scheduler import CancellationToken
from litagent.runner import LitAgent, derive_delivery

logger = get_logger("api")

FLOW_DEMO_STATIC_ROOT = Path(__file__).resolve().parent / "static"
FLOW_DEMO_STATIC = FLOW_DEMO_STATIC_ROOT / "flow-demo" / "index.html"
FLOW_DEMO_QUERY = "few-shot learning in computer vision"
_TASK_TTL_SECONDS = 3600
_CLEANUP_INTERVAL = 600
_TERMINAL_STATUSES = {"completed", "failed", "cancelled"}


class SurveyRequest(BaseModel):
    """Validate an untrusted survey request without accepting local paths."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def _strip_query(cls, value: str) -> str:
        """Reject whitespace-only requests and normalize surrounding whitespace."""
        value = value.strip()
        if not value:
            raise ValueError("query must not be blank")
        return value


class SurveyStatus(BaseModel):
    """Represent the current status of an API-owned survey task."""

    task_id: str
    status: str
    progress: str = ""
    error: str | None = None
    delivery_status: str | None = None


class SurveyReport(SurveyResult):
    """Add API task identity to the shared survey-result contract."""

    task_id: str


@dataclass
class SurveyTaskEntry:
    """Own one survey's cancellation, task handle, trace, and terminal result."""

    task_id: str
    query: str
    status: Literal[
        "queued", "running", "completed", "failed", "cancelling", "cancelled"
    ]
    progress: str
    cancellation: CancellationToken
    recorder: RunRecorder
    task: asyncio.Task[None] | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    flow_demo: bool = False


def _entry_value(entry: SurveyTaskEntry | dict[str, Any], name: str, default=None):
    """Read new task entries while preserving legacy in-memory test fixtures."""
    if isinstance(entry, dict):
        return entry.get(name, entry.get(f"_{name}", default))
    return getattr(entry, name, default)


async def _cleanup_old_tasks() -> None:
    """Remove terminal tasks after their in-memory TTL expires."""
    while True:
        await asyncio.sleep(_CLEANUP_INTERVAL)
        now = time.time()
        expired = [
            task_id
            for task_id, entry in app.state.tasks.items()
            if _entry_value(entry, "status") in _TERMINAL_STATUSES
            and now - float(_entry_value(entry, "created_at", 0) or 0)
            > _TASK_TTL_SECONDS
        ]
        for task_id in expired:
            app.state.tasks.pop(task_id, None)
            app.state.flow_runs.pop(task_id, None)
        if expired:
            logger.info("Cleaned %d expired tasks", len(expired))


async def _shutdown_owned_tasks(application: FastAPI) -> None:
    """Cooperatively stop API-owned work, then force only grace-timeout stragglers."""
    entries = [
        entry
        for entry in application.state.tasks.values()
        if isinstance(entry, SurveyTaskEntry) and entry.status not in _TERMINAL_STATUSES
    ]
    for entry in entries:
        entry.status = "cancelling"
        entry.progress = "cancelling"
        entry.cancellation.cancel()

    tasks = [entry.task for entry in entries if entry.task is not None]
    if not tasks:
        return
    grace = application.state.config.api.cancellation_grace_seconds
    done, pending = await asyncio.wait(tasks, timeout=grace)
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    if done:
        await asyncio.gather(*done, return_exceptions=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Initialize stable runtime state and close every owned task on shutdown."""
    application.state.config = load_config()
    application.state.survey_semaphore = asyncio.Semaphore(
        application.state.config.api.max_concurrent_surveys
    )
    application.state.flow_archive_index = {
        artifact["run_id"]: artifact
        for artifact in application.state.flow_repository.list()
    }
    cleanup_task = asyncio.create_task(_cleanup_old_tasks())
    logger.info("LitAgent API started")
    try:
        yield
    finally:
        await _shutdown_owned_tasks(application)
        cleanup_task.cancel()
        await asyncio.gather(cleanup_task, return_exceptions=True)
        logger.info("LitAgent API shut down")


app = FastAPI(
    title="LitAgent API",
    description="Multi-agent adversarial literature review framework REST API",
    version="0.1.0",
    lifespan=lifespan,
)
app.state.tasks: dict[str, SurveyTaskEntry | dict[str, Any]] = {}
app.state.flow_repository = ArchiveRepository()
app.state.flow_runs: dict[str, SurveyTaskEntry] = {}
app.state.flow_archive_index = {
    artifact["run_id"]: artifact for artifact in app.state.flow_repository.list()
}
app.state.config = None
app.state.survey_semaphore = None

app.mount(
    "/static",
    StaticFiles(directory=str(FLOW_DEMO_STATIC_ROOT)),
    name="static",
)


def _runtime_config() -> AppConfig:
    """Return lifespan configuration, with a narrow fallback for direct tests."""
    if app.state.config is None:
        app.state.config = load_config()
    return app.state.config


def _runtime_semaphore() -> asyncio.Semaphore:
    """Return the semaphore created for the active application event loop."""
    if app.state.survey_semaphore is None:
        app.state.survey_semaphore = asyncio.Semaphore(
            _runtime_config().api.max_concurrent_surveys
        )
    return app.state.survey_semaphore


def _survey_progress_hook(entry: SurveyTaskEntry):
    """Project trace events into small status updates without copying payloads."""

    def hook(event: str, data: dict[str, Any]) -> None:
        try:
            task_id = str(data.get("task_id") or "")
            agent_type = str(data.get("agent_type") or "")
            if event == "worker.start":
                entry.progress = task_id or agent_type or "worker"
            elif event == "subspan.start":
                entry.progress = str(data.get("name") or task_id or "processing")
            elif event == "survey.complete":
                entry.progress = "done"
            elif event == "survey.error":
                entry.progress = "failed"
            entry.events.append(
                {
                    "event": event,
                    "task_id": task_id,
                    "agent_type": agent_type,
                    "status": entry.status,
                }
            )
        except Exception:
            logger.debug("Survey progress hook failed", exc_info=True)

    return hook


def _build_survey_trace_hook(config: AppConfig, entry: SurveyTaskEntry):
    """Combine full local recording, progress projection, and redacted telemetry."""
    hooks: list[Any] = [entry.recorder, _survey_progress_hook(entry)]
    if config.observability.enabled:
        tracer = LangFuseTracer(
            host=config.observability.langfuse_host,
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        )
        hooks.append(
            RedactingTraceHook(tracer, payload_mode=config.observability.payload_mode)
        )
    return CompositeTraceHook(*hooks)


def _new_entry(query: str, *, task_id: str | None = None, flow_demo: bool = False):
    """Create and register one task entry before scheduling its coroutine."""
    task_id = task_id or uuid.uuid4().hex[:12]
    recorder = RunRecorder(task_id, query, app.state.flow_repository)
    recorder.set_config_summary(build_config_summary(_runtime_config()))
    entry = SurveyTaskEntry(
        task_id=task_id,
        query=query,
        status="queued",
        progress="queued",
        cancellation=CancellationToken(),
        recorder=recorder,
        flow_demo=flow_demo,
    )
    app.state.tasks[task_id] = entry
    if flow_demo:
        app.state.flow_runs[task_id] = entry
    entry.task = asyncio.create_task(_run_survey(entry))
    return entry


async def _run_survey(entry: SurveyTaskEntry) -> None:
    """Run one owned survey and atomically align API and artifact terminal state."""
    config = _runtime_config()
    try:
        async with _runtime_semaphore():
            if entry.cancellation.is_cancelled:
                result = LitAgent(config)._build_cancelled_report(entry.query)
            else:
                entry.status = "running"
                entry.progress = "planning"
                trace_hook = _build_survey_trace_hook(config, entry)
                async with LitAgent(config, trace_hook=trace_hook) as agent:
                    result = await agent.run(
                        entry.query, cancellation=entry.cancellation
                    )

            cancelled = entry.cancellation.is_cancelled or (
                "run_cancelled"
                in (
                    result.get("metadata", {})
                    .get("execution", {})
                    .get("reason_codes", [])
                )
            )
            terminal_status = "cancelled" if cancelled else "completed"
            artifact = entry.recorder.finalize(
                report=result, terminal_status=terminal_status
            )
            entry.result = result
            entry.status = terminal_status
            entry.progress = "cancelled" if cancelled else "done"
            if entry.flow_demo:
                app.state.flow_archive_index[entry.task_id] = artifact
    except asyncio.CancelledError:
        entry.status = "failed"
        entry.progress = "failed"
        entry.error = "task_force_cancelled"
        try:
            entry.recorder.finalize(error=entry.error, terminal_status="failed")
        except Exception:
            logger.exception(
                "Failed to finalize force-cancelled task %s", entry.task_id
            )
        raise
    except Exception as exc:
        entry.status = "failed"
        entry.progress = "failed"
        entry.error = f"{type(exc).__name__}: {exc}"[:1024]
        try:
            artifact = entry.recorder.finalize(
                error=entry.error, terminal_status="failed"
            )
            if entry.flow_demo:
                app.state.flow_archive_index[entry.task_id] = artifact
        except Exception:
            logger.exception("Failed to finalize survey %s", entry.task_id)
        logger.exception("Survey %s failed", entry.task_id)


def _status(entry: SurveyTaskEntry | dict[str, Any]) -> SurveyStatus:
    """Map one internal task entry to the public status model."""
    result = _entry_value(entry, "result") or {}
    delivery_status = None
    if _entry_value(entry, "status") in {"completed", "cancelled"}:
        delivery = result.get("delivery") or derive_delivery(
            result.get("partial", False), result.get("quality")
        )
        delivery_status = delivery.get("status")
    return SurveyStatus(
        task_id=str(_entry_value(entry, "task_id", "")),
        status=str(_entry_value(entry, "status", "unknown")),
        progress=str(_entry_value(entry, "progress", "") or ""),
        error=_entry_value(entry, "error"),
        delivery_status=delivery_status,
    )


@app.get("/health")
async def health():
    """Return the API health status."""
    return {"status": "ok"}


@app.post("/survey", status_code=202, response_model=SurveyStatus)
async def create_survey(req: SurveyRequest):
    """Submit a survey and return its API-owned queued task."""
    return _status(_new_entry(req.query))


@app.delete("/survey/{task_id}", status_code=202, response_model=SurveyStatus)
async def cancel_survey(task_id: str, response: Response):
    """Request cooperative cancellation for a queued or running survey."""
    entry = app.state.tasks.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    if not isinstance(entry, SurveyTaskEntry):
        raise HTTPException(status_code=409, detail="Legacy task cannot be cancelled")
    if entry.status in {"completed", "failed"}:
        raise HTTPException(status_code=409, detail="Task is already terminal")
    if entry.status in {"cancelling", "cancelled"}:
        response.status_code = 200
        return _status(entry)

    entry.status = "cancelling"
    entry.progress = "cancelling"
    entry.cancellation.cancel()
    response.status_code = 202
    return _status(entry)


@app.get("/survey/{task_id}", response_model=SurveyStatus)
async def get_survey_status(task_id: str):
    """Return the current status of a survey task."""
    entry = app.state.tasks.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    status = _status(entry)
    if not status.task_id:
        status.task_id = task_id
    return status


@app.get("/survey/{task_id}/report", response_model=SurveyReport)
async def get_survey_report(task_id: str):
    """Return a completed or cooperatively cancelled survey report."""
    entry = app.state.tasks.get(task_id)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    status = _entry_value(entry, "status")
    if status in {"queued", "running", "cancelling"}:
        raise HTTPException(status_code=409, detail="Survey is not terminal")
    if status == "failed":
        raise HTTPException(
            status_code=500, detail=_entry_value(entry, "error", "unknown")
        )
    result = dict(_entry_value(entry, "result") or {})
    if not result:
        raise HTTPException(status_code=500, detail="Terminal report is unavailable")
    result.setdefault(
        "delivery",
        derive_delivery(result.get("partial", False), result.get("quality")),
    )
    return SurveyReport(task_id=task_id, **result)


@app.get("/survey/{task_id}/events")
async def stream_survey_events(task_id: str):
    """Stream the recorder timeline and close after the task reaches terminal state."""
    entry = app.state.tasks.get(task_id)
    if not isinstance(entry, SurveyTaskEntry):
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    async def event_stream():
        offset = 0
        while True:
            events = entry.recorder.snapshot().get("events", [])
            for event in events[offset:]:
                yield f"event: trace\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
            offset = len(events)
            if entry.status in _TERMINAL_STATUSES:
                yield f"event: done\ndata: {json.dumps(_status(entry).model_dump())}\n\n"
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/flow-demo", include_in_schema=False)
async def flow_demo_page():
    """Return the packaged flow-demo user interface."""
    return FileResponse(FLOW_DEMO_STATIC)


@app.post("/flow-demo/runs", status_code=201)
async def create_flow_demo():
    """Start the fixed live query through the shared survey task owner."""
    entry = _new_entry(
        FLOW_DEMO_QUERY,
        task_id=f"flow-{uuid.uuid4().hex[:12]}",
        flow_demo=True,
    )
    return {"run_id": entry.task_id, "query": entry.query, "status": entry.status}


def _flow_artifact(run_id: str) -> dict[str, Any] | None:
    """Load one live or archived flow-demo artifact."""
    entry = app.state.flow_runs.get(run_id)
    if entry is not None:
        return entry.recorder.snapshot()
    artifact = app.state.flow_archive_index.get(run_id)
    if artifact is None:
        artifact = app.state.flow_repository.get(run_id)
        if artifact is not None:
            app.state.flow_archive_index[run_id] = artifact
    return artifact


def _flow_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    """Build a compact summary for a flow-demo run."""
    report = artifact.get("report") or {}
    return {
        "run_id": artifact["run_id"],
        "query": artifact.get("query", ""),
        "status": artifact.get("status", "unknown"),
        "started_at": artifact.get("started_at"),
        "completed_at": artifact.get("completed_at"),
        "error": artifact.get("error"),
        "quality": report.get("quality"),
        "delivery": report.get("delivery"),
        "config_fingerprint": artifact.get("config_fingerprint"),
    }


@app.get("/flow-demo/runs")
async def list_flow_demo_runs():
    """List live and archived flow-demo runs."""
    artifacts = dict(app.state.flow_archive_index)
    artifacts.update(
        {item["run_id"]: item for item in app.state.flow_repository.list()}
    )
    for run_id in app.state.flow_runs:
        artifact = _flow_artifact(run_id)
        if artifact:
            artifacts[run_id] = artifact
    return [
        _flow_summary(item)
        for item in sorted(
            artifacts.values(),
            key=lambda item: item.get("completed_at") or item.get("started_at", ""),
            reverse=True,
        )
    ]


@app.get("/flow-demo/runs/{run_id}")
async def get_flow_demo_run(run_id: str):
    """Return one live or archived flow-demo run."""
    artifact = _flow_artifact(run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Flow run '{run_id}' not found")
    return artifact


@app.get("/flow-demo/runs/{run_id}/events")
async def stream_flow_demo_events(run_id: str):
    """Stream flow-demo trace events over server-sent events."""
    entry = app.state.flow_runs.get(run_id)
    if entry is not None:
        return await stream_survey_events(run_id)
    artifact = _flow_artifact(run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Flow run '{run_id}' not found")

    async def archived_stream():
        for event in artifact.get("events", []):
            yield f"event: trace\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
        yield f"event: done\ndata: {json.dumps(_flow_summary(artifact), ensure_ascii=False)}\n\n"

    return StreamingResponse(archived_stream(), media_type="text/event-stream")


@app.get("/flow-demo/runs/{run_id}/download")
async def download_flow_demo_run(run_id: str):
    """Download a completed flow-demo artifact."""
    artifact = app.state.flow_repository.get(run_id)
    if artifact is None:
        raise HTTPException(
            status_code=404, detail=f"Completed flow run '{run_id}' not found"
        )
    return FileResponse(
        app.state.flow_repository.path_for(run_id),
        media_type="application/json",
        filename=f"{run_id}.json",
    )
