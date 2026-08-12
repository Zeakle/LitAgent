"""Expose the LitAgent REST API and persistent flow-demo endpoints."""

from __future__ import annotations

import asyncio as _asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from litagent.config import load_config
from litagent.contracts import SurveyResult, build_config_summary
from litagent.logging import get_logger
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.runner import LitAgent, derive_delivery

logger = get_logger("api")


# API models
# Static assets belong to the installed package, not the caller's cwd.
FLOW_DEMO_STATIC_ROOT = Path(__file__).resolve().parent / "static"
FLOW_DEMO_STATIC = FLOW_DEMO_STATIC_ROOT / "flow-demo" / "index.html"


class SurveyRequest(BaseModel):
    """Validate a survey query and its optional configuration path."""

    query: str = Field(..., min_length=1, max_length=2000)
    config_path: str | None = None


class SurveyStatus(BaseModel):
    """Represent the current status of a survey task."""

    task_id: str
    status: str
    progress: str = ""
    error: str | None = None
    delivery_status: str | None = None


class SurveyReport(SurveyResult):
    """Add API task identity to the shared survey-result contract."""

    task_id: str


# Background cleanup

_TASK_TTL_SECONDS = 3600
_CLEANUP_INTERVAL = 600
FLOW_DEMO_QUERY = "few-shot learning in computer vision"


async def _cleanup_old_tasks() -> None:
    """Remove terminal tasks after their in-memory TTL expires."""
    while True:
        await _asyncio.sleep(_CLEANUP_INTERVAL)
        now = time.time()
        expired = [
            tid
            for tid, t in app.state.tasks.items()
            if t["status"] in ("completed", "failed")
            and (now - t.get("_created_at", 0)) > _TASK_TTL_SECONDS
        ]
        for tid in expired:
            del app.state.tasks[tid]
        if expired:
            logger.info("Cleaned %d expired tasks", len(expired))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Restore archived runs and manage the cleanup task lifecycle."""
    _app.state.flow_archive_index = {
        artifact["run_id"]: artifact for artifact in _app.state.flow_repository.list()
    }
    cleanup_task = _asyncio.create_task(_cleanup_old_tasks())
    logger.info("LitAgent API started, TTL cleanup running")
    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except _asyncio.CancelledError:
        pass
    logger.info("LitAgent API shut down")


# Application state

app = FastAPI(
    title="LitAgent API",
    description="Multi-agent adversarial literature review framework — REST API",
    version="0.1.0",
    lifespan=lifespan,
)


app.state.tasks: dict[str, dict[str, Any]] = {}
app.state.flow_repository = ArchiveRepository()
app.state.flow_runs: dict[str, dict[str, Any]] = {}
app.state.flow_archive_index = {
    artifact["run_id"]: artifact for artifact in app.state.flow_repository.list()
}


app.mount(
    "/static",
    StaticFiles(directory=str(FLOW_DEMO_STATIC_ROOT)),
    name="static",
)


# Routes


@app.get("/health")
async def health():
    """Return the API health status."""
    return {"status": "ok"}


@app.get("/flow-demo", include_in_schema=False)
async def flow_demo_page():
    """Return the packaged flow-demo user interface."""
    return FileResponse(FLOW_DEMO_STATIC)


@app.post("/flow-demo/runs", status_code=201)
async def create_flow_demo(background_tasks: BackgroundTasks):
    """Start the fixed, live survey without changing the general /survey API."""
    run_id = f"flow-{uuid.uuid4().hex[:12]}"
    recorder = RunRecorder(run_id, FLOW_DEMO_QUERY, app.state.flow_repository)
    app.state.flow_runs[run_id] = {
        "status": "running",
        "recorder": recorder,
        "error": None,
        "_created_at": time.time(),
    }
    background_tasks.add_task(_run_flow_demo, run_id, recorder)
    return {"run_id": run_id, "query": FLOW_DEMO_QUERY, "status": "running"}


async def _run_flow_demo(run_id: str, recorder: RunRecorder) -> None:
    """Persist the complete real-run timeline after cleanup has emitted its events."""
    entry = app.state.flow_runs[run_id]
    try:
        config = load_config()
        config_summary = build_config_summary(config)
        recorder.set_config_summary(config_summary)

        trace_hook: Any = recorder
        if config.observability.enabled:
            langfuse = LangFuseTracer(
                host=config.observability.langfuse_host,
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            )
            trace_hook = CompositeTraceHook(
                recorder,
                RedactingTraceHook(
                    langfuse, payload_mode=config.observability.payload_mode
                ),
            )
        async with LitAgent(config, trace_hook=trace_hook) as agent:
            report = await agent.run(FLOW_DEMO_QUERY)
        artifact = recorder.finalize(report=report)
        entry.update(
            status=artifact["status"],
            error=artifact.get("error"),
            artifact=artifact,
        )
        app.state.flow_archive_index[run_id] = artifact
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        artifact = recorder.finalize(error=error)
        entry.update(status="failed", error=error, artifact=artifact)
        app.state.flow_archive_index[run_id] = artifact
        logger.exception("Flow demo %s failed", run_id)


def _flow_artifact(run_id: str) -> dict[str, Any] | None:
    """Load one flow-demo artifact."""
    entry = app.state.flow_runs.get(run_id)
    if entry:
        return entry["recorder"].snapshot()
    artifact = app.state.flow_archive_index.get(run_id)
    if artifact is not None:
        return artifact
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
    if _flow_artifact(run_id) is None:
        raise HTTPException(status_code=404, detail=f"Flow run '{run_id}' not found")

    async def event_stream():
        """Yield flow-demo events as SSE frames."""
        offset = 0
        while True:
            artifact = _flow_artifact(run_id)
            if artifact is None:
                return
            events = artifact.get("events", [])
            for event in events[offset:]:
                yield (
                    "event: trace\ndata: "
                    f"{json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                )
            offset = len(events)
            if artifact.get("status") in {"completed", "failed"}:
                yield (
                    "event: done\ndata: "
                    f"{json.dumps(_flow_summary(artifact), ensure_ascii=False)}\n\n"
                )
                return
            await _asyncio.sleep(0.3)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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


@app.post("/survey", status_code=201, response_model=SurveyStatus)
async def create_survey(req: SurveyRequest, background_tasks: BackgroundTasks):
    """Submit a survey task for background execution."""
    task_id = str(uuid.uuid4())[:8]
    app.state.tasks[task_id] = {
        "status": "running",
        "progress": "planner",
        "result": None,
        "error": None,
        "_created_at": time.time(),
    }
    background_tasks.add_task(_run_survey, task_id, req.query, req.config_path)
    return SurveyStatus(task_id=task_id, status="running", progress="planner")


async def _run_survey(task_id: str, query: str, config_path: str | None) -> None:
    """Run a survey and store its terminal task state."""
    try:
        config = load_config(config_path)
        async with LitAgent(config) as agent:
            result = await agent.run(query)
            app.state.tasks[task_id].update(
                status="completed",
                progress="done",
                result=result,
            )
    except Exception as e:
        app.state.tasks[task_id].update(
            status="failed",
            progress="",
            error=str(e),
        )


@app.get("/survey/{task_id}", response_model=SurveyStatus)
async def get_survey_status(task_id: str):
    """Return the current status of a survey task."""
    task = app.state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    delivery_status = None
    if task["status"] == "completed":
        result = task.get("result") or {}
        d = result.get("delivery") or derive_delivery(
            result.get("partial", False), result.get("quality")
        )
        delivery_status = d.get("status")

    return SurveyStatus(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress", ""),
        error=task.get("error"),
        delivery_status=delivery_status,
    )


@app.get("/survey/{task_id}/report", response_model=SurveyReport)
async def get_survey_report(task_id: str):
    """Return the report for a completed survey task."""
    task = app.state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404)
    if task["status"] == "running":
        raise HTTPException(
            status_code=409, detail="Still running. Poll GET /survey/{task_id} first."
        )
    if task["status"] == "failed":
        raise HTTPException(status_code=500, detail=task.get("error", "unknown"))

    result = dict(task["result"] or {})
    result.setdefault(
        "delivery",
        derive_delivery(result.get("partial", False), result.get("quality")),
    )

    return SurveyReport(task_id=task_id, **result)
