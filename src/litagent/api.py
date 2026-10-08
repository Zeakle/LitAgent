"""Expose the LitAgent REST API and persistent flow-demo endpoints."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from litagent.benchmark.survey_archive import SurveyBenchmarkArchive
from litagent.citation_formats import (
    build_bibliographic_snapshot,
    export_bibliography,
    format_capabilities,
)
from litagent.config import AppConfig, load_config
from litagent.contracts import SurveyResult, build_config_summary
from litagent.demo import (
    extract_report_evidence,
    load_bundled_samples,
    project_flow_demo_view,
)
from litagent.interaction_store import SqliteReviewRepository
from litagent.interactive_service import InteractiveRunService
from litagent.logging import get_logger
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.scheduler import CancellationToken
from litagent.report_audit import ReportAuditService
from litagent.report_export import LocalReportExporter
from litagent.report_render import (
    build_reader_report,
    paper_catalog_from_nodes,
    reader_report_projection,
)
from litagent.runner import LitAgent, derive_delivery
from litagent.workbench import (
    AutoAuditMode,
    CitationStyle,
    ExportResult,
    HumanDecision,
    ReportVersion,
    ReviewStructure,
    RunOptions,
)

logger = get_logger("api")


def _benchmark_archive() -> SurveyBenchmarkArchive:
    """Allow tests to inject a read-only archive without exposing a path parameter."""
    return getattr(app.state, "benchmark_archive", None) or SurveyBenchmarkArchive()


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
    report_language: Literal["en", "zh-CN"] = "en"
    human_review: bool = False
    review_structure: ReviewStructure = "legacy"
    citation_style: CitationStyle = "numbered"
    auto_audit_mode: AutoAuditMode = "off"
    auto_audit_max_versions: int = Field(default=1, ge=1, le=3, strict=True)

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
    tracer: LangFuseTracer | None = None
    options: RunOptions = field(default_factory=RunOptions)


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
    application.state.report_audit_service = ReportAuditService(
        application.state.config
    )
    await application.state.report_audit_service.initialize()
    application.state.interactive_service = InteractiveRunService(
        application.state.config,
        application.state.interactive_repository,
        semaphore=application.state.survey_semaphore,
        archive_repository=application.state.flow_repository,
        auto_audit_submitter=_submit_automatic_audit,
    )
    await application.state.interactive_service.initialize()
    application.state.flow_archive_index = {
        artifact["run_id"]: artifact
        for artifact in application.state.flow_repository.list()
    }
    cleanup_task = asyncio.create_task(_cleanup_old_tasks())
    logger.info("LitAgent API started")
    try:
        yield
    finally:
        await application.state.interactive_service.shutdown()
        await _shutdown_owned_tasks(application)
        await application.state.report_audit_service.shutdown()
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
app.state.flow_samples = load_bundled_samples()
app.state.flow_archive_index = {
    artifact["run_id"]: artifact for artifact in app.state.flow_repository.list()
}
app.state.config = None
app.state.survey_semaphore = None
app.state.interactive_repository = SqliteReviewRepository()
app.state.interactive_service = None
app.state.operation_token = secrets.token_urlsafe(24)
app.state.report_audit_service = None

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


async def _interactive_service() -> InteractiveRunService:
    """Return the durable service, initializing it for direct ASGI tests."""
    if app.state.interactive_service is None:
        app.state.interactive_service = InteractiveRunService(
            _runtime_config(),
            app.state.interactive_repository,
            semaphore=_runtime_semaphore(),
            archive_repository=app.state.flow_repository,
            auto_audit_submitter=_submit_automatic_audit,
        )
        await app.state.interactive_service.initialize()
    return app.state.interactive_service


def _require_operation_token(request: Request, token: str | None) -> None:
    """Protect local interactive mutations from drive-by browser requests."""
    expected = str(app.state.operation_token)
    if token is None or not secrets.compare_digest(token, expected):
        raise HTTPException(status_code=403, detail="invalid_operation_token")
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        raise HTTPException(status_code=403, detail="origin_not_allowed")


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
            trace_seed=entry.task_id,
        )
        entry.tracer = tracer
        hooks.append(
            RedactingTraceHook(tracer, payload_mode=config.observability.payload_mode)
        )
    return CompositeTraceHook(*hooks)


def _capture_trace_reference(entry: SurveyTaskEntry) -> None:
    """Copy the optional LangFuse reference into the persistent artifact."""
    if entry.tracer is None:
        return
    entry.recorder.set_trace_reference(entry.tracer.trace_id, entry.tracer.trace_url)


def _new_entry(
    query: str,
    *,
    task_id: str | None = None,
    flow_demo: bool = False,
    options: RunOptions | None = None,
):
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
        options=options or RunOptions(),
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
                        entry.query,
                        cancellation=entry.cancellation,
                        options=entry.options,
                    )

            _capture_trace_reference(entry)

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
            _capture_trace_reference(entry)
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
            _capture_trace_reference(entry)
            artifact = entry.recorder.finalize(
                error=entry.error, terminal_status="failed"
            )
            if entry.flow_demo:
                app.state.flow_archive_index[entry.task_id] = artifact
        except Exception:
            logger.exception("Failed to finalize survey %s", entry.task_id)
        logger.exception("Survey %s failed", entry.task_id)
    if entry.status == "completed" and entry.options.auto_audit_mode != "off":
        # Keep audit submission outside the Survey failure handler: cancellation
        # after finalization must not rewrite the successfully saved artifact.
        try:
            await _submit_automatic_artifact(artifact, entry.options)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Automatic audit submission failed for %s: %s",
                entry.task_id,
                type(exc).__name__,
            )


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
async def create_survey(
    req: SurveyRequest,
    request: Request,
    x_litagent_operation_token: str | None = Header(default=None),
):
    """Submit a survey and return its API-owned queued task."""
    options = RunOptions(
        report_language=req.report_language,
        human_review=req.human_review,
        review_structure=req.review_structure,
        auto_audit_mode=req.auto_audit_mode,
        auto_audit_max_versions=req.auto_audit_max_versions,
    )
    if req.auto_audit_mode != "off":
        _require_operation_token(request, x_litagent_operation_token)
        capabilities = (await _report_audit_service()).capabilities()
        if not capabilities["judge_available"]:
            raise HTTPException(409, capabilities["judge_reason"])
        if req.auto_audit_mode == "judge_jev" and not capabilities["jev_available"]:
            raise HTTPException(409, capabilities["jev_reason"])
    if req.human_review:
        _require_operation_token(request, x_litagent_operation_token)
        service = await _interactive_service()
        run_id = await service.start(
            req.query,
            options,
            **(
                {"citation_style": req.citation_style}
                if req.citation_style != "numbered"
                else {}
            ),
        )
        return SurveyStatus(task_id=run_id, status="queued", progress="queued")
    entry = _new_entry(req.query, options=options)
    entry.recorder.set_citation_style(req.citation_style)
    return _status(entry)


@app.delete("/survey/{task_id}", status_code=202, response_model=SurveyStatus)
async def cancel_survey(
    task_id: str,
    response: Response,
    request: Request = None,
    x_litagent_operation_token: str | None = Header(default=None),
):
    """Request cooperative cancellation for a queued or running survey."""
    entry = app.state.tasks.get(task_id)
    if entry is None:
        if request is None:
            raise HTTPException(status_code=401, detail="Operation token required")
        _require_operation_token(request, x_litagent_operation_token)
        try:
            await (await _interactive_service()).cancel(task_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Task '{task_id}' not found"
            ) from exc
        return SurveyStatus(task_id=task_id, status="cancelled", progress="cancelled")
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
        try:
            state = await (await _interactive_service()).get(task_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Task '{task_id}' not found"
            ) from exc
        delivery = (state.get("result") or {}).get("delivery") or {}
        return SurveyStatus(
            task_id=task_id,
            status=state["status"],
            progress=(
                state.get("review", {}).get("stage", state["status"])
                if state.get("review")
                else state["status"]
            ),
            error=state.get("error"),
            delivery_status=delivery.get("status"),
        )
    status = _status(entry)
    if not status.task_id:
        status.task_id = task_id
    return status


@app.get("/survey/{task_id}/report", response_model=SurveyReport)
async def get_survey_report(task_id: str):
    """Return a completed or cooperatively cancelled survey report."""
    entry = app.state.tasks.get(task_id)
    if entry is None:
        try:
            state = await (await _interactive_service()).get(task_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Task '{task_id}' not found"
            ) from exc
        if state["status"] in {"queued", "running", "awaiting_human"}:
            raise HTTPException(status_code=409, detail="Survey is not terminal")
        if state["status"] in {"failed", "interrupted"}:
            raise HTTPException(
                status_code=500, detail=state.get("error") or state["status"]
            )
        result = dict(state.get("result") or {})
        if not result:
            raise HTTPException(
                status_code=500, detail="Terminal report is unavailable"
            )
        return SurveyReport(task_id=task_id, **result)
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
async def stream_survey_events(
    task_id: str,
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
):
    """Stream the recorder timeline and close after the task reaches terminal state."""
    entry = app.state.tasks.get(task_id)
    if not isinstance(entry, SurveyTaskEntry):
        service = await _interactive_service()
        try:
            await service.get(task_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404, detail=f"Task '{task_id}' not found"
            ) from exc

        async def interactive_stream():
            try:
                offset = max(0, int(last_event_id or 0))
            except ValueError:
                offset = 0
            while True:
                for event in await service.events(task_id, after=offset):
                    offset = event["seq"]
                    yield (
                        f"id: {offset}\nevent: trace\ndata: "
                        f"{json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                    )
                state = await service.get(task_id)
                if state["status"] in {
                    "awaiting_human",
                    "completed",
                    "cancelled",
                    "rejected",
                    "failed",
                    "interrupted",
                }:
                    yield f"event: state\ndata: {json.dumps(state, ensure_ascii=False, default=str)}\n\n"
                    return
                await asyncio.sleep(0.3)

        return StreamingResponse(interactive_stream(), media_type="text/event-stream")

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


@app.get("/workbench/context")
async def get_workbench_context():
    """Return local UI capabilities and the same-origin mutation token."""
    mode = os.getenv("LITAGENT_DEMO_MODE", "live").lower()
    live_enabled = mode == "live"
    audit_service = app.state.report_audit_service or ReportAuditService(
        app.state.config or load_config()
    )
    return {
        "schema_version": 1,
        "operation_token": app.state.operation_token,
        "mode": "live" if live_enabled else "offline",
        "live_enabled": live_enabled,
        "report_languages": ["zh-CN", "en"],
        "export_formats": ["md", "pdf", "docx"],
        "format_capabilities": format_capabilities(),
        "review_structures": ["academic_v2", "academic_v1", "method_transparent_v1"],
        "human_review_stages": ["paper_selection", "final_report"],
        "auto_audit_capabilities": audit_service.capabilities(),
    }


@app.get("/workbench/runs")
async def list_workbench_runs():
    """List durable interactive runs for the local workbench."""
    return await (await _interactive_service()).list()


@app.get("/workbench/benchmarks")
async def list_survey_benchmarks():
    """List valid, unambiguous local Survey benchmark archives."""
    return _benchmark_archive().list()


@app.get("/workbench/benchmarks/{benchmark_id}")
async def get_survey_benchmark(benchmark_id: str):
    """Return one validated Survey benchmark without changing its contents."""
    try:
        archive = _benchmark_archive()
        result = archive.get(benchmark_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="benchmark_not_found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    payload = result.model_dump(mode="json")
    payload["corrected_diagnostic"] = archive.corrected_diagnostic(benchmark_id)
    return payload


@app.get("/workbench/benchmarks/{benchmark_id}/observations/{observation_id}")
async def get_survey_benchmark_observation(benchmark_id: str, observation_id: str):
    """Return an observation only when its linked artifact is intact."""
    try:
        archive = _benchmark_archive()
        payload = archive.observation(benchmark_id, observation_id)
        corrected = archive.corrected_diagnostic(benchmark_id)
        payload["corrected_observation"] = next(
            (
                item
                for item in (corrected or {}).get("observations", [])
                if item.get("run_id") == observation_id
            ),
            None,
        )
        return payload
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="observation_not_found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/survey/{task_id}/review")
async def get_survey_review(task_id: str):
    """Return the current server-owned review payload, if any."""
    try:
        state = await (await _interactive_service()).get(task_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_id}' not found"
        ) from exc
    return {"task_id": task_id, "status": state["status"], "pending": state["review"]}


@app.post("/survey/{task_id}/review/decision", status_code=202)
async def submit_survey_review(
    task_id: str,
    decision: HumanDecision,
    request: Request,
    x_litagent_operation_token: str | None = Header(default=None),
):
    """Persist a version-bound decision and queue one continuation."""
    _require_operation_token(request, x_litagent_operation_token)
    try:
        return await (await _interactive_service()).submit_decision(task_id, decision)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_id}' not found"
        ) from exc
    except ValueError as exc:
        code = str(exc)
        status = (
            409
            if code
            in {
                "decision_conflict",
                "review_not_pending",
                "stale_review",
            }
            else 422
        )
        raise HTTPException(status_code=status, detail=code) from exc


@app.get("/survey/{task_id}/versions")
async def list_survey_versions(task_id: str):
    """Return immutable server-owned report revisions newest-first."""
    try:
        versions = await (await _interactive_service()).versions(task_id)
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_id}' not found"
        ) from exc
    return {"task_id": task_id, "versions": versions}


@app.get("/survey/{task_id}/view")
async def get_survey_view(task_id: str, citation_style: CitationStyle | None = None):
    """Return the complete durable projection used by the research workbench."""
    try:
        return await (await _interactive_service()).get_view(
            task_id, **({"citation_style": citation_style} if citation_style else {})
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_id}' not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/survey/{task_id}/versions/{revision}")
async def get_survey_version(
    task_id: str, revision: int, citation_style: CitationStyle = "numbered"
):
    """Return one immutable report revision without rendering or mutation."""
    try:
        return await (await _interactive_service()).get_version(
            task_id,
            revision,
            **(
                {"citation_style": citation_style}
                if citation_style != "numbered"
                else {}
            ),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Report revision '{revision}' not found"
        ) from exc
    except ValueError as exc:
        code = str(exc)
        status = 422 if code == "invalid_report_revision" else 409
        raise HTTPException(status_code=status, detail=code) from exc

    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


async def _resolve_formatting_source(task_id: str, revision: int):
    """Resolve an immutable revision before rendering or exporting its bibliography."""
    service = await _interactive_service()
    try:
        version = await service.get_version(task_id, revision)
        report = version["report"]
        artifact = await service.get_audit_artifact(task_id)
    except KeyError as exc:
        if exc.args and isinstance(exc.args[0], tuple):
            raise ValueError("report_revision_not_found") from exc
        artifact = _flow_artifact(task_id)
        if artifact is None:
            raise KeyError(task_id)
        if revision != 1:
            raise ValueError("report_revision_not_found")
        report = artifact.get("report") or {}
    artifact = artifact or {}
    if not report.get("survey"):
        raise ValueError("report_unavailable")
    metadata = report.get("metadata") or {}
    version = ReportVersion(
        run_id=task_id,
        revision=revision,
        report_language=metadata.get("report_language", "en"),
        survey=report["survey"],
        human_status=(metadata.get("human_review") or {}).get(
            "status", "not_requested"
        ),
    )
    ledger = {
        i["evidence_id"]: i
        for i in extract_report_evidence(artifact.get("nodes") or {})
    }
    return (
        version,
        report,
        ledger,
        paper_catalog_from_nodes(artifact.get("nodes") or {}),
    )


class ReportAuditRequest(BaseModel):
    """Accept only an immutable target and bounded server-owned audit choices."""

    model_config = ConfigDict(extra="forbid")
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    include_jev: bool = False
    new_attempt: bool = False


async def _report_audit_service() -> ReportAuditService:
    """Reuse one application-owned executor and its shutdown lifecycle."""
    if app.state.report_audit_service is None:
        app.state.report_audit_service = ReportAuditService(
            app.state.config or load_config()
        )
        await app.state.report_audit_service.initialize()
    return app.state.report_audit_service


async def _report_audit_snapshot(task_id: str, revision: int) -> dict:
    """Freeze exact source text and host-selected evidence without rendering styles."""
    try:
        version, report, ledger, _ = await _resolve_formatting_source(task_id, revision)
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=404, detail="report_revision_not_found"
        ) from exc
    try:
        artifact = await (await _interactive_service()).get_audit_artifact(task_id)
    except KeyError:
        artifact = _flow_artifact(task_id) or {}
    return _audit_snapshot_from_source(version, report, ledger, artifact)


def _audit_snapshot_from_source(
    version: ReportVersion, report: dict, ledger: dict, artifact: dict
) -> dict:
    """Freeze version-owned input without rendering or reading a newer report."""
    metadata = report.get("metadata") or {}
    generation = (metadata.get("config_summary") or artifact.get("config") or {}).get(
        "llm"
    ) or {}
    # Reconstructed observation timestamps are not evidence. Excluding them keeps
    # identical pending versions deduplicated even after local event replay.
    source_nodes = {
        key: {
            "kind": node.get("kind"),
            "task_id": node.get("task_id"),
            "output": copy.deepcopy(node.get("output")),
        }
        for key, node in (artifact.get("nodes") or {}).items()
        if isinstance(node, dict)
    }
    return {
        "generation_provider": {
            "model": generation.get("model"),
            "origin": generation.get("origin") or generation.get("base_url"),
        },
        "source_run_id": version.run_id,
        "source_report_revision": version.revision,
        "source_report_sha256": version.content_sha256,
        "report": copy.deepcopy(report),
        "ledger": copy.deepcopy(ledger),
        "context": {
            "query": metadata.get("query") or artifact.get("query") or "",
            "report_language": version.report_language,
            "review_structure": metadata.get("review_structure") or "legacy",
            "review_methodology": copy.deepcopy(
                metadata.get("review_methodology") or {}
            ),
        },
        "artifact": {
            "run_id": version.run_id,
            "query": artifact.get("query"),
            "report": copy.deepcopy(report),
            "nodes": source_nodes,
        },
    }


async def _submit_automatic_audit(
    task_id: str, revision: int, options: RunOptions
) -> None:
    """Submit only a newly persisted interactive version with stored consent."""
    if options.auto_audit_mode == "off":
        return
    snapshot = await _report_audit_snapshot(task_id, revision)
    await (await _report_audit_service()).submit_automatic(
        snapshot,
        mode=options.auto_audit_mode,
        max_versions=options.auto_audit_max_versions,
    )


async def _submit_automatic_artifact(artifact: dict, options: RunOptions) -> None:
    """Submit a finalized ordinary API run without archive lookup or replay."""
    if options.auto_audit_mode == "off":
        return
    report = artifact.get("report") or {}
    if not report.get("survey"):
        return
    metadata = report.get("metadata") or {}
    version = ReportVersion(
        run_id=artifact["run_id"],
        revision=metadata.get("report_revision", 1),
        report_language=metadata.get("report_language", options.report_language),
        survey=report["survey"],
    )
    ledger = {
        item["evidence_id"]: item
        for item in extract_report_evidence(artifact.get("nodes") or {})
    }
    snapshot = _audit_snapshot_from_source(version, report, ledger, artifact)
    await (await _report_audit_service()).submit_automatic(
        snapshot,
        mode=options.auto_audit_mode,
        max_versions=options.auto_audit_max_versions,
    )


@app.get("/survey/{task_id}/versions/{revision}/audits")
async def list_report_audits(task_id: str, revision: int):
    """Expose capabilities and attempts for exactly one available report version."""
    snapshot = await _report_audit_snapshot(task_id, revision)
    result = await (await _report_audit_service()).list(task_id, revision)
    result["capabilities"].update(
        (await _report_audit_service()).model_comparison(snapshot)
    )
    return {**result, "source_report_sha256": snapshot["source_report_sha256"]}


@app.get("/survey/{task_id}/versions/{revision}/audits/dry-run")
async def dry_run_report_audit(task_id: str, revision: int, include_jev: bool = False):
    """Report credential and input gates without creating tasks or provider calls."""
    snapshot = await _report_audit_snapshot(task_id, revision)
    return (await _report_audit_service()).dry_run(snapshot, include_jev=include_jev)


@app.post("/survey/{task_id}/versions/{revision}/audits", status_code=202)
async def start_report_audit(
    task_id: str,
    revision: int,
    body: ReportAuditRequest,
    request: Request,
    x_litagent_operation_token: str | None = Header(default=None),
):
    """Queue a manually authorized independent audit after exact SHA comparison."""
    _require_operation_token(request, x_litagent_operation_token)
    snapshot = await _report_audit_snapshot(task_id, revision)
    if snapshot["source_report_sha256"] != body.report_sha256:
        raise HTTPException(status_code=409, detail="report_sha_mismatch")
    try:
        return await (await _report_audit_service()).submit(
            snapshot, include_jev=body.include_jev, new_attempt=body.new_attempt
        )
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/survey/{task_id}/versions/{revision}/audits/{audit_id}")
async def get_report_audit(task_id: str, revision: int, audit_id: str):
    """Read one immutable-version audit; never fall back to another revision."""
    try:
        return await (await _report_audit_service()).get(task_id, revision, audit_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="audit_not_found") from exc


@app.post("/survey/{task_id}/versions/{revision}/audits/{audit_id}/cancel")
async def cancel_report_audit(
    task_id: str,
    revision: int,
    audit_id: str,
    request: Request,
    x_litagent_operation_token: str | None = Header(default=None),
):
    """Cancel this audit without touching the source Survey or approval state."""
    _require_operation_token(request, x_litagent_operation_token)
    try:
        return await (await _report_audit_service()).cancel(task_id, revision, audit_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="audit_not_found") from exc


@app.get("/survey/{task_id}/bibliography")
async def download_survey_bibliography(
    task_id: str, revision: int, format: Literal["bibtex", "ris", "csl_json"]
):
    """Export only the current revision's cited metadata without network enrichment."""
    try:
        version, _, ledger, catalog = await _resolve_formatting_source(
            task_id, revision
        )
        snapshot = build_bibliographic_snapshot(version, ledger, catalog)
        content = export_bibliography(snapshot, format)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="report_not_found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    types = {
        "bibtex": ("bib", "application/x-bibtex; charset=utf-8"),
        "ris": ("ris", "application/x-research-info-systems; charset=utf-8"),
        "csl_json": ("json", "application/vnd.citationstyles.csl+json; charset=utf-8"),
    }
    extension, media = types[format]
    return Response(
        content=content,
        media_type=media,
        headers={
            "Content-Disposition": f'attachment; filename="litagent-{task_id}-r{revision}.{extension}"',
            "X-Report-SHA256": version.content_sha256,
            "X-Bibliography-Warning-Count": str(len(snapshot["warnings"])),
        },
    )


@app.get("/survey/{task_id}/export")
async def export_survey(
    task_id: str,
    revision: int,
    format: str,
    citation_style: CitationStyle = "evidence",
):
    """Download one immutable current revision in a supported format."""
    try:
        service = await _interactive_service()
        if citation_style == "evidence":
            exported = await service.export(task_id, revision, format)
        else:
            exported = await service.export(
                task_id, revision, format, citation_style=citation_style
            )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail=f"Task '{task_id}' not found"
        ) from exc
    except ValueError as exc:
        code = str(exc)
        if code == "report_revision_not_found" and _flow_artifact(task_id) is not None:
            try:
                version, report, ledger, catalog = await _resolve_formatting_source(
                    task_id, revision
                )
                reader = await asyncio.to_thread(
                    build_reader_report,
                    version,
                    evidence_ledger=ledger,
                    paper_catalog=catalog,
                    citation_style=citation_style,
                )
                exported = await LocalReportExporter().export(
                    version,
                    format=format,
                    publishable=bool((report.get("delivery") or {}).get("publishable")),
                    citation_style=citation_style,
                    reader_report=reader,
                )
            except KeyError as missing:
                raise HTTPException(
                    status_code=404, detail="report_revision_not_found"
                ) from missing
            except ValueError as invalid:
                raise HTTPException(status_code=422, detail=str(invalid)) from invalid
            except RuntimeError as unavailable:
                raise HTTPException(
                    status_code=503, detail=str(unavailable)
                ) from unavailable
        else:
            status = 404 if code == "report_revision_not_found" else 422
            raise HTTPException(status_code=status, detail=code) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(
        content=exported.content,
        media_type=exported.media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{exported.filename}"',
            "X-Report-SHA256": exported.report_sha256,
        },
    )


@app.get("/flow-demo", include_in_schema=False)
async def flow_demo_page():
    """Return the packaged flow-demo user interface."""
    return FileResponse(FLOW_DEMO_STATIC)


@app.post("/flow-demo/runs", status_code=201)
async def create_flow_demo():
    """Start the fixed live query through the shared survey task owner."""
    if os.getenv("LITAGENT_DEMO_MODE", "live").lower() != "live":
        raise HTTPException(status_code=403, detail="live_demo_disabled")
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
    if artifact is None:
        artifact = app.state.flow_samples.get(run_id)
    return artifact


def _flow_source(run_id: str) -> Literal["sample", "archive", "live"]:
    """Identify the owner of a flow-demo run without trusting its payload."""
    if run_id in app.state.flow_runs:
        return "live"
    if run_id in app.state.flow_samples:
        return "sample"
    return "archive"


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
        "source": _flow_source(str(artifact["run_id"])),
        "partial": bool(report.get("partial", False)),
        "trace_url": (artifact.get("trace") or {}).get("trace_url"),
    }


@app.get("/flow-demo/context")
async def get_flow_demo_context():
    """Return public demo capabilities without initializing runtime services."""
    mode = os.getenv("LITAGENT_DEMO_MODE", "live").lower()
    live_enabled = mode == "live"
    return {
        "schema_version": 1,
        "mode": "live" if live_enabled else "offline",
        "live_enabled": live_enabled,
        "default_run_id": "sample-ready",
        "benchmarks": {
            "rag": {
                "status": "completed",
                "result": "benchmarks/rag/RESULTS.md",
            },
            "survey": {
                "status": "implementation_complete_live_acceptance_pending",
                "result": None,
            },
        },
    }


@app.get("/flow-demo/runs")
async def list_flow_demo_runs():
    """List live and archived flow-demo runs."""
    artifacts = dict(app.state.flow_samples)
    artifacts.update(app.state.flow_archive_index)
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


@app.get("/flow-demo/runs/{run_id}/view")
async def get_flow_demo_view(run_id: str, citation_style: CitationStyle | None = None):
    """Return one normalized, browser-facing flow-demo projection."""
    artifact = _flow_artifact(run_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Flow run '{run_id}' not found")
    style = citation_style or artifact.get("citation_style", "numbered")
    try:
        view = await asyncio.to_thread(
            project_flow_demo_view,
            artifact,
            source=_flow_source(run_id),
            citation_style=style,
        )
        view["format_capabilities"] = format_capabilities()
        view["citation_style"] = style
        return view
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


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
    if artifact is None and run_id in app.state.flow_samples:
        return Response(
            content=json.dumps(
                app.state.flow_samples[run_id], ensure_ascii=False, indent=2
            ),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.json"'},
        )
    if artifact is None:
        raise HTTPException(
            status_code=404, detail=f"Completed flow run '{run_id}' not found"
        )
    return FileResponse(
        app.state.flow_repository.path_for(run_id),
        media_type="application/json",
        filename=f"{run_id}.json",
    )
