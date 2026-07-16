"""FastAPI REST API for LitAgent.

启动:
    uvicorn litagent.api:app --host 0.0.0.0 --port 8000 --reload
Swagger UI: http://localhost:8000/docs
"""

from __future__ import annotations
import asyncio as _asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from litagent.config import load_config
from litagent.logging import get_logger
from litagent.runner import LitAgent

logger = get_logger("api")


# ═══════════════════════════════════════════════════════
# Pydantic models
# ═══════════════════════════════════════════════════════

class SurveyRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    config_path: str | None = None


class SurveyStatus(BaseModel):
    task_id: str
    status: str
    progress: str = ""
    error: str | None = None


class SurveyReport(BaseModel):
    task_id: str
    survey: str
    metadata: dict[str, Any]
    review_history: list[dict[str, Any]]
    graph_data: dict[str, Any]
    partial: bool
    evaluation: dict[str, Any] = {}
    quality: dict[str, Any] = {
        'status': 'unverified',
        'failed_metrics': [],
        'unverified_metrics': []
    }


# ═══════════════════════════════════════════════════════
# Background: TTL cleanup
# ═══════════════════════════════════════════════════════

_TASK_TTL_SECONDS = 3600   # 1 小时后清理旧任务
_CLEANUP_INTERVAL = 600    # 每 10 分钟检查一次


async def _cleanup_old_tasks() -> None:
    """后台轮询，清理已超过 TTL 的 completed/failed 任务。"""
    while True:
        await _asyncio.sleep(_CLEANUP_INTERVAL)
        now = time.time()
        expired = [
            tid for tid, t in app.state.tasks.items()
            if t["status"] in ("completed", "failed")
            and (now - t.get("_created_at", 0)) > _TASK_TTL_SECONDS
        ]
        for tid in expired:
            del app.state.tasks[tid]
        if expired:
            logger.info("Cleaned %d expired tasks", len(expired))


# ═══════════════════════════════════════════════════════
# Lifespan — 替代已废弃的 @app.on_event("startup")
# ═══════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(_app: FastAPI):
    # startup: 启动 TTL 清理协程
    cleanup_task = _asyncio.create_task(_cleanup_old_tasks())
    logger.info("LitAgent API started, TTL cleanup running")
    yield
    # shutdown: 取消清理协程
    cleanup_task.cancel()
    try:
        await cleanup_task
    except _asyncio.CancelledError:
        pass
    logger.info("LitAgent API shut down")


# ═══════════════════════════════════════════════════════
# App
# ═══════════════════════════════════════════════════════

app = FastAPI(
    title="LitAgent API",
    description="Multi-agent adversarial literature review framework — REST API",
    version="0.1.0",
    lifespan=lifespan,
)

# In-memory 任务状态（必须在 lifespan 外初始化，否则 TestClient 不可见）
app.state.tasks: dict[str, dict[str, Any]] = {}

# Phase 14 前端静态文件预留
try:
    app.mount("/static", StaticFiles(directory="static"), name="static")
except RuntimeError:
    pass


# ═══════════════════════════════════════════════════════
# Routes
# ═══════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/survey", status_code=201, response_model=SurveyStatus)
async def create_survey(req: SurveyRequest, background_tasks: BackgroundTasks):
    """提交文献综述任务，后台异步执行。"""
    task_id = str(uuid.uuid4())[:8]
    app.state.tasks[task_id] = {
        "status": "running", "progress": "planner", "result": None, "error": None,
        "_created_at": time.time(),
    }
    background_tasks.add_task(_run_survey, task_id, req.query, req.config_path)
    return SurveyStatus(task_id=task_id, status="running", progress="planner")


async def _run_survey(task_id: str, query: str, config_path: str | None) -> None:
    """后台执行 survey，完成后更新 app.state.tasks。"""
    try:
        config = load_config(config_path)
        async with LitAgent(config) as agent:
            result = await agent.run(query)
            app.state.tasks[task_id].update(
                status="completed", progress="done", result=result,
            )
    except Exception as e:
        app.state.tasks[task_id].update(
            status="failed", progress="", error=str(e),
        )


@app.get("/survey/{task_id}", response_model=SurveyStatus)
async def get_survey_status(task_id: str):
    """查询任务状态。"""
    task = app.state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
    return SurveyStatus(
        task_id=task_id, status=task["status"],
        progress=task.get("progress", ""), error=task.get("error"),
    )


@app.get("/survey/{task_id}/report", response_model=SurveyReport)
async def get_survey_report(task_id: str):
    """获取已完成 survey 的完整报告。"""
    task = app.state.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404)
    if task["status"] == "running":
        raise HTTPException(status_code=409, detail="Still running. Poll GET /survey/{task_id} first.")
    if task["status"] == "failed":
        raise HTTPException(status_code=500, detail=task.get("error", "unknown"))

    result = task["result"] or {}
    return SurveyReport(
        task_id=task_id,
        survey=result.get("survey", ""),
        metadata=result.get("metadata", {}),
        review_history=result.get("review_history", []),
        graph_data=result.get("graph_data", {}),
        partial=result.get("partial", False),
        evaluation=result.get("evaluation", {}),
        quality=result.get('quality',
            {
                'status': 'unverified',
                'failed_metrics': [],
                'unverified_metrics': []
            }
        )
    )
