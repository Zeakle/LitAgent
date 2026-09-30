"""Execute local human-reviewed surveys across durable safe boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from collections.abc import Callable
from typing import Any

from litagent.config import AppConfig
from litagent.contracts import build_config_summary
from litagent.demo import project_flow_demo_view
from litagent.interaction import ExecutionCheckpoint, RunAdvanceResult
from litagent.interaction_store import SqliteReviewRepository
from litagent.logging import get_logger
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.task_graph import TaskGraph
from litagent.report_export import LocalReportExporter
from litagent.report_render import (
    build_reader_report,
    paper_catalog_from_nodes,
    reader_report_projection,
    render_report_projection,
)
from litagent.runner import LitAgent
from litagent.workbench import (
    DecisionReceipt,
    ExportResult,
    HumanDecision,
    ReportVersion,
    ReviewRequest,
    RunOptions,
)

logger = get_logger("interactive_service")

AgentFactory = Callable[[AppConfig, Callable[[str, dict], None]], LitAgent]


class InteractiveRunService:
    """Own local execution, durable decisions, and segment resource lifetimes."""

    def __init__(
        self,
        config: AppConfig,
        repository: SqliteReviewRepository,
        *,
        agent_factory: AgentFactory | None = None,
        external_trace_factory: (
            Callable[[str, str], Callable[[str, dict], None] | None] | None
        ) = None,
        semaphore: asyncio.Semaphore | None = None,
        archive_repository: ArchiveRepository | None = None,
    ) -> None:
        """Save dependencies without starting background work."""
        self._config = config
        self._repository = repository
        self._agent_factory = agent_factory or (
            lambda cfg, hook: LitAgent(cfg, trace_hook=hook)
        )
        self._external_trace_factory = external_trace_factory
        self._semaphore = semaphore
        self._archive_repository = archive_repository or ArchiveRepository()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._recorders: dict[str, RunRecorder] = {}
        self._exporter = LocalReportExporter()

    async def initialize(self) -> None:
        """Initialize storage, mark abandoned active calls, and resume queued work."""
        await self._repository.initialize()
        await self._repository.recover_startup()
        for row in await self._repository.list_runs():
            if row["status"] == "queued":
                self._schedule(str(row["run_id"]))

    async def shutdown(self) -> None:
        """Cancel process-owned segments; startup will mark them interrupted."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def start(self, query: str, options: RunOptions) -> str:
        """Create and asynchronously start one human-reviewed run."""
        if not options.human_review:
            raise ValueError("interactive_service_requires_human_review")
        run_id = uuid.uuid4().hex[:16]
        await self._repository.create_run(run_id, query, options)
        self._schedule(run_id)
        return run_id

    async def submit_decision(
        self, run_id: str, decision: HumanDecision
    ) -> DecisionReceipt:
        """Persist one decision and queue exactly one continuation."""
        receipt = await self._repository.decide(run_id, decision)
        if receipt.execution_status == "queued":
            self._schedule(run_id)
        return receipt

    async def cancel(self, run_id: str) -> None:
        """Cancel a process-owned segment and persist a terminal cancellation."""
        task = self._tasks.get(run_id)
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._repository.mark_terminal(run_id, "cancelled")

    async def get(self, run_id: str) -> dict[str, Any]:
        """Return a browser-safe durable projection for one run."""
        row = await self._repository.get_run(run_id)
        result = json.loads(row["result_json"]) if row.get("result_json") else None
        review = json.loads(row["review_json"]) if row.get("review_json") else None
        review_payload = None
        if row["status"] == "awaiting_human" and review is not None:
            checkpoint = ExecutionCheckpoint.model_validate(
                await self._repository.load_checkpoint(run_id)
            )
            review_payload = self._review_projection(checkpoint, review)
        return {
            "run_id": row["run_id"],
            "query": row["query"],
            "status": row["status"],
            "options": json.loads(row["options_json"]),
            "review": review_payload,
            "result": result,
            "error": row.get("error"),
            "updated_at": row["updated_at"],
        }

    async def get_view(self, run_id: str) -> dict[str, Any]:
        """Return one complete browser projection from durable server state."""
        row = await self._repository.get_run(run_id)
        state = await self.get(run_id)
        report = dict(state.get("result") or {}) or None
        graph = {"tasks": {}, "dependencies": {}, "nodes": {}}
        checkpoint: ExecutionCheckpoint | None = None

        if row.get("checkpoint_path"):
            checkpoint = ExecutionCheckpoint.model_validate(
                await self._repository.load_checkpoint(run_id)
            )
            validated_graph = TaskGraph.from_snapshot(checkpoint.graph).to_snapshot()
            graph["tasks"] = {
                str(task["task_id"]): task for task in validated_graph["tasks"]
            }
            graph["dependencies"] = validated_graph["dependencies"]
            if report is None and checkpoint.report_data:
                report = dict(checkpoint.report_data)

        artifact = self._archive_repository.get(run_id)
        archive_view = (
            project_flow_demo_view(artifact, source="archive")
            if artifact is not None
            else None
        )
        if archive_view is not None:
            archive_graph = archive_view.get("dag") or {}
            if not graph["tasks"]:
                graph["tasks"] = dict(archive_graph.get("tasks") or {})
                graph["dependencies"] = dict(archive_graph.get("dependencies") or {})
            graph["nodes"] = dict(archive_graph.get("nodes") or {})
            if report is None and artifact.get("report"):
                report = dict(artifact["report"])

        timeline = await self._repository.list_events(run_id)
        versions = await self._repository.list_report_versions(run_id)
        evaluation = dict((report or {}).get("evaluation") or {})
        quality = (report or {}).get("quality")
        delivery = (report or {}).get("delivery")
        metadata = dict((report or {}).get("metadata") or {})
        archive_summary = (archive_view or {}).get("summary") or {}
        evidence = list((archive_view or {}).get("evidence") or [])
        if not evidence:
            raw_evidence = (report or {}).get("evidence") or metadata.get(
                "selected_evidence"
            )
            if isinstance(raw_evidence, list):
                evidence = [item for item in raw_evidence if isinstance(item, dict)]

        status = str(row["status"])
        review = state.get("review")
        trace = dict((archive_view or {}).get("trace") or {})
        capabilities = {
            "can_cancel": status in {"queued", "running", "awaiting_human"},
            "can_review": status == "awaiting_human" and review is not None,
            "can_export": bool(versions),
            "can_download_artifact": artifact is not None,
            "has_trace": bool(trace.get("trace_url")),
        }
        token_usage = metadata.get("token_usage") or {}
        summary = {
            "run_id": run_id,
            "query": row["query"],
            "status": status,
            "started_at": row.get("created_at"),
            "completed_at": (
                row.get("updated_at")
                if status
                in {"completed", "cancelled", "rejected", "failed", "interrupted"}
                else None
            ),
            "elapsed_ms": archive_summary.get("elapsed_ms")
            or (metadata.get("execution") or {}).get("elapsed_ms"),
            "total_tokens": (
                archive_summary.get("total_tokens")
                if archive_summary.get("total_tokens") is not None
                else token_usage.get("total_tokens")
            ),
            "quality_status": (quality or {}).get("status"),
            "delivery_status": (delivery or {}).get("status"),
            "publishable": (delivery or {}).get("publishable"),
            "reason_codes": list((delivery or {}).get("reason_codes") or []),
            "report_language": json.loads(row["options_json"]).get("report_language"),
            "error": row.get("error"),
        }
        survey_text = str((report or {}).get("survey") or "")
        rendered_report = render_report_projection(survey_text)
        if survey_text and evidence:
            report_version = ReportVersion(
                run_id=run_id,
                revision=max((item["revision"] for item in versions), default=1),
                report_language=metadata.get("report_language", "en"),
                survey=survey_text,
            )
            reader = build_reader_report(
                report_version,
                evidence_ledger={
                    item["evidence_id"]: item
                    for item in evidence
                    if isinstance(item, dict) and item.get("evidence_id")
                },
                paper_catalog=paper_catalog_from_nodes(graph["nodes"]),
            )
            rendered_report = reader_report_projection(reader)
        return {
            "schema_version": 1,
            "source": "interactive",
            "summary": summary,
            "graph": graph,
            "timeline": timeline,
            "report": report,
            "rendered_report": rendered_report,
            "evidence": evidence,
            "evaluation": evaluation,
            "quality": quality,
            "delivery": delivery,
            "review": review,
            "versions": versions,
            "trace": trace,
            "capabilities": capabilities,
        }

    async def get_version(self, run_id: str, revision: int) -> dict[str, Any]:
        """Return one immutable report revision with its verified content hash."""
        if revision < 1:
            raise ValueError("invalid_report_revision")
        report = await self._repository.get_report_version(run_id, revision)
        survey = report.get("survey")
        if not isinstance(survey, str) or not survey:
            raise ValueError("report_unavailable")
        content_sha = hashlib.sha256(survey.encode("utf-8")).hexdigest()
        summaries = await self._repository.list_report_versions(run_id)
        summary = next(
            (item for item in summaries if item["revision"] == revision), None
        )
        if summary is None or summary["report_sha256"] != content_sha:
            raise ValueError("report_revision_sha_mismatch")
        archive = self._archive_repository.get(run_id)
        archive_view = (
            project_flow_demo_view(archive, source="archive")
            if archive is not None
            else None
        )
        rendered_report = render_report_projection(survey)
        if archive_view is not None and archive_view.get("evidence"):
            evidence = archive_view["evidence"]
            version = ReportVersion(
                run_id=run_id,
                revision=revision,
                report_language=(report.get("metadata") or {}).get(
                    "report_language", "en"
                ),
                survey=survey,
            )
            reader = build_reader_report(
                version,
                evidence_ledger={
                    item["evidence_id"]: item
                    for item in evidence
                    if isinstance(item, dict) and item.get("evidence_id")
                },
                paper_catalog=paper_catalog_from_nodes(
                    (archive or {}).get("nodes") or {}
                ),
            )
            rendered_report = reader_report_projection(reader)
        return {
            "run_id": run_id,
            "revision": revision,
            "content_sha256": content_sha,
            "report_sha256": summary["report_sha256"],
            "created_at": summary["created_at"],
            "report": report,
            "rendered_report": rendered_report,
        }

    @staticmethod
    def _review_projection(
        checkpoint: ExecutionCheckpoint, review: dict[str, Any]
    ) -> dict[str, Any]:
        projection = dict(review)
        projection["allowed_actions"] = list(
            ReviewRequest.model_validate(review).allowed_actions
        )
        if checkpoint.boundary == "paper_selection":
            graph = checkpoint.graph
            task = next(
                (
                    item
                    for item in graph.get("tasks", [])
                    if item.get("task_id") == "dedup"
                ),
                {},
            )
            candidates = []
            selected = set(review.get("automatic_selected_paper_ids", []))
            for item in task.get("result") or []:
                if not isinstance(item, dict):
                    continue
                item_id = item.get("paper_id") or item.get("id")
                if not item_id:
                    continue
                candidates.append(
                    {
                        "paper_id": item_id,
                        "title": item.get("title", ""),
                        "abstract": item.get("abstract", ""),
                        "source": item.get("source", ""),
                        "automatically_selected": item_id in selected,
                    }
                )
            projection["candidates"] = candidates
        else:
            projection["report"] = checkpoint.report_data
        return projection

    async def export(
        self,
        run_id: str,
        revision: int,
        format: str,
        citation_style: str = "evidence",
    ) -> ExportResult:
        """Export an immutable server-owned report revision."""
        try:
            report_data = await self._repository.get_report_version(run_id, revision)
        except KeyError as exc:
            raise ValueError("report_revision_not_found") from exc
        if not isinstance(report_data, dict) or not report_data.get("survey"):
            raise ValueError("report_unavailable")
        metadata = report_data.get("metadata", {})
        human_status = (metadata.get("human_review") or {}).get(
            "status", "not_requested"
        )
        version = ReportVersion(
            run_id=run_id,
            revision=revision,
            report_language=metadata.get("report_language", "en"),
            survey=report_data["survey"],
            human_status=human_status,
        )
        reader = None
        if citation_style == "numbered":
            archive = self._archive_repository.get(run_id)
            archive_view = (
                project_flow_demo_view(archive, source="archive")
                if archive is not None
                else None
            )
            evidence = (archive_view or {}).get("evidence") or []
            if not evidence:
                raise ValueError("reader_projection_unavailable")
            reader = build_reader_report(
                version,
                evidence_ledger={
                    item["evidence_id"]: item
                    for item in evidence
                    if isinstance(item, dict) and item.get("evidence_id")
                },
                paper_catalog=paper_catalog_from_nodes(
                    (archive or {}).get("nodes") or {}
                ),
            )
        return await self._exporter.export(
            version,
            format=format,
            publishable=bool((report_data.get("delivery") or {}).get("publishable")),
            citation_style=citation_style,
            reader_report=reader,
        )

    async def versions(self, run_id: str) -> list[dict[str, Any]]:
        """Return immutable report-version summaries newest-first."""
        return await self._repository.list_report_versions(run_id)

    async def list(self) -> list[dict[str, Any]]:
        """Return compact durable runs in newest-first order."""
        rows = await self._repository.list_runs()
        return [
            {
                "run_id": row["run_id"],
                "query": row["query"],
                "status": row["status"],
                "options": json.loads(row["options_json"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def events(self, run_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        """Return persisted execution events after the requested cursor."""
        await self._repository.get_run(run_id)
        return await self._repository.list_events(run_id, after=after)

    def _schedule(self, run_id: str) -> None:
        existing = self._tasks.get(run_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.create_task(self._execute(run_id))
        self._tasks[run_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(run_id, None))

    async def _trace_hook(self, run_id: str, query: str):
        local = self._repository.event_hook(run_id)
        recorder = self._recorders.get(run_id)
        if recorder is None:
            recorder = RunRecorder(run_id, query, self._archive_repository)
            recorder.set_config_summary(build_config_summary(self._config))
            # Rebuild the local artifact after a safe-boundary process restart.
            for item in await self._repository.list_events(run_id):
                recorder(item["event"], item["data"])
            self._recorders[run_id] = recorder
        hooks: list[Any] = [local, recorder]
        if self._config.observability.enabled:
            tracer = LangFuseTracer(
                host=self._config.observability.langfuse_host,
                public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=run_id,
            )
            recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            hooks.append(
                RedactingTraceHook(
                    tracer,
                    payload_mode=self._config.observability.payload_mode,
                )
            )
        external = (
            self._external_trace_factory(run_id, query)
            if self._external_trace_factory
            else None
        )

        if external is not None:

            def safe_external(event: str, payload: dict[str, Any]) -> None:
                try:
                    external(event, payload)
                except Exception:
                    logger.debug("External trace hook failed", exc_info=True)

            hooks.append(safe_external)
        return CompositeTraceHook(*hooks)

    async def _execute(self, run_id: str) -> None:
        if not await self._repository.claim_execution(run_id):
            return
        row = await self._repository.get_run(run_id)
        query = str(row["query"])
        options = RunOptions.model_validate_json(row["options_json"])
        checkpoint = None
        decision = None
        if row.get("checkpoint_path"):
            checkpoint = ExecutionCheckpoint.model_validate(
                await self._repository.load_checkpoint(run_id)
            )
        if row.get("decision_json"):
            decision = HumanDecision.model_validate_json(row["decision_json"])
        hook = await self._trace_hook(run_id, query)
        hook("run.resumed" if checkpoint else "survey.start", {"run_id": run_id})
        try:
            agent = self._agent_factory(self._config, hook)
            if self._semaphore is None:
                async with agent:
                    result: RunAdvanceResult = await agent.advance(
                        query,
                        options=options,
                        checkpoint=checkpoint,
                        decision=decision,
                        run_id=run_id,
                    )
            else:
                async with self._semaphore:
                    async with agent:
                        result = await agent.advance(
                            query,
                            options=options,
                            checkpoint=checkpoint,
                            decision=decision,
                            run_id=run_id,
                        )
            await self._persist_result(run_id, result)
        except asyncio.CancelledError:
            await self._repository.mark_terminal(run_id, "interrupted")
            raise
        except Exception as exc:
            logger.exception("Interactive run %s failed", run_id)
            await self._repository.mark_terminal(
                run_id,
                "failed",
                error=f"{type(exc).__name__}: {exc}"[:1024],
            )
            recorder = self._recorders.get(run_id)
            if recorder is not None:
                try:
                    recorder.finalize(error=f"{type(exc).__name__}: {exc}"[:1024])
                except Exception:
                    logger.exception(
                        "Failed to finalize interactive artifact %s", run_id
                    )

    async def _persist_result(self, run_id: str, result: RunAdvanceResult) -> None:
        if result.status == "awaiting_human":
            if result.checkpoint is None or result.review is None:
                raise ValueError("awaiting_result_missing_payload")
            if result.checkpoint.report_data:
                await self._repository.save_report_version(
                    run_id,
                    result.checkpoint.report_revision,
                    result.checkpoint.report_data,
                )
            await self._repository.pause(
                run_id,
                result.checkpoint.model_dump(mode="json"),
                result.review,
            )
            self._repository.event_hook(run_id)(
                "run.paused",
                {
                    "run_id": run_id,
                    "request_id": result.review.request_id,
                    "stage": result.review.stage,
                    "revision": result.review.revision,
                },
            )
            return
        if result.report and result.report.get("survey"):
            revision = int(
                (result.report.get("metadata") or {}).get("report_revision", 1)
            )
            await self._repository.save_report_version(run_id, revision, result.report)
        recorder = self._recorders.get(run_id)
        if recorder is not None and result.status in {"completed", "rejected"}:
            recorder.finalize(report=result.report, terminal_status="completed")
        await self._repository.mark_terminal(
            run_id, result.status, result=result.report
        )
