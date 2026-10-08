"""Persist and execute independent audits of immutable report versions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from litagent.audit_context import SurveyAuditContext, check_structure
from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.citation_audit import JevAuditConfig, JevCitationAuditor
from litagent.benchmark.survey_judge import (
    SurveyBenchmarkJudge,
    judge_prompt_fingerprint,
)
from litagent.benchmark.survey_models import SurveyJudgeConfig
from litagent.config import AppConfig
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.recorder import RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer

TERMINAL = {"completed", "partial", "failed", "cancelled", "interrupted"}


def digest(value: Any) -> str:
    """Hash canonical JSON for immutable audit bindings."""
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    """Replace one sidecar only after a complete flushed write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class ReportAuditService:
    """Own local audit tasks, reservations, clients and append-only attempts."""

    def __init__(
        self,
        config: AppConfig,
        root: Path = Path("artifacts/audits"),
        *,
        client_factory=None,
        jev_factory=None,
        trace_factory=None,
    ):
        """Allow offline tests to replace transports without changing production policy."""
        self.config = config
        self.root = root
        self.client_factory = client_factory
        self.jev_factory = jev_factory
        self.trace_factory = trace_factory
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(1)
        self._initialized = False

    def _db(self):
        """Open the small control index; payloads live in atomic sidecars."""
        connection = sqlite3.connect(self.root / "index.sqlite3")
        connection.row_factory = sqlite3.Row
        return connection

    async def initialize(self) -> None:
        """Mark interrupted work without restarting any paid request."""
        if self._initialized:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS audits (audit_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, revision INTEGER NOT NULL, binding TEXT NOT NULL, status TEXT NOT NULL, created_at REAL NOT NULL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS automatic_audits ("
                "run_id TEXT NOT NULL, revision INTEGER NOT NULL, "
                "source_sha TEXT NOT NULL, mode TEXT NOT NULL, "
                "max_versions INTEGER NOT NULL, status TEXT NOT NULL, "
                "reason_codes TEXT NOT NULL, audit_id TEXT, "
                "budget_reserved INTEGER NOT NULL, created_at REAL NOT NULL, "
                "updated_at REAL NOT NULL, PRIMARY KEY (run_id, revision))"
            )
            # A committed reservation may precede a provider call whose outcome is
            # unknown after a crash. Keep its budget spent and never submit it again.
            db.execute(
                "UPDATE automatic_audits SET status='interrupted', "
                "reason_codes=?, updated_at=? WHERE status='reserved'",
                (
                    json.dumps(["automatic_reservation_interrupted_no_retry"]),
                    time.time(),
                ),
            )
            interrupted = db.execute(
                "SELECT audit_id FROM audits WHERE status IN ('queued','running')"
            ).fetchall()
            for row in interrupted:
                result = self._read(row["audit_id"])
                result.update(
                    status="interrupted",
                    reason_codes=["process_interrupted_no_automatic_retry"],
                )
                atomic_json(self._path(row["audit_id"]), result)
            db.execute(
                "UPDATE audits SET status='interrupted' WHERE status IN ('queued','running')"
            )
            db.execute(
                "UPDATE automatic_audits SET status='interrupted', "
                "reason_codes=?, updated_at=? WHERE audit_id IN "
                "(SELECT audit_id FROM audits WHERE status='interrupted')",
                (json.dumps(["process_interrupted_no_automatic_retry"]), time.time()),
            )
        self._initialized = True

    def _path(self, audit_id: str) -> Path:
        """Resolve only server-generated IDs to prevent path traversal."""
        if not audit_id.startswith("audit-") or len(audit_id) != 38:
            raise KeyError(audit_id)
        try:
            int(audit_id[6:], 16)
        except ValueError as exc:
            raise KeyError(audit_id) from exc
        return self.root / audit_id / "result.json"

    def _read(self, audit_id: str) -> dict:
        """Read one committed sidecar, never the source report."""
        try:
            return json.loads(self._path(audit_id).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise KeyError(audit_id) from exc

    def judge_config(self) -> SurveyJudgeConfig:
        """Resolve the explicit Judge provider or inherit the generation provider."""
        cfg = self.config.report_audit
        return SurveyJudgeConfig(
            base_url=cfg.judge_base_url or self.config.llm.base_url,
            model=cfg.judge_model or self.config.llm.model,
            api_key_env=cfg.judge_api_key_env
            or (
                "LLM_API_KEY"
                if os.getenv("LLM_API_KEY") or not os.getenv("OPENAI_API_KEY")
                else "OPENAI_API_KEY"
            ),
            max_tokens=16384,
            timeout_seconds=180,
            max_retries=0,
            response_format_mode=cfg.judge_response_format,
            thinking_mode=cfg.judge_thinking_mode,
        )

    def capabilities(self) -> dict:
        """Return models, credential availability and limits without secrets."""
        judge = self.judge_config()
        cfg = self.config.report_audit
        live = os.getenv("LITAGENT_DEMO_MODE", "live").lower() == "live"
        enabled = cfg.enabled and live
        judge_ready = enabled and bool(os.getenv(judge.api_key_env))
        jev_ready = enabled and cfg.jev_enabled and bool(os.getenv("TYPESAFE_API_KEY"))
        return {
            "enabled": enabled,
            "judge_available": judge_ready,
            "judge_model": judge.model,
            "self_judged": judge.model == self.config.llm.model
            and judge.base_url == self.config.llm.base_url,
            "judge_reason": (
                None
                if judge_ready
                else (
                    "audit_offline_or_disabled"
                    if not enabled
                    else f"missing_key:{judge.api_key_env}"
                )
            ),
            "jev_available": jev_ready,
            "jev_reason": (
                None
                if jev_ready
                else (
                    "jev_disabled"
                    if not cfg.jev_enabled
                    else "missing_key:TYPESAFE_API_KEY"
                )
            ),
            "jev_model": "jev-1.13.0",
            "limits": {
                "concurrency": 1,
                "judge_calls": 2,
                "judge_max_output_tokens_per_call": 16384,
                "judge_timeout_seconds": 180,
                "judge_input_budget_tokens": cfg.judge_input_tokens,
                "judge_context_tokens": cfg.judge_context_tokens,
                "jev": JevAuditConfig().model_dump(
                    exclude={"base_url", "api_key_env", "enabled"}
                ),
            },
        }

    def model_comparison(self, snapshot: dict) -> dict:
        """Compare against the source run's provider, not today's generation config."""
        judge = self.judge_config()
        generation = snapshot.get(
            "generation_provider",
            {"model": self.config.llm.model, "origin": self.config.llm.base_url},
        )
        model = generation.get("model")
        origin = generation.get("origin")
        same = (
            None
            if not (model and origin)
            else model == judge.model
            and urlsplit(origin).netloc == urlsplit(judge.base_url).netloc
        )
        return {
            "self_judged": same,
            "source_generation_model": model,
            "model_comparison": (
                "unavailable"
                if same is None
                else (
                    "same_provider_and_model" if same else "different_provider_or_model"
                )
            ),
        }

    async def list(self, run_id: str, revision: int) -> dict:
        """List only attempts bound to the requested version."""
        await self.initialize()
        with self._db() as db:
            rows = db.execute(
                "SELECT audit_id FROM audits WHERE run_id=? AND revision=? ORDER BY created_at DESC",
                (run_id, revision),
            ).fetchall()
        return {
            "capabilities": self.capabilities(),
            "audits": [self._read(row["audit_id"]) for row in rows],
            "automatic": await self.get_automatic(run_id, revision),
        }

    @staticmethod
    def _automatic_record(row: sqlite3.Row) -> dict:
        """Expose the durable version reservation without provider credentials."""
        return {
            "source_run_id": row["run_id"],
            "source_report_revision": row["revision"],
            "source_report_sha256": row["source_sha"],
            "mode": row["mode"],
            "max_versions": row["max_versions"],
            "status": row["status"],
            "reason_codes": json.loads(row["reason_codes"]),
            "audit_id": row["audit_id"],
            "budget_reserved": bool(row["budget_reserved"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _update_automatic(
        self,
        run_id: str,
        revision: int,
        *,
        status: str,
        reason_codes: list[str],
        audit_id: str | None = None,
    ) -> None:
        """Finish a reservation while retaining its original identity and budget."""
        with self._db() as db:
            db.execute(
                "UPDATE automatic_audits SET status=?, reason_codes=?, "
                "audit_id=COALESCE(?, audit_id), updated_at=? "
                "WHERE run_id=? AND revision=?",
                (
                    status,
                    json.dumps(reason_codes),
                    audit_id,
                    time.time(),
                    run_id,
                    revision,
                ),
            )

    async def get_automatic(self, run_id: str, revision: int) -> dict | None:
        """Read one version's automatic decision without scheduling any work."""
        await self.initialize()
        with self._db() as db:
            row = db.execute(
                "SELECT * FROM automatic_audits WHERE run_id=? AND revision=?",
                (run_id, revision),
            ).fetchone()
        if row is None:
            return None
        result = self._automatic_record(row)
        if result["audit_id"]:
            try:
                audit = self._read(result["audit_id"])
            except KeyError:
                result.update(
                    status="failed", reason_codes=["automatic_bound_audit_missing"]
                )
            else:
                if (
                    audit["source_run_id"] != run_id
                    or audit["source_report_revision"] != revision
                    or audit["source_report_sha256"] != result["source_report_sha256"]
                ):
                    result.update(
                        status="failed",
                        reason_codes=["automatic_bound_audit_mismatch"],
                    )
                else:
                    result.update(
                        status=audit["status"],
                        reason_codes=list(audit.get("reason_codes") or []),
                    )
        return result

    async def submit_automatic(
        self, snapshot: dict, *, mode: str, max_versions: int
    ) -> dict:
        """Reserve a bounded version once before invoking the manual executor."""
        if mode not in {"off", "judge", "judge_jev"}:
            raise ValueError("automatic_mode_invalid")
        if (
            not isinstance(max_versions, int)
            or isinstance(max_versions, bool)
            or not 1 <= max_versions <= 3
        ):
            raise ValueError("automatic_max_versions_invalid")
        if mode == "off":
            return {
                "source_run_id": snapshot.get("source_run_id"),
                "source_report_revision": snapshot.get("source_report_revision"),
                "source_report_sha256": snapshot.get("source_report_sha256"),
                "mode": mode,
                "max_versions": max_versions,
                "status": "disabled",
                "reason_codes": ["auto_audit_off"],
                "audit_id": None,
                "budget_reserved": False,
                "created_at": None,
                "updated_at": None,
            }
        snapshot = json.loads(json.dumps(snapshot))
        run_id = snapshot.get("source_run_id")
        revision = snapshot.get("source_report_revision")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("automatic_run_id_invalid")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 1:
            raise ValueError("automatic_revision_invalid")
        report = snapshot.get("report") or {}
        survey = report.get("survey")
        sha = snapshot.get("source_report_sha256")
        actual_sha = (
            hashlib.sha256(survey.encode()).hexdigest()
            if isinstance(survey, str)
            else None
        )
        await self.initialize()
        reason = None
        status = "skipped"
        human_status = ((report.get("metadata") or {}).get("human_review") or {}).get(
            "status"
        )
        if not isinstance(sha, str) or sha != actual_sha:
            reason, status = "report_sha_mismatch", "failed"
        elif not survey or (report.get("quality") or {}).get("status") != "passed":
            reason = "automatic_quality_not_passed"
        elif report.get("partial"):
            reason = "automatic_report_partial"
        elif human_status == "rejected" or report.get("status") == "rejected":
            reason = "automatic_report_rejected"
        elif human_status == "cancelled" or report.get("status") in {
            "cancelled",
            "interrupted",
            "failed",
        }:
            reason = "automatic_report_cancelled"
        else:
            capabilities = self.capabilities()
            if not capabilities["judge_available"]:
                reason = capabilities["judge_reason"]
            elif mode == "judge_jev" and not capabilities["jev_available"]:
                reason = capabilities["jev_reason"]
        # BEGIN IMMEDIATE makes both the uniqueness check and run-wide count one
        # transaction, including callers in a different process or service instance.
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute(
                "SELECT * FROM automatic_audits WHERE run_id=? AND revision=?",
                (run_id, revision),
            ).fetchone()
            if existing is not None:
                if existing["source_sha"] != sha or sha != actual_sha:
                    raise ValueError("report_sha_mismatch")
                reused = True
            else:
                reserved = db.execute(
                    "SELECT COUNT(*) FROM automatic_audits "
                    "WHERE run_id=? AND budget_reserved=1",
                    (run_id,),
                ).fetchone()[0]
                if reason is None and reserved >= max_versions:
                    reason = "automatic_version_budget_exhausted"
                budget_reserved = reason is None
                status = "reserved" if budget_reserved else status
                now = time.time()
                db.execute(
                    "INSERT INTO automatic_audits VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        revision,
                        sha or "",
                        mode,
                        max_versions,
                        status,
                        json.dumps([reason] if reason else []),
                        None,
                        int(budget_reserved),
                        now,
                        now,
                    ),
                )
                reused = False
        if reused or reason is not None:
            return await self.get_automatic(run_id, revision)
        try:
            audit = await self.submit(
                snapshot, include_jev=mode == "judge_jev", new_attempt=False
            )
            self._update_automatic(
                run_id,
                revision,
                status=audit["status"],
                reason_codes=list(audit.get("reason_codes") or []),
                audit_id=audit["audit_id"],
            )
        except asyncio.CancelledError:
            self._update_automatic(
                run_id,
                revision,
                status="cancelled",
                reason_codes=["automatic_submission_cancelled"],
            )
            raise
        except Exception as exc:
            # Persist an error class rather than a provider message which may
            # contain secrets. The reserved budget never becomes a free retry.
            self._update_automatic(
                run_id,
                revision,
                status="failed",
                reason_codes=[f"automatic_submit_failed:{type(exc).__name__}"],
            )
        return await self.get_automatic(run_id, revision)

    async def get(self, run_id: str, revision: int, audit_id: str) -> dict:
        """Require both report identity and audit identity on reads."""
        await self.initialize()
        result = self._read(audit_id)
        if (
            result["source_run_id"] != run_id
            or result["source_report_revision"] != revision
        ):
            raise KeyError(audit_id)
        return result

    async def submit(
        self, snapshot: dict, *, include_jev: bool = False, new_attempt: bool = False
    ) -> dict:
        """Freeze inputs and deduplicate before scheduling any provider work."""
        await self.initialize()
        snapshot = json.loads(json.dumps(snapshot))
        survey = snapshot["report"]["survey"]
        sha = hashlib.sha256(survey.encode()).hexdigest()
        if snapshot["source_report_sha256"] != sha:
            raise ValueError("report_sha_mismatch")
        judge = self.judge_config()
        context = SurveyAuditContext.model_validate(snapshot["context"])
        snapshot["context"] = context.model_dump(mode="json")
        jev = JevAuditConfig(enabled=include_jev)
        binding = digest(
            {
                "snapshot": snapshot,
                "judge": judge.fingerprint,
                "prompt": judge_prompt_fingerprint(),
                "jev": jev.fingerprint,
                "limits": self.capabilities()["limits"],
            }
        )
        async with self._lock:
            with self._db() as db:
                pending = db.execute(
                    "SELECT audit_id FROM audits WHERE binding=? AND status IN ('queued','running') ORDER BY created_at DESC LIMIT 1",
                    (binding,),
                ).fetchone()
                if pending:
                    return {**self._read(pending["audit_id"]), "reused": True}
                if not new_attempt:
                    row = db.execute(
                        "SELECT audit_id FROM audits WHERE binding=? AND status IN ('completed','partial','failed','cancelled','interrupted') ORDER BY created_at DESC LIMIT 1",
                        (binding,),
                    ).fetchone()
                    if row:
                        return {**self._read(row["audit_id"]), "reused": True}
                capabilities = self.capabilities()
                if not capabilities["judge_available"]:
                    raise ValueError(capabilities["judge_reason"])
                if include_jev and not capabilities["jev_available"]:
                    raise ValueError(capabilities["jev_reason"])
                audit_id = f"audit-{uuid.uuid4().hex}"
                result = {
                    "schema_version": 1,
                    "protocol_version": "report-audit-v1",
                    "audit_id": audit_id,
                    "source_run_id": snapshot["source_run_id"],
                    "source_report_revision": snapshot["source_report_revision"],
                    "source_report_sha256": sha,
                    "source_snapshot_sha256": digest(snapshot),
                    "ledger_sha256": digest(snapshot["ledger"]),
                    "methodology_sha256": digest(context.review_methodology),
                    "binding": binding,
                    "judge_config_fingerprint": judge.fingerprint,
                    "judge_prompt_fingerprint": judge_prompt_fingerprint(),
                    "jev_config_fingerprint": jev.fingerprint,
                    "status": "queued",
                    "include_jev": include_jev,
                    **self.model_comparison(snapshot),
                    "judge_model": judge.model,
                    "structure": check_structure(survey, context),
                    "method_status": (
                        "pending" if context.review_methodology else "unverifiable"
                    ),
                    "method_assessments": [],
                    "method_inventory": [],
                    "judge": None,
                    "judge_usage": None,
                    "jev": {"status": "pending" if include_jev else "not_requested"},
                    "trace": {},
                    "reason_codes": [],
                    "created_at": time.time(),
                }
                atomic_json(self._path(audit_id).with_name("snapshot.json"), snapshot)
                atomic_json(self._path(audit_id), result)
                db.execute(
                    "INSERT INTO audits VALUES (?,?,?,?,?,?)",
                    (
                        audit_id,
                        snapshot["source_run_id"],
                        snapshot["source_report_revision"],
                        binding,
                        "queued",
                        result["created_at"],
                    ),
                )
            task = asyncio.create_task(self._execute(audit_id, snapshot, judge, jev))
            self._tasks[audit_id] = task
            task.add_done_callback(lambda done: self._tasks.pop(audit_id, None))
            return result

    def _save(self, result: dict) -> None:
        """Commit payload before its index status so interrupted writes fail closed."""
        atomic_json(self._path(result["audit_id"]), result)
        with self._db() as db:
            db.execute(
                "UPDATE audits SET status=? WHERE audit_id=?",
                (result["status"], result["audit_id"]),
            )
            db.execute(
                "UPDATE automatic_audits SET status=?, reason_codes=?, updated_at=? "
                "WHERE audit_id=?",
                (
                    result["status"],
                    json.dumps(result.get("reason_codes") or []),
                    time.time(),
                    result["audit_id"],
                ),
            )

    async def _execute(
        self,
        audit_id: str,
        snapshot: dict,
        judge_config: SurveyJudgeConfig,
        jev_config: JevAuditConfig,
    ) -> None:
        """Run bounded stages and close owned transports on every terminal path."""
        result = self._read(audit_id)
        client = None
        tracer = None
        hook = None
        state: dict = {}
        jev_attempted = False
        try:
            async with self._semaphore:
                result["status"] = "running"
                self._save(result)
                if self.trace_factory:
                    hook = self.trace_factory(audit_id)
                elif self.config.observability.enabled:
                    tracer = LangFuseTracer(
                        self.config.observability.langfuse_host,
                        os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                        os.getenv("LANGFUSE_SECRET_KEY", ""),
                        trace_seed=audit_id,
                    )
                    hook = RedactingTraceHook(
                        tracer, payload_mode=self.config.observability.payload_mode
                    )
                if hook:
                    original_hook = hook

                    def safe_hook(event, payload):
                        try:
                            original_hook(event, payload)
                        except Exception:
                            pass

                    hook = safe_hook
                if hook:
                    hook(
                        "survey.start",
                        {"query": snapshot["context"]["query"], "session_id": audit_id},
                    )
                    hook(
                        "subspan.start",
                        {
                            "task_id": "audit.freeze",
                            "name": "audit.freeze",
                            "input": {
                                "report_sha256": result["source_report_sha256"],
                                "binding": result["binding"],
                            },
                        },
                    )
                    hook(
                        "subspan.end",
                        {
                            "task_id": "audit.freeze",
                            "output": {
                                "snapshot_sha256": result["source_snapshot_sha256"]
                            },
                        },
                    )
                if tracer:
                    result["trace"] = {
                        "trace_id": tracer.trace_id,
                        "trace_url": tracer.trace_url,
                    }
                self._save(result)
                client = (
                    self.client_factory(judge_config, hook)
                    if self.client_factory
                    else OpenAICompatibleClient(
                        judge_config.base_url,
                        judge_config.model,
                        api_key=os.getenv(judge_config.api_key_env),
                        transport_max_attempts=1,
                        trace_hook=hook,
                    )
                )
                checkpoint = self._path(audit_id).with_name("judge-checkpoint.json")

                def reserve() -> bool:
                    # Missing usage or an in-flight attempt cannot become a free retry.
                    calls = state.get("calls", [])
                    allowed = len(calls) < 2 and all(
                        call.get("usage_complete") is True for call in calls
                    )
                    if hook:
                        hook(
                            "subspan.start",
                            {
                                "task_id": "audit.budget",
                                "name": "audit.budget",
                                "input": {"reserved_calls": len(calls), "max_calls": 2},
                            },
                        )
                        hook(
                            "subspan.end",
                            {"task_id": "audit.budget", "output": {"allowed": allowed}},
                        )
                    return allowed

                judge, usage = await SurveyBenchmarkJudge(
                    client, judge_config, trace_hook=hook
                ).evaluate(
                    case=SurveyAuditContext.model_validate(snapshot["context"]),
                    survey=snapshot["report"]["survey"],
                    ledger=snapshot["ledger"],
                    checkpoint=state,
                    save_checkpoint=lambda value: atomic_json(checkpoint, value),
                    before_call=reserve,
                    context_max_tokens=self.config.report_audit.judge_context_tokens,
                    input_budget_tokens=self.config.report_audit.judge_input_tokens,
                )
                result.update(
                    judge=judge.model_dump(mode="json"),
                    judge_usage=usage.model_dump(mode="json"),
                    method_inventory=state.get("method_inventory", []),
                    method_assessments=state.get("method_assessments", []),
                )
                result["judge_actual_models"] = sorted(
                    {
                        call["actual_model"]
                        for call in state.get("calls", [])
                        if call.get("actual_model")
                    }
                )
                result["method_status"] = (
                    "unverifiable"
                    if not snapshot["context"]["review_methodology"]
                    else ("completed" if judge.status == "completed" else "failed")
                )
                self._save(result)
                if jev_config.enabled:
                    if (
                        judge.status != "completed"
                        or not judge.inventory_snapshot
                        or not all(
                            item.source_faithful is True
                            for item in judge.claim_assessments
                        )
                    ):
                        result["jev"] = {
                            "status": "blocked",
                            "reason_codes": ["judge_source_fidelity_gate"],
                        }
                    elif not judge.inventory_snapshot.claims:
                        result["jev"] = {
                            "status": "skipped",
                            "reason_codes": ["no_scientific_claims"],
                        }
                    else:
                        auditor = (
                            self.jev_factory(audit_id, hook)
                            if self.jev_factory
                            else JevCitationAuditor(
                                repository=BenchmarkArtifactRepository(
                                    self._path(audit_id).parent / "jev"
                                ),
                                trace_hook=hook,
                                trace_id=result["trace"].get("trace_id"),
                                trace_url=result["trace"].get("trace_url"),
                            )
                        )
                        artifact = dict(snapshot["artifact"])
                        artifact.update(
                            run_id=snapshot["source_run_id"],
                            _source_artifact_sha256="sha256:"
                            + result["source_snapshot_sha256"],
                        )
                        jev_attempted = True
                        jev = await auditor.evaluate(
                            judge.inventory_snapshot,
                            snapshot["ledger"],
                            jev_config,
                            artifact=artifact,
                            reference_judge=result["judge"],
                        )
                        result["jev"] = jev.model_dump(mode="json")
                        result["jev"].update(
                            source_report_revision=result["source_report_revision"],
                            source_report_sha256=result["source_report_sha256"],
                        )
                result["status"] = (
                    "failed"
                    if judge.status == "failed"
                    else (
                        "partial"
                        if jev_config.enabled
                        and result["jev"]["status"] not in {"completed", "skipped"}
                        else "completed"
                    )
                )
                result["reason_codes"] = list(judge.reason_codes)
        except asyncio.CancelledError:
            result.update(
                status="cancelled",
                reason_codes=["audit_cancelled_usage_may_be_incomplete"],
            )
        except Exception as exc:
            # Do not persist provider error messages which may contain credentials.
            result.update(
                status="failed", reason_codes=[f"audit_error:{type(exc).__name__}"]
            )
        finally:
            if (
                not jev_attempted
                and result["jev"].get("status") == "pending"
                and result["status"] in {"failed", "cancelled"}
            ):
                result["jev"] = {
                    "status": "blocked",
                    "reason_codes": ["judge_not_completed"],
                }
            if jev_attempted and result["jev"].get("status") == "pending":
                sidecars = sorted(
                    (self._path(audit_id).parent / "jev").glob("citation-audit-*.json")
                )
                if sidecars:
                    result["jev"] = json.loads(sidecars[-1].read_text(encoding="utf-8"))
                    result["jev"].update(
                        source_report_revision=result["source_report_revision"],
                        source_report_sha256=result["source_report_sha256"],
                    )
                else:
                    result["jev"] = {
                        "status": result["status"],
                        "usage": {
                            "usage_complete": False,
                            "call_count": None,
                            "input_tokens": None,
                            "output_tokens": None,
                        },
                        "reason_codes": ["jev_interrupted_accounting_incomplete"],
                    }
            if result.get("judge_usage") is None:
                calls = state.get("calls", [])
                result["judge_usage"] = {
                    "call_count": len(calls),
                    "usage_complete": not calls,
                    "missing_usage_calls": sum(
                        not call.get("usage_complete", False) for call in calls
                    ),
                }
            result["completed_at"] = time.time()
            self._save(result)
            if hook:
                hook(
                    "subspan.start",
                    {
                        "task_id": "audit.terminal",
                        "name": "audit.terminal",
                        "input": {"status": result["status"]},
                    },
                )
                hook(
                    "subspan.end",
                    {
                        "task_id": "audit.terminal",
                        "output": {
                            "status": result["status"],
                            "judge_usage": result["judge_usage"],
                            "reason_codes": result["reason_codes"],
                        },
                    },
                )
                hook(
                    "survey.complete",
                    {
                        "judge": result["judge"],
                        "quality_status": "independent_audit",
                        "delivery_status": result["status"],
                    },
                )
            if client and hasattr(client, "close"):
                try:
                    await client.close()
                except Exception:
                    pass
            if tracer:
                await asyncio.to_thread(tracer.flush)

    async def cancel(self, run_id: str, revision: int, audit_id: str) -> dict:
        """Cancel only the bound audit, including a queued task before its first turn."""
        result = await self.get(run_id, revision, audit_id)
        if result["status"] in TERMINAL:
            return result
        task = self._tasks.get(audit_id)
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        result = self._read(audit_id)
        if result["status"] not in TERMINAL:
            result.update(
                status="cancelled", reason_codes=["audit_cancelled_before_call"]
            )
            self._save(result)
        return result

    async def shutdown(self) -> None:
        """Stop owned tasks without restarting requests during application shutdown."""
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def dry_run(self, snapshot: dict, *, include_jev: bool = False) -> dict:
        """Validate frozen inputs and report gates without scheduling paid work."""
        context = SurveyAuditContext.model_validate(snapshot["context"])
        sha = hashlib.sha256(snapshot["report"]["survey"].encode()).hexdigest()
        return {
            "dry_run": True,
            "paid_calls": 0,
            "new_surveys": 0,
            "source_run_id": snapshot["source_run_id"],
            "source_report_revision": snapshot["source_report_revision"],
            "source_report_sha256": sha,
            "snapshot_sha256": digest(snapshot),
            "ledger_sha256": digest(snapshot["ledger"]),
            "methodology_sha256": digest(context.review_methodology),
            "sha_matches": sha == snapshot["source_report_sha256"],
            "structure": check_structure(snapshot["report"]["survey"], context),
            "capabilities": self.capabilities(),
            "include_jev": include_jev,
            "jev_dependency": "requires_successful_source_faithful_judge_inventory",
            "judge_prompt_fingerprint": judge_prompt_fingerprint(),
        }
