"""Persist local interactive runs and immutable safe-boundary checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from litagent.workbench import (
    DecisionReceipt,
    HumanDecision,
    ReviewRequest,
    RunOptions,
    validate_decision,
)


class SqliteReviewRepository:
    """Coordinate one local service process with SQLite compare-and-set updates."""

    def __init__(self, root: str | Path = "artifacts/interactive") -> None:
        """Set storage paths without opening long-lived database handles."""
        self.root = Path(root)
        self.database_path = self.root / "state.sqlite3"
        self.checkpoints = self.root / "checkpoints"
        self._lock = asyncio.Lock()
        self._event_lock = threading.Lock()

    async def initialize(self) -> None:
        """Create storage and schema, then leave existing rows unchanged."""
        await asyncio.to_thread(self._initialize_sync)

    def _initialize_sync(self) -> None:
        self.checkpoints.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS interactive_runs (
                    run_id TEXT PRIMARY KEY,
                    query TEXT NOT NULL,
                    options_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    checkpoint_path TEXT,
                    checkpoint_sha256 TEXT,
                    review_json TEXT,
                    decision_json TEXT,
                    decision_sha256 TEXT,
                    receipt_json TEXT,
                    result_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS interactive_runs_status
                    ON interactive_runs(status);
                CREATE TABLE IF NOT EXISTS interactive_events (
                    run_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, seq)
                );
                CREATE TABLE IF NOT EXISTS report_versions (
                    run_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    report_sha256 TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (run_id, revision),
                    FOREIGN KEY (run_id) REFERENCES interactive_runs(run_id)
                );
                """
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(interactive_runs)"
                ).fetchall()
            }
            if "result_json" not in columns:
                connection.execute(
                    "ALTER TABLE interactive_runs ADD COLUMN result_json TEXT"
                )
            if "error" not in columns:
                connection.execute("ALTER TABLE interactive_runs ADD COLUMN error TEXT")

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    async def create_run(
        self, run_id: str, query: str, options: RunOptions
    ) -> dict[str, Any]:
        """Insert one queued run; duplicate IDs are explicit conflicts."""
        return await asyncio.to_thread(
            self._create_run_sync, run_id, query, options.model_dump_json()
        )

    def _create_run_sync(
        self, run_id: str, query: str, options_json: str
    ) -> dict[str, Any]:
        now = time.time()
        try:
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO interactive_runs "
                    "(run_id, query, options_json, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'queued', ?, ?)",
                    (run_id, query, options_json, now, now),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError("run_already_exists") from exc
        return {"run_id": run_id, "query": query, "status": "queued"}

    def checkpoint_path(self, run_id: str) -> Path:
        """Return a server-owned path for one run identifier."""
        if not run_id or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in run_id.lower()
        ):
            raise ValueError("invalid_run_id")
        return self.checkpoints / f"{run_id}.json"

    @staticmethod
    def _canonical_bytes(payload: dict[str, Any]) -> bytes:
        try:
            return json.dumps(
                payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint_not_json_serializable") from exc

    def _write_checkpoint_sync(
        self, run_id: str, payload: dict[str, Any]
    ) -> tuple[Path, str]:
        target = self.checkpoint_path(run_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        content = self._canonical_bytes(payload)
        digest = hashlib.sha256(content).hexdigest()
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent, prefix=f".{run_id}-", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return target, digest

    async def pause(
        self,
        run_id: str,
        checkpoint: dict[str, Any],
        request: ReviewRequest,
    ) -> None:
        """Write the checkpoint first, then expose its pending review atomically."""
        if request.run_id != run_id or request.status != "pending":
            raise ValueError("review_run_mismatch")
        path, digest = await asyncio.to_thread(
            self._write_checkpoint_sync, run_id, checkpoint
        )
        async with self._lock:
            await asyncio.to_thread(
                self._pause_sync,
                run_id,
                str(path),
                digest,
                request.model_dump_json(),
            )

    def _pause_sync(
        self,
        run_id: str,
        path: str,
        digest: str,
        review_json: str,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE interactive_runs SET status='awaiting_human', "
                "checkpoint_path=?, checkpoint_sha256=?, review_json=?, "
                "decision_json=NULL, decision_sha256=NULL, receipt_json=NULL, "
                "updated_at=? WHERE run_id=? AND status IN ('running','queued')",
                (path, digest, review_json, time.time(), run_id),
            )
            if cursor.rowcount != 1:
                self._raise_missing_or_conflict(connection, run_id)

    async def get_pending(self, run_id: str) -> ReviewRequest | None:
        """Return the pending request without treating storage errors as empty."""
        row = await self.get_run(run_id)
        if row["status"] != "awaiting_human":
            return None
        raw = row.get("review_json")
        if not isinstance(raw, str):
            raise ValueError("pending_review_missing")
        return ReviewRequest.model_validate_json(raw)

    async def load_checkpoint(self, run_id: str) -> dict[str, Any]:
        """Verify stored bytes before returning a checkpoint payload."""
        row = await self.get_run(run_id)
        path_value = row.get("checkpoint_path")
        expected = row.get("checkpoint_sha256")
        if not path_value or not expected:
            raise ValueError("checkpoint_unavailable")
        path = Path(path_value)
        try:
            content = await asyncio.to_thread(path.read_bytes)
        except OSError as exc:
            raise ValueError("checkpoint_unavailable") from exc
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("checkpoint_sha_mismatch")
        try:
            payload = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("checkpoint_invalid_json") from exc
        if not isinstance(payload, dict):
            raise ValueError("checkpoint_invalid_payload")
        return payload

    async def decide(self, run_id: str, decision: HumanDecision) -> DecisionReceipt:
        """Consume one review transactionally and replay identical decisions."""
        async with self._lock:
            return await asyncio.to_thread(self._decide_sync, run_id, decision)

    def _decide_sync(self, run_id: str, decision: HumanDecision) -> DecisionReceipt:
        decision_json = decision.model_dump_json()
        decision_sha = hashlib.sha256(decision_json.encode("utf-8")).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM interactive_runs WHERE run_id=?", (run_id,)
            ).fetchone()
            if row is None:
                raise KeyError(run_id)
            if row["decision_sha256"]:
                if row["decision_sha256"] != decision_sha:
                    raise ValueError("decision_conflict")
                return DecisionReceipt.model_validate_json(row["receipt_json"])
            if row["status"] != "awaiting_human" or not row["review_json"]:
                raise ValueError("review_not_pending")
            request = ReviewRequest.model_validate_json(row["review_json"])
            validate_decision(request, decision)
            status = {
                "cancel": "cancelled",
                "reject": "rejected",
            }.get(decision.action, "queued")
            receipt = DecisionReceipt(
                request_id=decision.request_id,
                run_id=run_id,
                revision=decision.revision,
                action=decision.action,
                execution_status=status,
            )
            connection.execute(
                "UPDATE interactive_runs SET status=?, decision_json=?, "
                "decision_sha256=?, receipt_json=?, updated_at=? WHERE run_id=?",
                (
                    status,
                    decision_json,
                    decision_sha,
                    receipt.model_dump_json(),
                    time.time(),
                    run_id,
                ),
            )
            return receipt

    async def claim_execution(self, run_id: str) -> bool:
        """Grant one executor ownership of a queued run."""
        async with self._lock:
            return await asyncio.to_thread(self._claim_execution_sync, run_id)

    def _claim_execution_sync(self, run_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE interactive_runs SET status='running', updated_at=? "
                "WHERE run_id=? AND status='queued'",
                (time.time(), run_id),
            )
            return cursor.rowcount == 1

    async def get_run(self, run_id: str) -> dict[str, Any]:
        """Return one durable run or raise for an unknown identity."""
        row = await asyncio.to_thread(self._get_run_sync, run_id)
        if row is None:
            raise KeyError(run_id)
        return row

    def _get_run_sync(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM interactive_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    async def list_runs(self) -> list[dict[str, Any]]:
        """List newest durable runs for the local workbench."""
        return await asyncio.to_thread(self._list_runs_sync)

    def _list_runs_sync(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM interactive_runs ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    async def recover_startup(self) -> list[str]:
        """Mark abandoned active work interrupted; never replay paid calls."""
        return await asyncio.to_thread(self._recover_startup_sync)

    def _recover_startup_sync(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT run_id FROM interactive_runs WHERE status='running'"
            ).fetchall()
            run_ids = [str(row["run_id"]) for row in rows]
            connection.execute(
                "UPDATE interactive_runs SET status='interrupted', updated_at=? "
                "WHERE status='running'",
                (time.time(),),
            )
        return run_ids

    async def mark_terminal(
        self,
        run_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """Persist one explicit terminal outcome without rewriting checkpoints."""
        if status not in {
            "completed",
            "cancelled",
            "rejected",
            "failed",
            "interrupted",
        }:
            raise ValueError("invalid_terminal_status")
        result_json = (
            json.dumps(result, ensure_ascii=False, sort_keys=True)
            if result is not None
            else None
        )
        await asyncio.to_thread(
            self._mark_terminal_sync, run_id, status, result_json, error
        )

    def _mark_terminal_sync(
        self,
        run_id: str,
        status: str,
        result_json: str | None,
        error: str | None,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE interactive_runs SET status=?, result_json=?, error=?, "
                "updated_at=? WHERE run_id=?",
                (status, result_json, error, time.time(), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(run_id)

    async def save_report_version(
        self, run_id: str, revision: int, report: dict[str, Any]
    ) -> None:
        """Persist one immutable server-owned report revision idempotently."""
        if revision < 1 or not isinstance(report.get("survey"), str):
            raise ValueError("invalid_report_version")
        report_json = self._canonical_bytes(report).decode("utf-8")
        digest = hashlib.sha256(report["survey"].encode("utf-8")).hexdigest()
        async with self._lock:
            await asyncio.to_thread(
                self._save_report_version_sync,
                run_id,
                revision,
                digest,
                report_json,
            )

    def _save_report_version_sync(
        self,
        run_id: str,
        revision: int,
        digest: str,
        report_json: str,
    ) -> None:
        with self._connect() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM interactive_runs WHERE run_id=?", (run_id,)
                ).fetchone()
                is None
            ):
                raise KeyError(run_id)
            existing = connection.execute(
                "SELECT report_sha256, report_json FROM report_versions "
                "WHERE run_id=? AND revision=?",
                (run_id, revision),
            ).fetchone()
            if existing is not None:
                if (
                    existing["report_sha256"] == digest
                    and existing["report_json"] == report_json
                ):
                    return
                raise ValueError("report_revision_conflict")
            connection.execute(
                "INSERT INTO report_versions "
                "(run_id, revision, report_sha256, report_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, revision, digest, report_json, time.time()),
            )

    async def get_report_version(self, run_id: str, revision: int) -> dict[str, Any]:
        """Return one exact report revision or raise when it does not exist."""
        row = await asyncio.to_thread(self._get_report_version_sync, run_id, revision)
        if row is None:
            raise KeyError((run_id, revision))
        return json.loads(row["report_json"])

    def _get_report_version_sync(
        self, run_id: str, revision: int
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT report_json FROM report_versions "
                "WHERE run_id=? AND revision=?",
                (run_id, revision),
            ).fetchone()
        return dict(row) if row is not None else None

    async def list_report_versions(self, run_id: str) -> list[dict[str, Any]]:
        """List immutable report revisions newest-first with compact metadata."""
        await self.get_run(run_id)
        return await asyncio.to_thread(self._list_report_versions_sync, run_id)

    def _list_report_versions_sync(self, run_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT revision, report_sha256, report_json, created_at "
                "FROM report_versions WHERE run_id=? ORDER BY revision DESC",
                (run_id,),
            ).fetchall()
        versions = []
        for row in rows:
            report = json.loads(row["report_json"])
            metadata = report.get("metadata") or {}
            versions.append(
                {
                    "revision": int(row["revision"]),
                    "report_sha256": row["report_sha256"],
                    "report_language": metadata.get("report_language", "en"),
                    "quality": report.get("quality"),
                    "delivery": report.get("delivery"),
                    "human_review": metadata.get("human_review"),
                    "created_at": float(row["created_at"]),
                }
            )
        return versions

    def event_hook(self, run_id: str):
        """Return a synchronous trace sink backed by ordered SQLite rows."""

        def hook(event: str, payload: dict[str, Any]) -> None:
            try:
                payload_json = json.dumps(
                    payload, ensure_ascii=False, sort_keys=True, default=str
                )
                with self._event_lock, self._connect() as connection:
                    row = connection.execute(
                        "SELECT COALESCE(MAX(seq), 0) + 1 AS seq "
                        "FROM interactive_events WHERE run_id=?",
                        (run_id,),
                    ).fetchone()
                    connection.execute(
                        "INSERT INTO interactive_events "
                        "(run_id, seq, event, payload_json, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (run_id, int(row["seq"]), event, payload_json, time.time()),
                    )
            except Exception:
                # Observability must never change the product result.
                return

        return hook

    async def list_events(self, run_id: str, *, after: int = 0) -> list[dict[str, Any]]:
        """Return ordered events after a stable per-run sequence."""
        return await asyncio.to_thread(self._list_events_sync, run_id, after)

    def _list_events_sync(self, run_id: str, after: int) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT seq, event, payload_json, created_at FROM interactive_events "
                "WHERE run_id=? AND seq>? ORDER BY seq",
                (run_id, after),
            ).fetchall()
        return [
            {
                "seq": int(row["seq"]),
                "event": row["event"],
                "data": json.loads(row["payload_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    @staticmethod
    def _raise_missing_or_conflict(connection: sqlite3.Connection, run_id: str) -> None:
        row = connection.execute(
            "SELECT status FROM interactive_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if row is None:
            raise KeyError(run_id)
        raise ValueError(f"invalid_run_state:{row['status']}")
