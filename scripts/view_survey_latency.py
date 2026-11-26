"""Serve a Survey benchmark and its exact archived workbench runs on loopback."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from starlette.middleware.trustedhost import TrustedHostMiddleware

from litagent.benchmark.survey_latency import (
    CONTRACT,
    SUMMARY_CHECKPOINT_SCOPE,
    archive_failure_reasons,
    render_benchmark_page,
    run_slots,
    summary_checkpoint_duration_ms,
)
from litagent.citation_formats import format_capabilities
from litagent.demo import project_flow_demo_view
from litagent.eval.results import BENCHMARK_FAITHFULNESS_POLICY
from litagent.observability.recorder import summarize_archive
from litagent.workbench import CitationStyle

STATIC = Path(__file__).resolve().parents[1] / "src/litagent/static/flow-demo"
REVALIDATION_CONTRACT = "survey-benchmark-matrix-delivery-revalidation-v1"


def sha256(raw: bytes) -> str:
    """Hash the stored bytes rather than a reserialized artifact."""
    return hashlib.sha256(raw).hexdigest()


def read_json_object(path: Path) -> dict:
    """Load one local JSON object."""
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


@dataclass(frozen=True)
class Archive:
    """The private path and evidence needed to read one archive."""

    path: Path
    run_id: str
    snapshot_sha256: str
    report_sha256: str | None

    def read(self) -> dict:
        """Read the exact bytes bound to this run and manuscript."""
        raw = self.path.read_bytes()
        if sha256(raw) != self.snapshot_sha256:
            raise ValueError("run snapshot SHA-256 mismatch")
        value = json.loads(raw)
        if not isinstance(value, dict) or value.get("run_id") != self.run_id:
            raise ValueError("run identity mismatch")
        report = value.get("report")
        if report is not None and not isinstance(report, dict):
            raise ValueError("invalid archived report")
        survey = (report or {}).get("survey")
        if (
            sha256(survey.encode()) if isinstance(survey, str) else None
        ) != self.report_sha256:
            raise ValueError("report SHA-256 mismatch")
        return value


@dataclass(frozen=True)
class Campaign:
    """One validated matrix and its exact local workbench entries."""

    digest: str
    status: str
    manifest: dict
    rows: list[dict]
    archives: dict[str, Archive]
    summaries: list[dict]


def read_campaign(root: Path) -> Campaign:
    """Verify the began campaign, matrix slots and every linked run."""
    root = root.resolve()
    manifest = read_json_object(root / "manifest.json")
    started = read_json_object(root / "started.json")
    approval = manifest.pop("approval_sha256", None)
    if (
        manifest.get("contract") != CONTRACT
        or approval != sha256(json.dumps(manifest, sort_keys=True).encode())
        or started.get("approval_sha256") != approval
        or manifest.get("faithfulness_policy") != BENCHMARK_FAITHFULNESS_POLICY
        or manifest.get("threshold") != 0.95
    ):
        raise ValueError("campaign manifest does not match its start receipt")
    data = (root / "benchmark.json").read_bytes()
    benchmark = json.loads(data)
    if not isinstance(benchmark, dict):
        raise ValueError("expected a terminal Survey benchmark matrix")
    slots = run_slots()
    rows = benchmark.get("rows")
    if (
        benchmark.get("contract") != CONTRACT
        or benchmark.get("status") not in {"completed", "partial"}
        or manifest.get("runs") != [asdict(slot) for slot in slots]
        or not isinstance(rows, list)
        or len(rows) != len(slots)
    ):
        raise ValueError("expected a terminal Survey benchmark matrix")
    archives, summaries = {}, []
    for row, slot in zip(rows, slots, strict=True):
        if not isinstance(row, dict) or any(
            row.get(key) != value for key, value in asdict(slot).items()
        ):
            raise ValueError("benchmark slot does not match the campaign manifest")
        if not row.get("snapshot"):
            continue
        run_id = row.get("run_id")
        relative = Path("snapshots") / slot.key / "run.json"
        path = (root / relative).resolve()
        if (
            not isinstance(run_id, str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_id)
            or run_id in archives
            or row["snapshot"] != relative.as_posix()
            or not path.is_relative_to(root)
        ):
            raise ValueError("benchmark archive path or run identity is invalid")
        archive = Archive(
            path, run_id, row["snapshot_sha256"], row.get("report_sha256")
        )
        artifact = archive.read()
        if row.get("delivery") == "blocked":
            row["reasons"] = list(
                dict.fromkeys(
                    [*row.get("reasons", []), *archive_failure_reasons(artifact)]
                )
            )
        archives[run_id] = archive
        summaries.append({**summarize_archive(artifact), "source": "archive"})
    return Campaign(
        sha256(data), benchmark["status"], manifest, rows, archives, summaries
    )


def retry_rows(baseline: Campaign, retries: Campaign) -> dict[str, dict]:
    """Index archived retry attempts by the matching original run ID."""
    slots = run_slots()
    selected = retries.manifest.get("retry_replay_slots")
    origins = retries.manifest.get("retry_of")
    timing_scope = retries.manifest.get("timing_scope")
    if (
        not isinstance(selected, list)
        or not selected
        or not all(isinstance(key, str) for key in selected)
        or len(set(selected)) != len(selected)
        or not isinstance(origins, dict)
        or set(selected) != set(origins)
        or not set(selected) <= {slot.key for slot in slots if slot.source == "replay"}
        or timing_scope not in (None, SUMMARY_CHECKPOINT_SCOPE)
    ):
        raise ValueError("invalid retry campaign manifest lineage")

    archived_deliveries = {
        summary["run_id"]: (summary.get("delivery") or {}).get("status")
        for summary in baseline.summaries
    }
    linked = {}
    for slot, original, row in zip(slots, baseline.rows, retries.rows, strict=True):
        origin = {key: original.get(key) for key in ("run_id", "snapshot_sha256")}
        if slot.key not in origins:
            if any(row.get(key) != value for key, value in origin.items()):
                raise ValueError("retry campaign changed a carried baseline run")
            continue
        if (
            origins[slot.key] != origin
            or original.get("delivery") != "blocked"
            or archived_deliveries.get(origin["run_id"]) != "blocked"
        ):
            raise ValueError(
                "retry campaign origin does not match the blocked baseline run"
            )
        if not row.get("snapshot"):
            continue
        run_id = row["run_id"]
        fingerprint = original.get("external_inputs")
        if (
            row.get("retry_of") != origin
            or row.get("timing_scope") != timing_scope
            or run_id in baseline.archives
            or not re.fullmatch(re.escape(slot.key) + r"-retry-[0-9a-f]{12}", run_id)
            or not fingerprint
            or row.get("external_inputs") != fingerprint
        ):
            raise ValueError(
                "retry run identity or external inputs do not match the baseline"
            )

        artifact = retries.archives[run_id].read()
        events = artifact.get("events") or []
        checkpoint_events = [
            event.get("data")
            for event in events
            if event.get("event") == "benchmark.summary_checkpoint"
        ]
        if timing_scope == SUMMARY_CHECKPOINT_SCOPE:
            witness = checkpoint_events[0] if len(checkpoint_events) == 1 else None
            reused = (
                witness.get("reused_task_ids") if isinstance(witness, dict) else None
            )
            duration = summary_checkpoint_duration_ms(events)
            if (
                not isinstance(witness, dict)
                or witness.get("source_run_id") != origin["run_id"]
                or witness.get("source_snapshot_sha256") != origin["snapshot_sha256"]
                or not isinstance(reused, list)
                or any(not isinstance(task, str) or not task for task in reused)
                or len(reused) != len(set(reused))
                or "extract" not in reused
                or "adversarial_review" in reused
                or row.get("downstream_ms") != duration
                or (row.get("delivery") == "ready" and duration is None)
            ):
                raise ValueError(
                    "retry summary checkpoint provenance or timing invalid"
                )
        elif checkpoint_events:
            raise ValueError("retry summary checkpoint timing scope missing")
        delivery = ((artifact.get("report") or {}).get("delivery") or {}).get("status")
        if row.get("delivery") != delivery:
            raise ValueError("retry delivery does not match the archived report")
        if delivery == "ready" and [
            (event.get("data") or {}).get("fingerprint")
            for event in events
            if event.get("event") == "external_inputs_ready"
        ] != [fingerprint]:
            raise ValueError("ready retry did not record the frozen external inputs")
        linked[origin["run_id"]] = {**row, "source": "retry"}
    return linked


def read_revalidation(path: Path, campaign: Campaign) -> dict:
    """Bind the offline acceptance results to the exact benchmark and run SHAs."""
    value = read_json_object(path)
    rows = value.get("rows")
    recorded_ready = sum(row.get("delivery") == "ready" for row in campaign.rows)
    if (
        value.get("contract") != REVALIDATION_CONTRACT
        or value.get("mode") != "offline_revalidation_of_real_surveys"
        or (value.get("dataset") or {}).get("sha256") != campaign.digest
        or value.get("policy") != BENCHMARK_FAITHFULNESS_POLICY
        or value.get("threshold") != 0.95
        or value.get("additional_model_requests") != 0
        or value.get("recorded_ready") != recorded_ready
        or value.get("total") != len(campaign.rows)
        or not isinstance(rows, list)
        or len(rows) != len(campaign.rows)
    ):
        raise ValueError("offline revalidation does not match the benchmark")
    for recorded, revalidated in zip(campaign.rows, rows, strict=True):
        if (
            not isinstance(revalidated, dict)
            or any(
                recorded.get(key) != revalidated.get(key)
                for key in (
                    "run_id",
                    "case_id",
                    "profile_id",
                    "repetition",
                    "report_sha256",
                )
            )
            or (
                recorded.get("snapshot_sha256")
                != (revalidated.get("run") or {}).get("sha256")
                or recorded.get("delivery") != revalidated.get("recorded_status")
                or (recorded.get("evaluation") or {})
                .get("faithfulness", {})
                .get("score")
                != revalidated.get("faithfulness")
                or revalidated.get("revalidated_status") not in {"ready", "blocked"}
            )
        ):
            raise ValueError("offline revalidation row has a different run or result")
        if revalidated["revalidated_status"] == "ready":
            score = revalidated.get("faithfulness")
            if (
                type(score) not in (int, float)
                or not math.isfinite(score)
                or not value["threshold"] <= score <= 1
                or (revalidated.get("gates") or {}).get("evaluation") != "passed"
            ):
                raise ValueError(
                    "offline revalidation row did not meet the acceptance gate"
                )
    if value.get("revalidated_ready") != sum(
        row["revalidated_status"] == "ready" for row in rows
    ):
        raise ValueError("offline revalidation count does not match its rows")
    return value


def viewer_app(
    campaign_path: Path,
    diagnostics_path: Path | None = None,
    revalidation_path: Path | None = None,
    *,
    retries_path: Path | Sequence[Path] | None = None,
) -> FastAPI:
    """Build a provider-free copy of the archive view with only selected runs."""
    campaign = read_campaign(campaign_path)
    rows, archives, summaries = (
        list(campaign.rows),
        dict(campaign.archives),
        list(campaign.summaries),
    )
    retry_campaigns = (
        (retries_path,) if isinstance(retries_path, Path) else retries_path
    )
    attempts: dict[str, list[dict]] = {}
    for path in retry_campaigns or ():
        retries = read_campaign(path)
        linked = retry_rows(campaign, retries)
        added_ids = {row["run_id"] for row in linked.values()}
        if archives.keys() & added_ids:
            raise ValueError("benchmark retry run identities overlap")
        archives.update({run_id: retries.archives[run_id] for run_id in added_ids})
        summaries.extend(row for row in retries.summaries if row["run_id"] in added_ids)
        for run_id, row in linked.items():
            attempts.setdefault(run_id, []).append(row)
    if attempts:
        rows = [
            entry
            for row in campaign.rows
            for entry in (row, *attempts.get(row.get("run_id"), ()))
        ]
    if diagnostics_path is not None:
        diagnostics = read_campaign(diagnostics_path)
        if archives.keys() & diagnostics.archives.keys():
            raise ValueError("benchmark and diagnostic run identities overlap")
        rows.extend(
            {**row, "source": "workbench-diagnostic"}
            for row in diagnostics.rows
            if row.get("run_id") in diagnostics.archives
        )
        archives.update(diagnostics.archives)
        summaries.extend(diagnostics.summaries)
    revalidated = (
        read_revalidation(revalidation_path, campaign) if revalidation_path else None
    )
    page = render_benchmark_page(rows, status=campaign.status, revalidated=revalidated)
    capabilities = format_capabilities()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.middleware("http")
    async def read_only(request: Request, call_next):
        if request.method not in {"GET", "HEAD"}:
            return Response(status_code=405, headers={"Allow": "GET, HEAD"})
        return await call_next(request)

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    @app.api_route("/presentation", methods=["GET", "HEAD"], include_in_schema=False)
    def overview_link():
        return RedirectResponse("/benchmark")

    @app.api_route("/benchmark", methods=["GET", "HEAD"], include_in_schema=False)
    def overview():
        return HTMLResponse(page)

    @app.api_route("/flow-demo", methods=["GET", "HEAD"], include_in_schema=False)
    def workbench(view: str | None = None):
        if view == "benchmark":
            return RedirectResponse("/benchmark")
        return FileResponse(STATIC / "index.html")

    @app.api_route("/workbench/context", methods=["GET", "HEAD"])
    def context():
        return {
            "schema_version": 1,
            "mode": "offline",
            "live_enabled": False,
            "benchmark_overview": "/benchmark",
            "format_capabilities": capabilities,
        }

    @app.api_route("/workbench/runs", methods=["GET", "HEAD"])
    @app.api_route("/workbench/benchmarks", methods=["GET", "HEAD"])
    def empty_workbench_list():
        return []

    @app.api_route("/flow-demo/runs", methods=["GET", "HEAD"])
    def history():
        return summaries

    @app.api_route("/flow-demo/runs/{run_id}/view", methods=["GET", "HEAD"])
    def workbench_view(run_id: str, citation_style: CitationStyle | None = None):
        archive = archives.get(run_id)
        if archive is None:
            raise HTTPException(status_code=404, detail="benchmark_run_not_found")
        try:
            artifact = archive.read()
        except (OSError, ValueError) as exc:
            raise HTTPException(
                status_code=409, detail="benchmark_archive_changed"
            ) from exc
        style = citation_style or artifact.get("citation_style", "numbered")
        view = project_flow_demo_view(artifact, source="archive", citation_style=style)
        view.update(
            format_capabilities=capabilities,
            citation_style=style,
            capabilities={"can_download_artifact": False, "can_export": False},
        )
        return view

    @app.api_route("/static/flow-demo/{filename}", methods=["GET", "HEAD"])
    def workbench_assets(filename: str):
        if not re.fullmatch(r"[a-z0-9-]+\.(js|css)", filename):
            raise HTTPException(status_code=404)
        path = STATIC / filename
        if not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path)

    return app


def main() -> None:
    """Open the verified local benchmark on the default viewer port."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--revalidation", type=Path)
    parser.add_argument(
        "--retries",
        type=Path,
        action="append",
        help="Add attempts from a completed or partial retry campaign (repeatable)",
    )
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        app = viewer_app(
            args.campaign,
            args.diagnostics,
            args.revalidation,
            retries_path=args.retries,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    uvicorn.run(app, host="127.0.0.1", port=args.port, reload=False)


if __name__ == "__main__":
    main()
