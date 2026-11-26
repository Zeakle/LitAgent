"""Run the bounded three-topic Survey latency experiment from the project root."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import socket
import subprocess
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from uuid import uuid4

import httpx

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.datasets import (
    load_survey_dataset,
    load_survey_judge_config,
    load_survey_profiles,
)
from litagent.benchmark.survey_latency import (
    CONTRACT,
    PROFILES,
    llm_outcomes,
    publish,
    run_slots,
    save_observation,
    stage_spans,
    write_bytes,
    write_json,
)
from litagent.benchmark.survey_models import ExternalInputSnapshot
from litagent.benchmark.survey_runner import (
    SurveyBenchmarkRunner,
    _capture_external_inputs_from_artifact,
)
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.eval.results import BENCHMARK_FAITHFULNESS_POLICY
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.run_limits import configured_run_limits
from litagent.runner import ExternalInputReplay, LitAgent, RunPolicy
from litagent.workbench import RunOptions, default_verification_options

DATASET = Path("benchmarks/survey/dataset.yaml")
PROFILES_PATH = Path("benchmarks/survey/profiles.yaml")
TIMEOUT_SECONDS = 4 * 60 * 60


def sha(value) -> str:
    """Hash a canonical manifest value."""
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def source_identity() -> str:
    """Bind an approval to executable bytes, including uncommitted changes."""
    paths = sorted(Path("src/litagent").rglob("*.py"))
    paths.extend(sorted(Path("src/litagent/static/flow-demo").glob("*.js")))
    paths.extend(
        [
            Path(__file__),
            Path(__file__).with_name("survey_latency_browser.js"),
            Path("src/litagent/static/flow-demo/index.html"),
        ]
    )
    return sha(
        [(str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths]
    )


def cache_identity(root: Path) -> str:
    """Freeze the local PDFs that will seed every timed run."""
    entries = []
    for path in sorted(root.rglob("*.pdf")):
        if path.is_file():
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            entries.append((str(path.relative_to(root)), digest))
    return sha(entries)


def isolated_cache(source: Path, root: Path) -> None:
    """Clone a raw cache whose downloads replace files atomically."""

    def link_or_copy(src, dst):
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)

    if source.is_dir():
        shutil.copytree(source, root, copy_function=link_or_copy)
    else:
        root.mkdir(parents=True)


def configure_cache(config, root):
    """Redirect acquisition and audit storage to this run's private cache."""
    value = config.model_copy(deep=True)
    value.rag.raw_root = str(root / "raw")
    value.rag.parsed_root = str(root / "parsed")
    value.rag.quarantine_root = str(root / "quarantine")
    return value


def options_for(config) -> RunOptions:
    """Match the explicit options selected in the ordinary workbench form."""
    verification = default_verification_options(config).model_dump()
    verification.update(judge=False, jev=False)
    return RunOptions(
        report_language="en",
        review_structure="academic_v2",
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v7",
        capacity_policy="adaptive-capacity-v1",
        verification=verification,
    )


@dataclass
class ResumedCampaign:
    """Hold authenticated prior rows and replay inputs for the same matrix."""

    manifest: dict
    identity: str
    observations: dict
    snapshots: dict
    files: dict
    retry_slots: tuple[str, ...] = ()
    retry_of: dict = field(default_factory=dict)

    def materialize(self, destination: Path) -> None:
        """Copy frozen bytes so every exported table link stays usable."""
        for relative, content in self.files.items():
            write_bytes(destination / relative, content)

    def select_failed_replays(self, keys) -> ResumedCampaign:
        """Reuse baseline inputs and schedule only explicitly selected failed replays."""
        selected = set(keys)
        if len(selected) != len(keys) or any(
            key not in self.observations
            or self.observations[key]["source"] != "replay"
            or self.observations[key]["delivery"] != "blocked"
            or self.observations[key]["usage"]["usage_uncertain"]
            for key in selected
        ):
            raise ValueError("survey_retry_replay_slot_invalid")
        return replace(
            self,
            observations={
                key: row
                for key, row in self.observations.items()
                if key not in selected
            },
            files={
                path: content
                for path, content in self.files.items()
                if not any(path.startswith(f"snapshots/{key}/") for key in selected)
            },
            retry_slots=tuple(sorted(selected)),
            retry_of={
                key: {
                    field_name: self.observations[key][field_name]
                    for field_name in ("run_id", "snapshot_sha256")
                }
                for key in selected
            },
        )


def reconcile_ark_usage(row, artifact, receipt: bytes) -> int:
    """Match an Ark usage response to the archive's one unaccounted request."""
    calls = [
        event.get("data") or {}
        for event in artifact.get("events", [])
        if event.get("event") in {"llm.complete", "llm.failed"}
    ]
    missing = [call for call in calls if call.get("usage_available") is False]
    usage = row["usage"]
    reported = (artifact.get("usage") or {}).get("total_tokens")
    if (
        len(missing) != 1
        or usage.get("missing_usage_calls") != 1
        or usage.get("paid_requests") != len(calls)
        or type(reported) is not int
        or reported < 0
        or reported != usage.get("total_tokens")
    ):
        raise ValueError("survey_resume_usage_not_reconcilable")
    request_id = missing[0].get("provider_request_id")
    response = json.loads(receipt)
    if not isinstance(response, dict):
        raise ValueError("survey_resume_usage_receipt_invalid")
    metadata = response.get("ResponseMetadata")
    result = response.get("Result")
    records = result.get("Records") if isinstance(result, dict) else None
    if (
        not request_id
        or not isinstance(metadata, dict)
        or metadata.get("Action") != "GetInferenceRequestUsage"
        or metadata.get("Service") != "ark"
        or metadata.get("Error")
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("survey_resume_usage_receipt_invalid")
    for record in records:
        if (
            not isinstance(record, dict)
            or record.get("RequestID") != request_id
            or any(
                type(record.get(key)) is not int or record[key] < 0
                for key in ("InputTokens", "OutputTokens")
            )
        ):
            raise ValueError("survey_resume_usage_receipt_invalid")
    return reported + sum(r["InputTokens"] + r["OutputTokens"] for r in records)


def read_resumed_campaign(
    root: Path,
    slots,
    dataset,
    usage_receipt: bytes | None = None,
    *,
    carry_unsettled: bool = False,
) -> ResumedCampaign:
    """Authenticate a completed segment and recover its immutable replay inputs."""
    manifest_bytes = (root / "manifest.json").read_bytes()
    benchmark_bytes = (root / "benchmark.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    approval = manifest.get("approval_sha256")
    started = json.loads((root / "started.json").read_bytes())
    if (
        approval
        != sha(
            {key: value for key, value in manifest.items() if key != "approval_sha256"}
        )
        or started.get("approval_sha256") != approval
    ):
        raise ValueError("survey_resume_manifest_mismatch")
    report = json.loads(benchmark_bytes)
    observations, snapshots, files = {}, {}, {}
    rows = report.get("rows")
    if (
        report.get("status") not in {"completed", "partial"}
        or not isinstance(rows, list)
        or len(rows) != len(slots)
    ):
        raise ValueError("survey_resume_requires_terminal_campaign")
    for slot, row in zip(slots, rows, strict=True):
        if any(row.get(key) != value for key, value in asdict(slot).items()):
            raise ValueError("survey_resume_slot_mismatch")
        if row.get("status") == "未运行":
            continue
        relative = f"snapshots/{slot.key}/run.json"
        if row.get("snapshot") != relative:
            raise ValueError("survey_resume_snapshot_missing")
        raw = (root / relative).read_bytes()
        artifact = json.loads(raw)
        report_data = artifact.get("report") or {}
        source_sha = (
            hashlib.sha256(report_data["survey"].encode()).hexdigest()
            if isinstance(report_data.get("survey"), str)
            else None
        )
        events = artifact.get("events") or []
        usage = row.get("usage") or {}
        accounting = next(
            (
                e.get("data") or {}
                for e in reversed(events)
                if e.get("event") == "benchmark.accounting"
            ),
            artifact.get("budget_state") or {},
        )
        if (
            not row.get("run_id")
            or hashlib.sha256(raw).hexdigest() != row.get("snapshot_sha256")
            or artifact.get("run_id") != row["run_id"]
            or row.get("report_sha256") != source_sha
            or row.get("config_fingerprint") != artifact.get("config_fingerprint")
            or type(usage.get("usage_uncertain")) is not bool
            or usage["usage_uncertain"] is not accounting.get("usage_uncertain")
            or usage.get("paid_requests") != accounting.get("requests")
            or (
                usage["usage_uncertain"] is False
                and usage.get("billed_tokens") != accounting.get("known_tokens")
            )
        ):
            raise ValueError("survey_resume_snapshot_mismatch")
        files[relative] = raw
        for version in row.get("versions", []):
            path = Path(version["path"])
            if path.parent != Path(relative).parent:
                raise ValueError("survey_resume_version_path_mismatch")
            files[str(path)] = (root / path).read_bytes()
        observation = {
            **row,
            "usage": {**usage, **llm_outcomes(events)},
        }
        observation["stages_ms"], missing_stages = stage_spans(events)
        observation["missing"] = {**row.get("missing", {}), "stages": missing_stages}
        if usage["usage_uncertain"]:
            receipt_path = f"usage-receipts/{slot.key}.json"
            previous = row.get("usage_receipt")
            receipt = None
            if previous:
                receipt = (root / receipt_path).read_bytes()
            elif usage_receipt is not None:
                receipt, usage_receipt = usage_receipt, None
            elif carry_unsettled:
                run_tokens = manifest.get("per_run_tokens")
                run_requests = manifest.get("per_run_requests")
                if (
                    not all(
                        type(v) is int and v > 0 for v in (run_tokens, run_requests)
                    )
                    or type(accounting.get("known_tokens")) is not int
                    or not 0 <= accounting["known_tokens"] <= run_tokens
                    or type(accounting.get("requests")) is not int
                    or not 0 < accounting["requests"] <= run_requests
                ):
                    raise ValueError("survey_resume_unsettled_run_limit_invalid")
                observation["usage"].update(
                    held_run_token_cap=run_tokens, held_run_request_cap=run_requests
                )
            else:
                raise ValueError("survey_resume_usage_receipt_required")
            if receipt is not None:
                link = {
                    "path": receipt_path,
                    "sha256": hashlib.sha256(receipt).hexdigest(),
                }
                if previous and previous != link:
                    raise ValueError("survey_resume_usage_receipt_mismatch")
                observation["usage"].pop("held_run_token_cap", None)
                observation["usage"].pop("held_run_request_cap", None)
                observation["usage"]["reconciled_billed_tokens"] = reconcile_ark_usage(
                    row, artifact, receipt
                )
                observation["usage_receipt"] = link
                files[receipt_path] = receipt
        observations[slot.key] = observation
        if slot.source == "workbench":
            input_file = f"inputs/{slot.case_id}.json"
            if row.get("input_snapshot_path") != input_file:
                raise ValueError("survey_resume_external_input_missing")
            raw_input = (root / input_file).read_bytes()
            snapshot = ExternalInputSnapshot.model_validate_json(raw_input)
            captured = _capture_external_inputs_from_artifact(
                case=dataset.case_by_id(slot.case_id), repetition=1, artifact=artifact
            )
            if snapshot.fingerprint != captured.fingerprint:
                raise ValueError("survey_resume_external_input_mismatch")
            snapshots[slot.case_id] = snapshot
            files[input_file] = raw_input
        if (
            slot.case_id not in snapshots
            or row.get("external_inputs") != snapshots[slot.case_id].fingerprint
        ):
            raise ValueError("survey_resume_external_input_mismatch")
    if usage_receipt is not None:
        raise ValueError("survey_resume_unused_usage_receipt")
    if not observations:
        raise ValueError("survey_resume_requires_recorded_observation")
    identity = sha(
        [hashlib.sha256(data).hexdigest() for data in (manifest_bytes, benchmark_bytes)]
        + [
            (name, hashlib.sha256(data).hexdigest())
            for name, data in sorted(files.items())
        ]
    )
    return ResumedCampaign(manifest, identity, observations, snapshots, files)


def remaining_approved_budget(resumed: ResumedCampaign) -> dict[str, int]:
    """Deduct settled spend and full unresolved-run limits from the source approval."""
    earlier = set(resumed.manifest.get("resumed_slots", []))
    tokens, requests = 0, 0
    for key, row in resumed.observations.items():
        if key in earlier:
            continue
        usage = row["usage"]
        spent = usage.get(
            "held_run_token_cap",
            usage.get("reconciled_billed_tokens", usage.get("billed_tokens")),
        )
        submitted = usage.get("held_run_request_cap", usage.get("paid_requests"))
        if not all(type(v) is int and v >= 0 for v in (spent, submitted)):
            raise ValueError("survey_resume_prior_budget_unknown")
        tokens += spent
        requests += submitted
    return {
        "prior_tokens_spent_or_reserved": tokens,
        "prior_requests_sent_or_reserved": requests,
        "remaining_tokens": resumed.manifest["max_tokens"] - tokens,
        "remaining_requests": resumed.manifest["max_requests"] - requests,
    }


def make_manifest(
    config,
    dataset,
    profiles,
    slots,
    token_cap,
    request_cap,
    max_runs=None,
    resumed=None,
):
    """Produce the maximum permitted exposure before any request can be sent."""
    limits = configured_run_limits(config)
    if not limits or min(limits["max_tokens"], limits["max_requests"]) <= 0:
        raise ValueError("configured run limits are required")
    max_tokens, max_requests = limits["max_tokens"], limits["max_requests"]
    available = len(slots) - (len(resumed.observations) if resumed else 0)
    run_limit = available if max_runs is None else max_runs
    if not 1 <= run_limit <= available:
        raise ValueError("max-runs must be within the planned matrix")
    carried_budget = None
    if (
        resumed
        and resumed.retry_slots
        and (
            token_cap is None
            or request_cap is None
            or run_limit != len(resumed.retry_slots)
        )
    ):
        raise ValueError("survey_retry_requires_explicit_full_scope_and_budget")
    if (
        resumed
        and not resumed.retry_slots
        and any(
            "held_run_token_cap" in row["usage"]
            for row in resumed.observations.values()
        )
    ):
        carried_budget = remaining_approved_budget(resumed)
        token_hold = carried_budget["remaining_tokens"]
        request_hold = carried_budget["remaining_requests"]
        token_cap = token_hold if token_cap is None else min(token_cap, token_hold)
        request_cap = (
            request_hold if request_cap is None else min(request_cap, request_hold)
        )
    if token_cap is not None and token_cap < max_tokens:
        raise ValueError("total token cap must admit at least one complete run")
    if request_cap is not None and request_cap < max_requests:
        raise ValueError("total request cap must admit at least one complete run")
    payload = {
        "contract": CONTRACT,
        "faithfulness_policy": BENCHMARK_FAITHFULNESS_POLICY,
        "threshold": 0.95,
        "config": build_config_summary(config, "strict_auto_v2"),
        "run_options": options_for(config).model_dump(mode="json"),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_identity": source_identity(),
        "dataset_fingerprint": dataset.fingerprint,
        "profiles_fingerprints": {p.profile_id: p.fingerprint for p in profiles},
        "corpus_manifests": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in dataset.manifest_paths
        },
        "raw_pdf_cache_sha256": cache_identity(Path(config.rag.raw_root)),
        "runs": [asdict(slot) for slot in slots],
        "max_runs": run_limit,
        "per_run_tokens": max_tokens,
        "per_run_requests": max_requests,
        "max_tokens": min(token_cap or max_tokens * run_limit, max_tokens * run_limit),
        "max_requests": min(
            request_cap or max_requests * run_limit, max_requests * run_limit
        ),
        "judge_requests": 0,
    }
    if resumed:
        for key in (
            "contract",
            "faithfulness_policy",
            "threshold",
            "config",
            "run_options",
            "dataset_fingerprint",
            "profiles_fingerprints",
            "corpus_manifests",
            "raw_pdf_cache_sha256",
            "runs",
            "per_run_tokens",
            "per_run_requests",
        ):
            if resumed.manifest.get(key) != payload[key]:
                raise ValueError(f"survey_resume_config_mismatch:{key}")
        payload["resumed_campaign"] = resumed.identity
        payload["resumed_slots"] = sorted(resumed.observations)
        if resumed.retry_slots:
            payload["retry_replay_slots"] = list(resumed.retry_slots)
            payload["retry_of"] = resumed.retry_of
        if carried_budget is not None:
            payload["carried_budget"] = carried_budget
    return {**payload, "approval_sha256": sha(payload)}


@asynccontextmanager
async def workbench(config, root, archive):
    """Serve the ordinary workbench on loopback with campaign-owned storage."""
    import uvicorn

    from litagent import api
    from litagent.interaction_store import SqliteReviewRepository

    api.app.state.latency_benchmark_config = config
    api.app.state.latency_benchmark_root = root
    api.app.state.flow_repository = archive
    api.app.state.interactive_repository = SqliteReviewRepository(root / "interactive")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(api.app, log_level="warning", access_log=False)
        )
        task = asyncio.create_task(server.serve(sockets=[listener]))
        try:
            async with asyncio.timeout(15):
                while not server.started:
                    if task.done():
                        await task
                        raise RuntimeError("benchmark_workbench_start_failed")
                    await asyncio.sleep(0.05)
            yield f"http://127.0.0.1:{port}", api.app.state
        finally:
            server.should_exit = True
            await task


async def browser_run(chrome, base_url, profile_path, query):
    """Use local Chromium DevTools to time the real workbench DOM."""
    from websockets.asyncio.client import connect

    profile_path.mkdir(parents=True)
    port_path = profile_path / "DevToolsActivePort"
    process = await asyncio.create_subprocess_exec(
        chrome,
        "--headless=new",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-renderer-backgrounding",
        "--no-first-run",
        "--remote-debugging-port=0",
        f"--user-data-dir={profile_path}",
        f"{base_url}/flow-demo?view=execution",
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(20):
            while not port_path.exists():
                if process.returncode is not None:
                    raise RuntimeError("benchmark_chromium_exited")
                await asyncio.sleep(0.05)
            port = port_path.read_text().splitlines()[0]
            async with httpx.AsyncClient(trust_env=False) as client:
                while True:
                    tabs = (
                        await client.get(f"http://127.0.0.1:{port}/json/list")
                    ).json()
                    tab = next(
                        (t for t in tabs if t.get("url", "").startswith(base_url)), None
                    )
                    if tab:
                        break
                    await asyncio.sleep(0.05)
        async with connect(
            tab["webSocketDebuggerUrl"], max_size=32 * 1024 * 1024
        ) as connection:
            counter = 0

            async def evaluate(expression):
                nonlocal counter
                counter += 1
                await connection.send(
                    json.dumps(
                        {
                            "id": counter,
                            "method": "Runtime.evaluate",
                            "params": {
                                "expression": expression,
                                "awaitPromise": True,
                                "returnByValue": True,
                            },
                        }
                    )
                )
                async for payload in connection:
                    response = json.loads(payload)
                    if response.get("id") != counter:
                        continue
                    result = response.get("result") or {}
                    if response.get("error") or result.get("exceptionDetails"):
                        raise RuntimeError("benchmark_browser_script_failed")
                    return result["result"].get("value")
                raise RuntimeError("benchmark_browser_connection_closed")

            async with asyncio.timeout(20):
                while not await evaluate(
                    "document.querySelector('#open-new-run')?.disabled === false"
                ):
                    await asyncio.sleep(0.05)
            script = Path(__file__).with_name("survey_latency_browser.js").read_text()
            data = json.dumps({"query": query, "timeoutMs": TIMEOUT_SECONDS * 1000})
            async with asyncio.timeout(TIMEOUT_SECONDS + 10):
                return await evaluate(f"({script})({data})")
    finally:
        if process.returncode is None:
            process.terminate()
        await process.wait()


async def replay_run(config, case, root, archive, snapshot, *, run_id):
    """Run the same workflow with one authenticated replay input."""
    recorder = RunRecorder(run_id, case.query, archive)
    recorder.set_config_summary(build_config_summary(config, "strict_auto_v2"))
    replay = ExternalInputReplay(
        query=snapshot.query,
        decomposition=tuple(snapshot.decomposition),
        search_outputs=snapshot.search_outputs,
        fingerprint=snapshot.fingerprint,
    )
    agent = LitAgent(
        config,
        trace_hook=recorder,
        run_policy=RunPolicy.isolated_benchmark(),
        external_inputs=replay,
    )
    agent._session_id = run_id
    agent._evaluation_progress_root = root / "evaluations"
    agent._targeted_repair_state_path = root / "repairs" / f"{run_id}.json"
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            async with agent:
                report = await agent.run(case.query, options=options_for(config))
    except BaseException as exc:
        recorder("benchmark.accounting", agent.benchmark_accounting())
        recorder.finalize(
            error=f"{type(exc).__name__}: {getattr(exc, 'code', '')}",
            terminal_status="failed",
        )
        if isinstance(exc, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
            raise
    else:
        recorder("benchmark.accounting", agent.benchmark_accounting())
        recorder.finalize(report=report, terminal_status="completed")
    return archive.path_for(run_id)


async def execute(
    root, config, dataset, profiles, slots, manifest, chrome, resumed=None
):
    """Persist every observed run and stop ahead of unknown or unapproved spend."""
    archive = ArchiveRepository(root / "runs")
    seed_root = root / "cache-seed"
    isolated_cache(Path(config.rag.raw_root), seed_root / "raw")
    if cache_identity(seed_root / "raw") != manifest["raw_pdf_cache_sha256"]:
        raise ValueError("benchmark_raw_pdf_cache_changed")
    index_root = root / "index-cache"
    isolated_cache(seed_root / "raw", index_root / "raw")
    config = configure_cache(config, index_root)
    runner = SurveyBenchmarkRunner(
        base_config=config,
        judge_config=load_survey_judge_config(
            Path("benchmarks/survey/judge.example.yaml")
        ),
        artifacts=BenchmarkArtifactRepository(root),
        run_archive=archive,
    )
    observed = dict(resumed.observations) if resumed else {}
    snapshots = dict(resumed.snapshots) if resumed else {}
    remaining = [slot for slot in slots if slot.key not in observed]
    eligible = remaining[: manifest.get("max_runs", len(remaining))]
    profile_ids = {slot.profile_id for slot in eligible}
    await runner._prepare_corpora(
        dataset, [profile for profile in profiles if profile.profile_id in profile_ids]
    )
    configs = {
        profile.profile_id: runner._config_for(profile, dataset.dataset_version)
        for profile in profiles
    }
    merged = str(root / "corpus" / "merged_manifest.yaml")
    for effective in configs.values():
        effective.rag.manifest_path = merged
    tokens, requests = 0, 0
    remaining_reason = "pending"
    active = None
    active_replay_id = None
    try:
        async with workbench(configs["external-abstract"], root, archive) as (
            base_url,
            state,
        ):
            for slot in eligible:
                if (
                    tokens + manifest["per_run_tokens"] > manifest["max_tokens"]
                    or requests + manifest["per_run_requests"]
                    > manifest["max_requests"]
                ):
                    remaining_reason = "benchmark_budget_exhausted"
                    break
                case = dataset.case_by_id(slot.case_id)
                snapshot = snapshots.get(slot.case_id)
                cache = root / "cache" / slot.key
                isolated_cache(seed_root / "raw", cache / "raw")
                effective = configure_cache(configs[slot.profile_id], cache)
                browser = None
                retry_of = manifest.get("retry_of", {}).get(slot.key)
                if slot.source == "workbench":
                    state.config = effective
                    active = slot
                    browser = await browser_run(
                        chrome, base_url, root / "chrome" / slot.key, case.query
                    )
                    run_id = browser.get("runId")
                    if (
                        not isinstance(run_id, str)
                        or len(run_id) != 12
                        or not run_id.isalnum()
                    ):
                        observed[slot.key] = {
                            "status": "failed",
                            "reasons": [
                                browser.get("error") or "missing_browser_receipt"
                            ],
                            "browser_ms": browser.get("markers") or {},
                        }
                        remaining_reason = "benchmark_usage_unknown"
                        break
                    path = archive.path_for(run_id)
                    if (
                        browser.get("error") == "workbench_timeout"
                        and not path.is_file()
                    ):
                        async with httpx.AsyncClient(
                            base_url=base_url, trust_env=False
                        ) as client:
                            await client.delete(f"/survey/{run_id}")
                        async with asyncio.timeout(30):
                            while not path.is_file():
                                await asyncio.sleep(0.2)
                    if path.is_file():
                        artifact = json.loads(path.read_bytes())
                        try:
                            snapshot = _capture_external_inputs_from_artifact(
                                case=case, repetition=1, artifact=artifact
                            )
                        except ValueError:
                            remaining_reason = "external_input_snapshot_invalid"
                        else:
                            write_json(
                                root / "inputs" / f"{case.case_id}.json",
                                snapshot.model_dump(mode="json"),
                            )
                            snapshots[slot.case_id] = snapshot
                elif snapshot is None:
                    remaining_reason = "external_input_snapshot_unavailable"
                    break
                else:
                    active = slot
                    run_id = (
                        f"{slot.key}-retry-{uuid4().hex[:12]}" if retry_of else slot.key
                    )
                    active_replay_id = run_id
                    path = await replay_run(
                        effective, case, root, archive, snapshot, run_id=run_id
                    )

                if not path.is_file():
                    row = {
                        "status": "failed",
                        "reasons": ["run_artifact_missing"],
                    }
                    if browser is not None:
                        row.update(
                            run_id=run_id, browser_ms=browser.get("markers") or {}
                        )
                        if browser.get("error"):
                            row["reasons"].append(browser["error"])
                    if retry_of:
                        row.update(run_id=run_id, retry_of=retry_of)
                    observed[slot.key] = row
                    remaining_reason = "benchmark_usage_unknown"
                    break
                row = save_observation(
                    slot,
                    case,
                    path,
                    root,
                    set(dataset.corpus_paper_ids),
                    browser=(browser or {}).get("markers"),
                    snapshot=snapshot.fingerprint if snapshot else None,
                    search_tasks=(
                        snapshot.search_outputs if slot.source == "replay" else None
                    ),
                )
                if snapshot:
                    row["input_snapshot_path"] = f"inputs/{case.case_id}.json"
                if browser and browser.get("error"):
                    row["reasons"].append(browser["error"])
                if retry_of:
                    if row["run_id"] != run_id:
                        raise ValueError("benchmark_retry_run_identity_mismatch")
                    row["retry_of"] = retry_of
                observed[slot.key] = row
                publish(root, slots, observed, "running", "pending")
                usage = row["usage"]
                if usage["usage_uncertain"] or not all(
                    type(usage[key]) is int
                    for key in ("billed_tokens", "paid_requests")
                ):
                    remaining_reason = "benchmark_usage_unknown"
                    break
                tokens += usage["billed_tokens"]
                requests += usage["paid_requests"]
                if remaining_reason != "pending":
                    break
                invalid_evaluation = {
                    "evaluation_report_identity_mismatch",
                    "benchmark_evaluation_policy_mismatch",
                }
                if invalid_evaluation.intersection(row["reasons"]):
                    remaining_reason = "invalid_evaluation_snapshot"
                    break
                if "external_input_fingerprint_mismatch" in row["reasons"] or (
                    slot.source == "replay"
                    and row["downstream_ms"] is None
                    and row["status"] == "completed"
                ):
                    remaining_reason = "invalid_controlled_replay"
                    break
    except BaseException as exc:
        remaining_reason = f"benchmark_interrupted:{type(exc).__name__}"
        if active is not None and active.key not in observed:
            # Replay cancellation is persisted by its recorder before the
            # cancellation reaches the campaign owner.
            path = archive.path_for(active_replay_id or active.key)
            snapshot = snapshots.get(active.case_id)
            if active.source == "replay" and path.is_file():
                row = save_observation(
                    active,
                    dataset.case_by_id(active.case_id),
                    path,
                    root,
                    set(dataset.corpus_paper_ids),
                    snapshot=snapshot.fingerprint,
                    search_tasks=snapshot.search_outputs,
                )
                row["input_snapshot_path"] = f"inputs/{active.case_id}.json"
            else:
                status = "failed"
                if isinstance(exc, asyncio.CancelledError):
                    status = "cancelled"
                elif isinstance(exc, TimeoutError):
                    status = "timeout"
                row = {
                    "status": status,
                    "reasons": [remaining_reason, "run_artifact_unavailable"],
                }
                if active.source == "workbench" and browser is not None:
                    row.update(
                        run_id=browser.get("runId"),
                        browser_ms=browser.get("markers") or {},
                    )
                    if browser.get("error"):
                        row["reasons"].append(browser["error"])
            if retry_of := manifest.get("retry_of", {}).get(active.key):
                row.update(run_id=active_replay_id, retry_of=retry_of)
            observed[active.key] = row
        raise
    finally:
        if remaining_reason == "pending" and len(eligible) < len(remaining):
            remaining_reason = "outside_approved_run_scope"
        done = len(observed) == len(slots) and remaining_reason == "pending"
        publish(
            root, slots, observed, "completed" if done else "partial", remaining_reason
        )


def main() -> None:
    """Create a dry-run ledger or execute one explicitly approved manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/agent-format-repair.yaml")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--max-runs",
        type=int,
        help="Only execute this many rows from the start of the matrix",
    )
    parser.add_argument(
        "--resume-from", type=Path, help="Reuse recorded rows from a terminal campaign"
    )
    parser.add_argument(
        "--usage-receipt",
        type=Path,
        help="Ark usage response for one unresolved request",
    )
    parser.add_argument(
        "--carry-unsettled",
        action="store_true",
        help="Continue an approved segment with the full prior run limit reserved",
    )
    parser.add_argument(
        "--retry-replay-slot",
        action="append",
        help="Select one settled failed replay for a separately approved retry",
    )
    parser.add_argument("--chrome", default=os.getenv("LITAGENT_CHROME_EXECUTABLE"))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument(
        "--approve-manifest", help="Exact dry-run SHA-256, authorizing paid execution"
    )
    args = parser.parse_args()
    if args.usage_receipt and not args.resume_from:
        parser.error("--usage-receipt requires --resume-from")
    if args.carry_unsettled and not args.resume_from:
        parser.error("--carry-unsettled requires --resume-from")
    if args.retry_replay_slot and not args.resume_from:
        parser.error("--retry-replay-slot requires --resume-from")

    config = load_config(args.config)
    config.eval.benchmark_faithfulness_policy = BENCHMARK_FAITHFULNESS_POLICY
    config.rag.writeback_enabled = False
    config.api.max_concurrent_surveys = 1
    dataset = load_survey_dataset(DATASET)
    profiles = load_survey_profiles(PROFILES_PATH)
    if {p.profile_id for p in profiles} != set(PROFILES):
        parser.error("expected the three latency benchmark profiles")
    slots = run_slots()
    resumed = (
        read_resumed_campaign(
            args.resume_from.resolve(),
            slots,
            dataset,
            args.usage_receipt.read_bytes() if args.usage_receipt else None,
            carry_unsettled=args.carry_unsettled,
        )
        if args.resume_from
        else None
    )
    if args.retry_replay_slot:
        if len(resumed.observations) != len(slots):
            parser.error("--retry-replay-slot requires a fully observed baseline")
        resumed = resumed.select_failed_replays(args.retry_replay_slot)
    manifest = make_manifest(
        config,
        dataset,
        profiles,
        slots,
        args.max_tokens,
        args.max_requests,
        args.max_runs,
        resumed,
    )
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "started.json").exists():
        parser.error("this campaign already started; choose a new output directory")
    prior = resumed.observations if resumed else {}
    if args.dry_run:
        if resumed:
            resumed.materialize(root)
        write_json(root / "manifest.json", manifest)
        publish(root, slots, prior, "dry-run", "awaiting_confirmed_budget")
        print(f"{root / 'manifest.json'}\n{manifest['approval_sha256']}")
        return

    stored = json.loads((root / "manifest.json").read_bytes())
    if manifest != stored or args.approve_manifest != manifest["approval_sha256"]:
        parser.error("approval hash or effective inputs differ from the dry-run")
    if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
        parser.error("generation credentials are unavailable")
    chrome = args.chrome or shutil.which("chromium") or shutil.which("google-chrome")
    if chrome is None:
        mac_chrome = Path(
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        )
        if mac_chrome.is_file():
            chrome = str(mac_chrome)
    if not chrome or not Path(chrome).is_file():
        parser.error("pass --chrome with the executable path")
    with (root / "started.json").open("x", encoding="utf-8") as handle:
        json.dump({"approval_sha256": args.approve_manifest}, handle)
    try:
        if resumed:
            resumed.materialize(root)
        asyncio.run(
            execute(root, config, dataset, profiles, slots, manifest, chrome, resumed)
        )
    except BaseException:
        if (
            not json.loads((root / "benchmark.json").read_bytes())["status"]
            == "partial"
        ):
            publish(root, slots, prior, "partial", "benchmark_preparation_failed")
        raise


if __name__ == "__main__":
    main()
