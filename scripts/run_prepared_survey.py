"""Continue a known public preparation under its original token and request cap."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

from litagent.agents.extractor import ExtractorWorker
from litagent.agents.search import SearchSourceOutcome, SearchSourceStatus
from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.datasets import load_survey_dataset, load_survey_profiles
from litagent.benchmark.quality_suite import suite_lock
from litagent.benchmark.survey_runner import SurveyBenchmarkRunner
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.evidence import collect_ledger
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import build_fact_cards
from litagent.report_audit import ReportAuditService, atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.injection import InjectionDetector
from scripts.rebuild_extraction_preparation import file_sha, known_preparation_usage
from scripts.run_stage_model_quality import check_models, options

VERSION = "prepared-survey-continuation-v1"
OWNER_ROOT = Path("artifacts/acceptance/preparation-continuation-owners")


def runtime_identity():
    """Pin the libraries that compile, parse, and transmit the frozen inputs."""
    return {
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in (
                "openai",
                "httpx",
                "pydantic",
                "langgraph",
                "PyMuPDF",
                "sentence-transformers",
            )
        },
    }


def effective_config():
    """Use the same fulltext product profile as the parent quality trial."""
    config = load_config()
    dataset = load_survey_dataset(Path("benchmarks/survey/dataset.yaml"))
    profile = next(
        p
        for p in load_survey_profiles(Path("benchmarks/survey/profiles.yaml"))
        if p.profile_id == "external-fulltext"
    )
    factory = SurveyBenchmarkRunner(
        base_config=config,
        judge_config=ReportAuditService(config).judge_config(),
        artifacts=BenchmarkArtifactRepository(Path("artifacts/benchmarks/survey")),
        run_archive=ArchiveRepository(),
    )
    return factory._config_for(profile, dataset.dataset_version)


def validate_preparation(path, config):
    """Require complete known accounting, exact parents, and untouched source bytes."""
    check_models(config)
    if config.extractor.record_identity != "source_fields_v1":
        raise ValueError("prepared_record_contract_invalid")
    if config.rag.content_mode != "abstract_and_selected_fulltext":
        raise ValueError("prepared_product_profile_invalid")
    p = json.loads(path.read_text())
    if p.get("version") != "received-preparation-rebuild-v1" or p.get(
        "binding"
    ) != digest({k: v for k, v in p.items() if k != "binding"}):
        raise ValueError("prepared_input_binding_invalid")
    parent_path = Path(p["source_artifact"])
    if file_sha(parent_path) != p["source_artifact_sha256"]:
        raise ValueError("prepared_parent_changed")
    parent = json.loads(parent_path.read_text())
    _, _, budget = known_preparation_usage(parent)
    if (
        budget != p["inherited_budget"]
        or budget["usage_uncertain"]
        or budget["requests_sent"] >= 64
    ):
        raise ValueError("prepared_accounting_invalid")
    case = load_survey_dataset(Path("benchmarks/survey/dataset.yaml")).case_by_id(
        "few-shot-metric"
    )
    if (
        p["query"] != case.query
        or parent["query"] != p["query"]
        or parent["run_id"] != p["source_run_id"]
    ):
        raise ValueError("prepared_question_or_parent_invalid")
    for name, sha in p["asset_hashes"].items():
        if file_sha(Path(name)) != sha:
            raise ValueError("prepared_cached_asset_changed")
    ledger = collect_ledger(p["extractions"])
    if (
        digest(ledger) != p["evidence_sha256"]
        or build_fact_cards(ledger) != p["fact_cards"]
    ):
        raise ValueError("prepared_evidence_changed")
    graph = TaskGraph.from_snapshot(p["graph"])
    if graph.get_task("extract").result != p["extractions"]:
        raise ValueError("prepared_graph_evidence_changed")
    extracted_ids = [item["paper_id"] for item in p["extractions"]]
    if extracted_ids != [
        item["paper_id"]
        for item in graph.get_task("relevance_gate").result
        if item["paper_id"] in extracted_ids
    ]:
        raise ValueError("prepared_candidate_order_changed")
    return p


def pending_papers(preparation):
    """Identify only candidates for which no original extraction request was sent."""
    graph = TaskGraph.from_snapshot(preparation["graph"])
    papers = graph.get_task("relevance_gate").result
    extracted = [x["paper_id"] for x in preparation["extractions"]]
    pids = [p["paper_id"] for p in papers]
    if (
        len(extracted) != len(set(extracted))
        or len(pids) != len(set(pids))
        or not set(extracted).issubset(pids)
    ):
        raise ValueError("prepared_paper_ids_invalid")
    pending = [p for p in papers if p["paper_id"] not in extracted]
    if [p["paper_id"] for p in pending] != preparation["scope"][
        "not_extracted_paper_ids"
    ]:
        raise ValueError("prepared_pending_scope_changed")
    return copy.deepcopy(pending)


def prepare(root, preparation_path, config):
    """Freeze a single continuation; its cap includes all inherited paid operations."""
    p = validate_preparation(preparation_path, config)
    pending = pending_papers(p)
    files = set(Path("src/litagent").rglob("*.py")) | {
        Path(__file__),
        Path("scripts/rebuild_extraction_preparation.py"),
        Path("scripts/run_stage_model_quality.py"),
        Path("config/default.yaml"),
        Path("benchmarks/survey/dataset.yaml"),
        Path("benchmarks/survey/profiles.yaml"),
        preparation_path,
        Path(p["source_artifact"]),
    }
    files.update(
        path
        for path in Path("src/litagent/skills").rglob("*")
        if path.is_file() and path.suffix in {".md", ".yaml", ".yml"}
    )
    manifest = {
        "version": VERSION,
        "run_id": root.name,
        "preparation_path": str(preparation_path.resolve()),
        "preparation_sha256": file_sha(preparation_path),
        "parent_sha256": p["source_artifact_sha256"],
        "config": build_config_summary(config, "strict_auto_v2"),
        "options": options().model_dump(mode="json"),
        "hashes": {str(f.resolve()): file_sha(f) for f in sorted(files)},
        "inherited_budget": p["inherited_budget"],
        "pending_paper_ids": [v["paper_id"] for v in pending],
        "total_tokens": 500000,
        "total_requests": 64,
        "new_token_ceiling": 500000 - p["inherited_budget"]["used_tokens"],
        "new_request_ceiling": 64 - p["inherited_budget"]["requests_sent"],
        "retries": 0,
        "scope": "development continuation, not a first-pass benchmark success",
        "runtime": runtime_identity(),
        "pending_concurrency": min(config.extractor.max_concurrent, 2),
    }
    root.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for path in sorted(files):
        if path.suffix not in {".py", ".yaml", ".yml", ".md"}:
            continue
        try:
            relative = path.resolve().relative_to(Path.cwd())
        except ValueError:
            continue
        target = root / "frozen-files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if file_sha(target) != manifest["hashes"][str(path.resolve())]:
            raise ValueError("prepared_prepare_input_changed")
        snapshots[str(target.resolve())] = file_sha(target)
    manifest["snapshots"] = snapshots
    manifest["binding"] = digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def validate_manifest(root, config):
    """Reject changed code or accounting before acquiring a paid owner."""
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("version") != VERSION
        or m.get("run_id") != root.name
        or m.get("binding") != digest({k: v for k, v in m.items() if k != "binding"})
        or m.get("config") != build_config_summary(config, "strict_auto_v2")
        or m.get("options") != options().model_dump(mode="json")
        or m.get("runtime") != runtime_identity()
        or m.get("pending_concurrency") != min(config.extractor.max_concurrent, 2)
    ):
        raise ValueError("prepared_manifest_invalid")
    for name, sha in {**m["hashes"], **m.get("snapshots", {})}.items():
        if file_sha(Path(name)) != sha:
            raise ValueError("prepared_frozen_source_changed")
    p = validate_preparation(Path(m["preparation_path"]), config)
    if m["parent_sha256"] != p["source_artifact_sha256"] or m[
        "preparation_sha256"
    ] != file_sha(Path(m["preparation_path"])):
        raise ValueError("prepared_manifest_parent_invalid")
    if (
        m["inherited_budget"] != p["inherited_budget"]
        or m["total_tokens"] != 500000
        or m["total_requests"] != 64
        or m["new_token_ceiling"] != 500000 - p["inherited_budget"]["used_tokens"]
        or m["new_request_ceiling"] != 64 - p["inherited_budget"]["requests_sent"]
        or m["retries"] != 0
    ):
        raise ValueError("prepared_manifest_budget_invalid")
    if m["pending_paper_ids"] != [v["paper_id"] for v in pending_papers(p)]:
        raise ValueError("prepared_pending_scope_changed")
    return m, p


class FrozenPreparedPlanner:
    """Return the verified graph without repeating planning, search, or extraction."""

    def __init__(self, preparation, extractions):
        """Keep a private snapshot whose only remaining task is report generation."""
        self.query = preparation["query"]
        self.snapshot = copy.deepcopy(preparation["graph"])
        for task in self.snapshot["tasks"]:
            if task["task_id"] == "extract":
                task["result"] = copy.deepcopy(extractions)

    async def plan(self, query):
        """Restore the exact public question and verified dependency graph."""
        if query != self.query:
            raise ValueError("prepared_query_changed")
        return TaskGraph.from_snapshot(copy.deepcopy(self.snapshot))


class NoAdditionalFulltext:
    """Keep the original per-round download cap; pending candidates use frozen chunks."""

    async def enrich(self, papers, requirements):
        """Reuse supplied public chunks without claiming another fulltext review."""
        result = copy.deepcopy(papers)
        for p in result:
            p["fulltext_acquisition"] = {
                "status": "reused_original_preparation_input",
                "full_document_reviewed": False,
                "reason_code": "no_additional_download_in_continuation",
            }
        return result


def restore_shared_budget(agent, preparation):
    """Restore into the actual object already borrowed by all clients and scheduler."""
    budget = agent._cost_budget
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.restore(preparation["inherited_budget"])
    for role in ("extraction", "generation", "evaluation"):
        if agent._stage_llm(role)._cost_budget is not budget:
            raise ValueError("prepared_shared_budget_invalid")
    if agent._scheduler._cost_budget is not budget:
        raise ValueError("prepared_scheduler_budget_invalid")


def claim_owner(root, manifest):
    """Allow one child identity for the exact known parent preparation."""
    registry = OWNER_ROOT
    registry.mkdir(parents=True, exist_ok=True)
    key = manifest["parent_sha256"]
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("prepared_owner_key_invalid")
    target = registry / (key + ".json")
    expected = {
        "child_binding": manifest["binding"],
        "child_root": str(root.resolve()),
        "parent_sha256": key,
    }
    with suite_lock(registry / (key + ".lock")):
        if target.exists():
            previous = json.loads(target.read_text())
            if previous.get("status") == "released_before_model_call":
                proof = verify_zero_call_child(
                    Path(previous["child_root"]), previous["child_binding"]
                )
                if proof != previous:
                    raise ValueError("prepared_zero_call_release_changed")
            elif previous != expected:
                raise ValueError("prepared_parent_already_owned")
        atomic_json(target, expected)


def verify_zero_call_child(root, binding):
    """Require immutable proof that no model request or budget increment occurred."""
    manifest = json.loads((root / "manifest.json").read_text())
    state = json.loads((root / "checkpoint.json").read_text())
    if (
        manifest.get("binding") != binding
        or digest({k: v for k, v in manifest.items() if k != "binding"}) != binding
        or state.get("binding") != binding
        or state.get("status") != "failed_no_replay"
        or state.get("incremental_cost")
        != {"tokens": 0, "requests": 0, "usage_complete": True}
        or state.get("budget") != manifest["inherited_budget"]
    ):
        raise ValueError("prepared_zero_call_proof_invalid")
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    if any(
        event.get("event") in {"llm.start", "llm.complete", "llm.failed"}
        for event in events
    ):
        raise ValueError("prepared_zero_call_transport_observed")
    return {
        "status": "released_before_model_call",
        "child_root": str(root.resolve()),
        "child_binding": binding,
        "parent_sha256": manifest["parent_sha256"],
        "checkpoint_sha256": file_sha(root / "checkpoint.json"),
        "journal_sha256": file_sha(root / "events.jsonl"),
    }


def release_zero_call_owner(root):
    """Release only an unused owner; retain its signed-input and zero-call history."""
    manifest = json.loads((root / "manifest.json").read_text())
    proof = verify_zero_call_child(root, manifest["binding"])
    key = proof["parent_sha256"]
    if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
        raise ValueError("prepared_owner_key_invalid")
    with suite_lock(OWNER_ROOT / (key + ".lock")):
        path = OWNER_ROOT / (key + ".json")
        owner = json.loads(path.read_text())
        if owner.get("child_binding") != manifest["binding"] or owner.get(
            "child_root"
        ) != str(root.resolve()):
            raise ValueError("prepared_zero_call_owner_conflict")
        history = OWNER_ROOT / "zero-call-history"
        history.mkdir(exist_ok=True)
        atomic_json(history / (manifest["binding"] + ".json"), proof)
        atomic_json(path, proof)
    return proof


async def run(root, config, *, live=False):
    """Run real workers once and retain unknown work without automatic resend."""
    manifest, p = validate_manifest(root, config)
    if not live:
        return {
            "status": "planned",
            "new_token_ceiling": manifest["new_token_ceiling"],
            "new_request_ceiling": manifest["new_request_ceiling"],
            "pending_papers": len(pending_papers(p)),
        }
    with suite_lock(root / ".trial.lock"):
        checkpoint = root / "checkpoint.json"
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get("binding") != manifest["binding"]:
                raise ValueError("prepared_checkpoint_conflict")
            if saved["status"] == "started_no_replay":
                saved.update(status="interrupted_no_replay")
                atomic_json(checkpoint, saved)
            return saved
        claim_owner(root, manifest)
        state = {
            "status": "started_no_replay",
            "binding": manifest["binding"],
            "new_reserved_tokens": manifest["new_token_ceiling"],
            "new_reserved_requests": manifest["new_request_ceiling"],
        }
        atomic_json(checkpoint, state)
        recorder = RunRecorder(root.name, p["query"], ArchiveRepository())
        recorder.set_config_summary(manifest["config"])
        recorder.set_citation_style("ieee")

        def journal(event, data):
            with (root / "events.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {"event": event, "data": data}, ensure_ascii=False, default=str
                    )
                    + "\n"
                )
                f.flush()
                os.fsync(f.fileno())

        tracer = LangFuseTracer(
            config.observability.langfuse_host,
            os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            os.getenv("LANGFUSE_SECRET_KEY", ""),
            trace_seed=root.name,
        )
        hook = CompositeTraceHook(
            recorder, journal, RedactingTraceHook(tracer, "metadata_only")
        )
        agent = LitAgent(
            config, trace_hook=hook, run_policy=RunPolicy.isolated_benchmark()
        )
        agent._session_id = root.name
        agent._targeted_repair_state_path = root / "targeted-repair-state.json"
        restored = False
        report = None
        try:
            async with agent:
                restore_shared_budget(agent, p)
                restored = True
                hook(
                    "survey.start",
                    {
                        "session_id": root.name,
                        "query": p["query"],
                        "phase": "prepared_continuation",
                    },
                )
                pending = pending_papers(p)
                new = []
                if pending:
                    hook(
                        "worker.start",
                        {
                            "task_id": "extract_pending",
                            "agent_type": "extractor",
                            "pending_papers": len(pending),
                        },
                    )
                    task = SubTask(
                        "extract_pending",
                        "Extract only unrequested frozen candidates",
                        "extractor",
                        input_data={
                            "quality_policy": "strict_auto_v2",
                            "upstream_results": {"relevance_gate": pending},
                        },
                        max_retries=0,
                    )
                    worker = ExtractorWorker(
                        agent._extraction_strategy,
                        max_concurrent=manifest["pending_concurrency"],
                        detector=InjectionDetector(),
                        fulltext_loader=NoAdditionalFulltext(),
                        failure_callback=agent._cost_budget.stop_after_worker_failure,
                    )
                    token = set_task_id("extract_pending")
                    try:
                        new = await worker.execute(task)
                    finally:
                        reset_task_id(token)
                    hook(
                        "worker.complete",
                        {
                            "task_id": "extract_pending",
                            "agent_type": "extractor",
                            "output": new,
                        },
                    )
                extractions = [*p["extractions"], *new]
                if len(extractions) != p["scope"]["relevant_papers"]:
                    raise ValueError("prepared_extraction_scope_incomplete")
                by_id = {item["paper_id"]: item for item in extractions}
                original = (
                    TaskGraph.from_snapshot(p["graph"])
                    .get_task("relevance_gate")
                    .result
                )
                if len(by_id) != len(original):
                    raise ValueError("prepared_extraction_identity_conflict")
                extractions = [by_id[item["paper_id"]] for item in original]
                atomic_json(
                    root / "completed-preparation.json",
                    {
                        "extractions": extractions,
                        "budget": agent._cost_budget.snapshot(),
                        "parent_binding": p["binding"],
                    },
                )
                agent._planner = FrozenPreparedPlanner(p, extractions)
                outcomes = [
                    SearchSourceOutcome(
                        **{**o, "status": SearchSourceStatus(o["status"])}
                    )
                    for o in p["search_source_outcomes"]
                ]
                agent._search = SimpleNamespace(
                    reset_source_outcomes=lambda: None,
                    get_source_outcomes=lambda: tuple(outcomes),
                )
                graph = await agent._planner.plan(p["query"])
                hook.capture_graph(graph)
                for tid in graph.get_results():
                    hook(
                        "worker.complete",
                        {
                            "task_id": tid,
                            "agent_type": graph.get_task(tid).agent_type,
                            "output": graph.get_task(tid).result,
                            "execution": "reused_verified_preparation",
                            "elapsed_ms": 0,
                            "source_run_id": p["source_run_id"],
                        },
                    )
                report = await agent.run(p["query"], options=options())
                report["metadata"]["prepared_continuation"] = {
                    "parent_run_id": p["source_run_id"],
                    "parent_sha256": p["source_artifact_sha256"],
                    "inherited_tokens": p["inherited_budget"]["used_tokens"],
                    "inherited_requests": p["inherited_budget"]["requests_sent"],
                    "newly_extracted_papers": len(new),
                    "first_pass_benchmark": False,
                }
                state["status"] = (
                    "completed"
                    if report.get("delivery", {}).get("publishable")
                    else "blocked"
                )
        except BaseException as exc:
            state.update(
                status=(
                    "interrupted_no_replay"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed_no_replay"
                ),
                reason_type=type(exc).__name__,
                reason=str(exc)[:240],
            )
        finally:
            budget = (
                agent._cost_budget.snapshot() if restored else p["inherited_budget"]
            )
            state.update(
                budget=budget,
                incremental_cost={
                    "tokens": budget["used_tokens"]
                    - p["inherited_budget"]["used_tokens"],
                    "requests": budget["requests_sent"]
                    - p["inherited_budget"]["requests_sent"],
                    "usage_complete": not budget["usage_uncertain"],
                },
                report_sha256=(
                    hashlib.sha256(report["survey"].encode()).hexdigest()
                    if report
                    else None
                ),
                trace_id=tracer.trace_id,
            )
            recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            recorder.finalize(
                report=report,
                error=state.get("reason"),
                terminal_status="completed" if report else "failed",
            )
            atomic_json(checkpoint, state)
            hook.flush()
        return state


async def main(args):
    """Keep preparation and paid execution separate and print only bounded status."""
    config = effective_config()
    if args.prepare:
        prepare(args.root, args.preparation, config)
    state = await run(args.root, config, live=args.live)
    print(
        json.dumps(
            {
                k: state.get(k)
                for k in (
                    "status",
                    "new_token_ceiling",
                    "new_request_ceiling",
                    "pending_papers",
                    "budget",
                    "incremental_cost",
                    "reason_type",
                    "report_sha256",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--preparation",
        type=Path,
        default=Path(
            "artifacts/acceptance/stage-model-quality-20261017-002/preparation-rebuild-ranked.json"
        ),
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    asyncio.run(main(parser.parse_args()))
