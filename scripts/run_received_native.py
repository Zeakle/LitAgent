"""Repair a known received draft under its inherited round and attempt budget."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

from litagent.agents.search import SearchSourceOutcome, SearchSourceStatus
from litagent.benchmark.quality_suite import suite_lock
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.evidence import collect_ledger
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.quality import (
    RECEIVED_NATIVE_CONTRACT,
    GroundedWritingError,
    build_fact_cards,
)
from litagent.quality import digest as quality_digest
from litagent.quality import (
    grounded_messages,
    native_task_identity,
    prepare_grounded_request,
    prepare_native_repair_request,
    render_grounded_report_traced,
    save_targeted_repairs,
    targeted_repair_state,
)
from litagent.report_audit import atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.budget import CostBudget, strict_request_reservation
from scripts.rebuild_extraction_preparation import file_sha
from scripts.run_prepared_survey import (
    FrozenPreparedPlanner,
    claim_owner,
    effective_config,
    restore_shared_budget,
    runtime_identity,
)
from scripts.run_stage_model_quality import check_models, options

VERSION = "received-native-continuation-v1"


def execution_config(config):
    """Allow review but no extra whole-report rewrite after the last repair slot."""
    result = config.model_copy(deep=True)
    result.adversarial.max_rounds = 1
    return result


def snapshot_path(root, path):
    """Preserve external input fixtures without assuming they live under the repo."""
    try:
        relative = path.resolve().relative_to(Path.cwd())
    except ValueError:
        relative = Path("external") / (file_sha(path) + "-" + path.name)
    return root / "frozen-files" / relative


def reconcile_parent(root):
    """Reconcile every sent operation before allowing a known worker stop to clear."""
    manifest = json.loads((root / "manifest.json").read_text())
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    if (
        manifest["binding"]
        != digest({k: v for k, v in manifest.items() if k != "binding"})
        or checkpoint.get("binding") != manifest["binding"]
        or checkpoint.get("status") != "failed_no_replay"
        or checkpoint.get("budget", {}).get("stop_reason") != "strict_worker_failed"
    ):
        raise ValueError("native_parent_identity_invalid")
    for name, expected in manifest["hashes"].items():
        path = Path(name)
        frozen = root / "frozen-files" / path.relative_to(Path.cwd())
        if file_sha(frozen if frozen.is_file() else path) != expected:
            raise ValueError("native_parent_source_changed")
    events = [
        json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()
    ]
    starts = [e["data"] for e in events if e["event"] == "llm.start"]
    completed = [e["data"] for e in events if e["event"] == "llm.complete"]
    by_id = {s["operation_id"]: s for s in starts}
    if (
        not starts
        or len(by_id) != len(starts)
        or len(starts) != len(completed)
        or len({x["operation_id"] for x in completed}) != len(completed)
        or any(e["event"] == "llm.failed" for e in events)
    ):
        raise ValueError("native_parent_operation_unknown")
    phases = [0, 0, 0]
    for item in completed:
        start = by_id.get(item["operation_id"], {})
        phase = 1 if ":synthesis:" in item["task_id"] else 0
        expected_model = manifest["config"]["effective"]["llm"]["model_routing"][
            "generation" if phase else "extraction"
        ]
        if (
            start.get("task_id") != item["task_id"]
            or start.get("model") != expected_model
            or item.get("actual_model") != expected_model
            or item.get("model") != expected_model
            or item.get("usage_available") is not True
            or item.get("finish_reason") != "stop"
            or (phase == 0 and item["task_id"] != "extract_pending")
            or any(
                type(item.get(k)) is not int or item[k] < 0
                for k in ("prompt_tokens", "completion_tokens")
            )
        ):
            raise ValueError("native_parent_usage_invalid")
        phases[phase] += item["prompt_tokens"] + item["completion_tokens"]
    before, after = manifest["inherited_budget"], checkpoint["budget"]
    budget = CostBudget.from_snapshot(after)
    if (
        not budget.strict
        or after["usage_uncertain"]
        or after["max_tokens"] != 500000
        or after["requests_sent"] != before["requests_sent"] + len(completed)
        or after["used_tokens"] != before["used_tokens"] + sum(phases)
        or after["call_count"] != before["call_count"] + len(completed)
        or after["phase_usage"]
        != [a + b for a, b in zip(before["phase_usage"], phases)]
        or after["requests_sent"] >= 64
        or after["used_tokens"] >= 500000
        or checkpoint.get("incremental_cost")
        != {"tokens": sum(phases), "requests": len(completed), "usage_complete": True}
    ):
        raise ValueError("native_parent_accounting_invalid")
    return manifest, checkpoint, events


def _read_input(path, config):
    """Rebuild the complete host inputs; reject any forged counters or source drift."""
    check_models(config)
    value = json.loads(path.read_text())
    if (
        value.get("version") != "received-native-recovery-input-v1"
        or value.get("binding")
        != digest({k: v for k, v in value.items() if k != "binding"})
        or value.get("config") != build_config_summary(config, "strict_auto_v2")
        or value.get("options") != options().model_dump(mode="json")
    ):
        raise ValueError("native_input_identity_invalid")
    parent = Path(value["parent_root"])
    for name, expected in value["parent_hashes"].items():
        if file_sha(parent / name) != expected:
            raise ValueError("native_input_parent_changed")
    manifest, checkpoint, events = reconcile_parent(parent)
    before = checkpoint["budget"]
    if (
        value["source_manifest_binding"] != manifest["binding"]
        or value["source_run_id"] != manifest["run_id"]
        or value["inherited_budget"] != before
        or value["remaining_requests"] != 64 - before["requests_sent"]
        or value["remaining_tokens"] != 500000 - before["used_tokens"]
        or value["remaining_repairs"] != 1
    ):
        raise ValueError("native_input_allowance_invalid")
    original = json.loads(Path(manifest["preparation_path"]).read_text())
    if file_sha(Path(manifest["preparation_path"])) != manifest[
        "preparation_sha256"
    ] or original["binding"] != digest(
        {k: v for k, v in original.items() if k != "binding"}
    ):
        raise ValueError("native_preparation_changed")
    for asset, expected in original["asset_hashes"].items():
        if file_sha(Path(asset)) != expected:
            raise ValueError("native_cached_source_changed")
    completed = json.loads((parent / "completed-preparation.json").read_text())
    ledger = collect_ledger(completed["extractions"])
    raw = next(e["data"] for e in events if e["event"] == "evidence.retrieve.complete")
    selection = EvidenceSelection.from_dict(
        {
            k: raw[k]
            for k in (
                "candidate_count",
                "selected_items",
                "section_evidence_ids",
                "method_by_section",
                "omitted_count",
                "estimated_tokens",
                "ranking_decisions",
            )
        }
    )
    if (
        selection.candidate_count != len(ledger)
        or len(completed["extractions"]) != original["scope"]["relevant_papers"]
        or any(ledger.get(k) != v for k, v in selection.selected_items.items())
    ):
        raise ValueError("native_selected_sources_invalid")
    for key in ("abstract", "conclusion"):
        selection.section_evidence_ids[key] = list(selection.selected_items)
        selection.method_by_section[key] = "selected_union"
    cards = build_fact_cards(selection.selected_items)
    repair = json.loads((parent / "targeted-repair-state.json").read_text())
    state = targeted_repair_state(repair["state"], value["source_run_id"])
    pro = [
        e["data"]
        for e in events
        if e["event"] == "llm.complete" and ":synthesis:" in e["data"]["task_id"]
    ]
    if (
        len(pro) != 2
        or len(state["attempts"]) != 1
        or state["attempts"][0]["status"] != "failed"
        or state != value["source_repair_state"]
        or repair["native_candidate"] != pro[0]["content"]
        or repair["native_response"] != pro[1]["content"]
        or value["candidate"] != repair["native_candidate"]
        or state["attempts"][0].get("response_sha256")
        != hashlib.sha256(pro[1]["content"].encode()).hexdigest()
        or state["attempts"][0].get("usage")
        != {
            "prompt_tokens": pro[1]["prompt_tokens"],
            "completion_tokens": pro[1]["completion_tokens"],
            "usage_available": True,
        }
        or quality_digest(cards) != repair["native_binding"]["fact_cards_sha256"]
        or cards != value["fact_cards"]
        or selection.to_dict() != value["selection"]
    ):
        raise ValueError("native_received_response_invalid")
    task_input = next(
        e["data"]["input"]
        for e in events
        if e["event"] == "worker.input" and e["data"]["task_id"] == "adversarial_review"
    )
    if task_input != value["source_input"]:
        raise ValueError("native_source_task_changed")
    try:
        render_grounded_report_traced(
            value["candidate"],
            cards,
            "academic_v2",
            "zh-CN",
            evidence_selection=selection,
            require_native=True,
            require_full_sources=True,
            require_scope_notices=True,
            analysis_mode=True,
        )
    except GroundedWritingError as failure:
        diagnostics = failure.diagnostics
    else:
        raise ValueError("native_recovery_not_required")
    messages, _ = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            "zh-CN",
            task_input["query"],
            coverage=task_input["coverage_requirements"],
            evidence_selection=selection,
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
    )
    request, schema, binding = prepare_native_repair_request(
        messages,
        value["candidate"],
        diagnostics,
        cards=cards,
        structure="academic_v2",
        language="zh-CN",
        evidence_selection=selection,
    )
    reservation = strict_request_reservation(request, 20000, schema)
    if (
        value["diagnostics"] != diagnostics
        or value["native_messages"] != request
        or value["native_schema"] != schema
        or value["native_binding"] != binding
        or value["native_reservation"] != reservation
        or reservation > int(131072 * 0.9)
        or reservation > 300000 - before["used_tokens"]
    ):
        raise ValueError("native_repair_input_invalid")
    return value, original, completed["extractions"]


def read_input(path, config):
    """Report malformed host artifacts without dispatching or hiding their shape error."""
    try:
        return _read_input(path, config)
    except (KeyError, TypeError, AttributeError, StopIteration) as exc:
        raise ValueError("native_input_shape_invalid") from exc


def prepare(root, source, config):
    """Freeze one explicit child while retaining its paid parent's owner and counters."""
    value, _, _ = read_input(source, config)
    files = set(Path("src/litagent").rglob("*.py")) | {
        Path(__file__),
        Path("scripts/run_prepared_survey.py"),
        Path("scripts/rebuild_extraction_preparation.py"),
        Path("scripts/run_stage_model_quality.py"),
        Path("config/default.yaml"),
        source,
    }
    files.update(
        p
        for p in Path("src/litagent/skills").rglob("*")
        if p.is_file() and p.suffix in {".md", ".yaml", ".yml"}
    )
    manifest = {
        "version": VERSION,
        "run_id": root.name,
        "input": str(source.resolve()),
        "input_sha256": file_sha(source),
        "source_binding": value["binding"],
        "parent_sha256": value["parent_hashes"]["checkpoint.json"],
        "config": build_config_summary(execution_config(config), "strict_auto_v2"),
        "options": options().model_dump(mode="json"),
        "runtime": runtime_identity(),
        "inherited_budget": value["inherited_budget"],
        "remaining_repairs": 1,
        "review_rounds": 1,
        "new_token_ceiling": value["remaining_tokens"],
        "new_request_ceiling": value["remaining_requests"],
        "hashes": {str(p.resolve()): file_sha(p) for p in sorted(files)},
        "retries": 0,
    }
    root.mkdir(parents=True, exist_ok=False)
    for path in files:
        target = snapshot_path(root, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if file_sha(target) != manifest["hashes"][str(path.resolve())]:
            raise ValueError("native_prepare_source_changed")
    manifest["binding"] = digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def validate_manifest(root, config):
    """Check the current executable and original accounting before any model call."""
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("version") != VERSION
        or m.get("run_id") != root.name
        or m.get("binding") != digest({k: v for k, v in m.items() if k != "binding"})
        or m.get("config")
        != build_config_summary(execution_config(config), "strict_auto_v2")
        or m.get("options") != options().model_dump(mode="json")
        or m.get("runtime") != runtime_identity()
        or m.get("retries") != 0
        or m.get("review_rounds") != 1
    ):
        raise ValueError("native_manifest_invalid")
    for path, expected in m["hashes"].items():
        if file_sha(Path(path)) != expected:
            raise ValueError("native_frozen_source_changed")
        if file_sha(snapshot_path(root, Path(path))) != expected:
            raise ValueError("native_frozen_snapshot_changed")
    value, original, extractions = read_input(Path(m["input"]), config)
    if (
        file_sha(Path(m["input"])) != m["input_sha256"]
        or value["binding"] != m["source_binding"]
        or m["parent_sha256"] != value["parent_hashes"]["checkpoint.json"]
        or m["inherited_budget"] != value["inherited_budget"]
        or m["new_token_ceiling"] != value["remaining_tokens"]
        or m["new_request_ceiling"] != value["remaining_requests"]
        or m["remaining_repairs"] != 1
    ):
        raise ValueError("native_manifest_allowance_invalid")
    return m, value, original, extractions


class FrozenSelection:
    """Reuse exact ranked sources instead of ranking them again after failure."""

    def __init__(self, value, extractions):
        self.query = value["source_input"]["query"]
        self.value = copy.deepcopy(value["selection"])
        self.ledger_sha = quality_digest(collect_ledger(extractions))

    async def select(self, query, ledger, **kwargs):
        """Reject changed evidence before returning the original section permissions."""
        if query != self.query or quality_digest(ledger) != self.ledger_sha:
            raise ValueError("native_frozen_selection_changed")
        return EvidenceSelection.from_dict(copy.deepcopy(self.value))


def resume_receipt(value, run_id):
    """Rebase only the session label, preserving every consumed parent attempt."""
    state = copy.deepcopy(value["source_repair_state"])
    state["run_id"] = run_id
    receipt = {
        "version": RECEIVED_NATIVE_CONTRACT,
        "run_id": run_id,
        "source_run_id": value["source_run_id"],
        "source_checkpoint_sha256": value["parent_hashes"]["checkpoint.json"],
        "candidate": value["candidate"],
        "candidate_sha256": hashlib.sha256(value["candidate"].encode()).hexdigest(),
        "fact_cards_sha256": quality_digest(value["fact_cards"]),
        "selection_sha256": quality_digest(value["selection"]),
        "task_sha256": native_task_identity(value["source_input"]),
        "repair_state": state,
    }
    receipt["binding_sha256"] = quality_digest(receipt)
    return receipt


async def run(root, config, *, live=False):
    """Execute the existing worker chain once without resending the received draft."""
    m, value, original, extractions = validate_manifest(root, config)
    if not live:
        return {
            "status": "planned",
            "new_requests": m["new_request_ceiling"],
            "new_tokens": m["new_token_ceiling"],
            "native_repairs": 1,
        }
    with suite_lock(root / ".trial.lock"):
        path = root / "checkpoint.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get("binding") != m["binding"]:
                raise ValueError("native_checkpoint_conflict")
            if saved["status"] == "started_no_replay":
                saved["status"] = "interrupted_no_replay"
                atomic_json(path, saved)
            return saved
        claim_owner(root, m)
        state = {
            "status": "started_no_replay",
            "binding": m["binding"],
            "inherited_budget": value["inherited_budget"],
            "remaining_repairs": 1,
        }
        atomic_json(path, state)
        recorder = RunRecorder(
            root.name, value["source_input"]["query"], ArchiveRepository()
        )
        recorder.set_config_summary(m["config"])
        recorder.set_citation_style("ieee")

        def journal(event, data):
            with (root / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {"event": event, "data": data}, ensure_ascii=False, default=str
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())

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
            execution_config(config),
            trace_hook=hook,
            run_policy=RunPolicy.isolated_benchmark(),
        )
        agent._session_id = root.name
        agent._targeted_repair_state_path = root / "targeted-repair-state.json"
        agent._evaluation_progress_root = root / "evaluations"
        restored, report = False, None
        try:
            async with agent:
                inherited = copy.deepcopy(value["inherited_budget"])
                # A new explicit child clears only the proven known worker stop.
                inherited["stop_reason"] = None
                restore_shared_budget(agent, {"inherited_budget": inherited})
                restored = True
                receipt = resume_receipt(value, root.name)
                save_targeted_repairs(
                    agent._targeted_repair_state_path,
                    receipt["repair_state"],
                    received_parent={
                        "source_run_id": value["source_run_id"],
                        "source_state": value["source_repair_state"],
                        "source_binding": value["binding"],
                    },
                )
                agent._synthesis._received_native = receipt
                agent._synthesis._evidence_selector = FrozenSelection(
                    value, extractions
                )
                planner = FrozenPreparedPlanner(original, extractions)
                for task in planner.snapshot["tasks"]:
                    if task["task_id"] == "adversarial_review":
                        task["input_data"] = copy.deepcopy(value["source_input"])
                agent._planner = planner
                outcomes = tuple(
                    SearchSourceOutcome(
                        **{**o, "status": SearchSourceStatus(o["status"])}
                    )
                    for o in original["search_source_outcomes"]
                )
                agent._search = SimpleNamespace(
                    reset_source_outcomes=lambda: None,
                    get_source_outcomes=lambda: outcomes,
                )
                graph = await planner.plan(value["source_input"]["query"])
                hook.capture_graph(graph)
                for tid, output in graph.get_results().items():
                    hook(
                        "worker.complete",
                        {
                            "task_id": tid,
                            "agent_type": graph.get_task(tid).agent_type,
                            "output": output,
                            "execution": "reused_verified_preparation",
                            "source_run_id": value["source_run_id"],
                            "elapsed_ms": 0,
                        },
                    )
                report = await agent.run(
                    value["source_input"]["query"], options=options()
                )
                report["metadata"]["native_continuation"] = {
                    "source_run_id": value["source_run_id"],
                    "source_checkpoint_sha256": receipt["source_checkpoint_sha256"],
                    "source_candidate_sha256": receipt["candidate_sha256"],
                    "inherited_requests": inherited["requests_sent"],
                    "inherited_tokens": inherited["used_tokens"],
                    "inherited_repairs": 1,
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
                agent._cost_budget.snapshot() if restored else value["inherited_budget"]
            )
            state.update(
                budget=budget,
                incremental_cost={
                    "requests": budget["requests_sent"]
                    - value["inherited_budget"]["requests_sent"],
                    "tokens": budget["used_tokens"]
                    - value["inherited_budget"]["used_tokens"],
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
            atomic_json(path, state)
            hook.flush()
        return state


async def main(args):
    """Keep the default path read-only and summarize only status and accounting."""
    config = effective_config()
    if args.prepare:
        prepare(args.root, args.source, config)
    result = await run(args.root, config, live=args.live)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    asyncio.run(main(parser.parse_args()))
