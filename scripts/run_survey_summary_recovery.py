"""Resume the five authenticated Survey summary-capacity failures without regenerating their bodies."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from uuid import uuid4

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.datasets import (
    load_survey_dataset,
    load_survey_judge_config,
    load_survey_profiles,
)
from litagent.benchmark.survey_latency import (
    publish,
    run_slots,
    save_observation,
    write_json,
)
from litagent.benchmark.survey_runner import SurveyBenchmarkRunner
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.eval.results import BENCHMARK_FAITHFULNESS_POLICY
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.orchestrator.task_graph import TaskGraph
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.safety.planned_budget import (
    EnvelopeExceeded,
    PlannedBudget,
    check_writer_envelope,
    planned_backend,
)
from litagent.writing_v2 import (
    CHECKPOINT_SUMMARY_REBUILD,
    WriterError,
    WriterSession,
    compile_document,
    finalize_synthesis,
)
from scripts.run_survey_latency import (
    DATASET,
    PROFILES_PATH,
    TIMEOUT_SECONDS,
    options_for,
    read_resumed_campaign,
    sha,
    source_identity,
)
from scripts.survey_summary_recovery_accounting import (
    checked_failed_summary_retry,
    checked_parent_ledger,
)
from scripts.survey_summary_recovery_inputs import load_frozen_summary_case
from scripts.view_survey_latency import read_campaign, retry_rows

TARGETS = {
    "replay-few-shot-metric-external-abstract-r1",
    "replay-few-shot-metric-external-fulltext-r1",
    "replay-vit-foundations-external-abstract-r1",
    "replay-few-shot-metric-external-only-r2",
    "replay-vit-foundations-external-fulltext-r2",
}
BODY_STAGES = {"body", "body_initial", "format_body"}
REMAINING_STAGES = {
    "summary",
    "reviewer",
    "repair",
    "inventory",
    "assessment",
    "global",
    "format_summary",
    "format_repair",
}


class CheckedRequest(Exception):
    """Stop the production patch immediately before its paid dispatch."""


class SummaryRecoveryBudget(PlannedBudget):
    """Fence adaptive reservations to the child's separately approved spend."""

    def __init__(self, parent):
        super().__init__(parent["plan"])
        self.restore(
            {**parent, "stop_reason": None, "closed_stages": sorted(BODY_STAGES)}
        )
        self.parent_tokens = parent["used_tokens"]
        self.parent_requests = parent["requests_sent"]
        self.exposure = remaining_exposure(self)
        self.approved_exhausted = False

    def reserve(self, tokens):
        """Check actual pending commitments before delegating durable reservation."""
        stage, _ = self._owner()
        if stage not in REMAINING_STAGES:
            raise ValueError("summary_recovery_stage_not_authorized")
        pending = [
            row
            for row in self.entries.values()
            if row["status"] in {"reserved_not_sent", "started_no_replay"}
        ]
        committed = sum(row["reserved_tokens"] for row in pending)
        reserved_requests = sum(row["status"] == "reserved_not_sent" for row in pending)
        if (
            self.approved_exhausted
            or self.used - self.parent_tokens + committed + tokens
            > self.exposure["max_tokens"]
            or self.snapshot()["requests_sent"]
            - self.parent_requests
            + reserved_requests
            >= self.exposure["max_requests"]
        ):
            self.approved_exhausted = True
            raise ValueError("summary_recovery_approved_budget_exhausted")
        return super().reserve(tokens)


def task_graph(artifact):
    """Restore the exact completed prerequisites and open only adversarial review."""
    source = artifact["graph"]
    outputs = {}
    for event in artifact["events"]:
        if event["event"] != "worker.complete":
            continue
        data = event["data"]
        name = data["task_id"]
        if (
            name in outputs
            or name not in source["tasks"]
            or data["agent_type"] != source["tasks"][name]["agent_type"]
        ):
            raise ValueError("summary_recovery_worker_receipt_conflict")
        outputs[name] = data["output"]
    expected = set(source["tasks"]) - {"adversarial_review"}
    if (
        set(outputs) != expected
        or source["tasks"]["adversarial_review"]["status"] != "failed"
        or any(source["tasks"][name]["status"] != "done" for name in expected)
    ):
        raise ValueError("summary_recovery_failed_boundary_invalid")
    tasks = [
        {
            **record,
            "input_data": record["input"],
            "status": "pending" if name == "adversarial_review" else "done",
            "result": outputs.get(name),
            "error": None,
        }
        for name, record in source["tasks"].items()
    ]
    return TaskGraph.from_snapshot(
        {"schema_version": 1, "tasks": tasks, "dependencies": source["dependencies"]}
    )


def restore_child_budget(budget, parent, path=None, *, after_capacity_failure=False):
    """Bind a separate owner to settled parent spend and close inherited body stages."""
    if (
        parent["stop_reason"] != "strict_worker_failed"
        or parent["usage_uncertain"] is not False
        or set(parent["closed_stages"])
        != (BODY_STAGES if after_capacity_failure else set())
        or parent["capacity_revisions"]
        or set(parent["plan"]["stages"]) != BODY_STAGES | REMAINING_STAGES
    ):
        raise ValueError("summary_recovery_budget_boundary_invalid")
    budget.configure_strict(halt_on_unknown_usage=True)
    if planned_backend(budget) is not None or budget.snapshot()["requests_sent"]:
        raise ValueError("summary_recovery_child_budget_not_empty")
    owner = SummaryRecoveryBudget(parent)
    budget._planned_backend = owner
    if path is not None:
        owner.path = path
        owner._persist()
    return owner


def remaining_exposure(owner):
    """Sum only the still open request slots and enforce their frozen caps."""
    tokens = requests = 0
    for stage in sorted(REMAINING_STAGES):
        spec = owner.plan["stages"][stage]
        entries = owner._stage_entries(stage)
        count = spec["count"] - len(entries)
        reserved = spec["request_cap"] * count
        settled = sum(entry["actual_tokens"] for entry in entries)
        if count < 0 or reserved + settled > spec["reserved_tokens"]:
            raise ValueError("summary_recovery_stage_budget_invalid")
        tokens += reserved
        requests += count
    if (
        tokens + owner.used > owner.plan["maximum_tokens"]
        or requests + owner.snapshot()["requests_sent"] > owner.plan["maximum_requests"]
    ):
        raise ValueError("summary_recovery_total_budget_invalid")
    return {"max_tokens": tokens, "max_requests": requests}


def session_at(case, llm, path, run_id, trace_hook=None, *, capacity_history=None):
    """Reopen the authenticated initial summary under a distinct child run ID."""
    write_json(Path(f"{path}.writer-v2.json"), case.journal)
    session = WriterSession(
        llm,
        path,
        run_id,
        case.packet,
        case.aliases,
        trace_hook=trace_hook,
        **case.session_kwargs(),
    )
    case.validate_session(session)
    if capacity_history is not None:
        journal, repair_state = capacity_history
        write_json(Path(f"{path}.writer-v2.json"), journal)
        write_json(path, {"state": {**repair_state, "run_id": run_id}})
        session = WriterSession(
            llm,
            path,
            run_id,
            case.packet,
            case.aliases,
            trace_hook=trace_hook,
            **case.session_kwargs(),
        )
        if (
            session.state["attempts"] != repair_state["attempts"]
            or session.state["limit"] != repair_state["limit"]
            or any(
                session.journal["calls"].get(name) != receipt
                for name, receipt in case.journal["calls"].items()
            )
        ):
            raise ValueError("summary_recovery_fallback_session_history_invalid")
    session.journal["benchmark_summary_checkpoint"] = True
    session.save()
    return session


def compact_frozen_conclusion(case, session):
    """Keep the opening and limitations paragraphs of the known overlong conclusion."""
    if [issue["field"] for issue in case.summary_issues] != ["conclusion_words"] or len(
        case.summaries["conclusion"]
    ) != 3:
        raise ValueError("summary_recovery_compaction_scope_invalid")
    candidate = copy.deepcopy(case.summaries)
    received, origins = session.reference_input(case.summaries)
    del candidate["conclusion"][1]
    del received["conclusion"][1]
    retained = []
    for origin in origins:
        path = origin["path"]
        if path[:2] == ["conclusion", 1]:
            continue
        if path[:2] == ["conclusion", 2]:
            origin = {**origin, "path": ["conclusion", 1, *path[2:]]}
        retained.append(origin)
    session.remember_reference_input(candidate, received, retained)
    exact_received, _ = session.reference_input(candidate)
    check_writer_envelope(exact_received, session.envelope, stage="summary")
    return candidate


async def check_summary_request(case, parent, *, capacity_history=None):
    """Run the real attempt path against an unreachable transport with no credentials."""
    after_capacity_failure = capacity_history is not None
    budget = CostBudget()
    owner = restore_child_budget(
        budget, parent, after_capacity_failure=after_capacity_failure
    )
    exposure = remaining_exposure(owner)
    with TemporaryDirectory(prefix="survey-summary-preflight-") as temp:
        session = session_at(
            case,
            SimpleNamespace(_cost_budget=budget),
            Path(temp) / "summary.json",
            "preflight",
            capacity_history=capacity_history,
        )
        if after_capacity_failure:
            compact = compact_frozen_conclusion(case, session)
            compile_document(
                case.body,
                compact,
                case.cards,
                case.aliases,
                case.packet["language"],
                case.methodology,
                writing_plan=session.writing_plan,
                envelope=session.envelope,
            )
        measurement = {}

        async def measured_call(name, system, payload, schema, *, stage, **dispatch):
            final_system, final_payload = session.planned_input(system, payload)
            messages = session.messages(final_system, final_payload, schema)
            guidance = (
                "no more than 60 words in each sentence"
                if after_capacity_failure
                else "120–160 English words total"
            )
            if stage == "summary" and not any(
                guidance in msg["content"]
                for msg in messages
                if msg["role"] == "system"
            ):
                raise ValueError("summary_recovery_length_target_missing_on_wire")
            fmt = session.response_format(stage, schema)
            spec = owner.plan["stages"][stage]
            if (
                dispatch.get("max_tokens", spec["output_tokens"])
                != spec["output_tokens"]
            ):
                raise ValueError("summary_recovery_output_budget_mismatch")
            amount = strict_request_reservation(messages, spec["output_tokens"], fmt)
            owner.check_stage_requests(stage, [amount])
            measurement[stage] = {
                "reserved_tokens": amount,
                "request_token_cap": spec["request_cap"],
                "output_tokens": spec["output_tokens"],
            }
            raise CheckedRequest

        session.call = measured_call
        if not after_capacity_failure:
            try:
                await session.repair_initial_summary_length(
                    case.body, case.summaries, "summary_initial", shorter_target=True
                )
            except CheckedRequest:
                pass
        try:
            await session.summaries(
                case.body,
                (
                    CHECKPOINT_SUMMARY_REBUILD
                    if after_capacity_failure
                    else "summary_repair_preflight"
                ),
            )
        except CheckedRequest:
            pass
        stages = {"summary"} if after_capacity_failure else {"repair", "summary"}
        if set(measurement) != stages or (
            owner.used,
            owner.snapshot()["requests_sent"],
        ) != (
            parent["used_tokens"],
            parent["requests_sent"],
        ):
            raise ValueError("summary_recovery_preflight_unexpected_dispatch")
    return {**exposure, **measurement}


def original_sidecars(baseline, run_id):
    """Locate the immutable writer ledger where the selected run executed."""
    paths = [
        folder / "repairs" / f"{run_id}.json.budget-ledger.json"
        for folder in baseline.parent.iterdir()
        if folder.is_dir()
        and (folder / "repairs" / f"{run_id}.json.budget-ledger.json").is_file()
    ]
    if len(paths) != 1:
        raise ValueError("summary_recovery_parent_ledger_ambiguous")
    ledger = paths[0]
    return ledger.with_name(f"{run_id}.json.budget-plan.json"), ledger


def capacity_fallback(path, campaign):
    """Find the one settled length failure in the already approved five cases."""
    if (path / "capacity-continuation-started.json").exists():
        raise ValueError("summary_recovery_fallback_already_started")
    previous = read_campaign(path)
    linked = retry_rows(campaign, previous)
    if (
        set(previous.manifest["retry_replay_slots"]) != TARGETS
        or previous.status != "completed"
        or len(linked) != len(TARGETS)
        or previous.manifest.get("capacity_continuation_of")
        or any(row["usage"]["usage_uncertain"] for row in linked.values())
    ):
        raise ValueError("summary_recovery_fallback_campaign_invalid")
    targets = [
        key
        for key, row in linked.items()
        if row["status"] == "failed"
        and "writer_summary_capacity_exceeded" in row["reasons"]
    ]
    if len(targets) != 1:
        raise ValueError("summary_recovery_fallback_slot_invalid")
    key = targets[0]
    run_id = linked[key]["run_id"]
    row = next(r for r in previous.rows if r.get("run_id") == run_id)
    source = previous.archives[run_id]
    stem = path / "repairs" / f"{run_id}.json"
    journal = json.loads(source.path.with_name("writer-revisions.json").read_bytes())
    if json.loads(Path(f"{stem}.writer-v2.json").read_bytes()) != journal:
        raise ValueError("summary_recovery_fallback_writer_journal_changed")
    state = json.loads(stem.read_bytes())["state"]
    ledger = json.loads(Path(f"{stem}.budget-ledger.json").read_bytes())
    ancestry = {
        "campaign_sha256": previous.digest,
        "run_id": run_id,
        "snapshot_sha256": source.snapshot_sha256,
        "writer_journal_sha256": sha(journal),
        "repair_state_sha256": sha(state),
        "ledger_sha256": sha(ledger),
    }
    return key, (source.read(), row, journal, state, ledger), ancestry, previous, linked


async def prepare(baseline, dataset, slots, config, profiles, fallback_from=None):
    """Authenticate the fixed comparison, all parent writers and future request bounds."""
    resumed = read_resumed_campaign(baseline, slots, dataset, carry_unsettled=True)
    campaign = read_campaign(baseline)
    configuration = build_config_summary(config, "strict_auto_v2")
    if options_for(config).model_dump(mode="json") != resumed.manifest["run_options"]:
        raise ValueError("summary_recovery_run_options_changed")
    if {
        name: profile.fingerprint for name, profile in profiles.items()
    } != resumed.manifest["profiles_fingerprints"]:
        raise ValueError("summary_recovery_profiles_changed")
    fallback = capacity_fallback(fallback_from, campaign) if fallback_from else None
    target_keys = {fallback[0]} if fallback else TARGETS
    targets = [slot for slot in slots if slot.key in target_keys]
    if len(targets) != len(target_keys):
        raise ValueError("summary_recovery_scope_invalid")
    frozen = []
    for slot in targets:
        case = load_frozen_summary_case(campaign, slot.key)
        if (
            configuration["effective"]["llm"]["model_routing"]
            != case.archive["config"]["effective"]["llm"]["model_routing"]
        ):
            raise ValueError("summary_recovery_model_routing_changed")
        plan_path, ledger_path = original_sidecars(baseline, case.row["run_id"])
        parent = checked_parent_ledger(
            case.archive, case.row, case.journal, plan_path, ledger_path
        )
        case.validate_writer_ledger_wire(parent)
        task_graph(case.archive)
        history = None
        if fallback:
            parent = checked_failed_summary_retry(*fallback[1], parent)
            history = fallback[1][2:4]
        measurements = await check_summary_request(
            case, parent, capacity_history=history
        )
        if fallback:
            _, (_, old_row, _, _, _), _, previous, linked = fallback
            prior_slot = previous.manifest["preflight"][slot.key]
            for metric, usage in (
                ("max_requests", "paid_requests"),
                ("max_tokens", "billed_tokens"),
            ):
                if (
                    measurements[metric] + old_row["usage"][usage] > prior_slot[metric]
                    or measurements[metric]
                    + sum(r["usage"][usage] for r in linked.values())
                    > previous.manifest[metric]
                ):
                    raise ValueError(
                        "summary_recovery_fallback_approved_budget_exhausted"
                    )
        reservations = ", ".join(
            f"{stage} {measurements[stage]['reserved_tokens']} tokens"
            for stage in ("repair", "summary")
            if stage in measurements
        )
        print(f"Verified {slot.key}: {reservations}", flush=True)
        frozen.append((slot, case, parent, measurements, history))
    selected = resumed.select_failed_replays([slot.key for slot in targets])
    manifest = {
        **{
            key: resumed.manifest[key]
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
                "per_run_tokens",
                "per_run_requests",
            )
        },
        "runs": [asdict(slot) for slot in slots],
        "source_identity": sha(
            [
                source_identity(),
                *[
                    Path(path).read_text()
                    for path in (
                        __file__,
                        "scripts/survey_summary_recovery_inputs.py",
                        "scripts/survey_summary_recovery_accounting.py",
                        "scripts/view_survey_latency.py",
                    )
                ],
            ]
        ),
        "resumed_campaign": selected.identity,
        "resumed_slots": sorted(selected.observations),
        "retry_replay_slots": list(selected.retry_slots),
        "retry_of": selected.retry_of,
        "timing_scope": "summary_checkpoint",
        "continuation_config": configuration,
        "max_runs": len(frozen),
        "max_tokens": sum(value["max_tokens"] for _, _, _, value, _ in frozen),
        "max_requests": sum(value["max_requests"] for _, _, _, value, _ in frozen),
        "judge_requests": 0,
        "preflight": {
            slot.key: {
                **measurement,
                "writer_journal_sha256": sha(case.journal),
                "parent_budget_sha256": sha(parent),
            }
            for slot, case, parent, measurement, _ in frozen
        },
    }
    if fallback:
        manifest["capacity_continuation_of"] = {fallback[0]: fallback[2]}
    return {**manifest, "approval_sha256": sha(manifest)}, selected, frozen


def summary_capacity_failure(exc):
    """Retry a settled summary length or packaging failure within the same authority."""
    if isinstance(exc, WriterError):
        return exc.code in {
            "writer_summary_capacity_exceeded",
            "writer_patch_capacity_exceeded",
        }
    return isinstance(exc, EnvelopeExceeded) and all(
        d.get("stage") == "summary" for d in exc.diagnostics
    )


async def shorten_summary(session, case, owner):
    """Use up to the three existing local repair slots with an explicit length margin."""
    for attempt in range(3):
        before = owner.snapshot()
        try:
            return await session.repair_initial_summary_length(
                case.body, case.summaries, "summary_initial", shorter_target=True
            )
        except (WriterError, EnvelopeExceeded) as exc:
            after = owner.snapshot()
            sent = after["requests_sent"] - before["requests_sent"]
            if (
                attempt == 2
                or not summary_capacity_failure(exc)
                or owner.stop_reason
                or after["usage_uncertain"]
                or not 0 < sent == after["call_count"] - before["call_count"]
            ):
                raise
    raise ValueError("summary_recovery_attempt_limit_invalid")


async def rebuild_summary(session, case, owner):
    """Restore a verified concise excerpt only if settled regeneration remains too long."""
    before = owner.snapshot()
    try:
        return await session.summaries(case.body, CHECKPOINT_SUMMARY_REBUILD)
    except (WriterError, EnvelopeExceeded) as exc:
        after = owner.snapshot()
        sent = after["requests_sent"] - before["requests_sent"]
        if (
            not summary_capacity_failure(exc)
            or owner.stop_reason
            or after["usage_uncertain"]
            or not 0 < sent == after["call_count"] - before["call_count"]
        ):
            raise
        candidate = compact_frozen_conclusion(case, session)
        session.journal["benchmark_summary_compaction"] = {
            "removed_conclusion_paragraph_index": 1,
            "source_summaries_sha256": sha(case.summaries),
            "removed_paragraph_sha256": sha(case.summaries["conclusion"][1]),
            "candidate_summaries_sha256": sha(candidate),
            "failed_regeneration_requests": sent,
        }
        session.journal.pop("failure", None)
        session.save()
        return candidate


async def continue_case(
    config,
    root,
    case,
    parent,
    recorder,
    run_id,
    input_fingerprint,
    *,
    capacity_parent=None,
    capacity_history=None,
):
    """Run only the recovered summary, production review, Evaluation and delivery."""
    graph = task_graph(case.archive)
    task = graph.get_task("adversarial_review")
    task.input_data = case.task_input
    opts = options_for(config)
    agent = LitAgent(
        config, trace_hook=recorder, run_policy=RunPolicy.isolated_benchmark()
    )
    agent._session_id = run_id
    agent._evaluation_progress_root = root / "evaluations"
    agent._targeted_repair_state_path = root / "repairs" / f"{run_id}.json"
    summary = build_config_summary(config, "strict_auto_v2")
    if (
        summary["effective"]["llm"]["model_routing"]
        != case.archive["config"]["effective"]["llm"]["model_routing"]
    ):
        raise ValueError("summary_recovery_model_routing_changed")
    agent._config_summary = summary
    recorder.set_config_summary(summary)
    recorder("survey.start", {"query": case.packet["query"], "session_id": run_id})
    recorder("external_inputs_ready", {"fingerprint": input_fingerprint})
    recorder.capture_worker_checkpoint(
        graph,
        source_run_id=case.row["run_id"],
        source_snapshot_sha256=case.row["snapshot_sha256"],
    )
    if capacity_parent:
        recorder("benchmark.summary_capacity_continuation", capacity_parent)
    report = None
    error = None
    owner = None
    try:
        async with agent:
            owner = restore_child_budget(
                agent._cost_budget,
                parent,
                Path(f"{agent._targeted_repair_state_path}.budget-ledger.json"),
                after_capacity_failure=capacity_parent is not None,
            )
            recorder("budget.plan", {"budget_plan": owner.plan})
            session = session_at(
                case,
                agent._synthesis._llm,
                agent._targeted_repair_path(),
                run_id,
                recorder,
                capacity_history=capacity_history,
            )
            graph.mark_running(task.task_id)
            recorder(
                "worker.start", {"task_id": task.task_id, "agent_type": task.agent_type}
            )
            async with agent._adversarial._sub_span(
                task.task_id, "synthesis", 1, input_data={"scope": "summary_checkpoint"}
            ) as span:
                summaries = (
                    await rebuild_summary(session, case, owner)
                    if capacity_parent
                    else await shorten_summary(session, case, owner)
                )
                synthesis = finalize_synthesis(
                    session,
                    case.body,
                    summaries,
                    case.cards,
                    case.selection,
                    case.methodology,
                    case.packet["requirements"],
                    verification=task.input_data["verification"],
                )
                span["output"] = synthesis
            reviewed = await agent._adversarial.review_synthesis(task, synthesis)
            graph.mark_done(task.task_id, reviewed)
            recorder(
                "worker.complete",
                {
                    "task_id": task.task_id,
                    "agent_type": task.agent_type,
                    "output": reviewed,
                },
            )
            results = graph.get_results()
            old = case.archive["report"]["metadata"]
            report, degradation = await agent._prepare_report(
                query=case.packet["query"],
                graph=graph,
                results=results,
                options=opts,
                search_outcomes=old["search_source_outcomes"],
                degradation_codes=list(old["degradation_reason_codes"]),
                cancellation=None,
            )
            await agent._automatic_review(report, case.packet["query"], results, None)
            report = await agent._finalize_report(
                query=case.packet["query"],
                report_data=report,
                results=results,
                degradation_codes=degradation,
                cancellation=None,
                allow_trusted_writes=False,
            )
    except BaseException as exc:
        error = str(exc)
        if task.status.value == "running":
            graph.mark_failed(task.task_id, error)
            if owner is not None:
                owner.stop_after_worker_failure()
            recorder(
                "worker.failed",
                {
                    "task_id": task.task_id,
                    "error": error,
                    "error_type": type(exc).__name__,
                },
            )
        if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise
    finally:
        recorder.capture_graph_state(graph)
        accounting = agent.benchmark_accounting()
        if owner is not None:
            recorder("budget.updated", {"budget_state": owner.snapshot()})
            accounting["known_tokens"] -= parent["used_tokens"]
            accounting["requests"] -= parent["requests_sent"]
        elif agent._cost_budget is None:
            accounting = {
                "known_tokens": 0,
                "requests": 0,
                "usage_uncertain": False,
                "stop_reason": None,
            }
        recorder("benchmark.accounting", accounting)
        recorder.finalize(
            report=report,
            error=error,
            terminal_status="completed" if error is None else "failed",
        )
    return owner is not None and owner.approved_exhausted


async def execute(root, manifest, resumed, frozen, slots, dataset, config, profiles):
    """Execute each admitted continuation once, preserving every ledger and result."""
    archives = ArchiveRepository(root / "runs")
    runner = SurveyBenchmarkRunner(
        base_config=config,
        judge_config=load_survey_judge_config(
            Path("benchmarks/survey/judge.example.yaml")
        ),
        artifacts=BenchmarkArtifactRepository(root),
        run_archive=archives,
    )
    observed = dict(resumed.observations)
    remaining_reason = "benchmark_remaining"
    try:
        publish(root, slots, observed, "running", remaining_reason)
        for slot, case, parent, _, history in frozen:
            configured = runner._config_for(
                profiles[slot.profile_id], dataset.dataset_version
            )
            run_id = f"{slot.key}-retry-{uuid4().hex[:12]}"
            recorder = RunRecorder(run_id, case.packet["query"], archives)
            snapshot = resumed.snapshots[slot.case_id]
            capacity_parent = manifest.get("capacity_continuation_of", {}).get(slot.key)
            print(f"Starting summary checkpoint {run_id}", flush=True)
            try:
                async with asyncio.timeout(TIMEOUT_SECONDS):
                    approved_exhausted = await continue_case(
                        configured,
                        root,
                        case,
                        parent,
                        recorder,
                        run_id,
                        snapshot.fingerprint,
                        capacity_parent=capacity_parent,
                        capacity_history=history,
                    )
            finally:
                path = archives.path_for(run_id)
                if path.is_file():
                    row = save_observation(
                        slot,
                        dataset.case_by_id(slot.case_id),
                        path,
                        root,
                        set(dataset.corpus_paper_ids),
                        snapshot=snapshot.fingerprint,
                    )
                    row.update(
                        retry_of=manifest["retry_of"][slot.key],
                        input_snapshot_path=f"inputs/{slot.case_id}.json",
                        timing_scope="summary_checkpoint",
                    )
                    if capacity_parent:
                        row["capacity_continuation_of"] = capacity_parent
                    observed[slot.key] = row
                    publish(root, slots, observed, "running", remaining_reason)
                    print(
                        f"Finished {run_id}: {row['status']}/{row['delivery']}, "
                        f"{row['usage']['paid_requests']} new requests",
                        flush=True,
                    )
            if observed[slot.key]["usage"]["usage_uncertain"]:
                remaining_reason = "benchmark_usage_unknown"
                break
            if approved_exhausted:
                remaining_reason = "benchmark_budget_exhausted"
                break
    finally:
        publish(
            root,
            slots,
            observed,
            "completed" if len(observed) == len(slots) else "partial",
            remaining_reason,
        )


async def main():
    """Seal a no-send preflight or run only the explicitly approved manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fallback-from",
        type=Path,
        help="Rebuild one capacity failure within the approved batch",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--approve-manifest")
    args = parser.parse_args()
    config = load_config("config/agent-format-repair.yaml")
    config.eval.benchmark_faithfulness_policy = BENCHMARK_FAITHFULNESS_POLICY
    config.rag.writeback_enabled = False
    config.api.max_concurrent_surveys = 1
    dataset = load_survey_dataset(DATASET)
    profiles = {
        profile.profile_id: profile for profile in load_survey_profiles(PROFILES_PATH)
    }
    slots = run_slots()
    root = args.output.resolve()
    if root == args.baseline.resolve() or (root / "started.json").exists():
        parser.error(
            "summary continuation needs a separate output directory that has not started"
        )
    manifest, resumed, frozen = await prepare(
        args.baseline.resolve(), dataset, slots, config, profiles, args.fallback_from
    )
    path = root / "manifest.json"
    if args.dry_run:
        resumed.materialize(root)
        write_json(path, manifest)
        publish(
            root, slots, resumed.observations, "dry-run", "awaiting_confirmed_budget"
        )
        print(
            path,
            manifest["approval_sha256"],
            manifest["max_tokens"],
            manifest["max_requests"],
            flush=True,
        )
        return
    if (
        json.loads(path.read_bytes()) != manifest
        or args.approve_manifest != manifest["approval_sha256"]
    ):
        parser.error("approved manifest or effective sources changed")
    if args.fallback_from is not None:
        claim = args.fallback_from / "capacity-continuation-started.json"
        with claim.open("x", encoding="utf-8") as receipt:
            json.dump(
                {"approval_sha256": manifest["approval_sha256"], "output": str(root)},
                receipt,
            )
    with (root / "started.json").open("x", encoding="utf-8") as receipt:
        json.dump({"approval_sha256": manifest["approval_sha256"]}, receipt)
    resumed.materialize(root)
    await execute(root, manifest, resumed, frozen, slots, dataset, config, profiles)


if __name__ == "__main__":
    asyncio.run(main())
