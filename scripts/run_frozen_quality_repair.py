"""Run one explicitly authorized repair from an immutable completed survey."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
from pathlib import Path

from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import ReportAuditService, atomic_json, digest
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget

ORIGIN = "https://ark.cn-beijing.volces.com/api/v3"
MODEL = "doubao-seed-2-0-mini-260428"
VERSION = "frozen-repair-trial-v1"


class TrialBudget(CostBudget):
    """Retain prior charges and impose an additional transport-count ceiling."""

    def __init__(
        self,
        snapshot: dict | None = None,
        new_requests: int = 7,
        *,
        max_tokens: int = 500000,
        warn_ratio: float = 0.8,
    ):
        """Restore an intact completed source without refunding historical use."""
        if type(new_requests) is not int or not 1 <= new_requests <= 7:
            raise ValueError("frozen_trial_request_limit_invalid")
        super().__init__(max_tokens=max_tokens, warn_ratio=warn_ratio)
        self.configure_strict(halt_on_unknown_usage=True)
        if snapshot is not None:
            self.restore(snapshot)
        self.initial_requests = self.snapshot()["requests_sent"]
        self.new_requests = new_requests

    def begin_request(self) -> None:
        """Refuse a new attempt before sending if either request ceiling is spent."""
        if (
            self.snapshot()["requests_sent"] - self.initial_requests
            >= self.new_requests
        ):
            raise ValueError("strict_request_budget_exceeded")
        super().begin_request()


def _record_completed_usage(
    artifact: dict, budget: CostBudget, *, trial=False, operation_ids=None
) -> None:
    """Reconstruct only matched, known provider usage in the actual task phase."""
    opened, completed = {}, set()
    charged = 0
    for event in artifact["events"]:
        if event["event"] == "llm.start":
            op, task = event["data"].get("operation_id"), event["data"].get("task_id")
            if (
                not isinstance(op, str)
                or op in opened
                or not isinstance(task, str)
                or operation_ids is not None
                and op in operation_ids
            ):
                raise ValueError("frozen_trial_source_lifecycle_invalid")
            opened[op] = task
        elif event["event"] == "llm.failed":
            raise ValueError("frozen_trial_source_usage_unknown")
        elif event["event"] == "llm.complete":
            data = event["data"]
            op, task = data.get("operation_id"), data.get("task_id")
            if op not in opened or op in completed or task != opened[op]:
                raise ValueError("frozen_trial_source_lifecycle_invalid")
            completed.add(op)
            if data.get("usage_available") is not True or any(
                type(data.get(key)) is not int or data[key] < 0
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                raise ValueError("frozen_trial_source_usage_unknown")
            if (
                data["total_tokens"]
                != data["prompt_tokens"] + data["completion_tokens"]
            ):
                raise ValueError("frozen_trial_source_usage_mismatch")
            if trial and not (
                task == "evidence_rewrite" or task.startswith("evaluation")
            ):
                raise ValueError("frozen_trial_parent_phase_invalid")
            token = set_task_id(task)
            try:
                budget.begin_request()
                budget.record(data)
            finally:
                reset_task_id(token)
            charged += data["total_tokens"]
    if (
        set(opened) != completed
        or type(artifact["usage"].get("total_tokens")) is not int
        or artifact["usage"].get("total_tokens") != charged
    ):
        raise ValueError("frozen_trial_source_usage_mismatch")
    if operation_ids is not None:
        operation_ids.update(completed)


def _inherit_parent_budget(
    root: Path,
    source: dict,
    source_sha: str,
    budget: CostBudget,
    seen=None,
    operation_ids=None,
) -> dict:
    """Validate immutable parent receipts before extending their cumulative ledger."""
    root = root.resolve()
    seen = set() if seen is None else seen
    operation_ids = (
        {
            e["data"]["operation_id"]
            for e in source["events"]
            if e["event"] == "llm.start"
        }
        if operation_ids is None
        else operation_ids
    )
    if root in seen:
        raise ValueError("frozen_trial_parent_cycle")
    seen.add(root)
    state_path, input_path = root / "checkpoint.json", root / "input-snapshot.json"
    state_raw, input_raw = state_path.read_bytes(), input_path.read_bytes()
    state, inputs = json.loads(state_raw), json.loads(input_raw)
    plan = state["plan"]
    expected_binding = digest({k: v for k, v in plan.items() if k != "binding"})
    if (
        state.get("status") != "blocked"
        or state.get("usage_complete") is not True
        or state.get("reason") == "committed"
        or state.get("binding") != expected_binding
        or plan.get("binding") != expected_binding
        or plan.get("version") != VERSION
        or plan.get("source_artifact_sha256") != source_sha
        or plan.get("source_run_id") != source["run_id"]
        or plan.get("source_report_sha256")
        != hashlib.sha256(source["report"]["survey"].encode()).hexdigest()
        or digest(inputs) != digest(source)
        or re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(plan.get("run_id"))) is None
    ):
        raise ValueError("frozen_trial_parent_binding_invalid")
    if plan.get("parent") is not None:
        receipt = _inherit_parent_budget(
            Path(plan["parent"]["root"]),
            source,
            source_sha,
            budget,
            seen,
            operation_ids,
        )
        if receipt != plan["parent"]:
            raise ValueError("frozen_trial_parent_receipt_changed")
    if digest(budget.snapshot()) != digest(plan["source_budget"]):
        raise ValueError("frozen_trial_parent_budget_reset")
    run_path = root / "runs" / f"{plan['run_id']}.json"
    run_raw = run_path.read_bytes()
    record = json.loads(run_raw)
    if (
        record.get("run_id") != plan["run_id"]
        or record.get("status") != "completed"
        or record.get("query") != source["query"]
        or record.get("report", {}).get("survey") != source["report"]["survey"]
        or record.get("report", {}).get("delivery", {}).get("publishable") is not False
        or digest(state.get("report")) != digest(record.get("report"))
    ):
        raise ValueError("frozen_trial_parent_report_invalid")
    before = budget.snapshot()
    _record_completed_usage(record, budget, trial=True, operation_ids=operation_ids)
    requests = budget.snapshot()["requests_sent"] - before["requests_sent"]
    tokens = budget.used - before["used_tokens"]
    if (
        digest(budget.snapshot()) != digest(state["budget"])
        or type(state.get("new_survey_requests")) is not int
        or type(state.get("new_survey_tokens")) is not int
        or state.get("new_survey_requests") != requests
        or state.get("new_survey_tokens") != tokens
        or requests > plan["limits"]["new_survey_requests"]
        or tokens > plan["limits"]["new_survey_tokens"]
        or budget.stop_reason is not None
        or budget.snapshot()["usage_uncertain"]
    ):
        raise ValueError("frozen_trial_parent_usage_mismatch")
    return {
        "root": str(root),
        "run_id": plan["run_id"],
        "binding": expected_binding,
        "hashes": {
            str(p): hashlib.sha256(raw).hexdigest()
            for p, raw in (
                (state_path, state_raw),
                (input_path, input_raw),
                (run_path, run_raw),
            )
        },
    }


def _frozen_repair_scope(artifact: dict, config) -> dict | None:
    """Reuse recorded chapter ranks within the exact frozen card/source pool."""
    from litagent.context.evidence_selector import (
        DEFAULT_SECTIONS,
        SHARED_SCOPE_CONTRACT,
    )
    from litagent.quality import validate_fact_snapshot

    records = [
        e["data"]
        for e in artifact.get("events", [])
        if e["event"] == "evidence.retrieve.complete"
    ]
    if not records:
        return None
    if len(records) != 1:
        raise ValueError("frozen_trial_scope_producer_ambiguous")
    record = records[0]
    summary = next(
        (
            n["output"].get("evidence_selection")
            for n in artifact["nodes"].values()
            if n.get("kind") == "worker" and n.get("task_id") == "adversarial_review"
        ),
        None,
    )
    metadata = artifact["report"]["metadata"]
    cards = metadata["fact_cards"]
    pool = {c["evidence_id"] for c in cards.values()}
    core = {s.key for s in DEFAULT_SECTIONS}
    if (
        not isinstance(summary, dict)
        or not summary.get("valid")
        or set(record.get("selected_items", {})) != pool
        or not isinstance(summary.get("section_evidence_ids"), dict)
        or not isinstance(record.get("section_evidence_ids"), dict)
        or set(record["section_evidence_ids"]) != core
        or any(
            record["section_evidence_ids"][key]
            != summary["section_evidence_ids"].get(key)
            for key in core
        )
        or summary.get("selected_count") != len(pool)
        or not isinstance(record.get("ranking_decisions"), list)
    ):
        raise ValueError("frozen_trial_scope_binding_invalid")
    validate_fact_snapshot(
        cards, record["selected_items"], metadata["evidence_snapshot_sha256"]
    )
    rows = {key: [] for key in core}
    seen = set()
    seen_ids = set()
    for row in record["ranking_decisions"]:
        if row.get("reason_code") == "invalid_entry" and "section" not in row:
            continue
        key, rank, eid = row.get("section"), row.get("rank"), row.get("evidence_id")
        if (
            key not in core
            or type(rank) is not int
            or rank <= 0
            or not isinstance(eid, str)
            or (key, rank) in seen
            or (key, eid) in seen_ids
        ):
            raise ValueError("frozen_trial_scope_rank_invalid")
        seen.add((key, rank))
        seen_ids.add((key, eid))
        # Unselected candidates remain unavailable, even if their old rank was high.
        if eid in pool:
            rows[key].append((rank, eid))
    if any(not values for values in rows.values()):
        raise ValueError("frozen_trial_scope_rank_incomplete")
    updated = copy.deepcopy(summary)
    for key, values in rows.items():
        updated["section_evidence_ids"][key] = [
            eid
            for _, eid in sorted(values)[: config.context.evidence_top_k_per_section]
        ]
    return {
        "contract": SHARED_SCOPE_CONTRACT,
        "selection": updated,
        "selected_source_count": len(pool),
        "producer_sha256": digest(record),
        "original_scope_sha256": digest(summary),
        "updated_scope_sha256": digest(updated),
    }


def prepare(
    source: Path, config, run_id: str, *, parent_root: Path | None = None
) -> dict:
    """Freeze source, configuration, code and exact known usage for one trial."""
    if re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_id) is None:
        raise ValueError("frozen_trial_id_invalid")
    if config.llm.base_url.rstrip("/") != ORIGIN or config.llm.model != MODEL:
        raise ValueError("frozen_trial_provider_invalid")
    if config.safety.max_cost_tokens != 500000:
        raise ValueError("frozen_trial_budget_invalid")
    raw = source.read_bytes()
    artifact = json.loads(raw)
    if artifact["run_id"] == run_id:
        raise ValueError("frozen_trial_requires_new_identity")
    report = artifact["report"]
    if (
        artifact["status"] != "completed"
        or report["metadata"].get("quality_policy") != "strict_auto_v1"
        or not LitAgent._completed_bound_repair(report["evaluation"], report["survey"])
    ):
        raise ValueError("frozen_trial_source_not_complete_bound")
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    _record_completed_usage(artifact, budget)
    snapshot = budget.snapshot()
    snapshot["strict_phase"] = 2
    budget.restore(snapshot)
    parent = None
    if parent_root is not None:
        parent = _inherit_parent_budget(
            parent_root, artifact, hashlib.sha256(raw).hexdigest(), budget
        )
        if parent["run_id"] == run_id:
            raise ValueError("frozen_trial_requires_new_identity")
        snapshot = budget.snapshot()
    remaining_requests = min(7, 64 - snapshot["requests_sent"])
    if budget.used >= 500000 or remaining_requests <= 0:
        raise ValueError("frozen_trial_source_allowance_spent")
    if parent is not None and remaining_requests < 4:
        raise ValueError("frozen_trial_complete_evaluation_capacity_unavailable")
    config = config.model_copy(deep=True)
    config.eval.strict_max_batches = min(4, remaining_requests - 3) if parent else 4
    config.eval.strict_max_claims = config.eval.strict_max_batches * 16
    files = sorted((Path(__file__).parents[1] / "src/litagent").rglob("*.py"))
    files.append(Path(__file__))
    source_hashes = {
        str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
    }
    plan = {
        "version": VERSION,
        "run_id": run_id,
        "source_run_id": artifact["run_id"],
        "source_artifact_sha256": hashlib.sha256(raw).hexdigest(),
        "source_report_sha256": hashlib.sha256(report["survey"].encode()).hexdigest(),
        "source_budget": snapshot,
        "limits": {
            "new_survey_tokens": 500000 - budget.used,
            "new_survey_requests": remaining_requests,
            "judge_total_tokens": 100000,
            "judge_requests": 2,
            "jev_input_tokens": 200000,
            "jev_requests": 20,
        },
        "config": build_config_summary(config, "strict_auto_v1"),
        "source_hashes": source_hashes,
        "formal_benchmark": False,
        "retries": 0,
    }
    if parent is not None:
        plan["parent"] = parent
        repair_scope = _frozen_repair_scope(artifact, config)
        if repair_scope is not None:
            plan["repair_scope"] = repair_scope
    plan["binding"] = digest(plan)
    return {"plan": plan, "artifact": artifact, "config": config}


async def run_trial(
    source: Path,
    config,
    *,
    root: Path,
    run_id: str,
    live: bool = False,
    parent_root: Path | None = None,
    authorization: dict | None = None,
    client_factory=None,
    service_factory=None,
    agent_factory=None,
    trace_factory=None,
) -> dict:
    """Execute once, keep failures and never replay an interrupted attempt."""
    prepared = prepare(source, config, run_id, parent_root=parent_root)
    plan, artifact, config = (prepared[k] for k in ("plan", "artifact", "config"))
    if not live:
        return {"status": "planned", "plan": plan, "paid_calls": 0}
    if (
        not authorization
        or authorization.get("actor") != "user"
        or authorization.get("binding") != plan["binding"]
    ):
        raise ValueError("frozen_trial_authorization_required")
    root.mkdir(parents=True, exist_ok=True)
    path = root / "checkpoint.json"
    with suite_lock(root / ".trial.lock"):
        if path.exists():
            state = json.loads(path.read_text())
            if state["binding"] != plan["binding"]:
                raise ValueError("frozen_trial_identity_conflict")
            if state["status"] == "started_no_replay":
                state.update(
                    status="interrupted", reason="frozen_trial_interrupted_no_replay"
                )
                atomic_json(path, state)
            return state
        if (
            any(
                (root / name).exists()
                for name in ("candidate.json", "input-snapshot.json")
            )
            or (root / "runs" / f"{run_id}.json").exists()
            or (root / "audits" / "index.sqlite3").exists()
        ):
            raise ValueError("frozen_trial_output_collision")
        budget = TrialBudget(
            plan["source_budget"], plan["limits"]["new_survey_requests"]
        )
        state = {
            "binding": plan["binding"],
            "plan": plan,
            "status": "started_no_replay",
            "budget": budget.snapshot(),
        }
        atomic_json(path, state)
        atomic_json(root / "input-snapshot.json", artifact)
        client = service = tracer = recorder = hook = None
        report = None
        try:
            recorder = RunRecorder(
                run_id, artifact["query"], ArchiveRepository(root / "runs")
            )
            recorder.set_config_summary(plan["config"])
            if trace_factory:
                tracer = trace_factory(run_id)
            elif config.observability.enabled and client_factory is None:
                tracer = LangFuseTracer(
                    config.observability.langfuse_host,
                    os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                    os.getenv("LANGFUSE_SECRET_KEY", ""),
                    trace_seed=run_id,
                )
            hook = CompositeTraceHook(
                recorder,
                (
                    RedactingTraceHook(tracer, payload_mode="metadata_only")
                    if tracer
                    else None
                ),
            )
            hook(
                "survey.start",
                {
                    "session_id": run_id,
                    "query": artifact["query"],
                    "quality_policy": "strict_auto_v1",
                },
            )
            factory = client_factory or OpenAICompatibleClient
            client = factory(
                base_url=ORIGIN,
                model=MODEL,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                cost_budget=budget,
                trace_hook=hook,
                transport_max_attempts=1,
            )
            service = (service_factory or ReportAuditService)(
                config, root=root / "audits"
            )
            capabilities = service.capabilities()
            if not capabilities.get("judge_available") or not capabilities.get(
                "jev_available"
            ):
                raise ValueError("frozen_trial_audit_capability_missing")
            judge = service.judge_config()
            if judge.base_url.rstrip("/") != ORIGIN or judge.model != MODEL:
                raise ValueError("frozen_trial_judge_provider_invalid")
            agent = (agent_factory or LitAgent)(
                config, trace_hook=hook, audit_service=service
            )
            agent._session_id = run_id
            agent._llm = client
            agent._cost_budget = budget
            agent._synthesis = SynthesisWorker(client)
            agent._evaluators = [
                CitationEvaluator(client),
                ConsistencyEvaluator(client),
                RagasFaithfulnessEvaluator(config, llm=client),
            ]
            original = artifact["report"]
            metadata = {
                k: copy.deepcopy(original["metadata"][k])
                for k in (
                    "query",
                    "quality_policy",
                    "report_language",
                    "review_structure",
                    "review_methodology",
                    "fact_cards",
                    "evidence_snapshot_sha256",
                    "coverage_requirements",
                    "coverage_matrix",
                )
                if k in original["metadata"]
            }
            metadata.update(
                config_summary=plan["config"],
                frozen_repair_provenance={
                    "run_id": artifact["run_id"],
                    "report_sha256": plan["source_report_sha256"],
                    "artifact_sha256": plan["source_artifact_sha256"],
                },
                execution_scope="frozen_repair_trial",
            )
            report = {
                "schema_version": 1,
                "survey": original["survey"],
                "metadata": metadata,
                "evaluation": {},
                "quality": {"status": "unverified"},
                "delivery": {"status": "blocked", "publishable": False},
                "partial": False,
                "graph_data": copy.deepcopy(original.get("graph_data", {})),
                "review_history": [],
            }
            results = {
                n["task_id"]: copy.deepcopy(n["output"])
                for n in artifact["nodes"].values()
                if n["kind"] == "worker"
            }
            if plan.get("repair_scope"):
                results["adversarial_review"]["evidence_selection"] = copy.deepcopy(
                    plan["repair_scope"]["selection"]
                )
                report["metadata"]["repair_scope_provenance"] = {
                    k: v for k, v in plan["repair_scope"].items() if k != "selection"
                }
            outcome = await agent._attempt_evidence_rewrite(
                query=artifact["query"],
                report_data=report,
                results=results,
                initial_evaluation=original["evaluation"],
                initial_quality=original["quality"],
                report_language=metadata["report_language"],
            )
            atomic_json(
                root / "candidate.json",
                {
                    "committed": outcome.committed,
                    "candidate": outcome.candidate,
                    "evaluation": outcome.candidate_evaluation,
                    "quality": outcome.candidate_quality,
                    "reason": outcome.reason_code,
                },
            )
            if outcome.committed:
                await agent._automatic_review(report, artifact["query"], results, None)
            state.update(
                status="completed" if report["delivery"]["publishable"] else "blocked",
                reason=outcome.reason_code,
                report=report,
            )
        except BaseException as exc:
            state.update(
                status=(
                    "interrupted"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed"
                ),
                reason=f"frozen_trial_error:{type(exc).__name__}",
            )
            if report is not None:
                report["metadata"]["trial_execution_error"] = state["reason"]
                report["delivery"] = {"status": "blocked", "publishable": False}
            if isinstance(exc, (asyncio.CancelledError, KeyboardInterrupt)):
                raise
        finally:
            state["budget"] = budget.snapshot()
            state["new_survey_requests"] = (
                budget.snapshot()["requests_sent"]
                - plan["source_budget"]["requests_sent"]
            )
            state["new_survey_tokens"] = (
                budget.used - plan["source_budget"]["used_tokens"]
            )
            state["usage_complete"] = not budget.snapshot()["usage_uncertain"]
            atomic_json(path, state)
            if recorder:
                failed_execution = state["status"] in {"failed", "interrupted"}
                if hook:
                    hook(
                        "survey.error" if failed_execution else "survey.complete",
                        {
                            "session_id": run_id,
                            "status": state["status"],
                            "quality_status": (report or {})
                            .get("quality", {})
                            .get("status", "unverified"),
                            "delivery_status": (report or {})
                            .get("delivery", {})
                            .get("status", "blocked"),
                            "accepted": state["status"] == "completed",
                            "total_tokens": state["new_survey_tokens"],
                            "reason_code": state.get("reason"),
                            **(
                                {"error": state.get("reason")}
                                if failed_execution
                                else {}
                            ),
                        },
                    )
                if tracer:
                    recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
                    tracer.flush()
                recorder.finalize(
                    report=report,
                    terminal_status="failed" if failed_execution else "completed",
                    error=state.get("reason") if failed_execution else None,
                )
            if service:
                await service.shutdown()
            if client:
                await client.close()
        return state


def main() -> None:
    """Require an explicit bound authorization before enabling live execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--parent-root", type=Path)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv

    load_dotenv(".env")
    auth = json.loads(args.authorization.read_text()) if args.authorization else None
    state = asyncio.run(
        run_trial(
            args.source,
            load_config(),
            root=args.root,
            run_id=args.run_id,
            live=args.live,
            parent_root=args.parent_root,
            authorization=auth,
        )
    )
    print(
        json.dumps(
            (
                state
                if not args.live
                else {k: v for k, v in state.items() if k not in {"report", "plan"}}
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
