"""Evaluate an immutable report once using the remaining diagnostic allowance."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path

from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.contracts import build_config_summary, normalize_survey_result
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions, VerificationOptions
from scripts.run_frozen_generation import MODEL, ORIGIN, file_sha
from scripts.run_inventory_diagnostic import file_sha_text


class EvaluationClient(OpenAICompatibleClient):
    """Keep the experiment's smaller request cap at the actual transport boundary."""

    async def chat(self, messages, **kwargs):
        """Count diagnostic and evaluation requests against one persisted allowance."""
        if self._cost_budget.snapshot()["requests_sent"] >= 13:
            raise ValueError("evaluation_diagnostic_request_limit")
        response = await super().chat(messages, **kwargs)
        if response.model != MODEL:
            self._cost_budget.stop_after_worker_failure()
            raise ValueError("evaluation_actual_model_invalid")
        return response


def verify_frozen_sources(root: Path, plan: dict) -> None:
    """Check prior readable code instead of requiring current code to be unchanged."""
    for name, expected in plan["hashes"].items():
        original = Path(name)
        try:
            relative = original.relative_to(Path.cwd())
        except ValueError:
            relative = None
        frozen = root / "frozen-code" / relative if relative else None
        checked = frozen if frozen and frozen.is_file() else original
        if file_sha(checked) != expected:
            raise ValueError("frozen_evaluation_parent_source_changed")


def continuation_budget(prior: Path, payload: dict, diagnostic_budget: dict) -> dict:
    """Reconcile a completed evaluation with its real request events before reuse."""
    plan = json.loads((prior / "manifest.json").read_text())
    state = json.loads((prior / "checkpoint.json").read_text())
    if (
        digest({k: v for k, v in plan.items() if k != "binding"}) != plan["binding"]
        or state["binding"] != plan["binding"]
        or state["status"] != "completed"
        or state.get("usage_complete") is not True
        or state.get("pending")
        or plan["max_tokens"] != 200000
        or plan["max_requests"] != 13
        or plan["source_budget"] != diagnostic_budget
        or any(
            plan.get(key) is not False
            for key in ("generation", "search", "repair", "judge", "jev")
        )
        or json.loads((prior / "input.json").read_text()) != payload
        or state["report"]["survey"] != payload["report"]["survey"]
    ):
        raise ValueError("frozen_evaluation_continuation_invalid")
    verify_frozen_sources(prior, plan)
    budget = CostBudget.from_snapshot(state["budget"])
    after, before = budget.snapshot(), plan["source_budget"]
    if (
        not budget.strict
        or budget.stop_reason
        or after["usage_uncertain"]
        or after["max_tokens"] != 200000
        or not 0 <= after["used_tokens"] < 200000
        or not before["requests_sent"] < after["requests_sent"] < 13
    ):
        raise ValueError("frozen_evaluation_continuation_budget_invalid")
    events = [
        json.loads(line) for line in (prior / "events.jsonl").read_text().splitlines()
    ]
    starts = [e["data"] for e in events if e["event"] == "llm.start"]
    completed = [e["data"] for e in events if e["event"] == "llm.complete"]
    start_ids = [e.get("operation_id") for e in starts]
    complete_ids = [e.get("operation_id") for e in completed]
    if (
        any(e["event"] == "llm.failed" for e in events)
        or not all(isinstance(i, str) and i for i in start_ids)
        or not all(isinstance(i, str) and i for i in complete_ids)
        or len(set(start_ids)) != len(start_ids)
        or sorted(start_ids) != sorted(complete_ids)
        or after["requests_sent"] - before["requests_sent"] != len(starts)
        or after["call_count"] - before["call_count"] != len(completed)
        or any(
            e.get("usage_available") is not True
            or e.get("actual_model") != MODEL
            or e.get("finish_reason") != "stop"
            or any(
                type(e.get(k)) is not int or e[k] < 0
                for k in ("prompt_tokens", "completion_tokens")
            )
            for e in completed
        )
        or after["used_tokens"] - before["used_tokens"]
        != sum(e["prompt_tokens"] + e["completion_tokens"] for e in completed)
    ):
        raise ValueError("frozen_evaluation_continuation_usage_invalid")
    return after


def prepare(
    root: Path,
    diagnostic: Path,
    config,
    *,
    archive_root=Path("artifacts/runs"),
    prior_evaluation: Path | None = None,
) -> dict:
    """Bind the finished diagnostic, original report and new evaluation code."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() or (root / "checkpoint.json").exists():
        raise ValueError("frozen_evaluation_identity_exists")
    parent_plan = json.loads((diagnostic / "manifest.json").read_text())
    parent = json.loads((diagnostic / "checkpoint.json").read_text())
    if (
        digest({k: v for k, v in parent_plan.items() if k != "binding"})
        != parent_plan["binding"]
        or parent["binding"] != parent_plan["binding"]
        or parent["status"] != "completed"
        or parent.get("pending")
        or not parent["flat_full_inventory_complete"]
        or parent["budget"]["usage_uncertain"]
        or parent["budget"]["max_tokens"] != 200000
        or parent["budget"]["requests_sent"] != 3
        or config.llm.base_url.rstrip("/") != ORIGIN
        or not config.llm.uses_single_model(MODEL)
    ):
        raise ValueError("frozen_evaluation_parent_invalid")
    verify_frozen_sources(diagnostic, parent_plan)
    source = Path(parent_plan["source"])
    inputs = json.loads((source / "vit-baseline/input-snapshot.json").read_text())
    source_case = json.loads((source / "vit-baseline/checkpoint.json").read_text())
    source_archive = archive_root / f"{source.name}-vit-baseline.json"
    archive = json.loads(source_archive.read_text())
    if (
        archive["report"] != source_case["report"]
        or file_sha_text(archive["report"]["survey"])
        != parent_plan["source_report_sha256"]
    ):
        raise ValueError("frozen_evaluation_report_changed")
    results = {
        n["task_id"]: copy.deepcopy(n["output"])
        for n in archive["nodes"].values()
        if n["kind"] == "worker"
    }
    metadata = archive["report"]["metadata"]
    for key in (
        "quality_policy",
        "body_quality",
        "review_structure",
        "report_language",
        "review_methodology",
        "fact_cards",
        "evidence_snapshot_sha256",
        "coverage_requirements",
        "writing_bindings",
    ):
        results["adversarial_review"][key] = copy.deepcopy(metadata[key])
    payload = {
        "report": archive["report"],
        "results": results,
        "query": inputs["question"],
    }
    source_budget = parent["budget"]
    continuation = None
    if prior_evaluation is not None:
        source_budget = continuation_budget(prior_evaluation, payload, source_budget)
        prior_plan = json.loads((prior_evaluation / "manifest.json").read_text())
        if (
            prior_plan["config"] != build_config_summary(config, "strict_auto_v2")
            or prior_plan["source_report_sha256"] != parent_plan["source_report_sha256"]
            or prior_plan["historical_before_diagnostic"] != parent_plan["historical"]
        ):
            raise ValueError("frozen_evaluation_continuation_identity_invalid")
        continuation = {
            "parent": str(prior_evaluation.resolve()),
            "parent_binding": prior_plan["binding"],
            "receipt": str(
                prior_evaluation.with_name(
                    prior_evaluation.name + "-successor.json"
                ).resolve()
            ),
        }
    atomic_json(root / "input.json", payload)
    hashes = {
        str(p.resolve()): file_sha(p)
        for p in [
            diagnostic / "manifest.json",
            diagnostic / "checkpoint.json",
            source_archive,
            root / "input.json",
        ]
    }
    if prior_evaluation is not None:
        hashes.update(
            {
                str((prior_evaluation / name).resolve()): file_sha(
                    prior_evaluation / name
                )
                for name in (
                    "manifest.json",
                    "checkpoint.json",
                    "events.jsonl",
                    "input.json",
                )
            }
        )
    for path in [*sorted(Path("src/litagent").rglob("*.py")), Path(__file__)]:
        hashes[str(path.resolve())] = file_sha(path)
        target = root / "frozen-code" / path.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        verification=VerificationOptions(),
    )
    plan = {
        "version": "frozen-evaluation-only-v1",
        "run_id": root.name,
        "source_report_sha256": parent_plan["source_report_sha256"],
        "source_run_id": archive["run_id"],
        "config": build_config_summary(config, "strict_auto_v2"),
        "run_options_fingerprint": options.fingerprint(),
        "source_budget": source_budget,
        "diagnostic_budget": parent["budget"],
        "historical_before_diagnostic": parent_plan["historical"],
        "hashes": hashes,
        "max_tokens": 200000,
        "max_requests": 13,
        "retries": 0,
        "generation": False,
        "search": False,
        "repair": False,
        "judge": False,
        "jev": False,
        **({"continuation": continuation} if continuation else {}),
    }
    plan["binding"] = digest(plan)
    if continuation:
        # An exclusive sibling receipt prevents two attempts spending the same remainder.
        # It stays outside the immutable parent and is never automatically released.
        receipt = {
            "root": str(root.resolve()),
            "binding": plan["binding"],
            "parent_binding": continuation["parent_binding"],
        }
        try:
            with Path(continuation["receipt"]).open("x", encoding="utf-8") as stream:
                json.dump(receipt, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
        except FileExistsError as exc:
            raise ValueError("frozen_evaluation_successor_already_reserved") from exc
    atomic_json(root / "manifest.json", plan)
    return plan


async def run(
    root: Path, config, *, live=False, client_factory=None, repository=None
) -> dict:
    """Run the existing evaluator and automatic verdict without invoking writing."""
    plan = json.loads((root / "manifest.json").read_text())
    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        verification=VerificationOptions(),
    )
    if (
        digest({k: v for k, v in plan.items() if k != "binding"}) != plan["binding"]
        or options.fingerprint() != plan["run_options_fingerprint"]
        or build_config_summary(config, "strict_auto_v2") != plan["config"]
        or any(file_sha(Path(p)) != h for p, h in plan["hashes"].items())
    ):
        raise ValueError("frozen_evaluation_identity_changed")
    if continuation := plan.get("continuation"):
        if json.loads(Path(continuation["receipt"]).read_text()) != {
            "root": str(root.resolve()),
            "binding": plan["binding"],
            "parent_binding": continuation["parent_binding"],
        }:
            raise ValueError("frozen_evaluation_successor_receipt_invalid")
    if not live:
        return {
            "status": "planned",
            "paid_calls": 0,
            "binding": plan["binding"],
            "source_budget": plan["source_budget"],
        }
    with suite_lock(root / ".evaluation.lock"):
        cp = root / "checkpoint.json"
        if cp.exists():
            state = json.loads(cp.read_text())
            if state["binding"] != plan["binding"]:
                raise ValueError("frozen_evaluation_checkpoint_conflict")
            if state["status"] == "started_no_replay":
                # Hold the entire remaining allowance when the last outcome is unknown.
                state.update(
                    status="interrupted_no_replay",
                    conservative_tokens=200000,
                    conservative_requests=13,
                    usage_complete=False,
                )
                atomic_json(cp, state)
            return state
        repo = repository if repository is not None else ArchiveRepository()
        if repo.path_for(plan["run_id"]).exists():
            raise ValueError("frozen_evaluation_archive_exists")
        payload = json.loads((root / "input.json").read_text())
        budget = CostBudget.from_snapshot(plan["source_budget"])
        state = {
            "binding": plan["binding"],
            "status": "started_no_replay",
            "budget": budget.snapshot(),
        }
        atomic_json(cp, state)
        recorder = RunRecorder(plan["run_id"], payload["query"], repo)
        recorder.set_config_summary(plan["config"])
        tracer = None
        if client_factory is None and config.observability.enabled:
            tracer = LangFuseTracer(
                config.observability.langfuse_host,
                os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=plan["run_id"],
            )

        def journal(event, data):
            """Persist complete local events before forwarding metadata-only traces."""
            with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"event": event, "data": data}, ensure_ascii=False, default=str
                    )
                    + "\n"
                )

        hook = CompositeTraceHook(
            recorder,
            journal,
            RedactingTraceHook(tracer, "metadata_only") if tracer else None,
        )
        client = (
            client_factory(budget, hook)
            if client_factory
            else EvaluationClient(
                ORIGIN,
                MODEL,
                cost_budget=budget,
                trace_hook=hook,
                transport_max_attempts=1,
            )
        )
        agent = LitAgent(
            config, trace_hook=hook, run_policy=RunPolicy.isolated_benchmark()
        )
        agent._session_id, agent._llm, agent._cost_budget = (
            plan["run_id"],
            client,
            budget,
        )
        agent._config_summary = plan["config"]
        agent._evaluators = [
            CitationEvaluator(client),
            ConsistencyEvaluator(client),
            RagasFaithfulnessEvaluator(config, llm=client),
        ]
        report = copy.deepcopy(payload["report"])
        metadata = report["metadata"]
        for key in (
            "automatic_acceptance",
            "evaluation_history",
            "evidence_rewrite",
            "evidence_rewrite_attempts",
            "human_review",
            "approval",
        ):
            metadata.pop(key, None)
        metadata.update(
            accepted=False,
            verification=options.verification.model_dump(),
            run_options_fingerprint=options.fingerprint(),
            config_summary=plan["config"],
            evaluation_only={
                "source_run_id": plan["source_run_id"],
                "source_report_sha256": plan["source_report_sha256"],
                "generation_calls": 0,
                "search_calls": 0,
                "repair_calls": 0,
                "approval_inherited": False,
                "shared_token_limit": 200000,
                "shared_request_limit": 13,
                "prior_diagnostic_tokens": plan.get(
                    "diagnostic_budget", plan["source_budget"]
                )["used_tokens"],
                "prior_experiment_tokens": plan["source_budget"]["used_tokens"],
                "prior_experiment_requests": plan["source_budget"]["requests_sent"],
                "continuation": plan.get("continuation"),
            },
        )
        report["review_history"] = []
        report["config_fingerprint"] = plan["config"]["fingerprint"]
        hook(
            "survey.start", {"session_id": plan["run_id"], "phase": "frozen_evaluation"}
        )
        try:
            for task_id, result in payload["results"].items():
                if task_id != "adversarial_review":
                    hook(
                        "worker.complete",
                        {
                            "task_id": task_id,
                            "output": result,
                            "execution": "reused_frozen_input",
                            "source_run_id": plan["source_run_id"],
                        },
                    )
            report["evaluation"] = await agent._evaluate(
                query=payload["query"],
                survey=report["survey"],
                results=payload["results"],
                phase="initial",
            )
            report["quality"] = agent._derive_quality(report["evaluation"])
            await agent._automatic_review(
                report, payload["query"], payload["results"], None
            )
            assert file_sha_text(report["survey"]) == plan["source_report_sha256"]
            report = normalize_survey_result(
                report, config_fingerprint=plan["config"]["fingerprint"]
            )
            hook(
                "survey.complete",
                {
                    "quality_status": report["quality"]["status"],
                    "delivery_status": report["delivery"]["status"],
                    "total_tokens": budget.used - plan["source_budget"]["used_tokens"],
                },
            )
            if tracer:
                recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            recorder.finalize(report=report)
            state.update(
                status="completed",
                report=report,
                budget=budget.snapshot(),
                usage_complete=not budget.snapshot()["usage_uncertain"],
            )
        except BaseException as exc:
            state.update(
                status="failed",
                reason=type(exc).__name__,
                budget=budget.snapshot(),
                usage_complete=not budget.snapshot()["usage_uncertain"],
            )
            recorder.finalize(error=type(exc).__name__)
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            atomic_json(cp, state)
            await client.close()
            hook.flush()
        return state


def main():
    """Prepare or inspect by default; live performs one explicitly identified attempt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path)
    parser.add_argument("--prior-evaluation", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.prepare:
        if args.diagnostic is None:
            parser.error("--diagnostic is required with --prepare")
        prepare(
            args.root, args.diagnostic, config, prior_evaluation=args.prior_evaluation
        )
    result = asyncio.run(run(args.root, config, live=args.live))
    print(
        json.dumps(
            {k: result.get(k) for k in ("status", "binding", "budget", "reason")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
