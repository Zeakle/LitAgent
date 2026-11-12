"""Repair one frozen report once, then assess the complete candidate without audits."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
from pathlib import Path

from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.quality_suite import suite_lock
from litagent.body_quality import ANALYSIS_REPAIR_CAUSE_CONTRACT
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
from litagent.quality import (
    remove_assessed_repetition,
    strict_acceptance,
    targeted_repair_state,
)
from litagent.report_audit import atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions, VerificationOptions
from scripts.run_frozen_evaluation import continuation_budget, verify_frozen_sources
from scripts.run_frozen_generation import MODEL, ORIGIN, file_sha
from scripts.run_inventory_diagnostic import file_sha_text

MAX_REQUESTS = 11


class RepairClient(OpenAICompatibleClient):
    """Enforce this attempt's smaller request allowance at the provider boundary."""

    async def chat(self, messages, **kwargs):
        """Stop before transport on exhaustion and after an unexpected model."""
        if self._cost_budget.snapshot()["requests_sent"] >= MAX_REQUESTS:
            raise ValueError("bound_repair_request_limit")
        response = await super().chat(messages, **kwargs)
        if response.model != MODEL:
            self._cost_budget.stop_after_worker_failure()
            raise ValueError("bound_repair_actual_model_invalid")
        return response


def options() -> RunOptions:
    """Keep independent reviewers disabled throughout the frozen repair."""
    return RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        review_structure="academic_v2",
        report_language="zh-CN",
        verification=VerificationOptions(),
    )


def prior_repair_receipt(path: Path, parent_binding: str, payload: dict) -> dict:
    """Verify the one known local rejection before spending its remaining allowance."""
    plan = json.loads((path / "manifest.json").read_text())
    state = json.loads((path / "checkpoint.json").read_text())
    candidate = json.loads((path / "candidate.json").read_text())
    if (
        digest({k: v for k, v in plan.items() if k != "binding"}) != plan["binding"]
        or state["binding"] != plan["binding"]
        or state["status"] != "completed"
        or state.get("usage_complete") is not True
        or plan.get("prior_repair") is not None
        or plan["parent_binding"] != parent_binding
        or plan["max_tokens"] != 500000
        or plan["max_requests"] != MAX_REQUESTS
        or plan["max_repairs"] != 1
        or state["reason"] != "strict_repair_host_content_invalid"
        or candidate["committed"]
        or candidate["candidate_evaluation"] is not None
        or state["report"]["survey"] != payload["report"]["survey"]
        or json.loads((path / "input.json").read_text()) != payload
        or json.loads(Path(plan["receipt"]).read_text())
        != {
            "root": str(path.resolve()),
            "binding": plan["binding"],
            "parent_binding": parent_binding,
        }
    ):
        raise ValueError("bound_repair_prior_invalid")
    verify_frozen_sources(path, plan)
    budget = CostBudget.from_snapshot(state["budget"])
    repair_state = targeted_repair_state(
        state["report"]["metadata"]["targeted_repair_state"], plan["run_id"]
    )
    if (
        json.loads((path / "targeted-repair-state.json").read_text())["state"]
        != repair_state
    ):
        raise ValueError("bound_repair_prior_counter_invalid")
    events = [
        json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()
    ]
    opened = [e["data"] for e in events if e["event"] == "llm.start"]
    completed = [e["data"] for e in events if e["event"] == "llm.complete"]
    if (
        budget.stop_reason
        or budget.snapshot()["usage_uncertain"]
        or budget.snapshot()["max_tokens"] != 500000
        or budget.snapshot()["call_count"] != 1
        or budget.snapshot()["phase_usage"] != [0, 0, budget.used]
        or len(repair_state["attempts"]) != 1
        or len(opened) != 1
        or len(completed) != 1
        or any(e["event"] == "llm.failed" for e in events)
        or opened[0]["operation_id"] != completed[0]["operation_id"]
        or completed[0].get("usage_available") is not True
        or completed[0].get("actual_model") != MODEL
        or completed[0].get("finish_reason") != "stop"
        or budget.snapshot()["requests_sent"] != 1
        or any(
            type(completed[0].get(k)) is not int or completed[0][k] < 0
            for k in ("prompt_tokens", "completion_tokens")
        )
        or budget.used
        != completed[0]["prompt_tokens"] + completed[0]["completion_tokens"]
    ):
        raise ValueError("bound_repair_prior_usage_invalid")
    return {
        "root": str(path.resolve()),
        "binding": plan["binding"],
        "budget": budget.snapshot(),
        "repair_state": repair_state,
        "candidate": candidate,
    }


def prepare(
    root: Path, parent: Path, config, *, prior_repair: Path | None = None
) -> dict:
    """Bind complete parent usage, exact edit permissions and readable current code."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", root.name):
        raise ValueError("bound_repair_run_id_invalid")
    if (root / "manifest.json").exists() or (root / "checkpoint.json").exists():
        raise ValueError("bound_repair_identity_exists")
    prior = json.loads((parent / "manifest.json").read_text())
    state = json.loads((parent / "checkpoint.json").read_text())
    original = json.loads((parent / "input.json").read_text())
    # The immediate parent itself inherited earlier requests; do not reconstruct
    # historical expenditure from only its three newest provider events.
    before = prior["source_budget"]
    if continuation := prior.get("continuation"):
        previous = json.loads(
            (Path(continuation["parent"]) / "checkpoint.json").read_text()
        )
        before = previous["budget"]
        receipt = json.loads(Path(continuation["receipt"]).read_text())
        if receipt != {
            "root": str(parent.resolve()),
            "binding": prior["binding"],
            "parent_binding": continuation["parent_binding"],
        }:
            raise ValueError("bound_repair_parent_receipt_invalid")
    inherited = continuation_budget(parent, original, before)
    report = state["report"]
    if (
        config.llm.base_url.rstrip("/") != ORIGIN
        or not config.llm.uses_single_model(MODEL)
        or config.safety.max_cost_tokens != 500000
        or prior["config"] != build_config_summary(config, "strict_auto_v2")
        or config.safety.max_cost_tokens != 500000
        or report["metadata"].get("review_structure") != "academic_v2"
        or report["metadata"].get("report_language") != "zh-CN"
        or report["metadata"].get("body_quality") != "analysis_v1"
        or report["metadata"].get("verification") != VerificationOptions().model_dump()
        or report["metadata"].get("targeted_repair_state", {}).get("attempts")
        or report["delivery"].get("publishable") is not False
        or not LitAgent._completed_bound_repair(report["evaluation"], report["survey"])
    ):
        raise ValueError("bound_repair_parent_invalid")
    targets = LitAgent._strict_repair_targets(
        report["evaluation"], report["survey"], "academic_v2", "zh-CN"
    )
    if not targets:
        raise ValueError("bound_repair_no_targets")
    payload = {**copy.deepcopy(original), "report": copy.deepcopy(report)}
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
        payload["results"]["adversarial_review"][key] = copy.deepcopy(
            report["metadata"][key]
        )
    history = copy.deepcopy(prior["historical_before_diagnostic"])
    history["survey"]["charged"] += inherited["used_tokens"]
    history["survey"]["requests"] += inherited["requests_sent"]
    previous = (
        prior_repair_receipt(prior_repair, prior["binding"], payload)
        if prior_repair
        else None
    )
    root.mkdir(parents=True, exist_ok=True)
    atomic_json(root / "input.json", payload)
    atomic_json(root / "targets.json", targets)
    hashes = {
        str(p.resolve()): file_sha(p)
        for p in [
            *(
                parent / n
                for n in (
                    "manifest.json",
                    "checkpoint.json",
                    "input.json",
                    "events.jsonl",
                )
            ),
            root / "input.json",
            root / "targets.json",
        ]
    }
    if previous:
        hashes.update(
            {
                str((prior_repair / name).resolve()): file_sha(prior_repair / name)
                for name in (
                    "manifest.json",
                    "checkpoint.json",
                    "input.json",
                    "events.jsonl",
                    "candidate.json",
                    "targeted-repair-state.json",
                )
            }
        )
    for p in [
        *sorted(Path("src/litagent").rglob("*.py")),
        Path(__file__),
        Path("scripts/run_frozen_evaluation.py"),
        Path("scripts/run_frozen_generation.py"),
        Path("scripts/run_inventory_diagnostic.py"),
        Path("tests/test_bound_body_repair.py"),
        Path("tests/fixtures/repair-cause-real.json"),
    ]:
        hashes[str(p.resolve())] = file_sha(p)
        frozen = root / "frozen-code" / p.resolve().relative_to(Path.cwd())
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(p.read_bytes())
    fresh = CostBudget()
    fresh.configure_strict(halt_on_unknown_usage=True)
    plan = {
        "version": "bound-body-repair-v1",
        "run_id": root.name,
        "parent": str(parent.resolve()),
        "parent_binding": prior["binding"],
        "source_run_id": prior["run_id"],
        "source_report_sha256": file_sha_text(report["survey"]),
        "historical_before": history,
        "parent_budget": inherited,
        "config": build_config_summary(config, "strict_auto_v2"),
        "run_options_fingerprint": options().fingerprint(),
        "repair_cause_contract": ANALYSIS_REPAIR_CAUSE_CONTRACT,
        "max_tokens": 500000,
        "max_requests": MAX_REQUESTS,
        "max_repairs": 1,
        "retries": 0,
        "search": False,
        "judge": False,
        "jev": False,
        "hashes": hashes,
        "receipt": str(
            (
                prior_repair.with_name(prior_repair.name + "-successor.json")
                if previous
                else parent.with_name(parent.name + "-repair-successor.json")
            ).resolve()
        ),
        "source_budget": previous["budget"] if previous else fresh.snapshot(),
        **({"prior_repair": previous} if previous else {}),
    }
    plan["binding"] = digest(plan)
    with Path(plan["receipt"]).open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "root": str(root.resolve()),
                "binding": plan["binding"],
                "parent_binding": prior["binding"],
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(root / "manifest.json", plan)
    return plan


def bind_agent(config, run_id, client, budget, hook, state_path) -> LitAgent:
    """Use the production repair compiler and all real report evaluators."""
    agent = LitAgent(config, trace_hook=hook, run_policy=RunPolicy.isolated_benchmark())
    agent._session_id, agent._llm, agent._cost_budget = run_id, client, budget
    agent._targeted_repair_state_path = state_path
    agent._config_summary = build_config_summary(config, "strict_auto_v2")
    agent._synthesis = SynthesisWorker(
        client, context_config=config.context, trace_hook=hook
    )
    agent._evaluators = [
        CitationEvaluator(client),
        ConsistencyEvaluator(client),
        RagasFaithfulnessEvaluator(config, llm=client),
    ]
    return agent


def prepare_editorial(root: Path, parent: Path, config) -> dict:
    """Bind a proven local redundancy removal to the existing paid ledger."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", root.name):
        raise ValueError("bound_repair_run_id_invalid")
    if root.exists():
        raise ValueError("bound_repair_identity_exists")
    prior = json.loads((parent / "manifest.json").read_text())
    state = json.loads((parent / "checkpoint.json").read_text())
    payload = json.loads((parent / "input.json").read_text())
    if (
        digest({k: v for k, v in prior.items() if k != "binding"}) != prior["binding"]
        or state["binding"] != prior["binding"]
        or state["status"] != "completed"
        or state.get("usage_complete") is not True
        or not prior.get("prior_repair")
        or prior.get("editorial")
        or prior["max_tokens"] != 500000
        or prior["max_requests"] != MAX_REQUESTS
        or prior["config"] != build_config_summary(config, "strict_auto_v2")
        or config.llm.base_url.rstrip("/") != ORIGIN
        or not config.llm.uses_single_model(MODEL)
        or json.loads(Path(prior["receipt"]).read_text())
        != {
            "root": str(parent.resolve()),
            "binding": prior["binding"],
            "parent_binding": prior["parent_binding"],
        }
    ):
        raise ValueError("strict_editorial_parent_invalid")
    verify_frozen_sources(parent, prior)
    first = prior_repair_receipt(
        Path(prior["prior_repair"]["root"]), prior["parent_binding"], payload
    )
    if first != prior["prior_repair"] or first["budget"] != prior["source_budget"]:
        raise ValueError("strict_editorial_inherited_budget_invalid")
    budget = CostBudget.from_snapshot(state["budget"])
    events = [json.loads(l) for l in (parent / "events.jsonl").read_text().splitlines()]
    starts = [e["data"] for e in events if e["event"] == "llm.start"]
    done = [e["data"] for e in events if e["event"] == "llm.complete"]
    if (
        budget.stop_reason
        or budget.snapshot()["usage_uncertain"]
        or budget.snapshot()["max_tokens"] != 500000
        or len(starts) != len(done)
        or not done
        or len({e["operation_id"] for e in starts}) != len(starts)
        or {e["operation_id"] for e in starts} != {e["operation_id"] for e in done}
        or any(e["event"] == "llm.failed" for e in events)
        or any(
            e.get("usage_available") is not True
            or e.get("actual_model") != MODEL
            or e.get("finish_reason") != "stop"
            for e in done
        )
        or any(
            type(e.get(k)) is not int or e[k] < 0
            for e in done
            for k in ("prompt_tokens", "completion_tokens")
        )
        or budget.used
        != prior["source_budget"]["used_tokens"]
        + sum(e["prompt_tokens"] + e["completion_tokens"] for e in done)
        or budget.snapshot()["requests_sent"]
        != prior["source_budget"]["requests_sent"] + len(done)
        or budget.snapshot()["call_count"] != budget.snapshot()["requests_sent"]
        or budget.snapshot()["requests_sent"] >= MAX_REQUESTS
    ):
        raise ValueError("strict_editorial_parent_usage_invalid")
    report = copy.deepcopy(state["report"])
    committed = json.loads((parent / "candidate.json").read_text())
    if (
        committed.get("committed") is not True
        or committed.get("candidate") != report["survey"]
        or committed.get("candidate_evaluation") != report["evaluation"]
        or report["metadata"]["verification"] != VerificationOptions().model_dump()
    ):
        raise ValueError("strict_editorial_parent_commit_invalid")
    assessed = report["evaluation"]["faithfulness"]["details"]
    if strict_acceptance(assessed)["status"] != "passed" or assessed.get(
        "source_report_sha256"
    ) != file_sha_text(report["survey"]):
        raise ValueError("strict_editorial_parent_assessment_incomplete")
    counter = targeted_repair_state(
        report["metadata"]["targeted_repair_state"], prior["run_id"]
    )
    if (
        len(counter["attempts"]) != 2
        or json.loads((parent / "targeted-repair-state.json").read_text())["state"]
        != counter
    ):
        raise ValueError("strict_editorial_counter_invalid")
    body, bindings, cleanup = remove_assessed_repetition(
        payload["report"]["survey"],
        payload["report"]["evaluation"],
        report["survey"],
        report["evaluation"],
        report["metadata"]["writing_bindings"],
    )
    report["survey"] = body
    report["metadata"]["writing_bindings"] = bindings
    report["metadata"]["editorial_cleanup"] = cleanup
    payload["report"] = report
    payload["results"]["adversarial_review"]["writing_bindings"] = bindings
    root.mkdir(parents=True)
    atomic_json(root / "input.json", payload)
    atomic_json(root / "editorial-cleanup.json", cleanup)
    hashes = {
        str((parent / n).resolve()): file_sha(parent / n)
        for n in (
            "manifest.json",
            "checkpoint.json",
            "input.json",
            "candidate.json",
            "targeted-repair-state.json",
            "events.jsonl",
        )
    }
    for p in [root / "input.json", root / "editorial-cleanup.json"]:
        hashes[str(p.resolve())] = file_sha(p)
    for p in [
        *sorted(Path("src/litagent").rglob("*.py")),
        *sorted(Path("scripts").glob("run_*evaluation*.py")),
        Path(__file__),
        Path("scripts/run_frozen_generation.py"),
        Path("scripts/run_inventory_diagnostic.py"),
        Path("tests/test_bound_body_repair.py"),
        Path("tests/test_assessed_repetition.py"),
        Path("tests/fixtures/assessed-repetition-real.json"),
    ]:
        hashes[str(p.resolve())] = file_sha(p)
        frozen = root / "frozen-code" / p.resolve().relative_to(Path.cwd())
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes(p.read_bytes())
    plan = {
        "version": "bound-editorial-evaluation-v1",
        "editorial": True,
        "run_id": root.name,
        "parent": str(parent.resolve()),
        "parent_binding": prior["binding"],
        "source_run_id": prior["run_id"],
        "source_report_sha256": cleanup["candidate_sha256"],
        "historical_before": prior["historical_before"],
        "source_budget": budget.snapshot(),
        "editorial_repair_state": counter,
        "editorial_history": report["metadata"].get("evaluation_history"),
        "config": build_config_summary(config, "strict_auto_v2"),
        "run_options_fingerprint": options().fingerprint(),
        "max_tokens": 500000,
        "max_requests": MAX_REQUESTS,
        "max_repairs": 0,
        "retries": 0,
        "search": False,
        "judge": False,
        "jev": False,
        "hashes": hashes,
        "receipt": str(
            parent.with_name(parent.name + "-editorial-successor.json").resolve()
        ),
    }
    plan["binding"] = digest(plan)
    with Path(plan["receipt"]).open("x", encoding="utf-8") as stream:
        json.dump(
            {
                "root": str(root.resolve()),
                "binding": plan["binding"],
                "parent_binding": prior["binding"],
            },
            stream,
        )
        stream.flush()
        os.fsync(stream.fileno())
    atomic_json(root / "manifest.json", plan)
    return plan


async def run(
    root: Path, config, *, live=False, client_factory=None, repository=None
) -> dict:
    """Perform one repair and complete reassessment, preserving failed candidates."""
    plan = json.loads((root / "manifest.json").read_text())
    if (
        digest({k: v for k, v in plan.items() if k != "binding"}) != plan["binding"]
        or build_config_summary(config, "strict_auto_v2") != plan["config"]
        or options().fingerprint() != plan["run_options_fingerprint"]
        or any(file_sha(Path(p)) != h for p, h in plan["hashes"].items())
        or json.loads(Path(plan["receipt"]).read_text())
        != {
            "root": str(root.resolve()),
            "binding": plan["binding"],
            "parent_binding": plan["parent_binding"],
        }
    ):
        raise ValueError("bound_repair_identity_changed")
    if not live:
        return {
            "status": "planned",
            "binding": plan["binding"],
            "paid_calls": 0,
            "historical_before": plan["historical_before"],
        }
    with suite_lock(root / ".repair.lock"):
        checkpoint = root / "checkpoint.json"
        if checkpoint.exists():
            state = json.loads(checkpoint.read_text())
            if state["binding"] != plan["binding"]:
                raise ValueError("bound_repair_checkpoint_conflict")
            if state["status"] == "started_no_replay":
                state.update(
                    status="interrupted_no_replay",
                    usage_complete=False,
                    conservative_tokens=500000,
                    conservative_requests=MAX_REQUESTS,
                )
                atomic_json(checkpoint, state)
            return state
        repo = repository if repository is not None else ArchiveRepository()
        if repo.path_for(plan["run_id"]).exists():
            raise ValueError("bound_repair_archive_exists")
        budget = CostBudget(max_tokens=plan["max_tokens"])
        budget.configure_strict(halt_on_unknown_usage=True)
        if plan.get("prior_repair") or plan.get("editorial"):
            budget.restore(plan["source_budget"])
        state = {
            "binding": plan["binding"],
            "status": "started_no_replay",
            "budget": budget.snapshot(),
        }
        atomic_json(checkpoint, state)
        payload = json.loads((root / "input.json").read_text())
        report = copy.deepcopy(payload["report"])
        metadata = report["metadata"]
        for key in (
            "approval",
            "human_review",
            "automatic_acceptance",
            "evaluation_only",
            "evaluation_history",
            "evidence_rewrite",
            "evidence_rewrite_attempts",
        ):
            metadata.pop(key, None)
        metadata.update(
            accepted=False,
            execution_scope="bound_body_repair",
            verification=options().verification.model_dump(),
            config_summary=plan["config"],
            run_options_fingerprint=options().fingerprint(),
            targeted_repair_state=targeted_repair_state(None, plan["run_id"]),
            frozen_repair_provenance={
                "source_run_id": plan["source_run_id"],
                "source_report_sha256": plan["source_report_sha256"],
                "parent_binding": plan["parent_binding"],
                "historical_before": plan["historical_before"],
            },
        )
        if previous := plan.get("prior_repair"):
            shared = copy.deepcopy(previous["repair_state"])
            shared["run_id"] = plan["run_id"]
            metadata["targeted_repair_state"] = shared
            metadata["inherited_targeted_repair_run_id"] = Path(previous["root"]).name
            metadata["local_repair_diagnostics"] = {
                "source_report_sha256": plan["source_report_sha256"],
                "reason_code": "strict_repair_host_content_invalid",
                "instruction": "The prior response copied bracketed IDs into text and was rejected. Keep source IDs only in fact_card_ids. Correct the actual original task/conditions; do not substitute unrelated or already protected findings.",
            }
        if plan.get("editorial"):
            shared = copy.deepcopy(plan["editorial_repair_state"])
            shared["run_id"] = plan["run_id"]
            metadata["targeted_repair_state"] = shared
            metadata["inherited_targeted_repair_run_id"] = plan["source_run_id"]
            metadata["evaluation_only"] = {
                "model_rewrite_calls": 0,
                "source_report_sha256": plan["source_report_sha256"],
                "approval_inherited": False,
            }
            metadata["execution_scope"] = "local_editorial_full_evaluation"
            report["evaluation"] = {}
            report["quality"] = {"status": "unverified", "failed_metrics": []}
            report["delivery"] = {"status": "blocked", "publishable": False}
        recorder = RunRecorder(plan["run_id"], payload["query"], repo)
        recorder.set_config_summary(plan["config"])
        recorder.set_citation_style("ieee")
        tracer = None
        if client_factory is None and config.observability.enabled:
            tracer = LangFuseTracer(
                config.observability.langfuse_host,
                os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=plan["run_id"],
            )

        def journal(event, data):
            """Retain raw evidence only in the local journal, before redacted forwarding."""
            with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"event": event, "data": data}, ensure_ascii=False, default=str
                    )
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())

        hook = CompositeTraceHook(
            recorder,
            journal,
            RedactingTraceHook(tracer, "metadata_only") if tracer else None,
        )
        client = (
            client_factory(budget, hook)
            if client_factory
            else RepairClient(
                ORIGIN,
                MODEL,
                max_tokens=config.llm.max_tokens,
                temperature=config.llm.temperature,
                cost_budget=budget,
                trace_hook=hook,
                transport_max_attempts=1,
            )
        )
        agent = bind_agent(
            config,
            plan["run_id"],
            client,
            budget,
            hook,
            root / "targeted-repair-state.json",
        )
        hook(
            "survey.start", {"session_id": plan["run_id"], "phase": "bound_body_repair"}
        )
        try:
            for task_id, output in payload["results"].items():
                if task_id != "adversarial_review":
                    hook(
                        "worker.complete",
                        {
                            "task_id": task_id,
                            "output": output,
                            "execution": "reused_frozen_input",
                            "source_run_id": plan["source_run_id"],
                        },
                    )
            if not plan.get("editorial") and agent._strict_repair_targets(
                report["evaluation"], report["survey"], "academic_v2", "zh-CN"
            ) != json.loads((root / "targets.json").read_text()):
                raise ValueError("bound_repair_targets_changed")
            original_evaluation = copy.deepcopy(report["evaluation"])
            if plan.get("editorial"):
                await evaluate_editorial(agent, report, payload, root, plan)
                outcome = None
            else:
                outcome = await agent._attempt_evidence_rewrite(
                    query=payload["query"],
                    report_data=report,
                    results=payload["results"],
                    initial_evaluation=original_evaluation,
                    initial_quality=report["quality"],
                    report_language="zh-CN",
                )
            candidate = (
                {
                    "attempted": outcome.attempted,
                    "committed": outcome.committed,
                    "candidate": outcome.candidate,
                    "candidate_evaluation": outcome.candidate_evaluation,
                    "candidate_quality": outcome.candidate_quality,
                    "candidate_bindings": outcome.candidate_bindings,
                    "reason_code": outcome.reason_code,
                }
                if outcome
                else None
            )
            if outcome:
                atomic_json(root / "candidate.json", candidate)
                metadata["evidence_rewrite"] = candidate
            attempts = (
                [plan["prior_repair"]["candidate"]] if plan.get("prior_repair") else []
            ) + [candidate]
            if outcome:
                metadata["evidence_rewrite_attempts"] = attempts
                metadata["evaluation_history"] = {
                    "version": "version-bound-evaluation-history-v1",
                    "source_report_sha256": file_sha_text(report["survey"]),
                    "initial_evaluation": original_evaluation,
                    "attempts": attempts,
                    "final_evaluation": report["evaluation"],
                }
            # Failed candidates never replace the frozen report, even when locally valid.
            await agent._automatic_review(
                report, payload["query"], payload["results"], None
            )
            report = normalize_survey_result(
                report, config_fingerprint=plan["config"]["fingerprint"]
            )
            state.update(
                status="completed",
                reason=(
                    outcome.reason_code if outcome else "editorial_full_reassessment"
                ),
                report=report,
                budget=budget.snapshot(),
                usage_complete=not budget.snapshot()["usage_uncertain"],
            )
            hook(
                "survey.complete",
                {
                    "quality_status": report["quality"]["status"],
                    "delivery_status": report["delivery"]["status"],
                    "total_tokens": budget.used - plan["source_budget"]["used_tokens"],
                },
            )
        except BaseException as exc:
            state.update(
                status="failed",
                reason=type(exc).__name__,
                report=report,
                budget=budget.snapshot(),
                usage_complete=not budget.snapshot()["usage_uncertain"],
            )
            hook("survey.error", {"error_type": type(exc).__name__})
            if isinstance(exc, asyncio.CancelledError):
                raise
        finally:
            atomic_json(checkpoint, state)
            if tracer:
                recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            recorder.finalize(
                report=state.get("report"),
                error=state.get("reason") if state["status"] == "failed" else None,
            )
            await client.close()
            hook.flush()
        return state


async def evaluate_editorial(agent, report, payload, root, plan):
    """Assess every new-version unit without invoking any rewrite producer."""
    report["evaluation"] = await agent._evaluate(
        query=payload["query"],
        survey=report["survey"],
        results=payload["results"],
        phase="initial",
    )
    report["quality"] = agent._derive_quality(report["evaluation"])
    report["metadata"]["evaluation_history"] = {
        "version": "version-bound-evaluation-history-v1",
        "source_report_sha256": file_sha_text(report["survey"]),
        "parent_report_sha256": plan["source_report_sha256"],
        "prior_history": plan["editorial_history"],
        "editorial_cleanup": json.loads((root / "editorial-cleanup.json").read_text()),
        "final_evaluation": report["evaluation"],
        "approval_inherited": False,
    }


def main():
    """Prepare and inspect first; --live consumes the fixed one-attempt allowance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--prior-repair", type=Path)
    parser.add_argument("--editorial", action="store_true")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.prepare:
        if args.parent is None:
            parser.error("--parent is required for preparation")
        if args.editorial:
            prepare_editorial(args.root, args.parent, config)
        else:
            prepare(args.root, args.parent, config, prior_repair=args.prior_repair)
    result = asyncio.run(run(args.root, config, live=args.live))
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "binding", "reason", "budget", "paid_calls")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
