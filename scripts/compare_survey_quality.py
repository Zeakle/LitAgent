"""Replay frozen writing inputs under captured and current prompt contracts."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.eval.scoped import EVALUATION_VERSION
from litagent.evidence import build_citation_aliases, collect_ledger
from litagent.interaction import interaction_source_fingerprint
from litagent.llm.client import BaseLLMClient, OpenAICompatibleClient
from litagent.observability.context import get_task_id
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.orchestrator.task_graph import SubTask
from litagent.runner import LitAgent, derive_delivery
from litagent.safety.budget import CostBudget


def execution_options(args) -> tuple[str, list[str]]:
    """Keep historical paired defaults and reject old prompts for new structures."""
    structure = getattr(args, "review_structure", "method_transparent_v1")
    arm = getattr(args, "arm", "both")
    if structure not in {"method_transparent_v1", "academic_v2"}:
        raise ValueError("comparison_structure_invalid")
    if arm not in {"both", "baseline", "candidate"}:
        raise ValueError("comparison_arm_invalid")
    arms = ["baseline", "candidate"] if arm == "both" else [arm]
    if structure == "academic_v2" and arms != ["candidate"]:
        raise ValueError("academic_v2_requires_current_single_arm")
    return structure, arms


def fixed_parameters(cfg) -> dict:
    """Describe every fixed generation and evaluation limit used by this replay."""
    return {
        "model": cfg.llm.model,
        "max_cost_tokens": cfg.safety.max_cost_tokens,
        "generation_max_tokens": cfg.llm.max_tokens,
        "context_max_tokens": cfg.context.max_tokens,
        "revision_max_rounds": cfg.adversarial.max_rounds,
        "evaluation_version": EVALUATION_VERSION,
        "faithfulness_threshold": RagasFaithfulnessEvaluator(cfg)._threshold,
        "evaluation_calls_per_stage": 2,
        "evaluation_max_output_tokens_per_call": cfg.eval.max_tokens,
        "evaluation_context_tokens": cfg.eval.context_tokens,
        "evaluation_max_evidence_items": cfg.eval.faithfulness_max_evidence_items,
        "evaluation_max_context_chars": cfg.eval.faithfulness_max_context_chars,
        "evaluation_timeout_seconds": cfg.eval.timeout_seconds,
        "evaluation_transport_attempts": 1,
        "auto_audit_mode": "off",
        "judge_jev_delivery_gate": False,
    }


async def initial_request(data: dict, cfg, structure: str) -> dict:
    """Build the actual synthesis request locally without constructing a provider."""
    captured = {}
    synthesis = SynthesisWorker(
        None,
        evidence_selector=FrozenSelector(data["selection"]),
        agent_config=cfg.agent,
        context_config=cfg.context,
    )

    async def capture(**kwargs):
        captured["messages"] = transform_messages(
            [
                {"role": "system", "content": kwargs["system_prompt"]},
                {"role": "user", "content": kwargs["primary_context"]},
            ],
            "adversarial_review:synthesis:r1",
            "candidate",
            data,
        )
        return "", []

    synthesis._generate_with_bounded_fallback = capture
    await synthesis.execute(
        SubTask(
            "synthesis",
            "Offline request preflight",
            "synthesis",
            input_data={
                "query": data["frozen"]["query"],
                "report_language": "zh-CN",
                "review_structure": structure,
                "review_methodology": data["record"],
                "upstream_results": data["upstream"],
            },
        )
    )
    captured["max_tokens"] = cfg.llm.max_tokens
    captured["input_output_reservation"] = (
        len(json.dumps(captured["messages"], ensure_ascii=False).encode())
        + 1024
        + cfg.llm.max_tokens
    )
    if captured["input_output_reservation"] > min(131072, cfg.safety.max_cost_tokens):
        raise ValueError("comparison_initial_request_budget_exceeded")
    return captured


def save(path: Path, value: dict) -> None:
    """Atomically persist a sidecar without changing any historical artifact."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def prepare(root: Path) -> dict:
    """Verify the immutable inputs and restore selected items from the full ledger."""
    plan = json.loads((root / "controlled-comparison-plan.json").read_text())
    path = root / plan["input"]["path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan["input"]["sha256"]:
        raise ValueError("frozen_materials_sha_conflict")
    frozen = json.loads(path.read_text())
    source = Path(frozen["source_artifact"])
    if (
        hashlib.sha256(source.read_bytes()).hexdigest()
        != frozen["source_artifact_sha256"]
    ):
        raise ValueError("historical_artifact_sha_conflict")
    artifact = json.loads(source.read_text())
    extractions = frozen["worker_outputs"]["worker:extract"]["output"]
    ledger = collect_ledger(extractions)
    summary = frozen["selected_evidence"]
    ids = summary["section_evidence_ids"]["abstract"]
    if len(set(ids)) != summary["selected_count"] or any(i not in ledger for i in ids):
        raise ValueError("selected_inventory_invalid")
    selection = EvidenceSelection.from_dict(
        {**summary, "selected_items": {i: ledger[i] for i in ids}}
    )
    captured = {
        node["task_id"]: node["input"]["messages"]
        for node in artifact["nodes"].values()
        if node.get("kind") == "llm"
        and node.get("task_id", "").startswith("adversarial_review:")
    }
    for component in ["synthesis", "reviewer"]:
        for round_number in [1, 2]:
            if f"adversarial_review:{component}:r{round_number}" not in captured:
                raise ValueError("captured_prompt_missing")
    aliases = build_citation_aliases(ids)
    expected = set(aliases.evidence_id_to_alias.values())
    initial = captured["adversarial_review:synthesis:r1"][-1]["content"]
    if set(re.findall(r"\[E:([^\]]+)\]", initial)) != expected:
        raise ValueError("captured_alias_inventory_invalid")
    return {
        "plan": plan,
        "frozen": frozen,
        "selection": selection,
        "captured": captured,
        "upstream": {"extract": extractions, "graph_analysis": frozen["paper_catalog"]},
    }


def transform_messages(
    messages: list[dict], stage: str, arm: str, data: dict
) -> list[dict]:
    """Replay old writing instructions while preserving current drafts and feedback."""
    value = copy.deepcopy(messages)
    if not stage.startswith("adversarial_review:"):
        return value
    component = "reviewer" if ":reviewer:" in stage else "synthesis"
    round_number = 1 if stage.endswith(":r1") else 2
    captured = data["captured"][f"adversarial_review:{component}:r{round_number}"]
    # Keep skill metadata equal without initializing an embedder or tool executor.
    skills = re.search(
        r"<available_skills>.*?</available_skills>", captured[0]["content"], re.S
    )
    if arm == "candidate":
        if skills and "<available_skills>" not in value[0]["content"]:
            value[0]["content"] += "\n" + skills.group()
        return value
    value[0]["content"] = captured[0]["content"]
    record = json.dumps(data["record"], ensure_ascii=False)
    value[0]["content"] = re.sub(
        r"(?<=<run_methodology>).*?(?=</run_methodology>)",
        lambda _: record,
        value[0]["content"],
        flags=re.S,
    )
    for tag in ["evidence_plan", "evidence_ledger"]:
        match = re.search(rf"<{tag}>.*?</{tag}>", captured[-1]["content"], re.S)
        if not match:
            raise ValueError("captured_evidence_block_missing")
        value[-1]["content"], count = re.subn(
            rf"<{tag}>.*?</{tag}>",
            lambda _: match.group(),
            value[-1]["content"],
            flags=re.S,
        )
        if count != 1:
            raise ValueError("current_evidence_block_invalid")
    return value


class FrozenSelector:
    """Return the same complete evidence selection without retrieval calls."""

    def __init__(self, selection: EvidenceSelection):
        self.selection = selection

    async def select(self, *args, **kwargs):
        """Do not rerank or change the frozen section allocation."""
        return copy.deepcopy(self.selection)


def record_frozen_sources(recorder: RunRecorder, data: dict) -> None:
    """Import existing source outputs without claiming to execute retrieval again."""
    frozen = data["frozen"]
    provenance = {
        "origin": "frozen_input_replay",
        "source_artifact": frozen["source_artifact"],
        "source_artifact_sha256": frozen["source_artifact_sha256"],
    }
    for task_id in ("dedup", "relevance_gate", "extract", "graph_analysis"):
        node = frozen["worker_outputs"][f"worker:{task_id}"]
        recorder(
            "worker.complete",
            {"task_id": task_id, "output": node["output"], **provenance},
        )
    recorder(
        "evidence.retrieve.complete",
        {
            "operation_id": "frozen-selection",
            "task_id": "synthesis",
            "output": data["selection"].to_dict(),
            **provenance,
        },
    )


def evaluation_complete(evaluation: dict) -> bool:
    """Keep incomplete results from starting another paid comparison arm."""
    required = {"citation_accuracy", "faithfulness", "internal_consistency"}
    if not isinstance(evaluation, dict) or not required <= evaluation.keys():
        return False
    for key in required:
        metric = evaluation[key]
        if not isinstance(metric, dict):
            return False
        score = metric.get("score")
        if (
            metric.get("skipped") is not False
            or not isinstance(metric.get("passed"), bool)
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not 0 <= score <= 1
        ):
            return False
    details = evaluation["faithfulness"].get("details")
    return (
        isinstance(details, dict)
        and details.get("evaluation_version") == EVALUATION_VERSION
        and isinstance(details.get("coverage"), dict)
        and details["coverage"].get("complete") is True
    )


class BoundedClient(BaseLLMClient):
    """Persist attempted calls and block retries after an uncertain transport result."""

    def __init__(self, provider, budget, arm, data, root):
        self.provider = provider
        self._cost_budget = budget
        self.arm, self.data, self.root = arm, data, root
        self.calls = 0
        self.stopped = False

    async def chat(self, messages, **kwargs):
        """Check complete input/output reservations before each paid request."""
        if self.stopped or self.calls >= 16:
            raise ValueError("comparison_call_gate_closed")
        stage = get_task_id() or "unassigned"
        messages = transform_messages(messages, stage, self.arm, self.data)
        maximum = kwargs.get("max_tokens", self.provider._max_tokens)
        estimate = (
            len(json.dumps(messages, ensure_ascii=False).encode()) + 1024 + maximum
        )
        # Both evaluation and runner repair reserve the full request at the caller.
        caller_reserved = stage.startswith("evaluation.") or stage == "evidence_rewrite"
        if not caller_reserved and estimate > self._cost_budget.remaining():
            self.stopped = True
            raise ValueError("comparison_run_budget_exceeded")
        if estimate > 131072:
            self.stopped = True
            raise ValueError("comparison_context_budget_exceeded")
        if not caller_reserved:
            self._cost_budget.reserve(estimate)
        self.calls += 1
        path = self.root / f"{self.arm}-call-{self.calls:02}.json"
        record = {
            "stage": stage,
            "status": "started_no_replay",
            "messages": messages,
            "max_tokens": maximum,
        }
        save(path, record)
        print(
            json.dumps(
                {
                    "arm": self.arm,
                    "call": self.calls,
                    "stage": stage,
                    "status": "started",
                }
            ),
            flush=True,
        )
        try:
            kwargs["transport_max_attempts"] = 1
            response = await asyncio.wait_for(
                self.provider.chat(messages, **kwargs), timeout=180
            )
            record.update(status="completed", response=asdict(response))
            save(path, record)
            if not response.usage.get("usage_available"):
                self.stopped = True
                raise ValueError("comparison_usage_incomplete")
            print(
                json.dumps(
                    {
                        "arm": self.arm,
                        "call": self.calls,
                        "status": "completed",
                        "usage": response.usage,
                    }
                ),
                flush=True,
            )
            return response
        except BaseException as exc:
            self.stopped = True
            record.update(status="interrupted_no_replay", error_type=type(exc).__name__)
            save(path, record)
            raise
        finally:
            if not caller_reserved:
                self._cost_budget.release(estimate)


async def run_arm(arm: str, data: dict, cfg, root: Path) -> dict:
    """Execute writing/review/evaluation with no search, memory or audit services."""
    structure = data.get("execution", {}).get(
        "review_structure", "method_transparent_v1"
    )
    execution_options(argparse.Namespace(review_structure=structure, arm=arm))
    marker = root / f"{arm}-started.json"
    if marker.exists():
        raise ValueError("comparison_attempt_already_started_no_replay")
    run_id = f"quality-v2-compare-{root.name}-{arm}"
    repository = ArchiveRepository()
    if repository.path_for(run_id).exists():
        raise ValueError("comparison_artifact_already_exists_no_replay")
    save(marker, {"arm": arm, "status": "started_no_replay"})
    recorder = RunRecorder(run_id, data["frozen"]["query"], repository)
    record_frozen_sources(recorder, data)
    recorder.set_config_summary(build_config_summary(cfg))
    tracer = LangFuseTracer(
        cfg.observability.langfuse_host,
        os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        os.getenv("LANGFUSE_SECRET_KEY", ""),
        trace_seed=run_id,
    )
    hook = RedactingTraceHook(CompositeTraceHook(recorder, tracer))
    budget = CostBudget(cfg.safety.max_cost_tokens)
    provider = OpenAICompatibleClient(
        cfg.llm.base_url,
        cfg.llm.model,
        cfg.llm.max_tokens,
        cfg.llm.temperature,
        budget,
        hook,
        transport_max_attempts=1,
    )
    client = BoundedClient(provider, budget, arm, data, root)
    synthesis = SynthesisWorker(
        client,
        evidence_selector=FrozenSelector(data["selection"]),
        agent_config=cfg.agent,
        context_config=cfg.context,
        trace_hook=hook,
    )
    reviewer = ReviewerWorker(
        client, config=cfg.adversarial, context_config=cfg.context
    )
    adversarial = AdversarialReviewWorker(
        client,
        synthesis,
        reviewer,
        cfg.adversarial.max_rounds,
        cfg.adversarial.pass_threshold,
        cfg.adversarial.thinking_mode,
        hook,
    )
    runner = LitAgent(cfg, trace_hook=hook)
    runner._llm, runner._synthesis, runner._cost_budget = client, synthesis, budget
    runner._evaluators = [
        CitationEvaluator(llm=client),
        RagasFaithfulnessEvaluator(cfg, llm=client),
        ConsistencyEvaluator(llm=client),
    ]
    report = None
    hook("survey.start", {"session_id": run_id, "query": data["frozen"]["query"]})
    try:
        task = SubTask(
            "adversarial_review",
            "Frozen writing comparison",
            "adversarial_review",
            input_data={
                "query": data["frozen"]["query"],
                "report_language": "zh-CN",
                "review_structure": structure,
                "review_methodology": data["record"],
                "upstream_results": data["upstream"],
            },
            max_retries=0,
        )
        result = await adversarial.execute(task)
        results = {**data["upstream"], "adversarial_review": result}
        report = runner._extract_report(results, data["frozen"]["query"])
        report["metadata"].update(
            {
                "query": data["frozen"]["query"],
                "report_language": "zh-CN",
                "review_structure": structure,
                "citation_style": "ieee",
                "evidence_selection": result.get("evidence_selection"),
                "review_methodology": result["review_methodology"],
                "comparison_arm": arm,
                "comparison_scope": "frozen_writing_chain_not_full_survey",
                "frozen_input_sha256": data["plan"]["input"]["sha256"],
                "source_fingerprint": interaction_source_fingerprint(),
            }
        )
        report["partial"] = False
        evaluation = await runner._evaluate(
            query=data["frozen"]["query"],
            survey=report["survey"],
            results=results,
            phase="initial",
        )
        quality = runner._derive_quality(evaluation)
        report.update(evaluation=evaluation, quality=quality)
        save(root / f"{arm}-initial.json", report)
        rewrite = None
        if (
            quality["status"] == "failed"
            and evaluation_complete(evaluation)
            and not client.stopped
        ):
            rewrite = await runner._attempt_evidence_rewrite(
                query=data["frozen"]["query"],
                report_data=report,
                results=results,
                initial_evaluation=evaluation,
                initial_quality=quality,
                report_language="zh-CN",
            )
            save(root / f"{arm}-rewrite.json", asdict(rewrite))
        report["delivery"] = derive_delivery(False, report["quality"])
        hook("survey.complete", {"output": report})
        recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
        artifact = recorder.finalize(report)
        output = {
            "run_id": run_id,
            "quality": report["quality"],
            "evaluation": report["evaluation"],
            "body_sha256": hashlib.sha256(report["survey"].encode()).hexdigest(),
            "tokens": budget.snapshot(),
            "calls": client.calls,
            "trace": artifact["trace"],
            "contract_complete": not client.stopped
            and evaluation_complete(report["evaluation"]),
        }
        save(root / f"{arm}-result.json", output)
        return output
    except BaseException as exc:
        hook("survey.error", {"error": type(exc).__name__})
        recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
        recorder.finalize(report, error=type(exc).__name__)
        save(
            root / f"{arm}-failure.json",
            {
                "error_type": type(exc).__name__,
                "calls": client.calls,
                "tokens": budget.snapshot(),
            },
        )
        raise
    finally:
        await provider._client.close()
        tracer.flush()


async def main(args) -> None:
    """Validate all fixed parameters before creating any network client."""
    structure, arms = execution_options(args)
    data = prepare(args.input_root)
    cfg = load_config()
    fixed = data["plan"]["fixed_parameters"]
    if fixed != fixed_parameters(cfg):
        raise ValueError("comparison_config_conflict")
    if cfg.llm.base_url.rstrip("/") != "https://ark.cn-beijing.volces.com/api/v3":
        raise ValueError("comparison_endpoint_conflict")
    data["execution"] = {"review_structure": structure, "arms": arms}
    data["record"] = {
        **data["frozen"]["host_runtime_record"],
        "evidence_counts": {
            "candidate": data["selection"].candidate_count,
            "selected": len(data["selection"].selected_items),
        },
    }
    synthesis = SynthesisWorker(None, context_config=cfg.context)
    aliases = build_citation_aliases(list(data["selection"].selected_items))
    context, context_tokens = await synthesis._build_synthesis_context(
        query=data["frozen"]["query"],
        aliases=aliases,
        extractions=data["upstream"]["extract"],
        graph_data=data["frozen"]["paper_catalog"],
        selection=data["selection"],
    )
    block = re.search(r"<evidence_ledger>(.*?)</evidence_ledger>", context, re.S)
    expected_aliases = set(aliases.evidence_id_to_alias.values())
    if not block or set(re.findall(r"\[E:([^\]]+)\]", block[1])) != expected_aliases:
        raise ValueError("candidate_generation_materials_incomplete")
    request = await initial_request(data, cfg, structure)
    args.output.mkdir(parents=True, exist_ok=True)
    save(args.output / "initial-request-dry-run.json", request)
    save(
        args.output / "preflight.json",
        {
            "status": "ready",
            "live": args.live,
            "selected": len(data["selection"].selected_items),
            "candidate": data["selection"].candidate_count,
            "fixed_parameters": fixed,
            "execution": data["execution"],
            "endpoint": cfg.llm.base_url,
            "requests_per_arm_max": 16,
            "total_tokens_max": cfg.safety.max_cost_tokens * len(arms),
            "initial_request_reservation": request["input_output_reservation"],
            "credentials_available": bool(
                os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
            ),
            "input_sha256": data["plan"]["input"]["sha256"],
            "source_fingerprint": interaction_source_fingerprint(),
            "initial_context_tokens": context_tokens,
            "initial_ledger_alias_count": len(expected_aliases),
        },
    )
    print(
        f"preflight ready: {structure}; arms={arms}; "
        f"{len(data['selection'].selected_items)} selected / "
        f"{data['selection'].candidate_count} candidate; no search or Judge/Jev",
        flush=True,
    )
    if not args.live:
        return
    if not (os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise ValueError("missing_llm_credentials")
    results = {}
    for arm in arms:
        results[arm] = await run_arm(arm, data, cfg, args.output)
        save(
            args.output / "comparison.json",
            {
                "scope": (
                    "single_frozen_writing_pair"
                    if len(arms) == 2
                    else "single_frozen_writing_attempt"
                ),
                "execution": data["execution"],
                "results": results,
                "statistical_pass_rate_gain_established": False,
            },
        )
        if not results[arm]["contract_complete"]:
            print(
                "comparison stopped: incomplete evaluation; next arm not submitted",
                flush=True,
            )
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("artifacts/acceptance/quality-v2-20261003"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--arm", choices=["both", "baseline", "candidate"], default="both"
    )
    parser.add_argument(
        "--review-structure",
        choices=["method_transparent_v1", "academic_v2"],
        default="method_transparent_v1",
    )
    asyncio.run(main(parser.parse_args()))
