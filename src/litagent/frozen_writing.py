"""Replay public frozen evidence through existing workers and the sole report exit."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.eval.citation import CitationEvaluator
from litagent.eval.consistency import ConsistencyEvaluator
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.orchestrator.scheduler import Scheduler
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import build_fact_cards, coverage_requirements, digest
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions, VerificationOptions
from litagent.writing_v2 import (
    EOF_CONTAINER_POLICY,
    INTACT_UNIT_REEMISSION_CONTRACT,
    SCOPE,
    STRATEGY,
    source_packet,
    writing_plan_spec,
)

CASES = (
    ("few-shot-metric", "survey-reliability-20261022-004-pilot-few-shot-metric"),
    ("vit-data-efficient", "quality-auto-native-002-pilot-vit-data-efficient"),
    ("nerf-foundations", "quality-auto-native-002-pilot-nerf-foundations"),
)


def frozen_options():
    """Keep optional paid checks off for this explicitly bounded validation."""
    return RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy=STRATEGY,
        report_language="zh-CN",
        review_structure="academic_v2",
        verification=VerificationOptions(judge=False, jev=False),
    )


def read_bundle(path):
    """Validate a local source archive without importing benchmark gold labels."""
    path = Path(path)
    raw = path.read_bytes()
    archive = json.loads(raw)
    nodes = [
        n["output"]
        for n in archive["nodes"].values()
        if n.get("kind") == "evidence.retrieve"
    ]
    if len(nodes) != 1:
        raise ValueError("frozen_writing_selection_ambiguous")
    selection = EvidenceSelection.from_dict(nodes[0])
    cards = build_fact_cards(selection.selected_items)
    # This is a new execution snapshot; the historical selection bytes are untouched.
    selection.writing_scope_policy = SCOPE
    requirements = coverage_requirements(archive["query"])
    source_packet(cards, selection, archive["query"], requirements, "zh-CN")
    extracts = next(
        (
            e["data"]["output"]
            for e in archive["events"]
            if e["event"] == "worker.complete"
            and e["data"].get("agent_type") == "extractor"
        ),
        None,
    )
    if not extracts:
        extracts = next(
            (
                n["output"]
                for n in archive["nodes"].values()
                if n.get("agent_type") == "extractor" and n.get("output")
            ),
            None,
        )
    if not extracts:
        raise ValueError("frozen_writing_extractions_missing")
    # Prefer the trusted methodology saved with the writer, never LLM prose.
    methodology = (archive.get("report") or {}).get("metadata", {}).get(
        "review_methodology"
    ) or {}
    if not methodology:
        for event in reversed(archive["events"]):
            data = event["data"]
            if event["event"] == "llm.start":
                for message in data.get("messages", []):
                    if message.get("role") == "user":
                        try:
                            value = json.loads(message.get("content", ""))
                        except (ValueError, TypeError):
                            continue
                        if isinstance(value, dict) and isinstance(
                            value.get("review_methodology"), dict
                        ):
                            methodology = value["review_methodology"]
                            break
                if methodology:
                    break
    if not methodology:
        from litagent.review_structure import build_methodology

        tasks = archive.get("graph", {}).get("tasks", {})
        upstream = {
            n["task_id"]: n["output"]
            for n in archive["nodes"].values()
            if n.get("kind") == "worker" and "output" in n
        }
        upstream["__search_queries__"] = {
            tid: t.get("input", {}).get("query")
            for tid, t in tasks.items()
            if tid.startswith("search_")
        }
        methodology = build_methodology(
            archive["query"],
            upstream,
            "abstract_and_selected_fulltext",
            source_outcomes=(archive.get("report") or {})
            .get("metadata", {})
            .get("search_source_outcomes", []),
            started_at=tasks.get("adversarial_review", {})
            .get("input", {})
            .get("search_started_at"),
        )
        methodology["full_text_review"] = (
            "selected blocks only; complete document review not performed"
        )
    return {
        "query": archive["query"],
        "source_run_id": archive["run_id"],
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "selection": selection.to_dict(),
        "fact_cards_sha256": digest(cards),
        "extractions": extracts,
        "methodology": methodology,
        "requirements": requirements,
        "historical_usage": archive.get("usage", {}),
    }


class FrozenSelector:
    """Supply an immutable selection without executing retrieval or extraction."""

    def __init__(self, bundle, *, trace_hook=None):
        """Initialize immutable inputs and durable ownership."""
        self.bundle = copy.deepcopy(bundle)
        self.trace_hook = trace_hook

    async def select(self, query, ledger, **kwargs):
        """Return the frozen selection only for its original question and ledger."""
        if query != self.bundle["query"]:
            raise ValueError("frozen_writing_query_conflict")
        selection = EvidenceSelection.from_dict(self.bundle["selection"])
        for eid, item in selection.selected_items.items():
            if eid not in ledger or ledger[eid].get("paper_id") != item["paper_id"]:
                raise ValueError("frozen_writing_ledger_conflict")
        if self.trace_hook:
            self.trace_hook(
                "evidence.retrieve.complete",
                {
                    "operation_id": "frozen-selection:" + self.bundle["source_run_id"],
                    "name": "Frozen prior evidence selection",
                    "output": selection.to_dict(),
                    "execution": "reused_frozen_input",
                    "source_run_id": self.bundle["source_run_id"],
                    "source_artifact_sha256": self.bundle.get("archive_sha256"),
                    "elapsed_ms": 0,
                },
            )
        return selection


async def run_frozen(
    agent,
    bundle,
    run_id,
    *,
    transport=None,
    native_reasoning_effort=None,
    native_provider_api="chat",
    native_shape_recovery=False,
    native_reemission_contract=None,
    native_reviewer_thinking_mode=None,
    native_syntax_policy=None,
    native_writing_plan=None,
):
    """Use real Synthesis, Reviewer, scheduler, assembly, Evaluation and acceptance."""
    if agent._wired:
        raise ValueError("frozen_writing_requires_fresh_runtime")
    if native_reviewer_thinking_mode not in (None, "disabled"):
        raise ValueError("native_reviewer_thinking_configuration_invalid")
    writing_plan_spec(native_writing_plan)
    if native_writing_plan is not None and native_syntax_policy != EOF_CONTAINER_POLICY:
        raise ValueError("native_writing_plan_configuration_invalid")
    if native_syntax_policy not in (None, EOF_CONTAINER_POLICY) or (
        native_syntax_policy is not None
        and (native_shape_recovery or native_reemission_contract)
    ):
        raise ValueError("native_syntax_policy_invalid")
    if native_reasoning_effort not in (None, "low") or (
        native_reasoning_effort is not None
        and agent._config.adversarial.thinking_mode != "enabled"
    ):
        raise ValueError("native_reasoning_configuration_invalid")
    if native_provider_api not in ("chat", "responses"):
        raise ValueError("native_provider_api_invalid")
    if type(native_shape_recovery) is not bool:
        raise ValueError("native_shape_recovery_configuration_invalid")
    if native_reemission_contract not in (None, INTACT_UNIT_REEMISSION_CONTRACT) or (
        native_shape_recovery and native_reemission_contract is not None
    ):
        raise ValueError("native_reemission_configuration_invalid")
    selection = EvidenceSelection.from_dict(bundle["selection"])
    cards = build_fact_cards(selection.selected_items)
    if bundle.get("fact_cards_sha256") and digest(cards) != bundle["fact_cards_sha256"]:
        raise ValueError("frozen_writing_fact_card_identity_conflict")
    cfg = agent._config
    options = frozen_options()
    agent._session_id = run_id
    agent._cost_budget = CostBudget(max_tokens=cfg.safety.max_cost_tokens)
    agent._cost_budget.configure_strict(halt_on_unknown_usage=True)
    agent._config_summary = build_config_summary(cfg, "strict_auto_v2")
    if transport is None:
        agent._wire_llms()
    else:
        agent._llm = transport
    generation = agent._stage_llm("generation")
    evaluation = agent._stage_llm("evaluation")
    agent._synthesis = SynthesisWorker(
        generation,
        evidence_selector=FrozenSelector(bundle, trace_hook=agent._trace_hook),
        context_config=cfg.context,
        trace_hook=agent._trace_hook,
        methodology_provider=lambda task: copy.deepcopy(bundle["methodology"]),
        thinking_mode=cfg.adversarial.thinking_mode,
        native_reasoning_effort=native_reasoning_effort,
        native_provider_api=native_provider_api,
        native_shape_recovery=native_shape_recovery,
        native_reemission_contract=native_reemission_contract,
        native_syntax_policy=native_syntax_policy,
        native_writing_plan=native_writing_plan,
        run_id_provider=lambda: agent._session_id,
        repair_state_path_provider=agent._targeted_repair_path,
    )
    # Preserve the writer's immutable mode while explicitly clearing inherited
    # effort for the answer-only Reviewer. Never mutate the shared app config.
    reviewer_config = cfg.adversarial
    reviewer_effort = native_reasoning_effort
    if native_reviewer_thinking_mode == "disabled":
        reviewer_config = cfg.adversarial.model_copy(
            update={"thinking_mode": "disabled"}, deep=True
        )
        reviewer_effort = None
    agent._reviewer = ReviewerWorker(
        evaluation,
        config=reviewer_config,
        native_reasoning_effort=reviewer_effort,
        native_provider_api=native_provider_api,
        context_config=cfg.context,
        trace_hook=agent._trace_hook,
    )
    agent._adversarial = AdversarialReviewWorker(
        generation,
        agent._synthesis,
        agent._reviewer,
        trace_hook=agent._trace_hook,
        thinking_mode=cfg.adversarial.thinking_mode,
    )
    agent._evaluators = [
        CitationEvaluator(evaluation),
        ConsistencyEvaluator(evaluation),
        RagasFaithfulnessEvaluator(cfg, llm=evaluation),
    ]
    graph = TaskGraph()
    graph.add_task(
        SubTask("extract", "Frozen extraction replay", "extractor", max_retries=0)
    )
    graph.mark_done("extract", copy.deepcopy(bundle["extractions"]))
    graph.add_task(
        SubTask(
            "adversarial_review",
            "Frozen writing validation",
            "adversarial_review",
            input_data={
                "query": bundle["query"],
                "coverage_requirements": bundle["requirements"],
            },
            max_retries=0,
            timeout_ms=cfg.adversarial.task_timeout_ms,
        ),
        depends_on=["extract"],
    )
    agent._apply_run_options_to_graph(graph, options)
    agent._scheduler = Scheduler(
        workers=[agent._adversarial],
        max_concurrent=1,
        timeout_ms=cfg.orchestrator.timeout_ms,
        cost_budget=agent._cost_budget,
        trace_hook=agent._trace_hook,
    )
    agent._wired = True
    agent._emit(
        "survey.start",
        {"query": bundle["query"], "session_id": run_id, "report_language": "zh-CN"},
    )
    agent._emit(
        "writing.input.frozen",
        {
            "source_run_id": bundle["source_run_id"],
            "input_sha256": digest(bundle),
            "contract_version": "grounded-writer-v1",
        },
    )
    if hasattr(agent._trace_hook, "capture_graph"):
        agent._trace_hook.capture_graph(graph)
    results = await agent._scheduler.run(graph)
    if hasattr(agent._trace_hook, "capture_graph_state"):
        agent._trace_hook.capture_graph_state(graph)
    report, _ = await agent._prepare_report(
        query=bundle["query"],
        graph=graph,
        results=results,
        options=options,
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )
    await agent._automatic_review(report, bundle["query"], results, None)
    return await agent._finalize_report(
        query=bundle["query"],
        report_data=report,
        results=results,
        degradation_codes=[],
        cancellation=None,
        allow_trusted_writes=False,
    )
