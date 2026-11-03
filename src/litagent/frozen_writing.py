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
from litagent.llm.client import BaseLLMClient
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


def plan_frozen_budget(
    bundle,
    *,
    budget_policy="planned-budget-v2",
    capacity=None,
    inventory_recovery=False,
    mixed_inventory_recovery=False,
    classification_review=False,
    repair_feedback=False,
    reference_contract=None,
    transport_revision=None,
    support_validation_revision=None,
    repair_mapping_contract=None,
    repair_guidance_contract=None,
    repair_guidance_wire_revision=None,
    semantic_conflict_policy=None,
    evaluation_response_contract=None,
    repair_prompt_revision=None,
    round_resource_contract=None,
    writer_source_view=None,
    evaluation_dimensions_contract=None,
    closeout_profile=None,
):
    """Use the same immutable source packet and first wire as ordinary synthesis."""
    from litagent.eval.scoped import SUPPORT_VALIDATION_REVISIONS
    from litagent.writing_v2 import writer_budget_plan

    if (
        closeout_profile is not None
        and round_resource_contract != "joint-round-escrow-v1"
    ):
        raise ValueError("closeout_joint_budget_required")

    if evaluation_dimensions_contract is not None and (
        type(evaluation_dimensions_contract) is not str
        or evaluation_dimensions_contract != "separated-evaluation-dimensions-v1"
        or budget_policy != "planned-budget-v7"
    ):
        raise ValueError("evaluation_dimensions_policy_invalid")

    if writer_source_view is not None and (
        type(writer_source_view) is not str
        or writer_source_view != "source-only-writer-v1"
        or budget_policy != "planned-budget-v7"
    ):
        raise ValueError("writer_source_view_policy_invalid")

    if round_resource_contract is not None:
        from litagent.safety.planned_budget import JOINT_ROUND_ESCROW_CONTRACT

        if (
            type(round_resource_contract) is not str
            or round_resource_contract != JOINT_ROUND_ESCROW_CONTRACT
            or budget_policy != "planned-budget-v7"
        ):
            raise ValueError("round_resource_policy_invalid")

    support_selected = (
        type(support_validation_revision) is str
        and support_validation_revision in SUPPORT_VALIDATION_REVISIONS
    )

    if repair_prompt_revision is not None:
        from litagent.writing_flat import REPAIR_PROMPT_REVISION

        if (
            type(repair_prompt_revision) is not str
            or repair_prompt_revision != REPAIR_PROMPT_REVISION
            or budget_policy != "planned-budget-v7"
        ):
            raise ValueError("repair_prompt_policy_invalid")

    if repair_mapping_contract == "capacity-source-map-v4" and (
        support_validation_revision != "host-support-receipts-v2"
        or repair_guidance_contract != "host-structured-repair-v1"
        or repair_guidance_wire_revision != "host-repair-task-view-v1"
    ):
        raise ValueError("repair_mapping_policy_invalid")

    if repair_guidance_wire_revision is not None:
        from litagent.writing_flat import REPAIR_GUIDANCE_WIRE_REVISIONS

        if (
            type(repair_guidance_wire_revision) is not str
            or repair_guidance_wire_revision not in REPAIR_GUIDANCE_WIRE_REVISIONS
            or repair_guidance_contract != "host-structured-repair-v1"
            or repair_mapping_contract
            not in ("capacity-source-map-v3", "capacity-source-map-v4")
        ):
            raise ValueError("repair_guidance_wire_policy_invalid")
    if repair_guidance_contract is not None and (
        repair_guidance_contract != "host-structured-repair-v1"
        or budget_policy != "planned-budget-v7"
        or repair_mapping_contract
        not in ("capacity-source-map-v3", "capacity-source-map-v4")
        or inventory_recovery is not True
        or mixed_inventory_recovery is not True
        or classification_review is not True
        or repair_feedback is not True
        or reference_contract != "ordered-reference-set-v1"
        or transport_revision != "host-lossless-transport-v1"
        or not support_selected
    ):
        raise ValueError("repair_guidance_policy_invalid")
    if (
        repair_mapping_contract
        not in (
            None,
            "capacity-edit-map-v2",
            "capacity-source-map-v3",
            "capacity-source-map-v4",
        )
        or (
            repair_mapping_contract is not None and budget_policy != "planned-budget-v7"
        )
        or (
            repair_mapping_contract
            in ("capacity-source-map-v3", "capacity-source-map-v4")
            and not support_selected
        )
    ):
        raise ValueError("repair_mapping_policy_invalid")
    if support_validation_revision is not None and (
        not support_selected
        or budget_policy != "planned-budget-v7"
        or inventory_recovery is not True
        or mixed_inventory_recovery is not True
        or classification_review is not True
        or repair_feedback is not True
        or reference_contract != "ordered-reference-set-v1"
        or transport_revision != "host-lossless-transport-v1"
    ):
        raise ValueError("support_validation_policy_invalid")
    if transport_revision is not None and (
        transport_revision != "host-lossless-transport-v1"
        or budget_policy != "planned-budget-v7"
        or mixed_inventory_recovery is not True
        or repair_feedback is not True
        or reference_contract != "ordered-reference-set-v1"
    ):
        raise ValueError("transport_revision_policy_invalid")
    if reference_contract is not None:
        from litagent.writing_flat import REFERENCE_SET_CONTRACT

        if (
            reference_contract != REFERENCE_SET_CONTRACT
            or budget_policy != "planned-budget-v7"
        ):
            raise ValueError("reference_contract_policy_invalid")
    if type(mixed_inventory_recovery) is not bool or (
        mixed_inventory_recovery
        and (budget_policy != "planned-budget-v7" or inventory_recovery is not True)
    ):
        raise ValueError("mixed_inventory_recovery_policy_invalid")
    if type(repair_feedback) is not bool or (
        repair_feedback and budget_policy != "planned-budget-v7"
    ):
        raise ValueError("repair_feedback_policy_invalid")
    selection = EvidenceSelection.from_dict(bundle["selection"])
    cards = build_fact_cards(selection.selected_items)
    if bundle.get("fact_cards_sha256") != digest(cards):
        raise ValueError("frozen_writing_fact_card_identity_conflict")
    packet, aliases = source_packet(
        cards,
        selection,
        bundle["query"],
        bundle["requirements"],
        (
            frozen_options().report_language
            if budget_policy
            in (
                "planned-budget-v4",
                "planned-budget-v5",
                "planned-budget-v6",
                "planned-budget-v7",
            )
            else "zh-CN"
        ),
    )
    methodology = bundle.get("methodology")
    if budget_policy in (
        "planned-budget-v4",
        "planned-budget-v5",
        "planned-budget-v6",
        "planned-budget-v7",
    ):
        from litagent.review_structure import prepare_writer_methodology

        methodology = prepare_writer_methodology(
            methodology,
            selection=selection,
            cards=cards,
            extractions=bundle.get("extractions", []),
            requested_methods=bundle.get("requested_methods", []),
            upstream_results={"extract": bundle.get("extractions", [])},
        )
    plan = writer_budget_plan(
        packet,
        aliases,
        budget_policy=budget_policy,
        methodology=methodology,
        capacity=capacity,
        inventory_recovery=inventory_recovery,
        **({"mixed_inventory_recovery": True} if mixed_inventory_recovery else {}),
        classification_review=classification_review,
        **({"repair_feedback": True} if repair_feedback else {}),
        **(
            {"reference_contract": reference_contract}
            if reference_contract is not None
            else {}
        ),
        **(
            {"transport_revision": transport_revision}
            if transport_revision is not None
            else {}
        ),
        **(
            {"repair_mapping_contract": repair_mapping_contract}
            if repair_mapping_contract is not None
            else {}
        ),
        **(
            {"repair_guidance_contract": repair_guidance_contract}
            if repair_guidance_contract is not None
            else {}
        ),
        **(
            {"repair_guidance_wire_revision": repair_guidance_wire_revision}
            if repair_guidance_wire_revision is not None
            else {}
        ),
        **(
            {"support_validation_revision": support_validation_revision}
            if support_validation_revision is not None
            else {}
        ),
        **(
            {"semantic_conflict_policy": semantic_conflict_policy}
            if semantic_conflict_policy is not None
            else {}
        ),
        **(
            {"evaluation_response_contract": evaluation_response_contract}
            if evaluation_response_contract is not None
            else {}
        ),
        **(
            {"repair_prompt_revision": repair_prompt_revision}
            if repair_prompt_revision is not None
            else {}
        ),
        **(
            {"writer_source_view": writer_source_view}
            if writer_source_view is not None
            else {}
        ),
        **(
            {"evaluation_dimensions_contract": evaluation_dimensions_contract}
            if evaluation_dimensions_contract is not None
            else {}
        ),
        **(
            {"closeout_profile": closeout_profile}
            if closeout_profile is not None
            else {}
        ),
    )
    if round_resource_contract is not None:
        from litagent.safety.planned_budget import with_joint_round_escrow

        plan = with_joint_round_escrow(plan)
    return plan


def validate_frozen_writer_plan(expected_plan: dict, actual_plan: dict) -> None:
    """Reject any prepared-input or allowance drift before activating the owner."""
    from litagent.writing_v2 import WriterError

    try:
        if not isinstance(expected_plan, dict) or not isinstance(actual_plan, dict):
            raise ValueError("plan_type")
        expected = json.dumps(expected_plan, sort_keys=True, allow_nan=False)
        actual = json.dumps(actual_plan, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError):
        raise WriterError("frozen_budget_plan_drift", "identity") from None
    if expected != actual:
        raise WriterError(
            "frozen_budget_plan_drift",
            "identity",
            [
                {
                    "expected_sha256": digest(expected_plan),
                    "actual_sha256": digest(actual_plan),
                }
            ],
        )


class FrozenGuardedClient(BaseLLMClient):
    """Borrow one existing client while checking a server seal around each await."""

    def __init__(self, base, guard):
        """Retain the original transport, budget, model and cleanup ownership."""
        self._base, self._guard = base, guard

    def __getattr__(self, name):
        """Expose the existing reservation and timeout policies without copying."""
        return getattr(self._base, name)

    async def chat(self, messages, **kwargs):
        """Reject drift before sending and preserve a paid response on late drift."""
        self._guard()
        response = await self._base.chat(messages, **kwargs)
        try:
            self._guard()
        except Exception as exc:
            # The underlying owner has already settled known usage. The writer
            # still needs the actual response and receipt in its failed journal.
            exc.response = response
            raise
        return response


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
    budget_policy="legacy",
    expected_budget_plan=None,
    frozen_input_guard=None,
    inventory_recovery=False,
    mixed_inventory_recovery=False,
    classification_review=False,
    repair_feedback=False,
    reference_contract=None,
    transport_revision=None,
    support_validation_revision=None,
    repair_mapping_contract=None,
    repair_guidance_contract=None,
    repair_guidance_wire_revision=None,
    semantic_conflict_policy=None,
    evaluation_response_contract=None,
    repair_prompt_revision=None,
    round_resource_contract=None,
    writer_source_view=None,
    evaluation_dimensions_contract=None,
    closeout_profile=None,
):
    """Use real Synthesis, Reviewer, scheduler, assembly, Evaluation and acceptance."""
    from litagent.closeout import profile_fields
    from litagent.eval.dimensions import selected_dimensions
    from litagent.eval.scoped import SUPPORT_VALIDATION_REVISIONS
    from litagent.safety.planned_budget import JOINT_ROUND_ESCROW_CONTRACT
    from litagent.writer_source_view import selected_source_view

    view_envelope = (expected_budget_plan or {}).get("envelope")
    closeout = profile_fields(view_envelope)
    if closeout_profile != closeout.get("closeout_profile"):
        raise ValueError("closeout_frozen_plan_required")
    dimension_selected = selected_dimensions(view_envelope)
    if (evaluation_dimensions_contract is not None) != dimension_selected or (
        dimension_selected
        and (
            budget_policy != "planned-budget-v7"
            or evaluation_dimensions_contract
            != view_envelope["evaluation_dimensions_contract"]
        )
    ):
        raise ValueError("evaluation_dimensions_frozen_plan_required")
    view_selected = selected_source_view(view_envelope)
    if (writer_source_view is not None) != view_selected or (
        view_selected
        and (
            budget_policy != "planned-budget-v7"
            or writer_source_view != view_envelope["writer_source_view"]
        )
    ):
        raise ValueError("writer_source_view_frozen_plan_required")

    resource_descriptor = (expected_budget_plan or {}).get("joint_round_escrow")
    if round_resource_contract is not None or resource_descriptor is not None:
        if (
            type(round_resource_contract) is not str
            or round_resource_contract != JOINT_ROUND_ESCROW_CONTRACT
            or budget_policy != "planned-budget-v7"
            or not isinstance(resource_descriptor, dict)
            or resource_descriptor.get("contract") != round_resource_contract
        ):
            raise ValueError("round_resource_frozen_plan_required")

    support_selected = (
        type(support_validation_revision) is str
        and support_validation_revision in SUPPORT_VALIDATION_REVISIONS
    )

    from litagent.eval.batched import selected_response_contract

    response_envelope = (expected_budget_plan or {}).get("envelope")
    response_selected = selected_response_contract(response_envelope)
    if (evaluation_response_contract is not None) != response_selected or (
        response_selected
        and (
            budget_policy != "planned-budget-v7"
            or evaluation_response_contract
            != response_envelope["evaluation_response_contract"]
        )
    ):
        raise ValueError("evaluation_response_frozen_plan_required")

    from litagent.writing_flat import selected_repair_prompt

    prompt_selected = selected_repair_prompt(response_envelope)
    if (repair_prompt_revision is not None) != prompt_selected or (
        prompt_selected
        and (
            budget_policy != "planned-budget-v7"
            or repair_prompt_revision != response_envelope["repair_prompt_revision"]
        )
    ):
        raise ValueError("repair_prompt_frozen_plan_required")
    if prompt_selected:
        from litagent.safety.planned_budget import validate_writer_envelope

        validate_writer_envelope(response_envelope)

    if semantic_conflict_policy is not None:
        from litagent.eval.semantic_conflict import selected_policy

        envelope = (expected_budget_plan or {}).get("envelope", {})
        if (
            budget_policy != "planned-budget-v7"
            or not selected_policy(envelope)
            or semantic_conflict_policy != envelope.get("semantic_conflict_policy")
        ):
            raise ValueError("semantic_conflict_frozen_plan_required")

    if repair_mapping_contract == "capacity-source-map-v4" and (
        support_validation_revision != "host-support-receipts-v2"
        or repair_guidance_contract != "host-structured-repair-v1"
        or repair_guidance_wire_revision != "host-repair-task-view-v1"
    ):
        raise ValueError("repair_mapping_policy_invalid")

    if repair_guidance_wire_revision is not None and expected_budget_plan is None:
        raise ValueError("repair_guidance_wire_frozen_plan_required")
    if repair_guidance_contract is not None and expected_budget_plan is None:
        raise ValueError("repair_guidance_frozen_plan_required")
    if (
        repair_mapping_contract in ("capacity-source-map-v3", "capacity-source-map-v4")
        and expected_budget_plan is None
    ):
        raise ValueError("repair_mapping_frozen_plan_required")
    if agent._wired:
        raise ValueError("frozen_writing_requires_fresh_runtime")
    if repair_guidance_wire_revision is not None:
        from litagent.writing_flat import REPAIR_GUIDANCE_WIRE_REVISIONS

        if (
            type(repair_guidance_wire_revision) is not str
            or repair_guidance_wire_revision not in REPAIR_GUIDANCE_WIRE_REVISIONS
            or repair_guidance_contract != "host-structured-repair-v1"
            or repair_mapping_contract
            not in ("capacity-source-map-v3", "capacity-source-map-v4")
        ):
            raise ValueError("repair_guidance_wire_policy_invalid")
    if repair_guidance_contract is not None and (
        repair_guidance_contract != "host-structured-repair-v1"
        or budget_policy != "planned-budget-v7"
        or repair_mapping_contract
        not in ("capacity-source-map-v3", "capacity-source-map-v4")
        or inventory_recovery is not True
        or mixed_inventory_recovery is not True
        or classification_review is not True
        or repair_feedback is not True
        or reference_contract != "ordered-reference-set-v1"
        or transport_revision != "host-lossless-transport-v1"
        or not support_selected
    ):
        raise ValueError("repair_guidance_policy_invalid")
    if (
        repair_mapping_contract
        not in (
            None,
            "capacity-edit-map-v2",
            "capacity-source-map-v3",
            "capacity-source-map-v4",
        )
        or (
            repair_mapping_contract is not None and budget_policy != "planned-budget-v7"
        )
        or (
            repair_mapping_contract
            in ("capacity-source-map-v3", "capacity-source-map-v4")
            and not support_selected
        )
    ):
        raise ValueError("repair_mapping_policy_invalid")
    if support_validation_revision is not None and (
        not support_selected
        or budget_policy != "planned-budget-v7"
        or inventory_recovery is not True
        or mixed_inventory_recovery is not True
        or classification_review is not True
        or repair_feedback is not True
        or reference_contract != "ordered-reference-set-v1"
        or transport_revision != "host-lossless-transport-v1"
    ):
        raise ValueError("support_validation_policy_invalid")
    if transport_revision is not None and (
        transport_revision != "host-lossless-transport-v1"
        or budget_policy != "planned-budget-v7"
        or mixed_inventory_recovery is not True
        or repair_feedback is not True
        or reference_contract != "ordered-reference-set-v1"
    ):
        raise ValueError("transport_revision_policy_invalid")
    if reference_contract is not None:
        from litagent.writing_flat import REFERENCE_SET_CONTRACT

        if (
            reference_contract != REFERENCE_SET_CONTRACT
            or budget_policy != "planned-budget-v7"
        ):
            raise ValueError("reference_contract_policy_invalid")
    if type(inventory_recovery) is not bool or (
        inventory_recovery and budget_policy != "planned-budget-v7"
    ):
        raise ValueError("inventory_recovery_policy_invalid")
    if type(mixed_inventory_recovery) is not bool or (
        mixed_inventory_recovery
        and (budget_policy != "planned-budget-v7" or inventory_recovery is not True)
    ):
        raise ValueError("mixed_inventory_recovery_policy_invalid")
    if type(classification_review) is not bool or (
        classification_review and budget_policy != "planned-budget-v7"
    ):
        raise ValueError("classification_review_policy_invalid")
    if type(repair_feedback) is not bool or (
        repair_feedback and budget_policy != "planned-budget-v7"
    ):
        raise ValueError("repair_feedback_policy_invalid")
    if expected_budget_plan is not None and budget_policy not in (
        "planned-budget-v4",
        "planned-budget-v5",
        "planned-budget-v6",
        "planned-budget-v7",
    ):
        raise ValueError("frozen_budget_plan_policy_conflict")
    if frozen_input_guard is not None:
        if budget_policy not in (
            "planned-budget-v4",
            "planned-budget-v5",
            "planned-budget-v6",
            "planned-budget-v7",
        ) or not callable(frozen_input_guard):
            raise ValueError("frozen_input_guard_invalid")
        frozen_input_guard()
    # Caller-owned dictionaries may change while selection or HTTP is awaited.
    # Only the validated server manifest supplies this optional certificate.
    if budget_policy in (
        "planned-budget-v4",
        "planned-budget-v5",
        "planned-budget-v6",
        "planned-budget-v7",
    ):
        bundle = copy.deepcopy(bundle)
        expected_budget_plan = copy.deepcopy(expected_budget_plan)
        if expected_budget_plan is not None:
            validate_frozen_writer_plan(
                expected_budget_plan,
                plan_frozen_budget(
                    bundle,
                    budget_policy=budget_policy,
                    capacity=agent._config.capacity,
                    inventory_recovery=inventory_recovery,
                    **(
                        {"mixed_inventory_recovery": True}
                        if mixed_inventory_recovery
                        else {}
                    ),
                    classification_review=classification_review,
                    **({"repair_feedback": True} if repair_feedback else {}),
                    **(
                        {"reference_contract": reference_contract}
                        if reference_contract is not None
                        else {}
                    ),
                    **(
                        {"transport_revision": transport_revision}
                        if transport_revision is not None
                        else {}
                    ),
                    **(
                        {"repair_mapping_contract": repair_mapping_contract}
                        if repair_mapping_contract is not None
                        else {}
                    ),
                    **(
                        {"repair_guidance_contract": repair_guidance_contract}
                        if repair_guidance_contract is not None
                        else {}
                    ),
                    **(
                        {"repair_guidance_wire_revision": repair_guidance_wire_revision}
                        if repair_guidance_wire_revision is not None
                        else {}
                    ),
                    **(
                        {"support_validation_revision": support_validation_revision}
                        if support_validation_revision is not None
                        else {}
                    ),
                    **(
                        {"semantic_conflict_policy": semantic_conflict_policy}
                        if semantic_conflict_policy is not None
                        else {}
                    ),
                    **(
                        {"evaluation_response_contract": evaluation_response_contract}
                        if evaluation_response_contract is not None
                        else {}
                    ),
                    **(
                        {"repair_prompt_revision": repair_prompt_revision}
                        if repair_prompt_revision is not None
                        else {}
                    ),
                    **(
                        {"round_resource_contract": round_resource_contract}
                        if round_resource_contract is not None
                        else {}
                    ),
                    **(
                        {"writer_source_view": writer_source_view}
                        if writer_source_view is not None
                        else {}
                    ),
                    **(
                        {"closeout_profile": closeout_profile}
                        if closeout_profile is not None
                        else {}
                    ),
                    **(
                        {
                            "evaluation_dimensions_contract": evaluation_dimensions_contract
                        }
                        if evaluation_dimensions_contract is not None
                        else {}
                    ),
                ),
            )
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
    if budget_policy != "legacy":
        options = RunOptions(
            **{
                **options.model_dump(),
                "budget_policy": budget_policy,
                **({"classification_review": True} if classification_review else {}),
                **({"repair_feedback": True} if repair_feedback else {}),
                **(
                    {"reference_contract": reference_contract}
                    if reference_contract is not None
                    else {}
                ),
                **(
                    {"transport_revision": transport_revision}
                    if transport_revision is not None
                    else {}
                ),
                **(
                    {"support_validation_revision": support_validation_revision}
                    if support_validation_revision is not None
                    else {}
                ),
                **(
                    {"mixed_inventory_recovery": True}
                    if mixed_inventory_recovery
                    else {}
                ),
                **(
                    {"capacity_policy": "adaptive-capacity-v1"}
                    if budget_policy in ("planned-budget-v6", "planned-budget-v7")
                    else {}
                ),
                **(
                    {"generation_strategy": "body_summary_v3"}
                    if budget_policy
                    in ("planned-budget-v5", "planned-budget-v6", "planned-budget-v7")
                    else {}
                ),
            }
        )
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
    if frozen_input_guard is not None:
        generation = FrozenGuardedClient(generation, frozen_input_guard)
        evaluation = FrozenGuardedClient(evaluation, frozen_input_guard)
        agent._stage_llms.update(generation=generation, evaluation=evaluation)
    agent._synthesis = SynthesisWorker(
        generation,
        evidence_selector=FrozenSelector(bundle, trace_hook=agent._trace_hook),
        context_config=cfg.context,
        trace_hook=agent._trace_hook,
        methodology_provider=lambda task: copy.deepcopy(bundle["methodology"]),
        frozen_writer_plan_provider=(
            (lambda: copy.deepcopy(expected_budget_plan))
            if expected_budget_plan is not None
            else None
        ),
        capacity_config=cfg.capacity,
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
                **(
                    {"requested_methods": bundle.get("requested_methods", [])}
                    if budget_policy
                    in (
                        "planned-budget-v4",
                        "planned-budget-v5",
                        "planned-budget-v6",
                        "planned-budget-v7",
                    )
                    else {}
                ),
            },
            max_retries=0,
            timeout_ms=cfg.adversarial.task_timeout_ms,
        ),
        depends_on=["extract"],
    )
    agent._apply_run_options_to_graph(graph, options)
    if semantic_conflict_policy is not None:
        # The verified frozen plan owns selection; mutable run context does not.
        locked = expected_budget_plan["envelope"]
        graph.tasks["adversarial_review"].input_data.update(
            {
                key: locked[key]
                for key in ("semantic_conflict_policy", "semantic_conflict_identity")
            }
        )
    if response_selected:
        # Only the verified plan can grant the optional response contract.
        locked = expected_budget_plan["envelope"]
        graph.tasks["adversarial_review"].input_data.update(
            {
                key: locked[key]
                for key in (
                    "evaluation_response_contract",
                    "evaluation_response_identity",
                )
            }
        )
    if dimension_selected:
        # Only the authenticated frozen envelope grants evaluation dimensions.
        from litagent.eval.dimensions import dimensions_fields

        graph.tasks["adversarial_review"].input_data.update(
            dimensions_fields(expected_budget_plan["envelope"])
        )
    if view_selected:
        # The verified frozen plan alone grants the model source view.
        locked = expected_budget_plan["envelope"]
        graph.tasks["adversarial_review"].input_data.update(
            {
                key: locked[key]
                for key in ("writer_source_view", "writer_source_view_identity")
            }
        )
    if round_resource_contract is not None:
        graph.tasks["adversarial_review"].input_data[
            "round_resource_contract"
        ] = round_resource_contract
    if prompt_selected:
        # The frozen plan alone selects repair instructions; tasks carry its pair.
        locked = expected_budget_plan["envelope"]
        graph.tasks["adversarial_review"].input_data.update(
            {
                key: locked[key]
                for key in ("repair_prompt_revision", "repair_prompt_identity")
            }
        )
    if repair_mapping_contract is not None:
        graph.tasks["adversarial_review"].input_data[
            "repair_mapping_contract"
        ] = repair_mapping_contract
    if repair_guidance_contract is not None:
        graph.tasks["adversarial_review"].input_data[
            "repair_guidance_contract"
        ] = repair_guidance_contract
    if repair_guidance_wire_revision is not None:
        graph.tasks["adversarial_review"].input_data[
            "repair_guidance_wire_revision"
        ] = repair_guidance_wire_revision
    if inventory_recovery:
        graph.tasks["adversarial_review"].input_data["inventory_recovery"] = True
    if mixed_inventory_recovery:
        graph.tasks["adversarial_review"].input_data["mixed_inventory_recovery"] = True
    if budget_policy in (
        "planned-budget-v4",
        "planned-budget-v5",
        "planned-budget-v6",
        "planned-budget-v7",
    ):
        # The generic graph builder derives requirements from the question. A
        # server-frozen campaign already owns the exact reviewed requirements.
        graph.tasks["adversarial_review"].input_data["coverage_requirements"] = (
            copy.deepcopy(bundle["requirements"])
        )
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
    if frozen_input_guard is not None:
        frozen_input_guard()
    failed = graph.tasks["adversarial_review"]
    if budget_policy in (
        "planned-budget-v4",
        "planned-budget-v5",
        "planned-budget-v6",
        "planned-budget-v7",
    ) and failed.error in {
        "frozen_budget_plan_drift",
        "frozen_campaign_source_changed",
    }:
        from litagent.writing_v2 import WriterError

        raise WriterError(failed.error, "identity")
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
    if frozen_input_guard is not None:
        frozen_input_guard()
    return await agent._finalize_report(
        query=bundle["query"],
        report_data=report,
        results=results,
        degradation_codes=[],
        cancellation=None,
        allow_trusted_writes=False,
    )
