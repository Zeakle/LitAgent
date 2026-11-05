"""Plan and persist complete bounded workflows before sending paid requests."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Callable

from litagent.safety.budget import CostBudget, strict_request_reservation

CONTRACT = "planned-budget-v2"
ADAPTIVE_CONTRACT = "planned-budget-v3"
ENVELOPE_BUDGET_CONTRACT = "planned-budget-v4"
FLAT_BUDGET_CONTRACT = "planned-budget-v5"
CAPACITY_BUDGET_CONTRACT = "planned-budget-v6"
ROUND_BUDGET_CONTRACT = "planned-budget-v7"
JOINT_ROUND_ESCROW_CONTRACT = "joint-round-escrow-v1"
CAPACITY_BUDGETS = (CAPACITY_BUDGET_CONTRACT, ROUND_BUDGET_CONTRACT)
FLAT_BUDGETS = (FLAT_BUDGET_CONTRACT, *CAPACITY_BUDGETS)
ENVELOPE_CONTRACT = "budgeted-writer-envelope-v1"
FLAT_ENVELOPE_CONTRACT = "flat-writer-envelope-v1"
ENVELOPE_BUDGETS = (ENVELOPE_BUDGET_CONTRACT, *FLAT_BUDGETS)
CONTRACTS = (CONTRACT, ADAPTIVE_CONTRACT, *ENVELOPE_BUDGETS)
CONTEXT_LIMIT = int(131072 * 0.9)
_active_request: ContextVar[tuple[str, str] | None] = ContextVar(
    "planned_budget_request", default=None
)


def _digest(value):
    """Bind public plans without embedding source materials."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def nested_json_size(value) -> int:
    """Measure a complete JSON value as embedded in the outer message JSON."""
    return (
        len(
            json.dumps(
                json.dumps(value, ensure_ascii=False), ensure_ascii=False
            ).encode()
        )
        - 2
    )


class EnvelopeExceeded(ValueError):
    """Reject output outside declared scope without calling it a budget shortfall."""

    def __init__(self, stage, field, actual, maximum):
        super().__init__(
            "evaluation_envelope_exceeded"
            if stage in {"inventory", "assessment", "global"}
            else "writer_envelope_exceeded"
        )
        self.diagnostics = [
            {"stage": stage, "field": field, "actual": actual, "maximum": maximum}
        ]


def validate_writer_envelope(value: dict) -> dict:
    """Validate a complete server-selected output and derived-request contract."""
    fields = {
        "contract",
        "body_max_units",
        "summary_max_units_per_section",
        "model_assertions",
        "assessment_batches",
        "body_json_bytes",
        "summary_json_bytes",
        "claim_json_bytes",
        "request_caps",
        "source_identity",
        "builder_identity",
        "frozen_component_reservation",
        "identity",
    }
    if isinstance(value, dict) and "output_tokens" in value:
        fields.add("output_tokens")
        if value.get("contract") != FLAT_ENVELOPE_CONTRACT or (
            "repair_mapping_contract" not in value
        ):
            raise ValueError("stage_output_capacity_policy_invalid")
    if isinstance(value, dict) and value.get("contract") == FLAT_ENVELOPE_CONTRACT:
        fields.update(("diagnostic_json_bytes", "claim_max_json_bytes"))
    if isinstance(value, dict) and {
        "closeout_profile",
        "closeout_profile_identity",
    }.intersection(value):
        from litagent.closeout import FIELDS, profile_fields

        fields.update(FIELDS)
        profile_fields(value)
    adaptive = isinstance(value, dict) and "capacity" in value
    if adaptive:
        fields.add("capacity")
    if isinstance(value, dict) and "repair_mapping_contract" in value:
        fields.add("repair_mapping_contract")
        if not adaptive or value["repair_mapping_contract"] not in (
            "capacity-edit-map-v2",
            "capacity-source-map-v3",
            "capacity-source-map-v4",
        ):
            raise ValueError("writer_envelope_mapping_invalid")
    if isinstance(value, dict) and "global_findings_contract" in value:
        fields.add("global_findings_contract")
        if value.get("repair_mapping_contract") not in (
            "capacity-edit-map-v2",
            "capacity-source-map-v3",
            "capacity-source-map-v4",
        ) or value["global_findings_contract"] not in (
            "located-global-findings-v1",
            "anchored-global-findings-v2",
        ):
            raise ValueError("writer_envelope_findings_invalid")
    if isinstance(value, dict) and "inventory_recovery_contract" in value:
        from litagent.eval.inventory_recovery import RECOVERY_CONTRACTS

        fields.add("inventory_recovery_contract")
        if (
            value.get("global_findings_contract") != "anchored-global-findings-v2"
            or not isinstance(value["inventory_recovery_contract"], str)
            or value["inventory_recovery_contract"] not in RECOVERY_CONTRACTS
        ):
            raise ValueError("writer_envelope_recovery_invalid")
    if isinstance(value, dict) and "summary_constraints_contract" in value:
        fields.add("summary_constraints_contract")
        if (
            value.get("contract") != FLAT_ENVELOPE_CONTRACT
            or value.get("repair_mapping_contract")
            not in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            )
            or value["summary_constraints_contract"] != "effective-summary-limits-v1"
        ):
            raise ValueError("writer_envelope_summary_constraints_invalid")
    if isinstance(value, dict) and "classification_review_contract" in value:
        from litagent.eval.scoped import INDEPENDENT_CLASSIFICATION_CONTRACT

        fields.add("classification_review_contract")
        if (
            value.get("summary_constraints_contract") != "effective-summary-limits-v1"
            or value.get("global_findings_contract") != "anchored-global-findings-v2"
            or value["classification_review_contract"]
            != INDEPENDENT_CLASSIFICATION_CONTRACT
        ):
            raise ValueError("writer_envelope_classification_review_invalid")
    if isinstance(value, dict) and "repair_feedback_contract" in value:
        fields.add("repair_feedback_contract")
        if (
            not adaptive
            or value.get("repair_mapping_contract")
            not in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            )
            or value["repair_feedback_contract"] != "host-repair-feedback-v1"
        ):
            raise ValueError("writer_envelope_repair_feedback_invalid")
    if isinstance(value, dict) and "reference_contract" in value:
        from litagent.writing_flat import REFERENCE_SET_CONTRACT

        fields.add("reference_contract")
        if (
            value.get("contract") != FLAT_ENVELOPE_CONTRACT
            or not adaptive
            or value.get("repair_mapping_contract")
            not in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            )
            or value["reference_contract"] != REFERENCE_SET_CONTRACT
        ):
            raise ValueError("writer_envelope_reference_contract_invalid")
    if isinstance(value, dict) and (
        "transport_revision" in value or "transport_identity" in value
    ):
        from litagent.writing_flat import transport_identity

        fields.update({"transport_revision", "transport_identity"})
        if (
            value.get("contract") != FLAT_ENVELOPE_CONTRACT
            or not adaptive
            or value.get("repair_mapping_contract")
            not in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            )
            or value.get("inventory_recovery_contract")
            != "bounded-inventory-recovery-v2"
            or value.get("repair_feedback_contract") != "host-repair-feedback-v1"
            or value.get("reference_contract") != "ordered-reference-set-v1"
            or value.get("transport_revision") != "host-lossless-transport-v1"
            or value.get("transport_identity") != transport_identity()
        ):
            raise ValueError("writer_envelope_transport_revision_invalid")
    if isinstance(value, dict) and (
        "support_validation_revision" in value or "support_validation_identity" in value
    ):
        from litagent.eval.scoped import (
            SUPPORT_VALIDATION_REVISIONS,
            support_validation_identity,
        )

        fields.update({"support_validation_revision", "support_validation_identity"})
        if (
            value.get("contract") != FLAT_ENVELOPE_CONTRACT
            or not adaptive
            or value.get("repair_mapping_contract")
            not in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            )
            or value.get("inventory_recovery_contract")
            != "bounded-inventory-recovery-v2"
            or value.get("classification_review_contract")
            != "independent-assertion-kinds-v1"
            or value.get("repair_feedback_contract") != "host-repair-feedback-v1"
            or value.get("reference_contract") != "ordered-reference-set-v1"
            or value.get("transport_revision") != "host-lossless-transport-v1"
            or type(value.get("support_validation_revision")) is not str
            or value["support_validation_revision"] not in SUPPORT_VALIDATION_REVISIONS
            or value.get("support_validation_identity")
            != (
                support_validation_identity()
                if value["support_validation_revision"] == "host-support-receipts-v1"
                else support_validation_identity(value["support_validation_revision"])
            )
        ):
            raise ValueError("writer_envelope_support_validation_invalid")
    source_map = isinstance(value, dict) and value.get("repair_mapping_contract") in (
        "capacity-source-map-v3",
        "capacity-source-map-v4",
    )
    if (
        source_map
        or isinstance(value, dict)
        and ("source_map_identity" in value or "source_map_json_bytes" in value)
    ):
        from litagent.writing_flat import source_map_wire_identity

        fields.update({"source_map_identity", "source_map_json_bytes"})
        if (
            not source_map
            or (
                type(value.get("support_validation_revision")) is not str
                or value["support_validation_revision"]
                not in SUPPORT_VALIDATION_REVISIONS
            )
            or value.get("source_map_identity")
            != (
                source_map_wire_identity()
                if value.get("repair_mapping_contract") == "capacity-source-map-v3"
                else source_map_wire_identity(contract=value["repair_mapping_contract"])
            )
            or type(value.get("source_map_json_bytes")) is not int
            or value["source_map_json_bytes"] != value.get("diagnostic_json_bytes")
        ):
            raise ValueError("writer_envelope_source_map_invalid")
    guidance_fields = {
        "repair_guidance_contract",
        "repair_guidance_identity",
        "repair_guidance_json_bytes",
    }
    if isinstance(value, dict) and guidance_fields.intersection(value):
        from litagent.writing_flat import (
            STRUCTURED_REPAIR_GUIDANCE_CONTRACT,
            structured_repair_guidance_identity,
        )

        fields.update(guidance_fields)
        if (
            not guidance_fields.issubset(value)
            or value.get("contract") != FLAT_ENVELOPE_CONTRACT
            or not adaptive
            or not source_map
            or value.get("global_findings_contract") != "anchored-global-findings-v2"
            or value.get("summary_constraints_contract")
            != "effective-summary-limits-v1"
            or value.get("inventory_recovery_contract")
            != "bounded-inventory-recovery-v2"
            or value.get("classification_review_contract")
            != "independent-assertion-kinds-v1"
            or value.get("repair_feedback_contract") != "host-repair-feedback-v1"
            or value.get("reference_contract") != "ordered-reference-set-v1"
            or value.get("transport_revision") != "host-lossless-transport-v1"
            or (
                type(value.get("support_validation_revision")) is not str
                or value["support_validation_revision"]
                not in SUPPORT_VALIDATION_REVISIONS
            )
            or value.get("repair_guidance_contract")
            != STRUCTURED_REPAIR_GUIDANCE_CONTRACT
            or value.get("repair_guidance_identity")
            != structured_repair_guidance_identity()
            or type(value.get("repair_guidance_json_bytes")) is not int
            or not 1 <= value["repair_guidance_json_bytes"] <= 12288
            or value["repair_guidance_json_bytes"] != value.get("diagnostic_json_bytes")
        ):
            raise ValueError("writer_envelope_repair_guidance_invalid")
    guidance_wire_fields = {
        "repair_guidance_wire_revision",
        "repair_guidance_wire_identity",
    }
    if isinstance(value, dict) and (
        guidance_wire_fields | {"repair_guidance_capacity_rule"}
    ).intersection(value):
        from litagent.writing_flat import (
            REPAIR_GUIDANCE_WIRE_REVISION,
            REPAIR_GUIDANCE_WIRE_REVISIONS,
            REPAIR_TASK_VIEW_ADMISSION_RULE,
            REPAIR_TASK_VIEW_REVISION,
            repair_guidance_wire_identity,
        )

        fields.update(guidance_wire_fields)
        revision = value.get("repair_guidance_wire_revision")
        task_view = revision == REPAIR_TASK_VIEW_REVISION
        if task_view:
            fields.add("repair_guidance_capacity_rule")
        if (
            not guidance_wire_fields.issubset(value)
            or value.get("repair_guidance_contract") != "host-structured-repair-v1"
            or value.get("repair_mapping_contract")
            not in ("capacity-source-map-v3", "capacity-source-map-v4")
            or type(revision) is not str
            or revision not in REPAIR_GUIDANCE_WIRE_REVISIONS
            or value.get("repair_guidance_wire_identity")
            != (
                repair_guidance_wire_identity()
                if revision == REPAIR_GUIDANCE_WIRE_REVISION
                else repair_guidance_wire_identity(revision)
            )
            or ("repair_guidance_capacity_rule" in value) is not task_view
            or task_view
            and value["repair_guidance_capacity_rule"]
            != REPAIR_TASK_VIEW_ADMISSION_RULE
        ):
            raise ValueError("writer_envelope_repair_guidance_wire_invalid")
    if (
        isinstance(value, dict)
        and value.get("repair_mapping_contract") == "capacity-source-map-v4"
        and (
            value.get("support_validation_revision") != "host-support-receipts-v2"
            or value.get("repair_guidance_contract") != "host-structured-repair-v1"
            or value.get("repair_guidance_wire_revision") != "host-repair-task-view-v1"
        )
    ):
        raise ValueError("writer_envelope_source_owner_invalid")
    if isinstance(value, dict) and {
        "semantic_conflict_policy",
        "semantic_conflict_identity",
    }.intersection(value):
        from litagent.eval.semantic_conflict import selected_policy

        fields.update(("semantic_conflict_policy", "semantic_conflict_identity"))
        selected_policy(value)
    if isinstance(value, dict) and {
        "evaluation_response_contract",
        "evaluation_response_identity",
    }.intersection(value):
        from litagent.eval.batched import selected_response_contract

        fields.update(("evaluation_response_contract", "evaluation_response_identity"))
        selected_response_contract(value)
    if isinstance(value, dict) and {
        "writer_source_view",
        "writer_source_view_identity",
    }.intersection(value):
        from litagent.writer_source_view import selected_source_view

        fields.update(("writer_source_view", "writer_source_view_identity"))
        selected_source_view(value)
    if isinstance(value, dict) and {
        "evaluation_dimensions_contract",
        "evaluation_dimensions_identity",
    }.intersection(value):
        from litagent.eval.dimensions import selected_dimensions

        fields.update(
            ("evaluation_dimensions_contract", "evaluation_dimensions_identity")
        )
        selected_dimensions(value)
    if isinstance(value, dict) and {
        "evaluation_result_contract",
        "evaluation_result_identity",
    }.intersection(value):
        from litagent.eval.results import FIELDS, selected

        fields.update(FIELDS)
        selected(value)
    if isinstance(value, dict) and {
        "assessment_decision_contract",
        "assessment_decision_identity",
    }.intersection(value):
        from litagent.eval.decision_output import FIELDS, selected

        fields.update(FIELDS)
        selected(value)
    if isinstance(value, dict) and {
        "repair_prompt_revision",
        "repair_prompt_identity",
    }.intersection(value):
        from litagent.writing_flat import selected_repair_prompt

        fields.update(("repair_prompt_revision", "repair_prompt_identity"))
        selected_repair_prompt(value)
        if (
            value.get("contract") != FLAT_ENVELOPE_CONTRACT
            or not adaptive
            or value.get("repair_mapping_contract") != "capacity-source-map-v4"
            or value.get("repair_guidance_contract") != "host-structured-repair-v1"
            or value.get("repair_guidance_wire_revision") != "host-repair-task-view-v1"
            or value.get("support_validation_revision") != "host-support-receipts-v2"
        ):
            raise ValueError("writer_envelope_repair_prompt_invalid")
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("writer_envelope_invalid")
    limits = {
        "body_max_units": (5, 66),
        "summary_max_units_per_section": (1, 3),
        "model_assertions": (1, 128),
        "assessment_batches": (1, 8),
        "body_json_bytes": (1, 1000000),
        "summary_json_bytes": (1, 1000000),
        "claim_json_bytes": (1, 1000000),
        "frozen_component_reservation": (0, 2**63 - 1),
    }
    if value.get("contract") == FLAT_ENVELOPE_CONTRACT:
        limits["diagnostic_json_bytes"] = (1, 12288)
        limits["claim_max_json_bytes"] = (1, 2000)
    batch_size = 16
    if adaptive:
        from litagent.capacity import resolve_capacity

        capacity = resolve_capacity(value["capacity"])
        batch_size = capacity.batch_size
        limits.update(
            body_max_units=(5, 256),
            model_assertions=(1, 1024),
            assessment_batches=(1, 64),
        )
        if (
            value["body_max_units"] != capacity.max_body_units
            or value["summary_max_units_per_section"] != capacity.max_summary_units
            or value["model_assertions"] < capacity.initial_assertions
            or (
                capacity.max_assertions is not None
                and value["model_assertions"] > capacity.max_assertions
            )
            or (
                capacity.max_assessment_batches is not None
                and value["assessment_batches"] > capacity.max_assessment_batches
            )
        ):
            raise ValueError("capacity_envelope_configuration_conflict")
    outputs = {
        "body": 20000,
        "summary": 20000,
        "reviewer": 4096,
        "repair": 20000,
        "inventory": 16384,
        "assessment": 16384,
        "global": 16384,
    }
    if "output_tokens" in value:
        from litagent.output_capacity import envelope_output_tokens

        outputs = envelope_output_tokens(value)
    if (
        value["contract"] not in (ENVELOPE_CONTRACT, FLAT_ENVELOPE_CONTRACT)
        or any(
            type(value[k]) is not int or not lo <= value[k] <= hi
            for k, (lo, hi) in limits.items()
        )
        or value["body_max_units"] + 2 * value["summary_max_units_per_section"]
        > value["model_assertions"]
        or value["model_assertions"] > batch_size * value["assessment_batches"]
        or not isinstance(value["request_caps"], dict)
        or set(value["request_caps"]) != set(outputs)
        or any(
            type(value["request_caps"][k]) is not int
            or not out <= value["request_caps"][k] <= 2**63 - 1
            for k, out in outputs.items()
        )
        or any(
            not isinstance(value[k], str)
            or len(value[k]) != 64
            or any(c not in "0123456789abcdef" for c in value[k])
            for k in ("source_identity", "builder_identity")
        )
        or value["identity"]
        != _digest({k: v for k, v in value.items() if k != "identity"})
    ):
        raise ValueError("writer_envelope_invalid")
    return copy.deepcopy(value)


def check_writer_envelope(document, envelope, *, stage):
    """Reject full authored outputs; never truncate or discard over-limit units."""
    envelope = validate_writer_envelope(envelope)
    sections = (
        ("abstract", "conclusion")
        if stage == "summary"
        else ("introduction", "taxonomy", "methods", "experiments", "open_problems")
    )
    counts = {
        k: sum(len(p["sentences"]) for p in document.get(k, [])) for k in sections
    }
    if stage == "summary":
        for key, count in counts.items():
            if count > envelope["summary_max_units_per_section"]:
                raise EnvelopeExceeded(
                    stage,
                    key + ".units",
                    count,
                    envelope["summary_max_units_per_section"],
                )
    else:
        maximum = envelope["body_max_units"] - sum(k not in document for k in sections)
        if sum(counts.values()) > maximum:
            raise EnvelopeExceeded(stage, "body.units", sum(counts.values()), maximum)
    if "capacity" in envelope:
        max_chars = envelope["capacity"]["max_unit_characters"]
        for key in sections:
            for paragraph in document.get(key, []):
                for unit in paragraph["sentences"]:
                    if len(unit["text"]) > max_chars:
                        raise EnvelopeExceeded(
                            stage, "unit.characters", len(unit["text"]), max_chars
                        )
    size = nested_json_size(document)
    limit = envelope["summary_json_bytes" if stage == "summary" else "body_json_bytes"]
    if size > limit:
        raise EnvelopeExceeded(stage, "nested_json_bytes", size, limit)


def check_inventory_envelope(claims, envelope):
    """Check the entire extracted list before any scientific batch is sent."""
    envelope = validate_writer_envelope(envelope)
    maximum = envelope["model_assertions"]
    if "capacity" in envelope:
        cap = envelope["capacity"]
        maximum = cap["max_assertions"] or 64 * cap["batch_size"]
    if len(claims) > maximum:
        raise EnvelopeExceeded("inventory", "model_assertions", len(claims), maximum)
    size = nested_json_size([c["claim_text"] for c in claims])
    if envelope["contract"] == FLAT_ENVELOPE_CONTRACT:
        maximum = max((nested_json_size(c["claim_text"]) for c in claims), default=0)
        if maximum > envelope["claim_max_json_bytes"]:
            raise EnvelopeExceeded(
                "inventory",
                "claim_max_json_bytes",
                maximum,
                envelope["claim_max_json_bytes"],
            )
    if size > envelope["claim_json_bytes"]:
        raise EnvelopeExceeded(
            "inventory", "claim_json_bytes", size, envelope["claim_json_bytes"]
        )


def locked_writer_envelope(budget):
    """Return the shared owner's explicit contract only for the new strategy."""
    backend = planned_backend(budget)
    if backend is None or backend.plan["contract"] not in ENVELOPE_BUDGETS:
        return None
    return validate_writer_envelope(backend.effective_envelope)


def make_writer_envelope(
    *,
    body_units,
    assertions,
    body_bytes,
    source_identity,
    builder_identity,
    body_cap=CONTEXT_LIMIT,
    frozen_component_reservation=0,
    contract=ENVELOPE_CONTRACT,
    capacity=None,
    round_budget=False,
    global_findings_contract=None,
    inventory_recovery=False,
    mixed_inventory_recovery=False,
    summary_constraints=False,
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
    writer_source_view=None,
    evaluation_dimensions_contract=None,
    evaluation_result_contract=None,
    closeout_profile=None,
    output_tokens=None,
):
    """Define explicit producer limits, not estimates of unconstrained outputs."""
    from litagent.eval.scoped import SUPPORT_VALIDATION_REVISIONS

    support_selected = (
        type(support_validation_revision) is str
        and support_validation_revision in SUPPORT_VALIDATION_REVISIONS
    )

    if evaluation_dimensions_contract is not None and (
        type(evaluation_dimensions_contract) is not str
        or evaluation_dimensions_contract != "separated-evaluation-dimensions-v1"
        or contract != FLAT_ENVELOPE_CONTRACT
        or round_budget is not True
    ):
        raise ValueError("evaluation_dimensions_policy_invalid")

    if writer_source_view is not None:
        from litagent.writer_source_view import CONTRACT as SOURCE_VIEW_CONTRACT

        if (
            type(writer_source_view) is not str
            or writer_source_view != SOURCE_VIEW_CONTRACT
            or contract != FLAT_ENVELOPE_CONTRACT
            or round_budget is not True
        ):
            raise ValueError("writer_source_view_policy_invalid")

    if repair_prompt_revision is not None:
        from litagent.writing_flat import REPAIR_PROMPT_REVISION

        if (
            type(repair_prompt_revision) is not str
            or repair_prompt_revision != REPAIR_PROMPT_REVISION
            or contract != FLAT_ENVELOPE_CONTRACT
            or round_budget is not True
        ):
            raise ValueError("repair_prompt_policy_invalid")

    if evaluation_response_contract is not None:
        from litagent.eval.batched import EVALUATION_RESPONSE_CONTRACTS

        if (
            type(evaluation_response_contract) is not str
            or evaluation_response_contract not in EVALUATION_RESPONSE_CONTRACTS
            or round_budget is not True
        ):
            raise ValueError("evaluation_response_contract_invalid")

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
    if repair_guidance_contract is not None:
        from litagent.writing_flat import STRUCTURED_REPAIR_GUIDANCE_CONTRACT

        if (
            repair_guidance_contract != STRUCTURED_REPAIR_GUIDANCE_CONTRACT
            or contract != FLAT_ENVELOPE_CONTRACT
            or round_budget is not True
            or capacity is None
            or repair_mapping_contract
            not in ("capacity-source-map-v3", "capacity-source-map-v4")
            or global_findings_contract != "anchored-global-findings-v2"
            or summary_constraints is not True
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
        or (repair_mapping_contract is not None and round_budget is not True)
        or (
            repair_mapping_contract
            in ("capacity-source-map-v3", "capacity-source-map-v4")
            and not support_selected
        )
    ):
        raise ValueError("repair_mapping_policy_invalid")
    if support_validation_revision is not None and (
        not support_selected
        or contract != FLAT_ENVELOPE_CONTRACT
        or round_budget is not True
        or capacity is None
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
        or contract != FLAT_ENVELOPE_CONTRACT
        or round_budget is not True
        or capacity is None
        or mixed_inventory_recovery is not True
        or repair_feedback is not True
        or reference_contract != "ordered-reference-set-v1"
    ):
        raise ValueError("transport_revision_policy_invalid")
    if type(mixed_inventory_recovery) is not bool or (
        mixed_inventory_recovery
        and (not round_budget or capacity is None or inventory_recovery is not True)
    ):
        raise ValueError("mixed_inventory_recovery_policy_invalid")
    if type(repair_feedback) is not bool or (
        repair_feedback and (not round_budget or capacity is None)
    ):
        raise ValueError("repair_feedback_policy_invalid")
    if reference_contract is not None:
        from litagent.writing_flat import REFERENCE_SET_CONTRACT

        if (
            reference_contract != REFERENCE_SET_CONTRACT
            or contract != FLAT_ENVELOPE_CONTRACT
            or round_budget is not True
            or capacity is None
        ):
            raise ValueError("reference_contract_policy_invalid")
    value = {
        "contract": contract,
        "body_max_units": body_units,
        "summary_max_units_per_section": 2,
        "model_assertions": assertions,
        "assessment_batches": (assertions + 15) // 16,
        "body_json_bytes": body_bytes,
        "summary_json_bytes": 4000,
        "claim_json_bytes": assertions * 512,
        "request_caps": {
            "body": body_cap,
            "summary": 60000,
            "reviewer": 90000,
            "repair": CONTEXT_LIMIT,
            "inventory": 90000,
            "assessment": 90000,
            "global": 90000,
        },
        "source_identity": source_identity,
        "builder_identity": builder_identity,
        "frozen_component_reservation": frozen_component_reservation,
    }
    if output_tokens is not None:
        from litagent.output_capacity import validate_output_tokens

        if contract != FLAT_ENVELOPE_CONTRACT or not round_budget:
            raise ValueError("stage_output_capacity_policy_invalid")
        value["output_tokens"] = validate_output_tokens(output_tokens)
    if contract == FLAT_ENVELOPE_CONTRACT:
        # Full-capacity prose is no longer squeezed to the v4 small-body path.
        value["request_caps"]["reviewer"] = CONTEXT_LIMIT
        value["request_caps"].update(
            summary=70000, assessment=100000, **{"global": CONTEXT_LIMIT}
        )
        value["claim_json_bytes"] = 30000
        value["claim_max_json_bytes"] = 2000
        value["diagnostic_json_bytes"] = 12288
    if output_tokens is not None:
        value["request_caps"].update(
            inventory=100000,
            **{"global": 120000},
            repair=CONTEXT_LIMIT + output_tokens["repair"] - 20000,
        )
    if capacity is not None:
        from litagent.capacity import resolve_capacity

        cap = resolve_capacity(capacity)
        value.update(
            capacity=cap.model_dump(),
            summary_max_units_per_section=cap.max_summary_units,
            assessment_batches=(assertions + cap.batch_size - 1) // cap.batch_size,
            summary_json_bytes=cap.summary_json_bytes,
            claim_json_bytes=cap.claim_json_bytes,
            claim_max_json_bytes=cap.claim_max_json_bytes,
            diagnostic_json_bytes=cap.diagnostic_json_bytes,
        )
    if round_budget:
        value["repair_mapping_contract"] = "capacity-edit-map-v2"
    if global_findings_contract is not None:
        value["global_findings_contract"] = global_findings_contract
    if inventory_recovery:
        from litagent.eval.inventory_recovery import CONTRACT, MIXED_CONTRACT

        value["inventory_recovery_contract"] = (
            MIXED_CONTRACT if mixed_inventory_recovery else CONTRACT
        )
    if summary_constraints:
        value["summary_constraints_contract"] = "effective-summary-limits-v1"
    if classification_review:
        from litagent.eval.scoped import INDEPENDENT_CLASSIFICATION_CONTRACT

        value["classification_review_contract"] = INDEPENDENT_CLASSIFICATION_CONTRACT
    if repair_feedback:
        value["repair_feedback_contract"] = "host-repair-feedback-v1"
    if reference_contract is not None:
        value["reference_contract"] = reference_contract
    if transport_revision is not None:
        from litagent.writing_flat import transport_identity

        value["transport_revision"] = transport_revision
        value["transport_identity"] = transport_identity()
    if support_validation_revision is not None:
        from litagent.eval.scoped import support_validation_identity

        value["support_validation_revision"] = support_validation_revision
        value["support_validation_identity"] = (
            support_validation_identity()
            if support_validation_revision == "host-support-receipts-v1"
            else support_validation_identity(support_validation_revision)
        )
    if repair_mapping_contract in ("capacity-source-map-v3", "capacity-source-map-v4"):
        from litagent.writing_flat import source_map_wire_identity

        value["repair_mapping_contract"] = repair_mapping_contract
        value["source_map_identity"] = (
            source_map_wire_identity()
            if repair_mapping_contract == "capacity-source-map-v3"
            else source_map_wire_identity(contract=repair_mapping_contract)
        )
        value["source_map_json_bytes"] = value["diagnostic_json_bytes"]
    if repair_guidance_contract is not None:
        from litagent.writing_flat import structured_repair_guidance_identity

        value["repair_guidance_contract"] = repair_guidance_contract
        value["repair_guidance_identity"] = structured_repair_guidance_identity()
        # This bounds the complete guidance value, not the complete SDK request.
        value["repair_guidance_json_bytes"] = value["diagnostic_json_bytes"]
    if repair_guidance_wire_revision is not None:
        from litagent.writing_flat import (
            REPAIR_GUIDANCE_WIRE_REVISION,
            REPAIR_TASK_VIEW_ADMISSION_RULE,
            REPAIR_TASK_VIEW_REVISION,
            repair_guidance_wire_identity,
        )

        value["repair_guidance_wire_revision"] = repair_guidance_wire_revision
        value["repair_guidance_wire_identity"] = (
            repair_guidance_wire_identity()
            if repair_guidance_wire_revision == REPAIR_GUIDANCE_WIRE_REVISION
            else repair_guidance_wire_identity(repair_guidance_wire_revision)
        )
        if repair_guidance_wire_revision == REPAIR_TASK_VIEW_REVISION:
            value["repair_guidance_capacity_rule"] = REPAIR_TASK_VIEW_ADMISSION_RULE
    if semantic_conflict_policy is not None:
        from litagent.eval.semantic_conflict import semantic_conflict_identity

        if round_budget is not True:
            raise ValueError("semantic_conflict_policy_invalid")
        value["semantic_conflict_policy"] = semantic_conflict_policy
        value["semantic_conflict_identity"] = semantic_conflict_identity()
    if evaluation_response_contract is not None:
        from litagent.eval.batched import response_contract_identity

        value["evaluation_response_contract"] = evaluation_response_contract
        value["evaluation_response_identity"] = response_contract_identity(
            contract=evaluation_response_contract
        )
    if repair_prompt_revision is not None:
        from litagent.writing_flat import repair_prompt_identity

        value["repair_prompt_revision"] = repair_prompt_revision
        value["repair_prompt_identity"] = repair_prompt_identity()
    if writer_source_view is not None:
        from litagent.writer_source_view import source_view_identity

        value["writer_source_view"] = writer_source_view
        value["writer_source_view_identity"] = source_view_identity()
    if evaluation_dimensions_contract is not None:
        from litagent.eval.dimensions import dimensions_identity

        value["evaluation_dimensions_contract"] = evaluation_dimensions_contract
        value["evaluation_dimensions_identity"] = dimensions_identity()
    if evaluation_result_contract is not None:
        from litagent.eval.results import CONTRACT as RESULT_CONTRACT
        from litagent.eval.results import contract_identity

        if evaluation_result_contract != RESULT_CONTRACT or round_budget is not True:
            raise ValueError("evaluation_result_contract_invalid")
        value["evaluation_result_contract"] = evaluation_result_contract
        value["evaluation_result_identity"] = contract_identity()
        if evaluation_dimensions_contract is not None:
            from litagent.eval.decision_output import fields as decision_fields

            value.update(decision_fields())
    if closeout_profile is not None:
        from litagent.closeout import PROFILE, profile_identity

        if closeout_profile != PROFILE or round_budget is not True:
            raise ValueError("closeout_profile_invalid")
        value.update(
            closeout_profile=PROFILE, closeout_profile_identity=profile_identity()
        )
    value["identity"] = _digest(value)
    return validate_writer_envelope(value)


def envelope_budget_plan(
    envelope, *, identity, first_request, repairs, spent_tokens=0, spent_requests=0
):
    """Protect the complete path for an explicitly enforced new output scope."""
    envelope = validate_writer_envelope(envelope)
    counts = {
        "body": 4,
        "summary": 1 + repairs,
        "reviewer": 1,
        "inventory": 1 + repairs,
        "assessment": envelope["assessment_batches"] * (1 + repairs),
        "global": 1 + repairs,
        **({"repair": repairs} if repairs else {}),
    }
    outputs = {
        "body": 20000,
        "summary": 20000,
        "reviewer": 4096,
        "repair": 20000,
        "inventory": 16384,
        "assessment": 16384,
        "global": 16384,
    }
    if "output_tokens" in envelope:
        from litagent.output_capacity import envelope_output_tokens

        outputs = envelope_output_tokens(envelope)
    stages = {
        k: {
            "count": n,
            "request_cap": envelope["request_caps"][k],
            "output_tokens": outputs[k],
            "bound": "host_envelope_upper_bound",
        }
        for k, n in counts.items()
    }
    stages["body_initial"] = {
        "count": 1,
        "output_tokens": outputs["body"],
        "bound": "exact_wire",
        "request_cap": wire_reservation(
            first_request["messages"], outputs["body"], first_request["response_format"]
        ),
    }
    return build_budget_plan(
        stages,
        identity=identity,
        spent_tokens=spent_tokens,
        spent_requests=spent_requests,
        contract=(
            (
                ROUND_BUDGET_CONTRACT
                if "repair_mapping_contract" in envelope
                else CAPACITY_BUDGET_CONTRACT
            )
            if "capacity" in envelope
            else (
                FLAT_BUDGET_CONTRACT
                if envelope["contract"] == FLAT_ENVELOPE_CONTRACT
                else ENVELOPE_BUDGET_CONTRACT
            )
        ),
        repairs=repairs,
        envelope=envelope,
    )


def build_budget_plan(
    stages: dict,
    *,
    identity: str,
    spent_tokens: int = 0,
    spent_requests: int = 0,
    contract: str = CONTRACT,
    repairs: int = 2,
    envelope: dict | None = None,
) -> dict:
    """Admit a complete worst-case path; stage counts include all shared repairs."""
    if (
        not isinstance(identity, str)
        or not identity
        or type(spent_tokens) is not int
        or spent_tokens < 0
        or type(spent_requests) is not int
        or not 0 <= spent_requests <= 64
        or not isinstance(stages, dict)
        or not stages
        or contract not in CONTRACTS
        or type(repairs) is not int
        or not 0 <= repairs <= 2
        or (contract == CONTRACT and repairs != 2)
        or (contract not in ENVELOPE_BUDGETS and envelope is not None)
    ):
        raise ValueError("budget_plan_input_invalid")
    projected = {}
    for name, stage in stages.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(stage, dict)
            or set(stage) != {"count", "request_cap", "output_tokens", "bound"}
            or any(
                type(stage[k]) is not int or stage[k] <= 0
                for k in ("count", "request_cap", "output_tokens")
            )
            or stage["request_cap"] < stage["output_tokens"]
            or stage["bound"]
            not in (
                {"exact_wire", "context_upper_bound", "host_envelope_upper_bound"}
                if contract in ENVELOPE_BUDGETS
                else {"exact_wire", "context_upper_bound"}
            )
        ):
            raise ValueError("budget_plan_stage_invalid")
        projected[name] = {
            **stage,
            "reserved_tokens": stage["count"] * stage["request_cap"],
        }
    if contract in ENVELOPE_BUDGETS:
        envelope = validate_writer_envelope(envelope)
        if "semantic_conflict_policy" in envelope and contract != ROUND_BUDGET_CONTRACT:
            raise ValueError("semantic_conflict_policy_invalid")
        if (
            "evaluation_response_contract" in envelope
            and contract != ROUND_BUDGET_CONTRACT
        ):
            raise ValueError("evaluation_response_contract_invalid")
        if "writer_source_view" in envelope and contract != ROUND_BUDGET_CONTRACT:
            raise ValueError("writer_source_view_policy_invalid")
        if (
            "evaluation_dimensions_contract" in envelope
            and contract != ROUND_BUDGET_CONTRACT
        ):
            raise ValueError("evaluation_dimensions_policy_invalid")
        if "repair_prompt_revision" in envelope and contract != ROUND_BUDGET_CONTRACT:
            raise ValueError("repair_prompt_policy_invalid")
    if contract in (ADAPTIVE_CONTRACT, *ENVELOPE_BUDGETS):
        counts = {k: s["count"] for k, s in projected.items()}
        body = (counts.pop("body", 0), counts.pop("body_initial", 0))
        required_counts = {
            "summary": 1 + repairs,
            "reviewer": 1,
            "inventory": 1 + repairs,
            "assessment": (envelope["assessment_batches"] if envelope else 8)
            * (1 + repairs),
            "global": 1 + repairs,
            **({"repair": repairs} if repairs else {}),
        }
        outputs = {
            "body": 20000,
            "body_initial": 20000,
            "summary": 20000,
            "reviewer": 4096,
            "repair": 20000,
            "inventory": 16384,
            "assessment": 16384,
            "global": 16384,
        }
        if envelope and "output_tokens" in envelope:
            from litagent.output_capacity import envelope_output_tokens

            outputs = envelope_output_tokens(envelope)
            outputs["body_initial"] = outputs["body"]
        if (
            body not in {(5, 0), (4, 1)}
            or counts != required_counts
            or any(s["output_tokens"] != outputs[k] for k, s in projected.items())
        ):
            raise ValueError("adaptive_budget_complete_path_invalid")
        if envelope and any(
            s["request_cap"] != envelope["request_caps"][k]
            for k, s in projected.items()
            if k != "body_initial"
        ):
            raise ValueError("writer_envelope_stage_identity_invalid")
    maximum_tokens = 4500000 if contract in FLAT_BUDGETS else 1000000
    if contract in FLAT_BUDGETS and (
        repairs != 2
        or envelope["contract"] != FLAT_ENVELOPE_CONTRACT
        or (
            contract == FLAT_BUDGET_CONTRACT
            and (
                envelope["body_max_units"] != 66 or envelope["model_assertions"] != 128
            )
        )
        or (contract in CAPACITY_BUDGETS and "capacity" not in envelope)
        or (
            (contract == ROUND_BUDGET_CONTRACT)
            != ("repair_mapping_contract" in envelope)
        )
        or (contract == FLAT_BUDGET_CONTRACT and "capacity" in envelope)
    ):
        raise ValueError("flat_budget_complete_scope_required")
    required = spent_tokens + sum(s["reserved_tokens"] for s in projected.values())
    requests = spent_requests + sum(s["count"] for s in projected.values())
    # Integer arithmetic avoids rounding an allowance below the required margin.
    allowance = max(500000, ((required * 11 + 99999) // 100000) * 10000)
    from litagent.output_capacity import envelope_context_limit, envelope_context_tokens

    context_limit = envelope_context_limit(envelope)
    reason = (
        "budget_plan_context_exceeded"
        if any(s["request_cap"] > context_limit for s in projected.values())
        else (
            "writer_envelope_frozen_components_exceeded"
            if envelope
            and envelope["frozen_component_reservation"]
            > min(
                envelope["request_caps"][k]
                for k in ("inventory", "assessment", "global")
            )
            else (
                "budget_plan_requests_exceeded"
                if requests > 64
                else (
                    "budget_plan_total_exceeded" if allowance > maximum_tokens else None
                )
            )
        )
    )
    result = {
        "contract": contract,
        "measurement": "complete-utf8-reservation-v1",
        "input_identity": identity,
        "status": "preflight_rejected" if reason else "admitted",
        "reason": reason,
        "required_tokens": required,
        "planned_tokens": allowance,
        "maximum_tokens": maximum_tokens,
        "planned_requests": requests,
        "maximum_requests": 64,
        "context_tokens": envelope_context_tokens(envelope),
        "context_reservation_limit": context_limit,
        "context_margin": 0.1,
        "spent_tokens": spent_tokens,
        "spent_requests": spent_requests,
        "scope_bounds": {
            "report_max_units": 72,
            "authored_text_max_characters": 2000,
            "diagnostic_string_max_characters": 2000,
            "model_assertions": 128,
            "assertions_per_batch": 16,
            "assessment_batches": 8,
            "shared_repairs": repairs,
        },
        "stages": projected,
    }
    if contract in (ADAPTIVE_CONTRACT, *ENVELOPE_BUDGETS):
        from litagent.body_quality import ADAPTIVE_ASSESSMENT_INPUT_CONTRACT

        result.update(
            repair_limit=repairs,
            repair_selection=(
                "fixed-two-complete-repairs-v1"
                if contract in FLAT_BUDGETS
                else "largest-admissible-complete-path-v1"
            ),
            repair_allocation="unallocated" if reason else "locked",
            initial_evaluation_included=True,
            initial_evaluation_reserved=reason is None,
            assessment_input_contract=ADAPTIVE_ASSESSMENT_INPUT_CONTRACT,
        )
    if envelope:
        result["envelope"] = envelope
        result["scope_bounds"].update(
            report_max_units=envelope["body_max_units"]
            + 2 * envelope["summary_max_units_per_section"],
            model_assertions=envelope["model_assertions"],
            assessment_batches=envelope["assessment_batches"],
            envelope_contract=envelope["contract"],
        )
    result["identity"] = _digest(result)
    return result


def _joint_baseline_plan(plan):
    """Rebuild the original fixed-count root without trusting aggregate fields."""
    stages = {
        name: {
            field: spec[field]
            for field in ("count", "request_cap", "output_tokens", "bound")
        }
        for name, spec in plan["stages"].items()
    }
    stages["assessment"]["count"] = plan["envelope"]["assessment_batches"] * (
        1 + plan["scope_bounds"]["shared_repairs"]
    )
    return build_budget_plan(
        stages,
        identity=plan["input_identity"],
        spent_tokens=plan["spent_tokens"],
        spent_requests=plan["spent_requests"],
        contract=plan["contract"],
        repairs=plan["scope_bounds"]["shared_repairs"],
        envelope=plan["envelope"],
    )


def joint_round_escrow_identity() -> str:
    """Bind only the opt-in aggregate rule and its live root validation."""
    from litagent.capacity_evaluation import preflight_round
    from litagent.capacity_rounds import (
        RoundRequestCapacityError,
        _allocate,
        choose_refinement,
    )

    return _digest(
        [
            JOINT_ROUND_ESCROW_CONTRACT,
            inspect.getsource(_joint_baseline_plan),
            inspect.getsource(with_joint_round_escrow),
            inspect.getsource(PlannedBudget.__init__),
            inspect.getsource(_allocate),
            inspect.getsource(choose_refinement),
            inspect.getsource(RoundRequestCapacityError),
            inspect.getsource(preflight_round),
        ]
    )


def with_joint_round_escrow(plan: dict) -> dict:
    """Freeze complete round escrows inside the original admitted allowance."""
    try:
        if (
            not isinstance(plan, dict)
            or plan.get("contract") != ROUND_BUDGET_CONTRACT
            or "joint_round_escrow" in plan
            or plan.get("scope_bounds", {}).get("shared_repairs") != 2
        ):
            raise ValueError("joint_round_escrow_plan_invalid")
        baseline = _joint_baseline_plan(plan)
        if json.dumps(plan, sort_keys=True, allow_nan=False) != json.dumps(
            baseline, sort_keys=True, allow_nan=False
        ):
            raise ValueError("joint_round_escrow_plan_invalid")
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("joint_round_escrow_plan_invalid") from exc
    if baseline["status"] != "admitted":
        return copy.deepcopy(baseline)

    from litagent.capacity import resolve_capacity

    capacity = resolve_capacity(baseline["envelope"]["capacity"])
    rounds = 1 + baseline["scope_bounds"]["shared_repairs"]
    other = {
        name: spec for name, spec in baseline["stages"].items() if name != "assessment"
    }
    nonassessment_tokens = baseline["spent_tokens"] + sum(
        spec["reserved_tokens"] for spec in other.values()
    )
    nonassessment_requests = baseline["spent_requests"] + sum(
        spec["count"] for spec in other.values()
    )
    count = min(
        capacity.max_assessment_batches or 64,
        (baseline["maximum_requests"] - nonassessment_requests) // rounds,
    )
    assessment = baseline["stages"]["assessment"]
    cap = assessment["request_cap"]
    total = min(
        count * cap,
        ((baseline["planned_tokens"] * 10 // 11) - nonassessment_tokens) // rounds,
    )
    if (
        cap != 100000
        or count < baseline["envelope"]["assessment_batches"]
        or total < assessment["output_tokens"]
    ):
        raise ValueError("joint_round_escrow_domain_invalid")
    result = copy.deepcopy(baseline)
    result["joint_round_escrow"] = {
        "contract": JOINT_ROUND_ESCROW_CONTRACT,
        "rule_identity": joint_round_escrow_identity(),
        "base_plan_identity": baseline["identity"],
        "rounds": rounds,
        "assessment": {
            "max_requests": count,
            "request_cap": cap,
            "reserved_tokens": total,
        },
    }
    result["stages"]["assessment"].update(
        count=rounds * count, reserved_tokens=rounds * total
    )
    result["required_tokens"] = nonassessment_tokens + rounds * total
    result["planned_requests"] = nonassessment_requests + rounds * count
    required = result["required_tokens"]
    with_margin = max(500000, ((required * 11 + 99999) // 100000) * 10000)
    if (
        with_margin > baseline["planned_tokens"]
        or result["planned_requests"] > baseline["maximum_requests"]
    ):
        raise ValueError("joint_round_escrow_domain_invalid")
    result.pop("identity")
    result["identity"] = _digest(result)
    return result


def complete_writer_plan(
    *,
    identity: str,
    first_request=None,
    spent_tokens=0,
    spent_requests=0,
    contract=CONTRACT,
    repairs=2,
    stage_bounds=None,
) -> dict:
    """Bound unknown future wire inputs by the enforced transport context ceiling.

    No tokenizer or statistical claim-density assumption is available for future
    inventory and repair outputs. A context ceiling is a real enforced upper bound,
    not a fabricated future report. This deliberately rejects unaffordable plans.
    Narrower bounds require a separately validated, versioned producer contract.
    """
    counts = {
        "body": (5, 20000),
        "summary": (1 + repairs, 20000),
        "reviewer": (1, 4096),
        "repair": (repairs, 20000),
        "inventory": (1 + repairs, 16384),
        "assessment": (8 * (1 + repairs), 16384),
        "global": (1 + repairs, 16384),
    }
    bounded = {
        name: {
            "count": count,
            "request_cap": CONTEXT_LIMIT,
            "output_tokens": output,
            "bound": "context_upper_bound",
        }
        for name, (count, output) in counts.items()
        if count
    }
    if stage_bounds is not None:
        if (
            contract != ADAPTIVE_CONTRACT
            or not isinstance(stage_bounds, dict)
            or set(stage_bounds) != set(counts)
        ):
            raise ValueError("adaptive_budget_bounds_invalid")
        for name, spec in bounded.items():
            bound = stage_bounds[name]
            if not isinstance(bound, dict) or set(bound) != {"request_cap", "bound"}:
                raise ValueError("adaptive_budget_bounds_invalid")
            spec.update(bound)
    if first_request is not None:
        bounded["body"]["count"] -= 1
        bounded["body_initial"] = {
            "count": 1,
            "request_cap": wire_reservation(
                first_request["messages"], 20000, first_request["response_format"]
            ),
            "output_tokens": 20000,
            "bound": "exact_wire",
        }
    return build_budget_plan(
        bounded,
        identity=identity,
        spent_tokens=spent_tokens,
        spent_requests=spent_requests,
        contract=contract,
        repairs=repairs,
    )


def adaptive_writer_plan(**kwargs) -> dict:
    """Lock the largest affordable complete path before writing starts."""
    for repairs in (2, 1, 0):
        result = complete_writer_plan(
            **kwargs, contract=ADAPTIVE_CONTRACT, repairs=repairs
        )
        if result["status"] == "admitted":
            return result
    # Even zero repairs must include every first-pass assessment batch/global call.
    return result


def repair_limit(budget) -> int:
    """Read the host-locked allowance without changing historical two-attempt runs."""
    backend = planned_backend(budget)
    return backend.plan["scope_bounds"]["shared_repairs"] if backend else 2


def check_response_bounds(value, *, field=None):
    """Bound model diagnostics while preserving complete copied source passages."""
    if (
        field in {"explanation", "reason", "exclusion_reason"}
        and isinstance(value, str)
        and len(value) > 2000
    ):
        raise ValueError("planned_response_size_exceeded")
    if isinstance(value, dict):
        for key, item in value.items():
            check_response_bounds(item, field=key)
    elif isinstance(value, list):
        for item in value:
            check_response_bounds(item, field=field)


@contextmanager
def request_scope(stage: str, request_id: str):
    """Bind ownership explicitly across async transport and usage recording."""
    token = _active_request.set((stage, request_id))
    try:
        yield
    finally:
        _active_request.reset(token)


class PlannedBudget(CostBudget):
    """Protect all future stages using durable, identity-bound request escrows."""

    def __init__(
        self,
        plan: dict,
        *,
        path: Path | None = None,
        checkpoint: Callable | None = None,
    ):
        """Validate the immutable plan before accepting any request."""
        if not isinstance(plan, dict) or not isinstance(plan.get("stages"), dict):
            raise ValueError("budget_plan_identity_invalid")
        try:
            if "joint_round_escrow" in plan:
                expected = with_joint_round_escrow(_joint_baseline_plan(plan))
                if json.dumps(expected, sort_keys=True, allow_nan=False) != json.dumps(
                    plan, sort_keys=True, allow_nan=False
                ):
                    raise ValueError("budget_plan_identity_invalid")
            else:
                expected = build_budget_plan(
                    {
                        k: {
                            f: v[f]
                            for f in ("count", "request_cap", "output_tokens", "bound")
                        }
                        for k, v in plan.get("stages", {}).items()
                    },
                    identity=plan.get("input_identity"),
                    spent_tokens=plan.get("spent_tokens", 0),
                    spent_requests=plan.get("spent_requests", 0),
                    contract=plan.get("contract"),
                    repairs=plan.get("scope_bounds", {}).get("shared_repairs", 2),
                    envelope=plan.get("envelope"),
                )
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError("budget_plan_identity_invalid") from exc
        if expected != plan:
            raise ValueError("budget_plan_identity_invalid")
        if plan["status"] != "admitted":
            raise ValueError("budget_preflight_rejected")
        super().__init__(max_tokens=plan["planned_tokens"])
        self.strict = True
        self._halt_on_unknown_usage = True
        self.plan = copy.deepcopy(plan)
        self.path, self.checkpoint = (
            Path(path) if path is not None else None
        ), checkpoint
        self._used_tokens = plan["spent_tokens"]
        self._call_count = plan["spent_requests"]
        self._requests_sent = plan["spent_requests"]
        self.entries: dict = {}
        self.revisions: list[dict] = []
        self.allocation: dict | None = None
        self.closed_stages: set[str] = set()
        if self.path and self.path.exists():
            self._restore_planned(json.loads(self.path.read_text()))
        self._persist()

    def configure_strict(self, *, halt_on_unknown_usage=False):
        """Keep the v2 guarantee independent of the legacy 500k phase policy."""
        self.strict = True
        self._halt_on_unknown_usage = True

    def _owner(self):
        """Require a host-bound stage and immutable request identity."""
        owner = _active_request.get()
        if (
            owner is None
            or not all(isinstance(v, str) and v for v in owner)
            or owner[0] not in self.plan["stages"]
        ):
            raise ValueError("planned_request_scope_missing")
        return owner

    @property
    def effective_envelope(self):
        """Return the current allocation without changing the admitted root identity."""
        return (self.allocation or self.plan)["envelope"]

    @property
    def effective_stages(self):
        """Read only verified mutable allocations for the explicit v6 policy."""
        return (self.allocation or self.plan)["stages"]

    def _stage_spec(self, stage):
        """Resolve a stage from the shared effective allocation."""
        return self.effective_stages[stage]

    def _restore_capacity(self, value):
        """Replay allocation proofs before accepting their enlarged request slots."""
        if self.plan["contract"] == ROUND_BUDGET_CONTRACT:
            from litagent.capacity_rounds import restore_rounds

            return restore_rounds(self, value)
        from litagent.capacity_budget import restore_capacity

        restore_capacity(self, value)

    def expand_capacity(
        self, assertions, batches, *, request_caps=None, reason="inventory"
    ):
        """Lock every remaining assessment before permitting additional batches."""
        from litagent.capacity_budget import expand_capacity

        return expand_capacity(
            self, assertions, batches, request_caps=request_caps, reason=reason
        )

    def _stage_entries(self, stage):
        """Return only reservations owned by one protected stage."""
        return [e for e in self.entries.values() if e["stage"] == stage]

    def remaining(self, *, advance=True):
        """Expose only this stage's unused protected allocation."""
        if self.stop_reason:
            return 0
        stage, request_id = self._owner()
        entries = self._stage_entries(stage)
        spec = self._stage_spec(stage)
        if stage in self.closed_stages or len(entries) >= spec["count"]:
            return 0
        if request_id in self.entries:
            raise ValueError("planned_request_no_replay")
        locked = sum(
            e["actual_tokens"] if e["status"] == "settled" else e["reserved_tokens"]
            for e in entries
        )
        return min(spec["request_cap"], spec["reserved_tokens"] - locked)

    def check_wire(self, messages, output_tokens, response_format=None):
        """Check the actual complete wire against its immutable admitted envelope."""
        stage, _ = self._owner()
        spec = self._stage_spec(stage)
        from litagent.eval.inventory_recovery import recovery_wire_binding

        recovery = recovery_wire_binding(self, messages, output_tokens, response_format)
        amount = wire_reservation(messages, output_tokens, response_format)
        if self.plan["contract"] == ROUND_BUDGET_CONTRACT and stage in {
            "assessment",
            "global",
        }:
            from litagent.capacity_rounds import check_round_wire

            check_round_wire(self, stage, messages, output_tokens, response_format)
        if self.plan["contract"] in FLAT_BUDGETS and amount > spec["request_cap"]:
            raise ValueError("budget_plan_contract_breached")
        if self.plan["contract"] in ENVELOPE_BUDGETS and amount > spec["request_cap"]:
            raise EnvelopeExceeded(
                stage, "complete_wire_reservation", amount, spec["request_cap"]
            )
        if (
            output_tokens
            != (
                self.plan["stages"]["inventory"]["output_tokens"]
                if recovery
                else spec["output_tokens"]
            )
            or amount > spec["request_cap"]
        ):
            raise ValueError("planned_request_envelope_exceeded")
        return amount

    def bind_wire(self, identity):
        """Save the exact outbound request identity before its send marker."""
        _, request_id = self._owner()
        entry = self.entries.get(request_id)
        if not entry or entry["status"] != "reserved_not_sent":
            raise ValueError("planned_request_no_replay")
        entry["wire_sha256"] = _digest(identity)
        self._persist()

    def stop_after_worker_failure(self):
        """Persist a technical terminal reason without refunding any requests."""
        if self._stop_reason is None:
            self._stop_reason = "strict_worker_failed"
        self._persist()

    def check_stage_requests(self, stage, amounts):
        """Preflight every assessment batch before paying for its first batch."""
        spec = self._stage_spec(stage) if stage in self.plan["stages"] else None
        if self.plan["contract"] in ENVELOPE_BUDGETS and spec is not None:
            if (
                stage == "assessment"
                and len(amounts) > self.effective_envelope["assessment_batches"]
            ):
                raise EnvelopeExceeded(
                    stage,
                    "batch_count",
                    len(amounts),
                    self.effective_envelope["assessment_batches"],
                )
            for n in amounts:
                if type(n) is int and n > spec["request_cap"]:
                    if self.plan["contract"] in FLAT_BUDGETS:
                        raise ValueError("budget_plan_contract_breached")
                    raise EnvelopeExceeded(
                        stage, "complete_wire_reservation", n, spec["request_cap"]
                    )
        if not spec or any(
            type(n) is not int or n <= 0 or n > spec["request_cap"] for n in amounts
        ):
            raise ValueError("planned_request_envelope_exceeded")
        entries = self._stage_entries(stage)
        if stage in self.closed_stages or len(entries) + len(amounts) > spec["count"]:
            raise ValueError("planned_stage_capacity_exceeded")
        locked = sum(
            e["actual_tokens"] if e["status"] == "settled" else e["reserved_tokens"]
            for e in entries
        )
        if sum(amounts) + locked > spec["reserved_tokens"]:
            raise ValueError("planned_stage_capacity_exceeded")

    def reserve(self, tokens):
        """Persist a reservation before any await or outbound request."""
        stage, request_id = self._owner()
        if self.stop_reason:
            raise ValueError(self.stop_reason)
        if type(tokens) is not int or tokens <= 0 or tokens > self.remaining():
            raise ValueError("planned_request_envelope_exceeded")
        binding = {}
        from litagent.eval.inventory_recovery import _active

        if _active.get() is not None:
            checked = getattr(self, "_checked_inventory_recovery", None)
            if not checked or checked[0] != self._owner() or checked[2] != tokens:
                raise ValueError("inventory_recovery_budget_identity_invalid")
            binding["inventory_recovery"] = copy.deepcopy(checked[1])
        if self.plan["contract"] == ROUND_BUDGET_CONTRACT and stage in {
            "assessment",
            "global",
        }:
            from litagent.capacity_rounds import reserve_round_wire

            binding = reserve_round_wire(self, tokens)
        self.entries[request_id] = {
            "stage": stage,
            "status": "reserved_not_sent",
            "reserved_tokens": tokens,
            "actual_tokens": None,
            "usage_sha256": None,
            "sent": False,
            **binding,
        }
        self._persist()

    def begin_request(self):
        """Count a request durably before sending; retries are never implicit."""
        _, request_id = self._owner()
        entry = self.entries.get(request_id)
        if not entry or entry["status"] != "reserved_not_sent":
            raise ValueError("planned_request_no_replay")
        if self.stop_reason or self._requests_sent >= 64:
            raise ValueError(self.stop_reason or "strict_request_budget_exceeded")
        entry["status"] = "started_no_replay"
        entry["sent"] = True
        self._requests_sent += 1
        self._persist()

    def record(self, usage):
        """Settle complete usage once, retaining a fingerprint for idempotence."""
        _, request_id = self._owner()
        entry = self.entries.get(request_id)
        fields = ("prompt_tokens", "completion_tokens")
        if (
            not entry
            or not isinstance(usage, dict)
            or any(type(usage.get(k)) is not int or usage[k] < 0 for k in fields)
        ):
            raise ValueError("planned_usage_invalid")
        fingerprint = _digest({k: usage[k] for k in fields})
        if entry["status"] == "settled":
            if fingerprint != entry["usage_sha256"]:
                raise ValueError("planned_usage_identity_conflict")
            return
        if entry["status"] != "started_no_replay":
            raise ValueError("planned_usage_state_invalid")
        total = sum(usage[k] for k in fields)
        entry.update(status="settled", actual_tokens=total, usage_sha256=fingerprint)
        self._used_tokens += total
        self._call_count += 1
        if total > entry["reserved_tokens"]:
            self._stop_reason = "planned_usage_bound_violation"
        self._persist()
        if self._stop_reason:
            raise ValueError(self._stop_reason)

    def charge_unknown_usage(self, reservation):
        """Hold the request ceiling without counting a known response twice."""
        _, request_id = self._owner()
        entry = self.entries.get(request_id)
        if not entry:
            raise ValueError("planned_request_scope_missing")
        if entry["status"] == "settled":
            return
        if reservation != entry["reserved_tokens"]:
            raise ValueError("planned_reservation_identity_conflict")
        entry["status"] = "interrupted_no_replay"
        self._usage_uncertain = True
        self._stop_reason = "strict_usage_unknown"
        self._persist()

    def release(self, tokens):
        """Release only known responses; never refund unknown/in-flight usage."""
        _, request_id = self._owner()
        entry = self.entries.get(request_id)
        if not entry or tokens != entry["reserved_tokens"]:
            raise ValueError("planned_reservation_identity_conflict")
        # A crash between reservation and send remains no-replay on recovery.
        if entry["status"] == "started_no_replay":
            self.charge_unknown_usage(tokens)
        self._persist()

    def close_stage(self, stage):
        """Close a completed branch without granting new request slots elsewhere."""
        if stage not in self.plan["stages"] or any(
            e["status"] != "settled" for e in self._stage_entries(stage)
        ):
            raise ValueError("planned_stage_close_invalid")
        self.closed_stages.add(stage)
        self._persist()

    def snapshot(self):
        """Keep actual usage distinct from future and uncertain reservations."""
        return copy.deepcopy(
            {
                "schema_version": 2,
                "strict_policy": self.plan["contract"],
                "plan": self.plan,
                "max_tokens": self._max_tokens,
                "used_tokens": self._used_tokens,
                "call_count": self._call_count,
                "requests_sent": self._requests_sent,
                "usage_uncertain": self._usage_uncertain,
                "stop_reason": self._stop_reason,
                "entries": self.entries,
                "closed_stages": sorted(self.closed_stages),
                **(
                    {
                        "capacity_revisions": self.revisions,
                        "capacity_allocation": self.allocation,
                    }
                    if self.plan["contract"] in CAPACITY_BUDGETS
                    else {}
                ),
            }
        )

    def _restore_planned(self, value):
        """Validate accounting consistency before reopening a durable ledger."""
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != 2
            or value.get("strict_policy") != self.plan["contract"]
            or value.get("plan") != self.plan
            or value.get("max_tokens") != self.plan["planned_tokens"]
            or any(
                type(value.get(k)) is not int or value[k] < 0
                for k in ("used_tokens", "call_count", "requests_sent")
            )
            or type(value.get("usage_uncertain")) is not bool
            or value.get("stop_reason")
            not in {
                None,
                "strict_worker_failed",
                "strict_usage_unknown",
                "planned_usage_bound_violation",
                "planned_ledger_persistence_failed",
            }
            or not isinstance(value.get("entries"), dict)
        ):
            raise ValueError("planned_budget_identity_conflict")
        if self.plan["contract"] in CAPACITY_BUDGETS:
            self._restore_capacity(value)
        self.entries = copy.deepcopy(value["entries"])
        totals, calls, sent = (
            self.plan["spent_tokens"],
            self.plan["spent_requests"],
            self.plan["spent_requests"],
        )
        for rid, entry in self.entries.items():
            if (
                not isinstance(rid, str)
                or not rid
                or not isinstance(entry, dict)
                or entry.get("stage") not in self.plan["stages"]
            ):
                raise ValueError("planned_budget_snapshot_invalid")
            spec = self._stage_spec(entry["stage"])
            amount = entry.get("reserved_tokens")
            if type(amount) is not int or not 0 < amount <= spec["request_cap"]:
                raise ValueError("planned_budget_snapshot_invalid")
            status = entry.get("status")
            if status not in {
                "reserved_not_sent",
                "started_no_replay",
                "settled",
                "interrupted_no_replay",
            }:
                raise ValueError("planned_budget_snapshot_invalid")
            if type(entry.get("sent")) is not bool:
                raise ValueError("planned_budget_snapshot_invalid")
            if entry["sent"]:
                sent += 1
            if status == "settled":
                actual = entry.get("actual_tokens")
                if (
                    type(actual) is not int
                    or actual < 0
                    or not isinstance(entry.get("usage_sha256"), str)
                    or len(entry["usage_sha256"]) != 64
                    or not entry["sent"]
                    or (
                        actual > amount
                        and value.get("stop_reason") != "planned_usage_bound_violation"
                    )
                ):
                    raise ValueError("planned_budget_snapshot_invalid")
                totals += actual
                calls += 1
            else:
                entry["status"] = "interrupted_no_replay"
                self._usage_uncertain = True
                self._stop_reason = "strict_usage_unknown"
        if any(
            len(self._stage_entries(s)) > p["count"]
            for s, p in self.effective_stages.items()
        ):
            raise ValueError("planned_budget_snapshot_invalid")
        if (totals, calls, sent) != (
            value.get("used_tokens"),
            value.get("call_count"),
            value.get("requests_sent"),
        ):
            raise ValueError("planned_budget_snapshot_invalid")
        self._used_tokens, self._call_count, self._requests_sent = totals, calls, sent
        if value.get("stop_reason"):
            self._stop_reason = value["stop_reason"]
        self._usage_uncertain |= value.get("usage_uncertain") is True
        closed = value.get("closed_stages")
        if not isinstance(closed, list) or any(not isinstance(k, str) for k in closed):
            raise ValueError("planned_budget_snapshot_invalid")
        self.closed_stages = set(closed)
        if not self.closed_stages.issubset(self.plan["stages"]):
            raise ValueError("planned_budget_snapshot_invalid")

    def restore(self, data):
        """Validate a forward-only checkpoint before changing a shared owner."""
        # A rejected checkpoint must not partially replace live paid identities.
        # The temporary owner has no path/callback and cannot touch durable state.
        candidate = PlannedBudget(self.plan)
        candidate._restore_planned(data)
        if candidate.revisions[: len(self.revisions)] != self.revisions:
            raise ValueError("capacity_revision_restore_regression")
        if (
            candidate._used_tokens < self._used_tokens
            or candidate._call_count < self._call_count
            or candidate._requests_sent < self._requests_sent
            or not self.closed_stages.issubset(candidate.closed_stages)
            or (self._usage_uncertain and not candidate._usage_uncertain)
            or (self._stop_reason and candidate._stop_reason != self._stop_reason)
        ):
            raise ValueError("planned_budget_restore_regression")
        for request_id, entry in self.entries.items():
            restored = candidate.entries.get(request_id)
            if (
                restored is None
                or any(
                    restored.get(key) != entry.get(key)
                    for key in (
                        "stage",
                        "reserved_tokens",
                        "wire_sha256",
                        "inventory_recovery",
                    )
                )
                or (entry["sent"] and not restored["sent"])
                or (entry["status"] == "settled" and restored != entry)
                or (
                    entry["status"] == "interrupted_no_replay"
                    and restored["status"] != "interrupted_no_replay"
                )
            ):
                raise ValueError("planned_budget_restore_regression")
        self._used_tokens = candidate._used_tokens
        self._call_count = candidate._call_count
        self._requests_sent = candidate._requests_sent
        self._usage_uncertain = candidate._usage_uncertain
        self._stop_reason = candidate._stop_reason
        self.revisions, self.allocation = candidate.revisions, candidate.allocation
        self.entries = candidate.entries
        self.closed_stages = candidate.closed_stages
        self._persist()

    def _persist(self):
        """Persist synchronously before transport, stopping on storage failure."""
        try:
            if self.path:
                from litagent.report_audit import atomic_json

                atomic_json(self.path, self.snapshot())
            if self.checkpoint:
                self.checkpoint(self.snapshot())
        except BaseException:
            self._stop_reason = "planned_ledger_persistence_failed"
            raise


def wire_reservation(messages, output_tokens, response_format=None):
    """Use the existing provider-boundary measurement for known wire requests."""
    return strict_request_reservation(messages, output_tokens, response_format)


def planned_backend(budget):
    """Resolve an opt-in backend without replacing already-shared client owners."""
    return (
        budget
        if isinstance(budget, PlannedBudget)
        else getattr(budget, "_planned_backend", None)
    )


def _assessment_settlement(budget, request_id):
    """Read the original settled slot without advancing the live round cursor."""
    from litagent.eval.semantic_conflict import selected_policy
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    if not isinstance(request_id, str) or not request_id:
        raise ValueError("semantic_conflict_settlement_request_invalid")
    owner = planned_backend(budget)
    snapshot = owner.snapshot() if owner is not None else budget
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("plan"), dict):
        raise ValueError("semantic_conflict_settlement_owner_missing")
    snapshot = validate_accounting_snapshot(snapshot, expected_plan=snapshot["plan"])
    plan = snapshot["plan"]
    if (
        plan["contract"] != ROUND_BUDGET_CONTRACT
        or not selected_policy(plan.get("envelope"))
        or snapshot["usage_uncertain"]
        or any(e["status"] != "settled" for e in snapshot["entries"].values())
    ):
        raise ValueError("semantic_conflict_settlement_unsettled")
    entry = snapshot["entries"].get(request_id)
    if (
        not isinstance(entry, dict)
        or entry.get("stage") != "assessment"
        or entry.get("status") != "settled"
        or entry.get("sent") is not True
        or "inventory_recovery" in entry
    ):
        raise ValueError("semantic_conflict_settlement_request_invalid")
    index = [
        key
        for key, value in snapshot["entries"].items()
        if value["stage"] == "assessment"
    ].index(request_id)
    matches = []
    for event in snapshot["capacity_revisions"]:
        allocation = event["allocation"]
        round_ = allocation["round"]
        slots = [s for s in round_["requests"] if s["stage"] == "assessment"]
        offset = index - round_["offsets"]["assessment"]
        if round_["mode"] == "assess" and 0 <= offset < len(slots):
            matches.append((allocation, slots[offset]))
    if len(matches) != 1:
        raise ValueError("semantic_conflict_settlement_round_invalid")
    allocation, slot = matches[0]
    if (
        entry.get("round_number") != allocation["round"]["number"]
        or entry.get("round_wire_sha256") != slot["wire_sha256"]
        or entry["reserved_tokens"] != slot["reserved_tokens"]
        or slot["output_tokens"] != plan["stages"]["assessment"]["output_tokens"]
        or not isinstance(entry.get("wire_sha256"), str)
        or len(entry["wire_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in entry["wire_sha256"])
    ):
        raise ValueError("semantic_conflict_settlement_wire_invalid")
    envelope = allocation["envelope"]
    receipt = {
        "contract": "host-settled-assessment-receipt-v1",
        "plan_identity": plan["identity"],
        "input_identity": plan["input_identity"],
        "envelope_identity": envelope["identity"],
        "semantic_conflict_policy": envelope["semantic_conflict_policy"],
        "semantic_conflict_identity": envelope["semantic_conflict_identity"],
        "stage": "assessment",
        "request_id": request_id,
        "input_sha256": slot["wire_sha256"],
        "round_number": entry["round_number"],
        "round_wire_sha256": entry["round_wire_sha256"],
        "wire_sha256": entry["wire_sha256"],
        "usage_sha256": entry["usage_sha256"],
        "reserved_tokens": entry["reserved_tokens"],
        "actual_tokens": entry["actual_tokens"],
        "entry_sha256": _digest(entry),
    }
    receipt["identity"] = _digest(receipt)
    return receipt, slot


def settled_assessment_receipt(
    budget, messages, max_tokens, response_format, usage, *, request_id=None
):
    """Authenticate a received response against its already-settled live entry."""
    from litagent.capacity_rounds import request_proof

    if request_id is None:
        current = _active_request.get()
        if current is None or current[0] != "assessment":
            raise ValueError("semantic_conflict_settlement_scope_missing")
        request_id = current[1]
    receipt, slot = _assessment_settlement(budget, request_id)
    if request_proof("assessment", messages, max_tokens, response_format) != slot:
        raise ValueError("semantic_conflict_settlement_input_invalid")
    fields = ("prompt_tokens", "completion_tokens")
    if (
        not isinstance(usage, dict)
        or usage.get("usage_available") is not True
        or any(type(usage.get(k)) is not int or usage[k] < 0 for k in fields)
        or (
            "total_tokens" in usage
            and (
                type(usage["total_tokens"]) is not int
                or usage["total_tokens"] != sum(usage[k] for k in fields)
            )
        )
        or _digest({k: usage[k] for k in fields}) != receipt["usage_sha256"]
        or sum(usage[k] for k in fields) != receipt["actual_tokens"]
    ):
        raise ValueError("semantic_conflict_settlement_usage_invalid")
    return receipt


def validate_settled_assessment_receipt(budget, receipt, *, request_id):
    """Replay a frozen receipt against the actual ledger, granting no owner."""
    expected, _ = _assessment_settlement(budget, request_id)
    if not isinstance(receipt, dict) or _digest(receipt) != _digest(expected):
        raise ValueError("semantic_conflict_settlement_receipt_invalid")
    return expected
