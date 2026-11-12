"""Validate frozen accounting data without restoring an executable budget owner."""

from __future__ import annotations

import copy
import hashlib
import json
import re

from litagent.capacity import resolve_capacity
from litagent.output_capacity import (
    envelope_context_limit,
    envelope_context_tokens,
    envelope_output_tokens,
)
from litagent.review_policy import parent_stage

_CONTRACTS = {f"planned-budget-v{version}" for version in range(2, 8)}
_PINNED_SUPPORT_REVISIONS = frozenset(
    ("host-support-receipts-v1", "host-support-receipts-v2")
)
_OUTPUTS = {
    "body": 20000,
    "body_initial": 20000,
    "summary": 20000,
    "reviewer": 4096,
    "repair": 20000,
    "inventory": 16384,
    "assessment": 16384,
    "global": 16384,
}
_STOP_REASONS = {
    None,
    "strict_worker_failed",
    "strict_usage_unknown",
    "planned_usage_bound_violation",
    "planned_ledger_persistence_failed",
}


def _require(condition, code):
    """Reject malformed cost evidence with a stable, data-only error family."""
    if not condition:
        raise ValueError("planned_accounting_" + code)


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _hashed(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _integer(value, minimum=0, maximum=2**63 - 1):
    return type(value) is int and minimum <= value <= maximum


def _keys(value, required, optional=()):
    _require(
        isinstance(value, dict)
        and set(required).issubset(value)
        and set(value).issubset(set(required) | set(optional)),
        "shape_invalid",
    )


def _envelope(value):
    """Check historical shape and arithmetic; semantic tags remain pinned data."""
    required = {
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
    optional = {
        "run_limits",
        "semantic_engine",
        "output_tokens",
        "capacity",
        "repair_mapping_contract",
        "source_map_identity",
        "source_map_json_bytes",
        "repair_guidance_contract",
        "repair_guidance_identity",
        "repair_guidance_json_bytes",
        "repair_guidance_wire_revision",
        "repair_guidance_wire_identity",
        "repair_guidance_capacity_rule",
        "global_findings_contract",
        "inventory_recovery_contract",
        "summary_constraints_contract",
        "classification_review_contract",
        "repair_feedback_contract",
        "reference_contract",
        "transport_revision",
        "transport_identity",
        "support_validation_revision",
        "support_validation_identity",
        "semantic_conflict_policy",
        "semantic_conflict_identity",
        "evaluation_response_contract",
        "evaluation_response_identity",
        "repair_prompt_revision",
        "repair_prompt_identity",
        "writer_source_view",
        "writer_source_view_identity",
        "evaluation_dimensions_contract",
        "evaluation_dimensions_identity",
        "evaluation_result_contract",
        "evaluation_result_identity",
        "assessment_decision_contract",
        "assessment_decision_identity",
        "closeout_profile",
        "closeout_profile_identity",
    }
    _require(isinstance(value, dict), "envelope_invalid")
    flat = value.get("contract") == "flat-writer-envelope-v1"
    if flat:
        required |= {"diagnostic_json_bytes", "claim_max_json_bytes"}
    _keys(value, required, optional)
    if "output_tokens" in value:
        _require(flat and "repair_mapping_contract" in value, "output_policy_invalid")
        envelope_output_tokens(value)
    _require(
        value["contract"] in {"flat-writer-envelope-v1", "budgeted-writer-envelope-v1"},
        "envelope_invalid",
    )
    if {"closeout_profile", "closeout_profile_identity"}.intersection(value):
        _require(
            value.get("closeout_profile") == "engineering-closeout-v1"
            and _hashed(value.get("closeout_profile_identity"))
            and value.get("writer_source_view") == "source-only-writer-v1"
            and value.get("evaluation_dimensions_contract")
            == "separated-evaluation-dimensions-v1",
            "envelope_closeout_profile_invalid",
        )
    if "semantic_engine" in value:
        # Historical accounting reads the frozen identity without importing live RAGAS.
        engine = value["semantic_engine"]
        _keys(
            engine,
            {"contract", "ragas_version", "identity", "batch_size"},
            {
                "global_findings_contract",
                "global_input_contract",
                "global_unit_response_contract",
                "assessment_response_contract",
            },
        )
        _require(
            engine["contract"] == "ragas-cited-faithfulness-v1"
            and engine["ragas_version"] == "0.4.3"
            and _hashed(engine["identity"])
            and _integer(engine["batch_size"], 1, 20),
            "ragas_engine_invalid",
        )
        _require(
            "global_findings_contract" not in engine
            or engine["global_findings_contract"] == "anchored-global-findings-v2",
            "ragas_global_findings_contract_invalid",
        )
        _require(
            "global_input_contract" not in engine
            or (
                engine["global_input_contract"] == "ragas-global-unit-input-v2"
                and engine.get("global_findings_contract")
                == "anchored-global-findings-v2"
            ),
            "ragas_global_input_contract_invalid",
        )
        _require(
            "global_unit_response_contract" not in engine
            or (
                engine["global_unit_response_contract"]
                in {
                    "ragas-global-unit-keys-v1",
                    "ragas-global-unit-keys-v2",
                    "ragas-global-unit-keys-v3",
                    "ragas-global-unit-keys-v4",
                }
                and engine.get("global_findings_contract")
                == "anchored-global-findings-v2"
            ),
            "ragas_global_unit_response_contract_invalid",
        )
        _require(
            "assessment_response_contract" not in engine
            or engine["assessment_response_contract"]
            == "ragas-assessment-statement-keys-v1",
            "ragas_assessment_response_contract_invalid",
        )
    if "run_limits" in value:
        from litagent.run_limits import validate_run_limits

        validate_run_limits(value["run_limits"])
        _require(flat and "repair_mapping_contract" in value, "run_limits_invalid")
    cap = resolve_capacity(value["capacity"]) if "capacity" in value else None
    limits = {
        "body_max_units": (5, 256 if cap else 66),
        "summary_max_units_per_section": (1, 3),
        "model_assertions": (1, 1024 if cap else 128),
        "assessment_batches": (1, 64 if cap else 8),
        "body_json_bytes": (1, 1000000),
        "summary_json_bytes": (1, 1000000),
        "claim_json_bytes": (1, 1000000),
        "frozen_component_reservation": (0, 2**63 - 1),
        **(
            {
                "diagnostic_json_bytes": (
                    1,
                    cap.diagnostic_json_bytes if cap and cap.revision else 12288,
                ),
                "claim_max_json_bytes": (
                    1,
                    cap.claim_max_json_bytes if cap and cap.revision else 2000,
                ),
            }
            if flat
            else {}
        ),
    }
    _require(
        all(_integer(value[key], low, high) for key, (low, high) in limits.items()),
        "envelope_limits_invalid",
    )
    if cap:
        _require(
            value["body_max_units"] == cap.max_body_units
            and value["summary_max_units_per_section"] == cap.max_summary_units
            and value["model_assertions"] >= cap.initial_assertions
            and (
                cap.max_assertions is None
                or value["model_assertions"] <= cap.max_assertions
            )
            and (
                cap.max_assessment_batches is None
                or value["assessment_batches"] <= cap.max_assessment_batches
            ),
            "capacity_invalid",
        )
    _require(
        value["body_max_units"] + 2 * value["summary_max_units_per_section"]
        <= value["model_assertions"]
        <= (cap.batch_size if cap else 16) * value["assessment_batches"],
        "envelope_scope_invalid",
    )
    _keys(value["request_caps"], set(_OUTPUTS) - {"body_initial"})
    _require(
        all(
            _integer(amount, _OUTPUTS[key])
            for key, amount in value["request_caps"].items()
        ),
        "envelope_caps_invalid",
    )
    _require(
        all(_hashed(value[key]) for key in ("source_identity", "builder_identity")),
        "envelope_identity_invalid",
    )
    dependencies = {
        "repair_mapping_contract": (
            {
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            },
            bool(cap),
        ),
        "inventory_recovery_contract": (
            {"bounded-inventory-recovery-v1", "bounded-inventory-recovery-v2"},
            value.get("global_findings_contract") == "anchored-global-findings-v2",
        ),
        "summary_constraints_contract": (
            "effective-summary-limits-v1",
            flat
            and value.get("repair_mapping_contract")
            in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            ),
        ),
        "classification_review_contract": (
            "independent-assertion-kinds-v1",
            value.get("summary_constraints_contract") == "effective-summary-limits-v1"
            and value.get("global_findings_contract") == "anchored-global-findings-v2",
        ),
        "repair_feedback_contract": (
            "host-repair-feedback-v1",
            bool(cap)
            and value.get("repair_mapping_contract")
            in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            ),
        ),
        "reference_contract": (
            "ordered-reference-set-v1",
            flat
            and bool(cap)
            and value.get("repair_mapping_contract")
            in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            ),
        ),
    }
    for key, (expected, valid) in dependencies.items():
        if key in value:
            _require(
                valid
                and (
                    isinstance(value[key], str) and value[key] in expected
                    if isinstance(expected, set)
                    else value[key] == expected
                ),
                "envelope_contract_invalid",
            )
    if "global_findings_contract" in value:
        _require(
            value.get("repair_mapping_contract")
            in (
                "capacity-edit-map-v2",
                "capacity-source-map-v3",
                "capacity-source-map-v4",
            )
            and value["global_findings_contract"]
            in {"located-global-findings-v1", "anchored-global-findings-v2"},
            "envelope_contract_invalid",
        )
    for field, identity, markers in (
        (
            "transport_revision",
            "transport_identity",
            frozenset(("host-lossless-transport-v1",)),
        ),
        (
            "support_validation_revision",
            "support_validation_identity",
            _PINNED_SUPPORT_REVISIONS,
        ),
    ):
        if field in value or identity in value:
            _require(
                isinstance(value.get(field), str)
                and value[field] in markers
                and _hashed(value.get(identity))
                and flat
                and bool(cap)
                and value.get("repair_mapping_contract")
                in (
                    "capacity-edit-map-v2",
                    "capacity-source-map-v3",
                    "capacity-source-map-v4",
                )
                and value.get("inventory_recovery_contract")
                == "bounded-inventory-recovery-v2"
                and value.get("repair_feedback_contract") == "host-repair-feedback-v1"
                and value.get("reference_contract") == "ordered-reference-set-v1"
                and (
                    field == "transport_revision"
                    or value.get("transport_revision") == "host-lossless-transport-v1"
                    and value.get("classification_review_contract")
                    == "independent-assertion-kinds-v1"
                ),
                "envelope_semantic_binding_invalid",
            )
    source_map = value.get("repair_mapping_contract") in (
        "capacity-source-map-v3",
        "capacity-source-map-v4",
    )
    if value.get("repair_mapping_contract") == "capacity-source-map-v4":
        _require(
            value.get("support_validation_revision") == "host-support-receipts-v2"
            and value.get("repair_guidance_contract") == "host-structured-repair-v1"
            and value.get("repair_guidance_wire_revision")
            == "host-repair-task-view-v1",
            "envelope_source_owner_invalid",
        )
    if source_map or "source_map_identity" in value or "source_map_json_bytes" in value:
        # Historical accounting authenticates pinned data, not current prompts.
        _require(
            source_map
            and value.get("support_validation_revision") in _PINNED_SUPPORT_REVISIONS
            and _hashed(value.get("source_map_identity"))
            and _integer(
                value.get("source_map_json_bytes"),
                1,
                cap.diagnostic_json_bytes if cap and cap.revision else 12288,
            )
            and value["source_map_json_bytes"] == value.get("diagnostic_json_bytes"),
            "envelope_source_map_invalid",
        )
    guidance_fields = {
        "repair_guidance_contract",
        "repair_guidance_identity",
        "repair_guidance_json_bytes",
    }
    if guidance_fields.intersection(value):
        # Fee evidence binds frozen semantics, never today's executable prompt.
        _require(
            guidance_fields.issubset(value)
            and value["repair_guidance_contract"] == "host-structured-repair-v1"
            and _hashed(value["repair_guidance_identity"])
            and _integer(
                value["repair_guidance_json_bytes"],
                1,
                cap.diagnostic_json_bytes if cap and cap.revision else 12288,
            )
            and value["repair_guidance_json_bytes"]
            == value.get("diagnostic_json_bytes")
            and flat
            and bool(cap)
            and source_map
            and value.get("global_findings_contract") == "anchored-global-findings-v2"
            and value.get("summary_constraints_contract")
            == "effective-summary-limits-v1"
            and value.get("inventory_recovery_contract")
            == "bounded-inventory-recovery-v2"
            and value.get("classification_review_contract")
            == "independent-assertion-kinds-v1"
            and value.get("repair_feedback_contract") == "host-repair-feedback-v1"
            and value.get("reference_contract") == "ordered-reference-set-v1"
            and value.get("transport_revision") == "host-lossless-transport-v1"
            and value.get("support_validation_revision") in _PINNED_SUPPORT_REVISIONS,
            "envelope_repair_guidance_invalid",
        )
    wire_fields = {
        "repair_guidance_wire_revision",
        "repair_guidance_wire_identity",
    }
    if wire_fields.intersection(value):
        # Historical fees bind the complete pinned pair, not today's codec code.
        _require(
            wire_fields.issubset(value)
            and guidance_fields.issubset(value)
            and type(value["repair_guidance_wire_revision"]) is str
            and value["repair_guidance_wire_revision"]
            in ("host-unit-ref-guidance-v1", "host-repair-task-view-v1")
            and _hashed(value["repair_guidance_wire_identity"])
            and source_map
            and value["repair_guidance_contract"] == "host-structured-repair-v1",
            "envelope_repair_guidance_wire_invalid",
        )
    task_view = value.get("repair_guidance_wire_revision") == "host-repair-task-view-v1"
    if {"semantic_conflict_policy", "semantic_conflict_identity"}.intersection(value):
        # Historical identities stay pinned data even after the live rule changes.
        _require(
            value.get("semantic_conflict_policy") == "host-unverified-continue-v1"
            and _hashed(value.get("semantic_conflict_identity"))
            and flat
            and bool(cap)
            and value.get("support_validation_revision") == "host-support-receipts-v2"
            and value.get("repair_mapping_contract") == "capacity-source-map-v4"
            and value.get("repair_guidance_contract") == "host-structured-repair-v1"
            and task_view,
            "envelope_semantic_conflict_invalid",
        )
    if {
        "evaluation_response_contract",
        "evaluation_response_identity",
    }.intersection(value):
        # Historical fees authenticate the pinned pair, not today's schema code.
        _require(
            value.get("evaluation_response_contract")
            in ("host-bound-evaluation-v1", "host-bound-evaluation-v2")
            and _hashed(value.get("evaluation_response_identity"))
            and flat
            and bool(cap)
            and value.get("support_validation_revision") == "host-support-receipts-v2"
            and value.get("inventory_recovery_contract")
            == "bounded-inventory-recovery-v2"
            and value.get("classification_review_contract")
            == "independent-assertion-kinds-v1"
            and value.get("global_findings_contract") == "anchored-global-findings-v2"
            and value.get("summary_constraints_contract")
            == "effective-summary-limits-v1"
            and value.get("repair_feedback_contract") == "host-repair-feedback-v1"
            and value.get("reference_contract") == "ordered-reference-set-v1"
            and value.get("transport_revision") == "host-lossless-transport-v1",
            "envelope_evaluation_response_invalid",
        )
    if {"repair_prompt_revision", "repair_prompt_identity"}.intersection(value):
        # Historical prompt identities authenticate data without executing renderers.
        _require(
            value.get("repair_prompt_revision") == "concise-grounded-repair-v1"
            and _hashed(value.get("repair_prompt_identity"))
            and flat
            and bool(cap)
            and value.get("repair_mapping_contract") == "capacity-source-map-v4"
            and value.get("support_validation_revision") == "host-support-receipts-v2"
            and value.get("repair_guidance_contract") == "host-structured-repair-v1"
            and task_view,
            "envelope_repair_prompt_invalid",
        )
    if {"writer_source_view", "writer_source_view_identity"}.intersection(value):
        # Historical accounting authenticates pinned data, never today's view.
        _require(
            value.get("writer_source_view") == "source-only-writer-v1"
            and _hashed(value.get("writer_source_view_identity"))
            and flat
            and bool(cap)
            and value.get("repair_mapping_contract") == "capacity-source-map-v4"
            and value.get("support_validation_revision") == "host-support-receipts-v2"
            and value.get("inventory_recovery_contract")
            == "bounded-inventory-recovery-v2"
            and value.get("classification_review_contract")
            == "independent-assertion-kinds-v1"
            and value.get("global_findings_contract") == "anchored-global-findings-v2"
            and value.get("summary_constraints_contract")
            == "effective-summary-limits-v1"
            and value.get("repair_feedback_contract") == "host-repair-feedback-v1"
            and value.get("reference_contract") == "ordered-reference-set-v1"
            and value.get("transport_revision") == "host-lossless-transport-v1"
            and value.get("repair_guidance_contract") == "host-structured-repair-v1"
            and task_view
            and value.get("evaluation_response_contract") == "host-bound-evaluation-v2"
            and value.get("repair_prompt_revision") == "concise-grounded-repair-v1",
            "envelope_writer_source_view_invalid",
        )
    if {
        "evaluation_dimensions_contract",
        "evaluation_dimensions_identity",
    }.intersection(value):
        # Historical identities are pinned data, not today's executable rules.
        _require(
            value.get("evaluation_dimensions_contract")
            == "separated-evaluation-dimensions-v1"
            and _hashed(value.get("evaluation_dimensions_identity"))
            and flat
            and bool(cap)
            and value.get("support_validation_revision") == "host-support-receipts-v2"
            and value.get("evaluation_response_contract") == "host-bound-evaluation-v2"
            and value.get("inventory_recovery_contract")
            == "bounded-inventory-recovery-v2"
            and value.get("classification_review_contract")
            == "independent-assertion-kinds-v1"
            and value.get("global_findings_contract") == "anchored-global-findings-v2"
            and value.get("summary_constraints_contract")
            == "effective-summary-limits-v1",
            "envelope_evaluation_dimensions_invalid",
        )
    if {"evaluation_result_contract", "evaluation_result_identity"}.intersection(value):
        _require(
            value.get("evaluation_result_contract") == "independent-evaluation-v1"
            and _hashed(value.get("evaluation_result_identity"))
            and flat
            and bool(cap)
            and value.get("support_validation_revision") == "host-support-receipts-v2"
            and value.get("global_findings_contract") == "anchored-global-findings-v2",
            "envelope_evaluation_results_invalid",
        )
    if {"assessment_decision_contract", "assessment_decision_identity"}.intersection(
        value
    ):
        _require(
            value.get("assessment_decision_contract")
            in {
                "reason-only-decision-v1",
                "reason-only-decision-v2",
                "reason-only-decision-v3",
            }
            and _hashed(value.get("assessment_decision_identity"))
            and value.get("evaluation_result_contract") == "independent-evaluation-v1"
            and value.get("evaluation_dimensions_contract")
            == "separated-evaluation-dimensions-v1",
            "envelope_assessment_decision_invalid",
        )
    _require(
        task_view == ("repair_guidance_capacity_rule" in value)
        and (
            not task_view
            or value["repair_guidance_capacity_rule"]
            == "complete-locked-request-reservation-v1"
        ),
        "envelope_repair_guidance_capacity_rule_invalid",
    )
    _require(
        value["identity"]
        == _digest({key: item for key, item in value.items() if key != "identity"}),
        "envelope_digest_invalid",
    )
    return cap


def _margin(required):
    return max(500000, ((required * 11 + 99999) // 100000) * 10000)


def _joint_plan(plan):
    """Authenticate historical round quotas without executing the live rule."""
    _require(plan.get("contract") == "planned-budget-v7", "joint_plan_invalid")
    domain = plan["joint_round_escrow"]
    _keys(
        domain,
        {"contract", "rule_identity", "base_plan_identity", "rounds", "assessment"},
    )
    limits = domain["assessment"]
    _keys(limits, {"max_requests", "request_cap", "reserved_tokens"})
    _require(
        domain["contract"] == "joint-round-escrow-v1"
        and _hashed(domain["rule_identity"])
        and _hashed(domain["base_plan_identity"])
        and _integer(domain["rounds"], 3, 3)
        and _integer(limits["max_requests"], 1, 64)
        and _integer(
            limits["request_cap"],
            (
                plan["envelope"]["request_caps"]["assessment"]
                if "run_limits" in plan["envelope"]
                else 100000
            ),
            (
                plan["envelope"]["request_caps"]["assessment"]
                if "run_limits" in plan["envelope"]
                else 100000
            ),
        )
        and _integer(limits["reserved_tokens"], 16384),
        "joint_plan_invalid",
    )
    baseline = copy.deepcopy(plan)
    baseline.pop("joint_round_escrow")
    assessment = baseline["stages"]["assessment"]
    assessment["count"] = baseline["envelope"]["assessment_batches"] * (
        1 + baseline["scope_bounds"]["shared_repairs"]
    )
    assessment["reserved_tokens"] = assessment["count"] * assessment["request_cap"]
    baseline["required_tokens"] = baseline["spent_tokens"] + sum(
        spec["reserved_tokens"] for spec in baseline["stages"].values()
    )
    baseline["planned_requests"] = baseline["spent_requests"] + sum(
        spec["count"] for spec in baseline["stages"].values()
    )
    baseline["planned_tokens"] = _margin(baseline["required_tokens"])
    baseline.pop("identity", None)
    baseline["identity"] = _digest(baseline)
    _plan(baseline)
    _require(
        baseline["scope_bounds"]["shared_repairs"] == 2
        and domain["base_plan_identity"] == baseline["identity"],
        "joint_base_plan_invalid",
    )
    capacity = resolve_capacity(baseline["envelope"]["capacity"])
    rounds = domain["rounds"]
    others = [spec for name, spec in baseline["stages"].items() if name != "assessment"]
    fixed_tokens = baseline["spent_tokens"] + sum(
        spec["reserved_tokens"] for spec in others
    )
    fixed_requests = baseline["spent_requests"] + sum(spec["count"] for spec in others)
    count = min(
        capacity.max_assessment_batches or 64,
        (baseline["maximum_requests"] - fixed_requests) // rounds,
    )
    total = min(
        count * assessment["request_cap"],
        ((baseline["planned_tokens"] * 10 // 11) - fixed_tokens) // rounds,
    )
    _require(
        count >= baseline["envelope"]["assessment_batches"]
        and limits
        == {
            "max_requests": count,
            "request_cap": assessment["request_cap"],
            "reserved_tokens": total,
        },
        "joint_domain_invalid",
    )
    result = copy.deepcopy(baseline)
    result["joint_round_escrow"] = copy.deepcopy(domain)
    result["stages"]["assessment"].update(
        count=rounds * count, reserved_tokens=rounds * total
    )
    result["required_tokens"] = fixed_tokens + rounds * total
    result["planned_requests"] = fixed_requests + rounds * count
    _require(
        _margin(result["required_tokens"]) <= baseline["planned_tokens"]
        and result["planned_requests"] <= baseline["maximum_requests"],
        "joint_unaffordable",
    )
    result.pop("identity")
    result["identity"] = _digest(result)
    _require(_json(result) == _json(plan), "joint_plan_identity_invalid")


def _plan(plan):
    """Reconstruct the published v2--v7 accounting plan, without current prompts."""
    _require(
        isinstance(plan, dict) and plan.get("contract") in _CONTRACTS, "plan_invalid"
    )
    context_limit = envelope_context_limit(plan.get("envelope"))
    if "joint_round_escrow" in plan:
        _joint_plan(plan)
        return
    contract = plan["contract"]
    adaptive, flat, capacity = (
        contract != "planned-budget-v2",
        contract in {"planned-budget-v5", "planned-budget-v6", "planned-budget-v7"},
        contract in {"planned-budget-v6", "planned-budget-v7"},
    )
    enveloped = contract in {
        "planned-budget-v4",
        "planned-budget-v5",
        "planned-budget-v6",
        "planned-budget-v7",
    }
    stages = plan.get("stages")
    _require(
        isinstance(stages, dict)
        and stages
        and isinstance(plan.get("input_identity"), str)
        and plan["input_identity"],
        "plan_invalid",
    )
    from litagent.run_limits import (
        envelope_run_limits,
        extended_repair_stages,
        format_recovery_stages,
        repair_allowances,
    )

    resources = envelope_run_limits(plan.get("envelope"))
    maximum_requests, maximum_repairs = (
        resources["max_requests"],
        resources["max_repairs"],
    )
    spent, calls = plan.get("spent_tokens"), plan.get("spent_requests")
    repairs = plan.get("scope_bounds", {}).get("shared_repairs")
    _require(
        _integer(spent)
        and _integer(calls, 0, maximum_requests)
        and _integer(repairs, 0, maximum_repairs)
        and (contract != "planned-budget-v2" or repairs == 2),
        "plan_invalid",
    )
    projected = {}
    for name, spec in stages.items():
        _keys(
            spec, {"count", "request_cap", "output_tokens", "bound", "reserved_tokens"}
        )
        _require(
            isinstance(name, str)
            and name
            and all(
                _integer(spec[k], 1) for k in ("count", "request_cap", "output_tokens")
            )
            and spec["output_tokens"] <= spec["request_cap"] <= context_limit
            and spec["bound"]
            in (
                {"exact_wire", "context_upper_bound", "host_envelope_upper_bound"}
                if enveloped
                else {"exact_wire", "context_upper_bound"}
            ),
            "plan_stage_invalid",
        )
        projected[name] = {
            **spec,
            "reserved_tokens": spec["count"] * spec["request_cap"],
        }
    envelope = plan.get("envelope")
    if enveloped:
        _envelope(envelope)
        _require(
            "semantic_conflict_policy" not in envelope
            or contract == "planned-budget-v7",
            "semantic_conflict_plan_invalid",
        )
        _require(
            "evaluation_response_contract" not in envelope
            or contract == "planned-budget-v7",
            "evaluation_response_plan_invalid",
        )
        _require(
            "repair_prompt_revision" not in envelope or contract == "planned-budget-v7",
            "repair_prompt_plan_invalid",
        )
        _require(
            "writer_source_view" not in envelope or contract == "planned-budget-v7",
            "writer_source_view_plan_invalid",
        )
        _require(
            "evaluation_dimensions_contract" not in envelope
            or contract == "planned-budget-v7",
            "evaluation_dimensions_plan_invalid",
        )
    else:
        _require(envelope is None, "plan_invalid")
    format_stages = format_recovery_stages(envelope)
    if adaptive:
        outputs = (
            {
                **envelope_output_tokens(envelope),
                "body_initial": envelope_output_tokens(envelope)["body"],
            }
            if envelope and "output_tokens" in envelope
            else _OUTPUTS
        )
        outputs = {
            **outputs,
            **{
                key: outputs[parent_stage(parent)]
                for key, (parent, _) in format_stages.items()
            },
        }
        counts = {key: item["count"] for key, item in projected.items()}
        for key, (_, count) in format_stages.items():
            _require(counts.pop(key, None) == count, "plan_format_path_invalid")
        body = counts.pop("body", 0), counts.pop("body_initial", 0)
        _require(
            body in {(5, 0), (4, 1)}
            and counts
            == (
                extended_repair_stages(envelope, repairs)
                or {
                    "summary": 1 + repairs,
                    "reviewer": 1,
                    "inventory": (
                        (
                            envelope["body_max_units"]
                            + 2 * envelope["summary_max_units_per_section"]
                            + envelope["semantic_engine"]["batch_size"]
                            - 1
                        )
                        // envelope["semantic_engine"]["batch_size"]
                        if envelope and "semantic_engine" in envelope
                        else 1
                    )
                    * (1 + repairs),
                    "assessment": (
                        (
                            (
                                envelope["body_max_units"]
                                + 2 * envelope["summary_max_units_per_section"]
                                + envelope["semantic_engine"]["batch_size"]
                                - 1
                            )
                            // envelope["semantic_engine"]["batch_size"]
                            if "semantic_engine" in envelope
                            else envelope["assessment_batches"]
                        )
                        if envelope
                        else 8
                    )
                    * (1 + repairs),
                    "global": 1 + repairs,
                    **({"repair": repairs} if repairs else {}),
                }
            )
            and all(
                item["output_tokens"]
                == outputs.get(key, outputs.get(parent_stage(key)))
                for key, item in projected.items()
            ),
            "plan_complete_path_invalid",
        )
    if envelope:
        _require(
            all(
                spec["request_cap"]
                == envelope["request_caps"][
                    parent_stage(format_stages.get(key, (key,))[0])
                ]
                for key, spec in projected.items()
                if key != "body_initial"
            )
            and envelope["frozen_component_reservation"]
            <= min(
                envelope["request_caps"][key]
                for key in ("inventory", "assessment", "global")
            ),
            "plan_envelope_invalid",
        )
    if flat:
        _require(
            2 <= repairs <= maximum_repairs
            and envelope["contract"] == "flat-writer-envelope-v1"
            and ("capacity" in envelope) == capacity
            and ("repair_mapping_contract" in envelope)
            == (contract == "planned-budget-v7")
            and (
                contract != "planned-budget-v5"
                or envelope["body_max_units"] == 66
                and envelope["model_assertions"] == 128
            ),
            "plan_scope_invalid",
        )
    required = spent + sum(item["reserved_tokens"] for item in projected.values())
    requests = calls + sum(item["count"] for item in projected.values())
    maximum = 4500000 if flat else 1000000
    if envelope and "run_limits" in envelope:
        from litagent.run_limits import validate_run_limits

        _require(contract == "planned-budget-v7", "run_limits_invalid")
        maximum = validate_run_limits(envelope["run_limits"])["max_tokens"]
    _require(
        requests <= maximum_requests and _margin(required) <= maximum, "plan_unadmitted"
    )
    result = {
        "contract": contract,
        "measurement": "complete-utf8-reservation-v1",
        "input_identity": plan["input_identity"],
        "status": "admitted",
        "reason": None,
        "required_tokens": required,
        "planned_tokens": _margin(required),
        "maximum_tokens": maximum,
        "planned_requests": requests,
        "maximum_requests": maximum_requests,
        "context_tokens": envelope_context_tokens(plan.get("envelope")),
        "context_reservation_limit": context_limit,
        "context_margin": 0.1,
        "spent_tokens": spent,
        "spent_requests": calls,
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
    if adaptive:
        result.update(
            repair_limit=repairs,
            repair_selection=(
                (
                    "bounded-complete-repairs-v2"
                    if maximum_repairs > 2
                    else "fixed-two-complete-repairs-v1"
                )
                if flat
                else "largest-admissible-complete-path-v1"
            ),
            repair_allocation="locked",
            initial_evaluation_included=True,
            initial_evaluation_reserved=True,
            assessment_input_contract="host-citation-support-aliases-v2",
        )
    if envelope:
        result["envelope"] = envelope
        if limits := repair_allowances(envelope):
            result["scope_bounds"]["repair_allowances"] = limits
        result["scope_bounds"].update(
            report_max_units=envelope["body_max_units"]
            + 2 * envelope["summary_max_units_per_section"],
            model_assertions=envelope["model_assertions"],
            assessment_batches=envelope["assessment_batches"],
            envelope_contract=envelope["contract"],
        )
        if envelope.get("capacity", {}).get("revision"):
            result["scope_bounds"].update(
                authored_text_max_characters=envelope["capacity"][
                    "max_unit_characters"
                ],
                diagnostic_string_max_characters=envelope["capacity"][
                    "max_unit_characters"
                ],
                assertions_per_batch=envelope["capacity"]["batch_size"],
            )
    result["identity"] = _digest(result)
    _require(_json(result) == _json(plan), "plan_identity_invalid")


def _entries(entries, stages, *, stop=None):
    """Validate raw entry states without normalizing pending or unsent requests."""
    _require(isinstance(entries, dict), "entries_invalid")
    for request_id, entry in entries.items():
        _require(isinstance(request_id, str) and request_id, "entry_invalid")
        _keys(
            entry,
            {
                "stage",
                "status",
                "reserved_tokens",
                "actual_tokens",
                "usage_sha256",
                "sent",
            },
            {"wire_sha256", "round_number", "round_wire_sha256", "inventory_recovery"},
        )
        stage = entry["stage"]
        _require(
            stage in stages
            and _integer(entry["reserved_tokens"], 1, stages[stage]["request_cap"])
            and type(entry["sent"]) is bool
            and entry["status"]
            in {
                "reserved_not_sent",
                "started_no_replay",
                "settled",
                "interrupted_no_replay",
            },
            "entry_invalid",
        )
        if "wire_sha256" in entry:
            _require(
                entry["wire_sha256"] is None or _hashed(entry["wire_sha256"]),
                "entry_wire_invalid",
            )
        if entry["status"] == "settled":
            _require(
                entry["sent"]
                and _integer(entry["actual_tokens"])
                and _hashed(entry["usage_sha256"])
                and (
                    entry["actual_tokens"] <= entry["reserved_tokens"]
                    or stop == "planned_usage_bound_violation"
                ),
                "entry_usage_invalid",
            )
        else:
            _require(
                entry["actual_tokens"] is None
                and entry["usage_sha256"] is None
                and (entry["status"] != "reserved_not_sent" or not entry["sent"])
                and (entry["status"] != "started_no_replay" or entry["sent"]),
                "entry_status_invalid",
            )
    for stage, spec in stages.items():
        own = [entry for entry in entries.values() if entry["stage"] == stage]
        _require(
            len(own) <= spec["count"]
            and (
                stop == "planned_usage_bound_violation"
                or sum(
                    (
                        entry["actual_tokens"]
                        if entry["status"] == "settled"
                        else entry["reserved_tokens"]
                    )
                    for entry in own
                )
                <= spec["reserved_tokens"]
            ),
            "entry_stage_capacity_invalid",
        )


def _recovery(plan, entry, previous):
    """Verify historical recovery purpose against its exact parent allocation."""
    value = entry.get("inventory_recovery")
    if value is None:
        return False
    _keys(
        value,
        {
            "contract",
            "parent_allocation",
            "source_report_sha256",
            "feedback_sha256",
            "wire_sha256",
        },
    )
    _require(
        plan["contract"] == "planned-budget-v7"
        and entry["stage"] == "repair"
        and value["contract"] == plan["envelope"].get("inventory_recovery_contract")
        and value["contract"]
        in {"bounded-inventory-recovery-v1", "bounded-inventory-recovery-v2"}
        and previous
        and previous["round"]["mode"] == "assess"
        and value["parent_allocation"] == previous["identity"]
        and value["source_report_sha256"] == previous["round"]["report_sha256"]
        and all(_hashed(item) for key, item in value.items() if key != "contract"),
        "recovery_binding_invalid",
    )
    return True


def _allocation(plan, entries, arguments, previous, *, round_budget):
    """Recompute allocation arithmetic using immutable dictionaries only."""
    cap = resolve_capacity(plan["envelope"]["capacity"])
    assertions, batches = arguments["assertions"], arguments["batches"]
    _require(
        _integer(assertions, 1)
        and _integer(batches, 1, 64)
        and assertions <= batches * cap.batch_size
        and (cap.max_assertions is None or assertions <= cap.max_assertions)
        and (
            cap.max_assessment_batches is None or batches <= cap.max_assessment_batches
        ),
        "allocation_scope_invalid",
    )
    _require(
        entries and all(entry["status"] == "settled" for entry in entries.values()),
        "allocation_boundary_invalid",
    )
    last = list(entries.values())[-1]
    _require(
        last["stage"] == "inventory"
        or round_budget
        and _recovery(plan, last, previous),
        "allocation_boundary_invalid",
    )
    own = {
        key: [entry for entry in entries.values() if entry["stage"] == key]
        for key in plan["stages"]
    }
    stages = copy.deepcopy(plan["stages"])
    envelope = copy.deepcopy(
        plan["envelope"] if round_budget else (previous or plan)["envelope"]
    )
    joint = plan.get("joint_round_escrow")
    envelope["model_assertions"] = max(
        cap.initial_assertions if round_budget else envelope["model_assertions"],
        assertions,
    )
    envelope["assessment_batches"] = max(envelope["assessment_batches"], batches)
    if round_budget:
        _keys(arguments, {"assertions", "batches", "requests", "report_sha256", "mode"})
        requests, mode = arguments["requests"], arguments["mode"]
        number = len(own["inventory"]) + sum(
            "inventory_recovery" in entry for entry in own.get("repair", [])
        )
        repairs = plan["repair_limit"] - len(own.get("repair", []))
        _require(
            all(
                len(own[key]) == plan["stages"][key]["count"]
                for key in ("body_initial", "body", "reviewer")
            )
            and mode in {"assess", "refine"}
            and _hashed(arguments["report_sha256"])
            and repairs >= 0
            and number <= 1 + len(own.get("repair", []))
            and (not previous or number > previous["round"]["number"])
            and (mode != "refine" or repairs > 0)
            and isinstance(requests, list)
            and len(requests) == (batches + 1 if mode == "assess" else 0),
            "round_boundary_invalid",
        )
        for index, request in enumerate(requests):
            _keys(request, {"stage", "wire_sha256", "reserved_tokens", "output_tokens"})
            _require(
                request["stage"] == ("global" if index == batches else "assessment")
                and _hashed(request["wire_sha256"])
                and _integer(
                    request["reserved_tokens"],
                    plan["stages"]["assessment"]["output_tokens"],
                    (
                        plan["stages"][request["stage"]]["request_cap"]
                        if joint
                        else plan["context_reservation_limit"]
                    ),
                )
                and type(request["output_tokens"]) is int
                and request["output_tokens"]
                == plan["stages"][request["stage"]]["output_tokens"],
                "round_slot_invalid",
            )
        for name, spec in stages.items():
            exact = [request for request in requests if request["stage"] == name]
            future = (
                repairs
                * (
                    joint["assessment"]["max_requests"]
                    if joint
                    else envelope["assessment_batches"]
                )
                if name == "assessment"
                else (
                    repairs
                    if name in {"summary", "inventory", "global", "repair"}
                    else 0
                )
            )
            spec.update(
                count=len(own[name]) + len(exact) + future,
                request_cap=max(
                    [
                        spec["request_cap"],
                        *[request["reserved_tokens"] for request in exact],
                    ]
                ),
                reserved_tokens=sum(entry["actual_tokens"] for entry in own[name])
                + sum(request["reserved_tokens"] for request in exact)
                + (
                    repairs * joint["assessment"]["reserved_tokens"]
                    if joint and name == "assessment"
                    else future * plan["stages"][name]["request_cap"]
                ),
            )
        round_ = {
            "number": number,
            "report_sha256": arguments["report_sha256"],
            "mode": mode,
            "status": "allocated" if mode == "assess" else "superseded_unverified",
            "assertions": assertions,
            "batches": batches,
            "remaining_repairs": repairs,
            "offsets": {key: len(own[key]) for key in ("assessment", "global")},
            "requests": requests,
        }
    else:
        _keys(arguments, {"assertions", "batches", "request_caps"})
        caps = arguments["request_caps"]
        _require(
            isinstance(caps, dict)
            and set(caps).issubset({"assessment", "global"})
            and all(
                _integer(amount, 1, plan["context_reservation_limit"])
                for amount in caps.values()
            ),
            "allocation_caps_invalid",
        )
        for stage, amount in caps.items():
            envelope["request_caps"][stage] = max(
                envelope["request_caps"][stage], amount
            )
        remaining = stages["inventory"]["count"] - len(own["inventory"]) + 1
        _require(remaining >= 1, "allocation_boundary_invalid")
        stages["assessment"]["count"] = (
            len(own["assessment"]) + remaining * envelope["assessment_batches"]
        )
        for name, spec in stages.items():
            if name in envelope["request_caps"]:
                spec["request_cap"] = envelope["request_caps"][name]
            _require(len(own[name]) <= spec["count"], "allocation_stage_invalid")
            spec["reserved_tokens"] = (
                sum(entry["actual_tokens"] for entry in own[name])
                + (spec["count"] - len(own[name])) * spec["request_cap"]
            )
    envelope["identity"] = _digest(
        {key: item for key, item in envelope.items() if key != "identity"}
    )
    _envelope(envelope)
    required = plan["spent_tokens"] + sum(
        spec["reserved_tokens"] for spec in stages.values()
    )
    requests_count = plan["spent_requests"] + sum(
        spec["count"] for spec in stages.values()
    )
    _require(
        _margin(required) <= plan["planned_tokens"]
        and requests_count <= plan["maximum_requests"],
        "allocation_unaffordable",
    )
    result = {
        "envelope": envelope,
        "stages": stages,
        "required_tokens": required,
        "with_margin_tokens": _margin(required),
        "planned_requests": requests_count,
        **({"round": round_} if round_budget else {}),
    }
    result["identity"] = _digest(result)
    return result


def _revisions(snapshot, plan):
    """Authenticate each ordered settled prefix and every actual round slot."""
    native = "semantic_engine" in plan.get("envelope", {})
    round_budget = plan["contract"] == "planned-budget-v7" and not native
    revisions = snapshot["capacity_revisions"]
    if native:
        _require(
            not revisions and snapshot["capacity_allocation"] is None,
            "ragas_static_escrow_invalid",
        )
    _require(
        isinstance(revisions, list) and len(revisions) <= (3 if round_budget else 64),
        "revision_chain_invalid",
    )
    final, parent, previous, prefix = snapshot["entries"], plan["identity"], None, {}
    for event in revisions:
        fields = {"parent", "entries", "allocation", "identity"} | (
            {"arguments"}
            if round_budget
            else {"assertions", "batches", "request_caps", "reason"}
        )
        _keys(event, fields)
        entries = event["entries"]
        _require(
            event["parent"] == parent
            and event["identity"]
            == _digest({key: item for key, item in event.items() if key != "identity"})
            and isinstance(entries, dict)
            and list(entries.items())[: len(prefix)] == list(prefix.items())
            and list(final.items())[: len(entries)] == list(entries.items()),
            "revision_chain_invalid",
        )
        prior = (previous or plan)["stages"]
        _entries(entries, prior)
        _require(
            all(entry["status"] == "settled" for entry in entries.values()),
            "revision_receipt_invalid",
        )
        if round_budget:
            arguments = event["arguments"]
        else:
            _require(
                isinstance(event["reason"], str) and event["reason"],
                "revision_reason_invalid",
            )
            arguments = {
                key: event[key] for key in ("assertions", "batches", "request_caps")
            }
        rebuilt = _allocation(
            plan, entries, arguments, previous, round_budget=round_budget
        )
        _require(
            _json(rebuilt) == _json(event["allocation"]), "revision_allocation_invalid"
        )
        parent, previous, prefix = event["identity"], rebuilt, entries
    _require(
        _json(snapshot["capacity_allocation"]) == _json(previous),
        "revision_allocation_invalid",
    )
    if round_budget and "semantic_engine" not in plan["envelope"]:
        slots = {}
        for event in revisions:
            round_ = event["allocation"]["round"]
            for stage in ("assessment", "global"):
                for index, slot in enumerate(
                    slot for slot in round_["requests"] if slot["stage"] == stage
                ):
                    key = stage, round_["offsets"][stage] + index
                    _require(key not in slots, "round_slot_reused")
                    slots[key] = round_["number"], slot
        for stage in ("assessment", "global"):
            for index, entry in enumerate(
                entry for entry in final.values() if entry["stage"] == stage
            ):
                expected = slots.get((stage, index))
                _require(
                    expected
                    and _integer(entry.get("round_number"), 1)
                    and (
                        entry["round_number"],
                        entry.get("round_wire_sha256"),
                        entry["reserved_tokens"],
                    )
                    == (
                        expected[0],
                        expected[1]["wire_sha256"],
                        expected[1]["reserved_tokens"],
                    ),
                    "round_receipt_invalid",
                )
    for index, entry in enumerate(final.values()):
        if "inventory_recovery" in entry:
            candidates = [
                event["allocation"]
                for event in revisions
                if len(event["entries"]) <= index
            ]
            _require(
                _recovery(plan, entry, candidates[-1] if candidates else None),
                "recovery_binding_invalid",
            )
            prior_entries = list(final.values())[:index]
            round_ = candidates[-1]["round"]
            _require(
                all(value["status"] == "settled" for value in prior_entries)
                and sum(value["stage"] == "assessment" for value in prior_entries)
                == round_["offsets"]["assessment"] + round_["batches"]
                and sum(value["stage"] == "global" for value in prior_entries)
                == round_["offsets"]["global"] + 1,
                "recovery_boundary_invalid",
            )
        _require(
            ("round_number" in entry or "round_wire_sha256" in entry)
            == (round_budget and entry["stage"] in {"assessment", "global"}),
            "entry_round_invalid",
        )
    return (previous or plan)["stages"]


def validate_accounting_snapshot(snapshot: dict, *, expected_plan: dict) -> dict:
    """Return exact frozen schema2 data, granting no resume or spending authority."""
    try:
        _plan(expected_plan)
        _require(
            isinstance(snapshot, dict)
            and _json(snapshot.get("plan")) == _json(expected_plan),
            "pinned_plan_conflict",
        )
        capacity = expected_plan["contract"] in {
            "planned-budget-v6",
            "planned-budget-v7",
        }
        fields = {
            "schema_version",
            "strict_policy",
            "plan",
            "max_tokens",
            "used_tokens",
            "call_count",
            "requests_sent",
            "usage_uncertain",
            "stop_reason",
            "entries",
            "closed_stages",
        } | ({"capacity_revisions", "capacity_allocation"} if capacity else set())
        _keys(snapshot, fields)
        _require(
            type(snapshot["schema_version"]) is int
            and snapshot["schema_version"] == 2
            and snapshot["strict_policy"] == expected_plan["contract"]
            and type(snapshot["max_tokens"]) is int
            and snapshot["max_tokens"] == expected_plan["planned_tokens"]
            and all(
                _integer(snapshot[key])
                for key in ("used_tokens", "call_count", "requests_sent")
            )
            and type(snapshot["usage_uncertain"]) is bool
            and snapshot["stop_reason"] in _STOP_REASONS,
            "snapshot_invalid",
        )
        _require(isinstance(snapshot["entries"], dict), "entries_invalid")
        # Validate entry fields before any revision arithmetic dereferences them.
        broad = {
            key: {
                **spec,
                "count": expected_plan["maximum_requests"],
                "request_cap": expected_plan["context_reservation_limit"],
                "reserved_tokens": 2**63 - 1,
            }
            for key, spec in expected_plan["stages"].items()
        }
        _entries(snapshot["entries"], broad, stop=snapshot["stop_reason"])
        stages = (
            _revisions(snapshot, expected_plan) if capacity else expected_plan["stages"]
        )
        _entries(snapshot["entries"], stages, stop=snapshot["stop_reason"])
        if expected_plan["contract"] != "planned-budget-v7":
            _require(
                all(
                    not {
                        "round_number",
                        "round_wire_sha256",
                        "inventory_recovery",
                    }.intersection(entry)
                    for entry in snapshot["entries"].values()
                ),
                "entry_round_invalid",
            )
        settled = [
            entry
            for entry in snapshot["entries"].values()
            if entry["status"] == "settled"
        ]
        _require(
            (snapshot["used_tokens"], snapshot["call_count"], snapshot["requests_sent"])
            == (
                expected_plan["spent_tokens"]
                + sum(entry["actual_tokens"] for entry in settled),
                expected_plan["spent_requests"] + len(settled),
                expected_plan["spent_requests"]
                + sum(entry["sent"] for entry in snapshot["entries"].values()),
            )
            and snapshot["requests_sent"] <= expected_plan["maximum_requests"],
            "totals_invalid",
        )
        closed = snapshot["closed_stages"]
        _require(
            isinstance(closed, list)
            and all(isinstance(stage, str) for stage in closed)
            and len(closed) == len(set(closed))
            and set(closed).issubset(stages)
            and all(
                entry["status"] == "settled"
                for entry in snapshot["entries"].values()
                if entry["stage"] in closed
            ),
            "closed_stages_invalid",
        )
        _require(
            not any(
                entry["status"] == "interrupted_no_replay"
                for entry in snapshot["entries"].values()
            )
            or snapshot["usage_uncertain"],
            "uncertainty_invalid",
        )
        return copy.deepcopy(snapshot)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("planned_accounting_snapshot_invalid") from exc
