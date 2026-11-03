"""Prepare complete evaluations under the explicitly selected capacity policy."""

from collections import Counter

from litagent.capacity import POLICY, resolve_capacity
from litagent.observability.context import remaining_deadline_seconds


def capacity_diagnostic(progress, reason, *, batches=None):
    """Describe capacity without calling any unassessed claim scientifically false."""
    claims = progress.get("inventory_assertions", [])
    return {
        "policy": POLICY,
        "authored_unit_counts": progress.get("authored_unit_counts", {}),
        "status": "needs_refinement",
        "reason": reason,
        "actual_assertions": len(claims),
        "actual_batches": batches,
        "assertions_by_fragment": dict(
            Counter(c["source_fragment_id"] for c in claims)
        ),
        "assertions_by_section": dict(Counter(c["section"] for c in claims)),
        "scientific_verdict": "unverified",
    }


def adaptive_eval_config(config, owner):
    """Apply operational limits without reusing the historical 128/8 validators."""
    capacity = resolve_capacity(owner.plan["envelope"]["capacity"])
    return config.model_copy(
        update={
            "strict_max_claims": capacity.max_assertions
            or owner.plan["maximum_requests"] * capacity.batch_size,
            "strict_max_batches": capacity.max_assessment_batches
            or owner.plan["maximum_requests"],
            "strict_batch_size": capacity.batch_size,
        }
    )


def refine_rejected_round(owner, progress, reason, report_sha256, *, batches=None):
    """Lock a feasible candidate after a complete inventory exceeds a resource cap."""
    from litagent.capacity_rounds import choose_refinement

    actual = len(progress["inventory_assertions"])
    cap = resolve_capacity(owner.plan["envelope"]["capacity"])
    batches = batches or (actual + cap.batch_size - 1) // cap.batch_size
    progress["capacity"] = {
        **capacity_diagnostic(progress, reason, batches=batches),
        "budget_policy": "planned-budget-v7",
        "source_report_sha256": report_sha256,
        "initial_assertions": cap.initial_assertions,
        "fixed_allowance": owner.plan["planned_tokens"],
    }
    try:
        allocation = choose_refinement(
            owner, assertions=actual, batches=batches, report_sha256=report_sha256
        )
    except ValueError as exc:
        if str(exc) not in {
            "capacity_no_affordable_refinement",
            "capacity_no_repair_available",
        }:
            raise
        progress["capacity"].update(status="exhausted", reason=str(exc))
        raise ValueError("evaluation_capacity_exhausted") from exc
    progress["capacity"].update(
        target_assertions=allocation["round"]["assertions"],
        target_batches=allocation["round"]["batches"],
        effective_assertions=allocation["envelope"]["model_assertions"],
        allocation_identity=allocation["identity"],
        round=allocation["round"],
        reserved_with_margin=allocation["with_margin_tokens"],
        known_tokens_at_allocation=owner.snapshot()["used_tokens"],
        future_locked_tokens=allocation["required_tokens"]
        - owner.snapshot()["used_tokens"],
    )
    raise ValueError("evaluation_capacity_refinement_required")


def preflight_capacity(
    owner, plans, payload, config, progress, global_reservation, *, report_sha256=None
):
    """Check complete requests and lock every remaining reevaluation before send."""
    from litagent.eval.batched import batch_prompt, request_size

    if owner.plan["contract"] == "planned-budget-v7":
        return preflight_round(
            owner, plans, payload, config, progress, global_reservation, report_sha256
        )

    try:
        remaining = remaining_deadline_seconds()
        required_seconds = (len(plans) + 1) * (config.timeout_seconds + 5)
        if remaining is not None and remaining < required_seconds:
            raise ValueError("capacity_stage_deadline_insufficient")
        amounts = [request_size(batch_prompt(p), p, config.max_tokens) for p in plans]
        allocation = owner.expand_capacity(
            len(payload["assertions"]),
            len(plans),
            request_caps={"assessment": max(amounts), "global": global_reservation},
        )
        progress["capacity"] = {
            "policy": POLICY,
            "authored_unit_counts": progress.get("authored_unit_counts", {}),
            "operational_limits": {
                k: owner.plan["envelope"]["capacity"][k]
                for k in ("max_assertions", "max_assessment_batches")
            },
            "status": "allocated",
            "initial_assertions": owner.plan["envelope"]["model_assertions"],
            "effective_assertions": allocation["envelope"]["model_assertions"],
            "actual_assertions": len(payload["assertions"]),
            "actual_batches": len(plans),
            "allocation_identity": allocation["identity"],
            "reserved_with_margin": allocation["with_margin_tokens"],
            "fixed_allowance": owner.plan["planned_tokens"],
            "known_tokens_at_allocation": owner.snapshot()["used_tokens"],
            "future_locked_tokens": allocation["required_tokens"]
            - owner.snapshot()["used_tokens"],
            "settled_reservation_savings": sum(
                e["reserved_tokens"] - e["actual_tokens"]
                for e in owner.entries.values()
            ),
        }
    except ValueError as exc:
        progress["capacity"] = capacity_diagnostic(
            progress, str(exc), batches=len(plans)
        )
        raise ValueError("evaluation_capacity_refinement_required") from exc


def preflight_round(
    owner, plans, payload, config, progress, global_reservation, report_sha256
):
    """Choose one fully funded branch and publish an executable refinement goal."""
    from litagent.capacity_rounds import choose_refinement, plan_round, request_proof
    from litagent.eval.batched import batch_prompt, build_global_payload, global_prompt
    from litagent.eval.scoped import build_evaluation_request

    requests = []
    for batch in plans:
        messages, schema = build_evaluation_request(batch_prompt(batch), batch)
        requests.append(
            request_proof("assessment", messages, config.max_tokens, schema)
        )
    gp = build_global_payload(payload, progress)
    messages, schema = build_evaluation_request(
        global_prompt(
            gp,
            native_unit_ownership=payload.get("native_unit_ownership_contract")
            == "grounded-writer-v1",
        ),
        gp,
    )
    requests.append(request_proof("global", messages, config.max_tokens, schema))
    if requests[-1]["reserved_tokens"] != global_reservation:
        raise ValueError("budget_plan_contract_breached")
    actual = len(payload["assertions"])
    current_total = sum(r["reserved_tokens"] for r in requests)
    remaining = remaining_deadline_seconds()
    if remaining is not None and remaining < (len(plans) + 1) * (
        config.timeout_seconds + 5
    ):
        raise ValueError("evaluation_capacity_stage_deadline_insufficient")
    failure = None
    try:
        allocation = plan_round(
            owner,
            assertions=actual,
            batches=len(plans),
            requests=requests,
            report_sha256=report_sha256,
        )
    except ValueError as exc:
        if str(exc) != "capacity_remaining_workflow_unaffordable":
            raise
        failure = str(exc)
        try:
            allocation = choose_refinement(
                owner,
                assertions=actual,
                batches=len(plans),
                report_sha256=report_sha256,
            )
        except ValueError as resource:
            if str(resource) not in {
                "capacity_no_affordable_refinement",
                "capacity_no_repair_available",
            }:
                raise
            progress["capacity"] = {
                **capacity_diagnostic(progress, str(resource), batches=len(plans)),
                "status": "exhausted",
                "source_report_sha256": report_sha256,
                "assessment_reserved_tokens": current_total,
            }
            raise ValueError("evaluation_capacity_exhausted") from resource
    progress["capacity"] = {
        "policy": POLICY,
        "budget_policy": "planned-budget-v7",
        "source_report_sha256": report_sha256,
        "authored_unit_counts": progress.get("authored_unit_counts", {}),
        "status": "needs_refinement" if failure else "allocated",
        "reason": failure,
        "initial_assertions": owner.plan["envelope"]["model_assertions"],
        "effective_assertions": allocation["envelope"]["model_assertions"],
        "actual_assertions": actual,
        "actual_batches": len(plans),
        "assessment_reserved_tokens": current_total,
        "allocation_identity": allocation["identity"],
        "round": allocation["round"],
        "reserved_with_margin": allocation["with_margin_tokens"],
        "fixed_allowance": owner.plan["planned_tokens"],
        "known_tokens_at_allocation": owner.snapshot()["used_tokens"],
        "future_locked_tokens": allocation["required_tokens"]
        - owner.snapshot()["used_tokens"],
        "settled_reservation_savings": sum(
            e["reserved_tokens"] - e["actual_tokens"] for e in owner.entries.values()
        ),
    }
    if failure:
        progress["capacity"].update(
            capacity_diagnostic(progress, failure, batches=len(plans)),
            target_assertions=allocation["round"]["assertions"],
            target_batches=allocation["round"]["batches"],
            scientific_verdict="unverified",
        )
        raise ValueError("evaluation_capacity_refinement_required")
