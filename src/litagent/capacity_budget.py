"""Reallocate settled savings while preserving a complete remaining workflow."""

import copy

from litagent.capacity import resolve_capacity
from litagent.safety.planned_budget import (
    CAPACITY_BUDGET_CONTRACT,
    CONTEXT_LIMIT,
    _digest,
    validate_writer_envelope,
)


def _allocation(owner, entries, assertions, batches, request_caps, previous):
    """Compute a bounded complete path from known receipts, never average usage."""
    if owner.plan["contract"] != CAPACITY_BUDGET_CONTRACT:
        raise ValueError("capacity_policy_inactive")
    cap = resolve_capacity(owner.plan["envelope"]["capacity"])
    if (
        type(assertions) is not int
        or assertions < 1
        or type(batches) is not int
        or batches < 1
        or assertions > batches * cap.batch_size
        or (cap.max_assertions is not None and assertions > cap.max_assertions)
        or (
            cap.max_assessment_batches is not None
            and batches > cap.max_assessment_batches
        )
    ):
        raise ValueError("capacity_operational_limit")
    if any(e.get("status") != "settled" for e in entries.values()):
        raise ValueError("capacity_reallocation_unsafe")
    if not entries or list(entries.values())[-1]["stage"] != "inventory":
        raise ValueError("capacity_inventory_boundary_required")
    stages = copy.deepcopy(owner.plan["stages"])
    old_envelope = (previous or owner.plan)["envelope"]
    envelope = copy.deepcopy(old_envelope)
    envelope["model_assertions"] = max(old_envelope["model_assertions"], assertions)
    envelope["assessment_batches"] = max(old_envelope["assessment_batches"], batches)
    request_caps = request_caps or {}
    if set(request_caps) - {"assessment", "global"}:
        raise ValueError("capacity_request_cap_invalid")
    for stage, amount in request_caps.items():
        if type(amount) is not int or not 0 < amount <= CONTEXT_LIMIT:
            raise ValueError("capacity_context_exceeded")
        envelope["request_caps"][stage] = max(envelope["request_caps"][stage], amount)
    envelope["identity"] = _digest(
        {k: v for k, v in envelope.items() if k != "identity"}
    )
    validate_writer_envelope(envelope)
    inventories = sum(e["stage"] == "inventory" for e in entries.values())
    remaining_rounds = stages["inventory"]["count"] - inventories + 1
    if remaining_rounds < 1:
        raise ValueError("capacity_inventory_round_exceeded")
    assessed = sum(e["stage"] == "assessment" for e in entries.values())
    stages["assessment"]["count"] = (
        assessed + remaining_rounds * envelope["assessment_batches"]
    )
    for name, spec in stages.items():
        own = [e for e in entries.values() if e["stage"] == name]
        if len(own) > spec["count"]:
            raise ValueError("capacity_stage_count_invalid")
        if name in envelope["request_caps"]:
            spec["request_cap"] = envelope["request_caps"][name]
        spec["reserved_tokens"] = (
            sum(e["actual_tokens"] for e in own)
            + (spec["count"] - len(own)) * spec["request_cap"]
        )
    required = owner.plan["spent_tokens"] + sum(
        s["reserved_tokens"] for s in stages.values()
    )
    with_margin = max(500000, ((required * 11 + 99999) // 100000) * 10000)
    requests = owner.plan["spent_requests"] + sum(s["count"] for s in stages.values())
    if (
        with_margin > owner.plan["planned_tokens"]
        or requests > owner.plan["maximum_requests"]
    ):
        raise ValueError("capacity_remaining_workflow_unaffordable")
    value = dict(
        envelope=envelope,
        stages=stages,
        required_tokens=required,
        with_margin_tokens=with_margin,
        planned_requests=requests,
    )
    value["identity"] = _digest(value)
    return value


def expand_capacity(
    owner, assertions, batches, *, request_caps=None, reason="inventory"
):
    """Append a proof before exposing additional request slots to transports."""
    if owner.stop_reason or owner._usage_uncertain:
        raise ValueError("capacity_reallocation_unsafe")
    proposed = _allocation(
        owner, owner.entries, assertions, batches, request_caps, owner.allocation
    )
    if proposed == owner.allocation:
        return copy.deepcopy(proposed)
    event = dict(
        parent=(
            owner.revisions[-1]["identity"]
            if owner.revisions
            else owner.plan["identity"]
        ),
        assertions=assertions,
        batches=batches,
        request_caps=request_caps or {},
        reason=reason,
        entries=copy.deepcopy(owner.entries),
        allocation=proposed,
    )
    event["identity"] = _digest(event)
    owner.revisions.append(event)
    owner.allocation = proposed
    owner._persist()
    return copy.deepcopy(proposed)


def restore_capacity(owner, snapshot):
    """Validate every revision and its settled prefix against final receipts."""
    revisions = snapshot.get("capacity_revisions")
    if not isinstance(revisions, list) or len(revisions) > 64:
        raise ValueError("capacity_revision_chain_invalid")
    parent, previous, old_entries = owner.plan["identity"], None, {}
    final = snapshot.get("entries", {})
    for event in revisions:
        if (
            not isinstance(event, dict)
            or set(event)
            != {
                "parent",
                "assertions",
                "batches",
                "request_caps",
                "reason",
                "entries",
                "allocation",
                "identity",
            }
            or event["parent"] != parent
            or event["identity"]
            != _digest({k: v for k, v in event.items() if k != "identity"})
            or not isinstance(event["entries"], dict)
            or any(final.get(k) != v for k, v in event["entries"].items())
            or any(event["entries"].get(k) != v for k, v in old_entries.items())
        ):
            raise ValueError("capacity_revision_chain_invalid")
        # A proof cannot invent larger historical slots to justify its own growth.
        prior_stages = (previous or owner.plan)["stages"]
        for name, spec in prior_stages.items():
            own = [v for v in event["entries"].values() if v.get("stage") == name]
            if len(own) > spec["count"] or any(
                type(v.get("actual_tokens")) is not int
                or not 0
                <= v["actual_tokens"]
                <= v.get("reserved_tokens", -1)
                <= spec["request_cap"]
                for v in own
            ):
                raise ValueError("capacity_revision_receipt_invalid")
        proposed = _allocation(
            owner,
            event["entries"],
            event["assertions"],
            event["batches"],
            event["request_caps"],
            previous,
        )
        if proposed != event["allocation"]:
            raise ValueError("capacity_revision_allocation_invalid")
        parent, previous, old_entries = event["identity"], proposed, event["entries"]
    if snapshot.get("capacity_allocation") != previous:
        raise ValueError("capacity_revision_allocation_invalid")
    owner.revisions, owner.allocation = copy.deepcopy(revisions), copy.deepcopy(
        previous
    )
