"""Lock exact current requests and bounded future candidates without rebilling."""

import copy

from litagent.capacity import resolve_capacity
from litagent.safety.planned_budget import (
    CONTEXT_LIMIT,
    ROUND_BUDGET_CONTRACT,
    PlannedBudget,
    _digest,
    validate_writer_envelope,
    wire_reservation,
)


def request_proof(stage, messages, output_tokens, response_format):
    """Bind the complete wire measured by the transport, excluding credentials."""
    return {
        "stage": stage,
        "wire_sha256": _digest([messages, output_tokens, response_format]),
        "reserved_tokens": wire_reservation(messages, output_tokens, response_format),
        "output_tokens": output_tokens,
    }


def _safe_entries(owner, entries, previous):
    """Require a fully settled inventory boundary and complete writing stages."""
    if owner.plan["contract"] != ROUND_BUDGET_CONTRACT:
        raise ValueError("capacity_round_policy_inactive")
    if not entries or any(e.get("status") != "settled" for e in entries.values()):
        raise ValueError("capacity_reallocation_unsafe")
    from litagent.eval.inventory_recovery import validate_recovery_binding

    last = list(entries.values())[-1]
    if last["stage"] != "inventory" and not validate_recovery_binding(
        owner, last, previous
    ):
        raise ValueError("capacity_inventory_boundary_required")
    for stage in ("body_initial", "body", "reviewer"):
        if (
            sum(e["stage"] == stage for e in entries.values())
            != owner.plan["stages"][stage]["count"]
        ):
            raise ValueError("capacity_writing_boundary_incomplete")


def _allocate(
    owner, entries, *, assertions, batches, requests, report_sha256, mode, previous
):
    """Rebuild a complete branch using exact current wires and future ceilings."""
    _safe_entries(owner, entries, previous)
    cap = resolve_capacity(owner.plan["envelope"]["capacity"])
    if (
        mode not in {"assess", "refine"}
        or type(assertions) is not int
        or assertions < 1
        or type(batches) is not int
        or not 1 <= batches <= 64
        or assertions > batches * cap.batch_size
        or (cap.max_assertions is not None and assertions > cap.max_assertions)
        or (
            cap.max_assessment_batches is not None
            and batches > cap.max_assessment_batches
        )
        or not isinstance(report_sha256, str)
        or len(report_sha256) != 64
        or any(c not in "0123456789abcdef" for c in report_sha256)
    ):
        raise ValueError("capacity_round_input_invalid")
    own = {
        k: [e for e in entries.values() if e["stage"] == k]
        for k in owner.plan["stages"]
    }
    round_number = len(own["inventory"]) + sum(
        "inventory_recovery" in e for e in own["repair"]
    )
    if previous and round_number <= previous["round"]["number"]:
        raise ValueError("capacity_round_identity_conflict")
    repairs = owner.plan["repair_limit"] - len(own["repair"])
    if repairs < 0 or round_number > 1 + len(own["repair"]):
        raise ValueError("capacity_round_count_invalid")
    if mode == "refine" and not repairs:
        raise ValueError("capacity_no_repair_available")
    if not isinstance(requests, list) or len(requests) != (
        batches + 1 if mode == "assess" else 0
    ):
        raise ValueError("capacity_round_requests_invalid")
    for i, request in enumerate(requests):
        stage = "global" if i == batches else "assessment"
        if (
            not isinstance(request, dict)
            or set(request)
            != {"stage", "wire_sha256", "reserved_tokens", "output_tokens"}
            or request["stage"] != stage
            or request["output_tokens"] != 16384
            or type(request["reserved_tokens"]) is not int
            or not 16384 <= request["reserved_tokens"] <= CONTEXT_LIMIT
            or not isinstance(request["wire_sha256"], str)
            or len(request["wire_sha256"]) != 64
            or any(c not in "0123456789abcdef" for c in request["wire_sha256"])
        ):
            raise ValueError("capacity_round_requests_invalid")
        if owner.plan.get("joint_round_escrow") and (
            request["reserved_tokens"] > owner.plan["stages"][stage]["request_cap"]
        ):
            raise ValueError("capacity_round_requests_invalid")
    envelope = copy.deepcopy(owner.plan["envelope"])
    envelope["model_assertions"] = max(cap.initial_assertions, assertions)
    envelope["assessment_batches"] = max(envelope["assessment_batches"], batches)
    envelope["identity"] = _digest(
        {k: v for k, v in envelope.items() if k != "identity"}
    )
    validate_writer_envelope(envelope)
    stages = copy.deepcopy(owner.plan["stages"])
    joint = owner.plan.get("joint_round_escrow", {}).get("assessment")
    for name, spec in stages.items():
        exact = [r for r in requests if r["stage"] == name]
        future = (
            repairs
            * (joint["max_requests"] if joint else envelope["assessment_batches"])
            if name == "assessment"
            else repairs if name in {"summary", "inventory", "global", "repair"} else 0
        )
        spec["count"] = len(own[name]) + len(exact) + future
        spec["request_cap"] = max(
            [spec["request_cap"], *[r["reserved_tokens"] for r in exact]]
        )
        spec["reserved_tokens"] = (
            sum(e["actual_tokens"] for e in own[name])
            + sum(r["reserved_tokens"] for r in exact)
            + (
                repairs * joint["reserved_tokens"]
                if name == "assessment" and joint
                else future * owner.plan["stages"][name]["request_cap"]
            )
        )
    required = owner.plan["spent_tokens"] + sum(
        s["reserved_tokens"] for s in stages.values()
    )
    with_margin = max(500000, ((required * 11 + 99999) // 100000) * 10000)
    request_count = owner.plan["spent_requests"] + sum(
        s["count"] for s in stages.values()
    )
    if (
        with_margin > owner.plan["planned_tokens"]
        or request_count > owner.plan["maximum_requests"]
    ):
        raise ValueError("capacity_remaining_workflow_unaffordable")
    value = {
        "envelope": envelope,
        "stages": stages,
        "required_tokens": required,
        "with_margin_tokens": with_margin,
        "planned_requests": request_count,
        "round": {
            "number": round_number,
            "report_sha256": report_sha256,
            "mode": mode,
            "status": "allocated" if mode == "assess" else "superseded_unverified",
            "assertions": assertions,
            "batches": batches,
            "remaining_repairs": repairs,
            "offsets": {k: len(own[k]) for k in ("assessment", "global")},
            "requests": requests,
        },
    }
    value["identity"] = _digest(value)
    return value


def _append(owner, arguments, allocation):
    """Persist an allocation before any caller can consume its slots."""
    event = {
        "parent": (
            owner.revisions[-1]["identity"]
            if owner.revisions
            else owner.plan["identity"]
        ),
        "arguments": copy.deepcopy(arguments),
        "entries": copy.deepcopy(owner.entries),
        "allocation": allocation,
    }
    event["identity"] = _digest(event)
    previous = owner.allocation
    owner.revisions.append(event)
    owner.allocation = allocation
    try:
        owner._persist()
    except BaseException:
        owner.revisions.pop()
        owner.allocation = previous
        owner._stop_reason = "capacity_allocation_persist_failed"
        raise
    return copy.deepcopy(allocation)


def plan_round(owner, *, assertions, batches, requests, report_sha256):
    """Freeze every actual request and both remaining complete repair paths."""
    if owner.stop_reason or owner._usage_uncertain:
        raise ValueError("capacity_reallocation_unsafe")
    arguments = dict(
        assertions=assertions,
        batches=batches,
        requests=requests,
        report_sha256=report_sha256,
        mode="assess",
    )
    allocation = _allocate(owner, owner.entries, **arguments, previous=owner.allocation)
    return _append(owner, arguments, allocation)


def choose_refinement(owner, *, assertions, batches, report_sha256):
    """Lock a candidate target; complete candidate wires still require preflight."""
    if owner.stop_reason or owner._usage_uncertain:
        raise ValueError("capacity_reallocation_unsafe")
    cap = resolve_capacity(owner.plan["envelope"]["capacity"])
    joint = owner.plan.get("joint_round_escrow", {}).get("assessment")
    minimum = 1 if joint else owner.plan["envelope"]["assessment_batches"]
    upper = min(
        batches if joint else batches - 1,
        cap.max_assessment_batches or 64,
        joint["max_requests"] if joint else 64,
    )
    for count in range(upper, minimum - 1, -1):
        target = min(
            assertions if joint else assertions - 1,
            count * cap.batch_size,
            cap.max_assertions or 1024,
        )
        if target < (1 if joint else cap.initial_assertions):
            continue
        arguments = dict(
            assertions=target,
            batches=count,
            requests=[],
            report_sha256=report_sha256,
            mode="refine",
        )
        try:
            allocation = _allocate(
                owner, owner.entries, **arguments, previous=owner.allocation
            )
        except ValueError as exc:
            if str(exc) != "capacity_remaining_workflow_unaffordable":
                raise
            continue
        return _append(owner, arguments, allocation)
    raise ValueError("capacity_no_affordable_refinement")


def _slot(owner, stage):
    """Resolve exactly one current request; future escrows are never spendable."""
    current = (owner.allocation or {}).get("round", {})
    if current.get("mode") != "assess":
        raise ValueError("capacity_round_request_not_authorized")
    requests = [r for r in current["requests"] if r["stage"] == stage]
    index = len(owner._stage_entries(stage)) - current["offsets"][stage]
    if not 0 <= index < len(requests):
        raise ValueError("capacity_round_request_not_authorized")
    return current, requests[index]


def round_request_identity(owner, stage, base_key):
    """Scope repeated content to one verified round without permitting replay."""
    if (
        owner is None
        or owner.plan["contract"] != ROUND_BUDGET_CONTRACT
        or stage not in {"assessment", "global"}
    ):
        return base_key
    if not isinstance(owner, PlannedBudget):
        raise ValueError("capacity_round_request_not_authorized")
    if not isinstance(base_key, str) or not base_key:
        raise ValueError("planned_request_scope_missing")
    allocation = owner.allocation
    revisions = owner.revisions
    if (
        not isinstance(allocation, dict)
        or not revisions
        or revisions[-1].get("allocation") != allocation
        or allocation.get("identity")
        != _digest({k: v for k, v in allocation.items() if k != "identity"})
        or revisions[-1].get("identity")
        != _digest({k: v for k, v in revisions[-1].items() if k != "identity"})
    ):
        raise ValueError("capacity_round_request_not_authorized")
    current = allocation.get("round", {})
    if (
        current.get("mode") != "assess"
        or current.get("status") != "allocated"
        or type(current.get("number")) is not int
        or current["number"] < 1
    ):
        raise ValueError("capacity_round_request_not_authorized")
    key = _digest(
        ["host-round-request-scope-v1", allocation["identity"], stage, base_key]
    )
    # Legacy ledgers remain readable, but changing the key format cannot retry
    # a current-round request or move an unsettled old request into new funds.
    if key in owner.entries:
        raise ValueError("planned_request_no_replay")
    previous = owner.entries.get(base_key)
    if previous is not None and (
        previous.get("stage") != stage
        or type(previous.get("round_number")) is not int
        or not 1 <= previous["round_number"] < current["number"]
        or previous.get("status") != "settled"
        or previous.get("sent") is not True
    ):
        raise ValueError("planned_request_no_replay")
    _slot(owner, stage)
    return key


def check_round_wire(owner, stage, messages, output_tokens, response_format):
    """Reject changed content even when it would fit the old request amount."""
    _, slot = _slot(owner, stage)
    if request_proof(stage, messages, output_tokens, response_format) != slot:
        raise ValueError("capacity_round_wire_identity_conflict")
    owner._checked_round_wire = (owner._owner(), copy.deepcopy(slot))


def reserve_round_wire(owner, tokens):
    """Require a checked exact wire before creating an assessment reservation."""
    stage, _ = owner._owner()
    current, slot = _slot(owner, stage)
    if (
        getattr(owner, "_checked_round_wire", None) != (owner._owner(), slot)
        or tokens != slot["reserved_tokens"]
    ):
        raise ValueError("capacity_round_wire_identity_conflict")
    return {"round_number": current["number"], "round_wire_sha256": slot["wire_sha256"]}


def restore_rounds(owner, snapshot):
    """Replay every safe allocation and verify receipts against exact old slots."""
    revisions = snapshot.get("capacity_revisions")
    if not isinstance(revisions, list) or len(revisions) > 3:
        raise ValueError("capacity_round_chain_invalid")
    final = snapshot.get("entries", {})
    parent, previous, prefix = owner.plan["identity"], None, {}
    for event in revisions:
        if (
            not isinstance(event, dict)
            or set(event)
            != {"parent", "arguments", "entries", "allocation", "identity"}
            or event["parent"] != parent
            or event["identity"]
            != _digest({k: v for k, v in event.items() if k != "identity"})
            or not isinstance(event["entries"], dict)
            or any(final.get(k) != v for k, v in event["entries"].items())
            or list(event["entries"].items())[: len(prefix)] != list(prefix.items())
            or list(final.items())[: len(event["entries"])]
            != list(event["entries"].items())
        ):
            raise ValueError("capacity_round_chain_invalid")
        prior = (previous or owner.plan)["stages"]
        for stage, spec in prior.items():
            entries = [e for e in event["entries"].values() if e.get("stage") == stage]
            if len(entries) > spec["count"] or any(
                type(e.get("actual_tokens")) is not int
                or not 0
                <= e["actual_tokens"]
                <= e.get("reserved_tokens", -1)
                <= spec["request_cap"]
                for e in entries
            ):
                raise ValueError("capacity_round_receipt_invalid")
        try:
            rebuilt = _allocate(
                owner, event["entries"], **event["arguments"], previous=previous
            )
        except (TypeError, KeyError) as exc:
            raise ValueError("capacity_round_chain_invalid") from exc
        if rebuilt != event["allocation"]:
            raise ValueError("capacity_round_allocation_invalid")
        parent, previous, prefix = event["identity"], rebuilt, event["entries"]
    if snapshot.get("capacity_allocation") != previous:
        raise ValueError("capacity_round_allocation_invalid")
    from litagent.eval.inventory_recovery import validate_recovery_binding

    allocations = {e["allocation"]["identity"]: e["allocation"] for e in revisions}
    for entry in final.values():
        if "inventory_recovery" in entry:
            binding = entry["inventory_recovery"]
            parent = (
                allocations.get(binding.get("parent_allocation"))
                if isinstance(binding, dict)
                else None
            )
            validate_recovery_binding(owner, entry, parent)
    bound = {}
    for event in revisions:
        r = event["allocation"]["round"]
        for stage in ("assessment", "global"):
            for i, slot in enumerate(s for s in r["requests"] if s["stage"] == stage):
                key = (stage, r["offsets"][stage] + i)
                if key in bound:
                    raise ValueError("capacity_round_slot_reused")
                bound[key] = (r["number"], slot)
    for stage in ("assessment", "global"):
        for i, entry in enumerate(e for e in final.values() if e.get("stage") == stage):
            expected = bound.get((stage, i))
            if not expected or (
                entry.get("round_number"),
                entry.get("round_wire_sha256"),
                entry.get("reserved_tokens"),
            ) != (
                expected[0],
                expected[1]["wire_sha256"],
                expected[1]["reserved_tokens"],
            ):
                raise ValueError("capacity_round_receipt_invalid")
    owner.revisions = copy.deepcopy(revisions)
    owner.allocation = copy.deepcopy(previous)
