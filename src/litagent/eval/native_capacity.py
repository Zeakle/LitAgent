"""Bind native inventory capacity and its independent recovery reservations."""

from __future__ import annotations

import hashlib
from collections import Counter
from functools import lru_cache

from litagent.quality import digest

CONTRACT = "native-capacity-recovery-v1"
INVENTORY_CONTEXT_CONTRACT = "ragas-inventory-local-context-v1"
REQUIRED = "evaluation_native_capacity_refinement_required"


class NativeCapacityRequired(ValueError):
    """Stop before NLI while retaining a complete, settled extraction."""

    def __init__(self):
        super().__init__(REQUIRED)


def selected(envelope):
    """Select only the new, explicitly frozen inline Evaluation contract."""
    return (envelope or {}).get("semantic_engine", {}).get(
        "capacity_recovery_contract"
    ) == CONTRACT


@lru_cache(maxsize=32)
def inventory_frame(batch_size, width, output, *, inventory_context=False):
    """Measure the largest host-owned job frame using the production builder."""
    from litagent.eval.ragas_adapter import batch_request
    from litagent.safety.budget import strict_request_reservation

    jobs = [
        {
            "sample": {
                "id": "u" + str(index).zfill(width),
                "question": "",
                "response": "",
            }
        }
        for index in range(batch_size)
    ]
    if inventory_context:
        for index, job in enumerate(jobs):
            job["sample"]["report_context"] = {
                "id": "p" + str(index).zfill(width),
                "text": "",
                "start": 0,
                "end": 0,
            }
    messages, schema, _ = batch_request(
        "inventory", jobs, inventory_context=inventory_context
    )
    # The empty frame has one digit per offset. Validated offsets can grow to
    # `width` digits; their decimal representation is unchanged by JSON escaping.
    offsets = 2 * batch_size * (width - 1) if inventory_context else 0
    return strict_request_reservation(messages, output, schema) + offsets


def _inventory_context(envelope):
    """Select the frozen extraction contract without upgrading historical plans."""
    return envelope.get("semantic_engine", {}).get("inventory_context_contract") == (
        INVENTORY_CONTEXT_CONTRACT
    )


def inventory_input_bound(envelope, question):
    """Bound extraction input with its real builder and complete authored text."""
    from litagent.output_capacity import envelope_output_tokens
    from litagent.safety.planned_budget import nested_json_size

    if not isinstance(question, str) or not question.strip():
        raise ValueError("native_capacity_question_invalid")
    maximum = envelope["body_json_bytes"] + envelope["summary_json_bytes"]
    width = len(str(maximum))
    batch_size = 2 * envelope["semantic_engine"]["batch_size"]
    question_size = nested_json_size(question)
    frame = inventory_frame(
        batch_size,
        width,
        envelope_output_tokens(envelope)["inventory"],
        inventory_context=_inventory_context(envelope),
    )
    # Empty answer strings are already in the measured frame. Adding the entire
    # nested JSON text allowance is conservative, including character escaping.
    return {
        "contract": CONTRACT,
        "question_sha256": digest(question),
        "question_json_bytes": question_size,
        "authored_text_json_bytes": maximum,
        "sample_id_width": width,
        "batch_size": batch_size,
        "frame_reservation": frame,
        **(
            {"report_context_json_bytes": maximum}
            if _inventory_context(envelope)
            else {}
        ),
        "request_cap": frame
        + maximum
        + batch_size * question_size
        + (maximum if _inventory_context(envelope) else 0),
    }


def validate_bound(envelope, *, live=True):
    """Authenticate frozen dimensions; archived accounting need not import RAGAS."""
    from litagent.output_capacity import envelope_output_tokens

    bound = envelope.get("native_inventory_bound")
    if not selected(envelope):
        if bound is not None:
            raise ValueError("native_capacity_contract_invalid")
        return
    fields = {
        "contract",
        "question_sha256",
        "question_json_bytes",
        "authored_text_json_bytes",
        "sample_id_width",
        "batch_size",
        "frame_reservation",
        "request_cap",
    }
    local_context = _inventory_context(envelope)
    if local_context:
        fields.add("report_context_json_bytes")
    if (
        not isinstance(bound, dict)
        or set(bound) != fields
        or bound["contract"] != CONTRACT
        or not isinstance(bound["question_sha256"], str)
        or len(bound["question_sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in bound["question_sha256"])
        or any(
            type(bound[k]) is not int or bound[k] <= 0
            for k in fields - {"contract", "question_sha256"}
        )
        or bound["authored_text_json_bytes"]
        != envelope["body_json_bytes"] + envelope["summary_json_bytes"]
        or bound["sample_id_width"] != len(str(bound["authored_text_json_bytes"]))
        or bound["batch_size"] != 2 * envelope["semantic_engine"]["batch_size"]
        or (
            local_context
            and bound["report_context_json_bytes"] != bound["authored_text_json_bytes"]
        )
        or bound["request_cap"]
        != bound["frame_reservation"]
        + bound["authored_text_json_bytes"]
        + bound["batch_size"] * bound["question_json_bytes"]
        + bound.get("report_context_json_bytes", 0)
        or envelope["request_caps"]["inventory"] != bound["request_cap"]
    ):
        raise ValueError("native_capacity_bound_invalid")
    if live and bound["frame_reservation"] != inventory_frame(
        bound["batch_size"],
        bound["sample_id_width"],
        envelope_output_tokens(envelope)["inventory"],
        inventory_context=local_context,
    ):
        raise ValueError("native_capacity_builder_drift")


def check_inventory_inputs(samples, bound):
    """Verify the executable input bound without truncating a single sample."""
    from litagent.safety.planned_budget import nested_json_size

    ids = [sample["id"] for sample in samples]
    if len(ids) != len(set(ids)) or any(
        not isinstance(key, str)
        or not key.startswith("u")
        or not key[1:].isdigit()
        or len(key) > bound["sample_id_width"] + 1
        for key in ids
    ):
        raise ValueError("native_capacity_sample_identity_invalid")
    if any(
        digest(sample["question"]) != bound["question_sha256"] for sample in samples
    ):
        raise ValueError("native_capacity_question_identity_invalid")
    actual = sum(nested_json_size(sample["response"]) for sample in samples)
    if actual > bound["authored_text_json_bytes"]:
        raise ValueError("native_capacity_authored_text_exceeded")
    if "report_context_json_bytes" in bound:
        from litagent.eval.ragas_adapter import report_context

        pool = {}
        for sample in samples:
            background = report_context(sample)
            key, text = background["id"], background["text"]
            if len(key) > bound["sample_id_width"] + 1:
                raise ValueError("native_capacity_context_identity_invalid")
            if key in pool and pool[key] != text:
                raise ValueError("ragas_inventory_context_conflict")
            pool[key] = text
        if (
            sum(nested_json_size(text) for text in pool.values())
            > bound["report_context_json_bytes"]
        ):
            raise ValueError("native_capacity_report_context_exceeded")
    return actual


def capacity_record(
    survey, assertions, *, envelope, plan_identity, parent, amount, limit
):
    """Describe a measured resource failure without issuing scientific verdicts."""
    value = {
        "contract": CONTRACT,
        "policy": "adaptive-capacity-v1",
        "status": "needs_refinement",
        "reason": REQUIRED,
        "scientific_verdict": "unverified",
        "source_report_sha256": hashlib.sha256(survey.encode()).hexdigest(),
        "inventory_sha256": digest(assertions),
        "budget_plan_identity": plan_identity,
        "budget_partition": parent,
        "actual_assertions": len(assertions),
        "global_reserved_tokens": amount,
        "field": "global_request_reservation",
        "actual": amount,
        "maximum": limit,
        "assertions_by_fragment": dict(
            Counter(c["source_fragment_id"] for c in assertions)
        ),
        "assertions_by_section": dict(Counter(c["section"] for c in assertions)),
    }
    return {**value, "identity": digest(value)}


def candidate_receipt(metadata, sha):
    """Read the persisted parent of one completed capacity candidate."""
    from litagent.quality import targeted_repair_state

    raw = metadata["targeted_repair_state"]
    state = targeted_repair_state(raw, raw["run_id"])
    if metadata["writing_bindings"]["report_sha256"] != sha:
        raise ValueError("native_capacity_candidate_identity_invalid")
    body_sha = digest(metadata["writer_document"]["body"])
    matches = [
        attempt
        for attempt in state["attempts"]
        if attempt.get("native_capacity")
        and attempt.get("candidate_sha256") == body_sha
    ]
    if len(matches) != 1:
        raise ValueError("native_capacity_candidate_identity_invalid")
    attempt = matches[0]
    receipt = attempt["native_capacity"]
    if (
        attempt.get("category") != "capacity"
        or attempt["status"] != "completed"
        or set(receipt)
        != {
            "source_report_sha256",
            "budget_partition",
            "capacity_identity",
        }
        or receipt["budget_partition"] not in (None, "evaluation_revision", "inventory")
        or receipt["source_report_sha256"] == sha
        or attempt["source_body_sha256"] == body_sha
        or any(
            not isinstance(receipt[key], str)
            or len(receipt[key]) != 64
            or any(c not in "0123456789abcdef" for c in receipt[key])
            for key in (
                "source_report_sha256",
                "capacity_identity",
            )
        )
    ):
        raise ValueError("native_capacity_candidate_identity_invalid")
    return {**receipt, "candidate_report_sha256": sha}


def retry_feedback(failure, prior, directory):
    """Keep an oversized candidate's resource feedback outside science guidance."""
    details = prior["evaluation"]["faithfulness"]["details"]
    capacity = details.get("stages", {}).get("capacity", {})
    sha = prior["report_sha256"]
    if (
        failure.get("family") != "capacity"
        or failure.get("code") != "writer_candidate_failed"
        or failure.get("diagnostics") != []
        or failure.get("candidate_sha256") != sha
        or hashlib.sha256(prior["survey"].encode()).hexdigest() != sha
        or prior["bindings"].get("report_sha256") != sha
        or prior.get("committed") is not False
        or details.get("execution_error") != REQUIRED
        or details.get("component_errors")
        or capacity.get("contract") != CONTRACT
        or capacity.get("source_report_sha256") != sha
        or capacity.get("identity")
        != digest({k: v for k, v in capacity.items() if k != "identity"})
        or capacity.get("inventory_sha256") != digest(details.get("assertions"))
        or capacity.get("actual", 0) <= capacity.get("maximum", 0)
        or directory.get("identity")
        != digest({k: v for k, v in directory.items() if k != "identity"})
        or directory.get("prior_report_sha256") != sha
        or directory.get("diagnostics_sha256") != digest([])
    ):
        raise ValueError("native_capacity_retry_identity_invalid")
    return {
        "policy": CONTRACT,
        "previous_failure": failure,
        "capacity": capacity,
        "feedback_directory_sha256": directory["identity"],
        "explanation": (
            f"Previous candidate {sha} remained too large: "
            f"{capacity['actual_assertions']} extracted assertions; complete global "
            f"request reservation {capacity['actual']} exceeds {capacity['maximum']}. "
            "This is a resource observation, not a scientific verdict. Retry from "
            "the authorized original; preserve core questions and necessary facts."
        ),
    }


def authorize(report, context, owner):
    """Authenticate capacity recovery against source bindings and settled receipts."""
    from litagent import review_policy
    from litagent.capacity_repair import capacity_targets
    from litagent.eval.engine_policy import (
        anchored_native_review,
        native_global_input_contract,
        native_unit_response_contract,
        phase_batches,
    )
    from litagent.eval.ragas_native import (
        canonical_assertions,
        global_review_request,
        organization_options,
        source_samples,
    )
    from litagent.quality import remaining_repair_allowances, targeted_repair_state
    from litagent.safety.budget import strict_request_reservation
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    def require(condition, reason="native_capacity_receipt_invalid"):
        if not condition:
            raise ValueError(reason)

    env = owner.plan["envelope"]
    require(
        selected(env) and review_policy.enabled(env), "native_capacity_contract_invalid"
    )
    snapshot = validate_accounting_snapshot(owner.snapshot(), expected_plan=owner.plan)
    require(not snapshot["usage_uncertain"] and not snapshot["stop_reason"])
    require(
        all(
            e["status"] == "settled" and e["sent"] for e in snapshot["entries"].values()
        )
    )
    survey, meta = report["survey"], report["metadata"]
    details = report["evaluation"]["faithfulness"]["details"]
    stages = details["stages"]
    record = stages["capacity"]
    require(
        details.get("execution_error") == REQUIRED
        and not details.get("component_errors")
    )
    require(
        stages.get("global_check") in (None, "not_started")
        and not stages.get("global_request_id")
    )
    require(
        details["source_report_sha256"] == hashlib.sha256(survey.encode()).hexdigest()
    )
    require(context.get("writing_bindings") == meta["writing_bindings"])
    require(context.get("evidence_snapshot_sha256") == meta["evidence_snapshot_sha256"])
    require(context.get("fact_cards") == meta["fact_cards"])
    _, units, _, samples = source_samples(
        survey, context, inventory_context=_inventory_context(env)
    )
    partial = details["partial_samples"]
    require(
        len(partial) == len(samples)
        and all(s["id"] == p["id"] for s, p in zip(samples, partial))
    )
    for sample, saved in zip(samples, partial):
        require(
            isinstance(saved["statements"], list)
            and all(isinstance(s, str) and s.strip() for s in saved["statements"])
        )
        sample["statements"] = saved["statements"]
        sample["verdicts"] = saved["verdicts"] or []
        if not sample["contexts"]:
            sample["assessment_origin"] = "host_uncited"
        else:
            require(not sample["verdicts"])
    check_inventory_inputs(samples, env["native_inventory_bound"])
    claims, _ = canonical_assertions(
        samples,
        **organization_options(env, survey, context),
    )
    require(digest(claims) == digest(details["assertions"]))
    parent = record["budget_partition"]
    require(parent in (None, "evaluation_revision", "inventory"))
    state = targeted_repair_state(
        meta["targeted_repair_state"], meta["targeted_repair_state"]["run_id"]
    )
    balances = remaining_repair_allowances(state)
    require(balances["capacity"] > 0, "native_capacity_attempts_exhausted")
    rows = stages["requests"]
    ids, keys = [], []
    inventory_stages = set()
    for row in rows:
        require(row["phase"] == "inventory" and row["status"] == "completed")
        entry = snapshot["entries"][row["request_id"]]
        usage = row["usage"]
        require(
            entry.get("wire_sha256")
            and entry["usage_sha256"]
            == digest({k: usage[k] for k in ("prompt_tokens", "completion_tokens")})
        )
        require(
            usage.get("usage_available") is True
            and entry["actual_tokens"]
            == usage["prompt_tokens"] + usage["completion_tokens"]
        )
        require(row["request_id"].startswith(stages["evaluation_identity"] + ":"))
        inventory_stages.add(entry["stage"])
        ids.extend(row["sample_ids"])
        keys.append(row["request_id"])
    require(
        ids == [s["id"] for s in samples] and len(keys) == len(set(keys)) and bool(keys)
    )
    require(
        set(keys)
        == {
            key
            for key in snapshot["entries"]
            if key.startswith(stages["evaluation_identity"] + ":")
        }
    )
    original_inventory = f"{parent}/inventory" if parent else "inventory"
    if inventory_stages == {"capacity/inventory"}:
        require(
            candidate_receipt(meta, details["source_report_sha256"])["budget_partition"]
            == parent
        )
    else:
        require(inventory_stages == {original_inventory})
    # Rebuild the exact rejected global wire from every preserved assertion.
    from types import SimpleNamespace

    config = SimpleNamespace(
        context_tokens=owner.plan["context_reservation_limit"],
        strict_context_margin=0,
        max_tokens=owner.plan["stages"]["global"]["output_tokens"],
    )
    _, messages, fmt = global_review_request(
        survey,
        context,
        config,
        units,
        claims,
        force_compact=True,
        anchored=anchored_native_review(env),
        global_scope=env["semantic_engine"].get("global_scope"),
        input_contract=native_global_input_contract(env),
        unit_response_contract=native_unit_response_contract(env),
    )
    amount = strict_request_reservation(messages, config.max_tokens, fmt)
    limit = min(owner.plan["context_reservation_limit"], env["request_caps"]["global"])
    expected = capacity_record(
        survey,
        claims,
        envelope=env,
        plan_identity=owner.plan["identity"],
        parent=parent,
        amount=amount,
        limit=limit,
    )
    require(record == expected and amount > limit)
    counts = {
        "capacity/repair": 1,
        "capacity/summary": 1,
        "capacity/inventory": phase_batches(env),
    }
    for phase, count in (("assessment", phase_batches(env)), ("global", 1)):
        counts[f"{parent}/{phase}" if parent else phase] = count
    # Format slots stay independently locked by the original full plan.
    reservations = {}
    for stage, count in counts.items():
        spec = owner.plan["stages"][stage]
        entries = [e for e in snapshot["entries"].values() if e["stage"] == stage]
        maximum = count * spec["request_cap"]
        require(
            stage not in snapshot["closed_stages"]
            and len(entries) + count <= spec["count"]
            and sum(e["actual_tokens"] for e in entries) + maximum
            <= spec["reserved_tokens"],
            "native_capacity_followup_budget_insufficient",
        )
        reservations[stage] = {"requests": count, "tokens": maximum}
    targets = capacity_targets(
        meta["writer_document"]["body"], record, maximum=env["body_max_units"]
    )
    return {
        "contract": "ragas-recovery-v1",
        "action": "capacity",
        "reason_code": REQUIRED,
        "source_report_sha256": details["source_report_sha256"],
        "evaluation_sha256": digest(report["evaluation"]),
        "targets": list(targets),
        "remaining_attempts": state["limit"] - len(state["attempts"]),
        "remaining_by_category": balances,
        "budget_partition": parent,
        "capacity_identity": record["identity"],
        "budget": {"status": "admitted", "reservations": reservations},
    }
