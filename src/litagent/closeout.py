"""Freeze the engineering release policy without changing historical runs."""

import inspect

from litagent.quality import digest

PROFILE = "engineering-closeout-v1"
FIELDS = ("closeout_profile", "closeout_profile_identity")
PRESET = {
    "inventory_recovery": True,
    "mixed_inventory_recovery": True,
    "classification_review": True,
    "repair_feedback": True,
    "reference_contract": "ordered-reference-set-v1",
    "transport_revision": "host-lossless-transport-v1",
    "support_validation_revision": "host-support-receipts-v2",
    "repair_mapping_contract": "capacity-source-map-v4",
    "repair_guidance_contract": "host-structured-repair-v1",
    "repair_guidance_wire_revision": "host-repair-task-view-v1",
    "semantic_conflict_policy": "host-unverified-continue-v1",
    "evaluation_response_contract": "host-bound-evaluation-v2",
    "repair_prompt_revision": "concise-grounded-repair-v1",
    "writer_source_view": "source-only-writer-v1",
    "evaluation_dimensions_contract": "separated-evaluation-dimensions-v1",
}
RECHECK_INSTRUCTION = (
    "An unchanged inventory may use this run's single inventory-recovery slot for "
    "fresh independent assessment and complete global review. A changed hash, "
    "wording or verdict alone does not prove a split or resolve a report defect. "
    "Reconsider prior advisory findings; retain every proposition and qualifier."
)


def profile_flags(profile):
    """Return one fixed server preset, never arbitrary user configuration."""
    if type(profile) is not str or profile != PROFILE:
        raise ValueError("closeout_profile_invalid")
    return {
        **PRESET,
        "round_resource_contract": "joint-round-escrow-v1",
        "closeout_profile": PROFILE,
    }


def profile_identity():
    """Bind recovery routing, request rules and their stored policy version."""
    return digest(
        [
            PROFILE,
            PRESET,
            RECHECK_INSTRUCTION,
            "two-shared-attempts;one-inventory-at-most;unchanged-recheck-allowed",
            *[
                inspect.getsource(fn)
                for fn in (
                    profile_flags,
                    profile_fields,
                    recovery_allowed,
                    recheck_prompt,
                    recovery_summary,
                )
            ],
        ]
    )


def profile_fields(envelope):
    """Validate the optional policy only after authenticating the frozen owner."""
    if envelope is None:
        return {}
    if type(envelope) is not dict:
        raise ValueError("closeout_profile_invalid")
    if not set(FIELDS).intersection(envelope):
        return {}
    from litagent.eval.dimensions import selected_dimensions
    from litagent.writer_source_view import selected_source_view

    if (
        envelope.get(FIELDS[0]) != PROFILE
        or envelope.get(FIELDS[1]) != profile_identity()
        or envelope.get("inventory_recovery_contract")
        != "bounded-inventory-recovery-v2"
        or envelope.get("semantic_conflict_policy")
        != PRESET["semantic_conflict_policy"]
        or not selected_dimensions(envelope)
        or not selected_source_view(envelope)
    ):
        raise ValueError("closeout_profile_invalid")
    return {key: envelope[key] for key in FIELDS}


def recovery_allowed(envelope, state):
    """Count sent attempts across failures, cancellation, report changes and restart."""
    if not profile_fields(envelope):
        return True
    if type(state) is not dict or type(state.get("attempts")) is not list:
        raise ValueError("closeout_repair_state_invalid")
    if any(type(row) is not dict for row in state["attempts"]):
        raise ValueError("closeout_repair_state_invalid")
    return not any(
        row.get("purpose")
        in ("bounded-inventory-recovery-v1", "bounded-inventory-recovery-v2")
        for row in state["attempts"]
    )


def recheck_prompt(prompt, envelope):
    """Replace only the historical no-progress sentence on the explicit new path."""
    if not profile_fields(envelope):
        return prompt
    return prompt.replace(
        "Repeating or\ncycling inventories is no progress.", RECHECK_INSTRUCTION
    ).replace(
        "Repeating an\nunchanged inventory or cycling to an earlier inventory is no progress.",
        RECHECK_INSTRUCTION,
    )


def recovery_summary(feedback, evaluation):
    """Record content changes and review outcomes without claiming semantic splits."""
    from litagent.eval.inventory_recovery import inventory_identity

    details = evaluation.get("faithfulness", {}).get("details", {})
    claims = details.get("assertions", [])
    complete = (
        details.get("assessment_stage") == "host_validated"
        and details.get("coverage", {}).get("complete") is True
        and details.get("stages", {}).get("global_check") == "completed"
    )
    inventory = details.get("stages", {}).get("inventory_assertions", [])
    required = {"source_fragment_id", "kind", "claim_text", "evidence_ids"}
    current = (
        inventory_identity(inventory)
        if inventory
        and all(isinstance(c, dict) and required <= c.keys() for c in inventory)
        else None
    )
    valid = (
        complete
        and bool(claims)
        and details.get("inventory_check", {}).get("status") == "passed"
        and all(
            all(
                c.get(key) is True
                for key in ("atomic", "source_faithful", "classification_valid")
            )
            for c in claims
        )
        and not any(
            issue.get("owner") == "inventory"
            for issue in details.get("global_issues", [])
        )
    )
    fragments = list(
        dict.fromkeys(
            fid
            for finding in feedback.get("findings", [])
            for fid in finding.get("source_fragment_ids", [])
        )
    )
    return {
        "report_sha256": feedback["source_report_sha256"],
        "previous_inventory_sha256": feedback["previous_inventory_sha256"],
        "current_inventory_sha256": current,
        "content_changed": (
            current != feedback["previous_inventory_sha256"] if current else None
        ),
        "operation": "independent_recheck",
        "status": "resolved" if valid else "unresolved",
        "complete_review": complete,
        "fragments": [
            {
                "source_fragment_id": fid,
                "current_checks": [
                    {
                        key: c.get(key)
                        for key in (
                            "claim_id",
                            "atomic",
                            "source_faithful",
                            "classification_valid",
                        )
                    }
                    for c in claims
                    if c.get("source_fragment_id") == fid
                ],
            }
            for fid in fragments
        ],
    }


def result_summary(report):
    """Expose distinct generation, valid evaluation and repair states."""
    meta = report.get("metadata", {})
    metric = report.get("evaluation", {}).get("faithfulness", {})
    details = metric.get("details", {})
    attempts = meta.get("targeted_repair_state", {}).get("attempts", [])
    inventory = sum(
        row.get("purpose")
        in ("bounded-inventory-recovery-v1", "bounded-inventory-recovery-v2")
        for row in attempts
    )
    valid = (
        metric.get("skipped") is False
        and details.get("overall_valid") is not False
        and details.get("coverage", {}).get("complete") is True
        and details.get("inventory_check", {}).get("status") == "passed"
        and details.get("stages", {}).get("global_check") == "completed"
    )
    return {
        "closeout_profile": PROFILE,
        "generated": bool(meta.get("writer_document")),
        "reviewer_completed": bool(meta.get("reviewer_diagnostics")),
        "evaluation": "completed" if valid else "not_complete",
        "evaluation_valid": valid,
        "inventory_recoveries": inventory,
        "prose_repairs": len(attempts) - inventory,
        "repair_limit": 2,
        "inventory_recovery_limit": 1,
        "inventory_reviews": [
            row["inventory_review"] for row in attempts if "inventory_review" in row
        ],
    }
