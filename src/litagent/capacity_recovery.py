"""Retry a rejected initial capacity patch against its original authority."""

import inspect
from functools import cache
from pathlib import Path

CONTRACT = "initial-capacity-source-retry-v1"
ERROR_CODE = "writer_capacity_source_map_invalid"


@cache
def identity():
    """Identify the narrowly scoped recovery rule in new frozen plans."""
    from litagent.quality import digest
    from litagent.stage_repair import authenticate_call
    from litagent.writing_v2 import repair_capacity_snapshot

    return digest(
        [
            Path(__file__).read_text(encoding="utf-8"),
            inspect.getsource(authenticate_call),
            inspect.getsource(repair_capacity_snapshot),
        ]
    )


def enabled(envelope):
    """Select recovery for the frozen source-owner and guidance contracts."""
    return (
        (envelope or {}).get("run_limits", {}).get("initial_capacity_recovery")
        == CONTRACT
        and envelope.get("repair_mapping_contract") == "capacity-source-map-v4"
        and envelope.get("repair_guidance_wire_revision") == "host-repair-task-view-v1"
    )


def source_findings(raw, schema, targets):
    """Locate owner errors in a complete response without accepting any patches."""
    from litagent.capacity_repair import original_unit_directory
    from litagent.quality import digest
    from litagent.stage_repair import source_owner_diagnostics
    from litagent.writing_flat import prepare
    from litagent.writing_v2 import parse_response

    plan = prepare(schema)
    directory = original_unit_directory(targets)
    if plan.get("source_owner_directory") != directory:
        raise ValueError("capacity_source_directory_conflict")
    response = parse_response(raw, plan["wire_schema"])
    if response["outcome"] != "patched":
        return None
    findings = source_owner_diagnostics(response["patches"], directory)
    # Hash the failed operation independently of whitespace and free-form explanation.
    return {
        "diagnostics": findings,
        "patches_sha256": digest([response["patches"], response["deleted_sources"]]),
    }


def authenticate_rejection(session, name, body, targets, failure):
    """Bind the rejected patch to its exact original request and settled usage."""
    from litagent.quality import digest
    from litagent.stage_repair import authenticate_call
    from litagent.writing_v2 import (
        WriterError,
        check_repair_guidance_record,
        repair_request_input,
        validate_targets,
    )

    if not enabled(session.envelope) or failure.get("code") != ERROR_CODE:
        return None
    try:
        call = session.journal["calls"][name]
        if (
            call["status"] != "failed_no_replay"
            or call["failure"] != failure
            or "format_recovery" in call
        ):
            return None
        attempt = session.state["attempts"][-1]
        if (
            name != f"repair_{attempt['index']}"
            or attempt["status"] != "failed"
            or attempt["cause"] != "capacity"
            or attempt["source_body_sha256"] != digest(body)
        ):
            return None
        validate_targets(
            body,
            targets,
            allow_ranges=True,
            maximum_units=session.envelope["body_max_units"],
        )
        guide = call["guidance_wire_record"]
        check_repair_guidance_record(
            session.envelope, guide, packet=session.packet, targets=targets
        )
        args = guide["arguments"]
        if (
            args["cause"] != "capacity"
            or args["summary"]
            or guide["prior_replay_witness"] is not None
        ):
            return None
        system, payload, schema = repair_request_input(
            session.packet,
            body,
            targets,
            session.aliases,
            cause="capacity",
            previous_failure=args["previous_failure"],
            writing_plan=session.writing_plan,
            envelope=session.envelope,
            wire_contract=session.wire_contract,
            capacity_diagnostic=args["capacity_diagnostic"],
            guidance_context=guide["authority"]["guidance_context"],
        )
        system, payload = session.planned_input(system, payload)
        request = {
            "stage": "repair",
            "category": "capacity",
            "messages": session.messages(system, payload, schema),
            "response_format": session.response_format("repair", schema),
            "schema": schema,
        }
        call, _, settlement = authenticate_call(
            session, name, "capacity", initial_capacity_request=request
        )
        replay = source_findings(call["raw"], schema, targets)
        if not replay or not replay["diagnostics"]:
            return None
        return {
            "contract": CONTRACT,
            "request": name,
            "body_sha256": attempt["source_body_sha256"],
            "raw_sha256": call["raw_sha256"],
            "actual_tokens": settlement["actual_tokens"],
            **replay,
        }
    except (KeyError, IndexError, TypeError, ValueError, WriterError):
        return None


def feedback(rejection):
    """Return bounded host diagnostics using positions and authorized keys only."""
    return [
        {
            "reason": issue["reason"],
            "field": "source_unit_ids",
            "actual": issue["wire_path"][1] + 1,
            "target_ids": issue["authorized_target_ids"],
        }
        for issue in rejection["diagnostics"][:4]
    ]
