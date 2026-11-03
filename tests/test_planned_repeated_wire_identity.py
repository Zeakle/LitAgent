"""Bind repeated paid wires by immutable serial occurrences, never usage values."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_rounds import plan_round, request_proof
from litagent.eval.inventory_recovery import recovery_scope
from litagent.eval.scoped import _request_identity
from litagent.quality import digest
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.writing_v2 import writer_budget_plan
from scripts import run_frozen_writing as campaign


def repeated_history(case, *, identical_usage=False, rounds=3, unknown=False):
    """Produce real escrow/round snapshots and portable sequential archive data."""
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        classification_review=True,
    )
    owner = PlannedBudget(plan)
    manifest = {
        "config": {
            "effective": {
                "eval": {"max_tokens": 16384, "thinking_mode": "disabled"},
                "llm": {"model_routing": {"evaluation": "lite"}},
            }
        },
        "budget_plans": {"offline": plan},
    }
    archive, records, starts = {"events": []}, [], []
    report_sha = "a" * 64
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "portable", "schema": {"type": "object"}},
    }

    def emit(name, data):
        archive["events"].append(
            {
                "sequence": len(archive["events"]),
                "event": name,
                "data": copy.deepcopy(data),
            }
        )

    # These are synthetic accounting records, not scientific gold or HTTP calls.
    for stage, count in (
        ("body_initial", 1),
        ("body", 4),
        ("summary", 1),
        ("reviewer", 1),
    ):
        for index in range(count):
            key = f"writer:{stage}:{index}"
            with request_scope(stage, key):
                owner.reserve(1000)
                owner.begin_request()
                owner.record({"prompt_tokens": 10, "completion_tokens": 2})
            start = {
                "operation_id": key,
                "messages": [{"role": "user", "content": key}],
            }
            starts.append(start)
            emit("llm.start", start)
            emit("llm.complete", {"operation_id": key})

    def messages(phase, index, round_index, feedback=None):
        payload = (
            {
                "host_methodology_validation": {"source_sha256": report_sha},
                "inventory": round_index,
            }
            if phase == "inventory"
            else {"batch": index} if phase == "assessment" else {"global": round_index}
        )
        if feedback:
            payload["inventory_recovery"] = feedback
        return [
            {"role": "system", "content": phase},
            {"role": "user", "content": json.dumps(payload)},
        ]

    def send(phase, wire, progress, round_index, index, feedback=None):
        stage = "repair" if feedback is not None else phase
        key = f"round:{round_index}:{phase}:{index}"
        kwargs = {
            "response_format": rf,
            "max_tokens": 16384,
            "temperature": 0,
            "model": "lite",
            "extra_body": {"thinking": {"type": "disabled"}},
        }
        with (
            request_scope(stage, key),
            recovery_scope(feedback) if feedback else nullcontext(),
        ):
            amount = owner.check_wire(wire, 16384, rf)
            owner.reserve(amount)
            owner.bind_wire([wire, kwargs, "chat"])
            owner.begin_request()
            start = {
                "operation_id": key,
                "messages": wire,
                "response_format": rf,
                "provider_api": "chat",
                "output_limit_parameter": "max_tokens",
                "max_tokens": 16384,
                "temperature": 0,
                "model": "lite",
                "reasoning_effort": None,
            }
            starts.append(start)
            emit("llm.start", start)
            is_unknown = unknown and round_index == rounds - 1 and phase == "global"
            usage = {
                "prompt_tokens": 100,
                "completion_tokens": 7 if identical_usage else 7 + round_index,
            }
            if is_unknown:
                owner.charge_unknown_usage(amount)
                emit("llm.failed", {"operation_id": key, "usage_available": False})
            else:
                owner.record(usage)
                emit(
                    "llm.complete",
                    {
                        "operation_id": key,
                        **usage,
                        "total_tokens": sum(usage.values()),
                        "usage_available": True,
                    },
                )
        progress["requests"].append(
            {
                "phase": phase,
                "input_sha256": _request_identity(wire, rf),
                "status": (
                    "interrupted_no_replay" if is_unknown else "response_received"
                ),
                "reserved_tokens": amount,
                "usage": (
                    {"usage_available": False}
                    if is_unknown
                    else {**usage, "usage_available": True}
                ),
            }
        )
        return key

    for round_index, batches in enumerate((10, 11, 10)[:rounds]):
        identity = digest(["portable-non-gold", round_index])
        phase = "initial" if round_index == 0 else "post_rewrite"
        progress = {
            "requests": [],
            "writer_envelope": copy.deepcopy(owner.effective_envelope),
        }
        feedback = None
        if round_index:
            feedback = {
                "contract": plan["envelope"]["inventory_recovery_contract"],
                "source_report_sha256": report_sha,
                "previous_inventory_sha256": str(round_index) * 64,
                "bindings_sha256": "b" * 64,
                "attempt_index": round_index - 1,
            }
            progress["inventory_recovery"] = feedback
        emit(
            "evaluation.batch",
            {"identity": identity, "phase": phase, "budget": owner.snapshot()},
        )
        inv_wire = messages("inventory", 0, round_index, feedback)
        send("inventory", inv_wire, progress, round_index, 0, feedback)
        wires = [
            ("assessment", messages("assessment", index, round_index))
            for index in range(batches)
        ] + [("global", messages("global", 0, round_index))]
        plan_round(
            owner,
            assertions=batches * 16,
            batches=batches,
            requests=[request_proof(stage, wire, 16384, rf) for stage, wire in wires],
            report_sha256=report_sha,
        )
        for index, (stage, wire) in enumerate(wires):
            send(stage, wire, progress, round_index, index)
        checkpoint = {
            "phase": phase,
            "report_sha256": report_sha,
            "budget": owner.snapshot(),
            "progress": progress,
        }
        record = {"identity": identity, "checkpoint": checkpoint}
        if feedback:
            record["canonical_recovery_feedback"] = feedback
            record["repair_state"] = {
                "attempts": [
                    {
                        "index": round_index - 1,
                        "stage": "evaluation",
                        "input_sha256": progress["requests"][0]["input_sha256"],
                        "purpose": feedback["contract"],
                        "source_report_sha256": report_sha,
                    }
                ]
            }
        records.append(record)
        emit(
            "evaluation.batch",
            {"identity": identity, "phase": phase, "budget": owner.snapshot()},
        )
    return SimpleNamespace(
        archive=archive,
        starts=starts,
        records=records,
        ledger=owner.snapshot(),
        manifest=manifest,
    )


def bindings(value):
    """Call the real occurrence consumer; no validator or result is mocked."""
    return campaign._planned_evaluation_occurrence_bindings(
        value.archive, value.starts, value.records, value.ledger, value.manifest
    )


def authenticate(value, expected):
    """Run the unchanged wire/options/round guards after occurrence matching."""
    start = next(
        row for row in value.starts if row["operation_id"] == expected["operation_id"]
    )
    record = next(
        row for row in value.records if row["identity"] == expected["progress_identity"]
    )
    return campaign.authenticate_planned_evaluation_start(
        start, record, value.ledger, value.manifest, expected_binding=expected
    )


@pytest.mark.parametrize("identical_usage", [False, True])
def test_three_rounds_repeated_wire_occurrences_remain_separately_billed(
    case, identical_usage
):
    value = repeated_history(case, identical_usage=identical_usage)
    before = copy.deepcopy(value.__dict__)
    result = bindings(value)
    assert [len(r["checkpoint"]["progress"]["requests"]) for r in value.records] == [
        12,
        13,
        12,
    ]
    assert [r["checkpoint"]["budget"]["requests_sent"] for r in value.records] == [
        19,
        32,
        44,
    ]
    first = [
        e["data"]["budget"]["requests_sent"]
        for e in value.archive["events"]
        if e["event"] == "evaluation.batch"
    ][::2]
    assert first == [7, 19, 32]
    assert len(result) == 37
    accepted = [authenticate(value, expected) for expected in result.values()]
    assert len({row["request_id"] for row in accepted}) == 37
    assert {
        row["round_number"] for row in result.values() if row["phase"] == "assessment"
    } == {1, 2, 3}
    assert value.__dict__ == before
    repeated = [
        row
        for row in accepted
        if row["phase"] == "assessment"
        and row["wire_sha256"] == accepted[1]["wire_sha256"]
    ]
    assert len(repeated) == 3
    assert len({row["progress_usage"]["completion_tokens"] for row in repeated}) == (
        1 if identical_usage else 3
    )


def test_old_four_argument_unique_wire_result_is_unchanged(case):
    value = repeated_history(case, rounds=1)
    expected = next(
        row for row in bindings(value).values() if row["phase"] == "assessment"
    )
    record = value.records[0]
    start = next(
        s for s in value.starts if s["operation_id"] == expected["operation_id"]
    )
    original = campaign.authenticate_planned_evaluation_start(
        start, record, value.ledger, value.manifest
    )
    assert original == campaign.authenticate_planned_evaluation_start(
        start, record, value.ledger, value.manifest, expected_binding=None
    )
    assert original == authenticate(value, expected)
    repeated = repeated_history(case)
    with pytest.raises(ValueError, match="evaluation_wire_ambiguous_or_missing"):
        campaign.authenticate_planned_evaluation_start(
            repeated.starts[8], repeated.records[0], repeated.ledger, repeated.manifest
        )


@pytest.mark.parametrize(
    "fault",
    [
        "final_as_first",
        "bad_final_count",
        "sequence_bool",
        "sequence_reorder",
        "starts_reorder",
        "requests_reorder",
        "missing_request",
        "missing_anchor",
        "duplicate_record",
        "duplicate_operation",
        "overlap",
        "missing_terminal",
        "bad_offset",
        "bad_round",
        "ledger_order",
    ],
)
def test_repeated_wire_history_requires_complete_unique_serial_provenance(case, fault):
    value = repeated_history(case)
    events = value.archive["events"]
    batches = [e for e in events if e["event"] == "evaluation.batch"]
    if fault == "final_as_first":
        batches[0]["data"]["budget"] = copy.deepcopy(
            value.records[0]["checkpoint"]["budget"]
        )
    elif fault == "bad_final_count":
        batches[-1]["data"]["budget"]["requests_sent"] -= 1
        value.records[-1]["checkpoint"]["budget"]["requests_sent"] -= 1
    elif fault == "sequence_bool":
        events[0]["sequence"] = False
    elif fault == "sequence_reorder":
        events[1]["sequence"] = events[0]["sequence"]
    elif fault == "starts_reorder":
        value.starts[8], value.starts[9] = value.starts[9], value.starts[8]
    elif fault == "requests_reorder":
        requests = value.records[0]["checkpoint"]["progress"]["requests"]
        requests[1], requests[2] = requests[2], requests[1]
    elif fault == "missing_request":
        value.records[0]["checkpoint"]["progress"]["requests"].pop()
    elif fault == "missing_anchor":
        value.records[0]["checkpoint"]["progress"]["requests"][0]["input_sha256"] = (
            value.records[0]["checkpoint"]["progress"]["requests"][1]["input_sha256"]
        )
    elif fault == "duplicate_record":
        value.records.insert(1, copy.deepcopy(value.records[0]))
    elif fault == "duplicate_operation":
        first, second = value.starts[8]["operation_id"], value.starts[9]["operation_id"]
        value.starts[9]["operation_id"] = first
        for event in events:
            if (
                event["event"] in {"llm.start", "llm.complete"}
                and event["data"].get("operation_id") == second
            ):
                event["data"]["operation_id"] = first
    elif fault == "overlap":
        index = next(
            i
            for i, e in enumerate(events)
            if e["event"] == "llm.start"
            and e["data"]["operation_id"] == value.starts[8]["operation_id"]
        )
        complete = events.pop(index + 1)
        events.insert(index + 2, complete)
        for i, event in enumerate(events):
            event["sequence"] = i
    elif fault == "missing_terminal":
        events[:] = [
            e
            for e in events
            if not (
                e["event"] == "llm.complete"
                and e["data"]["operation_id"] == value.starts[8]["operation_id"]
            )
        ]
        for i, event in enumerate(events):
            event["sequence"] = i
    elif fault == "bad_offset":
        value.ledger["capacity_revisions"][1]["allocation"]["round"]["offsets"][
            "assessment"
        ] = 0
    elif fault == "bad_round":
        value.ledger["capacity_revisions"][1]["allocation"]["round"]["number"] = 1
    elif fault == "ledger_order":
        entries = list(value.ledger["entries"].items())
        entries[0], entries[1] = entries[1], entries[0]
        value.ledger["entries"] = dict(entries)
    with pytest.raises(ValueError):
        bindings(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("progress_index", False),
        ("progress_index", -1),
        ("request_id", "unknown"),
        ("round_number", True),
        ("round_number", 99),
        ("progress_identity", "f" * 64),
        ("wire_sha256", "e" * 64),
        ("phase", "global"),
        ("operation_id", "other"),
    ],
)
def test_expected_binding_is_not_a_wire_or_round_bypass(case, field, value):
    history = repeated_history(case)
    expected = next(
        row for row in bindings(history).values() if row["phase"] == "assessment"
    )
    bad = {**expected, field: value}
    # Keep the actual start independent from a deliberately forged expected key.
    start = next(
        s for s in history.starts if s["operation_id"] == expected["operation_id"]
    )
    with pytest.raises(ValueError):
        campaign.authenticate_planned_evaluation_start(
            start,
            history.records[0],
            history.ledger,
            history.manifest,
            expected_binding=bad,
        )


def test_unknown_last_occurrence_keeps_full_hold_without_owner_restore(case):
    value = repeated_history(case, identical_usage=True, unknown=True)
    before = copy.deepcopy(value.ledger)
    result = bindings(value)
    expected = result[value.starts[-1]["operation_id"]]
    accepted = authenticate(value, expected)
    assert accepted["progress_status"] == "interrupted_no_replay"
    assert accepted["progress_usage"] == {"usage_available": False}
    known = value.ledger["used_tokens"]
    assert (
        campaign.historical_unknown_token_hold(
            value.ledger, known, expected_plan=value.ledger["plan"]
        )
        == accepted["reservation"]
    )
    assert value.ledger == before


@pytest.mark.parametrize(
    "fault", ["source", "schema", "model", "reservation", "extra_binding"]
)
def test_occurrence_does_not_replace_existing_complete_wire_validation(case, fault):
    value = repeated_history(case)
    expected = next(
        row for row in bindings(value).values() if row["phase"] == "assessment"
    )
    start = copy.deepcopy(
        next(
            row
            for row in value.starts
            if row["operation_id"] == expected["operation_id"]
        )
    )
    if fault == "source":
        start["messages"][1]["content"] = "Different original material."
    elif fault == "schema":
        start["response_format"]["json_schema"]["schema"][
            "description"
        ] = "Changed schema."
    elif fault == "model":
        start["model"] = "other"
    elif fault == "reservation":
        value.ledger["entries"][expected["request_id"]]["reserved_tokens"] -= 1
    else:
        expected["skip_validation"] = True
    with pytest.raises(ValueError):
        campaign.authenticate_planned_evaluation_start(
            start,
            value.records[0],
            value.ledger,
            value.manifest,
            expected_binding=expected,
        )


def test_read_only_occurrence_authentication_cannot_restore_or_construct_owner(
    case, monkeypatch
):
    value = repeated_history(case)
    before = copy.deepcopy(value.__dict__)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("historical semantic owner forbidden")

    monkeypatch.setattr(PlannedBudget, "__init__", forbidden)
    monkeypatch.setattr(PlannedBudget, "from_snapshot", forbidden)
    result = bindings(value)
    assert len([authenticate(value, expected) for expected in result.values()]) == 37
    assert value.__dict__ == before
