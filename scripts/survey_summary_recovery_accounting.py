"""Authenticate settled benchmark requests before starting a summary continuation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from litagent.llm.responses import decode_private_receipt, normalize_responses
from litagent.quality import digest
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.writing_v2 import BODY

_LENGTH_FAILURE = "writer_summary_capacity_exceeded"
_WRITER_TASK = "adversarial_review:synthesis:r1"
_WORKER_STOP = "strict_worker_failed"
_CLOSED_WRITING_STAGES = ["body", "body_initial", "format_body"]


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"summary_recovery_parent_{reason}")


def _tokens(receipt: dict) -> dict[str, int]:
    usage = {key: receipt[key] for key in ("prompt_tokens", "completion_tokens")}
    _require(
        receipt.get("usage_available") is True
        and all(type(value) is int and value >= 0 for value in usage.values()),
        "usage_unavailable",
    )
    return usage


def _paid_operations(events: list[dict]) -> dict[str, dict]:
    """Join each authenticated trace start to its complete provider usage."""
    starts, completes = {}, {}
    for event in events:
        kind = event["event"]
        if kind == "llm.failed":
            raise ValueError("summary_recovery_parent_unknown_operation")
        if kind not in ("llm.start", "llm.complete"):
            continue
        data = event["data"]
        index = starts if kind == "llm.start" else completes
        operation_id = data["operation_id"]
        _require(
            type(operation_id) is str
            and bool(operation_id)
            and operation_id not in index,
            "duplicate_operation",
        )
        index[operation_id] = data

    _require(bool(starts) and starts.keys() == completes.keys(), "unpaired_operation")
    for operation_id, end in completes.items():
        start = starts[operation_id]
        _require(
            type(end.get("task_id")) is str
            and start.get("task_id") == end["task_id"]
            and type(end.get("actual_model")) is str
            and bool(end["actual_model"])
            and start.get("model") == end.get("model") == end["actual_model"]
            and end.get("finish_reason") == "stop"
            and end.get("provider_error") is None,
            "provider_receipt_conflict",
        )
        _require(
            type(end.get("total_tokens")) is int
            and end["total_tokens"] == sum(_tokens(end).values()),
            "provider_usage_conflict",
        )
    return completes


def _check_writer_entries(
    operations: list[dict],
    journal: dict,
    entries: dict,
    generation_model: str,
    call_stages: dict[str, str],
) -> None:
    """Tie each actual provider receipt to its writer call and frozen ledger entry."""
    calls = journal["calls"]
    identity = journal["identity"]
    _require(
        type(identity) is str
        and bool(identity)
        and calls.keys() == call_stages.keys()
        and set(entries) == {f"{identity}:{name}" for name in call_stages},
        "writer_call_directory_conflict",
    )
    traced = {}
    for event in operations:
        request_id = event["request_id"]
        _require(
            type(request_id) is str
            and bool(request_id)
            and request_id not in traced
            and event["task_id"] == _WRITER_TASK
            and event["actual_model"] == generation_model
            and event.get("provider_status") == "completed",
            "writer_call_trace_conflict",
        )
        traced[request_id] = event

    for name, call in calls.items():
        request_id = call["request_id"]
        event = traced.pop(request_id)
        entry = entries[f"{identity}:{name}"]
        raw = call["raw"]
        usage = _tokens(event)
        _require(
            call.get("status") == "response_received"
            and call.get("provider_status") == "completed"
            and call.get("provider_error") is None
            and call.get("finish_reason") == "stop"
            and call.get("model") == generation_model
            and type(raw) is str
            and raw == event.get("content")
            and hashlib.sha256(raw.encode()).hexdigest() == call.get("raw_sha256")
            and _tokens(call["usage"]) == usage
            and entry["stage"] == call_stages[name]
            and entry["status"] == "settled"
            and entry["sent"] is True
            and entry["usage_sha256"] == digest(usage)
            and entry["actual_tokens"] == sum(usage.values()),
            "writer_settlement_conflict",
        )
    _require(not traced, "unclaimed_writer_receipt")


def checked_parent_ledger(
    archive: dict, row: dict, journal: dict, plan_path: Path, ledger_path: Path
) -> dict:
    """Return the unchanged terminal ledger after checking all paid parent receipts.

    The caller first authenticates the archive and journal bytes against the
    benchmark row. This function grants no authority to resume or clear a stop.
    """
    try:
        plan = json.loads(Path(plan_path).read_bytes())
        ledger = json.loads(Path(ledger_path).read_bytes())
        validate_accounting_snapshot(ledger, expected_plan=plan)
        _check_parent_binding(archive, row, journal, ledger, plan)
        return ledger
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("summary_recovery_parent_accounting_shape_invalid") from exc


def _check_parent_binding(
    archive: dict, row: dict, journal: dict, ledger: dict, plan: dict
) -> None:
    """Confirm the known length stop and reconcile the whole settled campaign slot."""
    run_id = row["run_id"]
    report = archive["report"]
    _require(
        type(run_id) is str
        and bool(run_id)
        and archive["run_id"] == archive["session_id"] == run_id
        and row["source"] == "replay"
        and row["delivery"] == report["delivery"]["status"] == "blocked"
        and report["delivery"]["publishable"] is False
        and report["partial"] is True
        and plan["contract"] == "planned-budget-v7"
        and archive["budget_plan"] == plan
        and ledger["stop_reason"] == _WORKER_STOP
        and ledger["usage_uncertain"] is False
        and report["metadata"]["writer_failure"] == journal["failure"]
        and journal["failure"]["code"] == _LENGTH_FAILURE,
        "stop_binding_conflict",
    )
    events = archive["events"]
    failures = [event["data"] for event in events if event["event"] == "worker.failed"]
    _require(
        len(failures) == 1
        and failures[0].get("task_id") == "adversarial_review"
        and failures[0].get("error") == _LENGTH_FAILURE,
        "worker_stop_conflict",
    )
    uses = {
        "known_tokens": ledger["used_tokens"],
        "requests": ledger["requests_sent"],
        "usage_uncertain": False,
    }
    _require(
        [event["data"] for event in events if event["event"] == "benchmark.accounting"]
        == [{**uses, "stop_reason": _WORKER_STOP}]
        and all(archive["budget_state"][key] == value for key, value in uses.items())
        and archive["budget_state"]["plan_identity"] == plan["identity"]
        and row["usage"]["usage_uncertain"] is False
        and row["usage"]["usage_complete"] is True
        and row["usage"]["paid_requests"] == uses["requests"]
        and row["usage"]["billed_tokens"] == uses["known_tokens"],
        "receipt_totals_conflict",
    )
    paid = _paid_operations(events)
    extraction = [item for item in paid.values() if item["task_id"] == "extract"]
    writing = [item for item in paid.values() if item["task_id"] != "extract"]
    routing = archive["config"]["effective"]["llm"]["model_routing"]
    _require(
        all(item["actual_model"] == routing["extraction"] for item in extraction)
        and len(extraction) == plan["spent_requests"]
        and sum(item["total_tokens"] for item in extraction) == plan["spent_tokens"]
        and len(paid) == uses["requests"]
        and sum(item["total_tokens"] for item in paid.values()) == uses["known_tokens"],
        "extraction_carry_conflict",
    )
    call_stages = {
        "summary_initial": "summary",
        **{
            f"body:{section}": "body_initial" if section == "introduction" else "body"
            for section in BODY
        },
    }
    _check_writer_entries(
        writing, journal, ledger["entries"], routing["generation"], call_stages
    )


def _check_retry_wire(call: dict, entry: dict) -> None:
    """Bind the settled repair to the stored private request and response."""
    receipt = call["provider_receipt"]
    request, response = decode_private_receipt(receipt)
    normalized = normalize_responses(json.dumps(response).encode())
    output = request["text"]["format"]
    kwargs = {
        "provider_api": "responses",
        "model": request["model"],
        "max_output_tokens": request["max_output_tokens"],
        "response_format": {
            "type": output["type"],
            "json_schema": {
                key: value for key, value in output.items() if key != "type"
            },
        },
    }
    if "thinking" in request:
        kwargs["extra_body"] = {"thinking": request["thinking"]}
    for key in ("reasoning_effort", "responses_transport"):
        if call.get(key) is not None:
            kwargs[key] = call[key]
    _require(
        call["stage"] == "repair"
        and call["provider_api"] == "responses"
        and call["max_tokens"] == request["max_output_tokens"]
        and call["reservation"] == entry["reserved_tokens"]
        and receipt["status_code"] == 200
        and receipt["headers"]["x-request-id"] == call["request_id"]
        and normalized["content"] == call["raw"]
        and normalized["model"] == request["model"] == call["model"]
        and normalized["finish_reason"] == "stop"
        and normalized["provider_status"] == "completed"
        and normalized["provider_error"] is None
        and _tokens(normalized["usage"]) == _tokens(call["usage"])
        and entry["wire_sha256"] == digest([request["input"], kwargs, "responses"]),
        "retry_wire_conflict",
    )


def checked_failed_summary_retry(
    archive: dict,
    row: dict,
    journal: dict,
    state: dict,
    ledger: dict,
    baseline_ledger: dict,
) -> dict:
    """Authenticate the fourth retry's failure and settled spend without authorizing it."""
    try:
        _check_failed_summary_retry(
            archive, row, journal, state, ledger, baseline_ledger
        )
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("summary_recovery_parent_retry_shape_invalid") from exc
    return ledger


def _check_failed_summary_retry(archive, row, journal, state, ledger, baseline):
    """Verify the retry stopped on three distinct received summary edit attempts."""
    plan = baseline["plan"]
    validate_accounting_snapshot(baseline, expected_plan=plan)
    validate_accounting_snapshot(ledger, expected_plan=plan)
    run_id = row["run_id"]
    origin = row["retry_of"]
    events = archive["events"]
    failures = [e["data"] for e in events if e["event"] == "worker.failed"]
    checkpoints = [
        e["data"] for e in events if e["event"] == "benchmark.summary_checkpoint"
    ]
    _require(
        (row["case_id"], row["profile_id"], row["repetition"])
        == ("few-shot-metric", "external-only", 2)
        and row["source"] == "replay"
        and row["timing_scope"] == "summary_checkpoint"
        and origin["run_id"] == "replay-few-shot-metric-external-only-r2"
        and isinstance(run_id, str)
        and run_id.startswith(f'{origin["run_id"]}-retry-')
        and archive["run_id"] == archive["session_id"] == run_id
        and archive["status"] == row["status"] == "failed"
        and archive["report"] is None
        and archive["delivery"] is row["delivery"] is None
        and archive["error"] == journal["failure"]["code"] == _LENGTH_FAILURE
        and journal["failure"]["family"] == "quality"
        and journal["benchmark_summary_checkpoint"] is True
        and len(failures) == 1
        and failures[0]["task_id"] == "adversarial_review"
        and failures[0]["error"] == _LENGTH_FAILURE
        and len(checkpoints) == 1
        and checkpoints[0]["source_run_id"] == origin["run_id"]
        and checkpoints[0]["source_snapshot_sha256"] == origin["snapshot_sha256"],
        "retry_stop_binding_conflict",
    )
    _require(
        plan["contract"] == "planned-budget-v7"
        and archive["budget_plan"] == plan
        and archive["budget_state"] == ledger
        and baseline["stop_reason"] == ledger["stop_reason"] == _WORKER_STOP
        and baseline["usage_uncertain"] is ledger["usage_uncertain"] is False
        and baseline["closed_stages"] == []
        and ledger["closed_stages"] == _CLOSED_WRITING_STAGES
        and baseline["capacity_revisions"] == ledger["capacity_revisions"] == []
        and baseline["capacity_allocation"] is ledger["capacity_allocation"] is None,
        "retry_budget_binding_conflict",
    )
    identity = journal["identity"]
    stages = {f"repair_{index}": "repair" for index in range(3)}
    inherited = baseline["entries"]
    inherited_calls = {f"body:{section}" for section in BODY} | {"summary_initial"}
    _require(
        set(inherited) == {f"{identity}:{name}" for name in inherited_calls}
        and all(ledger["entries"].get(key) == value for key, value in inherited.items())
        and set(journal["calls"]) == inherited_calls | stages.keys(),
        "retry_inherited_entries_conflict",
    )
    new_entries = {
        key: entry for key, entry in ledger["entries"].items() if key not in inherited
    }
    new_calls = {name: journal["calls"][name] for name in stages}
    paid = _paid_operations(events)
    routing = archive["config"]["effective"]["llm"]["model_routing"]
    _check_writer_entries(
        list(paid.values()),
        {"identity": identity, "calls": new_calls},
        new_entries,
        routing["generation"],
        stages,
    )
    for name, call in new_calls.items():
        _check_retry_wire(call, new_entries[f"{identity}:{name}"])
    if "state" in state:
        state = state["state"]
    attempts = state["attempts"]
    _require(
        archive["targeted_repair_state"] == state
        and state["version"] == "independent-targeted-repair-v1"
        and state["run_id"] == run_id
        and state["limit"] == plan["repair_limit"]
        and len(attempts) == len(stages)
        and all(
            attempt["index"] == index
            and attempt["stage"] == "native_initial"
            and attempt["status"] == "failed"
            and attempt["category"] == "capacity"
            and attempt["error_code"] == _LENGTH_FAILURE
            and attempt["input_sha256"]
            == new_calls[f"repair_{index}"]["guidance_wire_record"]["payload_sha256"]
            for index, attempt in enumerate(attempts)
        ),
        "retry_state_conflict",
    )
    tokens = sum(operation["total_tokens"] for operation in paid.values())
    requests = len(paid)
    row_usage = row["usage"]
    _require(
        tokens == ledger["used_tokens"] - baseline["used_tokens"]
        and requests
        == ledger["requests_sent"] - baseline["requests_sent"]
        == ledger["call_count"] - baseline["call_count"]
        and [
            event["data"]
            for event in events
            if event["event"] == "benchmark.accounting"
        ]
        == [
            {
                "known_tokens": tokens,
                "requests": requests,
                "usage_uncertain": False,
                "stop_reason": _WORKER_STOP,
            }
        ]
        and row_usage["usage_uncertain"] is False
        and row_usage["usage_complete"] is True
        and row_usage["paid_requests"] == row_usage["llm_calls"] == requests
        and row_usage["billed_tokens"] == row_usage["total_tokens"] == tokens
        and archive["usage"]["total_tokens"] == tokens,
        "retry_receipt_totals_conflict",
    )
