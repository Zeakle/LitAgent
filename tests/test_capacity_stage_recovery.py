"""Retry authenticated capacity candidates through the real runtime consumers."""

import copy
import json

import pytest
from test_llm_stage_repair import stage_config
from test_native_capacity_recovery import replay

from litagent.safety.planned_accounting import validate_accounting_snapshot


def configuration():
    """Use the shipped experimental policy with only the transport mocked."""
    config = stage_config()
    config.llm.base_url = "https://offline.invalid/api/v3"
    config.observability.enabled = False
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode", ["recovered", "repeated", "exhausted", "unknown_usage"]
)
async def test_capacity_candidate_repair_is_bounded_and_settled(
    tmp_path, monkeypatch, mode
):
    feedbacks = []
    authentication_faults = []
    if mode == "recovered":
        from litagent import stage_repair
        from litagent.safety.planned_budget import planned_backend

        reject = stage_repair.reject_candidate

        def authenticate_faults(session, error, retained, **kwargs):
            if kwargs["category"] != "capacity":
                return reject(session, error, retained, **kwargs)
            name = f"repair_{session.state['attempts'][-1]['index']}"
            owner = planned_backend(session.llm._cost_budget)
            settlement = owner.entries[f"{session.identity}:{name}"]
            stage = settlement["stage"]
            settlement["stage"] = "evaluation_revision/repair"
            try:
                assert reject(session, error, retained, **kwargs) is None
                authentication_faults.append("wrong_partition")
            finally:
                settlement["stage"] = stage
            call = session.journal["calls"][name]
            original_usage = copy.deepcopy(call["usage"])
            call["usage"]["completion_tokens"] += 1
            try:
                assert reject(session, error, retained, **kwargs) is None
                authentication_faults.append("usage_mismatch")
            finally:
                call["usage"] = original_usage
            invalid = copy.deepcopy(retained)
            invalid["draft"] += " changed host manuscript"
            assert reject(session, error, invalid, **kwargs) is None
            authentication_faults.append("retained_manuscript_changed")
            return reject(session, error, retained, **kwargs)

        monkeypatch.setattr(stage_repair, "reject_candidate", authenticate_faults)

    def capacity_response(gateway, payload, reply):
        feedbacks.append(payload.get("candidate_feedback", {}))
        if mode == "recovered" and gateway.capacity_calls > 1:
            return
        content = next(row for row in reply["output"] if row["type"] == "message")[
            "content"
        ][0]
        candidate = json.loads(content["text"])
        suffix = gateway.capacity_calls if mode == "exhausted" else 1
        candidate["patches"][0]["source_unit_ids"].append(
            payload["targets"]["r2"]["original_unit_ids"][0]
        )
        if mode == "exhausted":
            candidate["patches"][0]["text"] += f" Candidate {suffix}."
        content["text"] = json.dumps(candidate)
        if mode == "unknown_usage":
            reply.pop("usage")

    report, gateway, plan = await replay(
        tmp_path,
        monkeypatch,
        mode="candidate",
        config=configuration(),
        capacity_response=capacity_response,
    )
    expected = {"recovered": 2, "repeated": 2, "exhausted": 3, "unknown_usage": 1}[mode]
    assert gateway.capacity_calls == expected, report["metadata"].get("writer_failure")
    state = report["metadata"]["targeted_repair_state"]
    assert len(state["attempts"]) == expected
    assert all(attempt["category"] == "capacity" for attempt in state["attempts"])
    journal = json.loads((tmp_path / "repairs.json.writer-v2.json").read_bytes())
    snapshot = json.loads((tmp_path / "budget-ledger.json").read_bytes())
    validate_accounting_snapshot(snapshot, expected_plan=plan)
    assert not report["delivery"]["publishable"]
    if mode == "unknown_usage":
        assert snapshot["usage_uncertain"]
        assert not journal.get("candidate_rejections")
        assert "summary" not in gateway.calls
        return
    rejections = list(journal["candidate_rejections"].values())
    assert len(rejections) == (1 if mode == "recovered" else expected)
    assert all(row["category"] == "capacity" for row in rejections)
    assert all(
        row["diagnostics"][0]["reason"] == "multiple_authorized_targets"
        for row in rejections
    )
    assert feedbacks[1]["diagnostics_complete"]
    assert feedbacks[1]["category"] == "capacity"
    assert not snapshot["usage_uncertain"]
    assert all(entry["status"] == "settled" for entry in snapshot["entries"].values())
    if mode == "recovered":
        assert authentication_faults == [
            "wrong_partition",
            "usage_mismatch",
            "retained_manuscript_changed",
        ]
        assert gateway.calls.count("summary") == 1
        assert "assessment" in gateway.calls and "global" in gateway.calls
        candidate = journal["candidates"][0]
        assert not candidate["committed"]
        details = candidate["evaluation"]["faithfulness"]["details"]
        assert details["stages"]["global_check"] == "completed"
        assert details["unsupported_count"] > 0
        assert (
            rejections[0]["retained_report_sha256"]
            == report["metadata"]["writing_bindings"]["report_sha256"]
        )
    else:
        assert rejections[-1]["action"] == "stop"
        assert "summary" not in gateway.calls
        assert "assessment" not in gateway.calls and "global" not in gateway.calls
        assert not journal["candidates"]
