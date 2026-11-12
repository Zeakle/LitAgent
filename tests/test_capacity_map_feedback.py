"""Keep invalid edit maps rejected while exercising bounded corrective feedback."""

import asyncio
import copy
import json

import pytest
from test_capacity_repair import OverflowInventoryGateway, run_capacity_case
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_repair import (
    CONTENT_GUIDANCE,
    MAP_GUIDANCE,
    capacity_targets,
    normalize_edit_map,
    retryable_map_failure,
)
from litagent.quality import digest
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    chapter_request_input,
    prepare_planned_input,
    repair_request_input,
    writer_budget_plan,
)


def orphan_map(case):
    """Reproduce 030's two deleted originals and one unaccounted replacement."""
    body = copy.deepcopy(case.body)
    body["taxonomy"][0]["sentences"].append(
        copy.deepcopy(body["taxonomy"][0]["sentences"][0])
    )
    targets = capacity_targets(body, {})
    tid = next(k for k, t in targets.items() if t["path"][0] == "taxonomy")
    result = {
        "outcome": "patched",
        "explanation": "Consolidate the classification paragraph.",
        "patches": [{"target_id": tid, "sentences": targets[tid]["original"][:1]}],
        "edit_records": [
            {
                "unit_id": f"writer:taxonomy:0:{i}",
                "replacement_indexes": [],
                "reason": "Removed redundant exposition.",
            }
            for i in range(2)
        ],
    }
    return body, targets, result, tid


def test_030_orphan_is_located_but_never_filled_by_the_host(case):
    """Keep the exact orphan pattern rejected until a separate response repairs it."""
    body, targets, result, tid = orphan_map(case)
    original = copy.deepcopy((body, result))
    with pytest.raises(WriterError) as caught:
        normalize_edit_map(result, targets)
    assert str(caught.value) == "writer_capacity_new_unit_unaccounted"
    assert caught.value.family == "protocol"
    assert caught.value.diagnostics == [
        {
            "target_id": tid,
            "path": ["taxonomy", 0, 0],
            "end": 2,
            "replacement_count": 1,
            "unmapped_indexes": [0],
        }
    ]
    assert (body, result) == original
    # Only a separate, explicitly corrected response may supply this mapping.
    fixed = copy.deepcopy(result)
    for record in fixed["edit_records"]:
        record["replacement_indexes"] = [0]
    normalized = normalize_edit_map(fixed, targets)
    assert all(
        r["replacement_indexes"] == [0]
        for r in normalized["edit_records"]
        if r["unit_id"].startswith("writer:taxonomy:")
    )


@pytest.mark.parametrize(
    "fault",
    [None, "sha", "json", "family", "empty", "path", "bool", "legacy", "missing_body"],
)
def test_only_current_located_map_feedback_can_use_a_remaining_slot(case, fault):
    """Reject stale, forged or legacy feedback before it can trigger another call."""
    body, targets, result, _ = orphan_map(case)
    with pytest.raises(WriterError) as caught:
        normalize_edit_map(result, targets)
    envelope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )["envelope"]
    failure = {
        "code": caught.value.code,
        "family": "quality",
        "source_body_sha256": digest(body),
        "diagnostics": caught.value.diagnostics,
    }
    if fault == "sha":
        failure["source_body_sha256"] = "0" * 64
    if fault == "json":
        failure["code"] = "writer_json_invalid"
    if fault == "family":
        failure["family"] = "protocol"
    if fault == "empty":
        failure["diagnostics"] = []
    if fault == "path":
        failure["diagnostics"][0]["path"] = ["methods", 0, 0]
    if fault == "bool":
        failure["diagnostics"][0]["unmapped_indexes"] = [False]
    if fault == "legacy":
        envelope.pop("global_findings_contract")
    if fault == "missing_body":
        body = None
    assert retryable_map_failure(failure, body, envelope) is (fault is None)


class MapFeedbackGateway(OverflowInventoryGateway):
    """Return 030's map defect through HTTP, then an independently corrected map."""

    def __init__(self, case, *, fault="once", unsupported=False):
        """Configure only the external transport fault and keep real consumers."""
        super().__init__(case, unsupported=unsupported)
        self.map_fault = fault
        self.map_requests = []

    def answer(self, payload, native_schema):
        """Create a complete response whose first map deliberately contradicts it."""
        result = super().answer(payload, native_schema)
        if payload.get("cause") != "capacity":
            return result
        self.map_requests.append(copy.deepcopy(payload))
        if len(self.map_requests) == 1 or self.map_fault == "repeat":
            for record in result["edit_records"]:
                if record["unit_id"].startswith("writer:methods:0:"):
                    record["replacement_indexes"] = []
        return result

    async def __call__(self, request):
        """Exercise SDK JSON, usage and cancellation boundaries at HTTP."""
        value = json.loads(request.content)
        payload = json.loads(value.get("input", value.get("messages"))[1]["content"])
        if payload.get("cause") == "capacity" and self.map_fault == "cancel":
            raise asyncio.CancelledError()
        response = await super().__call__(request)
        if payload.get("cause") == "capacity" and self.map_fault in {"json", "unknown"}:
            import httpx

            result = response.json()
            if self.map_fault == "unknown":
                result.pop("usage", None)
            else:
                for output in result["output"]:
                    if output["type"] == "message":
                        output["content"][0]["text"] = '{"outcome":'
            return httpx.Response(200, json=result)
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["once", "repeat", "json", "unknown"])
async def test_real_consumer_chain_shares_two_slots_and_never_commits_invalid_map(
    case, tmp_path, monkeypatch, fault
):
    """Use the second shared slot only for eligible localized map defects."""
    gateway = MapFeedbackGateway(case, fault=fault)
    report, _ = await run_capacity_case(
        case, tmp_path, monkeypatch, gateway=gateway, allow_uncertain=fault == "unknown"
    )
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    expected = 2 if fault in {"once", "repeat"} else 1
    assert len(attempts) == len(gateway.map_requests) == expected
    assert all(a["cause"] == "capacity" for a in attempts)
    assert attempts[0]["status"] == "failed"
    assert report["delivery"]["publishable"] is (fault == "once")
    if expected == 2:
        first, second = gateway.map_requests
        assert first["body_sha256"] == second["body_sha256"]
        assert first["current_body"] == second["current_body"]
        assert (
            second["previous_local_failure"]["source_body_sha256"]
            == first["body_sha256"]
        )
        assert list(
            second["previous_local_failure"]["diagnostics"]["unmapped_indexes"].values()
        ) == [[0]]
    if fault == "once":
        assert gateway.inventory_calls == 2
        details = report["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert details["stages"]["global_check"] == "completed"
        assert details["unverified_count"] == details["unsupported_count"] == 0
        assert len(report["metadata"]["evidence_rewrite_attempts"]) == 2
    else:
        assert gateway.inventory_calls == 1
        assert report["evaluation"]["faithfulness"]["skipped"]
        assert not any(
            x.get("committed") for x in report["metadata"]["evidence_rewrite_attempts"]
        )
        if fault == "unknown":
            ledger = json.loads(
                (tmp_path / "repair.json.budget-ledger.json").read_text()
            )
            assert ledger["usage_uncertain"]
            assert [
                e["status"]
                for e in ledger["entries"].values()
                if e["stage"] == "repair"
            ] == ["interrupted_no_replay"]


@pytest.mark.asyncio
async def test_cancelled_map_response_keeps_one_attempt_and_never_replays(
    case, tmp_path, monkeypatch
):
    """Keep an unknown cancelled response occupied without a second request."""
    gateway = MapFeedbackGateway(case, fault="cancel")
    with pytest.raises(asyncio.CancelledError):
        await run_capacity_case(case, tmp_path, monkeypatch, gateway=gateway)
    state = json.loads((tmp_path / "repair.json").read_text())["state"]
    assert len(state["attempts"]) == 1
    assert state["attempts"][0]["status"] == "cancelled"
    ledger = json.loads((tmp_path / "repair.json.budget-ledger.json").read_text())
    assert ledger["usage_uncertain"]
    assert [
        e["status"] for e in ledger["entries"].values() if e["stage"] == "repair"
    ] == ["interrupted_no_replay"]


@pytest.mark.asyncio
async def test_corrected_map_still_requires_scientific_support(
    case, tmp_path, monkeypatch
):
    """Reject the candidate despite a structurally correct map when science fails."""
    gateway = MapFeedbackGateway(case, unsupported=True)
    report, _ = await run_capacity_case(case, tmp_path, monkeypatch, gateway=gateway)
    assert len(gateway.map_requests) == 2 and gateway.inventory_calls == 2
    assert not report["delivery"]["publishable"]
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 2
    latest = report["metadata"]["evidence_rewrite_attempts"][-1]
    assert (
        latest["candidate_evaluation"]["faithfulness"]["details"]["unsupported_count"]
        > 0
    )
    assert not latest["committed"]
    assert report["metadata"]["writer_failure"]["code"] == "writer_candidate_failed"


def test_v7_guidance_shares_actual_builders_and_legacy_wire_stays_unchanged(case):
    """Attach the same concise guidance at admission and actual transmission."""
    for policy in ("planned-budget-v6", "planned-budget-v7"):
        envelope = writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy=policy
        )["envelope"]
        system, payload, _ = chapter_request_input(
            case.packet, case.aliases, BODY[0], {}, {}, RANGE_WRITING_PLAN, envelope
        )
        prepared, _ = prepare_planned_input(
            system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, envelope
        )
        assert (CONTENT_GUIDANCE in prepared) is (policy == "planned-budget-v7")
        targets = capacity_targets(case.body, {})
        system, _, _ = repair_request_input(
            case.packet,
            case.body,
            targets,
            case.aliases,
            cause="capacity",
            previous_failure=None,
            writing_plan=RANGE_WRITING_PLAN,
            envelope=envelope,
            wire_contract="flat-writer-v1",
        )
        assert (MAP_GUIDANCE in system) is (policy == "planned-budget-v7")


def test_feedback_size_limit_rejects_without_cutting_diagnostics(case):
    """Retain oversized guidance and reject it instead of silently truncating it."""
    envelope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )["envelope"]
    previous = {
        "code": "writer_capacity_new_unit_unaccounted",
        "diagnostics": [{"explanation": "x" * 13000}],
    }
    original = copy.deepcopy(previous)
    with pytest.raises(ValueError, match="envelope"):
        repair_request_input(
            case.packet,
            case.body,
            capacity_targets(case.body, {}),
            case.aliases,
            cause="capacity",
            previous_failure=previous,
            writing_plan=RANGE_WRITING_PLAN,
            envelope=envelope,
            wire_contract="flat-writer-v1",
        )
    assert previous == original


def test_feedback_shares_only_positions_proven_by_current_target_bindings(case):
    """Preserve every localized defect while sharing duplicated range metadata."""
    from litagent.writing_flat import repair_projection

    body, targets, result, tid = orphan_map(case)
    with pytest.raises(WriterError) as caught:
        normalize_edit_map(result, targets)
    feedback = {
        "family": "quality",
        "code": caught.value.code,
        "source_body_sha256": digest(body),
        "diagnostics": caught.value.diagnostics,
    }
    original = copy.deepcopy(feedback)
    envelope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )["envelope"]
    _, payload, _ = repair_request_input(
        case.packet,
        body,
        targets,
        case.aliases,
        cause="capacity",
        previous_failure=feedback,
        writing_plan=RANGE_WRITING_PLAN,
        envelope=envelope,
        wire_contract="flat-writer-v1",
    )
    shared = payload["previous_local_failure"]["diagnostics"]
    assert shared == {"unmapped_indexes": {tid: [0]}, "replacement_counts": {tid: 1}}
    assert payload["targets"][tid]["path"] == original["diagnostics"][0]["path"]
    assert payload["targets"][tid]["end"] == original["diagnostics"][0]["end"]
    assert payload["current_body"] == body and feedback == original
    with pytest.raises(WriterError, match="writer_patch_identity_invalid"):
        repair_projection(
            {
                "cause": "capacity",
                "current_body": body,
                "targets": targets,
                "body_sha256": "0" * 64,
                "previous_local_failure": feedback,
            },
            envelope,
        )
