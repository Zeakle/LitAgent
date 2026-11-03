"""Exercise the opt-in transport through actual SDK and report consumers."""

import json

import pytest
import test_flat_writer_pipeline as flat_tests
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_reference_set_pipeline import DuplicateGateway
from test_repair_feedback import run_feedback_case
from test_writer_v2 import case  # noqa: F401

from litagent.writing_flat import (
    TRANSPORT_REPAIR_INSTRUCTION,
    TRANSPORT_REVISION,
    transport_identity,
)
from litagent.writing_v2 import WriterError, writer_budget_plan


def selected_options():
    """Select the single new revision and its existing required ancestors."""
    return {
        "inventory_recovery": True,
        "mixed_inventory_recovery": True,
        "repair_feedback": True,
        "reference_contract": "ordered-reference-set-v1",
        "transport_revision": TRANSPORT_REVISION,
    }


def select_frozen(monkeypatch):
    """Keep real frozen planning and execution while selecting server policy."""
    import litagent.frozen_writing as frozen

    original_plan, original_run = frozen.plan_frozen_budget, frozen.run_frozen

    def plan(*args, **kwargs):
        for key, value in selected_options().items():
            kwargs.setdefault(key, value)
        return original_plan(*args, **kwargs)

    async def run(*args, **kwargs):
        for key, value in selected_options().items():
            kwargs.setdefault(key, value)
        return await original_run(*args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", plan)
    monkeypatch.setattr(frozen, "run_frozen", run)


async def setup(case, tmp_path, gateway):
    """Activate the real locked budget, preserving the normal SDK boundary."""
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **selected_options(),
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    return owner, make_session(owner, case, tmp_path)


@pytest.mark.asyncio
async def test_frozen_synthesis_reviewer_report_and_evaluation_remain_real(
    case, tmp_path, monkeypatch
):
    select_frozen(monkeypatch)
    await flat_tests.test_flat_real_frozen_worker_runner_and_report_exit(
        case, tmp_path, monkeypatch, False, False, "planned-budget-v7"
    )
    journal = json.loads((tmp_path / "repair.json.writer-v2.json").read_text())
    writing = [
        call
        for name, call in journal["calls"].items()
        if name.startswith("body:") or name == "summary_initial"
    ]
    assert len(writing) == 6
    assert all(call["transport_revision"] == TRANSPORT_REVISION for call in writing)
    assert all(call["transport_identity"] == transport_identity() for call in writing)
    assert all(
        call["wire_bindings"]["transport_revision"] == TRANSPORT_REVISION
        for call in writing
    )


@pytest.mark.asyncio
async def test_body_summary_and_two_repairs_share_revision_and_occurrence_receipts(
    case, tmp_path
):
    gateway = DuplicateGateway(case)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        summaries = await session.summaries(body, "summary_initial")
        received, _ = session.reference_input(body)
        body_target = {
            "m": {
                "path": ["methods", 0, 0],
                "original": body["methods"][0]["sentences"][0],
                "explanation": "Clarify the supported comparison.",
            }
        }
        candidate = await session.patch(body, body_target, cause="evaluation")
        assert (
            session.reference_input(candidate)[0]["introduction"]
            == received["introduction"]
        )
        summary_target = {
            "a": {
                "path": ["abstract", 0, 0],
                "original": summaries["abstract"][0]["sentences"][0],
                "explanation": "Clarify the selected frozen finding.",
            }
        }
        revised = await session.patch(
            summaries, summary_target, cause="evaluation", summary_body=body
        )
        assert session.reference_input(revised)[0]["conclusion"] == (
            session.reference_input(summaries)[0]["conclusion"]
        )
        assert len(session.state["attempts"]) == 2
        for sent in gateway.sent:
            assert sent["payload"]["writing_envelope"]["transport_revision"] == (
                TRANSPORT_REVISION
            )
            if "targets" in sent["payload"]:
                assert (
                    TRANSPORT_REPAIR_INSTRUCTION
                    in sent["request"]["input"][0]["content"]
                )
                assert all(
                    t["original_unit_ids"] for t in sent["payload"]["targets"].values()
                )
        assert len(gateway.sent) == 8
        with pytest.raises(WriterError, match="repair_limit"):
            await session.patch(body, body_target, cause="evaluation")
        assert len(gateway.sent) == 8
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing_revision", "wrong_revision", "identity"])
async def test_call_transport_tampering_stops_before_another_request(
    case, tmp_path, fault
):
    gateway = DuplicateGateway(case)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        call = session.journal["calls"]["body:introduction"]
        if fault == "missing_revision":
            call.pop("transport_revision")
        elif fault == "wrong_revision":
            call["transport_revision"] = "wrong"
        else:
            call["transport_identity"] = "0" * 64
        sent = len(gateway.sent)
        with pytest.raises(
            WriterError, match="writer_reference_input_identity_invalid"
        ):
            await session.summaries(body, "summary_initial")
        assert len(gateway.sent) == sent
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [1, 2])
async def test_failed_feedback_replays_same_revision_and_never_submits_bad_candidate(
    case, tmp_path, monkeypatch, failing_candidates
):
    select_frozen(monkeypatch)
    report, gateway = await run_feedback_case(
        case, tmp_path, monkeypatch, failing_candidates
    )
    assert gateway.repair_count == 2
    assert gateway.inventories == 3
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2
    for attempt in attempts:
        receipt = attempt["provenance_receipt"]
        assert receipt["transport_revision"] == TRANSPORT_REVISION
        assert receipt["transport_identity"] == transport_identity()
        assert len(receipt["original_unit_directory_sha256"]) == 64
    repairs = [r["payload"] for r in gateway.sent if "targets" in r["payload"]]
    assert all("FLOPs by 44%" in json.dumps(r["current_body"]) for r in repairs)
    assert all("FLOPs by 45%" not in json.dumps(r["current_body"]) for r in repairs)
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["coverage"]["complete"]
    if failing_candidates == 1:
        assert final["supported_count"] == final["factual_claim_count"]
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert report["delivery"]["publishable"]
    else:
        assert "FLOPs by 44%" in report["survey"]
        assert not report["delivery"]["publishable"]
