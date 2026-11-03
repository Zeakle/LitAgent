"""Exercise explicit unit-reference guidance through SDK and full evaluation."""

import copy
import json

import pytest
import test_evaluation_stability_pipeline as stability
import test_scientific_repair_pipeline as scientific
import test_structured_repair_pipeline as structured
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import products, run_source_case, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.safety.planned_budget import nested_json_size
from litagent.writing_v2 import WriterError, body_units, writer_budget_plan

REVISION = "host-unit-ref-guidance-v1"


def select_unit_reference_policy(monkeypatch, *, module=None):
    """Select the trusted option without replacing any application consumer."""
    monkeypatch.setattr(
        structured,
        "FLAGS",
        {**structured.FLAGS, "repair_guidance_wire_revision": REVISION},
    )
    structured.select_structured_policy(monkeypatch, module=module)


def assert_complete_model_wire(entry, *, prior):
    """Require readable origins, complete concerns, and actual target linkage."""
    payload = entry["payload"]
    wire = payload["repair_guidance"]
    assert set(wire) == {"contract", "wire_revision", "current", "prior_candidate"}
    assert wire["contract"] == structured.CONTRACT
    assert wire["wire_revision"] == REVISION
    assert (wire["prior_candidate"] is not None) is prior
    assert nested_json_size(wire) <= 12288
    actual_request = json.dumps(entry["request"])
    assert structured.LOCAL_MARKER not in actual_request
    assert "fN" in actual_request and "iN" in actual_request
    linked_targets = set()
    for name in ("current", "prior_candidate"):
        origin = wire[name]
        if origin is None:
            continue
        assert set(origin) == {
            "header",
            "host_catalog_sha256",
            "units",
            "issue_columns",
            "issues",
        }
        assert origin["header"]["coverage"]["complete"] is True
        assert origin["header"]["coverage"]["issues_total"] == len(origin["issues"])
        assert origin["issues"]
        for unit in origin["units"]:
            assert unit[0] in {"b", "u", "l"}
            if name == "prior_candidate":
                # A failed candidate has its own provenance, even for equal prose.
                assert unit[0] == "l"
        for issue in origin["issues"]:
            assert type(issue[0]) is int and issue[0] in {0, 1}
            if issue[0] == 0:
                linked_targets.update(issue[2])
            else:
                linked_targets.update(issue[1]["target_ids"])
    assert set(payload["targets"]) <= linked_targets
    return wire


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_unit_reference_sdk_keeps_full_reassessment_and_shared_repair_limit(
    case, tmp_path, monkeypatch, failing_candidates
):
    """Only full positive candidates commit; a second failed draft never does."""
    gateway = structured.LongScientificGateway(case, failing_candidates, reviewer=True)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_unit_reference_policy(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, failing_candidates)
    state, journal, ledger = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    expected = min(2, failing_candidates + 1)
    assert len(repairs) == gateway.repair_count == expected
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    for index, entry in enumerate(repairs):
        assert_complete_model_wire(entry, prior=index > 0)
        assert "FLOPs by 44%" in json.dumps(entry["payload"]["current_body"])
        assert "FLOPs by 45%" not in json.dumps(entry["payload"]["current_body"])
        for card in case.cards.values():
            assert card["source_quote"] in json.dumps(entry["request"])
    assert gateway.inventories == expected + 1
    assert len(sent_with(gateway, "body_units")) == expected + 1
    structured.assert_local_notes_retained(tmp_path, gateway)
    assert ledger["requests_sent"] == len(gateway.sent)
    assert ledger["usage_uncertain"] is False
    assert all(row["status"] == "settled" for row in ledger["entries"].values())
    assert state and journal["candidates"]
    if failing_candidates < 2:
        assert report["delivery"]["publishable"] is True
        final = report["evaluation"]["faithfulness"]["details"]
        assert final["coverage"]["complete"] is True
        assert final["supported_count"] == final["factual_claim_count"]
        assert final["unsupported_count"] == final["unverified_count"] == 0
        assert "FLOPs by" not in report["survey"]
    else:
        assert report["delivery"]["publishable"] is False
        assert "FLOPs by 44%" in report["survey"]
        assert not any(candidate["committed"] for candidate in journal["candidates"])


@pytest.mark.asyncio
async def test_summary_reference_view_exists_before_final_sdk_projection(
    case, tmp_path, monkeypatch
):
    """A summary repair reads the exact frozen body and still gets evaluated."""
    gateway = structured.SummaryGateway(case)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_unit_reference_policy(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, 0)
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    assert_complete_model_wire(repairs[0], prior=False)
    first_summary = next(
        entry
        for entry in sent_with(gateway, "body_units")
        if "targets" not in entry["payload"]
    )
    assert repairs[0]["payload"]["body_units"] == first_summary["payload"]["body_units"]
    assert body_units(report["metadata"]["writer_document"]["body"]) == (
        first_summary["payload"]["body_units"]
    )
    assert report["delivery"]["publishable"] is True
    assert "FLOPs by" not in report["survey"]
    structured.assert_local_notes_retained(tmp_path, gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "wrong_paper", "missing_core"])
async def test_reference_input_does_not_exempt_wrong_sources_or_core_coverage(
    case, tmp_path, monkeypatch, mode
):
    """Capacity metadata changes neither citation support nor delivery gates."""
    select_unit_reference_policy(monkeypatch)
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, mode=mode)
    repairs = sent_with(gateway, "targets")
    assert 1 <= len(repairs) <= 2
    assert_complete_model_wire(repairs[0], prior=False)
    assert report["delivery"]["publishable"] is (mode == "valid")


@pytest.mark.asyncio
async def test_unknown_reference_repair_preserves_hold_and_sends_no_retry(
    case, tmp_path, monkeypatch
):
    """An interrupted request stays occupied regardless of its input encoding."""
    select_unit_reference_policy(monkeypatch)
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, unknown=True)
    _, _, ledger = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    assert_complete_model_wire(repairs[0], prior=False)
    assert report["delivery"]["publishable"] is False
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert ledger["usage_uncertain"] is True
    assert ledger["stop_reason"] == "strict_usage_unknown"
    unknown = [
        row
        for row in ledger["entries"].values()
        if row["sent"] and row["status"] != "settled"
    ]
    assert len(unknown) == 1
    assert unknown[0]["actual_tokens"] is None
    assert unknown[0]["reserved_tokens"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("science_failure", [False, True])
async def test_inventory_recovery_and_reference_prose_edit_share_two_requests(
    case, tmp_path, monkeypatch, science_failure
):
    """A compact input grants no extra attempt after inventory recovery."""
    select_unit_reference_policy(monkeypatch, module=stability.mixed)
    gateway = stability.StableGateway(case, science_failure=science_failure)
    report, _ = await stability.execute(case, tmp_path, monkeypatch, gateway)
    _, journal, ledger = products(tmp_path)
    attempts = report["metadata"]["targeted_repair_state"]["attempts"]
    assert len(attempts) == 2
    assert sum("purpose" in attempt for attempt in attempts) == 1
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    assert_complete_model_wire(repairs[0], prior=False)
    assert len(sent_with(gateway, "inventory_recovery")) == 1
    assert report["delivery"]["publishable"] is (not science_failure)
    if science_failure:
        assert not any(candidate["committed"] for candidate in journal["candidates"])
    assert ledger["requests_sent"] == len(gateway.sent)


@pytest.mark.asyncio
async def test_local_reference_primitives_have_no_fabricated_report_offsets(
    case, tmp_path, monkeypatch
):
    """Local targets stay reportless and cannot acquire a third edit allowance."""
    select_unit_reference_policy(monkeypatch)
    gateway = structured.LocalGateway(case)
    owner, budget, _ = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **structured.FLAGS,
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = make_session(owner, case, tmp_path)
    try:
        original = await session.initial_body()
        targets = {
            "writer:methods:0:0": {
                "path": ["methods", 0, 0],
                "original": copy.deepcopy(original["methods"][0]["sentences"][0]),
                "explanation": structured.long_note("local-wire-target"),
            }
        }
        for _ in range(2):
            assert await session.patch(original, targets, cause="local") != original
        repairs = sent_with(gateway, "targets")
        assert len(repairs) == len(session.state["attempts"]) == 2
        for entry in repairs:
            origin = assert_complete_model_wire(entry, prior=False)["current"]
            assert origin["header"]["source_report_sha256"] is None
            assert origin["header"]["source_bindings_sha256"] is None
            assert {issue[1]["category"] for issue in origin["issues"]} == {"local"}
            assert all(issue[1]["body_paths"] for issue in origin["issues"])
        before = copy.deepcopy(budget.snapshot())
        with pytest.raises(WriterError, match="writer_repair_limit"):
            await session.patch(original, targets, cause="local")
        assert budget.snapshot() == before
        assert gateway.local_calls == 2
    finally:
        await owner.close()
