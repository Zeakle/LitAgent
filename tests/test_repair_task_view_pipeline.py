"""Exercise task views through real consumers, replacing only model transport."""

import copy
import json

import pytest
import test_scientific_repair_pipeline as scientific
import test_structured_repair_pipeline as structured
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.writing_flat import (
    REPAIR_TASK_VIEW_ADMISSION_RULE,
    REPAIR_TASK_VIEW_REVISION,
)
from litagent.writing_v2 import check_guidance_wire_record


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_task_view_real_consumers_reassess_whole_report_before_commit(
    case, tmp_path, monkeypatch, failing_candidates
):
    """Both HTTP attempts share limits and cannot publish a failed candidate."""
    gateway = structured.LongScientificGateway(case, failing_candidates, reviewer=True)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    monkeypatch.setattr(
        structured,
        "FLAGS",
        {
            **structured.FLAGS,
            "repair_guidance_wire_revision": REPAIR_TASK_VIEW_REVISION,
        },
    )
    structured.select_structured_policy(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, failing_candidates)
    state, journal, ledger = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    expected = min(2, failing_candidates + 1)
    assert len(repairs) == gateway.repair_count == expected
    assert gateway.inventories == expected + 1
    for index, entry in enumerate(repairs):
        wire = entry["payload"]["repair_guidance"]
        assert wire["wire_revision"] == REPAIR_TASK_VIEW_REVISION
        assert wire["admission_rule"] == REPAIR_TASK_VIEW_ADMISSION_RULE
        assert (wire["prior_candidate"] is not None) is (index > 0)
        model_scope = entry["payload"]["writing_envelope"]
        assert "repair_guidance_json_bytes" not in model_scope
        assert (
            model_scope["repair_guidance_capacity_rule"]
            == REPAIR_TASK_VIEW_ADMISSION_RULE
        )
        for card in case.cards.values():
            assert card["source_quote"] in json.dumps(entry["request"])
        assert structured.LOCAL_MARKER not in json.dumps(entry["request"])
        assert "FLOPs by 44%" in json.dumps(entry["payload"]["current_body"])
        assert "FLOPs by 45%" not in json.dumps(entry["payload"]["current_body"])
    structured.assert_local_notes_retained(tmp_path, gateway)
    assert ledger["requests_sent"] == len(gateway.sent)
    assert ledger["usage_uncertain"] is False
    assert all(row["status"] == "settled" for row in ledger["entries"].values())
    for entry in journal["calls"].values():
        if entry.get("stage") == "repair" and entry.get("status") == "completed":
            record = copy.deepcopy(entry["guidance_wire_record"])
            envelope = report["metadata"]["writer_document"]["budget_envelope"]
            check_guidance_wire_record(envelope, record)
    if failing_candidates < 2:
        assert report["delivery"]["publishable"] is True
        details = report["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"] is True
        assert details["supported_count"] == details["factual_claim_count"]
        assert details["unsupported_count"] == details["unverified_count"] == 0
    else:
        assert report["delivery"]["publishable"] is False
        assert not any(candidate["committed"] for candidate in journal["candidates"])
        assert "FLOPs by 44%" in report["survey"]


def select_policy(monkeypatch):
    """Opt into the finite trusted task view without bypassing consumers."""
    import test_source_map_pipeline as source

    from litagent.writing_flat import _canonical_task_payload

    class TaskGateway(source.SourceMapGateway):
        """Read the declared input layout only at the mocked HTTP boundary."""

        def answer(self, payload, native_schema):
            return super().answer(_canonical_task_payload(payload), native_schema)

    monkeypatch.setattr(source, "SourceMapGateway", TaskGateway)
    monkeypatch.setattr(
        structured,
        "FLAGS",
        {
            **structured.FLAGS,
            "repair_guidance_wire_revision": REPAIR_TASK_VIEW_REVISION,
        },
    )
    structured.select_structured_policy(monkeypatch)


@pytest.mark.asyncio
async def test_task_view_summary_repair_keeps_readonly_body_premises(
    case, tmp_path, monkeypatch
):
    """Compact summary inputs retain their exact complete frozen body."""
    gateway = structured.SummaryGateway(case)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_policy(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, 0)
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    first = next(
        e for e in sent_with(gateway, "body_units") if "targets" not in e["payload"]
    )
    assert repairs[0]["payload"]["body_units"] == first["payload"]["body_units"]
    assert report["delivery"]["publishable"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "wrong_paper", "missing_core"])
async def test_task_view_does_not_exempt_wrong_sources_or_missing_core(
    case, tmp_path, monkeypatch, mode
):
    """The full scientific and core gates still decide candidate delivery."""
    from test_source_map_pipeline import run_source_case

    select_policy(monkeypatch)
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, mode=mode)
    assert 1 <= len(sent_with(gateway, "targets")) <= 2
    assert report["delivery"]["publishable"] is (mode == "valid")


@pytest.mark.asyncio
async def test_task_view_unknown_repair_keeps_reservation_and_never_retries(
    case, tmp_path, monkeypatch
):
    """An unknown response cannot release tokens or acquire another attempt."""
    from test_source_map_pipeline import run_source_case

    select_policy(monkeypatch)
    report, gateway = await run_source_case(case, tmp_path, monkeypatch, unknown=True)
    _, _, ledger = products(tmp_path)
    assert len(sent_with(gateway, "targets")) == 1
    assert report["delivery"]["publishable"] is False
    assert ledger["usage_uncertain"] is True
    assert ledger["stop_reason"] == "strict_usage_unknown"
    unknown = [
        r for r in ledger["entries"].values() if r["sent"] and r["status"] != "settled"
    ]
    assert len(unknown) == 1
    assert unknown[0]["actual_tokens"] is None
    assert unknown[0]["reserved_tokens"] > 0
