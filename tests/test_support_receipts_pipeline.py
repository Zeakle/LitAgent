"""Keep the new receipt revision on real frozen workers and SDK consumers."""

import json

import pytest
import test_flat_writer_pipeline as flat_tests
from test_repair_feedback import run_feedback_case
from test_writer_v2 import case  # noqa: F401

from litagent.eval.scoped import SUPPORT_VALIDATION_INSTRUCTION

SELECTION = {
    "inventory_recovery": True,
    "mixed_inventory_recovery": True,
    "classification_review": True,
    "repair_feedback": True,
    "reference_contract": "ordered-reference-set-v1",
    "transport_revision": "host-lossless-transport-v1",
    "support_validation_revision": "host-support-receipts-v1",
}


def select_frozen(monkeypatch):
    """Select the server policy while retaining the actual planner and runner."""
    import litagent.frozen_writing as frozen

    original_plan, original_run = frozen.plan_frozen_budget, frozen.run_frozen

    def plan(*args, **kwargs):
        for key, value in SELECTION.items():
            kwargs.setdefault(key, value)
        return original_plan(*args, **kwargs)

    async def run(*args, **kwargs):
        for key, value in SELECTION.items():
            kwargs.setdefault(key, value)
        return await original_run(*args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", plan)
    monkeypatch.setattr(frozen, "run_frozen", run)


@pytest.mark.asyncio
async def test_new_revision_reaches_synthesis_summary_reviewer_and_full_evaluation(
    case, tmp_path, monkeypatch
):
    select_frozen(monkeypatch)
    gateways = []
    original_gateway = flat_tests.FlatGateway

    class RecordingGateway(original_gateway):
        """Capture only the already-mocked external boundary."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            gateways.append(self)

    monkeypatch.setattr(flat_tests, "FlatGateway", RecordingGateway)
    await flat_tests.test_flat_real_frozen_worker_runner_and_report_exit(
        case, tmp_path, monkeypatch, False, False, "planned-budget-v7"
    )
    assert len(gateways) == 1
    gateway = gateways[0]
    journal = json.loads((tmp_path / "repair.json.writer-v2.json").read_text())
    received = [
        row for row in journal["calls"].values() if row["status"] == "response_received"
    ]
    assert len(received) == 7
    writing_requests = [r for r in gateway.sent if "writing_envelope" in r["payload"]]
    assert len(writing_requests) == 7
    for sent in writing_requests:
        envelope = sent["payload"]["writing_envelope"]
        assert (
            envelope["support_validation_revision"]
            == SELECTION["support_validation_revision"]
        )
    science = [r for r in gateway.sent if "support_passages" in r["payload"]]
    assert science
    inventories = [
        r for r in gateway.sent if "allowed_citation_aliases" in r["payload"]
    ]
    global_checks = [
        r for r in gateway.sent if "global_output_contract" in r["payload"]
    ]
    assert len(inventories) == len(global_checks) == 1
    assert len(gateway.sent) == 7 + len(science) + 2
    for sent in inventories + global_checks:
        prompt = sent["request"]["messages"][0]["content"]
        assert "support_basis may use a sufficient subset" not in prompt
        assert SUPPORT_VALIDATION_INSTRUCTION not in prompt
        assert sent["request"]["max_tokens"] == 16384
    for sent in science:
        payload = sent["payload"]
        prompt = sent["request"]["messages"][0]["content"]
        assert prompt.count(SUPPORT_VALIDATION_INSTRUCTION) == 1
        assert prompt.count("support_basis may use a sufficient subset") == 1
        assert all(
            source["text"] in {item["supporting_text"] for item in case.ledger.values()}
            for source in payload["original_supports"].values()
        )
        if "assertion_columns" in payload:
            assert "allowed_support_ids" in payload["assertion_columns"]
        else:
            assert all("allowed_support_ids" in row for row in payload["assertions"])
        request = sent["request"]
        assert request["max_tokens"] == 16384
    progress_files = list((tmp_path / "evaluation").glob("*/progress.json"))
    assert len(progress_files) == 1
    progress = json.loads(progress_files[0].read_text())
    entries = list(progress["budget"]["entries"].values())
    assert len(entries) == len(gateway.sent)
    assert all(row["sent"] and row["status"] == "settled" for row in entries)
    assert all(
        type(row["actual_tokens"]) is int and row["actual_tokens"] == 120
        for row in entries
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [1, 2])
async def test_new_receipts_preserve_two_shared_repairs_full_reassessment_and_rollback(
    case, tmp_path, monkeypatch, failing_candidates
):
    select_frozen(monkeypatch)
    report, gateway = await run_feedback_case(
        case, tmp_path, monkeypatch, failing_candidates
    )
    assert gateway.repair_count == 2
    assert gateway.inventories == 3
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 2
    envelope = report["metadata"]["writer_document"]["budget_envelope"]
    assert (
        envelope["support_validation_revision"]
        == SELECTION["support_validation_revision"]
    )
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"]
    assert details["stages"]["global_check"] == "completed"
    if failing_candidates == 1:
        assert report["delivery"]["publishable"]
        assert details["supported_count"] == details["factual_claim_count"]
        assert details["unsupported_count"] == details["unverified_count"] == 0
    else:
        assert not report["delivery"]["publishable"]
        assert "FLOPs by 44%" in report["survey"]
        assert "FLOPs by 45%" not in report["survey"]
