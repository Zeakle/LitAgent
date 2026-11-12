"""Replay the real workbench publication decision from its local receipts."""

import hashlib
import json
from pathlib import Path

import pytest

from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.quality import digest
from litagent.safety.budget import CostBudget

ARTIFACTS = Path(__file__).resolve().parents[1] / "artifacts"
ACCEPTANCE = ARTIFACTS / "acceptance"
RUN = "235d5bf1dba2"
FINAL_SHA = "bd983eb7f6b7f208277641aa86fbd854d01885069d21f20dd4ff82d7fc1370f1"


@pytest.fixture(scope="module")
def published_run():
    """Load the sealed real acceptance run when available on this machine."""
    terminal_path = ACCEPTANCE / "workbench-vit-402-20261111" / "terminal.json"
    if not terminal_path.is_file():
        pytest.skip("local 402 acceptance archive is unavailable")

    terminal = json.loads(terminal_path.read_bytes())
    archive_path = ACCEPTANCE / "workbench-vit-386-20261110" / "runs" / f"{RUN}.json"
    archive_bytes = archive_path.read_bytes()
    assert hashlib.sha256(archive_bytes).hexdigest() == terminal["archive_sha256"]
    archive = json.loads(archive_bytes)
    assert archive["run_id"] == terminal["status"]["task_id"] == RUN

    ledger = archive["nodes"]["worker:adversarial_review"]["output"][
        "evidence_selection"
    ]["selected_items"]
    budget_path = ARTIFACTS / "targeted-repairs" / f"{RUN}.json.budget-ledger.json"
    budget = CostBudget.from_snapshot(json.loads(budget_path.read_bytes()))
    return archive, ledger, budget


def test_402_replays_exact_automatic_acceptance(published_run):
    """Only the fourth, fully evaluated candidate binds to the ready report."""
    archive, ledger, budget = published_run
    report = archive["report"]
    metadata = report["metadata"]
    evaluation = report["evaluation"]
    history = metadata["evaluation_history"]
    attempts = history["attempts"]

    assert [attempt["committed"] for attempt in attempts] == [False, False, False, True]
    assert [
        attempt["candidate_evaluation"]["faithfulness"]["details"]["unsupported_count"]
        for attempt in attempts
    ] == [20, 31, 11, 0]
    assert all(attempt["candidate"] != report["survey"] for attempt in attempts[:-1])
    assert attempts[-1]["candidate"] == report["survey"]
    assert (
        attempts[-1]["candidate_evaluation"]
        == history["final_evaluation"]
        == evaluation
    )

    snapshot = freeze_automatic_inputs(report, ledger, run_id=RUN, budget=budget)
    replayed = automatic_acceptance(snapshot, evaluation)
    assert replayed == metadata["automatic_acceptance"]
    assert replayed["publishable"] is True and replayed["actor"] == "system"
    assert (
        archive["delivery"]
        == report["delivery"]
        == {
            "status": "ready",
            "publishable": True,
            "reason_codes": [],
        }
    )
    details = evaluation["faithfulness"]["details"]
    assert (
        hashlib.sha256(report["survey"].encode("utf-8")).hexdigest()
        == snapshot["source_report_sha256"]
        == history["source_report_sha256"]
        == details["source_report_sha256"]
        == replayed["report_sha256"]
        == FINAL_SHA
    )
    assert replayed["evaluation_sha256"] == digest(evaluation)
    assert details["factual_claim_count"] == details["supported_count"] == 454
    assert details["unsupported_count"] == details["unverified_count"] == 0


def test_402_rejects_mixing_report_and_evaluation_versions(published_run):
    """Passing evaluation cannot be reused for an earlier candidate or vice versa."""
    archive, ledger, budget = published_run
    report = archive["report"]
    first = report["metadata"]["evaluation_history"]["attempts"][0]
    with pytest.raises(
        ValueError, match="^strict_writing_scope_notice_binding_invalid$"
    ):
        freeze_automatic_inputs(
            {**report, "survey": first["candidate"]}, ledger, run_id=RUN, budget=budget
        )
    with pytest.raises(ValueError, match="automatic_evaluation_"):
        freeze_automatic_inputs(
            {**report, "evaluation": first["candidate_evaluation"]},
            ledger,
            run_id=RUN,
            budget=budget,
        )
