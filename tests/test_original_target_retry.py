"""Verify that rejected candidate feedback cannot block an authorized retry."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from test_native_responses import envelope, owner_for

from litagent.body_quality import bound_repair_diagnostics
from litagent.llm.routing import ModelBoundClient
from litagent.quality import digest
from litagent.writing_v2 import (
    FORMAT_JSON_POLICY,
    LOSSLESS_JSON_POLICY,
    ORIGINAL_TARGET_RETRY_POLICY,
    WriterError,
    check_repair_guidance_record,
    defer_disjoint_prior_feedback,
    repair_report,
)


def feedback(*, outside=True):
    """Describe a candidate already authenticated by the directory producer."""
    failure = {"candidate_sha256": "candidate", "diagnostics": [{"id": "claim_1"}]}
    directory = {
        "current_report_sha256": "original",
        "prior_report_sha256": failure["candidate_sha256"],
        "diagnostics_sha256": digest(failure["diagnostics"]),
        "fragments": {
            "fragment1": {"current_target_ids": ["target1"]},
            "fragment2": {"current_target_ids": [] if outside else ["target1"]},
        },
    }
    directory["identity"] = digest(directory)
    return directory, failure


@pytest.mark.parametrize(
    ("policy", "action", "outside", "deferred"),
    [
        (FORMAT_JSON_POLICY, "prose", True, True),
        (FORMAT_JSON_POLICY, "prose", False, False),
        (FORMAT_JSON_POLICY, "combined", True, True),
        (FORMAT_JSON_POLICY, "combined", False, False),
        (FORMAT_JSON_POLICY, "inventory", True, False),
        (LOSSLESS_JSON_POLICY, "prose", True, False),
    ],
)
def test_original_report_retries_defer_disjoint_feedback(
    policy, action, outside, deferred
):
    directory, failure = feedback(outside=outside)
    result = defer_disjoint_prior_feedback(
        {"run_limits": {"syntax_policy": policy}}, directory, failure, action=action
    )
    if not deferred:
        assert result is None
        return
    assert result == {
        "policy": ORIGINAL_TARGET_RETRY_POLICY,
        "current_report_sha256": "original",
        "feedback_directory_sha256": directory["identity"],
        "outside_target_fragment_ids": ["fragment2"],
        "previous_failure": failure,
    }
    assert result["previous_failure"]["diagnostics"] is not failure["diagnostics"]


@pytest.mark.parametrize("field", ["identity", "candidate_sha256", "diagnostics"])
def test_retry_deferral_never_accepts_tampered_feedback(field):
    directory, failure = feedback()
    (directory if field == "identity" else failure)[field] = "tampered"
    with pytest.raises(WriterError, match="writer_feedback_identity_invalid"):
        defer_disjoint_prior_feedback(
            {"run_limits": {"syntax_policy": FORMAT_JSON_POLICY}},
            directory,
            failure,
            action="prose",
        )


@pytest.mark.asyncio
async def test_076_original_retry_reaches_sdk_with_only_authorized_targets(tmp_path):
    """Use copied 076 files and replace only the provider's HTTP response."""
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/acceptance/ragas-native-single-20261109-076"
    )
    paths = [
        "report.json",
        "repair.json",
        "repair.json.writer-v2.json",
        "final-budget.json",
    ]
    if not all((root / name).is_file() for name in paths):
        pytest.skip("local 076 acceptance archive is unavailable")
    frozen = {name: (root / name).read_bytes() for name in paths}
    report, state, journal, snapshot = (json.loads(frozen[name]) for name in paths)
    metadata = report["metadata"]
    original = report["survey"]
    prior = copy.deepcopy(journal["candidates"][-1])
    assert not prior["committed"]
    previous_failure = {
        "family": "quality",
        "code": "writer_candidate_regressed",
        "candidate_sha256": prior["report_sha256"],
        "diagnostics": bound_repair_diagnostics(
            prior["evaluation"]["faithfulness"]["details"]
        ),
    }
    assert len(previous_failure["diagnostics"]) == 35
    journal["failure"] = previous_failure
    (tmp_path / "repair.json").write_bytes(frozen["repair.json"])
    journal_path = tmp_path / "repair.json.writer-v2.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    decision = metadata["evaluation_history"]["attempts"][2]["recovery_decision"]
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(json.loads(body["input"][1]["content"]))
        return httpx.Response(
            200,
            json=envelope(
                json.dumps(
                    {
                        "outcome": "declined",
                        "explanation": "Offline retry of authorized targets.",
                        "patches": [],
                    }
                ),
                model=body["model"],
                usage={
                    "input_tokens": 20,
                    "output_tokens": 100,
                    "total_tokens": 120,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            ),
        )

    owner, budget, _ = await owner_for(handler)
    try:
        budget.restore(snapshot)
        client = ModelBoundClient(
            owner, journal["calls"]["repair_1"]["model"], "generation"
        )
        runner = SimpleNamespace(
            _stage_llm=lambda _: client,
            _targeted_repair_path=lambda: tmp_path / "repair.json",
            _session_id=state["state"]["run_id"],
            _trace_hook=None,
            _cost_budget=budget,
        )
        outcome = await repair_report(
            runner,
            query=metadata["query"],
            report_data=report,
            results={},
            initial_evaluation=report["evaluation"],
            initial_quality=report["quality"],
            report_language=metadata["report_language"],
            recovery=decision,
        )
    finally:
        await owner.close()

    assert outcome.reason_code == "writer_repair_declined", metadata.get(
        "writer_failure"
    )
    assert outcome.attempted and not outcome.committed
    assert len(requests) == 1
    sent = requests[0]
    assert set(sent["targets"]) == set(decision["targets"])
    assert sent["body_sha256"] == digest(metadata["writer_document"]["body"])
    assert sent["previous_local_failure"] == {
        "code": "writer_prior_feedback_outside_targets",
        "family": "quality",
        "guidance_ref": "current",
    }
    assert prior["report_sha256"] not in json.dumps(sent)
    after = budget.snapshot()
    assert after["requests_sent"] == snapshot["requests_sent"] + 1
    assert after["used_tokens"] == snapshot["used_tokens"] + 120
    assert not after["usage_uncertain"]
    saved = json.loads(journal_path.read_text())
    assert saved["candidates"] == [prior]
    wire_record = saved["calls"]["repair_2"]["guidance_wire_record"]
    assert wire_record["canonical"]["prior_candidate"] is None
    assert wire_record["prior_replay_witness"] is None
    document = metadata["writer_document"]
    check_repair_guidance_record(
        document["budget_envelope"],
        wire_record,
        packet=document["packet"],
        document=document,
        survey=original,
        bindings=metadata["writing_bindings"],
        targets=wire_record["authorized_targets"],
    )
    audit = saved["deferred_prior_feedback"]
    assert len(audit) == 1
    assert audit[0]["previous_failure"] == previous_failure
    assert len(audit[0]["outside_target_fragment_ids"]) == 9
    assert audit[0]["current_report_sha256"] == decision["source_report_sha256"]
    assert report["survey"] == original
    assert not report["delivery"]["publishable"]
    assert all(
        (root / name).read_bytes() == contents for name, contents in frozen.items()
    )


@pytest.mark.asyncio
async def test_400_combined_retry_reaches_sdk_with_original_authority(tmp_path):
    """Reproduce the blocked combined action with the actual archived material."""
    from litagent.eval.ragas_repair import native_repair_diagnostics

    artifacts = Path(__file__).resolve().parents[1] / "artifacts"
    run = "a909cb995cd6"
    root = artifacts / "targeted-repairs"
    archive_path = (
        artifacts / "acceptance/workbench-vit-386-20261110/runs" / f"{run}.json"
    )
    if not archive_path.is_file():
        pytest.skip("local 400 acceptance archive is unavailable")
    report = json.loads(archive_path.read_bytes())["report"]
    state = (root / f"{run}.json").read_bytes()
    journal = json.loads((root / f"{run}.json.writer-v2.json").read_bytes())
    snapshot = json.loads((root / f"{run}.json.budget-ledger.json").read_bytes())
    assert journal["failure"]["code"] == "writer_guidance_unrepairable"
    (prior,) = journal["candidates"]
    assert prior["committed"] is False
    previous_failure = {
        "family": "quality",
        "code": "writer_candidate_failed",
        "candidate_sha256": prior["report_sha256"],
        "diagnostics": native_repair_diagnostics(
            prior["evaluation"]["faithfulness"]["details"]
        ),
    }
    assert len(previous_failure["diagnostics"]) == 29
    journal["failure"] = previous_failure
    (tmp_path / "repair.json").write_bytes(state)
    journal_path = tmp_path / "repair.json.writer-v2.json"
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    metadata = report["metadata"]
    decision = metadata["evaluation_history"]["attempts"][-1]["recovery_decision"]
    assert decision["action"] == "combined" and len(decision["targets"]) == 27
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(json.loads(body["input"][1]["content"]))
        return httpx.Response(
            200,
            json=envelope(
                json.dumps(
                    {
                        "outcome": "declined",
                        "explanation": "Offline fixture.",
                        "patches": [],
                    }
                ),
                model=body["model"],
                usage={
                    "input_tokens": 20,
                    "output_tokens": 100,
                    "total_tokens": 120,
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            ),
        )

    owner, budget, _ = await owner_for(handler)
    try:
        budget.restore(snapshot)
        client = ModelBoundClient(
            owner, journal["calls"]["repair_0"]["model"], "generation"
        )
        runner = SimpleNamespace(
            _stage_llm=lambda _: client,
            _targeted_repair_path=lambda: tmp_path / "repair.json",
            _session_id=run,
            _trace_hook=None,
            _cost_budget=budget,
        )
        outcome = await repair_report(
            runner,
            query=metadata["query"],
            report_data=report,
            results={},
            initial_evaluation=report["evaluation"],
            initial_quality=report["quality"],
            report_language=metadata["report_language"],
            recovery=decision,
        )
    finally:
        await owner.close()

    assert outcome.reason_code == "writer_repair_declined", metadata.get(
        "writer_failure"
    )
    assert outcome.attempted and not outcome.committed
    (sent,) = requests
    assert set(sent["targets"]) == set(decision["targets"])
    assert prior["report_sha256"] not in json.dumps(sent)
    saved = json.loads(journal_path.read_text())
    (audit,) = saved["deferred_prior_feedback"]
    assert audit["previous_failure"] == previous_failure
    assert len(audit["outside_target_fragment_ids"]) == 1
    assert saved["candidates"] == [prior]
    after = budget.snapshot()
    assert after["requests_sent"] == snapshot["requests_sent"] + 1
    assert after["used_tokens"] == snapshot["used_tokens"] + 120
