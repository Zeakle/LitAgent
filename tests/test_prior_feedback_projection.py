"""Preserve authenticated prior feedback within the current edit authority."""

import copy
import json

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.quality import digest
from litagent.writing_v2 import WriterError


def feedback_case(total=74, outside=(5, 38)):
    """Keep each diagnostic distinct even when several share one source span."""
    diagnostics = [
        {
            "claim_id": f"claim_{i}",
            "source_fragment_id": f"f{i}",
            "status": "unsupported",
        }
        for i in range(total)
    ]
    failure = {
        "code": "writer_candidate_failed",
        "family": "quality",
        "candidate_sha256": "b" * 64,
        "diagnostics": diagnostics,
    }
    directory = {
        "contract": "host-repair-feedback-directory-v1",
        "current_report_sha256": "a" * 64,
        "prior_report_sha256": failure["candidate_sha256"],
        "diagnostics_sha256": digest(diagnostics),
        "fragments": {
            f"f{i}": {"current_target_ids": [] if i in outside else ["original_target"]}
            for i in range(total)
        },
    }
    directory["identity"] = digest(directory)
    return directory, failure


def test_mixed_feedback_retains_every_authorized_diagnostic_without_changing_input():
    from litagent.stage_repair import select_prior_feedback

    directory, failure = feedback_case()
    before = copy.deepcopy((directory, failure))
    result = select_prior_feedback(directory, failure)
    selected = result["failure"]["diagnostics"]
    assert len(selected) == 72
    assert selected == [
        d for i, d in enumerate(failure["diagnostics"]) if i not in (5, 38)
    ]
    assert result["audit"]["included_diagnostic_indexes"] == [
        i for i in range(74) if i not in (5, 38)
    ]
    assert result["audit"]["deferred_diagnostic_indexes"] == [5, 38]
    assert result["audit"]["previous_failure"] == failure
    assert result["directory"]["diagnostics_sha256"] == digest(selected)
    assert result["directory"]["identity"] == digest(
        {k: v for k, v in result["directory"].items() if k != "identity"}
    )
    assert set(result["directory"]["fragments"]) == {
        d["source_fragment_id"] for d in selected
    }
    assert (directory, failure) == before


@pytest.mark.parametrize("total,outside", [(3, ()), (3, (0, 1, 2)), (0, ())])
def test_all_inside_outside_and_empty_prior_feedback_are_explicit(total, outside):
    from litagent.stage_repair import select_prior_feedback

    directory, failure = feedback_case(total, outside)
    result = select_prior_feedback(directory, failure)
    if total and not outside:
        assert result == {"directory": directory, "failure": failure, "audit": None}
    else:
        assert result["directory"] is None and result["failure"] is None
        assert result["audit"]["included_diagnostic_indexes"] == []
        assert result["audit"]["deferred_diagnostic_indexes"] == list(range(total))


@pytest.mark.parametrize(
    "fault", ["identity", "diagnostic_hash", "candidate", "fragment", "targets"]
)
def test_projection_rejects_invalid_authentication_instead_of_dropping_rows(fault):
    from litagent.stage_repair import select_prior_feedback

    directory, failure = feedback_case()
    if fault == "identity":
        directory["identity"] = "0" * 64
    elif fault == "diagnostic_hash":
        failure["diagnostics"][0]["status"] = "supported"
    elif fault == "candidate":
        failure["candidate_sha256"] = "0" * 64
    else:
        if fault == "fragment":
            directory["fragments"].pop("f0")
        else:
            directory["fragments"]["f0"]["current_target_ids"] = [True]
        directory["identity"] = digest(
            {k: v for k, v in directory.items() if k != "identity"}
        )
    with pytest.raises(WriterError, match="writer_feedback_identity_invalid"):
        select_prior_feedback(directory, failure)


def test_shared_source_span_does_not_merge_independent_diagnostics():
    from litagent.stage_repair import select_prior_feedback

    directory, failure = feedback_case(3, (2,))
    failure["diagnostics"][1]["source_fragment_id"] = "f0"
    del directory["fragments"]["f1"]
    directory["diagnostics_sha256"] = digest(failure["diagnostics"])
    directory["identity"] = digest(
        {k: v for k, v in directory.items() if k != "identity"}
    )
    result = select_prior_feedback(directory, failure)
    assert [d["claim_id"] for d in result["failure"]["diagnostics"]] == [
        "claim_0",
        "claim_1",
    ]
    assert list(result["directory"]["fragments"]) == ["f0"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [1, 2])
async def test_real_retry_sends_mapped_prior_feedback_and_replays_full_witness(
    case, tmp_path, monkeypatch, failures
):
    """Change only mocked provider responses, including one unedited negative."""
    from test_native_ragas_evaluation import verdict_pairs
    from test_separate_review_pipeline import execute
    from test_stream_writer_pipeline import StreamingGateway, journals, stream_config

    from litagent import stage_repair
    from litagent.writing_v2 import check_repair_guidance_record

    outside_text = "The comparison setting uses labeled examples"

    class Gateway(StreamingGateway):
        def __init__(self, fixture):
            super().__init__(fixture, reviews=0, failures=failures)
            self.repairs_sent = []
            delegate = self.semantic.chat

            async def semantic_response(messages, **kwargs):
                response = await delegate(messages, **kwargs)
                payload = json.loads(messages[1]["content"])
                if "context_pool" in payload and 1 <= self.repair_count <= failures:
                    raw = json.loads(response.content)
                    for key, value in raw.items():
                        for statement, verdict in verdict_pairs(
                            payload["jobs"][key], value
                        ):
                            if outside_text in statement:
                                verdict.update(
                                    verdict=0,
                                    reason="Offline independent reassessment negative.",
                                )
                    response.content = json.dumps(raw)
                return response

            self.semantic.chat = semantic_response

        def answer(self, payload, schema):
            response = super().answer(payload, schema)
            if payload.get("requested_body_section") == "taxonomy":
                response["units"][0]["text"] = outside_text + "[C:c1]."
            if "response_target_keys" in payload and self.global_calls:
                self.repairs_sent.append(copy.deepcopy(payload))
            return response

    gateway = Gateway(case)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stream_config())
    assert gateway.repair_count == failures + 1, report["metadata"].get(
        "writer_failure"
    )
    assert gateway.global_calls == failures + 2
    assert report["delivery"]["publishable"]
    first, *retries = gateway.repairs_sent
    for retry in retries:
        assert first["response_target_keys"] == retry["response_target_keys"]
        assert retry["repair_guidance"]["prior_candidate"] is not None
    (journal,) = journals(tmp_path)
    audits = [
        r
        for r in journal["deferred_prior_feedback"]
        if r["policy"] == stage_repair.PRIOR_FEEDBACK_POLICY
    ]
    records = [
        c["guidance_wire_record"]
        for c in journal["calls"].values()
        if c.get("guidance_wire_record", {}).get("prior_replay_witness")
    ]
    assert len(audits) == len(records) == failures
    envelope = report["metadata"]["writer_document"]["budget_envelope"]
    for audit, record in zip(audits, records):
        assert audit["included_diagnostic_indexes"]
        assert audit["deferred_diagnostic_indexes"]
        full = audit["previous_failure"]["diagnostics"]
        assert record["prior_replay_witness"]["previous_failure"]["diagnostics"] == full
        selected = record["arguments"]["previous_failure"]["diagnostics"]
        assert selected == [full[i] for i in audit["included_diagnostic_indexes"]]
        assert len(selected) < len(full)
        check_repair_guidance_record(envelope, record)
        forged = copy.deepcopy(record)
        forged["prior_replay_witness"]["previous_failure"]["diagnostics"] = selected
        forged["identity"] = digest(
            {k: v for k, v in forged.items() if k != "identity"}
        )
        with pytest.raises(
            WriterError,
            match="writer_(feedback_identity|guidance_wire_receipt)_invalid",
        ):
            check_repair_guidance_record(envelope, forged)
    first_attempt = report["metadata"]["evaluation_history"]["attempts"][0]
    assert not first_attempt["committed"]
    assert not first_attempt["working_draft_decision"]["promote"]
