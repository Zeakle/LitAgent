"""Keep separated evaluation decisions on the actual scientific repair chain."""

import copy
import json

import pytest
import test_writer_source_view_pipeline as writer_view
from jsonschema import Draft202012Validator
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.eval.dimensions import CONTRACT, FIELDS, selected_dimensions


def nested_reply(value):
    """Rearrange the controlled peer's fields without changing any verdict."""
    result = copy.deepcopy(value)
    for index, decision in enumerate(value["decisions"]):
        report = {
            key: decision[key]
            for key in ("correct_kind", "atomic", "source_faithful", "explanation")
        }
        result["decisions"][index] = {
            "claim_id": decision["claim_id"],
            "report_check": report,
            "evidence_check": {
                key: decision[key]
                for key in ("status", "reason_code", "support_basis", "aspect_checks")
            },
        }
    return result


def separated_peer(gateway):
    """Replace only the controlled external response, not the real decoder."""
    original = gateway.evaluator.chat
    gateway.dimension_replies = []

    async def chat(messages, **options):
        response = await original(messages, **options)
        payload = gateway.evaluator.base.calls[-1]
        if "cited_materials" not in payload:
            return response
        value = nested_reply(json.loads(response.content))
        if payload.get("assessment_decision_contract") == "reason-only-decision-v3":
            # The controlled provider must obey the actually selected wire, while
            # preserving its original reason and every negative scientific flag.
            for decision in value["decisions"]:
                decision["evidence_check"].pop("status")
                decision["evidence_check"].pop("support_basis")
                decision["evidence_check"]["aspect_checks"] = {
                    check["aspect"]: {
                        key: val for key, val in check.items() if key != "aspect"
                    }
                    for check in decision["evidence_check"]["aspect_checks"]
                }
        schema = options["response_format"]["json_schema"]["schema"]
        Draft202012Validator(schema).validate(value)
        response.content = json.dumps(value)
        gateway.dimension_replies.append(copy.deepcopy(value))
        return response

    gateway.evaluator.chat = chat
    return gateway


def select_boundary(monkeypatch):
    """Select the same independent option before the genuine plan is built."""
    original = writer_view.source_flags

    def flags(view=writer_view.VIEW):
        return {**original(view), "evaluation_dimensions_contract": CONTRACT}

    monkeypatch.setattr(writer_view, "source_flags", flags)
    monkeypatch.setattr(writer_view, "conform_evaluation_replies", separated_peer)


def assert_boundary(result):
    """Bind saved raw decisions and host products to one frozen selector."""
    envelope = result.ledger["plan"]["envelope"]
    assert selected_dimensions(envelope)
    for key in FIELDS:
        assert result.journal[key] == envelope[key]
        assert all(
            call[key] == envelope[key] for call in result.journal["calls"].values()
        )
    assert result.gateway.dimension_replies
    for raw in result.gateway.dimension_replies:
        assert all(
            set(row) == {"claim_id", "report_check", "evidence_check"}
            for row in raw["decisions"]
        )
    assert result.ledger["plan"]["repair_limit"] == 2
    assert result.ledger["plan"]["maximum_tokens"] == 4500000
    assert result.ledger["plan"]["maximum_requests"] == 64
    # Evaluation raw is stored separately from canonical host decisions. Do not
    # mistake the writer journal's body responses for evaluation provider data.
    progress = result.report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    for key in FIELDS:
        assert progress[key] == envelope[key]
    return progress


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 2])
async def test_separated_decisions_keep_complete_two_attempt_scientific_pipeline(
    source_case, tmp_path, monkeypatch, failing_candidates
):
    select_boundary(monkeypatch)
    packet = copy.deepcopy(source_case.packet)
    result = await writer_view.run_scientific_case(
        source_case, tmp_path, monkeypatch, failures=failing_candidates
    )
    initial = assert_boundary(result)
    assert initial["coverage"]["complete"]
    assert initial["stages"]["global_check"] == "completed"
    assert result.report["metadata"]["writer_document"]["packet"] == packet
    assert len(result.report["metadata"]["targeted_repair_state"]["attempts"]) <= 2
    assert result.report["delivery"]["publishable"] is (failing_candidates == 0)
    if failing_candidates:
        assert not any(row["committed"] for row in result.journal["candidates"])
        assert "FLOPs by 44%" in result.report["survey"]
    else:
        final = result.report["evaluation"]["faithfulness"]["details"]
        assert final["supported_count"] == final["factual_claim_count"] > 0
        assert final["unverified_count"] == final["unsupported_count"] == 0
        assert result.journal["candidates"][-1]["committed"]
    assert not result.ledger["usage_uncertain"]
    assert all(
        entry["status"] == "settled" for entry in result.ledger["entries"].values()
    )


@pytest.mark.asyncio
async def test_separated_decisions_do_not_approve_wrong_paper_candidate(
    source_case, tmp_path, monkeypatch
):
    select_boundary(monkeypatch)
    result = await writer_view.run_capacity_case(
        source_case, tmp_path, monkeypatch, mode="wrong_paper"
    )
    assert_boundary(result)
    assert not result.report["delivery"]["publishable"]
    assert not any(row["committed"] for row in result.journal["candidates"])
    assert len(result.report["metadata"]["targeted_repair_state"]["attempts"]) == 2
    details = result.journal["candidates"][0]["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"]
    assert details["stages"]["global_check"] == "completed"
    assert details["unsupported_count"] > 0
