"""Run independent metrics through the real frozen writer and report consumers."""

import copy
import json

import pytest
import test_writer_source_view_pipeline as writer_view
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.eval.results import CONTRACT, EvaluationRunResult, selected


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [0, 2])
@pytest.mark.parametrize("reason_only", [False, True])
async def test_real_pipeline_keeps_shared_repairs_and_independent_metrics(
    source_case, tmp_path, monkeypatch, failures, reason_only
):
    original = writer_view.source_flags

    def flags(view=writer_view.VIEW):
        if reason_only:
            from litagent.closeout import PROFILE, profile_flags

            return {**profile_flags(PROFILE), "evaluation_result_contract": CONTRACT}
        return {**original(view), "evaluation_result_contract": CONTRACT}

    monkeypatch.setattr(writer_view, "source_flags", flags)
    if reason_only:
        from test_evaluation_dimensions_pipeline import separated_peer

        monkeypatch.setattr(writer_view, "conform_evaluation_replies", separated_peer)
    outcome = await writer_view.run_scientific_case(
        source_case, tmp_path / "run", monkeypatch, failures=failures
    )
    report = outcome.report
    assert selected(outcome.ledger["plan"]["envelope"])
    if reason_only:
        assert (
            outcome.ledger["plan"]["envelope"]["assessment_decision_contract"]
            == "reason-only-decision-v3"
        )
        assert outcome.gateway.dimension_replies
        assert all(
            "status" not in decision["evidence_check"]
            for reply in outcome.gateway.dimension_replies
            for decision in reply["decisions"]
        )
    assert report["evaluation"]["faithfulness"]["status"] in {"passed", "failed"}
    assert report["evaluation"]["internal_consistency"]["status"] == "passed"
    assert report["evaluation"]["internal_consistency"]["score"] == 1.0
    assert report["evaluation_checks"]
    initial = report["metadata"]["evaluation_history"]["initial_evaluation"]
    assert initial["faithfulness"]["score"] < 1.0
    from litagent.quality import strict_acceptance

    assert strict_acceptance(initial["faithfulness"]["details"])["status"] == "failed"
    assert initial["internal_consistency"]["status"] == "passed"
    assert report["delivery"]["publishable"] is (failures == 0)
    assert outcome.ledger["plan"]["repair_limit"] == 2
    # The full input reaches the real global constructor and provider peer.
    global_calls = [
        p
        for p in outcome.gateway.evaluator.base.calls
        if p.get("global_output_contract") == "host-global-completeness-v1"
    ]
    assert global_calls
    for payload in global_calls:
        assert payload["source_units"]
    for name in ("faithfulness", "internal_consistency", "citation_accuracy"):
        assert "status" in report["evaluation"][name]


@pytest.mark.asyncio
async def test_invalid_aspect_source_stops_before_global_or_prose_repair(
    source_case, tmp_path, monkeypatch
):
    """Treat an ignored native permission as protocol failure at the real decoder."""
    from test_evaluation_dimensions_pipeline import separated_peer

    from litagent.closeout import PROFILE, profile_flags

    monkeypatch.setattr(
        writer_view,
        "source_flags",
        lambda *_: {**profile_flags(PROFILE), "evaluation_result_contract": CONTRACT},
    )

    def adapter(gateway):
        gateway = separated_peer(gateway)
        real = gateway.evaluator.chat

        async def chat(messages, **options):
            response = await real(messages, **options)
            value = json.loads(response.content)
            if "decisions" in value:
                first = next(
                    row["evidence_check"]["aspect_checks"]
                    for row in value["decisions"]
                    if row["evidence_check"]["aspect_checks"]
                )
                next(iter(first.values()))["support_ids"].append(
                    "unknown-provider-passage"
                )
                response.content = json.dumps(value)
            return response

        gateway.evaluator.chat = chat
        return gateway

    monkeypatch.setattr(writer_view, "conform_evaluation_replies", adapter)
    outcome = await writer_view.run_scientific_case(source_case, tmp_path, monkeypatch)
    assert outcome.report["evaluation"]["faithfulness"]["score"] is None
    assert outcome.report["evaluation_checks"]["execution"]["status"] == "error"
    assert not outcome.report["delivery"]["publishable"]
    assert not outcome.journal["candidates"]
    assert not outcome.ledger["usage_uncertain"]
    assert not any(
        p.get("global_output_contract") == "host-global-completeness-v1"
        for p in outcome.gateway.evaluator.base.calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("persistent", [False, True])
async def test_atomicity_has_one_owner_and_recovery_preserves_prose_slot(
    source_case, tmp_path, monkeypatch, persistent
):
    import test_structured_repair_pipeline as structured
    from test_closeout_pipeline import RecheckGateway
    from test_evaluation_dimensions_pipeline import separated_peer

    from litagent.closeout import PROFILE, profile_flags

    monkeypatch.setattr(
        writer_view,
        "source_flags",
        lambda *_: {**profile_flags(PROFILE), "evaluation_result_contract": CONTRACT},
    )
    monkeypatch.setattr(writer_view, "conform_evaluation_replies", separated_peer)
    gateway = RecheckGateway(source_case, persistent=persistent)
    monkeypatch.setattr(
        structured, "LongScientificGateway", lambda *_args, **_kwargs: gateway
    )
    outcome = await writer_view.run_scientific_case(source_case, tmp_path, monkeypatch)
    assert selected(outcome.ledger["plan"]["envelope"])
    attempts = outcome.report["metadata"]["targeted_repair_state"]["attempts"]
    assert (
        sum(row.get("purpose") == "bounded-inventory-recovery-v2" for row in attempts)
        == 1
    )
    assert len(attempts) == (1 if persistent else 2)
    assert outcome.report["delivery"]["publishable"] is not persistent
    if persistent:
        assert outcome.report["evaluation"]["faithfulness"]["score"] is None
        assert outcome.report["evaluation_checks"]["inventory"]["status"] == "failed"
        assert not outcome.journal["candidates"]
    else:
        assert outcome.report["evaluation"]["faithfulness"]["score"] == 1.0
        assert outcome.journal["candidates"][-1]["committed"]


def test_actual_cli_freezes_new_contract_and_rejects_selector_drift(
    case, tmp_path, monkeypatch
):
    import sys

    from test_closeout_policy import (
        test_closeout_cli_reaches_actual_prepare_and_freezes_every_policy,
    )

    from scripts import run_frozen_writing as campaign

    original = campaign.main

    def main():
        sys.argv.extend(["--evaluation-result-contract", CONTRACT])
        return original()

    monkeypatch.setattr(campaign, "main", main)
    test_closeout_cli_reaches_actual_prepare_and_freezes_every_policy(
        case, tmp_path, monkeypatch
    )
    manifest = json.loads((tmp_path / "prepared/manifest.json").read_text())
    assert manifest["evaluation_result_contract"] == CONTRACT
    assert manifest["writing_parameters"]["evaluation_result_contract"] == CONTRACT
    assert all(selected(plan["envelope"]) for plan in manifest["budget_plans"].values())
    manifest["evaluation_result_identity"] = "0" * 64
    with pytest.raises(ValueError, match="results_configuration_invalid"):
        campaign.evaluation_results_options(manifest)


def test_maximum_legal_path_keeps_two_repairs_and_final_evaluation_budget(
    case, tmp_path, monkeypatch
):
    import test_joint_round_escrow as joint

    original = joint._base_plan

    def plan(*args, **kwargs):
        value = original(*args, evaluation_result_contract=CONTRACT, **kwargs)
        assert selected(value["envelope"])
        return value

    monkeypatch.setattr(joint, "_base_plan", plan)
    joint.test_joint_maximum_usage_funds_three_full_rounds_and_both_repairs(
        case, tmp_path
    )


def test_reason_only_wire_keeps_the_complete_maximum_resource_path(
    case, tmp_path, monkeypatch
):
    import test_joint_round_escrow as joint

    from litagent.closeout import PROFILE, profile_flags
    from litagent.eval.decision_output import selected as reason_selected
    from litagent.writing_v2 import writer_budget_plan

    flags = profile_flags(PROFILE)
    flags.pop("round_resource_contract")
    flags.pop("closeout_profile")

    def plan(packet, aliases, **kwargs):
        value = writer_budget_plan(
            packet,
            aliases,
            methodology={},
            budget_policy="planned-budget-v7",
            evaluation_result_contract=CONTRACT,
            **flags,
            **kwargs,
        )
        assert reason_selected(value["envelope"])
        return value

    monkeypatch.setattr(joint, "_base_plan", plan)
    joint.test_joint_maximum_usage_funds_three_full_rounds_and_both_repairs(
        case, tmp_path
    )


@pytest.mark.asyncio
async def test_malformed_analysis_keeps_science_but_stops_protocol_repairs(
    source_case, tmp_path, monkeypatch
):
    original_flags = writer_view.source_flags
    original_adapter = writer_view.conform_evaluation_replies

    def flags(view=writer_view.VIEW):
        return {**original_flags(view), "evaluation_result_contract": CONTRACT}

    def adapter(gateway):
        gateway = original_adapter(gateway)
        real = gateway.evaluator.chat

        async def chat(messages, **kwargs):
            response = await real(messages, **kwargs)
            value = json.loads(response.content)
            if "analyses" in value:
                value["analyses"] = "invalid component"
                response.content = json.dumps(value)
            return response

        gateway.evaluator.chat = chat
        return gateway

    monkeypatch.setattr(writer_view, "source_flags", flags)
    monkeypatch.setattr(writer_view, "conform_evaluation_replies", adapter)
    outcome = await writer_view.run_scientific_case(source_case, tmp_path, monkeypatch)
    metrics = outcome.report["evaluation"]
    assert metrics["faithfulness"]["score"] is not None
    assert metrics["internal_consistency"]["score"] == 1.0
    assert outcome.report["evaluation_checks"]["analysis"]["status"] == "error"
    assert outcome.report["evaluation_checks"]["execution"]["status"] == "error"
    assert not outcome.report["delivery"]["publishable"]
    assert not outcome.journal["candidates"]
