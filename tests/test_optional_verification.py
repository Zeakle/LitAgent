"""Exercise optional reviewers at their real generation and delivery consumers."""

import copy
import json
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from model_config_helpers import load_previous_default_config as load_config
from pydantic import ValidationError
from test_automatic_audit_integration import audit_api  # noqa: F401
from test_extraction_verification import selection
from test_scope_notice_contract import notice_document
from test_strict_automatic import frozen, passing_audit
from test_verified_extraction_freeze import ready

from litagent import api
from litagent.agents.synthesis import SynthesisWorker
from litagent.api import SurveyRequest
from litagent.contracts import normalize_survey_result
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.eval.base import CTX_EVIDENCE
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import build_fact_cards, digest
from litagent.runner import LitAgent
from litagent.workbench import RunOptions, VerificationOptions, verification_enabled


def switches(judge=False, jev=False):
    """Serialize the same strict contract sent by the new Survey dialog."""
    return VerificationOptions(judge=judge, jev=jev).model_dump(mode="json")


def optional_report(judge=False, jev=False):
    """Reuse complete scientific evaluation while keeping original source lineage."""
    old = frozen()
    report = copy.deepcopy(old["report"])
    report["metadata"].update(
        quality_policy="strict_auto_v2", verification=switches(judge, jev)
    )
    report["evaluation"]["faithfulness"]["details"]["quality_policy"] = "strict_auto_v2"
    return report, old["ledger"]


@pytest.mark.parametrize(
    "judge,jev", [(False, False), (True, False), (False, True), (True, True)]
)
def test_options_identity_and_task_propagation(judge, jev):
    options = RunOptions(
        quality_policy="strict_auto_v2", verification=switches(judge, jev)
    )
    assert not options.human_review
    assert options.auto_audit_mode == ("judge" if judge else "off")
    graph = TaskGraph()
    graph.add_task(SubTask("synthesis", "test", "synthesis"))
    LitAgent._apply_run_options_to_graph(graph, options)
    assert graph.tasks["synthesis"].input_data["verification"] == switches(judge, jev)
    assert verification_enabled(graph.tasks["synthesis"].input_data, "jev") is jev
    identities = {
        RunOptions(
            quality_policy="strict_auto_v2", verification=switches(j, e)
        ).fingerprint()
        for j in (False, True)
        for e in (False, True)
    }
    assert len(identities) == 4
    assert (
        options.fingerprint()
        != RunOptions(quality_policy="strict_auto_v2").fingerprint()
    )


@pytest.mark.parametrize(
    "value",
    [
        {"judge": "false"},
        {"jev": 0},
        {"judge": None},
        {"other": True},
        {"version": "unknown"},
    ],
)
def test_invalid_switches_are_not_coerced(value):
    with pytest.raises(ValidationError):
        SurveyRequest(
            query="topic", quality_policy="strict_auto_v2", verification=value
        )


def test_old_options_keep_their_fingerprints_and_mandatory_gates():
    before = {
        "legacy": "a66dbe38a4c2354825f756e5e219b6f8139716683617cfa8e6afd00b7b4ee75b",
        "strict_v1": "080b78d5390d2d222eefa22ee3eab19451b827e617da90e306b619ef52c7bd4a",
        "strict_auto_v1": "22cd987efe9e5e0aa1633e8354b5f951db4a542fb9cf3cf4d5f6faf9c362e7d7",
        "strict_auto_v2": "b659e1a6a9c783ae713de3713032dde97bc0c7cd58726d1c8474888376581760",
    }
    for policy, original in before.items():
        options = RunOptions(quality_policy=policy)
        assert options.fingerprint() == original
        assert verification_enabled(options.model_dump(), "judge")
        assert verification_enabled(options.model_dump(), "jev")
    with pytest.raises(ValidationError, match="optional_verification_requires"):
        SurveyRequest(query="topic", verification={})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "judge,jev", [(False, False), (True, False), (False, True), (True, True)]
)
async def test_actual_runner_accepts_only_enabled_gates(
    tmp_path, monkeypatch, judge, jev
):
    monkeypatch.chdir(tmp_path)
    if jev:
        report, ledger, _, _ = await ready(tmp_path / "verified", monkeypatch)
        report["metadata"]["verification"] = switches(judge, jev)
    else:
        report, ledger = optional_report(judge, jev)
        ledger["unused"] = {
            "paper_id": "unselected",
            "supporting_text": "Unselected original",
        }
    service = AsyncMock()
    service.submit.return_value = {"audit_id": "optional-audit"}

    async def audited(snapshot, **kwargs):
        assert kwargs == {"include_jev": False}
        service.wait.return_value = passing_audit(snapshot)
        return {"audit_id": "optional-audit"}

    service.submit.side_effect = audited
    agent = LitAgent(load_config(), audit_service=service)
    agent._session_id = "test-auto"
    agent._build_eval_context = lambda **kwargs: {CTX_EVIDENCE: ledger}
    with patch(
        "litagent.report_audit.ReportAuditService",
        side_effect=AssertionError("disabled service constructed"),
    ):
        await agent._automatic_review(report, "EViT", {}, None)
    accepted = report["metadata"]["automatic_acceptance"]
    assert report["delivery"]["publishable"], accepted
    assert accepted["gates"] == {
        "evaluation": "passed",
        "judge": "passed" if judge else "disabled",
        "extraction_jev": "passed" if jev else "disabled",
    }
    assert accepted["verification_sha256"] == digest(switches(judge, jev))
    assert service.submit.await_count == int(judge)
    assert service.wait.await_count == int(judge)
    service.shutdown.assert_not_awaited()


@pytest.mark.parametrize(
    "mutation", ["skipped", "unsupported", "missing_inventory", "cards", "source"]
)
def test_disabled_reviewers_do_not_bypass_evaluation_or_frozen_sources(mutation):
    report, ledger = optional_report()
    if mutation == "skipped":
        report["evaluation"]["faithfulness"]["skipped"] = True
    elif mutation == "unsupported":
        report["evaluation"]["faithfulness"]["details"]["assertions"][0][
            "status"
        ] = "unsupported"
    elif mutation == "missing_inventory":
        report["evaluation"]["faithfulness"]["details"]["assertions"] = []
    elif mutation == "cards":
        report["metadata"]["fact_cards"] = {}
    else:
        ledger["a"]["supporting_text"] = "A different experiment."
    with pytest.raises(ValueError):
        freeze_automatic_inputs(report, ledger, run_id="test-auto")


def test_enabled_judge_failure_still_blocks_and_option_changes_invalidate():
    report, ledger = optional_report(True, False)
    snapshot = freeze_automatic_inputs(report, ledger, run_id="test-auto")
    bad = passing_audit(snapshot)
    bad["judge"]["claim_assessments"][0]["supported"] = False
    assert not automatic_acceptance(snapshot, report["evaluation"], bad)["publishable"]
    report["metadata"]["automatic_acceptance"] = automatic_acceptance(
        snapshot, report["evaluation"], passing_audit(snapshot)
    )
    report["metadata"]["verification"]["jev"] = True
    value = normalize_survey_result(report, config_fingerprint="offline")
    assert value["metadata"]["automatic_acceptance"]["status"] == "invalidated"
    assert not value["delivery"]["publishable"]


@pytest.mark.asyncio
async def test_agent_does_not_create_an_external_judge_for_old_policy(
    tmp_path, monkeypatch
):
    """Only an explicitly supplied historical benchmark owner can run Judge."""
    report, ledger = optional_report(True, False)
    agent = LitAgent(load_config())
    agent._session_id = "test-auto"
    agent._build_eval_context = lambda **kwargs: {CTX_EVIDENCE: ledger}
    monkeypatch.chdir(tmp_path)
    with patch(
        "litagent.report_audit.ReportAuditService",
        side_effect=AssertionError("Agent constructed an external Judge"),
    ):
        await agent._automatic_review(report, "EViT", {}, None)
    acceptance = report["metadata"]["automatic_acceptance"]
    assert "automatic_report_audit_retired" in acceptance["failure_codes"]
    assert acceptance["publishable"] is False


@pytest.mark.asyncio
async def test_enabled_jev_without_receipt_blocks_even_when_judge_off():
    report, ledger = optional_report(False, True)
    with pytest.raises(ValueError, match="automatic_extraction_not_passed"):
        freeze_automatic_inputs(report, ledger, run_id="test-auto")
    acceptance = automatic_acceptance(
        {"report": report, "ledger": ledger}, report["evaluation"]
    )
    assert not acceptance["publishable"]
    assert acceptance["gates"]["judge"] == "disabled"
    assert "extraction_jev_not_passed" in acceptance["failure_codes"]


@pytest.mark.asyncio
async def test_synthesis_with_jev_off_preserves_cards_without_verifier_or_key():
    selected = selection()
    cards = build_fact_cards(selected.selected_items)
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(notice_document(cards)), model="offline", usage={}
    )
    selector = AsyncMock()
    selector.select.return_value = selected
    verifier = AsyncMock()
    verifier.verify.side_effect = AssertionError("disabled Jev called")
    worker = SynthesisWorker(
        llm, evidence_selector=selector, extraction_verifier=verifier
    )
    result = await worker.execute(
        SubTask(
            "synthesis",
            "test",
            "synthesis",
            input_data={
                "query": "EViT",
                "quality_policy": "strict_auto_v2",
                "review_structure": "academic_v2",
                "verification": switches(),
            },
        )
    )
    verifier.verify.assert_not_awaited()
    assert result["verification"] == switches()
    assert result["extraction_verification"] is None
    assert result["fact_cards"] == cards
    assert set(result["evidence_selection"]["selected_items"]) == {"a", "b"}


@pytest.mark.asyncio
@pytest.mark.parametrize("jev", [False, True])
async def test_api_accepts_explicit_extraction_jev_switch(audit_api, jev):
    """The Jev switch affects evidence admission without enabling a report audit."""
    _, create, _ = audit_api
    verification = {**switches(jev=jev), "version": "optional-verification-v2"}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/survey",
            json={
                "query": "question",
                "quality_policy": "strict_auto_v2",
                "verification": verification,
            },
            headers={"X-LitAgent-Operation-Token": api.app.state.operation_token},
        )
    assert response.status_code == 202, response.text
    options = create.call_args.kwargs["options"]
    assert options.verification.jev is jev
    assert options.auto_audit_mode == "off"


@pytest.mark.asyncio
async def test_missing_jev_key_rejects_default_before_generation(
    audit_api, monkeypatch
):
    """The default cannot silently proceed with unverified evidence."""
    _, create, _ = audit_api
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        body = {"query": "question", "quality_policy": "strict_auto_v2"}
        headers = {"X-LitAgent-Operation-Token": api.app.state.operation_token}
        response = await client.post("/survey", json=body, headers=headers)
        assert response.status_code == 409
        assert response.json()["detail"] == "missing_key:TYPESAFE_API_KEY"
        create.assert_not_called()
        off = await client.post(
            "/survey", json={**body, "verification": {"jev": False}}
        )
        assert off.status_code == 202


@pytest.mark.asyncio
async def test_retired_post_selection_jev_version_rejects_new_run(audit_api):
    """New Jev requests must check claims before ledger admission."""
    _, create, _ = audit_api
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/survey",
            json={
                "query": "question",
                "quality_policy": "strict_auto_v2",
                "verification": switches(jev=True),
            },
        )
    assert response.status_code == 422
    assert response.json()["detail"] == "extraction_admission_v2_required"
    create.assert_not_called()
