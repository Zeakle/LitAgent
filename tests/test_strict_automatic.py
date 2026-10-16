"""Test automatic gates and matrix execution without paid transports."""

import asyncio
import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from test_batched_evaluation import BatchLLM
from test_strict_quality import valid_details

from litagent.benchmark.citation_audit import build_citation_audit_inputs
from litagent.benchmark.datasets import load_survey_dataset, load_survey_profiles
from litagent.benchmark.quality_suite import QualitySuiteRunner, public_projection
from litagent.benchmark.survey_judge import SurveyBenchmarkJudge
from litagent.benchmark.survey_models import SurveyJudgeConfig
from litagent.config import load_config
from litagent.contracts import build_config_summary, normalize_survey_result
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.observability.recorder import ArchiveRepository
from litagent.quality import build_fact_cards, digest
from litagent.report_audit import digest as compact_digest
from litagent.report_render import build_report_source_units
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions, is_strict_policy


def frozen():
    """Create a faithfully located fixture with a misleading compressed extraction."""
    survey = "EViT improves speed by 50% on DeiT-S. [E:a]"
    ledger = {
        "a": {
            "paper_id": "evit",
            "evidence_id": "a",
            "text": "Evo-ViT improves FLOPs.",
            "supporting_text": "EViT improves speed by 50% on DeiT-S.",
            "chunk_key": "abstract",
        }
    }
    cards = build_fact_cards(ledger)
    unit = build_report_source_units(survey)[0]
    details = valid_details()
    details["quality_policy"] = "strict_auto_v1"
    details["assertions"][0].update(
        claim_text="EViT improves speed by 50% on DeiT-S.",
        source_unit_id=unit.unit_id,
        source_fragment_id=f"{unit.unit_id}:fragment:0",
        source_quote=survey,
        source_start=0,
        source_end=len(survey),
        evidence_ids=["a"],
        local_error=None,
    )
    report = {
        "survey": survey,
        "quality": {"status": "passed"},
        "metadata": {
            "quality_policy": "strict_auto_v1",
            "query": "EViT speed",
            "review_structure": "legacy",
            "report_language": "en",
            "fact_cards": cards,
            "evidence_snapshot_sha256": digest(cards),
        },
        "evaluation": {
            "citation_accuracy": {"passed": True},
            "internal_consistency": {"passed": True},
            "faithfulness": {"passed": True, "details": details},
        },
    }
    return freeze_automatic_inputs(report, ledger, run_id="test-auto")


def passing_audit(snapshot):
    """Keep individual citations and the claim set as separate denominators."""
    inventory = snapshot["frozen_inventory"]["inventory"]
    return {
        **{
            k: snapshot[k]
            for k in ("source_run_id", "source_report_revision", "source_report_sha256")
        },
        "ledger_sha256": compact_digest(snapshot["ledger"]),
        "source_snapshot_sha256": compact_digest(snapshot),
        "judge": {
            "status": "completed",
            "inventory_snapshot": inventory,
            "faithfulness": 1.0,
            "contradiction_rate": 0,
            "claim_assessments": [
                {"claim_id": "c1", "supported": True, "source_faithful": True}
            ],
        },
        "judge_full_report_check": {"status": "passed"},
        "judge_usage": {"usage_complete": True},
        "jev": {
            "inventory_sha256": inventory["inventory_sha256"],
            "ledger_sha256": inventory["ledger_sha256"],
            "source_report_sha256": snapshot["source_report_sha256"],
            "status": "completed",
            "actual_model": "jev-1.13.0",
            "usage": {"usage_complete": True},
            "inputs": [{"claim_id": "c1", "status": "ready"}],
            "decisions": [
                {
                    "question_id": "g:c1",
                    "choice": "supported",
                    "confidence": 0.95,
                    "uncertain": False,
                    "status": "valid",
                    "evidence_id": None,
                },
                {
                    "question_id": "m:c1:0",
                    "choice": "contributes",
                    "confidence": 0.95,
                    "uncertain": False,
                    "status": "valid",
                    "evidence_id": "a",
                },
            ],
        },
    }


def test_policy_enforces_no_human_and_preserves_legacy_identity():
    old = RunOptions()
    assert old.fingerprint() == RunOptions(quality_policy="legacy").fingerprint()
    automatic = RunOptions(
        quality_policy="strict_auto_v1",
        human_review=True,
        auto_audit_mode="off",
        auto_audit_max_versions=3,
    )
    assert (
        not automatic.human_review
        and automatic.auto_audit_mode == "judge_jev"
        and automatic.auto_audit_max_versions == 1
    )
    assert (
        is_strict_policy(automatic.quality_policy)
        and automatic.review_structure == "academic_v2"
    )
    assert RunOptions(quality_policy="strict_v1", human_review=False).human_review
    assert automatic.report_language == "zh-CN"
    assert (
        RunOptions(
            quality_policy="strict_auto_v1", report_language="en"
        ).report_language
        == "en"
    )
    from litagent.api import SurveyRequest

    request = SurveyRequest(query="topic", quality_policy="strict_auto_v1")
    assert request.report_language == "zh-CN" and request.citation_style == "ieee"
    old_request = SurveyRequest(query="topic")
    assert (
        old_request.report_language == "en" and old_request.citation_style == "numbered"
    )
    cfg = load_config()
    assert (
        build_config_summary(cfg, "strict_auto_v1")["fingerprint"]
        != build_config_summary(cfg, "strict_v1")["fingerprint"]
    )


@pytest.mark.asyncio
async def test_unknown_usage_stops_next_transport_and_survives_restore():
    """A cancelled sent request must stop later phases without resetting charges."""
    from litagent.llm.client import OpenAICompatibleClient

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = object.__new__(OpenAICompatibleClient)
    client._cost_budget = budget
    client._transport_max_attempts = 3
    client._max_tokens = 1
    sent = AsyncMock(side_effect=asyncio.CancelledError())
    client._chat_attempt = sent
    with pytest.raises(asyncio.CancelledError):
        await client.chat([{"role": "user", "content": "first"}])
    snapshot = budget.snapshot()
    assert snapshot["usage_uncertain"] and snapshot["requests_sent"] == 1
    assert budget.is_exceeded() and budget.remaining() == 0
    with pytest.raises(ValueError, match="strict_usage_unknown"):
        await client.chat([{"role": "user", "content": "next phase"}])
    assert sent.await_count == 1 and budget.snapshot() == snapshot
    restored = CostBudget.from_snapshot(snapshot)
    assert restored.is_exceeded() and restored.remaining() == 0
    with pytest.raises(ValueError, match="strict_usage_unknown"):
        restored.begin_request()
    assert restored.snapshot() == snapshot


def test_existing_strict_budget_snapshot_keeps_old_semantics():
    budget = CostBudget()
    budget.configure_strict()
    budget.charge_unknown_usage(100)
    assert "halt_on_unknown_usage" not in budget.snapshot()
    assert not budget.is_exceeded() and budget.remaining() == 199900


@pytest.mark.asyncio
async def test_automatic_worker_failure_stops_queued_siblings_and_followup():
    """Invalid worker output stops new dispatch while preserving known usage."""
    from litagent.orchestrator.scheduler import Scheduler, Worker
    from litagent.orchestrator.task_graph import SubTask, TaskGraph, TaskStatus

    attempted = []

    class InvalidWorker(Worker):
        agent_type = "invalid"

        async def execute(self, task):
            attempted.append(task.task_id)
            raise ValueError("grounded_report_contract_invalid")

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    graph = TaskGraph()
    for task_id in ("first", "sibling"):
        graph.add_task(
            SubTask(
                task_id=task_id,
                description="invalid response",
                agent_type="invalid",
                max_retries=0,
            )
        )
    scheduler = Scheduler([InvalidWorker()], cost_budget=budget, max_concurrent=1)
    await scheduler.run(graph)
    assert attempted == ["first"]
    assert graph.get_task("first").status == TaskStatus.FAILED
    assert budget.stop_reason == "strict_worker_failed"
    assert budget.used == 0 and not budget.snapshot()["usage_uncertain"]
    restored = CostBudget.from_snapshot(budget.snapshot())
    with pytest.raises(ValueError, match="strict_worker_failed"):
        restored.reserve(1)


def test_freeze_and_original_text_do_not_use_compressed_claim():
    snapshot = frozen()
    from litagent.benchmark.survey_models import JudgeInventorySnapshot

    inputs = build_citation_audit_inputs(
        JudgeInventorySnapshot.model_validate(
            snapshot["frozen_inventory"]["inventory"]
        ),
        snapshot["ledger"],
        artifact=snapshot["artifact"],
    )
    assert inputs[0].evidence[0].text == snapshot["ledger"]["a"]["supporting_text"]
    assert "DeiT-S" in inputs[0].evidence[0].text
    damaged = copy.deepcopy(snapshot)
    damaged["report"]["evaluation"]["faithfulness"]["details"]["assertions"][0][
        "source_end"
    ] -= 1
    with pytest.raises(ValueError, match="fidelity"):
        freeze_automatic_inputs(
            damaged["report"], damaged["ledger"], run_id="test-auto"
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "low",
        "unsupported",
        "unrelated",
        "missing",
        "duplicate",
        "stale",
        "model",
        "global",
        "usage",
        "judge_failed",
    ],
)
def test_completed_semantic_failures_never_deliver(mutation):
    snapshot = frozen()
    audit = passing_audit(snapshot)
    assert automatic_acceptance(snapshot, snapshot["report"]["evaluation"], audit)[
        "publishable"
    ]
    if mutation == "low":
        audit["jev"]["decisions"][0]["confidence"] = 0.89
    if mutation == "unsupported":
        audit["jev"]["decisions"][0]["choice"] = "insufficient"
    if mutation == "unrelated":
        audit["jev"]["decisions"][1]["choice"] = "unrelated"
    if mutation == "missing":
        audit["jev"]["decisions"].pop()
    if mutation == "duplicate":
        audit["jev"]["decisions"][1] = audit["jev"]["decisions"][0]
    if mutation == "stale":
        audit["source_report_sha256"] = "0" * 64
    if mutation == "model":
        audit["jev"]["actual_model"] = "jev-1.14"
    if mutation == "global":
        audit["judge_full_report_check"]["status"] = "failed"
    if mutation == "usage":
        audit["judge_usage"]["usage_complete"] = False
    if mutation == "judge_failed":
        audit["judge"]["claim_assessments"][0]["supported"] = False
    result = automatic_acceptance(snapshot, snapshot["report"]["evaluation"], audit)
    assert (
        result["status"] == "failed"
        and not result["publishable"]
        and result["actor"] == "system"
    )


@pytest.mark.asyncio
async def test_frozen_judge_two_calls_original_support_and_total_reservation():
    snapshot = frozen()
    llm = BatchLLM()
    state = {}
    judge, usage = await SurveyBenchmarkJudge(
        llm, SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline")
    ).evaluate(
        case=__import__(
            "litagent.audit_context", fromlist=["SurveyAuditContext"]
        ).SurveyAuditContext.model_validate(snapshot["context"]),
        survey=snapshot["report"]["survey"],
        ledger=snapshot["ledger"],
        frozen_inventory=snapshot["frozen_inventory"],
        checkpoint=state,
    )
    assert judge.status == "completed" and judge.faithfulness == 1
    assert (
        usage.call_count == 2
        and state["automatic_full_report_check"]["status"] == "passed"
    )
    assert "Evo-ViT" not in json.dumps(llm.calls[0]["cited_materials"])
    blocked = BatchLLM()
    result, cost = await SurveyBenchmarkJudge(
        blocked,
        SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline"),
    ).evaluate(
        case=__import__(
            "litagent.audit_context", fromlist=["SurveyAuditContext"]
        ).SurveyAuditContext.model_validate(snapshot["context"]),
        survey=snapshot["report"]["survey"],
        ledger=snapshot["ledger"],
        frozen_inventory=snapshot["frozen_inventory"],
        total_budget_tokens=100,
    )
    assert result.status == "failed" and not blocked.calls and cost.call_count == 0
    with pytest.raises(ValueError, match="no_replay"):
        await SurveyBenchmarkJudge(
            llm,
            SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline"),
        ).evaluate(
            case=__import__(
                "litagent.audit_context", fromlist=["SurveyAuditContext"]
            ).SurveyAuditContext.model_validate(snapshot["context"]),
            survey=snapshot["report"]["survey"],
            ledger=snapshot["ledger"],
            frozen_inventory=snapshot["frozen_inventory"],
            checkpoint=state,
        )


def suite(tmp_path, executor, quality_policy="strict_auto_v1"):
    return QualitySuiteRunner(
        load_config(),
        load_survey_dataset(Path("benchmarks/survey/dataset.yaml")),
        next(
            p
            for p in load_survey_profiles(Path("benchmarks/survey/profiles.yaml"))
            if p.profile_id == "external-fulltext"
        ),
        root=tmp_path,
        run_archive=ArchiveRepository(tmp_path / "runs"),
        suite_id="offline-auto",
        executor=executor,
        quality_policy=quality_policy,
    )


def observation(passed=True, complete=True, technical=None):
    return {
        "status": "delivered" if passed else "blocked",
        "acceptance": {"publishable": passed},
        "complete_verification": complete,
        "costs": {
            k: {"tokens": 10, "requests": 1, "usage_complete": True}
            for k in ("survey", "judge", "jev")
        },
        "technical_failure": technical,
        "private": "secret-private-body",
    }


@pytest.mark.asyncio
async def test_fixed_matrix_science_failure_retained_and_replay_free(tmp_path):
    executor = AsyncMock(side_effect=[observation(False)] + [observation()] * 17)
    runner = suite(tmp_path, executor)
    result = await runner.run()
    assert (
        len(result["observations"]) == 18
        and result["counts"]["formal"]["automatic_passed"] == 15
    )
    assert result["counts"]["pilot"]["blocked"] == 1 and result["counts"]["goal_met"]
    assert executor.await_count == 18
    again = await runner.run()
    assert again == result and executor.await_count == 18
    public = public_projection(result)
    assert "secret-private-body" not in json.dumps(public)
    assert public["service_usage"]["survey"]["known_requests"] == 18
    bundle = tmp_path / "presentation" / runner.suite_id
    assert all(
        (bundle / f"benchmark.{ext}").is_file() for ext in ("json", "csv", "md", "html")
    )
    import csv

    rows = list(csv.DictReader((bundle / "benchmark.csv").open()))
    assert len(rows) == 18 and rows[0]["survey_requests"] == "1"
    assert rows[0]["survey_usage_complete"] == "True"
    assert "人工校准未验证" in (bundle / "benchmark.md").read_text()


@pytest.mark.asyncio
async def test_technical_failure_stops_and_preserves_formal_denominator(tmp_path):
    executor = AsyncMock(return_value=observation(False, technical="auth_failed"))
    result = await suite(tmp_path, executor).run()
    assert executor.await_count == 1 and result["status"] == "stopped"
    assert (
        result["counts"]["formal"]["not_run"] == 15
        and result["counts"]["formal"]["pass_rate_planned"] == 0
    )


@pytest.mark.asyncio
async def test_unknown_tokens_preserve_known_request_count_and_full_reservation(
    tmp_path,
):
    partial = observation(False)
    partial["costs"]["survey"].update(usage_complete=False, requests=2)
    executor = AsyncMock(return_value=partial)
    runner = suite(tmp_path, executor)
    result = await runner.run()
    assert result["stop_reasons"] == ["quality_usage_unknown"]
    assert result["budgets"]["survey"]["charged"] == 500000
    assert result["budgets"]["survey"]["requests"] == 2
    assert await runner.run() == result and executor.await_count == 1


@pytest.mark.asyncio
async def test_pilot_no_judge_jev_contract_does_not_run_full_suite(tmp_path):
    executor = AsyncMock(return_value=observation(False, complete=False))
    result = await suite(tmp_path, executor).run()
    assert executor.await_count == 3 and result["status"] == "stopped"


@pytest.mark.asyncio
async def test_suite_interruption_charges_reservation_without_replay(tmp_path):
    executor = AsyncMock(side_effect=asyncio.CancelledError())
    runner = suite(tmp_path, executor)
    with pytest.raises(asyncio.CancelledError):
        await runner.run()
    result = await runner.run()
    assert (
        executor.await_count == 1 and result["budgets"]["survey"]["charged"] == 500000
    )


def test_changed_body_or_cards_invalidates_only_automatic_result():
    snapshot = frozen()
    acceptance = automatic_acceptance(
        snapshot, snapshot["report"]["evaluation"], passing_audit(snapshot)
    )
    report = copy.deepcopy(snapshot["report"])
    report["metadata"]["automatic_acceptance"] = acceptance
    report["delivery"] = {"publishable": True, "status": "ready"}
    assert normalize_survey_result(report, config_fingerprint="test")["delivery"][
        "publishable"
    ]
    report["survey"] += " Unverified extra assertion."
    output = normalize_survey_result(report, config_fingerprint="test")
    assert (
        not output["delivery"]["publishable"]
        and output["metadata"]["automatic_acceptance"]["status"] == "invalidated"
    )
    assert output["evaluation"] == report["evaluation"]


@pytest.mark.asyncio
async def test_runner_finalization_keeps_system_record_and_no_human_request(
    tmp_path, monkeypatch
):
    from litagent.eval.base import CTX_EVIDENCE
    from litagent.runner import LitAgent

    snapshot = frozen()
    report = copy.deepcopy(snapshot["report"])
    service = AsyncMock()
    service.submit.return_value = {"audit_id": "offline"}

    async def finish(_):
        submitted = service.submit.await_args.args[0]
        return passing_audit(submitted)

    service.wait.side_effect = finish
    agent = LitAgent(load_config(), audit_service=service)
    agent._session_id = "test-auto"
    agent._build_eval_context = lambda **kw: {CTX_EVIDENCE: snapshot["ledger"]}
    monkeypatch.chdir(tmp_path)
    await agent._automatic_review(report, "EViT speed", {}, None)
    assert report["delivery"]["publishable"]
    assert report["metadata"]["automatic_acceptance"]["actor"] == "system"
    assert "human_review" not in report["metadata"]
    normalize_survey_result(report, config_fingerprint="offline")
