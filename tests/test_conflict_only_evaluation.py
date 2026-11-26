"""Exercise the three-metric evaluation contract and its strict delivery gate."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_native_ragas_evaluation import SemanticTransport, native_case  # noqa: F401
from test_quality_view import render_repair_state
from test_writer_v2 import case  # noqa: F401

from litagent.config import EvalConfig, load_config
from litagent.eval import conflict_review, ragas_native
from litagent.eval.automatic import automatic_acceptance
from litagent.eval.citation import CitationEvaluator
from litagent.eval.engine_policy import (
    DIRECT_INVENTORY_PROMPT,
    STAGE_BATCH_LIMITS,
    configured_engine,
    engine_descriptor,
    native_writer_budget_plan,
    validate_engine,
)
from litagent.eval.ragas_adapter import planned_batches
from litagent.eval.ragas_repair import native_repair_diagnostics
from litagent.eval.results import acceptance, derive_quality, public_summary, result
from litagent.output_capacity import configured_output_tokens
from litagent.quality import build_fact_cards, digest
from litagent.report_render import build_report_source_units
from litagent.run_limits import configured_run_limits
from litagent.runner import LitAgent
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.safety.planned_budget import PlannedBudget
from litagent.workbench import VerificationOptions


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["clear", "unsupported", "conflict", "invalid"])
async def test_native_scope_only_checks_contradictions(
    native_case, monkeypatch, tmp_path, mode
):
    """Use real RAGAS inventory and support consumers and replace only model IO."""
    survey, context = native_case
    context.update(
        structure_checks={"status": "failed"},
        method_checks={"status": "failed"},
        language_checks={"status": "failed"},
    )
    monkeypatch.setattr(
        ragas_native,
        "locked_writer_envelope",
        lambda _: {
            "semantic_engine": engine_descriptor(global_scope=conflict_review.CONTRACT)
        },
    )

    class Transport(SemanticTransport):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "jobs" in payload and "context_pool" not in payload:
                assert (
                    "Extract statements only from this job's answer"
                    in messages[0]["content"]
                )
                assert "For NLI" not in messages[0]["content"]
                assert all(
                    set(job) == {"question", "answer"}
                    for job in payload["jobs"].values()
                )
            if payload.get("global_scope") == conflict_review.CONTRACT:
                schema = kwargs["response_format"]["json_schema"]["schema"]
                assert set(schema["properties"]) == {"conflicts"}
                assert set(payload) == {"global_scope", "report_passages", "claims"}
                if mode == "conflict":
                    first, second = list(payload["claims"])[:2]
                    response.content = json.dumps(
                        {
                            "conflicts": [
                                {
                                    "claim_a": first,
                                    "claim_b": second,
                                    "reason": "Same method and setting conflict.",
                                }
                            ]
                        }
                    )
                if mode == "invalid":
                    response.content = '{"conflicts": [], "issues": []}'
            return response

    llm = Transport(
        reject="Method A learns an embedding." if mode == "unsupported" else None
    )
    run = await ragas_native.execute_native(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    evaluation = {"citation_accuracy": result("passed", score=1), **run.metrics()}
    details = evaluation["faithfulness"]["details"]
    assert details["stages"]["inventory_prompt_contract"] == DIRECT_INVENTORY_PROMPT
    assert details["stages"]["stage_batch_contract"] == STAGE_BATCH_LIMITS
    assert len(llm.calls) == 3
    assert set(run.checks()) == {"execution", "inventory"}
    assert (
        not {
            "core_coverage",
            "analysis_check",
            "limitations_check",
            "structure_checks",
            "method_checks",
            "language_checks",
        }
        & details.keys()
    )
    gate = derive_quality(evaluation)
    assert (gate["status"] == "passed") == (mode == "clear")
    if mode == "conflict":
        assert evaluation["faithfulness"]["score"] == 1
        assert evaluation["internal_consistency"]["score"] == 0.8
        assert gate["strict_acceptance"]["failure_codes"] == [
            "internal_consistency_not_passed"
        ]
        assert [item["reason_code"] for item in native_repair_diagnostics(details)] == [
            "evaluation_report_contradiction"
        ] * 2
    if mode == "unsupported":
        assert evaluation["faithfulness"]["score"] == 6 / 7
        assert evaluation["internal_consistency"]["score"] == 1
        assert gate["strict_acceptance"]["failure_codes"] == [
            "science_not_fully_supported"
        ]
        report = {
            "survey": survey,
            "evaluation": evaluation,
            "quality": gate,
            "metadata": {
                "quality_policy": "strict_auto_v2",
                "verification": VerificationOptions().model_dump(),
            },
        }
        metadata = report["metadata"]
        agent = LitAgent(load_config())
        monkeypatch.chdir(tmp_path)
        await agent._automatic_review(report, "Compare the two methods.", {}, None)

        automatic = metadata["automatic_acceptance"]
        assert automatic["inventory_sha256"] is None
        assert automatic["gates"]["evaluation"] == "failed"
        assert report["delivery"] == {
            "status": "blocked",
            "publishable": False,
            "reason_codes": [
                "evaluation_not_passed",
                "automatic_evaluation_not_passed",
            ],
        }
        assert run.checks()["inventory"]["status"] == "passed"
        assert {
            "automatic_global_scope_mismatch",
            "automatic_inventory_missing",
        }.issubset(
            automatic_acceptance(
                {"report": report, "frozen_inventory": {}}, evaluation
            )["failure_codes"]
        )
    if mode == "invalid":
        assert evaluation["internal_consistency"]["score"] is None
        assert "incomplete_verification" in gate["strict_acceptance"]["failure_codes"]
    projection = public_summary({"survey": survey, "evaluation": evaluation})
    assert set(projection["metrics"]) == {
        "citation_accuracy",
        "faithfulness",
        "internal_consistency",
    }
    assert set(projection["checks"]) == {"execution", "inventory"}


def test_direct_inventory_keeps_existing_wire_identity_read_only():
    """Keep the previously settled profile readable without using its old prompt."""
    config = load_config("config/agent-format-repair.yaml")
    current = configured_engine(config)
    assert current["inventory_prompt_contract"] == DIRECT_INVENTORY_PROMPT
    assert current["stage_batch_contract"] == STAGE_BATCH_LIMITS
    assert validate_engine(current) == current
    previous_split = {
        **current,
        "identity": "82bda69d7b8df2a45381b5c29ccdd6db4c089304cd83ea6c0b164bc1a0f1ef22",
    }
    previous_split.pop("stage_batch_contract")
    old = {
        **previous_split,
        "identity": "9c6545ada24c173adbd22f5b8b174a064fc7fc5e02a9aed783f88fe8f11ed405",
    }
    old.pop("inventory_prompt_contract")
    for archived in (old, previous_split):
        assert validate_engine(archived, archived=True) == archived
        with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
            validate_engine(archived)
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(
            {**old, "inventory_prompt_contract": DIRECT_INVENTORY_PROMPT},
            archived=True,
        )
    with pytest.raises(ValueError, match="ragas_engine_identity_invalid"):
        validate_engine(
            {**previous_split, "stage_batch_contract": STAGE_BATCH_LIMITS},
            archived=True,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("oversized_single", [False, True])
async def test_stage_batch_limit_matches_admission_and_send(
    native_case, case, oversized_single
):
    """Rebatch at the assessment cap before reserving or sending to the provider."""
    cfg = load_config("config/agent-format-repair.yaml")
    plan = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        semantic_engine=configured_engine(cfg),
        capacity=cfg.capacity,
        methodology={},
        output_tokens=configured_output_tokens(cfg),
        run_limits=configured_run_limits(cfg),
    )
    assert plan["status"] == "admitted"
    cap = plan["stages"]["assessment"]["request_cap"]
    survey, context = native_case
    parts = survey.split("[E:e1]")
    survey = (
        "".join(
            f"{part}[E:e{1 if index < 3 else 2}]"
            for index, part in enumerate(parts[:-1])
        )
        + parts[-1]
    )
    evidence = context["evidence"]
    evidence["e2"] = {
        **evidence["e1"],
        "evidence_id": "e2",
        "paper_id": "paper:B",
        "supporting_text": "b" * 90000,
    }
    evidence["e1"]["supporting_text"] = "a" * (177000 if oversized_single else 90000)
    context["fact_cards"] = build_fact_cards(evidence)
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    *_, samples = ragas_native.source_samples(survey, context)

    class Transport(SemanticTransport):
        async def chat(self, messages, **kwargs):
            payload = json.loads(messages[-1]["content"])
            if "context_pool" in payload:
                amount = strict_request_reservation(
                    messages, cfg.eval.max_tokens, kwargs["response_format"]
                )
                assert amount <= cap
                sent.append((list(payload["jobs"]), amount))
            return await super().chat(messages, **kwargs)

    sent = []
    llm = Transport()
    llm._cost_budget = PlannedBudget(plan)
    if not oversized_single:
        for sample in samples:
            sample["statements"] = [sample["response"]]
        (before,) = planned_batches(
            "assessment", samples, cfg.eval, 20, 5, keyed_assessment=True
        )
        assert cap < before[3] <= plan["stages"]["global"]["request_cap"]
        expected = planned_batches(
            "assessment", samples, cfg.eval, 20, 5, keyed_assessment=True, limit=cap
        )
        assert len(expected) == 2

    run = await ragas_native.execute_native(
        llm=llm, survey=survey, context=context, config=cfg.eval
    )
    admission = run.details["stages"]["capacity_admission"]
    if oversized_single:
        assert not sent and len(llm.calls) == 1
        assert admission["rejected_stage"] == "assessment"
        assert admission["reason"] == "ragas_request_context_exceeded"
        assert run.metrics()["faithfulness"]["score"] is None
        return
    assert sent == [(ids, tokens) for ids, _, _, tokens in expected]
    assert admission["assessment"]["batch_reservations"] == [
        amount for _, amount in sent
    ]
    assert admission["assessment"]["context_limit"] == cap
    assert (
        admission["global"]["context_limit"] == plan["stages"]["global"]["request_cap"]
    )
    assert run.metrics()["faithfulness"]["score"] == 1.0
    assert len(run.assertions) == len(samples)
    assert run.details["coverage"]["complete"]
    assert validate_accounting_snapshot(llm._cost_budget.snapshot(), expected_plan=plan)


def test_direct_inventory_does_not_change_nli_or_job_schema():
    """Freeze the same jobs and schema while isolating the inventory instruction."""
    config = EvalConfig()
    samples = [
        {
            "id": "u0",
            "question": "Compare two methods.",
            "response": "A computes distances when labels are sparse.",
            "contexts": ["Original evidence."],
            "statements": ["A computes distances when labels are sparse."],
        }
    ]
    for phase in ("inventory", "assessment"):
        (old,) = planned_batches(phase, samples, config, 16, 1, keyed_assessment=True)
        (new,) = planned_batches(
            phase, samples, config, 16, 1, keyed_assessment=True, direct_inventory=True
        )
        if phase == "inventory":
            assert old[0] == new[0] and old[2] == new[2]
            assert old[1][1] == new[1][1]
            assert "For NLI" in old[1][0]["content"]
            assert "For NLI" not in new[1][0]["content"]
        else:
            assert old == new


def test_conflict_request_and_parser_preserve_all_science_and_reject_false_ids():
    """Preflight sees the same contradictory candidates as the actual request."""
    units = [SimpleNamespace(unit_id="u1", text="Complete original report passage.")]
    claims = [
        {
            "claim_id": f"claim_{i}",
            "claim_text": f"Statement {i}",
            "kind": "scientific",
            "source_unit_id": "u1",
            "status": "unverifiable",
        }
        for i in range(3)
    ]
    before = conflict_review.build_request(units, claims)
    for row, status in zip(
        claims, ("supported", "unsupported", "supported"), strict=True
    ):
        row["status"] = status
    after = conflict_review.build_request(units, claims)
    assert before == after
    payload = before[0]
    assert list(payload["claims"]) == [row["claim_id"] for row in claims]
    pair = {
        "claim_a": "claim_0",
        "claim_b": "claim_1",
        "reason": "Incompatible results.",
    }
    assert conflict_review.parse_response(
        json.dumps({"conflicts": [pair]}), payload
    ) == [pair]
    for invalid in (
        [{**pair, "claim_b": "unknown"}],
        [{**pair, "claim_b": "claim_0"}],
        [pair, {**pair, "claim_a": "claim_1", "claim_b": "claim_0"}],
        [{**pair, "reason": "  "}],
    ):
        with pytest.raises(ValueError, match="evaluation_conflict"):
            conflict_review.parse_response(json.dumps({"conflicts": invalid}), payload)


def test_unknown_reference_blocks_strict_citation_ids_even_above_old_threshold():
    """The three-metric delivery policy requires every cited ID to exist."""
    evaluator = CitationEvaluator(None)
    ledger = {f"e{i}": {} for i in range(10)}
    text = " ".join(f"[E:e{i}]" for i in range(11))
    assert evaluator.evaluate_ids(text, ledger).passed
    invalid = evaluator.evaluate_ids(text, ledger, strict=True)
    assert invalid.score == 10 / 11 and not invalid.passed
    assert invalid.details["unknown_evidence_ids"] == ["e10"]


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_three_metrics_quality_view_hides_additional_report_judgments(locale):
    """Render the real workbench projection without a report issue table."""
    details = {
        "evaluation_version": "survey-evaluation-ragas-v1",
        "evaluation_result_contract": "independent-evaluation-v1",
        "body_analysis_contract": "body-analysis-v1",
        "stages": {
            "global_scope": conflict_review.CONTRACT,
            "global_check": "completed",
        },
        "evaluation_checks": {
            "execution": {"status": "passed"},
            "inventory": {"status": "passed"},
        },
    }
    view = render_repair_state(
        {
            "evaluation": {
                "citation_accuracy": {"score": 1, "status": "passed"},
                "faithfulness": {"score": 1, "status": "passed", "details": details},
                "internal_consistency": {"score": 1, "status": "passed"},
            }
        },
        locale,
    )
    assert view["metricStatuses"] == ["passed"] * 3
    assert ("矛盾核验" if locale == "zh-CN" else "Contradiction check") in view["text"]
    for label in (
        "章节分析",
        "全局正文问题",
        "核心问题覆盖",
        "Chapter analysis",
        "Global report issues",
        "Core coverage",
        "analytical depth",
    ):
        assert label not in view["text"]


def test_real_fifth_candidate_is_read_only_and_new_request_keeps_478_claims():
    """Read the real previous failure with its old policy, then rebuild a new request."""
    base = Path("artifacts/targeted-repairs/1225f3fe0282.json")
    writer, accounting = (
        Path(f"{base}.{suffix}.json") for suffix in ("writer-v2", "budget-ledger")
    )
    if not writer.is_file() or not accounting.is_file():
        pytest.skip("requires the local Live run archive")
    raw = writer.read_bytes()
    candidate = json.loads(raw)["candidates"][-1]
    details = candidate["evaluation"]["faithfulness"]["details"]
    assert acceptance(details)["failure_codes"] == ["global_inconsistency"]
    ledger = json.loads(accounting.read_bytes())
    assert (
        validate_accounting_snapshot(ledger, expected_plan=ledger["plan"])[
            "used_tokens"
        ]
        == 1_536_937
    )
    units = build_report_source_units(candidate["survey"])
    payload, _, fmt = conflict_review.build_request(
        units, copy.deepcopy(details["assertions"])
    )
    assert len(payload["claims"]) == 478
    assert set(fmt["json_schema"]["schema"]["properties"]) == {"conflicts"}
    assert writer.read_bytes() == raw and candidate["committed"] is False
