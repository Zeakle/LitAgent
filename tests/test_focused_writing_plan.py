"""Bind focused authoring limits without relaxing scientific or coverage gates."""

import copy
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import EvaluatorTransport, Provider, case  # noqa: F401

from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped
from litagent.quality import digest, strict_acceptance
from litagent.report_audit import atomic_json
from litagent.report_audit import digest as manifest_digest
from litagent.safety.budget import CostBudget
from litagent.writing_v2 import (
    BODY,
    EOF_CONTAINER_POLICY,
    FOCUSED_WRITING_PLAN,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    check_writing_plan,
    compile_document,
    writing_plan_spec,
)
from scripts import run_frozen_writing as campaign


def test_old_api_and_eof_profile_do_not_silently_enable_a_writing_plan(case, tmp_path):
    """Absence and explicit None retain the exact legacy session identity."""
    one = WriterSession(
        Provider(case),
        tmp_path / "one",
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
    )
    two = WriterSession(
        Provider(case),
        tmp_path / "two",
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=None,
    )
    assert one.identity == two.identity and "writing_plan" not in one.journal
    assert "native_writing_plan" not in campaign.profile_parameters(
        campaign.EOF_PROFILE
    )
    assert "writing_plan" not in campaign.profile_description(campaign.EOF_PROFILE, {})
    one.save()
    with pytest.raises(WriterError, match="identity_conflict"):
        WriterSession(
            Provider(case),
            tmp_path / "one",
            "r",
            case.packet,
            case.aliases,
            syntax_policy=EOF_CONTAINER_POLICY,
            writing_plan=FOCUSED_WRITING_PLAN,
        )


def test_budget_is_explicit_and_never_a_scientific_result():
    """The envelope intentionally does not equate 72 units with 128 atomic claims."""
    spec = writing_plan_spec(FOCUSED_WRITING_PLAN)
    assert spec["body_max_units"] == 66 and spec["report_max_units"] == 72
    assert spec["atomic_claim_capacity"] == 128
    assert spec["scientific_assessment"] == "not_verified"
    assert spec["core_coverage_required"] is True
    assert sum(spec["body_target_units"].values()) < spec["body_max_units"]
    spec["body_max_units"] = 1
    assert writing_plan_spec(FOCUSED_WRITING_PLAN)["body_max_units"] == 66
    assert writing_plan_spec(None) is None


@pytest.mark.asyncio
async def test_repair_reserves_full_candidate_capacity_before_spending(case, tmp_path):
    """Small authored-unit counts cannot underreserve a changed inventory."""
    provider = Provider(case)
    provider._cost_budget = CostBudget(max_tokens=180000)
    provider._cost_budget.configure_strict()
    session = WriterSession(
        provider,
        tmp_path / "focused-repair",
        "r",
        case.packet,
        case.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=FOCUSED_WRITING_PLAN,
    )
    before = provider._cost_budget.snapshot()
    with pytest.raises(WriterError, match="transaction_budget") as error:
        await session.patch(
            case.body,
            {
                "x": {
                    "path": ["methods", 0, 0],
                    "original": case.body["methods"][0]["sentences"][0],
                }
            },
            cause="evaluation",
            claim_count=1,
        )
    detail = error.value.diagnostics[0]
    assert detail["input_claim_count"] == 1
    assert detail["projected_model_assertions"] == 128
    assert detail["projected_scientific_batches"] == 8
    assert not provider.calls and not session.state["attempts"]
    assert provider._cost_budget.snapshot() == before
    assert session.journal["failure"]["request_sent"] is False


def test_complete_old_body_is_preserved_but_not_accepted_under_smaller_plan(case):
    """Local budget rejection cannot truncate authored text or alter legacy behavior."""
    body = copy.deepcopy(case.body)
    limits = writing_plan_spec(FOCUSED_WRITING_PLAN)["body_section_max_units"]
    for key in BODY:
        sentence = case.body[key][0]["sentences"][0]
        body[key] = [
            {
                "sentences": [
                    {**sentence, "text": sentence["text"] + f" Finding {i}."}
                    for i in range(limits[key])
                ]
            }
        ]
    before = copy.deepcopy(body)
    draft, _ = compile_document(body, case.summaries, case.cards, case.aliases, "en")
    assert draft
    with pytest.raises(WriterError, match="writing_plan_capacity_exceeded") as error:
        compile_document(
            body,
            case.summaries,
            case.cards,
            case.aliases,
            "en",
            writing_plan=FOCUSED_WRITING_PLAN,
        )
    assert error.value.family == "budget" and body == before


@pytest.mark.asyncio
async def test_overlong_chapter_keeps_complete_raw_response_and_stops_followups(
    case, tmp_path
):
    """A rejected unit budget cannot become a shorter accepted report."""
    expanded = copy.deepcopy(case)
    unit = expanded.body["introduction"][0]["sentences"][0]
    expanded.body["introduction"] = [
        {"sentences": [copy.deepcopy(unit) for _ in range(9)]}
    ]
    provider = Provider(expanded)
    session = WriterSession(
        provider,
        tmp_path / "r",
        "r",
        expanded.packet,
        expanded.aliases,
        syntax_policy=EOF_CONTAINER_POLICY,
        writing_plan=FOCUSED_WRITING_PLAN,
    )
    with pytest.raises(WriterError, match="writing_plan_capacity_exceeded"):
        await session.initial_body()
    assert len(provider.calls) == 1 and session.state["attempts"] == []
    raw = session.journal["calls"]["body:introduction"]["raw"]
    assert len(json.loads(raw)["introduction"][0]["sentences"]) == 9
    assert session.journal["failure"]["family"] == "budget"


def test_summary_budget_keeps_dependency_and_source_boundaries(case):
    """Different sections may reuse body findings but cannot expand without limit."""
    summaries = copy.deepcopy(case.summaries)
    summaries["abstract"][0]["sentences"] *= 4
    before = copy.deepcopy(summaries)
    with pytest.raises(WriterError, match="writing_plan_capacity_exceeded"):
        compile_document(
            case.body,
            summaries,
            case.cards,
            case.aliases,
            "en",
            writing_plan=FOCUSED_WRITING_PLAN,
        )
    assert summaries == before
    with pytest.raises(WriterError, match="writing_plan_invalid"):
        check_writing_plan({}, "unknown")


def test_manifest_binds_the_plan_and_its_complete_numeric_spec(manifest):
    """Rehashing a changed numeric budget cannot authorize another execution."""
    root, config = manifest
    target = root.with_name("focused-plan")
    config = campaign.profile_config(config, campaign.EOF_PROFILE)
    campaign.prepare(
        target,
        config,
        writing_profile=campaign.EOF_PROFILE,
        writing_plan=FOCUSED_WRITING_PLAN,
    )
    value = campaign.validate(target, config)
    assert value["writing_parameters"]["native_writing_plan"] == FOCUSED_WRITING_PLAN
    assert value["writing_plan_spec"] == writing_plan_spec(FOCUSED_WRITING_PLAN)
    value["writing_plan_spec"]["body_max_units"] += 1
    value["identity"] = manifest_digest(
        {k: v for k, v in value.items() if k != "identity"}
    )
    atomic_json(target / "manifest.json", value)
    with pytest.raises(ValueError, match="writing_plan_identity_conflict"):
        campaign.validate(target, config)


def test_all_export_formats_use_the_frozen_plan_without_enabling_old_profiles(manifest):
    """The optional control is visible and identical in every result format."""
    root, config = manifest
    target = root.with_name("focused-export")
    config = campaign.profile_config(config, campaign.EOF_PROFILE)
    campaign.prepare(
        target,
        config,
        writing_profile=campaign.EOF_PROFILE,
        writing_plan=FOCUSED_WRITING_PLAN,
    )
    value = campaign.validate(target, config)
    public = campaign.export(
        target,
        {"identity": value["identity"], "status": "offline_preview", "cases": []},
        value,
    )
    assert public["writing_profile"]["writing_plan"] == writing_plan_spec(
        FOCUSED_WRITING_PLAN
    )
    out = Path("benchmarks/survey/results") / target.name
    rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert all(row["writing_plan"] == FOCUSED_WRITING_PLAN for row in rows)
    page = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            page.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )
    assert FOCUSED_WRITING_PLAN in (out / "benchmark.md").read_text()
    assert (
        public["new_usage"]["requests"] == public["generated"] == public["passed"] == 0
    )
    with pytest.raises(ValueError, match="writing_plan_configuration_invalid"):
        campaign.profile_parameters(
            campaign.ROLE_PROFILE, writing_plan=FOCUSED_WRITING_PLAN
        )


@pytest.mark.asyncio
async def test_invalid_runtime_plan_stops_before_any_client_or_budget_initialization():
    """The fixed frozen boundary cannot become an arbitrary model parameter API."""
    from litagent.frozen_writing import run_frozen

    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="writing_plan_configuration_invalid"):
        await run_frozen(agent, {}, "no-call", native_writing_plan=FOCUSED_WRITING_PLAN)
    with pytest.raises(WriterError, match="writing_plan_invalid"):
        await run_frozen(agent, {}, "no-call", native_writing_plan="unknown")
    assert vars(agent) == {"_wired": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("plan", [FOCUSED_WRITING_PLAN, RANGE_WRITING_PLAN])
async def test_compact_fully_supported_fixture_still_fails_missing_core_coverage(
    case, plan
):
    """Small prose and a scripted support ratio of 1.0 cannot replace the question."""
    draft, bindings = compile_document(
        case.body,
        case.summaries,
        case.cards,
        case.aliases,
        "en",
        writing_plan=plan,
    )

    class MissingCore(EvaluatorTransport):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = self.base.calls[-1]
            if (
                "allowed_citation_aliases" not in payload
                and "cited_materials" not in payload
            ):
                result = json.loads(response.content)
                for coverage in result["coverage"]:
                    coverage.update(
                        answered=False,
                        claim_ids=[],
                        explanation="The requested method is missing.",
                    )
                response.content = json.dumps(result)
            return response

    context = {
        "query": case.packet["query"],
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "report_language": "en",
        "review_structure": "academic_v2",
        "writing_bindings": bindings,
        "scope_notice_contract": "scope-notice-v1",
        "fact_cards": case.cards,
        "evidence_snapshot_sha256": digest(case.cards),
        "evidence": case.ledger,
        "coverage_requirements": case.packet["requirements"],
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
        "language_checks": {"status": "passed"},
    }
    result = await evaluate_scoped(
        llm=MissingCore(), survey=draft, context=context, config=EvalConfig()
    )
    assert result["score"] == 1.0 and result["coverage"]["complete"]
    acceptance = strict_acceptance(result)
    assert acceptance["status"] == "failed"
    assert "core_coverage_unverified" in acceptance["failure_codes"]


@pytest.mark.asyncio
@pytest.mark.parametrize("plan", [FOCUSED_WRITING_PLAN, RANGE_WRITING_PLAN])
async def test_focused_unit_budget_cannot_hide_150_atomic_claims(case, plan):
    """Even a locally legal plan must fail an independently expanded inventory."""
    from litagent.eval.scoped import ScopedEvaluationError

    body = copy.deepcopy(case.body)
    summaries = copy.deepcopy(case.summaries)
    propositions = [
        "Relation networks learn a comparison function.",
        "Relation networks model nonlinear relationships.",
        "Relation networks classify images.",
    ]
    for key, count in writing_plan_spec(FOCUSED_WRITING_PLAN)[
        "body_target_units"
    ].items():
        unit = body[key][0]["sentences"][0]
        body[key] = [
            {
                "sentences": [
                    {**unit, "text": " ".join(propositions)} for _ in range(count)
                ]
            }
        ]
    for paragraphs in summaries.values():
        paragraphs[0]["sentences"][0]["text"] = " ".join(propositions)
    draft, bindings = compile_document(
        body,
        summaries,
        case.cards,
        case.aliases,
        "en",
        writing_plan=plan,
    )
    context = {
        "query": case.packet["query"],
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "report_language": "en",
        "review_structure": "academic_v2",
        "writing_bindings": bindings,
        "scope_notice_contract": "scope-notice-v1",
        "fact_cards": case.cards,
        "evidence_snapshot_sha256": digest(case.cards),
        "evidence": case.ledger,
        "coverage_requirements": case.packet["requirements"],
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
        "language_checks": {"status": "passed"},
    }

    class ThreeAssertions(EvaluatorTransport):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            assert "allowed_citation_aliases" in self.base.calls[-1]
            result = json.loads(response.content)
            for fragment in result["fragments"].values():
                fragment["assertions"] = [
                    {"kind": "scientific", "claim_text": text} for text in propositions
                ]
            response.content = json.dumps(result)
            return response

    llm = ThreeAssertions()
    with pytest.raises(ScopedEvaluationError) as caught:
        await evaluate_scoped(
            llm=llm, survey=draft, context=context, config=EvalConfig()
        )
    assert str(caught.value) == "evaluation_assertion_capacity_exceeded"
    assert caught.value.details["unverified_count"] == 150
    assert len(caught.value.details["assertions"]) == 150
    assert len(llm.base.calls) == 1
