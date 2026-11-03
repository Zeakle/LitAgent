"""Exercise typed findings through the real SDK, frozen worker and report outlet."""

import copy
import json

import pytest
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.eval.batched import (
    CURRENT_FINDINGS_CONTRACT,
    GLOBAL_FINDINGS_CONTRACT,
    AnalysisGlobalReview,
    LocatedAnalysisGlobalReview,
    global_review_type,
)
from litagent.frozen_writing import frozen_options, plan_frozen_budget, run_frozen
from litagent.quality import digest, strict_acceptance
from litagent.review_structure import build_methodology
from litagent.runner import LitAgent, RunPolicy
from litagent.writing_v2 import EOF_CONTAINER_POLICY, RANGE_WRITING_PLAN


class FindingsGateway(FlatGateway):
    """Change only external responses while retaining all production consumers."""

    def __init__(self, case, fault):
        super().__init__(case)
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            value = json.loads(response.content)
            if "cited_materials" in payload and fault == "atomic":
                value["decisions"][0].update(
                    atomic=False,
                    status="unverifiable",
                    reason_code="unverifiable",
                    support_basis=[],
                    aspect_checks=[],
                )
            if "cited_materials" in payload and fault == "source_fidelity":
                value["decisions"][0].update(
                    source_faithful=False,
                    status="unverifiable",
                    reason_code="unverifiable",
                    support_basis=[],
                    aspect_checks=[],
                )
            if "cited_materials" in payload and fault == "batch_duplicate":
                response.content = (
                    json.dumps(value)[:-1]
                    + ',"decisions":'
                    + json.dumps(value["decisions"])
                    + "}"
                )
                return response
            if "analyses" in value:
                cid = payload["analysis_claim_ids_by_section"]["methods"][0]
                claim = next(c for c in payload["assertions"] if c["claim_id"] == cid)
                unit_id = next(
                    u["source_unit_id"]
                    for u in payload["source_units"]
                    if any(
                        f["source_fragment_id"] == claim["source_fragment_id"]
                        for f in u["source_fragments"]
                    )
                )
                if fault in {"inventory", "report", "forged", "wrong_owner"}:
                    value["issues"] = [
                        {
                            "source_unit_id": unit_id,
                            "code": "repetition" if fault == "report" else "non_atomic",
                            "claim_ids": ["forged"] if fault == "forged" else [cid],
                            "explanation": "A located test finding; not scientific evidence.",
                        }
                    ]
                    if (
                        payload.get("global_findings_contract")
                        == "anchored-global-findings-v2"
                    ):
                        value["issues"][0].pop("source_unit_id")
                        value["issues"][0]["anchor_ids"] = value["issues"][0].pop(
                            "claim_ids"
                        )
                    if fault == "wrong_owner":
                        value["issues"][0]["owner"] = "report"
                if fault == "incomplete":
                    value["units"][0]["complete"] = False
                if fault == "global_json":
                    response.content = '{"units":'
                    return response
                if fault == "global_duplicate":
                    response.content = json.dumps(value)[:-1] + ',"issues":[]}'
                    return response
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


async def run_case(
    case,
    tmp_path,
    monkeypatch,
    fault,
    *,
    inventory_recovery=False,
    classification_review=False,
    allow_uncertain=False,
):
    """Run Synthesis, Reviewer, Evaluation and automatic refusal with HTTP mocked."""
    gateway = FindingsGateway(case, fault)
    owner, _, _ = await owner_for(gateway)
    config = load_config()
    config.observability.enabled = False
    config.mcp_servers = {}
    config.llm.model = config.llm.extraction_model = config.llm.evaluation_model = (
        "lite"
    )
    config.llm.generation_model = "pro"
    config.llm.max_tokens = 20000
    config.adversarial.review_max_tokens = 4096
    agent = LitAgent(config, run_policy=RunPolicy.isolated_benchmark())
    agent._targeted_repair_state_path = tmp_path / "repair.json"
    agent._evaluation_progress_root = tmp_path / "evaluation"

    def factory(**kwargs):
        owner._model = kwargs["model"]
        owner._cost_budget = kwargs["cost_budget"]
        owner._trace_hook = kwargs["trace_hook"]
        return owner

    monkeypatch.setattr("litagent.runner.OpenAICompatibleClient", factory)
    monkeypatch.setattr(
        "litagent.frozen_writing.frozen_options",
        lambda: frozen_options().model_copy(update={"report_language": "en"}),
    )
    bundle = {
        "query": case.packet["query"],
        "selection": case.selection.to_dict(),
        "source_run_id": "offline-source",
        "requirements": case.packet["requirements"],
        "extractions": [
            {"paper_id": i["paper_id"], "claims": [i["text"]], "evidence_items": [i]}
            for i in case.ledger.values()
        ],
        "methodology": build_methodology(
            case.packet["query"], {}, source_outcomes=[], started_at=1700000000
        ),
        "fact_cards_sha256": digest(case.cards),
    }
    before = copy.deepcopy(bundle)
    plan = plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        inventory_recovery=inventory_recovery,
        classification_review=classification_review,
    )
    try:
        report = await run_frozen(
            agent,
            bundle,
            "located-offline",
            native_provider_api="responses",
            native_reviewer_thinking_mode="disabled",
            native_syntax_policy=EOF_CONTAINER_POLICY,
            native_writing_plan=RANGE_WRITING_PLAN,
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            inventory_recovery=inventory_recovery,
            classification_review=classification_review,
        )
        assert bundle == before
        if not allow_uncertain:
            assert not agent._cost_budget.snapshot()["usage_uncertain"]
        return report, gateway
    finally:
        await agent.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["inventory", "atomic", "incomplete", "source_fidelity"]
)
async def test_inventory_defect_keeps_host_checked_decisions_without_rewrite(
    case, tmp_path, monkeypatch, fault
):
    report, gateway = await run_case(case, tmp_path, monkeypatch, fault)
    details = report["evaluation"]["faithfulness"]["details"]
    assert details["global_findings_contract"] == CURRENT_FINDINGS_CONTRACT
    assert details["inventory_check"]["status"] == "failed"
    assert details["assessment_stage"] == "host_validated"
    assert not details["coverage"]["complete"]
    assert details["supported_count"] > 0
    assert details["stages"]["global_check"] == "completed"
    if fault == "inventory":
        binding = details["stages"]["global_finding_bindings"][0]
        assert binding["finding_index"] == 0
        assert binding["source_unit_ids"] == [
            details["global_issues"][0]["source_unit_id"]
        ]
        assert binding["anchor_ids"][0].startswith("claim:")
    assert report["evaluation"]["faithfulness"]["skipped"]
    assert not report["delivery"]["publishable"]
    assert not report["metadata"]["targeted_repair_state"]["attempts"]
    assert not any("targets" in row["payload"] for row in gateway.sent)
    assert not LitAgent._completed_bound_repair(report["evaluation"], report["survey"])
    assert strict_acceptance(details)["status"] == "failed"
    tampered = copy.deepcopy(details)
    tampered["coverage"]["complete"] = True
    assert "inventory_invalid" in strict_acceptance(tampered)["failure_codes"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["forged", "wrong_owner", "global_json", "global_duplicate"]
)
async def test_invalid_global_protocol_keeps_partial_diagnostics_but_never_a_score(
    case, tmp_path, monkeypatch, fault
):
    report, gateway = await run_case(case, tmp_path, monkeypatch, fault)
    details = report["evaluation"]["faithfulness"]["details"]
    assert report["evaluation"]["faithfulness"]["skipped"]
    assert not report["delivery"]["publishable"]
    assert not details.get("global_issues")
    partial = details["partial_assessment"]
    assert partial["stage"] == "host_validated" and not partial["overall_valid"]
    assert partial["processed_assertions"] > 0 and partial["supported_count"] > 0
    assert not any("targets" in row["payload"] for row in gateway.sent)


@pytest.mark.asyncio
async def test_duplicate_batch_keys_are_rejected_before_any_scientific_score(
    case, tmp_path, monkeypatch
):
    report, _ = await run_case(case, tmp_path, monkeypatch, "batch_duplicate")
    details = report["evaluation"]["faithfulness"]["details"]
    assert report["evaluation"]["faithfulness"]["skipped"]
    assert details["skipped_reason"] == "evaluation_response_size_or_json_invalid"
    assert details["stages"]["batches"][0]["status"] == "interrupted_no_replay"
    assert details["stages"]["requests"][-1]["response"].count('"decisions"') == 2
    assert not report["delivery"]["publishable"]
    assert not report["metadata"]["targeted_repair_state"]["attempts"]


@pytest.mark.asyncio
async def test_report_findings_keep_complete_science_and_authorize_bounded_repair(
    case, tmp_path, monkeypatch
):
    report, _ = await run_case(case, tmp_path, monkeypatch, "report")
    history = report["metadata"]["evaluation_history"]["initial_evaluation"]
    details = history["faithfulness"]["details"]
    assert details["coverage"]["complete"]
    assert details["inventory_check"]["status"] == "passed"
    assert details["global_issues"][0]["owner"] == "report"
    assert strict_acceptance(details)["status"] == "failed"
    assert 0 < len(report["metadata"]["targeted_repair_state"]["attempts"]) <= 2
    assert not report["delivery"]["publishable"]


def test_old_global_response_type_stays_unchanged_and_no_keyword_owner_inference():
    assert (
        global_review_type({"body_analysis_contract": "body-analysis-v1"})
        is AnalysisGlobalReview
    )
    assert (
        global_review_type(
            {
                "body_analysis_contract": "body-analysis-v1",
                "global_findings_contract": GLOBAL_FINDINGS_CONTRACT,
            }
        )
        is LocatedAnalysisGlobalReview
    )
    old = AnalysisGlobalReview.model_json_schema()["$defs"]["GlobalIssue"]
    assert set(old["properties"]) == {"source_unit_id", "explanation"}
    new = LocatedAnalysisGlobalReview.model_json_schema()["$defs"]["LocatedGlobalIssue"]
    assert "owner" not in new["properties"]


@pytest.mark.parametrize("language", ["zh-CN", "en"])
def test_discovery_counts_include_recorded_memory_recall_without_changing_units(
    language,
):
    from litagent.review_structure import render_methodology

    upstream = {f"search_{i}": [None] * 20 for i in range(6)}
    upstream.update({f"recall_{i}": [None] * 20 for i in range(3)})
    upstream["dedup"] = [None] * 132
    record = build_methodology("question", upstream)
    original = copy.deepcopy(record)
    rendered = render_methodology(record, language)
    for count in ["120", "60", "180", "132"]:
        assert count in rendered
    assert (
        "Memory recall records" if language == "en" else "记忆召回记录条数"
    ) in rendered
    assert record == original
    legacy = render_methodology(
        build_methodology("question", {"search_a": [None]}), language
    )
    assert (
        "Memory recall records" if language == "en" else "记忆召回记录条数"
    ) not in legacy


def test_only_typed_inventory_failure_is_quality_and_unknown_errors_remain_protocol():
    from litagent.eval.scoped import ScopedEvaluationError
    from litagent.writing_v2 import classify_failure

    typed = ScopedEvaluationError(
        "evaluation_inventory_invalid",
        {
            "global_findings_contract": GLOBAL_FINDINGS_CONTRACT,
            "inventory_check": {"status": "failed"},
        },
    )
    assert classify_failure(typed)["family"] == "quality"
    assert (
        classify_failure(ValueError("evaluation_inventory_invalid"))["family"]
        == "protocol"
    )
    assert (
        classify_failure(
            ScopedEvaluationError("evaluation_global_structure_invalid", {})
        )["family"]
        == "protocol"
    )


def test_old_v7_envelope_keeps_its_explicit_contract_without_implicit_upgrade(case):
    from litagent.safety.planned_budget import validate_writer_envelope
    from litagent.writing_v2 import writer_budget_plan

    old = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v6"
    )
    new = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
    )
    assert "global_findings_contract" not in old["envelope"]
    assert new["envelope"]["global_findings_contract"] == CURRENT_FINDINGS_CONTRACT
    explicit_v1 = copy.deepcopy(new["envelope"])
    explicit_v1["global_findings_contract"] = GLOBAL_FINDINGS_CONTRACT
    explicit_v1["identity"] = digest(
        {k: v for k, v in explicit_v1.items() if k != "identity"}
    )
    assert validate_writer_envelope(explicit_v1) == explicit_v1
    legacy = copy.deepcopy(new["envelope"])
    legacy.pop("global_findings_contract")
    legacy["identity"] = digest({k: v for k, v in legacy.items() if k != "identity"})
    assert validate_writer_envelope(legacy) == legacy
    invalid = copy.deepcopy(old["envelope"])
    invalid["global_findings_contract"] = GLOBAL_FINDINGS_CONTRACT
    with pytest.raises(ValueError, match="findings_invalid"):
        validate_writer_envelope(invalid)
