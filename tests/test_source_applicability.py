"""Keep real task boundaries in writing, review, evaluation and optional Judge."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_batched_evaluation import materials
from test_body_analysis import analysis_document, render_analysis
from test_evaluation_wire_contract import AnalysisTransport
from test_support_passages import selected
from test_writing_passages import facts

from litagent.body_quality import APPLICABILITY_CONTRACT, APPLICABILITY_INSTRUCTION
from litagent.config import EvalConfig
from litagent.eval.scoped import (
    ScopedEvaluationError,
    evaluate_scoped,
    required_support_aspects,
    validate_support_basis,
)
from litagent.quality import (
    build_fact_cards,
    digest,
    grounded_messages,
    prepare_grounded_request,
)

REAL = json.loads(Path("tests/fixtures/evaluation-ownership-real.json").read_text())[
    "scope_case"
]
SOURCE = REAL["source"]["source_quote"]
SCOPED = "在心脏MRI重建中，冻结或参数高效适配的视觉编码器在有限监督与跨数据集测试中展示出数据效率和泛化表现的提升。"


@pytest.mark.parametrize("analysis", [False, True])
def test_applicability_requires_conditions_even_when_the_claim_omits_them(analysis):
    claim = {
        "claim_id": "c",
        "claim_text": "Visual encoders improve performance.",
        "evidence_ids": ["a"],
    }
    ledger = {"a": {"text": claim["claim_text"], "supporting_text": SOURCE}}
    decision, passages = selected(claim, ledger, {"a": "e1"})
    result = validate_support_basis(
        claim,
        decision,
        ledger,
        {"a": "e1"},
        passages=passages,
        original_only=True,
        applicability=analysis,
    )
    assert result["status"] == ("unverified" if analysis else "passed")
    if analysis:
        assert result["issue"] == "support_aspect_coverage_invalid"
    assert (
        "conditions"
        in required_support_aspects(claim["claim_text"], applicability=analysis)
    ) == analysis


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scoped,contradictory", [(True, False), (False, False), (False, True)]
)
async def test_real_mri_source_pair_passes_through_actual_evaluation(
    scoped, contradictory
):
    _, context = materials(1)
    context.update(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    context["evidence"]["a0"].update(
        text=REAL["claim"]["claim_text"], supporting_text=SOURCE
    )
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    claim = SCOPED if scoped else REAL["claim"]["claim_text"]

    class ScopeTransport(AnalysisTransport):
        """Replace provider judgments only, retaining complete sources and host checks."""

        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "cited_materials" in payload:
                assert APPLICABILITY_INSTRUCTION in messages[0]["content"]
                assert all(
                    "conditions" in c["required_support_aspects"]
                    for c in payload["assertions"]
                )
                assert any(
                    x["text"] == SOURCE for x in payload["original_supports"].values()
                )
                result = json.loads(response.content)
                if not scoped:
                    d = result["decisions"][0]
                    d.update(
                        status="supported" if contradictory else "unsupported",
                        reason_code=(
                            "supported" if contradictory else "insufficient_support"
                        ),
                        explanation=REAL["claim"]["claim_text"]
                        + ": scope exceeds cardiac MRI evidence.",
                    )
                    for aspect in d["aspect_checks"]:
                        if aspect["aspect"] == "conditions":
                            aspect.update(
                                supported=False,
                                support_ids=[],
                                explanation="The general visual-task conclusion omits the actual cardiac MRI task.",
                            )
                    if not contradictory:
                        d["support_basis"] = []
                response.content = json.dumps(result)
            return response

    llm = ScopeTransport()
    if contradictory:
        with pytest.raises(
            ScopedEvaluationError, match="evaluation_decision_consistency_invalid"
        ):
            await evaluate_scoped(
                llm=llm, survey=claim + " [E:a0]", context=context, config=EvalConfig()
            )
        assert len(llm.calls) == 2  # No global call after an illegal semantic contract.
        return
    result = await evaluate_scoped(
        llm=llm, survey=claim + " [E:a0]", context=context, config=EvalConfig()
    )
    assert result["score"] == (1.0 if scoped else 0.0)
    assert result["unsupported_count"] == (0 if scoped else 1)
    assert result["source_applicability_contract"] == APPLICABILITY_CONTRACT
    assert len(llm.calls) == 3
    assert context["evidence"]["a0"]["supporting_text"] == SOURCE


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_generation_and_whole_revision_share_scope_contract(language):
    cards = facts(SOURCE)
    for draft in ("", "Current draft [E:a]"):
        messages = grounded_messages(
            cards,
            "academic_v2",
            language,
            "data-efficient visual methods",
            draft=draft,
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
        prepared, _ = prepare_grounded_request(messages)
        packet = json.loads(prepared[1]["content"])
        assert APPLICABILITY_INSTRUCTION in prepared[0]["content"]
        assert packet["source_applicability_contract"] == APPLICABILITY_CONTRACT
        assert any(
            s.get("text") == SOURCE or s.get("source_quote") == SOURCE
            for s in packet["sources"].values()
        )


@pytest.mark.asyncio
async def test_actual_reviewer_and_revision_receive_full_original_scope():
    from litagent.agents.reviewer import ReviewerWorker
    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.llm.client import LLMResponse
    from litagent.review_structure import structure_sections

    ledger = {
        "a": {
            "evidence_id": "a",
            "paper_id": REAL["source"]["paper_id"],
            "text": REAL["claim"]["claim_text"],
            "supporting_text": SOURCE,
            "chunk_key": "abstract",
        }
    }
    scope = {k: ["a"] for k, _ in structure_sections("academic_v2", "en")}
    selection = EvidenceSelection(
        1, ledger, scope, {k: "lexical_fallback" for k in scope}, 0, 1
    )
    client = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(
                    {
                        "score": 0.1,
                        "strengths": [],
                        "weaknesses": [],
                        "issues": [],
                        "missing_coverage": [],
                        "verdict": "revise",
                        "evidence_compliance": {
                            "passed": False,
                            "unsupported_claims": [],
                            "unknown_evidence_ids": [],
                        },
                    }
                )
            )
        )
    )
    reviewer = ReviewerWorker(client)
    for previous in (None, {"overall_score": 0.1}):
        await reviewer._request_review_with_compact_retry(
            system="Review actual citations.",
            draft=REAL["claim"]["source_quote"].replace(
                REAL["source"]["evidence_id"], "a"
            ),
            selection=selection,
            previous_review=previous,
            query="data-efficient visual methods",
            report_language="en",
            review_structure="academic_v2",
            quality_policy="strict_auto_v2",
            body_quality="analysis_v1",
        )
        messages = client.chat.call_args.args[0]
        assert APPLICABILITY_INSTRUCTION in messages[0]["content"]
        packet = json.loads(messages[1]["content"])
        assert packet["source_applicability_contract"] == APPLICABILITY_CONTRACT
        assert SOURCE in json.dumps(packet, ensure_ascii=False)
    assert client.chat.await_count == 2


def test_both_targeted_repair_requests_retain_the_shared_applicability_contract():
    from test_fragment_repair import case

    from litagent.quality import (
        GroundedWritingError,
        prepare_fragment_repair_request,
        prepare_native_repair_request,
    )

    body, _, cards, targets, _, _ = case()
    messages = grounded_messages(
        cards,
        "academic_v2",
        "en",
        "Compare methods",
        draft=body,
        use_full_sources=True,
        use_scope_notices=True,
        analysis_mode=True,
    )
    prepared, _ = prepare_fragment_repair_request(messages, targets)
    assert APPLICABILITY_INSTRUCTION in prepared[0]["content"]
    assert (
        json.loads(prepared[1]["content"])["source_applicability_contract"]
        == APPLICABILITY_CONTRACT
    )

    cards = facts()
    doc = analysis_document(cards)
    doc["sections"]["open_problems"]["first_block"]["first_sentence"] = {
        "text": "The field lacks systematic comparisons.",
        "kind": "gap",
        "fact_card_ids": [],
        "scope_notice_id": "none",
    }
    messages = grounded_messages(
        cards,
        "academic_v2",
        "en",
        "EViT",
        use_full_sources=True,
        use_scope_notices=True,
        analysis_mode=True,
    )
    request, _ = prepare_grounded_request(messages)
    raw = json.dumps(doc)
    with pytest.raises(GroundedWritingError) as failure:
        render_analysis(doc, cards)
    native, _, _ = prepare_native_repair_request(
        request,
        raw,
        failure.value.diagnostics,
        cards=cards,
        structure="academic_v2",
        language="en",
    )
    assert APPLICABILITY_INSTRUCTION in native[0]["content"]


@pytest.mark.asyncio
async def test_optional_frozen_judge_consumes_the_same_conditions_requirement():
    from test_batched_evaluation import BatchLLM
    from test_strict_automatic import frozen

    from litagent.audit_context import SurveyAuditContext
    from litagent.benchmark.survey_judge import SurveyBenchmarkJudge, SurveyJudgeConfig
    from litagent.eval.automatic import freeze_automatic_inputs
    from litagent.report_render import build_report_source_units

    report = copy.deepcopy(frozen()["report"])
    report["survey"] = SCOPED + " [E:a]"
    ledger = {
        "a": {
            "paper_id": REAL["source"]["paper_id"],
            "text": SCOPED,
            "supporting_text": SOURCE,
        }
    }
    cards = build_fact_cards(ledger)
    report["metadata"].update(fact_cards=cards, evidence_snapshot_sha256=digest(cards))
    unit = build_report_source_units(report["survey"])[0]
    details = report["evaluation"]["faithfulness"]["details"]
    details["source_applicability_contract"] = APPLICABILITY_CONTRACT
    details["assertions"][0].update(
        claim_text=SCOPED,
        source_quote=report["survey"],
        source_unit_id=unit.unit_id,
        source_fragment_id=f"{unit.unit_id}:fragment:0",
        source_start=0,
        source_end=len(report["survey"]),
    )
    snapshot = freeze_automatic_inputs(report, ledger, run_id="applicability-offline")
    assert (
        snapshot["frozen_inventory"]["source_applicability_contract"]
        == APPLICABILITY_CONTRACT
    )

    class Capture(BatchLLM):
        async def chat(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            if "cited_materials" in payload:
                assert APPLICABILITY_INSTRUCTION in messages[0]["content"]
                assert (
                    "conditions" in payload["assertions"][0]["required_support_aspects"]
                )
            return await super().chat(messages, **kwargs)

    result, usage = await SurveyBenchmarkJudge(
        Capture(),
        SurveyJudgeConfig(base_url="https://offline.invalid/v1", model="offline"),
    ).evaluate(
        case=SurveyAuditContext.model_validate(snapshot["context"]),
        survey=report["survey"],
        ledger=ledger,
        frozen_inventory=snapshot["frozen_inventory"],
    )
    assert result.status == "completed" and result.faithfulness == 1.0
    assert usage.call_count == 2
