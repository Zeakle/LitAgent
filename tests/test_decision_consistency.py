"""Reject inconsistent model diagnostics without promoting scientific verdicts."""

import copy
import json

import pytest
from test_batched_evaluation import BatchLLM, materials
from test_quality_protocol_stop import live_runner, report

from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.eval.batched import _validate_decision_consistency
from litagent.eval.scoped import PassageDecision, ScopedEvaluationError, evaluate_scoped
from litagent.llm.client import LLMResponse


def packet(status="unsupported"):
    """Build an explicit synthetic claim and its actual cited passage inventory."""
    claim = {
        "claim_id": "claim_0",
        "kind": "scientific",
        "claim_text": "A achieves 83.1% on D1.",
        "source_unit_id": "u1",
        "source_fragment_id": "u1:f0",
        "required_support_aspects": ["predicate", "conditions", "quantities"],
        "evidence": [{"evidence_alias": "e1"}],
    }
    payload = {
        "support_passages": {"p1": ["e1", "s1", 0, 25]},
        "original_supports": {"s1": {"text": "A achieves 83.1% on D1."}},
    }
    decision = PassageDecision.model_validate(
        {
            "claim_id": "claim_0",
            "correct_kind": "scientific",
            "atomic": True,
            "classification_valid": True,
            "source_faithful": True,
            "status": status,
            "reason_code": (
                "supported" if status == "supported" else "insufficient_support"
            ),
            "explanation": "Synthetic diagnostic response, not a gold label.",
            "support_basis": [{"support_id": "p1"}],
            "aspect_checks": [
                {
                    "aspect": a,
                    "supported": True,
                    "support_ids": ["p1"],
                    "explanation": "Synthetic aspect check.",
                }
                for a in claim["required_support_aspects"]
            ],
        }
    )
    return claim, decision, payload


def test_negative_status_and_complete_positive_diagnostics_are_not_promoted():
    claim, decision, payload = packet()
    before = copy.deepcopy((claim, decision.model_dump(), payload))
    with pytest.raises(
        ScopedEvaluationError, match="decision_consistency_invalid"
    ) as caught:
        _validate_decision_consistency(claim, decision, payload)
    issue = caught.value.details["decision_inconsistencies"][0]
    assert issue["source_unit_id"] == "u1" and issue["source_fragment_id"] == "u1:f0"
    assert issue["original_decision"]["status"] == "unsupported"
    assert issue["subreason"] == "negative_status_with_complete_positive_diagnostics"
    assert (claim, decision.model_dump(), payload) == before


@pytest.mark.parametrize(
    "aspect", ["predicate", "conditions", "quantities", "causal_relation"]
)
def test_positive_status_cannot_hide_a_negative_required_aspect(aspect):
    claim, decision, payload = packet("supported")
    claim["required_support_aspects"].append("causal_relation")
    decision.aspect_checks.append(
        type(decision.aspect_checks[0])(
            aspect="causal_relation",
            supported=True,
            support_ids=["p1"],
            explanation="Synthetic causal check.",
        )
    )
    next(a for a in decision.aspect_checks if a.aspect == aspect).supported = False
    with pytest.raises(ScopedEvaluationError, match="decision_consistency_invalid"):
        _validate_decision_consistency(claim, decision, payload)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_aspect",
        "duplicate_aspect",
        "empty_basis",
        "unknown_passage",
        "other_paper",
        "empty_aspect_ids",
        "unfaithful",
        "misclassified",
        "nonatomic",
        "non_scientific",
        "negative_aspect",
    ],
)
def test_incomplete_or_negative_diagnostics_are_not_mistaken_for_positive_support(
    fault,
):
    claim, decision, payload = packet()
    if fault == "missing_aspect":
        decision.aspect_checks.pop()
    elif fault == "duplicate_aspect":
        decision.aspect_checks.append(copy.deepcopy(decision.aspect_checks[0]))
    elif fault == "empty_basis":
        decision.support_basis.clear()
    elif fault == "unknown_passage":
        payload["support_passages"].clear()
    elif fault == "other_paper":
        payload["support_passages"]["p1"][0] = "e2"
    elif fault == "empty_aspect_ids":
        decision.aspect_checks[0].support_ids.clear()
    elif fault == "unfaithful":
        decision.source_faithful = False
    elif fault == "misclassified":
        decision.correct_kind = "report_structure"
    elif fault == "nonatomic":
        decision.atomic = False
    elif fault == "non_scientific":
        claim["kind"] = "runtime_method"
    else:
        decision.aspect_checks[0].supported = False
    _validate_decision_consistency(claim, decision, payload)
    assert decision.status == "unsupported"


def test_consistent_positive_and_uncertain_results_are_preserved():
    claim, decision, payload = packet("supported")
    _validate_decision_consistency(claim, decision, payload)
    decision.status, decision.reason_code = "unverifiable", "unverifiable"
    _validate_decision_consistency(claim, decision, payload)
    assert decision.status == "unverifiable"


def test_explanation_text_never_promotes_an_unsupported_decision():
    claim, decision, payload = packet()
    decision.aspect_checks[0].supported = False
    decision.explanation = "It is supported and correct."
    _validate_decision_consistency(claim, decision, payload)
    assert decision.status == "unsupported"


@pytest.mark.asyncio
async def test_scoped_resource_error_before_response_preserves_interrupted_state():
    class BudgetErrorLLM(BatchLLM):
        async def chat(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            if "cited_materials" in payload:
                self.calls.append(payload)
                raise ScopedEvaluationError("evaluation_context_budget_exceeded", {})
            return await super().chat(messages, **kwargs)

    survey, context = materials(2)
    llm = BudgetErrorLLM()
    with pytest.raises(
        ScopedEvaluationError, match="context_budget_exceeded"
    ) as caught:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=load_config().eval
        )
    batch = caught.value.details["stages"]["batches"][0]
    assert batch["status"] == "interrupted_no_replay"
    assert "usage" not in batch and len(llm.calls) == 2


class ConflictingBatchLLM(BatchLLM):
    """Change one model status while keeping its actual synthetic checks positive."""

    def __init__(self, conflict_batch=1):
        super().__init__()
        self.conflict_batch = conflict_batch
        self.raw_conflict = None

    async def chat(self, messages, **kwargs):
        response = await super().chat(messages, **kwargs)
        payload = json.loads(messages[1]["content"])
        if "cited_materials" in payload and self.batches == self.conflict_batch:
            content = json.loads(response.content)
            content["decisions"][0].update(
                status="unsupported", reason_code="insufficient_support"
            )
            self.raw_conflict = content["decisions"][0]
            return LLMResponse(
                content=json.dumps(content), model="offline", usage=response.usage
            )
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("batch,count", [(1, 4), (2, 20)])
async def test_real_scoped_consumer_preserves_invalid_response_usage_and_stops_batches(
    batch, count
):
    survey, context = materials(count)
    llm = ConflictingBatchLLM(batch)
    with pytest.raises(
        ScopedEvaluationError, match="decision_consistency_invalid"
    ) as caught:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=load_config().eval
        )
    details = caught.value.details
    assert len(llm.calls) == 1 + batch
    assert details["coverage"]["complete"] is False
    assert details["unverified_count"] == count and details["supported_count"] == 0
    progress = details["stages"]
    assert progress["processed_assertions"] == 16 * (batch - 1)
    assert progress["batches"][-1]["status"] == "invalid_response"
    assert progress["batches"][-1]["usage"]["usage_available"] is True
    assert (
        progress["batches"][-1]["diagnostics"]["decision_inconsistencies"][0][
            "original_decision"
        ]
        == llm.raw_conflict
    )
    assert "global_check" not in progress
    assert details["source_report_sha256"] and details["fact_card_map"]


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate", [False, True])
async def test_consistency_contract_error_stops_the_real_benchmark_consumer(
    tmp_path, monkeypatch, candidate
):
    code = "evaluation_decision_consistency_invalid"
    value = report(code if not candidate else "evaluation_assertions_unverifiable")
    if candidate:
        value["metadata"]["evidence_rewrite"] = {
            "candidate": "A draft",
            "candidate_evaluation": {
                "faithfulness": {"details": {"skipped_reason": code}}
            },
        }
    runner, calls = live_runner(
        tmp_path, monkeypatch, [value, report(None, verified=True)]
    )
    result = await runner.run()
    assert len(calls) == 1 and result["stop_reasons"] == [code]
    assert result["budgets"]["survey"]["charged"] == 15
    runner.service.wait.assert_not_awaited()
    assert await runner.run() == result and len(calls) == 1


def test_consistency_contract_is_strict_only_and_changes_restore_identity(monkeypatch):
    cfg = load_config()
    before = {
        p: build_config_summary(cfg, p)
        for p in ("legacy", "strict_v1", "strict_auto_v1")
    }
    monkeypatch.setattr(
        "litagent.eval.batched.STRICT_DECISION_CONTRACT", "old-consistency-contract"
    )
    after = {p: build_config_summary(cfg, p) for p in before}
    assert before["legacy"] == after["legacy"]
    assert before["strict_v1"] != after["strict_v1"]
    assert before["strict_auto_v1"] != after["strict_auto_v1"]
