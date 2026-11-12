"""Separate report fidelity from scientific support in actual v3 requests."""

import json

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.config import EvalConfig
from litagent.eval.batched import (
    BATCH_PROMPT,
    SOURCE_FIDELITY_CONTRACT,
    batch_prompt,
    plan_batches,
    request_size,
)
from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped
from litagent.llm.client import LLMResponse


class UncitedLLM(BatchLLM):
    """Return unsupported paper judgments with independently controlled report fidelity."""

    def __init__(self, faithful=True):
        super().__init__()
        self.faithful = faithful
        self.prompts = []

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        response = await super().chat(messages, **kwargs)
        payload = json.loads(messages[1]["content"])
        if "cited_materials" in payload:
            data = json.loads(response.content)
            for decision in data["decisions"]:
                decision.update(
                    source_faithful=self.faithful,
                    status="unsupported",
                    reason_code="uncited_claim",
                    explanation="No actual citation; this says nothing about report fidelity.",
                    support_basis=[],
                )
                for check in decision["aspect_checks"]:
                    check.update(
                        supported=False,
                        support_ids=[],
                        explanation="No actual cited material.",
                    )
            return LLMResponse(
                content=json.dumps(data), model="offline", usage=response.usage
            )
        return response


@pytest.mark.asyncio
@pytest.mark.parametrize("faithful", [True, False])
async def test_uncited_science_never_passes_and_invalid_fidelity_is_not_corrected(
    faithful,
):
    survey, context = materials(1)
    survey = survey.replace(" [E:a0]", "")
    context["quality_policy"] = "strict_auto_v2"
    llm = UncitedLLM(faithful)
    if not faithful:
        with pytest.raises(
            ScopedEvaluationError, match="evaluation_classification_or_source_invalid"
        ) as rejected:
            await evaluate_scoped(
                llm=llm, survey=survey, context=context, config=EvalConfig()
            )
        result = rejected.value.details
    else:
        result = await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    if faithful:
        assert result["score"] != 1.0
        assert result["assertions"] and all(
            c["status"] != "supported" for c in result["assertions"]
        )
    prompts = [p for p in llm.prompts if "BATCH CONTRACT" in p]
    assert len(prompts) == 1 and SOURCE_FIDELITY_CONTRACT in prompts[0]
    assert "report fragment" in prompts[0] and "uncited_claim" in prompts[0]
    assert result["coverage"]["processed_units"] == result["coverage"]["source_units"]
    if faithful:
        assert (
            result.get("reason_code") != "evaluation_classification_or_source_invalid"
        )


def test_old_prompt_and_actual_new_budget_use_the_same_explicit_contract():
    assert batch_prompt({}) == BATCH_PROMPT
    payload = {"source_fidelity_contract": SOURCE_FIDELITY_CONTRACT}
    assert batch_prompt(payload) != BATCH_PROMPT
    # A request fitting only the shorter historical prompt must still be blocked.
    from test_batched_evaluation import materials

    from litagent.eval.batched import batch_payload

    sample = {
        **payload,
        "assertions": [{"claim_id": "c", "source_unit_id": "u", "evidence": []}],
        "source_units": [],
        "report_structure": [],
        "cited_materials": {},
        "original_supports": {},
        "support_passages": {},
        "unit_inventory": [],
    }
    old = request_size(BATCH_PROMPT, sample, 16384)
    new = request_size(batch_prompt(sample), sample, 16384)
    assert new > old
    config = EvalConfig(context_tokens=int(old / 0.9) + 1)
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        plan_batches(sample, config)
