"""Keep authored units, atomic claims and complete verification capacity distinct."""

import copy
import json

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.config import EvalConfig
from litagent.eval.batched import plan_batches
from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped
from litagent.quality import build_fact_cards, digest


class CompoundInventory(BatchLLM):
    """Replace only transport output with two actual propositions per source unit."""

    async def chat(self, messages, **kwargs):
        response = await super().chat(messages, **kwargs)
        payload = json.loads(messages[1]["content"])
        if "allowed_citation_aliases" not in payload:
            return response
        result = json.loads(response.content)

        def expand(value):
            if isinstance(value, list):
                for item in value:
                    expand(item)
            elif isinstance(value, dict):
                if "assertions" in value:
                    expanded = []
                    for assertion in value["assertions"]:
                        for text in (
                            "ViT uses image patches.",
                            "ViT learns visual representations.",
                        ):
                            expanded.append({**assertion, "claim_text": text})
                    value["assertions"] = expanded
                else:
                    for item in value.values():
                        expand(item)

        expand(result)
        response.content = json.dumps(result)
        return response


def compound_materials(count):
    """Retain complete source text and update its actual frozen card identity."""
    survey, context = materials(count)
    sentence = "ViT uses image patches and learns visual representations."
    survey = survey.replace("ViT uses image patches.", sentence)
    for item in context["evidence"].values():
        item["text"] = item["supporting_text"] = sentence
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    return survey, context


@pytest.mark.asyncio
async def test_sixty_four_units_still_need_eight_batches_for_128_atomic_claims():
    """The real evaluator checks both findings rather than treating each unit as one."""
    survey, context = compound_materials(64)
    llm = CompoundInventory()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    assert result["coverage"]["complete"]
    assert result["factual_claim_count"] == result["supported_count"] == 128
    assert llm.batches == 8 and len(llm.calls) == 10
    ids = [c["claim_id"] for batch in llm.calls[1:-1] for c in batch["assertions"]]
    assert len(ids) == len(set(ids)) == 128


@pytest.mark.asyncio
async def test_sixty_five_legal_units_cannot_drop_two_claims_to_fit_capacity():
    """An over-capacity inventory stays fully unverified without support requests."""
    survey, context = compound_materials(65)
    original = copy.deepcopy(context)
    llm = CompoundInventory()
    with pytest.raises(ScopedEvaluationError) as caught:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    details = caught.value.details
    assert str(caught.value) == "evaluation_assertion_capacity_exceeded"
    assert details["factual_claim_count"] == details["unverified_count"] == 130
    assert len(details["assertions"]) == 130
    assert not details["coverage"]["complete"]
    assert llm.batches == 0 and len(llm.calls) == 1
    assert context == original


def test_organization_statements_still_consume_model_inventory_capacity():
    """Host disclosure is separate, but modeled organization cannot erase count."""
    claims = [
        {"claim_id": f"c{i}", "kind": "scientific" if i < 124 else "report_structure"}
        for i in range(129)
    ]
    with pytest.raises(ValueError, match="evaluation_assertion_capacity_exceeded"):
        plan_batches({"assertions": claims}, EvalConfig())
    assert len(claims) == 129


def test_native_global_preflight_matches_the_existing_caller_wire():
    """Include the caller-owned instruction without changing a valid wire request."""
    from litagent.body_quality import NATIVE_OWNERSHIP_INSTRUCTION
    from litagent.eval.batched import global_prompt
    from litagent.eval.scoped import _request_identity, build_evaluation_request
    from litagent.safety.budget import strict_request_reservation

    payload = {"source_units": [{"text": "unchanged source"}], "assertions": []}
    original = copy.deepcopy(payload)
    old_actual_system = global_prompt(payload) + NATIVE_OWNERSHIP_INSTRUCTION
    new_system = global_prompt(payload, native_unit_ownership=True)
    old_messages, old_format = build_evaluation_request(old_actual_system, payload)
    messages, response_format = build_evaluation_request(new_system, payload)
    assert messages == old_messages and response_format == old_format
    assert messages[0]["content"].count(NATIVE_OWNERSHIP_INSTRUCTION) == 1
    assert _request_identity(messages, response_format) == _request_identity(
        old_messages, old_format
    )
    assert strict_request_reservation(
        messages, 16384, response_format
    ) == strict_request_reservation(old_messages, 16384, old_format)
    assert payload == original


@pytest.mark.asyncio
async def test_global_context_failure_is_known_before_any_support_request(monkeypatch):
    """An oversized full-report check must not waste already predictable batch calls."""
    from litagent.eval import batched

    original_prompt = batched.global_prompt
    monkeypatch.setattr(
        batched,
        "global_prompt",
        lambda payload, **kwargs: original_prompt(payload, **kwargs) + "x" * 150000,
    )
    survey, context = materials(2)
    llm = BatchLLM()
    with pytest.raises(ScopedEvaluationError) as caught:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert str(caught.value) == "evaluation_context_budget_exceeded"
    assert len(llm.calls) == 1 and llm.batches == 0
    progress = caught.value.details["stages"]
    assert progress["global_preflight"]["status"] == "rejected_not_sent"
    assert progress["rejected_requests"][-1]["phase"] == "global_planning"
    assert not progress["batches"]
