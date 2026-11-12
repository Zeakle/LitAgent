"""Reject assertion omissions without guessing support or invalidating source availability."""

import json

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.config import EvalConfig
from litagent.eval.scoped import (
    ScopedEvaluationError,
    _strict_inventory_response,
    evaluate_scoped,
)
from litagent.quality import strict_acceptance


class OmittingLLM(BatchLLM):
    """Return otherwise affirmative decisions after one deliberately hidden fragment."""

    def __init__(self, mode="empty"):
        super().__init__()
        self.mode = mode

    async def chat(self, messages, **kwargs):
        result = await super().chat(messages, **kwargs)
        if "allowed_citation_aliases" in self.calls[-1]:
            value = json.loads(result.content)
            fragment = list(next(iter(value["units"].values())).values())[1]
            if self.mode == "empty":
                fragment.update(
                    assertions=[], exclusion_reason="No assertion to verify"
                )
            else:
                for assertion in fragment["assertions"]:
                    assertion["kind"] = "report_structure"
            result.content = json.dumps(value)
        return result


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("mode", ["empty", "organization"])
async def test_partial_scientific_omission_cannot_return_a_perfect_support_score(
    policy, language, mode
):
    _, context = materials(2)
    context["quality_policy"] = policy
    survey = (
        "# Report\n\nViT uses image patches. [E:a0] "
        "DeiT improves accuracy by 20%. [E:a1]"
        if language == "en"
        else "# 报告\n\nViT采用图像分块 [E:a0]。DeiT提升了准确率20% [E:a1]。"
    )
    llm = OmittingLLM(mode)
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_extraction_incomplete"
    ) as error:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    details = error.value.details
    assert len(llm.calls) == 1 and llm.batches == 0
    assert details["coverage"]["complete"] is False
    assert len(details["stages"]["fragment_inventory"]) == 2
    issue = details["unsupported_claims"][0]
    assert "DeiT" in issue["source_quote"] and "20%" in issue["source_quote"]
    assert survey[issue["source_start"] : issue["source_end"]] == issue["source_quote"]
    assert issue["evidence_refs"] == ["a1"]
    assert issue["diagnostic_source"] == "survey-evaluation-v3"
    assert issue["source_fragment_id"].endswith("fragment:1")
    assert details["fact_card_map"]
    acceptance = strict_acceptance(details)
    assert acceptance["status"] != "passed"
    assert "fact_cards_unavailable" not in acceptance["failure_codes"]


@pytest.mark.asyncio
async def test_uncited_explicit_science_cannot_be_excluded_either():
    _, context = materials(2)
    survey = "# Report\n\nViT uses patches. [E:a0] DeiT improves accuracy by 20%."
    llm = OmittingLLM()
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_extraction_incomplete"
    ) as error:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    issue = error.value.details["unsupported_claims"][0]
    assert issue["evidence_refs"] == [] and "DeiT" in issue["source_quote"]
    assert len(llm.calls) == 1


def partition(text, aliases, *, assertions, exclusion=""):
    """Use a host-bound semantic transport fixture with no model-owned position."""
    units = [
        {
            "source_unit_id": "u",
            "source_fragments": [
                {
                    "source_fragment_id": "u:f",
                    "text": text,
                    "citation_aliases": aliases,
                }
            ],
        }
    ]
    value = {
        "units": {
            "u": {
                "u:f": {
                    "assertions": assertions,
                    "exclusion_reason": exclusion,
                }
            }
        }
    }
    return _strict_inventory_response(json.dumps(value), units, {})


def test_cited_organization_remains_organization_without_being_forced_into_science():
    inventory = partition(
        "This review compares ViT and CNN. [E:e1]",
        ["e1"],
        assertions=[
            {
                "kind": "report_structure",
                "claim_text": "This review compares ViT and CNN.",
            }
        ],
    )
    assert inventory.units[0].assertions[0].kind == "report_structure"
    assert inventory.units[0].assertions[0].evidence_aliases == ["e1"]


@pytest.mark.parametrize("text,aliases", [("Overview", []), ("[E:e1]", ["e1"])])
def test_nonassertive_labels_and_bare_reference_markers_can_be_recorded(text, aliases):
    inventory = partition(text, aliases, assertions=[], exclusion="Nonassertive label")
    assert not inventory.units[0].assertions
    assert inventory.units[0].exclusion_reason == "Nonassertive label"


@pytest.mark.asyncio
async def test_invalid_native_shape_preserves_validated_fact_card_provenance():
    class InvalidShape(BatchLLM):
        async def chat(self, messages, **kwargs):
            result = await super().chat(messages, **kwargs)
            result.content = '{"units":[]}'
            return result

    survey, context = materials(1)
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_inventory_shape_invalid"
    ) as error:
        await evaluate_scoped(
            llm=InvalidShape(), survey=survey, context=context, config=EvalConfig()
        )
    assert set(error.value.details["fact_card_map"]) == set(context["fact_cards"])
    assert error.value.details["coverage"]["complete"] is False


@pytest.mark.asyncio
async def test_forged_fact_card_is_not_marked_available_after_failed_validation():
    survey, context = materials(1)
    next(iter(context["fact_cards"].values()))["source_quote"] = "forged source"
    llm = BatchLLM()
    with pytest.raises(ScopedEvaluationError) as error:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert not error.value.details.get("fact_card_map")
    assert llm.calls == []
