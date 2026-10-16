"""Separate canonical host disclosures from the requested model inventory."""

import json

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.config import EvalConfig
from litagent.eval.scoped import (
    ScopedEvaluationError,
    evaluate_scoped,
)
from litagent.llm.client import LLMResponse
from litagent.review_structure import assemble_methodology, build_methodology


class ScopeLLM(BatchLLM):
    """Preserve the actual protocol and optionally add an unauthorized unit."""

    def __init__(self, extra=False):
        super().__init__()
        self.prompts = []
        self.extra = extra

    async def chat(self, messages, **kwargs):
        self.prompts.append(messages[0]["content"])
        response = await super().chat(messages, **kwargs)
        if self.extra and "allowed_citation_aliases" in self.calls[-1]:
            value = json.loads(response.content)
            extra_unit = {
                "source_unit_id": "host-unrequested",
                "assertions": [],
                "exclusion_reason": "No fragments provided",
            }
            if isinstance(value["units"], dict):
                value["units"]["host-unrequested"] = {}
            else:
                value["units"].append(extra_unit)
            return LLMResponse(
                content=json.dumps(value), model="offline", usage=response.usage
            )
        return response


def with_methodology(policy="strict_v1", language="en"):
    """Render the exact host record while retaining independently authored science."""
    survey, context = materials(1)
    record = build_methodology("ViT mechanisms", {}, started_at=1791435077.4)
    survey = assemble_methodology(survey, "academic_v2", language, record)
    context.update(quality_policy=policy, review_methodology=record)
    return survey, context


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh-CN"])
async def test_host_ids_stay_local_and_exact_model_scope_is_preserved(language):
    survey, context = with_methodology(language=language)
    llm = ScopeLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    inventory = llm.calls[0]
    ids = [u["source_unit_id"] for u in inventory["source_units"]]
    host = result["host_methodology_validation"]
    assert host["source_unit_ids"] and host["processed_units"] > 0
    assert "source_unit_ids" not in inventory["host_methodology_validation"]
    assert "source_unit_ids" not in llm.calls[1]["host_methodology_validation"]
    assert set(host["source_unit_ids"]).isdisjoint(ids)
    partitions = inventory["schema"]["properties"]["units"]
    assert set(partitions["required"]) == set(ids)
    assert set(partitions["properties"]) == set(ids)
    assert partitions["additionalProperties"] is False
    assert "REQUESTED UNIT SCOPE" in llm.prompts[0]
    assert result["coverage"]["complete"]
    assert result["coverage"]["processed_units"] == result["coverage"]["source_units"]


@pytest.mark.asyncio
async def test_extra_unit_is_still_rejected_instead_of_filtered():
    survey, context = with_methodology()
    llm = ScopeLLM(extra=True)
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_inventory_unit_coverage_invalid"
    ):
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_legacy_model_contract_retains_its_previous_shape():
    survey, context = with_methodology(policy="legacy")
    llm = ScopeLLM()
    await evaluate_scoped(llm=llm, survey=survey, context=context, config=EvalConfig())
    inventory = llm.calls[0]
    assert "source_unit_ids" in inventory["host_methodology_validation"]
    assert (
        "enum"
        not in inventory["schema"]["$defs"]["UnitInventory"]["properties"][
            "source_unit_id"
        ]
    )
    assert "minItems" not in inventory["schema"]["properties"]["units"]
    assert "REQUESTED UNIT SCOPE" not in llm.prompts[0]


@pytest.mark.asyncio
async def test_added_science_outside_canonical_block_is_not_hidden():
    survey, context = with_methodology()
    survey += "\n\n## Additional science\n\nViT uses image patches. [E:a0]"
    llm = ScopeLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    assert len(llm.calls[0]["source_units"]) == 2
    assert result["factual_claim_count"] == 2
