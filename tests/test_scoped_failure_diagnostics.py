"""Keep parsed coverage diagnostics without turning incomplete evaluation into a score."""

import json

import pytest

from litagent.config import EvalConfig, load_config
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.eval.scoped import EVALUATION_VERSION, evaluate_scoped
from litagent.llm.client import LLMResponse

SURVEY = (
    "# Review\n\n## Methods\n\nViT uses image patches [E:paper-a].\n\n"
    "## Runtime record\n\nSearch source: arXiv; retrieved records: 2.\n"
)
EXCLUSION_EXPLANATION = (
    "This unit contains only retrieval metadata and no valid assertions."
)
CONTEXT = {
    "evaluation_version": EVALUATION_VERSION,
    "evidence": {
        "paper-a": {
            "evidence_id": "paper-a",
            "paper_id": "arxiv:2010.11929",
            "paper_title": "An Image is Worth 16x16 Words",
            "text": "ViT uses image patches.",
        }
    },
}


class IncompleteMetadataLLM:
    """Reproduce false completeness on metadata after successful science extraction."""

    def __init__(self):
        self.calls = []
        self.incomplete_unit_id = None

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        payload = json.loads(messages[1]["content"])
        assert len(payload["source_units"]) == 2
        if len(self.calls) == 1:
            scientific, metadata = payload["source_units"]
            fragment = scientific["source_fragments"][0]
            self.incomplete_unit_id = metadata["source_unit_id"]
            output = {
                "units": [
                    {
                        "source_unit_id": scientific["source_unit_id"],
                        "assertions": [
                            {
                                "kind": "scientific",
                                "claim_text": "ViT uses image patches.",
                                "source_fragment_id": fragment["source_fragment_id"],
                                "evidence_aliases": fragment["citation_aliases"],
                            }
                        ],
                        "exclusion_reason": "",
                    },
                    {
                        "source_unit_id": self.incomplete_unit_id,
                        "assertions": [],
                        "exclusion_reason": EXCLUSION_EXPLANATION,
                    },
                ]
            }
        else:
            claim = payload["assertions"][0]
            support_id = next(
                pid
                for pid, entry in payload["support_passages"].items()
                if entry[0] == claim["evidence"][0]["evidence_alias"]
            )
            output = {
                "units": [
                    {
                        "source_unit_id": unit["source_unit_id"],
                        "complete": unit["source_unit_id"] != self.incomplete_unit_id,
                        "explanation": (
                            EXCLUSION_EXPLANATION
                            if unit["source_unit_id"] == self.incomplete_unit_id
                            else "All scientific assertions were extracted."
                        ),
                    }
                    for unit in payload["source_units"]
                ],
                "decisions": [
                    {
                        "claim_id": claim["claim_id"],
                        "correct_kind": "scientific",
                        "atomic": True,
                        "classification_valid": True,
                        "source_faithful": True,
                        "status": "supported",
                        "reason_code": "supported",
                        "explanation": "The model claims the actual paper supports it.",
                        "support_basis": [{"support_id": support_id}],
                        "aspect_checks": [
                            {
                                "aspect": aspect,
                                "supported": True,
                                "support_ids": [support_id],
                                "explanation": "The model claims this aspect is supported.",
                            }
                            for aspect in claim["required_support_aspects"]
                        ],
                    }
                ],
                "conflicts": [],
            }
        return LLMResponse(
            content=json.dumps(output),
            model="offline-incomplete-metadata",
            usage={
                "prompt_tokens": 37,
                "completion_tokens": 11,
                "usage_available": True,
            },
        )


def assert_incomplete_diagnostics(details, llm):
    """Treat completed extraction counts as diagnostics, never supported facts."""
    coverage = details["coverage"]
    assert coverage["source_units"] == 2
    assert coverage["processed_units"] == 2
    assert coverage["complete"] is False
    assert coverage["incomplete_unit_ids"] == [llm.incomplete_unit_id]
    assert details["incomplete_units"] == [
        {
            "source_unit_id": llm.incomplete_unit_id,
            "complete": False,
            "explanation": EXCLUSION_EXPLANATION,
        }
    ]
    assert details["completed_scientific_claim_count"] == 1
    assert details["provider"]["call_count"] == 2
    assert details["provider"]["model"] == ["offline-incomplete-metadata"] * 2
    assert details["provider"]["usage"] == {
        "prompt_tokens": 74,
        "completion_tokens": 22,
    }
    assert details["provider"]["usage_complete"] is True
    assert "score" not in details
    assert "supported_count" not in details
    assert "assertions" not in details
    assert len(llm.calls) == 2


@pytest.mark.asyncio
async def test_incomplete_assessment_rejects_score_but_preserves_parsed_diagnostics():
    llm = IncompleteMetadataLLM()
    with pytest.raises(ValueError, match="evaluation_extraction_incomplete") as raised:
        await evaluate_scoped(
            llm=llm, survey=SURVEY, context=CONTEXT, config=EvalConfig()
        )
    details = getattr(raised.value, "details", None)
    assert isinstance(
        details, dict
    ), "A parsed completeness failure needs host diagnostics."
    assert_incomplete_diagnostics(details, llm)


@pytest.mark.asyncio
async def test_faithfulness_skip_retains_incomplete_diagnostics_without_verified_score():
    llm = IncompleteMetadataLLM()
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        SURVEY, CONTEXT
    )
    assert result.skipped is True
    assert result.details["skipped_reason"] == "evaluation_extraction_incomplete"
    # EvalResult.skip's legacy numeric slot is not an evaluated or valid score.
    assert result.score == 0.0
    assert_incomplete_diagnostics(result.details, llm)
