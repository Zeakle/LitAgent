"""Keep canonical host disclosures out of probabilistic author-text extraction."""

import json

import pytest

from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped
from litagent.llm.client import LLMResponse
from litagent.report_render import build_report_source_units
from litagent.review_structure import (
    METHOD_END,
    assemble_methodology,
    build_methodology,
)


class CapturedRequest(RuntimeError):
    """Stop after capturing the real request without creating provider output."""


class CaptureLLM:
    """Observe the exact inventory payload without performing model requests."""

    def __init__(self):
        self.requests = []

    async def chat(self, messages, **kwargs):
        self.requests.append(json.loads(messages[1]["content"]))
        raise CapturedRequest()


@pytest.fixture
def record():
    """Build trusted run metadata through the production producer."""
    return build_methodology("ViT", {}, source_outcomes=[], started_at=1791435077.4)


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
async def test_only_byte_verified_host_metadata_uses_deterministic_validation(
    record, language, profile
):
    body = "# Survey\n\n## Results\n\nViT uses patches [E:a].\n\n## Authored runtime claim\n\nWe independently screened 500 papers.\n"
    survey = assemble_methodology(body, profile, language, record)
    llm = CaptureLLM()
    with pytest.raises(CapturedRequest):
        await evaluate_scoped(
            llm=llm,
            survey=survey,
            context={"review_methodology": record},
            config=EvalConfig(),
        )
    payload = llm.requests[0]
    originals = {u.unit_id: u for u in build_report_source_units(survey)}
    assert len(payload["source_units"]) == 2
    sent = "\n".join(
        f["text"] for unit in payload["source_units"] for f in unit["source_fragments"]
    )
    assert "ViT uses patches" in sent
    assert "independently screened 500 papers" in sent
    assert "1791435077.4" not in sent
    for unit in payload["source_units"]:
        original = originals[unit["source_unit_id"]]
        assert survey[original.start : original.end] == original.text
    host = payload["host_methodology_validation"]
    assert host["status"] == "passed"
    assert host["processed_units"] == len(originals) - 2
    assert host["source_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "damage", ["timestamp", "scientific_insertion", "missing_record"]
)
async def test_modified_managed_metadata_cannot_exempt_science_or_start_a_model(
    record, damage
):
    survey = assemble_methodology(
        "# Survey\n\nViT uses patches [E:a].", "academic_v2", "en", record
    )
    context = {"review_methodology": record}
    if damage == "timestamp":
        survey = survey.replace("1791435077.4", "0.0")
    elif damage == "scientific_insertion":
        survey = survey.replace(
            METHOD_END,
            "ViT achieves 99999% accuracy.\n" + METHOD_END,
        )
        assert "99999%" in survey
    else:
        context = {}
    llm = CaptureLLM()
    with pytest.raises(ValueError, match="evaluation_host_methodology_invalid"):
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert llm.requests == []


@pytest.mark.asyncio
async def test_unmarked_methodology_like_science_stays_in_author_inventory(record):
    survey = "# Survey\n\n## Run methodology\n\nViT achieves 99999% accuracy.\n"
    llm = CaptureLLM()
    with pytest.raises(CapturedRequest):
        await evaluate_scoped(
            llm=llm,
            survey=survey,
            context={"review_methodology": record},
            config=EvalConfig(),
        )
    assert "99999%" in json.dumps(llm.requests[0]["source_units"])


@pytest.mark.asyncio
async def test_complete_host_coverage_cannot_raise_science_score_or_hide_author_runtime(
    record,
):
    survey = assemble_methodology(
        "# Survey\n\nViT achieves 99999% accuracy.\n\n"
        "We independently screened 500 papers.\n",
        "academic_v2",
        "en",
        record,
    )

    class UnsupportedLLM:
        """Return complete extraction while rejecting both authored assertions."""

        def __init__(self):
            self.requests = []

        async def chat(self, messages, **kwargs):
            payload = json.loads(messages[1]["content"])
            self.requests.append(payload)
            if len(self.requests) == 1:
                output = {
                    "units": [
                        {
                            "source_unit_id": unit["source_unit_id"],
                            "assertions": [
                                {
                                    "kind": kind,
                                    "claim_text": unit["source_fragments"][0]["text"],
                                    "source_fragment_id": unit["source_fragments"][0][
                                        "source_fragment_id"
                                    ],
                                    "evidence_aliases": [],
                                }
                            ],
                            "exclusion_reason": "",
                        }
                        for unit, kind in zip(
                            payload["source_units"], ["scientific", "runtime_method"]
                        )
                    ]
                }
            else:
                output = {
                    "units": [
                        {
                            "source_unit_id": unit["source_unit_id"],
                            "complete": True,
                            "explanation": "All authored assertions were extracted.",
                        }
                        for unit in payload["source_units"]
                    ],
                    "decisions": [
                        {
                            "claim_id": claim["claim_id"],
                            "correct_kind": claim["kind"],
                            "atomic": True,
                            "classification_valid": True,
                            "source_faithful": True,
                            "status": "unsupported",
                            "reason_code": (
                                "uncited_claim"
                                if claim["kind"] == "scientific"
                                else "record_mismatch"
                            ),
                            "explanation": "The actual evidence or run record does not support this claim.",
                            "support_basis": [],
                            "aspect_checks": [],
                        }
                        for claim in payload["assertions"]
                    ],
                    "conflicts": [],
                }
            return LLMResponse(content=json.dumps(output), model="offline-rejected")

    llm = UnsupportedLLM()
    result = await evaluate_scoped(
        llm=llm,
        survey=survey,
        context={"review_methodology": record},
        config=EvalConfig(),
    )
    assert len(llm.requests) == 2
    assert all(len(request["source_units"]) == 2 for request in llm.requests)
    assert result["score"] == 0.0
    assert result["factual_claim_count"] == result["unsupported_count"] == 1
    assert result["supported_count"] == 0
    assert len(result["method_assertion_errors"]) == 1
    coverage = result["coverage"]
    assert coverage["complete"] is True
    assert coverage["model_source_units"] == coverage["model_processed_units"] == 2
    assert coverage["host_processed_units"] > 0
    assert coverage["source_units"] == coverage["processed_units"]
    assert coverage["source_units"] == 2 + coverage["host_processed_units"]
