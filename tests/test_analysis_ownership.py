"""Replay real chapter misbindings without rewriting prose or forcing support."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from litagent.body_quality import (
    ANALYSIS_OWNERSHIP_CONTRACT,
    ROLES,
    analysis_claim_ids_by_section,
    analysis_repair_diagnostics,
    validate_analysis_decisions,
)
from litagent.eval.batched import global_review_schema


def real_case():
    """Load the frozen automatic observations, not a human scientific gold set."""
    return json.loads(Path("tests/fixtures/evaluation-ownership-real.json").read_text())


def bound_rows(payload):
    """Make chapter-local negative decisions with no invented findings."""
    return [
        {
            "section_key": k,
            "adequate": False,
            "claim_ids": [],
            "explanation": "No verified analytical judgment.",
        }
        for k in ROLES
    ]


def test_actual_global_response_cannot_borrow_summary_claims_for_body():
    case = real_case()
    original = copy.deepcopy(case)
    payload = case["payload"]
    schema = global_review_schema(payload)
    validator = Draft202012Validator(schema)
    validator.check_schema(schema)
    assert list(validator.iter_errors(case["response"]))
    result = validate_analysis_decisions(
        case["response"]["analyses"],
        payload["assertions"],
        set(case["supported_claim_ids"]),
        payload["report_structure"],
    )
    methods = next(r for r in result["sections"] if r["section_key"] == "methods")
    assert methods["model_adequate"] and not methods["adequate"]
    assert "analysis_wrong_section_claim" in methods["host_validation"]["failure_codes"]
    assert "claim_2" not in methods["host_validation"]["allowed_claim_ids"]
    assert case == original
    # A protocol failure cannot authorize rewriting already supported methods.
    details = {
        "body_analysis_contract": "body-analysis-v1",
        "analysis_check": result,
        "assertions": [
            {
                **c,
                "status": "supported",
                "section": next(
                    r["section"]
                    for r in payload["report_structure"]
                    if r["source_unit_id"] == c["source_unit_id"]
                ),
            }
            for c in payload["assertions"]
        ],
    }
    assert not analysis_repair_diagnostics(details)


def test_complete_negative_response_remains_expressible_and_blocks():
    case = real_case()
    payload = case["payload"]
    negative = {
        **case["response"],
        "analyses": bound_rows(payload),
        "limitations": {
            "adequate": False,
            "claim_ids": [],
            "explanation": "No established limitations.",
        },
    }
    Draft202012Validator(global_review_schema(payload)).validate(negative)
    result = validate_analysis_decisions(
        negative["analyses"], payload["assertions"], set(), payload["report_structure"]
    )
    assert result["status"] == "failed"
    assert all(not r["adequate"] for r in result["sections"])


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_same_finding_and_citation_in_each_section_keep_distinct_claim_ids(language):
    from litagent.review_structure import structure_sections

    titles = dict(structure_sections("academic_v2", language))
    structure = [{"source_unit_id": key, "section": titles[key]} for key in ROLES]
    assertions = [
        {
            "claim_id": f"c-{key}",
            "source_unit_id": key,
            "kind": "scientific",
            "claim_text": "Same finding [E:a]",
        }
        for key in ROLES
    ]
    grouped = analysis_claim_ids_by_section(assertions, structure)
    assert grouped == {key: [f"c-{key}"] for key in ROLES}
    rows = [
        {
            "section_key": k,
            "adequate": True,
            "claim_ids": ids,
            "explanation": "Controlled chapter-local judgment.",
        }
        for k, ids in grouped.items()
    ]
    good = validate_analysis_decisions(
        rows, assertions, {c["claim_id"] for c in assertions}, structure
    )
    assert good["status"] == "passed"
    # Final host scientific checks may revoke an earlier positive model decision.
    again = validate_analysis_decisions(good["sections"], assertions, set(), structure)
    assert again["status"] == "failed"
    assert all(
        "analysis_claim_not_supported" in r["host_validation"]["failure_codes"]
        for r in again["sections"]
    )


@pytest.mark.parametrize(
    "fault", ["duplicate_claim", "duplicate_unit", "unknown_unit", "section_drift"]
)
def test_host_identity_errors_fail_before_provider_submission(fault):
    case = real_case()
    payload = case["payload"]
    if fault == "duplicate_claim":
        payload["assertions"].append(copy.deepcopy(payload["assertions"][0]))
    elif fault == "duplicate_unit":
        payload["report_structure"].append(
            copy.deepcopy(payload["report_structure"][0])
        )
    elif fault == "unknown_unit":
        payload["assertions"][0]["source_unit_id"] = "unknown"
    else:
        payload["assertions"][0]["section"] = "Different section"
    with pytest.raises(ValueError, match="evaluation_global_input_identity_invalid"):
        global_review_schema(payload)


@pytest.mark.parametrize(
    "fault,code",
    [
        ("unknown", "analysis_unknown_claim"),
        ("duplicate", "analysis_duplicate_claim"),
        ("foreign", "analysis_wrong_section_claim"),
        ("unsupported", "analysis_claim_not_supported"),
    ],
)
def test_host_rechecks_model_ids_even_when_native_schema_is_ignored(fault, code):
    case = real_case()
    payload = case["payload"]
    grouped = analysis_claim_ids_by_section(
        payload["assertions"], payload["report_structure"]
    )
    rows = bound_rows(payload)
    row = next(r for r in rows if r["section_key"] == "methods")
    cid = grouped["methods"][0]
    row.update(adequate=True, claim_ids=[cid])
    supported = {c["claim_id"] for c in payload["assertions"]}
    if fault == "unknown":
        row["claim_ids"] = ["invented"]
    elif fault == "duplicate":
        row["claim_ids"] *= 2
    elif fault == "foreign":
        row["claim_ids"] = grouped["abstract"][:1]
    else:
        supported.remove(cid)
    result = validate_analysis_decisions(
        rows, payload["assertions"], supported, payload["report_structure"]
    )
    methods = next(r for r in result["sections"] if r["section_key"] == "methods")
    assert (
        not methods["adequate"] and code in methods["host_validation"]["failure_codes"]
    )


def test_empty_scientific_chapter_cannot_get_a_positive_native_decision():
    case = real_case()
    payload = case["payload"]
    payload["assertions"] = []
    negative = {
        **case["response"],
        "analyses": bound_rows(payload),
        "limitations": {
            "adequate": False,
            "claim_ids": [],
            "explanation": "Missing evidence.",
        },
    }
    validator = Draft202012Validator(global_review_schema(payload))
    validator.validate(negative)
    negative["analyses"][0]["adequate"] = True
    assert list(validator.iter_errors(negative))


@pytest.mark.asyncio
async def test_actual_evaluator_sends_host_maps_and_preserves_negative_science():
    from test_batched_evaluation import materials
    from test_evaluation_wire_contract import AnalysisTransport

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped

    survey, context = materials(2)
    context["body_quality"] = "analysis_v1"
    llm = AnalysisTransport("unsupported")
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    global_payload = llm.calls[-1]
    assert global_payload["analysis_ownership_contract"] == ANALYSIS_OWNERSHIP_CONTRACT
    assert global_payload[
        "analysis_claim_ids_by_section"
    ] == analysis_claim_ids_by_section(
        global_payload["assertions"], global_payload["report_structure"]
    )
    assert result["unsupported_count"] == 1
    assert result["analysis_check"]["status"] == "failed"
    assert len(llm.calls) == 3


@pytest.mark.asyncio
async def test_invalid_global_ownership_retains_science_but_has_no_delivery_score():
    from test_batched_evaluation import materials
    from test_evaluation_wire_contract import AnalysisTransport

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped

    class InvalidOwnership(AnalysisTransport):
        """Simulate a provider ignoring the native schema so host guards are exercised."""

        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "analysis_claim_ids_by_section" in payload:
                result = json.loads(response.content)
                result["analyses"][0].update(
                    adequate=True, claim_ids=[payload["assertions"][0]["claim_id"]]
                )
                response.content = json.dumps(result)
            return response

    survey, context = materials(2)
    context["body_quality"] = "analysis_v1"
    client = InvalidOwnership()
    snapshots = []
    context["evaluation_checkpoint"] = lambda p: snapshots.append(copy.deepcopy(p))
    result = await evaluate_scoped(
        llm=client, survey=survey, context=context, config=EvalConfig()
    )
    assert len(client.calls) == 3
    assert result["score"] is None
    assert result["unverified_reason"] == "evaluation_analysis_claim_ownership_invalid"
    assert not result["coverage"]["complete"]
    assert result["supported_count"] == 2 and result["unsupported_count"] == 0
    assert result["stages"]["global_check"] == "invalid_response"
    assert snapshots[-1]["global_check"] == "invalid_response"
    assert (
        snapshots[-1]["global_reason_code"]
        == "evaluation_analysis_claim_ownership_invalid"
    )
