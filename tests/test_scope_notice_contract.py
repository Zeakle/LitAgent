"""Exercise host scope statements without granting scientific support."""

import copy
import hashlib
import json

import pytest
from test_host_full_source_binding import full_document
from test_writing_passages import facts

from litagent.quality import (
    grounded_messages,
    prepare_grounded_request,
    render_grounded_report_traced,
)
from litagent.writing_passages import (
    bind_scope_notices,
    expand_scope_notice,
    scope_notice_catalog,
    validate_scope_notice_bindings,
)


def notice_document(cards, language="en"):
    """Build v21 input while retaining the original v20 fixture for compatibility."""
    doc = full_document(cards, language)
    for key, section in doc["sections"].items():
        if key in {"abstract", "conclusion"}:
            continue
        for block in [section["first_block"], *section["additional_blocks"]]:
            for unit in [block["first_sentence"], *block["additional_sentences"]]:
                unit["scope_notice_id"] = "none"
    return doc


def render(doc, cards, language="en"):
    bindings = {}
    text = render_grounded_report_traced(
        json.dumps(doc),
        cards,
        "academic_v2",
        language,
        require_native=True,
        require_full_sources=True,
        require_scope_notices=True,
        bindings=bindings,
    )
    return text, bindings


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_notice_binds_exact_section_offsets_and_report_identity(language):
    cards = facts()
    doc = notice_document(cards, language)
    doc["sections"]["introduction"]["first_block"]["additional_sentences"].append(
        {
            "kind": "organization",
            "text": "",
            "fact_card_ids": [],
            "scope_notice_id": "review_scope",
        }
    )
    report, bindings = render(doc, cards, language)
    validate_scope_notice_bindings(report, bindings, language)
    row = next(iter(bindings["scope_notices"].values()))
    assert (
        report[row["source_start"] : row["source_end"]]
        == scope_notice_catalog(language)["review_scope"]
    )
    for field, value in [
        ("source_start", 0),
        ("text_sha256", "fake"),
        ("notice_id", "fake"),
    ]:
        altered = copy.deepcopy(bindings)
        next(iter(altered["scope_notices"].values()))[field] = value
        with pytest.raises(ValueError, match="binding_invalid"):
            validate_scope_notice_bindings(report, altered, language)
    missing = copy.deepcopy(bindings)
    missing["scope_notices"] = {}
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_scope_notice_bindings(report, missing, language)
    with pytest.raises(ValueError, match="binding_invalid"):
        validate_scope_notice_bindings(report + "extra", bindings, language)


@pytest.mark.parametrize(
    "unit",
    [
        None,
        [],
        "text",
        {},
        {
            "text": "Research lacks broad systematic evaluation.",
            "kind": "gap",
            "fact_card_ids": [],
            "scope_notice_id": "evidence_limit",
        },
        {
            "text": "",
            "kind": "gap",
            "fact_card_ids": ["f:bad"],
            "scope_notice_id": "evidence_limit",
        },
        {"text": "", "kind": "gap", "fact_card_ids": [], "scope_notice_id": {}},
        {"text": "", "kind": "gap", "fact_card_ids": [], "scope_notice_id": "unknown"},
    ],
)
def test_arbitrary_scientific_prose_and_malformed_notices_are_rejected(unit):
    with pytest.raises(ValueError, match="scope_notice_invalid"):
        expand_scope_notice(unit, "en")


def test_repeated_body_notice_and_notice_only_limitations_fail():
    cards = facts()
    doc = notice_document(cards)
    notice = {
        "text": "",
        "kind": "gap",
        "fact_card_ids": [],
        "scope_notice_id": "evidence_limit",
    }
    doc["sections"]["open_problems"]["first_block"]["first_sentence"] = notice
    with pytest.raises(ValueError, match="scientific_section_missing"):
        render(doc, cards)
    doc = notice_document(cards)
    for key in ("introduction", "methods"):
        doc["sections"][key]["first_block"]["additional_sentences"].append(
            copy.deepcopy(notice)
        )
    with pytest.raises(ValueError, match="scope_notice_repeated"):
        render(doc, cards)


def test_actual_v21_schema_and_prompt_use_the_same_fields():
    cards = facts()
    prepared, schema = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            "en",
            "EViT",
            use_full_sources=True,
            use_scope_notices=True,
        )
    )
    assert (
        json.loads(prepared[1]["content"])["writing_contract"] == "grounded-writing-v21"
    )
    bodies = [
        x
        for x in schema["json_schema"]["schema"]["$defs"].values()
        if "kind" in x.get("properties", {})
    ]
    assert bodies and all(
        set(x["properties"]) == {"text", "kind", "fact_card_ids", "scope_notice_id"}
        for x in bodies
    )
    assert "ONLY text, kind and fact_card_ids" not in prepared[0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("adequate", [True, False])
async def test_full_evaluation_requires_semantic_limitations_not_merely_a_scientific_sentence(
    adequate,
):
    from test_batched_evaluation import BatchLLM

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import digest, strict_acceptance

    class ScopeLLM(BatchLLM):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "limitations" in payload["schema"].get("properties", {}):
                result = json.loads(response.content)
                section_by_unit = {
                    u["source_unit_id"]: u["section"]
                    for u in payload["report_structure"]
                }
                ids = [
                    c["claim_id"]
                    for c in payload["assertions"]
                    if section_by_unit[c["source_unit_id"]]
                    == "Limitations and Open Problems"
                ]
                result["limitations"] = {
                    "adequate": adequate,
                    "claim_ids": ids if adequate else [],
                    "explanation": (
                        "Concrete applicability boundary."
                        if adequate
                        else "Only an unrelated performance statement."
                    ),
                }
                response.content = json.dumps(result)
            return response

    source = "EViT is limited to image classification."
    cards = facts(source)
    doc = notice_document(cards)
    for key, section in doc["sections"].items():
        if key not in {"abstract", "conclusion"}:
            section["first_block"]["first_sentence"]["text"] = source
    report, bindings = render(doc, cards)
    llm = ScopeLLM()
    result = await evaluate_scoped(
        llm=llm,
        survey=report,
        config=EvalConfig(),
        context={
            "query": "EViT limitations",
            "quality_policy": "strict_auto_v2",
            "evaluation_version": "survey-evaluation-v3",
            "report_language": "en",
            "review_structure": "academic_v2",
            "scope_notice_contract": "scope-notice-v1",
            "writing_bindings": bindings,
            "fact_cards": cards,
            "evidence_snapshot_sha256": digest(cards),
            "evidence": {
                "a": {
                    "evidence_id": "a",
                    "paper_id": "evit",
                    "text": source,
                    "supporting_text": source,
                    "chunk_key": "abstract",
                }
            },
            "coverage_requirements": [
                {
                    "requirement_id": "question",
                    "kind": "question",
                    "text": "EViT limitations",
                }
            ],
        },
    )
    assert len(llm.calls) == 3
    result.update(
        structure_checks={"status": "passed"},
        method_checks={"status": "not_applicable"},
        language_checks={"status": "passed", "requested": "en"},
    )
    failure_codes = strict_acceptance(result)["failure_codes"]
    assert ("scientific_limitations_missing" in failure_codes) is (not adequate)
    assert result["coverage"]["complete"]


@pytest.mark.asyncio
async def test_fake_notice_binding_stops_before_any_evaluation_request():
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litagent.config import EvalConfig
    from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped

    llm = SimpleNamespace(chat=AsyncMock())
    with pytest.raises(ScopedEvaluationError):
        await evaluate_scoped(
            llm=llm,
            survey="invented",
            config=EvalConfig(),
            context={
                "quality_policy": "strict_auto_v2",
                "scope_notice_contract": "scope-notice-v1",
                "writing_bindings": {},
                "report_language": "en",
            },
        )
    llm.chat.assert_not_awaited()


@pytest.mark.parametrize("ids", [["invented"], ["c1", "c1"], ["c2"], []])
def test_limitations_cannot_use_unknown_duplicate_or_other_section_claims(ids):
    from litagent.eval.batched import ScopeGlobalReview, valid_limitations

    result = ScopeGlobalReview(
        units=[],
        conflicts=[],
        issues=[],
        coverage=[],
        limitations={
            "adequate": True,
            "claim_ids": ids,
            "explanation": "Claimed limitation.",
        },
    )
    claims = [
        {
            "claim_id": "c1",
            "section": "Limitations and Open Problems",
            "kind": "scientific",
        },
        {"claim_id": "c2", "section": "Introduction", "kind": "scientific"},
    ]
    assert not valid_limitations(result, claims, {"c1", "c2"})
