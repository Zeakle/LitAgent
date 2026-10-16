"""Exercise analytical writing contracts through real compilers and evaluators."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_optional_verification import switches
from test_scope_notice_contract import notice_document
from test_writing_passages import facts

from litagent.body_quality import (
    ANALYSIS_CONTRACT,
    ANALYSIS_VERSION,
    ROLES,
    body_diagnostics,
    build_analysis_plan,
    validate_analysis_decisions,
)
from litagent.quality import (
    GroundedWritingError,
    grounded_messages,
    prepare_grounded_request,
    render_grounded_report_traced,
    strict_acceptance,
)
from litagent.workbench import RunOptions
from litagent.writing_passages import compile_passage_report


def analysis_document(cards, language="en"):
    """Keep the real v21 body and use the explicit v22 summary producer."""
    doc = notice_document(cards, language)
    for key in ("abstract", "conclusion"):
        for block in [
            doc["sections"][key]["first_block"],
            *doc["sections"][key]["additional_blocks"],
        ]:
            for sentence in [block["first_sentence"], *block["additional_sentences"]]:
                reference = sentence.pop("body_reference")
                section = doc["sections"][reference["section_key"]]
                b = (
                    section["first_block"]
                    if reference["block_index"] == 0
                    else section["additional_blocks"][reference["block_index"] - 1]
                )
                unit = (
                    b["first_sentence"]
                    if reference["sentence_index"] == 0
                    else b["additional_sentences"][reference["sentence_index"] - 1]
                )
                sentence.update(text=unit["text"], body_references=[reference])
    return doc


def render_analysis(doc, cards, language="en"):
    """Render the new summary codec through the production trace wrapper."""
    bindings = {}
    text = render_grounded_report_traced(
        json.dumps(doc),
        cards,
        "academic_v2",
        language,
        require_native=True,
        require_full_sources=True,
        require_scope_notices=True,
        analysis_mode=True,
        bindings=bindings,
    )
    return text, bindings


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_native_schema_and_renderer_share_the_same_summary_contract(language):
    """Exercise native schema and renderer share the same summary contract."""
    cards = facts()
    messages, schema = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            language,
            "EViT speed",
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
        )
    )
    payload = json.loads(messages[1]["content"])
    assert payload["writing_contract"] == ANALYSIS_VERSION
    assert payload["analysis_plan"]["requirements_status"] == "not_verified"
    definitions = schema["json_schema"]["schema"]["$defs"]
    summary_types = [
        x for x in definitions.values() if "body_references" in x.get("properties", {})
    ]
    assert len(summary_types) == 2
    assert all(
        set(x["properties"]) == {"text", "body_references"} for x in summary_types
    )
    text, bindings = render_analysis(
        analysis_document(cards, language), cards, language
    )
    assert text.count("[E:a]") >= 7
    assert bindings["body_analysis_contract"] == ANALYSIS_CONTRACT
    assert bindings["summary_derivations"]


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "summary_target",
        "duplicate",
        "empty",
        "new_card",
        "free_kind",
        "notice",
    ],
)
def test_summary_cannot_introduce_sources_or_use_organization_as_science(fault):
    """Exercise summary cannot introduce sources or use organization as science."""
    cards = facts()
    doc = analysis_document(cards)
    s = doc["sections"]["abstract"]["first_block"]["first_sentence"]
    if fault == "unknown":
        s["body_references"][0]["block_index"] = 90
    elif fault == "summary_target":
        s["body_references"][0]["section_key"] = "conclusion"
    elif fault == "duplicate":
        s["body_references"] *= 2
    elif fault == "empty":
        s["body_references"] = []
    elif fault == "new_card":
        s["fact_card_ids"] = ["invented"]
    elif fault == "free_kind":
        s["kind"] = "organization"
    else:
        ref = s["body_references"][0]
        doc["sections"][ref["section_key"]]["first_block"]["first_sentence"] = {
            "text": "",
            "kind": "gap",
            "fact_card_ids": [],
            "scope_notice_id": "evidence_limit",
        }
    with pytest.raises((GroundedWritingError, ValueError)):
        render_analysis(doc, cards)


def test_rephrased_summary_preserves_original_support_and_is_not_auto_verified():
    """Exercise rephrased summary preserves original support and is not auto verified."""
    cards = facts()
    doc = analysis_document(cards)
    summary = doc["sections"]["abstract"]["first_block"]["first_sentence"]
    summary["text"] = "On DeiT-S, EViT yields a speed improvement of 50%."
    normalized, spans = compile_passage_report(
        json.dumps(doc),
        cards,
        binding_mode="full_sources",
        require_scope_notices=True,
        language="en",
        analysis_mode=True,
    )
    value = json.loads(normalized)["sections"]["abstract"]["first_block"][
        "first_sentence"
    ]
    assert value["text"] == summary["text"]
    assert value["fact_card_ids"] == list(cards)
    assert (
        spans["abstract.blocks.0.sentences.0"][0]["verification"]
        == "source_location_only"
    )
    assert value["support_quotes"] == [next(iter(cards.values()))["source_quote"]]


def test_body_counts_exclude_appendix_and_do_not_claim_a_quality_score():
    """Exercise body counts exclude appendix and do not claim a quality score."""
    text = (
        "# Report\n\n## Methods\n\nA claim [E:a].\n\n## Appendix A: Search\n\n"
        + "log " * 100
    )
    d = body_diagnostics(text)
    assert d["academic_characters"] < 100
    assert d["non_body_characters"] > 300
    assert d["interpretation"] == "diagnostics_only_not_a_quality_score"


def test_preparation_is_bound_to_originals_but_does_not_assert_coverage():
    """Exercise preparation is bound to originals but does not assert coverage."""
    cards = facts()
    plan = build_analysis_plan(
        "Compare EViT", [{"requirement_id": "question", "text": "Compare EViT"}], cards
    )
    assert plan["requirements_status"] == "not_verified"
    assert all(r["semantic_status"] == "not_verified" for r in plan["sources"])
    assert plan["missing_fields"]["conditions"] == list(cards)
    assert set(r["section_key"] for r in plan["section_duties"]) == set(ROLES)
    with pytest.raises(ValueError):
        build_analysis_plan("Question", [], cards, {"methods": ["unknown"]})


def test_depth_requires_supported_science_in_the_actual_section():
    """Exercise depth requires supported science in the actual section."""
    from litagent.review_structure import structure_sections

    titles = dict(structure_sections("academic_v2", "en"))
    claims = [
        {"claim_id": k, "kind": "scientific", "section": titles[k]} for k in ROLES
    ]
    decisions = [
        {
            "section_key": k,
            "adequate": True,
            "claim_ids": [k],
            "explanation": "Mechanism and limitations supported.",
        }
        for k in ROLES
    ]
    result = validate_analysis_decisions(decisions, claims, set(ROLES), [])
    assert result["status"] == "passed"
    for fault in ("unsupported", "organization", "wrong_section"):
        altered = copy.deepcopy(claims)
        supported = set(ROLES)
        if fault == "unsupported":
            supported.remove("methods")
        elif fault == "organization":
            altered[2]["kind"] = "report_structure"
        else:
            altered[2]["section"] = titles["abstract"]
        assert (
            validate_analysis_decisions(decisions, altered, supported, [])["status"]
            == "failed"
        )
    with pytest.raises(ValueError):
        validate_analysis_decisions(decisions[:-1], claims, set(ROLES), [])


def test_analysis_options_change_identity_and_disabled_audits_stay_optional():
    """Exercise analysis options change identity and disabled audits stay optional."""
    old = RunOptions(quality_policy="strict_auto_v2", verification=switches())
    new = RunOptions(
        quality_policy="strict_auto_v2",
        verification=switches(),
        body_quality="analysis_v1",
    )
    assert new.fingerprint() != old.fingerprint()
    assert (
        not new.verification.judge
        and not new.verification.jev
        and new.auto_audit_mode == "off"
    )
    with pytest.raises(ValueError):
        RunOptions(body_quality="analysis_v1")


@pytest.mark.asyncio
async def test_real_synthesis_uses_the_analysis_request_and_new_compiler():
    """Exercise real synthesis uses the analysis request and new compiler."""
    from test_extraction_verification import selection

    from litagent.agents.synthesis import SynthesisWorker
    from litagent.llm.client import LLMResponse
    from litagent.orchestrator.task_graph import SubTask
    from litagent.quality import build_fact_cards

    selected = selection()
    cards = build_fact_cards(selected.selected_items)
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(analysis_document(cards)), model="offline", usage={}
    )
    selector = AsyncMock()
    selector.select.return_value = selected
    worker = SynthesisWorker(llm, evidence_selector=selector)
    result = await worker.execute(
        SubTask(
            "synthesis",
            "test",
            "synthesis",
            input_data={
                "query": "EViT",
                "review_structure": "academic_v2",
                "quality_policy": "strict_auto_v2",
                "verification": switches(),
                "body_quality": "analysis_v1",
            },
        )
    )
    assert result["body_quality"] == "analysis_v1"
    assert result["analysis_plan"]["requirements_status"] == "not_verified"
    assert result["writing_bindings"]["body_analysis_contract"] == ANALYSIS_CONTRACT
    assert (
        llm.chat.call_args.kwargs["response_format"]["json_schema"]["name"]
        == "grounded_report_v22"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "inadequate",
        "unknown_claim",
        "omit",
        "wrong_number",
        "empty_global",
        "inventory_nesting",
    ],
)
async def test_real_evaluation_adds_analysis_to_existing_global_call(fault):
    """Exercise real evaluation adds analysis to existing global call."""
    from test_batched_evaluation import BatchLLM

    from litagent.body_quality import bound_repair_diagnostics
    from litagent.config import EvalConfig
    from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped
    from litagent.llm.client import LLMResponse
    from litagent.quality import digest
    from litagent.review_structure import structure_sections

    source = "EViT improves speed by 50% on DeiT-S."
    cards = facts(source)
    doc = analysis_document(cards)
    for key in ROLES:
        if key not in {"abstract", "conclusion"}:
            doc["sections"][key]["first_block"]["first_sentence"]["text"] = source
    if fault == "wrong_number":
        doc["sections"]["abstract"]["first_block"]["first_sentence"][
            "text"
        ] = "EViT improves speed by 70% on DeiT-S."
    report, bindings = render_analysis(doc, cards)

    class AnalysisLLM(BatchLLM):
        async def chat(self, messages, **kwargs):
            """Keep batch and global response contracts separate in the fake transport."""
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            if "allowed_citation_aliases" in payload:
                native = kwargs["response_format"]["json_schema"]
                assert native["name"] == "host_fragment_inventory_v2"
                assert native["schema"] == payload["schema"]
                assert native["strict"] is True
                if fault == "inventory_nesting":
                    result = json.loads(response.content)
                    first, second, *_ = result["units"]
                    result["units"][first][second] = result["units"].pop(second)
                    response.content = json.dumps(result)
            if "analyses" in payload.get("schema", {}).get("properties", {}):
                assert "unit_inventory" not in payload
                assert payload["expected_source_unit_ids"] == [
                    u["source_unit_id"] for u in payload["source_units"]
                ]
                assert kwargs["response_format"]["type"] == "json_schema"
                assert (
                    kwargs["response_format"]["json_schema"]["schema"]
                    == payload["schema"]
                )
                result = json.loads(response.content)
                section_by_unit = {
                    u["source_unit_id"]: u["section"]
                    for u in payload["report_structure"]
                }
                titles = dict(structure_sections("academic_v2", "en"))
                grouped = {
                    k: [
                        c["claim_id"]
                        for c in payload["assertions"]
                        if section_by_unit[c["source_unit_id"]] == titles[k]
                    ]
                    for k in ROLES
                }
                result["limitations"] = {
                    "adequate": True,
                    "claim_ids": grouped["open_problems"],
                    "explanation": "Fixture concrete boundary.",
                }
                result["analyses"] = [
                    {
                        "section_key": k,
                        "adequate": not (fault == "inadequate" and k == "methods"),
                        "claim_ids": grouped[k],
                        "explanation": "Located analytical decision.",
                    }
                    for k in ROLES
                ]
                if fault == "unknown_claim":
                    result["analyses"][2]["claim_ids"] = ["unknown"]
                if fault == "omit":
                    result["analyses"].pop()
                if fault == "empty_global":
                    result.update(units=[], coverage=[], analyses=[])
                response.content = json.dumps(result)
            return response

    llm = AnalysisLLM()
    context = {
        "query": "EViT mechanism",
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
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
            {"requirement_id": "question", "kind": "question", "text": "EViT mechanism"}
        ],
    }
    if fault in {"omit", "empty_global", "inventory_nesting"}:
        with pytest.raises(ScopedEvaluationError):
            await evaluate_scoped(
                llm=llm, survey=report, context=context, config=EvalConfig()
            )
        return
    try:
        result = await evaluate_scoped(
            llm=llm, survey=report, context=context, config=EvalConfig()
        )
    except ScopedEvaluationError:
        assert fault == "wrong_number"
        return
    if fault == "wrong_number":
        assert not strict_acceptance(result)["status"] == "passed"
        return
    assert len(llm.calls) == 3
    assert result["body_analysis_contract"] == ANALYSIS_CONTRACT
    assert result["analysis_check"]["status"] == (
        "passed" if fault is None else "failed"
    )
    if fault == "inadequate":
        assert bound_repair_diagnostics(result)
        assert all(
            d["preserve_supported_claim"] for d in bound_repair_diagnostics(result)
        )
    result.update(
        structure_checks={"status": "passed"},
        method_checks={"status": "not_applicable"},
        language_checks={"status": "passed", "requested": "en"},
    )
    assert (
        "body_analysis_incomplete" in strict_acceptance(result)["failure_codes"]
    ) == (fault in {"inadequate", "unknown_claim"})


def test_real_empty_global_response_is_rejected_by_frozen_identity_schema():
    """Replay the actual empty response and accept explicit negative coverage instead."""
    from pathlib import Path

    from jsonschema import Draft202012Validator

    from litagent.eval.batched import global_review_schema

    fixture = json.loads(
        Path("tests/fixtures/analysis-global-empty-real.json").read_text()
    )
    schema = global_review_schema(
        {**fixture, "body_analysis_contract": ANALYSIS_CONTRACT}
    )
    validator = Draft202012Validator(schema)
    assert len(list(validator.iter_errors(fixture["empty_response"]))) >= 3
    negative = {
        **fixture["empty_response"],
        "units": [
            {
                "source_unit_id": u["source_unit_id"],
                "complete": False,
                "explanation": "Missing assertions.",
            }
            for u in fixture["source_units"]
        ],
        "coverage": [
            {
                "requirement_id": r["requirement_id"],
                "answered": False,
                "claim_ids": [],
                "explanation": "Unanswered.",
            }
            for r in fixture["coverage_requirements"]
        ],
        "analyses": [
            {
                "section_key": k,
                "adequate": False,
                "claim_ids": [],
                "explanation": "Inadequate.",
            }
            for k in ROLES
        ],
    }
    validator.validate(negative)
    negative["units"][0]["source_unit_id"] = "unknown"
    assert list(validator.iter_errors(negative))


def test_real_nested_inventory_is_rejected_without_reassigning_source_units():
    """Replay malformed host partitions; native schema never repairs or guesses ownership."""
    from pathlib import Path

    from jsonschema import Draft202012Validator

    from litagent.eval.scoped import (
        _strict_inventory_response,
        _strict_inventory_schema,
    )

    fixture = json.loads(
        Path("tests/fixtures/analysis-inventory-nesting-real.json").read_text()
    )
    schema = _strict_inventory_schema(fixture["source_units"])
    validator = Draft202012Validator(schema)
    assert list(validator.iter_errors(fixture["nested_response"]))
    with pytest.raises(ValueError, match="evaluation_inventory_unit_coverage_invalid"):
        _strict_inventory_response(
            json.dumps(fixture["nested_response"]), fixture["source_units"], {}
        )
    complete = {
        "units": {
            u["source_unit_id"]: {
                f["source_fragment_id"]: {
                    "assertions": [
                        {
                            "kind": "scientific",
                            "claim_text": "Host-owned scientific fragment.",
                        }
                    ],
                    "exclusion_reason": "",
                }
                for f in u["source_fragments"]
            }
            for u in fixture["source_units"]
        }
    }
    validator.validate(complete)
    inventory = _strict_inventory_response(
        json.dumps(complete), fixture["source_units"], {}
    )
    assert len(inventory.units) == len(fixture["source_units"]) == 7


def test_native_repair_preserves_new_summary_dependencies():
    """Exercise native repair preserves new summary dependencies."""
    from litagent.quality import apply_native_repairs, prepare_native_repair_request

    cards = facts()
    doc = analysis_document(cards)
    doc["sections"]["abstract"]["first_block"]["first_sentence"]["body_references"] = [
        {"section_key": "open_problems", "block_index": 0, "sentence_index": 0}
    ]
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
    prepared, _ = prepare_grounded_request(messages)
    raw = json.dumps(doc)
    try:
        render_analysis(doc, cards)
    except GroundedWritingError as failure:
        diagnostics = failure.diagnostics
    else:
        raise AssertionError("uncited research-gap fixture was accepted")
    _, _, binding = prepare_native_repair_request(
        prepared, raw, diagnostics, cards=cards, structure="academic_v2", language="en"
    )
    patches = {
        k: {
            "kind": "scientific",
            "text": "EViT improves speed by 50% on DeiT-S.",
            "fact_card_ids": list(cards),
            "scope_notice_id": "none",
        }
        for k in binding["targets"]
    }
    changed, _ = apply_native_repairs(
        raw, json.dumps({"can_repair": True, "patches": patches}), binding, cards=cards
    )
    _, bound = render_analysis(json.loads(changed), cards)
    assert bound["body_analysis_contract"] == ANALYSIS_CONTRACT


def test_frozen_consumer_retains_all_unverified_sources_when_jev_is_disabled():
    """Exercise frozen consumer retains all unverified sources when jev is disabled."""
    from test_extraction_verification import selection

    from litagent.quality import build_fact_cards, digest
    from scripts.run_frozen_generation import validate_frozen_worker_evidence

    selected = selection()
    source = selected.to_dict()
    for key in ("abstract", "conclusion"):
        selected.section_evidence_ids[key] = list(selected.selected_items)
        selected.method_by_section[key] = "selected_union"
    cards = build_fact_cards(selected.selected_items)
    result = {
        "fact_cards": cards,
        "evidence_snapshot_sha256": digest(cards),
        "evidence_selection": selected.to_dict(),
    }
    assert (
        validate_frozen_worker_evidence(
            result,
            {
                "fact_cards": cards,
                "evidence_selection": source,
                "review_structure": "academic_v2",
            },
            require_verification=False,
        )
        == selected.selected_items
    )
    with pytest.raises(ValueError):
        validate_frozen_worker_evidence(
            result,
            {
                "fact_cards": cards,
                "evidence_selection": source,
                "review_structure": "academic_v2",
            },
            require_verification=True,
        )
