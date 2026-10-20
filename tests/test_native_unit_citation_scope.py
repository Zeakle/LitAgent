"""Stop native citation-scope conflicts before paid review and evaluation."""

import copy
import json
from pathlib import Path

import pytest
from test_batched_evaluation import materials
from test_body_analysis import analysis_document, facts
from test_evaluation_wire_contract import AnalysisTransport

from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped
from litagent.quality import (
    GroundedWritingError,
    apply_native_repairs,
    grounded_messages,
    prepare_native_repair_request,
    render_grounded_report_traced,
)
from litagent.writing_passages import validate_unit_citation_scope


def render(candidate, cards, *, enabled=True, language="en"):
    """Use actual native compilation with an explicit new-contract opt-in."""
    return render_grounded_report_traced(
        candidate,
        cards,
        "academic_v2",
        language,
        require_native=True,
        require_full_sources=True,
        require_scope_notices=True,
        analysis_mode=True,
        require_unit_citation_scope=enabled,
    )


def test_real_received_body_exposes_all_eight_uncited_scientific_clauses():
    """Replay public native text; neither add references nor rescore the archive."""
    f = json.loads(Path("tests/fixtures/native-unit-scope-real.json").read_text())
    before = copy.deepcopy(f)
    assert len(f["body_claims"]) == 8
    old = render(f["candidate"], f["cards"], enabled=False, language=f["language"])
    with pytest.raises(GroundedWritingError) as error:
        render(f["candidate"], f["cards"], language=f["language"])
    diagnostics = [
        d
        for d in error.value.diagnostics
        if d["reason_code"] == "strict_writing_unit_citation_scope_ambiguous"
    ]
    assert len(diagnostics) == 7
    assert any(d["location"].startswith("conclusion.") for d in error.value.diagnostics)
    assert all(d["raw_json_path"].startswith("sections.") for d in diagnostics)
    assert all(d["actual_fact_card_ids"] for d in diagnostics)
    for claim in f["body_claims"]:
        assert claim["source_quote"] in old
        assert any(
            claim["source_quote"] in d["candidate_text"]
            for d in error.value.diagnostics
        )
    assert f == before and f["scientific_verdict"] == "not_verified"


@pytest.mark.parametrize(
    "text",
    [
        "A result improves accuracy; another result reduces memory",
        "方法提高准确率；另一个结果降低显存",
    ],
)
def test_exposed_semicolon_does_not_inherit_a_future_reference(text):
    with pytest.raises(ValueError, match="unit_citation_scope_ambiguous"):
        validate_unit_citation_scope(text, ["a"], "en")


@pytest.mark.parametrize(
    "text",
    [
        "On DeiT-S, EViT improves speed by 50%",
        "Accuracy decreases by 0.3 percentage points",
        "The setting (shots=1; dataset=D) limits this result",
        'The source calls the setting "one shot; fixed data"',
        "The label `A;B` is quoted",
        "The source is https://example.test/a;b",
    ],
)
def test_conditions_lists_numbers_and_protected_semicolons_remain_legal(text):
    validate_unit_citation_scope(text, ["a"], "en")


def test_native_repair_restores_the_scientific_unit_without_editing_other_prose():
    """Use compile, target freezing, patch validation and recompilation together."""
    cards = facts()
    doc = analysis_document(cards)
    expected = copy.deepcopy(doc)
    unit = doc["sections"]["methods"]["first_block"]["first_sentence"]
    original_text = unit["text"]
    unit["text"] += "; the same finding is repeated"
    candidate = json.dumps(doc)
    messages = grounded_messages(
        cards,
        "academic_v2",
        "en",
        "Explain EViT",
        use_full_sources=True,
        use_scope_notices=True,
        analysis_mode=True,
        require_unit_citation_scope=True,
    )
    with pytest.raises(GroundedWritingError) as failure:
        render(candidate, cards)
    prompt, schema, binding = prepare_native_repair_request(
        messages,
        candidate,
        failure.value.diagnostics,
        cards=cards,
        structure="academic_v2",
        language="en",
    )
    assert "unit-citation-scope-v1" in prompt[0]["content"]
    assert len(binding["targets"]) == 1
    key, target = next(iter(binding["targets"].items()))
    assert target["must_remain_scientific"]
    props = schema["json_schema"]["schema"]["properties"]["patches"]["properties"][key][
        "properties"
    ]
    assert props["kind"]["enum"] == ["scientific"]
    patch = {**copy.deepcopy(unit), "text": original_text}
    response = json.dumps({"can_repair": True, "patches": {key: patch}})
    repaired, _ = apply_native_repairs(candidate, response, binding, cards=cards)
    assert json.loads(repaired) == expected
    render(repaired, cards)
    patch.update(
        kind="gap", text="", fact_card_ids=[], scope_notice_id="evidence_limit"
    )
    with pytest.raises(ValueError, match="strict_native_repair_patch_invalid"):
        apply_native_repairs(
            candidate,
            json.dumps({"can_repair": True, "patches": {key: patch}}),
            binding,
            cards=cards,
        )


def test_summary_scope_failure_requires_explicit_summary_repair_target():
    """Authorize only the actual failing summary, retaining all body units."""
    cards = facts()
    doc = analysis_document(cards)
    unit = doc["sections"]["abstract"]["first_block"]["first_sentence"]
    unit["text"] += "; another independently testable finding"
    candidate = json.dumps(doc)
    with pytest.raises(GroundedWritingError) as failure:
        render(candidate, cards)
    assert any(
        d["location"].startswith("abstract.")
        and d["reason_code"] == "strict_writing_unit_citation_scope_ambiguous"
        for d in failure.value.diagnostics
    )
    messages = grounded_messages(
        cards,
        "academic_v2",
        "en",
        "Explain EViT",
        use_full_sources=True,
        use_scope_notices=True,
        analysis_mode=True,
        require_unit_citation_scope=True,
    )
    _, schema, binding = prepare_native_repair_request(
        messages,
        candidate,
        failure.value.diagnostics,
        cards=cards,
        structure="academic_v2",
        language="en",
    )
    assert len(binding["targets"]) == 1
    key, target = next(iter(binding["targets"].items()))
    assert target["unit_type"] == "summary"
    assert target["path"] == "sections.abstract.first_block.first_sentence"
    fields = schema["json_schema"]["schema"]["properties"]["patches"]["properties"][
        key
    ]["properties"]
    assert set(fields) == {"text", "body_unit_ids", "additional_units"}
    assert (
        "additional_units"
        not in schema["json_schema"]["schema"]["properties"]["patches"]["properties"][
            key
        ]["required"]
    )
    assert target["allowed_body_unit_ids"]


class AtomicTransport(AnalysisTransport):
    """Keep a supported premise separate from a faithfully extracted false one."""

    async def chat(self, messages, **kwargs):
        response = await super().chat(messages, **kwargs)
        payload = json.loads(messages[1]["content"])
        result = json.loads(response.content)
        if "allowed_citation_aliases" in payload:
            for fragment in result["fragments"].values():
                fragment["assertions"] = [
                    {"kind": "scientific", "claim_text": "ViT uses image patches"},
                    {"kind": "scientific", "claim_text": "ViT has unbounded accuracy"},
                ]
        elif "cited_materials" in payload:
            result["decisions"][1].update(
                status="unsupported",
                reason_code="insufficient_support",
                support_basis=[],
            )
            for aspect in result["decisions"][1]["aspect_checks"]:
                aspect.update(supported=False, support_ids=[])
        response.content = json.dumps(result)
        return response


@pytest.mark.asyncio
async def test_one_fragment_can_emit_multiple_atomic_claims_without_rescuing_falsehood():
    """Exercise real inventory and batch consumers while replacing only transport."""
    _, context = materials(1)
    context["body_quality"] = "analysis_v1"
    llm = AtomicTransport()
    result = await evaluate_scoped(
        llm=llm,
        survey="## Representative Methods\n\nViT uses image patches and has unbounded accuracy [E:a0].",
        context=context,
        config=EvalConfig(),
    )
    assert result["factual_claim_count"] == 2
    assert result["unsupported_count"] == 1 and result["score"] < 1.0
    a, b = result["assertions"]
    assert a["source_fragment_id"] == b["source_fragment_id"]
    assert a["claim_id"] != b["claim_id"]
    assert a["evidence_ids"] == b["evidence_ids"] == ["a0"]
