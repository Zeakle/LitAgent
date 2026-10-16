"""Prevent extracted self-proof and exercise complete strict writing boundaries."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_batched_evaluation import BatchLLM, materials
from test_strict_quality import draft, to_native
from test_support_passages import selected

from litagent.agents.extraction_strategy import build_extraction_context
from litagent.agents.reviewer import ReviewerWorker
from litagent.config import EvalConfig, load_config
from litagent.contracts import build_config_summary
from litagent.eval.scoped import (
    build_support_passages,
    evaluate_scoped,
    validate_support_basis,
)
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    build_fact_cards,
    digest,
    grounded_messages,
    render_grounded_report,
)
from litagent.review_structure import assemble_methodology, structure_sections


def source_pair(claim_text, original):
    """Place a deliberately wrong extraction beside a real original source."""
    claim = {"claim_id": "c", "claim_text": claim_text, "evidence_ids": ["a"]}
    ledger = {"a": {"text": claim_text, "supporting_text": original}}
    return claim, ledger, {"a": "e1"}


@pytest.mark.parametrize(
    "claim_text, original",
    [
        ("Model-A accuracy is 90%.", "Model-A accuracy is 70%."),
        ("Evo-ViT improves speed by 50%.", "EViT improves speed by 50%."),
        ("DeiT uses distillation.", "ConViT uses soft convolutional inductive biases."),
        ("Model-A improves speed by 50%.", "Model-A reduces FLOPs by 50%."),
    ],
)
def test_extracted_claim_cannot_prove_itself(claim_text, original):
    claim, ledger, aliases = source_pair(claim_text, original)
    decision, passages = selected(claim, ledger, aliases, source="evidence_text")
    assert (
        validate_support_basis(claim, decision, ledger, aliases, passages=passages)[
            "status"
        ]
        == "passed"
    )
    strict = validate_support_basis(
        claim, decision, ledger, aliases, passages=passages, original_only=True
    )
    assert (
        strict["status"] == "unverified"
        and strict["issue"] == "support_basis_not_original"
    )
    actual, passages = selected(claim, ledger, aliases)
    assert (
        validate_support_basis(
            claim, actual, ledger, aliases, passages=passages, original_only=True
        )["status"]
        == "unverified"
    )


@pytest.mark.parametrize("original", [None, "", "   "])
def test_missing_original_never_creates_strict_support_from_extraction(original):
    _, ledger, aliases = source_pair("Model-A accuracy is 90%.", original)
    assert build_support_passages(ledger, aliases, original_only=True) == {}
    assert build_support_passages(ledger, aliases)


def test_actual_original_remains_locatable_and_supported():
    claim, ledger, aliases = source_pair(
        "Model-A accuracy is 70%.", "Model-A accuracy is 70%."
    )
    decision, passages = selected(claim, ledger, aliases)
    result = validate_support_basis(
        claim, decision, ledger, aliases, passages=passages, original_only=True
    )
    assert result["status"] == "passed"
    assert all(p["source"] == "original_support" for p in result["spans"])


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
@pytest.mark.parametrize("incorrect", [False, True])
async def test_real_v3_requests_only_original_and_rejects_wrong_extraction(
    policy, incorrect
):
    _, context = materials(1)
    context["quality_policy"] = policy
    context["evidence"]["a0"].update(
        text="Model-A accuracy is 90%.", supporting_text="Model-A accuracy is 70%."
    )
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    survey = (
        "# Report\n\n## Methods\n\nModel-A accuracy is "
        + ("90%" if incorrect else "70%")
        + ". [E:a0]"
    )
    llm = BatchLLM()
    result = await evaluate_scoped(
        llm=llm, survey=survey, context=context, config=EvalConfig()
    )
    batch = next(p for p in llm.calls if "cited_materials" in p)
    assert all(m["text"] == "" for m in batch["cited_materials"].values())
    assert all(p[1] != p[0] for p in batch["support_passages"].values())
    assert "90%" not in json.dumps(batch["original_supports"])
    science = [c for c in result["assertions"] if c["kind"] == "scientific"]
    assert science and all(
        c["status"] == ("unverifiable" if incorrect else "supported") for c in science
    )
    if incorrect:
        assert science[0]["support_validation"]["issue"] == "support_number_unverified"


def writing_material():
    ledger = {
        "a": {
            "evidence_id": "a",
            "paper_id": "evit",
            "text": "EViT improves speed.",
            "supporting_text": "EViT improves speed.",
            "chunk_key": "abstract",
        }
    }
    cards = build_fact_cards(ledger)
    selection = {
        "section_evidence_ids": {k: ["a"] for k, _ in structure_sections("academic_v2")}
    }
    requirements = [
        {
            "requirement_id": "question",
            "kind": "question",
            "text": "Compare EViT under its reported settings",
        },
        {"requirement_id": "evit", "kind": "literal", "text": "EViT"},
    ]
    return cards, selection, requirements


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
def test_shared_writing_contract_keeps_responsibilities_question_and_allowed_cards(
    language, structure
):
    cards, selection, requirements = writing_material()
    packet = json.loads(
        grounded_messages(
            cards,
            structure,
            language,
            requirements[0]["text"],
            coverage=requirements,
            evidence_selection=selection,
        )[1]["content"]
    )
    expected = [
        (k, t)
        for k, t in structure_sections(structure, language)
        if k != "run_methodology"
    ]
    assert [
        (s["section_key"], s["title"]) for s in packet["section_contract"]
    ] == expected
    assert packet["coverage_requirements"] == requirements
    assert packet["report_language"] == language
    assert packet["summary_rules"]["max_sentence_units"] == 6
    assert all(
        s["responsibility"] and s["allowed_fact_card_ids"] == list(cards)
        for s in packet["section_contract"]
    )
    assert packet["coverage_matrix"]["requirements"][1]["status"] == "source_available"
    assert all(c["semantic_status"] == "not_verified" for c in cards.values())
    assert (
        render_grounded_report(
            json.dumps(draft(cards, language)),
            cards,
            "academic_v2",
            language,
            evidence_selection=selection,
        ).count("[E:a]")
        == 7
    )


def test_card_outside_section_is_blocked_and_missing_source_is_exposed():
    cards, selection, requirements = writing_material()
    selection["section_evidence_ids"]["experiments"] = []
    packet = json.loads(
        grounded_messages(
            cards,
            "academic_v2",
            "en",
            "EViT",
            coverage=requirements,
            evidence_selection=selection,
        )[1]["content"]
    )
    assert (
        next(
            s for s in packet["section_contract"] if s["section_key"] == "experiments"
        )["allowed_fact_card_ids"]
        == []
    )
    with pytest.raises(ValueError, match="strict_writing_out_of_scope_fact_card"):
        render_grounded_report(
            json.dumps(draft(cards)),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
async def test_reviewer_receives_same_frozen_question_and_scope(revision, structure):
    from litagent.context.evidence_selector import EvidenceSelection

    cards, selection, requirements = writing_material()
    ledger = {
        "a": {
            "evidence_id": "a",
            "paper_id": "evit",
            "text": "EViT improves speed.",
            "supporting_text": "EViT improves speed.",
            "chunk_key": "abstract",
        }
    }
    selection = EvidenceSelection(
        1,
        ledger,
        selection["section_evidence_ids"],
        {
            k: (
                "selected_union"
                if k in ("abstract", "conclusion")
                else "lexical_fallback"
            )
            for k in selection["section_evidence_ids"]
        },
        0,
        20,
    )
    body = render_grounded_report(
        json.dumps(draft(cards)),
        cards,
        "academic_v2",
        "en",
        evidence_selection=selection,
    )
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(
            {
                "score": 0.9,
                "strengths": [],
                "weaknesses": [],
                "issues": [],
                "missing_coverage": [],
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
                "language_compliance": {"passed": True, "reason": "English"},
            }
        ),
        model="offline",
        usage={},
    )
    worker = ReviewerWorker(llm)
    kwargs = dict(
        query=requirements[0]["text"],
        review_structure=structure,
        quality_policy="strict_auto_v1",
        coverage_requirements=requirements,
    )
    if revision:
        await worker.review_revision(body, {}, selection, **kwargs)
    else:
        await worker.execute(
            SubTask(
                "review",
                "review",
                "reviewer",
                input_data={
                    **kwargs,
                    "upstream_results": {
                        "synthesis": {
                            "draft": body,
                            "evidence_selection": selection.to_dict(),
                            "coverage_requirements": requirements,
                        }
                    },
                },
            )
        )
    packet = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert packet["coverage_requirements"] == requirements
    assert [(s["section_key"], s["title"]) for s in packet["section_contract"]] == [
        (k, t) for k, t in structure_sections(structure) if k != "run_methodology"
    ]
    assert all("allowed_fact_card_ids" not in s for s in packet["section_contract"])
    for section in packet["section_contract"]:
        assert {
            cid
            for cid, card in packet["cards"].items()
            if section["section_key"] in card["allowed_section_keys"]
        } == set(cards)
    assert packet["current_draft"] == body
    assert packet["report_language"] == "en"
    assert packet["summary_rules"]["max_sentence_units"] == 6
    llm.chat.assert_awaited_once()


@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
def test_scientific_sources_are_not_crowded_out_by_headers_or_oversized_blocks(policy):
    chunks = [
        {
            "chunk_key": f"header:{i}",
            "section": section,
            "text": "metadata " * 30,
            "content_scope": "selected_fulltext",
        }
        for i, section in enumerate(["title", "authors", "references", "affiliation"])
    ]
    chunks += [
        {
            "chunk_key": "large",
            "section": "abstract",
            "text": "x" * 5000,
            "content_scope": "selected_fulltext",
        },
        {
            "chunk_key": "finding",
            "section": "results",
            "text": "EViT improves speed by 50% under the reported settings.",
            "content_scope": "selected_fulltext",
        },
    ]
    context, keys = build_extraction_context(
        {"title": "Paper", "quality_policy": policy, "chunks": chunks},
        max_chars=500,
        max_chunks=4,
    )
    assert keys == ["finding"]
    assert (
        chunks[-1]["text"] in context
        and "metadata" not in context
        and "x" * 200 not in context
    )
    assert len(context) <= 500


def test_unknown_fulltext_blocks_keep_complete_science_before_short_page_headers():
    chunks = [
        {"chunk_key": f"h{i}", "text": "Title / author", "section": "unknown"}
        for i in range(4)
    ]
    source = "Model-A achieves the reported result under the specified conditions. " * 3
    chunks.append({"chunk_key": "science", "text": source, "section": "unknown"})
    context, keys = build_extraction_context(
        {"quality_policy": "strict_auto_v1", "chunks": chunks}, max_chunks=1
    )
    assert keys == ["science"] and source in context
    old, keys = build_extraction_context(
        {"quality_policy": "legacy", "chunks": chunks}, max_chunks=1
    )
    assert keys == ["h0"] and "Title / author" in old


def test_strict_contract_versions_change_identity_without_changing_legacy():
    config = load_config()
    legacy = build_config_summary(config)
    assert "quality_policy" not in legacy["effective"]
    for policy in ("strict_v1", "strict_auto_v1"):
        versions = build_config_summary(config, policy)["effective"]["quality_policy"][
            "contracts"
        ]
        assert {
            "original-support-only-v1",
            "grounded-writing-v14",
            "source-selection-v1",
        }.issubset(versions)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope, expected",
    [
        ("abstract", "abstract_only"),
        ("selected_fulltext", "abstract_and_selected_fulltext"),
    ],
)
async def test_method_disclosure_does_not_turn_failed_fulltext_attempt_into_review(
    scope, expected
):
    from litagent.agents.synthesis import SynthesisWorker
    from litagent.context.evidence_selector import EvidenceSelection

    cards, _, requirements = writing_material()
    item = {
        "evidence_id": "a",
        "paper_id": "evit",
        "text": "EViT improves speed.",
        "supporting_text": "EViT improves speed.",
        "content_scope": scope,
        "chunk_key": "abstract",
    }
    cards = build_fact_cards({"a": item})
    selection = EvidenceSelection(
        1,
        {"a": item},
        {
            k: ["a"]
            for k, _ in structure_sections("academic_v2")
            if k not in ("abstract", "conclusion")
        },
        {
            k: "lexical_fallback"
            for k, _ in structure_sections("academic_v2")
            if k not in ("abstract", "conclusion")
        },
        0,
        20,
    )
    selector = AsyncMock()
    selector.select.return_value = selection
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(to_native(draft(cards), cards)), model="offline", usage={}
    )
    worker = SynthesisWorker(llm, evidence_selector=selector)
    result = await worker.execute(
        SubTask(
            "write",
            "write",
            "synthesis",
            input_data={
                "query": requirements[0]["text"],
                "review_structure": "academic_v2",
                "quality_policy": "strict_auto_v1",
                "report_language": "en",
                "upstream_results": {
                    "extract": [
                        {
                            "paper_id": "evit",
                            "fulltext_acquisition": {
                                "status": "failed",
                                "reason_code": "fulltext_unavailable",
                            },
                            "evidence_items": [item],
                        }
                    ]
                },
            },
        )
    )
    assert result["review_methodology"]["content_mode"] == expected
    assert (
        result["review_methodology"]["fulltext_acquisition"]["evit"]["status"]
        == "failed"
    )
    if scope == "abstract":
        assert result["review_methodology"]["full_text_review"].startswith(
            "not performed"
        )


@pytest.mark.parametrize(
    "text, scientific",
    [
        ("ViT uses image patches.", True),
        ("ViT encodes image patches.", True),
        ("DeiT采用蒸馏训练。", True),
        ("本综述讨论ViT使用图像块的机制。", True),
        ("This review discusses ViT.", False),
        ("This review uses two groups to discuss ViT.", False),
        ("本综述介绍ViT与CNN。", False),
    ],
)
def test_named_mechanism_cannot_be_exempted_as_report_organization(text, scientific):
    from litagent.eval.scoped import has_scientific_predicate

    assert has_scientific_predicate(text, mechanism_subjects=True) is scientific


@pytest.mark.asyncio
async def test_real_v3_cannot_accept_named_mechanism_as_organization():
    class Misclassified(BatchLLM):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = json.loads(messages[1]["content"])
            data = json.loads(response.content)
            if "allowed_citation_aliases" in payload:
                for fragments in data["units"].values():
                    for fragment in fragments.values():
                        for assertion in fragment["assertions"]:
                            assertion["kind"] = "report_structure"
            elif "cited_materials" in payload:
                for decision in data["decisions"]:
                    decision.update(
                        correct_kind="report_structure",
                        support_basis=[],
                        aspect_checks=[],
                    )
            response.content = json.dumps(data)
            return response

    survey, context = materials(1)
    from litagent.eval.scoped import ScopedEvaluationError

    llm = Misclassified()
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_extraction_incomplete"
    ) as error:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert error.value.details["unsupported_claims"]
    assert error.value.details["coverage"]["complete"] is False
    assert len(llm.calls) == 1 and llm.batches == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_auto_v1", "strict_v1"])
async def test_invalid_review_protocol_stops_auto_without_schema_retry(policy):
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(content="{}", model="offline", usage={})
    from litagent.context.evidence_selector import EvidenceSelection

    worker = ReviewerWorker(llm)
    cards, scope, requirements = writing_material()
    selection = EvidenceSelection(
        1,
        {
            "a": {
                "evidence_id": "a",
                "paper_id": "evit",
                "text": "EViT improves speed.",
                "supporting_text": "EViT improves speed.",
                "chunk_key": "abstract",
            }
        },
        scope["section_evidence_ids"],
        {key: "lexical_fallback" for key in scope["section_evidence_ids"]},
        0,
        100,
    )
    task = SubTask(
        "review",
        "review",
        "reviewer",
        input_data={
            "quality_policy": policy,
            "review_structure": "academic_v2",
            "coverage_requirements": requirements,
            "upstream_results": {
                "synthesis": {
                    "draft": "EViT improves speed. [E:a]",
                    "evidence_selection": selection.to_dict(),
                }
            },
        },
    )
    if policy == "strict_auto_v1":
        with pytest.raises(ValueError, match="strict_review_contract_invalid"):
            await worker.execute(task)
    else:
        assert (await worker.execute(task))["review_unverified"] is True
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "invalid", "context", "language", "summary"])
async def test_actual_strict_adversarial_revision_has_json_budget_and_failure_stop(
    mode,
):
    from litagent.agents.adversarial import AdversarialReviewWorker
    from litagent.agents.synthesis import SynthesisWorker
    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.safety.budget import CostBudget

    cards, _, requirements = writing_material()
    item = {
        "evidence_id": "a",
        "paper_id": "evit",
        "text": "EViT improves speed.",
        "supporting_text": "EViT improves speed.",
        "chunk_key": "abstract",
    }
    selection = EvidenceSelection(
        1,
        {"a": item},
        {k: ["a"] for k, _ in structure_sections("academic_v2")},
        {
            k: (
                "selected_union"
                if k in ("abstract", "conclusion")
                else "lexical_fallback"
            )
            for k, _ in structure_sections("academic_v2")
        },
        0,
        20,
    )
    original = render_grounded_report(
        json.dumps(draft(cards)),
        cards,
        "academic_v2",
        "en",
        evidence_selection=selection,
    )
    original = assemble_methodology(original, "academic_v2", "en", {})
    revised = draft(cards)
    revised["title"] = "Revision"
    if mode == "language":
        revised = draft(cards, "zh-CN")
    if mode == "summary":
        revised["sections"][0]["blocks"][0]["sentences"] *= 7
    llm = AsyncMock()
    budget = CostBudget(max_tokens=500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    llm._cost_budget = budget
    llm.chat.return_value = LLMResponse(
        content=json.dumps(to_native(revised, cards)) if mode != "invalid" else "{}",
        model="offline",
        usage={},
    )
    synthesis = SynthesisWorker(llm)
    synthesis.execute = AsyncMock(
        return_value={
            "draft": original,
            "quality_policy": "strict_auto_v1",
            "fact_cards": cards,
            "coverage_requirements": requirements,
            "evidence_selection": selection.to_dict(),
        }
    )
    reviewer = AsyncMock()
    reviewer.execute.return_value = {
        "score": 0.5,
        "verdict": "revise",
        "bound_feedback": [
            {
                "source_quote": original,
                "existing_evidence_ids": ["a"],
                "issue": "x" * 150000 if mode == "context" else "Located review",
            }
        ],
        "evidence_compliance": {"passed": False},
        "language_compliance": {"passed": True},
    }
    reviewer.review_revision.return_value = {
        "score": 0.9,
        "verdict": "accept",
        "evidence_compliance": {"passed": True},
        "language_compliance": {"passed": True},
    }
    events = []
    worker = AdversarialReviewWorker(
        llm,
        synthesis,
        reviewer,
        max_rounds=2,
        trace_hook=lambda e, d: events.append((e, d)),
    )
    result = await worker.execute(
        SubTask(
            "adv",
            "adv",
            "adversarial_review",
            input_data={
                "query": requirements[0]["text"],
                "quality_policy": "strict_auto_v1",
                "review_structure": "academic_v2",
                "report_language": "en",
            },
        )
    )
    if mode == "context":
        llm.chat.assert_not_awaited()
        detail = next(d for e, d in events if e == "writing.request.rejected")
        assert detail["phase"] == "revision" and detail["max_tokens"] == 20000
        assert detail["budget"]["reservation"] > 117964
        material = json.loads(detail["messages"][1]["content"])
        assert "x" * 150000 in material["located_feedback"]
        assert detail["response_format"]["type"] == "json_schema"
    else:
        options = llm.chat.await_args.kwargs
        assert options["response_format"]["type"] == "json_schema"
        assert options["response_format"]["json_schema"]["strict"] is True
        assert options["max_tokens"] == 20000 and options["transport_max_attempts"] == 1
        packet = json.loads(llm.chat.await_args.args[0][1]["content"])
        assert (
            packet["coverage_requirements"] == requirements
            and len(packet["section_contract"]) == 7
        )
        llm.chat.assert_awaited_once()
    if mode == "valid":
        assert (
            result["final_draft"].startswith("# Revision")
            and budget.stop_reason is None
        )
    else:
        assert result["final_draft"] == original
        assert budget.stop_reason == "strict_worker_failed" and budget.remaining() == 0
        reviewer.review_revision.assert_not_awaited()
