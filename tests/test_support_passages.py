"""Verify source-selected support against self-contained real positive/negative cases."""

import copy
import json
from pathlib import Path

import pytest
from test_scoped_evaluation import ScriptedLLM, ctx

from litagent.agents.reviewer import ReviewerWorker
from litagent.config import EvalConfig
from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval.scoped import (
    PassageDecision,
    build_support_passages,
    evaluate_scoped,
    required_support_aspects,
    validate_support_basis,
)

CASES = {
    c["claim"]["claim_id"]: c
    for c in json.loads(
        Path(__file__)
        .with_name("fixtures")
        .joinpath("run004-support-cases.json")
        .read_text()
    )["cases"]
}


def real_case(ident):
    """Restore only the assertion's actual citations from the public fixture."""
    c = CASES[ident]
    ledger = {
        m["evidence_id"]: {
            "text": m["text"],
            "supporting_text": m["original_support"]["text"],
        }
        for m in c["cited_materials"].values()
    }
    aliases = {m["evidence_id"]: alias for alias, m in c["cited_materials"].items()}
    return copy.deepcopy(c["claim"]), ledger, aliases


def selected(claim, ledger, aliases, *, source="original_support", quote=None):
    """Select exact host passages with complete aspect decisions."""
    passages = build_support_passages(ledger, aliases)
    ids = [
        pid
        for pid, p in passages.items()
        if p["source"] == source
        and (
            ledger[p["evidence_id"]][
                "supporting_text" if source == "original_support" else "text"
            ][p["source_start"] : p["source_end"]]
            == quote
            if quote is not None
            else p["source_start"] == 0
            and p["source_end"]
            == len(
                ledger[p["evidence_id"]][
                    "supporting_text" if source == "original_support" else "text"
                ]
            )
        )
    ]
    assert ids
    d = PassageDecision(
        claim_id="claim_0",
        correct_kind="scientific",
        atomic=True,
        classification_valid=True,
        source_faithful=True,
        status="supported",
        reason_code="supported",
        explanation="Controlled source check.",
        support_basis=[{"support_id": pid} for pid in ids],
        aspect_checks=[
            {
                "aspect": aspect,
                "supported": True,
                "support_ids": ids,
                "explanation": "Controlled aspect check.",
            }
            for aspect in required_support_aspects(claim["claim_text"])
        ],
    )
    return d, passages


@pytest.mark.parametrize("ident", ["claim_3", "claim_30", "claim_29"])
def test_actual_run004_legitimate_science_preserves_support(ident):
    claim, ledger, aliases = real_case(ident)
    if ident == "claim_30":
        # Positive control uses the source's exact benchmark label, not a guessed alias.
        claim["claim_text"] = claim["claim_text"].replace("ImageNet", "ImageNet1K")
    decision, passages = selected(claim, ledger, aliases)
    result = validate_support_basis(claim, decision, ledger, aliases, passages=passages)
    assert result["status"] == "passed"
    for span in result["spans"]:
        field = "text" if span["source"] == "evidence_text" else "supporting_text"
        assert (
            span["quote"]
            == ledger[span["evidence_id"]][field][
                span["source_start"] : span["source_end"]
            ]
        )


def test_source_explicit_acronym_accepts_compressed_long_form():
    claim, ledger, aliases = real_case("claim_3")
    decision, passages = selected(claim, ledger, aliases, source="evidence_text")
    assert (
        validate_support_basis(claim, decision, ledger, aliases, passages=passages)[
            "status"
        ]
        == "passed"
    )
    for item in ledger.values():
        item["supporting_text"] = item["text"]
    assert (
        validate_support_basis(claim, decision, ledger, aliases)["issue"]
        == "support_entity_unverified"
    )


def test_original_mctf_benchmark_label_is_not_silently_normalized():
    claim, ledger, aliases = real_case("claim_30")
    decision, passages = selected(claim, ledger, aliases)
    result = validate_support_basis(claim, decision, ledger, aliases, passages=passages)
    assert result["status"] == "unverified" and result["entity"] == "ImageNet"


@pytest.mark.parametrize("ident", ["claim_37", "claim_2", "claim_11", "claim_44"])
def test_real_supported_model_response_cannot_override_wrong_science(ident):
    claim, ledger, aliases = real_case(ident)
    decision, passages = selected(claim, ledger, aliases)
    assert (
        validate_support_basis(claim, decision, ledger, aliases, passages=passages)[
            "status"
        ]
        == "unverified"
    )
    if ident in {"claim_11", "claim_44"}:
        assert (
            validate_support_basis(claim, decision, ledger, aliases)["issue"]
            == "support_relation_unverified"
        )


@pytest.mark.parametrize(
    "claim,source,expected",
    [
        (
            "Model-A has quadratic attention cost.",
            "Model-A has quadratic attention cost.",
            "passed",
        ),
        (
            "Model-A has quadratic attention cost limiting edge deployment.",
            "Model-A has quadratic attention cost.",
            "unverified",
        ),
        (
            "Model-A has quadratic attention cost increasing parameter count.",
            "Model-A has quadratic attention cost.",
            "unverified",
        ),
        (
            "Model-A enables mobile deployment.",
            "Model-A enables mobile deployment.",
            "passed",
        ),
        ("Model-A outperforms Model-B.", "Model-B outperforms Model-A.", "unverified"),
        ("Model-A outperforms Model-B.", "Model-A outperforms Model-B.", "passed"),
        (
            "Model-A improves efficiency via distillation.",
            "Model-B outperforms Model-A. Vision models require distillation.",
            "unverified",
        ),
        (
            "Model-A improves efficiency via distillation.",
            "Model-A improves efficiency via distillation.",
            "passed",
        ),
    ],
)
def test_relation_guards_preserve_positive_and_reject_extra_scopes(
    claim, source, expected
):
    ledger = {"a": {"text": source, "supporting_text": source}}
    item = {"claim_text": claim, "evidence_ids": ["a"]}
    decision, _ = selected(item, ledger, {"a": "e1"})
    assert (
        validate_support_basis(item, decision, ledger, {"a": "e1"})["status"]
        == expected
    )


@pytest.mark.parametrize(
    "failure",
    [
        "unknown",
        "duplicate",
        "cross_citation",
        "aspect_missing",
        "aspect_false",
        "aspect_foreign",
        "quote_in_response",
    ],
)
@pytest.mark.asyncio
async def test_current_request_contract_fails_closed_with_two_calls(failure):
    def mutate(stage, output, payload):
        if stage != 2:
            return
        d = output["decisions"][0]
        if failure == "unknown":
            d["support_basis"] = [{"support_id": "invented"}]
        elif failure == "duplicate":
            d["support_basis"] *= 2
        elif failure == "cross_citation":
            d["support_basis"] = [
                {"support_id": pid}
                for pid in payload["assertions"][1]["allowed_support_ids"]
            ][:1]
        elif failure == "aspect_missing":
            d["aspect_checks"] = []
        elif failure == "aspect_false":
            d["aspect_checks"][0]["supported"] = False
        elif failure == "aspect_foreign":
            d["aspect_checks"][0]["support_ids"] = ["invented"]
        else:
            d["support_basis"] = [
                {
                    "evidence_alias": "e1",
                    "source": "evidence_text",
                    "quote": "EViT accelerates DeiT-S by 50%.",
                }
            ]

    llm = ScriptedLLM(mutate=mutate)
    result = (
        await evaluate_scoped(
            llm=llm,
            survey="# Review\n\nEViT accelerates DeiT-S by 50%. [E:a]\n\nOther evidence. [E:b]\n",
            context=ctx(),
            config=EvalConfig(),
        )
        if failure != "quote_in_response"
        else None
    )
    if failure == "quote_in_response":
        with pytest.raises(ValueError):
            await evaluate_scoped(
                llm=llm,
                survey="# Review\n\nEViT accelerates DeiT-S by 50%. [E:a]\n",
                context=ctx(),
                config=EvalConfig(),
            )
    else:
        assert result["score"] is None and not result["coverage"]["complete"]
    assert len(llm.calls) == 2
    packet = json.loads(llm.calls[1][0][1]["content"])
    assert set(packet["schema"]["$defs"]["SupportSelection"]["properties"]) == {
        "support_id"
    }
    assert all("allowed_support_ids" not in c for c in packet["assertions"])


@pytest.mark.asyncio
async def test_full_passage_index_is_counted_before_second_call():
    evidence = {"a": {"text": "EViT improves accuracy. " * 700}}
    llm = ScriptedLLM()
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        await evaluate_scoped(
            llm=llm,
            survey="# Review\n\nEViT improves accuracy. [E:a]\n",
            context=ctx(evidence=evidence),
            config=EvalConfig(context_tokens=32768),
        )
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_wire_passage_offsets_resolve_to_host_diagnostics():
    llm = ScriptedLLM()
    result = await evaluate_scoped(
        llm=llm,
        survey="# Review\n\nEViT accelerates DeiT-S by 50%. [E:a]\n",
        context=ctx(),
        config=EvalConfig(),
    )
    packet = json.loads(llm.calls[1][0][1]["content"])
    for span in result["assertions"][0]["support_validation"]["spans"]:
        alias, ref, start, end = packet["support_passages"][span["support_id"]]
        source = (
            packet["cited_materials"][ref]["text"]
            if alias == ref
            else packet["original_supports"][ref]["text"]
        )
        assert span["quote"] == source[start:end]
        assert span["source_start"] == start and span["source_end"] == end


def test_passages_are_deterministic_continuous_and_protect_decimal_urls():
    text = "Model-A improves accuracy by 2.96%. Code: https://example.org/a.b.\nMore context."
    ledger = {"a": {"text": text}}
    passages = build_support_passages(ledger, {"a": "e1"})
    assert passages == build_support_passages(ledger, {"a": "e1"})
    assert any(
        text[p["source_start"] : p["source_end"]]
        == "Model-A improves accuracy by 2.96%."
        for p in passages.values()
    )
    assert all(p["source_end"] - p["source_start"] <= 6000 for p in passages.values())
    assert build_support_passages({"a": {"text": "x" * 6001}}, {"a": "e1"}) == {}


@pytest.mark.parametrize(
    "variant", ["present", "missing", "ambiguous", "semantic", "uncited_neighbor"]
)
def test_reviewer_presence_correction_keeps_semantic_failure_and_score(variant):
    claim = "模型提升准确率"
    draft = f"# Report\n\n{claim} [E:a]。\n"
    if variant == "missing":
        draft = f"# Report\n\n{claim}。\n"
    elif variant == "ambiguous":
        draft += f"\n{claim} [E:a]。\n"
    elif variant == "uncited_neighbor":
        draft = f"# Report\n\n{claim}；另一个模型的结论 [E:a]。\n"
    selection = EvidenceSelection(
        selected_items={"a": {"text": "Evidence"}},
        section_evidence_ids={},
        method_by_section={},
        candidate_count=1,
        omitted_count=0,
        estimated_tokens=1,
    )
    review = {
        "score": 0.6,
        "verdict": "revise",
        "evidence_compliance": {
            "passed": False,
            "unsupported_claims": [
                {
                    "claim": claim,
                    "reason": "No citation",
                    "action": "downgrade" if variant == "semantic" else "bind",
                }
            ],
            "unknown_evidence_ids": [],
        },
    }
    result = ReviewerWorker(None)._enforce_evidence_contract(draft, selection, review)
    diagnostic = result["evidence_compliance"]["unsupported_claims"][0]
    assert (diagnostic.get("citation_presence") == "present") == (variant == "present")
    assert result["score"] == 0.6 and result["verdict"] == "revise"
    assert not result["evidence_compliance"]["passed"]
    if variant == "present":
        assert (
            diagnostic["original_reason"] == "No citation"
            and diagnostic["action"] == "verify"
        )
