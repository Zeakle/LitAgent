"""Protect supported occurrences while allowing fully diagnosed wrong citations."""

import copy
import hashlib

import pytest

from litagent.eval.scoped import build_source_fragments
from litagent.report_render import build_report_source_units
from litagent.runner import LitAgent


def bound(body):
    """Bind every model fragment and its diagnosed scientific assertion."""
    units = build_report_source_units(body)
    assertions = []
    for unit in units:
        for fragment in build_source_fragments(unit):
            assertions.append(
                {
                    "claim_id": f"c{len(assertions)}",
                    "kind": "scientific",
                    "status": "unverifiable",
                    "source_unit_id": unit.unit_id,
                    "source_fragment_id": fragment["source_fragment_id"],
                    "source_quote": fragment["source_quote"],
                    "source_start": unit.start + fragment["start"],
                    "source_end": unit.start + fragment["end"],
                    "evidence_ids": fragment["citation_aliases"],
                    "claim_text": fragment["source_quote"],
                }
            )
    details = {
        "evaluation_version": "survey-evaluation-v3",
        "quality_policy": "strict_auto_v1",
        "source_report_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "assertions": assertions,
        "unsupported_claims": [
            {
                **a,
                "diagnostic_source": "survey-evaluation-v3",
                "alignment_status": "exact",
                "citation_scope": "exact",
            }
            for a in assertions
        ],
        "coverage": {
            "source_units": len(units),
            "processed_units": len(units),
            "model_source_units": len(units),
            "model_processed_units": len(units),
        },
        "stages": {
            "expected_assertions": len(assertions),
            "processed_assertions": len(assertions),
            "expected_batches": 1,
            "batches": [
                {
                    "status": "completed",
                    "claim_ids": [a["claim_id"] for a in assertions],
                }
            ],
            "global_check": "completed",
            "requests": [{"phase": "inventory", "status": "response_received"}],
        },
    }
    return {"faithfulness": {"details": details}}


def test_repeated_reference_can_be_replaced_only_when_all_occurrences_are_diagnosed():
    body = "EViT improves speed. [E:a]\n\nEViT uses tokens. [E:a]"
    evaluation = bound(body)
    assert LitAgent._completed_bound_repair(evaluation, body)
    assert LitAgent._removable_strict_references(evaluation, body) == {"a"}


@pytest.mark.parametrize(
    "defect",
    [
        "supported",
        "organization",
        "not_diagnosed",
        "missing_batch",
        "stale_sha",
        "fake_quote",
        "wrong_position",
    ],
)
def test_a_protected_or_unverified_occurrence_cannot_be_removed(defect):
    body = "EViT improves speed. [E:a]\n\nEViT uses tokens. [E:a]"
    evaluation = bound(body)
    d = evaluation["faithfulness"]["details"]
    if defect == "supported":
        d["assertions"][1]["status"] = "supported"
        d["unsupported_claims"].pop()
    elif defect == "organization":
        d["assertions"][1]["kind"] = "report_structure"
        d["unsupported_claims"][1]["kind"] = "report_structure"
    elif defect == "not_diagnosed":
        d["unsupported_claims"].pop()
    elif defect == "missing_batch":
        d["stages"]["batches"] = []
    elif defect == "stale_sha":
        d["source_report_sha256"] = "stale"
    elif defect == "fake_quote":
        d["unsupported_claims"][1]["source_quote"] = "Other original"
    else:
        d["unsupported_claims"][1]["source_start"] += 1
    assert LitAgent._removable_strict_references(evaluation, body) == set()


def test_combined_citations_are_considered_separately_for_each_paper_occurrence():
    body = "EViT improves speed. [E:a] [E:b]\n\nEViT uses tokens. [E:a]"
    evaluation = bound(body)
    d = evaluation["faithfulness"]["details"]
    d["assertions"][1]["status"] = "supported"
    d["unsupported_claims"].pop()
    assert LitAgent._removable_strict_references(evaluation, body) == {"b"}


def test_compound_fragment_with_one_supported_assertion_protects_its_citations():
    body = "EViT improves speed and uses tokens. [E:a]"
    evaluation = bound(body)
    d = evaluation["faithfulness"]["details"]
    supported = {
        **copy.deepcopy(d["assertions"][0]),
        "claim_id": "c1",
        "status": "supported",
    }
    d["assertions"].append(supported)
    d["stages"].update(expected_assertions=2, processed_assertions=2)
    d["stages"]["batches"][0]["claim_ids"].append("c1")
    assert LitAgent._completed_bound_repair(evaluation, body)
    assert LitAgent._removable_strict_references(evaluation, body) == set()


def test_strict_duplicate_heading_cleanup_does_not_block_a_new_complete_candidate():
    from litagent.config import load_config

    original = "# Review\n\n## Introduction\n\n### Introduction\n\nEViT improves speed. [E:a]\n\n## Methods\n\n### Methods\n\nEViT uses tokens. [E:a]"
    candidate = original.replace("### Introduction\n\n", "").replace(
        "### Methods\n\n", ""
    )
    agent = LitAgent(load_config())
    ledger = {"a": {"text": "EViT improves speed and uses tokens."}}
    assert agent._validate_rewrite_candidate(
        original=original, candidate=candidate, evidence_ledger=ledger
    ).accepted
    distinct = original.replace("## Methods", "## Specific mechanism")
    assert not agent._validate_rewrite_candidate(
        original=distinct,
        candidate=candidate,
        evidence_ledger=ledger,
    ).accepted


def test_removing_all_wrong_occurrences_still_requires_known_replacement_citations():
    from litagent.config import load_config

    original = "EViT improves speed. [E:a]\n\nEViT uses tokens. [E:a]"
    evaluation = bound(original)
    allowed = LitAgent._removable_strict_references(evaluation, original)
    candidate = original.replace("[E:a]", "[E:b]")
    agent = LitAgent(load_config())
    missing = agent._validate_rewrite_candidate(
        original=original,
        candidate=candidate,
        evidence_ledger={"a": {"text": "wrong paper"}},
        allowed_removed_refs=sorted(allowed),
    )
    assert not missing.accepted and "unknown_evidence_refs" in missing.reason_codes


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [True, False])
async def test_real_rewrite_request_and_host_validation_share_occurrence_policy(strict):
    import json
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from litagent.agents.synthesis import SynthesisWorker
    from litagent.config import load_config
    from litagent.eval.base import CTX_EVIDENCE, CTX_SELECTED_EVIDENCE
    from litagent.llm.client import LLMResponse
    from litagent.quality import build_fact_cards, digest

    body = "EViT improves speed. [E:a]\n\nEViT uses tokens. [E:a]"
    if strict:
        body = "# Review\n\n## Introduction\n\n" + body
    ledger = {
        "a": {
            "paper_id": "paper-a",
            "evidence_id": "a",
            "text": "EViT improves speed.",
            "supporting_text": "EViT improves speed.",
        },
        "b": {
            "paper_id": "paper-b",
            "evidence_id": "b",
            "text": "EViT uses tokens.",
            "supporting_text": "EViT uses tokens.",
        },
    }
    facts = build_fact_cards(ledger)
    evaluation = bound(body)
    agent = LitAgent(load_config())
    agent._llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                content="{}", model="offline", usage={"prompt_tokens": 1}
            )
        )
    )
    agent._synthesis = SynthesisWorker(agent._llm)
    agent._build_eval_context = lambda **kwargs: {
        CTX_EVIDENCE: ledger,
        CTX_SELECTED_EVIDENCE: list(ledger.values()),
    }
    report = {
        "survey": body,
        "metadata": {
            "quality_policy": "strict_auto_v1" if strict else "legacy",
            "review_structure": "academic_v2" if strict else "legacy",
            "report_language": "en",
            "fact_cards": facts,
            "evidence_snapshot_sha256": digest(facts),
        },
    }
    before = copy.deepcopy(report)
    outcome = await agent._attempt_evidence_rewrite(
        query="EViT",
        report_data=report,
        results={},
        initial_evaluation=evaluation,
        initial_quality={"status": "unverified"},
        report_language="en",
    )
    assert not outcome.committed and report == before
    policy = json.loads(agent._llm.chat.await_args.args[0][-1]["content"])
    if strict:
        cid = next(cid for cid, c in facts.items() if c["evidence_id"] == "a")
        assert (
            policy["allowed_removed_fact_card_ids"] == [cid]
            and policy["must_preserve_fact_card_ids"] == []
        )
    else:
        assert policy["allowed_removed_evidence_ids"] == []
        assert len(policy["must_preserve_evidence_ids"]) == 1
