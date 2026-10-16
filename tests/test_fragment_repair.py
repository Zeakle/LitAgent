"""Repair only fully diagnosed fragments while preserving source-bound science."""

import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_reference_occurrence_policy import bound

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.eval.base import CTX_EVIDENCE, CTX_SELECTED_EVIDENCE
from litagent.llm.client import LLMResponse
from litagent.quality import (
    apply_fragment_repairs,
    build_fact_cards,
    digest,
    grounded_messages,
    prepare_fragment_repair_request,
)
from litagent.runner import LitAgent


def case(language="en", layout="paragraph"):
    heading = "引言" if language == "zh-CN" else "Introduction"
    good = "EViT improves speed [E:a]." if language == "en" else "EViT提高速度 [E:a]。"
    bad = (
        "Evo-ViT improves speed [E:b]."
        if language == "en"
        else "Evo-ViT提高速度 [E:b]。"
    )
    if layout == "table":
        content = f"{good}\n\n| Method | Finding |\n|---|---|\n| EViT | {bad} |"
    elif layout == "list":
        content = f"{good}\n\n- {bad}"
    else:
        content = good + " " + bad
    body = f"# Review\n\n## {heading}\n\n{content}\n"
    evaluation = bound(body)
    details = evaluation["faithfulness"]["details"]
    for c in details["assertions"]:
        if bad not in c["source_quote"]:
            c["status"] = "supported"
    details["unsupported_claims"] = [
        c for c in details["unsupported_claims"] if bad in c["source_quote"]
    ]
    facts = build_fact_cards(
        {
            "a": {"paper_id": "evit", "supporting_text": "EViT improves speed."},
            "b": {"paper_id": "evovit", "supporting_text": "Evo-ViT uses tokens."},
        }
    )
    targets = LitAgent._strict_repair_targets(evaluation, body, "academic_v2", language)
    replacement = {
        "text": "Evo-ViT uses tokens." if language == "en" else "Evo-ViT使用token。",
        "fact_card_ids": [next(k for k, c in facts.items() if c["evidence_id"] == "b")],
    }
    replacement["support_quotes"] = [
        facts[cid]["source_quote"] for cid in replacement["fact_card_ids"]
    ]
    response = {
        "patches": {
            t["source_fragment_id"]: {
                "first_sentence": replacement,
                "additional_sentences": [],
            }
            for t in targets
        }
    }
    return body, evaluation, facts, targets, response, good


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("layout", ["paragraph", "list", "table"])
def test_patch_preserves_every_byte_outside_host_target(language, layout):
    body, _, facts, targets, response, good = case(language, layout)
    assert len(targets) == 1
    candidate = apply_fragment_repairs(
        json.dumps(response), body, targets, facts, "academic_v2", language
    )
    t = targets[0]
    prefix = t.get("source_prefix", "")
    left, right = t["source_start"] + len(prefix), t["source_end"]
    assert candidate[:left] == body[:left]
    assert candidate.endswith(body[right:])
    assert good in candidate
    if layout == "table":
        assert "| EViT |" in candidate and candidate.count("|") == body.count("|")
    if layout == "list":
        assert "\n- Evo-ViT" in candidate


@pytest.mark.parametrize(
    "protected", ["supported", "report_structure", "not_diagnosed"]
)
def test_mixed_or_unbound_fragment_is_not_editable(protected):
    body, evaluation, _, _, _, _ = case()
    d = evaluation["faithfulness"]["details"]
    bad = copy.deepcopy(d["unsupported_claims"][0])
    bad["claim_id"] = "another"
    if protected == "supported":
        bad["status"] = "supported"
    if protected == "report_structure":
        bad["kind"] = "report_structure"
    d["assertions"].append(bad)
    d["stages"].update(
        expected_assertions=len(d["assertions"]),
        processed_assertions=len(d["assertions"]),
    )
    d["stages"]["batches"][0]["claim_ids"].append("another")
    assert LitAgent._strict_repair_targets(evaluation, body, "academic_v2", "en") == []


def test_request_keeps_complete_sources_with_native_patch_identity():
    body, _, facts, targets, _, _ = case()
    raw = grounded_messages(facts, "academic_v2", "en", "Compare EViT", draft=body)
    before = copy.deepcopy(raw)
    prepared, fmt = prepare_fragment_repair_request(raw, targets)
    packet = json.loads(prepared[1]["content"])
    schema = fmt["json_schema"]["schema"]
    assert raw == before
    assert "sections" not in schema["properties"]
    assert set(schema["properties"]["patches"]["properties"]) == {
        targets[0]["source_fragment_id"]
    }
    assert list(packet["sources"].values()) == list(
        json.loads(raw[1]["content"])["sources"].values()
    )
    assert "[E:" not in packet["current_draft"] and "[F:" in packet["current_draft"]
    assert packet["repair_contract"] == "source-locked-fragment-repair-v1"


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "card",
        "duplicate_card",
        "empty",
        "host",
        "newline",
        "pipe",
        "unknown_field",
        "stale_sha",
        "range",
        "quote",
        "overlap",
        "scope",
        "forged_source",
    ],
)
def test_patch_rejects_bad_identity_and_never_mutates_original(fault):
    body, _, facts, targets, response, _ = case()
    target = targets[0]
    patch = response["patches"][target["source_fragment_id"]]
    sentence = patch["first_sentence"]
    selection = None
    if fault == "missing":
        response["patches"].clear()
    elif fault == "extra":
        response["patches"]["forged"] = copy.deepcopy(patch)
    elif fault == "card":
        sentence["fact_card_ids"] = ["unknown"]
    elif fault == "duplicate_card":
        sentence["fact_card_ids"] *= 2
    elif fault == "empty":
        sentence["text"] = ""
    elif fault == "host":
        sentence["text"] += " [F:forged]"
    elif fault == "newline":
        sentence["text"] += "\n## forged"
    elif fault == "pipe":
        sentence["text"] += " | forged"
    elif fault == "unknown_field":
        patch["offset"] = 0
    elif fault == "stale_sha":
        target["source_report_sha256"] = "wrong"
    elif fault == "range":
        target["source_start"] += 1
    elif fault == "quote":
        target["source_quote"] = "invented"
    elif fault == "overlap":
        targets.append(copy.deepcopy(target))
    elif fault == "scope":
        selection = {"section_evidence_ids": {"introduction": ["a"]}}
    else:
        next(iter(facts.values()))["source_quote"] += " forged"
    before = copy.deepcopy((body, facts, targets, response))
    with pytest.raises(ValueError):
        apply_fragment_repairs(
            json.dumps(response),
            body,
            targets,
            facts,
            "academic_v2",
            "en",
            evidence_selection=selection,
        )
    assert (body, facts, targets, response) == before


def test_duplicate_json_key_cannot_discard_a_target():
    body, _, facts, targets, _, _ = case()
    with pytest.raises(ValueError, match="duplicate"):
        apply_fragment_repairs(
            '{"patches":{},"patches":{}}', body, targets, facts, "academic_v2", "en"
        )


@pytest.mark.asyncio
async def test_actual_runner_uses_patch_and_still_requires_entire_candidate_evaluation():
    body, evaluation, facts, _, response, good = case()
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(response), model="offline", usage={"total_tokens": 1}
            )
        )
    )
    agent = LitAgent(load_config())
    agent._llm = llm
    agent._synthesis = SynthesisWorker(llm)
    ledger = {
        c["evidence_id"]: {
            "evidence_id": c["evidence_id"],
            "paper_id": c["paper_id"],
            "text": c["source_quote"],
            "supporting_text": c["source_quote"],
        }
        for c in facts.values()
    }
    agent._build_eval_context = lambda **kw: {
        CTX_EVIDENCE: ledger,
        CTX_SELECTED_EVIDENCE: list(ledger.values()),
    }
    agent._evaluate = AsyncMock(
        return_value={
            "faithfulness": {
                "skipped": True,
                "details": {"evaluation_version": "survey-evaluation-v3"},
            },
            "citation_accuracy": {"passed": True},
        }
    )
    report = {
        "survey": body,
        "metadata": {
            "quality_policy": "strict_auto_v1",
            "review_structure": "academic_v2",
            "fact_cards": facts,
            "evidence_snapshot_sha256": digest(facts),
        },
    }
    before = copy.deepcopy(report)
    outcome = await agent._attempt_evidence_rewrite(
        query="Compare EViT",
        report_data=report,
        results={},
        initial_evaluation=evaluation,
        initial_quality={"status": "unverified"},
        report_language="en",
    )
    assert not outcome.committed and report == before
    assert good in outcome.candidate
    agent._evaluate.assert_awaited_once()
    assert agent._evaluate.await_args.kwargs["survey"] == outcome.candidate
    fmt = llm.chat.await_args.kwargs["response_format"]
    assert fmt["json_schema"]["name"] == "fragment_repair_v2"


def fully_supported(body):
    """Bind a complete offline evaluation to every actual source fragment."""
    from test_strict_quality import valid_details

    details = bound(body)["faithfulness"]["details"]
    details["coverage"]["complete"] = True
    details["unsupported_claims"] = []
    for c in details["assertions"]:
        c["status"] = "supported"
    for key in (
        "fact_card_map",
        "coverage_requirements",
        "core_coverage",
        "structure_checks",
        "method_checks",
        "language_checks",
    ):
        details[key] = valid_details()[key]
    details["core_coverage"][0]["claim_ids"] = [details["assertions"][0]["claim_id"]]
    return {
        "faithfulness": {"passed": True, "skipped": False, "details": details},
        "citation_accuracy": {"passed": True, "skipped": False},
        "internal_consistency": {"passed": True, "skipped": False},
    }


def test_reextracted_claim_wording_does_not_regress_unchanged_verified_source():
    body, old, facts, targets, response, good = case()
    candidate = apply_fragment_repairs(
        json.dumps(response), body, targets, facts, "academic_v2", "en"
    )
    new = fully_supported(candidate)
    new["faithfulness"]["details"]["assertions"][0][
        "claim_text"
    ] = "A newly atomized paraphrase of the unchanged source."
    assert LitAgent._passed_metric_regressed(old, new)
    assert not LitAgent._passed_metric_regressed(
        old, new, original_survey=body, candidate_survey=candidate
    )
    assert good in candidate


@pytest.mark.parametrize(
    "fault",
    [
        "incomplete",
        "unsupported",
        "wrong_kind",
        "wrong_quote",
        "wrong_refs",
        "wrong_offset",
        "stale",
        "missing_unit",
        "global_failure",
        "source_changed",
    ],
)
def test_source_preservation_is_never_a_substitute_for_complete_new_verification(fault):
    body, old, facts, targets, response, _ = case()
    candidate = apply_fragment_repairs(
        json.dumps(response), body, targets, facts, "academic_v2", "en"
    )
    new = fully_supported(candidate)
    d = new["faithfulness"]["details"]
    c = d["assertions"][0]
    c["claim_text"] = "Re-atomized"
    if fault == "incomplete":
        d["stages"]["global_check"] = "not_started"
    elif fault == "unsupported":
        c["status"] = "unsupported"
    elif fault == "wrong_kind":
        c["kind"] = "report_structure"
        d["core_coverage"][0]["claim_ids"] = [d["assertions"][1]["claim_id"]]
    elif fault == "wrong_quote":
        c["source_quote"] = "forged"
    elif fault == "wrong_refs":
        c["evidence_ids"] = ["b"]
    elif fault == "wrong_offset":
        c["source_start"] += 1
    elif fault == "stale":
        d["source_report_sha256"] = "wrong"
    elif fault == "missing_unit":
        d["coverage"]["processed_units"] -= 1
    elif fault == "global_failure":
        d["global_issues"] = ["An unresolved conflict"]
    else:
        candidate = candidate.replace("EViT improves speed", "EViT may improve speed")
        new = fully_supported(candidate)
        new["faithfulness"]["details"]["assertions"][0]["claim_text"] = "Re-atomized"
    assert LitAgent._passed_metric_regressed(
        old, new, original_survey=body, candidate_survey=candidate
    )


@pytest.mark.asyncio
async def test_complete_candidate_can_commit_without_rewriting_supported_sources():
    body, old, facts, _, response, good = case()
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                content=json.dumps(response), model="offline", usage={"total_tokens": 1}
            )
        )
    )
    agent = LitAgent(load_config())
    agent._llm = llm
    agent._synthesis = SynthesisWorker(llm)
    ledger = {
        c["evidence_id"]: {
            "evidence_id": c["evidence_id"],
            "paper_id": c["paper_id"],
            "text": c["source_quote"],
            "supporting_text": c["source_quote"],
        }
        for c in facts.values()
    }
    agent._build_eval_context = lambda **kw: {
        CTX_EVIDENCE: ledger,
        CTX_SELECTED_EVIDENCE: list(ledger.values()),
    }

    async def evaluate(**kwargs):
        new = fully_supported(kwargs["survey"])
        for c in new["faithfulness"]["details"]["assertions"]:
            c["claim_text"] = "Re-atomized " + c["claim_id"]
        return new

    agent._evaluate = AsyncMock(side_effect=evaluate)
    report = {
        "survey": body,
        "metadata": {
            "quality_policy": "strict_auto_v1",
            "review_structure": "academic_v2",
            "fact_cards": facts,
            "evidence_snapshot_sha256": digest(facts),
        },
    }
    outcome = await agent._attempt_evidence_rewrite(
        query="Compare EViT",
        report_data=report,
        results={},
        initial_evaluation=old,
        initial_quality={"status": "unverified"},
        report_language="en",
    )
    assert outcome.committed and good in report["survey"]
    agent._evaluate.assert_awaited_once()
    assert report["quality"]["status"] == "passed"


@pytest.mark.asyncio
async def test_mixed_supported_fragment_has_no_model_request():
    body, evaluation, facts, _, _, _ = case()
    d = evaluation["faithfulness"]["details"]
    protected = copy.deepcopy(d["unsupported_claims"][0])
    protected.update(claim_id="protected", status="supported")
    d["assertions"].append(protected)
    d["stages"].update(
        expected_assertions=len(d["assertions"]),
        processed_assertions=len(d["assertions"]),
    )
    d["stages"]["batches"][0]["claim_ids"].append("protected")
    agent = LitAgent(load_config())
    agent._llm = SimpleNamespace(chat=AsyncMock())
    agent._synthesis = SynthesisWorker(agent._llm)
    agent._build_eval_context = lambda **kw: {
        CTX_EVIDENCE: {},
        CTX_SELECTED_EVIDENCE: [],
    }
    report = {
        "survey": body,
        "metadata": {
            "quality_policy": "strict_auto_v1",
            "review_structure": "academic_v2",
            "fact_cards": facts,
        },
    }
    outcome = await agent._attempt_evidence_rewrite(
        query="EViT",
        report_data=report,
        results={},
        initial_evaluation=evaluation,
        initial_quality={"status": "unverified"},
        report_language="en",
    )
    assert (
        not outcome.attempted
        and outcome.reason_code == "strict_repair_no_editable_fragment"
    )
    agent._llm.chat.assert_not_awaited()


def test_no_authorized_card_fails_preflight_without_dropping_source_material():
    body, _, facts, targets, _, _ = case()
    raw = grounded_messages(
        facts,
        "academic_v2",
        "en",
        "Compare EViT",
        draft=body,
        evidence_selection={"section_evidence_ids": {"introduction": []}},
    )
    before = copy.deepcopy(raw)
    with pytest.raises(ValueError, match="strict_repair_source_unavailable"):
        prepare_fragment_repair_request(raw, targets)
    assert raw == before


def test_complete_partial_candidate_can_enter_second_repair_without_committing():
    body, old, facts, targets, response, _ = case()
    candidate = apply_fragment_repairs(
        json.dumps(response), body, targets, facts, "academic_v2", "en"
    )
    new = fully_supported(candidate)
    d = new["faithfulness"]["details"]
    d["coverage"]["complete"] = False
    d["assertions"][0]["claim_text"] = "New wording for unchanged supported fact"
    unresolved = d["assertions"][1]
    unresolved["status"] = "unverifiable"
    d["unsupported_claims"] = [
        {
            **unresolved,
            "diagnostic_source": "survey-evaluation-v3",
            "alignment_status": "exact",
            "citation_scope": "exact",
        }
    ]
    new["faithfulness"].update(skipped=True, passed=False)
    assert LitAgent._completed_bound_repair(new, candidate)
    assert not LitAgent._passed_metric_regressed(
        old, new, original_survey=body, candidate_survey=candidate
    )
    assert LitAgent._derive_quality(new)["status"] != "passed"
    assert (
        len(LitAgent._strict_repair_targets(new, candidate, "academic_v2", "en")) == 1
    )


@pytest.mark.asyncio
async def test_patch_capacity_refusal_keeps_budget_and_never_sends():
    from litagent.safety.budget import CostBudget

    body, evaluation, facts, _, _, _ = case()
    config = load_config()
    config.eval.context_tokens = 4096
    budget = CostBudget()
    budget.configure_strict()
    llm = SimpleNamespace(chat=AsyncMock(), _cost_budget=budget)
    agent = LitAgent(config)
    agent._llm = llm
    agent._synthesis = SynthesisWorker(llm)
    ledger = {
        c["evidence_id"]: {
            "evidence_id": c["evidence_id"],
            "paper_id": c["paper_id"],
            "text": c["source_quote"],
            "supporting_text": c["source_quote"],
        }
        for c in facts.values()
    }
    agent._build_eval_context = lambda **kw: {
        CTX_EVIDENCE: ledger,
        CTX_SELECTED_EVIDENCE: list(ledger.values()),
    }
    report = {
        "survey": body,
        "metadata": {
            "quality_policy": "strict_auto_v1",
            "review_structure": "academic_v2",
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
    assert (
        outcome.reason_code == "rewrite_context_budget_exceeded"
        and not outcome.committed
    )
    llm.chat.assert_not_awaited()
    assert budget.used == 0 and budget.snapshot()["requests_sent"] == 0
    assert report == before
