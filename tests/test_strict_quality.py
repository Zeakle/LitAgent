"""Exercise strict source binding, stage reservations and delivery intersection."""

import copy
import hashlib
import json

import pytest

from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.interaction_store import SqliteReviewRepository
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.quality import (
    bind_review_feedback,
    build_fact_cards,
    coverage_requirements,
    digest,
    grounded_messages,
    render_grounded_report,
    strict_acceptance,
    validate_fact_snapshot,
)
from litagent.review_structure import structure_sections
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget
from litagent.workbench import HumanDecision, ReviewRequest, RunOptions


def to_native(report, cards=None):
    """Represent the same synthetic units in the required native transport shape."""

    def block(value):
        sentences = copy.deepcopy(value["sentences"])
        for sentence in sentences:
            sentence.setdefault(
                "support_quotes",
                [
                    cards[cid]["source_quote"]
                    for cid in sentence["fact_card_ids"]
                    if cid in (cards or {})
                ],
            )
        return {
            "heading": value.get("heading", ""),
            "type": value.get("type", "paragraph"),
            "first_sentence": sentences[0],
            "additional_sentences": sentences[1:],
        }

    return {
        "title": report["title"],
        "sections": {
            s["section_key"]: {
                "first_block": block(s["blocks"][0]),
                "additional_blocks": [block(b) for b in s["blocks"][1:]],
            }
            for s in report["sections"]
        },
    }


def cards():
    """Keep an intentionally incorrect extraction beside its genuine source."""
    return build_fact_cards(
        {
            "a": {
                "paper_id": "evit",
                "text": "Evo-ViT is 50% faster.",
                "supporting_text": "EViT improves speed by 50% on DeiT-S.",
                "chunk_key": "abstract",
                "fact_fields": {"method": "EViT", "dataset": "ImageNet"},
            }
        }
    )


def draft(facts, language="en"):
    """Use host template keys and explicit per-sentence binding."""
    return {
        "title": "文献综述" if language == "zh-CN" else "Review",
        "sections": [
            {
                "section_key": key,
                "blocks": [
                    {
                        "sentences": [
                            {
                                "kind": "scientific",
                                "text": (
                                    "EViT 在 DeiT-S 上将速度提高了50%。"
                                    if language == "zh-CN"
                                    else "EViT improves speed by 50% on DeiT-S."
                                ),
                                "fact_card_ids": list(facts),
                            }
                        ]
                    }
                ],
            }
            for key, _ in structure_sections("academic_v2", language)
        ],
    }


def valid_details():
    """Specify a fully checked report without declaring a card scientifically true."""
    return {
        "evaluation_version": "survey-evaluation-v3",
        "coverage": {"complete": True},
        "stages": {
            "global_check": "completed",
            "processed_assertions": 1,
            "expected_assertions": 1,
            "expected_batches": 1,
            "batches": [{"status": "completed", "claim_ids": ["c1"]}],
        },
        "assertions": [{"claim_id": "c1", "kind": "scientific", "status": "supported"}],
        "fact_card_map": {"f1": {"evidence_id": "a"}},
        "coverage_requirements": [{"requirement_id": "question"}],
        "core_coverage": [
            {"requirement_id": "question", "answered": True, "claim_ids": ["c1"]}
        ],
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
        "language_checks": {"status": "passed"},
    }


def test_cards_keep_source_and_do_not_trust_extraction_or_missing_fields():
    fact = next(iter(cards().values()))
    assert fact["semantic_status"] == "not_verified"
    assert fact["method"] == "EViT"
    assert fact["dataset"] is None and "dataset" in fact["missing_fields"]
    assert "Evo-ViT" in fact["finding_candidate"]
    assert fact["source_quote"][fact["source_start"] : fact["source_end"]].startswith(
        "EViT"
    )
    assert build_fact_cards({"b": {"text": "compressed claim", "title": "EViT"}}) == {}


def test_equal_source_text_does_not_merge_different_paper_identities():
    facts = build_fact_cards(
        {
            eid: {
                "paper_id": eid,
                "text": "candidate",
                "supporting_text": "same original",
            }
            for eid in ["a", "b"]
        }
    )
    payload = json.loads(
        grounded_messages(facts, "academic_v2", "en", "compare")[1]["content"]
    )
    assert {s["paper_id"] for s in payload["sources"].values()} == {"a", "b"}


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_writing_renders_citations_directly_in_every_scientific_unit(language):
    facts = cards()
    body = render_grounded_report(
        json.dumps(draft(facts, language)), facts, "academic_v2", language
    )
    assert body.count("[E:a]") == 7
    assert "[E:a]" in body.split("## ")[1]
    assert "<!--" not in body and "附录" not in body


@pytest.mark.parametrize(
    "invalid", ["unknown", "uncited", "source", "host", "title", "section"]
)
def test_invalid_writing_cannot_gain_a_canonical_reference(invalid):
    facts = cards()
    report = draft(facts)
    sentence = report["sections"][0]["blocks"][0]["sentences"][0]
    if invalid == "unknown":
        sentence["fact_card_ids"] = ["unknown"]
    if invalid == "uncited":
        sentence["fact_card_ids"] = []
    if invalid == "source":
        next(iter(facts.values()))["source_quote"] = "forged"
    if invalid == "host":
        sentence["text"] += " [E:a]"
    if invalid == "title":
        report["title"] += "\n## injected"
    if invalid == "section":
        report["sections"].reverse()
    with pytest.raises(ValueError):
        render_grounded_report(json.dumps(report), facts, "academic_v2", "en")


def test_fact_snapshot_verifies_the_actual_frozen_source():
    facts = cards()
    item = next(iter(facts.values()))
    ledger = {
        "a": {
            "paper_id": "evit",
            "supporting_text": item["source_quote"],
            "chunk_key": "abstract",
        }
    }
    validate_fact_snapshot(facts, ledger, digest(facts))
    ledger["a"]["paper_id"] = "evovit"
    with pytest.raises(ValueError, match="fact_source_invalid"):
        validate_fact_snapshot(facts, ledger, digest(facts))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_start", 1),
        ("page", 999),
        ("chunk_key", "forged"),
        ("source_version", "forged"),
    ],
)
def test_fact_snapshot_rejects_forged_host_positions_even_with_updated_snapshot_hash(
    field, value
):
    facts = cards()
    item = next(iter(facts.values()))
    ledger = {
        "a": {
            "paper_id": "evit",
            "supporting_text": item["source_quote"],
            "chunk_key": "abstract",
        }
    }
    item[field] = value
    with pytest.raises(ValueError, match="fact_source_invalid"):
        validate_fact_snapshot(facts, ledger, digest(facts))


def test_review_feedback_must_match_current_source_and_actual_references():
    body = "EViT is faster [E:a]."
    review = {
        "issues": [
            {
                "source_quote": body,
                "existing_evidence_ids": ["a"],
                "issue": "check speed",
            },
            {"source_quote": "old draft", "existing_evidence_ids": []},
            {"source_quote": body, "existing_evidence_ids": ["b"]},
        ]
    }
    result = bind_review_feedback(body, review)
    assert len(result["bound_feedback"]) == 1 and len(result["unbound_feedback"]) == 2
    assert (
        result["bound_feedback"][0]["source_sha256"]
        == hashlib.sha256(body.encode()).hexdigest()
    )
    assert not bind_review_feedback(body + body, review)["bound_feedback"]


@pytest.mark.parametrize(
    "defect",
    [
        "unsupported",
        "unverified",
        "empty",
        "coverage",
        "missing_requirement",
        "invented_answer",
        "method",
        "conflict",
        "cards",
        "missing_stages",
        "interrupted_batch",
        "global_pending",
        "duplicate_batch_id",
    ],
)
def test_strict_acceptance_rejects_every_incomplete_or_false_dimension(defect):
    details = valid_details()
    if defect in {"unsupported", "unverified"}:
        details["assertions"][0]["status"] = defect
    if defect == "empty":
        details["assertions"] = []
    if defect == "coverage":
        details["coverage"]["complete"] = False
    if defect == "missing_requirement":
        details["core_coverage"] = []
    if defect == "invented_answer":
        details["core_coverage"][0]["claim_ids"] = ["unknown"]
    if defect == "method":
        details["method_checks"]["status"] = "unverified"
    if defect == "conflict":
        details["conflicts"] = [{"claim_a": "c1"}]
    if defect == "cards":
        details["fact_card_map"] = {}
    if defect == "missing_stages":
        details.pop("stages")
    if defect == "interrupted_batch":
        details["stages"]["batches"][0]["status"] = "interrupted_no_replay"
    if defect == "global_pending":
        details["stages"]["global_check"] = "unverified"
    if defect == "duplicate_batch_id":
        details["stages"]["batches"][0]["claim_ids"].append("c1")
    assert strict_acceptance(details)["status"] == "failed"


def test_strict_gate_is_separate_from_the_legacy_point_eight_threshold():
    details = valid_details()
    assert strict_acceptance(details)["status"] == "passed"
    details["assertions"].append(
        {"claim_id": "c2", "kind": "scientific", "status": "unsupported"}
    )
    evaluation = {
        "citation_accuracy": {"passed": True},
        "faithfulness": {"passed": True, "score": 0.9, "details": details},
    }
    assert LitAgent._derive_quality(evaluation)["status"] == "failed"
    evaluation["faithfulness"]["details"] = {
        "evaluation_version": "survey-evaluation-v2",
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
    }
    assert LitAgent._derive_quality(evaluation)["status"] == "passed"


def test_strict_options_require_humans_and_preserve_legacy_identity_algorithm():
    legacy = RunOptions()
    # The original identity predates every optional policy field.
    old = {"report_language": "en", "human_review": False}
    assert (
        legacy.fingerprint()
        == hashlib.sha256(json.dumps(old, sort_keys=True).encode()).hexdigest()
    )
    strict = RunOptions(quality_policy="strict_v1", human_review=False)
    assert strict.human_review and strict.review_structure == "academic_v2"
    assert (
        strict.fingerprint()
        != RunOptions(human_review=True, review_structure="academic_v2").fingerprint()
    )
    config = load_config()
    before = build_config_summary(config)
    config.eval.strict_batch_size = 8
    assert before["fingerprint"] == build_config_summary(config)["fingerprint"]
    assert (
        build_config_summary(config, "strict_v1")["fingerprint"]
        != before["fingerprint"]
    )


def test_stage_reservations_transfer_only_forward_and_failed_restore_is_atomic():
    budget = CostBudget()
    budget.configure_strict()
    assert budget.remaining() == 200000
    budget.reserve(180000)
    with pytest.raises(ValueError):
        budget.reserve(20001)
    budget.record({"prompt_tokens": 100000})
    budget.release(180000)
    token = set_task_id("adversarial_review.synthesis.1")
    try:
        assert budget.remaining() == 200000
    finally:
        reset_task_id(token)
    assert budget.remaining() == 0
    token = set_task_id("evaluation.initial")
    try:
        assert budget.remaining() == 400000
    finally:
        reset_task_id(token)
    before = budget.snapshot()
    with pytest.raises(ValueError):
        budget.restore(CostBudget().snapshot())
    assert budget.snapshot() == before
    assert CostBudget.from_snapshot(before).snapshot() == before


@pytest.mark.asyncio
async def test_provider_attempt_cap_and_zero_retry_apply_even_to_failed_transport():
    budget = CostBudget()
    budget.configure_strict()
    client = object.__new__(OpenAICompatibleClient)
    client._cost_budget = budget
    client._transport_max_attempts = 3
    client._max_tokens = 1
    sent = []

    async def send(messages, **kwargs):
        sent.append(messages)
        raise TimeoutError()

    client._chat_attempt = send
    with pytest.raises(TimeoutError):
        await client.chat([{"role": "user", "content": "test"}])
    assert (
        len(sent) == 1 and budget.snapshot()["requests_sent"] == 1 and budget.used > 0
    )
    budget._requests_sent = 64
    with pytest.raises(ValueError, match="request_budget_exceeded"):
        await client.chat([])
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_strict_approval_cannot_override_failed_automatic_gate(tmp_path):
    repo = SqliteReviewRepository(tmp_path)
    await repo.initialize()
    await repo.create_run("run", "question", RunOptions(quality_policy="strict_v1"))
    request = ReviewRequest(
        request_id="request",
        run_id="run",
        stage="final_report",
        revision=1,
        report_revision=1,
        report_sha256="a" * 64,
    )
    await repo.pause(
        "run",
        {"report_data": {"metadata": {"automatic_delivery": {"publishable": False}}}},
        request,
    )
    with pytest.raises(ValueError, match="automatic_acceptance_required"):
        await repo.decide(
            "run", HumanDecision(request_id="request", revision=1, action="approve")
        )
    assert (await repo.get_run("run"))["status"] == "awaiting_human"
