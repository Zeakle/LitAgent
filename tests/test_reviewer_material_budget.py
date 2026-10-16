"""Keep complete strict review inputs and bind failures to the actual report."""

import copy
import json
from unittest.mock import AsyncMock

import pytest
from test_batched_evaluation import BatchLLM, materials
from test_quality_control import writing_material
from test_strict_automatic import frozen

from litagent.agents.reviewer import (
    STRICT_REVIEW_REQUEST_CONTRACT,
    ReviewerWorker,
)
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary, normalize_survey_result
from litagent.eval.ragas_eval import RagasFaithfulnessEvaluator
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest, strict_acceptance
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget, strict_request_reservation


def reviewer_fixture():
    """Preserve unused evidence for coverage and complete cited original support."""
    cards, scope, requirements = writing_material()
    selection = EvidenceSelection(
        candidate_count=len(cards),
        selected_items={
            card["evidence_id"]: {
                "evidence_id": card["evidence_id"],
                "paper_id": card["paper_id"],
                "text": card["source_quote"],
                "supporting_text": card["source_quote"],
                "chunk_key": "abstract",
            }
            for card in cards.values()
        },
        section_evidence_ids=scope["section_evidence_ids"],
        method_by_section={
            key: "lexical_fallback" for key in scope["section_evidence_ids"]
        },
        omitted_count=0,
        estimated_tokens=100,
    )
    draft = "EViT improves speed by 50% on DeiT-S. [E:a]"
    review = {
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
        "language_compliance": {"passed": True, "reason": "fixture"},
    }
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(review), model="offline", usage={}
    )
    return cards, selection, requirements, draft, llm


async def request_review(worker, draft, selection, requirements, *, revision, **kwargs):
    """Drive both public consumers rather than bypassing their system prompts."""
    common = dict(
        query="Compare EViT speed under the recorded conditions.",
        review_structure="academic_v2",
        coverage_requirements=requirements,
        **kwargs,
    )
    if revision:
        return await worker.review_revision(draft, {}, selection, **common)
    return await worker.execute(
        SubTask(
            "review",
            "review",
            "reviewer",
            input_data={
                **common,
                "upstream_results": {
                    "synthesis": {
                        "draft": draft,
                        "evidence_selection": selection.to_dict(),
                    }
                },
            },
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
@pytest.mark.parametrize("language", ["en", "zh-CN"])
async def test_complete_review_record_once_without_trimming(revision, policy, language):
    cards, selection, requirements, draft, llm = reviewer_fixture()
    record = {"search_queries": {"q0": "UNIQUE_HOST_QUERY_検索" * 2000}}
    before = copy.deepcopy(selection.to_dict())
    await request_review(
        ReviewerWorker(llm),
        draft,
        selection,
        requirements,
        revision=revision,
        quality_policy=policy,
        report_language=language,
        review_methodology=record,
    )
    messages = llm.chat.await_args.args[0]
    packet = json.loads(messages[1]["content"])
    assert "UNIQUE_HOST_QUERY_検索" not in messages[0]["content"]
    assert packet["review_methodology"] == record
    assert packet["current_draft"] == draft
    assert packet["coverage_requirements"] == requirements
    assert packet["review_request_contract"] == STRICT_REVIEW_REQUEST_CONTRACT
    assert set(packet["cards"]) == set(cards)
    for cid, card in cards.items():
        ref = packet["cards"][cid]["source_ref"]
        assert packet["sources"][ref]["source_quote"] == card["source_quote"]
        assert packet["cards"][cid]["evidence_id"] == card["evidence_id"]
    assert "ONLY against its actual citations" in messages[0]["content"]
    assert selection.to_dict() == before
    assert llm.chat.await_args.kwargs["max_tokens"] == 4096
    assert strict_request_reservation(messages, 4096) < 117964
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_oversized_complete_record_refuses_before_transport(revision, policy):
    _, selection, requirements, draft, llm = reviewer_fixture()
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        await request_review(
            ReviewerWorker(llm),
            draft,
            selection,
            requirements,
            revision=revision,
            quality_policy=policy,
            review_methodology={"query": "完整记录" * 20000},
        )
    llm.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
async def test_legacy_keeps_system_method_record(revision):
    _, selection, requirements, draft, llm = reviewer_fixture()
    await request_review(
        ReviewerWorker(llm),
        draft,
        selection,
        requirements,
        revision=revision,
        quality_policy="legacy",
        review_methodology={"query": "legacy-record"},
    )
    assert "legacy-record" in llm.chat.await_args.args[0][0]["content"]


@pytest.mark.asyncio
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize(
    "mutation", [None, "survey", "fact_cards", "method", "evaluation"]
)
async def test_failed_automatic_snapshot_binding_stays_fail_closed(
    tmp_path, monkeypatch, partial, mutation
):
    report = copy.deepcopy(frozen()["report"])
    report["partial"] = partial
    report["quality"]["status"] = "failed"
    report["metadata"]["review_methodology"] = {"search_queries": ["real query"]}
    before = copy.deepcopy(report)
    service = AsyncMock()
    agent = LitAgent(load_config(), audit_service=service)
    agent._session_id = "offline-review-failed"
    monkeypatch.chdir(tmp_path)
    await agent._automatic_review(report, "EViT speed", {}, None)
    service.submit.assert_not_awaited()
    assert report["survey"] == before["survey"]
    assert report["evaluation"] == before["evaluation"]
    acceptance = report["metadata"]["automatic_acceptance"]
    assert acceptance["actor"] == "system"
    assert acceptance["fact_cards_sha256"] == digest(report["metadata"]["fact_cards"])
    assert acceptance["methodology_sha256"] == digest(
        report["metadata"]["review_methodology"]
    )
    expected = (
        "automatic_report_partial" if partial else "automatic_evaluation_not_passed"
    )
    assert expected in acceptance["failure_codes"]
    if mutation == "survey":
        report["survey"] += " Changed."
    elif mutation == "fact_cards":
        report["metadata"]["fact_cards"] = {}
    elif mutation == "method":
        report["metadata"]["review_methodology"]["search_queries"] = []
    elif mutation == "evaluation":
        report["evaluation"]["faithfulness"]["passed"] = False
    output = normalize_survey_result(report, config_fingerprint="offline")
    actual = output["metadata"]["automatic_acceptance"]
    assert actual["status"] == ("invalidated" if mutation else "failed")
    assert not output["delivery"]["publishable"]
    assert "human_review" not in output["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_unverified_evaluation_preserves_policy_and_upstream_stop(policy):
    survey, context = materials(1)
    context["quality_policy"] = policy
    llm = BatchLLM()
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.stop_after_worker_failure()
    llm._cost_budget = budget
    result = await RagasFaithfulnessEvaluator(load_config(), llm=llm).evaluate(
        survey, context
    )
    assert result.skipped
    assert result.details["quality_policy"] == policy
    assert result.details["upstream_stop_reason"] == "strict_worker_failed"
    gate = strict_acceptance(result.details)
    assert gate["policy"] == policy
    assert gate["human_approval_required"] == (policy == "strict_v1")
    assert gate["status"] == "failed"
    assert llm.calls == []


def test_review_contract_changes_strict_identity_only():
    cfg = load_config()
    legacy = build_config_summary(cfg, "legacy")
    assert "complete-review-materials-v1" not in json.dumps(legacy)
    for policy in ["strict_v1", "strict_auto_v1"]:
        summary = build_config_summary(cfg, policy)
        assert STRICT_REVIEW_REQUEST_CONTRACT in json.dumps(summary)
        assert summary["fingerprint"] != legacy["fingerprint"]
    assert "failed-acceptance-binding-v1" in json.dumps(
        build_config_summary(cfg, "strict_auto_v1")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize(
    "draft, reason",
    [
        ("Claim. [E:absent]", "strict_review_unknown_evidence_refs"),
        ("Claim. [E:a;b]", "strict_review_malformed_evidence_refs"),
    ],
)
async def test_local_citation_errors_make_no_request(revision, draft, reason):
    _, selection, requirements, _, llm = reviewer_fixture()
    with pytest.raises(ValueError, match=reason):
        await request_review(
            ReviewerWorker(llm),
            draft,
            selection,
            requirements,
            revision=revision,
            quality_policy="strict_auto_v1",
        )
    llm.chat.assert_not_awaited()


@pytest.mark.asyncio
async def test_missing_strict_selection_cannot_enter_legacy_context():
    _, _, _, draft, llm = reviewer_fixture()
    with pytest.raises(ValueError, match="strict_review_evidence_missing"):
        await ReviewerWorker(llm).execute(
            SubTask(
                "r",
                "r",
                "reviewer",
                input_data={
                    "quality_policy": "strict_auto_v1",
                    "upstream_results": {"synthesis": {"draft": draft}},
                },
            )
        )
    llm.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize("oversized", [False, True])
async def test_unused_original_sources_are_not_dropped_to_fit(revision, oversized):
    _, selection, requirements, draft, llm = reviewer_fixture()
    original = "UNUSED ORIGINAL source preserves conditions and limitations. "
    original *= 3000 if oversized else 2
    selection.selected_items["unused"] = {
        "evidence_id": "unused",
        "paper_id": "unused-paper",
        "text": "Compressed extraction is not support.",
        "supporting_text": original,
        "chunk_key": "abstract",
    }
    for ids in selection.section_evidence_ids.values():
        ids.append("unused")
    selection.candidate_count += 1
    call = request_review(
        ReviewerWorker(llm),
        draft,
        selection,
        requirements,
        revision=revision,
        quality_policy="strict_auto_v1",
    )
    if oversized:
        with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
            await call
        llm.chat.assert_not_awaited()
    else:
        await call
        packet = json.loads(llm.chat.await_args.args[0][1]["content"])
        assert len(packet["cards"]) == 2
        assert original in [s["source_quote"] for s in packet["sources"].values()]
        assert "Compressed extraction is not support." not in json.dumps(packet)


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_review_scope_projection_retains_exact_reverse_permissions(
    revision, policy
):
    from litagent.quality import build_fact_cards, grounded_messages

    _, selection, requirements, draft, llm = reviewer_fixture()
    expected = json.loads(
        grounded_messages(
            build_fact_cards(selection.selected_items),
            "academic_v2",
            "en",
            "scope",
            evidence_selection=selection,
        )[1]["content"]
    )
    before = copy.deepcopy(selection.to_dict())
    await request_review(
        ReviewerWorker(llm),
        draft,
        selection,
        requirements,
        revision=revision,
        quality_policy=policy,
    )
    packet = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert (
        "complete section permission map" in llm.chat.await_args.args[0][0]["content"]
    )
    for section in packet["section_contract"]:
        key = section["section_key"]
        original = next(
            s for s in expected["section_contract"] if s["section_key"] == key
        )
        actual = {
            cid
            for cid, card in packet["cards"].items()
            if key in card["allowed_section_keys"]
        }
        assert actual == set(original["allowed_fact_card_ids"])
        assert section == {
            k: v for k, v in original.items() if k != "allowed_fact_card_ids"
        }
    assert list(packet["sources"].values()) == list(expected["sources"].values())
    for cid, card in packet["cards"].items():
        original_card = expected["cards"][cid]
        assert {k: v for k, v in card.items() if k != "source_ref"} == {
            k: v for k, v in original_card.items() if k != "source_ref"
        }
        assert (
            packet["sources"][card["source_ref"]]
            == expected["sources"][original_card["source_ref"]]
        )
    assert packet["current_draft"] == draft
    assert packet["coverage_requirements"] == requirements
    assert before == selection.to_dict()
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
async def test_revision_review_preserves_full_previous_advisory():
    _, selection, requirements, draft, llm = reviewer_fixture()
    previous = {
        "issues": [
            {
                "source_quote": "OLD_FULL_QUOTE" * 100,
                "existing_evidence_ids": ["a"],
                "issue": "old issue",
            }
        ],
        "binding": {"status": "unlocated", "diagnostic": "preserve"},
    }
    await ReviewerWorker(llm).review_revision(
        draft,
        previous,
        selection,
        query="EViT speed",
        report_language="en",
        review_structure="academic_v2",
        quality_policy="strict_auto_v1",
        coverage_requirements=requirements,
    )
    packet = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert packet["previous_review_advisory"] == previous
    assert packet["current_draft"] == draft
    assert packet["coverage_requirements"] == requirements


@pytest.mark.asyncio
@pytest.mark.parametrize("revision", [False, True])
async def test_review_budget_rejection_keeps_local_input_and_redacted_trace(revision):
    from litagent.observability.context import reset_task_id, set_task_id
    from litagent.observability.recorder import RedactingTraceHook

    _, selection, requirements, draft, llm = reviewer_fixture()
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    llm._cost_budget = budget
    before = budget.snapshot()
    local, external = [], []
    redacted = RedactingTraceHook(
        lambda e, d: external.append((e, d)), payload_mode="metadata_only"
    )

    def emit(event, data):
        local.append((event, data))
        redacted(event, data)

    worker = ReviewerWorker(llm, trace_hook=emit)
    token = set_task_id("adversarial:reviewer:r2")
    try:
        with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
            await request_review(
                worker,
                draft,
                selection,
                requirements,
                revision=revision,
                quality_policy="strict_auto_v1",
                review_methodology={"query": "PRIVATE_RECORD" * 20000},
            )
    finally:
        reset_task_id(token)
    llm.chat.assert_not_awaited()
    assert budget.snapshot() == before
    assert len(local) == len(external) == 1
    event, data = local[0]
    assert event == "writing.request.rejected"
    assert data["task_id"] == "adversarial:reviewer:r2"
    assert data["phase"] == ("revision_review" if revision else "review")
    assert data["contract_version"] == STRICT_REVIEW_REQUEST_CONTRACT
    assert data["response_format"] == {"type": "json_object"}
    assert data["max_tokens"] == 4096
    assert data["budget"]["reservation"] == strict_request_reservation(
        data["messages"], 4096, data["response_format"]
    )
    assert json.loads(data["messages"][1]["content"])["current_draft"] == draft
    assert data["input_sha256"] == digest([data["messages"], data["response_format"]])
    assert "PRIVATE_RECORD" in json.dumps(local)
    assert "PRIVATE_RECORD" not in json.dumps(external)
    assert "messages" not in external[0][1]
    assert external[0][1]["budget"] == data["budget"]


@pytest.mark.asyncio
async def test_review_budget_hook_failure_does_not_change_original_refusal():
    _, selection, requirements, draft, llm = reviewer_fixture()

    def broken_hook(*args):
        raise RuntimeError("trace unavailable")

    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        await request_review(
            ReviewerWorker(llm, trace_hook=broken_hook),
            draft,
            selection,
            requirements,
            revision=False,
            quality_policy="strict_auto_v1",
            review_methodology={"query": "large" * 50000},
        )
    llm.chat.assert_not_awaited()
