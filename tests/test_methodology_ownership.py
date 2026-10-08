"""Guard host-owned methodology against model revisions and historical repair."""

import hashlib
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.review_structure import (
    METHOD_BEGIN,
    METHOD_END,
    appendix_title,
    assemble_methodology,
    build_methodology,
    check_methodology,
    prepare_authored_body,
    render_methodology,
    structure_sections,
)
from litagent.runner import LitAgent


@pytest.fixture
def recorded_methodology():
    """Use the recorded timestamp whose model mutation broke the live survey."""
    record = build_methodology(
        "ViT and DeiT",
        {
            "__search_queries__": {"search_0": "ViT DeiT image classification"},
            "search_0": [{"paper_id": "p"}],
            "dedup": [{"paper_id": "p"}],
            "relevance_gate": [{"paper_id": "p"}],
        },
        source_outcomes=[
            {
                "task_id": "search_0",
                "source": "arxiv",
                "status": "success",
                "result_count": 1,
                "reason_code": None,
            }
        ],
        started_at=1791435077.4014654,
    )
    record["evidence_counts"] = {"candidate": 1, "selected": 1}
    return record


@pytest.fixture
def selected_evidence():
    """Provide the real serialized evidence contract without external services."""
    return EvidenceSelection(
        candidate_count=1,
        selected_items={
            "a": {"evidence_id": "a", "paper_id": "p", "text": "A supported finding"}
        },
        section_evidence_ids={"methods": ["a"]},
        method_by_section={"methods": "lexical_fallback"},
        omitted_count=0,
        estimated_tokens=20,
    )


def authored_body(profile, language, finding="A supported finding"):
    """Return a complete authored body independent from host disclosure syntax."""
    return "# Review\n\n" + "\n\n".join(
        f"## {heading}\n\n{finding} [E:a]."
        for key, heading in structure_sections(profile, language)
        if key != "run_methodology"
    )


def damaged_block(record, language, damage):
    """Model the live timestamp corruption and malformed ownership boundaries."""
    block = render_methodology(record, language)
    if damage == "timestamp":
        return block.replace(
            "2026-10-08T04:51:17.401465+00:00 (UTC; raw=1791435077.4014654)",
            "2026-10-08T04:51:17.4014654)",
        )
    if damage == "missing_begin":
        return block.replace(METHOD_BEGIN, "")
    if damage == "missing_end":
        return block.replace(METHOD_END, "")
    if damage == "duplicate":
        return block + "\n\n" + block
    raise AssertionError(f"Unknown damage: {damage}")


@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("builder", ["revise", "rewrite_with_evidence"])
def test_revision_messages_exclude_canonical_host_disclosures(
    profile, language, builder, recorded_methodology, selected_evidence
):
    """Only authored prose may be offered to a model for revision."""
    body = authored_body(profile, language)
    draft = assemble_methodology(body, profile, language, recorded_methodology)
    worker = SynthesisWorker(ScriptedLLM("unused"))
    kwargs = {
        "review_structure": profile,
        "report_language": language,
        "review_methodology": recorded_methodology,
    }
    if builder == "revise":
        messages = worker.revise(
            draft, "Clarify the finding", selected_evidence, **kwargs
        )
    else:
        messages = worker.rewrite_with_evidence(
            draft, "ledger", "diagnostics", **kwargs
        )
    current_draft = (
        messages[1]["content"]
        .split("<current_draft>", 1)[1]
        .split("</current_draft>", 1)[0]
    )
    assert METHOD_BEGIN not in current_draft
    assert METHOD_END not in current_draft
    assert "2026-10-08T04:51:17" not in current_draft
    assert "A supported finding" in current_draft
    assert "<run_methodology>" in messages[0]["content"]


@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_canonical_assembly_is_unique_idempotent_and_method_verified(
    profile, language, recorded_methodology
):
    """Repeated host assembly preserves scientific prose and exact runtime data."""
    record_before = deepcopy(recorded_methodology)
    first = assemble_methodology(
        authored_body(profile, language), profile, language, recorded_methodology
    )
    second = assemble_methodology(first, profile, language, recorded_methodology)
    assert second == first
    assert second.count(METHOD_BEGIN) == second.count(METHOD_END) == 1
    assert (
        check_methodology(second, profile, language, recorded_methodology)["status"]
        == "passed"
    )
    assert recorded_methodology == record_before


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_unmarked_authored_appendix_is_preserved_and_remains_blocked(
    language, recorded_methodology
):
    """A matching heading does not grant ownership of user-authored prose."""
    authored = "Scientific appendix content remains visible [E:a]."
    body = authored_body("academic_v2", language)
    draft = body + f"\n\n## {appendix_title(language)}\n\n{authored}"
    assembled = assemble_methodology(
        draft, "academic_v2", language, recorded_methodology
    )
    assert authored in assembled
    assert (
        check_methodology(assembled, "academic_v2", language, recorded_methodology)[
            "status"
        ]
        == "failed"
    )


@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
@pytest.mark.parametrize(
    "damage", ["timestamp", "missing_begin", "missing_end", "duplicate"]
)
def test_damaged_historical_managed_input_is_rejected_without_silent_repair(
    profile, damage, recorded_methodology
):
    """Never replace or append over a previously corrupted managed disclosure."""
    draft = assemble_methodology(
        authored_body(profile, "zh-CN"), profile, "zh-CN", recorded_methodology
    )
    canonical = render_methodology(recorded_methodology, "zh-CN")
    damaged = draft.replace(
        canonical, damaged_block(recorded_methodology, "zh-CN", damage)
    )
    assert damaged != draft
    with pytest.raises(ValueError, match="methodology_managed_block_invalid"):
        assemble_methodology(damaged, profile, "zh-CN", recorded_methodology)


class ScriptedLLM:
    """Return deterministic revision output while recording the actual messages."""

    def __init__(self, output):
        """Store the single model response for an isolated bounded review."""
        self.output = output
        self.messages = []

    async def chat(self, messages, **kwargs):
        """Capture revision messages without contacting a paid model."""
        self.messages.append(deepcopy(messages))
        return LLMResponse(content=self.output)


class PreparedSynthesis(SynthesisWorker):
    """Reuse real revision builders with a precomputed initial synthesis result."""

    def __init__(self, llm, initial, record, selection):
        """Avoid unrelated retrieval while preserving production revision behavior."""
        super().__init__(llm)
        self.initial = initial
        self.record = record
        self.selection = selection

    async def execute(self, task):
        """Return a legitimate initial report with a real evidence-selection schema."""
        return {
            "draft": self.initial,
            "evidence_selection": self.selection.to_dict(),
            "review_methodology": self.record,
        }


class PermissiveReviewer:
    """Request one revision then accept to expose missing host validation."""

    async def execute(self, task):
        """Ask for a clarity revision on the current valid report."""
        return {
            "score": 0.5,
            "verdict": "revise",
            "evidence_compliance": {"passed": True},
        }

    async def review_revision(self, draft, comments, selection, **kwargs):
        """A model reviewer cannot authorize replacement of host-owned fields."""
        return {
            "score": 1.0,
            "verdict": "accept",
            "evidence_compliance": {"passed": True},
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
@pytest.mark.parametrize(
    "damage",
    [
        "timestamp",
        "missing_begin",
        "missing_end",
        "duplicate",
        "canonical_copy",
        "forged_appendix",
    ],
)
async def test_adversarial_rejects_model_authored_managed_disclosures(
    profile, damage, recorded_methodology, selected_evidence
):
    """Even canonical-looking model output must not own reserved runtime blocks."""
    original_body = authored_body(profile, "zh-CN")
    initial = assemble_methodology(
        original_body, profile, "zh-CN", recorded_methodology
    )
    new_body = authored_body(profile, "zh-CN", "A clearer supported finding")
    block = (
        render_methodology(recorded_methodology, "zh-CN")
        if damage == "canonical_copy"
        else (
            "Claimed system search disclosure: fabricated counts"
            if damage == "forged_appendix"
            else damaged_block(recorded_methodology, "zh-CN", damage)
        )
    )
    heading = (
        appendix_title("zh-CN") if profile == "academic_v2" else "检索与证据选择方法"
    )
    llm = ScriptedLLM(new_body + f"\n\n## {heading}\n\n{block}")
    synthesis = PreparedSynthesis(llm, initial, recorded_methodology, selected_evidence)
    worker = AdversarialReviewWorker(llm, synthesis, PermissiveReviewer(), max_rounds=2)
    result = await worker.execute(
        SubTask(
            task_id="adversarial",
            description="review",
            agent_type="adversarial_review",
            input_data={
                "review_structure": profile,
                "report_language": "zh-CN",
                "query": "ViT",
            },
        )
    )
    assert result["final_draft"] == initial
    assert result["accepted"] is False
    assert result["degradation_reasons"]
    assert (
        check_methodology(
            result["final_draft"], profile, "zh-CN", recorded_methodology
        )["status"]
        == "passed"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
async def test_adversarial_accepts_authored_only_revision_and_reassembles_host_block(
    profile, recorded_methodology, selected_evidence
):
    """A valid body revision remains usable and receives one trusted disclosure."""
    initial = assemble_methodology(
        authored_body(profile, "zh-CN"), profile, "zh-CN", recorded_methodology
    )
    new_body = authored_body(profile, "zh-CN", "A clearer supported finding")
    llm = ScriptedLLM(new_body)
    worker = AdversarialReviewWorker(
        llm,
        PreparedSynthesis(llm, initial, recorded_methodology, selected_evidence),
        PermissiveReviewer(),
        max_rounds=2,
    )
    result = await worker.execute(
        SubTask(
            task_id="adversarial",
            description="review",
            agent_type="adversarial_review",
            input_data={"review_structure": profile, "report_language": "zh-CN"},
        )
    )
    assert "A clearer supported finding" in result["final_draft"]
    assert result["accepted"] is True
    assert result["final_draft"].count(METHOD_BEGIN) == 1
    assert (
        check_methodology(
            result["final_draft"], profile, "zh-CN", recorded_methodology
        )["status"]
        == "passed"
    )


@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_authored_body_preparation_preserves_nonmanaged_appendix_and_prose(
    profile, language, recorded_methodology
):
    """Remove canonical disclosure while retaining distinct authored appendices."""
    body = authored_body(profile, language)
    extra = "## Supplementary scientific evidence\n\nAuthored equations and qualifications [E:a]."
    body += "\n\n" + extra
    report = assemble_methodology(body, profile, language, recorded_methodology)
    prepared = prepare_authored_body(report, profile, language, recorded_methodology)
    assert METHOD_BEGIN not in prepared
    assert METHOD_END not in prepared
    assert extra in prepared
    for _, heading in structure_sections(profile, language):
        if heading not in {
            "检索与证据选择方法",
            "Search and Evidence Selection Methods",
        }:
            assert f"## {heading}" in prepared
    assert prepared.count("A supported finding [E:a].") == 7


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_authored_body_preparation_refuses_mixed_owned_appendix_without_deleting_prose(
    language, recorded_methodology
):
    """Mixed ownership is an error instead of permission to erase authored prose."""
    report = assemble_methodology(
        authored_body("academic_v2", language),
        "academic_v2",
        language,
        recorded_methodology,
    )
    mixed = report + "\nAuthor's scientific caveat remains important [E:a]."
    with pytest.raises(ValueError, match="methodology_managed_block_invalid"):
        prepare_authored_body(mixed, "academic_v2", language, recorded_methodology)
    assert "Author's scientific caveat" in mixed


def runner_report_and_results(profile, record):
    """Build frozen input, real extraction output, and repair diagnostics."""
    survey = assemble_methodology(
        authored_body(profile, "zh-CN"), profile, "zh-CN", record
    )
    diagnostic = {
        "claim_id": "claim_0",
        "kind": "scientific",
        "claim_text": "A supported finding",
        "source_quote": "A supported finding [E:a].",
        "evidence_ids": ["a"],
        "alignment_status": "exact",
        "citation_scope": "exact",
        "reason_code": "insufficient_support",
        "explanation": "Clarify the conditions.",
    }
    evaluation = {
        "citation_accuracy": {"score": 1.0, "passed": True},
        "faithfulness": {
            "score": 0.4,
            "passed": False,
            "details": {"unsupported_claims": [diagnostic]},
        },
    }
    report = {
        "survey": survey,
        "evaluation": evaluation,
        "quality": {"status": "failed", "failed_metrics": ["faithfulness"]},
        "delivery": {"status": "blocked", "publishable": False},
        "metadata": {
            "report_revision": 1,
            "source_sha256": hashlib.sha256(survey.encode("utf-8")).hexdigest(),
            "review_structure": profile,
            "review_methodology": deepcopy(record),
        },
    }
    evidence = {
        "evidence_id": "a",
        "paper_id": "p",
        "paper_title": "Original paper",
        "text": "A supported finding",
        "supporting_text": "A supported finding holds under the stated conditions.",
    }
    results = {
        "extract": [
            {
                "paper_id": "p",
                "claims": ["A supported finding"],
                "evidence_items": [evidence],
            }
        ],
        "adversarial_review": {
            "review_structure": profile,
            "review_methodology": deepcopy(record),
            "report_language": "zh-CN",
            "evidence_selection": {
                "valid": True,
                "candidate_count": 1,
                "section_evidence_ids": {"methods": ["a"]},
                "method_by_section": {"methods": "lexical_fallback"},
            },
        },
    }
    return report, results


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ["academic_v2", "method_transparent_v1"])
@pytest.mark.parametrize("flow", ["evidence_rewrite", "human_revision"])
@pytest.mark.parametrize(
    "damage", ["canonical_copy", "timestamp", "missing_end", "forged_appendix"]
)
async def test_runner_rejects_model_disclosures_before_extra_evaluation_or_review(
    profile, flow, damage, recorded_methodology
):
    """Reject reserved output atomically before spending on another quality stage."""
    report, results = runner_report_and_results(profile, recorded_methodology)
    before_report, before_results = deepcopy(report), deepcopy(results)
    initial_evaluation, initial_quality = report["evaluation"], report["quality"]
    body = authored_body(profile, "zh-CN", "A clearer supported finding")
    heading = (
        appendix_title("zh-CN") if profile == "academic_v2" else "检索与证据选择方法"
    )
    if damage == "canonical_copy":
        block = render_methodology(recorded_methodology, "zh-CN")
    elif damage == "forged_appendix":
        block = "Claimed system search disclosure: fabricated counts"
    else:
        block = damaged_block(recorded_methodology, "zh-CN", damage)
    llm = ScriptedLLM(body + f"\n\n## {heading}\n\n{block}")
    agent = LitAgent(load_config())
    agent._llm = llm
    agent._synthesis = SynthesisWorker(llm)
    agent._evaluate = AsyncMock(
        side_effect=AssertionError("Unexpected paid evaluator stage")
    )
    agent._reviewer = PermissiveReviewer()
    agent._reviewer.review_revision = AsyncMock(
        side_effect=AssertionError("Unexpected paid reviewer stage")
    )
    if flow == "evidence_rewrite":
        outcome = await agent._attempt_evidence_rewrite(
            query="ViT",
            report_data=report,
            results=results,
            initial_evaluation=initial_evaluation,
            initial_quality=initial_quality,
            report_language="zh-CN",
        )
        assert outcome.attempted is True
        assert outcome.committed is False
        assert outcome.reason_code == "methodology_model_authored"
        assert outcome.candidate_evaluation is None
        assert outcome.candidate_quality is None
    else:
        returned, outcome = await agent._attempt_human_revision(
            query="ViT",
            report_data=report,
            results=results,
            feedback="Clarify the conditions",
            report_language="zh-CN",
        )
        assert returned is report
        assert outcome["committed"] is False
        assert outcome["reason_code"] == "methodology_model_authored"
    assert report == before_report
    assert results == before_results
    assert report["evaluation"] is initial_evaluation
    assert report["quality"] is initial_quality
    assert (
        hashlib.sha256(report["survey"].encode("utf-8")).hexdigest()
        == before_report["metadata"]["source_sha256"]
    )
    assert len(llm.messages) == 1
    agent._evaluate.assert_not_awaited()
    agent._reviewer.review_revision.assert_not_awaited()
    current_draft = (
        llm.messages[0][1]["content"]
        .split("<current_draft>", 1)[1]
        .split("</current_draft>", 1)[0]
    )
    assert METHOD_BEGIN not in current_draft
    assert METHOD_END not in current_draft
