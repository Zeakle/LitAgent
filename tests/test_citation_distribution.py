"""Keep selected evidence optional and summaries free of whole-inventory citations."""

import hashlib
import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.audit_context import SurveyAuditContext, check_structure
from litagent.context.evidence_selector import EvidenceSelection
from litagent.demo import project_flow_demo_view
from litagent.llm.client import MockLLMClient, OpenAICompatibleClient
from litagent.observability.recorder import ArchiveRepository
from litagent.review_structure import (
    find_summary_citation_dumps,
    structure_instruction,
    summary_section_spans,
)
from litagent.runner import LitAgent

spec = importlib.util.spec_from_file_location(
    "citation_repair", Path(__file__).parents[1] / "scripts/repair_report_citations.py"
)
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def selection(n=8):
    """Expose available evidence without imposing a citation coverage target."""
    items = {
        f"e{i}": {"evidence_id": f"e{i}", "text": f"Finding {i}", "paper_id": f"p{i}"}
        for i in range(n)
    }
    return EvidenceSelection(
        n,
        items,
        {"methods": list(items), "abstract": list(items)},
        {"methods": "lexical_fallback", "abstract": "selected_union"},
        0,
        100,
    )


def cluster(n=8):
    return " ".join(f"[E:e{i}]" for i in range(n))


@pytest.mark.parametrize(
    "heading", ["摘要", "结论", "Abstract", "Conclusion", "Conclusions"]
)
def test_summary_inventory_dump_is_located_exactly(heading):
    survey = f"# Title\n\n## {heading}\n\nThis review has eight evidence items. {cluster()}\n\n## Methods\nFinding [E:e0]."
    issues = find_summary_citation_dumps(survey, list(selection().selected_items))
    assert len(issues) == 1
    issue = issues[0]
    assert issue["section"] == heading and issue["source_quote"] == cluster()
    assert survey[issue["source_start"] : issue["source_end"]] == cluster()


@pytest.mark.parametrize(
    "content",
    [
        "# Title\n\n## Methods\nFacts {refs}",
        "# Title\n\n```markdown\n## Abstract\nExample {refs}\n```",
        "# Title\n\n> ## Abstract\n> Example {refs}",
        "# Title\n\n## Abstract\nExample `{refs}`",
        "# Title\n\n## Abstract\nFirst fact [E:e0]. Second fact [E:e1].",
        "# Title\n\n## Abstract\nFirst [E:e0] [E:e1] [E:e2] [E:e3]. Second [E:e4] [E:e5] [E:e6] [E:e7].",
    ],
)
def test_direct_distributed_body_or_code_citations_are_not_inventory_dumps(content):
    assert (
        find_summary_citation_dumps(
            content.format(refs=cluster()), list(selection().selected_items)
        )
        == []
    )


@pytest.mark.parametrize("n,expected", [(2, False), (7, False), (8, True), (60, True)])
def test_inventory_boundary_does_not_ban_small_scientific_summary_citations(
    n, expected
):
    issues = find_summary_citation_dumps(
        f"# T\n\n## Abstract\nA supported comparison {cluster(n)}",
        list(selection(n).selected_items),
    )
    assert bool(issues) == expected


def test_unused_selected_evidence_does_not_fail_local_reviewer():
    worker = ReviewerWorker(MockLLMClient())
    review = {
        "score": 0.95,
        "verdict": "accept",
        "evidence_compliance": {
            "passed": True,
            "unsupported_claims": [],
            "unknown_evidence_ids": [],
        },
    }
    result = worker._enforce_evidence_contract(
        "# Title\n\n## Abstract\nFinding [E:e0].", selection(), review
    )
    assert result["verdict"] == "accept" and result["score"] == 0.95
    assert result["evidence_compliance"]["passed"]


def test_reviewer_local_guard_rejects_model_accept_without_calling_again():
    llm = MockLLMClient()
    worker = ReviewerWorker(llm)
    review = {
        "score": 0.99,
        "verdict": "accept",
        "evidence_compliance": {
            "passed": True,
            "unsupported_claims": [],
            "unknown_evidence_ids": [],
        },
    }
    result = worker._enforce_evidence_contract(
        "# T\n\n## Abstract\nStudy scope. " + cluster(), selection(), review
    )
    assert result["verdict"] == "revise" and result["score"] < 0.8
    assert result["evidence_compliance"]["unsupported_claims"] == []
    assert result["evidence_compliance"]["citation_placement_issues"]
    adv = AdversarialReviewWorker(llm, SynthesisWorker(llm), worker)
    assert not adv._review_passes(result)
    feedback = adv._format_review_for_revision(result)
    assert "not a mandatory citation list" in feedback and "Do not relocate" in feedback


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_structure_quality_gate_blocks_inventory_without_changing_support_score(
    language,
):
    profile = "academic_v1"
    titles = (
        structure_instruction(profile, language)
        .split("Use these H2 sections in order: ")[1]
        .split(".\n")[0]
        .split(" -> ")
    )
    survey = "# T\n\n" + "\n\n".join(
        f"## {title}\n\nContent." + (cluster() if i == 0 else "")
        for i, title in enumerate(titles)
    )
    context = SurveyAuditContext(
        query="q", review_structure=profile, report_language=language
    )
    assert (
        check_structure(survey, context)["status"] == "passed"
    )  # Old caller is compatible.
    checks = check_structure(
        survey, context, selected_evidence_ids=list(selection().selected_items)
    )
    assert checks["status"] == "failed" and checks["citation_placement_issues"]
    assert (
        LitAgent._derive_quality(
            {
                "citation_accuracy": {"passed": True},
                "faithfulness": {
                    "passed": True,
                    "score": 0.9,
                    "details": {
                        "evaluation_version": "survey-evaluation-v2",
                        "structure_checks": checks,
                        "method_checks": {"status": "not_applicable"},
                    },
                },
            }
        )["status"]
        == "failed"
    )


@pytest.mark.parametrize("profile", ["academic_v1", "method_transparent_v1"])
def test_generation_review_and_all_revisions_share_placement_contract(profile):
    from litagent.agents.reviewer import REVIEWER_INSTRUCTIONS
    from litagent.review_structure import EVIDENCE_ATTRIBUTION_INSTRUCTION

    assert "OPTIONAL allowance" in structure_instruction(profile)
    assert "An unused selected ID alone" in REVIEWER_INSTRUCTIONS
    worker = SynthesisWorker(MockLLMClient())
    for messages in [
        worker.revise("draft", "feedback", selection(), review_structure=profile),
        worker.rewrite_with_evidence(
            "draft", "ledger", "diagnostic", review_structure=profile
        ),
    ]:
        assert EVIDENCE_ATTRIBUTION_INSTRUCTION in messages[0]["content"]
        assert "Do not append" in messages[0]["content"]
    assert (
        "Do not demand citation of every allowed ID" in EVIDENCE_ATTRIBUTION_INSTRUCTION
    )


def source_artifact(repository):
    papers = [
        {
            "paper_id": f"p{i}",
            "title": f"Paper {i}",
            "authors": ["Test Author"],
            "year": 2022,
        }
        for i in range(8)
    ]
    items = [
        {
            **item,
            "paper_title": f"Paper {i}",
            "source_locator": "abstract",
            "content_scope": "abstract",
        }
        for i, item in enumerate(selection().selected_items.values())
    ]
    artifact = {
        "run_id": "original",
        "query": "Compare methods",
        "report": {
            "survey": "# Title\n\n## Abstract\nScope. "
            + cluster()
            + "\n\n## Methods\nFinding 0 [E:e0]. Finding 1 [E:e1].\n",
            "metadata": {
                "accepted": True,
                "final_score": 0.97,
                "human_review": {"status": "approved"},
                "citation_style": "apa7",
                "report_language": "en",
            },
            "review_history": [{"score": 0.97}],
            "evaluation": {"faithfulness": {"score": 0.97, "passed": True}},
            "quality": {"status": "passed"},
            "delivery": {"status": "ready"},
        },
        "nodes": {
            f"worker:{k}": {"kind": "worker", "task_id": k, "output": out}
            for k, out in {
                "dedup": papers,
                "relevance_gate": papers,
                "extract": [{"evidence_items": items}],
                "graph_analysis": {},
            }.items()
        },
    }
    repository.save(artifact)
    return artifact


def test_local_repair_preserves_source_body_and_clears_scores_approval_and_trace(
    tmp_path, monkeypatch
):
    no_provider = AsyncMock(
        side_effect=AssertionError("No paid call authorized for a layout edit")
    )
    monkeypatch.setattr(OpenAICompatibleClient, "chat", no_provider)
    repository = ArchiveRepository(tmp_path)
    original = source_artifact(repository)
    raw = repository.path_for("original").read_bytes()
    result = repair.create_repair_draft(
        repository,
        source_id="original",
        source_sha256=hashlib.sha256(raw).hexdigest(),
        run_id="repaired",
        abstract="This review compares supplied methods.",
        abstract_title="Abstract",
    )
    assert repository.path_for("original").read_bytes() == raw
    report = result["report"]
    assert (
        report["survey"].split("## Methods")[1]
        == original["report"]["survey"].split("## Methods")[1]
    )
    assert "[E:" not in report["survey"].split("## Methods")[0]
    assert (
        report["metadata"]["final_score"] is None and not report["metadata"]["accepted"]
    )
    assert (
        report["metadata"]["human_review"]["status"] == "not_requested"
        and report["review_history"] == []
    )
    assert all(v["skipped"] for v in report["evaluation"].values())
    assert (
        result["quality"]["status"] == "unverified"
        and result["delivery"]["status"] == "needs_review"
    )
    assert result["usage"]["total_tokens"] == 0 and result["trace"]["trace_id"] is None
    assert (
        result["report"]["metadata"]["source_artifact_sha256"]
        == hashlib.sha256(raw).hexdigest()
    )
    assert no_provider.await_count == 0
    for style in ["numbered", "ieee", "apa7"]:
        view = project_flow_demo_view(result, source="archive", citation_style=style)
        reader = view["rendered_report"]
        assert not [
            w for w in reader["warnings"] if w.startswith("unresolved_evidence:")
        ]
        assert len(reader["citation_occurrences"]) == 2
        if style != "numbered":
            assert len(reader["citation_links"]) == 2
        assert "This review compares supplied methods." in reader["html"]
    with pytest.raises(ValueError, match="already_exists"):
        repair.create_repair_draft(
            repository,
            source_id="original",
            source_sha256=hashlib.sha256(raw).hexdigest(),
            run_id="repaired",
            abstract="Change",
            abstract_title="Abstract",
        )


@pytest.mark.parametrize("case", ["bad_sha", "same_run", "unsafe_run", "unsafe_source"])
def test_repair_binding_and_path_guards(tmp_path, case):
    repository = ArchiveRepository(tmp_path)
    source_artifact(repository)
    args = {
        "source_id": "original",
        "source_sha256": hashlib.sha256(
            repository.path_for("original").read_bytes()
        ).hexdigest(),
        "run_id": "repaired",
        "abstract": "Scope",
        "abstract_title": "Abstract",
    }
    if case == "bad_sha":
        args["source_sha256"] = "0" * 64
    elif case == "same_run":
        args["run_id"] = "original"
    elif case == "unsafe_run":
        args["run_id"] = "../overwrite"
    else:
        args["source_id"] = "../secret"
    with pytest.raises(ValueError):
        repair.create_repair_draft(repository, **args)
    assert not repository.path_for("repaired").exists()


@pytest.mark.parametrize("replacement", ["", "# Another title", "## Another section"])
def test_repair_cannot_replace_structure_or_empty_summary(replacement):
    with pytest.raises(ValueError):
        repair.replace_abstract(
            "# T\n\n## Abstract\nScope.\n\n## Body\nFact.", replacement, "Abstract"
        )
