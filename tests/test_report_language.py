"""Verify report-language propagation across every generation path."""

import json

import pytest

from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.llm.client import MockLLMClient
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.runner import LitAgent
from litagent.workbench import RunOptions, language_instruction


def test_language_instruction_is_explicit_and_preserves_identifiers():
    chinese = language_instruction("zh-CN")
    assert "Simplified Chinese" in chinese
    assert "Evidence IDs" in chinese
    assert "do not translate" in chinese
    assert "English" in language_instruction("en")


def test_revision_and_rewrite_receive_the_same_language_contract():
    worker = SynthesisWorker(llm=None)
    revision = worker.revise(
        "# Draft",
        "Fix this",
        {
            "candidate_count": 0,
            "selected_items": {},
            "section_evidence_ids": {},
            "method_by_section": {},
            "omitted_count": 0,
            "estimated_tokens": 0,
        },
        report_language="zh-CN",
    )
    rewrite = worker.rewrite_with_evidence(
        "# Draft", "ledger", "diagnostics", report_language="zh-CN"
    )
    assert "Simplified Chinese" in revision[0]["content"]
    assert "Simplified Chinese" in rewrite[0]["content"]


def test_runner_injects_options_without_overwriting_query():
    graph = TaskGraph()
    graph.add_task(SubTask("synthesis", "write", "synthesis", {"query": "q"}))
    options = RunOptions(report_language="zh-CN", human_review=False)
    LitAgent._apply_run_options_to_graph(graph, options)
    task = graph.get_task("synthesis")
    assert task.input_data == {"query": "q", "report_language": "zh-CN"}


def test_run_options_do_not_expose_human_review_as_a_worker_input():
    graph = TaskGraph()
    graph.add_task(SubTask("search", "search", "search"))
    LitAgent._apply_run_options_to_graph(
        graph, RunOptions(report_language="en", human_review=True)
    )
    assert graph.get_task("search").input_data == {"report_language": "en"}


def test_survey_cli_exposes_report_language_without_enabling_interactive_review():
    from litagent.cli import build_parser

    parsed = build_parser().parse_args(["survey", "query", "--language", "zh-CN"])
    assert parsed.language == "zh-CN"


@pytest.mark.asyncio
async def test_reviewer_rejects_obvious_wrong_language_without_changing_evidence():
    response = json.dumps(
        {
            "score": 0.95,
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
            "language_compliance": {"passed": True, "reason": "ok"},
        }
    )
    worker = ReviewerWorker(MockLLMClient(responses=[response]))
    draft = (
        "This literature survey remains entirely in English even though the "
        "requested report language is Simplified Chinese and every paragraph "
        "continues to use English prose without any translated explanation."
    )

    result = await worker.review_revision(
        draft,
        {},
        {
            "candidate_count": 0,
            "selected_items": {},
            "section_evidence_ids": {},
            "method_by_section": {},
            "omitted_count": 0,
            "estimated_tokens": 0,
        },
        report_language="zh-CN",
    )

    assert result["language_compliance"]["passed"] is False
    assert result["verdict"] == "revise"
    assert result["evidence_compliance"]["passed"] is True
