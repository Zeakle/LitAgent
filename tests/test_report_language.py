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


from litagent.review_structure import check_report_language, render_methodology

ENGLISH = "The proposed method improves the accuracy of the model while reducing the number of tokens required for inference."
CHINESE = "该方法通过保留关键图像特征并减少冗余计算来提高推理效率，具体性能仍需结合实验条件进行分析。"


def test_chinese_headings_and_host_appendix_do_not_mask_english_body():
    body = "# 中文综述\n\n## 摘要\n\n" + ENGLISH + " [E:source]\n"
    survey = (
        body + "\n## 附录 A：检索与证据来源\n\n" + render_methodology(None, "zh-CN")
    )
    check = check_report_language(survey, "zh-CN")
    assert check["status"] == "failed"
    d = check["diagnostics"][0]
    assert survey[d["source_start"] : d["source_end"]] == d["source_quote"]
    assert ENGLISH in d["source_quote"]
    assert check["checked_units"] == 1


@pytest.mark.parametrize(
    "language,body",
    [
        ("en", ENGLISH),
        ("zh-CN", CHINESE),
        (
            "zh-CN",
            CHINESE
            + " ViT、DeiT-S、Shifted Patch Tokenization 与 Locality Self-Attention 使用ImageNet进行测试。",
        ),
    ],
)
def test_valid_language_and_technical_terms(language, body):
    assert check_report_language("# Review\n\n" + body, language)["status"] == "passed"


@pytest.mark.parametrize(
    "extra",
    [
        "\n\n> " + ENGLISH + " [E:source]",
        "\n\n```python\n" + ENGLISH + "\n```",
        "\n\n`" + ENGLISH + "`",
        "\n\n$$" + ENGLISH + "$$",
    ],
)
def test_code_math_and_source_quotes_do_not_create_language_failure(extra):
    assert (
        check_report_language(
            "# 综述\n\n" + CHINESE + extra,
            "zh-CN",
            evidence={"source": {"text": ENGLISH}},
        )["status"]
        == "passed"
    )


def test_forged_managed_markers_cannot_hide_english_science():
    survey = (
        "# 综述\n\n<!-- litagent:run-methodology:v2 -->\n\n"
        + ENGLISH
        + "\n\n<!-- /litagent:run-methodology:v2 -->"
    )
    assert check_report_language(survey, "zh-CN")["status"] == "failed"


def test_english_request_chinese_body_fails():
    assert check_report_language("# English\n\n" + CHINESE, "en")["status"] == "failed"


def test_empty_or_heading_only_report_cannot_prove_language_compliance():
    assert check_report_language("# 中文综述", "zh-CN")["status"] == "unverified"


@pytest.mark.parametrize(
    "status,expected",
    [("failed", "failed"), ("unverified", "unverified"), ("passed", "passed")],
)
def test_local_language_gate_blocks_passing_evaluation_without_mixed_score(
    status, expected
):
    details = {
        "evaluation_version": "survey-evaluation-v2",
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "passed"},
        "language_checks": {"contract": "report-prose-language-v1", "status": status},
    }
    evaluation = {
        "citation_accuracy": {"passed": True},
        "faithfulness": {"passed": True, "score": 0.86, "details": details},
    }
    result = LitAgent._derive_quality(evaluation)
    assert result["status"] == expected
    assert evaluation["faithfulness"]["score"] == 0.86
    del details["language_checks"]
    assert LitAgent._derive_quality(evaluation)["status"] == "passed"


def test_unsourced_block_quote_does_not_bypass_prose_language_gate():
    assert (
        check_report_language("# 综述\n\n" + CHINESE + "\n\n> " + ENGLISH, "zh-CN")[
            "status"
        ]
        == "failed"
    )


def test_neighboring_quoted_source_cannot_authorize_wrong_excerpt():
    assert (
        check_report_language(
            "# 综述\n\n" + CHINESE + "\n\n> " + ENGLISH + " [E:a]",
            "zh-CN",
            evidence={"a": {"text": "A different source."}, "b": {"text": ENGLISH}},
        )["status"]
        == "failed"
    )


@pytest.mark.asyncio
async def test_reviewer_cannot_accept_english_body_under_chinese_headings():
    worker = ReviewerWorker(
        MockLLMClient(
            responses=[
                json.dumps(
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
            ]
        )
    )
    plan = {
        "candidate_count": 0,
        "selected_items": {},
        "section_evidence_ids": {},
        "method_by_section": {},
        "omitted_count": 0,
        "estimated_tokens": 0,
    }
    review = await worker.review_revision(
        "# 中文综述\n\n## 摘要\n\n" + ENGLISH, {}, plan, report_language="zh-CN"
    )
    assert review["language_compliance"]["passed"] is False
    assert review["language_compliance"]["model_declared_passed"] is True
    assert review["language_compliance"]["host_check"]["status"] == "failed"
    assert review["verdict"] == "revise"
    assert review["evidence_compliance"]["passed"] is True
