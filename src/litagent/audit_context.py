"""Define report audit inputs and deterministic structure checks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict, Field

from litagent.review_structure import (
    appendix_title,
    find_summary_citation_dumps,
    structure_sections,
)


class AuditTopic(BaseModel):
    """Describe optional human-authored topic expectations."""

    model_config = ConfigDict(extra="forbid")
    topic_id: str
    description: str
    required: bool = True


class SurveyAuditContext(BaseModel):
    """Supply runtime truth without manufacturing benchmark annotations."""

    model_config = ConfigDict(extra="forbid")
    query: str
    report_language: Literal["en", "zh-CN"] = "en"
    review_structure: Literal[
        "legacy", "academic_v1", "method_transparent_v1", "academic_v2"
    ] = "legacy"
    review_methodology: dict[str, Any] = Field(default_factory=dict)
    expected_topics: list[AuditTopic] = Field(default_factory=list)


class MethodAssessment(BaseModel):
    """Keep runtime assertions outside scientific support metrics."""

    model_config = ConfigDict(extra="forbid")
    claim_id: str
    source_faithful: bool
    classification_valid: bool
    status: Literal["verified", "unsupported", "unverifiable"]
    record_paths: list[str]
    explanation: str


def record_paths(value: dict[str, Any], prefix: str = "") -> set[str]:
    """List existing runtime fields that an assessment may reference."""
    paths = set()
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        paths.add(path)
        if isinstance(item, dict):
            paths.update(record_paths(item, path))
    return paths


def check_structure(
    survey: str,
    context: SurveyAuditContext,
    *,
    selected_evidence_ids: Sequence[str] | None = None,
) -> dict:
    """Check exact template sections without assigning content quality scores."""
    if context.review_structure == "legacy":
        return {"status": "not_applicable", "issues": []}
    expected = [
        title
        for _, title in structure_sections(
            context.review_structure, context.report_language
        )
    ]
    tokens = MarkdownIt("commonmark").parse(survey)
    # Fenced code and quoted headings cannot satisfy report structure requirements.
    headings = [
        (token.tag, tokens[index + 1].content, token.map)
        for index, token in enumerate(tokens)
        if token.type == "heading_open" and token.level == 0
    ]
    matches = [item for item in headings if item[0] == "h2"]
    titles = [item[1] for item in matches]
    lines = survey.splitlines()
    issues = []
    if context.review_structure == "academic_v2":
        allowed = set(expected) | {appendix_title(context.report_language)}
        issues += [
            f"unexpected_section:{title}" for title in titles if title not in allowed
        ]

    positions = []
    for title in expected:
        if title not in titles:
            issues.append(f"missing_section:{title}")
            continue
        index = titles.index(title)
        positions.append(index)
        if titles.count(title) > 1:
            issues.append(f"duplicate_section:{title}")
        end = matches[index + 1][2][0] if index + 1 < len(matches) else len(lines)
        if not "\n".join(lines[matches[index][2][1] : end]).strip():
            issues.append(f"empty_section:{title}")
    if positions != sorted(positions):
        issues.append("section_order_invalid")
    if sum(item[0] == "h1" for item in headings) != 1:
        issues.append("single_title_required")
    placement = (
        find_summary_citation_dumps(survey, selected_evidence_ids)
        if selected_evidence_ids is not None
        else []
    )
    issues.extend(f"{p['reason_code']}:{p['section']}" for p in placement)
    return {
        "citation_placement_issues": placement,
        "status": "passed" if not issues else "failed",
        "expected_sections": expected,
        "issues": issues,
    }
