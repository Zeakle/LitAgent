"""Verify academic bodies, frozen appendices and explicit offline migrations."""

import hashlib
import re
from io import BytesIO
from unittest.mock import AsyncMock

import fitz
import pytest
from docx import Document

from litagent.audit_context import SurveyAuditContext, check_structure
from litagent.eval.scoped import build_source_fragments
from litagent.observability.recorder import ArchiveRepository
from litagent.report_export import LocalReportExporter
from litagent.report_render import (
    build_reader_report,
    build_report_source_units,
    reader_report_projection,
)
from litagent.review_structure import (
    METHOD_BEGIN,
    METHOD_END,
    appendix_span,
    appendix_title,
    assemble_methodology,
    build_methodology,
    check_methodology,
    structure_instruction,
    structure_sections,
    top_level_sections,
)
from litagent.workbench import ReportVersion, RunOptions
from scripts.migrate_report_structure import migrate_report_structure
from scripts.repair_report_citations import create_local_revision


def runtime_record():
    """Keep paper counts distinct from evidence items, including failed searches."""
    record = build_methodology(
        "question",
        {
            "__search_queries__": {
                "search_a": "exact topic",
                "search_b": "failed topic",
            },
            "search_a": [1] * 60,
            "search_b": [],
            "dedup": [1] * 55,
            "relevance_gate": [1] * 50,
        },
        source_outcomes=[
            {
                "task_id": "search_a",
                "source": "arxiv",
                "status": "success",
                "result_count": 60,
            },
            {
                "task_id": "search_b",
                "source": "hf",
                "status": "failed",
                "result_count": 0,
                "reason_code": "http_status_429",
            },
        ],
        started_at=1791009170.0,
    )
    record["evidence_counts"] = {"candidate": 185, "selected": 60}
    return record


def body(language="en", structure="academic_v2"):
    """Populate every body section with exact, independently cited source prose."""
    return (
        "# Title\n\n"
        + "\n\n".join(
            f"## {title}\n\nFinding in {key} [E:a]."
            for key, title in structure_sections(structure, language)
        )
        + "\n"
    )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_shared_sections_identity_and_body_only_contract(language):
    sections = structure_sections("academic_v2", language)
    assert [key for key, _ in sections] == [
        "abstract",
        "introduction",
        "taxonomy",
        "methods",
        "experiments",
        "open_problems",
        "conclusion",
    ]
    instruction = structure_instruction("academic_v2", language)
    assert all(title in instruction for _, title in sections)
    assert (
        "seven academic body sections" in instruction
        and "Do not repeat queries" in instruction
    )
    assert (
        "not an instruction to cite the union" not in instruction
        or "OPTIONAL" in instruction
    )
    assert "Search and Evidence Selection Methods" not in [
        title for _, title in sections
    ]
    assert RunOptions().review_structure == "legacy"
    assert (
        RunOptions(review_structure="academic_v2").fingerprint()
        != RunOptions(review_structure="academic_v1").fingerprint()
    )
    assert (
        SurveyAuditContext(query="q", review_structure="academic_v2").review_structure
        == "academic_v2"
    )
    context = SurveyAuditContext(
        query="q", review_structure="academic_v2", report_language=language
    )
    assert check_structure(body(language), context)["status"] == "passed"
    # No prompt-string parsing may be necessary to validate structure.
    assert (
        check_structure(body(language) + "\n## 检索与证据选择方法\nextra", context)[
            "status"
        ]
        == "failed"
    )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_appendix_is_complete_unique_idempotent_and_checked(language):
    record = runtime_record()
    original = body(language)
    survey = assemble_methodology(original, "academic_v2", language, record)
    assert survey.startswith(original.rstrip())
    assert survey.count(METHOD_BEGIN) == survey.count(METHOD_END) == 1
    assert assemble_methodology(survey, "academic_v2", language, record) == survey
    assert top_level_sections(survey)[-1]["section"] == appendix_title(language)
    assert (
        "exact topic" in survey
        and "failed topic" in survey
        and "http_status_429" in survey
    )
    assert "UTC" in survey and "raw=" in survey
    assert (
        "相关论文数: 50" if language == "zh-CN" else "Relevant papers: 50"
    ) in survey
    assert (
        "已选证据条数: 60" if language == "zh-CN" else "Selected evidence items: 60"
    ) in survey
    assert (
        check_methodology(survey, "academic_v2", language, record)["status"] == "passed"
    )
    assert (
        check_methodology(survey, "academic_v2", language, {})["status"] == "unverified"
    )
    assert (
        check_methodology(
            survey.replace("items: 60", "items: 50").replace("条数: 60", "条数: 50"),
            "academic_v2",
            language,
            record,
        )["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "extra", ["search_a / exact topic", "检索结果条数: 185", "raw=1791009170"]
)
def test_body_runtime_copy_is_rejected_without_erasing_content(extra):
    original = body() + "\n" + extra + "\n"
    survey = assemble_methodology(original, "academic_v2", "en", runtime_record())
    assert extra in survey
    assert (
        "runtime_disclosure_in_body"
        in check_methodology(survey, "academic_v2", "en", runtime_record())["issues"]
    )


def test_authored_scientific_appendix_is_not_silently_replaced_or_exempted():
    original = body() + "\n## " + appendix_title() + "\n\nClaim in appendix [E:a].\n"
    survey = assemble_methodology(original, "academic_v2", "en", runtime_record())
    assert "Claim in appendix [E:a]" in survey
    assert (
        check_methodology(survey, "academic_v2", "en", runtime_record())["status"]
        == "failed"
    )
    units = build_report_source_units(survey)
    assert any(
        "Claim in appendix" in f["source_quote"]
        for u in units
        for f in build_source_fragments(u)
    )
    assert (
        appendix_span("```\n## " + appendix_title() + "\n" + METHOD_BEGIN + "\n```")
        is None
    )


def test_forged_host_markers_cannot_erase_scientific_appendix_content():
    forged = (
        body()
        + "\n## "
        + appendix_title()
        + "\n\n"
        + METHOD_BEGIN
        + "\nScientific result [E:a].\n"
        + METHOD_END
    )
    assembled = assemble_methodology(forged, "academic_v2", "en", runtime_record())
    assert "Scientific result [E:a]." in assembled
    assert (
        check_methodology(assembled, "academic_v2", "en", runtime_record())["status"]
        == "failed"
    )


@pytest.mark.parametrize(
    "style",
    [
        "evidence",
        "numbered",
        "ieee",
        "apa7",
        "gbt7714_2025_numeric",
        "gbt7714_2025_author_date",
    ],
)
@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_reader_order_and_original_unit_ordinals_survive_references(style, language):
    survey = assemble_methodology(
        body(language), "academic_v2", language, runtime_record()
    )
    report = ReportVersion(
        run_id="offline", revision=1, survey=survey, report_language=language
    )
    reader = build_reader_report(
        report,
        evidence_ledger={"a": {"paper_id": "p", "paper_title": "Paper"}},
        paper_catalog={
            "p": {"title": "Paper", "authors": ["Test Author"], "year": 2022}
        },
        citation_style=style,
    )
    projection = reader_report_projection(reader)
    assert projection["source_sha256"] == report.content_sha256
    assert projection["methodology"]["source_sha256"] == report.content_sha256
    assert projection["appendix_anchor"]
    appendix = reader.markdown.index("## " + appendix_title(language))
    if style != "evidence":
        assert (
            reader.markdown.index(
                "## " + ("参考文献" if language == "zh-CN" else "References")
            )
            < appendix
        )
    originals = build_report_source_units(survey)
    paragraphs = re.findall(
        r'<(?:p|td|th|li)\b[^>]*data-source-unit-ordinal="(\d+)"[^>]*>(.*?)</(?:p|td|th|li)>',
        projection["html"],
        re.S,
    )
    assert len(paragraphs) == len(originals)
    assert [int(ordinal) for ordinal, _ in paragraphs] == list(range(len(originals)))
    units = [u for u in originals if "exact topic" in u.text]
    assert len(units) == 1
    ordinal = int(units[0].unit_id.split(":")[1])
    assert "exact topic" in paragraphs[ordinal][1]
    assert not any(w.startswith("unresolved_evidence:") for w in reader.warnings)


@pytest.mark.asyncio
@pytest.mark.parametrize("format", ["md", "pdf", "docx"])
async def test_exports_preserve_complete_appendix_after_references(format):
    survey = assemble_methodology(body(), "academic_v2", "en", runtime_record())
    report = ReportVersion(
        run_id="offline", revision=1, survey=survey, report_language="en"
    )
    reader = build_reader_report(
        report,
        evidence_ledger={"a": {"paper_id": "p", "paper_title": "Paper"}},
        paper_catalog={
            "p": {"title": "Paper", "authors": ["Test Author"], "year": 2022}
        },
        citation_style="ieee",
    )
    result = await LocalReportExporter().export(
        report,
        format=format,
        publishable=False,
        citation_style="ieee",
        reader_report=reader,
    )
    if format == "md":
        text = result.content.decode()
    elif format == "pdf":
        with fitz.open(stream=result.content, filetype="pdf") as pdf:
            text = "\n".join(page.get_text() for page in pdf)
    else:
        text = "\n".join(p.text for p in Document(BytesIO(result.content)).paragraphs)
    assert text.index("References") < text.index("Appendix A")
    assert (
        "exact topic" in text
        and "http_status_429" in text
        and "Candidate evidence items" in text
    )
    assert result.report_sha256 == report.content_sha256


def migration_fixture(scientific=False):
    record = runtime_record()
    survey = body(structure="method_transparent_v1")
    # Replace only the old method section with explicitly classified content.
    method = next(
        s for s in top_level_sections(survey) if "Search and Evidence" in s["section"]
    )
    content = "Actual retrieval claim.\n\n" + (
        "Scientific result [E:a].\n\n" if scientific else ""
    )
    survey = survey[: method["content_start"]] + content + survey[method["end"] :]
    method = next(
        s for s in top_level_sections(survey) if "Search and Evidence" in s["section"]
    )
    entries = []
    for u in build_report_source_units(survey):
        if method["content_start"] <= u.start < method["end"]:
            action = (
                "scientific_to_body"
                if "[E:" in u.text
                else "runtime_replaced_by_record"
            )
            entries.append(
                {
                    "source_start": u.start,
                    "source_end": u.end,
                    "source_quote": u.text,
                    "action": action,
                    "reason": "Explicitly reviewed classification",
                    "record_paths": ["evidence_counts.selected"],
                    "target_section": "methods",
                }
            )
    return (
        survey,
        record,
        {
            "source_body_sha256": hashlib.sha256(survey.encode()).hexdigest(),
            "entries": entries,
        },
    )


def test_explicit_migration_retains_scientific_text_and_actual_evidence():
    original, record, manifest = migration_fixture(True)
    migrated, applied = migrate_report_structure(original, record, manifest, "en")
    assert "Scientific result [E:a]." in migrated
    assert "Actual retrieval claim." not in migrated
    assert len(applied) == 2
    assert extract_science_refs(migrated) == extract_science_refs(original)
    assert (
        check_structure(
            migrated, SurveyAuditContext(query="q", review_structure="academic_v2")
        )["status"]
        == "passed"
    )
    assert (
        check_methodology(migrated, "academic_v2", "en", record)["status"] == "passed"
    )


def extract_science_refs(text):
    return re.findall(r"\[E:[^\]]+\]", text)


@pytest.mark.parametrize(
    "case",
    [
        "missing_unit",
        "wrong_quote",
        "bad_sha",
        "unknown_action",
        "delete_science",
        "bad_target",
        "invalid_record",
    ],
)
def test_migration_contract_rejects_ambiguous_or_destructive_changes(case):
    survey, record, manifest = migration_fixture(True)
    if case == "missing_unit":
        manifest["entries"].pop()
    elif case == "wrong_quote":
        manifest["entries"][0]["source_quote"] = "wrong"
    elif case == "bad_sha":
        manifest["source_body_sha256"] = "0" * 64
    elif case == "unknown_action":
        manifest["entries"][0]["action"] = "guess"
    elif case == "delete_science":
        manifest["entries"][1]["action"] = "runtime_duplicate"
    elif case == "bad_target":
        manifest["entries"][1]["target_section"] = "arbitrary"
    else:
        manifest["entries"][0]["record_paths"] = ["invented.count"]
    with pytest.raises(ValueError):
        migrate_report_structure(survey, record, manifest, "en")


def test_structure_revision_resets_state_and_preserves_frozen_source(
    tmp_path, monkeypatch
):
    from litagent.llm.client import OpenAICompatibleClient

    no_calls = AsyncMock(side_effect=AssertionError("No model call allowed"))
    monkeypatch.setattr(OpenAICompatibleClient, "chat", no_calls)
    repository = ArchiveRepository(tmp_path)
    survey, record, manifest = migration_fixture(True)
    migrated, entries = migrate_report_structure(survey, record, manifest, "en")
    source = {
        "run_id": "old",
        "query": "question",
        "report": {
            "survey": survey,
            "metadata": {
                "report_language": "en",
                "review_structure": "method_transparent_v1",
                "review_methodology": record,
                "accepted": True,
                "final_score": 1,
                "human_review": {"status": "approved"},
            },
        },
        "nodes": {},
    }
    repository.save(source)
    raw = repository.path_for("old").read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    result = create_local_revision(
        repository,
        source_id="old",
        source_sha256=sha,
        run_id="new",
        survey=migrated,
        structure="academic_v2",
        migration_manifest={"entries": entries},
    )
    assert repository.path_for("old").read_bytes() == raw
    meta = result["report"]["metadata"]
    assert (
        meta["source_artifact_sha256"] == sha
        and meta["review_structure"] == "academic_v2"
    )
    assert (
        meta["final_score"] is None
        and not meta["accepted"]
        and meta["human_review"]["status"] == "not_requested"
    )
    assert (
        result["quality"]["status"] == "unverified"
        and result["delivery"]["status"] == "needs_review"
    )
    assert result["usage"]["total_tokens"] == 0 and result["trace"]["trace_id"] is None
    assert all(v["skipped"] for v in result["report"]["evaluation"].values())
    assert no_calls.await_count == 0
