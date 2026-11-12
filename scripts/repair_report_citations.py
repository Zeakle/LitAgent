"""Create a local citation-layout draft without reusing scores or making model calls."""

from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import asdict
from pathlib import Path

from markdown_it import MarkdownIt

from litagent.config import load_config
from litagent.eval.base import EvalResult
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.review_structure import summary_section_spans
from litagent.runner import LitAgent, derive_delivery
from litagent.workbench import RunOptions


def replace_abstract(survey: str, replacement: str, title: str) -> str:
    """Replace one explicitly selected summary; keep every other source byte."""
    matches = [s for s in summary_section_spans(survey) if s["section"] == title]
    if len(matches) != 1:
        raise ValueError("unique_abstract_required")
    if not replacement.strip() or any(
        t.type == "heading_open" for t in MarkdownIt("commonmark").parse(replacement)
    ):
        raise ValueError("replacement_must_be_nonempty_summary_text")
    span = matches[0]
    return (
        survey[: span["content_start"]]
        + "\n"
        + replacement.strip()
        + "\n\n"
        + survey[span["end"] :]
    )


def _revision_source(
    repository: ArchiveRepository, source_id: str, source_sha256: str, run_id: str
) -> dict:
    """Validate safe paths and immutable binding before reading source content."""
    if (
        not re.fullmatch(r"[A-Za-z0-9._-]+", run_id)
        or not re.fullmatch(r"[A-Za-z0-9._-]+", source_id)
        or run_id == source_id
    ):
        raise ValueError("new_safe_run_id_required")
    if repository.path_for(run_id).exists():
        raise ValueError("repair_artifact_already_exists")
    source_path = repository.path_for(source_id)
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source_sha256:
        raise ValueError("source_artifact_sha_conflict")
    source = repository.get(source_id)
    if source is None:
        raise ValueError("source_artifact_missing")
    return source


def create_repair_draft(
    repository: ArchiveRepository,
    *,
    source_id: str,
    source_sha256: str,
    run_id: str,
    abstract: str,
    abstract_title: str = "摘要",
) -> dict:
    """Create a summary-only draft using the shared local revision recorder."""
    source = _revision_source(repository, source_id, source_sha256, run_id)
    survey = replace_abstract(source["report"]["survey"], abstract, abstract_title)
    return create_local_revision(
        repository,
        source_id=source_id,
        source_sha256=source_sha256,
        run_id=run_id,
        survey=survey,
    )


def create_local_revision(
    repository: ArchiveRepository,
    *,
    source_id: str,
    source_sha256: str,
    run_id: str,
    survey: str,
    structure: str | None = None,
    migration_manifest: dict | None = None,
) -> dict:
    """Bind a local draft to immutable input and fresh unverified state."""
    source = _revision_source(repository, source_id, source_sha256, run_id)
    source_path = repository.path_for(source_id)
    original = source["report"]
    if structure is not None:
        RunOptions(review_structure=structure)
    if not survey.strip():
        raise ValueError("local_revision_empty_body")
    original_metadata = original.get("metadata") or {}
    research_query = original_metadata.get("query") or source["query"]
    config = load_config()
    # Use the shared report assembler, with no review rounds or accepted verdict.
    runner = LitAgent(config)
    report = runner._extract_report(
        {
            "adversarial_review": {
                "final_draft": survey,
                "rounds": [],
                "accepted": False,
            },
            "graph_analysis": original.get("graph_data", {}),
        },
        research_query,
    )
    provenance = {
        "origin": (
            "manual_structure_revision"
            if structure
            else "manual_citation_layout_revision"
        ),
        "source_run_id": source_id,
        "source_artifact_sha256": source_sha256,
        "source_body_sha256": hashlib.sha256(original["survey"].encode()).hexdigest(),
        "repair_scope": (
            "explicit_structure_migration"
            if structure
            else "abstract_only_other_sections_unchanged"
        ),
    }
    report["metadata"].update(
        {
            key: original_metadata[key]
            for key in (
                "report_language",
                "review_structure",
                "review_methodology",
                "evidence_selection",
            )
            if key in original_metadata
        }
    )
    if structure:
        report["metadata"]["review_structure"] = structure
        report["metadata"]["structure_migration"] = migration_manifest or {}
    report["metadata"].update(
        **provenance,
        final_score=None,
        total_rounds=0,
        accepted=False,
        human_review={"status": "not_requested"},
        citation_style="ieee",
    )
    report["evaluation"] = {
        name: asdict(
            EvalResult.skip(
                name,
                (
                    "manual_structure_revision_not_evaluated"
                    if structure
                    else "manual_layout_revision_not_evaluated"
                ),
            )
        )
        for name in ("citation_accuracy", "faithfulness", "internal_consistency")
    }
    report["quality"] = runner._derive_quality(report["evaluation"])
    report["partial"] = False
    report["delivery"] = derive_delivery(False, report["quality"])
    recorder = RunRecorder(
        run_id,
        ("正文结构修订草稿 · " if structure else "引用布局修订草稿 · ")
        + research_query,
        repository,
    )
    recorder.set_citation_style("ieee")
    for node in source.get("nodes", {}).values():
        if node.get("kind") == "worker" and node.get("task_id") in {
            "dedup",
            "relevance_gate",
            "extract",
            "graph_analysis",
        }:
            recorder(
                "worker.complete",
                {
                    "task_id": node["task_id"],
                    "output": node["output"],
                    "origin": "source_snapshot_reuse",
                    **{k: v for k, v in provenance.items() if k != "origin"},
                },
            )
    recorder("report.manual_revision", {**provenance, "paid_requests": 0})
    # A manual edit is not a fresh paid Survey or independent audit.
    if repository.path_for(run_id).exists():
        raise ValueError("repair_artifact_already_exists")
    artifact = recorder.finalize(report)
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_sha256
    return artifact


def main() -> None:
    """Require exact source identity and explicit human-authored replacement text."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--abstract-file", required=True, type=Path)
    parser.add_argument("--abstract-title", default="摘要")
    args = parser.parse_args()
    artifact = create_repair_draft(
        ArchiveRepository(),
        source_id=args.source_id,
        source_sha256=args.source_sha256,
        run_id=args.run_id,
        abstract=args.abstract_file.read_text(encoding="utf-8"),
        abstract_title=args.abstract_title,
    )
    print(f"Created unverified local draft: {artifact['run_id']}; paid calls: 0")


if __name__ == "__main__":
    main()
