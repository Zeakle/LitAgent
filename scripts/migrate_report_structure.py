"""Migrate an explicitly reviewed source-unit manifest without any model calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from litagent.audit_context import record_paths
from litagent.evidence import extract_evidence_refs_ordered
from litagent.observability.recorder import ArchiveRepository
from litagent.report_render import build_report_source_units
from litagent.review_structure import (
    METHOD_BEGIN,
    METHOD_END,
    assemble_methodology,
    structure_sections,
    top_level_sections,
)
from scripts.repair_report_citations import _revision_source, create_local_revision


def migrate_report_structure(
    survey: str, record: Mapping, manifest: Mapping, language: str = "zh-CN"
) -> tuple[str, list[dict]]:
    """Apply reviewed exact units; unknown or citation-destructive actions fail closed."""
    if (
        manifest.get("source_body_sha256")
        != hashlib.sha256(survey.encode()).hexdigest()
    ):
        raise ValueError("migration_source_sha_conflict")
    sections = top_level_sections(survey)
    method_title = (
        "检索与证据选择方法"
        if language == "zh-CN"
        else "Search and Evidence Selection Methods"
    )
    methods = [s for s in sections if s["section"] == method_title]
    if len(methods) != 1:
        raise ValueError("unique_source_method_section_required")
    method = methods[0]
    units = build_report_source_units(survey)
    by_range = {(u.start, u.end): u for u in units}
    method_ranges = {
        (u.start, u.end)
        for u in units
        if method["content_start"] <= u.start < method["end"]
    }
    targets = dict(structure_sections("academic_v2", language))
    entries = manifest.get("entries", [])
    seen = set()
    edits = []
    relocated = {}
    for entry in entries:
        pair = (entry.get("source_start"), entry.get("source_end"))
        unit = by_range.get(pair)
        if (
            unit is None
            or pair in seen
            or entry.get("source_quote") != unit.text
            or not entry.get("reason")
        ):
            raise ValueError("migration_unit_contract_invalid")
        seen.add(pair)
        refs = extract_evidence_refs_ordered(unit.text)
        action = entry.get("action")
        paths = entry.get("record_paths", [])
        runtime_actions = {
            "runtime_duplicate",
            "runtime_replaced_by_record",
            "runtime_tail_to_appendix",
        }
        if action in runtime_actions and (
            not paths or not set(paths) <= record_paths(dict(record))
        ):
            raise ValueError("migration_runtime_record_binding_required")
        if action in {"runtime_duplicate", "runtime_replaced_by_record"}:
            if pair not in method_ranges or refs or "[E:" in unit.text:
                raise ValueError("migration_scientific_content_cannot_be_deleted")
            replacement = ""
        elif action == "scientific_to_body":
            if pair not in method_ranges or entry.get("target_section") not in {
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            }:
                raise ValueError("migration_scientific_target_invalid")
            relocated.setdefault(entry["target_section"], []).append(unit.text)
            replacement = ""
        elif action == "runtime_tail_to_appendix":
            replacement = entry.get("retained_text", "")
            if (
                pair in method_ranges
                or not replacement.strip()
                or not unit.text.startswith(replacement)
                or extract_evidence_refs_ordered(replacement) != refs
                or "[E:" in unit.text[len(replacement) :]
            ):
                raise ValueError("migration_runtime_tail_invalid")
        else:
            raise ValueError("migration_action_unknown")
        edits.append((unit.start, unit.end, replacement))
    if not method_ranges <= seen:
        raise ValueError("migration_unclassified_method_units")
    # Reconstruct the method section separately before removing its heading.
    content = survey[method["content_start"] : method["end"]]
    for start, end, replacement in sorted(edits, reverse=True):
        if (start, end) in method_ranges:
            lo, hi = start - method["content_start"], end - method["content_start"]
            content = content[:lo] + replacement + content[hi:]
    if content.replace(METHOD_BEGIN, "").replace(METHOD_END, "").strip():
        raise ValueError("migration_unclassified_method_content")
    edits = [(a, b, v) for a, b, v in edits if (a, b) not in method_ranges]
    edits.append((method["start"], method["end"], ""))
    body = survey
    for start, end, replacement in sorted(edits, reverse=True):
        body = body[:start] + replacement + body[end:]
    old_titles = dict(structure_sections("method_transparent_v1", language))
    # Real top-level headings only; identical text in source prose is not renamed.
    for span in reversed(top_level_sections(body)):
        key = next(
            (key for key, title in old_titles.items() if title == span["section"]), None
        )
        if key not in targets:
            raise ValueError("migration_unexpected_source_section")
        heading = f"## {targets[key]}\n"
        body = body[: span["start"]] + heading + body[span["content_start"] :]
    for span in reversed(top_level_sections(body)):
        key = next(key for key, title in targets.items() if title == span["section"])
        if key in relocated:
            body = (
                body[: span["end"]].rstrip()
                + "\n\n"
                + "\n".join(relocated[key])
                + "\n\n"
                + body[span["end"] :]
            )
    return assemble_methodology(body, "academic_v2", language, record), [
        dict(e) for e in entries
    ]


def main() -> None:
    """Require a new archive ID, immutable source SHA and explicit migration manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    repository = ArchiveRepository()
    source = _revision_source(
        repository, args.source_id, args.source_sha256, args.run_id
    )
    metadata = source["report"]["metadata"]
    record = metadata.get("review_methodology") or {}
    manifest = json.loads(args.manifest.read_text())
    body, applied = migrate_report_structure(
        source["report"]["survey"],
        record,
        manifest,
        metadata.get("report_language", "en"),
    )
    provenance = {
        "entries": applied,
        "source_body_sha256": manifest["source_body_sha256"],
        "methodology_sha256": hashlib.sha256(
            json.dumps(record, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
        "new_body_sha256": hashlib.sha256(body.encode()).hexdigest(),
    }
    result = create_local_revision(
        repository,
        source_id=args.source_id,
        source_sha256=args.source_sha256,
        run_id=args.run_id,
        survey=body,
        structure="academic_v2",
        migration_manifest=provenance,
    )
    print(f"Created unverified structure draft: {result['run_id']}; paid calls: 0")


if __name__ == "__main__":
    main()
