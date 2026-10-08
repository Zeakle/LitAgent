"""Prepare provenance-backed annotation candidates without inventing human labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from litagent.eval.scoped import build_source_fragments
from litagent.evidence import extract_evidence_refs_ordered
from litagent.report_render import build_report_source_units


def _digest(text: str) -> str:
    """Return a stable UTF-8 identity."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _domain(query: str) -> str | None:
    """Classify only the three declared calibration domains."""
    for domain, pattern in (
        ("fewshot", r"few.?shot|小样本"),
        ("nerf", r"\bnerf\b|radiance|辐射场"),
        ("vit", r"\bvit\b|vision transformer|视觉.*transformer"),
    ):
        if re.search(pattern, query, re.I):
            return domain
    return None


def _ledger(value: object) -> dict:
    """Collect exported evidence, excluding unrelated event payloads."""
    result = {}

    def visit(item: object) -> None:
        if isinstance(item, dict):
            for evidence in item.get("evidence_items", []) or []:
                if isinstance(evidence, dict) and evidence.get("evidence_id"):
                    result.setdefault(evidence["evidence_id"], evidence)
            for key, child in item.items():
                if key not in {"events", "evidence_items", "config"}:
                    visit(child)
        elif isinstance(item, list):
            for child in item:
                visit(child)

    visit(value)
    return result


def build_dataset(artifacts: list[tuple[str, dict]], per_domain: int = 40) -> dict:
    """Build exact-source candidates and source-isolated annotation splits."""
    if per_domain < 1:
        raise ValueError("per_domain_must_be_positive")
    pools = {domain: [] for domain in ("vit", "nerf", "fewshot")}
    seen = set()
    for path, artifact in sorted(artifacts, key=lambda item: item[0]):
        domain = _domain(str(artifact.get("query") or ""))
        report = artifact.get("report") or {}
        survey = report.get("survey") or ""
        if not domain or not isinstance(survey, str) or not survey.strip():
            continue
        ledger = _ledger(artifact)
        report_sha = _digest(survey)
        assertions = (
            report.get("evaluation", {})
            .get("faithfulness", {})
            .get("details", {})
            .get("assertions", [])
        )
        fragments = []
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            start, end = assertion.get("source_start"), assertion.get("source_end")
            quote = assertion.get("source_quote")
            if (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(survey)
                and isinstance(quote, str)
                and survey[start:end] == quote
            ):
                fragments.append({"source_quote": quote, "start": start, "end": end})
        if not fragments:
            for unit in build_report_source_units(survey):
                for fragment in build_source_fragments(unit):
                    fragments.append(
                        {
                            "source_quote": fragment["source_quote"],
                            "start": unit.start + fragment["start"],
                            "end": unit.start + fragment["end"],
                        }
                    )
        for fragment in fragments:
            quote = fragment["source_quote"]
            start = fragment["start"]
            end = fragment["end"]
            if survey[start:end] != quote:
                raise ValueError("source_roundtrip_failed")
            identity = (report_sha, start, end)
            if identity in seen:
                continue
            seen.add(identity)
            ids = extract_evidence_refs_ordered(quote)
            supports = [
                {
                    key: ledger[eid].get(key)
                    for key in (
                        "evidence_id",
                        "paper_id",
                        "text",
                        "supporting_text",
                        "source_locator",
                        "content_scope",
                        "content_hash",
                    )
                }
                for eid in ids
                if eid in ledger
            ]
            pools[domain].append(
                {
                    "sample_id": _digest(f"{report_sha}:{start}:{end}:observed"),
                    "domain": domain,
                    "source_artifact": path,
                    "source_report_sha256": report_sha,
                    "source_start": start,
                    "source_end": end,
                    "source_quote": quote,
                    "claim_text": quote,
                    "evidence_ids": ids,
                    "missing_evidence_ids": [eid for eid in ids if eid not in ledger],
                    "original_supports": supports,
                    "sample_type": "observed",
                    "annotation_status": "annotation_pending",
                    "reviewer": None,
                    "labels": dict.fromkeys(
                        ("kind", "atomic", "source_faithful", "support_status")
                    ),
                }
            )

    samples = []
    for domain, pool in pools.items():
        # Spread candidates across reports instead of taking only the first report.
        groups = {}
        for row in pool:
            groups.setdefault(row["source_report_sha256"], []).append(row)
        ordered = []
        while any(groups.values()):
            for rows in groups.values():
                if rows:
                    ordered.append(rows.pop(0))
        observed_target = per_domain - min(10, per_domain // 4)
        observed = ordered[:observed_target]
        samples.extend(observed)
        supported_sources = [row for row in observed if row["original_supports"]]
        for parent in supported_sources[: per_domain - len(observed)]:
            row = dict(parent)
            row.update(
                sample_id=_digest(parent["sample_id"] + ":entity_quantity"),
                sample_type="constructed_entity_quantity",
                parent_sample_id=parent["sample_id"],
                claim_text="UnreportedMethod999 achieved 99999% accuracy. "
                + parent["claim_text"],
                labels=dict(parent["labels"]),
            )
            samples.append(row)

    parents = {row["sample_id"]: row["sample_id"] for row in samples}

    def root(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    owners = {}
    for row in samples:
        keys = ["report:" + row["source_report_sha256"]]
        keys.extend(
            "paper:" + str(item["paper_id"])
            for item in row["original_supports"]
            if item.get("paper_id")
        )
        for key in keys:
            if key in owners:
                parents[root(row["sample_id"])] = root(owners[key])
            owners[key] = row["sample_id"]
    components = {}
    for row in samples:
        components.setdefault(root(row["sample_id"]), []).append(row)
    sizes = {"dev": 0, "holdout": 0}
    for component in sorted(
        components.values(), key=lambda rows: (-len(rows), rows[0]["sample_id"])
    ):
        split = min(sizes, key=sizes.get)
        sizes[split] += len(component)
        group_id = _digest("|".join(sorted(row["sample_id"] for row in component)))
        for row in component:
            row.update(split=split, source_group_id=group_id)
    counts = Counter(row["domain"] for row in samples)
    return {
        "schema_version": "quality-calibration-candidates-v1",
        "samples": samples,
        "summary": {
            "per_domain_counts": {domain: counts[domain] for domain in pools},
            "shortfall": {domain: per_domain - counts[domain] for domain in pools},
            "annotation_status": "annotation_pending",
            "split_counts": sizes,
            "domains_without_holdout": [
                domain
                for domain in pools
                if not any(
                    row["domain"] == domain and row["split"] == "holdout"
                    for row in samples
                )
            ],
            "note": "Candidates are not human ground truth; constructed claims are explicitly synthetic.",
        },
    }


def calibration_metrics(dataset: dict, predictions: dict) -> dict:
    """Score only complete, explicitly human-reviewed annotation records."""
    accepted, rejected, false_accept, false_reject = 0, 0, 0, 0
    for row in dataset.get("samples", []):
        labels = row.get("labels") or {}
        prediction = predictions.get(row.get("sample_id"))
        if not (
            row.get("annotation_status") == "human_reviewed"
            and row.get("reviewer")
            and labels.get("kind")
            in {"scientific", "runtime_method", "report_structure"}
            and isinstance(labels.get("atomic"), bool)
            and isinstance(labels.get("source_faithful"), bool)
            and labels.get("support_status")
            in {"supported", "unsupported", "unverifiable"}
            and prediction in {"supported", "unsupported", "unverifiable"}
        ):
            continue
        if labels["support_status"] == "supported":
            accepted += 1
            false_reject += prediction != "supported"
        else:
            rejected += 1
            false_accept += prediction == "supported"
    return {
        "human_reviewed_count": accepted + rejected,
        "false_accept": {
            "count": false_accept,
            "denominator": rejected,
            "rate": false_accept / rejected if rejected else None,
        },
        "false_reject": {
            "count": false_reject,
            "denominator": accepted,
            "rate": false_reject / accepted if accepted else None,
        },
    }


def main() -> None:
    """Write an annotation package without accessing any model or provider."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts/runs"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-domain", type=int, default=40)
    args = parser.parse_args()
    artifacts = [
        (str(path), json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(args.artifacts.glob("*.json"))
    ]
    dataset = build_dataset(artifacts, args.per_domain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(dataset["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
