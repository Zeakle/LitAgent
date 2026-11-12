"""Prepare source-isolated two-human calibration without inventing gold labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import yaml


def build_package(manifest, previous, dataset=None):
    """Create forty candidates per domain from actual saved original sources."""
    pools = {"vit": [], "nerf": [], "fewshot": []}
    if dataset is None:
        path = Path(__file__).resolve().parents[1] / "benchmarks/survey/dataset.yaml"
        dataset = yaml.safe_load(path.read_text(encoding="utf-8"))
    domains = {"vision_transformer": "vit", "nerf": "nerf"}
    identities = {name: set() for name in domains.values()}
    for case in dataset["cases"]:
        domain = domains.get(case.get("domain"))
        if domain is None:
            continue
        identities[domain].update(case.get("required_paper_ids", []))
        identities[domain].update(case.get("relevant_paper_ids", []))
        for topic in case.get("expected_topics", []):
            identities[domain].update(topic.get("source_paper_ids", []))
    for paper in manifest["papers"]:
        source = paper.get("abstract", "")
        sha = hashlib.sha256(source.encode()).hexdigest()
        if sha != paper.get("abstract_sha256"):
            raise ValueError("calibration_source_sha_mismatch")
        source_record = {
            "paper_id": paper["paper_id"],
            "source": source,
            "source_sha256": sha,
            "source_url": paper.get("abstract_url"),
            "source_version": paper.get("abstract_source_version"),
            "source_kind": "saved_arxiv_api_abstract",
        }
        for domain, ids in identities.items():
            if paper["paper_id"] in ids:
                pools[domain].append(source_record)
    seen = set()
    for row in previous["samples"]:
        if row.get("domain") != "fewshot":
            continue
        for item in row["original_supports"]:
            source = item.get("supporting_text")
            if not source or item["paper_id"] in seen:
                continue
            seen.add(item["paper_id"])
            pools["fewshot"].append(
                {
                    "paper_id": item["paper_id"],
                    "source": source,
                    "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "source_locator": item.get("source_locator"),
                    "source_kind": "saved_run_original_support",
                    "source_artifact": row["source_artifact"],
                }
            )
    samples = []
    for domain, papers in pools.items():
        if len(papers) < 10:
            raise ValueError("calibration_sources_insufficient:" + domain)
        papers = sorted(papers, key=lambda p: p["paper_id"])[:10]
        for split, group in (("dev", papers[:5]), ("holdout", papers[5:])):
            for index, paper in enumerate(group):
                spans = [
                    (m.start(), m.end(), m.group().strip())
                    for m in re.finditer(r"[^.!?]+[.!?]|[^.!?]+$", paper["source"])
                    if len(m.group().strip()) >= 40
                ]
                if len(spans) < 2:
                    raise ValueError("calibration_source_fragments_insufficient")
                for variant in range(4):
                    start, end, quote = spans[variant % 2]
                    start += len(paper["source"][start:end]) - len(
                        paper["source"][start:end].lstrip()
                    )
                    end = start + len(quote)
                    support = paper if variant < 3 else group[(index + 1) % len(group)]
                    claim = (
                        quote
                        if variant != 2
                        else quote
                        + " These findings hold on every dataset and experimental condition without exception."
                    )
                    sid = hashlib.sha256(
                        f"{domain}:{paper['paper_id']}:{variant}".encode()
                    ).hexdigest()
                    samples.append(
                        {
                            "sample_id": sid,
                            "domain": domain,
                            "split": split,
                            "sample_type": (
                                "verbatim_candidate"
                                if variant < 2
                                else (
                                    "constructed_universal_extension"
                                    if variant == 2
                                    else "constructed_wrong_source"
                                )
                            ),
                            "claim_text": claim,
                            "original_quote": quote,
                            "original_start": start,
                            "original_end": end,
                            "original_source": paper,
                            "cited_source": support,
                            "annotation_status": "two_human_annotations_pending",
                            "annotations": [],
                            "adjudication": None,
                            "gold": None,
                        }
                    )
    return {
        "schema_version": "strict-calibration-candidates-v1",
        "samples": samples,
        "summary": {
            "domain_split_counts": {
                d: {
                    s: sum(r["domain"] == d and r["split"] == s for r in samples)
                    for s in ("dev", "holdout")
                }
                for d in pools
            },
            "gold_count": 0,
            "status": "two_human_annotations_pending",
            "warning": "Candidate transformations are explicit; neither source copying nor construction supplies a human gold label.",
        },
    }


def calibration_gate(package, predictions):
    """Score only independent, attested human labels and explicit adjudication."""
    groups = []
    source_splits = {}
    for row in package["samples"]:
        for key in ("original_source", "cited_source"):
            source_splits.setdefault(row[key]["paper_id"], set()).add(row["split"])
    if any(len(splits) != 1 for splits in source_splits.values()):
        raise ValueError("calibration_source_split_leak")

    def reviewed(row):
        votes = row.get("annotations") or []
        if (
            len(votes) != 2
            or len({v.get("reviewer") for v in votes if v.get("reviewer")}) != 2
            or not all(
                v.get("human_attested") is True
                and v.get("support_status")
                in {"supported", "unsupported", "unverifiable"}
                and type(v.get("atomic")) is bool
                and type(v.get("source_faithful")) is bool
                for v in votes
            )
        ):
            return False
        gold = row.get("gold")
        if gold not in {"supported", "unsupported", "unverifiable"}:
            return False
        if votes[0]["support_status"] == votes[1]["support_status"]:
            return gold == votes[0]["support_status"]
        resolution = row.get("adjudication") or {}
        return (
            resolution.get("human_attested") is True
            and bool(resolution.get("reviewer"))
            and resolution.get("support_status") == gold
        )

    for domain in ("vit", "nerf", "fewshot"):
        rows = [
            r
            for r in package["samples"]
            if r["domain"] == domain and r["split"] == "holdout"
        ]
        labeled = []
        for row in rows:
            if not reviewed(row):
                continue
            if predictions.get(row["sample_id"]) not in {
                "supported",
                "unsupported",
                "unverifiable",
            }:
                continue
            labeled.append(row)
        positive = [r for r in labeled if r["gold"] == "supported"]
        negative = [r for r in labeled if r["gold"] != "supported"]
        false_reject = sum(predictions[r["sample_id"]] != "supported" for r in positive)
        false_accept = sum(predictions[r["sample_id"]] == "supported" for r in negative)
        n = len(negative)
        # A zero observed error still has a nonzero upper confidence bound.
        upper = 1 - 0.05 ** (1 / n) if n and not false_accept else None
        development = [
            r
            for r in package["samples"]
            if r["domain"] == domain and r["split"] == "dev"
        ]
        development_reviewed = sum(reviewed(r) for r in development)
        ready = (
            len(development) >= 20
            and development_reviewed == len(development)
            and any(r.get("gold") == "supported" for r in development)
            and any(r.get("gold") == "unsupported" for r in development)
            and len(rows) >= 20
            and len(labeled) == len(rows)
            and bool(positive)
            and bool(negative)
        )
        groups.append(
            {
                "domain": domain,
                "development_human_reviewed_count": development_reviewed,
                "holdout_count": len(rows),
                "human_scored_count": len(labeled),
                "supported_count": len(positive),
                "unsupported_count": n,
                "false_accept_count": false_accept,
                "false_reject_count": false_reject,
                "false_reject_rate": false_reject / len(positive) if positive else None,
                "zero_false_accept_one_sided_95_upper": upper,
                "status": (
                    "passed"
                    if ready
                    and false_accept == 0
                    and false_reject / len(positive) <= 0.1
                    else "failed" if ready else "annotation_pending"
                ),
            }
        )
    return {
        "status": (
            "passed"
            if all(g["status"] == "passed" for g in groups)
            else (
                "failed"
                if any(g["status"] == "failed" for g in groups)
                else "annotation_pending"
            )
        ),
        "groups": groups,
    }


def main():
    """Write a local annotation package; this command performs no model calls."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("benchmarks/survey/vit_nerf_manifest.yaml"),
    )
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = build_package(
        yaml.safe_load(args.manifest.read_text()), json.loads(args.previous.read_text())
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(package["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
