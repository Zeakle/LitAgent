"""Publish an allowlisted, offline closeout replay without private request data."""

import argparse
import csv
import hashlib
import html
import io
import json
import re
from pathlib import Path

from litagent.quality import digest

CASES = ("few-shot-metric", "vit-data-efficient", "nerf-foundations")
ROW_FIELDS = (
    "case_id",
    "run_id",
    "status",
    "generated",
    "reviewer_completed",
    "evaluation",
    "evaluation_valid",
    "publishable",
    "supported",
    "unsupported",
    "unverified",
    "inventory_recoveries",
    "prose_repairs",
    "requests",
    "known_tokens",
    "unknown_hold",
    "usage_complete",
    "elapsed_ms",
    "report_sha256",
    "failure_family",
    "failure_code",
)


def publication(benchmark, manifest, archives):
    """Validate fixed identities, then select public result and report fields only."""
    if (
        manifest.get("closeout_profile") != "engineering-closeout-v1"
        or benchmark.get("suite_id") != manifest.get("suite_id")
        or benchmark.get("identity") != manifest.get("identity")
        or manifest.get("identity")
        != digest({k: v for k, v in manifest.items() if k != "identity"})
        or tuple(row.get("case_id") for row in benchmark.get("cases", [])) != CASES
        or benchmark.get("planned") != 3
        or benchmark.get("state")
        not in ("completed", "stopped", "interrupted", "preflight_rejected")
    ):
        raise ValueError("closeout_publication_identity_invalid")
    rows = []
    for item in benchmark["cases"]:
        row = {key: item.get(key) for key in ROW_FIELDS}
        archive = archives.get(item["case_id"])
        if archive is not None:
            if archive.get("run_id") != row["run_id"]:
                raise ValueError("closeout_publication_run_invalid")
            report = archive["report"]
            text = report["survey"]
            sha = hashlib.sha256(text.encode()).hexdigest()
            if row["report_sha256"] and sha != row["report_sha256"]:
                raise ValueError("closeout_publication_report_sha_invalid")
            if row["report_sha256"]:
                row["report"] = text
                packet = (
                    report.get("metadata", {})
                    .get("writer_document", {})
                    .get("packet", {})
                )
                sources = []
                for ref, source in packet.get("sources", {}).items():
                    paper = source["paper_id"]
                    arxiv = paper.removeprefix("arxiv:")
                    sources.append(
                        {
                            "source_ref": ref,
                            "paper_id": paper,
                            "content_sha256": source["content_sha256"],
                            "url": (
                                f"https://arxiv.org/abs/{arxiv}"
                                if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv)
                                else None
                            ),
                        }
                    )
                row["sources"] = sources
                details = (
                    report.get("evaluation", {})
                    .get("faithfulness", {})
                    .get("details", {})
                )
                row["diagnostics"] = [
                    {
                        key: claim.get(key)
                        for key in (
                            "claim_id",
                            "claim_text",
                            "source_start",
                            "source_end",
                            "evidence_ids",
                            "status",
                            "reason_code",
                            "local_error",
                        )
                    }
                    for claim in details.get("assertions", [])
                ]
        elif row["report_sha256"]:
            raise ValueError("closeout_publication_report_missing")
        rows.append(row)
    value = {
        "publication_version": "engineering-closeout-public-v1",
        "suite_id": benchmark["suite_id"],
        "manifest_identity": manifest["identity"],
        "source_commit": benchmark.get("source_commit"),
        "source_fingerprint": benchmark.get("source_fingerprint"),
        "models": benchmark.get("writing_profile", {}).get("configured_models", {}),
        "state": benchmark.get("state"),
        "stop_reason": benchmark.get("stop_reason"),
        "planned": 3,
        "run": benchmark.get("run"),
        "generated": sum(r["generated"] is True for r in rows),
        "evaluation_completed": sum(r["evaluation_valid"] is True for r in rows),
        "delivered": sum(r["publishable"] is True for r in rows),
        "cases": rows,
        "new_usage": {
            key: benchmark.get("new_usage", {}).get(key)
            for key in ("requests", "known_tokens", "unknown_hold", "request_hold")
        },
        "historical_usage": {
            key: benchmark.get("historical_usage", {}).get(key)
            for key in (
                "model_requests",
                "model_known_tokens",
                "model_unknown_conservative_tokens",
                "jev_requests",
                "jev_conservative_input_tokens",
                "judge_requests",
                "jev_usage_complete",
            )
        },
        "human_accuracy": None,
        "interpretation": "Automatic evaluation; human calibration unavailable. Live Survey is experimental. Unrun and invalid evaluations have no scientific score.",
    }
    # Local paths, provider credentials and raw request containers are not public fields.
    raw = json.dumps(value, ensure_ascii=False)
    if any(
        token in raw for token in ("/Users/", "Authorization:", "Bearer ", "api_key")
    ) or re.search(r"\bsk-[A-Za-z0-9_-]{20,}\b", raw):
        raise ValueError("closeout_publication_sensitive_content")
    return value


def write_publication(value, destination):
    """Derive all four formats from one immutable public object."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    outputs = {
        kind: destination / ("closeout." + kind)
        for kind in ("json", "csv", "md", "html")
    }
    if any(p.exists() for p in outputs.values()):
        raise ValueError("closeout_publication_exists")
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=ROW_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(value["cases"])
    intro = f"计划3题；运行{value['run']}；生成{value['generated']}；有效完整核验{value['evaluation_completed']}；严格交付{value['delivered']}。"
    md = (
        "# LitAgent 工程收尾 / Engineering closeout\n\n"
        + intro
        + "\n\n"
        + value["interpretation"]
        + "\n\n"
    )
    md += "|Case|Generated|Reviewer|Valid Evaluation|Delivered|Inventory / prose repairs|\n|---|---|---|---|---|---|\n"
    md += "".join(
        f"|{r['case_id']}|{r['generated']}|{r['reviewer_completed']}|{r['evaluation_valid']}|{r['publishable']}|{r['inventory_recoveries']} / {r['prose_repairs']}|\n"
        for r in value["cases"]
    )
    for row in value["cases"]:
        md += (
            "\n## "
            + row["case_id"]
            + "\n\n"
            + row.get("report", "未生成报告 / No report generated.")
            + "\n"
        )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LitAgent · Engineering closeout</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:24px auto;padding:0 16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}summary{cursor:pointer;padding:12px}a{color:#1753ae}*:focus-visible{outline:3px solid #1753ae}</style><h1>LitAgent 工程收尾 / Engineering closeout</h1><p>'
        + html.escape(intro)
        + "</p><p>"
        + html.escape(value["interpretation"])
        + '</p><p><a href="closeout.json">JSON</a> · <a href="closeout.csv">CSV</a> · <a href="closeout.md">Markdown</a> · <a href="/flow-demo">Workbench</a></p>'
    )
    page += "".join(
        "<details><summary>"
        + html.escape(r["case_id"])
        + " · generated="
        + str(r["generated"])
        + " · evaluated="
        + str(r["evaluation_valid"])
        + " · delivered="
        + str(r["publishable"])
        + "</summary><pre>"
        + html.escape(json.dumps(r, ensure_ascii=False, indent=2))
        + "</pre></details>"
        for r in value["cases"]
    )
    page += (
        '<script type="application/json" id="publication">'
        + json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")
        + "</script></html>"
    )
    data = {
        "json": json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        "csv": stream.getvalue(),
        "md": md,
        "html": page,
    }
    for kind, path in outputs.items():
        path.write_text(data[kind], encoding="utf-8")
    return {
        kind: hashlib.sha256(path.read_bytes()).hexdigest()
        for kind, path in outputs.items()
    }


def main():
    """Read a trusted local frozen suite only after its controller has terminated."""
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument(
        "--destination", type=Path, default=Path("src/litagent/static/presentation")
    )
    args = parser.parse_args()
    manifest = json.loads((args.suite / "manifest.json").read_text())
    benchmark = json.loads(
        (
            Path("benchmarks/survey/results") / args.suite.name / "benchmark.json"
        ).read_text()
    )
    archives = {}
    for row in benchmark["cases"]:
        run_id = row.get("run_id")
        if not run_id:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("closeout_publication_run_invalid")
        path = Path("artifacts/runs") / (run_id + ".json")
        if path.exists():
            archives[row["case_id"]] = json.loads(path.read_text())
    print(
        json.dumps(
            write_publication(
                publication(benchmark, manifest, archives), args.destination
            )
        )
    )


if __name__ == "__main__":
    main()
