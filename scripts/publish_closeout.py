"""Publish an allowlisted, offline closeout replay without private request data."""

import argparse
import csv
import hashlib
import html
import io
import json
import re
from pathlib import Path

from litagent.quality import digest as evidence_digest
from litagent.report_audit import digest as manifest_digest

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
    "compiled_report_sha256",
    "draft_sha256",
    "material_status",
    "failure_family",
    "failure_code",
)


def public_sources(packet):
    """Keep source identities and citation ownership without publishing originals."""
    sources = []
    for ref, source in packet.get("sources", {}).items():
        paper = source["paper_id"]
        arxiv = paper.removeprefix("arxiv:")
        sources.append(
            {
                "source_ref": ref,
                "paper_id": paper,
                "content_sha256": source["content_sha256"],
                "evidence_ids": sorted(
                    {
                        card["evidence_id"]
                        for card in packet.get("cards", {}).values()
                        if card["source_ref"] == ref
                    }
                ),
                "url": (
                    f"https://arxiv.org/abs/{arxiv}"
                    if re.fullmatch(r"\d{4}\.\d{4,5}(?:v\d+)?", arxiv)
                    else None
                ),
            }
        )
    return sources


def compiled_draft(manifest, row, journal):
    """Reproduce an existing compiled draft; never repair or promote its status."""
    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.quality import build_fact_cards
    from litagent.review_structure import prepare_writer_methodology
    from litagent.writing_v2 import compile_document, source_packet

    expected = row.get("compiled_report_sha256")
    if (
        not expected
        or row.get("generated") is not True
        or journal.get("status") != "generated"
        or journal.get("draft_sha256") != expected
    ):
        raise ValueError("closeout_publication_draft_identity_invalid")
    case = next(c for c in manifest["cases"] if c["case_id"] == row["case_id"])
    bundle = case["bundle"]
    selection = EvidenceSelection.from_dict(bundle["selection"])
    cards = build_fact_cards(selection.selected_items)
    if evidence_digest(cards) != bundle["fact_cards_sha256"]:
        raise ValueError("closeout_publication_draft_sources_invalid")
    language = manifest.get("options", {}).get("report_language", "zh-CN")
    packet, aliases = source_packet(
        cards, selection, bundle["query"], bundle["requirements"], language
    )
    methodology = prepare_writer_methodology(
        bundle["methodology"],
        selection=selection,
        cards=cards,
        extractions=bundle.get("extractions", []),
        requested_methods=bundle.get("requested_methods", []),
        upstream_results={"extract": bundle.get("extractions", [])},
    )
    text, bindings = compile_document(
        journal["body"],
        journal["summaries"],
        cards,
        aliases,
        language,
        methodology,
        writing_plan=manifest.get("writing_plan"),
        envelope=manifest["budget_plans"][row["case_id"]]["envelope"],
    )
    if bindings["report_sha256"] != expected:
        raise ValueError("closeout_publication_draft_sha_invalid")
    failed = next(
        (c for c in journal.get("calls", {}).values() if c.get("failure")), {}
    )
    return {
        "draft": text,
        "draft_sha256": expected,
        "sources": public_sources(packet),
        "failure_details": {
            "stage": failed.get("stage"),
            "code": (failed.get("failure") or {}).get("code"),
            "model": failed.get("model"),
            "max_output_tokens": failed.get("max_tokens"),
            "completion_tokens": failed.get("usage", {}).get("completion_tokens"),
            "reasoning_tokens": failed.get("usage", {}).get("reasoning_tokens"),
        },
    }


def publication(benchmark, manifest, archives, journals=None):
    """Validate fixed identities, then select public result and report fields only."""
    if (
        manifest.get("closeout_profile") != "engineering-closeout-v1"
        or benchmark.get("suite_id") != manifest.get("suite_id")
        or benchmark.get("identity") != manifest.get("identity")
        or manifest.get("identity")
        != manifest_digest({k: v for k, v in manifest.items() if k != "identity"})
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
                row["sources"] = public_sources(packet)
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
        if not row["report_sha256"] and row["compiled_report_sha256"]:
            journal = (journals or {}).get(item["case_id"])
            if journal is None:
                raise ValueError("closeout_publication_draft_missing")
            row.update(compiled_draft(manifest, row, journal))
        row["material_status"] = (
            "archived_report"
            if "report" in row
            else "unreviewed_draft" if "draft" in row else "no_report"
        )
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
        "reviewer_completed": sum(r["reviewer_completed"] is True for r in rows),
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


def case_html(row):
    """Render readable, safely escaped results without inventing missing scores."""
    from litagent.report_render import render_report_html
    from litagent.review_structure import appendix_span

    escape = lambda value: html.escape(str(value))
    case = row["case_id"]
    page = f'<section id="{case}"><h2>{case}</h2><dl>'
    for key, title in (
        ("generated", "正文生成 / Generated"),
        ("reviewer_completed", "Reviewer完成 / Reviewer"),
        ("evaluation_valid", "有效完整核验 / Valid Evaluation"),
        ("publishable", "严格交付 / Delivered"),
    ):
        if row[key] is True:
            status = "完成 / Yes"
        elif key == "publishable":
            status = "阻止交付 / Blocked"
        elif row.get("status") == "not_run":
            status = "未运行 / Not run"
        elif key == "evaluation_valid":
            status = "未核验 / Incomplete"
        else:
            status = "未完成 / Incomplete"
        page += f"<dt>{title}</dt><dd>{status}</dd>"
    page += "</dl>"
    if row.get("status") == "not_run":
        page += "<p>前题技术停止，未运行；保留在计划分母。Not run after a technical stop.</p>"
    if row.get("failure_code"):
        page += (
            "<p>停止原因 / Stop reason: <code>"
            + escape(row["failure_code"])
            + "</code></p>"
        )
    if row.get("failure_details"):
        failure = row["failure_details"]
        page += (
            "<p>阶段 / Stage: "
            + escape(failure["stage"])
            + " · 输出 / Output: "
            + escape(failure["completion_tokens"])
            + " / "
            + escape(failure["max_output_tokens"])
            + " tokens · 思考 / Reasoning: "
            + escape(failure["reasoning_tokens"])
            + "</p>"
        )
    text = row.get("report") or row.get("draft")
    sources = row.get("sources", [])
    anchors = {
        eid: f"{case}-source-{i}"
        for i, source in enumerate(sources)
        for eid in source.get("evidence_ids", [])
    }

    def render(markdown):
        """Link only known citation identities; unknown IDs stay plain text."""
        linked = re.sub(
            r"\[E:([^\]\r\n]+)\]",
            lambda match: (
                f"[[E:{match[1]}]](#{anchors[match[1]]})"
                if match[1] in anchors
                else match[0]
            ),
            markdown,
        )
        return render_report_html(linked).html

    if text:
        if row["material_status"] == "unreviewed_draft":
            page += "<aside>未审阅、未核验草稿。Unreviewed and unevaluated draft; not deliverable.</aside>"
        else:
            page += "<aside>归档报告；交付状态以上方门禁为准。An archived report is not automatically accepted.</aside>"
        page += '<article class="report">'
        appendix = appendix_span(text)
        if appendix:
            page += render(text[: appendix["start"]])
            page += '<details class="appendix"><summary>附录：检索与证据来源 / Search and evidence appendix</summary>'
            page += render(text[appendix["start"] :]) + "</details>"
        else:
            page += render(text)
        page += "</article>"
    else:
        page += "<p>没有已编译正文可展示。No compiled report is available.</p>"
    if sources:
        page += "<h3>实际选中来源 / Selected sources</h3><ol>"
        for i, source in enumerate(sources):
            paper = escape(source["paper_id"])
            link = (
                f'<a href="{html.escape(source["url"], quote=True)}" rel="noopener noreferrer">{paper}</a>'
                if source["url"]
                else paper
            )
            page += (
                f'<li id="{case}-source-{i}" tabindex="-1">{link}'
                + " · "
                + escape(", ".join(source.get("evidence_ids", [])))
                + "<br>SHA: <code>"
                + escape(source["content_sha256"])
                + "</code></li>"
            )
        page += "</ol><p>公开包仅含来源定位与链接；完整选中原文保留本地。Original support texts remain local.</p>"
    metadata = {
        key: value for key, value in row.items() if key not in ("report", "draft")
    }
    return (
        page
        + "<details><summary>结果、问题与用量 / Result, diagnostics and usage</summary><pre>"
        + html.escape(json.dumps(metadata, ensure_ascii=False, indent=2))
        + "</pre></details></section>"
    )


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
    writer = csv.DictWriter(
        stream, fieldnames=ROW_FIELDS, extrasaction="ignore", lineterminator="\n"
    )
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
            + (
                "> 未审阅、未核验草稿；不可交付。Unreviewed draft, not deliverable.\n\n"
                if row["material_status"] == "unreviewed_draft"
                else ""
            )
            + (
                row.get("report")
                or row.get("draft")
                or "未生成报告 / No report generated."
            )
            + "\n"
        )
    page = (
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LitAgent · Engineering closeout</title><style>*{box-sizing:border-box}body{font:16px/1.7 system-ui;max-width:1100px;margin:24px auto;padding:0 16px;color:#18273c}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}summary{cursor:pointer;padding:12px}a{color:#1753ae;overflow-wrap:anywhere}*:focus-visible{outline:3px solid #1753ae}aside{background:#fff3ce;padding:16px}section{border-top:1px solid #ccd6e2;margin-top:28px;padding-top:12px}article{max-width:78ch}article h1{font-size:1.7em}dl{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:8px}dd{margin:0}code,li{overflow-wrap:anywhere}.appendix{background:#f1f5fa;padding:8px;margin:20px 0}li:target{background:#fff3ce}table{display:block;overflow:auto;max-width:100%}td,th{padding:8px;border:1px solid #ccd6e2}@media(max-width:500px){h1{font-size:1.5em}dl{display:block}dd{margin-bottom:10px}}</style><h1>LitAgent 工程收尾 / Engineering closeout</h1><p>'
        + html.escape(intro)
        + "</p><p>"
        + html.escape(value["interpretation"])
        + '</p><p><a href="closeout.json">JSON</a> · <a href="closeout.csv">CSV</a> · <a href="closeout.md">Markdown</a> · <a href="/flow-demo">Workbench</a></p>'
    )
    page += "".join(case_html(row) for row in value["cases"])
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
    archives, journals = {}, {}
    for row in benchmark["cases"]:
        run_id = row.get("run_id")
        if not run_id:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("closeout_publication_run_invalid")
        path = Path("artifacts/runs") / (run_id + ".json")
        if path.exists():
            archives[row["case_id"]] = json.loads(path.read_text())
        if row.get("compiled_report_sha256") and not row.get("report_sha256"):
            path = Path("artifacts/targeted-repairs") / (
                run_id + ".json.writer-v2.json"
            )
            if path.exists():
                journals[row["case_id"]] = json.loads(path.read_text())
    print(
        json.dumps(
            write_publication(
                publication(benchmark, manifest, archives, journals), args.destination
            )
        )
    )


if __name__ == "__main__":
    main()
