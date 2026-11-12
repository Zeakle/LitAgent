"""Publish allowlisted measurements and replay metadata without private payloads."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

from litagent.benchmark.quality_suite import (
    QualitySuiteResult,
    export_bundle,
    public_projection,
)


def build(suite_path: Path, run_path: Path, output: Path) -> dict:
    """Keep original identities while publishing no source text, credentials or paths."""
    original = json.loads(suite_path.read_text())
    run = json.loads(run_path.read_text())
    public = public_projection(original)
    payload = {k: v for k, v in public.items() if k in QualitySuiteResult.model_fields}
    for key in ("binding", "dataset_fingerprint"):
        payload[key] = original[key]
    for row, source in zip(
        payload["observations"], original["observations"], strict=True
    ):
        row["run_id"] = source["run_id"]
        row["acceptance"] = {
            "publishable": source.get("acceptance", {}).get("publishable") is True
        }
    QualitySuiteResult.model_validate(payload)
    source_sha = hashlib.sha256(run_path.read_bytes()).hexdigest()
    diagnostics = []
    for event in run.get("events", []):
        if event["event"] == "writing.wire.failed":
            data = event["data"]
            diagnostics.append(
                {
                    "event": event["event"],
                    "data": {
                        k: data[k]
                        for k in (
                            "phase",
                            "candidate_sha256",
                            "raw_response_sha256",
                            "contract_version",
                            "error_type",
                            "json_error",
                            "json_position",
                            "json_line",
                            "json_column",
                            "task_id",
                            "reason_code",
                            "location",
                        )
                        if k in data
                    },
                }
            )
    report = run.get("report") or {}
    replay = {
        "version": 1,
        "run_id": run["run_id"],
        "query": run["query"],
        "status": run["status"],
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "elapsed_ms": run.get("elapsed_ms"),
        "provenance": "sanitized_real_diagnostic_replay",
        "source_archive_sha256": source_sha,
        "formal_benchmark_evidence": False,
        "events": diagnostics,
        "nodes": {},
        "graph": {},
        "trace": {},
        "usage": {
            k: run.get("usage", {}).get(k) for k in ("total_tokens", "llm_calls")
        },
        "report": {
            "partial": bool(report.get("partial", False)),
            "survey": report.get("survey", ""),
            "evaluation": report.get("evaluation", {}),
            "quality": report.get("quality", {}),
            "delivery": report.get("delivery", {}),
            "metadata": {
                "report_language": "zh-CN",
                "quality_policy": "strict_auto_v2",
                "verification": {
                    "version": "optional-verification-v1",
                    "judge": False,
                    "jev": False,
                },
                "public_diagnostic_only": True,
            },
        },
    }
    # Never publish error strings or arbitrary metadata from external responses.
    output.mkdir(parents=True, exist_ok=True)
    dump = lambda obj: json.dumps(obj, ensure_ascii=False, indent=2) + "\n"
    (output / "suite.json").write_text(dump(payload))
    (output / "runs.json").write_text(dump([replay]))
    export_bundle(output / "presentation" / payload["run_id"], payload)
    csv_path = output / "presentation" / payload["run_id"] / "benchmark.csv"
    # Public text assets use LF in Git; the CSV records and fields are unchanged.
    csv_path.write_text(csv_path.read_text(encoding="utf-8"), encoding="utf-8")
    rag_path = Path("benchmarks/rag/phase14_3_results.json")
    rag = json.loads(rag_path.read_text())
    (output / "retrieval.json").write_text(dump(rag))
    rows = "".join(
        "<tr>"
        + "".join(
            f"<td>{html.escape(str(value))}</td>"
            for value in (
                r["profile_id"],
                r["recall_at_5"],
                r["mrr_at_10"],
                r["ndcg_at_10"],
                r["latency_p50_ms"],
                r["latency_p95_ms"],
                r["index_points_count"],
            )
        )
        + "</tr>"
        for r in rag["profiles"]
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LitAgent · 工程演示与真实证据</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.7;color:#18273c}}a{{color:#1b57bd}}table{{border-collapse:collapse;min-width:720px}}td,th{{padding:10px;border-bottom:1px solid #ccd6e2;text-align:left}}.scroll{{overflow:auto}}aside{{background:#fff3ce;padding:16px}}code{{overflow-wrap:anywhere}}</style><h1>LitAgent · 工程演示与真实证据</h1><p>Evidence-grounded multi-agent literature review · 自动评测，人工校准未验证。</p><p><a href="/flow-demo?run=sample%3Asample-ready&amp;view=report">离线契约样例 / Offline fixture</a> · <a href="/flow-demo?run=archive%3A{run['run_id']}&amp;view=quality">真实失败定位 / Real diagnostic</a></p><aside>Live Survey 仍属实验能力。内置 ready 样例为合成夹具，不代表模型质量；正式 Survey baseline 未建立。失败不计为通过，未运行不计为零分。</aside><h2>已完成：检索对比</h2><p>25 篇论文、15 条查询、每条重复 3 次；source_reviewed，未独立人工双审。源码 <code>{rag['benchmark_git_sha']}</code>，数据集 {rag['dataset_id']} / {rag['dataset_version']}。查询延迟不含模型下载，不能外推为通用检索性能。</p><div class="scroll"><table><thead><tr><th>Profile</th><th>Recall@5</th><th>MRR@10</th><th>nDCG@10</th><th>p50 ms</th><th>p95 ms</th><th>Points</th></tr></thead><tbody>{rows}</tbody></table></div><p><a href="/static/presentation/retrieval.json">原始公开检索汇总 JSON</a></p><h2>未完成：严格 Survey 验证</h2><p>最新诊断运行：{run['run_id']}。三题试点已运行 1 题；27 请求 / 124,448 tokens，usage 完整。Pro 原始 JSON 在字符 8299 解析失败，Evaluation 未运行，正式 0/15。Judge / Jev 未启用；不代表科学支持率为零。</p><p>完整原始响应与论文原文留在本地，本展示仅包含脱敏元数据。源归档 SHA：<code>{source_sha}</code>。</p><p><a href="/workbench/benchmarks/{payload['run_id']}/export/html">Survey JSON / CSV / Markdown / HTML 展示包</a> · <a href="/flow-demo?view=benchmark">工作台 Benchmark</a></p><h2>三分钟展示顺序</h2><ol><li>离线样例：正文 → 引用 → 来源与质量。</li><li>检索实验：解释 Recall 与 reranker 延迟取舍。</li><li>真实失败：响应定位 → 未运行评估 → 阻止交付 → 预算与恢复边界。</li></ol><p>科学准确率、误放行率与误拒绝率：不可用。当前能力边界和复现说明见仓库 README。</p></html>"""
    (output / "index.html").write_text(page, encoding="utf-8")
    manifest = {
        "publication_version": 1,
        "suite_id": payload["run_id"],
        "source_archive_sha256": source_sha,
        "source_suite_sha256": hashlib.sha256(suite_path.read_bytes()).hexdigest(),
        "retrieval_sha256": hashlib.sha256(rag_path.read_bytes()).hexdigest(),
        "source_binding": payload["binding"],
        "interpretation": "sanitized_diagnostic_not_scientific_acceptance",
    }
    (output / "manifest.json").write_text(dump(manifest))
    return manifest


def publish_protocol(summary_path: Path, output: Path) -> dict:
    """Publish exact known usage and protocol outcome without response content."""
    receipt = json.loads(summary_path.read_text())
    root = summary_path.parent
    state = json.loads((root / "probe-state.json").read_text())
    manifest = json.loads((root / "probe-manifest.json").read_text())
    received = state.get("received", {})
    public = {
        "protocol_id": manifest["run_id"],
        "request_binding": manifest["binding"],
        "status": receipt["status"],
        "scientific_verdict": "not_evaluated",
        "new_survey_rounds": 0,
        "requests": receipt["new_requests"],
        "known_tokens": receipt["new_known_tokens"],
        "unknown_hold": receipt["new_unknown_hold"],
        "model": received.get("actual_model"),
        "finish_reason": received.get("finish_reason"),
        "prompt_tokens": received.get("prompt_tokens"),
        "completion_tokens": received.get("completion_tokens"),
        "max_total_tokens": manifest["max_tokens"],
        "max_output_tokens": manifest["frame"]["max_tokens"],
        "response_sha256": state.get("raw_response_sha256"),
        "json_error": state.get("json_error"),
        "error_type": state.get("error_type"),
        "family_requests": receipt["family_requests"],
        "family_known_tokens": receipt["family_known_tokens"],
        "cumulative": receipt["cumulative"],
        "phase14_5": "in_progress",
        "followup_surveys_run": 0,
        "stop_reason": (
            "protocol_failed_no_followup"
            if receipt["status"] != "completed"
            else "protocol_only_not_scientific_acceptance"
        ),
        "source_receipt_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
    }
    (output / "protocol.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2) + "\n"
    )
    page = output / "index.html"
    status = html.escape(public["status"])
    block = f'<h2>收尾协议验证 / Release protocol</h2><p>{status} · {public["requests"]} request · {public["known_tokens"]:,} known tokens。完整冻结输入与内联 schema 验证未通过，未启动三域 Survey，未产生科学分数，不重试。</p><p><a href="/static/presentation/protocol.json">协议结果、原响应 SHA 与累计计账 JSON</a></p>'
    page.write_text(
        page.read_text().replace(
            "<h2>三分钟展示顺序</h2>", block + "<h2>三分钟展示顺序</h2>"
        )
    )
    return public


def main():
    """Require explicit local source files before publishing the fixed package."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = build(args.suite, args.run, args.output)
    if args.protocol:
        result["protocol"] = publish_protocol(args.protocol, args.output)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
