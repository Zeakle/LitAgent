"""Run exactly three frozen-evidence writing cases, with durable no-replay accounting."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import html
import io
import json
import os
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path

from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.frozen_writing import CASES, frozen_options, read_bundle, run_frozen
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.writing_v2 import CONTRACT, classify_failure
from scripts.run_stage_model_quality import check_models

VERSION = "frozen-writing-three-domain-v1"
LIMITS = {
    "cases": 3,
    "per_case_tokens": 500000,
    "per_case_requests": 64,
    "total_tokens": 1500000,
    "total_requests": 192,
    "repairs": 2,
    "retries": 0,
    "formal_cases": 0,
}


def runtime_identity():
    """Bind core transport, schema and workflow dependencies before execution."""
    return {
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in ("openai", "httpx", "pydantic", "langgraph", "PyMuPDF")
        },
    }


def source_paths():
    """Freeze the complete implementation and test inventory, including new files."""
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    paths = [
        Path(p)
        for p in tracked
        if p.startswith(("src/", "tests/", "scripts/", "config/"))
        or p in {"pyproject.toml", "requirements-demo.txt"}
    ]
    paths += (
        list(Path("src/litagent").rglob("*.py"))
        + list(Path("scripts").glob("*.py"))
        + list(Path("tests").glob("test_writer_v2*.py"))
    )
    return sorted({p for p in paths if p.is_file() and "__pycache__" not in str(p)})


def prepare(root, config):
    """Freeze all three materials and the same source before any paid request."""
    check_models(config)
    root = Path(root)
    if root.exists():
        raise ValueError("frozen_campaign_already_exists")
    rows = []
    for case, run in CASES:
        bundle = read_bundle(Path("artifacts/runs", run + ".json"))
        rows.append({"case_id": case, "bundle": bundle})
    paths = source_paths()
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
        if p.is_file() and "__pycache__" not in str(p)
    }
    history = json.loads(
        Path(
            "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
        ).read_text()
    )["cumulative"]
    manifest = {
        "version": VERSION,
        "suite_id": root.name,
        "contract": CONTRACT,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "options": frozen_options().model_dump(mode="json"),
        "config": build_config_summary(config, "strict_auto_v2"),
        "source_hashes": hashes,
        "cases": rows,
        "history": history,
        "limits": {
            "cases": 3,
            "per_case_tokens": 500000,
            "per_case_requests": 64,
            "total_tokens": 1500000,
            "total_requests": 192,
            "repairs": 2,
            "retries": 0,
            "formal_cases": 0,
        },
    }
    manifest["limits"] = dict(LIMITS)
    manifest["runtime"] = runtime_identity()
    manifest["identity"] = digest(manifest)
    root.mkdir(parents=True)
    atomic_json(root / "manifest.json", manifest)
    for name, h in hashes.items():
        p = root / "source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(Path(name).read_bytes())
    return {
        "identity": manifest["identity"],
        "cases": [
            {
                "id": r["case_id"],
                "source": r["bundle"]["source_run_id"],
                "cards": len(r["bundle"]["selection"]["selected_items"]),
            }
            for r in rows
        ],
    }


def validate(root, config):
    """Reject changed sources, materials, runtime config or campaign scope."""
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("version") != VERSION
        or m.get("limits") != LIMITS
        or m.get("runtime") != runtime_identity()
        or m.get("contract") != CONTRACT
        or set(m.get("source_hashes", {})) != {str(p) for p in source_paths()}
        or digest({k: v for k, v in m.items() if k != "identity"}) != m["identity"]
        or m["config"] != build_config_summary(config, "strict_auto_v2")
        or m["options"] != frozen_options().model_dump(mode="json")
        or [(r["case_id"], r["bundle"]["source_run_id"]) for r in m["cases"]]
        != list(CASES)
    ):
        raise ValueError("frozen_campaign_identity_conflict")
    check_models(config)
    for name, h in m["source_hashes"].items():
        if (
            hashlib.sha256(Path(name).read_bytes()).hexdigest() != h
            or hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() != h
        ):
            raise ValueError("frozen_campaign_source_changed")
    for row in m["cases"]:
        path = Path("artifacts/runs", row["bundle"]["source_run_id"] + ".json")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != row["bundle"]["archive_sha256"]
        ):
            raise ValueError("frozen_campaign_archive_changed")
    return m


def export(root, state, manifest):
    """Publish one sanitized data source with separate execution and quality outcomes."""
    rows = state["cases"]
    public = {
        "version": VERSION,
        "suite_id": root.name,
        "identity": manifest["identity"],
        "source_commit": manifest["source_commit"],
        "contract": CONTRACT,
        "state": state["status"],
        "stop_reason": state.get("stop_reason"),
        "planned": 3,
        "run": len(rows),
        "generated": sum(bool(r.get("generated")) for r in rows),
        "evaluation_completed": sum(r.get("evaluation") == "completed" for r in rows),
        "passed": sum(r.get("publishable", False) for r in rows),
        "not_run": 3 - len(rows),
        "judge": "disabled",
        "jev": "disabled",
        "human_calibration": "not_validated",
        "cases": rows,
        "historical_usage": manifest["history"],
        "new_usage": {
            "requests": sum(r.get("requests", 0) for r in rows),
            "known_tokens": sum(r.get("known_tokens", 0) for r in rows),
            "unknown_hold": sum(r.get("unknown_hold", 0) for r in rows),
            "request_hold": sum(r.get("request_hold", 0) for r in rows),
        },
        "limits": manifest["limits"],
    }
    out = Path("benchmarks/survey/results", root.name)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "benchmark.json", public)
    keys = [
        "case_id",
        "run_id",
        "status",
        "generated",
        "evaluation",
        "publishable",
        "supported",
        "unsupported",
        "unverified",
        "requests",
        "known_tokens",
        "unknown_hold",
        "elapsed_ms",
        "report_sha256",
        "failure_family",
        "failure_code",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    (out / "benchmark.csv").write_text(buf.getvalue())
    md = f"# 三域冻结正文验收\n\n计划3题，运行{public['run']}题，生成{public['generated']}题，Evaluation完成{public['evaluation_completed']}题，严格交付{public['passed']}/3。状态：{state['status']}。\n\n源码 `{manifest['source_commit']}`，协议 `{CONTRACT}`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。\n\n|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|\n|---|---|---|---|---|---|\n"
    md += "".join(
        f"|{r['case_id']}|{r.get('generated')}|{r.get('evaluation')}|{r.get('publishable')}|{r.get('supported')}/{r.get('unsupported')}/{r.get('unverified')}|{r.get('failure_code','')}|\n"
        for r in rows
    )
    md += f"\n新增已知用量{public['new_usage']['known_tokens']} tokens，未知保守占用{public['new_usage']['unknown_hold']}，{public['new_usage']['requests']}请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。\n\n[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)\n"
    (out / "benchmark.md").write_text(md)
    (out / "README.md").write_text(md)
    table = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(str(r.get(k, "—"))) + "</td>"
            for k in [
                "case_id",
                "generated",
                "evaluation",
                "publishable",
                "supported",
                "unsupported",
                "unverified",
            ]
        )
        + '<td><a href="http://127.0.0.1:8000/flow-demo?run=archive%3A'
        + r["run_id"]
        + '&view=quality">质量与原文 / Report</a></td></tr>'
        for r in rows
    )
    details = "".join(
        "<details><summary>"
        + html.escape(r["case_id"])
        + " · "
        + html.escape(str(r.get("failure_code") or r["status"]))
        + "</summary><pre>"
        + html.escape(json.dumps(r, ensure_ascii=False, indent=2))
        + "</pre></details>"
        for r in rows
    )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LitAgent · Frozen writing validation</title><style>*{box-sizing:border-box}body{font:16px/1.6 system-ui;background:#f4f6fa;color:#172c43;max-width:1100px;padding:20px;margin:auto}section,header{background:white;padding:20px;margin:16px 0;border:1px solid #dbe2ee;border-radius:12px}h1{font-size:1.7rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.stat{background:#edf2f9;padding:14px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85em}button,a,summary{cursor:pointer}button{font:inherit;padding:8px}a{color:#135a9e}*:focus-visible{outline:3px solid #1679d2}summary{padding:12px 0}p{overflow-wrap:anywhere}@media(max-width:430px){body{padding:10px}section,header{padding:14px}h1{font-size:1.3rem}}</style><header><button id="locale">English / 中文</button><h1 id="title">三域冻结正文验收</h1><p id="scope">固定证据 → 正文 → Reviewer → Evaluation → 自动交付或失败归档；不重新检索。Judge/Jev关闭。</p></header><section><div id="stats" class="grid"></div><p id="note">自动评测；人工校准未验证。未核验不等于通过，协议完成不等于科学通过。失败及未运行题均保留在分母。</p><p id="status"></p></section><section class="scroll"><table><thead><tr><th>Case</th><th>Generated</th><th>Evaluation</th><th>Delivered</th><th>Supported</th><th>Unsupported</th><th>Unverified</th><th>Detail</th></tr></thead><tbody>__ROWS__</tbody></table></section><section><h2>诊断 / Diagnostics</h2>__DETAILS__</section><section><a href="benchmark.json">JSON</a> · <a href="benchmark.csv">CSV</a> · <a href="benchmark.md">Markdown</a><pre id="usage"></pre></section><script id="data" type="application/json">__DATA__</script><script>const d=JSON.parse(document.getElementById('data').textContent);let en=false;function draw(){document.documentElement.lang=en?'en':'zh-CN';document.getElementById('title').textContent=en?'Three-domain frozen writing validation':'三域冻结正文验收';document.getElementById('scope').textContent=en?'Frozen evidence → writing → Reviewer → Evaluation → automatic delivery or archived failure. No retrieval. Judge/Jev disabled.':'固定证据 → 正文 → Reviewer → Evaluation → 自动交付或失败归档；不重新检索。Judge/Jev关闭。';document.getElementById('note').textContent=en?'Automatic evaluation; no human calibration. Unverified is not passed. Protocol completion is not scientific acceptance. Failures and unrun cases remain in the denominator.':'自动评测；人工校准未验证。未核验不等于通过，协议完成不等于科学通过。失败及未运行题均保留在分母。';document.getElementById('status').textContent=d.state+' · '+(d.stop_reason||'');let s=document.getElementById('stats');s.replaceChildren();for(const [k,v]of Object.entries({Planned:3,Run:d.run,Generated:d.generated,Evaluated:d.evaluation_completed,Delivered:d.passed})){let e=document.createElement('div');e.className='stat';e.textContent=k+' '+v+' / 3';s.append(e)}document.getElementById('usage').textContent=JSON.stringify({new_usage:d.new_usage,historical_usage:d.historical_usage,source_commit:d.source_commit,contract:d.contract},null,2)}document.getElementById('locale').onclick=()=>{en=!en;draw()};draw();</script></html>"""
    page = (
        page.replace("__ROWS__", table)
        .replace("__DETAILS__", details)
        .replace(
            "__DATA__", json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
        )
    )
    (out / "benchmark.html").write_text(page)
    return public


async def execute(root, config):
    """Reject concurrent submissions before touching the durable suite state."""
    with suite_lock(root / "execution.lock"):
        return await _execute(root, config)


async def _execute(root, config):
    """Never replay a started observation or expand the three-case manifest."""
    m = validate(root, config)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != m["identity"]:
            raise ValueError("frozen_campaign_state_conflict")
        if state["status"] != "prepared":
            if state["status"] == "running":
                state.update(
                    status="interrupted", stop_reason="process_interrupted_no_replay"
                )
                for entry in state["cases"]:
                    if entry["status"] == "started_no_replay":
                        entry.update(
                            status="interrupted_no_replay",
                            failure_family="interrupted",
                            failure_code="process_interrupted_no_replay",
                            unknown_hold=500000,
                            request_hold=64,
                            accounting_scope="unsettled_case_cap",
                            usage_complete=False,
                        )
                atomic_json(state_path, state)
            export(root, state, m)
            return state
    else:
        state = {"identity": m["identity"], "status": "prepared", "cases": []}
    state["status"] = "running"
    atomic_json(state_path, state)
    for row in m["cases"]:
        run_id = root.name + "-" + row["case_id"]
        entry = {
            "case_id": row["case_id"],
            "run_id": run_id,
            "status": "started_no_replay",
            "generated": False,
            "reviewer_completed": False,
            "evaluation": "not_run",
            "publishable": False,
            "supported": None,
            "unsupported": None,
            "unverified": None,
        }
        state["cases"].append(entry)
        atomic_json(state_path, state)
        recorder = RunRecorder(run_id, row["bundle"]["query"], ArchiveRepository())
        recorder.set_config_summary(build_config_summary(config, "strict_auto_v2"))
        recorder.set_citation_style("ieee")
        tracer = (
            LangFuseTracer(
                config.observability.langfuse_host,
                os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=run_id,
            )
            if config.observability.enabled
            else None
        )
        hook = CompositeTraceHook(
            recorder,
            (
                RedactingTraceHook(tracer, payload_mode="metadata_only")
                if tracer
                else None
            ),
        )
        agent = LitAgent(
            config, trace_hook=hook, run_policy=RunPolicy.isolated_benchmark()
        )
        start = time.monotonic()
        failure = None
        report = None
        try:
            report = await run_frozen(agent, row["bundle"], run_id)
        except BaseException as exc:
            failure = classify_failure(exc)
        finally:
            budget = agent._cost_budget.snapshot() if agent._cost_budget else {}
            await agent.cleanup()
        if tracer:
            recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            tracer.flush()
        artifact = recorder.finalize(
            report=report, error=(failure["code"] if failure else None)
        )
        events = artifact["events"]
        ends = [
            e["data"]
            for e in events
            if e["event"] == "llm.complete" and e["data"].get("usage_available")
        ]
        known = sum(e["total_tokens"] for e in ends)
        entry.update(
            requests=budget.get("requests_sent", 0),
            known_tokens=known,
            unknown_hold=max(0, budget.get("used_tokens", 0) - known),
            usage_complete=not budget.get("usage_uncertain", False),
            elapsed_ms=round((time.monotonic() - start) * 1000),
            trace_url=tracer.trace_url if tracer else None,
        )
        if report:
            meta = report.get("metadata", {})
            faith = report.get("evaluation", {}).get("faithfulness", {})
            details = faith.get("details", {})
            entry.update(
                generated=bool(meta.get("writer_document")),
                evaluation=(
                    "completed"
                    if not faith.get("skipped", True)
                    and details.get("stages", {}).get("global_check") == "completed"
                    else "not_complete"
                ),
                publishable=report.get("delivery", {}).get("publishable", False),
                report_sha256=(
                    hashlib.sha256(report["survey"].encode()).hexdigest()
                    if meta.get("writer_document")
                    else None
                ),
                supported=details.get("supported_count"),
                unsupported=details.get("unsupported_count"),
                unverified=details.get("unverified_count"),
                repairs=len(meta.get("targeted_repair_state", {}).get("attempts", [])),
                core_coverage=details.get("core_coverage", []),
                global_issues=details.get("global_issues", []),
                evaluation_history=[
                    {
                        "stage": label,
                        "supported": value.get("faithfulness", {})
                        .get("details", {})
                        .get("supported_count"),
                        "unsupported": value.get("faithfulness", {})
                        .get("details", {})
                        .get("unsupported_count"),
                        "unverified": value.get("faithfulness", {})
                        .get("details", {})
                        .get("unverified_count"),
                    }
                    for label, value in [
                        (
                            "initial",
                            meta.get("evaluation_history", {}).get(
                                "initial_evaluation", {}
                            ),
                        ),
                        *[
                            (f"repair_{i+1}", a.get("candidate_evaluation") or {})
                            for i, a in enumerate(
                                meta.get("evidence_rewrite_attempts", [])
                            )
                        ],
                    ]
                ],
            )
            failure = meta.get("writer_failure") or failure
            journal = Path(
                "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
            )
            if not failure and not entry["publishable"] and journal.exists():
                j = json.loads(journal.read_text())
                failure = j.get("failure") or (
                    None
                    if entry["generated"]
                    else next(
                        (
                            v.get("failure")
                            for v in reversed(list(j["calls"].values()))
                            if v.get("failure")
                        ),
                        None,
                    )
                )
            if not failure and entry["evaluation"] != "completed":
                failure = {
                    "family": "protocol",
                    "code": details.get("error")
                    or details.get("reason")
                    or "evaluation_not_completed",
                }
        journal_path = Path(
            "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
        )
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            entry["generated"] = (
                entry["generated"] or journal.get("status") == "generated"
            )
            reviewed = journal.get("calls", {}).get("reviewer", {})
            entry["reviewer_completed"] = reviewed.get(
                "status"
            ) == "response_received" and not reviewed.get("failure")
            entry["compiled_report_sha256"] = journal.get("draft_sha256")
        if entry["unknown_hold"] or not entry["usage_complete"]:
            failure = {"family": "interrupted", "code": "usage_unknown"}
        failure = failure or (
            None
            if entry["publishable"]
            else {"family": "quality", "code": "strict_quality_failed"}
        )
        entry.update(
            status="passed" if entry["publishable"] else "blocked",
            failure_family=failure["family"] if failure else None,
            failure_code=failure["code"] if failure else None,
        )
        atomic_json(state_path, state)
        export(root, state, m)
        if failure and failure["family"] not in {"quality", "evidence"}:
            state.update(status="stopped", stop_reason=failure["code"])
            break
    else:
        state["status"] = "completed"
    atomic_json(state_path, state)
    export(root, state, m)
    return state


def main():
    """Require explicit preparation or bounded live execution of the frozen suite."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--live", action="store_true")
    args = p.parse_args()
    cfg = load_config()
    result = (
        prepare(args.root, cfg)
        if args.prepare
        else asyncio.run(execute(args.root, cfg))
    )
    print(
        json.dumps(
            {
                "suite": args.root.name,
                **{
                    k: v
                    for k, v in result.items()
                    if k in {"identity", "status", "stop_reason", "cases"}
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
