"""Follow actual SDK budget snapshots through durable and browser consumers."""

import asyncio
import copy
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from model_config_helpers import load_previous_default_config as load_config
from test_native_responses import envelope, owner_for
from test_planned_budget import plan
from test_planned_mixed_receipts import mixed_parent
from test_writer_v2 import case  # noqa: F401

from litagent.demo import project_flow_demo_view
from litagent.interaction_store import SqliteReviewRepository
from litagent.interactive_service import InteractiveRunService
from litagent.llm.routing import ModelBoundClient
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.workbench import RunOptions
from litagent.writing_v2 import WriterSession, compile_document, schema_for

STATIC = Path(__file__).resolve().parents[1] / "src/litagent/static/flow-demo"
FIELDS = (
    "plan_identity",
    "known_tokens",
    "requests",
    "usage_uncertain",
    "stop_reason",
    "unknown_hold",
)


async def sdk_archive(case, tmp_path, mode):
    """Generate and settle real planned calls with only external HTTP replaced."""
    sent, events = [], []

    async def handler(request):
        sent.append(request)
        if len(sent) == 2:
            if mode == "cancelled":
                raise asyncio.CancelledError()
            if mode == "rejected":
                return httpx.Response(403, json={"error": {"code": "OfflineDenied"}})
            if mode == "missing_usage":
                value = envelope(json.dumps(case.body))
                value.pop("usage")
                return httpx.Response(200, json=value)
        return httpx.Response(200, json=envelope(json.dumps(case.body)))

    owner, budget, _ = await owner_for(handler)
    run_id = "budget-pipeline-" + mode
    repository = ArchiveRepository(tmp_path / "archives")
    recorder = RunRecorder(run_id, case.packet["query"], repository)

    def trace(event, data):
        events.append((event, copy.deepcopy(data)))
        recorder(event, data)

    owner._trace_hook = trace
    ledger_path = tmp_path / "ledger.json"
    budget.activate_plan(plan(), path=ledger_path)
    session = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "writer.json",
        run_id,
        case.packet,
        case.aliases,
        thinking_mode="disabled",
        provider_api="responses",
    )
    try:
        body = await session.call(
            "body:introduction",
            "Preserve the supplied sources.",
            case.packet,
            schema_for("body", case.aliases),
        )
        if mode == "settled":
            await session.call(
                "body:taxonomy",
                "Preserve all sources.",
                case.packet,
                schema_for("body", case.aliases),
            )
        else:
            expected = asyncio.CancelledError if mode == "cancelled" else ValueError
            with pytest.raises(expected):
                await session.call(
                    "body:taxonomy",
                    "Preserve all sources.",
                    case.packet,
                    schema_for("body", case.aliases),
                )
    finally:
        await owner.close()
    survey, bindings = compile_document(
        body, case.summaries, case.cards, case.aliases, "en"
    )
    # This fixture intentionally does not run scientific Evaluation. Actual SDK
    # completion and a valid cost snapshot must leave the draft unverified.
    report = {
        "survey": survey,
        "evaluation": {
            "faithfulness": {
                "score": None,
                "skipped": True,
                "passed": False,
                "details": {
                    "source_report_sha256": hashlib.sha256(survey.encode()).hexdigest(),
                    "coverage": {"complete": False},
                    "skipped_reason": "not_run",
                },
            }
        },
        "quality": {"status": "unverified"},
        "delivery": {
            "status": "blocked",
            "publishable": False,
            "reason_codes": ["evaluation_not_run"],
        },
        "metadata": {"report_language": "en", "writing_bindings": bindings},
    }
    recorder.finalize(
        report, error=None if mode == "settled" else "offline_unknown_usage"
    )
    archive = repository.get(run_id)
    updates = [
        data["budget_state"] for event, data in events if event == "budget.updated"
    ]
    assert len(updates) == len(sent) == 2
    assert archive["budget_state"] == updates[-1]
    assert set(updates[-1]) == set(FIELDS)
    snapshot = budget.snapshot()
    assert updates[-1] == {
        "plan_identity": snapshot["plan"]["identity"],
        "known_tokens": snapshot["used_tokens"],
        "requests": snapshot["requests_sent"],
        "usage_uncertain": snapshot["usage_uncertain"],
        "stop_reason": snapshot["stop_reason"],
        "unknown_hold": sum(
            e["reserved_tokens"]
            for e in snapshot["entries"].values()
            if e["status"] == "interrupted_no_replay"
        ),
    }
    return SimpleNamespace(
        run_id=run_id,
        repository=repository,
        archive=archive,
        report=report,
        snapshot=snapshot,
        events=events,
        sent=sent,
        ledger_path=ledger_path,
        journal_path=session.journal_path,
    )


async def interactive_view(archive, archive_repository, tmp_path):
    """Read a durable archive through the real interactive service bridge."""
    repository = SqliteReviewRepository(tmp_path / "interactive")
    await repository.initialize()
    run_id = archive["run_id"]
    await repository.create_run(
        run_id, archive["query"], RunOptions(report_language="en")
    )
    await repository.mark_terminal(run_id, archive["status"], result=archive["report"])
    await repository.save_report_version(run_id, 1, archive["report"])
    service = InteractiveRunService(
        load_config(), repository, archive_repository=archive_repository
    )
    return await service.get_view(run_id)


def browser_projection(raw, source, run_id, locale):
    """Execute both actual adapter and quality DOM, retaining input identities."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const adapters=await load('adapters.js'),{renderQuality}=await load('quality-view.js');
 const raw=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(raw),source=process.argv[2];
 const identity={runId:process.argv[3],source,key:source+':'+process.argv[3],status:raw.summary.status};
 const view=(source==='archive'?adapters.adaptArchiveView:adapters.adaptInteractiveView)(raw,identity);
 const frozen=JSON.stringify(view),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,view.runtimeEvaluation,view.quality,view.delivery,()=>{},x=>x,{locale:process.argv[4],runId:view.identity.runId,budgetState:view.budgetState,budgetPlan:view.budgetPlan,sha:view.reportSha});
 if(JSON.stringify(raw)!==before||JSON.stringify(view)!==frozen)throw Error('Consumer mutated its input');
 const nodes=walk(root),budget=root.children.find(e=>e.children.some(c=>c.textContent===(process.argv[4]==='en'?'Run budget usage (last recorded)':'运行预算用量（最后记录）')));
 console.log(JSON.stringify({budgetState:view.budgetState,text:nodes.map(e=>e.textContent).join(' '),budgetTexts:budget?walk(budget).map(e=>e.textContent):[],reportSha:view.reportSha,survey:view.report.survey,evaluation:view.runtimeEvaluation,quality:view.quality,delivery:view.delivery,metricStatuses:nodes.filter(e=>e.className==='metric-row').map(e=>e.dataset.status)}));
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(STATIC),
            source,
            run_id,
            locale,
        ],
        input=json.dumps(raw),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def assert_budget_pipeline(raw, source, archive, locale):
    """Check exact real amounts while preserving the report's science boundary."""
    before = copy.deepcopy(raw)
    actual = archive["budget_state"]
    state = raw["budget_state"]
    assert (
        state["contract"] == "run-budget-state-v1"
        and state["availability"] == "available"
    )
    assert state["scope"] == "run" and state["run_id"] == archive["run_id"]
    assert {key: state[key] for key in FIELDS} == actual
    rendered = browser_projection(raw, source, archive["run_id"], locale)
    assert rendered["budgetState"] == state
    en = locale == "en"
    for label, key in [
        ("Known usage" if en else "已知用量", "known_tokens"),
        ("Unknown reservation" if en else "未知占用", "unknown_hold"),
        ("Requests sent" if en else "已发请求", "requests"),
    ]:
        suffix = "" if key == "requests" else " tokens"
        assert f"{label}: {actual[key]}{suffix}" in rendered["budgetTexts"]
    if actual["usage_uncertain"]:
        assert ("Usage incomplete" if en else "用量不完整") in rendered["text"]
    else:
        assert (
            "No unknown reservation recorded in this snapshot"
            if en
            else "本次快照未记录未知占用"
        ) in rendered["text"]
    if actual["stop_reason"]:
        assert actual["stop_reason"] in rendered["text"]
    report = archive["report"]
    assert rendered["survey"] == report["survey"]
    assert (
        rendered["reportSha"] == hashlib.sha256(report["survey"].encode()).hexdigest()
    )
    assert rendered["evaluation"] == report["evaluation"]
    assert rendered["quality"] == report["quality"]
    assert rendered["delivery"] == report["delivery"]
    assert rendered["metricStatuses"][-1] in {"unverified", "failed"}
    assert raw == before
    return rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["settled", "rejected", "cancelled", "missing_usage"])
async def test_sdk_budget_event_archive_and_both_browser_consumers(
    case, tmp_path, mode
):
    fixture = await sdk_archive(case, tmp_path, mode)
    archive = fixture.archive
    before = copy.deepcopy(archive)
    protected = {
        p: p.read_bytes()
        for p in [
            fixture.ledger_path,
            fixture.journal_path,
            fixture.repository.path_for(fixture.run_id),
        ]
    }
    state = archive["budget_state"]
    assert state["known_tokens"] == (240 if mode == "settled" else 120)
    assert fixture.archive["usage"]["total_tokens"] == state["known_tokens"]
    assert state["requests"] == 2
    assert state["usage_uncertain"] is (mode != "settled")
    assert (state["unknown_hold"] == 0) is (mode == "settled")
    assert state["unknown_hold"] == sum(
        e["reserved_tokens"]
        for e in fixture.snapshot["entries"].values()
        if e["sent"] and e["actual_tokens"] is None
    )
    projected = project_flow_demo_view(archive, source="archive")
    interactive = await interactive_view(archive, fixture.repository, tmp_path)
    for locale in ["zh-CN", "en"]:
        assert_budget_pipeline(projected, "archive", archive, locale)
        assert_budget_pipeline(interactive, "interactive", archive, locale)
    assert archive == before
    assert all(p.read_bytes() == raw for p, raw in protected.items())
    assert len(fixture.sent) == 2


@pytest.mark.asyncio
async def test_real_future_global_allocation_is_not_a_sent_request(
    case, tmp_path, monkeypatch
):
    parent = await mixed_parent(case, tmp_path, monkeypatch, mode="rejected")
    archive = json.loads(parent.archive_path.read_text())
    ledger = parent.row["budget_snapshot"]
    allocations = [row["allocation"] for row in ledger["capacity_revisions"]]
    assert any(
        slot["stage"] == "global"
        for allocation in allocations
        for slot in allocation["round"]["requests"]
    )
    sent = [entry for entry in ledger["entries"].values() if entry["sent"]]
    assert not any(entry["stage"] == "global" for entry in sent)
    assert archive["budget_state"]["requests"] == len(sent) == len(parent.http_calls)
    assert archive["budget_state"]["unknown_hold"] == sum(
        entry["reserved_tokens"] for entry in sent if entry["actual_tokens"] is None
    )
    before = {
        p: p.read_bytes()
        for p in [parent.archive_path, parent.ledger_path, parent.journal_path]
    }
    projected = project_flow_demo_view(archive, source="archive")
    repository = ArchiveRepository("artifacts/runs")
    interactive = await interactive_view(archive, repository, tmp_path)
    for locale in ["zh-CN", "en"]:
        assert_budget_pipeline(projected, "archive", archive, locale)
        assert_budget_pipeline(interactive, "interactive", archive, locale)
    assert all(p.read_bytes() == raw for p, raw in before.items())
