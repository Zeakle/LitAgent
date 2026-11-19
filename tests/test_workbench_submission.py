"""Exercise live submission while unrelated workbench reads overlap."""

import asyncio
import shutil
import subprocess
import threading

import httpx
import pytest
from test_live_execution_contract import runtime  # noqa: F401
from test_run_budget_view import RENDER, node_result
from test_separate_review_pipeline import execute

from litagent import api
from litagent.config import load_config
from litagent.observability.recorder import ArchiveRepository


@pytest.mark.parametrize(
    "scenario",
    [
        "history_first",
        "receipt_first",
        "navigate",
        "interactive",
        "lost_receipt",
        "view_failed",
    ],
)
def test_survey_receipt_survives_workbench_navigation(scenario):
    """Run the production client, store and event handlers with deferred HTTP."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable")
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=process.argv[1],scenario=process.argv[2],app=fs.readFileSync(path.join(root,'app.js'),'utf8');
const load=name=>import('data:text/javascript;base64,'+fs.readFileSync(path.join(root,name)).toString('base64'));
const tick=()=>new Promise(resolve=>setImmediate(resolve));
(async()=>{
 const {createApiClient}=await load('api-client.js');
 const {createWorkbenchStore}=await load('state.js');
 const adapters=await load('adapters.js');
 const requests=[],urls=[],events=[],notices=[],elements=new Map();
 const location={search:''};
 global.fetch=(path,options)=>new Promise((resolve,reject)=>{
  const entry={path,options,reply:(json,status=200)=>resolve(new Response(JSON.stringify(json),{status})),reject};
  requests.push(entry);
  options.signal?.addEventListener('abort',()=>reject(new DOMException('Aborted','AbortError')),{once:true});
 });
 const store=createWorkbenchStore();
 const $=s=>{
  if(!elements.has(s))elements.set(s,{value:'',hidden:false,disabled:false,open:true,dataset:{},close(){this.open=false;}});
  return elements.get(s);
 };
 $('#query-input').value='Public few-shot question';
 $('#quality-policy').value=scenario==='interactive'?'strict_v1':'strict_auto_v2';
 $('#new-citation-style').value='ieee';
 const api=createApiClient(()=>'offline-token');
 const ctx=vm.createContext({store,api,$,context:{live_enabled:true},submittingSurvey:false,historyLoading:false,eventSource:null,
  citationPreferences:new Map(),URLSearchParams,location,history:{replaceState:(_,__,url)=>{urls.push(url);location.search=url;}},
  surveyVerification:{options:()=>({verification:{jev:true}})},t:x=>x,setNotice:message=>notices.push(message),
  liveViewRefresh:{cancel(){}},setInspector(){},closeRunSidebar(){},connectEvents:(...args)=>events.push(args),...adapters});
 const start=app.indexOf('async function refreshInteractive('),end=app.indexOf('async function initialize(');
 vm.runInContext(app.slice(start,end),ctx);
 const history=ctx.loadRunList();
 const formEvent={preventDefault(){}};
 const submission=ctx.startRun(formEvent);
 await ctx.startRun(formEvent);
 const post=requests.find(r=>r.path==='/survey');
 assert.equal(requests.filter(r=>r.options.method==='POST').length,1);
 assert.equal(post.options.signal,undefined);
 assert.equal(post.options.headers['X-LitAgent-Operation-Token'],'offline-token');
 assert.equal($('#new-run-form button[type="submit"]').disabled,true);
 const old={run_id:'old',query:'Old report',status:'completed'};
 function finishList(){requests[0].reply([]);requests[1].reply([old]);}
 const expected=scenario==='interactive'?'interactive:run':'legacy:run';
 const viewPath=scenario==='interactive'?'/survey/run/view?citation_style=ieee':'/flow-demo/runs/run/view?citation_style=ieee';
 const payload={source:'live',mode:scenario==='interactive'?'interactive':undefined,summary:{run_id:'run',query:'Public few-shot question',status:'running'},report:{survey:''}};
 if(scenario==='navigate'){
  store.update({runs:[{key:'archive:old',runId:'old',query:'Old report',status:'completed',source:'archive'},{key:'archive:other',runId:'other',query:'Other',status:'completed',source:'archive'}]});
  const oldSelection=ctx.selectRun('archive:old');
  const nextSelection=ctx.selectRun('archive:other');
  assert.equal(requests.find(r=>r.path==='/flow-demo/runs/old/view').options.signal.aborted,true);
  requests.find(r=>r.path==='/flow-demo/runs/other/view').reply({summary:{run_id:'other',status:'completed'}});
  await Promise.all([oldSelection,nextSelection]);
 }
 if(scenario!=='receipt_first'){
  finishList();
  await history;
  if(scenario!=='navigate')assert.equal(store.get().selectedRunKey,null);
 }
 if(scenario==='lost_receipt'){
  post.reject(new TypeError('Network error'));
  await submission;
  assert.equal($('#survey-submit-status').textContent,'surveyReceiptUnknown');
  assert.equal(store.get().selectedRunKey,null);
 } else {
  post.reply({task_id:'run',status:'queued'},202);
  await tick();
  assert.equal(urls.at(-1),'?run='+encodeURIComponent(expected)+'&view=execution');
  assert.equal(store.get().selectedRunKey,expected);
  assert.ok(store.get().runs.some(r=>r.runId==='run'));
  if(scenario==='receipt_first'){
   finishList(); await history;
   assert.equal(store.get().selectedRunKey,expected);
  }
  const request=requests.find(r=>r.path===viewPath);
  assert.ok(request.options.signal);
  if(scenario==='view_failed'){
   request.reply({detail:'Could not read view'},503);
   await submission;
   assert.match(notices.at(-1),/surveyViewUnavailable.*run/);
   assert.equal(store.get().selectedRunKey,expected);
   const retry=ctx.selectRun(expected,'execution');
   requests.at(-1).reply(payload);
   await retry;
  } else {
   request.reply(payload);
   await submission;
  }
  assert.equal(store.get().view.identity.runId,'run');
  assert.equal(events.at(-1)[0],'run');
 }
 assert.equal(requests.filter(r=>r.options.method==='POST').length,1);
 assert.equal($('#new-run-form button[type="submit"]').disabled,false);
})().catch(error=>{console.error(error);process.exit(1)});
"""
    completed = subprocess.run(
        [node, "-e", script, "src/litagent/static/flow-demo", scenario],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert completed.returncode == 0, completed.stderr


def test_archive_summary_cache_tracks_changed_added_corrupt_and_deleted_files(
    tmp_path, monkeypatch
):
    """Cache metadata only and reuse the existing validated archive reader."""
    repository = ArchiveRepository(tmp_path)
    repository.save(
        {
            "run_id": "old",
            "query": "First question",
            "report": {"survey": "x" * 2000, "quality": {"status": "passed"}},
        }
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_bytes(b"\xff")
    original = repository._read
    read = []

    def measured(path):
        read.append(path.name)
        return original(path)

    monkeypatch.setattr(repository, "_read", measured)
    first = repository.list_summaries()
    assert set(read) == {"old.json", "invalid.json"}
    assert len(first) == 1 and first[0]["quality"] == {"status": "passed"}
    assert "report" not in first[0] and "survey" not in first[0]
    read.clear()
    assert repository.list_summaries() == first
    assert not read
    repository.save({"run_id": "old", "query": "Updated question"})
    repository.save({"run_id": "new", "query": "New question"})
    updated = {item["run_id"]: item for item in repository.list_summaries()}
    assert set(read) == {"old.json", "new.json"}
    assert updated["old"]["query"] == "Updated question"
    assert updated["new"]["query"] == "New question"
    read.clear()
    repository.path_for("old").unlink()
    assert [item["run_id"] for item in repository.list_summaries()] == ["new"]
    assert not read


@pytest.mark.asyncio
async def test_archive_scan_does_not_block_survey_submission(runtime, monkeypatch):
    """Share a cold archive scan while health, config and task creation finish."""
    _, repository = runtime
    repository.save({"run_id": "older", "query": "An archived survey"})
    loop = asyncio.get_running_loop()
    scanning = asyncio.Event()
    release = threading.Event()
    scans = []
    original = repository.list_summaries

    def slow_scan():
        scans.append(True)
        loop.call_soon_threadsafe(scanning.set)
        assert release.wait(timeout=10)
        return original()

    async def local_survey(entry):
        pass

    monkeypatch.setattr(repository, "list_summaries", slow_scan)
    monkeypatch.setattr(api.app.state, "flow_archive_scan", None)
    monkeypatch.setattr(api, "_run_survey", local_survey)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api.app), base_url="http://localhost"
    ) as client:
        first = asyncio.create_task(client.get("/flow-demo/runs"))
        await asyncio.wait_for(scanning.wait(), 5)
        second = asyncio.create_task(client.get("/flow-demo/runs"))
        try:
            async with asyncio.timeout(5):
                health, settings, submitted = await asyncio.gather(
                    client.get("/health"),
                    client.get("/workbench/context"),
                    client.post("/survey", json={"query": "A new survey"}),
                )
            assert [
                health.status_code,
                settings.status_code,
                submitted.status_code,
            ] == [
                200,
                200,
                202,
            ]
            assert submitted.json()["task_id"]
            assert not first.done() and not second.done() and scans == [True]
        finally:
            release.set()
            one, two = await asyncio.gather(first, second)
    assert scans == [True]
    assert one.status_code == two.status_code == 200
    assert one.json() == two.json()
    assert {entry["run_id"] for entry in one.json()} == {
        "older",
        submitted.json()["task_id"],
    }


@pytest.mark.asyncio
async def test_provider_refusal_before_writing_stays_unverified(tmp_path, monkeypatch):
    """Replay the Ark refusal via HTTP and render the terminal quality projection."""

    class Refused:
        unexpected = ()

        def __init__(self):
            self.calls = []

        def __call__(self, request):
            self.calls.append(request.url.path)
            return httpx.Response(
                403,
                headers={"x-request-id": "offline-overdue"},
                json={"error": {"code": "AccountOverdueError", "message": "Overdue"}},
            )

    assistant = Refused()
    report = await execute(
        None,
        tmp_path,
        monkeypatch,
        assistant,
        config=load_config("config/agent-format-repair.yaml"),
    )
    assert assistant.calls == ["/api/v3/chat/completions"]
    assert report["partial"] is True
    assert report["delivery"]["publishable"] is False
    assert report["evaluation"]["faithfulness"]["skipped"] is True
    raw = {
        "report": report,
        **{k: report[k] for k in ("evaluation", "quality", "delivery")},
    }
    for locale, stage, unverified, writing in (
        ("zh-CN", "查询拆分请求被供应商拒绝", "未核验", "写作校验失败"),
        (
            "en",
            "The provider rejected query decomposition",
            "Unverified",
            "Writing validation failed",
        ),
    ):
        text = node_result(RENDER, {"raw": raw}, locale=locale)["text"]
        assert stage in text and unverified.lower() in text.lower()
        assert (
            "HTTP 403" in text and "AccountOverdueError" in text and "blocked" in text
        )
        assert writing not in text


@pytest.mark.parametrize(
    "failure,expected",
    [
        ({}, "报告未生成"),
        ({"writer_failure": {"code": "writer_flat_schema_invalid"}}, "写作校验失败"),
    ],
)
def test_missing_report_summary_uses_available_diagnostic(failure, expected):
    raw = {
        "report": {"metadata": failure},
        "evaluation": {
            "faithfulness": {
                "skipped": True,
                "details": {"skipped_reason": "final_report_unavailable"},
            }
        },
    }
    assert expected in node_result(RENDER, {"raw": raw})["text"]
