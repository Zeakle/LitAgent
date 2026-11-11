"""Keep recorded run usage separate from reservations and report quality."""

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

FLOW_ROOT = Path("src/litagent/static/flow-demo").resolve()
RUN_ID = "frozen-038-vit"
BUDGET_FIELDS = (
    "plan_identity",
    "known_tokens",
    "requests",
    "usage_uncertain",
    "stop_reason",
    "unknown_hold",
)


def recorded_budget(**changes):
    """Use the public shape of the retained 038 uncertain usage snapshot."""
    return {
        "contract": "run-budget-state-v1",
        "scope": "run",
        "run_id": RUN_ID,
        "availability": "available",
        "reason": None,
        "plan_identity": "a" * 64,
        "known_tokens": 218506,
        "requests": 14,
        "usage_uncertain": True,
        "stop_reason": "strict_usage_unknown",
        "unknown_hold": 78205,
        **changes,
    }


def node_result(script, value, *, locale="zh-CN", experimental=False):
    """Execute actual browser modules without a browser or external transport."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is unavailable for browser module execution")
    args = [node]
    if experimental:
        args.append("--experimental-vm-modules")
    process = subprocess.run(
        [*args, "-e", script, str(FLOW_ROOT), locale],
        input=json.dumps(value),
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


RENDER = r"""
const fs=require('node:fs');
(async()=>{
 class E {
  constructor(tag){this.tag=tag;this.children=[];this.dataset={};this.textContent='';this.attrs={};}
  append(...x){this.children.push(...x);} replaceChildren(...x){this.children=x;}
  addEventListener(){} setAttribute(k,v){this.attrs[k]=v;}
 }
 global.document={createElement:tag=>new E(tag)};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const adapters=await load('adapters.js'), {renderQuality}=await load('quality-view.js');
 const input=JSON.parse(fs.readFileSync(0,'utf8')), before=JSON.stringify(input), root=new E('main');
 const identity={source:input.source||'archive',runId:input.runId||'frozen-038-vit',status:'interrupted'};
 if(input.mutation)input.raw.budget_state[input.mutation.field]=input.mutation.value==='nan'?NaN:Infinity;
 const view=input.direct?null:adapters[input.source==='interactive'?'adaptInteractiveView':'adaptArchiveView'](input.raw,identity);
 const options={budgetState:input.direct?input.raw.budget_state:view.budgetState,runId:input.noRunId?undefined:identity.runId,locale:process.argv[2],budgetPlan:input.raw.budget_plan};
 const evaluation=input.raw.evaluation||{}, quality=input.raw.quality||{status:'unverified'}, delivery=input.raw.delivery||{status:'blocked',publishable:false};
 renderQuality(root,evaluation,quality,delivery,()=>{},x=>x,options);
 const walk=e=>[e,...e.children.flatMap(walk)],nodes=walk(root), title=process.argv[2]==='en'?'Run budget usage (last recorded)':'运行预算用量（最后记录）';
 const panel=root.children.find(e=>e.children.some(c=>c.textContent===title));
 if(!input.mutation && JSON.stringify(input)!==before)throw Error('Input was mutated');
 console.log(JSON.stringify({budget:view?.budgetState||null,panel:panel?walk(panel).map(e=>e.textContent).join(' '):null,text:nodes.map(e=>e.textContent).join(' '),scores:nodes.filter(e=>e.className==='metric-row').map(e=>e.dataset.status),body:view?.report,quality,delivery}));
})().catch(e=>{console.error(e);process.exit(1)});
"""


def payload(state, **changes):
    """Keep scientific and delivery records explicitly unverified in fixtures."""
    return {
        "budget_state": state,
        "report": {"survey": "# Retained draft", "source_sha256": "b" * 64},
        "evaluation": {"faithfulness": {"score": None, "skipped": True}},
        "quality": {"status": "unverified"},
        "delivery": {"status": "blocked", "publishable": False},
        **changes,
    }


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("source", ["archive", "interactive"])
def test_recorded_usage_reaches_actual_adapter_and_renderer_without_plan(
    locale, source
):
    """Display real values without treating protocol completion as science."""
    raw = payload(recorded_budget())
    before = copy.deepcopy(raw)
    result = node_result(RENDER, {"raw": raw, "source": source}, locale=locale)
    assert result["budget"] == raw["budget_state"]
    en = locale == "en"
    for expected in (
        "Known usage: 218506" if en else "已知用量: 218506",
        "Unknown reservation: 78205" if en else "未知占用: 78205",
        "Requests sent: 14" if en else "已发请求: 14",
        "Usage incomplete" if en else "用量不完整",
        "strict_usage_unknown",
    ):
        assert expected in result["panel"]
    assert "296711" not in result["text"]
    assert "758433" not in result["text"]
    assert result["scores"] == ["unverified"]
    assert result["body"] == raw["report"]
    assert result["quality"] == {"status": "unverified"}
    assert result["delivery"] == {"status": "blocked", "publishable": False}
    assert raw == before


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_real_zero_hold_is_not_inferred_as_final_complete_usage(locale):
    """Explicit zero is a recorded value, while uncertainty remains independent."""
    zero = recorded_budget(
        known_tokens=0,
        unknown_hold=0,
        requests=0,
        usage_uncertain=False,
        stop_reason=None,
    )
    result = node_result(RENDER, {"raw": payload(zero)}, locale=locale)
    assert ("Known usage: 0" if locale == "en" else "已知用量: 0") in result["panel"]
    assert (
        "No unknown reservation recorded in this snapshot"
        if locale == "en"
        else "本次快照未记录未知占用"
    ) in result["panel"]
    uncertain = node_result(
        RENDER,
        {"raw": payload({**zero, "usage_uncertain": True})},
        locale=locale,
    )
    assert ("Usage incomplete" if locale == "en" else "用量不完整") in uncertain[
        "panel"
    ]


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("source", ["archive", "interactive"])
def test_absent_and_declared_missing_usage_never_become_zero(locale, source):
    """Historical missing values stay unavailable rather than an invented bill."""
    declared = {
        "contract": "run-budget-state-v1",
        "scope": "run",
        "run_id": RUN_ID,
        "availability": "missing",
        "reason": "not_recorded",
        **dict.fromkeys(BUDGET_FIELDS),
    }
    for raw in (payload(None), payload(declared), {"summary": {"total_tokens": 77}}):
        result = node_result(RENDER, {"raw": raw, "source": source}, locale=locale)
        assert result["budget"] == declared
        assert ("Not recorded" if locale == "en" else "未记录") in result["panel"]
        assert "0 tokens" not in result["panel"]
        assert "77" not in result["panel"]


@pytest.mark.parametrize("source", ["archive", "interactive"])
def test_both_adapters_reject_invalid_or_cross_run_values_without_leaks(source):
    """Treat injected fields, machine reasons and unsafe numbers as unavailable."""
    states = [
        [],
        {},
        "raw-secret",
        recorded_budget(contract="other"),
        recorded_budget(scope="report"),
        recorded_budget(run_id="another-run"),
        recorded_budget(plan_identity="A" * 64),
        recorded_budget(known_tokens=True),
        recorded_budget(requests=-1),
        recorded_budget(unknown_hold=0.5),
        recorded_budget(known_tokens="218506"),
        recorded_budget(known_tokens=2**53),
        recorded_budget(usage_uncertain="true"),
        recorded_budget(usage_uncertain=False),
        recorded_budget(stop_reason="raw-secret"),
        recorded_budget(reason="raw-secret"),
        recorded_budget(messages=["raw-secret"]),
        {k: v for k, v in recorded_budget().items() if k != "requests"},
    ]
    for state in states:
        result = node_result(RENDER, {"raw": payload(state), "source": source})
        assert result["budget"]["availability"] == "invalid", state
        assert result["budget"]["reason"] == "invalid_record"
        assert all(result["budget"][key] is None for key in BUDGET_FIELDS)
        assert "预算记录不可用" in result["panel"]
        assert "raw-secret" not in json.dumps(result)
        assert "218506" not in result["panel"]


@pytest.mark.parametrize("mutation", ["nan", "infinity"])
@pytest.mark.parametrize("direct", [False, True])
def test_direct_renderer_and_adapter_reject_nonfinite_numbers(mutation, direct):
    """Defend direct renderer calls independently of server JSON validation."""
    result = node_result(
        RENDER,
        {
            "raw": payload(recorded_budget()),
            "mutation": {"field": "known_tokens", "value": mutation},
            "direct": direct,
        },
    )
    assert "预算记录不可用" in result["panel"]
    assert "NaN" not in result["panel"]
    assert "Infinity" not in result["panel"]
    assert "0 tokens" not in result["panel"]


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_direct_renderer_rejects_missing_owner_and_arbitrary_error_text(locale):
    """Do not borrow run identity or render a supplied provider exception."""
    cases = [
        {"raw": payload(recorded_budget()), "noRunId": True},
        {"raw": payload(recorded_budget(run_id="other"))},
        {"raw": payload(recorded_budget(reason="raw-secret"))},
        {"raw": payload(recorded_budget(stop_reason="raw-secret"))},
        {"raw": payload(recorded_budget(secret="raw-secret"))},
    ]
    for case in cases:
        result = node_result(RENDER, {**case, "direct": True}, locale=locale)
        assert (
            "Budget record unavailable" if locale == "en" else "预算记录不可用"
        ) in result["panel"]
        assert "raw-secret" not in result["panel"]
        assert "218506" not in result["panel"]


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_planned_and_allocation_snapshots_remain_separate_from_usage(locale):
    """Preserve timestamps of savings and never infer a usable final balance."""
    plan = {
        "contract": "planned-budget-v7",
        "identity": "a" * 64,
        "status": "admitted",
        "required_tokens": 4000000,
        "planned_tokens": 4480000,
        "maximum_tokens": 4500000,
        "planned_requests": 41,
        "maximum_requests": 64,
        "stages": {},
        "repair_limit": 2,
    }
    evaluation = {
        "faithfulness": {
            "score": None,
            "skipped": True,
            "details": {
                "stages": {
                    "capacity": {
                        "known_tokens_at_allocation": 134461,
                        "future_locked_tokens": 2931869,
                        "settled_reservation_savings": 483523,
                    }
                }
            },
        }
    }
    result = node_result(
        RENDER,
        {"raw": payload(recorded_budget(), budget_plan=plan, evaluation=evaluation)},
        locale=locale,
    )
    for amount in ("218506", "78205", "4480000", "134461", "2931869", "483523"):
        assert amount in result["text"]
    assert "483523" not in result["panel"]
    assert "4480000" not in result["panel"]
    assert "758433" not in result["text"]
    assert "4183289" not in result["text"]
    conflicting = {**plan, "identity": "c" * 64}
    invalid = node_result(
        RENDER,
        {"raw": payload(recorded_budget(), budget_plan=conflicting)},
        locale=locale,
    )
    assert invalid["budget"]["availability"] == "invalid"


@pytest.mark.parametrize("source", ["archive", "interactive"])
@pytest.mark.parametrize("direct", [False, True])
def test_present_null_plan_identity_conflicts_but_missing_key_does_not(source, direct):
    """Align defensive consumers with the projection's present-key contract."""
    base = {"raw": payload(recorded_budget()), "source": source, "direct": direct}
    no_identity = copy.deepcopy(base)
    no_identity["raw"]["budget_plan"] = {"contract": "planned-budget-v7"}
    valid = node_result(RENDER, no_identity)
    assert "已知用量: 218506" in valid["panel"]
    null_identity = copy.deepcopy(base)
    null_identity["raw"]["budget_plan"] = {
        "contract": "planned-budget-v7",
        "identity": None,
    }
    invalid = node_result(RENDER, null_identity)
    assert "预算记录不可用" in invalid["panel"]
    assert "218506" not in invalid["panel"]


def test_version_and_style_controls_preserve_run_usage_and_ignore_late_response():
    """Exercise actual format handlers and epochs without revising run accounts."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {createWorkbenchStore}=await load('state.js'),{adaptInteractiveView,adaptArchiveView}=await load('adapters.js'),{bindFormatControls}=await load('format-controls.js');
 const raw=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(raw), elements=new Map();
 const element=key=>{if(!elements.has(key))elements.set(key,{events:{},value:key==='#export-style'?'ieee':'2',addEventListener(name,fn){this.events[name]=fn;}});return elements.get(key);};
 global.document={querySelector:element,querySelectorAll:()=>[]};
 const view=adaptInteractiveView(raw,{runId:'frozen-038-vit',source:'interactive'}),store=createWorkbenchStore({view}),requests=[];
 const version={report_sha256:'c'.repeat(64),report:{survey:'# Revision',evaluation:{},quality:{status:'unverified'},delivery:{status:'blocked'}},rendered_report:{html:'revision'}};
 let pendingResolve=null;
 const api={json:async path=>{requests.push(path);return pendingResolve===false?await new Promise(resolve=>{pendingResolve=resolve;}):version;}};
 bindFormatControls({store,api,t:x=>x,setNotice(){},citationPreferences:new Map()});
 const originalBudget=JSON.stringify(view.budgetState);
 await element('#version-select').events.change({target:{value:'2'}});
 if(store.get().view.reportSha!=='c'.repeat(64)||JSON.stringify(store.get().view.budgetState)!==originalBudget)throw Error('Revision changed run usage');
 await element('#export-style').events.change({target:{value:'apa'}});
 if(JSON.stringify(store.get().view.budgetState)!==originalBudget)throw Error('Style changed run usage');
 pendingResolve=false;const late=element('#version-select').events.change({target:{value:'3'}});
 store.beginRequest();store.update({view:adaptArchiveView({summary:{run_id:'run-b'},report:{survey:'# B'}},{runId:'run-b',source:'archive'})});
 pendingResolve(version);await late;
 if(store.get().view.identity.runId!=='run-b'||store.get().view.budgetState.availability!=='missing'||store.get().view.budgetState.known_tokens!==null)throw Error('Late A response contaminated B');
 if(JSON.stringify(raw)!==before)throw Error('Controls changed archived input');
 if(requests.some(path=>!/\/versions\//.test(path)))throw Error('Unexpected paid or mutation request');
 console.log(JSON.stringify({revision_and_style_requests:requests.length,known_tokens:store.get().view.budgetState.known_tokens,model_requests:0}));
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = node_result(script, payload(recorded_budget()))
    assert result == {
        "revision_and_style_requests": 3,
        "known_tokens": None,
        "model_requests": 0,
    }


def test_actual_app_passes_snapshot_owner_and_clears_old_run_before_fetch():
    """Execute real app, adapters, store and renderer through archive selection."""
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
(async()=>{
 const raw=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(raw),elements=new Map(),calls=[];
 class E {
  constructor(tag='div'){this.tag=tag;this.children=[];this.dataset={};this.events={};this.attrs={};this.options=[];this.textContent='';this.value='';this.hidden=false;this.classList={add(){},remove(){}};}
  append(...x){this.children.push(...x);} replaceChildren(...x){this.children=x;}
  addEventListener(k,v){this.events[k]=v;}setAttribute(k,v){this.attrs[k]=v;}
  add(x){this.options.push(x);}querySelectorAll(){return [];}focus(){}
 }
 const element=key=>{if(!elements.has(key))elements.set(key,new E());return elements.get(key);};
 let pendingB;
 const api={abort(){},json:async url=>{calls.push(url);if(url==='/workbench/context')return {mode:'presentation',live_enabled:false};if(url.includes('/frozen-038-vit/'))return raw;if(url.includes('/run-b/'))return await new Promise(resolve=>{pendingB=resolve;});throw Error('Unexpected route '+url);}};
 const sandbox=vm.createContext({document:{documentElement:{},querySelector:element,querySelectorAll:()=>[],createElement:tag=>new E(tag),addEventListener(){}},
  localStorage:{getItem:()=>null,setItem(){}},location:{search:'?run=archive%3Afrozen-038-vit&view=quality'},history:{replaceState(){}},
  matchMedia:()=>({matches:false,addEventListener(){}}),URLSearchParams,Option:function(label,value){this.textContent=label;this.value=String(value);},console,setTimeout,clearTimeout});
 const mocks={createApiClient:()=>api,openEventStream(){throw Error('Unexpected event stream');},createExecutionWorkspace:()=>({render(){},bind(){}}),
  bindAuditControls(){},bindSurveyVerification:()=>({render(){},options:()=>({})}),bindFormatControls(){},
  createBenchmarkController:()=>({load(){},select(){},selectObservation(){}}),renderBenchmarkDetail(){},renderBenchmarkList(){},renderBenchmarkObservation(){},
  evidenceMap:()=>new Map(),renderEvidenceList(){},renderExecutionInspector(){},renderHistory(){},renderReview(){},
  renderOutline(){},navigateReportHeading(){},renderReportDocument:()=>({headings:[]}),renderSearchRecords(){}};
 const real=new Set(['app.js','adapters.js','state.js','quality-view.js','i18n.js']),cache=new Map();
 async function load(filename){if(cache.has(filename))return cache.get(filename);
  const source=real.has(filename)?fs.readFileSync(path.join(process.argv[1],filename),'utf8')+(filename==='app.js'?'\nexport {selectRun,store};':''):null;
  const m=source===null?new vm.SyntheticModule(Object.keys(mocks),function(){for(const [key,value]of Object.entries(mocks))this.setExport(key,value);},{context:sandbox,identifier:filename}):new vm.SourceTextModule(source,{context:sandbox,identifier:filename});
  cache.set(filename,m);await m.link(spec=>load(spec.split('?')[0].replace('./','')));return m;
 }
 const app=await load('app.js');await app.evaluate();await new Promise(resolve=>setImmediate(resolve));
 const walk=e=>[e,...e.children.flatMap(walk)],words=()=>walk(element('#quality-content')).map(e=>e.textContent).join(' '),store=app.namespace.store;
 if(!words().includes('已知用量: 218506')||!words().includes('未知占用: 78205'))throw Error('Actual app omitted budgetState or runId');
 if(element('#summary-tokens').textContent!==Number(218506).toLocaleString())throw Error('Summary mixed unknown into known');
 store.update({runs:[...store.get().runs,{key:'archive:run-b',runId:'run-b',source:'archive'}]});
 const changing=app.namespace.selectRun('archive:run-b','quality');
 if(store.get().view!==null||words().includes('218506'))throw Error('Old run budget survived selection');
 pendingB({summary:{run_id:'run-b'},report:{survey:'# Legacy B'},quality:{status:'unverified'},delivery:{status:'blocked'}});await changing;
 if(store.get().view.budgetState.availability!=='missing'||!words().includes('未记录')||words().includes('78205'))throw Error('Missing B budget borrowed A');
 if(JSON.stringify(raw)!==before||calls.length!==3||calls.some(url=>/audits|export|decision/.test(url)))throw Error('Unexpected mutation or extra request');
 console.log(JSON.stringify({actual_app_consumers:true,get_requests:calls.length,model_requests:0}));
})().catch(e=>{console.error(e);process.exit(1)});
"""
    raw = payload(recorded_budget(), summary={"total_tokens": 218506})
    assert node_result(script, raw, experimental=True) == {
        "actual_app_consumers": True,
        "get_requests": 3,
        "model_requests": 0,
    }
