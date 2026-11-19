"""Test real application selection and event handlers for ordinary live runs."""

import json

import pytest
from test_execution_evidence_inspection import run_js


@pytest.mark.parametrize(
    "scenario", ["running", "failed", "cancel", "late", "archive_live", "settle"]
)
def test_live_selection_preserves_graph_and_request_identity(scenario):
    """Exercise application code with only HTTP, DOM and stream boundaries replaced."""
    run_js(
        "const scenario="
        + json.dumps(scenario)
        + r""";
await (async () => {
const fs = await import('node:fs');
const vm = await import('node:vm');
const {createWorkbenchStore}=await load('state.js');
const adapters=await load('adapters.js');
const app=fs.readFileSync(new URL(root+'/app.js'),'utf8');
const store=createWorkbenchStore({activeSection:'evidence',sectionWasUserSelected:true});
const requests=[],streams=[],urls=[],elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{value:'',addEventListener(name,fn){this[name]=fn;}});return elements.get(id);};
const raw=status=>({source:status==='running'?'live':'archive',summary:{run_id:'r',query:'query',status},dag:{tasks:{extract:{task_id:'extract',agent_type:'extractor'}},nodes:{'worker:extract':{output:['actual claim']}},dependencies:{}},timeline:[],report:{survey:'',metadata:{}},delivery:{}});
let nextStatus=scenario==='failed'?'failed':'running';
const api={abort(){},async request(path,options){assert.equal(options.method,'DELETE');nextStatus='cancelled';},json(path){requests.push(path);if(scenario==='late')return new Promise(resolve=>requests.resolve=resolve);return Promise.resolve(raw(nextStatus));}};
const ctx=vm.createContext({store,api,$,...adapters,citationPreferences:new Map(),eventSource:null,URLSearchParams,
  history:{replaceState(_,__,url){urls.push(url);}},location:{search:''},liveViewRefresh:{cancel(){},schedule(){}},
  setNotice(message){throw Error(message);},t:x=>x,closeRunSidebar(){},setInspector(){},
  openEventStream(runId,handlers){const stream={runId,handlers,closed:false,close(){this.closed=true;}};streams.push(stream);return stream;}});
const start=app.indexOf('function connectEvents('),end=app.indexOf('async function loadRunList(');
vm.runInContext(app.slice(start,end),ctx);
ctx.loadRunList=async()=>{};
if(scenario==='late'){
  const pending=ctx.refreshLegacy('r','query');
  store.beginRequest();store.update({selectedRunKey:'archive:other',view:{identity:{key:'archive:other',runId:'other'}}});
  requests.resolve(raw('running'));assert.equal(await pending,false);
  assert.equal(store.get().selectedRunKey,'archive:other');assert.equal(streams.length,0);return;
}
if(scenario==='archive_live'){
  store.update({runs:[{key:'archive:r',runId:'r',source:'archive'}]});
  assert.equal(await ctx.selectRun('archive:r','evidence'),true);
}else assert.equal(await ctx.refreshLegacy('r','query'),true);
assert.equal(store.get().view.graph.nodes['worker:extract'].output[0],'actual claim');
assert.equal(store.get().activeSection,'evidence');
assert.equal(requests.length,1);assert.equal(requests[0],'/flow-demo/runs/r/view');
assert.equal(store.get().selectedRunKey,scenario==='failed'?'archive:r':'legacy:r');
assert.equal(streams.length,scenario==='failed'?0:1);
assert(urls.at(-1).includes(encodeURIComponent(store.get().selectedRunKey)));
const normalized=adapters.normalizeRunList([],[{run_id:'r',source:'live',status:'running'}]);
assert.equal(normalized[0].source,'legacy');
if(scenario==='cancel'){
  const line=app.split('\n').find(line=>line.startsWith('$("#cancel-run").addEventListener'));
  vm.runInContext(line,ctx);await $('#cancel-run').click();
  assert.equal(store.get().selectedRunKey,'archive:r');
  assert.equal(store.get().view.identity.status,'cancelled');
  assert.equal(store.get().view.capabilities.canCancel,false);
  assert.equal(streams[0].closed,true);
}else if(scenario==='settle'){
  nextStatus='completed';await streams[0].handlers.onSettle();
  assert.equal(store.get().selectedRunKey,'archive:r');
  assert.equal(store.get().activeSection,'evidence');
  assert.equal(store.get().view.graph.nodes['worker:extract'].output[0],'actual claim');
  assert.equal(streams[0].closed,true);
}else if(scenario==='running'){
  const previous=streams[0].handlers;
  store.beginRequest();store.update({selectedRunKey:'archive:other',view:{identity:{key:'archive:other',runId:'other'}}});
  await previous.onSettle();
  assert.equal(requests.length,1);assert.equal(store.get().selectedRunKey,'archive:other');
}
})();
"""
    )
