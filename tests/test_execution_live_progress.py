"""Verify live lifecycle updates in the production workbench modules."""

from test_execution_evidence_inspection import run_js


def test_live_steps_follow_parallel_workers_and_each_review_round():
    """Trace delivery moves the marker immediately, including both parallel tasks."""
    run_js(
        r"""
const elements = new Map();
document.querySelector = key => {
  if (!elements.has(key)) elements.set(key,new Element('div'));
  return elements.get(key);
};
document.querySelectorAll = () => [];
globalThis.requestAnimationFrame = () => {};
const {createExecutionWorkspace} = await load('execution-workspace.js');
const {createWorkbenchStore} = await load('state.js');
const {createLiveViewRefresh} = await load('live-view.js');
const identity = {key:'legacy:one',runId:'one',source:'legacy',status:'queued'};
const store = createWorkbenchStore({view:{identity,summary:{status:'queued'},graph:{tasks:{},nodes:{},dependencies:{}},events:[]}});
const workspace = createExecutionWorkspace({store,t:translator('en'),onInspect(){}});
const render = () => workspace.render(store.get().view);
store.subscribe(render); render();
const refresh = createLiveViewRefresh({current:store.get,apply:view=>store.update({view}),load:()=>{throw Error('Unexpected GET');}});
let seq=0;
const send=(event,data)=>refresh.receive({event,data,sequence:seq++});
const marked=()=>walk(elements.get('#execution-graph')).filter(n=>n.attrs['aria-current']==='step').map(n=>n.dataset.nodeId).sort();
const bar=()=>words(elements.get('#execution-current-steps'));
const state=()=>elements.get('#execution-run-state').textContent;
assert.equal(state(),'Queued'); assert.deepEqual(marked(),[]);
send('subspan.start',{task_id:'query_decomposition',name:'planner.query_decomposition',round:0});
assert.equal(state(),'Running'); assert.match(bar(),/Query rewriting/);
send('subspan.end',{task_id:'query_decomposition',output:{sub_queries:['one']}});
send('worker.start',{task_id:'search',agent_type:'search'});
send('worker.start',{task_id:'recall',agent_type:'recall'});
assert.deepEqual(marked(),['recall','search']); assert.equal(elements.get('#execution-current-steps').children.length,2);
send('worker.complete',{task_id:'search',output:['paper']});
assert.deepEqual(marked(),['recall']);
send('worker.failed',{task_id:'recall',error:'missing source'});
send('worker.start',{task_id:'write',agent_type:'adversarial_review'});
assert.deepEqual(marked(),['write']);
for (let round=1;round<=2;round++) for(const role of ['synthesis','reviewer']){
  const data={task_id:'write:'+role+':r'+round,name:role+'.r'+round,parent_task_id:'write',round};
  send('subspan.start',data);
  assert.deepEqual(marked(),['write::'+role+':r'+round]); assert.match(bar(),new RegExp('Round '+round));
  send('subspan.end',{...data,output:'recorded'});
}
send('worker.complete',{task_id:'write',output:{total_rounds:2}});
send('subspan.start',{task_id:'evaluation.initial',name:'evaluation.initial'});
assert.deepEqual(marked(),['subspan:evaluation.initial']);
send('subspan.end',{task_id:'evaluation.initial',output:{score:1}});
send('extraction_verification.start',{run_id:'one'});
assert.deepEqual(marked(),['system:extraction-verification']);
send('extraction_verification.complete',{run_id:'one'});
assert.deepEqual(marked(),[]);
const live=store.get().view;
store.update({view:{...live,identity:{...identity,status:'awaiting_human'},review:{stage:'paper_selection'}}});
assert.deepEqual(marked(),['system:human-review']); assert.match(bar(),/Paper selection/);
for (const status of ['completed','failed','rejected','interrupted','cancelled']) {
  store.update({view:{...live,identity:{...identity,status}}});
  assert.deepEqual(marked(),[]); assert.equal(bar(),''); assert(!state().startsWith('status'));
}
refresh.cancel();
"""
    )


def test_snapshot_in_flight_cannot_undo_a_newer_step_or_replay():
    """Both zero-based recorder and one-based SQLite cursors survive stale GETs."""
    run_js(
        r"""
const {createLiveViewRefresh}=await load('live-view.js');
let timer,resolve;
globalThis.setTimeout=fn=>{timer=fn;return 1;};globalThis.clearTimeout=()=>{};
for(const field of ['sequence','seq']){
  const first=field==='seq'?1:0;
  let state={view:{identity:{key:'one',runId:'one',status:'running'},graph:{tasks:{},nodes:{},dependencies:{}},events:[]},requestEpoch:1};
  const refresh=createLiveViewRefresh({current:()=>state,load:()=>new Promise(r=>resolve=r),apply:view=>{state={...state,view};}});
  const start={event:'worker.start',data:{task_id:'one',agent_type:'search'},[field]:first};
  refresh.receive(start);
  const snapshot=structuredClone(state.view);const inFlight=timer();
  refresh.receive({event:'worker.complete',data:{task_id:'one',output:['result']},[field]:first+1});
  resolve(snapshot);await inFlight;
  assert.equal(state.view.graph.nodes['worker:one'].status,'completed');
  assert.deepEqual(state.view.graph.nodes['worker:one'].output,['result']);
  refresh.receive(start);assert.equal(state.view.events.length,2);
  const current=state.view;
  state={...state,view:{...current,displayRevision:1}};
  refresh.receive({...start,[field]:first+2});assert.equal(state.view.events.length,2);
  refresh.cancel();
}
"""
    )
