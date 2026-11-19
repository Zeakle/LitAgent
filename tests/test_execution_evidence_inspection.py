"""Exercise read-only node inspection and version-bound evidence display."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path("src/litagent/static/flow-demo").resolve()


def run_js(body: str) -> None:
    """Run actual ES modules with a small DOM boundary, without model calls."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable")
    script = r"""
import assert from 'node:assert/strict';
const root = ROOT;
const load = name => import(root + '/' + name);
class Element {
  constructor(tag) { this.tag = tag; this.children = []; this.events = {}; this.attrs = {}; this.dataset = {}; this.style = {}; this.textContent = ''; this.classList = {add(){}}; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(name, fn) { this.events[name] = fn; }
  setAttribute(key, value) { this.attrs[key] = value; }
  set innerHTML(value) { throw Error('Unsafe HTML rendering'); }
}
globalThis.document = {createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag)};
const walk = e => [e, ...e.children.flatMap(walk)];
const words = e => walk(e).map(n => n.textContent).join('\n');
const {translator} = await load('i18n.js');
const item = {evidence_id:'card:1',paper_id:'paper:1',text:'Exact extracted claim',supporting_text:'<script>untrusted</script>\nFull original source, including its final qualifier.',source_locator:'page:3:block:8',confidence:0.2,fact_fields:{metric:'accuracy'}};
const decision = kind => ({question_id:kind+':extract:1',claim_id:'extract:1',kind,evidence_id:'card:1',choice:'ok',probabilities:{ok:0.93,bad:0.07},confidence:0.93,uncertain:false,status:'valid'});
const receipt = {status:'completed',stage:'extraction',source_run_id:'source-run',confidence_threshold:0.6,candidate_ledger:{'card:1':item},accepted_evidence_ids:['card:1'],rejected:{},jev:{actual_model:'jev-1.13.0',inputs:[{claim_id:'extract:1',source_unit_id:'card:1',evidence:[{evidence_id:'card:1'}]}],decisions:[decision('claim_validity'),decision('claim_format')]}};
const view = {identity:{key:'archive:run',runId:'run'},evidence:[item],report:{metadata:{extraction_verification:receipt}},graph:{nodes:{'worker:dedup':{output:[{paper_id:'paper:1',title:'Source paper',url:'https://example.org/paper'}]}}}};
""".replace(
        "ROOT", json.dumps(ROOT.as_uri())
    )
    result = subprocess.run(
        [node, "--input-type=module", "--eval", script + body],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_evidence_click_renders_exact_source_and_distinct_jev_scores(locale):
    """Click through the actual list and inspect both Jev questions safely."""
    run_js(
        f"const t = translator({json.dumps(locale)});"
        + r"""
const {buildEvidenceLedger,filterEvidence} = await load('evidence-model.js');
const {renderEvidenceList,renderEvidenceInspector} = await load('evidence-view.js');
const before = JSON.stringify(view), ledger = buildEvidenceLedger(view);
assert.equal(ledger[0].inspection.state, 'accepted');
assert.equal(filterEvidence(ledger,'final qualifier','accepted').length, 1);
assert.equal(filterEvidence(ledger,'absent','all').length, 0);
const list = new Element('div'), pane = new Element('aside');
renderEvidenceList(list,ledger,t,selection => renderEvidenceInspector(pane,selection,t,ledger));
walk(list).find(e=>e.tag==='button').events.click();
assert.ok(words(pane).includes(item.supporting_text));
assert.ok(words(pane).includes(item.text));
assert.ok(words(pane).includes('93.0%'));
assert.ok(words(pane).includes('20.0%'));
assert.ok(words(pane).includes(t('jev_claim_validity')));
assert.ok(words(pane).includes(t('jev_claim_format')));
assert.ok(words(pane).includes('page:3:block:8'));
assert.equal(walk(pane).find(e=>e.tag==='a').href, 'https://example.org/paper');
assert.equal(JSON.stringify(view),before);
"""
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "changed.evidence[0].text = 'Different claim'",
        "changed.evidence[0].supporting_text = 'Different source'",
        "changed.evidence[0].fact_fields.metric = 'latency'",
        "changed.evidence[0].source_locator = 'page:4'",
        "changed.evidence[0].paper_id = 'different-paper'",
    ],
)
def test_same_id_with_changed_source_never_inherits_jev(mutation):
    """Evidence identity alone cannot attach a previous source's scores."""
    run_js(
        "const changed = JSON.parse(JSON.stringify(view));"
        + mutation
        + r""";
const {buildEvidenceLedger} = await load('evidence-model.js');
const entry = buildEvidenceLedger(changed)[0];
assert.equal(entry.inspection.state, 'mismatch');
assert.deepEqual(entry.inspection.decisions, []);
"""
    )


def test_jev_missing_disabled_rejected_and_partial_records_remain_distinct():
    """Retain rejected candidates and avoid manufacturing a passed status."""
    run_js(
        r"""
const {buildEvidenceLedger,probability,safePaperUrl} = await load('evidence-model.js');
view.evidence = [];
receipt.accepted_evidence_ids = [];
receipt.rejected = {'card:1':['extraction_low_confidence']};
assert.equal(buildEvidenceLedger(view)[0].inspection.state,'rejected');
assert.equal(buildEvidenceLedger(view)[0].inspection.selected,false);
receipt.rejected = {}; receipt.accepted_evidence_ids = ['card:1'];
receipt.jev.decisions.pop();
assert.equal(buildEvidenceLedger(view)[0].inspection.state,'unverified');
receipt.jev.decisions.push({...decision('claim_format'),claim_id:'wrong-claim'});
assert.equal(buildEvidenceLedger(view)[0].inspection.state,'unverified');
view.evidence = [item]; view.report.metadata = {verification:{jev:false}};
assert.equal(buildEvidenceLedger(view)[0].inspection.state,'disabled');
view.report.metadata = {};
assert.equal(buildEvidenceLedger(view)[0].inspection.state,'unverified');
for (const value of [null,'0.9',NaN,Infinity,-1,1.1]) assert.equal(probability(value),'—');
assert.equal(probability(0),'0.0%');
for (const value of ['javascript:alert(1)','data:text/html,unsafe','file:///private','https://u:p@example.com']) assert.equal(safePaperUrl(value),null);
"""
    )


def test_inspector_uses_latest_node_payload_and_exact_call_parent():
    """The clicked worker links only its recorded calls and updates payloads."""
    run_js(
        r"""
const {renderExecutionInspector,relatedCalls} = await load('execution-inspector.js');
const nodes = {
 'worker:extract':{kind:'worker',task_id:'extract',status:'completed',input:{papers:['p1']},output:{claims:['latest']}},
 'llm:call1':{kind:'llm',task_id:'extract',input:{prompt:'actual request'},output:'actual response'},
 'tool:call2':{kind:'tool',task_id:'other',input:{parent_task_id:'extract'},output:{papers:2}},
 'llm:unrelated':{kind:'llm',task_id:'extract-other',output:'must not join'},
};
assert.deepEqual(relatedCalls(nodes,'extract').map(([id])=>id),['llm:call1','tool:call2']);
const pane = new Element('aside'), t=translator('en');
const inspect=selection=>{pane.replaceChildren();renderExecutionInspector(pane,selection,t,{view:{graph:{nodes}},onSelect:inspect});};
inspect({kind:'node',id:'extract',value:{id:'extract',status:'running',execution:{output:'stale'}}});
assert.ok(words(pane).includes('latest'));
assert.ok(!words(pane).includes('must not join'));
const buttons=walk(pane).filter(e=>e.tag==='button'); assert.equal(buttons.length,2);
buttons[0].events.click();
assert.ok(words(pane).includes('actual request')); assert.ok(words(pane).includes('actual response'));
inspect({kind:'observation',id:'system:delivery',value:{input:{quality:'failed'},output:{status:'blocked'},raw:{status:'blocked'}}});
assert.ok(words(pane).includes('quality'));
nodes['subspan:review:r1']={id:'subspan:review:r1',kind:'subspan',task_id:'review:r1',input:{parent_task_id:'extract'}};
nodes['llm:nested']={kind:'llm',task_id:'review:r1',input:'nested request',output:'nested response'};
inspect({kind:'observation',id:'subspan:review:r1',value:nodes['subspan:review:r1']});
const nested=walk(pane).filter(e=>e.tag==='button');assert.equal(nested.length,1);
nested[0].events.click();assert.ok(words(pane).includes('nested response'));
"""
    )


def test_live_refresh_coalesces_events_and_rejects_late_run_or_version_responses():
    """Old network responses must not overwrite another run or report revision."""
    run_js(
        r"""
const timers = new Map(); let counter=0;
globalThis.setTimeout = fn => {timers.set(++counter,fn);return counter;};
globalThis.clearTimeout = id => timers.delete(id);
const fire = () => {const [id,fn]=timers.entries().next().value;timers.delete(id);return fn();};
const {createLiveViewRefresh}=await load('live-view.js');
let state={view,requestEpoch:1}, resolve, applied=[];
const refresh=createLiveViewRefresh({current:()=>state,load:()=>new Promise(r=>resolve=r),apply:v=>applied.push(v)});
refresh.schedule(); refresh.schedule(); assert.equal(timers.size,1);
let pending=fire(); state={view:{identity:{key:'archive:other',runId:'other'}},requestEpoch:2};
resolve({latest:true}); await pending; assert.equal(applied.length,0);
state={view,requestEpoch:3}; refresh.schedule(); pending=fire();
state={...state,view:{...view,displayRevision:1}}; resolve({latest:true});await pending;
assert.equal(applied.length,0);
state={view,requestEpoch:4};refresh.schedule();pending=fire();refresh.cancel();resolve({old:true});await pending;
assert.equal(applied.length,0);
refresh.schedule();pending=fire();resolve({latest:true});await pending;
assert.deepEqual(applied,[{latest:true}]);
"""
    )


def test_business_graph_clicks_observed_stages_with_explicit_sequence_edges():
    """Render the workspace and drill into a recorded evaluation stage."""
    run_js(
        r"""
const elements=new Map();
document.querySelector=selector=>{if(!elements.has(selector))elements.set(selector,new Element('div'));return elements.get(selector);};
document.querySelectorAll=()=>[];
globalThis.requestAnimationFrame=()=>{};
const {createWorkbenchStore}=await load('state.js');
const {createExecutionWorkspace}=await load('execution-workspace.js');
const {renderExecutionInspector}=await load('execution-inspector.js');
view.graph={tasks:{extract:{task_id:'extract',agent_type:'extractor'}},dependencies:{extract:[]},nodes:{
 'worker:extract':{kind:'worker',task_id:'extract',status:'completed',input:['paper'],output:['claim']},
 'subspan:evaluation.initial':{kind:'subspan',task_id:'evaluation.initial',name:'evaluation.initial',status:'completed',input:{report:'exact draft'},output:{score:0.4}},
 'subspan:planner':{kind:'subspan',task_id:'planner',name:'planner',status:'completed',input:{query:'question'},output:{tasks:['extract']}}
}};
const store=createWorkbenchStore({view});
let selection;
const workspace=createExecutionWorkspace({store,t:translator('en'),onInspect:s=>selection=s});
workspace.render(view);
const nodes=walk(elements.get('#execution-graph')).filter(e=>e.className==='graph-node');
assert.equal(nodes.length,4);
const evaluation=nodes.find(e=>words(e).includes('evaluation.initial'));assert.ok(evaluation);
evaluation.events.click();
const pane=new Element('aside');renderExecutionInspector(pane,selection,translator('en'),{view});
assert.ok(words(pane).includes('exact draft'));assert.ok(words(pane).includes('0.4'));
const edges=walk(elements.get('#execution-graph')).filter(e=>e.tag==='path'&&e.attrs['marker-end']);
assert.ok(edges.length > 0);
assert.ok(edges.every(edge=>edge.dataset.kind==='sequence'));
const positions=new Map(nodes.map(node=>[node.dataset.nodeId,parseFloat(node.style.top)]));
assert.ok(positions.get('subspan:planner') < positions.get('extract'));
assert.ok(positions.get('extract') < positions.get('subspan:evaluation.initial'));
"""
    )


def test_workbench_serves_all_inspection_modules_without_runtime_services():
    """Check actual ASGI static routes without starting paid or network work."""
    from fastapi.testclient import TestClient

    from litagent.api import app

    client = TestClient(app)
    page = client.get("/flow-demo")
    assert page.status_code == 200
    assert 'id="evidence-search"' in page.text
    assert 'id="graph-search"' in page.text
    for module in (
        "app.js",
        "evidence-model.js",
        "evidence-view.js",
        "execution-workspace.js",
        "execution-flow.js",
        "execution-inspector.js",
        "execution-content.js",
        "live-view.js",
    ):
        response = client.get(f"/static/flow-demo/{module}?v=15.10.6.426")
        assert response.status_code == 200
        assert response.content == (ROOT / module).read_bytes()


@pytest.mark.parametrize("direction", ["LR", "TB"])
def test_packaged_sample_places_evaluation_delivery_after_review_with_return_arc(
    direction,
):
    """Exercise the exact sample displayed to the user through real producers."""
    from litagent.demo import load_bundled_samples, project_flow_demo_view

    raw = project_flow_demo_view(
        load_bundled_samples()["sample-ready"], source="sample"
    )
    run_js(
        "const direction = "
        + json.dumps(direction)
        + "; const raw = "
        + json.dumps(raw)
        + r""";
const {adaptArchiveView}=await load('adapters.js');
const {buildExecutionPresentation}=await load('execution-model.js');
const {buildBusinessFlow}=await load('execution-flow.js');
const {layoutExecutionGraph,renderExecutionGraph}=await load('execution-view.js');
const sample=adaptArchiveView(raw,{key:'sample:sample-ready',runId:'sample-ready',source:'sample'});
const original=JSON.stringify(sample),model=buildExecutionPresentation(sample);
const flow=buildBusinessFlow(model,sample.graph.nodes),layout=layoutExecutionGraph(flow,{direction});
const x=id=>layout.byId.get(id)[direction==='TB'?'y':'x'];
assert.ok(x('relevance_gate')<x('extract'));
assert.ok(x('extract')<x('adversarial_review::synthesis:r1'));
assert.ok(x('adversarial_review::synthesis:r1')<x('adversarial_review::reviewer:r2'));
assert.ok(x('adversarial_review::reviewer:r2')<x('subspan:evaluation.initial'));
assert.ok(x('subspan:evaluation.initial')<x('system:delivery'));
assert.equal(flow.cyclic,false);
assert.deepEqual(flow.edges.filter(e=>e.kind==='dependency').map(e=>e.sourceEdge),model.dependencies);
const feedback=flow.edges.find(e=>e.kind==='feedback');
assert.equal(feedback.from,'adversarial_review::reviewer:r1');assert.equal(feedback.to,'adversarial_review::synthesis:r2');
assert.ok(x('adversarial_review::reviewer:r1')<x('adversarial_review::synthesis:r2'));
assert.ok(x('system:query-decomposition')<x('relevance_gate'));
const pane=new Element('div');
renderExecutionGraph(pane,layout,()=>{}, {edgeLabel:edge=>edge.kind==='feedback'?'Recorded review loop':''});
const arc=walk(pane).find(e=>e.tag==='path'&&e.dataset.kind==='feedback');
assert.ok(arc.attrs.d.includes(' C '));assert.ok(arc.attrs['marker-end']);
assert.ok(words(pane).includes('Recorded review loop'));
if(direction==='TB') {
 const from=layout.byId.get(feedback.from),to=layout.byId.get(feedback.to);
 assert.ok(arc.attrs.d.startsWith(`M ${from.x+from.width} ${from.y+from.height/2} C `));
 assert.ok(arc.attrs.d.endsWith(`${to.x+to.width} ${to.y+to.height/2}`));
 assert.ok(layout.bounds.width>from.x+from.width+200);
 const edge=flow.edges.find(e=>e.kind!=='feedback'),path=walk(pane).find(e=>e.dataset.edgeId===edge.id);
 const a=layout.byId.get(edge.from),b=layout.byId.get(edge.to);
 assert.ok(path.attrs.d.startsWith(`M ${a.x+a.width/2} ${a.y+a.height} C `));
 assert.ok(path.attrs.d.endsWith(`${b.x+b.width/2} ${b.y}`));
 const parallel=layoutExecutionGraph({nodes:[{id:'a',rank:0},{id:'b',rank:0},{id:'c',rank:1}],edges:[]},{direction});
 assert.ok(parallel.byId.get('a').x+248<parallel.byId.get('b').x);
 assert.equal(parallel.byId.get('a').y,parallel.byId.get('b').y);
 assert.equal(parallel.byId.get('c').x,(parallel.byId.get('a').x+parallel.byId.get('b').x)/2);
} else assert.ok(layout.bounds.height > layout.byId.get(feedback.from).y+layout.byId.get(feedback.from).height+70);
assert.equal(JSON.stringify(sample),original);
"""
    )


def test_loop_roles_preserve_exact_parent_rounds_and_clickable_io():
    """Keep two adversarial tasks separate and drill into a real review round."""
    run_js(
        r"""
const {buildExecutionPresentation}=await load('execution-model.js');
const {buildBusinessFlow}=await load('execution-flow.js');
const {renderExecutionInspector}=await load('execution-inspector.js');
view.graph={tasks:{write:{task_id:'write',agent_type:'adversarial_review'}},dependencies:{write:[]},nodes:{}};
const span=(id,name,round,output,parent='write')=>({id,kind:'subspan',task_id:id,name,status:'completed',input:{parent_task_id:parent,round,prompt:'round '+round},output});
view.graph.nodes={
 'worker:write':{kind:'worker',task_id:'write',status:'completed',output:{total_rounds:2}},
 s1:span('s1','synthesis.r1',1,'draft 1'), r1:span('r1','reviewer.r1',1,'review 1'),
 s2:span('s2','revision.r2',2,'draft 2'), r2:span('r2','reviewer.r2',2,'review 2'),
 foreign:span('foreign','reviewer.r3',3,'foreign review','write-other'),
};
const flow=buildBusinessFlow(buildExecutionPresentation(view),view.graph.nodes);
const synthesis=flow.nodes.find(n=>n.role==='synthesis'),review=flow.nodes.find(n=>n.role==='reviewer');
assert.deepEqual(synthesis.roundRecords.map(r=>r.id),['s1','s2']);
assert.deepEqual(review.roundRecords.map(r=>r.id),['r1','r2']);
const pane=new Element('aside');let selection;
renderExecutionInspector(pane,{kind:'node',id:review.id,value:review},translator('en'),{view,onSelect:s=>selection=s});
assert.ok(words(pane).includes('review 1'));assert.ok(!words(pane).includes('review 2'));assert.ok(!words(pane).includes('foreign review'));
const second = flow.nodes.find(n=>n.role==='reviewer'&&n.round===2);
const secondPane = new Element('aside');
renderExecutionInspector(secondPane,{kind:'node',id:second.id,value:second},translator('en'),{view});
assert.ok(words(secondPane).includes('review 2'));assert.ok(!words(secondPane).includes('review 1'));
assert.deepEqual(flow.nodes.filter(n=>n.role).map(n=>[n.role,n.round]),[['synthesis',1],['reviewer',1],['synthesis',2],['reviewer',2]]);
walk(pane).find(e=>e.tag==='button'&&e.textContent.startsWith('reviewer.r1')).events.click();
pane.replaceChildren();renderExecutionInspector(pane,selection,translator('en'),{view});
assert.ok(words(pane).includes('review 1'));assert.ok(words(pane).includes('round 1'));
"""
    )


def test_single_round_keeps_feedback_conditional_and_does_not_fabricate_io():
    """A workflow arc is not evidence that an unrecorded revision happened."""
    run_js(
        r"""
const {buildExecutionPresentation}=await load('execution-model.js');
const {buildBusinessFlow}=await load('execution-flow.js');
const {renderExecutionInspector}=await load('execution-inspector.js');
view.graph={tasks:{write:{task_id:'write',agent_type:'adversarial'}},dependencies:{write:[]},nodes:{
 'worker:write':{kind:'worker',task_id:'write',status:'completed',input:'whole input',output:{total_rounds:1,final_draft:'whole output'}}
}};
const flow=buildBusinessFlow(buildExecutionPresentation(view),view.graph.nodes);
assert.equal(flow.edges.find(e=>e.kind==='feedback').labelKey,'flowFeedbackConditional');
const role=flow.nodes.find(n=>n.role==='synthesis');assert.equal(role.execution,null);assert.equal(role.status,'unknown');
const pane=new Element('aside');let selected;
renderExecutionInspector(pane,{kind:'node',id:role.id,value:role},translator('en'),{view,onSelect:s=>selected=s});
const io=pane.children.filter(e=>e.tag==='details').slice(0,2);
assert.ok(io.every(e=>words(e).includes('Not recorded')));
walk(pane).find(e=>e.tag==='button'&&e.textContent==='Inspect complete adversarial stage').events.click();
pane.replaceChildren();renderExecutionInspector(pane,selected,translator('en'),{view});
assert.ok(words(pane).includes('whole input'));assert.ok(words(pane).includes('whole output'));
"""
    )


@pytest.mark.parametrize("recorded", [True, False])
def test_query_rewriting_precedes_each_search_without_inventing_output(recorded):
    """Show the real subqueries or an explicit missing record before discovery."""
    run_js(
        "const recorded = "
        + json.dumps(recorded)
        + ";"
        + r"""
const {buildExecutionPresentation}=await load('execution-model.js');
const {buildBusinessFlow}=await load('execution-flow.js');
const {renderExecutionInspector}=await load('execution-inspector.js');
view.summary={query:'Compare method A and method B'};
view.graph={tasks:{a:{task_id:'a',agent_type:'search'},b:{task_id:'b',agent_type:'recall'}},dependencies:{a:[],b:[]},nodes:{}};
if(recorded) view.graph.nodes['subspan:query_decomposition']={
 kind:'subspan',task_id:'query_decomposition',name:'planner.query_decomposition',status:'completed',
 input:{parent_task_id:'',name:'planner.query_decomposition',round:0},output:{sub_queries:['method A original paper','method B original paper']}
};
const before=JSON.stringify(view),flow=buildBusinessFlow(buildExecutionPresentation(view),view.graph.nodes);
const query=flow.nodes.find(n=>n.labelKey==='flowQueryRewrite');
assert.ok(query);
for(const id of ['a','b']) {
 assert.ok(query.rank<flow.nodes.find(n=>n.id===id).rank);
 assert.ok(flow.edges.some(e=>e.from===query.id&&e.to===id&&e.kind==='sequence'));
}
const pane=new Element('aside');
renderExecutionInspector(pane,{kind:'node',id:query.id,value:query},translator('en'),{view});
assert.ok(words(pane).includes(view.summary.query));
assert.equal(words(pane).includes('method A original paper'),recorded);
if(!recorded) {assert.equal(query.status,'unknown');assert.equal(query.execution.output,null);}
assert.equal(JSON.stringify(view),before);
"""
    )


def test_round_lifecycle_metadata_is_not_presented_as_business_output():
    """Historical lifecycle-only records must not imply that drafts were saved."""
    run_js(
        r"""
const {buildExecutionPresentation}=await load('execution-model.js');
const {buildBusinessFlow}=await load('execution-flow.js');
const {renderExecutionInspector}=await load('execution-inspector.js');
view.graph={tasks:{write:{task_id:'write',agent_type:'adversarial_review'}},dependencies:{write:[]},nodes:{
 'worker:write':{kind:'worker',task_id:'write',output:{total_rounds:2,final_draft:'FINAL ONLY'}},
 'subspan:s1':{kind:'subspan',task_id:'s1',name:'synthesis.r1',status:'completed',input:{parent_task_id:'write',name:'synthesis.r1',round:1},output:{error:null,error_type:null}}
}};
const flow=buildBusinessFlow(buildExecutionPresentation(view),view.graph.nodes);
assert.equal(flow.nodes.filter(n=>n.role).length,4);
const role=flow.nodes.find(n=>n.role==='synthesis'&&n.round===1);
const pane=new Element('aside');
renderExecutionInspector(pane,{kind:'node',id:role.id,value:role},translator('en'),{view});
const io=pane.children.filter(e=>e.tag==='details').slice(0,2);
assert.ok(io.every(e=>words(e).includes('Not recorded')));
assert.ok(io.every(e=>!words(e).includes('FINAL ONLY')));
"""
    )


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_business_round_content_toggles_exact_safe_diff_and_collapsed_raw(locale):
    """Inspect a real-shaped phase payload, compare drafts, then open evidence."""
    run_js(
        "const t = translator("
        + json.dumps(locale)
        + ");"
        + r"""
const {buildExecutionPresentation}=await load('execution-model.js');
const {buildBusinessFlow}=await load('execution-flow.js');
const {renderExecutionInspector}=await load('execution-inspector.js');
view.graph={tasks:{write:{task_id:'write',agent_type:'adversarial_review'}},dependencies:{write:[]},nodes:{
 'worker:write':{kind:'worker',task_id:'write',output:{total_rounds:2}},
 'subspan:s2':{kind:'subspan',task_id:'s2',name:'synthesis.r2',status:'completed',input:{parent_task_id:'write',round:2,input:{query:'Actual question',previous_draft:'Common\nOld claim\n',review:{score:0,accepted:false}}},output:{draft:'Common\nNew <script>claim</script>\n',promoted:true,evidence_selection:{selected_items:{'card:1':item}}}}
}};
const before=JSON.stringify(view),flow=buildBusinessFlow(buildExecutionPresentation(view),view.graph.nodes);
const node=flow.nodes.find(n=>n.role==='synthesis'&&n.round===2),pane=new Element('aside');
const selection={kind:'node',id:node.id,value:node};let evidence;
renderExecutionInspector(pane,selection,t,{view,onSelect:s=>evidence=s});
const business=pane.children.filter(e=>e.className==='execution-business');
assert.ok(words(business[0]).includes('Actual question'));assert.ok(words(business[0]).includes('0'));assert.ok(words(business[0]).includes(t('ioNo')));
assert.ok(words(business[1]).includes('New <script>claim</script>'));
assert.ok(pane.children.filter(e=>e.tag==='details').every(e=>!e.open));
const compare=walk(pane).find(e=>e.tag==='button'&&e.textContent===t('ioDiff'));
assert.equal(compare.disabled,false);compare.events.click();
assert.equal(selection.diffMode,true);
assert.equal(walk(pane).find(e=>e.tag==='del').textContent,'Old claim\n');
assert.equal(walk(pane).find(e=>e.tag==='ins').textContent,'New <script>claim</script>\n');
walk(pane).find(e=>e.tag==='button'&&e.textContent==='card:1').events.click();
assert.equal(evidence.kind,'evidence');assert.equal(evidence.value.supporting_text,item.supporting_text);
walk(pane).find(e=>e.tag==='button'&&e.textContent===t('ioFullDraft')).events.click();
assert.equal(selection.diffMode,false);assert.equal(JSON.stringify(view),before);
"""
    )


def test_diff_preserves_both_versions_including_large_and_empty_documents():
    """Reconstruct both exact documents even when bounded diff uses fallback."""
    run_js(
        r"""
const {diffLines}=await load('execution-content.js');
for(const [before,after] of [['','first\n'],['last',''],['a\r\nb\n','a\r\nc'],['same','same'],[Array.from({length:1100},(_,i)=>'old'+i+'\n').join(''),Array.from({length:1100},(_,i)=>'new'+i+'\n').join('')]]) {
 const diff=diffLines(before,after);
 assert.equal(diff.filter(c=>c.kind!=='added').map(c=>c.text).join(''),before);
 assert.equal(diff.filter(c=>c.kind!=='removed').map(c=>c.text).join(''),after);
}
"""
    )


def test_missing_previous_round_disables_diff_without_borrowing_another_parent():
    """A similarly named task cannot provide this round's comparison baseline."""
    run_js(
        r"""
const {renderExecutionInspector}=await load('execution-inspector.js');
view.graph={tasks:{},dependencies:{},nodes:{
 s2:{kind:'subspan',task_id:'s2',name:'synthesis.r2',status:'completed',input:{parent_task_id:'ours',round:2},output:{draft:'Our second draft'}},
 s1:{kind:'subspan',task_id:'s1',name:'synthesis.r1',status:'completed',input:{parent_task_id:'theirs',round:1},output:{draft:'Foreign first draft'}}
}};
const pane=new Element('aside'),t=translator('en');
renderExecutionInspector(pane,{kind:'observation',id:'s2',value:view.graph.nodes.s2},t,{view});
assert.equal(walk(pane).find(e=>e.tag==='button'&&e.textContent===t('ioDiff')).disabled,true);
assert.ok(!words(pane).includes('Foreign first draft'));
"""
    )
