"""Execute real selection consumers without making archive access depend on history."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "scenario",
    ["direct", "missing", "identity", "late_selection", "history_error", "default"],
)
def test_direct_archive_bootstrap_and_late_history_preserve_exact_selection(scenario):
    """Only HTTP and DOM boundaries are replaced; actual store and adapters execute."""
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const root=process.argv[1],scenario=process.argv[2],app=fs.readFileSync(path.join(root,'app.js'),'utf8');
const moduleFrom=name=>import('data:text/javascript;base64,'+fs.readFileSync(path.join(root,name)).toString('base64'));
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const raw=id=>({summary:{run_id:id,query:'Question '+id,status:'completed'},report:{survey:'# '+id},rendered_report:{html:'<p>'+id+'</p>',source_sha256:'sha-'+id}});
const identity=id=>({key:'archive:'+id,runId:id,source:'archive',query:id,status:'completed'});
(async()=>{
 const adapters=await moduleFrom('adapters.js'),{createWorkbenchStore}=await moduleFrom('state.js');
 const store=createWorkbenchStore(),requests=[],notices=[],urls=[],elements=new Map();
 const $=s=>{if(!elements.has(s))elements.set(s,{dataset:{},attributes:{},hidden:false,disabled:false,textContent:'',setAttribute(name,value){this.attributes[name]=value;}});return elements.get(s)};
 const api={abort(){},json(p){if(p==='/workbench/context')return Promise.resolve({mode:'offline',live_enabled:false});return new Promise((resolve,reject)=>requests.push({path:p,resolve,reject}))}};
 const ctx=vm.createContext({store,api,$,context:null,historyLoading:false,eventSource:null,mobileSidebar:null,citationPreferences:new Map(),URLSearchParams,
   location:{search:scenario==='default'?'':'?run=archive%3Atarget&view=quality'},history:{replaceState:(_,__,url)=>urls.push(url)},
   surveyVerification:{render(){}},t:x=>x,setNotice:message=>notices.push(message),
   ...adapters,setSection(){},refreshInteractive(){throw Error('unexpected interactive fallback')}});
 const start=app.indexOf('async function selectRun('),end=app.indexOf('async function submitReview(');
 assert(start>0&&end>start);
 const closeStart=app.indexOf('function closeRunSidebar('),closeEnd=app.indexOf('function setInspector(');
 assert(closeStart>0&&closeEnd>closeStart);
 const initStart=app.indexOf('async function initialize('),initEnd=app.indexOf('store.subscribe(render);');
 vm.runInContext(app.slice(closeStart,closeEnd)+'\n'+app.slice(start,end)+'\n'+app.slice(initStart,initEnd),ctx);
 assert(app.includes('$("#load-run-history").addEventListener("click", reloadHistory)'));
 $('#run-sidebar').dataset.open='true';$('#sidebar-toggle').setAttribute('aria-expanded','true');
 const assertClosed=()=>{assert.equal($('#run-sidebar').dataset.open,'false');assert.equal($('#sidebar-scrim').hidden,true);assert.equal($('#sidebar-toggle').attributes['aria-expanded'],'false');};
 const initialization=ctx.initialize();await tick();
 if(scenario==='default'){
   assert.equal(requests.length,2);assert.equal(requests[0].path,'/workbench/runs');assert.equal(requests[1].path,'/flow-demo/runs');
   requests[0].resolve([]);requests[1].resolve([{run_id:'target',query:'Question target',status:'completed'}]);await tick();
   assert.equal(requests[2].path,'/flow-demo/runs/target/view');requests[2].resolve(raw('target'));await initialization;
   assert.equal(store.get().view.reportSha,'sha-target');assertClosed();return;
 }
 assert.equal(requests.length,1);assert.equal(requests[0].path,'/flow-demo/runs/target/view');
 assert.equal(store.get().selectedRunKey,'archive:target');assert.equal(store.get().view,null);
 if(scenario==='missing'){
   requests[0].reject({status:404,message:'missing exact report'});await initialization;
   assert.equal(requests.length,1);assert.equal(store.get().view,null);assert.equal(notices.length,1);assertClosed();return;
 }
 if(scenario==='identity'){
   requests[0].resolve(raw('other'));await initialization;
   assert.equal(store.get().view,null);assert.match(notices[0],/archive_run_identity_mismatch/);assertClosed();return;
 }
 requests[0].resolve(raw('target'));await initialization;
 assert.equal(requests.length,1);assert.equal(store.get().view.reportSha,'sha-target');
 assert.equal(store.get().view.identity.query,'Question target');assert.equal(store.get().activeSection,'quality');
 assertClosed();
 if(scenario==='direct')return;
 const history=ctx.reloadHistory();await tick();
 assert.equal(requests[1].path,'/workbench/runs');assert.equal(requests[2].path,'/flow-demo/runs');
 if(scenario==='history_error'){
   requests[1].reject(new Error('history unavailable'));requests[2].resolve([]);await history;
   assert.equal(store.get().view.reportSha,'sha-target');assert.equal(store.get().selectedRunKey,'archive:target');
   assert.equal($('#load-run-history').disabled,false);assert.equal(notices.length,1);return;
 }
 store.update({runs:[...store.get().runs,identity('next'),identity('old')]});
 const old=ctx.selectRun('archive:old','report');await tick();
 const next=ctx.selectRun('archive:next','report');await tick();
 assert.equal(store.get().view,null);assert.equal(store.get().selectedRunKey,'archive:next');
 requests[4].resolve(raw('next'));await next;
 requests[3].reject(new Error('late old selection failed'));await old;
 requests[1].resolve([]);requests[2].resolve([{run_id:'target',query:'old index',status:'completed'}]);await history;
 assert.equal(store.get().view.reportSha,'sha-next');assert.equal(store.get().selectedRunKey,'archive:next');
 assert(store.get().runs.some(r=>r.key==='archive:next'));assert.match(urls.at(-1),/archive%3Anext/);
 assert.equal(notices.length,0);assert.equal(requests.length,5);
 assertClosed();
})().catch(error=>{console.error(error);process.exit(1)});
"""
    completed = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo")),
            scenario,
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr


def test_source_inspector_shows_original_support_without_html_interpretation():
    """Display raw support separately from the extracted claim, using text nodes."""
    script = r"""
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const app=fs.readFileSync(process.argv[1],'utf8');
class Node {
 constructor(){this.children=[];this.dataset={};this.attrs={};this.textContent='';}
 append(...items){this.children.push(...items)}
 replaceChildren(){this.children=[]}
 setAttribute(k,v){this.attrs[k]=v}
 focus(){}
}
const elements=new Map(),$=id=>{if(!elements.has(id))elements.set(id,new Node());return elements.get(id)};
const context=vm.createContext({$,document:{createElement:()=>new Node()},inspectorTrigger:null,t:x=>x,
 renderExecutionInspector:()=>false});
vm.runInContext(app.slice(app.indexOf('function setInspector('),app.indexOf('function summaryValue(')),context);
const support='<script>untrusted()</script> Full original with conditions and a different number: 17%.';
context.setInspector({kind:'evidence',id:'E:a',value:{evidence_id:'a',text:'Extracted claim.',supporting_text:support}});
const fields=$('#inspector-content').children[0].children;
assert(fields.some(n=>n.textContent==='supporting_text'));
assert(fields.some(n=>n.textContent===support));
assert(fields.some(n=>n.textContent==='Extracted claim.'));
assert.equal($('#inspector-drawer').attrs['aria-hidden'],'false');
assert(!fields.some(n=>Object.hasOwn(n,'innerHTML')));
"""
    completed = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            "src/litagent/static/flow-demo/app.js",
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert completed.returncode == 0, completed.stderr
