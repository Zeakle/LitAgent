"""Exercise automatic-audit UI actions with the real client and offline fetch."""

import shutil
import subprocess
from pathlib import Path


def test_new_survey_submits_default_jev_and_operation_token():
    """Run the app module against mock DOM and fetch without creating a Survey."""
    script = r"""
const fs = require('node:fs'); const vm = require('node:vm'); const path = require('node:path');
(async () => {
 class Element {
  constructor() { this.events={}; this.children=[]; this.dataset={}; this.options=[]; this.value=''; this.hidden=false; this.checked=false; this.disabled=false; this.textContent=''; this.classList={add(){},remove(){}}; }
  addEventListener(name, fn) { this.events[name]=fn; } replaceChildren(...children) { this.children=children; }
  append(...children) { this.children.push(...children); } setAttribute(){} querySelectorAll(){return [];} close() { this.open=false; } showModal(){this.open=true;} add(option){this.options.push(option);} focus(){focused=this;}
 }
 const elements=new Map(); const $=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 $('#quality-policy').value='strict_auto_v2';
 $('#query-input').value='offline fixture'; $('#report-language').value='en'; $('#review-structure').value='academic_v1'; $('#new-citation-style').value='ieee'; $('#human-review').checked=true;
 const documentEvents={}; let focused=null; let activeDialog=null;
 const calls=[]; const caps={judge_available:false,jev_available:true};
 const automatic_acceptance={default_generation_strategy:'body_summary_v3',default_budget_policy:'planned-budget-v7',maximum_survey_tokens:10000000,default_verification:{version:'optional-verification-v2',judge:false,jev:true}};
 const context={mode:'live',live_enabled:true,operation_token:'offline-operation-token',auto_audit_capabilities:caps,automatic_acceptance};
 const sandbox=vm.createContext({
   document:{documentElement:{lang:''},addEventListener:(name,fn)=>{documentEvents[name]=fn;},querySelector:selector=>selector==='dialog[open]'?activeDialog:$(selector),querySelectorAll:()=>[],createElement:()=>new Element(),getElementById:()=>null}, localStorage:{getItem:()=>null,setItem(){}},
   location:{search:'',href:''},history:{replaceState(){}},URLSearchParams,AbortController,Option:function(label,value){this.textContent=label;this.value=String(value);},
   console,setTimeout,clearTimeout,fetch:async(url,options)=>{calls.push({url,options});return{ok:true,json:async()=>url==='/workbench/context'?context:url==='/survey'?{task_id:'offline-run'}:[]};},
 });
 let state={locale:'zh-CN',runs:[],view:null,draftReviewState:{},activeSection:'report'};
 const mocks={createWorkbenchStore:()=>({get:()=>state,subscribe(){},update:patch=>{state={...state,...patch};}}),createExecutionWorkspace:()=>({render(){},bind(){}}),
   evidenceMap:()=>new Map(),renderEvidenceList(){},renderExecutionInspector(){},renderHistory(){},renderQuality(){},locateReportDiagnostic:()=>false,renderReview(){},renderOutline(){},renderSearchRecords(){},renderReportDocument:()=>({headings:[]}),navigateReportHeading:()=>false,
   renderBenchmarkDetail(){},renderBenchmarkList(){},renderBenchmarkObservation(){},adaptArchiveView(){},adaptInteractiveView(){},chooseDefaultSection:()=> 'report',normalizeRunList:()=>[],bindFormatControls(){} };
 const real=new Set(['app.js','audit-controls.js','api-client.js','i18n.js','benchmark-controller.js']); const cache=new Map();
 async function load(filename) {
   if(cache.has(filename))return cache.get(filename);
   const module=real.has(filename)?new vm.SourceTextModule(fs.readFileSync(path.join(process.argv[1],filename),'utf8'),{context:sandbox,identifier:filename}):new vm.SyntheticModule(Object.keys(mocks),function(){for(const [name,value]of Object.entries(mocks))this.setExport(name,value);},{context:sandbox,identifier:filename});
   cache.set(filename,module); await module.link(specifier=>load(specifier.split('?')[0].replace('./',''))); return module;
 }
 await (await load('app.js')).evaluate(); await new Promise(resolve=>setImmediate(resolve));
 if(sandbox.document.documentElement.lang!=='zh-CN')throw Error('initial document language must match UI');
 if(!$('#verification-jev').checked)throw Error('default Jev must be on');
 $('#sidebar-toggle').events.click(); activeDialog={};
 documentEvents.keydown({key:'Escape',preventDefault(){throw Error('dialog Escape intercepted');}});
 if($('#run-sidebar').dataset.open!=='true')throw Error('dialog dismissal closed sidebar');
 activeDialog=null;let prevented=false;documentEvents.keydown({key:'Escape',preventDefault(){prevented=true;}});
 if(!prevented||$('#run-sidebar').dataset.open!=='false'||!$('#sidebar-scrim').hidden||focused!==$('#sidebar-toggle'))throw Error('Escape must close sidebar and restore focus');
 $('#new-run-dialog').showModal();$('#dismiss-new-run').events.click();
 if($('#new-run-dialog').open||calls.some(call=>call.url==='/survey'))throw Error('closing new Survey must not submit a paid run');
 async function submit(){await $('#new-run-form').events.submit({preventDefault(){}});return calls.filter(call=>call.url==='/survey').at(-1);}
 let request=await submit(); let body=JSON.parse(request.options.body);
 if(body.verification.jev!==true||'judge' in body.verification||'auto_audit_mode' in body)throw Error('incorrect Jev-only payload');
 if(body.generation_strategy!=='body_summary_v3'||body.budget_policy!=='planned-budget-v7')throw Error('native planned writer lost');
 if(request.options.headers['X-LitAgent-Operation-Token']!=='offline-operation-token')throw Error('missing mutation token');
 caps.jev_available=false;caps.jev_reason='missing_key:TYPESAFE_API_KEY';$('#quality-policy').events.change();
 const before=calls.filter(call=>call.url==='/survey').length;
 await submit();
 if(calls.filter(call=>call.url==='/survey').length!==before)throw Error('enabled Jev without key reached server');
 $('#verification-jev').checked=false;$('#verification-jev').events.change();
 body=JSON.parse((await submit()).options.body);
 if(body.verification.jev!==false||'judge' in body.verification)throw Error('explicit Jev opt-out lost');
 if(calls.some(call=>call.url.includes('/audits')&&call.options?.method==='POST'))throw Error('Survey preferences must not call report audits');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    _run_node(script, "src/litagent/static/flow-demo", modules=True)


def test_automatic_status_is_version_bound_and_does_not_change_quality():
    """Show budget skips and call failures without changing the report verdict."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {constructor(){this.events={};this.children=[];this.textContent='';}append(...children){this.children.push(...children);}replaceChildren(...children){this.children=children;}addEventListener(name,fn){this.events[name]=fn;}setAttribute(){}close(){}querySelectorAll(){return [];}}
 const elements=new Map();global.document={querySelector:id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement:()=>new Element()};
 let view={identity:{runId:'offline',source:'interactive'},displayRevision:1,report:{survey:'immutable [E:1]'},quality:{status:'passed'},delivery:{status:'awaiting_human'}};
 let subscriber;let locale='zh-CN';const store={get:()=>({view,locale}),subscribe:fn=>subscriber=fn};let response={capabilities:{judge_available:true},audits:[],automatic:{source_report_revision:1,mode:'judge',max_versions:1,status:'failed',reason_codes:['automatic_submit_failed:RuntimeError']}};
 const api={json:async()=>response};const {bindAuditControls}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
 const controls=bindAuditControls({store,api,setNotice(){},onEvidence(){}});await new Promise(resolve=>setImmediate(resolve));
 const text=()=>{const walk=node=>node.textContent+' '+node.children.map(walk).join(' ');return walk(document.querySelector('#audit-results'));};
 if(!text().includes('不是报告质量判定')||!text().includes('提交核验任务失败'))throw Error('call failure must be distinguished from quality');
 if(view.quality.status!=='passed'||view.delivery.status!=='awaiting_human')throw Error('automatic audit changed quality/approval');
 view={...view,displayRevision:2};response={capabilities:{judge_available:true},audits:[],automatic:{source_report_revision:2,mode:'judge',max_versions:1,status:'skipped',reason_codes:['automatic_version_budget_exhausted']}};subscriber();await new Promise(resolve=>setImmediate(resolve));
 if(!text().includes('预算已用尽')||text().includes('RuntimeError')||!text().includes('r2'))throw Error('version switch reused stale automatic result');
 locale='en';subscriber();if(!text().includes('budget is exhausted')||!text().includes('not submitted'))throw Error('English automatic status missing');
 response={capabilities:{judge_available:true},audits:[],automatic:null};await controls.refresh();if(text().includes('Automatic audit after'))throw Error('historical report invented an automatic submission');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    _run_node(script, "src/litagent/static/flow-demo/audit-controls.js")


def test_extraction_labels_have_distinct_chinese_and_english_translations():
    """Validate the real dictionaries used by the new Survey controls."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 const {translator}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
 const keys=['optionalVerification','optionalJev','optionalVerificationHelp','autoAuditJevUnavailable','autoAuditUnavailableSelection'];
 for(const locale of ['zh-CN','en'])for(const key of keys)if(translator(locale)(key)===key)throw Error(`missing ${locale}:${key}`);
 if(!translator('zh-CN')('optionalVerificationHelp').includes('默认开启')||!translator('en')('optionalVerificationHelp').includes('defaults to on'))throw Error('wrong locale dictionary');
 if(translator('zh-CN')('cancelDialog')!=='取消'||translator('en')('cancelDialog')!=='Cancel')throw Error('dialog dismissal is not localized');
 const markup=fs.readFileSync(require('node:path').join(require('node:path').dirname(process.argv[1]),'index.html'),'utf8');
 if(!markup.includes('id="close-new-run" data-i18n="cancelDialog"'))throw Error('dialog dismissal translation is not wired');
 if(/id="(verification-judge|auto-audit-mode|audit-start|audit-dialog)"/.test(markup))throw Error('a report Judge control remains exposed');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    _run_node(script, "src/litagent/static/flow-demo/i18n.js")


def _run_node(script: str, target: str, *, modules: bool = False) -> None:
    """Run a bounded local JavaScript fixture and include diagnostics on failure."""
    args = [shutil.which("node") or "node"]
    if modules:
        args.append("--experimental-vm-modules")
    args.extend(["-e", script, str(Path(target).resolve())])
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_native_ragas_default_uses_server_v7_capacity_and_budget():
    """Keep the new workbench request aligned with its server-owned RAGAS plan."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.value='';this.checked=false;this.options=[];}addEventListener(){}}
 const elements=new Map(),$=id=>{if(!elements.has(id))elements.set(id,new E());return elements.get(id)};
 global.document={querySelector:$};
 $('#quality-policy').value='strict_auto_v2';
 const caps={judge_available:true,jev_available:true,default_generation_strategy:'body_summary_v3',default_budget_policy:'planned-budget-v7',maximum_survey_tokens:10000000,default_verification:{version:'optional-verification-v2',judge:false,jev:true}};
 const {bindSurveyVerification}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
 const prefs=bindSurveyVerification({capabilities:()=>caps,t:x=>x});
 const options=prefs.options();
 if(options.generation_strategy!=='body_summary_v3'||options.budget_policy!=='planned-budget-v7'||options.capacity_policy!=='adaptive-capacity-v1')throw Error('native writer defaults lost');
 if(!$('#auto-audit-budget-note').textContent.includes('10,000,000'))throw Error('server limit not displayed');
 if(options.semantic_engine||options.endpoint||'judge' in options.verification)throw Error('client selected an internal engine, endpoint or Judge');
 $('#quality-policy').value='legacy'; prefs.render(); if(Object.keys(prefs.options()).length)throw Error('legacy request changed');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    _run_node(script, "src/litagent/static/flow-demo/audit-controls.js")
