"""Exercise automatic-audit UI actions with the real client and offline fetch."""

import shutil
import subprocess
from pathlib import Path


def test_new_survey_actions_submit_automatic_modes_budget_and_operation_token():
    """Run the app module against mock DOM and fetch without creating a Survey."""
    script = r"""
const fs = require('node:fs'); const vm = require('node:vm'); const path = require('node:path');
(async () => {
 class Element {
  constructor() { this.events={}; this.children=[]; this.dataset={}; this.options=[]; this.value=''; this.hidden=false; this.checked=false; this.disabled=false; this.textContent=''; this.classList={add(){},remove(){}}; }
  addEventListener(name, fn) { this.events[name]=fn; } replaceChildren(...children) { this.children=children; }
  append(...children) { this.children.push(...children); } setAttribute(){} querySelectorAll(){return [];} close() { this.open=false; } showModal(){this.open=true;} add(option){this.options.push(option);}
 }
 const elements=new Map(); const $=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
 $('#auto-audit-mode').options=['off','judge','judge_jev'].map(value=>({value,disabled:false})); $('#auto-audit-mode').value='off'; $('#auto-audit-max-versions').value='1';
 $('#query-input').value='offline fixture'; $('#report-language').value='en'; $('#review-structure').value='academic_v1'; $('#new-citation-style').value='ieee'; $('#human-review').checked=true;
 const calls=[]; const caps={judge_available:true,jev_available:true,judge_model:'offline',jev_model:'jev-1.13.0',limits:{judge_calls:2,judge_max_output_tokens_per_call:16384,judge_timeout_seconds:180,jev:{max_requests:20,max_input_tokens:200000}}};
 const context={mode:'live',live_enabled:true,operation_token:'offline-operation-token',auto_audit_capabilities:caps};
 const sandbox=vm.createContext({
   document:{querySelector:$,querySelectorAll:()=>[],createElement:()=>new Element(),getElementById:()=>null}, localStorage:{getItem:()=>null,setItem(){}},
   location:{search:'',href:''},history:{replaceState(){}},URLSearchParams,AbortController,Option:function(label,value){this.textContent=label;this.value=String(value);},
   console,setTimeout,clearTimeout,fetch:async(url,options)=>{calls.push({url,options});return{ok:true,json:async()=>url==='/workbench/context'?context:url==='/survey'?{task_id:'offline-run'}:[]};},
 });
 let state={locale:'zh-CN',runs:[],view:null,draftReviewState:{},activeSection:'report'};
 const mocks={createWorkbenchStore:()=>({get:()=>state,subscribe(){},update:patch=>{state={...state,...patch};}}),createExecutionWorkspace:()=>({render(){},bind(){}}),
   evidenceMap:()=>new Map(),renderEvidenceList(){},renderExecutionInspector(){},renderHistory(){},renderQuality(){},locateReportDiagnostic:()=>false,renderReview(){},renderOutline(){},renderSearchRecords(){},renderReportDocument:()=>({headings:[]}),
   renderBenchmarkDetail(){},renderBenchmarkList(){},renderBenchmarkObservation(){},adaptArchiveView(){},adaptInteractiveView(){},chooseDefaultSection:()=> 'report',normalizeRunList:()=>[],bindFormatControls(){} };
 const real=new Set(['app.js','audit-controls.js','api-client.js','i18n.js']); const cache=new Map();
 async function load(filename) {
   if(cache.has(filename))return cache.get(filename);
   const module=real.has(filename)?new vm.SourceTextModule(fs.readFileSync(path.join(process.argv[1],filename),'utf8'),{context:sandbox,identifier:filename}):new vm.SyntheticModule(Object.keys(mocks),function(){for(const [name,value]of Object.entries(mocks))this.setExport(name,value);},{context:sandbox,identifier:filename});
   cache.set(filename,module); await module.link(specifier=>load(specifier.split('?')[0].replace('./',''))); return module;
 }
 await (await load('app.js')).evaluate(); await new Promise(resolve=>setImmediate(resolve));
 if(!$('#auto-audit-max-versions').disabled)throw Error('default-off budget must be disabled');
 $('#new-run-dialog').showModal();$('#dismiss-new-run').events.click();
 if($('#new-run-dialog').open||calls.some(call=>call.url==='/survey'))throw Error('closing new Survey must not submit a paid run');
 async function submit(){await $('#new-run-form').events.submit({preventDefault(){}});return calls.filter(call=>call.url==='/survey').at(-1);}
 let request=await submit(); let body=JSON.parse(request.options.body);
 if(body.auto_audit_mode!=='off'||body.auto_audit_max_versions!==1)throw Error('default automatic authorization must be off/1');
 if(request.options.headers['X-LitAgent-Operation-Token']!=='offline-operation-token')throw Error('missing mutation token');
 $('#auto-audit-mode').value='judge'; $('#auto-audit-mode').events.change(); $('#auto-audit-max-versions').value='2'; $('#auto-audit-max-versions').events.change();
 if($('#auto-audit-max-versions').disabled||!$('#auto-audit-budget-note').textContent.includes('16384'))throw Error('Judge budget disclosure missing');
 body=JSON.parse((await submit()).options.body);if(body.auto_audit_mode!=='judge'||body.auto_audit_max_versions!==2)throw Error('Judge authorization lost');
 $('#auto-audit-mode').value='judge_jev'; $('#auto-audit-mode').events.change(); $('#auto-audit-max-versions').value='3';
 if(!$('#auto-audit-budget-note').textContent.includes('200000'))throw Error('Jev input budget disclosure missing');
 body=JSON.parse((await submit()).options.body);if(body.auto_audit_mode!=='judge_jev'||body.auto_audit_max_versions!==3)throw Error('Judge/Jev authorization lost');
 caps.jev_available=false;caps.jev_reason='missing_key:TYPESAFE_API_KEY';$('#auto-audit-mode').events.change();
 if($('#auto-audit-mode').value!=='off'||!$('#auto-audit-mode').options[2].disabled||!$('#auto-audit-capabilities').textContent.includes('TYPESAFE_API_KEY'))throw Error('missing TypeSafe key gate broken');
 body=JSON.parse((await submit()).options.body);if(body.auto_audit_max_versions!==1)throw Error('disabled budget must serialize default');
 const before=calls.filter(call=>call.url==='/survey').length;
 $('#auto-audit-mode').value='judge_jev';await submit();
 if(calls.filter(call=>call.url==='/survey').length!==before)throw Error('programmatic unsupported selection reached backend');
 caps.judge_available=false;caps.judge_reason='audit_offline_or_disabled';$('#auto-audit-mode').events.change();
 if(!$('#auto-audit-mode').options[1].disabled||$('#auto-audit-mode').options[0].disabled||$('#open-new-run').disabled)throw Error('missing Judge must preserve normal Survey creation');
 caps.judge_available=true;$('#auto-audit-mode').value='judge';$('#auto-audit-mode').events.change();$('#auto-audit-max-versions').value='4';await submit();
 if(calls.filter(call=>call.url==='/survey').length!==before)throw Error('invalid budget reached backend');
 if(calls.some(call=>call.url.includes('/audits')&&call.options?.method==='POST'))throw Error('Survey preferences must not directly call audit');
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


def test_automatic_labels_have_distinct_chinese_and_english_translations():
    """Validate the real dictionaries used by the new Survey controls."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 const {translator}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
 const keys=['autoAuditMode','autoAuditOff','autoAuditJudge','autoAuditJudgeJev','autoAuditMaxVersions','autoAuditOffHelp','autoAuditBudgetHelp','autoAuditPerVersion','autoAuditRules','autoAuditJudgeUnavailable','autoAuditJevUnavailable','autoAuditInvalidBudget'];
 for(const locale of ['zh-CN','en'])for(const key of keys)if(translator(locale)(key)===key)throw Error(`missing ${locale}:${key}`);
 if(translator('zh-CN')('autoAuditOff')!=='关闭'||translator('en')('autoAuditOff')!=='Off')throw Error('wrong locale dictionary');
 if(translator('zh-CN')('cancelDialog')!=='取消'||translator('en')('cancelDialog')!=='Cancel')throw Error('dialog dismissal is not localized');
 const markup=fs.readFileSync(require('node:path').join(require('node:path').dirname(process.argv[1]),'index.html'),'utf8');
 if(!markup.includes('id="close-new-run" data-i18n="cancelDialog"'))throw Error('dialog dismissal translation is not wired');
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
