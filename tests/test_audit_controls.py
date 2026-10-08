"""Verify audit UI authentication and version races without browser model calls."""

import shutil
import subprocess
from pathlib import Path


def test_audit_controls_bind_sha_mutation_token_and_ignore_stale_versions():
    """Exercise the actual module with controlled DOM and asynchronous API replies."""
    module = Path("src/litagent/static/flow-demo/audit-controls.js").resolve()
    script = r"""
const fs=require('node:fs');
(async()=>{
  class Element {
    constructor(tag='div') {this.tag=tag;this.events={};this.children=[];this.textContent='';this.disabled=false;this.checked=false;this.hidden=false;this.value='';}
    append(...children){this.children.push(...children);} replaceChildren(...children){this.children=children;}
    addEventListener(key,fn){this.events[key]=fn;} setAttribute(){} close(){this.open=false;} showModal(){this.open=true;}
    querySelectorAll(){return [];} scrollIntoView(){} cloneNode(){return this;}
  }
  const elements=new Map();
  global.document={querySelector(id){if(!elements.has(id))elements.set(id,new Element());return elements.get(id);},createElement:tag=>new Element(tag)};
  const $=id=>document.querySelector(id);
  let view={identity:{runId:'run',source:'interactive'},displayRevision:1,report:{survey:'raw [E:ev-1]'},quality:{status:'passed'}};
  let subscriber; const store={get:()=>({view,locale:'zh-CN'}),subscribe:fn=>subscriber=fn};
  let release;const calls=[];const notices=[];
  const caps={judge_available:true,judge_model:'fixture',self_judged:true,jev_available:false,jev_reason:'missing_key:TYPESAFE_API_KEY',jev_model:'jev-1.13.0'};
  const result={audit_id:'audit-1',source_report_revision:2,source_report_sha256:'sha2',status:'running',judge_model:'fixture',method_status:'unverifiable',structure:{status:'not_applicable'},jev:{status:'not_requested'},reason_codes:[]};
  const api={json:async(path,options)=>{
    calls.push({path,options});
    if(options?.method==='POST'){
      if(options.mutate!==true)throw Error('operation token omitted');
      if(path.endsWith('/cancel'))return {...result,status:'cancelled'};
      if(options.body.report_sha256!=='sha2'||options.body.include_jev!==false)throw Error('wrong audit target/default');
      return result;
    }
    if(path.includes('/versions/1/'))return new Promise(resolve=>release=resolve);
    return {capabilities:caps,source_report_sha256:'sha2',audits:[]};
  }};
  const {bindAuditControls}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
  bindAuditControls({store,api,setNotice:(...args)=>notices.push(args),onEvidence(){}});
  view={...view,displayRevision:2};subscriber(); await new Promise(resolve=>setImmediate(resolve));
  release({capabilities:caps,source_report_sha256:'sha1',audits:[{...result,source_report_revision:1}]});await new Promise(resolve=>setImmediate(resolve));
  $('#audit-start').events.click();
  if(!$('#audit-dialog-target').textContent.includes('sha2'))throw Error('stale response replaced target');
  if($('#audit-jev').checked||!$('#audit-jev').disabled)throw Error('Jev default/key gate broken');
  await $('#audit-submit').events.click();
  if(!calls.find(call=>call.options?.body?.report_sha256==='sha2'&&call.path.includes('/versions/2/')))throw Error('version binding broken');
  if(view.report.survey!=='raw [E:ev-1]'||view.quality.status!=='passed')throw Error('audit mutated report/quality');
  $('#audit-start').events.click();view={...view,displayRevision:3};subscriber();await new Promise(resolve=>setImmediate(resolve));
  await $('#audit-submit').events.click();
  if(calls.filter(call=>call.options?.method==='POST').length!==1)throw Error('stale modal submitted new version');
  if(calls.some(call=>call.path==='/survey'))throw Error('unexpected Survey call');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run(
        [shutil.which("node") or "node", "-e", script, str(module)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_new_format_labels_exist_in_each_locale():
    module = Path("src/litagent/static/flow-demo/i18n.js").resolve()
    script = r"""
const fs=require('node:fs');
(async()=>{
 const {translator}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
 for(const locale of ['zh-CN','en'])for(const key of ['reviewStructure','academicStructure','methodStructure','reportFiles','bibliographyFiles','gbtAuthorDate','formatHelp']){
  if(translator(locale)(key)===key)throw Error(`missing translation ${locale}:${key}`);
 }
 if(translator('zh-CN')('reportFiles')!=='报告文件')throw Error('wrong Chinese dictionary');
 if(translator('en')('reportFiles')!=='Report files')throw Error('wrong English dictionary');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run(
        [shutil.which("node") or "node", "-e", script, str(module)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_shared_quality_audit_snapshot_discards_late_version_and_reuses_selection():
    """Publish the existing controller's state without creating another polling loop."""
    import shutil
    import subprocess
    from pathlib import Path

    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element{constructor(){this.events={};this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}setAttribute(){}addEventListener(k,f){this.events[k]=f;}close(){}querySelectorAll(){return[];}}
 const elements=new Map();const $=s=>{if(!elements.has(s))elements.set(s,new Element());return elements.get(s);};global.document={querySelector:$,createElement:()=>new Element()};
 const {bindAuditControls}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/audit-controls.js').toString('base64'));
 let state={locale:'en',view:{identity:{runId:'run',source:'archive'},displayRevision:1,report:{survey:'body'}}};let sync;let last;const calls=[];const resolves=[];
 const store={get:()=>state,subscribe:f=>sync=f};const api={json:url=>{calls.push(url);return new Promise(resolve=>resolves.push(resolve));}};
 bindAuditControls({store,api,setNotice(){},onEvidence(){},onSnapshot:s=>last=s});
 if(calls.length!==1)throw Error('duplicate initial load');
 state={...state,view:{...state.view,displayRevision:2,reportSha:'b'.repeat(64)}};sync();if(calls.length!==2||last.data!==null||last.target.revision!==2)throw Error('stale results not cleared');
 const result={source_report_revision:2,source_report_sha256:'b'.repeat(64),capabilities:{judge_available:true},audits:[{audit_id:'latest-failed',source_report_revision:2,source_report_sha256:'b'.repeat(64),status:'failed',judge_model:'offline'},{audit_id:'old-success',source_report_revision:2,source_report_sha256:'b'.repeat(64),status:'completed',judge_model:'offline'}]};
 resolves[1](result);await new Promise(r=>setImmediate(r));if(last.selected!=='latest-failed')throw Error('latest failure hidden');
 resolves[0]({source_report_revision:1,source_report_sha256:'a'.repeat(64),audits:[]});await new Promise(r=>setImmediate(r));if(last.data!==result||last.target.revision!==2)throw Error('late result attached to newer version');
 last.select('old-success');if(last.selected!=='old-success'||calls.length!==2)throw Error('selection should not refetch or re-run audits');
 const before=JSON.stringify(result);sync();if(JSON.stringify(result)!==before||calls.length!==2)throw Error('snapshot modified results or created duplicate polling');
 state={...state,view:{...state.view,reportSha:'c'.repeat(64)}};sync();
 resolves[2](result);await new Promise(r=>setImmediate(r));if(last.data!==null||last.error!=='audit_input_binding_mismatch')throw Error('wrong SHA was shown for current version');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
