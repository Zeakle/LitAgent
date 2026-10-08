"""Keep skipped Evaluation metrics distinct from actual scores in the workbench."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_skipped_metric_is_unverified_without_overwriting_original_result(locale):
    """Render the real skipped+passed contract with actual locale dictionaries."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {constructor(){this.children=[];this.dataset={};this.events={};this.textContent='';}replaceChildren(...x){this.children=x;}append(...x){this.children.push(...x);}addEventListener(key,fn){this.events[key]=fn;}}
 global.document={createElement:()=>new Element()};
 const base=process.argv[1];const locale=process.argv[2];
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(base+'/'+name).toString('base64'));
 const {renderQuality}=await load('quality-view.js');const {translator}=await load('i18n.js');
 const evaluation={faithfulness:{score:0,passed:true,skipped:true,details:{skipped_reason:'no evidence'}},normal_zero:{score:0,passed:true,skipped:false},actual_failure:{score:0.4,passed:false}};
 const before=JSON.stringify(evaluation);const container=new Element();let inspected;
 renderQuality(container,evaluation,{status:'unverified'},{status:'needs_review'},value=>inspected=value,translator(locale));
 const rows=container.children[0].children;
 if(rows[0].dataset.status!=='unverified'||rows[0].children[1].textContent!=='—')throw Error('skipped score was displayed as valid');
 const expected=locale==='zh-CN'?'已跳过／未核验':'Skipped / unverified';
 if(rows[0].children[2].textContent!==expected)throw Error('skipped status is not localized');
 if(rows[1].dataset.status!=='passed'||rows[1].children[1].textContent!=='0.000')throw Error('valid zero score was hidden');
 if(rows[2].dataset.status!=='failed'||rows[2].children[1].textContent!=='0.400')throw Error('actual failure was relabeled');
 rows[0].events.click();if(inspected.value!==evaluation.faithfulness||JSON.stringify(evaluation)!==before)throw Error('render changed original result');
 if(container.children[1].children[0].textContent!=='unverified / needs_review')throw Error('aggregate quality was changed');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_visual_quality_counts_thresholds_diagnostics_and_independent_attempts(locale):
    """Render both scopes and navigate the actual diagnostic without mutating scores."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {constructor(tag=''){this.tag=tag;this.children=[];this.dataset={};this.events={};this.textContent='';this.attrs={};}replaceChildren(...x){this.children=x;}append(...x){this.children.push(...x);}addEventListener(key,fn){this.events[key]=fn;}setAttribute(k,v){this.attrs[k]=v;}}
 global.document={createElement:tag=>new Element(tag)};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {renderQuality}=await load('quality-view.js');const {translator}=await load('i18n.js');const locale=process.argv[2];
 const quote={claim_id:'claim_0',source_quote:'Wrong method result. [E:a]',source_start:20,source_end:49,evidence_ids:['a']};
 const issue={...quote,section:'Results',claim_text:'Wrong method result.',reason_code:'attribution_mismatch',explanation:'The number belongs to another entity.',local_error:'support_metric_unverified',support_validation:{spans:[{evidence_id:'a',source:'original_support',quote:'FLOPs reduced by 44%.'}]}};
 const evaluation={citation_accuracy:{score:1,passed:true,details:{threshold:1}},faithfulness:{score:0.9,passed:true,details:{evaluation_version:'survey-evaluation-v2',support_contract:'host-support-passages-v2',language_checks:{contract:'report-prose-language-v1',status:'failed',issues:['report_language_mismatch']},threshold:0.8,supported_count:9,unsupported_count:1,unverified_count:0,factual_claim_count:10,coverage:{source_units:8,processed_units:8},unsupported_claims:[issue],structure_checks:{status:'passed',issues:[]},method_checks:{status:'failed',issues:['incorrect_unit']}}}};
 const audit={audit_id:'latest-failed',status:'failed',source_report_revision:2,source_report_sha256:'a'.repeat(64),judge_model:'offline',self_judged:true,judge:{status:'completed',faithfulness:0.7,topic_coverage:null,inventory_snapshot:{claims:[quote]},claim_assessments:[{claim_id:'claim_0',claim_text:'Wrong method result.'}]},jev:{status:'partial',summary:{protocol_completion_rate:0.5,agreement:null,local_error_questions:1,uncertain_questions:1},decisions:[{claim_id:'claim_0',choice:'insufficient',confidence:0.6,uncertain:true}]}};
 const old={...audit,audit_id:'older-success',status:'completed'};let chosen;let inspected;
 const options={locale,revision:2,sha:'a'.repeat(64),audits:{selected:audit.audit_id,data:{audits:[audit,old]},select:id=>chosen=id}};
 const before=JSON.stringify({evaluation,audit,old});const container=new Element();
 renderQuality(container,evaluation,{status:'failed'},{status:'blocked'},x=>inspected=x,translator(locale),options);
 const walk=e=>[e,...e.children.flatMap(walk)];const all=walk(container);const words=all.map(e=>e.textContent).join(' ');
 if(!words.includes(locale==='zh-CN'?'引用 ID 有效性':'Citation ID validity'))throw Error('ID validity mislabeled');
 if(!words.includes('0.8')||!all.some(e=>e.tag==='progress'&&e.value===0.9))throw Error('threshold visualization missing');
 if(!words.includes('host-support-passages-v2')||!words.includes('FLOPs reduced by 44%.')||!words.includes('support_metric_unverified')||!words.includes(locale==='zh-CN'?'正文语言':'Report prose language'))throw Error('support contract/basis/language missing');
 if(!words.includes('8 / 8')||!words.includes('incorrect_unit'))throw Error('coverage/check missing');
 if(!words.includes('self_judged')||!words.includes(locale==='zh-CN'?'置信度不是正确率':'confidence is not accuracy'))throw Error('audit boundary missing');
 const select=all.find(e=>e.tag==='select');if(select.value!=='latest-failed')throw Error('old successful attempt hides failure');select.value='older-success';select.events.change();if(chosen!=='older-success')throw Error('shared selection unavailable');
 const button=all.find(e=>e.tag==='button'&&e.textContent===(locale==='zh-CN'?'定位原文及证据':'Locate source and evidence'));button.events.click();if(inspected.value!==issue)throw Error('wrong original diagnostic');
 if(JSON.stringify({evaluation,audit,old})!==before)throw Error('visualization changed immutable result');
 const quoted=JSON.parse(JSON.stringify(evaluation));quoted.faithfulness.details.support_contract='cited-support-spans-v1';
 renderQuality(container,quoted,{status:'failed'},{status:'blocked'},()=>{},translator(locale),{locale});
 if(!walk(container).some(e=>e.textContent.includes(locale==='zh-CN'?'前版支持原文口径':'Previous quoted-support scope')))throw Error('quoted v1 scope not distinguished');
 if(quoted.faithfulness.score!==0.9)throw Error('quoted v1 score overwritten');
 const priorV2=JSON.parse(JSON.stringify(evaluation));delete priorV2.faithfulness.details.support_contract;
 renderQuality(container,priorV2,{status:'failed'},{status:'blocked'},()=>{},translator(locale),{locale});
 if(!walk(container).some(e=>e.textContent.includes(locale==='zh-CN'?'历史 v2':'Historical v2')))throw Error('old v2 contract not distinguished');
 if(priorV2.faithfulness.score!==0.9)throw Error('old v2 score overwritten');
 renderQuality(container,{faithfulness:{score:0.62,passed:false,details:{factual_claim_count:50,unsupported_claims:[issue]}}},{status:'failed'},{status:'blocked'},()=>{},translator(locale),{locale});
 const historical=walk(container);if(!historical.some(e=>e.textContent.includes(locale==='zh-CN'?'历史评估口径':'Historical evaluation scope')))throw Error('legacy scope not labeled');
 if(historical.some(e=>e.className==='quality-threshold'))throw Error('invented legacy threshold');
 if(historical.filter(e=>e.className==='quality-count supported')[0].children[0].textContent!=='—')throw Error('legacy scientific counts fabricated');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_quality_module_cache_identity_follows_html_entrypoint():
    """Keep refreshed workbenches from importing an older cached quality module."""
    import re

    root = Path("src/litagent/static/flow-demo")
    entry = re.search(r'/app\.js\?v=([^"\s]+)', (root / "index.html").read_text())[1]
    quality = re.search(
        r'\./quality-view\.js\?v=([^"\s]+)', (root / "app.js").read_text()
    )[1]
    assert quality == entry
