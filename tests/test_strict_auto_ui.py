"""Exercise automatic Benchmark drill-down with real browser module contracts."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest


def test_new_workbench_contract_modules_share_the_bootstrap_cache_version():
    """A previously opened workbench must load new quality-suite consumers."""
    root = Path("src/litagent/static/flow-demo")
    version = re.search(r'app\.js\?v=([^"\s]+)', (root / "index.html").read_text())[1]
    script = (root / "app.js").read_text()
    for module in (
        "audit-controls.js",
        "adapters.js",
        "benchmark-view.js",
        "quality-view.js",
        "i18n.js",
    ):
        assert f'{module}?v={version}"' in script


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_automatic_benchmark_has_bound_detail_and_incomplete_usage_note(locale):
    """Keep blocked and unrun cases visible without inventing audit scores."""
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is unavailable for browser-module execution")
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {
  constructor(tag){this.tag=tag;this.children=[];this.events={};this.attrs={};this.textContent='';this.namespaceURI='http://www.w3.org/2000/svg';}
  replaceChildren(...x){this.children=x;} append(...x){this.children.push(...x);}
  addEventListener(k,fn){this.events[k]=fn;} setAttribute(k,v){this.attrs[k]=v;} getAttribute(k){return this.attrs[k];}
 }
 global.document={createElement:t=>new Element(t),createElementNS:(_,t)=>new Element(t)};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {renderBenchmarkDetail,renderBenchmarkObservation}=await load('benchmark-view.js');
 const {translator}=await load('i18n.js');const t=translator(process.argv[2]);
 const observation={phase:'pilot',case_id:'first',domain:'few_shot',run_id:'run-first',status:'blocked',artifact_path:'private',report_sha256:'bound-sha',acceptance:{publishable:false},gates:{evaluation:'failed',judge:'failed',jev:'failed'},costs:{survey:{tokens:123,requests:2,usage_complete:false},judge:{tokens:0,requests:0,usage_complete:true},jev:{tokens:0,requests:0,usage_complete:true}},failure_codes:['quality_usage_unknown']};
 const result={benchmark_type:'survey_quality',run_id:'suite',status:'stopped',planned:[{phase:'pilot',domain:'few_shot',case_id:'first'},{phase:'formal',domain:'vit',case_id:'not-run'}],observations:[observation],counts:{pilot:{planned:3,run:1,complete_verification:0,automatic_passed:0,blocked:1,not_run:2},formal:{planned:15,run:0,complete_verification:0,automatic_passed:0,blocked:0,not_run:15},goal_met:false},stop_reasons:['quality_usage_unknown']};
 const before=JSON.stringify(result);const target=new Element('div');let selected;
 renderBenchmarkDetail(target,result,id=>selected=id,t);
 const walk=e=>[e,...e.children.flatMap(walk)];const elements=walk(target);
 const pane=elements.find(e=>e.id==='benchmark-observation-detail');if(!pane)throw Error('missing detail target');
 const click=elements.find(e=>e.tag==='button'&&!e.disabled);click.events.click();if(selected!=='run-first')throw Error('wrong run');
 if(!elements.some(e=>e.textContent===t('incompleteUsageNote')))throw Error('false precision warning absent');
 if(!elements.some(e=>e.tag==='button'&&e.disabled))throw Error('unrun case hidden');
 renderBenchmarkObservation(pane,{observation,artifact:{report:{survey:'Retained blocked draft',evaluation:{}}}},t);
 const words=walk(pane).map(e=>e.textContent).join(' ');
 if(!words.includes('bound-sha')||!words.includes('Retained blocked draft')||!words.includes('quality_usage_unknown'))throw Error('missing source binding or failure');
 if(!walk(pane).some(e=>e.tag==='a'&&e.href.includes('archive%3Arun-first')&&e.href.includes('view=quality')))throw Error('wrong quality navigation');
 if(JSON.stringify(result)!==before)throw Error('display changed immutable result');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            node,
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
