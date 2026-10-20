"""Run actual workbench consumers with controlled out-of-order read responses."""

import shutil
import subprocess
from pathlib import Path


def test_benchmark_reads_discard_stale_selection_and_normal_cancellation():
    """Preserve the last user choice and report only current genuine read failures."""
    script = r"""
const fs=require('node:fs'),assert=require('node:assert/strict');
const source=fs.readFileSync(process.argv[1],'utf8');
assert(source.includes('createBenchmarkController({'));
assert(source.includes('benchmark-controller.js?v=15.10.6.124-evaluation-execution'));
const requests=[],notices=[],observations=[];
const state={results:[],selectedId:null,detail:null,observation:null,loaded:false,loadSequence:0,selectionSequence:0,observationSequence:0};
const context={benchmarkState:state,api:{json:path=>new Promise((resolve,reject)=>requests.push({path,resolve,reject}))},t:x=>x,setNotice:x=>notices.push(x),renderBenchmarkState:()=>{},renderBenchmarkObservation:(_,p)=>observations.push(p),$:()=>({replaceChildren(){}})};

const abort=Object.assign(new Error('cancelled'),{name:'AbortError'});
(async()=>{
 const {createBenchmarkController}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1].replace('app.js','benchmark-controller.js')).toString('base64'));
 const controller=createBenchmarkController({api:context.api,state,render:context.renderBenchmarkState,renderObservation:p=>observations.push(p),clearObservation:()=>{},onError:e=>notices.push(e.message)});
 context.loadBenchmarks=controller.load;context.selectBenchmark=controller.select;context.selectBenchmarkObservation=controller.selectObservation;
 const index=context.loadBenchmarks();
 const manual=context.selectBenchmark('manual');
 requests[1].resolve({run_id:'manual'});await manual;
 requests[0].resolve([{benchmark_id:'default'}]);await index;
 assert.equal(requests.length,2);assert.equal(state.selectedId,'manual');assert.equal(state.detail.run_id,'manual');
 const stale=context.selectBenchmark('old'),latest=context.selectBenchmark('new');
 assert.equal(state.detail,null);assert.equal(state.observation,null);
 requests[3].resolve({run_id:'new'});await latest;requests[2].resolve({run_id:'old'});await stale;
 assert.equal(state.detail.run_id,'new');assert.equal(state.selectedId,'new');
 const oldObservation=context.selectBenchmarkObservation('old-observation');
 const next=context.selectBenchmark('next');requests[5].resolve({run_id:'next'});await next;
 requests[4].resolve({observation:{run_id:'old-observation'}});await oldObservation;
 assert.equal(state.observation,null);assert.equal(observations.length,0);
 const first=context.selectBenchmarkObservation('first'),second=context.selectBenchmarkObservation('second');
 requests[7].resolve({observation:{run_id:'second'}});await second;
 requests[6].resolve({observation:{run_id:'first'}});await first;
 assert.equal(state.observation.observation.run_id,'second');assert.equal(observations.length,1);
 for(const action of [()=>context.loadBenchmarks(),()=>context.selectBenchmark('cancelled'),()=>context.selectBenchmarkObservation('cancelled')]){
  const promise=action();requests.at(-1).reject(abort);await promise;
 }
 assert.equal(notices.length,0);
 const oldError=context.selectBenchmark('error-old'),newChoice=context.selectBenchmark('survivor');
 requests.at(-1).resolve({run_id:'survivor'});await newChoice;
 requests.at(-2).reject(new Error('stale error'));await oldError;assert.equal(notices.length,0);
 const current=context.selectBenchmarkObservation('current');requests.at(-1).reject(new Error('integrity mismatch'));await current;
 assert.equal(notices.length,1);assert.match(notices[0],/integrity mismatch/);
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo/app.js")),
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
