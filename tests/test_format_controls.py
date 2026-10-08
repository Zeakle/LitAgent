"""Exercise browser format controls without a DOM server or model calls."""

import shutil
import subprocess
from pathlib import Path


def test_version_style_races_and_downloads_use_selected_revision():
    """Ignore stale style responses and bind downloads to the visible revision."""
    module = Path("src/litagent/static/flow-demo/format-controls.js").resolve()
    script = r"""
const fs = require('node:fs');
(async () => {
  const elements = new Map();
  for (const id of ['#version-select','#export-style']) elements.set(id, {value: id === '#export-style' ? 'ieee' : '2', events: {}, addEventListener(k, fn) {this.events[k] = fn;}});
  const download = {dataset:{format:'md'}, addEventListener(k,fn){this.click=fn;}};
  global.document = {querySelector:id=>elements.get(id), querySelectorAll:q=>q==='[data-format]'?[download]:[],createElement:()=>({click(){}})};
  let epoch=0;
  let view={identity:{runId:'run',source:'interactive'},displayRevision:2,report:{sha:'v2'},quality:{passed:true},renderedReport:{html:'v2'}};
  const store={get:()=>({view}),beginRequest:()=>++epoch,isCurrent:n=>n===epoch,update:next=>{if(next.view)view=next.view;}};
  let release; const calls=[]; const notices=[]; const preferences=new Map();
  const api={json:async path=>{calls.push(path);if(path.includes('citation_style=apa7'))return new Promise(resolve=>release=resolve);return {report:{sha:'v1',quality:{passed:true}},rendered_report:{html:path}};},request:async path=>{calls.push(path);throw Error('engine unavailable');}};
  const {bindFormatControls}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
  bindFormatControls({store,api,t:x=>x,setNotice:(...x)=>notices.push(x),citationPreferences:preferences});
  const stale=elements.get('#export-style').events.change({target:{value:'apa7'}});
  const versionLoad=elements.get('#version-select').events.change({target:{value:'1'}});
  if(view.report!==null||Object.keys(view.runtimeEvaluation).length!==0||view.quality!==null)throw Error('version change did not clear old quality immediately');
  await versionLoad;
  release({rendered_report:{html:'stale'}});await stale;
  if(view.displayRevision!==1||view.report.sha!=='v1'||view.renderedReport.html==='stale')throw Error('stale result overwrote version');
  await elements.get('#export-style').events.change({target:{value:'ieee'}});
  if(view.report.sha!=='v1'||!view.quality.passed)throw Error('presentation changed approval/content');
  await download.click();
  if(!calls.at(-1).includes('revision=1&format=md&citation_style=ieee'))throw Error('download revision mismatch');
  if(!notices.some(x=>String(x[0]).includes('engine unavailable')))throw Error('missing error message');
  if(calls.some(x=>!x.startsWith('/survey/run/')))throw Error('unexpected model/run request');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [shutil.which("node") or "node", "-e", script, str(module)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
