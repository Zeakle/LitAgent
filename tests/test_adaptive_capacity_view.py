"""Keep adaptive allocation counts separate from scientific verdicts and costs."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("policy", ["planned-budget-v6", "planned-budget-v7"])
@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("status", ["allocated", "needs_refinement"])
def test_capacity_view_is_read_only_and_does_not_promote_unverified_science(
    locale, status, policy
):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const capacity={status:'allocated',initial_assertions:128,effective_assertions:144,actual_assertions:144,actual_batches:9,authored_unit_counts:{body:54,summary:4},reserved_with_margin:4110000,fixed_allowance:4430000,known_tokens_at_allocation:119169,future_locked_tokens:3609820,settled_reservation_savings:591821};
 if(process.argv[3]==='needs_refinement'){capacity.status='needs_refinement';delete capacity.initial_assertions;delete capacity.effective_assertions;}
 if(process.argv[4]==='planned-budget-v7'){capacity.round={number:1,mode:'refine',status:'superseded_unverified'};capacity.target_assertions=176;capacity.target_batches=11;capacity.assessment_reserved_tokens=820816;}
 const writer_envelope={model_assertions:144,capacity:{initial_assertions:128}};
 const evaluation={faithfulness:{score:null,skipped:true,details:{evaluation_version:'survey-evaluation-v3',unverified_count:136,stages:{capacity,writer_envelope}}}};
 const before=JSON.stringify(evaluation),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,evaluation,{status:'unverified'},{status:'blocked'},()=>{},x=>x,{locale:process.argv[2],budgetPlan:{contract:process.argv[4],status:'admitted',repair_limit:2,envelope:writer_envelope}});
 const text=walk(root).map(e=>e.textContent).join(' ');
 for(const expected of ['54 / 4','128 / 144 / 144 / 9','4110000 / 4430000','119169 / 3609820 / 591821'])if(!text.includes(expected))throw Error(expected);
 if(process.argv[4]==='planned-budget-v7'&&(!text.includes('superseded_unverified')||!text.includes('176 / 11')||!text.includes('820816')))throw Error('missing round evidence');
 if(JSON.stringify(evaluation)!==before)throw Error('source result mutated');
 if(!text.includes(process.argv[2]==='en'?'Every retained scientific assertion requires assessment':'最终保留的每项科学断言均须核验'))throw Error('missing unverified explanation');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            status,
            policy,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
