"""Keep skipped Evaluation metrics distinct from actual scores in the workbench."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_writer_v2 import case  # noqa: F401


def render_repair_state(value, locale):
    """Run the actual quality renderer and expose its immutable DOM projection."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const input=JSON.parse(fs.readFileSync(0,'utf8')),before=JSON.stringify(input),root=new E();
 renderQuality(root,input.evaluation||{},input.quality||{},input.delivery||{},()=>{},x=>x,{...input.options,locale:process.argv[2]});
 const walk=e=>[e,...e.children.flatMap(walk)],nodes=walk(root);
 const heading=process.argv[2]==='en'?'Bounded targeted repairs':'有限定向修复';
 const repairs=root.children.find(e=>e.children.some(c=>c.textContent===heading));
 if(JSON.stringify(input)!==before)throw Error('Renderer mutated the producer input');
 console.log(JSON.stringify({text:nodes.map(e=>e.textContent).join(' '),repairTexts:repairs?walk(repairs).map(e=>e.textContent):[],metricStatuses:nodes.filter(e=>e.className==='metric-row').map(e=>e.dataset.status)}));
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
        ],
        input=json.dumps(value),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize(
    "purpose", ["bounded-inventory-recovery-v1", "bounded-inventory-recovery-v2"]
)
@pytest.mark.parametrize("status", ["completed", "failed", "started_no_replay"])
def test_inventory_attempt_purpose_and_status_remain_distinct_from_body_repair(
    locale, purpose, status
):
    """Label exact recovery contracts without changing counts or science gates."""
    state = {
        "version": "shared-targeted-repair-limit-v1",
        "run_id": "offline-quality-view",
        "limit": 2,
        "attempts": [
            {"index": 0, "stage": "evaluation", "purpose": purpose, "status": status}
        ],
    }
    value = {
        "evaluation": {"faithfulness": {"skipped": True, "score": None, "details": {}}},
        "quality": {"status": "unverified"},
        "delivery": {"status": "blocked", "publishable": False},
        "options": {"qualityPolicy": "strict_auto_v2", "targetedRepairState": state},
    }
    before = json.dumps(value, sort_keys=True)
    rendered = render_repair_state(value, locale)
    en = locale == "en"
    expected_status = {
        "completed": "Completed" if en else "已完成",
        "failed": "Failed" if en else "未通过",
        "started_no_replay": (
            "Sent; automatic replay forbidden" if en else "已发出，禁止自动重发"
        ),
    }[status]
    label = "Inventory re-extraction" if en else "断言清单重抽取"
    assert f"{label}: {expected_status}" in rendered["repairTexts"]
    assert ("Attempts consumed: 1 / 2" if en else "已消耗次数: 1 / 2") in rendered[
        "repairTexts"
    ]
    assert not any(
        text.startswith("Evaluation repair:" if en else "正文定向修复:")
        for text in rendered["repairTexts"]
    )
    assert ("does not establish scientific support" if en else "不代表科学支持") in (
        rendered["text"]
    )
    assert rendered["metricStatuses"] == ["unverified"]
    assert "1.000" not in rendered["text"]
    assert json.dumps(value, sort_keys=True) == before


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize(
    "attempt,labels",
    [
        (
            {"stage": "native_initial"},
            ("初稿单元", "Initial native unit"),
        ),
        (
            {"stage": "evaluation", "cause": "capacity"},
            ("正文容量精简", "Body capacity refinement"),
        ),
        (
            {"stage": "evaluation"},
            ("正文定向修复", "Evaluation repair"),
        ),
        (
            {"stage": "evaluation", "purpose": "bounded-inventory-recovery-v3"},
            ("正文定向修复", "Evaluation repair"),
        ),
        (
            {"stage": "evaluation", "purpose": "bounded-inventory-recovery-v2-extra"},
            ("正文定向修复", "Evaluation repair"),
        ),
    ],
)
def test_inventory_label_does_not_absorb_existing_repair_branches(
    locale, attempt, labels
):
    """Keep exact contract matching separate from stage and prefix guesses."""
    value = {
        "options": {
            "targetedRepairState": {
                "version": "shared-targeted-repair-limit-v1",
                "limit": 2,
                "attempts": [{"index": 0, "status": "completed", **attempt}],
            }
        }
    }
    rendered = render_repair_state(value, locale)
    en = locale == "en"
    assert f"{labels[en]}: {'Completed' if en else '已完成'}" in rendered["repairTexts"]
    assert not any(
        text.startswith("Inventory re-extraction:" if en else "断言清单重抽取:")
        for text in rendered["repairTexts"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("locale", ["zh-CN", "en"])
async def test_actual_mixed_recovery_state_renders_without_inventing_body_repair(
    case, tmp_path, monkeypatch, locale
):
    """Follow SDK recovery output to the actual DOM with no mocked producer."""
    from test_mixed_inventory_pipeline import execute

    static_root = Path("src/litagent/static/flow-demo").resolve()
    report, gateway = await execute(case, tmp_path, monkeypatch, fault="once")
    state = report["metadata"]["targeted_repair_state"]
    history = report["metadata"]["evaluation_history"]
    attempt = state["attempts"][0]
    assert attempt["stage"] == "evaluation"
    assert attempt["purpose"] == "bounded-inventory-recovery-v2"
    assert attempt["status"] == "completed" and attempt["index"] == 0
    assert len(state["attempts"]) == 1 and state["limit"] == 2
    assert gateway.repair_count == 0 and gateway.recovery_requests == 1
    sha = hashlib.sha256(report["survey"].encode()).hexdigest()
    first = history["initial_evaluation"]
    assert first["faithfulness"]["details"]["source_report_sha256"] == sha
    assert (
        report["evaluation"]["faithfulness"]["details"]["source_report_sha256"] == sha
    )
    assert history["attempts"][0]["kind"] == "inventory_recovery"
    assert history["attempts"][0]["committed"] is False
    before = json.dumps(report, sort_keys=True)
    # The helper's default relative static path must still resolve to project
    # source if the real fixture later isolates its output working directory.
    monkeypatch.chdir(static_root.parents[3])
    options = {
        "qualityPolicy": "strict_auto_v2",
        "sha": sha,
        "targetedRepairState": state,
        "evaluationHistory": history,
    }
    rendered = render_repair_state(
        {
            "evaluation": report["evaluation"],
            "quality": report["quality"],
            "delivery": report["delivery"],
            "options": options,
        },
        locale,
    )
    en = locale == "en"
    assert (
        "Inventory re-extraction: Completed" if en else "断言清单重抽取: 已完成"
    ) in (rendered["repairTexts"])
    assert ("Attempts consumed: 1 / 2" if en else "已消耗次数: 1 / 2") in rendered[
        "repairTexts"
    ]
    assert not any(
        text.startswith("Evaluation repair:" if en else "正文定向修复:")
        for text in rendered["repairTexts"]
    )
    # The real first inventory was invalid. Its existing unverified/failed metric
    # remains so even when the shared recovery-call state is completed.
    initial = render_repair_state({"evaluation": first, "options": options}, locale)
    assert initial["metricStatuses"][-1] in {"failed", "unverified"}
    assert ("does not establish scientific support" if en else "不代表科学支持") in (
        initial["text"]
    )
    assert json.dumps(report, sort_keys=True) == before


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("binding", ["current", "old_failure", "old_evaluation"])
def test_terminal_and_upstream_causes_are_distinct_and_sha_bound(locale, binding):
    """Expose current failures without importing another version's diagnosis."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const binding=process.argv[3],en=process.argv[2]==='en';
 const evaluation={faithfulness:{skipped:true,details:{source_report_sha256:binding==='old_evaluation'?'old':'current',skipped_reason:'evaluation_capacity_refinement_required'}}};
 const failure={source_report_sha256:binding==='old_failure'?'old':'current',family:'quality',code:'writer_capacity_new_unit_unaccounted'};
 const before=JSON.stringify({evaluation,failure}),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,evaluation,{}, {},()=>{},x=>x,{locale:process.argv[2],sha:'current',writerFailure:failure});
 const text=walk(root).map(e=>e.textContent).join(' ');
 const terminal=en?'Terminal failure:':'最终失败原因:',upstream=en?'Upstream Evaluation:':'上游 Evaluation 原因:';
 if(text.includes(terminal)!==(binding!=='old_failure'))throw Error(text);
 if(text.includes(upstream)!==(binding==='current'))throw Error(text);
 if(binding==='current'&&(!text.includes('writer_capacity_new_unit_unaccounted')||!text.includes('evaluation_capacity_refinement_required')))throw Error(text);
 if(binding==='old_failure'&&text.includes('writer_capacity_new_unit_unaccounted'))throw Error('Borrowed another version failure');
 if(text.includes('1.000')||JSON.stringify({evaluation,failure})!==before)throw Error('Promoted or mutated an unverified result');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            binding,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("partial", [False, True])
def test_host_diagnostics_do_not_promote_an_invalid_inventory(locale, partial):
    """Show the validation boundary while keeping the exact score and source intact."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const en=process.argv[2]==='en',partial=process.argv[3]==='true';
 const details={evaluation_version:'survey-evaluation-v3',global_findings_contract:'located-global-findings-v1',inventory_check:{status:'failed'},coverage:{complete:false},factual_claim_count:5,supported_count:0,unsupported_count:0,unverified_count:5};
 if(partial)details.partial_assessment={stage:'host_validated',overall_valid:false,processed_assertions:2,expected_assertions:5,supported_count:1,unsupported_count:1,unverified_count:0};
 else {details.assessment_stage='host_validated';details.global_issues=[{owner:'inventory',code:'non_atomic',explanation:'A located inventory defect.'}];}
 const evaluation={faithfulness:{score:null,skipped:true,details}},before=JSON.stringify(evaluation),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,evaluation,{status:'unverified'},{status:'blocked'},()=>{},x=>x,{locale:process.argv[2]});
 const text=walk(root).map(e=>e.textContent).join(' ');
 if(!text.includes(en?'Inventory validity':'抽取清单有效性'))throw Error(text);
 const expected=partial?(en?'Partial host-checked diagnostics · not a final score':'部分宿主校验诊断 · 非最终分数'):(en?'Host-checked decisions; the overall evaluation remains invalid':'以上为宿主校验过的局部判断；整体验证仍无效');
 if(!text.includes(expected))throw Error(text);
 if(JSON.stringify(evaluation)!==before||text.includes('1.000'))throw Error('Promoted or mutated an invalid assessment');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            str(partial).lower(),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_unsent_repair_preflight_is_distinct_from_a_failed_paid_candidate(locale):
    """Preserve both diagnostics without inventing a repair call or changing history."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const history={version:'version-bound-evaluation-history-v1',source_report_sha256:'bound',attempts:[
   {attempted:false,reason_code:'writer_assessment_incomplete',candidate_evaluation:null},
   {attempted:true,reason_code:'sent_candidate_failed',candidate_evaluation:null}
 ]};
 const original=JSON.stringify(history),root=new E(),walk=e=>[e,...e.children.flatMap(walk)],en=process.argv[2]==='en';
 renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],sha:'bound',evaluationHistory:history});
 const text=walk(root).map(e=>e.textContent).join(' ');
 for(const expected of [en?'Repair preflight · no request sent':'修复预检 · 未发送请求',en?'Repair candidate 1':'修复候选 1','writer_assessment_incomplete','sent_candidate_failed'])if(!text.includes(expected))throw Error(expected);
 if(text.includes(en?'Repair candidate 2':'修复候选 2'))throw Error('unsent preflight consumed candidate ordinal');
 if(JSON.stringify(history)!==original)throw Error('source history changed');
})().catch(e=>{console.error(e);process.exit(1)});
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
def test_failed_writer_is_not_historical_evaluation_or_placeholder_analysis(locale):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const root=new E(),walk=e=>[e,...e.children.flatMap(walk)],en=process.argv[2]==='en';
 const evaluation={faithfulness:{score:0,passed:false,skipped:true,details:{skipped_reason:'final_report_unavailable'}}};
 const pipeline={assessment_scope:'generation_only',stages:{writing:'failed',evaluation:'not_run'},writing_diagnostics:[{location:'open_problems.blocks.0.sentences.4',raw_json_path:'sections.open_problems.first_block.additional_sentences.3',candidate_text:'',reason_code:'strict_writing_scope_notice_repeated',phase:'native_initial_repair',candidate_sha256:'a'.repeat(64)}]};
 const before=JSON.stringify(evaluation);
 renderQuality(root,evaluation,{status:'failed'},{status:'blocked'},()=>{},x=>x,{locale:process.argv[2],sha:'b'.repeat(64),bodyDiagnostics:{academic_characters:49,non_body_characters:0,repeated_units:0},pipelineVerification:pipeline});
 let text=walk(root).map(e=>e.textContent).join(' ');
 if(!text.includes(en?'Writing validation failed; Evaluation was not run.':'写作校验失败；Evaluation 未运行。'))throw Error(text);
 if(text.includes(en?'Historical evaluation scope':'历史评估口径')||text.includes(en?'Academic characters':'学术正文字符')||text.includes('0.000'))throw Error('Invented assessment from placeholder');
 for(const value of ['strict_writing_scope_notice_repeated','native_initial_repair','a'.repeat(64),'sections.open_problems.first_block.additional_sentences.3',en?'not Evaluation scores':'不是 Evaluation 评分'])if(!text.includes(value))throw Error(value);
 if(before!==JSON.stringify(evaluation))throw Error('Original evaluation mutated');
 renderQuality(root,{faithfulness:{score:.5,passed:false,details:{}}},{status:'failed'},{status:'blocked'},()=>{},x=>x,{locale:process.argv[2]});
 text=walk(root).map(e=>e.textContent).join(' ');
 if(!text.includes(en?'Historical evaluation scope':'历史评估口径'))throw Error('History compatibility');
})().catch(e=>{console.error(e);process.exit(1)});
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
@pytest.mark.parametrize("shape", ["old_parent_sha", "current_sha"])
def test_editorial_history_requires_exact_parent_and_current_assessment_identities(
    locale, shape
):
    """Read older receipts without overwriting scores or mixing report versions."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const evaluation={faithfulness:{details:{source_report_sha256:'new',supported_count:15,unsupported_count:0,unverified_count:0}}};
 const prior={source_report_sha256:'old',initial_evaluation:{faithfulness:{details:{supported_count:14,unsupported_count:2}}},attempts:[{reason_code:'first-local-rejection',candidate_evaluation:null},{reason_code:'second-scientific-pass',candidate_evaluation:{faithfulness:{details:{supported_count:16,unsupported_count:0}}}}]};
 const history={version:'version-bound-evaluation-history-v1',source_report_sha256:'old',prior_history:prior,editorial_cleanup:{version:'assessed-repair-repetition-v1',requires_full_reassessment:true,candidate_sha256:'old',result_sha256:'new'},final_evaluation:evaluation};
 if(process.argv[3]==='current_sha'){history.source_report_sha256='new';history.parent_report_sha256='old';}
 const before=JSON.stringify(history),root=new E(),walk=e=>[e,...e.children.flatMap(walk)],en=process.argv[2]==='en';
 renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],sha:'new',evaluationHistory:history});
 let text=walk(root).map(e=>e.textContent).join(' ');
 for(const value of ['first-local-rejection','second-scientific-pass',en?'Source-version initial Evaluation':'来源版本首次 Evaluation',en?'assessed in full again':'重新完整核验'])if(!text.includes(value))throw Error(value);
 if(before!==JSON.stringify(history))throw Error('immutable source history changed');
 for(const fault of ['selected_sha','parent','evaluation']){
   const h=structuredClone(history);let sha='new';
   if(fault==='selected_sha')sha='wrong';
   else if(fault==='parent')h.prior_history.source_report_sha256='wrong';
   else h.final_evaluation.faithfulness.details.source_report_sha256='wrong';
   renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],sha,evaluationHistory:h});
   text=walk(root).map(e=>e.textContent).join(' ');if(text.includes('first-local-rejection'))throw Error('stale history displayed');
 }
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            shape,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("historical", [False, True])
def test_analysis_host_ownership_failure_cannot_hide_behind_positive_model_prose(
    locale, historical
):
    """Render separate decisions without deriving new verdicts for old artifacts."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const row={section_key:'methods',adequate:false,model_adequate:true,claim_ids:[],explanation:'The model praised this analysis.',host_validation:{status:'failed',failure_codes:['analysis_wrong_section_claim'],allowed_claim_ids:['method-claim']}};
 if(process.argv[3]==='True'){delete row.model_adequate;delete row.host_validation;}
 const d={evaluation_version:'survey-evaluation-v3',body_analysis_contract:'body-analysis-v1',analysis_check:{status:'failed',sections:[row]},stages:{global_check:'invalid_response',global_reason_code:'evaluation_analysis_claim_ownership_invalid'}};
 const before=JSON.stringify(d),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,{faithfulness:{details:d}},{status:'failed'},{status:'blocked'},()=>{},x=>x,{locale:process.argv[2]});
 const text=walk(root).map(e=>e.textContent).join(' '),en=process.argv[2]==='en';
 if(process.argv[3]==='True'){
   if(!text.includes(en?'Host ownership check: Not recorded / unavailable':'宿主归属核验: 未记录／不可用'))throw Error(text);
 }else{
   if(!text.includes(en?'Model analysis decision: Passed':'模型分析结论: 通过'))throw Error(text);
   if(!text.includes(en?'Host gate checks: Failed':'宿主门禁检查: 未通过'))throw Error(text);
   if(!text.includes(en?'Claims do not belong to this chapter':'断言不属于本章'))throw Error(text);
 }
 if(before!==JSON.stringify(d))throw Error('immutable evaluation changed');
 if(!text.includes(en?'Invalid response contract':'响应契约无效')||!text.includes(en?'overall evaluation unverified':'整体未核验'))throw Error('Invalid global result is not explained');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            str(historical),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("status", ["completed", "failed", "started_no_replay"])
def test_targeted_repair_count_is_not_a_scientific_success(locale, status):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node unavailable")
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const state={version:'shared-targeted-repair-limit-v1',limit:2,attempts:[{stage:'native_initial',status:process.argv[3]}]};
 const before=JSON.stringify(state),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,{}, {status:'unverified'}, {status:'blocked'},()=>{},x=>x,{locale:process.argv[2],qualityPolicy:'strict_auto_v2',targetedRepairState:state});
 const text=walk(root).map(e=>e.textContent).join(' ');
 if(!text.includes('1 / 2') || !text.includes(process.argv[2]==='en'?'does not establish scientific support':'不代表科学支持'))throw Error(text);
 if(text.includes('1.000')||before!==JSON.stringify(state))throw Error('Invented score or mutation');
 renderQuality(root,{}, {status:'unverified'}, {status:'blocked'},()=>{},x=>x,{locale:process.argv[2],qualityPolicy:'strict_auto_v2',targetedRepairState:state,verification:{version:'optional-verification-v1',judge:false,jev:false}});
 const optionalText=walk(root).map(e=>e.textContent).join(' ');
 if(optionalText.includes(process.argv[2]==='en'?'full Evaluation and Judge are still required':'仍需完整 Evaluation 和 Judge'))throw Error('Disabled Judge described as mandatory');
 if(!optionalText.includes(process.argv[2]==='en'?'any enabled reviews':'已启用的复核'))throw Error('Missing optional repair gate explanation');
})();
"""
    result = subprocess.run(
        [
            node,
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo")),
            locale,
            status,
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_pre_writing_jev_result_is_visible_without_a_fake_report_score(locale):
    """Keep real stage failures distinct from historical or completed evaluations."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const root=new E(),locale=process.argv[2],walk=e=>[e,...e.children.flatMap(walk)];
 const receipt={status:'completed',accepted_evidence_ids:['a'],candidate_ledger:{a:{text:'Claim',supporting_text:'Original'}},rejected:{},jev:{actual_model:'jev-1.13.0',planned_question_count:2,decisions:[{},{}]}};
 const pipeline={stages:{extraction_jev:'completed',writing:'failed',evaluation:'not_run',judge:'not_run'},costs:{jev:{tokens:100,requests:1,usage_complete:true}},writing_diagnostics:[{location:'taxonomy.blocks.0.sentences.0',candidate_text:'Rejected draft',subreason:'quote_not_original',raw_json_path:'sections.taxonomy.first_block.first_sentence.support_quotes.0'}]};
 const before=JSON.stringify({receipt,pipeline});
 renderQuality(root,{}, {status:'unverified'}, {status:'blocked'},()=>{},x=>x,{locale,qualityPolicy:'strict_auto_v2',extractionVerification:receipt,pipelineVerification:pipeline});
 const text=walk(root).map(e=>e.textContent).join(' ');
 for(const expected of ['jev-1.13.0','1 / 1','2 / 2','100 tokens','quote_not_original','Rejected draft'])if(!text.includes(expected))throw Error('missing '+expected);
 if(!text.includes(locale==='en'?'Evaluation was not run':'Evaluation 未运行'))throw Error('invented evaluation scope');
 if(text.includes(locale==='en'?'Historical evaluation scope':'历史评估口径'))throw Error('new run mislabeled historical');
 if(JSON.stringify({receipt,pipeline})!==before)throw Error('immutable result changed');
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
@pytest.mark.parametrize(
    "policy", ["legacy", "strict_v1", "strict_auto_v1", "strict_auto_v2"]
)
def test_judge_jev_gate_description_follows_current_quality_policy(locale, policy):
    """Automatic audits participate in delivery; historical policies stay diagnostic."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {constructor(){this.children=[];this.dataset={};this.textContent='';}replaceChildren(...x){this.children=x;}append(...x){this.children.push(...x);}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new Element()};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {renderQuality}=await load('quality-view.js');const {translator}=await load('i18n.js');
 const locale=process.argv[2],policy=process.argv[3],root=new Element();
 const walk=e=>[e,...e.children.flatMap(walk)];
 const evaluation={faithfulness:{skipped:true,details:{quality_policy:policy}}};
 const acceptance={policy,actor:'system',status:'failed',report_sha256:'frozen-sha',gates:{judge:'failed',jev:'failed'},failure_codes:['judge_not_passed']};
 const before=JSON.stringify({evaluation,acceptance});
 const expected=policy==='strict_auto_v2'?(locale==='en'?'Judge · report delivery gate':'Judge · 正文交付门禁'):policy==='strict_auto_v1'?(locale==='en'?'Judge/Jev review · automatic delivery gate':'Judge/Jev 复核 · 自动交付门禁'):(locale==='en'?'Independent review · outside the delivery gate':'独立复核 · 不参与交付门禁');
 for(const bound of [false,true]){
  renderQuality(root,evaluation,{status:'failed'},{status:'blocked'},()=>{},translator(locale),{locale,...(bound&&['strict_auto_v1','strict_auto_v2'].includes(policy)?{automaticAcceptance:acceptance}:{})});
  const text=walk(root).map(e=>e.textContent).join(' ');
  if(!text.includes(expected))throw Error('wrong policy explanation');
  if(['strict_auto_v1','strict_auto_v2'].includes(policy)&&text.includes(locale==='en'?'outside the delivery gate':'不参与交付门禁'))throw Error('automatic gate misrepresented');
  if(bound&&['strict_auto_v1','strict_auto_v2'].includes(policy)&&(!text.includes('frozen-sha')||!text.includes('judge_not_passed')))throw Error('system failure hidden');
 }
 if(JSON.stringify({evaluation,acceptance})!==before)throw Error('display changed immutable result');
})().catch(error=>{console.error(error);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            policy,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


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


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_strict_quality_and_human_checklist_never_imply_approval(locale):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {constructor(tag=''){this.tag=tag;this.children=[];this.dataset={};this.events={};this.textContent='';}replaceChildren(...x){this.children=x;}append(...x){this.children.push(...x);}addEventListener(key,fn){this.events[key]=fn;}setAttribute(){}}
 global.document={createElement:tag=>new Element(tag)};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {renderQuality}=await load('quality-view.js'); const {renderReview}=await load('review-view.js'); const {translator}=await load('i18n.js'); const locale=process.argv[2], t=translator(locale), root=new Element();
 const data={faithfulness:{score:.8,passed:true,details:{evaluation_version:'survey-evaluation-v3',threshold:.8,strict_acceptance:{status:'failed',failure_codes:['science_not_fully_supported']},stages:{processed_assertions:16,expected_assertions:32,global_check:'interrupted_no_replay',batches:[{index:0,status:'completed',claim_ids:['one']},{index:1,status:'interrupted_no_replay',claim_ids:['two']}]}}}};
 const before=JSON.stringify(data);renderQuality(root,data,{status:'failed'},{status:'blocked'},()=>{},t,{locale}); const walk=e=>[e,...e.children.flatMap(walk)];const text=walk(root).map(e=>e.textContent).join(' ');
 if(!text.includes('100%')||!text.includes('16 / 32')||!text.includes('science_not_fully_supported')||!text.includes(locale==='en'?'automatic replay forbidden':'禁止自动重发'))throw Error('strict failed coverage missing');
 if(JSON.stringify(data)!==before)throw Error('mutated score');
 const request={stage:'final_report',report_revision:3,report_sha256:'b'.repeat(64),allowed_actions:['approve','reject'],report:{metadata:{quality_policy:'strict_v1'}}};renderReview(root,request,{},t,()=>{},()=>{});
 let items=walk(root);if(!items.some(e=>e.textContent==='r3 · SHA '+'b'.repeat(64)))throw Error('human version SHA unavailable');const approve=items.find(e=>e.value==='approve'), checks=items.filter(e=>e.type==='checkbox');if(!approve.disabled||checks.length!==5)throw Error('human checklist bypass');for(const input of checks){input.checked=true;input.events.change();}if(approve.disabled)throw Error('completed checklist blocked');
 request.allowed_actions=['reject'];renderReview(root,request,{},t,()=>{},()=>{});if(walk(root).some(e=>e.value==='approve'))throw Error('failed automatic gate has approve');
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
@pytest.mark.parametrize(
    "version", ["automatic-acceptance-v1", "automatic-acceptance-v2"]
)
def test_unrun_audit_status_and_historical_warning_are_localized(locale, version):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class Element {constructor(){this.children=[];this.dataset={};this.textContent='';}replaceChildren(...x){this.children=x;}append(...x){this.children.push(...x);}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new Element()};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {renderQuality}=await load('quality-view.js'); const {translator}=await load('i18n.js');
 const locale=process.argv[2],version=process.argv[3],root=new Element();
 const acceptance={version,policy:'strict_auto_v1',status:'failed',report_sha256:'bound',gates:{evaluation:'failed',judge:'blocked',jev:'not_run'},failure_codes:['evaluation_not_passed','judge_not_run','jev_not_run']};
 const before=JSON.stringify(acceptance);
 renderQuality(root,{},{status:'failed'},{status:'blocked'},()=>{},translator(locale),{locale,automaticAcceptance:acceptance});
 const walk=e=>[e,...e.children.flatMap(walk)];const text=walk(root).map(e=>e.textContent).join(' ');
 if(!text.includes(locale==='en'?'judge: Blocked':'judge: 阻塞')||!text.includes(locale==='en'?'jev: Not run':'jev: 未运行'))throw Error('unrun state not localized');
 const historical=locale==='en'?'Historical v1 record':'历史 v1 判定记录';
 if(text.includes(historical)!==(version==='automatic-acceptance-v1'))throw Error('wrong evaluation version warning');
 if(JSON.stringify(acceptance)!==before)throw Error('render changed immutable acceptance');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            version,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
@pytest.mark.parametrize("state", ["verified", "incomplete", "unsupported", "missing"])
def test_core_coverage_uses_final_host_science_instead_of_model_answer_flag(
    locale, state
):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const load=async name=>import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/'+name).toString('base64'));
 const {renderQuality}=await load('quality-view.js'),{translator}=await load('i18n.js');
 const locale=process.argv[2],state=process.argv[3],details={evaluation_version:'survey-evaluation-v3',coverage:{complete:true},assertions:[{claim_id:'c',kind:'scientific',status:'supported'}],coverage_requirements:[{requirement_id:'q',text:'Question target'}],core_coverage:[{requirement_id:'q',answered:true,claim_ids:['c']}]};
 if(state==='incomplete')details.coverage.complete=false;
 if(state==='unsupported')details.assertions[0].status='unsupported';
 if(state==='missing')delete details.assertions;
 const before=JSON.stringify(details),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,{faithfulness:{details}},{status:'failed'},{status:'blocked'},()=>{},translator(locale),{locale});
 const text=walk(root).map(e=>e.textContent).find(t=>t.startsWith('Question target:'));
 const expected=locale==='en'?(state==='verified'?'Passed':'Unverified'):(state==='verified'?'通过':'未核验');
 if(text!=='Question target: '+expected)throw Error('wrong host coverage '+text);
 if(JSON.stringify(details)!==before)throw Error('old evaluation was changed');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            state,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_evaluation_history_is_sha_bound_and_does_not_invent_candidate_score(locale):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const walk=e=>[e,...e.children.flatMap(walk)],root=new E();
 const evaluation={faithfulness:{details:{coverage:{processed_units:75,source_units:75},supported_count:14,unsupported_count:1,unverified_count:0,stages:{global_check:'completed'}}}};
 const history={version:'version-bound-evaluation-history-v1',source_report_sha256:'abc',initial_evaluation:evaluation,final_evaluation:evaluation,attempts:[{reason_code:'strict_repair_duplicate_content',committed:false,candidate_evaluation:null}]};
 const before=JSON.stringify(history);
 renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],sha:'abc',evaluationHistory:history});
 const text=walk(root).map(e=>e.textContent).join(' ');
 for(const expected of ['strict_repair_duplicate_content','75 / 75',process.argv[2]==='en'?'Full Evaluation not run':'未运行完整 Evaluation'])if(!text.includes(expected))throw Error(text);
 renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],sha:'other',evaluationHistory:history});
 if(walk(root).some(e=>e.textContent.includes('strict_repair_duplicate_content')))throw Error('Stale version history');
 if(JSON.stringify(history)!==before)throw Error('History mutated');
})().catch(e=>{console.error(e);process.exit(1)});
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
def test_disabled_optional_reviews_are_not_shown_as_passed(locale):
    """Keep local Evaluation mandatory and disabled services distinguishable."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const locale=process.argv[2],root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 const verification={version:'optional-verification-v1',judge:false,jev:false};
 const acceptance={version:'automatic-acceptance-v4',policy:'strict_auto_v2',verification,gates:{evaluation:'failed',judge:'disabled',extraction_jev:'disabled'},status:'failed'};
 const before=JSON.stringify(acceptance);
 renderQuality(root,{faithfulness:{details:{evaluation_version:'survey-evaluation-v3'}}},{status:'failed'},{status:'blocked'},()=>{},x=>x,{locale,automaticAcceptance:acceptance});
 const text=walk(root).map(e=>e.textContent).join(' '),disabled=locale==='en'?'Disabled':'未启用';
 if(!text.includes('Judge · '+disabled)||!text.includes('Jev · '+disabled))throw Error(text);
 if(!text.includes(locale==='en'?'Complete Evaluation is required':'完整 Evaluation 为必需门禁'))throw Error('mandatory Evaluation missing');
 if(text.includes(locale==='en'?'must all pass':'全部通过'))throw Error('old gates shown for optional run');
 if(JSON.stringify(acceptance)!==before)throw Error('mutated result');
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
def test_v2_failure_is_visible_without_a_compiled_report(locale):
    """A failed initial response has a root cause even before native bindings exist."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent=''}append(...x){this.children.push(...x)}replaceChildren(...x){this.children=x}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 renderQuality(root,{}, {status:'unverified'}, {status:'blocked'},()=>{},x=>x,{locale:process.argv[2],writerFailure:{family:'protocol',code:'writer_json_invalid'}});
 const text=walk(root).map(e=>e.textContent).join(' ');
 if(!text.includes('protocol · writer_json_invalid')||text.includes('1.000'))throw Error(text);
})().catch(e=>{console.error(e);process.exit(1)});
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
@pytest.mark.parametrize("policy", ["planned-budget-v2", "planned-budget-v3"])
def test_budget_admission_never_appears_as_a_quality_score(locale, policy):
    """Render run-level conservative reservations with no invented paid usage."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const plan={contract:'planned-budget-v2',status:'preflight_rejected',required_tokens:4836524,planned_tokens:5330000,maximum_tokens:1000000,planned_requests:41,maximum_requests:64,reason:'budget_plan_total_exceeded',stages:{assessment:{count:24,request_cap:117964,reserved_tokens:2831136,bound:'context_upper_bound'}}};
 plan.contract=process.argv[3]; if(plan.contract==='planned-budget-v3'){plan.repair_limit=0;plan.initial_evaluation_reserved=true;}
 const original=JSON.stringify(plan),root=new E(),walk=e=>[e,...e.children.flatMap(walk)],en=process.argv[2]==='en';
 renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],budgetPlan:plan});
 const text=walk(root).map(e=>e.textContent).join(' ');
 for(const expected of [en?'Workflow budget admission':'完整流程预算准入',en?'Preflight rejected':'预算检查未通过',en?'Reservations are not actual charges':'预留不等于实际费用','4836524','1000000'])if(!text.includes(expected))throw Error(expected);
 if(JSON.stringify(plan)!==original)throw Error('plan was changed');
 if(walk(root).some(e=>e.dataset.status==='passed'))throw Error('budget became a passed scientific score');
 if(plan.contract==='planned-budget-v3'&&!text.includes(en?'no allocation':'尚未分配'))throw Error('rejected path masquerades as allocated');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            policy,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_inventory_recovery_history_has_distinct_label_and_same_sha(locale):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const walk=e=>[e,...e.children.flatMap(walk)],root=new E();
 const evaluation={faithfulness:{details:{coverage:{processed_units:75,source_units:75},supported_count:14,unsupported_count:1,unverified_count:0,stages:{global_check:'completed'}}}};
 const history={version:'version-bound-evaluation-history-v1',source_report_sha256:'abc',initial_evaluation:evaluation,final_evaluation:evaluation,attempts:[{kind:'inventory_recovery',reason_code:'inventory_recovery_evaluated',committed:false,candidate_evaluation:evaluation}]};
 const before=JSON.stringify(history);
 renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],sha:'abc',evaluationHistory:history});
 const text=walk(root).map(e=>e.textContent).join(' ');
 for(const expected of ['inventory_recovery_evaluated','75 / 75',process.argv[2]==='en'?'Inventory re-extraction · prose unchanged':'断言清单重抽取 · 正文未修改'])if(!text.includes(expected))throw Error(text);
 renderQuality(root,{}, {},{},()=>{},x=>x,{locale:process.argv[2],sha:'other',evaluationHistory:history});
 if(walk(root).some(e=>e.textContent.includes('inventory_recovery_evaluated')))throw Error('Stale version history');
 if(JSON.stringify(history)!==before)throw Error('History mutated');
})().catch(e=>{console.error(e);process.exit(1)});
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
@pytest.mark.parametrize("no_report", [False, True])
def test_summary_quantity_failure_is_visible_only_before_a_final_report(
    locale, no_report
):
    """Show host quantity failures without converting an unrun evaluation to zero."""
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
 global.document={createElement:()=>new E()};
 const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
 const noReport=process.argv[3]==='true',root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 const failure={code:'writer_flat_schema_invalid',diagnostics:[{section:'conclusion',actual:3,maximum:2,summary_constraints:{contract:'effective-summary-limits-v1'}}]};
 const evaluation={faithfulness:{score:null,skipped:true,details:{skipped_reason:noReport?'final_report_unavailable':'other'}}},before=JSON.stringify({failure,evaluation});
 renderQuality(root,evaluation,{status:'unverified'},{status:'blocked'},()=>{},x=>x,{locale:process.argv[2],writerFailure:failure});
 const text=walk(root).map(e=>e.textContent).join(' '),title=process.argv[2]==='en'?'Summary structure validation failed':'摘要结构校验失败';
 if(text.includes(title)!==noReport)throw Error(text);
 if(noReport&&!text.includes('3 / 2'))throw Error('Missing actual/maximum');
 if(text.includes('1.000')||JSON.stringify({failure,evaluation})!==before)throw Error('Changed scientific state');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
            str(no_report).lower(),
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
