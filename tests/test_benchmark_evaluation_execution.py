"""Keep scientific execution, acceptance gates and historical evidence distinct."""

import copy
import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from test_quality_protocol_stop import live_runner, report

from litagent.benchmark.quality_suite import (
    evaluation_execution,
    export_bundle,
    public_projection,
    summarize,
)


def assessment(status):
    """Describe execution proof independently of a scientific verdict."""
    if status == "not_run":
        return {
            "skipped": True,
            "details": {"skipped_reason": "final_report_unavailable"},
        }
    if status == "unrecorded":
        return {"score": 0, "passed": False, "details": {}}
    return {
        "score": 0.5 if status == "completed" else 0,
        "passed": False,
        "details": {
            "evaluation_version": "survey-evaluation-v3",
            "coverage": {"complete": status == "completed"},
            "stages": {
                "requests": [{"status": "completed"}],
                "global_check": "completed" if status == "completed" else "interrupted",
            },
        },
    }


@pytest.mark.parametrize("status", ["not_run", "incomplete", "completed", "unrecorded"])
@pytest.mark.asyncio
async def test_actual_live_consumer_keeps_execution_separate_from_failed_gate(
    tmp_path, monkeypatch, status
):
    """Replace only the external Survey and retain actual result classification."""
    value = report("scientific_failure")
    value["evaluation"]["faithfulness"] = assessment(status)
    original = copy.deepcopy(value)
    runner, calls = live_runner(tmp_path, monkeypatch, [value])
    result = await runner._live(
        runner.dataset.cases[0], runner.config, "execution-case"
    )
    assert len(calls) == 1 and result["evaluation_execution"] == status
    assert result["gates"]["evaluation"] == "failed"
    assert result["acceptance"]["publishable"] is False
    assert value == original


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"details": []},
        {"details": {"stages": [1]}},
        {"score": 1, "passed": True},
        {"skipped": True, "details": {"skipped_reason": "legacy_provider_failure"}},
    ],
)
def test_missing_or_invalid_proof_does_not_invent_execution(payload):
    assert evaluation_execution(payload) == "unrecorded"


def test_sent_request_cannot_be_labelled_not_run_by_a_skipped_score():
    value = assessment("incomplete")
    value.update(skipped=True)
    value["details"]["skipped_reason"] = "final_report_unavailable"
    assert evaluation_execution(value) == "incomplete"
    value["details"]["stages"]["requests"] = []
    assert evaluation_execution(value) == "not_run"


def result_fixture():
    """Use failed gates for all statuses so completion cannot imply delivery."""
    statuses = ["not_run", "incomplete", "completed", "unrecorded"]
    planned = [{"phase": "formal", "case_id": s, "domain": "vit"} for s in statuses]
    result = {
        "planned": planned,
        "observations": [
            {
                **p,
                "status": "blocked",
                "evaluation_execution": p["case_id"],
                "gates": {"evaluation": "failed"},
                "acceptance": {"publishable": False},
                "costs": {},
            }
            for p in planned
        ],
        "stop_reasons": [],
        "budgets": {},
    }
    result["counts"] = summarize(result)
    return result


def test_four_exports_share_execution_and_preserve_gates_and_denominators(tmp_path):
    result = result_fixture()
    before = copy.deepcopy(result)
    export_bundle(tmp_path, result)
    public = public_projection(result)
    assert json.loads((tmp_path / "benchmark.json").read_text()) == public
    with (tmp_path / "benchmark.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    for index, observation in enumerate(public["observations"]):
        assert (
            rows[index]["evaluation_execution"] == observation["evaluation_execution"]
        )
        assert rows[index]["evaluation"] == "failed"
        label = f"执行 / execution: {observation['evaluation_execution']} · 门禁 / gate: failed"
        assert label in (tmp_path / "benchmark.md").read_text()
        assert label in (tmp_path / "benchmark.html").read_text()
    assert result == before and public["counts"]["formal"]["blocked"] == 4
    assert not public["counts"]["goal_met"]


@pytest.mark.parametrize(
    "invalid", [None, {"private": "secret"}, "private_body_text", 1]
)
def test_public_execution_field_accepts_only_fixed_enums(invalid):
    result = result_fixture()
    result["observations"][0]["evaluation_execution"] = invalid
    assert (
        public_projection(result)["observations"][0]["evaluation_execution"]
        == "unrecorded"
    )
    assert "private_body_text" not in json.dumps(public_projection(result))


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_browser_dom_distinguishes_execution_from_delivery(locale):
    script = r"""
const fs=require('node:fs');
(async()=>{
 class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(k,v){this[k]=v;}getAttribute(k){return this[k];}}
 global.document={createElement:()=>new E(),createElementNS:()=>new E()};
 const dir=process.argv[1],locale=process.argv[2];
 const mod=async p=>import('data:text/javascript;base64,'+fs.readFileSync(dir+'/'+p).toString('base64'));
 const {translator}=await mod('i18n.js'),{renderBenchmarkDetail}=await mod('benchmark-view.js');
 const t=translator(locale),root=new E(),walk=e=>[e,...e.children.flatMap(walk)];
 const counts={planned:15,run:1,automatic_passed:0,complete_verification:0,not_run:14,pending:0,terminal:1,blocked:1};
 for(const status of [undefined,'not_run','incomplete','completed','absent']){
  const result={benchmark_type:'survey_quality',run_id:'case',status:'stopped',planned:[{phase:'pilot',case_id:'x',domain:'vit'}],counts:{pilot:counts,formal:counts},observations:[{phase:'pilot',case_id:'x',status:'blocked',evaluation_execution:status,gates:{evaluation:'failed',judge:'disabled',extraction_jev:'disabled'},counts:{},costs:{}}],budgets:{},stop_reasons:[]};
  if(status==='absent')result.observations=[];
  const before=JSON.stringify(result);renderBenchmarkDetail(root,result,()=>{},t);
  const text=walk(root).map(e=>e.textContent).join(' ');
  const labels={absent:'evaluationNotExecuted',not_run:'evaluationNotExecuted',incomplete:'evaluationIncomplete',completed:'evaluationCompleted'};
  if(!text.includes(t(labels[status]||'evaluationUnrecorded'))||!text.includes(t('evaluationDeliveryGate')+': '+(status==='absent'?t('notRun'):'failed')))throw Error(text);
  if(before!==JSON.stringify(result))throw Error('historical mutation');
 }
})().catch(e=>{console.error(e);process.exit(1)});
"""
    output = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            locale,
        ],
        text=True,
        capture_output=True,
        timeout=15,
    )
    assert output.returncode == 0, output.stderr


def test_summary_candidate_rejection_is_shared_by_four_exports(tmp_path):
    from litagent import review_policy
    from litagent.eval.results import CONTRACT
    from litagent.review_cycles import SUMMARY_REJECTION_CONTRACT

    value = result_fixture()
    rejection = {
        "contract": SUMMARY_REJECTION_CONTRACT,
        "category": "review_revision",
        "reason_code": "writer_inline_citation_unknown",
        "retained_report_sha256": "a" * 64,
        "call": "summary_repair_1",
        "remaining": 2,
        "actual_tokens": 12176,
        "raw": "PRIVATE RAW RESPONSE",
        "diagnostics": [
            {
                "path": ["conclusion", 0, 1],
                "offset": 10,
                "reference": "c29",
                "in_frozen_pool": True,
                "code": "writer_inline_citation_unknown",
            },
            {
                "path": ["conclusion", 0, 1],
                "offset": 20,
                "reference": "c48",
                "in_frozen_pool": False,
                "code": "writer_inline_citation_unknown",
            },
        ],
    }
    value["observations"][0]["evaluation_summary"] = {
        "contract": CONTRACT,
        "report_sha256": "a" * 64,
        "metrics": {"faithfulness": {"status": "unverified", "score": None}},
        "review_workflow": {
            "contract": review_policy.CONTRACT,
            "report_sha256": "a" * 64,
            "allowances": {
                k: {
                    "limit": 3,
                    "used": 1 if k == "review_revision" else 0,
                    "remaining": 2 if k == "review_revision" else 3,
                }
                for k in review_policy.ALLOWANCES
            },
            "reviews": [
                {
                    "round": 0,
                    "status": "candidate_rejected",
                    "responses": {},
                    "source_report_sha256": "a" * 64,
                    "summary_rejection": rejection,
                }
            ],
            "evaluations": [],
            "publishable": False,
        },
    }
    original = copy.deepcopy(value)
    export_bundle(tmp_path, value)
    projected = json.loads((tmp_path / "benchmark.json").read_text())
    summary = projected["observations"][0]["evaluation_summary"]
    assert summary["metrics"]["faithfulness"]["score"] is None
    assert (
        summary["review_workflow"]["reviews"][0]["summary_rejection"]["remaining"] == 2
    )
    with (tmp_path / "benchmark.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert json.loads(rows[0]["evaluation_summary"]) == summary
    for extension in ("md", "html"):
        text = (tmp_path / f"benchmark.{extension}").read_text()
        assert "保留稿继续核验" in text and "c29" in text and "c48" in text
        assert "12176" in text and "Unknown frozen card" in text
        assert "Outside body dependencies" in text
    assert all(
        "PRIVATE RAW RESPONSE" not in f.read_text()
        for f in tmp_path.glob("benchmark.*")
    )
    assert value == original


def test_patch_candidate_rejection_is_shared_by_four_exports(tmp_path):
    from litagent import review_policy
    from litagent.eval.results import CONTRACT
    from litagent.review_cycles import PATCH_REJECTION_CONTRACT

    value = result_fixture()
    rejection = {
        "contract": PATCH_REJECTION_CONTRACT,
        "category": "review_revision",
        "reason_code": "writer_patch_response_mismatch",
        "retained_report_sha256": "a" * 64,
        "call": "repair_0",
        "remaining": 2,
        "actual_tokens": 12176,
        "raw": "PRIVATE RAW RESPONSE",
        "diagnostics": [
            {
                "target_key": "t5",
                "target_id": "writer:methods:3:1",
                "action": "keep",
                "patch_index": 4,
                "reason": "patch_for_non_replace",
            },
            {
                "target_key": "t6",
                "target_id": "writer:methods:3:2",
                "action": "keep",
                "patch_index": 5,
                "reason": "patch_for_non_replace",
            },
        ],
    }
    value["observations"][0]["evaluation_summary"] = {
        "contract": CONTRACT,
        "report_sha256": "a" * 64,
        "metrics": {"faithfulness": {"status": "unverified", "score": None}},
        "review_workflow": {
            "contract": review_policy.CONTRACT,
            "report_sha256": "a" * 64,
            "allowances": {
                k: {
                    "limit": 3,
                    "used": 1 if k == "review_revision" else 0,
                    "remaining": 2 if k == "review_revision" else 3,
                }
                for k in review_policy.ALLOWANCES
            },
            "reviews": [
                {
                    "round": 0,
                    "status": "candidate_rejected",
                    "responses": {},
                    "source_report_sha256": "a" * 64,
                    "patch_rejection": rejection,
                }
            ],
            "evaluations": [],
            "publishable": False,
        },
    }
    original = copy.deepcopy(value)
    export_bundle(tmp_path, value)
    projected = json.loads((tmp_path / "benchmark.json").read_text())
    summary = projected["observations"][0]["evaluation_summary"]
    assert summary["metrics"]["faithfulness"]["score"] is None
    assert summary["review_workflow"]["reviews"][0]["patch_rejection"]["remaining"] == 2
    with (tmp_path / "benchmark.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert json.loads(rows[0]["evaluation_summary"]) == summary
    for extension in ("md", "html"):
        text = (tmp_path / f"benchmark.{extension}").read_text()
        assert "保留稿继续核验" in text and "t5" in text and "t6" in text
        assert "12176" in text and "Patch candidate rejected" in text
        assert "patch_for_non_replace" in text
    assert all(
        "PRIVATE RAW RESPONSE" not in f.read_text()
        for f in tmp_path.glob("benchmark.*")
    )
    assert value == original
