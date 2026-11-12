"""Exercise dual admission through real Jev transport and ledger consumers."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from test_automatic_audit_controls import _run_node
from test_extraction_verification import selection, verify
from test_jev_citation_audit import _valid_response
from test_optional_verification import optional_report
from test_quality_view import render_repair_state
from test_scope_notice_contract import notice_document

from litagent.agents.extractor import ExtractorWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.api import SurveyRequest
from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.evidence import SOURCE_FIELDS_IDENTITY, collect_ledger
from litagent.extraction_verification import (
    ADMISSION_VERSION,
    ExtractionVerifier,
    admitted_writing_ledger,
    validate_extraction_receipt,
)
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import build_fact_cards, digest
from litagent.runner import LitAgent
from litagent.workbench import (
    RunOptions,
    VerificationOptions,
    default_verification_options,
)


def dual_transport(captured, *, validity="ok", formatting="ok", fault=None):
    """Replace only external HTTP; preserve real packing, decoding and receipts."""

    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        response = _valid_response(request)
        for qid, answer in response["answers"].items():
            item = payload["state"]["audit_items"][qid]
            record = item["candidate_record"]
            decision = validity if qid.startswith("v:") else formatting
            if record.get("fact_fields", {}).get("metric") == "speed":
                decision = "not_ok"
            answer["choice"] = decision
            answer["probabilities"] = {
                key: float(key == decision) for key in answer["probabilities"]
            }
            if fault == "low":
                answer["confidence"] = 0.89
        if fault == "model":
            response["model"] = "wrong-model"
        elif fault == "usage":
            response.pop("usage")
        elif fault == "omitted":
            response["answers"].pop(next(iter(response["answers"])))
        elif fault == "probability":
            next(iter(response["answers"].values()))["probabilities"]["ok"] = -1
        return httpx.Response(200, json=response)

    return httpx.MockTransport(respond)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "validity,formatting,allowed",
    [
        ("ok", "ok", True),
        ("not_ok", "ok", False),
        ("ok", "not_ok", False),
        ("not_ok", "not_ok", False),
        ("uncertain", "ok", False),
    ],
)
async def test_both_questions_required_and_originals_preserved(
    tmp_path, monkeypatch, validity, formatting, allowed
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    ledger = {"a": selection().selected_items["a"]}
    captured = []
    async with httpx.AsyncClient(
        transport=dual_transport(captured, validity=validity, formatting=formatting)
    ) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        admitted, receipt = await gate.verify_candidates(ledger, run_id="dual")
        assert bool(admitted) is allowed
        assert receipt["candidate_ledger"] == ledger
        assert validate_extraction_receipt(receipt, admitted)
        assert len(captured) == 1 and len(captured[0]["questions"]) == 2
        assert {d["kind"] for d in receipt["jev"]["decisions"]} == {
            "claim_validity",
            "claim_format",
        }
        assert receipt["jev"]["summary"]["claim_completion_rate"] == 1
        for state in captured[0]["state"]["audit_items"].values():
            original = state["cited_evidence"]
            assert original[0]["text"] == ledger["a"]["supporting_text"]
        assert await gate.verify_candidates(ledger, run_id="dual") == (
            admitted,
            receipt,
        )
        assert len(captured) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["model", "usage", "omitted", "probability", "low"])
async def test_invalid_or_uncertain_verdict_cannot_admit_or_replay(
    tmp_path, monkeypatch, fault
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    captured = []
    async with httpx.AsyncClient(
        transport=dual_transport(captured, fault=fault)
    ) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        if fault == "low":
            admitted, _ = await gate.verify_candidates(
                selection().selected_items, run_id="bad"
            )
            assert not admitted
        else:
            with pytest.raises(ValueError, match="extraction_verification_incomplete"):
                await gate.verify_candidates(selection().selected_items, run_id="bad")
            with pytest.raises(ValueError, match="no_replay"):
                await gate.verify_candidates(selection().selected_items, run_id="bad")
        assert len(captured) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unknown_field", "field_type", "missing_original"])
async def test_invalid_local_record_never_reaches_jev(tmp_path, fault):
    ledger = selection().selected_items
    if fault == "unknown_field":
        ledger["a"]["fact_fields"]["injected"] = "allowed"
    elif fault == "field_type":
        ledger["a"]["fact_fields"]["value"] = 50
    else:
        ledger["a"]["supporting_text"] = ""
    auditor = AsyncMock()
    with pytest.raises(ValueError):
        await ExtractionVerifier(
            tmp_path, protocol=ADMISSION_VERSION, auditor=auditor
        ).verify_candidates(ledger, run_id="invalid")
    auditor.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_extractor_to_synthesis_quarantines_and_pays_once(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    text = "DeiT reports 81.8% accuracy on ImageNet."
    strategy = AsyncMock()
    strategy.extract.return_value = {
        "claims": [text],
        "record_identity": SOURCE_FIELDS_IDENTITY,
        "claim_records": [
            {
                "text": text,
                "source_chunk_key": "c1",
                "fact_fields": {"metric": metric, "value": "81.8%"},
            }
            for metric in ("accuracy", "speed")
        ],
    }
    loader = AsyncMock()
    loader.enrich.side_effect = lambda papers, _: papers
    captured = []
    options = default_verification_options().model_dump()
    async with httpx.AsyncClient(transport=dual_transport(captured)) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        worker = ExtractorWorker(
            strategy,
            fulltext_loader=loader,
            extraction_verifier=gate,
            run_id_provider=lambda: "pipeline",
        )
        output = await worker.execute(
            SubTask(
                "extract",
                "test",
                "extractor",
                input_data={
                    "quality_policy": "strict_auto_v2",
                    "verification": options,
                    "upstream_results": {
                        "dedup": [
                            {
                                "paper_id": "deit",
                                "title": "DeiT",
                                "chunks": [{"chunk_key": "c1", "text": text}],
                            }
                        ]
                    },
                },
            )
        )
        assert len(output[0]["candidate_evidence_items"]) == 2
        assert len(output[0]["evidence_items"]) == len(output[0]["claim_records"]) == 1
        assert output[0]["claim_records"][0]["fact_fields"]["metric"] == "accuracy"
        ledger = collect_ledger(output)
        selector = AsyncMock()
        picked = selection()
        picked.selected_items = ledger
        picked.candidate_count = len(ledger)
        picked.omitted_count = 0
        picked.section_evidence_ids = {
            key: list(ledger) for key in picked.section_evidence_ids
        }
        selector.select.return_value = picked
        llm = AsyncMock()
        llm.chat.return_value = LLMResponse(
            content=json.dumps(notice_document(build_fact_cards(ledger))),
            model="offline",
            usage={},
        )
        synthesis = SynthesisWorker(
            llm,
            evidence_selector=selector,
            extraction_verifier=gate,
            run_id_provider=lambda: "pipeline",
        )
        result = await synthesis.execute(
            SubTask(
                "synthesis",
                "test",
                "synthesis",
                input_data={
                    "quality_policy": "strict_auto_v2",
                    "verification": options,
                    "review_structure": "academic_v2",
                    "query": "DeiT",
                    "upstream_results": {"extract": output},
                },
            )
        )
        assert len(captured) == 1
        assert result["evidence_selection"]["selected_items"] == ledger
        assert result["extraction_verification"] == output[0]["extraction_verification"]


@pytest.mark.asyncio
async def test_receipt_allows_ranked_subset_but_rejects_tampering(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    captured = []
    async with httpx.AsyncClient(transport=dual_transport(captured)) as client:
        admitted, receipt = await ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        ).verify_candidates(
            selection().selected_items,
            run_id="test-auto",
            resource_policy=default_verification_options().extraction_resource_policy,
            failure_policy=default_verification_options().extraction_failure_policy,
            confidence_threshold=(
                default_verification_options().extraction_confidence_threshold
            ),
        )
    subset = {"a": admitted["a"]}
    cards = build_fact_cards(subset)
    assert (
        admitted_writing_ledger(receipt, admitted, cards, run_id="test-auto") == subset
    )
    report, _ = optional_report()
    report["metadata"].update(
        verification=default_verification_options().model_dump(),
        extraction_verification=receipt,
        fact_cards=cards,
        evidence_snapshot_sha256=digest(cards),
    )
    snapshot = freeze_automatic_inputs(report, admitted, run_id="test-auto")
    assert snapshot["ledger"] == subset
    assert automatic_acceptance(snapshot, report["evaluation"], None)["publishable"]
    corrupted = copy.deepcopy(subset)
    corrupted["a"]["supporting_text"] += "changed"
    with pytest.raises(ValueError):
        admitted_writing_ledger(receipt, corrupted, cards, run_id="test-auto")


@pytest.mark.asyncio
async def test_new_contract_cannot_downgrade_to_old_receipt(tmp_path, monkeypatch):
    selected, receipt, _ = await verify(tmp_path, monkeypatch, run_id="test-auto")
    report, _ = optional_report()
    cards = build_fact_cards(selected.selected_items)
    report["metadata"].update(
        verification=default_verification_options().model_dump(),
        extraction_verification=receipt,
        fact_cards=cards,
        evidence_snapshot_sha256=digest(cards),
    )
    with pytest.raises(ValueError, match="extraction"):
        freeze_automatic_inputs(report, selected.selected_items, run_id="test-auto")


@pytest.mark.asyncio
async def test_missing_receipt_fails_before_selection_or_writing():
    selector, llm = AsyncMock(), AsyncMock()
    worker = SynthesisWorker(
        llm, evidence_selector=selector, run_id_provider=lambda: "missing"
    )
    with pytest.raises(ValueError, match="admission_receipt_missing"):
        await worker.execute(
            SubTask(
                "synthesis",
                "test",
                "synthesis",
                input_data={
                    "quality_policy": "strict_auto_v2",
                    "verification": default_verification_options().model_dump(),
                },
            )
        )
    selector.select.assert_not_awaited()
    llm.chat.assert_not_awaited()


def test_new_request_defaults_and_old_options_remain_distinct():
    request = SurveyRequest(query="topic", quality_policy="strict_auto_v2")
    assert request.verification == default_verification_options()
    assert request.auto_audit_mode == "off"
    old = VerificationOptions()
    assert old.version == "optional-verification-v1" and not old.jev
    explicit = SurveyRequest(
        query="topic", quality_policy="strict_auto_v2", verification=old
    )
    assert explicit.verification == old
    new = RunOptions(quality_policy="strict_auto_v2", verification=request.verification)
    assert (
        new.fingerprint()
        != RunOptions(quality_policy="strict_auto_v2", verification=old).fingerprint()
    )
    assert SurveyRequest(query="old").verification is None


def test_task_deadline_covers_admission_without_repeated_extension():
    graph = TaskGraph()
    task = SubTask("extract", "test", "extractor", timeout_ms=1200000)
    graph.add_task(task)
    options = RunOptions(
        quality_policy="strict_auto_v2", verification=default_verification_options()
    )
    LitAgent._apply_run_options_to_graph(graph, options)
    assert task.timeout_ms == 2710000
    LitAgent._apply_run_options_to_graph(graph, options)
    assert task.timeout_ms == 2710000
    LitAgent._apply_run_options_to_graph(graph, RunOptions())
    assert task.timeout_ms == 1200000


def test_frontend_defaults_on_blocks_missing_key_and_retains_explicit_off():
    script = r"""
const fs=require('node:fs');
(async()=>{
class E{constructor(){this.value='';this.checked=false;this.options=[];this.events={};}addEventListener(k,f){this.events[k]=f;}}
const elements=new Map(),$=id=>{if(!elements.has(id))elements.set(id,new E());return elements.get(id)};
global.document={querySelector:$};
$('#quality-policy').value='strict_auto_v2';$('#auto-audit-mode').options=['off','judge','judge_jev'].map(value=>({value}));
const caps={judge_available:true,jev_available:false,default_generation_strategy:'body_summary_v2',default_budget_policy:'planned-budget-v4',default_verification:{version:'optional-verification-v2',judge:false,jev:true,extraction_failure_policy:'verified_subset_v1',extraction_confidence_threshold:0.6,extraction_resource_policy:"expanded-v1"}};
const {bindSurveyVerification}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]).toString('base64'));
const prefs=bindSurveyVerification({capabilities:()=>caps,t:x=>x});
if(!$('#verification-jev').checked||$('#verification-jev').disabled||$('#verification-judge').checked)throw Error('new defaults or explicit off unavailable');
let blocked=false;try{prefs.options()}catch{blocked=true}if(!blocked)throw Error('missing key silently disabled gate');
caps.jev_available=true;prefs.render();let v=prefs.options();if(!v.verification.jev||v.budget_policy!=='planned-budget-v4'||v.verification.version!=='optional-verification-v2')throw Error('admission default or budget lost');if(v.verification.extraction_failure_policy!=='verified_subset_v1')throw Error('fallback preference lost');
if(v.verification.extraction_resource_policy!=='expanded-v1')throw Error('resource preference lost');
if(v.verification.extraction_confidence_threshold!==0.6)throw Error('confidence preference lost');
$('#verification-jev').checked=false;$('#verification-jev').events.change();prefs.render();if(prefs.options().verification.jev)throw Error('explicit off overridden');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    _run_node(script, "src/litagent/static/flow-demo/audit-controls.js")


@pytest.mark.asyncio
async def test_admission_cancel_does_not_reissue_unknown_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    started = asyncio.Event()
    calls = []

    async def waiting(request):
        calls.append(request)
        started.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(waiting)) as client:
        gate = ExtractionVerifier(
            tmp_path,
            protocol=ADMISSION_VERSION,
            auditor=JevCitationAuditor(client=client),
        )
        task = asyncio.create_task(
            gate.verify_candidates(selection().selected_items, run_id="cancel")
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        with pytest.raises(ValueError, match="no_replay"):
            await gate.verify_candidates(selection().selected_items, run_id="cancel")
        assert len(calls) == 1
        receipt = json.loads((tmp_path / "cancel" / "receipt.json").read_text())
        assert receipt["status"] == "failed_no_replay"


@pytest.mark.asyncio
async def test_complete_batch_capacity_is_checked_before_dispatch(tmp_path):
    ledger = {}
    for i in range(161):
        item = copy.deepcopy(selection().selected_items["a"])
        item["evidence_id"] = f"a{i}"
        ledger[item["evidence_id"]] = item
    auditor = AsyncMock()
    with pytest.raises(ValueError, match="budget_exceeded"):
        await ExtractionVerifier(
            tmp_path, protocol=ADMISSION_VERSION, auditor=auditor
        ).verify_candidates(ledger, run_id="large")
    auditor.evaluate.assert_not_awaited()


@pytest.mark.parametrize("locale", ["zh-CN", "en"])
def test_quality_view_explains_independent_claim_checks(locale):
    receipt = {
        "version": ADMISSION_VERSION,
        "status": "completed",
        "candidate_ledger": {"a": {}},
        "accepted_evidence_ids": [],
        "rejected": {"a": ["extraction_claim_format_not_ok"]},
        "jev": {
            "decisions": [
                {
                    "kind": "claim_validity",
                    "choice": "ok",
                    "confidence": 0.97,
                    "uncertain": False,
                },
                {
                    "kind": "claim_format",
                    "choice": "not_ok",
                    "confidence": 0.97,
                    "uncertain": False,
                },
            ]
        },
    }
    result = render_repair_state(
        {
            "options": {
                "qualityPolicy": "strict_auto_v2",
                "verification": default_verification_options().model_dump(),
                "extractionVerification": receipt,
            }
        },
        locale,
    )
    assert (
        "Claim validity: 1 / 1" if locale == "en" else "Claim 合法性: 1 / 1"
    ) in result["text"]
    assert ("Claim format: 0 / 1" if locale == "en" else "Claim 格式: 0 / 1") in result[
        "text"
    ]
