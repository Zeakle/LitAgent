"""Exercise native Jev extraction verification and actual downstream writing gates."""

import asyncio
import copy
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from test_jev_citation_audit import _valid_response
from test_scope_notice_contract import notice_document as document
from test_strict_automatic import frozen, passing_audit
from test_strict_quality import draft, to_native

from litagent.agents.extractor import ExtractorWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.citation_audit import JevCitationAuditor
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary, normalize_survey_result
from litagent.eval.automatic import automatic_acceptance, freeze_automatic_inputs
from litagent.eval.base import CTX_EVIDENCE
from litagent.extraction_verification import (
    ExtractionVerifier,
    build_extraction_inventory,
    validate_extraction_receipt,
)
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import build_fact_cards, digest
from litagent.runner import LitAgent
from litagent.workbench import RunOptions, is_strict_policy


def selection():
    """Keep wrong compressed claims alongside unchanged originals and valid field names."""
    items = {
        "a": {
            "evidence_id": "a",
            "paper_id": "evit",
            "text": "EViT improves speed by 50% on DeiT-S.",
            "supporting_text": "EViT improves speed by 50% on DeiT-S.",
            "chunk_key": "abstract",
            "fact_fields": {"method": "EViT", "value": "50%"},
        },
        "b": {
            "evidence_id": "b",
            "paper_id": "evo",
            "text": "Evo-ViT improves FLOPs by 50%.",
            "supporting_text": "EViT improves speed by 50% on DeiT-S.",
            "chunk_key": "abstract",
        },
    }
    keys = ["introduction", "taxonomy", "methods", "experiments", "open_problems"]
    return EvidenceSelection(
        2,
        items,
        {k: ["a", "b"] for k in keys},
        {k: "lexical_fallback" for k in keys},
        0,
        100,
    )


def transport(captured, fault=None):
    """Choose negative fixtures by input identity without claiming human-calibrated truth."""

    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        response = _valid_response(request)
        for qid, value in payload["state"]["audit_items"].items():
            evidence = value["cited_evidence"]
            eid = (evidence[0] if isinstance(evidence, list) else evidence)[
                "evidence_id"
            ]
            if eid == "b":
                answer = response["answers"][qid]
                options = list(answer["probabilities"])
                answer["choice"] = options[-1]
                answer["probabilities"] = {o: float(o == options[-1]) for o in options}
            if fault == "low" and eid == "a":
                response["answers"][qid]["confidence"] = 0.89
        if fault == "model":
            response["model"] = "other-model"
        elif fault == "usage":
            response.pop("usage")
        elif fault == "omitted":
            response["answers"].pop(next(iter(response["answers"])))
        elif fault == "auth":
            return httpx.Response(401)
        return httpx.Response(200, json=response)

    return httpx.MockTransport(respond)


async def verify(tmp_path, monkeypatch, fault=None, run_id="extract-test"):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    captured = []
    async with httpx.AsyncClient(transport=transport(captured, fault)) as client:
        gate = ExtractionVerifier(tmp_path, auditor=JevCitationAuditor(client=client))
        selected, receipt = await gate.verify(selection(), run_id=run_id)
    return selected, receipt, captured


@pytest.mark.asyncio
async def test_filters_wrong_claim_keeps_full_sources_fields_and_counts(
    tmp_path, monkeypatch
):
    selected, receipt, captured = await verify(tmp_path, monkeypatch)
    assert set(selected.selected_items) == {"a"}
    assert selected.omitted_count == 1
    assert all(ids == ["a"] for ids in selected.section_evidence_ids.values())
    assert receipt["rejected"] == {"b": ["extraction_not_supported"]}
    assert receipt["candidate_ledger"] == selection().selected_items
    assert receipt["jev"]["usage"]["input_tokens"] == 100
    assert validate_extraction_receipt(receipt, selected.selected_items)
    assert captured
    for item in captured[0]["state"]["audit_items"].values():
        ev = item["cited_evidence"]
        ev = ev[0] if isinstance(ev, list) else ev
        assert (
            ev["text"]
            == selection().selected_items[ev["evidence_id"]]["supporting_text"]
        )
        if ev["evidence_id"] == "a":
            assert '"method": "EViT"' in item["atomic_claim"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["model", "usage", "omitted", "auth"])
async def test_protocol_failure_stops_and_is_not_replayed(tmp_path, monkeypatch, fault):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    captured = []
    async with httpx.AsyncClient(transport=transport(captured, fault)) as client:
        gate = ExtractionVerifier(tmp_path, auditor=JevCitationAuditor(client=client))
        with pytest.raises(ValueError, match="extraction_verification_incomplete"):
            await gate.verify(selection(), run_id="bad")
        record = json.loads((tmp_path / "bad" / "receipt.json").read_text())
        assert record["status"] == "failed_no_replay"
        assert record["jev"]["usage"]["call_count"] == 1
        with pytest.raises(ValueError, match="no_replay"):
            await gate.verify(selection(), run_id="bad")
        assert len(captured) == 1


@pytest.mark.asyncio
async def test_low_confidence_never_enters_writing_set(tmp_path, monkeypatch):
    selected, receipt, _ = await verify(tmp_path, monkeypatch, "low")
    assert not selected.selected_items
    assert receipt["rejected"]["a"] == ["extraction_low_confidence"]


@pytest.mark.asyncio
async def test_identical_completed_reuse_and_changed_input_rejected(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    captured = []
    async with httpx.AsyncClient(transport=transport(captured)) as client:
        gate = ExtractionVerifier(tmp_path, auditor=JevCitationAuditor(client=client))
        first = await gate.verify(selection(), run_id="cache")
        assert await gate.verify(selection(), run_id="cache") == first
        changed = selection()
        changed.selected_items["a"]["fact_fields"]["value"] = "70%"
        with pytest.raises(ValueError, match="identity_conflict"):
            await gate.verify(changed, run_id="cache")
        assert len(captured) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["text", "supporting_text", "paper_id"])
async def test_missing_sources_do_not_call_api(tmp_path, field):
    picked = selection()
    picked.selected_items["a"][field] = ""
    auditor = AsyncMock()
    with pytest.raises(ValueError, match="source_or_claim_missing"):
        await ExtractionVerifier(tmp_path, auditor=auditor).verify(
            picked, run_id="missing"
        )
    auditor.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_body_or_budget_overflow_preflight_has_no_requests(tmp_path):
    picked = selection()
    picked.selected_items["a"]["supporting_text"] = "original " * 24000
    auditor = AsyncMock()
    with pytest.raises(ValueError, match="budget_exceeded"):
        await ExtractionVerifier(tmp_path, auditor=auditor).verify(
            picked, run_id="large"
        )
    auditor.evaluate.assert_not_called()


@pytest.mark.asyncio
async def test_real_extractor_preserves_bad_fields_for_jev_to_reject(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    raw = "DeiT reports 81.8% accuracy on ImageNet; this is not a speed measurement."
    strategy = AsyncMock()
    strategy.extract.return_value = {
        "claims": ["DeiT improves speed by 81.8%."],
        "claim_records": [
            {
                "text": "DeiT improves speed by 81.8%.",
                "source_chunk_key": "c1",
                "fact_fields": {"method": "DeiT", "metric": "speed", "value": "81.8%"},
            }
        ],
    }
    loader = AsyncMock()
    loader.enrich.side_effect = lambda papers, _: papers
    extractor = ExtractorWorker(strategy, fulltext_loader=loader)
    output = await extractor.execute(
        SubTask(
            "extract",
            "test",
            "extractor",
            input_data={
                "quality_policy": "strict_auto_v2",
                "upstream_results": {
                    "dedup": [
                        {
                            "paper_id": "deit",
                            "title": "DeiT",
                            "chunks": [{"chunk_key": "c1", "text": raw}],
                        }
                    ]
                },
            },
        )
    )
    item = output[0]["evidence_items"][0]
    assert item["supporting_text"] == raw and item["fact_fields"]["metric"] == "speed"
    picked = selection()
    picked.selected_items = {item["evidence_id"]: item}
    picked.section_evidence_ids = {
        k: [item["evidence_id"]] for k in picked.section_evidence_ids
    }
    picked.candidate_count = 1
    captured = []

    def reject(request):
        captured.append(json.loads(request.content))
        response = _valid_response(request)
        for answer in response["answers"].values():
            options = list(answer["probabilities"])
            answer["choice"] = options[-1]
            answer["probabilities"] = {o: float(o == options[-1]) for o in options}
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
        filtered, receipt = await ExtractionVerifier(
            tmp_path, auditor=JevCitationAuditor(client=client)
        ).verify(picked, run_id="producer")
    assert not filtered.selected_items
    assert receipt["rejected"][item["evidence_id"]] == ["extraction_not_supported"]
    assert raw in json.dumps(captured)
    assert "81.8%" in json.dumps(captured) and "speed" in json.dumps(captured)


@pytest.mark.asyncio
async def test_cancellation_then_restart_never_dispatches_again(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    started = asyncio.Event()
    count = 0

    async def wait(request):
        nonlocal count
        count += 1
        started.set()
        await asyncio.Event().wait()

    async with httpx.AsyncClient(transport=httpx.MockTransport(wait)) as client:
        gate = ExtractionVerifier(tmp_path, auditor=JevCitationAuditor(client=client))
        task = asyncio.create_task(gate.verify(selection(), run_id="cancel"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        next_gate = ExtractionVerifier(
            tmp_path, auditor=JevCitationAuditor(client=client)
        )
        with pytest.raises(ValueError, match="no_replay"):
            await next_gate.verify(selection(), run_id="cancel")
        assert count == 1


@pytest.mark.asyncio
async def test_separate_verifier_duplicate_cannot_send_again(tmp_path, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-fake")
    started = asyncio.Event()
    release = asyncio.Event()
    captured = []

    async def slow(request):
        captured.append(request)
        started.set()
        await release.wait()
        return httpx.Response(200, json=_valid_response(request))

    async with httpx.AsyncClient(transport=httpx.MockTransport(slow)) as client:
        first = ExtractionVerifier(tmp_path, auditor=JevCitationAuditor(client=client))
        second = ExtractionVerifier(tmp_path, auditor=JevCitationAuditor(client=client))
        task = asyncio.create_task(first.verify(selection(), run_id="duplicate"))
        await started.wait()
        with pytest.raises(ValueError, match="no_replay"):
            await second.verify(selection(), run_id="duplicate")
        release.set()
        await task
        assert len(captured) == 1


@pytest.mark.asyncio
async def test_v2_benchmark_counts_front_jev_when_body_evaluation_fails(
    tmp_path, monkeypatch
):
    from test_quality_protocol_stop import live_runner, report

    _, receipt, _ = await verify(tmp_path / "extractions", monkeypatch)
    value = report("evaluation_assertions_unverifiable")
    value["metadata"]["extraction_verification"] = receipt
    runner, _ = live_runner(tmp_path / "suite", monkeypatch, [value])
    runner.quality_policy = "strict_auto_v2"

    # Exercise the actual consumer independently of the old fixture's run-option guard.
    class Agent:
        def __init__(self, *a, **kw):
            from litagent.safety.budget import CostBudget

            self._cost_budget = CostBudget()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            pass

        async def run(self, query, *, options):
            assert (
                options.quality_policy == "strict_auto_v2" and not options.human_review
            )
            return value

    monkeypatch.setattr("litagent.benchmark.quality_suite.LitAgent", Agent)
    result = await runner._live(runner.dataset.cases[0], runner.config, "local-v2")
    assert result["costs"]["jev"] == {
        "tokens": 100,
        "requests": 1,
        "usage_complete": True,
    }
    assert not result["complete_verification"]
    assert result["status"] == "blocked"
    assert result["costs"]["judge"]["requests"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["claim", "original", "fields", "decision", "status", "drop", "response", "usage"],
)
async def test_final_gate_revalidates_complete_receipt(tmp_path, monkeypatch, mutation):
    selected, receipt, _ = await verify(tmp_path, monkeypatch)
    receipt = copy.deepcopy(receipt)
    if mutation in {"claim", "original", "fields"}:
        key = {"claim": "text", "original": "supporting_text", "fields": "fact_fields"}[
            mutation
        ]
        receipt["candidate_ledger"]["a"][key] = (
            {"value": "70%"} if key == "fact_fields" else "changed"
        )
        receipt["candidate_sha256"] = digest(receipt["candidate_ledger"])
    elif mutation == "decision":
        receipt["jev"]["decisions"][0]["confidence"] = 0.99
    elif mutation == "status":
        receipt["status"] = "passed"
    elif mutation == "drop":
        receipt["jev"]["decisions"].pop()
    elif mutation == "response":
        receipt["jev"]["batches"][0]["response"]["model"] = "illegal"
    elif mutation == "usage":
        receipt["jev"]["usage"]["input_tokens"] = 0
    assert not validate_extraction_receipt(receipt, selected.selected_items)


@pytest.mark.asyncio
async def test_synthesis_uses_only_verified_selection_and_propagates_receipt(
    tmp_path, monkeypatch
):
    selected, receipt, _ = await verify(tmp_path, monkeypatch)
    cards = build_fact_cards(selected.selected_items)
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(document(cards)), model="offline", usage={}
    )
    selector = AsyncMock()
    selector.select.return_value = selection()
    gate = ExtractionVerifier(tmp_path)
    worker = SynthesisWorker(
        llm,
        evidence_selector=selector,
        extraction_verifier=gate,
        run_id_provider=lambda: "extract-test",
    )
    result = await worker.execute(
        SubTask(
            "synthesis",
            "test",
            "synthesis",
            input_data={
                "query": "EViT",
                "quality_policy": "strict_auto_v2",
                "review_structure": "academic_v2",
            },
        )
    )
    assert result["extraction_verification"] == receipt
    assert set(result["evidence_selection"]["selected_items"]) == {"a"}
    assert "[E:b]" not in result["draft"]
    assert result["fact_cards"] == cards
    assert "Evo-ViT improves FLOPs" not in str(llm.chat.call_args)


@pytest.mark.asyncio
async def test_upstream_failure_prevents_any_generation():
    llm = AsyncMock()
    gate = AsyncMock()
    gate.verify.side_effect = ValueError("extraction_verification_incomplete")
    selector = AsyncMock()
    selector.select.return_value = selection()
    worker = SynthesisWorker(
        llm,
        evidence_selector=selector,
        extraction_verifier=gate,
        run_id_provider=lambda: "failed",
    )
    with pytest.raises(ValueError, match="incomplete"):
        await worker.execute(
            SubTask(
                "synthesis",
                "test",
                "synthesis",
                input_data={"quality_policy": "strict_auto_v2"},
            )
        )
    llm.chat.assert_not_called()


def test_new_policy_is_strict_automatic_and_distinct_from_old_identity():
    v2 = RunOptions(
        quality_policy="strict_auto_v2", human_review=True, auto_audit_mode="judge_jev"
    )
    assert is_strict_policy(v2.quality_policy)
    assert not v2.human_review and v2.auto_audit_mode == "judge"
    assert v2.review_structure == "academic_v2" and v2.report_language == "zh-CN"
    v1 = RunOptions(quality_policy="strict_auto_v1")
    assert v1.auto_audit_mode == "judge_jev"
    assert v2.fingerprint() != v1.fingerprint()
    cfg = load_config()
    assert (
        build_config_summary(cfg, "strict_auto_v2")["fingerprint"]
        != build_config_summary(cfg, "strict_auto_v1")["fingerprint"]
    )


@pytest.mark.asyncio
async def test_final_judge_required_without_post_body_jev_and_sha_invalidates(
    tmp_path, monkeypatch
):
    selected, receipt, _ = await verify(tmp_path, monkeypatch, run_id="test-auto")
    prior = frozen()
    report = copy.deepcopy(prior["report"])
    report["metadata"].update(
        quality_policy="strict_auto_v2",
        extraction_verification=receipt,
        fact_cards=build_fact_cards(selected.selected_items),
    )
    report["metadata"]["evidence_snapshot_sha256"] = digest(
        report["metadata"]["fact_cards"]
    )
    report["evaluation"]["faithfulness"]["details"]["quality_policy"] = "strict_auto_v2"
    snapshot = freeze_automatic_inputs(
        report, selected.selected_items, run_id="test-auto"
    )
    audit = passing_audit(snapshot)
    audit["jev"] = {"status": "not_requested"}
    accepted = automatic_acceptance(snapshot, report["evaluation"], audit)
    assert accepted["publishable"] and set(accepted["gates"]) == {
        "extraction_jev",
        "evaluation",
        "judge",
    }
    audit["judge"]["faithfulness"] = 0.8
    assert not automatic_acceptance(snapshot, report["evaluation"], audit)[
        "publishable"
    ]
    report["metadata"]["automatic_acceptance"] = accepted
    report["delivery"] = {"publishable": True}
    report["metadata"]["extraction_verification"]["accepted_evidence_ids"] = []
    normalized = normalize_survey_result(report, config_fingerprint="offline")
    assert not normalized["delivery"]["publishable"]
    service = AsyncMock()
    agent = LitAgent(load_config(), audit_service=service)
    agent._session_id = "test-auto"
    agent._build_eval_context = lambda **_: {CTX_EVIDENCE: selected.selected_items}
    report["metadata"]["extraction_verification"] = receipt
    # Verify actual dispatch separately from final semantic judgments.
    monkeypatch.setattr(
        "litagent.eval.automatic.freeze_automatic_inputs", lambda *a, **kw: snapshot
    )
    service.submit.return_value = {"audit_id": "audit-test"}
    service.wait.return_value = audit
    monkeypatch.setattr("litagent.report_audit.atomic_json", lambda *a: None)
    await agent._automatic_review(report, "EViT", {}, None)
    assert service.submit.call_args.kwargs["include_jev"] is False
