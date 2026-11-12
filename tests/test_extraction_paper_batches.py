"""Exercise paper batches through real normalization, workers and SDK accounting."""

import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from litagent.agents.extraction_strategy import (
    LLMStrategy,
    ResilientExtractionStrategy,
    build_extraction_context,
)
from litagent.agents.extractor import ExtractorWorker
from litagent.benchmark.citation_audit import build_citation_audit_inputs
from litagent.evidence import collect_ledger
from litagent.extraction_verification import build_extraction_inventory
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import build_fact_cards
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions


def sources():
    """Use public original texts with deliberately identical host chunk keys."""
    first = json.loads(
        Path("tests/fixtures/extraction-multi-metric-real.json").read_text()
    )
    p0, r0 = copy.deepcopy(first["paper"]), copy.deepcopy(first["response"])
    # The received model joined disjoint phrases in this one field. The new
    # positive fixture leaves it unknown; a separate negative retains that bug.
    r0["claim_records"][1]["fact_fields"]["metric"] = None
    key = r0["claim_records"][0]["source_chunk_key"]
    for chunk in p0["chunks"]:
        if chunk["chunk_key"] == key:
            chunk["chunk_key"] = "shared-abstract"
    for record in r0["claim_records"]:
        record["source_chunk_key"] = "shared-abstract"
    cards = json.loads(Path("tests/fixtures/native-unit-scope-real.json").read_text())[
        "cards"
    ]
    card = next(c for c in cards.values() if c["paper_id"] != p0["paper_id"])
    quote = card["source_quote"]
    p1 = {
        "paper_id": card["paper_id"],
        "title": "Public source used only for transport regression",
        "abstract": quote,
        "chunks": [
            {
                "chunk_key": "shared-abstract",
                "text": quote,
                "section": "abstract",
                "content_scope": "abstract",
            }
        ],
    }
    r1 = {
        "claim_records": [
            {"text": quote, "source_chunk_key": "shared-abstract", "fact_fields": {}}
        ],
        "metrics": {},
        "methods": [],
        "datasets": [],
    }
    for paper in (p0, p1):
        paper["quality_policy"] = "strict_auto_v2"
    return [p0, p1], {"p0": r0, "p1": r1}


def strategy(llm):
    skills = Mock()
    skills.to_metadata_text_for.return_value = "Offline skill metadata"
    return LLMStrategy(llm, skills, record_identity="source_fields_v1")


def response(result):
    return LLMResponse(
        json.dumps({"papers": result}),
        model="offline",
        usage={"prompt_tokens": 10, "completion_tokens": 10},
        finish_reason="stop",
    )


def task(papers, body="analysis_v1"):
    return SubTask(
        "extract",
        "Extract public sources",
        "extractor",
        input_data={
            "quality_policy": "strict_auto_v2",
            "body_quality": body,
            "upstream_results": {"relevance_gate": papers},
        },
    )


def test_paper_batch_version_is_bound_to_analysis_but_not_legacy(monkeypatch):
    import litagent.body_quality as quality

    current = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    legacy = RunOptions()
    before, old = current.fingerprint(), legacy.fingerprint()
    monkeypatch.setattr(quality, "PAPER_BATCH_CONTRACT", "different-batch-contract")
    assert current.fingerprint() != before
    assert legacy.fingerprint() == old


@pytest.mark.asyncio
async def test_real_worker_retains_all_associations_sources_cards_and_jev_input():
    papers, payload = sources()
    untouched = copy.deepcopy(papers)
    llm = AsyncMock()
    llm.chat.return_value = response(payload)
    regex = AsyncMock()
    adapter = ResilientExtractionStrategy(strategy(llm), regex)
    loader = SimpleNamespace(
        enrich=AsyncMock(side_effect=lambda ps, _: copy.deepcopy(ps))
    )
    worker = ExtractorWorker(adapter, fulltext_loader=loader, strict_paper_batch_size=2)
    results = await worker.execute(task(papers))
    assert papers == untouched
    assert [r["paper_id"] for r in results] == [p["paper_id"] for p in papers]
    assert llm.chat.await_count == 1
    sent, kwargs = llm.chat.call_args.args[0], llm.chat.call_args.kwargs
    assert kwargs["max_tokens"] == 16384 and kwargs["transport_max_attempts"] == 1
    for index, paper in enumerate(papers):
        expected, keys = build_extraction_context(paper)
        item = json.loads(sent[1]["content"])["papers"][f"p{index}"]
        assert item["source_context"] == expected
        assert item["allowed_source_chunk_keys"] == keys
    ledger = collect_ledger(results)
    assert len(ledger) == 4
    cards = build_fact_cards(ledger)
    assert len(cards) == 4 and all(
        c["semantic_status"] == "not_verified" for c in cards.values()
    )
    assert [r["extraction_batch"]["member_key"] for r in results] == ["p0", "p1"]
    assert len({r["extraction_batch"]["response_sha256"] for r in results}) == 1
    inventory = build_extraction_inventory(ledger)
    inputs = build_citation_audit_inputs(
        inventory, ledger, artifact={"quality_policy": "strict_auto_v2"}
    )
    assert len(inputs) == 4
    for item in inputs:
        assert item.evidence[0].text in {v["supporting_text"] for v in ledger.values()}
    regex.extract.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "extra",
        "duplicate",
        "foreign_quote",
        "foreign_key",
        "foreign_field",
        "shape",
        "truncated",
    ],
)
async def test_any_bad_member_rejects_whole_batch_without_fallback_or_retry(fault):
    papers, data = sources()
    if fault == "missing":
        del data["p1"]
    elif fault == "extra":
        data["p2"] = copy.deepcopy(data["p1"])
    elif fault == "foreign_quote":
        data["p0"]["claim_records"][0]["text"] = data["p1"]["claim_records"][0]["text"]
    elif fault == "foreign_key":
        data["p1"]["claim_records"][0]["source_chunk_key"] = "other-source"
    elif fault == "foreign_field":
        data["p0"]["claim_records"][0]["fact_fields"][
            "method"
        ] = "invented-foreign-method"
    elif fault == "shape":
        data["p1"] = []
    result = response(data)
    if fault == "duplicate":
        result.content = '{"papers":{"p0":{},"p0":{},"p1":{}}}'
    if fault == "truncated":
        result.finish_reason = "length"
    llm, regex = AsyncMock(), AsyncMock()
    llm.chat.return_value = result
    adapter = ResilientExtractionStrategy(strategy(llm), regex)
    with pytest.raises(ValueError):
        await adapter.extract_batch(papers)
    assert llm.chat.await_count == 1
    regex.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_cancel_does_not_fall_back_or_retry():
    papers, _ = sources()
    llm, regex = AsyncMock(), AsyncMock()
    llm.chat.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await ResilientExtractionStrategy(strategy(llm), regex).extract_batch(papers)
    assert llm.chat.await_count == 1
    regex.extract.assert_not_awaited()


@pytest.mark.asyncio
async def test_real_received_noncontiguous_field_is_rejected_not_silently_repaired():
    papers, data = sources()
    original = json.loads(
        Path("tests/fixtures/extraction-multi-metric-real.json").read_text()
    )
    data["p0"]["claim_records"][1]["fact_fields"]["metric"] = original["response"][
        "claim_records"
    ][1]["fact_fields"]["metric"]
    llm = AsyncMock()
    llm.chat.return_value = response(data)
    with pytest.raises(ValueError, match="extraction_batch_field_source_invalid"):
        await strategy(llm).extract_batch(papers)
    assert llm.chat.await_count == 1


@pytest.mark.asyncio
async def test_empty_member_is_retained_and_missing_ids_stop_before_transport():
    papers, data = sources()
    data["p1"]["claim_records"] = []
    llm = AsyncMock()
    llm.chat.return_value = response(data)
    result = await strategy(llm).extract_batch(papers)
    assert len(result) == 2 and result[1]["claim_records"] == []
    papers[1]["paper_id"] = papers[0]["paper_id"]
    with pytest.raises(ValueError, match="paper_identity"):
        await strategy(llm).extract_batch(papers)
    assert llm.chat.await_count == 1


@pytest.mark.asyncio
async def test_legacy_body_keeps_individual_strategy_calls():
    papers, _ = sources()
    adapter = AsyncMock()
    adapter.extract.side_effect = [{"claims": [], "claim_records": []} for _ in papers]
    worker = ExtractorWorker(
        adapter,
        strict_paper_batch_size=2,
        fulltext_loader=SimpleNamespace(enrich=AsyncMock(side_effect=lambda ps, _: ps)),
    )
    result = await worker.execute(task(papers, body="legacy"))
    assert len(result) == adapter.extract.await_count == 2
    adapter.extract_batch.assert_not_awaited()


class BatchTransport:
    """Return per-member verbatim findings with controlled completion order."""

    def __init__(self):
        self.calls, self.active, self.max_active = [], 0, 0

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        items = json.loads(messages[1]["content"])["papers"]
        data = {
            key: {
                "claim_records": [
                    {
                        "text": "Public finding for " + item["paper_id"],
                        "source_chunk_key": "abstract",
                        "fact_fields": {},
                    }
                ],
                "metrics": {},
                "methods": [],
                "datasets": [],
            }
            for key, item in items.items()
        }
        self.active -= 1
        return response(data)


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [49, 50])
async def test_all_papers_reach_ledger_in_rank_order_with_25_transports(count):
    papers = [
        {
            "paper_id": f"p{i}",
            "title": "Offline paper",
            "abstract": f"Public finding for p{i}",
            "chunks": [
                {
                    "chunk_key": "abstract",
                    "text": f"Public finding for p{i}",
                    "section": "abstract",
                }
            ],
        }
        for i in range(count)
    ]
    llm = BatchTransport()
    worker = ExtractorWorker(
        strategy(llm),
        strict_paper_batch_size=2,
        fulltext_loader=SimpleNamespace(enrich=AsyncMock(side_effect=lambda ps, _: ps)),
    )
    result = await worker.execute(task(papers))
    assert len(llm.calls) == 25 and llm.max_active <= 2
    assert [r["paper_id"] for r in result] == [p["paper_id"] for p in papers]
    assert len(collect_ledger(result)) == count
    assert llm.calls[-1][1]["max_tokens"] == (8192 if count % 2 else 16384)


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_real_sdk_accounts_one_lite_call_and_never_retries_unknown_usage(
    monkeypatch, unknown
):
    papers, data = sources()
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "doubao-seed-2-1-lite-260915",
        api_key="offline",
        cost_budget=budget,
    )

    async def create(**kwargs):
        assert kwargs["model"] == "doubao-seed-2-1-lite-260915"
        assert kwargs["max_tokens"] == 16384
        assert kwargs["extra_body"]["thinking"]["type"] == "disabled"
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(
                        content=response(data).content, tool_calls=None
                    ),
                )
            ],
            usage=(
                None
                if unknown
                else SimpleNamespace(prompt_tokens=20, completion_tokens=10)
            ),
        )

    wire = AsyncMock(side_effect=create)
    monkeypatch.setattr(client._client.chat.completions, "create", wire)
    try:
        if unknown:
            with pytest.raises(ValueError, match="usage"):
                await strategy(client).extract_batch(papers)
            assert budget.snapshot()["usage_uncertain"]
        else:
            result = await strategy(client).extract_batch(papers)
            assert len(result) == 2
            assert budget.snapshot()["used_tokens"] == 30
        assert wire.await_count == budget.snapshot()["requests_sent"] == 1
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_worker_failure_or_cancel_stops_queued_sdk_batches_without_refund(
    monkeypatch, cancel
):
    source, payload = sources()
    papers = [
        {**copy.deepcopy(source[1]), "paper_id": f"offline-queued-{i}"}
        for i in range(8)
    ]
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "doubao-seed-2-1-lite-260915",
        api_key="offline",
        cost_budget=budget,
    )
    entered = asyncio.Event()
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            entered.set()
        if cancel:
            await asyncio.Event().wait()
        await asyncio.sleep(0)
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content='{"papers":{}}', tool_calls=None),
                )
            ],
            usage=SimpleNamespace(prompt_tokens=20, completion_tokens=10),
        )

    monkeypatch.setattr(
        client._client.chat.completions, "create", AsyncMock(side_effect=create)
    )
    worker = ExtractorWorker(
        strategy(client),
        strict_paper_batch_size=2,
        fulltext_loader=SimpleNamespace(enrich=AsyncMock(side_effect=lambda ps, _: ps)),
        failure_callback=budget.stop_after_worker_failure,
    )
    run = asyncio.create_task(worker.execute(task(papers)))
    try:
        if cancel:
            await asyncio.wait_for(entered.wait(), 2)
            run.cancel()
            with pytest.raises(asyncio.CancelledError):
                await run
            assert budget.snapshot()["usage_uncertain"]
        else:
            with pytest.raises(ValueError, match="member_coverage"):
                await run
            assert budget.snapshot()["used_tokens"] == 60
            assert budget.stop_reason == "strict_worker_failed"
        assert len(calls) == budget.snapshot()["requests_sent"] == 2
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_complete_context_overflow_refuses_before_transport():
    papers, _ = sources()
    llm = AsyncMock()
    adapter = strategy(llm)
    adapter._skill_manager.to_metadata_text_for.return_value = "x" * 120000
    with pytest.raises(ValueError, match="context_budget_exceeded"):
        await adapter.extract_batch(papers)
    llm.chat.assert_not_awaited()
