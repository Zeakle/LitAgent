"""Restore fact fields from host positions while retaining semantic verification."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_extraction_paper_batches import response, strategy, task

from litagent.agents.extraction_strategy import (
    decode_field_spans,
    field_source_tokens,
)
from litagent.agents.extractor import ExtractorWorker
from litagent.body_quality import EXTRACTION_FIELD_SPAN_CONTRACT
from litagent.evidence import collect_ledger
from litagent.llm.client import OpenAICompatibleClient
from litagent.quality import build_fact_cards
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.workbench import RunOptions

FIELDS = ("method", "conditions", "dataset", "metric", "value", "unit", "qualifiers")


def real_sources(index=0):
    """Keep original public contexts and all received invalid fields unchanged."""
    fixture = json.loads(
        Path("tests/fixtures/extraction-nonliteral-fields-real.json").read_text()
    )
    row = fixture["batches"][index]
    papers = []
    for data in row["input"]["papers"].values():
        context = data["source_context"]
        source = context.split("section=abstract\n", 1)[1]
        papers.append(
            {
                "paper_id": data["paper_id"],
                "title": context.splitlines()[0][7:],
                "abstract": source,
                "quality_policy": "strict_auto_v2",
                "_field_span_contract": EXTRACTION_FIELD_SPAN_CONTRACT,
                "chunks": [
                    {"chunk_key": "abstract", "section": "abstract", "text": source}
                ],
            }
        )
    return papers, copy.deepcopy(row["response"])


def selected_response(papers):
    """Construct a new positional transport fixture, never patch the old response."""
    result = {}
    for i, paper in enumerate(papers):
        source = paper["abstract"]
        quote = source.split(". ", 1)[0] + ("." if ". " in source else "")
        tokens = field_source_tokens(source)
        end = next(j for j, t in enumerate(tokens) if t[2] == len(quote))
        fields = {key: None for key in FIELDS}
        fields["qualifiers"] = {"first_token": 0, "last_token": end}
        result[f"p{i}"] = {
            "claim_records": [
                {
                    "source_quote_span": {"first_token": 0, "last_token": end},
                    "source_chunk_key": "abstract",
                    "confidence": None,
                    "fact_fields": fields,
                }
            ],
            "metrics": [],
            "methods": [],
            "datasets": [],
        }
    return result


def test_token_positions_preserve_unicode_punctuation_and_whitespace():
    source = "方法 A\t提高 1.3%\n  on CUB-200—not speed."
    tokens = field_source_tokens(source)
    assert all(source[start:end] == token for token, start, end in tokens)
    raw = {
        "metrics": [],
        "claim_records": [
            {"source_chunk_key": "s", "fact_fields": {k: None for k in FIELDS}}
        ],
    }
    start = next(i for i, t in enumerate(tokens) if t[0] == "1")
    end = next(i for i, t in enumerate(tokens) if t[0] == "%")
    raw["claim_records"][0]["fact_fields"]["value"] = {
        "first_token": start,
        "last_token": end,
    }
    before = copy.deepcopy(raw)
    decoded, bindings = decode_field_spans(raw, {"s": source})
    assert decoded["claim_records"][0]["fact_fields"]["value"] == "1.3%"
    assert bindings[0]["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert raw == before


@pytest.mark.asyncio
@pytest.mark.parametrize("batch", [0, 1])
async def test_received_nonliteral_strings_remain_rejected_in_both_contracts(batch):
    papers, received = real_sources(batch)
    llm = AsyncMock()
    llm.chat.return_value = response(received["papers"])
    with pytest.raises(ValueError, match="extraction_field_span_response_invalid"):
        await strategy(llm).extract_batch(papers)
    for p in papers:
        p.pop("_field_span_contract")
    with pytest.raises(ValueError, match="extraction_batch_field_source_invalid"):
        await strategy(llm).extract_batch(papers)
    assert llm.chat.await_count == 2


@pytest.mark.asyncio
async def test_actual_sdk_worker_ledger_and_cards_keep_position_only_status(
    monkeypatch,
):
    papers, _ = real_sources()
    original = copy.deepcopy(papers)
    data = selected_response(papers)
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    llm = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    wire = []

    async def create(**kwargs):
        wire.append(copy.deepcopy(kwargs))
        schema = kwargs["response_format"]["json_schema"]["schema"]
        Draft202012Validator(schema).validate({"papers": data})
        assert kwargs["max_tokens"] == 16384
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        return SimpleNamespace(
            model="offline",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"papers": data}), tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

    monkeypatch.setattr(
        llm._client.chat.completions, "create", AsyncMock(side_effect=create)
    )
    worker = ExtractorWorker(
        strategy(llm),
        strict_paper_batch_size=2,
        strict_field_spans=True,
        fulltext_loader=SimpleNamespace(enrich=AsyncMock(side_effect=lambda p, r: p)),
    )
    try:
        results = await worker.execute(task(papers))
    finally:
        await llm.close()
    assert (
        len(wire) == 1 and budget.snapshot()["requests_sent"] == 1 and budget.used == 20
    )
    ledger = collect_ledger(results)
    cards = build_fact_cards(ledger)
    assert len(cards) == 2
    assert all(c["semantic_status"] == "not_verified" for c in cards.values())
    for i, item in enumerate(results):
        record = item["claim_records"][0]
        binding = next(
            b for b in item["field_span_bindings"] if b["field"] == "qualifiers"
        )
        source = papers[i]["abstract"]
        assert (
            source[binding["source_start"] : binding["source_end"]]
            == record["fact_fields"]["qualifiers"]
        )
        assert binding["verification"] == "source_location_only"
        assert item["field_span_contract"] == EXTRACTION_FIELD_SPAN_CONTRACT
    payload = json.loads(wire[0]["messages"][1]["content"])
    for i, paper in enumerate(papers):
        source = payload["papers"][f"p{i}"]["source_chunks"]["abstract"]
        assert (
            "".join(piece for _, block in source["token_blocks"] for piece in block)
            == paper["abstract"]
        )
    assert papers == original
    assert (
        strict_request_reservation(
            wire[0]["messages"], 16384, wire[0]["response_format"]
        )
        < 117964
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "negative",
        "boolean",
        "reversed",
        "overflow",
        "string",
        "extra",
        "foreign_chunk",
        "missing_field",
        "unknown_record_key",
    ],
)
async def test_malformed_or_foreign_selectors_never_reach_evidence(fault):
    papers, _ = real_sources()
    data = selected_response(papers)
    r = data["p0"]["claim_records"][0]
    span = r["fact_fields"]["qualifiers"]
    if fault == "negative":
        span["first_token"] = -1
    elif fault == "boolean":
        span["first_token"] = True
    elif fault == "reversed":
        span.update(first_token=4, last_token=0)
    elif fault == "overflow":
        span["last_token"] = 99999
    elif fault == "string":
        r["fact_fields"]["qualifiers"] = "invented result"
    elif fault == "extra":
        span["text"] = "forged"
    elif fault == "foreign_chunk":
        r["source_chunk_key"] = "other-paper-source"
    elif fault == "missing_field":
        r["fact_fields"].pop("method")
    else:
        r["host_verified"] = True
    llm = AsyncMock()
    llm.chat.return_value = response(data)
    with pytest.raises(ValueError, match="extraction_field_span_"):
        await strategy(llm).extract_batch(papers)
    assert llm.chat.await_count == 1


def test_field_selector_version_affects_only_analysis_identity(monkeypatch):
    import litagent.body_quality as quality

    current = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    legacy = RunOptions()
    a, b = current.fingerprint(), legacy.fingerprint()
    monkeypatch.setattr(quality, "EXTRACTION_FIELD_SPAN_CONTRACT", "different-spans")
    assert current.fingerprint() != a and legacy.fingerprint() == b


@pytest.mark.asyncio
async def test_token_directory_and_native_schema_cannot_bypass_budget():
    """Count all locating material and native schema before any transport."""
    papers, _ = real_sources()
    llm = AsyncMock()
    adapter = strategy(llm)
    messages, kwargs, _ = adapter.prepare_batch(papers)
    reservation = strict_request_reservation(messages, 16384, kwargs["response_format"])
    assert reservation > strict_request_reservation(messages, 16384)
    llm._cost_budget = CostBudget(max_tokens=reservation - 1)
    with pytest.raises(ValueError, match="evaluation_run_budget_exceeded"):
        await adapter.extract_batch(papers)
    llm.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unknown_usage", "truncated", "cancel"])
async def test_native_selector_sdk_failures_remain_paid_and_never_retry(
    monkeypatch, fault
):
    """Native positioning does not release unknown requests or substitute output."""
    import asyncio

    papers, _ = real_sources()
    data = selected_response(papers)
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    llm = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )

    async def create(**kwargs):
        if fault == "cancel":
            raise asyncio.CancelledError()
        return SimpleNamespace(
            model="offline",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"papers": data}), tool_calls=None
                    ),
                    finish_reason="length" if fault == "truncated" else "stop",
                )
            ],
            usage=(
                None
                if fault == "unknown_usage"
                else SimpleNamespace(prompt_tokens=10, completion_tokens=10)
            ),
        )

    wire = AsyncMock(side_effect=create)
    monkeypatch.setattr(llm._client.chat.completions, "create", wire)
    try:
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else ValueError):
            await strategy(llm).extract_batch(papers)
        assert wire.await_count == budget.snapshot()["requests_sent"] == 1
        if fault in {"unknown_usage", "cancel"}:
            assert budget.snapshot()["usage_uncertain"]
        else:
            assert budget.used == 20
    finally:
        await llm.close()


@pytest.mark.parametrize("duplicate", [False, True])
def test_native_metric_entries_roundtrip_without_duplicate_overwrite(duplicate):
    """Keep arbitrary metric names without open objects in the native schema."""
    raw = {
        "claim_records": [],
        "metrics": [
            {"name": "准确率", "value": 85.7},
            {"name": "speed", "value": "2×"},
        ],
    }
    if duplicate:
        raw["metrics"].append({"name": "准确率", "value": 90})
        with pytest.raises(ValueError, match="metric_identity_invalid"):
            decode_field_spans(raw, {})
    else:
        decoded, bindings = decode_field_spans(raw, {})
        assert decoded["metrics"] == {"准确率": 85.7, "speed": "2×"}
        assert bindings == [] and isinstance(raw["metrics"], list)


@pytest.mark.parametrize(
    "fault",
    [
        "negative",
        "boolean",
        "reversed",
        "overflow",
        "string",
        "null",
        "missing",
        "free_text",
    ],
)
@pytest.mark.asyncio
async def test_quote_selectors_reject_invalid_positions_or_free_text(fault):
    """Reject bad passage selectors before a partial batch can enter the ledger."""
    papers, _ = real_sources()
    data = selected_response(papers)
    record = data["p0"]["claim_records"][0]
    span = record["source_quote_span"]
    if fault == "negative":
        span["first_token"] = -1
    elif fault == "boolean":
        span["last_token"] = True
    elif fault == "reversed":
        span.update(first_token=5, last_token=0)
    elif fault == "overflow":
        span["last_token"] = 999999
    elif fault == "string":
        record["source_quote_span"] = "copied text"
    elif fault == "null":
        record["source_quote_span"] = None
    elif fault == "missing":
        record.pop("source_quote_span")
    else:
        record["text"] = "model-supplied text"
    llm = AsyncMock()
    llm.chat.return_value = response(data)
    with pytest.raises(ValueError, match="extraction_field_span_"):
        await strategy(llm).extract_batch(papers)
    assert llm.chat.await_count == 1


def test_actual_v1_quote_failures_remain_unchanged_but_v2_restores_original():
    """Replay received paraphrases without normalizing them into accepted originals."""
    import re

    fixture = json.loads(
        Path("tests/fixtures/extraction-nonliteral-quotes-real.json").read_text()
    )
    untouched = copy.deepcopy(fixture)
    failures = []
    for batch in fixture["batches"]:
        for member, raw in batch["response"]["papers"].items():
            ctx = batch["input"]["papers"][member]["source_context"]
            markers = list(
                re.finditer(r"^\[CHUNK:([^\]]+)\] section=[^\n]+\n", ctx, re.MULTILINE)
            )
            sources = {
                m[1]: ctx[
                    m.end() : (
                        markers[i + 1].start() - 2 if i + 1 < len(markers) else len(ctx)
                    )
                ]
                for i, m in enumerate(markers)
            }
            old, _ = decode_field_spans(raw, sources)
            for original, decoded in zip(
                raw["claim_records"], old["claim_records"], strict=True
            ):
                assert decoded["text"] == original["text"]
                source = sources[original["source_chunk_key"]]
                if original["text"] in source:
                    continue
                failures.append(original["text"])
                tokens = field_source_tokens(source)
                new = {
                    "claim_records": [
                        {
                            "source_chunk_key": original["source_chunk_key"],
                            "source_quote_span": {
                                "first_token": 0,
                                "last_token": len(tokens) - 1,
                            },
                            "fact_fields": {key: None for key in FIELDS},
                        }
                    ],
                    "metrics": [],
                }
                restored, bindings = decode_field_spans(new, sources, quote_spans=True)
                assert (
                    restored["claim_records"][0]["text"]
                    == source[tokens[0][1] : tokens[-1][2]]
                )
                assert bindings[0]["field"] == "source_quote"
                assert bindings[0]["verification"] == "source_location_only"
    assert len(failures) == 2
    assert fixture == untouched


def test_v2_prompt_and_native_schema_never_require_free_quote_copying():
    """The final provider contract and prompt request the same host selector."""
    papers, _ = real_sources()
    messages, kwargs, _ = strategy(AsyncMock()).prepare_batch(papers)
    schema = kwargs["response_format"]["json_schema"]["schema"]
    record = schema["properties"]["papers"]["properties"]["p0"]["properties"][
        "claim_records"
    ]["items"]
    assert (
        "source_quote_span" in record["required"] and "text" not in record["properties"]
    )
    assert "claim_records.text MUST" not in messages[0]["content"]
    assert "Each record has exactly source_quote_span" in messages[0]["content"]
    assert (
        kwargs["response_format"]["json_schema"]["name"]
        == "host_extraction_field_spans_v2"
    )
