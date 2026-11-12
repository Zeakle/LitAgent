"""Keep semantic candidates, original support and multi-setting measurements distinct."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_batched_evaluation import BatchLLM, materials
from test_evaluation_wire_contract import AnalysisTransport
from test_extraction_paper_batches import strategy, task

from litagent.agents.extraction_strategy import (
    decode_field_spans,
    decode_grounded_claims,
)
from litagent.agents.extractor import ExtractorWorker
from litagent.benchmark.citation_audit import build_citation_audit_inputs
from litagent.body_quality import EXTRACTION_CLAIM_CONTRACT
from litagent.config import EvalConfig
from litagent.eval.scoped import evaluate_scoped
from litagent.evidence import collect_ledger
from litagent.extraction_verification import build_extraction_inventory
from litagent.llm.client import OpenAICompatibleClient
from litagent.quality import build_fact_cards, digest
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions


def source_and_candidates():
    """Construct protocol fixtures from the recorded public multi-backbone example."""
    fixture = json.loads(
        Path("tests/fixtures/extraction-metric-cases-real.json").read_text()
    )
    paper = {
        "paper_id": fixture["paper_id"],
        "title": fixture["title"],
        "abstract": fixture["sources"]["abstract"],
        "chunks": [
            {
                "chunk_key": "abstract",
                "text": fixture["sources"]["abstract"],
                "section": "abstract",
                "content_scope": "abstract",
            }
        ],
        "quality_policy": "strict_auto_v2",
        "_claim_contract": EXTRACTION_CLAIM_CONTRACT,
    }
    records = []
    for method, value in [
        ("ResNet-18", "98.93%"),
        ("ResNet-50", "98.60%"),
        ("VGG16", "33.33%"),
    ]:
        records.append(
            {
                "finding": f"Using {method} for feature extraction, the paper reports {value} classification accuracy on TBX11K.",
                "source_chunk_key": "abstract",
                "confidence": None,
                "fact_fields": {
                    "method": method,
                    "conditions": "feature extraction",
                    "dataset": "TBX11K Chest X-ray",
                    "metric": "classification accuracy",
                    "value": value,
                    "unit": "%",
                    "qualifiers": None,
                },
            }
        )
    return paper, {"claim_records": records}


def test_actual_duplicate_metric_names_are_distinct_unverified_associations():
    fixture = json.loads(
        Path("tests/fixtures/extraction-metric-cases-real.json").read_text()
    )
    original = copy.deepcopy(fixture)
    with pytest.raises(ValueError, match="metric_identity_invalid"):
        decode_field_spans(fixture["received_v2"], fixture["sources"], quote_spans=True)
    paper, raw = source_and_candidates()
    decoded, bindings = decode_grounded_claims(raw, {"abstract": paper["abstract"]})
    assert len(decoded["claim_records"]) == len(decoded["metrics"]) == 3
    assert {v["metric"] for v in decoded["metrics"].values()} == {
        "classification accuracy"
    }
    assert {(v["method"], v["value"]) for v in decoded["metrics"].values()} == {
        ("ResNet-18", "98.93%"),
        ("ResNet-50", "98.60%"),
        ("VGG16", "33.33%"),
    }
    assert all(
        v["semantic_status"] == "not_verified" for v in decoded["metrics"].values()
    )
    assert all(
        b["source_sha256"] == hashlib.sha256(paper["abstract"].encode()).hexdigest()
        for b in bindings
    )
    assert all(b["verification"] == "source_location_only" for b in bindings)
    assert fixture == original


@pytest.mark.asyncio
async def test_actual_sdk_worker_ledger_cards_and_jev_use_complete_original(
    monkeypatch,
):
    paper, raw = source_and_candidates()
    before = copy.deepcopy(paper)
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "https://offline.invalid/v1", "offline", api_key="offline", cost_budget=budget
    )
    wire = []

    async def create(**kwargs):
        wire.append(copy.deepcopy(kwargs))
        native = kwargs["response_format"]["json_schema"]
        assert native["name"] == "host_grounded_claims_v1" and native["strict"]
        Draft202012Validator(native["schema"]).validate({"papers": {"p0": raw}})
        payload = json.loads(kwargs["messages"][1]["content"])
        assert paper["abstract"] in payload["papers"]["p0"]["source_context"]
        assert "source_chunks" not in payload["papers"]["p0"]
        assert kwargs["max_tokens"] == 8192
        return SimpleNamespace(
            model="offline",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"papers": {"p0": raw}}), tool_calls=None
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )

    monkeypatch.setattr(
        client._client.chat.completions, "create", AsyncMock(side_effect=create)
    )
    worker = ExtractorWorker(
        strategy(client),
        strict_paper_batch_size=2,
        strict_grounded_claims=True,
        fulltext_loader=SimpleNamespace(enrich=AsyncMock(side_effect=lambda p, r: p)),
    )
    try:
        results = await worker.execute(task([paper]))
    finally:
        await client.close()
    assert (
        len(wire) == 1 and budget.snapshot()["requests_sent"] == 1 and budget.used == 20
    )
    ledger = collect_ledger(results)
    cards = build_fact_cards(ledger)
    assert len(ledger) == len(cards) == 3
    assert all(
        c["source_quote"] == paper["abstract"]
        and c["semantic_status"] == "not_verified"
        for c in cards.values()
    )
    assert all(c["finding_candidate"] != c["source_quote"] for c in cards.values())
    assert all(item["supporting_text"] == paper["abstract"] for item in ledger.values())
    inputs = build_citation_audit_inputs(
        build_extraction_inventory(ledger),
        ledger,
        artifact={"quality_policy": "strict_auto_v2"},
    )
    assert len(inputs) == 3
    for item in inputs:
        assert item.evidence[0].text == paper["abstract"]
        assert "every field and its association" in item.claim_text
        assert "TBX11K" in item.claim_text
    assert paper == before


@pytest.mark.parametrize(
    "fault",
    [
        "source",
        "empty_source",
        "source_type",
        "field_type",
        "missing_field",
        "finding_empty",
        "aggregate",
        "duplicate",
    ],
)
def test_malformed_candidates_fail_without_silently_dropping_measurements(fault):
    paper, raw = source_and_candidates()
    sources = {"abstract": paper["abstract"]}
    record = raw["claim_records"][0]
    if fault == "source":
        record["source_chunk_key"] = "foreign-source"
    elif fault == "empty_source":
        sources["abstract"] = " "
    elif fault == "source_type":
        sources["abstract"] = None
    elif fault == "field_type":
        record["fact_fields"]["value"] = 98.93
    elif fault == "missing_field":
        record["fact_fields"].pop("conditions")
    elif fault == "finding_empty":
        record["finding"] = " "
    elif fault == "aggregate":
        raw["metrics"] = {"accuracy": "invented summary"}
    else:
        raw["claim_records"].append(copy.deepcopy(record))
    with pytest.raises(ValueError):
        decode_grounded_claims(raw, sources)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["unknown_usage", "cancel", "truncated"])
async def test_grounded_sdk_failures_are_charged_and_not_retried(monkeypatch, fault):
    import asyncio

    paper, raw = source_and_candidates()
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
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
                        content=json.dumps({"papers": {"p0": raw}}), tool_calls=None
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
    monkeypatch.setattr(client._client.chat.completions, "create", wire)
    try:
        with pytest.raises(asyncio.CancelledError if fault == "cancel" else ValueError):
            await strategy(client).extract_batch([paper])
        assert wire.await_count == budget.snapshot()["requests_sent"] == 1
        if fault in {"unknown_usage", "cancel"}:
            assert budget.snapshot()["usage_uncertain"]
        else:
            assert budget.used == 20
    finally:
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [True, False])
@pytest.mark.parametrize("source_metric", ["accuracy", "f1"])
async def test_semantic_claim_is_never_verified_by_source_location_alone(
    supported, source_metric
):
    paper, raw = source_and_candidates()
    if source_metric == "f1":
        paper["abstract"] = paper["abstract"].replace(
            "classification accuracies", "F1 scores"
        )
        paper["chunks"][0]["text"] = paper["abstract"]
    raw["claim_records"] = raw["claim_records"][:1]
    if not supported:
        raw["claim_records"][0][
            "finding"
        ] = "Using ResNet-50 for feature extraction, the paper reports 98.93% classification accuracy on TBX11K."
        raw["claim_records"][0]["fact_fields"]["method"] = "ResNet-50"
    data, _ = decode_grounded_claims(raw, {"abstract": paper["abstract"]})
    llm = AsyncMock()
    llm.chat.return_value = SimpleNamespace(
        content=json.dumps({"papers": {"p0": raw}}),
        finish_reason="stop",
        usage={"prompt_tokens": 1, "completion_tokens": 1},
    )
    results = await ExtractorWorker(
        strategy(llm),
        strict_paper_batch_size=2,
        strict_grounded_claims=True,
        fulltext_loader=SimpleNamespace(enrich=AsyncMock(side_effect=lambda p, r: p)),
    ).execute(task([paper]))
    ledger = collect_ledger(results)
    facts = build_fact_cards(ledger)
    eid = next(iter(ledger))
    assert next(iter(facts.values()))["semantic_status"] == "not_verified"
    _, context = materials(1)
    context.update(
        body_quality="analysis_v1",
        quality_policy="strict_auto_v2",
        query="Reported backbone accuracy",
        evidence=ledger,
        fact_cards=facts,
        evidence_snapshot_sha256=digest(facts),
        coverage_requirements=[
            {
                "requirement_id": "question",
                "kind": "question",
                "text": "Reported backbone accuracy",
            }
        ],
    )
    survey = (
        "# Report\n\n## Experiments\n\n"
        + data["claim_records"][0]["text"]
        + f" [E:{eid}]"
    )

    class Responses(AnalysisTransport):
        async def chat(self, messages, **kwargs):
            response = await super().chat(messages, **kwargs)
            payload = self.calls[-1]
            if not supported and "cited_materials" in payload:
                result = json.loads(response.content)
                for d in result["decisions"]:
                    d.update(
                        status="unsupported",
                        reason_code="attribution_mismatch",
                        support_basis=[],
                    )
                    for aspect in d["aspect_checks"]:
                        aspect.update(supported=False, support_ids=[])
                response.content = json.dumps(result)
            return response

    result = await evaluate_scoped(
        llm=Responses(), survey=survey, context=context, config=EvalConfig()
    )
    assert result["factual_claim_count"] == 1
    if supported and source_metric == "f1":
        assert result["score"] is None and not result["coverage"]["complete"]
        assert result["assertions"][0]["local_error"] == "support_metric_unverified"
    else:
        assert result["coverage"]["complete"]
        assert result["score"] == (1.0 if supported else 0.0)


def test_new_claim_contract_changes_only_analysis_identity(monkeypatch):
    import litagent.body_quality as quality

    current = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    legacy = RunOptions()
    a, b = current.fingerprint(), legacy.fingerprint()
    monkeypatch.setattr(
        quality, "EXTRACTION_CLAIM_CONTRACT", "different-grounded-contract"
    )
    assert current.fingerprint() != a and legacy.fingerprint() == b


def test_metric_inflection_is_narrow_and_keeps_legacy_numeric_defaults():
    """Treat one inflection consistently without mapping arbitrary metrics."""
    from litagent.eval.scoped import _numeric_facts

    source = "The reported classification accuracies were 98.93%."
    assert _numeric_facts(source)[0][2] is None
    assert _numeric_facts(source, metric_inflections=True)[0][2] == "accuracy"
    assert (
        _numeric_facts("The F1 scores were 98.93%.", metric_inflections=True)[0][2]
        is None
    )


def test_metric_rule_version_is_bound_only_to_new_analysis_identity(monkeypatch):
    import litagent.body_quality as quality

    current = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    legacy = RunOptions()
    a, b = current.fingerprint(), legacy.fingerprint()
    monkeypatch.setattr(quality, "METRIC_INFLECTION_CONTRACT", "different-metric-rule")
    assert current.fingerprint() != a and legacy.fingerprint() == b


def test_candidate_binding_matches_normalized_ledger_text_without_touching_original():
    paper, raw = source_and_candidates()
    raw["claim_records"] = raw["claim_records"][:1]
    raw["claim_records"][0]["finding"] = (
        "  " + raw["claim_records"][0]["finding"] + "\n"
    )
    raw["claim_records"][0]["fact_fields"]["method"] = " ResNet-18 "
    before = copy.deepcopy(raw)
    data, bindings = decode_grounded_claims(raw, {"abstract": paper["abstract"]})
    record = data["claim_records"][0]
    assert record["text"] == before["claim_records"][0]["finding"].strip()
    assert record["fact_fields"]["method"] == "ResNet-18"
    assert (
        bindings[0]["claim_sha256"]
        == hashlib.sha256(record["text"].encode()).hexdigest()
    )
    assert (
        bindings[0]["source_sha256"]
        == hashlib.sha256(paper["abstract"].encode()).hexdigest()
    )
    assert raw == before
