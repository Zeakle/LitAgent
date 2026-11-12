"""Replay distinct measurements sharing a source through real evidence consumers."""

import copy
import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

from litagent.agents.extraction_strategy import LLMStrategy
from litagent.agents.extractor import ExtractorWorker
from litagent.benchmark.citation_audit import build_citation_audit_inputs
from litagent.evidence import build_evidence_items, collect_ledger
from litagent.extraction_verification import build_extraction_inventory
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import build_fact_cards


def real_case():
    return json.loads(
        (
            Path(__file__).parent / "fixtures/extraction-multi-metric-real.json"
        ).read_text()
    )


def strategy(payload, identity="source_fields_v1"):
    llm, skills = AsyncMock(), Mock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(payload), model="offline", usage={}
    )
    skills.to_metadata_text_for.return_value = ""
    return LLMStrategy(llm, skills, record_identity=identity)


@pytest.mark.asyncio
async def test_actual_multi_metric_response_reaches_cards_and_jev_without_overwrite():
    f = real_case()
    source = copy.deepcopy(f["paper"])
    loader = Mock()
    loader.enrich = AsyncMock(return_value=[source])
    worker = ExtractorWorker(strategy(f["response"]), fulltext_loader=loader)
    task = SubTask(
        task_id="extract",
        description="Replay the public multi-metric source",
        agent_type="extractor",
        input_data={
            "quality_policy": "strict_auto_v2",
            "upstream_results": {"relevance_gate": [source]},
        },
    )
    result = await worker.execute(task)
    ledger = collect_ledger(result)
    assert len(ledger) == 3
    assert [x["fact_fields"]["value"] for x in ledger.values()] == [
        "8.5%",
        "4.8%",
        "76.7%",
    ]
    assert len({x["evidence_id"] for x in ledger.values()}) == 3
    assert len({x["supporting_text"] for x in ledger.values()}) == 1
    cards = build_fact_cards(ledger)
    assert len(cards) == 3 and all(
        x["semantic_status"] == "not_verified" for x in cards.values()
    )
    assert {x["value"] for x in cards.values()} == {"8.5%", "4.8%", "76.7%"}
    # An inferred field is still sent to Jev, not hidden by card literal filtering.
    inventory = build_extraction_inventory(ledger)
    assert len(inventory.claims) == 3
    inputs = build_citation_audit_inputs(
        inventory, ledger, artifact={"quality_policy": "strict_auto_v2"}
    )
    assert len(inputs) == 3
    for item in inputs:
        assert item.evidence[0].text == next(iter(ledger.values()))["supporting_text"]
    assert (
        "absolute accuracy improvement over the state-of-the-art [17]"
        in inventory.claims[1].claim_text
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["exact", "alias", "confidence", "null-empty"])
async def test_same_association_remains_duplicate_even_with_display_changes(mutation):
    f = real_case()
    one = copy.deepcopy(f["response"]["claim_records"][0])
    other = copy.deepcopy(one)
    if mutation == "alias":
        other["source_chunk_key"] = "[CHUNK:" + other["source_chunk_key"] + "]"
    if mutation == "confidence":
        other["confidence"] = 0.5
    if mutation == "null-empty":
        other["fact_fields"]["conditions"] = "  "
    f["response"]["claim_records"] = [one, other]
    with pytest.raises(ValueError, match="duplicate_record"):
        await strategy(f["response"]).extract(f["paper"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field,value", [("value", 42), ("metric", []), ("invented", "unsafe")]
)
async def test_invalid_fields_are_not_a_way_to_create_an_extra_identity(field, value):
    f = real_case()
    f["response"]["claim_records"][0]["fact_fields"][field] = value
    with pytest.raises(ValueError, match="fact_field"):
        await strategy(f["response"]).extract(f["paper"])


@pytest.mark.asyncio
async def test_old_contract_still_rejects_same_source_text_and_keeps_old_ids():
    f = real_case()
    with pytest.raises(ValueError, match="duplicate_record"):
        await strategy(f["response"], None).extract(f["paper"])
    old = {**f["paper"], "claim_records": [f["response"]["claim_records"][0]]}
    item = build_evidence_items(old)[0]
    r = old["claim_records"][0]
    digest = hashlib.sha256(
        f"{r['text']}|{r['source_chunk_key']}".encode()
    ).hexdigest()[:12]
    assert item["evidence_id"].endswith(":claim:" + digest)
    assert "record_identity" not in item and "fact_fields" not in item


@pytest.mark.asyncio
async def test_identity_is_deterministic_under_record_order_and_field_key_order():
    f = real_case()
    original = await strategy(f["response"]).extract(f["paper"])
    original.update(f["paper"])
    before = {
        x["fact_fields"]["value"]: x["evidence_id"]
        for x in build_evidence_items(original)
    }
    payload = copy.deepcopy(f["response"])
    payload["claim_records"].reverse()
    for r in payload["claim_records"]:
        r["fact_fields"] = dict(reversed(list(r["fact_fields"].items())))
    second = await strategy(payload).extract(f["paper"])
    second.update(f["paper"])
    after = {
        x["fact_fields"]["value"]: x["evidence_id"]
        for x in build_evidence_items(second)
    }
    assert after == before


@pytest.mark.asyncio
async def test_distinct_identity_does_not_turn_a_false_value_into_verified_evidence(
    tmp_path, monkeypatch
):
    import httpx
    from test_jev_citation_audit import _valid_response

    from litagent.benchmark.citation_audit import JevCitationAuditor
    from litagent.context.evidence_selector import EvidenceSelection
    from litagent.extraction_verification import ExtractionVerifier

    fixture = real_case()
    fixture["response"]["claim_records"][0]["fact_fields"]["value"] = "999%"
    extracted = await strategy(fixture["response"]).extract(fixture["paper"])
    extracted.update(fixture["paper"])
    ledger = {i["evidence_id"]: i for i in build_evidence_items(extracted)}
    wrong_id = next(
        eid for eid, item in ledger.items() if item["fact_fields"]["value"] == "999%"
    )
    wrong_card = next(
        c for c in build_fact_cards(ledger).values() if c["evidence_id"] == wrong_id
    )
    assert wrong_card["value"] is None and "value" in wrong_card["missing_fields"]
    selected = EvidenceSelection(
        3,
        ledger,
        {
            k: list(ledger)
            for k in [
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            ]
        },
        {
            k: "lexical_fallback"
            for k in [
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            ]
        },
        0,
        100,
    )
    captured = []

    def respond(request):
        payload = json.loads(request.content)
        captured.append(payload)
        result = _valid_response(request)
        for qid, item in payload["state"]["audit_items"].items():
            if "999%" not in item["atomic_claim"]:
                continue
            answer = result["answers"][qid]
            negative = list(answer["probabilities"])[-1]
            answer["choice"] = negative
            answer["probabilities"] = {
                k: float(k == negative) for k in answer["probabilities"]
            }
        return httpx.Response(200, json=result)

    monkeypatch.setenv("TYPESAFE_API_KEY", "offline-test")
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        verifier = ExtractionVerifier(
            tmp_path, auditor=JevCitationAuditor(client=client)
        )
        actual, receipt = await verifier.verify(
            selected, run_id="distinct-record-negative"
        )
    assert len(actual.selected_items) == 2
    assert wrong_id not in actual.selected_items
    assert receipt["rejected"][wrong_id] == ["extraction_not_supported"]
    assert "999%" in json.dumps(captured)
