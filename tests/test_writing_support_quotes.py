"""Locate writer quotations without treating them as scientific verification."""

import copy
import hashlib
import json
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_chapter_scope_contract import materials, unit
from test_fragment_repair import case
from test_native_writing_schema import section_definition
from test_quality_protocol_stop import live_runner, report
from test_strict_quality import to_native, valid_details

from litagent.agents.synthesis import SynthesisWorker
from litagent.benchmark.quality_suite import public_projection
from litagent.llm.client import LLMResponse
from litagent.observability.recorder import RedactingTraceHook
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    GroundedWritingError,
    _locate_writing_support_quotes,
    apply_fragment_repairs,
    build_fact_cards,
    grounded_messages,
    prepare_grounded_request,
    render_grounded_report,
    render_grounded_report_traced,
    strict_acceptance,
)


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_located_quotes_preserve_card_identity_ranges_and_private_trace(language):
    cards, selection, authored = materials(language=language)
    native = to_native(authored, cards)
    raw = json.dumps(native, ensure_ascii=False)
    before = copy.deepcopy((cards, selection.to_dict(), native))
    events, remote = [], []
    redact = RedactingTraceHook(
        lambda name, data: remote.append((name, data)), "metadata_only"
    )

    def record(name, data):
        events.append((name, data))
        redact(name, data)

    body = render_grounded_report_traced(
        raw,
        cards,
        "academic_v2",
        language,
        evidence_selection=selection,
        require_native=True,
        require_source_quotes=True,
        trace_hook=record,
    )
    event, data = events[0]
    assert event == "writing.support.located" and data["count"] == 7
    assert data["candidate_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    for location in data["bindings"]:
        card = cards[location["fact_card_id"]]
        assert location["paper_id"] == card["paper_id"]
        assert location["evidence_id"] == card["evidence_id"]
        assert (
            card["source_quote"][location["quote_start"] : location["quote_end"]]
            == location["source_quote"]
        )
        assert location["verification"] == "source_location_only"
        assert location["offset_scope"] == "fact_card_source_quote"
        assert "scientifically_verified" not in location
    assert remote[0][1]["count"] == 7 and "bindings" not in remote[0][1]
    assert "EViT improves speed" not in json.dumps(remote)
    assert "support_quotes" not in body and "source_location_only" not in body
    assert (cards, selection.to_dict(), native) == before


@pytest.mark.parametrize(
    "mode",
    [
        "missing",
        "wrong_card",
        "paraphrase",
        "disjoint",
        "duplicate",
        "empty",
        "forged_position",
    ],
)
def test_invalid_quote_bindings_reject_the_whole_unchanged_native_candidate(mode):
    cards, selection, authored = materials()
    native = to_native(authored, cards)
    sentence = native["sections"]["introduction"]["first_block"]["first_sentence"]
    quote = sentence["support_quotes"][0]
    if mode == "missing":
        sentence.pop("support_quotes")
    elif mode == "wrong_card":
        sentence["support_quotes"][0] = cards[list(cards)[1]]["source_quote"]
    elif mode == "paraphrase":
        sentence["support_quotes"][0] = "EViT speeds up inference."
    elif mode == "disjoint":
        sentence["support_quotes"][0] = "EViT speed."
    elif mode == "duplicate":
        sentence["support_quotes"].append(copy.deepcopy(quote))
    elif mode == "empty":
        sentence["support_quotes"][0] = " "
    else:
        sentence["support_quotes"][0] = {"source_quote": quote, "quote_start": 123}
    raw = json.dumps(native)
    with pytest.raises(ValueError) as caught:
        render_grounded_report(
            raw,
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
            require_source_quotes=True,
        )
    if mode != "forged_position":
        assert isinstance(caught.value, GroundedWritingError)
        issue = next(
            i
            for i in caught.value.diagnostics
            if i.get("section_key") == "introduction"
        )
        assert issue["raw_json_path"].startswith(
            "sections.introduction.first_block.first_sentence.support_quotes"
        )
        assert caught.value.candidate_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert json.dumps(native) == raw


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_host_generates_character_offsets_and_rejects_ambiguous_originals(language):
    phrase = (
        "方法A在D1上提高速度。" if language == "zh-CN" else "A improves speed on D1."
    )
    original = "前言 " + phrase + " 后记"
    cards = build_fact_cards(
        {"a": {"paper_id": "paper-a", "supporting_text": original}}
    )
    cid = next(iter(cards))
    quotes = [phrase]
    located, errors = _locate_writing_support_quotes([cid], quotes, cards)
    assert not errors and located[0]["quote_start"] == 3
    assert located[0]["quote_end"] == 3 + len(phrase)
    repeated = build_fact_cards(
        {"a": {"paper_id": "paper-a", "supporting_text": original + original}}
    )
    other = next(iter(repeated))
    _, errors = _locate_writing_support_quotes(
        [other],
        [phrase],
        repeated,
    )
    assert errors[0]["subreason"] == "quote_ambiguous"


def test_same_quote_in_distinct_cards_keeps_both_actual_owners():
    cards = build_fact_cards(
        {
            eid: {"paper_id": eid, "supporting_text": "Shared original."}
            for eid in ("a", "b")
        }
    )
    quotes = ["Shared original." for cid in cards]
    located, errors = _locate_writing_support_quotes(list(cards), quotes, cards)
    assert not errors and [i["paper_id"] for i in located] == ["a", "b"]


@pytest.mark.parametrize(
    "field", ["paper_id", "evidence_id", "content_sha256", "source_start"]
)
def test_tampered_card_identity_is_not_a_valid_quote_location(field):
    cards, _, _ = materials()
    cid = next(iter(cards))
    original = cards[cid]["source_quote"]
    cards[cid][field] = 1 if field == "source_start" else "tampered"
    located, errors = _locate_writing_support_quotes([cid], [original], cards)
    assert not located and errors[0]["subreason"] == "source_identity_invalid"


def test_quote_position_metadata_does_not_share_mutable_source_locators():
    cards, _, _ = materials()
    cid = next(iter(cards))
    cards[cid]["source_locator"] = {"page": 2, "block": {"index": 1}}
    located, errors = _locate_writing_support_quotes(
        [cid], [cards[cid]["source_quote"]], cards
    )
    assert not errors
    located[0]["source_locator"]["block"]["index"] = 99
    assert cards[cid]["source_locator"]["block"]["index"] == 1


@pytest.mark.parametrize("key", ["abstract", "conclusion"])
def test_summary_new_quote_from_an_existing_body_card_is_still_rejected(key):
    cards, selection, authored = materials()
    native = to_native(authored, cards)
    sentence = native["sections"][key]["first_block"]["first_sentence"]
    sentence["support_quotes"][0] = "improves speed"
    with pytest.raises(GroundedWritingError) as caught:
        render_grounded_report(
            json.dumps(native),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
            require_source_quotes=True,
        )
    assert any(
        i.get("subreason") == "summary_quote_not_in_body"
        for i in caught.value.diagnostics
    )


def test_quote_in_an_organization_unit_does_not_establish_scientific_summary_reuse():
    cards, selection, authored = materials()
    aid = next(iter(cards))
    for section in authored["sections"]:
        if section["section_key"] not in {"abstract", "conclusion", "methods"}:
            section["blocks"][0]["sentences"][0]["kind"] = "organization"
    native = to_native(authored, cards)
    with pytest.raises(GroundedWritingError) as caught:
        render_grounded_report(
            json.dumps(native),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
            require_source_quotes=True,
        )
    assert any(
        "summary_card_not_in_body" in i["reason_code"]
        and aid in i["invalid_fact_card_ids"]
        for i in caught.value.diagnostics
    )


def test_valid_locations_do_not_override_wrong_science_or_strict_acceptance():
    cards, selection, authored = materials()
    unit(authored, "experiments")["text"] = "Evo-ViT guarantees 50% better accuracy."
    native = to_native(authored, cards)
    body = render_grounded_report(
        json.dumps(native),
        cards,
        "academic_v2",
        "en",
        evidence_selection=selection,
        require_native=True,
        require_source_quotes=True,
    )
    assert "Evo-ViT guarantees" in body
    # This synthetic semantic verdict is separate from the location check.
    details = valid_details()
    details["assertions"][0]["status"] = "unsupported"
    assert strict_acceptance(details)["status"] == "failed"


def test_native_schema_quotes_use_the_exact_same_chapter_permissions():
    cards, selection, authored = materials()
    prepared, fmt = prepare_grounded_request(
        grounded_messages(
            cards, "academic_v2", "en", "Compare methods", evidence_selection=selection
        )
    )
    schema = fmt["json_schema"]["schema"]
    Draft202012Validator(schema).validate(to_native(authored, cards))
    for key in schema["properties"]["sections"]["required"]:
        props = section_definition(schema, key, "sentence")["properties"]
        assert list(props).index("support_quotes") < list(props).index("text")
        assert props["support_quotes"]["items"] == {"type": "string"}
    assert (
        "A passage proposing an efficient method does not establish a best result"
        in prepared[0]["content"]
    )
    source_values = list(json.loads(prepared[1]["content"])["sources"].values())
    assert {v["source_quote"] for v in source_values} == {
        c["source_quote"] for c in cards.values()
    }


@pytest.mark.parametrize("mode", ["correct", "missing", "paraphrase", "malformed_json"])
def test_fragment_repair_uses_original_quotes_without_touching_other_prose(mode):
    original, _, cards, targets, response, good = case()
    sentence = response["patches"][targets[0]["source_fragment_id"]]["first_sentence"]
    cid = sentence["fact_card_ids"][0]
    if mode == "missing":
        sentence.pop("support_quotes")
    if mode != "missing":
        sentence["support_quotes"] = [
            cards[cid]["source_quote"] if mode == "correct" else "Invented paraphrase."
        ]
    raw = "{" if mode == "malformed_json" else json.dumps(response)
    if mode == "correct":
        candidate = apply_fragment_repairs(
            raw,
            original,
            targets,
            cards,
            "academic_v2",
            "en",
            require_source_quotes=True,
        )
        assert good in candidate and "Evo-ViT uses tokens" in candidate
    else:
        with pytest.raises(ValueError, match="contract_invalid"):
            apply_fragment_repairs(
                raw,
                original,
                targets,
                cards,
                "academic_v2",
                "en",
                require_source_quotes=True,
            )


@pytest.mark.asyncio
async def test_local_quote_compilation_error_stops_before_next_benchmark_case(
    tmp_path, monkeypatch
):
    value = report("evaluation_assertions_unverifiable")
    value["metadata"]["evidence_rewrite"] = {
        "attempted": True,
        "committed": False,
        "reason_code": "strict_writing_source_quote_contract_invalid",
        "candidate": None,
        "candidate_evaluation": None,
    }
    runner, calls = live_runner(
        tmp_path, monkeypatch, [value, report(None, verified=True)]
    )
    result = await runner.run()
    assert len(calls) == 1 and result["stop_reasons"] == [
        "strict_writing_source_quote_contract_invalid"
    ]
    assert result["budgets"]["survey"]["charged"] == 15
    assert public_projection(result)["counts"]["formal"]["not_run"] == 15


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_actual_initial_writer_requires_original_quotes_after_one_request(policy):
    cards, selection, authored = materials()
    raw = json.dumps(to_native(authored))
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(raw)
    selector = AsyncMock()
    selector.select.return_value = selection
    worker = SynthesisWorker(llm, evidence_selector=selector)
    with pytest.raises(GroundedWritingError, match="source_quote_contract_invalid"):
        await worker.execute(
            SubTask(
                "write",
                "offline",
                "synthesis",
                input_data={
                    "query": "EViT",
                    "quality_policy": policy,
                    "review_structure": "academic_v2",
                    "report_language": "en",
                },
            )
        )
    llm.chat.assert_awaited_once()
    assert llm.chat.await_args.kwargs["max_tokens"] == 20000
    assert llm.chat.await_args.kwargs["transport_max_attempts"] == 1
