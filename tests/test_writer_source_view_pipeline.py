"""Exercise the source-only writer view through real SDK and report consumers."""

import base64
import copy
import hashlib
import json
import re
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_scientific_repair_pipeline as scientific
import test_source_owner_pipeline as source_owner
import test_structured_repair_pipeline as structured
from jsonschema import Draft202012Validator
from test_claim_response_pipeline import decision_branches
from test_concise_repair_prompt import FLAGS as CONCISE_FLAGS
from test_native_responses import owner_for
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.eval.batched import EVALUATION_RESPONSE_CONTRACT_V2
from litagent.llm.responses import RESPONSES_CONTRACT
from litagent.llm.routing import ModelBoundClient
from litagent.quality import build_fact_cards, digest
from litagent.safety.budget import strict_request_reservation
from litagent.writer_source_view import (
    FINAL_CHAPTER_INSTRUCTION,
    GROUNDING_INSTRUCTION,
    selected_source_view,
    source_view_identity,
    validate_source_view_record,
)
from litagent.writing_v2 import (
    BODY,
    EOF_CONTAINER_POLICY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    body_units,
    check_repair_guidance_record,
    source_packet,
)

VIEW = "source-only-writer-v1"
PAIR = {"writer_source_view", "writer_source_view_identity"}
LOOKUP_FIELDS = {"source_ref"}
CANDIDATE_MARKER = "LOCAL_ONLY_EXTRACTION_CANDIDATE_"
OMITTED = object()


@pytest.fixture
def source_case(case):
    """Keep an unmistakable unverified candidate beside unchanged full originals."""
    value = copy.deepcopy(case)
    for eid, item in value.selection.selected_items.items():
        item["text"] = CANDIDATE_MARKER + eid + ": unverified extracted hint."
    value.ledger = copy.deepcopy(value.selection.selected_items)
    value.cards = build_fact_cards(value.ledger)
    value.packet, value.aliases = source_packet(
        value.cards,
        value.selection,
        case.packet["query"],
        copy.deepcopy(case.packet["requirements"]),
        case.packet["language"],
    )
    assert value.aliases == case.aliases
    assert value.packet["sources"] == case.packet["sources"]
    return value


def source_flags(view=VIEW):
    """Select only the trusted frozen option on the complete experimental stack."""
    flags = {
        **CONCISE_FLAGS,
        "evaluation_response_contract": EVALUATION_RESPONSE_CONTRACT_V2,
    }
    if view is not OMITTED:
        flags["writer_source_view"] = view
    return flags


def conform_evaluation_replies(gateway):
    """Adapt the controlled HTTP peer to the actual selected native definition."""
    original = gateway.evaluator.chat

    async def chat(messages, **options):
        response = await original(messages, **options)
        payload = gateway.evaluator.base.calls[-1]
        if "cited_materials" not in payload:
            return response
        schema = options["response_format"]["json_schema"]["schema"]
        value = json.loads(response.content)
        if all(
            "classification_valid" not in branch["properties"]
            for branch in decision_branches(schema)
        ):
            for decision in value["decisions"]:
                decision.pop("classification_valid", None)
        Draft202012Validator(schema).validate(value)
        response.content = json.dumps(value)
        return response

    gateway.evaluator.chat = chat
    return gateway


async def run_scientific_case(
    case, path, monkeypatch, *, failures=0, summary=False, view=VIEW, config=None
):
    """Replace HTTP responses while retaining synthesis, review and transactions."""
    path.mkdir(parents=True, exist_ok=True)
    gateway = conform_evaluation_replies(
        structured.SummaryGateway(case)
        if summary
        else structured.LongScientificGateway(case, failures, reviewer=True)
    )
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    monkeypatch.setattr(structured, "FLAGS", source_flags(view))
    structured.select_structured_policy(monkeypatch)
    report, actual_gateway = await run_feedback_case(
        case,
        path,
        monkeypatch,
        failures,
        **({"config": config} if config is not None else {}),
    )
    assert actual_gateway is gateway
    state, journal, ledger = products(path)
    return SimpleNamespace(
        report=report, gateway=gateway, state=state, journal=journal, ledger=ledger
    )


async def run_capacity_case(case, path, monkeypatch, *, mode="valid", unknown=False):
    """Keep source-owner derivation and full evaluation outside the HTTP mock."""
    gateway = conform_evaluation_replies(
        source_owner.SourceOwnerGateway(case, mode=mode, unknown=unknown)
    )
    monkeypatch.setattr(source_owner, "FLAGS", source_flags())
    monkeypatch.setattr(
        source_owner, "SourceOwnerGateway", lambda *_args, **_kwargs: gateway
    )
    report, actual_gateway = await source_owner.run_owner_case(
        case, path, monkeypatch, mode=mode, unknown=unknown
    )
    assert actual_gateway is gateway
    state, journal, ledger = products(path)
    return SimpleNamespace(
        report=report, gateway=gateway, state=state, journal=journal, ledger=ledger
    )


def request_parts(entry):
    """Read the unmodified Responses request recorded at the actual SDK boundary."""
    request = entry["request"]
    messages = request["input"]
    native = {
        "type": "json_schema",
        "json_schema": {
            key: value
            for key, value in request["text"]["format"].items()
            if key != "type"
        },
    }
    identity = digest(
        [
            messages,
            native,
            {"max_output_tokens": request["max_output_tokens"]},
            {"thinking": request["thinking"]},
            ["responses", RESPONSES_CONTRACT],
        ]
    )
    return messages, native, identity


def assert_selected_requests(result, canonical_packet):
    """Match complete originals and model views to the actual owner's receipts."""
    journal, ledger = result.journal, result.ledger
    envelope = ledger["plan"]["envelope"]
    assert selected_source_view(envelope)
    assert envelope["writer_source_view"] == VIEW
    assert envelope["writer_source_view_identity"] == source_view_identity()
    for key in PAIR:
        assert journal[key] == envelope[key]
    document = result.report["metadata"]["writer_document"]
    assert document["packet"] == canonical_packet
    assert all(
        CANDIDATE_MARKER in card["finding_candidate"]
        for card in document["packet"]["cards"].values()
    )
    writers = [entry for entry in result.gateway.sent if "input" in entry["request"]]
    assert writers
    for entry in writers:
        payload = entry["payload"]
        assert not PAIR.intersection(payload)
        assert not PAIR.intersection(payload.get("writing_envelope", {}))
        assert VIEW not in json.dumps(entry["request"])
        assert CANDIDATE_MARKER not in json.dumps(entry["request"])
        if "cards" in payload:
            assert list(payload["cards"]) == list(canonical_packet["cards"])
            assert payload["sources"] == {
                sid: {key: source[key] for key in ("paper_id", "source_quote")}
                for sid, source in canonical_packet["sources"].items()
            }
            for sid, source in payload["sources"].items():
                assert (
                    hashlib.sha256(source["source_quote"].encode()).hexdigest()
                    == canonical_packet["sources"][sid]["content_sha256"]
                )
            assert list(payload["sources"]) == list(canonical_packet["sources"])
            for ref, card in payload["cards"].items():
                assert set(card) == LOOKUP_FIELDS
                assert card == {
                    key: canonical_packet["cards"][ref][key] for key in LOOKUP_FIELDS
                }
                assert (
                    canonical_packet["cards"][ref]["semantic_status"] == "not_verified"
                )
        else:
            assert "sources" not in payload and "body_units" in payload
        messages, native, identity = request_parts(entry)
        if "targets" in payload:
            assert payload["grounding_instruction"] == GROUNDING_INSTRUCTION
        elif "requested_body_section" in payload:
            assert messages[0]["content"].count(FINAL_CHAPTER_INSTRUCTION) == 1
            assert "Read complete cited originals" in messages[0]["content"]
        else:
            assert messages[0]["content"].count(GROUNDING_INSTRUCTION) == 1
        matching = [
            call
            for call in journal["calls"].values()
            if call.get("input_sha256") == identity
        ]
        assert matching
        for call in matching:
            assert {key: call[key] for key in PAIR} == {
                key: envelope[key] for key in PAIR
            }
            record = call["writer_source_view_record"]
            before = copy.deepcopy(record)
            validate_source_view_record(envelope, record, canonical_packet)
            assert record == before
            assert record["source_snapshot"] == digest(canonical_packet)
            assert record["view_sha"] == digest(payload)
            assert record["system_sha"] == digest(messages[0]["content"])
            assert call["payload_sha256"] == digest(payload)
            assert call["reservation"] == strict_request_reservation(
                messages, entry["request"]["max_output_tokens"], native
            )
            assert call["provider_schema_sha256"] == digest(
                native["json_schema"]["schema"]
            )
            receipt = call["provider_receipt"]
            assert (
                base64.b64decode(receipt["request_body_base64"])
                == entry["request_bytes"]
            )
    assert ledger["requests_sent"] == len(result.gateway.sent)
    assert ledger["plan"]["maximum_tokens"] == 4500000
    assert ledger["plan"]["maximum_requests"] == 64
    assert ledger["plan"]["repair_limit"] == 2
    assert envelope["request_caps"]["assessment"] == 100000
    assert_original_prose_inputs(result)


def assert_original_prose_inputs(result):
    """Recover the complete initial report and every accumulated prior chapter."""
    context = {}
    chapters = sent_with(result.gateway, "requested_body_section")
    assert [entry["payload"]["requested_body_section"] for entry in chapters] == list(
        BODY
    )
    for section, entry in zip(BODY, chapters, strict=True):
        assert entry["payload"]["prior_chapters_unverified"] == context
        context.update(
            copy.deepcopy(
                result.journal["calls"]["body:" + section]["reference_canonical"]
            )
        )
    reviewer_identity = result.journal["calls"]["reviewer"]["input_sha256"]
    [reviewer] = [
        entry
        for entry in result.gateway.sent
        if "input" in entry["request"] and request_parts(entry)[2] == reviewer_identity
    ]
    payload = reviewer["payload"]
    assert {
        row["unit_id"] for row in payload["report_layout"] if "unit_id" in row
    } == set(payload["units"])
    projected = "".join(
        (
            row["text"]
            if "text" in row
            else payload["units"][row["unit_id"]]["source_quote"]
        )
        for row in payload["report_layout"]
    )
    restored = re.sub(
        r"\[E:([^\]]+)\]",
        lambda match: "[E:" + payload["citation_projection"][match[1]] + "]",
        projected,
    )
    assert hashlib.sha256(restored.encode()).hexdigest() == payload["report_sha256"]
    initial = result.report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    assert initial["source_report_sha256"] == payload["report_sha256"]


def assert_guidance_preserved(entry, call, envelope, *, prior):
    """Keep every typed issue, target and prose value outside the card projection."""
    record = call["guidance_wire_record"]
    before = copy.deepcopy(record)
    check_repair_guidance_record(envelope, record)
    assert record == before
    assert not PAIR.intersection(record)
    assert not PAIR.intersection(record["payload"]["writing_envelope"])
    current = record["canonical"]["current"]
    assert current["coverage"]["complete"]
    assert current["coverage"]["issues_total"] == len(current["issues"]) > 0
    assert (record["canonical"]["prior_candidate"] is not None) is prior
    for key in (
        "current_body",
        "targets",
        "previous_local_failure",
        "repair_guidance",
        "body_units",
        "summary_source_body_sha256",
    ):
        if key in record["payload"]:
            assert entry["payload"][key] == record["payload"][key]
    if prior:
        previous = record["canonical"]["prior_candidate"]
        assert previous["coverage"]["complete"]
        assert previous["coverage"]["issues_total"] == len(previous["issues"]) > 0
        assert previous["source_report_sha256"] != current["source_report_sha256"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_source_only_sdk_keeps_canonical_cards_and_full_current_prior_repairs(
    source_case, tmp_path, monkeypatch, failing_candidates
):
    """Candidate hints disappear while scientific failures retain both attempts."""
    before = copy.deepcopy(source_case.packet)
    cards_before = copy.deepcopy(source_case.cards)
    result = await run_scientific_case(
        source_case, tmp_path, monkeypatch, failures=failing_candidates
    )
    assert_selected_requests(result, before)
    assert source_case.packet == before and source_case.cards == cards_before
    expected = min(2, failing_candidates + 1)
    repairs = sent_with(result.gateway, "targets")
    assert len(repairs) == result.gateway.repair_count == expected
    assert result.gateway.inventories == expected + 1
    assert len(result.journal["candidates"]) == expected
    assert (
        len(result.report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    )
    for index, entry in enumerate(repairs):
        call = result.journal["calls"][f"repair_{index}"]
        assert_guidance_preserved(
            entry, call, result.ledger["plan"]["envelope"], prior=index > 0
        )
        assert call["guidance_wire_record"]["authority"]["packet"] == before
        assert "FLOPs by 44%" in json.dumps(entry["payload"]["current_body"])
        assert "FLOPs by 45%" not in json.dumps(entry["payload"]["current_body"])
    structured.assert_local_notes_retained(tmp_path, result.gateway)
    assert not result.ledger["usage_uncertain"]
    assert all(row["status"] == "settled" for row in result.ledger["entries"].values())
    assert result.report["delivery"]["publishable"] is (failing_candidates < 2)
    if failing_candidates < 2:
        details = result.report["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert details["stages"]["global_check"] == "completed"
        assert details["supported_count"] == details["factual_claim_count"] > 0
        assert details["unsupported_count"] == details["unverified_count"] == 0
        assert result.journal["candidates"][-1]["committed"]
    else:
        assert not any(row["committed"] for row in result.journal["candidates"])
        assert "FLOPs by 44%" in result.report["survey"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "wrong_paper"])
async def test_source_only_capacity_derivation_still_requires_real_paper_support(
    source_case, tmp_path, monkeypatch, mode
):
    """Removing hints neither authorizes foreign science nor weakens source maps."""
    before = copy.deepcopy(source_case.packet)
    result = await run_capacity_case(source_case, tmp_path, monkeypatch, mode=mode)
    assert_selected_requests(result, before)
    repairs = sent_with(result.gateway, "targets")
    assert 1 <= len(repairs) <= 2
    assert result.gateway.inventory_calls == len(result.journal["candidates"]) + 1
    for index, entry in enumerate(repairs):
        call = result.journal["calls"][f"repair_{index}"]
        assert_guidance_preserved(
            entry, call, result.ledger["plan"]["envelope"], prior=index > 0
        )
        parsed = source_owner.assert_source_owner_receipt(
            call, result.ledger["plan"]["envelope"]
        )
        assert parsed["canonical"]["source_relations"]
        assert all(
            "target_id" not in patch for patch in json.loads(call["raw"])["patches"]
        )
    assert result.report["delivery"]["publishable"] is (mode == "valid")
    if mode == "wrong_paper":
        # The second identical repair is rejected locally before reevaluation.
        assert len(repairs) == 2 and len(result.journal["candidates"]) == 1
        assert result.journal["failure"]["code"] == "writer_repair_no_progress"
        assert not any(row["committed"] for row in result.journal["candidates"])
        details = result.journal["candidates"][0]["evaluation"]["faithfulness"][
            "details"
        ]
        assert details["coverage"]["complete"]
        assert details["stages"]["global_check"] == "completed"
        assert details["unsupported_count"] > 0
    else:
        assert result.journal["candidates"][-1]["committed"]


@pytest.mark.asyncio
async def test_source_only_summary_repair_preserves_the_exact_readonly_body(
    source_case, tmp_path, monkeypatch
):
    """A source-view option adds no paper facts or new summary premise rights."""
    before = copy.deepcopy(source_case.packet)
    result = await run_scientific_case(source_case, tmp_path, monkeypatch, summary=True)
    assert_selected_requests(result, before)
    [repair] = sent_with(result.gateway, "targets")
    summary = next(
        entry
        for entry in sent_with(result.gateway, "body_units")
        if "targets" not in entry["payload"]
    )
    assert "cards" not in summary["payload"] and "sources" not in summary["payload"]
    assert repair["payload"]["body_units"] == summary["payload"]["body_units"]
    assert (
        body_units(result.report["metadata"]["writer_document"]["body"])
        == summary["payload"]["body_units"]
    )
    call = result.journal["calls"]["repair_0"]
    assert_guidance_preserved(
        repair, call, result.ledger["plan"]["envelope"], prior=False
    )
    assert call["guidance_wire_record"]["arguments"]["summary"] is True
    properties = repair["request"]["text"]["format"]["schema"]["properties"]["patches"][
        "items"
    ]["properties"]
    assert set(properties) == {"target_id", "text", "body_unit_refs"}
    assert result.report["delivery"]["publishable"]
    assert result.journal["candidates"][0]["committed"]
    structured.assert_local_notes_retained(tmp_path, result.gateway)


@pytest.mark.asyncio
async def test_source_only_unknown_usage_has_one_occupied_attempt_and_no_retry(
    source_case, tmp_path, monkeypatch
):
    """A valid source view does not turn uncertain transport usage into a retry."""
    before = copy.deepcopy(source_case.packet)
    result = await run_capacity_case(source_case, tmp_path, monkeypatch, unknown=True)
    assert_selected_requests(result, before)
    assert len(sent_with(result.gateway, "targets")) == 1
    assert len(result.report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert not result.journal["candidates"]
    assert result.ledger["usage_uncertain"]
    assert result.ledger["stop_reason"] == "strict_usage_unknown"
    held = [
        row
        for row in result.ledger["entries"].values()
        if row["sent"] and row["status"] != "settled"
    ]
    assert len(held) == 1 and held[0]["actual_tokens"] is None
    assert held[0]["reserved_tokens"] > 0
    assert not result.report["delivery"]["publishable"]


@pytest.mark.asyncio
async def test_omitted_and_explicit_none_preserve_complete_sdk_wires(
    source_case, tmp_path, monkeypatch
):
    """Inactive options keep full canonical cards and every old request identity."""
    results = []
    for label, selection in (("omitted", OMITTED), ("none", None)):
        with monkeypatch.context() as scoped_patch:
            results.append(
                await run_scientific_case(
                    copy.deepcopy(source_case),
                    tmp_path / label,
                    scoped_patch,
                    failures=1,
                    view=selection,
                )
            )
    first, second = results
    assert first.ledger["plan"] == second.ledger["plan"]
    assert [entry["request"] for entry in first.gateway.sent] == [
        entry["request"] for entry in second.gateway.sent
    ]
    assert [entry["request_bytes"] for entry in first.gateway.sent] == [
        entry["request_bytes"] for entry in second.gateway.sent
    ]
    assert first.report["survey"] == second.report["survey"]
    for result in results:
        assert not PAIR.intersection(result.ledger["plan"]["envelope"])
        assert not PAIR.intersection(result.journal)
        for call in result.journal["calls"].values():
            assert not PAIR.intersection(call)
            assert "writer_source_view_record" not in call
        body = sent_with(result.gateway, "requested_body_section")
        assert [entry["payload"]["requested_body_section"] for entry in body] == list(
            BODY
        )
        assert all(
            entry["payload"]["cards"] == source_case.packet["cards"] for entry in body
        )
        reviewer = next(
            entry
            for entry in result.gateway.sent
            if "report_layout" in entry["payload"]
        )
        assert reviewer["payload"]["cards"] == source_case.packet["cards"]
        assert result.report["delivery"]["publishable"]


@asynccontextmanager
async def restored_session(result, path, *, journal=None):
    """Restore a copied real SDK journal with a transport that permits no sends."""
    path.mkdir(parents=True, exist_ok=True)
    sent = []

    def transport(request):
        sent.append(request)
        pytest.fail("Reference restoration must not send another HTTP request")

    owner, budget, _ = await owner_for(transport)
    try:
        budget.activate_plan(
            copy.deepcopy(result.ledger["plan"]), path=path / "budget.json"
        )
        state_path = path / "repair.json"
        Path(str(state_path) + ".writer-v2.json").write_text(
            json.dumps(result.journal if journal is None else journal), encoding="utf-8"
        )
        document = result.report["metadata"]["writer_document"]
        session = WriterSession(
            ModelBoundClient(owner, "pro", "generation"),
            state_path,
            "repair-feedback-offline",
            document["packet"],
            document["aliases"],
            thinking_mode="disabled",
            provider_api="responses",
            syntax_policy=EOF_CONTAINER_POLICY,
            writing_plan=RANGE_WRITING_PLAN,
        )
        yield session, budget, sent
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_restored_reference_input_rejects_both_rebound_source_view_hashes(
    source_case, tmp_path, monkeypatch
):
    """Well-formed record hashes must still match the actual stored call hashes."""
    result = await run_scientific_case(source_case, tmp_path / "actual", monkeypatch)
    original_http = len(result.gateway.sent)
    canonical = result.journal["calls"]["body:introduction"]["reference_canonical"]
    async with restored_session(result, tmp_path / "restored") as (
        session,
        budget,
        sent,
    ):
        expected = session.journal["calls"]["body:introduction"]["received_canonical"]
        assert session.reference_input(canonical)[0] == expected
        baseline = copy.deepcopy(session.journal)
        budget_before = copy.deepcopy(budget.snapshot())
        envelope = result.ledger["plan"]["envelope"]
        for field in ("view_sha", "system_sha"):
            session.journal = copy.deepcopy(baseline)
            record = session.journal["calls"]["body:introduction"][
                "writer_source_view_record"
            ]
            record[field] = "f" * 64
            validate_source_view_record(envelope, record, source_case.packet)
            with pytest.raises(
                WriterError, match="writer_reference_input_identity_invalid"
            ):
                session.reference_input(canonical)
            assert budget.snapshot() == budget_before
            assert not sent and len(result.gateway.sent) == original_http


@pytest.mark.asyncio
async def test_inactive_none_pair_presence_cannot_be_added_to_restored_records(
    source_case, tmp_path, monkeypatch
):
    """An absent legacy pair differs from explicit None in journals and calls."""
    result = await run_scientific_case(
        source_case, tmp_path / "actual", monkeypatch, view=OMITTED
    )
    assert not PAIR.intersection(result.journal)
    canonical = result.journal["calls"]["body:introduction"]["reference_canonical"]
    original_http = len(result.gateway.sent)
    for field in sorted(PAIR):
        snapshot = copy.deepcopy(result.journal)
        snapshot[field] = None
        with pytest.raises(WriterError, match="writer_checkpoint_source_view_conflict"):
            async with restored_session(
                result, tmp_path / ("journal-" + field), journal=snapshot
            ):
                pytest.fail("A legacy journal with an explicit None pair was accepted")
        assert len(result.gateway.sent) == original_http
    async with restored_session(result, tmp_path / "calls") as (session, budget, sent):
        assert session.reference_input(canonical)[0]
        baseline = copy.deepcopy(session.journal)
        budget_before = copy.deepcopy(budget.snapshot())
        for field in sorted(PAIR):
            session.journal = copy.deepcopy(baseline)
            session.journal["calls"]["body:introduction"][field] = None
            with pytest.raises(
                WriterError, match="writer_reference_input_identity_invalid"
            ):
                session.reference_input(canonical)
            assert budget.snapshot() == budget_before
            assert not sent and len(result.gateway.sent) == original_http
