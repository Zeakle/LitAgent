"""Exercise concise repair through the SDK and unchanged scientific consumers."""

import base64
import copy
import json
from types import SimpleNamespace

import pytest
import test_scientific_repair_pipeline as scientific
import test_source_owner_pipeline as source_owner
import test_structured_repair_pipeline as structured
from jsonschema import Draft202012Validator
from test_concise_repair_prompt import FLAGS, PAIR
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import actual_start, products, sent_with
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
from litagent.llm.responses import RESPONSES_CONTRACT
from litagent.quality import digest
from litagent.safety.budget import strict_request_reservation
from litagent.writing_v2 import (
    WriterError,
    body_units,
    check_repair_guidance_record,
    provider_response_format,
)
from scripts import run_frozen_writing as campaign


def select_prompt(monkeypatch):
    """Pass the internal selector into the actual frozen planner and runner."""
    monkeypatch.setattr(structured, "FLAGS", dict(FLAGS))
    structured.select_structured_policy(monkeypatch)


def assert_final_repair_request(entry, call, envelope):
    """Match final SDK bytes to native schema, capture and checkpoint identity."""
    request = entry["request"]
    assert "input" in request and "messages" not in request
    messages = request["input"]
    schema = call["reference_schema"]
    rf = provider_response_format("repair", schema, wire_contract="flat-writer-v1")
    expected_system = flat.repair_prompt_system(schema)
    assert messages == flat.repair_prompt_messages(
        expected_system, entry["payload"], schema
    )
    assert request["text"]["format"]["schema"] == rf["json_schema"]["schema"]
    assert call["provider_schema_sha256"] == digest(rf["json_schema"]["schema"])
    assert call["schema_sha256"] == digest(schema)
    assert request["max_output_tokens"] == call["max_tokens"] == 20000
    assert request["thinking"] == {"type": "disabled"}
    assert call["thinking_mode"] == "disabled" and call["reasoning_effort"] is None
    record = call["guidance_wire_record"]
    assert record["system_sha256"] == digest(messages[0]["content"])
    assert record["schema_sha256"] == digest(schema)
    assert record["final_payload_sha256"] == digest(entry["payload"])
    assert record["identity"] == digest(
        {k: v for k, v in record.items() if k != "identity"}
    )
    for field in PAIR:
        assert entry["payload"]["writing_envelope"][field] == envelope[field]
        assert record[field] == envelope[field]
    original_record = copy.deepcopy(record)
    check_repair_guidance_record(envelope, record)
    assert record == original_record
    assert not PAIR.intersection(record["payload"]["writing_envelope"])
    assert call["input_sha256"] == digest(
        [
            messages,
            rf,
            {"max_output_tokens": 20000},
            {"thinking": {"type": "disabled"}},
            ["responses", RESPONSES_CONTRACT],
        ]
    )
    assert call["reservation"] == strict_request_reservation(messages, 20000, rf)
    assert call["reservation"] <= envelope["request_caps"]["repair"] == 117964
    receipt = call["provider_receipt"]
    assert base64.b64decode(receipt["request_body_base64"]) == entry["request_bytes"]
    Draft202012Validator(rf["json_schema"]["schema"]).validate(json.loads(call["raw"]))
    campaign._historical_writer_envelope(actual_start(entry), envelope, call=call)


def assert_pipeline_accounting(gateway, ledger):
    """Keep shared funding and attempts authoritative under the new prompt."""
    assert ledger["requests_sent"] == len(gateway.sent)
    assert ledger["call_count"] == sum(
        row["status"] == "settled" for row in ledger["entries"].values()
    )
    assert ledger["plan"]["maximum_tokens"] == 4500000
    assert ledger["plan"]["maximum_requests"] == 64
    assert ledger["plan"]["repair_limit"] == 2
    assert ledger["plan"]["envelope"]["request_caps"]["assessment"] == 100000
    assert flat.selected_repair_prompt(ledger["plan"]["envelope"])


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_body_candidates_need_full_reassessment_and_share_two_calls(
    case, tmp_path, monkeypatch, failing_candidates
):
    """An unsupported numeric relation remains unsafe until a full repair passes."""
    gateway = structured.LongScientificGateway(case, failing_candidates, reviewer=True)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_prompt(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, failing_candidates)
    _, journal, ledger = products(tmp_path)
    expected = min(2, failing_candidates + 1)
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == gateway.repair_count == expected
    assert gateway.inventories == expected + 1
    assert len(sent_with(gateway, "body_units")) == expected + 1
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    for index, entry in enumerate(repairs):
        call = journal["calls"][f"repair_{index}"]
        assert_final_repair_request(entry, call, ledger["plan"]["envelope"])
        guidance = call["guidance_wire_record"]["canonical"]
        assert guidance["current"]["coverage"]["complete"]
        assert (guidance["prior_candidate"] is not None) is (index > 0)
        assert {issue["category"] for issue in guidance["current"]["issues"]} >= {
            "evaluation",
            "reviewer",
        }
        assert "FLOPs by 44%" in json.dumps(entry["payload"]["current_body"])
        assert "FLOPs by 45%" not in json.dumps(entry["payload"]["current_body"])
        for card in case.cards.values():
            assert card["source_quote"] in json.dumps(entry["request"])
    structured.assert_local_notes_retained(tmp_path, gateway)
    assert_pipeline_accounting(gateway, ledger)
    assert not ledger["usage_uncertain"]
    assert all(row["status"] == "settled" for row in ledger["entries"].values())
    assert report["delivery"]["publishable"] is (failing_candidates < 2)
    if failing_candidates < 2:
        details = report["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert details["stages"]["global_check"] == "completed"
        assert details["supported_count"] == details["factual_claim_count"]
        assert details["unsupported_count"] == details["unverified_count"] == 0
        assert journal["candidates"][-1]["committed"]
        assert "FLOPs by" not in report["survey"]
    else:
        assert not any(candidate["committed"] for candidate in journal["candidates"])
        assert "FLOPs by 44%" in report["survey"]
        assert "FLOPs by 45%" not in report["survey"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_source_owned_capacity_candidates_keep_full_science_and_two_attempts(
    case, tmp_path, monkeypatch, failing_candidates
):
    """Target-free source ownership changes no evidence or transaction gate."""
    monkeypatch.setattr(source_owner, "FLAGS", dict(FLAGS))
    report, gateway = await source_owner.run_owner_case(
        case,
        tmp_path,
        monkeypatch,
        mode="candidate",
        failing_candidates=failing_candidates,
    )
    _, journal, ledger = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    expected = min(2, failing_candidates + 1)
    assert len(repairs) == len(gateway.capacity_responses) == expected
    assert gateway.inventory_calls == expected + 1
    assert len(journal["candidates"]) == expected
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    envelope = ledger["plan"]["envelope"]
    for index, entry in enumerate(repairs):
        call = journal["calls"][f"repair_{index}"]
        assert_final_repair_request(entry, call, envelope)
        parsed = source_owner.assert_source_owner_receipt(call, envelope)
        assert parsed["canonical"]["source_relations"]
        assert all(
            "target_id" not in patch for patch in json.loads(call["raw"])["patches"]
        )
    assert report["delivery"]["publishable"] is (failing_candidates < 2)
    assert journal["candidates"][-1]["committed"] is (failing_candidates < 2)
    if failing_candidates:
        assert journal["candidates"][0]["committed"] is False
    assert_pipeline_accounting(gateway, ledger)
    assert not ledger["usage_uncertain"]


@pytest.mark.asyncio
async def test_summary_local_repair_uses_the_actual_readonly_body_shape(
    case, tmp_path, monkeypatch
):
    """Summary replacements retain body premises rather than source-owner fields."""
    gateway = structured.SummaryGateway(case)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    select_prompt(monkeypatch)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, 0)
    _, journal, ledger = products(tmp_path)
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 1
    initial = next(
        row
        for row in sent_with(gateway, "body_units")
        if "targets" not in row["payload"]
    )
    assert repairs[0]["payload"]["body_units"] == initial["payload"]["body_units"]
    assert (
        body_units(report["metadata"]["writer_document"]["body"])
        == initial["payload"]["body_units"]
    )
    call = journal["calls"]["repair_0"]
    assert_final_repair_request(repairs[0], call, ledger["plan"]["envelope"])
    props = repairs[0]["request"]["text"]["format"]["schema"]["properties"]["patches"][
        "items"
    ]["properties"]
    assert set(props) == {"target_id", "text", "body_unit_refs"}
    assert call["guidance_wire_record"]["arguments"]["summary"] is True
    assert report["delivery"]["publishable"] is True
    assert "FLOPs by" not in report["survey"]
    assert journal["candidates"][0]["committed"] is True
    structured.assert_local_notes_retained(tmp_path, gateway)
    assert_pipeline_accounting(gateway, ledger)


class ConditionGateway(source_owner.SourceOwnerGateway):
    """Change only evaluation responses for a deliberately wrong model condition."""

    def __init__(self, case):
        super().__init__(case)
        self.condition_rejections = 0
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            value = json.loads(response.content)
            claims = {claim["claim_id"]: claim for claim in payload["assertions"]}
            for decision in value["decisions"]:
                if "On DeiT-B," not in claims[decision["claim_id"]]["claim_text"]:
                    continue
                self.condition_rejections += 1
                decision.update(
                    status="unsupported",
                    reason_code="insufficient_support",
                    support_basis=[],
                    explanation="The original establishes DeiT-S, not the stated DeiT-B condition.",
                )
                for aspect in decision["aspect_checks"]:
                    aspect.update(
                        supported=False,
                        support_ids=[],
                        explanation="The stated condition has no cited support.",
                    )
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["conditions", "wrong_paper", "missing_core"])
async def test_concise_instructions_never_exempt_conditions_paper_or_core(
    case, tmp_path, monkeypatch, fault
):
    """A native-valid repaired candidate cannot override scientific rejection."""
    monkeypatch.setattr(source_owner, "FLAGS", dict(FLAGS))
    if fault == "conditions":
        case.body["methods"][0]["sentences"][0].update(
            text="On DeiT-B, EViT reduces computation.",
            fact_card_refs=[case.by_eid["a"]],
        )
        gateway = ConditionGateway(case)
        monkeypatch.setattr(
            source_owner, "SourceOwnerGateway", lambda *_args, **_kwargs: gateway
        )
    report, gateway = await source_owner.run_owner_case(
        case, tmp_path, monkeypatch, mode="valid" if fault == "conditions" else fault
    )
    _, journal, ledger = products(tmp_path)
    assert report["delivery"]["publishable"] is False
    assert journal["candidates"] and not any(
        row["committed"] for row in journal["candidates"]
    )
    details = journal["candidates"][0]["evaluation"]["faithfulness"]["details"]
    assert details["coverage"]["complete"]
    assert details["stages"]["global_check"] == "completed"
    if fault == "missing_core":
        assert any(row["answered"] is False for row in details["core_coverage"])
    else:
        assert details["unsupported_count"] > 0
    if fault == "conditions":
        assert gateway.condition_rejections > 0
    for index, entry in enumerate(sent_with(gateway, "targets")):
        assert_final_repair_request(
            entry, journal["calls"][f"repair_{index}"], ledger["plan"]["envelope"]
        )
    assert_pipeline_accounting(gateway, ledger)


@pytest.mark.asyncio
async def test_unknown_usage_occupies_the_attempt_without_any_extra_http(
    case, tmp_path, monkeypatch
):
    """A valid prompt and native reply never turn missing usage into a retry."""
    monkeypatch.setattr(source_owner, "FLAGS", dict(FLAGS))
    report, gateway = await source_owner.run_owner_case(
        case, tmp_path, monkeypatch, unknown=True
    )
    _, journal, ledger = products(tmp_path)
    assert len(sent_with(gateway, "targets")) == len(gateway.capacity_responses) == 1
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == 1
    assert not journal["candidates"]
    assert report["delivery"]["publishable"] is False
    assert ledger["usage_uncertain"] and ledger["stop_reason"] == "strict_usage_unknown"
    held = [
        row
        for row in ledger["entries"].values()
        if row["sent"] and row["status"] != "settled"
    ]
    assert len(held) == 1
    assert held[0]["actual_tokens"] is None and held[0]["reserved_tokens"] > 0
    assert_pipeline_accounting(gateway, ledger)


@pytest.mark.asyncio
async def test_mutated_system_and_receipt_reject_before_a_second_sdk_send(
    case, tmp_path, monkeypatch
):
    """Replay consumers authenticate the actual prompt without issuing new HTTP."""
    monkeypatch.setattr(source_owner, "FLAGS", dict(FLAGS))
    report, gateway = await source_owner.run_owner_case(case, tmp_path, monkeypatch)
    _, journal, ledger = products(tmp_path)
    call = journal["calls"]["repair_0"]
    envelope = ledger["plan"]["envelope"]
    record = call["guidance_wire_record"]
    original_calls = len(gateway.sent)
    original_record = copy.deepcopy(record)
    for field in (*sorted(PAIR), "payload_sha256", "receipt_sha256"):
        forged = copy.deepcopy(record)
        forged[field] = "0" * 64
        forged["identity"] = digest(
            {k: v for k, v in forged.items() if k != "identity"}
        )
        with pytest.raises(WriterError, match="writer_guidance_wire_receipt_invalid"):
            check_repair_guidance_record(envelope, forged)
    for field in PAIR:
        forged = copy.deepcopy(record)
        forged.pop(field)
        forged["identity"] = digest(
            {k: v for k, v in forged.items() if k != "identity"}
        )
        with pytest.raises(WriterError, match="writer_guidance_wire_receipt_invalid"):
            check_repair_guidance_record(envelope, forged)
    forged = copy.deepcopy(record)
    forged["identity"] = "0" * 64
    with pytest.raises(WriterError, match="writer_guidance_wire_receipt_invalid"):
        check_repair_guidance_record(envelope, forged)
    assert record == original_record
    assert len(gateway.sent) == original_calls

    unsent = []

    def transport(request):
        unsent.append(request)
        pytest.fail("A modified system reached the external transport")

    owner, budget, _ = await owner_for(transport)
    budget.activate_plan(
        copy.deepcopy(ledger["plan"]), path=tmp_path / "mutated-budget.json"
    )
    document = report["metadata"]["writer_document"]
    session = make_session(
        owner,
        SimpleNamespace(packet=document["packet"], aliases=document["aliases"]),
        tmp_path / "mutated",
    )
    try:
        before = copy.deepcopy(budget.snapshot())
        for field in ("system_sha256", "schema_sha256"):
            forged = copy.deepcopy(record)
            forged[field] = "0" * 64
            forged["identity"] = digest(
                {k: v for k, v in forged.items() if k != "identity"}
            )
            with pytest.raises(
                WriterError, match="writer_guidance_wire_receipt_invalid"
            ):
                await session.call(
                    "repair_bad_" + field,
                    flat.repair_prompt_system(call["reference_schema"]),
                    record["payload"],
                    call["reference_schema"],
                    stage="repair",
                    guidance_wire_record=forged,
                )
            assert budget.snapshot() == before
        with pytest.raises(
            WriterError,
            match="writer_(repair_prompt_identity|guidance_wire_receipt)_invalid",
        ):
            await session.call(
                "repair_mutated",
                flat.repair_prompt_system(call["reference_schema"])
                + "Unknown extra rule.",
                record["payload"],
                call["reference_schema"],
                stage="repair",
                guidance_wire_record=record,
            )
        assert not unsent
        assert budget.snapshot() == before
        assert all(
            entry["status"] == "rejected_not_sent"
            for entry in session.journal["calls"].values()
        )
        assert len(gateway.sent) == original_calls
    finally:
        await owner.close()
