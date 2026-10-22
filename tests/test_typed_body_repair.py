"""Keep body typing, local repair accounting and scientific gates independent."""

import asyncio
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator, ValidationError
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_native_array_integration import request
from test_staged_native_generation import (
    provider_body,
    provider_repair,
    provider_summaries,
    split,
)
from test_writing_passages import facts

from litagent.llm.client import OpenAICompatibleClient
from litagent.native_wire import canonical_from_array
from litagent.observability.context import reset_task_id, set_task_id
from litagent.quality import (
    GroundedWritingError,
    _writing_scopes,
    apply_native_repairs,
    begin_targeted_repair,
    load_targeted_repairs,
    prepare_native_repair_request,
    render_grounded_report,
    save_targeted_repairs,
    strict_acceptance,
    targeted_repair_state,
)
from litagent.safety.budget import CostBudget
from litagent.staged_generation import (
    body_row_projection,
    card_reference_binding,
    decode_typed_body,
    encode_typed_body,
    expand_body,
    generate_staged_report,
    prepare_body_request,
    validate_body,
)
from litagent.writing_passages import (
    expand_scope_notice,
    scope_notice_catalog,
    validate_unit_citation_scope,
)

REAL = json.loads(Path("tests/fixtures/typed-body-errors-real.json").read_text())


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_notices_are_host_owned_and_positions_preserve_scientific_units(language):
    cards = facts()
    body, _ = split(analysis_document(cards, language))
    block = body["sections"]["introduction"]["blocks"][0]
    first = copy.deepcopy(block["sentences"][0])
    block["sentences"] += [
        {
            "text": "",
            "kind": "organization",
            "fact_card_ids": [],
            "scope_notice_id": "review_scope",
        },
        copy.deepcopy(first),
    ]
    binding = card_reference_binding(
        cards, _writing_scopes(cards, "academic_v2", language, None)
    )
    wire = encode_typed_body(body, binding)
    assert wire["scope_notices"] == [
        {
            "section_key": "introduction",
            "after_row": 0,
            "scope_notice_id": "review_scope",
        }
    ]
    assert wire["introduction"][1]["text"] == first["text"]
    assert expand_body(decode_typed_body(wire, binding)) == body
    messages, rf = request(cards, language, array_wire=True)
    _, schema = prepare_body_request(messages, rf)
    restored, _ = validate_body(
        json.dumps(wire), schema, cards, "academic_v2", language, None
    )
    assert restored == body
    projection = body_row_projection(body, binding, typed_units=True)
    assert next(
        r for r in projection["positions"] if r["wire_path"] == "scope_notices.0"
    )["array_path"].endswith(".sentences.1")
    assert next(
        r for r in projection["positions"] if r["wire_path"] == "introduction.1"
    )["array_path"].endswith(".sentences.2")
    unit = body["sections"]["introduction"]["blocks"][0]["sentences"][1]
    assert (
        expand_scope_notice(unit, language)["text"]
        == scope_notice_catalog(language)["review_scope"]
    )
    assert projection["scientific_verdict"] == "not_evaluated"


@pytest.mark.parametrize(
    "fault",
    [
        "empty_cards",
        "notice_prose",
        "notice_cards",
        "unknown_notice",
        "bad_anchor",
        "duplicate_notice",
        "notice_text_field",
        "scientific_notice",
    ],
)
def test_native_types_cannot_hide_unbound_science_or_free_notice_prose(fault):
    cards = facts()
    body, _ = split(analysis_document(cards))
    wire = provider_body(body, cards)
    notice = {
        "section_key": "introduction",
        "after_row": 0,
        "scope_notice_id": "review_scope",
    }
    wire["scope_notices"] = [notice]
    if fault == "empty_cards":
        wire["methods"][0]["fact_card_refs"] = []
    elif fault == "notice_prose":
        wire["methods"][0].update(kind="organization", scope_notice_id="review_scope")
    elif fault == "notice_cards":
        notice["fact_card_refs"] = ["i1"]
    elif fault == "unknown_notice":
        notice["scope_notice_id"] = "field_has_no_research"
    elif fault == "bad_anchor":
        notice["after_row"] = 100
    elif fault == "duplicate_notice":
        wire["scope_notices"].append(copy.deepcopy(notice))
    elif fault == "notice_text_field":
        notice["text"] = "There are no relevant studies."
    else:
        wire["methods"][0]["scope_notice_id"] = "evidence_limit"
    messages, rf = request(cards, array_wire=True)
    _, schema = prepare_body_request(messages, rf)
    with pytest.raises((ValueError, ValidationError)):
        validate_body(json.dumps(wire), schema, cards, "academic_v2", "en", None)


@pytest.mark.parametrize("item", REAL["errors"])
def test_real_failures_remain_negative_without_scientific_gold_claims(item):
    assert (
        REAL["source_response_sha256"]
        == "b80ca564a0c728b29b58a3086ac6e9965b4e79ef7e13423ec4190b31f63d5134"
    )
    row = copy.deepcopy(item["wire_unit"])
    code = item["reason_code"]
    if code == "strict_writing_scope_notice_invalid":
        unit = {k: row[k] for k in ("kind", "text", "scope_notice_id")}
        unit["fact_card_ids"] = item["actual_fact_card_ids"]
        with pytest.raises(ValueError, match=code):
            expand_scope_notice(unit, "zh-CN")
    elif code == "strict_writing_unit_citation_scope_ambiguous":
        with pytest.raises(ValueError, match=code):
            validate_unit_citation_scope(
                row["text"], item["actual_fact_card_ids"], "zh-CN"
            )
    else:
        assert (
            code == "strict_writing_passage_binding_invalid"
            and not row["fact_card_refs"]
        )
        cards = facts()
        messages, rf = request(cards, "zh-CN", array_wire=True)
        _, schema = prepare_body_request(messages, rf)
        with pytest.raises(ValidationError):
            Draft202012Validator(
                schema["json_schema"]["schema"]["properties"]["methods"]["items"]
            ).validate(row)
    assert Counter(x["reason_code"] for x in REAL["errors"]) == {
        "strict_writing_scope_notice_invalid": 5,
        "strict_writing_passage_binding_invalid": 3,
        "strict_writing_unit_citation_scope_ambiguous": 4,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "twice",
        "exhausted",
        "unknown_usage",
        "cancel",
        "model",
        "truncated",
        "no_progress",
        "regression",
    ],
)
async def test_real_sdk_repairs_share_attempts_then_freeze_before_summary(
    tmp_path, fault
):
    cards = facts()
    messages, rf = request(cards, array_wire=True)
    original = analysis_document(cards)
    body, summaries = split(original)
    body_wire = provider_body(body, cards)
    correct = copy.deepcopy(
        original["sections"]["methods"]["first_block"]["first_sentence"]
    )
    body_wire["methods"][0]["fact_card_refs"] = []
    if fault in {"no_progress", "twice"}:
        body_wire["methods"][0]["fact_card_refs"] = provider_body(body, cards)[
            "methods"
        ][0]["fact_card_refs"]
        body_wire["methods"][0][
            "text"
        ] += "; Independent experiments retain different conditions."
    state = targeted_repair_state(None, "sdk-body")
    path = tmp_path / "repair.json"
    if fault == "exhausted":
        for _ in range(2):
            begin_targeted_repair(state, "native_initial", "a" * 64)[
                "status"
            ] = "failed"
        save_targeted_repairs(path, state)
    calls = []
    events = []

    async def send(req):
        payload = json.loads(req.content)
        calls.append(payload)
        name = payload["response_format"]["json_schema"]["name"]
        if name == "grounded_body_v5_blocks":
            value = body_wire
        elif name == "grounded_summaries_v3_layout":
            assert state["attempts"][-1]["status"] == "completed"
            value = provider_summaries(summaries, payload)
        else:
            assert name == "native_body_repair_v7_refs"
            assert (
                json.loads(path.read_text())["state"]["attempts"][-1]["status"]
                == "started_no_replay"
            )
            packet = json.loads(payload["messages"][1]["content"])
            assert set(packet["current_candidate"]["sections"]) == set(body["sections"])
            assert "body_unit_catalog" not in packet
            assert packet["sources"] == json.loads(messages[1]["content"])["sources"]
            if fault == "cancel":
                raise asyncio.CancelledError()
            patch = {**correct, "additional_units": []}
            if fault == "twice" and len(state["attempts"]) == 1:
                patch["text"] = body_wire["methods"][0]["text"] + " Still unbound."
            if fault == "no_progress":
                patch["text"] = body_wire["methods"][0]["text"]
            if fault == "regression":
                patch.update(
                    kind="gap",
                    text="",
                    fact_card_ids=[],
                    scope_notice_id="evidence_limit",
                )
            value = {"can_repair": True, "patches": {"u0": patch}}
        if name == "native_body_repair_v7_refs":
            value = provider_repair(value, cards)
        reply = {
            "id": "sdk-offline",
            "object": "chat.completion",
            "created": 0,
            "model": "wrong" if fault == "model" and len(calls) > 1 else "offline-pro",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": (
                            "{"
                            if fault == "truncated" and len(calls) > 1
                            else json.dumps(value)
                        ),
                    },
                }
            ],
        }
        if not (fault == "unknown_usage" and len(calls) > 1):
            reply["usage"] = {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        return httpx.Response(200, json=reply)

    sdk = AsyncOpenAI(
        api_key="offline",
        base_url="http://offline/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)),
    )
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    client = OpenAICompatibleClient(
        "http://offline/v1",
        "offline-pro",
        cost_budget=budget,
        api_key="offline",
        transport_max_attempts=1,
    )
    await client._client.close()
    client._client = sdk
    token = set_task_id("synthesis")
    try:
        kwargs = dict(
            cards=cards,
            structure="academic_v2",
            language="en",
            selection=None,
            checkpoint_path=path,
            repair_state=state,
            trace_hook=lambda event, data: events.append((event, data)),
        )
        # Use the same selection-free permission producer as the original message.
        kwargs["selection"] = None
        if fault in {None, "twice"}:
            candidate, proof = await generate_staged_report(
                client, messages, rf, **kwargs
            )
            assert json.loads(candidate) == original
            assert len(state["attempts"]) == (2 if fault == "twice" else 1)
            assert state["attempts"][-1]["status"] == "completed"
            assert proof["body_row_projection"]["transport"] == "host_repaired_body"
            assert len(calls) == (4 if fault == "twice" else 3)
            assert render_grounded_report(
                candidate,
                cards,
                "academic_v2",
                "en",
                require_native=True,
                require_full_sources=True,
                require_scope_notices=True,
                analysis_mode=True,
            )
        else:
            with pytest.raises(
                asyncio.CancelledError if fault == "cancel" else ValueError
            ):
                await generate_staged_report(client, messages, rf, **kwargs)
            assert all(
                c["response_format"]["json_schema"]["name"]
                != "grounded_summaries_v3_layout"
                for c in calls
            )
            assert len(calls) == (1 if fault == "exhausted" else 2)
            assert len(state["attempts"]) == (2 if fault == "exhausted" else 1)
        before = len(calls)
        with pytest.raises(ValueError):
            await generate_staged_report(client, messages, rf, **kwargs)
        assert len(calls) == before
        assert load_targeted_repairs(path, "sdk-body") == state
        assert budget.snapshot()["requests_sent"] == len(calls)
        if fault == "unknown_usage":
            assert budget.snapshot()["usage_uncertain"]
        if fault == "cancel":
            assert state["attempts"][0]["status"] == "cancelled"
        if fault == "truncated":
            failed = next(
                data for event, data in events if event == "writing.wire.failed"
            )
            assert failed["phase"] == "initial/body_repair_0"
            assert failed["candidate_sha256"] == hashlib.sha256(b"{").hexdigest()
            assert failed["json_position"] == 1
    finally:
        reset_task_id(token)
        await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid_global", [False, True, "unsupported_science"])
async def test_synthesis_repair_reviewer_and_actual_v3_evaluator_share_frozen_sources(
    tmp_path, invalid_global
):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    from test_batched_evaluation import BatchLLM
    from test_complete_native_repair import source_selection

    from litagent.agents.reviewer import ReviewerWorker
    from litagent.agents.synthesis import SynthesisWorker
    from litagent.config import EvalConfig
    from litagent.eval.scoped import ScopedEvaluationError, evaluate_scoped
    from litagent.llm.routing import ModelBoundClient
    from litagent.orchestrator.task_graph import SubTask
    from litagent.report_audit import digest

    cards = facts()
    document = analysis_document(cards)
    if invalid_global == "unsupported_science":
        document["sections"]["methods"]["first_block"]["first_sentence"][
            "text"
        ] = "EViT improves speed by 50%, while an unrelated DeiT-B setting improves speed by 70%."
    body, summaries = split(document)
    selection = source_selection(cards)
    requests = []
    evaluator = BatchLLM(incomplete_global=invalid_global is True)

    async def send(request):
        p = json.loads(request.content)
        requests.append(p)
        name = p.get("response_format", {}).get("json_schema", {}).get("name")
        if name == "grounded_body_v5_blocks":
            value = provider_body(body, cards, selection=selection)
            value["methods"][0]["fact_card_refs"] = []
        elif name == "native_body_repair_v7_refs":
            value = {
                "can_repair": True,
                "patches": {
                    "u0": {
                        **document["sections"]["methods"]["first_block"][
                            "first_sentence"
                        ],
                        "additional_units": [],
                    }
                },
            }
        elif name == "grounded_summaries_v3_layout":
            value = provider_summaries(summaries, p)
        elif len(requests) == 4:
            value = {
                "score": 0.9,
                "verdict": "accept",
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
                "issues": [],
            }
        else:
            result = await evaluator.chat(
                p["messages"], response_format=p.get("response_format")
            )
            value = json.loads(result.content)
            if invalid_global == "unsupported_science" and value.get("decisions"):
                decision = value["decisions"][0]
                decision.update(
                    status="unsupported",
                    reason_code="insufficient_support",
                    support_basis=[],
                )
                decision["aspect_checks"][0].update(
                    supported=False,
                    support_ids=[],
                    explanation="Controlled unsupported premise, despite valid punctuation.",
                )
            if "analyses" in p.get("response_format", {}).get("json_schema", {}).get(
                "schema", {}
            ).get("properties", {}):
                from litagent.body_quality import ROLES
                from litagent.review_structure import structure_sections

                payload = json.loads(p["messages"][1]["content"])
                section_by_unit = {
                    u["source_unit_id"]: u["section"]
                    for u in payload["report_structure"]
                }
                titles = dict(structure_sections("academic_v2", "en"))
                grouped = {
                    key: [
                        c["claim_id"]
                        for c in payload["assertions"]
                        if section_by_unit[c["source_unit_id"]] == titles[key]
                    ]
                    for key in ROLES
                }
                value["limitations"] = {
                    "adequate": True,
                    "claim_ids": grouped["open_problems"],
                    "explanation": "Controlled source-backed fixture boundary.",
                }
                value["analyses"] = [
                    {
                        "section_key": key,
                        "adequate": True,
                        "claim_ids": grouped[key],
                        "explanation": "Controlled located fixture.",
                    }
                    for key in ROLES
                ]
        if name == "native_body_repair_v7_refs":
            value = provider_repair(value, cards)
        return httpx.Response(
            200,
            json={
                "id": "consumer",
                "object": "chat.completion",
                "created": 0,
                "model": p["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(value)},
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            },
        )

    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    owner = OpenAICompatibleClient(
        "http://offline/v1",
        "lite",
        api_key="offline",
        cost_budget=budget,
        transport_max_attempts=1,
    )
    await owner._client.close()
    owner._client = AsyncOpenAI(
        api_key="offline",
        base_url="http://offline/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)),
    )
    pro = ModelBoundClient(owner, "pro", "generation")
    lite = ModelBoundClient(owner, "lite", "evaluation")
    synthesis = SynthesisWorker(
        pro,
        evidence_selector=SimpleNamespace(select=AsyncMock(return_value=selection)),
        run_id_provider=lambda: "actual-chain",
        repair_state_path_provider=lambda: tmp_path / "repairs.json",
    )
    token = set_task_id("synthesis")
    try:
        from litagent.agents.adversarial import AdversarialReviewWorker
        from litagent.config import load_config
        from litagent.runner import LitAgent

        worker = AdversarialReviewWorker(
            pro, synthesis, ReviewerWorker(lite), max_rounds=1
        )
        result = await worker.execute(
            SubTask(
                "adversarial_review",
                "write",
                "adversarial_review",
                input_data={
                    "query": "Explain EViT on DeiT-S",
                    "quality_policy": "strict_auto_v2",
                    "review_structure": "academic_v2",
                    "report_language": "en",
                    "body_quality": "analysis_v1",
                    "generation_strategy": "body_summary_v1",
                    "verification": {"judge": False, "jev": False},
                },
            )
        )
        assert len(result["targeted_repair_state"]["attempts"]) == 1
        assert (
            result["writing_bindings"]["native_wire_input_projection"][
                "scientific_verdict"
            ]
            == "not_evaluated"
        )
        assert result["fact_cards"] == cards and result["accepted"]
        reset_task_id(token)
        token = set_task_id("evaluation")
        context = LitAgent(load_config())._build_eval_context(
            query="Explain EViT on DeiT-S",
            survey=result["final_draft"],
            results={
                "adversarial_review": result,
                "extract": [
                    {
                        "claims": [],
                        "evidence_items": list(selection.selected_items.values()),
                    }
                ],
            },
        )
        assert context["evidence_snapshot_sha256"] == result["evidence_snapshot_sha256"]
        if invalid_global is True:
            with pytest.raises(ScopedEvaluationError):
                await evaluate_scoped(
                    llm=lite,
                    survey=result["final_draft"],
                    context=context,
                    config=EvalConfig(),
                )
        else:
            assessment = await evaluate_scoped(
                llm=lite,
                survey=result["final_draft"],
                context=context,
                config=EvalConfig(),
            )
            assert assessment["factual_claim_count"] > 0
            if invalid_global == "unsupported_science":
                assert assessment["stages"]["global_check"] == "completed"
                assert any(
                    row["status"] == "unsupported" for row in assessment["assertions"]
                )
                assert strict_acceptance(assessment)["status"] == "failed"
            else:
                assert assessment["coverage"]["complete"]
        assert len(evaluator.calls) >= 3
        assert [p["model"] for p in requests[:4]] == ["pro", "pro", "pro", "lite"]
        assert all(p["model"] == "lite" for p in requests[4:])
        assert budget.snapshot()["requests_sent"] == len(requests)
        assert (
            load_targeted_repairs(tmp_path / "repairs.json", "actual-chain")
            == result["targeted_repair_state"]
        )
    finally:
        reset_task_id(token)
        await owner.close()


def test_body_repair_groups_keep_conditions_and_every_non_target_unit():
    cards = facts(
        "On DeiT-S, EViT improves speed by 50%. On DeiT-B, EViT improves speed by 20%."
    )
    body, _ = split(analysis_document(cards))
    binding = card_reference_binding(
        cards, _writing_scopes(cards, "academic_v2", "en", None)
    )
    wire = provider_body(body, cards)
    wire["methods"][0][
        "text"
    ] = "On DeiT-S, EViT improves speed by 50%; on DeiT-B, EViT improves speed by 20%."
    messages, rf = request(cards, array_wire=True)
    _, body_rf = prepare_body_request(messages, rf)
    with pytest.raises(GroundedWritingError) as failure:
        validate_body(json.dumps(wire), body_rf, cards, "academic_v2", "en", None)
    native, _ = canonical_from_array(
        expand_body(decode_typed_body(wire, binding)), body_only=True
    )
    candidate = json.dumps(native, ensure_ascii=False)
    prepared, schema, contract = prepare_native_repair_request(
        messages,
        candidate,
        failure.value.diagnostics,
        cards=cards,
        structure="academic_v2",
        language="en",
        body_only=True,
    )
    packet = json.loads(prepared[1]["content"])
    original = json.loads(messages[1]["content"])
    assert (
        packet["sources"] == original["sources"]
        and packet["current_candidate"] == native
    )
    assert (
        "body_unit_catalog" not in packet
        and packet["repair_permission_source"] == "native_patch_schema_enums"
    )
    assert contract["version"] == "complete-native-body-repair-v6"
    target = next(iter(contract["targets"].values()))
    assert target["must_remain_scientific"]
    patch = {
        "kind": "scientific",
        "scope_notice_id": "none",
        "text": "On DeiT-S, EViT improves speed by 50%.",
        "fact_card_ids": list(cards),
        "additional_units": [
            {
                "kind": "scientific",
                "scope_notice_id": "none",
                "text": "On DeiT-B, EViT improves speed by 20%.",
                "fact_card_ids": list(cards),
            }
        ],
    }
    result = {"can_repair": True, "patches": {"u0": patch}}
    restored, proof = apply_native_repairs(
        candidate, json.dumps(result), contract, cards=cards
    )
    document = json.loads(restored)
    for key in native["sections"]:
        if key != "methods":
            assert document["sections"][key] == native["sections"][key]
    methods = document["sections"]["methods"]["first_block"]
    assert methods["first_sentence"]["text"].endswith("50%.")
    assert methods["additional_sentences"][0]["text"].endswith("20%.")
    assert proof["modified_paths"] == [target["path"]]
    assert set(document["sections"]) == set(body["sections"])
    bad = copy.deepcopy(result)
    bad["patches"]["u0"].update(
        kind="gap", fact_card_ids=[], text="", scope_notice_id="evidence_limit"
    )
    with pytest.raises(ValueError):
        apply_native_repairs(candidate, json.dumps(bad), contract, cards=cards)
    bad = copy.deepcopy(result)
    bad["patches"]["extra"] = bad["patches"]["u0"]
    with pytest.raises(ValueError):
        apply_native_repairs(candidate, json.dumps(bad), contract, cards=cards)


def test_notice_order_is_not_silently_projected_to_another_provider_position():
    cards = facts()
    body, _ = split(analysis_document(cards))
    wire = provider_body(body, cards)
    wire["scope_notices"] = [
        {
            "section_key": "open_problems",
            "after_row": 0,
            "scope_notice_id": "evidence_limit",
        },
        {
            "section_key": "introduction",
            "after_row": 0,
            "scope_notice_id": "review_scope",
        },
    ]
    binding = card_reference_binding(
        cards, _writing_scopes(cards, "academic_v2", "en", None)
    )
    with pytest.raises(ValueError, match="typed_body_notice_order_invalid"):
        decode_typed_body(wire, binding)


def test_missing_chapter_sources_are_a_material_gap_before_transport():
    messages, rf = request(facts(), array_wire=True)
    schema = rf["json_schema"]["schema"]["properties"]["sections"]["properties"][
        "methods"
    ]["properties"]["blocks"]["items"]["properties"]["sentences"]["items"]
    schema["properties"]["fact_card_ids"]["items"]["enum"] = []
    with pytest.raises(ValueError, match="staged_body_section_sources_unavailable"):
        prepare_body_request(messages, rf)
