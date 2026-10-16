"""Apply native chapter enums without hiding schema tokens or source material."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_chapter_scope_contract import materials, unit
from test_strict_quality import to_native

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.eval.base import CTX_EVIDENCE, CTX_SELECTED_EVIDENCE
from litagent.eval.scoped import check_request_budget
from litagent.llm.client import LLMResponse, OpenAICompatibleClient
from litagent.observability.recorder import RedactingTraceHook
from litagent.quality import (
    grounded_messages,
    prepare_grounded_request,
    render_grounded_report,
)
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget, strict_request_reservation


def packet(structure="academic_v2", language="en"):
    """Build a complete writing request from the actual shared producer."""
    cards, selection, report = materials(structure, language)
    messages = grounded_messages(
        cards, structure, language, "Compare EViT", evidence_selection=selection
    )
    for section in report["sections"]:
        for block in section["blocks"]:
            block.setdefault("heading", "")
            block.setdefault("type", "paragraph")
    return cards, selection, report, messages


def section_definition(schema, key, kind):
    """Resolve a section's actual local references without deriving their names."""
    block_ref = schema["properties"]["sections"]["properties"][key]["properties"][
        "first_block"
    ]["$ref"]
    block = schema["$defs"][block_ref.removeprefix("#/$defs/")]
    if kind == "block":
        return block
    sentence_ref = block["properties"]["first_sentence"]["$ref"]
    return schema["$defs"][sentence_ref.removeprefix("#/$defs/")]


def assert_projected_sources(original, moved):
    """Verify complete source values and every card's original ownership edge."""
    assert list(moved["sources"].values()) == list(original["sources"].values())
    assert set(moved["cards"]) == set(original["cards"])
    for cid, card in moved["cards"].items():
        canonical = original["cards"][cid]
        assert {k: v for k, v in card.items() if k != "source_ref"} == {
            k: v for k, v in canonical.items() if k not in {"source_ref", "evidence_id"}
        }
        assert (
            moved["sources"][card["source_ref"]]
            == original["sources"][canonical["source_ref"]]
        )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
def test_native_schema_matches_host_permissions_and_preserves_full_sources(
    structure, language
):
    cards, selection, report, messages = packet(structure, language)
    before = copy.deepcopy(messages)
    prepared, fmt = prepare_grounded_request(messages)
    assert messages == before
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(to_native(report))
    original = json.loads(messages[1]["content"])
    moved = json.loads(prepared[1]["content"])
    expected = copy.deepcopy(original)
    del expected["grounded_report_schema"]
    for scope in expected["section_contract"]:
        del scope["allowed_fact_card_ids"]
    for card in expected["cards"].values():
        del card["evidence_id"]
    assert {
        k: v
        for k, v in moved.items()
        if k
        not in {"generation_section_order", "sources", "cards", "reference_contract"}
    } == {k: v for k, v in expected.items() if k not in {"sources", "cards"}}
    assert_projected_sources(original, moved)
    for cid, view in moved["cards"].items():
        assert original["cards"][cid]["evidence_id"] == cards[cid]["evidence_id"]
        assert moved["sources"][view["source_ref"]] == {
            "paper_id": cards[cid]["paper_id"],
            "source_quote": cards[cid]["source_quote"],
            "fact_card_ids": [
                other_id
                for other_id, other_view in moved["cards"].items()
                if other_view["source_ref"] == view["source_ref"]
            ],
        }
    assert "allowed_fact_card_ids" not in prepared[0]["content"]
    assert (
        schema["properties"]["sections"]["required"]
        == moved["generation_section_order"]
    )
    for scope in original["section_contract"]:
        allowed = section_definition(schema, scope["section_key"], "sentence")[
            "properties"
        ]["fact_card_ids"]["items"]["enum"]
        assert allowed == scope["allowed_fact_card_ids"]
    assert (
        render_grounded_report(
            json.dumps(to_native(report)),
            cards,
            structure,
            language,
            evidence_selection=selection,
            require_native=True,
        ).count("[E:")
        == 7
    )
    for banned in (
        '"allOf"',
        '"if"',
        '"then"',
        '"uniqueItems"',
        '"minItems"',
        '"maxItems"',
        '"minLength"',
        '"maxLength"',
    ):
        assert banned not in json.dumps(schema)


@pytest.mark.parametrize("mode", ["unknown", "outside", "missing", "extra", "field"])
def test_native_schema_rejects_wrong_cards_order_and_undeclared_fields(mode):
    cards, _, report, messages = packet()
    _, fmt = prepare_grounded_request(messages)
    if mode in {"unknown", "outside"}:
        unit(report, "introduction")["fact_card_ids"] = [
            "missing" if mode == "unknown" else list(cards)[1]
        ]
    native = to_native(report)
    if mode == "missing":
        del native["sections"]["introduction"]
    elif mode == "extra":
        native["sections"]["invented"] = copy.deepcopy(native["sections"]["conclusion"])
    elif mode == "field":
        native["approved"] = True
    assert list(Draft202012Validator(fmt["json_schema"]["schema"]).iter_errors(native))


def test_empty_chapter_cannot_borrow_visible_cards_and_host_still_checks_science():
    cards, selection, report, _ = packet()
    selection.section_evidence_ids["introduction"] = []
    messages = grounded_messages(
        cards, "academic_v2", "en", "EViT", evidence_selection=selection
    )
    _, fmt = prepare_grounded_request(messages)
    validator = Draft202012Validator(fmt["json_schema"]["schema"])
    assert list(validator.iter_errors(to_native(report)))
    unit(report, "introduction")["fact_card_ids"] = []
    validator.validate(to_native(report))
    with pytest.raises(ValueError, match="strict_writing_uncited_science"):
        render_grounded_report(
            json.dumps(to_native(report)),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )


@pytest.mark.parametrize("mode", ["version", "schema", "scope", "missing", "duplicate"])
def test_inconsistent_internal_schema_fails_before_transport(mode):
    _, _, _, messages = packet()
    data = json.loads(messages[1]["content"])
    if mode == "version":
        data["writing_contract"] = "obsolete"
    elif mode == "schema":
        data["grounded_report_schema"] = {}
    elif mode == "scope":
        data["section_contract"][0]["allowed_fact_card_ids"] = [10]
    elif mode == "missing":
        del data["section_contract"]
    else:
        data["section_contract"].append(data["section_contract"][0])
    messages[1]["content"] = json.dumps(data)
    with pytest.raises(ValueError, match="strict_writing_schema_contract_invalid"):
        prepare_grounded_request(messages)


@pytest.mark.parametrize("limit", ["context", "stage"])
def test_preflight_counts_entire_native_schema_without_borrowing_reserves(limit):
    _, _, _, messages = packet()
    messages, fmt = prepare_grounded_request(messages)
    old = strict_request_reservation(messages, 20000)
    full = strict_request_reservation(messages, 20000, fmt)
    assert full > old
    budget = CostBudget(500000)
    budget.configure_strict()
    llm = SimpleNamespace(_cost_budget=budget)
    context = full - 1 if limit == "context" else 117964
    if limit == "stage":
        budget.record({"prompt_tokens": 200000 - full + 1, "completion_tokens": 0})
    before = budget.snapshot()
    with pytest.raises(
        ValueError,
        match="evaluation_"
        + ("context" if limit == "context" else "run")
        + "_budget_exceeded",
    ):
        check_request_budget(llm, messages, 20000, context, response_format=fmt)
    assert budget.snapshot() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["complete", "provider_reject", "context"])
async def test_actual_sdk_schema_budget_and_trace_no_fallback(mode):
    _, _, report, messages = packet()
    messages, fmt = prepare_grounded_request(messages)
    if mode == "context":
        fmt = copy.deepcopy(fmt)
        fmt["json_schema"]["schema"]["description"] = "x" * 120000
    budget = CostBudget(500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    local, public = [], []
    safe = RedactingTraceHook(lambda e, d: public.append((e, d)), "metadata_only")

    def record(event, data):
        local.append((event, data))
        safe(event, data)

    client = OpenAICompatibleClient(
        "https://example.test/v1",
        "offline",
        api_key="offline",
        cost_budget=budget,
        trace_hook=record,
    )
    sdk = client._client
    create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(to_native(report)),
                        tool_calls=None,
                        reasoning_content=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=25, completion_tokens=35),
            model="offline",
        )
    )
    if mode == "provider_reject":
        create.side_effect = ValueError("schema unsupported")
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    reserved = []
    original_reserve = budget.reserve

    def reserve(tokens):
        reserved.append(tokens)
        return original_reserve(tokens)

    budget.reserve = reserve
    try:
        if mode == "complete":
            await client.chat(messages, max_tokens=20000, response_format=fmt)
        else:
            with pytest.raises(
                ValueError,
                match=(
                    "schema unsupported"
                    if mode == "provider_reject"
                    else "strict_context_budget_exceeded"
                ),
            ):
                await client.chat(messages, max_tokens=20000, response_format=fmt)
    finally:
        await sdk.close()
    if mode == "context":
        create.assert_not_awaited()
        assert not reserved and budget.used == 0
    else:
        create.assert_awaited_once()
        assert create.await_args.kwargs["response_format"] == fmt
        assert reserved == [strict_request_reservation(messages, 20000, fmt)]
        start = next(d for e, d in local if e == "llm.start")
        assert start["response_format"] == fmt
        assert "response_format" not in next(d for e, d in public if e == "llm.start")
        assert budget.used == (60 if mode == "complete" else reserved[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path,policy",
    [("repair", "strict_auto_v1"), ("repair", "strict_v1"), ("human", "strict_v1")],
)
async def test_actual_runner_revision_paths_send_native_schema_and_keep_rejected_original(
    path, policy
):
    from test_reference_occurrence_policy import bound

    cards, selection, candidate, _ = packet()
    original = render_grounded_report(
        json.dumps(candidate), cards, "academic_v2", "en", evidence_selection=selection
    )
    unit(candidate, "introduction")["fact_card_ids"] = [list(cards)[1]]
    budget = CostBudget(500000)
    budget.configure_strict()
    llm = SimpleNamespace(
        chat=AsyncMock(
            return_value=LLMResponse(
                json.dumps(to_native(candidate)), usage={"usage_available": True}
            )
        ),
        _cost_budget=budget,
    )
    agent = LitAgent(load_config())
    agent._llm = llm
    agent._synthesis = SynthesisWorker(llm)
    agent._evaluate = AsyncMock()
    agent._build_eval_context = lambda **kwargs: {
        CTX_EVIDENCE: selection.selected_items,
        CTX_SELECTED_EVIDENCE: list(selection.selected_items.values()),
    }
    report = {
        "survey": original,
        "metadata": {
            "quality_policy": policy,
            "fact_cards": cards,
            "review_structure": "academic_v2",
            "review_methodology": {},
        },
        "evaluation": {
            "faithfulness": {
                "details": {
                    "unsupported_claims": [
                        {
                            "reason_code": "uncited_claim",
                            "evidence_ids": [],
                            "source_quote": unit(candidate, "introduction")["text"],
                        }
                    ]
                }
            }
        },
        "quality": {"status": "failed"},
    }
    if path == "repair":
        report["evaluation"] = bound(original)
    before = copy.deepcopy(report)
    results = {"adversarial_review": {"evidence_selection": selection.to_dict()}}
    if path == "repair":
        outcome = await agent._attempt_evidence_rewrite(
            query="EViT",
            report_data=report,
            results=results,
            initial_evaluation=report["evaluation"],
            initial_quality=report["quality"],
        )
        assert not outcome.committed
    else:
        _, outcome = await agent._attempt_human_revision(
            query="EViT",
            report_data=report,
            results=results,
            feedback="Fix located issue",
            report_language="en",
        )
        assert not outcome["committed"]
    llm.chat.assert_awaited_once()
    options = llm.chat.await_args.kwargs
    assert options["response_format"]["type"] == "json_schema"
    assert options["response_format"]["json_schema"]["strict"] is True
    if path == "repair":
        assert options["response_format"]["json_schema"]["name"] == "fragment_repair_v2"
    assert options["transport_max_attempts"] == 1
    assert report == before
    agent._evaluate.assert_not_awaited()
