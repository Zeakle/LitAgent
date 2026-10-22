"""Keep local repair diagnostics, card encoding and scientific judgments separate."""

import copy
import hashlib
import json
from pathlib import Path

import httpx
import pytest
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_complete_native_repair import source_selection
from test_staged_native_generation import provider_body, provider_repair, split

from litagent.llm.client import OpenAICompatibleClient
from litagent.native_wire import canonical_from_array
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import CompositeTraceHook, RedactingTraceHook
from litagent.quality import (
    GroundedRequestBudgetError,
    GroundedWritingError,
    apply_native_repairs,
    build_fact_cards,
    grounded_messages,
    load_targeted_repairs,
    prepare_grounded_request,
    prepare_native_repair_request,
    targeted_repair_state,
)
from litagent.report_audit import digest
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.staged_generation import (
    BODY_REPAIR_WIRE,
    CONTRACT,
    _short_reference_fields,
    generate_staged_report,
    prepare_body_request,
    prepare_staged_body_repair,
    restore_body_repair_response,
    validate_body,
    validate_body_document,
)
from litagent.writing_passages import validate_unit_citation_scope

REAL = json.loads(Path("tests/fixtures/body-scope-errors-138.json").read_text())


@pytest.fixture
def repair_inputs():
    """Provide two sources with a genuinely narrower methods permission menu."""
    cards = build_fact_cards(
        {
            "a": {
                "evidence_id": "a",
                "paper_id": "evit",
                "chunk_key": "abstract",
                "text": "On DeiT-S, EViT improves speed by 50%. On DeiT-B, EViT improves speed by 20%.",
                "supporting_text": "On DeiT-S, EViT improves speed by 50%. On DeiT-B, EViT improves speed by 20%.",
            },
            "b": {
                "evidence_id": "b",
                "paper_id": "other",
                "chunk_key": "abstract",
                "text": "Other evidence is not permitted in the methods section.",
                "supporting_text": "Other evidence is not permitted in the methods section.",
            },
        }
    )
    selected = source_selection(cards)
    selected.section_evidence_ids["methods"] = ["a"]
    messages, rf = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            "en",
            "Explain EViT on DeiT-S",
            evidence_selection=selected,
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
            require_unit_citation_scope=True,
        ),
        array_wire=True,
    )
    original = analysis_document(cards)
    body, summaries = split(original)
    allowed = [cid for cid, c in cards.items() if c["evidence_id"] == "a"]
    for block in body["sections"]["methods"]["blocks"]:
        for unit in block["sentences"]:
            unit["fact_card_ids"] = allowed
    body["sections"]["methods"]["blocks"][0]["sentences"][0][
        "text"
    ] = "On DeiT-S, EViT improves speed by 50%; on DeiT-B, EViT improves speed by 20%."
    wire = provider_body(body, cards, selection=selected)
    _, body_rf = prepare_body_request(messages, rf)
    with pytest.raises(GroundedWritingError) as failure:
        validate_body(json.dumps(wire), body_rf, cards, "academic_v2", "en", selected)
    native, _ = canonical_from_array(body, body_only=True)
    candidate = json.dumps(native, ensure_ascii=False)
    args = dict(cards=cards, structure="academic_v2", language="en", selection=selected)
    prepared, schema, binding = prepare_staged_body_repair(
        messages, candidate, failure.value.diagnostics, **args
    )
    patch = {
        "kind": "scientific",
        "scope_notice_id": "none",
        "text": "On DeiT-S, EViT improves speed by 50%.",
        "fact_card_ids": allowed,
        "additional_units": [
            {
                "kind": "scientific",
                "scope_notice_id": "none",
                "text": "On DeiT-B, EViT improves speed by 20%.",
                "fact_card_ids": allowed,
            }
        ],
    }
    response = {"can_repair": True, "patches": {"u0": patch}}
    return dict(
        cards=cards,
        selected=selected,
        messages=messages,
        rf=rf,
        body=body,
        summaries=summaries,
        wire=wire,
        candidate=candidate,
        diagnostics=failure.value.diagnostics,
        prepared=prepared,
        schema=schema,
        binding=binding,
        response=response,
    )


def test_compact_repair_preserves_complete_sources_candidate_and_constraints(
    repair_inputs,
):
    x = repair_inputs
    packet = json.loads(x["prepared"][1]["content"])
    old_messages, old_schema, old_binding = prepare_native_repair_request(
        x["messages"],
        x["candidate"],
        x["diagnostics"],
        cards=x["cards"],
        structure="academic_v2",
        language="en",
        evidence_selection=x["selected"],
        body_only=True,
    )
    old = json.loads(old_messages[1]["content"])
    assert x["binding"]["contract"] == BODY_REPAIR_WIRE
    assert x["binding"]["canonical_binding"] == old_binding
    assert packet["sources"] == old["sources"]
    reading = x["binding"]["reading_cards"]
    assert _short_reference_fields(packet["current_candidate"], reading) == json.loads(
        x["candidate"]
    )
    assert {reading[k]: v for k, v in packet["cards"].items()} == old["cards"]
    assert (
        _short_reference_fields(packet["section_source_index"], reading)
        == old["section_source_index"]
    )
    assert (
        _short_reference_fields(packet["analysis_plan"]["requirements"], reading)
        == old["analysis_plan"]["requirements"]
    )
    for key, ids in packet["analysis_plan"]["missing_fields"].items():
        assert [reading[v] for v in ids] == old["analysis_plan"]["missing_fields"][key]
    target = x["binding"]["canonical_binding"]["targets"]["u0"]
    positions = packet["repair_targets"]["u0"]["scope_boundary_offsets"]
    assert positions and all(target["original"]["text"][i] == ";" for i in positions)
    assert x["schema"]["json_schema"]["name"] == "native_body_repair_v7_refs"

    def check_native_schema(node):
        if isinstance(node, dict):
            assert "$ref" not in node
            if node.get("type") == "object":
                assert set(node["required"]) == set(node["properties"])
            for item in node.values():
                check_native_schema(item)
        elif isinstance(node, list):
            for item in node:
                check_native_schema(item)

    check_native_schema(x["schema"]["json_schema"]["schema"])
    raw = json.dumps(provider_repair(x["response"], x["cards"]))
    restored = restore_body_repair_response(raw, x["binding"], cards=x["cards"])
    assert json.loads(restored) == x["response"]
    candidate, proof = apply_native_repairs(
        x["candidate"], restored, old_binding, cards=x["cards"]
    )
    proposed = json.loads(candidate)
    assert proof["modified_paths"] == [target["path"]]
    assert "50%" in candidate and "20%" in candidate
    for k in proposed["sections"]:
        if k != "methods":
            assert proposed["sections"][k] == json.loads(x["candidate"])["sections"][k]
    from litagent.native_wire import array_from_native

    validate_body_document(
        array_from_native(proposed, body_only=True),
        x["cards"],
        "academic_v2",
        "en",
        x["selected"],
    )


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "original_id",
        "foreign_section",
        "duplicate",
        "extra",
        "kind",
        "binding",
        "aliases",
        "null_binding",
    ],
)
def test_compact_response_rejects_forged_or_outside_permission_cards(
    repair_inputs, fault
):
    x = repair_inputs
    response = provider_repair(x["response"], x["cards"])
    binding = copy.deepcopy(x["binding"])
    patch = response["patches"]["u0"]
    if fault == "unknown":
        patch["fact_card_ids"] = ["c999"]
    elif fault == "original_id":
        patch["fact_card_ids"] = x["response"]["patches"]["u0"]["fact_card_ids"]
    elif fault == "foreign_section":
        patch["fact_card_ids"] = [
            ref
            for ref, cid in binding["reading_cards"].items()
            if x["cards"][cid]["evidence_id"] == "b"
        ]
    elif fault == "duplicate":
        patch["fact_card_ids"] *= 2
    elif fault == "extra":
        patch["outside"] = "unsupported"
    elif fault == "kind":
        patch["kind"] = "gap"
    elif fault == "binding":
        binding["sha256"] = "changed"
    elif fault == "aliases":
        binding["reading_cards"] = dict(
            reversed(list(binding["reading_cards"].items()))
        )
        keys = list(binding["reading_cards"])
        binding["reading_cards"][keys[0]], binding["reading_cards"][keys[1]] = (
            binding["reading_cards"][keys[1]],
            binding["reading_cards"][keys[0]],
        )
        binding["sha256"] = digest({k: v for k, v in binding.items() if k != "sha256"})
    else:
        binding = None
    with pytest.raises(ValueError):
        restored = restore_body_repair_response(
            json.dumps(response), binding, cards=x["cards"]
        )
        apply_native_repairs(
            x["candidate"],
            restored,
            x["binding"]["canonical_binding"],
            cards=x["cards"],
        )


@pytest.mark.parametrize("case", REAL["cases"], ids=lambda c: c["raw_json_path"])
def test_each_real_compound_error_stays_blocked_without_implicit_citations(case):
    # This tests format and identity only. No original paper or gold support
    # decision is fabricated from the writer's sentence.
    with pytest.raises(ValueError, match="citation_scope_ambiguous"):
        validate_unit_citation_scope(
            case["candidate_text"], case["actual_fact_card_ids"], "zh-CN"
        )
    assert "；" in case["candidate_text"] and case["raw_json_path"].startswith(
        "sections."
    )
    # Merely swapping punctuation can pass the local format check. It is not a
    # scientific verdict, and must still go through the independent evaluator.
    validate_unit_citation_scope(
        case["candidate_text"].replace("；", "，"),
        case["actual_fact_card_ids"],
        "zh-CN",
    )


@pytest.mark.parametrize(
    "text",
    [
        'A literal "condition A; condition B" appears in the source.',
        "The conditions (A; B) remain explicitly grouped.",
        "The source URL https://example.com/a;b describes the method.",
    ],
)
def test_protected_semicolons_are_not_mechanically_split(text):
    validate_unit_citation_scope(text, ["a"], "en")


@pytest.mark.asyncio
async def test_budget_refusal_keeps_initial_diagnostics_without_spending_a_repair(
    repair_inputs, tmp_path
):
    x = repair_inputs
    body_messages, body_rf = prepare_body_request(x["messages"], x["rf"])
    reservation = strict_request_reservation(body_messages, 20000, body_rf)
    b = CostBudget()
    b.configure_strict(halt_on_unknown_usage=True)
    snapshot = b.snapshot()
    inherited = 300000 - reservation
    snapshot.update(
        used_tokens=inherited,
        call_count=29,
        requests_sent=29,
        strict_phase=1,
        phase_usage=[98267, inherited - 98267, 0],
    )
    budget = CostBudget.from_snapshot(snapshot)
    state = targeted_repair_state(None, "budget-diagnostics")
    path = tmp_path / "repair.json"
    local, remote, calls = [], [], []
    hook = CompositeTraceHook(
        lambda e, d: local.append((e, d)),
        RedactingTraceHook(
            lambda e, d: remote.append((e, d)), payload_mode="metadata_only"
        ),
    )
    raw = json.dumps(x["wire"])

    async def send(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": "pro",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": raw},
                    }
                ],
                "usage": {
                    "prompt_tokens": reservation - 20000,
                    "completion_tokens": 15,
                    "total_tokens": reservation - 19985,
                },
            },
        )

    owner = OpenAICompatibleClient(
        "http://offline/v1",
        "pro",
        api_key="offline",
        cost_budget=budget,
        transport_max_attempts=1,
        trace_hook=hook,
    )
    await owner._client.close()
    owner._client = AsyncOpenAI(
        api_key="offline",
        base_url="http://offline/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(send)),
    )
    token = set_task_id("synthesis")
    try:
        with pytest.raises(GroundedRequestBudgetError):
            await generate_staged_report(
                owner,
                x["messages"],
                x["rf"],
                cards=x["cards"],
                structure="academic_v2",
                language="en",
                selection=x["selected"],
                checkpoint_path=path,
                repair_state=state,
                trace_hook=hook,
            )
        assert len(calls) == 1 and not state["attempts"]
        shared = json.loads(path.read_text())
        assert shared["staged_body_diagnostics"] == x["diagnostics"]
        assert shared["staged_body_wire_binding"]["contract"] == BODY_REPAIR_WIRE
        journal = json.loads(
            next(tmp_path.glob("repair.json.generation-*.json")).read_text()
        )
        failure = journal["body_validation_failure"]
        assert failure["diagnostics"] == x["diagnostics"]
        assert (
            failure["raw_response_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
        )
        assert failure["scientific_verdict"] == "not_evaluated"
        assert journal["error_type"] == "GroundedRequestBudgetError"
        assert journal["error_code"] == "evaluation_run_budget_exceeded"
        assert load_targeted_repairs(path, state["run_id"]) == state
        event = next(d for e, d in remote if e == "writing.wire.failed")
        assert event["diagnostic_count"] == len(x["diagnostics"])
        assert event["body_failure_codes"] == [
            "strict_writing_unit_citation_scope_ambiguous"
        ]
        assert raw not in json.dumps(remote) and x["diagnostics"][0][
            "candidate_text"
        ] not in json.dumps(remote)
        with pytest.raises(ValueError, match="restart_no_replay"):
            await generate_staged_report(
                owner,
                x["messages"],
                x["rf"],
                cards=x["cards"],
                structure="academic_v2",
                language="en",
                selection=x["selected"],
                checkpoint_path=path,
                repair_state=state,
                trace_hook=hook,
            )
        assert len(calls) == 1 and budget.snapshot()["requests_sent"] == 30
    finally:
        reset_task_id(token)
        await owner.close()


def test_new_staged_contract_changes_only_explicit_staged_identity(monkeypatch):
    from litagent import staged_generation
    from litagent.workbench import RunOptions

    single = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    staged = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v1",
    )
    assert CONTRACT == "native-body-summary-v11"
    previous_single = single.fingerprint()
    monkeypatch.setattr(staged_generation, "CONTRACT", "native-body-summary-v6")
    previous_staged = staged.fingerprint()
    monkeypatch.setattr(staged_generation, "CONTRACT", CONTRACT)
    assert staged.fingerprint() != previous_staged
    assert single.fingerprint() == previous_single
