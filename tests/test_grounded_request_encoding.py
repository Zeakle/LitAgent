"""Preserve complete science and permissions while refusing over-budget calls."""

import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_native_writing_schema import (
    assert_projected_sources,
    packet,
    section_definition,
)

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.observability.recorder import RedactingTraceHook
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    check_grounded_request_budget,
    digest,
    prepare_grounded_request,
)
from litagent.safety.budget import CostBudget, strict_request_reservation


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
def test_short_local_refs_preserve_entire_schema_and_restore_budget(
    structure, language
):
    _, _, _, messages = packet(structure=structure, language=language)
    prepared, fmt = prepare_grounded_request(messages)
    schema = fmt["json_schema"]["schema"]
    expanded = copy.deepcopy(fmt)
    expanded_schema = expanded["json_schema"]["schema"]
    replacements = {}
    for key in schema["properties"]["sections"]["required"]:
        block_ref = schema["properties"]["sections"]["properties"][key]["properties"][
            "first_block"
        ]["$ref"]
        sentence_ref = section_definition(schema, key, "block")["properties"][
            "first_sentence"
        ]["$ref"]
        replacements[block_ref.removeprefix("#/$defs/")] = key + "_block"
        replacements[sentence_ref.removeprefix("#/$defs/")] = key + "_sentence"
    assert set(replacements) == set(schema["$defs"])
    expanded_schema["$defs"] = {
        replacements[key]: definition
        for key, definition in expanded_schema["$defs"].items()
    }

    def rename_refs(value):
        if isinstance(value, dict):
            if "$ref" in value:
                name = value["$ref"].removeprefix("#/$defs/")
                # The producer deliberately shares ref dictionaries across
                # fields; a deep copy retains those aliases during this walk.
                assert name in replacements or name in replacements.values()
                value["$ref"] = "#/$defs/" + replacements.get(name, name)
            for child in value.values():
                rename_refs(child)
        elif isinstance(value, list):
            for child in value:
                rename_refs(child)

    rename_refs(expanded_schema)
    for key in schema["properties"]["sections"]["required"]:
        for kind in ("sentence", "block"):
            short = copy.deepcopy(section_definition(schema, key, kind))
            rename_refs(short)
            assert short == section_definition(expanded_schema, key, kind)
    full = strict_request_reservation(prepared, 20000, fmt)
    assert strict_request_reservation(prepared, 20000, expanded) > full
    budget = CostBudget(500000)
    budget.configure_strict()
    llm = SimpleNamespace(_cost_budget=budget, chat=AsyncMock())
    before = budget.snapshot()
    check_grounded_request_budget(llm, prepared, fmt, context_tokens=full)
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_grounded_request_budget(llm, prepared, expanded, context_tokens=full)
    assert budget.snapshot() == before
    llm.chat.assert_not_awaited()


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_only_host_permissions_and_renderer_identity_are_projected(language):
    _, _, _, messages = packet(language=language)
    original = json.loads(messages[1]["content"])
    original["current_draft"] = "完整原文 [E:a]"
    original["located_feedback"] = [{"source_quote": "完整原文", "reason": "条件缺失"}]
    messages[1]["content"] = json.dumps(original, ensure_ascii=False)
    frozen = copy.deepcopy(messages)
    prepared, fmt = prepare_grounded_request(messages)
    moved = json.loads(prepared[1]["content"])
    assert messages == frozen
    assert prepared[1]["content"] == json.dumps(
        moved, ensure_ascii=False, separators=(",", ":")
    )
    for field in original.keys() - {
        "section_contract",
        "grounded_report_schema",
        "cards",
        "sources",
        "current_draft",
        "located_feedback",
    }:
        assert moved[field] == original[field]
    assert_projected_sources(original, moved)
    cid = next(
        cid for cid, card in original["cards"].items() if card["evidence_id"] == "a"
    )
    assert moved["current_draft"] == f"完整原文 [F:{cid}]"
    assert moved["located_feedback"] == original["located_feedback"]
    for section, projected in zip(
        original["section_contract"], moved["section_contract"]
    ):
        assert projected == {
            k: v for k, v in section.items() if k != "allowed_fact_card_ids"
        }
        allowed = section["allowed_fact_card_ids"]
        enum = section_definition(
            fmt["json_schema"]["schema"], section["section_key"], "sentence"
        )["properties"]["fact_card_ids"]["items"]["enum"]
        assert enum == allowed
        assert [
            cid
            for cid, card in moved["cards"].items()
            if section["section_key"] in card["allowed_section_keys"]
        ] == allowed


@pytest.mark.parametrize("mode", ["missing", "empty", "null", "integer"])
def test_invalid_host_citation_identity_is_rejected_before_projection(mode):
    _, _, _, messages = packet()
    original = json.loads(messages[1]["content"])
    card = next(iter(original["cards"].values()))
    if mode == "missing":
        del card["evidence_id"]
    else:
        card["evidence_id"] = {"empty": "", "null": None, "integer": 1}[mode]
    messages[1]["content"] = json.dumps(original, ensure_ascii=False)
    before = copy.deepcopy(messages)
    with pytest.raises(ValueError, match="strict_writing_schema_contract_invalid"):
        prepare_grounded_request(messages)
    assert messages == before


@pytest.mark.parametrize("failure", ["context", "stage", "stopped"])
def test_rejected_request_keeps_full_local_input_and_private_trace_boundary(failure):
    _, _, _, messages = packet()
    prepared, fmt = prepare_grounded_request(messages)
    full = strict_request_reservation(prepared, 20000, fmt)
    budget = CostBudget(500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    context = full - 1 if failure == "context" else 117964
    if failure == "stage":
        budget.record({"prompt_tokens": 200000 - full + 1, "completion_tokens": 0})
    elif failure == "stopped":
        budget.stop_after_worker_failure()
    llm = SimpleNamespace(_cost_budget=budget, chat=AsyncMock())
    local, public = [], []
    safe = RedactingTraceHook(lambda e, d: public.append((e, d)), "metadata_only")

    def trace(event, data):
        local.append((event, data))
        safe(event, data)

    before = budget.snapshot()
    with pytest.raises(ValueError):
        check_grounded_request_budget(
            llm,
            prepared,
            fmt,
            trace_hook=trace,
            parent_task_id="writer",
            stage="revision",
            context_tokens=context,
        )
    llm.chat.assert_not_awaited()
    assert budget.snapshot() == before
    event, detail = local[0]
    assert event == "writing.request.rejected"
    assert detail["messages"] == prepared and detail["response_format"] == fmt
    assert detail["input_sha256"] == digest([prepared, fmt])
    assert detail["budget"]["reservation"] == full
    assert detail["max_tokens"] == 20000 and detail["phase"] == "revision"
    assert public[0][1]["budget"] == detail["budget"]
    assert "messages" not in public[0][1] and "response_format" not in public[0][1]
    assert "source_quote" not in json.dumps(public)


def test_failing_trace_sink_cannot_mask_refusal_or_issue_request():
    _, _, _, messages = packet()
    prepared, fmt = prepare_grounded_request(messages)

    def broken_sink(*args):
        raise RuntimeError("local recorder failed")

    llm = SimpleNamespace(chat=AsyncMock())
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_grounded_request_budget(
            llm, prepared, fmt, context_tokens=1, trace_hook=broken_sink
        )
    llm.chat.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_actual_first_writer_saves_full_oversized_sources_without_call(policy):
    _, selection, _, _ = packet()
    source = "EViT condition and metric. " * 10000
    for item in selection.selected_items.values():
        item["supporting_text"] = source
        item["text"] = source
    selector, llm = AsyncMock(), AsyncMock()
    selector.select.return_value = selection
    events = []
    writer = SynthesisWorker(
        llm, evidence_selector=selector, trace_hook=lambda e, d: events.append((e, d))
    )
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        await writer.execute(
            SubTask(
                "writer",
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
    llm.chat.assert_not_awaited()
    detail = next(d for e, d in events if e == "writing.request.rejected")
    material = json.loads(detail["messages"][1]["content"])
    assert all(s["source_quote"] == source for s in material["sources"].values())
    assert detail["budget"]["reservation"] > detail["budget"]["context_tokens"]


def test_request_encoding_enters_only_strict_configuration_identity():
    cfg = load_config()
    assert "grounded-request-encoding-v1" not in json.dumps(
        build_config_summary(cfg, "legacy")
    )
    for policy in ["strict_v1", "strict_auto_v1"]:
        assert "grounded-request-encoding-v1" in json.dumps(
            build_config_summary(cfg, policy)
        )


def test_langfuse_budget_event_never_forwards_local_materials():
    from unittest.mock import Mock

    from litagent.observability.tracing import LangFuseTracer

    tracer = LangFuseTracer.__new__(LangFuseTracer)
    tracer._root = Mock()
    tracer._spans = {}
    tracer._handle(
        "writing.request.rejected",
        {
            "task_id": "writer",
            "phase": "revision",
            "reason_code": "over_budget",
            "budget": {"reservation": 120000},
            "input_sha256": "frozen",
            "messages": [{"content": "private scientific material"}],
            "response_format": {"private_source_ids": ["secret"]},
        },
    )
    output = tracer._root.update.call_args.kwargs
    assert output["level"] == "ERROR"
    assert output["metadata"]["writing_request"]["budget"] == {"reservation": 120000}
    assert "private" not in json.dumps(output)
