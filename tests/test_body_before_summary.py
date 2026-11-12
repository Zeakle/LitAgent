"""Keep summary dependencies explicit and reject every located reuse error."""

import copy
import hashlib
import json
from unittest.mock import AsyncMock

import pytest
from test_native_writing_schema import assert_projected_sources, packet
from test_strict_quality import to_native

from litagent.agents.synthesis import SynthesisWorker
from litagent.llm.client import LLMResponse
from litagent.observability.recorder import RedactingTraceHook
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    GroundedWritingError,
    prepare_grounded_request,
    render_grounded_report,
    render_grounded_report_traced,
)


@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_native_body_precedes_summaries_without_changing_display_or_sources(
    structure, language
):
    cards, selection, report, messages = packet(structure, language)
    before = copy.deepcopy(messages)
    prepared, fmt = prepare_grounded_request(messages)
    original = json.loads(messages[1]["content"])
    current = json.loads(prepared[1]["content"])
    expected = [
        k for k in original["section_keys"] if k not in {"abstract", "conclusion"}
    ] + ["abstract", "conclusion"]
    assert current["generation_section_order"] == expected
    schema = fmt["json_schema"]["schema"]["properties"]["sections"]
    assert list(schema["properties"]) == schema["required"] == expected
    assert current["section_keys"] == original["section_keys"]
    assert current["section_contract"] == [
        {k: v for k, v in scope.items() if k != "allowed_fact_card_ids"}
        for scope in original["section_contract"]
    ]
    assert_projected_sources(original, current)
    assert "Use exactly the supplied section_key order." not in prepared[0]["content"]
    assert "emit the required section" not in prepared[0]["content"]
    assert "actually used in its scientific units" in prepared[0]["content"]
    native = to_native(report)
    native["sections"] = {key: native["sections"][key] for key in expected}
    body = render_grounded_report(
        json.dumps(native),
        cards,
        structure,
        language,
        evidence_selection=selection,
        require_native=True,
    )
    headings = [line for line in body.splitlines() if line.startswith("## ")]
    assert headings[0].endswith("摘要" if language == "zh-CN" else "Abstract")
    assert headings[-1].endswith("结论" if language == "zh-CN" else "Conclusion")
    assert messages == before and body.count("[E:") == 7


@pytest.mark.parametrize("native", [False, True])
def test_all_summary_reuse_and_scope_errors_are_bound_to_original_candidate(native):
    cards, selection, report, _ = packet()
    for section in report["sections"]:
        if section["section_key"] not in {"abstract", "conclusion"}:
            section["blocks"][0]["sentences"][0]["kind"] = "organization"
    intro = next(s for s in report["sections"] if s["section_key"] == "introduction")
    intro["blocks"][0]["sentences"][0]["fact_card_ids"] = [list(cards)[1]]
    raw = json.dumps(to_native(report) if native else report)
    events = []
    with pytest.raises(
        GroundedWritingError, match="summary_card_not_in_body:abstract"
    ) as caught:
        render_grounded_report_traced(
            raw,
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=native,
            trace_hook=lambda e, d: events.append((e, d)),
            parent_task_id="writer",
        )
    error = caught.value
    assert error.candidate_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert [d["reason_code"] for d in error.diagnostics] == [
        "strict_writing_summary_card_not_in_body:abstract",
        "strict_writing_summary_card_not_in_body:conclusion",
        "strict_writing_out_of_scope_fact_card",
    ]
    for d in error.diagnostics[:2]:
        assert d["invalid_fact_card_ids"] == [list(cards)[0]]
        assert d["allowed_fact_card_ids"] == []
        assert d["existing_evidence_ids"] == ["a"]
        if native:
            assert (
                d["raw_json_path"]
                == f"sections.{d['section_key']}.first_block.first_sentence"
            )
    local = next(d for e, d in events if e == "writing.validation.failed")
    assert local["diagnostic_count"] == 3 and local["diagnostics"] == error.diagnostics
    safe = []
    RedactingTraceHook(lambda e, d: safe.append((e, d)), payload_mode="metadata_only")(
        "writing.validation.failed", local
    )
    assert safe[0][1]["diagnostic_count"] == 3
    assert "diagnostics" not in safe[0][1] and "candidate_text" not in str(safe)
    assert json.dumps(to_native(report) if native else report) == raw


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_layout_and_reuse_failures_are_reported_together(language):
    cards, selection, report, _ = packet(language=language)
    for section in report["sections"]:
        if section["section_key"] in {"abstract", "conclusion"}:
            section["blocks"][0]["heading"] = "Summary" if language == "en" else "摘要"
        else:
            section["blocks"][0]["sentences"][0]["kind"] = "organization"
    raw = json.dumps(to_native(report, cards))
    with pytest.raises(
        GroundedWritingError, match="summary_layout_invalid:abstract"
    ) as caught:
        render_grounded_report(
            raw,
            cards,
            "academic_v2",
            language,
            evidence_selection=selection,
            require_native=True,
        )
    assert len(caught.value.diagnostics) == 4
    assert {d["section_key"] for d in caught.value.diagnostics} == {
        "abstract",
        "conclusion",
    }
    assert caught.value.candidate_sha256 == hashlib.sha256(raw.encode()).hexdigest()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_initial_writer_records_summary_dependency_errors_once_without_retry(
    policy,
):
    cards, selection, report, _ = packet()
    for section in report["sections"]:
        if section["section_key"] not in {"abstract", "conclusion"}:
            section["blocks"][0]["sentences"][0]["kind"] = "organization"
    raw = json.dumps(to_native(report, cards))
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(raw)
    selector = AsyncMock()
    selector.select.return_value = selection
    events = []
    worker = SynthesisWorker(
        llm, evidence_selector=selector, trace_hook=lambda e, d: events.append((e, d))
    )
    with pytest.raises(GroundedWritingError, match="summary_card_not_in_body:abstract"):
        await worker.execute(
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
    llm.chat.assert_awaited_once()
    sent = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert sent["generation_section_order"][-2:] == ["abstract", "conclusion"]
    local = next(d for e, d in events if e == "writing.validation.failed")
    assert local["candidate_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert local["diagnostic_count"] == 2 and local["phase"] == "initial"
