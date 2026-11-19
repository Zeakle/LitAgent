"""Keep rejected-stage feedback local, verbatim, bounded, and authenticated."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_separate_review_pipeline import execute
from test_stream_writer_pipeline import StreamingGateway, stream_config
from test_writer_v2 import case  # noqa: F401

from litagent import inline_citations, stage_repair
from litagent.writer_source_view import GROUNDING_INSTRUCTION
from litagent.writing_v2 import LIVE_SUMMARY_PROMPT, request_messages, schema_for


def record(diagnostics):
    return {
        "category": "evaluation_revision",
        "stage": "summary",
        "reason_code": "writer_inline_citation_without_prose",
        "raw_sha256": "a" * 64,
        "diagnostics": diagnostics,
    }


def test_late_failure_is_visible_even_when_candidate_prefix_does_not_reach_it():
    text = '引号"与反斜线\\均应保留。' * 90 + "需要核对的结果。[C:c7]"
    raw = json.dumps(
        {
            "abstract": [{"text": "前置摘要。" * 300}, {"text": "更多背景。" * 300}],
            "conclusion": [{"text": text}],
        },
        ensure_ascii=False,
    )
    offset = text.index("[C:c7]")
    feedback = stage_repair.make_feedback(
        record([{"wire_path": ["conclusion", 0, "text"], "offset": offset}]), raw
    )
    assert "需要核对的结果" not in feedback["failed_candidate_excerpt"]
    assert feedback["failed_regions_total"] == 1
    region = feedback["failed_regions"][0]
    assert region["diagnostic_index"] == 0
    assert region["start"] <= offset < region["end"]
    assert region["text"] == text[region["start"] : region["end"]]
    assert "需要核对的结果。[C:c7]" in region["text"]
    stage_repair.feedback_payload({"candidate_feedback": feedback})


@pytest.mark.parametrize("offset", [-1, True, 999, "1"])
def test_invalid_character_offsets_do_not_manufacture_candidate_text(offset):
    feedback = stage_repair.make_feedback(
        record([{"wire_path": ["conclusion", 0, "text"], "offset": offset}]),
        '{"conclusion":[{"text":"原文。"}]}',
    )
    assert feedback["failed_regions"] == []
    assert feedback["diagnostics_included"] == 1


@pytest.mark.parametrize(
    "raw",
    ['{"conclusion":[{"text":"残缺', '{"conclusion":[],"conclusion":[]}'],
)
def test_invalid_raw_json_never_supplies_guessed_failure_regions(raw):
    feedback = stage_repair.make_feedback(
        record([{"wire_path": ["conclusion", 0, "text"], "offset": 1}]), raw
    )
    assert feedback["failed_regions"] == []
    assert feedback["failed_candidate_excerpt"] == raw


def test_scope_prompt_explains_sentence_boundaries_and_verbatim_feedback():
    result = inline_citations.scope_instruction(inline_citations.INSTRUCTION)
    assert "before any sentence-ending punctuation" in result
    assert "Each scientific sentence" in result
    assert "failed_regions" in result
    assert "comma, enumeration separator or connective" in result
    assert " ".join(sorted(inline_citations.BOUNDARY)) in result
    # This change is opt-in; the old contract remains an independent string.
    assert "failed_regions" not in inline_citations.INSTRUCTION


def test_summary_local_citations_follow_the_actual_body_input():
    payload = {
        "body_units": {
            "u1": "The paper reports 87.7% top-1 accuracy[C:c3]. "
            "The setting differs from other studies."
        },
        "candidate_feedback": None,
    }
    schema = schema_for("summary", payload["body_units"])
    schema["x-citation-contract"] = inline_citations.CONTRACT
    base = LIVE_SUMMARY_PROMPT + GROUNDING_INSTRUCTION
    legacy = request_messages(base, payload, schema, wire_contract="flat-writer-v1")
    schema["x-citation-scope-contract"] = stage_repair.SCOPE_CONTRACT
    actual = request_messages(base, payload, schema, wire_contract="flat-writer-v1")

    assert legacy[1] == actual[1]
    assert json.loads(actual[1]["content"]) == payload
    assert GROUNDING_INSTRUCTION in legacy[0]["content"]
    assert GROUNDING_INSTRUCTION not in actual[0]["content"]
    assert "same cited passage in the selected body units" in actual[0]["content"]
    assert "cannot support a different passage" in actual[0]["content"]
    assert "List the body units used as premises" in actual[0]["content"]
    assert "Retain the core requirements" not in actual[0]["content"]


def test_vit14_feedback_replays_immutable_responses_and_reaches_each_late_error():
    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-14/repairs.json.writer-v2.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT14")
    before = path.read_bytes()
    journal = json.loads(before)
    for name in ("summary_repair_1", "summary_repair_2", "summary_repair_3"):
        call = journal["calls"][name]
        findings = stage_repair.summary_diagnostics(
            call["raw"],
            call["candidate_request"],
            {f"c{i}" for i in range(1, 30)},
            scope_contract=stage_repair.SCOPE_CONTRACT,
        )
        assert findings == journal["candidate_rejections"][name]["diagnostics"]
        feedback = stage_repair.make_feedback(record(findings), call["raw"])
        decoded = json.loads(call["raw"])
        assert feedback["diagnostics_complete"]
        assert len(feedback["failed_regions"]) == len(findings)
        for region in feedback["failed_regions"]:
            finding = findings[region["diagnostic_index"]]
            section, ordinal, field = finding["wire_path"]
            text = decoded[section][ordinal][field]
            assert region["text"] == text[region["start"] : region["end"]]
            assert region["start"] <= finding["offset"] < region["end"]
        stage_repair.feedback_payload({"candidate_feedback": feedback})
    assert path.read_bytes() == before


def test_vit14_recorded_summary_has_body_premises_without_source_text():
    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-14/repairs.json.writer-v2.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT14")
    raw = path.read_bytes()
    request = json.loads(raw)["calls"]["summary_repair_4"]["candidate_request"]
    original = request["messages"]
    payload = json.loads(original[1]["content"])
    assert len(payload["body_units"]) == 48
    assert "sources" not in payload and "cards" not in payload
    assert all("source_quote" not in unit for unit in payload["body_units"].values())
    assert GROUNDING_INSTRUCTION in original[0]["content"]
    actual = request_messages(
        LIVE_SUMMARY_PROMPT + GROUNDING_INSTRUCTION,
        payload,
        request["schema"],
        wire_contract="flat-writer-v1",
    )
    assert json.loads(actual[1]["content"]) == payload
    assert GROUNDING_INSTRUCTION not in actual[0]["content"]
    assert "same cited passage in the selected body units" in actual[0]["content"]
    assert path.read_bytes() == raw


@pytest.mark.asyncio
@pytest.mark.parametrize("category", ["review", "evaluation"])
async def test_streamed_summary_retry_receives_local_failure_and_reuses_body(
    case, tmp_path, monkeypatch, category
):
    class Gateway(StreamingGateway):
        rejected_text = None
        received_feedback = None

        def answer(self, payload, schema):
            answer = super().answer(payload, schema)
            ready = (
                self.editorial_edits > 0
                if category == "review"
                else self.repair_count > 0
            )
            if "abstract" not in schema.get("properties", {}) or not ready:
                return answer
            if self.rejected_text is None:
                item = answer["conclusion"][0]
                refs = inline_citations.MARK.findall(item["text"])
                item["text"] += f"[C:{refs[0]}]"
                self.rejected_text = item["text"]
            elif (
                payload.get("candidate_feedback", {}).get("category")
                == f"{category}_revision"
            ):
                self.received_feedback = copy.deepcopy(payload["candidate_feedback"])
            return answer

    gateway = Gateway(case, reviews=1 if category == "review" else 0)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=stream_config())
    summaries = [
        request["input"]
        for request in gateway.stream_requests
        if "body_units" in json.loads(request["input"][1]["content"])
        and "targets" not in json.loads(request["input"][1]["content"])
    ]
    assert len(summaries) >= 3
    for messages in summaries:
        assert "same cited passage in the selected body units" in messages[0]["content"]
        assert GROUNDING_INSTRUCTION not in messages[0]["content"]
        assert "sources" not in json.loads(messages[1]["content"])
    feedback = gateway.received_feedback
    assert feedback and feedback["failed_regions"]
    region = feedback["failed_regions"][0]
    assert region["text"] == gateway.rejected_text[region["start"] : region["end"]]
    assert gateway.global_calls >= 1
    assert (
        gateway.editorial_edits if category == "review" else gateway.repair_count
    ) == 1
    assert (
        report["evaluation"]["faithfulness"]["details"]["stages"]["global_check"]
        == "completed"
    )
    for path in tmp_path.rglob("*.writer-v2.json"):
        journal = json.loads(path.read_bytes())
        failures = list(journal.get("candidate_rejections", {}).values())
        if failures:
            assert len(failures) == 1
            pending = failures[0]["body_sha256"]
            assert (
                any(
                    candidate["repair_provenance"]["candidate_body_sha256"] == pending
                    for candidate in journal["candidates"]
                )
                or category == "review"
            )


def test_vit14_stream_identity_allows_archived_accounting_only():
    from litagent.run_limits import validate_run_limits
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    path = Path(
        "artifacts/acceptance/inline-citations-vit-235d5bf1dba2/"
        "vit-235d5bf1dba2-inline-reassessment-14/budget-ledger.json"
    )
    if not path.exists():
        pytest.skip("requires immutable local ViT14")
    original = path.read_bytes()
    ledger = json.loads(original)
    limits = ledger["plan"]["envelope"]["run_limits"]
    assert validate_run_limits(limits, archived=True) == limits
    validate_accounting_snapshot(ledger, expected_plan=ledger["plan"])
    assert ledger["used_tokens"] == 661697
    with pytest.raises(ValueError, match="stage_repair_identity_invalid"):
        validate_run_limits(limits)
    for key in ("responses_transport_identity", "candidate_recovery_identity"):
        with pytest.raises(ValueError):
            validate_run_limits({**limits, key: "0" * 64}, archived=True)
    without_stream = {
        k: v for k, v in limits.items() if not k.startswith("responses_transport")
    }
    with pytest.raises(ValueError):
        validate_run_limits(without_stream, archived=True)
    assert (
        hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(original).digest()
    )
