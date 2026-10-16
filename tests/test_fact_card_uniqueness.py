"""Reject duplicate known cards without misreporting unknown source identities."""

import copy
import hashlib
import json

import pytest
from test_chapter_scope_contract import materials, unit
from test_native_writing_schema import section_definition
from test_strict_quality import to_native

from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.quality import (
    GroundedWritingError,
    grounded_messages,
    prepare_grounded_request,
    render_grounded_report,
    render_grounded_report_traced,
)


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize(
    "mode,code",
    [
        ("duplicate", "strict_writing_duplicate_fact_card"),
        ("unknown", "strict_writing_unknown_fact_card"),
        ("mixed", "strict_writing_unknown_fact_card"),
        ("outside", "strict_writing_out_of_scope_fact_card"),
    ],
)
def test_reference_errors_keep_raw_ids_and_fail_the_entire_candidate(
    language, native, mode, code
):
    cards, selection, report = materials(language=language)
    aid, bid = list(cards)
    key = "introduction" if mode == "outside" else "experiments"
    actual = {
        "duplicate": [aid, bid, bid],
        "unknown": [aid, "f:unknown"],
        "mixed": [aid, aid, "f:unknown", "f:unknown"],
        "outside": [bid],
    }[mode]
    unit(report, key)["fact_card_ids"] = actual
    payload = to_native(report) if native else report
    raw = json.dumps(payload, ensure_ascii=False)
    before = copy.deepcopy((cards, selection.to_dict(), payload))
    events = []
    with pytest.raises(GroundedWritingError, match=code) as caught:
        render_grounded_report_traced(
            raw,
            cards,
            "academic_v2",
            language,
            evidence_selection=selection,
            require_native=native,
            trace_hook=lambda name, data: events.append((name, data)),
            parent_task_id="synthesis",
            stage="initial",
        )
    error = caught.value
    assert error.candidate_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    issue = error.diagnostics[0]
    assert issue["reason_code"] == code
    assert issue["actual_fact_card_ids"] == actual
    assert issue["location"] == f"{key}.blocks.0.sentences.0"
    if native:
        assert issue["raw_json_path"] == f"sections.{key}.first_block.first_sentence"
    if mode == "duplicate":
        assert issue["invalid_fact_card_ids"] == [bid]
        assert issue["existing_evidence_ids"] == ["a", "b"]
    elif mode in {"unknown", "mixed"}:
        assert issue["invalid_fact_card_ids"] == ["f:unknown"]
    assert events[0][0] == "writing.validation.failed"
    assert events[0][1]["candidate_sha256"] == error.candidate_sha256
    assert (cards, selection.to_dict(), payload) == before
    assert raw == json.dumps(payload, ensure_ascii=False)


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_same_paper_distinct_cards_and_cross_sentence_reuse_remain_legal(language):
    cards, selection, report = materials(language=language)
    aid, bid = list(cards)
    assert cards[aid]["paper_id"] == cards[bid]["paper_id"]
    scientific = unit(report, "experiments")
    scientific["fact_card_ids"] = [aid, bid]
    section = next(s for s in report["sections"] if s["section_key"] == "experiments")
    section["blocks"][0]["sentences"].append(copy.deepcopy(scientific))
    native = to_native(report)
    raw = json.dumps(native, ensure_ascii=False)
    body = render_grounded_report(
        raw,
        cards,
        "academic_v2",
        language,
        evidence_selection=selection,
        require_native=True,
    )
    assert body.count("[E:a]") == 7
    assert body.count("[E:b]") == 3
    assert json.dumps(native, ensure_ascii=False) == raw


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
@pytest.mark.parametrize("revision", [False, True])
def test_shared_native_request_explains_uniqueness_without_changing_scopes(
    language, structure, revision
):
    cards, selection, _ = materials(structure, language)
    messages = grounded_messages(
        cards,
        structure,
        language,
        "Compare approaches",
        evidence_selection=selection,
        draft="Current draft" if revision else "",
        feedback="Check conditions" if revision else "",
    )
    prepared, fmt = prepare_grounded_request(messages)
    assert "list each exact card ID once" in prepared[0]["content"]
    assert "host rejects repeated IDs without deduplication" in prepared[0]["content"]
    schema = fmt["json_schema"]["schema"]
    assert fmt["json_schema"]["name"] == "grounded_report_v13"
    for key in schema["properties"]["sections"]["required"]:
        refs = section_definition(schema, key, "sentence")["properties"][
            "fact_card_ids"
        ]
        assert "list each exact ID once" in refs["description"]
        assert "uniqueItems" not in refs
        allowed = [
            cid
            for cid, card in cards.items()
            if card["evidence_id"] in selection.section_evidence_ids[key]
        ]
        assert refs["items"]["enum"] == allowed


def test_new_writing_contract_changes_only_strict_config_identity(monkeypatch):
    cfg = load_config()
    current = {
        policy: build_config_summary(cfg, policy)
        for policy in ("legacy", "strict_v1", "strict_auto_v1")
    }
    monkeypatch.setattr("litagent.contracts.WRITING_VERSION", "grounded-writing-v11")
    previous = {policy: build_config_summary(cfg, policy) for policy in current}
    assert previous["legacy"] == current["legacy"]
    assert previous["strict_v1"] != current["strict_v1"]
    assert previous["strict_auto_v1"] != current["strict_auto_v1"]
