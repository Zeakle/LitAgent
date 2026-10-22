"""Preserve frozen scientific premises across flat summary reference transport."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError
from test_body_analysis import analysis_document
from test_native_array_integration import request
from test_staged_native_generation import split
from test_writing_passages import facts

from litagent.native_wire import unique
from litagent.report_audit import digest
from litagent.staged_generation import (
    CONTRACT,
    _short_reference_fields,
    decode_summaries,
    encode_summaries,
    prepare_summary_request,
    summary_reference_binding,
)

REAL = json.loads(Path("tests/fixtures/summary-eof-140.json").read_text())


@pytest.fixture
def parts():
    """Prepare actual canonical producer output and a complete source packet."""
    cards = facts()
    body, summaries = split(analysis_document(cards))
    messages, rf = request(cards, array_wire=True)
    binding = summary_reference_binding(body)
    return body, summaries, messages, rf, binding


def test_real_summary_eof_is_not_patched_or_interpreted_as_scientific_success():
    raw = REAL["raw_response"]
    assert hashlib.sha256(raw.encode()).hexdigest() == REAL["raw_response_sha256"]
    with pytest.raises(json.JSONDecodeError) as error:
        json.loads(raw, object_pairs_hook=unique)
    assert error.value.pos == len(raw) == REAL["json_error"]["offset"]
    assert REAL["finish_reason"] == "stop"
    assert REAL["usage"]["completion_tokens"] == 973 < 20000
    assert REAL["scientific_verdict"] == "not_evaluated"


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_round_trip_full_sources_prose_policies_and_native_strict_schema(language):
    cards = facts()
    body, summaries = split(analysis_document(cards, language))
    # Multi-block paragraph/list wire compatibility remains independent of final
    # readability rules, which the complete report compiler still evaluates.
    block = copy.deepcopy(summaries["sections"]["abstract"]["blocks"][0])
    block["heading"] = "Another" if language == "en" else "另一项"
    block["type"] = "list"
    summaries["sections"]["abstract"]["blocks"].append(block)
    messages, rf = request(cards, language, array_wire=True)
    messages.append(
        {
            "role": "system",
            "content": "Keep explicitly authorized references unchanged.",
        }
    )
    binding = summary_reference_binding(body)
    count = binding["body_unit_count"]
    old, old_rf = prepare_summary_request(body, count, messages, rf)
    new, new_rf = prepare_summary_request(body, count, messages, rf, short_refs=True)
    old_packet, new_packet = [json.loads(x[1]["content"]) for x in (old, new)]
    restored_packet = copy.deepcopy(new_packet)
    restored_packet["frozen_body"] = _short_reference_fields(
        new_packet["frozen_body"], binding["reading_cards"]
    )
    restored_packet["cards"] = {
        cid: new_packet["cards"][ref] for ref, cid in binding["reading_cards"].items()
    }
    for key in old_packet:
        assert old_packet[key] == restored_packet[key]
    assert new[2:] == messages[2:] == old[2:]
    assert new_packet["body_unit_index"] == binding["units"]
    assert restored_packet["frozen_body"] == body
    assert new_rf["json_schema"]["name"] == "grounded_summaries_v2_refs"
    assert old_rf["json_schema"]["name"] == "grounded_summaries_v1"
    wire = encode_summaries(summaries, binding)
    assert decode_summaries(wire, binding, body=body) == summaries
    for key, rows in wire.items():
        assert all(
            set(row) == {"block_index", "heading", "type", "text", "body_refs"}
            for row in rows
        )
        assert rows[0]["block_index"] == 0

    def check(node):
        if isinstance(node, dict):
            assert "$ref" not in node
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["properties"]) == set(node["required"])
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    check(new_rf["json_schema"]["schema"])
    Draft202012Validator(new_rf["json_schema"]["schema"]).validate(wire)


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "duplicate",
        "empty",
        "coordinate",
        "concat",
        "extra",
        "sections",
        "empty_section",
        "bool_index",
        "sparse",
        "reorder",
        "heading",
        "block_type",
        "empty_text",
        "stale_body",
        "binding",
    ],
)
def test_forged_dependencies_and_mutated_frozen_body_are_rejected(parts, fault):
    body, summaries, _, _, binding = parts
    wire = encode_summaries(summaries, binding)
    row = wire["abstract"][0]
    if fault == "unknown":
        row["body_refs"] = ["b999"]
    elif fault == "duplicate":
        row["body_refs"] *= 2
    elif fault == "empty":
        row["body_refs"] = []
    elif fault == "coordinate":
        row["body_refs"] = [binding["units"]["b1"]]
    elif fault == "concat":
        row["body_refs"] = ["b1,b2"]
    elif fault == "extra":
        row["fact_card_ids"] = ["invented"]
    elif fault == "sections":
        wire = {"sections": wire}
    elif fault == "empty_section":
        wire["abstract"] = []
    elif fault == "bool_index":
        row["block_index"] = True
    elif fault == "sparse":
        row["block_index"] = 1
    elif fault == "reorder":
        second = copy.deepcopy(row)
        second["block_index"] = 1
        wire["abstract"] = [row, second, copy.deepcopy(row)]
    elif fault == "heading":
        second = copy.deepcopy(row)
        second["heading"] = "Changed"
        wire["abstract"].append(second)
    elif fault == "block_type":
        row["type"] = "unrecognized"
    elif fault == "empty_text":
        row["text"] = ""
    elif fault == "stale_body":
        body["title"] += " changed"
    else:
        binding = copy.deepcopy(binding)
        binding["units"]["b1"]["sentence_index"] = 9
        binding["sha256"] = digest({k: v for k, v in binding.items() if k != "sha256"})
    with pytest.raises((ValueError, ValidationError)):
        decode_summaries(wire, binding, body=body)


def test_organization_units_are_not_selectable_scientific_premises(parts):
    body, _, _, _, _ = parts
    unit = copy.deepcopy(body["sections"]["introduction"]["blocks"][0]["sentences"][0])
    unit.update(kind="organization", fact_card_ids=[], scope_notice_id="none")
    body["sections"]["introduction"]["blocks"][0]["sentences"].insert(0, unit)
    binding = summary_reference_binding(body)
    assert binding["units"]["b1"]["sentence_index"] == 1
    assert not any(
        ref == {"section_key": "introduction", "block_index": 0, "sentence_index": 0}
        for ref in binding["units"].values()
    )


def test_total_capacity_and_premise_limit_remain_enforced(parts):
    body, summaries, messages, rf, _ = parts
    base = body["sections"]["methods"]["blocks"][0]["sentences"]
    while (
        sum(len(b["sentences"]) for s in body["sections"].values() for b in s["blocks"])
        < 126
    ):
        base.append(copy.deepcopy(base[0]))
    binding = summary_reference_binding(body)
    wire = encode_summaries(summaries, binding)
    wire["abstract"].append(copy.deepcopy(wire["abstract"][0]))
    with pytest.raises((ValueError, ValidationError)):
        decode_summaries(wire, binding, body=body)
    with pytest.raises(ValueError, match="count_invalid"):
        prepare_summary_request(body, 125, messages, rf, short_refs=True)
    base.append(copy.deepcopy(base[0]))
    with pytest.raises(ValueError, match="capacity_unavailable"):
        summary_reference_binding(body)


def test_new_contract_retains_single_identity_and_paid_guards_stay_closed(
    monkeypatch, tmp_path
):
    from litagent import staged_generation
    from litagent.workbench import RunOptions
    from scripts import probe_received_body as received
    from scripts import probe_staged_generation as probe

    assert CONTRACT == "native-body-summary-v9"
    single = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    staged = single.model_copy(update={"generation_strategy": "body_summary_v1"})
    before_single = single.fingerprint()
    before_staged = staged.fingerprint()
    monkeypatch.setattr(staged_generation, "CONTRACT", "native-body-summary-v7")
    assert single.fingerprint() == before_single
    assert staged.fingerprint() != before_staged
    assert probe.TYPED_CONTRACT == "native-body-summary-v4"
    assert received.EXPECTED_CONTRACT == "native-body-summary-v5"
    with pytest.raises(ValueError, match="migration_required"):
        received.prepare(tmp_path / "new", tmp_path / "parent", tmp_path / "tip")


def test_more_than_six_distinct_scientific_premises_still_rejected(parts):
    body, summaries, _, _, _ = parts
    base = body["sections"]["methods"]["blocks"][0]["sentences"]
    while len(summary_reference_binding(body)["units"]) < 7:
        base.append(copy.deepcopy(base[0]))
    binding = summary_reference_binding(body)
    wire = encode_summaries(summaries, binding)
    wire["abstract"][0]["body_refs"] = list(binding["units"])[:7]
    with pytest.raises(ValidationError):
        decode_summaries(wire, binding, body=body)
