"""Reject stale rule catalogs while preserving native JSON and unknown host rules."""

import copy

import pytest
from test_writer_source_view_contract import chapter_payload, source_envelope
from test_writer_v2 import case  # noqa: F401

import litagent.body_quality as quality
import litagent.writer_source_view as view
from litagent.writing_flat import REFERENCE_SET_INSTRUCTION, prepare
from litagent.writing_v2 import (
    RANGE_WRITING_PLAN,
    prepare_planned_input,
    request_messages,
)


def complete_messages(case, monkeypatch):
    """Capture the real full sender before its final selected-path compilation."""
    scope = source_envelope()
    system, payload, schema = chapter_payload(case, scope)
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
    )
    with monkeypatch.context() as patch:
        patch.setattr(view, "project_chapter_messages", lambda messages, *_: messages)
        messages = request_messages(
            system, payload, schema, wire_contract="flat-writer-v1"
        )
    return messages, payload, prepare(schema)["wire_schema"]


def test_final_compiler_retains_payload_and_unknown_rules_verbatim(case, monkeypatch):
    messages, payload, native = complete_messages(case, monkeypatch)
    original = copy.deepcopy((messages, payload, native))
    extra = "\nFuture host instruction must survive exactly.\n"
    messages[0]["content"] += extra
    projected = view.project_chapter_messages(messages, payload, native)
    assert projected[0]["content"].count(view.FINAL_CHAPTER_INSTRUCTION) == 1
    assert extra in projected[0]["content"]
    assert projected[1:] == original[0][1:]
    assert payload == original[1] and native == original[2]
    assert view.project_chapter_messages(projected, payload, native) is projected


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_unrecognized_rule_composition_cannot_silently_drop_obligations(
    case, monkeypatch, fault
):
    messages, payload, native = complete_messages(case, monkeypatch)
    if fault == "missing":
        messages[0]["content"] = messages[0]["content"].replace(
            REFERENCE_SET_INSTRUCTION, ""
        )
    else:
        messages[0]["content"] += REFERENCE_SET_INSTRUCTION
    with pytest.raises(ValueError, match="writer_source_view_invalid"):
        view.project_chapter_messages(messages, payload, native)


def test_new_shared_scientific_obligation_requires_catalog_review_before_sending(
    case, monkeypatch
):
    monkeypatch.setattr(
        quality,
        "WRITER_CONTENT_INSTRUCTION",
        quality.WRITER_CONTENT_INSTRUCTION + "\nNew host scientific obligation.\n",
    )
    scope = source_envelope()
    system, payload, _ = chapter_payload(case, scope)
    with pytest.raises(ValueError, match="writer_chapter_rule_catalog_changed"):
        prepare_planned_input(
            system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
        )


def test_new_upstream_obligation_cannot_be_removed_by_the_old_compiler(
    case, monkeypatch
):
    import litagent.writing_v2 as writer

    monkeypatch.setattr(
        writer, "INSTRUCTION", writer.INSTRUCTION + "New upstream obligation.\n"
    )
    scope = source_envelope()
    system, payload, _ = chapter_payload(case, scope)
    with pytest.raises(ValueError, match="writer_chapter_rule_catalog_changed"):
        prepare_planned_input(
            system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
        )


def test_unchanged_native_schema_cannot_gain_an_unrecognized_unit_field(
    case, monkeypatch
):
    messages, payload, native = complete_messages(case, monkeypatch)
    native["properties"]["units"]["items"]["properties"]["approved"] = {
        "type": "boolean"
    }
    with pytest.raises(ValueError, match="writer_source_view_invalid"):
        view.project_chapter_messages(messages, payload, native)


def test_unselected_stages_keep_the_original_message_objects():
    messages = [
        {"role": "system", "content": "Original summary or reviewer rules."},
        {"role": "user", "content": "Original payload."},
    ]
    assert view.project_chapter_messages(messages, {}, {}) is messages


def test_final_native_request_is_idempotent_for_the_actual_source_record(case):
    scope = source_envelope()
    system, payload, schema = chapter_payload(case, scope)
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    final = messages[0]["content"]
    projected, repeated = view.project_source_view(final, payload, case.packet, scope)
    assert projected == final and repeated == payload
    record = view.make_source_view_record(scope, final, payload, case.packet)
    view.validate_source_view_record(scope, record, case.packet)
    assert record["system_sha"] == view._digest(final)


@pytest.mark.parametrize("fault", ["duplicate", "mixed_old_rules", "wrong_prefix"])
def test_invalid_final_instruction_record_is_never_blessed(case, fault):
    scope = source_envelope()
    system, payload, schema = chapter_payload(case, scope)
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    final = messages[0]["content"]
    if fault == "duplicate":
        final += view.FINAL_CHAPTER_INSTRUCTION
    elif fault == "mixed_old_rules":
        final += view.GROUNDING_INSTRUCTION
    else:
        final = "Prefix changed.\n" + final
    with pytest.raises(ValueError, match="writer_source_view_invalid"):
        view.make_source_view_record(scope, final, payload, case.packet)
