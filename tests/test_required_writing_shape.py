"""Require real native units while preserving original candidate identities."""

import copy
import hashlib
import json
import re

import pytest
from jsonschema import Draft202012Validator
from test_chapter_scope_contract import unit
from test_native_writing_schema import packet
from test_strict_quality import to_native, valid_details

from litagent.quality import (
    GroundedWritingError,
    prepare_grounded_request,
    render_grounded_report,
    strict_acceptance,
)


@pytest.mark.parametrize(
    "mode",
    [
        "empty_array",
        "empty_map",
        "missing_chapter",
        "missing_block",
        "missing_sentence",
        "empty_text",
        "extra_field",
    ],
)
def test_native_empty_shapes_fail_without_fabricating_a_report(mode):
    cards, selection, report, messages = packet()
    native = to_native(report)
    if mode == "empty_array":
        native["sections"] = []
    elif mode == "empty_map":
        native["sections"] = {}
    elif mode == "missing_chapter":
        del native["sections"]["methods"]
    elif mode == "missing_block":
        del native["sections"]["methods"]["first_block"]
    elif mode == "missing_sentence":
        del native["sections"]["methods"]["first_block"]["first_sentence"]
    elif mode == "empty_text":
        native["sections"]["methods"]["first_block"]["first_sentence"]["text"] = ""
    else:
        native["sections"]["methods"]["invented"] = True
    _, fmt = prepare_grounded_request(messages)
    if mode != "empty_text":
        assert list(
            Draft202012Validator(fmt["json_schema"]["schema"]).iter_errors(native)
        )
    raw = json.dumps(native)
    with pytest.raises(ValueError):
        render_grounded_report(
            raw,
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )
    assert raw == json.dumps(native)


def test_native_conversion_preserves_all_units_citations_and_template_order():
    cards, selection, report, _ = packet()
    intro = report["sections"][1]
    intro["blocks"][0]["sentences"].append(
        copy.deepcopy(intro["blocks"][0]["sentences"][0])
    )
    intro["blocks"].append(copy.deepcopy(intro["blocks"][0]))
    native = to_native(report)
    native["sections"] = dict(reversed(list(native["sections"].items())))
    before = copy.deepcopy(native)
    legacy = render_grounded_report(
        json.dumps(report), cards, "academic_v2", "en", evidence_selection=selection
    )
    body = render_grounded_report(
        json.dumps(native),
        cards,
        "academic_v2",
        "en",
        evidence_selection=selection,
        require_native=True,
    )
    assert body == re.sub(r"(\.)((?: \[E:[^\]]+\])+)", r"\2\1", legacy)
    assert body.count("[E:") == legacy.count("[E:") == 10
    assert native == before
    with pytest.raises(ValueError, match="strict_writing_native_contract_invalid"):
        render_grounded_report(
            json.dumps(report),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )


def test_native_unknown_bindings_keep_raw_sha_and_exact_transport_paths():
    cards, selection, report, _ = packet()
    wrong = list(cards)[1]
    unit(report, "introduction")["fact_card_ids"] = [wrong]
    native = to_native(report)
    native["sections"]["introduction"]["first_block"]["additional_sentences"] = [
        copy.deepcopy(unit(report, "introduction"))
    ]
    raw = json.dumps(native)
    with pytest.raises(GroundedWritingError) as caught:
        render_grounded_report(
            raw,
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )
    assert caught.value.candidate_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert [d["raw_json_path"] for d in caught.value.diagnostics] == [
        "sections.introduction.first_block.first_sentence",
        "sections.introduction.first_block.additional_sentences.0",
    ]
    assert [d["location"] for d in caught.value.diagnostics] == [
        "introduction.blocks.0.sentences.0",
        "introduction.blocks.0.sentences.1",
    ]


def test_duplicate_native_object_keys_cannot_silently_discard_units():
    cards, selection, report, _ = packet()
    raw = json.dumps(to_native(report)).replace(
        '"sections": {', '"sections": {}, "sections": {', 1
    )
    with pytest.raises(ValueError, match="strict_writing_duplicate_object_key"):
        render_grounded_report(
            raw,
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )


@pytest.mark.parametrize("field", ["additional_sentences", "additional_blocks"])
def test_native_continuations_do_not_expand_existing_unit_caps(field):
    cards, selection, report, _ = packet()
    native = to_native(report)
    section = native["sections"]["methods"]
    if field == "additional_blocks":
        section[field] = [copy.deepcopy(section["first_block"])] * 30
    else:
        section["first_block"][field] = [
            copy.deepcopy(section["first_block"]["first_sentence"])
        ] * 30
    with pytest.raises(ValueError):
        render_grounded_report(
            json.dumps(native),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            require_native=True,
        )


def test_required_organizational_units_do_not_imply_scientific_acceptance():
    details = valid_details()
    for claim in details["assertions"]:
        claim["kind"] = "report_structure"
    result = strict_acceptance(details)
    assert result["status"] == "failed"
    assert "science_not_fully_supported" in result["failure_codes"]
