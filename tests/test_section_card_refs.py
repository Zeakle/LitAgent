"""Keep chapter aliases reversible without rewriting facts or relaxing evidence scope."""

import copy
import json
from pathlib import Path

import pytest
from test_body_analysis import analysis_document
from test_native_array_integration import request
from test_staged_native_generation import split
from test_writing_passages import facts

from litagent.native_wire import BODY
from litagent.quality import _writing_scopes
from litagent.staged_generation import (
    BodyRowError,
    body_row_projection,
    card_reference_binding,
    decode_body,
    encode_body,
    expand_body,
    prepare_body_request,
    validate_body,
)


def binding_for(cards, language="en"):
    """Use the same host permission producer as the actual writer consumers."""
    return card_reference_binding(
        cards, _writing_scopes(cards, "academic_v2", language, None)
    )


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_aliases_restore_exact_canonical_body_and_original_position_hashes(language):
    cards = facts()
    binding = binding_for(cards, language)
    body, _ = split(analysis_document(cards, language))
    wire = encode_body(body, binding)
    assert expand_body(decode_body(wire, binding)) == body
    for section in BODY:
        for row in wire[section]:
            assert "fact_card_ids" not in row
            assert set(row["fact_card_refs"]) <= set(binding["section_cards"][section])
    assert (
        card_reference_binding(
            dict(reversed(list(cards.items()))),
            _writing_scopes(cards, "academic_v2", language, None),
        )
        == binding
    )
    messages, rf = request(cards, language, array_wire=True)
    _, schema = prepare_body_request(messages, rf, typed_units=False)
    recovered, count = validate_body(
        json.dumps(wire), schema, cards, "academic_v2", language, None
    )
    assert recovered == body and count >= 5
    proof = body_row_projection(body, binding)
    assert proof["card_reference_binding_sha256"] == binding["sha256"]
    assert proof["scientific_verdict"] == "not_evaluated"
    assert len(proof["positions"]) == count


@pytest.mark.parametrize(
    "fault",
    [
        "unknown",
        "foreign",
        "reading",
        "long_id",
        "leading_zero",
        "duplicate",
        "mixed",
        "forged_mapping",
    ],
)
def test_unknown_foreign_or_ambiguous_references_are_never_guessed(fault):
    cards = facts()
    binding = binding_for(cards)
    body, _ = split(analysis_document(cards))
    wire = encode_body(body, binding)
    row = wire["methods"][0]
    valid = row["fact_card_refs"][0]
    if fault == "unknown":
        row["fact_card_refs"] = ["m999"]
    elif fault == "foreign":
        row["fact_card_refs"] = [next(iter(binding["section_cards"]["introduction"]))]
    elif fault == "reading":
        row["fact_card_refs"] = [next(iter(binding["reading_cards"]))]
    elif fault == "long_id":
        row["fact_card_refs"] = [next(iter(cards))]
    elif fault == "leading_zero":
        row["fact_card_refs"] = [valid[0] + "0" + valid[1:]]
    elif fault == "duplicate":
        row["fact_card_refs"] = [valid, valid]
    elif fault == "mixed":
        row["fact_card_ids"] = [next(iter(cards))]
    else:
        wire["card_reference_binding"] = binding
    before = copy.deepcopy(wire)
    with pytest.raises(ValueError):
        decode_body(wire, binding)
    assert wire == before


def test_model_schema_cannot_expand_host_section_permissions():
    cards = facts()
    body, _ = split(analysis_document(cards))
    binding = binding_for(cards)
    messages, rf = request(cards, array_wire=True)
    _, rf = prepare_body_request(messages, rf, typed_units=False)
    rf["json_schema"]["schema"]["properties"]["methods"]["items"]["properties"][
        "fact_card_refs"
    ]["items"]["enum"].append("m999")
    with pytest.raises(ValueError, match="section_card_schema_binding_invalid"):
        validate_body(
            json.dumps(encode_body(body, binding)), rf, cards, "academic_v2", "en", None
        )


def test_source_prose_and_policy_survive_defined_reference_slot_projection():
    cards = facts()
    messages, rf = request(cards, array_wire=True)
    original = json.loads(messages[1]["content"])
    cid = next(iter(cards))
    ref = original["cards"][cid]["source_ref"]
    # A literal card ID in scientific content is data, not an instruction to recode.
    original["research_question"] = "Literal " + cid
    original["sources"][ref]["source_quote"] = cid
    original["cards"][cid]["unverified_fields"]["method"] = cid
    original["current_draft"] = f"Exact 50% condition [F:{cid}]"
    original["located_feedback"] = json.dumps(
        {"source_quote": original["current_draft"], "fact_card_ids": [cid], "start": 7}
    )
    messages[1]["content"] = json.dumps(original)
    messages.append(
        {
            "role": "user",
            "content": json.dumps(
                {
                    "must_preserve_fact_card_ids": [cid],
                    "allowed_removed_fact_card_ids": [],
                    "instruction": "Keep the original 50% condition.",
                }
            ),
        }
    )
    before = copy.deepcopy(messages)
    prepared, _ = prepare_body_request(messages, rf, typed_units=False)
    packet = json.loads(prepared[1]["content"])
    binding = binding_for(cards)
    short = next(k for k, v in binding["reading_cards"].items() if v == cid)
    assert packet["sources"][ref]["source_quote"] == cid
    assert packet["cards"][short]["unverified_fields"]["method"] == cid
    assert packet["research_question"] == original["research_question"]
    assert packet["current_draft"] == f"Exact 50% condition [F:{short}]"
    feedback = json.loads(packet["located_feedback"])
    assert feedback == {
        "source_quote": packet["current_draft"],
        "fact_card_ids": [short],
        "start": 7,
    }
    policy = json.loads(prepared[2]["content"])
    assert policy["must_preserve_fact_card_ids"] == [short]
    assert policy["instruction"] == "Keep the original 50% condition."
    assert messages == before


def test_analysis_and_coverage_references_roundtrip_without_loss_of_sources():
    cards = facts()
    messages, rf = request(cards, array_wire=True)
    old = json.loads(messages[1]["content"])
    prepared, _ = prepare_body_request(messages, rf, typed_units=False)
    new = json.loads(prepared[1]["content"])
    b = binding_for(cards)
    reading = b["reading_cards"]
    for a, z in zip(
        old["analysis_plan"]["sources"], new["analysis_plan"]["sources"], strict=True
    ):
        assert {**z, "fact_card_id": reading[z["fact_card_id"]]} == a
    assert (
        old["analysis_plan"]["section_duties"] == new["analysis_plan"]["section_duties"]
    )
    for field in ("requirements",):
        for a, z in zip(
            old["analysis_plan"][field], new["analysis_plan"][field], strict=True
        ):
            assert {
                **z,
                "candidate_card_ids": [reading[x] for x in z["candidate_card_ids"]],
            } == a
    for key, ids in old["analysis_plan"]["missing_fields"].items():
        assert [reading[x] for x in new["analysis_plan"]["missing_fields"][key]] == ids
    for a, z in zip(
        old["coverage_matrix"]["requirements"],
        new["coverage_matrix"]["requirements"],
        strict=True,
    ):
        assert {**z, "source_card_ids": [reading[x] for x in z["source_card_ids"]]} == a
    for section in BODY:
        for a, z in zip(
            old["section_source_index"][section],
            new["section_source_index"][section],
            strict=True,
        ):
            assert (
                b["section_cards"][section][z["fact_card_ref"]]
                == a["fact_card_id"]
                == reading[z["card_key"]]
            )
            assert z["source_ref"] == a["source_ref"]


FIXTURE = json.loads(
    Path("tests/fixtures/section-card-reference-errors-real.json").read_text()
)


@pytest.mark.parametrize(
    "fault", ["missing_scope", "unknown_card", "duplicate_card", "invalid_type"]
)
def test_invalid_host_scopes_are_rejected_without_fallback(fault):
    cards = facts()
    scopes = _writing_scopes(cards, "academic_v2", "en", None)
    if fault == "missing_scope":
        del scopes["methods"]
    elif fault == "unknown_card":
        scopes["methods"] = ["unknown"]
    elif fault == "duplicate_card":
        scopes["methods"] *= 2
    else:
        scopes["methods"] = [{"not": "an ID"}]
    with pytest.raises(ValueError, match="section_card_binding_invalid"):
        card_reference_binding(cards, scopes)


@pytest.mark.parametrize("issue", FIXTURE["rejected_references"])
def test_received_seven_reference_errors_remain_rejected(issue):
    cards = dict.fromkeys(FIXTURE["card_ids"], {})
    b = card_reference_binding(cards, FIXTURE["section_scopes"])
    body = {
        "title": "Reference regression only",
        "sections": {
            k: {
                "blocks": [
                    {
                        "heading": "",
                        "type": "paragraph",
                        "sentences": [
                            {
                                "kind": "scientific",
                                "text": "Not a scientific acceptance fixture.",
                                "scope_notice_id": "none",
                                "fact_card_ids": [ids[0]],
                            }
                        ],
                    }
                ]
            }
            for k, ids in FIXTURE["section_scopes"].items()
        },
    }
    section = issue["path"].split(".")[0]
    bad = issue["rejected_card_id"]
    body["sections"][section]["blocks"][0]["sentences"][0]["fact_card_ids"] = [bad]
    assert bad not in b["section_cards"][section].values()
    with pytest.raises(BodyRowError, match="section_card_reference_invalid"):
        encode_body(body, b)
