"""Keep bidirectional source ownership explicit without weakening verification."""

import copy
import json

import pytest
from test_fragment_repair import case
from test_strict_quality import draft, to_native

from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.quality import (
    GroundedWritingError,
    build_fact_cards,
    grounded_messages,
    prepare_fragment_repair_request,
    prepare_grounded_request,
    render_grounded_report,
)


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_same_paper_same_original_keeps_every_card_next_to_one_source(language):
    facts = build_fact_cards(
        {
            eid: {
                "paper_id": "p1",
                "supporting_text": "DeiT uses ImageNet and pre-training.",
            }
            for eid in ("a", "b", "c")
        }
    )
    messages = grounded_messages(facts, "academic_v2", language, "DeiT")
    original = json.loads(messages[1]["content"])
    assert len(original["sources"]) == 1
    source = next(iter(original["sources"].values()))
    assert source["fact_card_ids"] == list(facts)
    assert source["source_quote"] == "DeiT uses ImageNet and pre-training."
    prepared, _ = prepare_grounded_request(messages)
    projected = json.loads(prepared[1]["content"])
    assert list(projected["sources"].values()) == [source]
    for cid, card in projected["cards"].items():
        assert cid in projected["sources"][card["source_ref"]]["fact_card_ids"]
    assert original["cards"].keys() == projected["cards"].keys()


def test_identical_text_from_different_papers_never_merges_owners():
    facts = build_fact_cards(
        {
            eid: {"paper_id": eid, "supporting_text": "Models use training data."}
            for eid in ("p1", "p2")
        }
    )
    raw = json.loads(
        grounded_messages(facts, "academic_v2", "en", "Compare")[1]["content"]
    )
    assert len(raw["sources"]) == 2
    for source in raw["sources"].values():
        assert len(source["fact_card_ids"]) == 1
        assert facts[source["fact_card_ids"][0]]["paper_id"] == source["paper_id"]


def test_inverse_index_preserves_existing_optional_source_metadata_selection():
    facts = build_fact_cards(
        {
            eid: {"paper_id": "p1", "supporting_text": "DeiT uses ImageNet."}
            for eid in ("a", "b")
        }
    )
    last = next(reversed(facts))
    facts[last]["document_subject"] = {"term": "DeiT", "basis": "explicit_source_title"}
    source = next(
        iter(
            json.loads(
                grounded_messages(facts, "academic_v2", "en", "DeiT")[1]["content"]
            )["sources"].values()
        )
    )
    assert source["document_subject"] == facts[last]["document_subject"]
    assert source["fact_card_ids"] == list(facts)


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_fragment_projection_preserves_owner_map_and_complete_originals(language):
    body, _, facts, targets, *_ = case(language)
    messages = grounded_messages(facts, "academic_v2", language, "Compare", draft=body)
    original = json.loads(messages[1]["content"])
    projected, _ = prepare_fragment_repair_request(messages, targets)
    packet = json.loads(projected[1]["content"])
    assert list(packet["sources"].values()) == list(original["sources"].values())
    for cid, card in packet["cards"].items():
        source = packet["sources"][card["source_ref"]]
        assert cid in source["fact_card_ids"]
        assert source["source_quote"] == facts[cid]["source_quote"]


@pytest.mark.parametrize("language", ["en", "zh-CN"])
@pytest.mark.parametrize("fault", [None, "other_paper", "hyphen_removed"])
def test_positive_and_wrong_attribution_or_changed_pdf_bytes(language, fault):
    facts = build_fact_cards(
        {
            "a": {
                "paper_id": "deit",
                "supporting_text": "DeiT uses pre-training on ImageNet.",
            },
            "b": {
                "paper_id": "datafree",
                "supporting_text": "Data-free distillation uses a teacher.",
            },
        }
    )
    payload = to_native(draft(facts, language), facts)
    # This local writing fixture tests lineage only; scientific truth is independent.
    scientific = payload["sections"]["methods"]["first_block"]["first_sentence"]
    if fault == "other_paper":
        scientific["support_quotes"][1] = facts[scientific["fact_card_ids"][0]][
            "source_quote"
        ]
    elif fault == "hyphen_removed":
        scientific["support_quotes"][0] = scientific["support_quotes"][0].replace(
            "pre-training", "pretraining"
        )
    kwargs = {"require_native": True, "require_source_quotes": True}
    if fault:
        with pytest.raises(GroundedWritingError, match="source_quote_contract_invalid"):
            render_grounded_report(
                json.dumps(payload), facts, "academic_v2", language, **kwargs
            )
    else:
        assert "[E:a]" in render_grounded_report(
            json.dumps(payload), facts, "academic_v2", language, **kwargs
        )


def test_shared_instructions_keep_pdf_characters_and_do_not_exempt_semantics():
    facts = build_fact_cards(
        {
            "a": {
                "paper_id": "p",
                "supporting_text": "pre-training\nDeiT⚗ improves performance.",
            }
        }
    )
    messages = grounded_messages(facts, "academic_v2", "en", "DeiT")
    prepared, _ = prepare_grounded_request(messages)
    instruction = prepared[0]["content"]
    assert "PDF hyphenation, spaces, spelling and symbols exactly" in instruction
    assert "Quotes establish locations, not scientific truth" in instruction
    assert (
        next(iter(json.loads(prepared[1]["content"])["sources"].values()))[
            "source_quote"
        ]
        == "pre-training\nDeiT⚗ improves performance."
    )


def test_v14_identity_changes_strict_only_and_does_not_touch_inputs(monkeypatch):
    from litagent import contracts

    config = load_config()
    legacy = build_config_summary(config)
    strict = build_config_summary(config, "strict_auto_v1")
    monkeypatch.setattr(contracts, "WRITING_VERSION", "grounded-writing-v13")
    assert build_config_summary(config) == legacy
    assert (
        build_config_summary(config, "strict_auto_v1")["fingerprint"]
        != strict["fingerprint"]
    )
    facts = build_fact_cards(
        {"a": {"paper_id": "p", "supporting_text": "Exact original."}}
    )
    frozen = copy.deepcopy(facts)
    grounded_messages(facts, "academic_v2", "en", "Question")
    assert facts == frozen
