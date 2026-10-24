"""Verify lossless native report projection and durable sending boundaries."""

import copy
import hashlib
import json

import pytest
from test_writer_v2 import Provider, case  # noqa: F401

from litagent.body_quality import NATIVE_REPORT_INPUT_CONTRACT
from litagent.eval.batched import global_input_payload
from litagent.eval.scoped import build_evaluation_request
from litagent.quality import digest
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.writing_v2 import (
    WriterError,
    WriterSession,
    compile_document,
    provider_response_format,
    request_messages,
    review_payload,
    schema_for,
)


def source(case):
    """Compile the same native source consumed by Reviewer and Evaluation."""
    draft, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    return {
        "draft": draft,
        "writing_bindings": bindings,
        "writer_document": {"packet": case.packet, "aliases": case.aliases},
        "report_language": "en",
        "coverage_requirements": case.packet["requirements"],
        "fact_cards": case.cards,
        "evidence_snapshot_sha256": digest(case.cards),
        "evidence_selection": case.selection.to_dict(),
    }


def test_review_projection_preserves_every_byte_and_actual_citation(case):
    """Headers, whitespace, citations and the whole appendix survive reconstruction."""
    original = source(case)
    before = copy.deepcopy(original)
    projected = review_payload(original)
    restored = "".join(
        p["text"] if "text" in p else projected["units"][p["unit_id"]]["source_quote"]
        for p in projected["report_layout"]
    )
    assert restored.encode() == original["draft"].encode()
    assert hashlib.sha256(restored.encode()).hexdigest() == projected["report_sha256"]
    assert original == before and "draft" not in projected
    assert projected["sources"] == case.packet["sources"]
    for unit in original["writing_bindings"]["units"]:
        value = projected["units"][unit["unit_id"]]
        assert value["source_quote"] == unit["source_quote"]
        assert [
            projected["cards"][r]["evidence_id"] for r in value["fact_card_refs"]
        ] == unit["evidence_ids"]


@pytest.mark.parametrize(
    "fault",
    ["hash", "paper", "missing", "duplicate", "quote", "unknown", "source", "card"],
)
def test_review_projection_rejects_corrupt_host_binding(case, fault):
    """A compact representation never licenses tampered input or missing units."""
    value = source(case)
    unit = value["writing_bindings"]["units"][0]
    if fault == "hash":
        value["writing_bindings"]["report_sha256"] = "bad"
    elif fault == "paper":
        next(iter(value["writer_document"]["packet"]["sources"].values()))[
            "paper_id"
        ] = "other"
    elif fault == "missing":
        value["writing_bindings"]["units"].pop()
    elif fault == "duplicate":
        value["writing_bindings"]["units"].append(copy.deepcopy(unit))
    elif fault == "quote":
        unit["source_quote"] += " extra"
    elif fault == "unknown":
        unit["evidence_ids"] = ["outside"]
    elif fault == "source":
        next(iter(value["writer_document"]["packet"]["sources"].values()))[
            "source_quote"
        ] += " altered"
    else:
        next(iter(value["writer_document"]["packet"]["cards"].values()))[
            "evidence_id"
        ] = "outside"
    with pytest.raises(WriterError):
        review_payload(value)


def global_payload():
    """Retain canonical claim texts, contexts, exclusions and real citation aliases."""
    return {
        "global_output_contract": "host-global-completeness-v1",
        "report_input_contract": NATIVE_REPORT_INPUT_CONTRACT,
        "source_units": [
            {
                "source_unit_id": "unit:1",
                "scope_context": "On dataset A only, at peak accuracy.",
                "citation_aliases": ["e1"],
                "source_fragments": [
                    {
                        "source_fragment_id": "native:1",
                        "text": "EViT reduces FLOPs [E:e1].",
                        "citation_aliases": ["e1"],
                    },
                    {
                        "source_fragment_id": "native:2",
                        "text": "Organization text.",
                        "citation_aliases": [],
                    },
                ],
            }
        ],
        "assertions": [
            {
                "claim_id": "a1",
                "kind": "scientific",
                "claim_text": "EViT reduces FLOPs on A.",
                "source_unit_id": "unit:1",
                "source_fragment_id": "native:1",
                "evidence": [{"evidence_alias": "e1", "material_ref": "material:e1"}],
                "required_support_aspects": ["method", "condition", "direction"],
            }
        ],
        "fragment_inventory": [
            {
                "source_unit_id": "unit:1",
                "source_fragment_id": "native:1",
                "assertion_count": 1,
                "exclusion_reason": "",
            },
            {
                "source_unit_id": "unit:1",
                "source_fragment_id": "native:2",
                "assertion_count": 0,
                "exclusion_reason": "organization",
            },
        ],
        "schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }


def test_global_projection_keeps_prose_conditions_and_claim_parent():
    """Only redundant identity fields disappear; every semantic string is retained."""
    value = global_payload()
    before = copy.deepcopy(value)
    result = global_input_payload(value)
    assert value == before
    assert (
        result["source_units"][0]["scope_context"]
        == value["source_units"][0]["scope_context"]
    )
    for new, old in zip(
        result["source_units"][0]["source_fragments"],
        value["source_units"][0]["source_fragments"],
    ):
        assert new["text"] == old["text"]
        assert new["citation_aliases"] == old["citation_aliases"]
    claim = result["assertions"][0]
    assert claim["claim_text"] == value["assertions"][0]["claim_text"]
    assert claim["evidence_aliases"] == ["e1"]
    parent = next(
        u["source_unit_id"]
        for u in result["source_units"]
        if any(
            f["source_fragment_id"] == claim["source_fragment_id"]
            for f in u["source_fragments"]
        )
    )
    assert parent == value["assertions"][0]["source_unit_id"]
    assert result["assertion_kind_default"] == "scientific"
    assert result["fragment_inventory"][0]["exclusion_reason"] == "organization"
    assert len(result["fragment_inventory"]) == 1
    messages, schema = build_evaluation_request("Check everything.", value)
    assert schema["json_schema"]["schema"] == value["schema"]
    assert json.loads(messages[1]["content"])["assertions"] == result["assertions"]
    value.pop("report_input_contract")
    assert global_input_payload(value) is value
    old_messages, _ = build_evaluation_request("Check everything.", value)
    assert old_messages[0]["content"] == "Check everything."
    assert json.loads(old_messages[1]["content"])["assertions"] == value["assertions"]


@pytest.mark.parametrize(
    "fault",
    ["parent", "citation", "missing", "count", "duplicate", "omission", "exclusion"],
)
def test_global_projection_refuses_ambiguous_or_missing_claims(fault):
    """The request adapter cannot erase invalid inputs to make a report pass."""
    value = global_payload()
    if fault == "parent":
        value["assertions"][0]["source_unit_id"] = "other"
    elif fault == "citation":
        value["assertions"][0]["evidence"][0]["evidence_alias"] = "e2"
    elif fault == "missing":
        value["fragment_inventory"].pop()
    elif fault == "count":
        value["fragment_inventory"][0]["assertion_count"] = 2
    elif fault == "duplicate":
        value["assertions"].append(copy.deepcopy(value["assertions"][0]))
    elif fault == "omission":
        value["assertions"].clear()
    else:
        value["fragment_inventory"][0]["exclusion_reason"] = "skip science"
    with pytest.raises(ValueError, match="evaluation_input_"):
        global_input_payload(value)


@pytest.mark.asyncio
async def test_preflight_rejection_is_durable_and_does_not_send_or_spend(
    case, tmp_path
):
    """A context rejection records its exact stage, leaving cost and attempts unchanged."""
    llm = Provider(case)
    llm._cost_budget = CostBudget()
    session = WriterSession(llm, tmp_path / "r.json", "r", case.packet, case.aliases)
    before = llm._cost_budget.snapshot()
    with pytest.raises(ValueError, match="context_budget"):
        await session.call(
            "reviewer",
            "Untrusted sources only.",
            {"draft": "x" * 140000},
            schema_for("body", case.aliases),
            stage="reviewer",
            max_tokens=4096,
        )
    entry = session.journal["calls"]["reviewer"]
    assert entry["status"] == "rejected_not_sent"
    assert entry["input_sha256"] and entry["reservation"] > 117964
    assert session.journal["failure"]["stage"] == "reviewer"
    assert session.journal["failure"]["request_sent"] is False
    assert not llm.calls and not session.state["attempts"]
    assert llm._cost_budget.snapshot() == before
    durable = session.journal_path.read_bytes()
    resumed = WriterSession(llm, tmp_path / "r.json", "r", case.packet, case.aliases)
    with pytest.raises(WriterError, match="no_replay"):
        await resumed.call("reviewer", "system", {}, {}, stage="reviewer")
    assert session.journal_path.read_bytes() == durable


@pytest.mark.asyncio
async def test_transaction_preflight_preserves_failure_without_spending_attempt(
    case, tmp_path
):
    """Unfunded full reassessment cannot buy a partial repair or consume its counter."""
    llm = Provider(case)
    llm._cost_budget = CostBudget(20000)
    session = WriterSession(llm, tmp_path / "r.json", "r", case.packet, case.aliases)
    targets = {
        "x": {
            "path": ["methods", 0, 0],
            "original": case.body["methods"][0]["sentences"][0],
        }
    }
    with pytest.raises(WriterError, match="transaction_budget"):
        await session.patch(case.body, targets, cause="evaluation", claim_count=128)
    assert session.journal["failure"]["stage"] == "repair_preflight"
    assert session.journal["failure"]["request_sent"] is False
    assert not session.state["attempts"] and not llm.calls


def test_global_non_scientific_classification_does_not_erase_cited_source():
    """The evaluator may challenge a classification; the adapter preserves it as input."""
    value = global_payload()
    value["assertions"][0].update(kind="runtime_method", evidence=[])
    projected = global_input_payload(value)
    assert projected["assertions"][0]["kind"] == "runtime_method"
    assert projected["assertions"][0]["evidence_aliases"] == []
    assert projected["source_units"][0]["source_fragments"][0]["citation_aliases"] == [
        "e1"
    ]


def test_large_native_report_retains_prose_without_duplicate_transmission(case):
    """A 103-unit shaped report exercises the real serializer and output reservation."""
    from test_complete_native_repair import source_selection

    from litagent.quality import build_fact_cards
    from litagent.writing_v2 import BODY, SCOPE, source_packet

    ledger = {}
    for i, item in enumerate(case.ledger.values()):
        eid = "paper:public-source:claim:" + hashlib.sha256(str(i).encode()).hexdigest()
        ledger[eid] = {**item, "evidence_id": eid}
    case.cards = build_fact_cards(ledger)
    case.selection = source_selection(case.cards)
    case.selection.writing_scope_policy = SCOPE
    case.packet, case.aliases = source_packet(
        case.cards, case.selection, "Compare conditions.", [], "en"
    )
    for i, key in enumerate(BODY):
        case.body[key] = [
            {
                "sentences": [
                    {
                        "text": f"Finding {key} {j}: "
                        + "比较保留方法、数据集及其条件。" * 5
                        + "Conditions, values and causal qualifiers stay attached to this finding.",
                        "fact_card_refs": [next(iter(case.aliases))],
                    }
                    for j in range(18 if i == 0 else 19)
                ]
            }
        ]
    for key, n in (("abstract", 5), ("conclusion", 4)):
        case.summaries[key] = [
            {
                "sentences": [
                    {"text": f"Summary finding {j}.", "body_unit_refs": [f"u{j+1}"]}
                    for j in range(n)
                ]
            }
        ]
    original = source(case)
    assert len(original["writing_bindings"]["units"]) == 103
    projected = review_payload(original)
    schema = {
        "type": "object",
        "properties": {"issues": {"type": "array", "items": {"type": "string"}}},
        "required": ["issues"],
        "additionalProperties": False,
    }
    rf = provider_response_format("reviewer", schema)
    old = {
        "draft": original["draft"],
        "units": {u["unit_id"]: u for u in original["writing_bindings"]["units"]},
        "cards": case.packet["cards"],
        "sources": case.packet["sources"],
    }
    old_size = strict_request_reservation(
        request_messages("Review.", old, schema), 4096, rf
    )
    new_size = strict_request_reservation(
        request_messages("Review.", projected, schema), 4096, rf
    )
    assert old_size > 117964 > new_size
    assert projected["sources"] == old["sources"]
    assert [u["source_quote"] for u in projected["units"].values()] == [
        u["source_quote"] for u in original["writing_bindings"]["units"]
    ]
