"""Replay shared references without relabeling the original 040 decisions."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from test_support_receipts import envelope

from litagent.body_quality import ADAPTIVE_ASSESSMENT_INPUT_CONTRACT
from litagent.eval.batched import (
    _validate_decision_consistency,
    batch_payload,
    batch_prompt,
    batch_support_aliases,
    restore_batch_support_ids,
)
from litagent.eval.scoped import (
    SUPPORT_VALIDATION_INSTRUCTION,
    Assessment,
    ScopedEvaluationError,
    build_evaluation_request,
    build_support_passages,
    decision_status_instruction,
    required_support_aspects,
    support_receipt_payload,
    validate_assessment_decisions,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures/run040-shared-citation-decisions.json"
FIXTURE_SHA256 = "b304f3d8b1cc11337fe881323a8e50d12c35185965524a0e3cc4db45756ede76"
REVISION = "host-support-receipts-v1"
SUBSET_RULE = "support_basis may use a sufficient subset"


@pytest.fixture(scope="module")
def real040():
    """Retain the public originals, actual references and unmodified diagnostics."""
    assert hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest() == FIXTURE_SHA256
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def replay_packet(fixture):
    """Rebuild host passages and batch ownership from the public source records."""
    aliases = fixture["aliases"]
    ledger = {}
    for eid, alias in aliases.items():
        material = fixture["cited_materials"][alias]
        source = fixture["original_supports"][material["original_support_ref"]]
        ledger[eid] = {
            "paper_id": source["paper_id"],
            "paper_title": material["paper_title"],
            "text": source["text"],
            "supporting_text": source["text"],
        }
    passages = build_support_passages(ledger, aliases, original_only=True)
    table = {}
    for alias, ref, start, end in fixture["actual_support_passages"].values():
        pid = next(
            pid
            for pid, row in passages.items()
            if row["evidence_alias"] == alias
            and row["source_start"] == start
            and row["source_end"] == end
        )
        table[pid] = [alias, ref, start, end]
    claims = []
    for case in fixture["cases"]:
        original = case["claim"]
        claims.append(
            {
                key: original[key]
                for key in (
                    "claim_id",
                    "kind",
                    "claim_text",
                    "source_unit_id",
                    "source_fragment_id",
                )
            }
            | {
                "evidence": [
                    {"evidence_alias": aliases[eid], "material_ref": aliases[eid]}
                    for eid in original["evidence_ids"]
                ],
                "required_support_aspects": required_support_aspects(
                    original["claim_text"],
                    applicability=True,
                    support_validation_revision=REVISION,
                ),
            }
        )
    units = []
    for fragment in fixture["fragments"].values():
        cited = fragment["citation_aliases"]
        units.append(
            {
                "source_unit_id": fragment["source_unit_id"],
                "scope_context": "",
                "citation_aliases": list(cited),
                "source_fragments": [
                    {
                        "source_fragment_id": fragment["native_source_fragment_id"],
                        "text": fragment["prose"]
                        + " "
                        + " ".join(f"[E:{a}]" for a in cited),
                        "citation_aliases": list(cited),
                    }
                ],
            }
        )
    payload = batch_payload(
        {
            "body_analysis_contract": "body-analysis-v1",
            "assessment_input_contract": ADAPTIVE_ASSESSMENT_INPUT_CONTRACT,
            "evaluation_input_contract": "flat-evaluation-input-v1",
            "writer_envelope": envelope(),
            "source_units": units,
            "report_structure": [
                {"source_unit_id": row["source_unit_id"], "section": "introduction"}
                for row in units
            ],
            "unit_inventory": [],
            "cited_materials": copy.deepcopy(fixture["cited_materials"]),
            "original_supports": copy.deepcopy(fixture["original_supports"]),
            "support_passages": table,
            "host_runtime_record": {},
        },
        claims,
    )
    return payload, ledger, passages


def restored_decision(payload, raw):
    """Decode the actual selected pN wire using the production consumer."""
    value = Assessment.model_validate(
        {"units": [], "decisions": [raw], "conflicts": []}
    )
    return restore_batch_support_ids(value, payload).decisions[0]


def controlled_decision(payload, case, *, support_id=None):
    """Create a separate positive control; never modify the archived decision."""
    raw = copy.deepcopy(case["raw_decision"])
    raw.update(
        status="supported",
        reason_code="supported",
        explanation="Separate controlled subset response, not an archived verdict.",
    )
    if support_id is not None:
        raw["support_basis"] = [{"support_id": support_id}]
        for aspect in raw["aspect_checks"]:
            aspect["support_ids"] = [support_id]
    return restored_decision(payload, raw)


def host_checked(fixture, payload, ledger, passages, case, decision, *, claim=None):
    """Run the strict host validator after the batch consistency consumer."""
    claim = copy.deepcopy(claim or case["claim"])
    return validate_assessment_decisions(
        [claim],
        [decision],
        ledger=ledger,
        aliases=fixture["aliases"],
        support_passages=passages,
        context={
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "review_methodology": True,
            "_independent_classification": True,
            "_support_validation_revision": REVISION,
            "_evaluation_progress": {
                "global_findings_contract": "anchored-global-findings-v2"
            },
        },
        strict=True,
    )[0]


def test_public_fixture_keeps_actual_shared_citations_and_complete_originals(real040):
    assert real040["science_gold_standard"] is False
    assert len(real040["cases"]) == 3
    assert len(real040["fragments"]) == 2
    assert len(real040["aliases"]) == len(real040["cited_materials"]) == 15
    assert len(real040["original_supports"]) == 11
    assert set(real040) == {
        "contract",
        "science_gold_standard",
        "provenance",
        "fragments",
        "aliases",
        "cited_materials",
        "original_supports",
        "actual_support_passages",
        "cases",
    }
    payload, _, _ = replay_packet(real040)
    assert list(payload["support_passages"].values()) == list(
        real040["actual_support_passages"].values()
    )
    permitted = support_receipt_payload(payload)["assertions"]
    reverse = {pid: alias for alias, pid in batch_support_aliases(payload).items()}
    for case, claim in zip(real040["cases"], permitted, strict=True):
        matching = [
            fragment
            for fragment in real040["fragments"].values()
            if fragment["native_source_fragment_id"] == claim["source_fragment_id"]
            and fragment["source_unit_id"] == claim["source_unit_id"]
        ]
        assert len(matching) == 1
        assert [row["evidence_alias"] for row in claim["evidence"]] == (
            matching[0]["citation_aliases"]
        )
        assert claim["required_support_aspects"] == case["required_support_aspects"]
        assert [reverse[p] for p in claim["allowed_support_ids"]] == (
            case["actual_allowed_support_ids"]
        )
    for alias, ref, start, end in payload["support_passages"].values():
        assert real040["cited_materials"][alias]["original_support_ref"] == ref
        assert start == 0 and end == len(real040["original_supports"][ref]["text"])


@pytest.mark.parametrize("index", range(3))
def test_raw040_all_positive_aspects_still_reject_unsupported(real040, index):
    before = copy.deepcopy(real040)
    payload, _, _ = replay_packet(real040)
    frozen = copy.deepcopy(payload)
    case = real040["cases"][index]
    decision = restored_decision(payload, case["raw_decision"])
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_decision_consistency_invalid"
    ) as caught:
        _validate_decision_consistency(payload["assertions"][index], decision, payload)
    issue = caught.value.details["decision_inconsistencies"][0]
    assert issue["claim_id"] == case["claim"]["claim_id"]
    assert issue["subreason"] == case["expected_consistency_error"]["subreason"]
    assert issue["original_decision"]["status"] == "unsupported"
    assert issue["original_decision"]["reason_code"] == "attribution_mismatch"
    assert payload == frozen and real040 == before


def test_controlled_sufficient_subset_does_not_require_all_shared_papers(real040):
    before = copy.deepcopy(real040)
    payload, ledger, passages = replay_packet(real040)
    case = real040["cases"][1]
    decision = controlled_decision(payload, case)
    claim = support_receipt_payload(payload)["assertions"][1]
    assert len(claim["evidence"]) == len(claim["allowed_support_ids"]) == 15
    assert len(decision.support_basis) == 1
    assert decision.support_basis[0].support_id == "e11:full_original"
    _validate_decision_consistency(claim, decision, payload)
    checked = host_checked(real040, payload, ledger, passages, case, decision)
    assert checked["status"] == "supported"
    assert checked["support_validation"]["status"] == "passed"
    assert real040 == before


@pytest.mark.parametrize("fault", ["wrong_paper", "entity", "condition"])
def test_permitted_shared_pool_cannot_license_an_unrelated_basis(real040, fault):
    payload, ledger, passages = replay_packet(real040)
    case = real040["cases"][1]
    claim = copy.deepcopy(case["claim"])
    decision = controlled_decision(
        payload, case, support_id="p13" if fault == "wrong_paper" else None
    )
    if fault == "entity":
        claim["claim_text"] = claim["claim_text"].replace("K-tuplet", "MSENet")
    elif fault == "condition":
        claim["claim_text"] = claim["claim_text"].replace(
            "Caltech-101", "TieredImageNet"
        )
    checked = host_checked(
        real040, payload, ledger, passages, case, decision, claim=claim
    )
    assert checked["status"] == "unverifiable"
    assert checked["support_validation"]["status"] != "passed"


@pytest.mark.parametrize("index", [0, 2])
def test_source_spelling_is_not_repaired_to_make_original040_supported(real040, index):
    payload, ledger, passages = replay_packet(real040)
    case = real040["cases"][index]
    assert "miniImageNet" in case["claim"]["claim_text"]
    assert "miniImagenNet" in real040["original_supports"]["s5"]["text"]
    checked = host_checked(
        real040, payload, ledger, passages, case, controlled_decision(payload, case)
    )
    assert checked["status"] == "unverifiable"
    assert checked["local_error"] == "support_entity_unverified"


def test_matching_source_spelling_still_cannot_license_a_wrong_quantity(real040):
    payload, ledger, passages = replay_packet(real040)
    case = real040["cases"][0]
    claim = copy.deepcopy(case["claim"])
    claim["claim_text"] = claim["claim_text"].replace("miniImageNet", "miniImagenNet")
    claim["claim_text"] = claim["claim_text"].replace("2%", "3%")
    checked = host_checked(
        real040,
        payload,
        ledger,
        passages,
        case,
        controlled_decision(payload, case),
        claim=claim,
    )
    assert checked["status"] == "unverifiable"
    assert checked["local_error"] == "support_number_unverified"


def test_another_fragment_citation_cannot_restore_missing_actual_ownership(real040):
    value = copy.deepcopy(real040)
    eid = next(eid for eid, alias in value["aliases"].items() if alias == "e11")
    value["fragments"]["f1"]["citation_aliases"].remove("e11")
    for case in value["cases"][:2]:
        case["claim"]["evidence_ids"].remove(eid)
    payload, ledger, passages = replay_packet(value)
    case = value["cases"][1]
    claim = support_receipt_payload(payload)["assertions"][1]
    assert "e11:full_original" in payload["support_passages"]
    assert "e11:full_original" not in claim["allowed_support_ids"]
    checked = host_checked(
        value, payload, ledger, passages, case, controlled_decision(payload, case)
    )
    assert checked["status"] == "unverifiable"
    assert checked["local_error"] == "support_basis_ownership_invalid"


def test_a_negative_required_aspect_remains_a_real_negative_decision(real040):
    payload, ledger, passages = replay_packet(real040)
    case = real040["cases"][1]
    decision = restored_decision(payload, case["raw_decision"])
    decision.aspect_checks[0].supported = False
    _validate_decision_consistency(payload["assertions"][1], decision, payload)
    checked = host_checked(real040, payload, ledger, passages, case, decision)
    assert checked["status"] == "unsupported"
    decision.status, decision.reason_code = "supported", "supported"
    with pytest.raises(ScopedEvaluationError, match="decision_consistency_invalid"):
        _validate_decision_consistency(payload["assertions"][1], decision, payload)


def test_actual_builder_adds_subset_rule_once_and_keeps_sources_and_schema(real040):
    payload, _, _ = replay_packet(real040)
    before = copy.deepcopy(payload)
    messages, schema = build_evaluation_request(batch_prompt(payload), payload)
    assert messages[0]["content"].count(SUBSET_RULE) == 1
    assert messages[0]["content"].count(SUPPORT_VALIDATION_INSTRUCTION) == 1
    wire = json.loads(messages[1]["content"])
    assert wire["original_supports"] == real040["original_supports"]
    assert list(wire["support_passages"].values()) == list(
        real040["actual_support_passages"].values()
    )
    assert schema["json_schema"]["schema"]["properties"]["decisions"]
    other = copy.deepcopy(payload)
    other["writer_envelope"] = envelope(active=False)
    built, old_schema = build_evaluation_request(batch_prompt(other), other)
    annotated = copy.deepcopy(schema)
    fields = annotated["json_schema"]["schema"]["$defs"]["PassageDecision"][
        "properties"
    ]
    assert fields["status"].pop("description") == "Must match reason_code."
    assert fields["reason_code"].pop("description") == decision_status_instruction()
    assert old_schema == annotated
    assert SUBSET_RULE not in built[0]["content"]
    assert SUPPORT_VALIDATION_INSTRUCTION not in built[0]["content"]
    assert payload == before
