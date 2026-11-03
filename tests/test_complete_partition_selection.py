"""Select complete feasible partitions using real source and wire consumers."""

import copy
import hashlib
import itertools
import json
from types import SimpleNamespace

import pytest

from litagent.body_quality import ADAPTIVE_ASSESSMENT_INPUT_CONTRACT
from litagent.capacity import CapacityConfig
from litagent.config import EvalConfig
from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval import batched, scoped
from litagent.evidence import extract_evidence_refs_ordered
from litagent.quality import build_fact_cards
from litagent.report_render import build_report_source_units
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import (
    check_inventory_envelope,
    check_writer_envelope,
    make_writer_envelope,
    nested_json_size,
)
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    SCOPE,
    bound_fragments,
    compile_document,
    source_packet,
    validate_body,
)

SOURCE_CHARACTERS = 1284
CLAIM_CHARACTERS = 285
LEGACY_WIRE_SHA256 = "a1e3ffc132e49857ef6d1434bad4a694c0efb32759bd8aa6246f2ca35e164d20"


def _partition_case(*, selected=True, duplicate_text=False):
    """Build bounded synthetic prose without loading a run or creating an owner."""
    originals = {}
    for index in range(14):
        prefix = f"Synthetic source {index:02d} states a controlled limitation. "
        originals[index] = prefix + "q" * (SOURCE_CHARACTERS - len(prefix))
    ledger = {}
    for index in range(20):
        eid = f"synthetic-evidence-{index:02d}-" + "r" * 40
        paper = index % len(originals)
        ledger[eid] = {
            "evidence_id": eid,
            "paper_id": f"synthetic-paper-{paper:02d}",
            "paper_title": f"Synthetic source title {paper:02d}",
            "chunk_key": "abstract",
            "text": "Controlled finding candidate; not scientific gold.",
            "supporting_text": originals[paper],
        }
    selection = EvidenceSelection(
        candidate_count=len(ledger),
        selected_items=ledger,
        section_evidence_ids={section: list(ledger) for section in BODY},
        method_by_section={section: "lexical_fallback" for section in BODY},
        omitted_count=0,
        estimated_tokens=1000,
        writing_scope_policy=SCOPE,
    )
    cards = build_fact_cards(ledger)
    packet, card_aliases = source_packet(
        cards, selection, "Explain controlled limitations.", [], "zh-CN"
    )
    envelope = make_writer_envelope(
        body_units=66,
        assertions=128,
        body_bytes=32000,
        source_identity="a" * 64,
        builder_identity="b" * 64,
        contract="flat-writer-envelope-v1",
        capacity=CapacityConfig(),
        round_budget=True,
        global_findings_contract="anchored-global-findings-v2",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        summary_constraints=True,
        classification_review=True,
        repair_feedback=True,
        reference_contract="ordered-reference-set-v1",
        transport_revision="host-lossless-transport-v1",
        support_validation_revision="host-support-receipts-v2",
        **(
            {"evaluation_response_contract": batched.EVALUATION_RESPONSE_CONTRACT}
            if selected
            else {}
        ),
    )
    refs = list(packet["cards"])
    exclusions = list(
        itertools.chain(
            itertools.combinations(range(len(refs)), 1),
            itertools.combinations(range(len(refs)), 2),
        )
    )
    body = {
        "title": "Synthetic partition shape witness",
        **{section: [] for section in BODY},
    }
    for index in range(65):
        label = 0 if duplicate_text and index in {0, 5} else index
        prefix = f"Method m{label} has a source-bound limitation. "
        text = prefix + "x" * (CLAIM_CHARACTERS - len(prefix))
        own_refs = [
            ref for offset, ref in enumerate(refs) if offset not in exclusions[index]
        ]
        section = BODY[index % len(BODY)]
        if not body[section] or len(body[section][-1]["sentences"]) == 3:
            body[section].append({"sentences": []})
        body[section][-1]["sentences"].append(
            {"text": text, "fact_card_refs": own_refs}
        )
    body_units = validate_body(
        body, card_aliases, writing_plan=RANGE_WRITING_PLAN, envelope=envelope
    )
    check_writer_envelope(body, envelope, stage="body")
    summaries = {
        section: [
            {
                "sentences": [
                    {
                        "text": "Synthetic shape only, not scientific gold.",
                        "body_unit_refs": [next(iter(body_units))],
                    }
                ]
            }
        ]
        for section in ("abstract", "conclusion")
    }
    report, bindings = compile_document(
        body,
        summaries,
        cards,
        card_aliases,
        "zh-CN",
        {},
        writing_plan=RANGE_WRITING_PLAN,
        envelope=envelope,
    )
    fragments_by_unit = bound_fragments(report, bindings)
    eids = extract_evidence_refs_ordered(report)
    aliases = {eid: f"e{index}" for index, eid in enumerate(eids, 1)}
    units, claims = [], []
    for unit in build_report_source_units(report):
        fragments = fragments_by_unit.get(unit.unit_id)
        if not fragments or any(
            f":native:writer:{section}:" in fragments[0]["source_fragment_id"]
            for section in ("abstract", "conclusion")
        ):
            continue
        units.append(
            {
                "source_unit_id": unit.unit_id,
                "scope_context": scoped._alias_text(
                    report[max(0, unit.start - 400) : unit.start], aliases
                ),
                "section": scoped._section(report, unit.start, academic_parent=True),
                "citation_aliases": extract_evidence_refs_ordered(
                    scoped._alias_text(unit.text, aliases)
                ),
                "source_fragments": [
                    {
                        "source_fragment_id": fragment["source_fragment_id"],
                        "text": scoped._alias_text(fragment["source_quote"], aliases),
                        "citation_aliases": [
                            aliases[eid] for eid in fragment["citation_aliases"]
                        ],
                    }
                    for fragment in fragments
                ],
            }
        )
        for fragment in fragments:
            record = next(
                record
                for record in bindings["units"]
                if record["source_start"] == unit.start + fragment["start"]
            )
            section, paragraph, sentence = record["path"]
            claims.append(
                {
                    "claim_id": f"claim_{len(claims)}",
                    "kind": "scientific",
                    "claim_text": body[section][paragraph]["sentences"][sentence][
                        "text"
                    ],
                    "source_unit_id": unit.unit_id,
                    "source_fragment_id": fragment["source_fragment_id"],
                    "evidence_ids": fragment["citation_aliases"],
                }
            )
    check_inventory_envelope(claims, envelope)
    _, materials, supports, passages, canonical = scoped._assessment_support_inputs(
        claims,
        ledger,
        eids,
        aliases,
        envelope=envelope,
        context={"quality_policy": "strict_auto_v2", "body_quality": "analysis_v1"},
    )
    payload = batched.batch_payload(
        {
            "body_analysis_contract": "body-analysis-v1",
            "assessment_input_contract": ADAPTIVE_ASSESSMENT_INPUT_CONTRACT,
            "evaluation_input_contract": batched.FLAT_EVALUATION_INPUT,
            "writer_envelope": envelope,
            "source_units": units,
            "report_structure": [
                {
                    "source_unit_id": unit["source_unit_id"],
                    "section": unit["section"],
                }
                for unit in units
            ],
            "unit_inventory": [],
            "assertions": canonical,
            "cited_materials": {
                aliases[eid]: material for eid, material in materials.items()
            },
            "original_supports": supports,
            "support_passages": scoped.support_passage_table(
                passages, aliases, materials
            ),
            "host_runtime_record": {},
            "coverage_requirements": [],
        },
        canonical,
    )
    batched.batch_alias_input(payload)
    return SimpleNamespace(
        payload=payload,
        body=body,
        claims=claims,
        report=report,
        bindings=bindings,
        source_texts=sorted(originals.values()),
        config=EvalConfig().model_copy(
            update={
                "strict_max_claims": 1024,
                "strict_max_batches": 64,
                "strict_batch_size": 16,
            }
        ),
    )


def _size(batch, config):
    """Measure the complete actual native request rather than claim counts."""
    messages, native = scoped.build_evaluation_request(
        batched.batch_prompt(batch), batch
    )
    amount = strict_request_reservation(messages, config.max_tokens, native)
    assert amount == batched.request_size(
        batched.batch_prompt(batch), batch, config.max_tokens
    )
    return amount


def _wire_digest(batches):
    """Bind all complete legacy messages and native formats in stable order."""
    wires = [
        scoped.build_evaluation_request(batched.batch_prompt(batch), batch)
        for batch in batches
    ]
    return hashlib.sha256(
        json.dumps(wires, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


@pytest.fixture(scope="module")
def partition_case():
    """Keep the finite regression independent of local acceptance artifacts."""
    return _partition_case()


def test_complete_candidate_selection_reproduces_ten_versus_nine(partition_case):
    case = partition_case
    payload, config = case.payload, case.config
    claims = payload["assertions"]
    balanced = [
        batched.batch_payload(payload, claims[offset::5]) for offset in range(5)
    ]
    ordered = [
        batched.batch_payload(payload, claims[index : index + 16])
        for index in range(0, len(claims), 16)
    ]
    balanced_sizes = [_size(batch, config) for batch in balanced]
    ordered_sizes = [_size(batch, config) for batch in ordered]
    assert 100000 < max(balanced_sizes) < max(ordered_sizes)
    assert all(amount > 100000 for amount in balanced_sizes)
    assert len(ordered[-1]["assertions"]) == 1
    balanced_halves = [
        batched.batch_payload(payload, batch["assertions"][start:end])
        for batch in balanced
        for start, end in ((0, 6), (6, 13))
    ]
    assert len(balanced_halves) == 10
    assert max(_size(batch, config) for batch in balanced_halves) <= 100000
    actual = batched.plan_batches(payload, config)
    assert [len(batch["assertions"]) for batch in actual] == [8] * 8 + [1]
    assert sum(_size(batch, config) for batch in actual) < sum(
        _size(batch, config) for batch in balanced_halves
    )


def test_partition_retains_every_claim_source_and_native_permission(partition_case):
    case = partition_case
    before = copy.deepcopy(case.payload)
    batches = batched.plan_batches(case.payload, case.config)
    actual_claims = {
        claim["claim_id"]: claim for batch in batches for claim in batch["assertions"]
    }
    expected = {claim["claim_id"]: claim for claim in case.payload["assertions"]}
    assert (
        len(actual_claims) == sum(len(batch["assertions"]) for batch in batches) == 65
    )
    assert actual_claims == expected and case.payload == before
    assert nested_json_size(case.body) <= 32000
    assert nested_json_size([claim["claim_text"] for claim in case.claims]) <= 30000
    assert (
        sorted(source["text"] for source in case.payload["original_supports"].values())
        == case.source_texts
    )
    for batch in batches:
        messages, native = scoped.build_evaluation_request(
            batched.batch_prompt(batch), batch
        )
        wire = json.loads(messages[1]["content"])
        assert wire["original_supports"] == batch["original_supports"]
        assert _size(batch, case.config) <= 100000
        schema = native["json_schema"]["schema"]
        by_id = {claim["claim_id"]: claim for claim in batch["assertions"]}
        for reference in schema["properties"]["decisions"]["items"]["anyOf"]:
            branch = schema["$defs"][reference["$ref"].rsplit("/", 1)[-1]]
            properties = branch["properties"]
            for claim_id in properties["claim_id"]["enum"]:
                cited = {item["evidence_alias"] for item in by_id[claim_id]["evidence"]}
                allowed = {
                    pid
                    for pid, passage in wire["support_passages"].items()
                    if passage[0] in cited
                }
                assert (
                    set(
                        properties["support_basis"]["items"]["properties"][
                            "support_id"
                        ]["enum"]
                    )
                    == allowed
                )
                assert (
                    set(
                        properties["aspect_checks"]["items"]["properties"][
                            "support_ids"
                        ]["items"]["enum"]
                    )
                    == allowed
                )


def test_unaffordable_balanced_candidate_does_not_hide_feasible_ordered(partition_case):
    config = partition_case.config.model_copy(update={"strict_max_batches": 9})
    rejected = []
    batches = batched.plan_batches(
        partition_case.payload, config, on_rejected=lambda *args: rejected.append(args)
    )
    assert [len(batch["assertions"]) for batch in batches] == [8] * 8 + [1]
    assert max(_size(batch, config) for batch in batches) <= 100000
    assert rejected == []


def test_both_unaffordable_candidates_keep_explicit_resource_error(partition_case):
    config = partition_case.config.model_copy(update={"strict_max_batches": 8})
    rejected = []
    with pytest.raises(ValueError, match="evaluation_capacity_context_unaffordable"):
        batched.plan_batches(
            partition_case.payload,
            config,
            on_rejected=lambda *args: rejected.append(args),
        )
    assert len(rejected) == 1
    assert rejected[0][0] == "evaluation_capacity_context_unaffordable"
    assert rejected[0][1] is partition_case.payload


def test_permission_failure_is_not_swallowed_as_a_partition_resource_error(
    partition_case,
):
    payload = copy.deepcopy(partition_case.payload)
    payload["assertions"][0]["allowed_support_ids"] = ["forged-source"]
    rejected = []
    with pytest.raises(ValueError, match="evaluation_support_permission_invalid"):
        batched.plan_batches(
            payload,
            partition_case.config,
            on_rejected=lambda *args: rejected.append(args),
        )
    assert rejected == []


def test_equal_text_at_distinct_positions_is_not_merged():
    case = _partition_case(duplicate_text=True)
    first, second = case.payload["assertions"][:2]
    assert first["claim_text"] == second["claim_text"]
    assert first["source_fragment_id"] != second["source_fragment_id"]
    batches = batched.plan_batches(case.payload, case.config)
    claims = [claim for batch in batches for claim in batch["assertions"]]
    assert len(claims) == len({claim["claim_id"] for claim in claims}) == 65
    assert {first["claim_id"], second["claim_id"]}.issubset(
        {claim["claim_id"] for claim in claims}
    )


def test_none_preserves_exact_frozen_legacy_partition_and_wire():
    case = _partition_case(selected=False)
    assert "evaluation_response_contract" not in case.payload["writer_envelope"]
    batches = batched.plan_batches(case.payload, case.config)
    assert [
        [claim["claim_id"] for claim in batch["assertions"]] for batch in batches
    ] == [
        [f"claim_{index}" for index in range(start, min(start + 16, 65))]
        for start in range(0, 65, 16)
    ]
    assert _wire_digest(batches) == LEGACY_WIRE_SHA256
