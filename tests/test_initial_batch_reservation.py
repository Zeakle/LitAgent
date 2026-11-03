"""Keep initial assertion capacity separate from complete native batch counts."""

import copy
import itertools
import json
import socket
from collections import Counter
from types import SimpleNamespace

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


def _initial_batch_case():
    """Compile 65 native body units and two summaries from synthetic originals."""
    originals = {}
    for index in range(14):
        prefix = f"Synthetic source {index:02d} states a controlled limitation. "
        originals[index] = prefix + "q" * (1284 - len(prefix))
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
        evaluation_response_contract=batched.EVALUATION_RESPONSE_CONTRACT,
    )
    refs = list(packet["cards"])
    exclusions = list(
        itertools.chain(
            itertools.combinations(range(len(refs)), 1),
            itertools.combinations(range(len(refs)), 2),
        )
    )
    body = {
        "title": "Synthetic initial batch capacity witness",
        **{section: [] for section in BODY},
    }
    index = 0
    for section in BODY:
        sentences = []
        for _ in range(61 if section == "methods" else 1):
            prefix = f"Method m{index} has a source-bound limitation. "
            text = prefix + "x" * (285 - len(prefix))
            sentences.append(
                {
                    "text": text,
                    "fact_card_refs": [
                        ref
                        for offset, ref in enumerate(refs)
                        if offset not in exclusions[index]
                    ],
                }
            )
            index += 1
        body[section].append({"sentences": sentences})
    body_units = validate_body(
        body, card_aliases, writing_plan=RANGE_WRITING_PLAN, envelope=envelope
    )
    check_writer_envelope(body, envelope, stage="body")
    first, last = next(iter(body_units)), list(body_units)[-1]
    summaries = {
        section: [
            {"sentences": [{"text": body_units[ref]["text"], "body_unit_refs": [ref]}]}
        ]
        for section, ref in (("abstract", first), ("conclusion", last))
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
    units, raw_inventory, unit_records = [], {}, {}
    for unit in build_report_source_units(report):
        fragments = fragments_by_unit.get(unit.unit_id)
        if not fragments:
            continue
        unit_records[unit.unit_id] = unit
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
            binding = next(
                record
                for record in bindings["units"]
                if record["source_start"] == unit.start + fragment["start"]
            )
            section, paragraph, sentence = binding["path"]
            document = summaries if section in summaries else body
            text = document[section][paragraph]["sentences"][sentence]["text"]
            raw_inventory[fragment["source_fragment_id"]] = {
                "assertions": [{"kind": "scientific", "claim_text": text}],
                "exclusion_reason": "",
            }
    inventory = scoped.flat_inventory_response(
        json.dumps({"fragments": raw_inventory}),
        units,
        {},
        research_gaps=True,
        classification_review=True,
    )
    claims = []
    for entry in inventory.units:
        unit = unit_records[entry.source_unit_id]
        fragments = {
            fragment["source_fragment_id"]: fragment
            for fragment in fragments_by_unit[unit.unit_id]
        }
        for assertion in entry.assertions:
            fragment = fragments[assertion.source_fragment_id]
            start = unit.start + fragment["start"]
            assert not scoped._pure_organization(assertion.claim_text)
            assert not scoped._source_scope_issue(
                assertion.claim_text,
                fragment["source_quote"],
                report[max(0, start - 400) : start],
            )
            assert set(assertion.evidence_aliases) == {
                aliases[eid] for eid in fragment["citation_aliases"]
            }
            claims.append(
                {
                    "claim_id": f"claim_{len(claims)}",
                    "kind": assertion.kind,
                    "claim_text": assertion.claim_text,
                    "source_unit_id": unit.unit_id,
                    "source_fragment_id": assertion.source_fragment_id,
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
    assert all(material["text"] == "" for material in materials.values())
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
    return SimpleNamespace(
        payload=payload,
        body=body,
        summaries=summaries,
        source_texts=sorted(originals.values()),
        config=EvalConfig().model_copy(
            update={
                "strict_max_claims": 1024,
                "strict_max_batches": 64,
                "strict_batch_size": 16,
            }
        ),
    )


def test_initial_assertion_capacity_does_not_bound_complete_assessment_batches(
    monkeypatch,
):
    """Preserve positions, full sources and permissions while reproducing 67/9."""

    def deny_network(*args, **kwargs):
        raise AssertionError("The pure batch regression must not use a transport")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket, "create_connection", deny_network)
    case = _initial_batch_case()
    payload = case.payload
    before = copy.deepcopy(payload)
    envelope = payload["writer_envelope"]
    assert len(payload["assertions"]) == 67 < envelope["model_assertions"] == 128
    assert envelope["assessment_batches"] == 8
    assert nested_json_size(case.body) <= envelope["body_json_bytes"]
    assert nested_json_size(case.summaries) <= envelope["summary_json_bytes"]
    assert (
        nested_json_size([claim["claim_text"] for claim in payload["assertions"]])
        <= envelope["claim_json_bytes"]
    )
    assert len(batched._response_permission_groups(payload, None)) == 65
    batches = batched.plan_batches(payload, case.config)
    assert [len(batch["assertions"]) for batch in batches] == [8] * 8 + [3]
    assert len(batches) == 9 > envelope["assessment_batches"]
    actual = [claim for batch in batches for claim in batch["assertions"]]
    assert Counter(claim["claim_id"] for claim in actual) == Counter(
        claim["claim_id"] for claim in payload["assertions"]
    )
    assert {claim["claim_id"]: claim for claim in actual} == {
        claim["claim_id"]: claim for claim in payload["assertions"]
    }
    # Summaries repeat body wording at independently bound source positions.
    repeated = {}
    for claim in actual:
        repeated.setdefault(claim["claim_text"], set()).add(claim["source_fragment_id"])
    assert sum(len(locations) > 1 for locations in repeated.values()) == 2
    assert (
        sorted(source["text"] for source in payload["original_supports"].values())
        == case.source_texts
    )
    sizes = []
    reached_sources = {}
    for batch in batches:
        messages, native = scoped.build_evaluation_request(
            batched.batch_prompt(batch), batch
        )
        size = strict_request_reservation(messages, case.config.max_tokens, native)
        assert size == batched.request_size(
            batched.batch_prompt(batch), batch, case.config.max_tokens
        )
        assert size <= envelope["request_caps"]["assessment"] == 100000
        sizes.append(size)
        wire = json.loads(messages[1]["content"])
        assert wire["original_supports"] == batch["original_supports"]
        for ref, source in batch["original_supports"].items():
            assert source == payload["original_supports"][ref]
            reached_sources[ref] = source
        schema = native["json_schema"]["schema"]
        claims_by_id = {claim["claim_id"]: claim for claim in batch["assertions"]}
        seen = []
        for reference in schema["properties"]["decisions"]["items"]["anyOf"]:
            branch = schema["$defs"][reference["$ref"].rsplit("/", 1)[-1]]
            properties = branch["properties"]
            for claim_id in properties["claim_id"]["enum"]:
                cited = {
                    item["evidence_alias"]
                    for item in claims_by_id[claim_id]["evidence"]
                }
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
                seen.append(claim_id)
        assert Counter(seen) == Counter(claims_by_id.keys())
    assert reached_sources == payload["original_supports"]
    assert payload == before
    print(
        json.dumps(
            {
                "claims": len(actual),
                "initial_batches": envelope["assessment_batches"],
                "actual_batches": len(batches),
                "reservations": sizes,
                "max_reservation": max(sizes),
                "sum_reservations": sum(sizes),
                "scientific_truth_verified": False,
            },
            sort_keys=True,
        )
    )
