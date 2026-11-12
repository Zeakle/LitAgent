"""Exercise authenticated input codecs without asserting scientific correctness."""

import copy
import json

import pytest
from test_structured_repair_guidance import (
    context_for,
    guidance_case,
    prior_context,
    rebind,
    request_payload,
    selected_scope,
)
from test_support_receipt_identity import selected_envelope

import litagent.writing_flat as flat
from litagent.capacity_repair import capacity_targets
from litagent.quality import digest
from litagent.safety.planned_budget import nested_json_size
from litagent.writing_v2 import WriterError, body_units


def codec_input(case, current=None, prior=None, old=None):
    """Build transport views while leaving all authorization to real consumers."""
    packet = case[2]
    current = current or context_for(case)
    failure = (
        {"candidate_sha256": prior["host_directory"]["source_report_sha256"]}
        if prior is not None
        else None
    )
    payload = request_payload(case, failure)
    payload["cards"] = {
        ref: {k: card[k] for k in ("source_ref", "evidence_id", "semantic_status")}
        for ref, card in packet["cards"].items()
    }
    for target in payload["targets"].values():
        target.pop("original")
    authority = {
        "guidance_context": {"current": current, "prior_candidate": prior},
        "feedback_context": old,
        "packet": copy.deepcopy(packet),
        "summary_body": None,
    }
    canonical = {
        "contract": flat.STRUCTURED_REPAIR_GUIDANCE_CONTRACT,
        "current": flat.project_structured_repair_guidance(
            current["diagnostics"],
            host_directory=current["host_directory"],
            packet=packet,
            current_body=case[0],
            targets={"r1": case[1]["r1"]},
        ),
        "prior_candidate": (
            flat.project_structured_repair_guidance(
                prior["diagnostics"],
                host_directory=prior["host_directory"],
                packet=packet,
                current_body=case[0],
                targets={"r1": case[1]["r1"]},
            )
            if prior is not None
            else None
        ),
    }
    return canonical, payload, authority


def encode(case, current=None, prior=None, old=None):
    """Use the public encoder and its independent inverse before returning."""
    canonical, payload, authority = codec_input(case, current, prior, old)
    wire, receipt = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    return canonical, payload, authority, wire, receipt


def test_complete_scientific_issue_sources_and_local_records_are_unchanged(
    guidance_case,
):
    args = codec_input(guidance_case)
    before = copy.deepcopy(args)
    wire, receipt = flat.encode_repair_guidance_wire(
        args[0], payload=args[1], authority=args[2]
    )
    restored = flat.restore_repair_guidance_wire(
        wire, payload=args[1], authority=args[2], receipt=receipt
    )
    assert restored == args[0]
    assert wire["current"]["units"] == [["b", ["introduction", 0, 0]]]
    issue = restored["current"]["issues"][0]
    assert issue["status"] == "unsupported"
    assert issue["aspect_checks"][0]["supported"] is False
    assert issue["support_validation"]["required_aspects"] == [
        "predicate",
        "conditions",
    ]
    assert args[1]["sources"] == guidance_case[2]["sources"]
    assert "LOCAL-ONLY-" not in json.dumps(wire)
    assert args == before


def test_packet_question_cannot_be_self_rehashed_into_new_wire_receipt(guidance_case):
    """A pure codec must not trust a changed view even if real builders are safe."""
    case = copy.deepcopy(guidance_case)
    case[2]["query"] = "Compare Method A under condition X."
    canonical, payload, authority = codec_input(case)
    wire, receipt = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    assert (
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )
        == canonical
    )
    payload["query"] = "A different unauthorized research question."
    # Re-run the actual encoder, not a fabricated hash shortcut. Before the
    # guard, this issued a fresh self-consistent wire/receipt for the wrong view.
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.encode_repair_guidance_wire(
            canonical, payload=payload, authority=authority
        )


@pytest.mark.parametrize(
    "field,original,changed",
    [
        ("language", "zh-CN", "en-US"),
        ("requirements", {"core": [0]}, {"core": [False]}),
        ("section_duties", {"introduction": "Context"}, {"introduction": "Invent"}),
    ],
)
def test_original_packet_fields_are_independently_bound(
    guidance_case, field, original, changed
):
    case = copy.deepcopy(guidance_case)
    case[2][field] = original
    canonical, payload, authority = codec_input(case)
    wire, receipt = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    payload[field] = changed
    receipt["wire_sha256"] = digest(wire)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


@pytest.mark.parametrize("unit_id", ["unit-1", "writer:introduction:0:0"])
def test_default_reviewer_retains_original_unit_identifier(guidance_case, unit_id):
    wrapper = context_for(
        guidance_case,
        [{"unit_id": unit_id, "explanation": "Concern only, not a verdict."}],
        category="reviewer",
    )
    canonical, payload, authority, wire, receipt = encode(guidance_case, wrapper)
    assert wire["current"]["issues"] == [[0, ["f1"], ["r1"]]]
    restored = flat.restore_repair_guidance_wire(
        wire, payload=payload, authority=authority, receipt=receipt
    )
    assert restored["current"]["issues"][0]["unit_id"] == unit_id
    assert restored == canonical


def test_native_uid_reviewer_variant_is_not_renamed(guidance_case):
    case = copy.deepcopy(guidance_case)
    fid = "rendered:native:native-uid"
    case = (*case[:3], fid, *case[4:])
    wrapper = context_for(
        case,
        [{"unit_id": "native-uid", "explanation": "Review this exact fragment."}],
        category="reviewer",
    )
    canonical, payload, authority, wire, receipt = encode(case, wrapper)
    assert wire["current"]["issues"][0][0] == 0
    assert (
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )
        == canonical
    )
    assert canonical["current"]["issues"][0]["unit_id"] == "native-uid"


def test_nondefault_reviewer_retains_complete_literal(guidance_case):
    wrapper = context_for(
        guidance_case,
        [{"unit_id": "unit-1", "explanation": "Not a scientific verdict."}],
        category="reviewer",
    )
    wrapper["host_directory"]["issues"][0]["body_paths"] = [["introduction", 0, 0]]
    rebind(wrapper)
    canonical, _, _, wire, _ = encode(guidance_case, wrapper)
    assert wire["current"]["issues"] == [[1, canonical["current"]["issues"][0]]]


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_root",
        "unknown_origin",
        "unknown_header",
        "unknown_literal_issue",
        "unknown_target",
        "unknown_fragment",
        "wrong_unit_view",
        "extra_unit_field",
        "bool_path",
        "float_path",
        "negative_path",
        "bool_tag",
        "float_tag",
        "missing_issue",
        "duplicate_issue",
        "unknown_catalog_field",
        "wrong_context",
    ],
)
def test_closed_wire_rejects_mutations_even_with_rehashed_receipt(
    guidance_case, mutation
):
    _, payload, authority, wire, receipt = encode(guidance_case)
    origin = wire["current"]
    issue = origin["issues"][0][1]
    if mutation == "unknown_root":
        wire["future"] = "cannot drop"
    elif mutation == "unknown_origin":
        origin["future"] = None
    elif mutation == "unknown_header":
        origin["header"]["future"] = False
    elif mutation == "unknown_literal_issue":
        issue["future"] = "cannot filter"
    elif mutation == "unknown_target":
        issue["target_ids"] = ["r999999"]
    elif mutation == "unknown_fragment":
        issue["fragment_refs"] = ["f999999"]
    elif mutation == "wrong_unit_view":
        origin["units"][0][0] = "prior_candidate"
    elif mutation == "extra_unit_field":
        origin["units"][0].append("cannot drop")
    elif mutation in {"bool_path", "float_path", "negative_path"}:
        origin["units"][0][1][1] = {
            "bool_path": False,
            "float_path": 0.0,
            "negative_path": -1,
        }[mutation]
    elif mutation in {"bool_tag", "float_tag"}:
        origin["issues"][0][0] = True if mutation == "bool_tag" else 1.0
    elif mutation == "missing_issue":
        origin["issues"] = []
    elif mutation == "duplicate_issue":
        origin["issues"] *= 2
    elif mutation == "unknown_catalog_field":
        receipt["catalogs"]["current"][0][2]["future"] = "cannot drop"
        origin["host_catalog_sha256"] = digest(receipt["catalogs"]["current"])
    elif mutation == "wrong_context":
        origin["header"]["context_identity"] = "f" * 64
    receipt["wire_sha256"] = digest(wire)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


def two_inputs(case):
    """Keep equal prose at two distinct positions with disjoint authority."""
    current = context_for(case)
    d = current["host_directory"]
    fid = "unit-2:native:writer:taxonomy:0:0"
    fragment = copy.deepcopy(case[4])
    fragment.update(
        source_unit_id="unit-2",
        native_paths=[["taxonomy", 0, 0]],
        source_start=100,
        source_end=100 + len(fragment["text"]),
        current_target_ids=["r2"],
    )
    d["fragments"][fid] = fragment
    raw = copy.deepcopy(case[-1])
    raw.update(
        claim_id="claim-2",
        statement_id="claim-2",
        source_unit_id="unit-2",
        source_fragment_id=fid,
        source_start=fragment["source_start"],
        source_end=fragment["source_end"],
        section="Taxonomy",
    )
    current["diagnostics"].append(raw)
    d["issues"].append(
        {
            "diagnostic_index": 1,
            "diagnostic_sha256": digest(raw),
            "category": "evaluation",
            "fragment_ids": [fid],
            "target_ids": ["r2"],
            "body_paths": [],
        }
    )
    rebind(current)
    canonical, payload, authority = codec_input(case)
    authority["guidance_context"]["current"] = current
    target = copy.deepcopy(case[1]["r2"])
    target.pop("original")
    payload["targets"]["r2"] = target
    canonical["current"] = flat.project_structured_repair_guidance(
        current["diagnostics"],
        host_directory=d,
        packet=case[2],
        current_body=case[0],
        targets={key: case[1][key] for key in ("r1", "r2")},
    )
    return canonical, payload, authority


@pytest.mark.parametrize(
    "mutation",
    ["reorder_both", "other_issue_target", "target_order", "authority_order"],
)
def test_ordinal_and_per_issue_authority_cannot_be_rehashed(guidance_case, mutation):
    canonical, payload, authority = two_inputs(guidance_case)
    wire, receipt = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    assert len(wire["current"]["units"]) == 2
    if mutation == "reorder_both":
        wire["current"]["units"].reverse()
        receipt["catalogs"]["current"].reverse()
        wire["current"]["host_catalog_sha256"] = digest(receipt["catalogs"]["current"])
    elif mutation == "other_issue_target":
        wire["current"]["issues"][0][1]["target_ids"] = ["r2"]
    elif mutation == "target_order":
        payload["targets"] = dict(reversed(list(payload["targets"].items())))
    else:
        directory = authority["guidance_context"]["current"]["host_directory"]
        directory["fragments"] = dict(reversed(list(directory["fragments"].items())))
        rebind(authority["guidance_context"]["current"])
    receipt["wire_sha256"] = digest(wire)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


def test_prior_text_never_borrows_same_current_path(guidance_case):
    prior, old = prior_context(guidance_case)
    canonical, payload, authority, wire, receipt = encode(
        guidance_case, prior=prior, old=old
    )
    row = wire["prior_candidate"]["units"][0]
    assert row[0] == "l"
    assert row[2] == guidance_case[-1]["claim_text"]
    assert row[2] + " [E:E1]" == canonical["prior_candidate"]["fragments"]["f1"]["text"]
    wire["prior_candidate"]["units"][0] = ["b", ["introduction", 0, 0]]
    receipt["wire_sha256"] = digest(wire)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


def prose_case(case, prose, refs):
    """Construct native text and host-rendered fragments, not scientific labels."""
    body, _, packet, fid, fragment, raw = copy.deepcopy(case)
    packet["cards"]["c2"] = {
        **copy.deepcopy(packet["cards"]["c1"]),
        "evidence_id": "E2",
    }
    evidence = list(dict.fromkeys(packet["cards"][ref]["evidence_id"] for ref in refs))
    rendered = prose + (
        " " + " ".join(f"[E:{eid}]" for eid in evidence) if evidence else ""
    )
    for section in body:
        if section == "title":
            continue
        body[section][0]["sentences"][0] = {"text": prose, "fact_card_refs": list(refs)}
    fragment.update(
        text=rendered,
        source_end=fragment["source_start"] + len(rendered),
        evidence_ids=evidence,
    )
    raw.update(
        claim_text=prose,
        source_quote=rendered,
        source_end=fragment["source_end"],
        evidence_ids=evidence,
    )
    if not refs:
        raw.update(support_basis=[], aspect_checks=[])
    targets = capacity_targets(body, {"reason": "capacity"}, maximum=66)
    return body, targets, packet, fid, fragment, raw


@pytest.mark.parametrize(
    "prose,refs",
    [
        ("Method A works only under condition X. ", ["c1"]),
        ('方法 A 仅在条件 X 下成立，"限定" 与 \\ 路径均保留。', ["c1", "c2"]),
        ("Two ordered sources preserve their conditions.", ["c2", "c1"]),
        ("An unsupported uncited statement remains unsupported.", []),
        ("Text [1] and a trailing tab are untouched.\t", ["c1"]),
    ],
)
def test_native_literal_algebra_preserves_every_character_and_reference(
    guidance_case, prose, refs
):
    case = prose_case(guidance_case, prose, refs)
    prior, old = prior_context(case)
    canonical, payload, authority, wire, receipt = encode(case, prior=prior, old=old)
    before = copy.deepcopy((canonical, payload, authority))
    row = wire["prior_candidate"]["units"][0]
    assert row[0] == "l" and row[2] == prose
    assert row[3] == list(dict.fromkeys(refs))
    evidence = list(dict.fromkeys(case[2]["cards"][r]["evidence_id"] for r in row[3]))
    rendered = prose + (
        " " + " ".join(f"[E:{eid}]" for eid in evidence) if evidence else ""
    )
    assert rendered == canonical["prior_candidate"]["fragments"]["f1"]["text"]
    assert (
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )
        == canonical
    )
    assert (canonical, payload, authority) == before


@pytest.mark.parametrize(
    "rendered",
    [
        "The expected host suffix is absent.",
        "A wrong host suffix [E:E2]",
        "A suffix followed by extra text [E:E1] extra",
        "A repeated suffix [E:E1] [E:E1]",
        "Do not erase an embedded [E:unowned] marker [E:E1]",
        "   [E:E1]",
    ],
)
def test_literal_suffix_must_have_unique_exact_native_reconstruction(
    guidance_case, rendered
):
    prior, old = prior_context(guidance_case)
    fragment = next(iter(prior["host_directory"]["fragments"].values()))
    fragment.update(text=rendered, source_end=fragment["source_start"] + len(rendered))
    prior["diagnostics"][0].update(
        source_quote=rendered, source_end=fragment["source_end"]
    )
    rebind(prior)
    old["diagnostics_sha256"] = prior["host_directory"]["diagnostics_sha256"]
    old["identity"] = digest({k: v for k, v in old.items() if k != "identity"})
    canonical, payload, authority = codec_input(guidance_case, prior=prior, old=old)
    before = copy.deepcopy((canonical, payload, authority))
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.encode_repair_guidance_wire(
            canonical, payload=payload, authority=authority
        )
    assert (canonical, payload, authority) == before


@pytest.mark.parametrize("change", ["prose", "refs", "premises", "prior_path"])
def test_native_literal_tampering_cannot_be_self_rehashed(guidance_case, change):
    prior, old = prior_context(guidance_case)
    _, payload, authority, wire, receipt = encode(guidance_case, prior=prior, old=old)
    row = wire["prior_candidate"]["units"][0]
    if change == "prose":
        row[2] += " unrelated finding"
    elif change == "refs":
        row[3] = ["c999999"]
    elif change == "premises":
        row[4] = [["taxonomy", 0, 0]]
    else:
        row[1] = ["taxonomy", 0, 0]
    receipt["wire_sha256"] = digest(wire)
    receipt["ordered_wire_sha256"] = flat._guidance_wire_order_hash(wire)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


@pytest.mark.parametrize("consumer", ["encode", "restore"])
@pytest.mark.parametrize(
    "missing",
    [
        "current_body",
        "sources",
        "cards",
        "target_path",
        "packet_sources",
        "context_directory",
    ],
)
def test_public_codec_missing_fields_have_typed_failures(
    guidance_case, consumer, missing
):
    canonical, payload, authority, wire, receipt = encode(guidance_case)
    if missing in {"current_body", "sources", "cards"}:
        payload.pop(missing)
    elif missing == "target_path":
        payload["targets"]["r1"].pop("path")
    elif missing == "packet_sources":
        authority["packet"].pop("sources")
    else:
        authority["guidance_context"]["current"].pop("host_directory")
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        if consumer == "encode":
            flat.encode_repair_guidance_wire(
                canonical, payload=payload, authority=authority
            )
        else:
            flat.restore_repair_guidance_wire(
                wire, payload=payload, authority=authority, receipt=receipt
            )


def test_reportless_local_issue_remains_reportless(guidance_case):
    wrapper = context_for(
        guidance_case,
        [{"code": "writer_local_contract_issue", "actual": 67, "maximum": 66}],
        category="local",
        reportless=True,
    )
    canonical, payload, authority, wire, receipt = encode(guidance_case, wrapper)
    assert wire["current"]["units"] == []
    assert (
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )
        == canonical
    )
    assert canonical["current"]["source_report_sha256"] is None


def test_capacity_count_vector_preserves_full_native_directory(guidance_case):
    raw = {
        "code": "capacity_exceeded",
        "actual_assertions": 176,
        "maximum": 128,
        "assertions_by_fragment": {guidance_case[3]: 176},
    }
    wrapper = context_for(guidance_case, [raw], category="capacity")
    directory = wrapper["host_directory"]
    directory["fragments"] = {guidance_case[3]: copy.deepcopy(guidance_case[4])}
    directory["issues"][0].update(fragment_ids=[guidance_case[3]], body_paths=[])
    rebind(wrapper)
    canonical, payload, authority, wire, receipt = encode(guidance_case, wrapper)
    counts = wire["current"]["issues"][0][1]["host_fields"]["assertions_by_fragment"]
    assert counts == {"catalog_count_values": [176]}
    assert (
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )
        == canonical
    )
    counts["catalog_count_values"][0] = True
    receipt["wire_sha256"] = digest(wire)
    receipt["ordered_wire_sha256"] = flat._guidance_wire_order_hash(wire)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


@pytest.mark.parametrize(
    "field", ["sources", "cards", "summary_body", "extra_authority", "extra_receipt"]
)
def test_full_authority_cannot_be_replaced_by_a_self_rehashed_receipt(
    guidance_case, field
):
    _, payload, authority, wire, receipt = encode(guidance_case)
    if field == "sources":
        payload["sources"]["s1"]["source_quote"] += " fabricated source"
    elif field == "cards":
        payload["cards"]["c1"]["evidence_id"] = "E-foreign"
    elif field == "summary_body":
        authority["summary_body"] = copy.deepcopy(guidance_case[0])
    elif field == "extra_authority":
        authority["future"] = False
    else:
        receipt["future"] = False
    receipt["wire_sha256"] = digest(wire)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


def test_summary_view_is_bound_to_independent_full_body(guidance_case):
    case = copy.deepcopy(guidance_case)
    source_body = case[0]
    summary = {
        "abstract": [
            {"sentences": [{"text": case[-1]["claim_text"], "body_unit_refs": ["u1"]}]}
        ]
    }
    wrapper = context_for(case)
    d = wrapper["host_directory"]
    fragment = next(iter(d["fragments"].values()))
    fragment["native_paths"] = [["abstract", 0, 0]]
    fragment["body_premise_paths"] = [["introduction", 0, 0]]
    d["current_body_sha256"] = d["source_body_sha256"] = digest(summary)
    rebind(wrapper)
    target = {
        "path": ["abstract", 0, 0],
        "original": summary["abstract"][0]["sentences"][0],
        "original_sha256": digest(summary["abstract"][0]["sentences"][0]),
        "maximum": 1,
    }
    canonical, payload, authority = codec_input(case)
    authority["guidance_context"]["current"] = wrapper
    authority["summary_body"] = source_body
    payload.update(
        current_body=summary,
        body_sha256=digest(summary),
        targets={"r1": {k: v for k, v in target.items() if k != "original"}},
        body_units=body_units(source_body),
        summary_source_body_sha256=digest(source_body),
    )
    canonical["current"] = flat.project_structured_repair_guidance(
        wrapper["diagnostics"],
        host_directory=d,
        packet=case[2],
        current_body=summary,
        targets={"r1": target},
    )
    wire, receipt = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    assert (
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )
        == canonical
    )
    payload["body_units"]["u1"]["text"] += " a forged premise"
    receipt["views_sha256"] = digest(payload["body_units"])
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )


def test_exact_encoded_cap_and_one_byte_over(guidance_case):
    canonical, payload, authority = codec_input(guidance_case)
    wire, _ = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    extra = 12288 - nested_json_size(wire)
    wrapper = authority["guidance_context"]["current"]
    wrapper["diagnostics"][0]["claim_text"] += "x" * extra
    rebind(wrapper)
    canonical["current"] = flat.project_structured_repair_guidance(
        wrapper["diagnostics"],
        host_directory=wrapper["host_directory"],
        packet=guidance_case[2],
        current_body=guidance_case[0],
        targets={"r1": guidance_case[1]["r1"]},
    )
    at_cap, _ = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    assert nested_json_size(at_cap) == 12288
    wrapper["diagnostics"][0]["claim_text"] += "x"
    rebind(wrapper)
    canonical["current"] = flat.project_structured_repair_guidance(
        wrapper["diagnostics"],
        host_directory=wrapper["host_directory"],
        packet=guidance_case[2],
        current_body=guidance_case[0],
        targets={"r1": guidance_case[1]["r1"]},
    )
    with pytest.raises(
        WriterError, match="writer_guidance_unrepairable_capacity"
    ) as error:
        flat.encode_repair_guidance_wire(
            canonical, payload=payload, authority=authority
        )
    assert error.value.diagnostics[0]["actual"] == 12289


def test_new_wire_identity_is_separate_from_existing_semantic_and_output_ids(
    monkeypatch,
):
    before = [
        flat.structured_repair_guidance_identity(),
        flat.source_map_wire_identity(),
        flat.transport_identity(),
    ]
    identity = flat.repair_guidance_wire_identity()
    monkeypatch.setattr(
        flat,
        "REPAIR_GUIDANCE_WIRE_INSTRUCTION",
        flat.REPAIR_GUIDANCE_WIRE_INSTRUCTION + " changed",
    )
    assert flat.repair_guidance_wire_identity() != identity
    assert before == [
        flat.structured_repair_guidance_identity(),
        flat.source_map_wire_identity(),
        flat.transport_identity(),
    ]


def test_none_rejects_complete_canonical_overflow_without_hidden_defer(guidance_case):
    raw = {"unit_id": "writer:introduction:0:0", "explanation": "Not gold."}
    wrapper = context_for(
        guidance_case, [copy.deepcopy(raw) for _ in range(150)], category="reviewer"
    )
    before = copy.deepcopy(wrapper)
    with pytest.raises(WriterError, match="writer_guidance_unrepairable_capacity"):
        flat.repair_projection(
            request_payload(guidance_case),
            selected_scope(),
            packet=guidance_case[2],
            guidance_context={"current": wrapper, "prior_candidate": None},
        )
    assert wrapper == before


def test_only_explicit_new_pair_defers_cap_then_final_encoder_checks_all_issues(
    guidance_case,
):
    raw = {
        "unit_id": "writer:introduction:0:0",
        "explanation": "Independent concern only.",
    }
    wrapper = context_for(
        guidance_case, [copy.deepcopy(raw) for _ in range(150)], category="reviewer"
    )
    scope = selected_envelope(
        repair_mapping_contract="capacity-source-map-v3",
        repair_guidance_contract=flat.STRUCTURED_REPAIR_GUIDANCE_CONTRACT,
        repair_guidance_wire_revision=flat.REPAIR_GUIDANCE_WIRE_REVISION,
    )
    payload = flat.repair_projection(
        request_payload(guidance_case),
        scope,
        packet=guidance_case[2],
        guidance_context={"current": wrapper, "prior_candidate": None},
    )
    canonical = payload["repair_guidance"]
    assert nested_json_size(canonical) > 12288
    authority = {
        "guidance_context": {"current": wrapper, "prior_candidate": None},
        "feedback_context": None,
        "packet": guidance_case[2],
        "summary_body": None,
    }
    wire, receipt = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    assert len(wire["current"]["issues"]) == 150
    assert (
        flat.restore_repair_guidance_wire(
            wire, payload=payload, authority=authority, receipt=receipt
        )
        == canonical
    )


@pytest.mark.parametrize(
    "field", ["repair_guidance_wire_revision", "repair_guidance_wire_identity"]
)
def test_partial_opt_in_marker_never_defers_old_cap(guidance_case, field):
    wrapper = context_for(guidance_case)
    scope = selected_scope()
    scope[field] = (
        flat.REPAIR_GUIDANCE_WIRE_REVISION
        if field.endswith("revision")
        else flat.repair_guidance_wire_identity()
    )
    with pytest.raises((WriterError, ValueError)):
        flat.repair_projection(
            request_payload(guidance_case),
            scope,
            packet=guidance_case[2],
            guidance_context={"current": wrapper, "prior_candidate": None},
        )
