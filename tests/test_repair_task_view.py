"""Check complete authenticated repair tasks without asserting scientific gold."""

import base64
import copy
import json

import pytest
from test_guidance_wire_contract import guidance_case  # noqa: F401
from test_guidance_wire_contract import (
    codec_input,
    context_for,
    prior_context,
    rebind,
    selected_scope,
    two_inputs,
)

import litagent.writing_flat as flat
from litagent.capacity_repair import capacity_targets
from litagent.quality import digest
from litagent.safety.planned_budget import nested_json_size
from litagent.writing_v2 import WriterError, body_units

ADMISSION = "complete-locked-request-reservation-v1"


def select_task_view(payload):
    """Select the explicit codec through a complete host envelope."""
    envelope = selected_scope()
    envelope.update(
        repair_guidance_wire_revision=flat.REPAIR_TASK_VIEW_REVISION,
        repair_guidance_wire_identity=flat.repair_guidance_wire_identity(
            flat.REPAIR_TASK_VIEW_REVISION
        ),
        repair_guidance_capacity_rule=ADMISSION,
    )
    envelope["identity"] = digest(
        {key: value for key, value in envelope.items() if key != "identity"}
    )
    payload["writing_envelope"] = envelope


def task_input(case, current=None, prior=None, old=None):
    """Keep complete canonical authority beside the selected model view."""
    canonical, payload, authority = codec_input(case, current, prior, old)
    select_task_view(payload)
    return canonical, payload, authority


def encode(args):
    """Call the public encoder with the real source and target witnesses."""
    return flat.encode_repair_guidance_wire(args[0], payload=args[1], authority=args[2])


def restore(args, wire, receipt):
    """Require independent host replay rather than a model-view-only inverse."""
    return flat.restore_repair_guidance_wire(
        wire, payload=args[1], authority=args[2], receipt=receipt
    )


def issue_rows(origin):
    """Read the public field table without relying on a column ordering."""
    output = []
    common = origin.get("issue_shared", {})
    for row in origin["issues"]:
        assert type(row[0]) is int
        fieldset = origin["issue_columns"][row[0]]
        assert set(fieldset) == {"fields", "shared"}
        columns = fieldset["fields"]
        assert len(columns) == len(set(columns))
        assert not set(columns) & fieldset["shared"].keys()
        assert not set(common) & (set(columns) | fieldset["shared"].keys())
        assert len(row) == len(columns) + 1
        values = {
            **copy.deepcopy(common),
            **copy.deepcopy(fieldset["shared"]),
            **dict(zip(columns, row[1:], strict=True)),
        }
        issue = {}
        for key, value in values.items():
            namespace, separator, child = key.partition(".")
            if separator and namespace in {
                "support_validation",
                "assessment_conflict_effect",
            }:
                issue.setdefault(namespace, {})[child] = value
            else:
                issue[key] = value
        output.append(issue)
    return output


def decoded_issue_rows(wire, origin_name):
    """Independently read the explicit template codec without a host-side inverse."""
    assert origin_name in {"current", "prior_candidate"}
    origin = wire[origin_name]
    markers = {"issue_templates", "aspect_columns", "effect_digest_wire"}
    if not markers.intersection(wire):
        return None if origin is None else issue_rows(origin)
    assert markers <= set(wire)
    assert wire["aspect_columns"] == ["aspect", "supported", "support_refs"]
    fields = [
        "assessment_conflict_effect.decision_sha256",
        "assessment_conflict_effect.binding_sha256",
    ]
    assert wire["effect_digest_wire"] == {
        "encoding": "python-base64-b85encode-32bytes-v1",
        "fields": fields,
        "decoded_format": "64-lowercase-hex",
    }
    templates = issue_rows(wire["issue_templates"])
    if origin is None:
        return None
    output = []
    for occurrence in issue_rows(origin):
        index = occurrence.pop("issue_template_index")
        assert type(index) is int and 0 <= index < len(templates)
        value = copy.deepcopy(templates[index])
        effect = value.get("assessment_conflict_effect")
        assert set(occurrence) == ({"assessment_conflict_effect"} if effect else set())
        if effect is not None:
            assert not {"decision_sha256", "binding_sha256"}.intersection(effect)
            digests = occurrence["assessment_conflict_effect"]
            assert set(digests) == {field.split(".")[1] for field in fields}
            for field, encoded in digests.items():
                assert type(encoded) is str and len(encoded) == 40 and encoded.isascii()
                raw = base64.b85decode(encoded.encode("ascii"))
                assert len(raw) == 32
                assert base64.b85encode(raw).decode("ascii") == encoded
                effect[field] = raw.hex()
        if "aspect_checks" in value:
            checks = []
            for row in value["aspect_checks"]:
                assert type(row) is list and len(row) == 3
                name, supported, refs = row
                assert type(name) is str and type(supported) is bool
                assert type(refs) is list and all(type(ref) is str for ref in refs)
                checks.append(dict(zip(wire["aspect_columns"], row, strict=True)))
            value["aspect_checks"] = checks
        output.append(value)
    return output


def replace_value(origin, index, field, value):
    """Tamper one named model field independently of its current column set."""
    row = origin["issues"][index]
    fieldset = origin["issue_columns"][row[0]]
    if field in {"support_validation", "assessment_conflict_effect"}:
        for key, child in value.items():
            replace_value(origin, index, field + "." + key, child)
    elif field in origin.get("issue_shared", {}):
        origin["issue_shared"][field] = value
    elif field in fieldset["shared"]:
        fieldset["shared"][field] = value
    else:
        row[fieldset["fields"].index(field) + 1] = value


def rehash_view(wire, receipt):
    """Show that derived receipt hashes cannot grant new edit authority."""
    receipt["wire_sha256"] = digest(wire)
    receipt["ordered_wire_sha256"] = flat._guidance_wire_order_hash(wire)


def rebind_prior(prior, old):
    """Refresh synthetic host hashes without bypassing any public codec checks."""
    rebind(prior)
    directory = prior["host_directory"]
    for previous, current in (
        ("diagnostics_sha256", "diagnostics_sha256"),
        ("fragments", "fragments"),
        ("evidence_aliases", "evidence_aliases"),
    ):
        old[previous] = copy.deepcopy(directory[current])
    old["identity"] = digest(
        {key: value for key, value in old.items() if key != "identity"}
    )


def test_complete_failure_tasks_round_trip_without_mutating_authority(guidance_case):
    prior, old = prior_context(guidance_case)
    args = task_input(guidance_case, prior=prior, old=old)
    before = copy.deepcopy(args)
    wire, receipt = encode(args)
    assert set(wire) == {
        "contract",
        "wire_revision",
        "admission_rule",
        "current",
        "prior_candidate",
    }
    assert wire["contract"] == flat.STRUCTURED_REPAIR_GUIDANCE_CONTRACT
    assert wire["wire_revision"] == flat.REPAIR_TASK_VIEW_REVISION
    assert wire["admission_rule"] == ADMISSION
    for origin in ("current", "prior_candidate"):
        content = wire[origin]
        assert set(content) == {
            "source_report_sha256",
            "support_refs",
            "units",
            "issue_columns",
            "issues",
        }
        assert (
            content["source_report_sha256"] == args[0][origin]["source_report_sha256"]
        )
        assert content["support_refs"] == args[0][origin]["support_refs"]
        rows = issue_rows(content)
        assert len(rows) == args[0][origin]["coverage"]["diagnostics_total"] == 1
        expected = args[0][origin]["issues"][0]
        for field in (
            "category",
            "claim_id",
            "claim_text",
            "kind",
            "status",
            "reason_code",
            "local_error",
            "fragment_refs",
            "target_ids",
            "body_paths",
            "support_basis_refs",
            "support_validation",
        ):
            assert rows[0][field] == expected[field]
    assert wire["current"]["units"] == [["b", ["introduction", 0, 0]]]
    assert wire["prior_candidate"]["units"] == [
        ["p", ["introduction", 0, 0], ["c1"], []]
    ]
    assert args[1]["current_body"] == guidance_case[0]
    assert args[1]["sources"] == guidance_case[2]["sources"]
    assert "LOCAL-ONLY-" not in json.dumps(wire)
    assert "LOCAL-CANDIDATE-ONLY" not in json.dumps(wire)
    assert restore(args, wire, receipt) == args[0]
    assert args == before


def test_all_failed_aspects_and_validation_parameters_remain_visible(guidance_case):
    raw = copy.deepcopy(guidance_case[-1])
    raw["aspect_checks"] = [
        {
            "aspect": aspect,
            "supported": supported,
            "support_ids": ["e1:full_original"],
            "explanation": "Local audit explanation.",
        }
        for aspect, supported in (
            ("predicate", True),
            ("conditions", False),
            ("quantities", False),
            ("causal_relation", True),
        )
    ]
    raw["support_validation"].update(
        actual=["percentage"],
        expected=["absolute_gain"],
        requested_role="improvement",
        unit="percent",
        value="10",
        context="Condition X",
    )
    args = task_input(guidance_case, context_for(guidance_case, [raw]))
    before = copy.deepcopy(args)
    wire, receipt = encode(args)
    issue = issue_rows(wire["current"])[0]
    canonical = args[0]["current"]["issues"][0]
    assert issue["aspect_checks"] == [
        item for item in canonical["aspect_checks"] if item["supported"] is False
    ]
    assert [item["aspect"] for item in issue["aspect_checks"]] == [
        "conditions",
        "quantities",
    ]
    assert issue["support_validation"] == canonical["support_validation"]
    restored = restore(args, wire, receipt)
    assert restored == args[0]
    assert len(restored["current"]["issues"][0]["aspect_checks"]) == 4
    assert args == before


def test_multiple_claims_at_one_location_are_not_grouped_or_reordered(guidance_case):
    first = copy.deepcopy(guidance_case[-1])
    second = copy.deepcopy(first)
    second.update(
        claim_id="claim-2",
        statement_id="claim-2",
        claim_text="A numeric result is 10 percent.",
        local_error="quantity_role_mismatch",
        reason_code="support_quantity_role_invalid",
    )
    args = task_input(guidance_case, context_for(guidance_case, [first, second]))
    wire, receipt = encode(args)
    rows = issue_rows(wire["current"])
    assert len(wire["current"]["units"]) == 1
    assert len(rows) == 2
    assert [row["claim_id"] for row in rows] == ["claim-1", "claim-2"]
    assert [row["claim_text"] for row in rows] == [
        first["claim_text"],
        second["claim_text"],
    ]
    assert rows[0]["fragment_refs"] == rows[1]["fragment_refs"] == ["f1"]
    assert rows[1]["local_error"] == "quantity_role_mismatch"
    assert restore(args, wire, receipt) == args[0]


def test_supported_claim_with_report_concern_is_still_a_repair_task(guidance_case):
    raw = copy.deepcopy(guidance_case[-1])
    raw.update(
        status="supported",
        diagnostic_owner="report",
        reason_code="evaluation_report_cross_section_contradiction",
    )
    args = task_input(guidance_case, context_for(guidance_case, [raw]))
    wire, receipt = encode(args)
    rows = issue_rows(wire["current"])
    assert len(rows) == 1
    assert rows[0]["status"] == "supported"
    assert rows[0]["diagnostic_owner"] == "report"
    assert rows[0]["reason_code"] == raw["reason_code"]
    assert rows[0]["claim_text"] == raw["claim_text"]
    assert restore(args, wire, receipt) == args[0]


@pytest.mark.parametrize("flag", ["atomic", "classification_valid", "source_faithful"])
def test_unsafe_prior_context_keeps_complete_literal_and_failed_flag(
    guidance_case, flag
):
    prior, old = prior_context(guidance_case)
    prior["diagnostics"][0][flag] = False
    rebind_prior(prior, old)
    canonical, payload, authority = codec_input(guidance_case, prior=prior, old=old)
    legacy, _ = flat.encode_repair_guidance_wire(
        canonical, payload=payload, authority=authority
    )
    select_task_view(payload)
    args = canonical, payload, authority
    before = copy.deepcopy(args)
    wire, receipt = encode(args)
    assert wire["prior_candidate"]["units"] == legacy["prior_candidate"]["units"]
    row = wire["prior_candidate"]["units"][0]
    assert row[0] == "l" and row[2] == guidance_case[-1]["claim_text"]
    assert issue_rows(wire["prior_candidate"])[0][flag] is False
    assert restore(args, wire, receipt) == canonical
    assert args == before


def test_one_unsafe_claim_keeps_shared_prior_fragment_literal(guidance_case):
    prior, old = prior_context(guidance_case)
    second = copy.deepcopy(prior["diagnostics"][0])
    second.update(claim_id="claim-2", statement_id="claim-2", atomic=False)
    prior["diagnostics"].append(second)
    issue = copy.deepcopy(prior["host_directory"]["issues"][0])
    issue["diagnostic_index"] = 1
    prior["host_directory"]["issues"].append(issue)
    rebind_prior(prior, old)
    args = task_input(guidance_case, prior=prior, old=old)
    wire, receipt = encode(args)
    assert len(wire["prior_candidate"]["units"]) == 1
    assert wire["prior_candidate"]["units"][0][0] == "l"
    rows = issue_rows(wire["prior_candidate"])
    assert len(rows) == 2 and rows[1]["atomic"] is False
    assert restore(args, wire, receipt) == args[0]


def test_reviewer_is_a_located_concern_without_inferred_scientific_verdict(
    guidance_case,
):
    raw = {"unit_id": "unit-1", "explanation": "Concern, not scientific gold."}
    args = task_input(
        guidance_case, context_for(guidance_case, [raw], category="reviewer")
    )
    wire, receipt = encode(args)
    row = issue_rows(wire["current"])[0]
    assert row["category"] == "reviewer"
    assert row["need_independent_review"] is True
    assert row["unit_id"] == raw["unit_id"]
    assert row["target_ids"] == ["r1"] and row["fragment_refs"] == ["f1"]
    assert not {"status", "reason_code", "actual", "expected"} & row.keys()
    assert restore(args, wire, receipt) == args[0]


@pytest.mark.parametrize("category", ["local", "capacity"])
def test_complete_host_parameters_remain_reportless_when_no_report_exists(
    guidance_case, category
):
    raw = {
        "code": "capacity_exceeded",
        "actual": 70,
        "expected": 64,
        "minimum": 1,
        "maximum": 66,
        "diagnostics": [{"field": "unit_count", "actual": 70, "expected": 64}],
    }
    args = task_input(
        guidance_case,
        context_for(guidance_case, [raw], category=category, reportless=True),
    )
    wire, receipt = encode(args)
    content = wire["current"]
    row = issue_rows(content)[0]
    assert content["source_report_sha256"] is None and content["units"] == []
    assert row["host_fields"] == args[0]["current"]["issues"][0]["host_fields"]
    assert row["host_fields"]["diagnostics"] == raw["diagnostics"]
    assert row["body_paths"] == [["introduction", 0, 0]]
    assert row["target_ids"] == ["r1"]
    assert restore(args, wire, receipt) == args[0]


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_field",
        "unknown_aspect",
        "unknown_validation",
        "missing_flag",
        "alignment",
    ],
)
def test_unknown_or_incomplete_scientific_variants_are_not_defaulted(
    guidance_case, fault
):
    raw = copy.deepcopy(guidance_case[-1])
    if fault == "unknown_field":
        raw["future_science_exemption"] = True
    elif fault == "unknown_aspect":
        raw["aspect_checks"][0]["aspect"] = "future_aspect"
    elif fault == "unknown_validation":
        raw["support_validation"]["future_validation"] = "cannot drop"
    elif fault == "missing_flag":
        raw.pop("source_faithful")
    else:
        raw["alignment_status"] = "ambiguous"
    wrapper = context_for(guidance_case, [raw])
    before = copy.deepcopy((guidance_case, wrapper))
    with pytest.raises(WriterError):
        encode(task_input(guidance_case, wrapper))
    assert (guidance_case, wrapper) == before


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_root",
        "unknown_origin",
        "unknown_column",
        "unknown_variant",
        "extra_unit_field",
        "bool_column_index",
        "float_column_index",
        "bool_native_index",
        "float_native_index",
        "bool_failed_aspect",
        "remove_issue",
        "duplicate_issue",
        "remove_unit",
        "changed_claim",
        "changed_kind",
        "changed_status",
        "changed_reason",
        "changed_local_error",
        "removed_failed_aspects",
        "changed_validation",
        "changed_fragment",
        "changed_target",
        "changed_origin_sha",
        "changed_support",
        "prior_aliases_current",
    ],
)
def test_task_wire_tampering_is_rejected_even_with_fresh_view_hashes(
    guidance_case, mutation
):
    prior, old = prior_context(guidance_case)
    args = task_input(guidance_case, prior=prior, old=old)
    wire, receipt = encode(args)
    origin = wire["current"]
    if mutation == "unknown_root":
        wire["future"] = False
    elif mutation == "unknown_origin":
        origin["future"] = False
    elif mutation == "unknown_column":
        origin["issue_columns"][0]["fields"].append("future")
        origin["issues"][0].append(True)
    elif mutation == "unknown_variant":
        replace_value(origin, 0, "category", "future_category")
    elif mutation == "extra_unit_field":
        origin["units"][0].append("cannot drop")
    elif mutation in {"bool_column_index", "float_column_index"}:
        origin["issues"][0][0] = False if mutation == "bool_column_index" else 0.0
    elif mutation in {"bool_native_index", "float_native_index"}:
        origin["units"][0][1][1] = False if mutation == "bool_native_index" else 0.0
    elif mutation == "bool_failed_aspect":
        checks = copy.deepcopy(issue_rows(origin)[0]["aspect_checks"])
        checks[0]["supported"] = 0
        replace_value(origin, 0, "aspect_checks", checks)
    elif mutation == "remove_issue":
        origin["issues"].clear()
    elif mutation == "duplicate_issue":
        origin["issues"] *= 2
    elif mutation == "remove_unit":
        origin["units"].clear()
    elif mutation == "changed_claim":
        replace_value(origin, 0, "claim_text", "A different unsupported claim.")
    elif mutation == "changed_kind":
        replace_value(origin, 0, "kind", "runtime_method")
    elif mutation == "changed_status":
        replace_value(origin, 0, "status", "supported")
    elif mutation == "changed_reason":
        replace_value(origin, 0, "reason_code", "supported")
    elif mutation == "changed_local_error":
        replace_value(origin, 0, "local_error", "invented_error")
    elif mutation == "removed_failed_aspects":
        replace_value(origin, 0, "aspect_checks", [])
    elif mutation == "changed_validation":
        validation = copy.deepcopy(issue_rows(origin)[0]["support_validation"])
        validation["required_aspects"].reverse()
        replace_value(origin, 0, "support_validation", validation)
    elif mutation == "changed_fragment":
        replace_value(origin, 0, "fragment_refs", ["f2"])
    elif mutation == "changed_target":
        replace_value(origin, 0, "target_ids", ["r2"])
    elif mutation == "changed_origin_sha":
        origin["source_report_sha256"] = "f" * 64
    elif mutation == "changed_support":
        origin["support_refs"]["s1"]["range"][0] = 1
    else:
        wire["prior_candidate"]["units"][0] = ["b", ["introduction", 0, 0]]
    rehash_view(wire, receipt)
    before = copy.deepcopy((args, wire, receipt))
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        restore(args, wire, receipt)
    assert (args, wire, receipt) == before


@pytest.mark.parametrize("mutation", ["issues", "units", "columns", "targets"])
def test_task_coordinates_and_full_authority_order_are_bound(guidance_case, mutation):
    canonical, payload, authority = two_inputs(guidance_case)
    select_task_view(payload)
    args = canonical, payload, authority
    wire, receipt = encode(args)
    if mutation == "issues":
        wire["current"]["issues"].reverse()
    elif mutation == "units":
        wire["current"]["units"].reverse()
    elif mutation == "columns":
        wire["current"]["issue_columns"][0]["fields"].reverse()
        for row in wire["current"]["issues"]:
            row[1:] = reversed(row[1:])
    else:
        payload["targets"] = dict(reversed(list(payload["targets"].items())))
    rehash_view(wire, receipt)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        restore(args, wire, receipt)


@pytest.mark.parametrize(
    "mutation",
    ["source_text", "card_identity", "current_body", "current_target", "authority"],
)
@pytest.mark.parametrize("consumer", ["encode", "restore"])
def test_model_view_never_substitutes_for_original_source_and_current_edit_authority(
    guidance_case, mutation, consumer
):
    args = task_input(guidance_case)
    wire, receipt = encode(args)
    _, payload, authority = args
    if mutation == "source_text":
        payload["sources"]["s1"]["source_quote"] += " Unauthorized source."
    elif mutation == "card_identity":
        payload["cards"]["c1"]["evidence_id"] = "E-foreign"
    elif mutation == "current_body":
        payload["current_body"]["introduction"][0]["sentences"][0][
            "text"
        ] += " Changed."
        payload["body_sha256"] = digest(payload["current_body"])
    elif mutation == "current_target":
        payload["targets"]["r1"]["path"] = ["taxonomy", 0, 0]
    else:
        authority["future_authority"] = True
    rehash_view(wire, receipt)
    before = copy.deepcopy((args, wire, receipt))
    expected = (
        "writer_guidance_identity_invalid"
        if mutation == "current_target"
        else "writer_guidance_wire_invalid"
    )
    with pytest.raises(WriterError, match=expected) as raised:
        if consumer == "encode":
            encode(args)
        else:
            restore(args, wire, receipt)
    if mutation == "current_target":
        assert raised.value.diagnostics == [
            {"reason": "fragment_target_overlap_invalid"}
        ]
    assert (args, wire, receipt) == before


def test_explicit_task_view_keeps_oversized_complete_host_failure_without_admission_claim(
    guidance_case,
):
    raw = copy.deepcopy(guidance_case[-1])
    raw["support_validation"]["context"] = "Complete host condition. " * 800
    current = context_for(guidance_case, [raw])
    canonical, payload, authority = codec_input(guidance_case, current)
    before = copy.deepcopy((canonical, payload, authority))
    with pytest.raises(WriterError, match="writer_guidance_unrepairable_capacity"):
        flat.encode_repair_guidance_wire(
            canonical, payload=payload, authority=authority
        )
    assert (canonical, payload, authority) == before
    select_task_view(payload)
    args = canonical, payload, authority
    wire, receipt = encode(args)
    assert nested_json_size(wire) > 12288
    assert wire["admission_rule"] == ADMISSION
    assert (
        issue_rows(wire["current"])[0]["support_validation"]["context"]
        == raw["support_validation"]["context"]
    )
    assert restore(args, wire, receipt) == canonical


def test_default_v1_keeps_full_prior_literal_and_exact_output_after_task_selection(
    guidance_case,
):
    prior, old = prior_context(guidance_case)
    args = codec_input(guidance_case, prior=prior, old=old)
    before = copy.deepcopy(args)
    old_wire, old_receipt = encode(args)
    assert set(old_wire) == {"contract", "wire_revision", "current", "prior_candidate"}
    assert old_wire["wire_revision"] == flat.REPAIR_GUIDANCE_WIRE_REVISION
    assert old_wire["prior_candidate"]["units"][0][0] == "l"
    selected = copy.deepcopy(args)
    select_task_view(selected[1])
    task_wire, _ = encode(selected)
    assert task_wire["prior_candidate"]["units"][0][0] == "p"
    repeated_wire, repeated_receipt = encode(args)
    assert repeated_wire == old_wire and repeated_receipt == old_receipt
    assert args == before


def multi_paragraph_case(case):
    """Keep distinct prose, paragraph boundaries, and ordered citation lists."""
    body, _, packet, fid, fragment, raw = copy.deepcopy(case)
    body["title"] = 'Exact title "引文" and \\ path'
    packet["cards"]["c2"] = {
        **copy.deepcopy(packet["cards"]["c1"]),
        "evidence_id": "E2",
    }
    body["introduction"][0]["sentences"].append(
        {"text": "Second native unit.\t", "fact_card_refs": ["c2", "c1"]}
    )
    body["introduction"].append(
        {
            "sentences": [
                {"text": 'Exact "quotation" and \\ path.', "fact_card_refs": []}
            ]
        }
    )
    targets = capacity_targets(body, {"reason": "capacity"}, maximum=66)
    return body, targets, packet, fid, fragment, raw


def summary_task_input(case):
    """Bind summary coordinates to an independent complete readonly body."""
    source_body = case[0]
    summary = {
        "abstract": [
            {"sentences": [{"text": case[-1]["claim_text"], "body_unit_refs": ["u1"]}]}
        ],
        "conclusion": [
            {
                "sentences": [
                    {
                        "text": "Ordered readonly premises.",
                        "body_unit_refs": ["u2", "u1"],
                    }
                ]
            }
        ],
    }
    wrapper = context_for(case)
    directory = wrapper["host_directory"]
    fragment = next(iter(directory["fragments"].values()))
    fragment["native_paths"] = [["abstract", 0, 0]]
    fragment["body_premise_paths"] = [["introduction", 0, 0]]
    directory["current_body_sha256"] = directory["source_body_sha256"] = digest(summary)
    rebind(wrapper)
    unit = summary["abstract"][0]["sentences"][0]
    target = {
        "path": ["abstract", 0, 0],
        "original": copy.deepcopy(unit),
        "original_sha256": digest(unit),
        "maximum": 1,
    }
    canonical, payload, authority = codec_input(case)
    authority["guidance_context"]["current"] = wrapper
    authority["summary_body"] = copy.deepcopy(source_body)
    payload.update(
        current_body=summary,
        body_sha256=digest(summary),
        targets={
            "r1": {key: value for key, value in target.items() if key != "original"}
        },
        body_units=body_units(source_body),
        summary_source_body_sha256=digest(source_body),
    )
    canonical["current"] = flat.project_structured_repair_guidance(
        wrapper["diagnostics"],
        host_directory=directory,
        packet=case[2],
        current_body=summary,
        targets={"r1": target},
    )
    select_task_view(payload)
    return canonical, payload, authority


@pytest.mark.parametrize("scope", ["body", "summary"])
def test_native_body_table_is_exact_and_uses_the_same_authenticated_receipt(
    guidance_case, scope
):
    case = multi_paragraph_case(guidance_case)
    args = task_input(case) if scope == "body" else summary_task_input(case)
    before = copy.deepcopy(args)
    wire, receipt = encode(args)
    projected = flat.repair_task_body_projection(args[1])
    assert projected is not args[1]
    assert projected["current_body_format"] == "host-native-prose-table-v1"
    expected = {}
    for section, value in args[1]["current_body"].items():
        if section == "title":
            expected[section] = value
            continue
        field = (
            "body_unit_refs"
            if section in {"abstract", "conclusion"}
            else "fact_card_refs"
        )
        expected[section] = [
            [[unit["text"], unit[field]] for unit in paragraph["sentences"]]
            for paragraph in value
        ]
    assert projected["current_body"] == expected
    assert flat._canonical_task_payload(projected) == args[1]
    if scope == "body":
        assert [
            len(paragraph) for paragraph in projected["current_body"]["introduction"]
        ] == [2, 1]
        assert projected["current_body"]["introduction"][0][1][1] == ["c2", "c1"]
        assert projected["current_body"]["title"] == case[0]["title"]
    else:
        assert projected["current_body"]["conclusion"][0][0][1] == ["u2", "u1"]
        assert projected["body_units"] == args[1]["body_units"]
    projected_args = args[0], projected, args[2]
    assert encode(projected_args) == (wire, receipt)
    assert restore(projected_args, wire, receipt) == args[0]
    assert args == before


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_format",
        "missing_format",
        "unknown_section",
        "row_extra_value",
        "row_missing_value",
        "paragraph_container",
        "boolean_text",
        "boolean_reference",
        "float_reference",
        "changed_text",
        "changed_refs",
        "deleted_unit",
        "reordered_units",
        "deleted_paragraph",
        "reordered_paragraphs",
        "current_target",
        "boolean_target_index",
        "float_target_index",
    ],
)
@pytest.mark.parametrize("consumer", ["encode", "restore"])
def test_native_body_table_tampering_cannot_replace_complete_authority(
    guidance_case, mutation, consumer
):
    args = task_input(multi_paragraph_case(guidance_case))
    original = copy.deepcopy(args)
    wire, receipt = encode(args)
    payload = flat.repair_task_body_projection(args[1])
    paragraphs = payload["current_body"]["introduction"]
    if mutation == "unknown_format":
        payload["current_body_format"] = "host-native-prose-table-future"
    elif mutation == "missing_format":
        payload.pop("current_body_format")
    elif mutation == "unknown_section":
        payload["current_body"]["future"] = []
    elif mutation == "row_extra_value":
        paragraphs[0][0].append("cannot silently drop")
    elif mutation == "row_missing_value":
        paragraphs[0][0].pop()
    elif mutation == "paragraph_container":
        paragraphs[0] = {"sentences": paragraphs[0]}
    elif mutation == "boolean_text":
        paragraphs[0][0][0] = False
    elif mutation == "boolean_reference":
        paragraphs[0][0][1] = [False]
    elif mutation == "float_reference":
        paragraphs[0][0][1] = [0.0]
    elif mutation == "changed_text":
        paragraphs[0][0][0] += " Unauthenticated change."
    elif mutation == "changed_refs":
        paragraphs[0][0][1] = ["c2", "c1"]
    elif mutation == "deleted_unit":
        paragraphs[0].pop(0)
    elif mutation == "reordered_units":
        paragraphs[0].reverse()
    elif mutation == "deleted_paragraph":
        paragraphs.pop()
    elif mutation == "reordered_paragraphs":
        paragraphs.reverse()
    elif mutation == "current_target":
        payload["targets"]["r1"]["path"] = ["taxonomy", 0, 0]
    else:
        payload["targets"]["r1"]["path"][1] = (
            False if mutation == "boolean_target_index" else 0.0
        )
    if mutation in {
        "boolean_text",
        "boolean_reference",
        "float_reference",
        "changed_text",
        "changed_refs",
        "deleted_unit",
        "reordered_units",
        "deleted_paragraph",
        "reordered_paragraphs",
    }:
        payload["body_sha256"] = digest(
            flat._canonical_task_payload(payload)["current_body"]
        )
    rehash_view(wire, receipt)
    projected_args = args[0], payload, args[2]
    before = copy.deepcopy((projected_args, wire, receipt))
    with pytest.raises(WriterError):
        if consumer == "encode":
            encode(projected_args)
        else:
            restore(projected_args, wire, receipt)
    assert (projected_args, wire, receipt) == before
    assert args == original


def test_body_projection_preserves_source_markers_and_rejects_second_projection(
    guidance_case,
):
    args = task_input(guidance_case)
    before = copy.deepcopy(args)
    projected = flat.repair_task_body_projection(args[1])
    assert projected["writing_envelope"] == args[1]["writing_envelope"]
    assert projected["sources"] == args[1]["sources"]
    assert projected["cards"] == args[1]["cards"]
    snapshot = copy.deepcopy(projected)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat.repair_task_body_projection(projected)
    assert projected == snapshot
    assert args == before


def test_explicit_shared_values_preserve_host_boolean_integer_distinctions(
    guidance_case,
):
    raw = [
        {"code": "host_parameter_mismatch", "actual": 0, "expected": False},
        {"code": "host_parameter_mismatch", "actual": False, "expected": 0},
    ]
    args = task_input(
        guidance_case,
        context_for(guidance_case, raw, category="local", reportless=True),
    )
    wire, receipt = encode(args)
    rows = issue_rows(wire["current"])
    assert len(rows) == 2
    assert type(rows[0]["host_fields"]["actual"]) is int
    assert rows[0]["host_fields"]["expected"] is False
    assert rows[1]["host_fields"]["actual"] is False
    assert type(rows[1]["host_fields"]["expected"]) is int
    assert restore(args, wire, receipt) == args[0]
