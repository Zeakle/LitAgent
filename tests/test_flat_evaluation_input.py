"""Check complete report sharing without changing scientific ownership."""

import copy
import json

import pytest
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_native_report_input import global_payload
from test_writer_v2 import case  # noqa: F401

from litagent.config import EvalConfig
from litagent.eval.batched import (
    GlobalReview,
    batch_alias_input,
    batch_payload,
    batch_prompt,
    flat_evaluation_instruction,
    global_input_payload,
    plan_batches,
)
from litagent.safety.planned_budget import (
    EnvelopeExceeded,
    check_inventory_envelope,
    nested_json_size,
)
from litagent.writing_flat import (
    EVALUATION_INPUT,
    evaluation_layout,
    evaluation_projection,
    global_identity_projection,
    repair_projection,
    restore_global_identities,
    restore_inventory_identities,
    validation_schema_projection,
)
from litagent.writing_v2 import digest, repair_targets, writer_budget_plan


def test_report_layout_uses_host_ranges_even_when_text_is_identical():
    report = "## Methods\nSame. [E:long]\nSame. [E:long]\n"
    first, second = report.index("Same"), report.rindex("Same")
    records = [("a", first, first + 14), ("b", second, second + 14)]
    layout = evaluation_layout(report, records, {"long": "e1"}, len(report))
    assert "".join(p["text"] for p in layout) == report.replace("[E:long]", "[E:e1]")
    assert [p["source_fragment_id"] for p in layout if "source_fragment_id" in p] == [
        "a",
        "b",
    ]
    payload = {
        "report_layout": layout,
        "source_units": [
            {
                "source_unit_id": "u",
                "scope_context": "duplicate context",
                "source_fragments": [
                    {
                        "source_fragment_id": p["source_fragment_id"],
                        "text": p["text"],
                        "citation_aliases": ["e1"],
                    }
                    for p in layout
                    if "source_fragment_id" in p
                ],
            }
        ],
    }
    before = copy.deepcopy(payload)
    wire = evaluation_projection(payload)
    assert payload == before
    for part, original in zip(wire["report_layout"], layout):
        if "source_fragment_id" in part:
            assert part["prose"] + " [E:e1]" == original["text"]
        else:
            assert part == original
    assert "scope_context" not in wire["source_units"][0]
    assert all("text" not in f for f in wire["source_units"][0]["source_fragments"])


@pytest.mark.parametrize("fault", ["text", "omit", "duplicate", "order"])
def test_invalid_layout_cannot_drop_or_rebind_report_text(fault):
    payload = global_payload()
    fragments = payload["source_units"][0]["source_fragments"]
    payload["report_layout"] = [
        {"source_fragment_id": f["source_fragment_id"], "text": f["text"]}
        for f in fragments
    ]
    if fault == "text":
        payload["report_layout"][0]["text"] = "A different dataset."
    elif fault == "omit":
        payload["report_layout"].pop()
    elif fault == "duplicate":
        payload["report_layout"].append(copy.deepcopy(payload["report_layout"][0]))
    else:
        payload["report_layout"].reverse()
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="evaluation_layout"):
        evaluation_projection(payload)
    assert payload == before


@pytest.mark.parametrize("uncited", [False, True])
def test_citation_inheritance_is_only_from_the_same_exact_fragment(uncited):
    payload = global_payload()
    payload["evaluation_input_contract"] = EVALUATION_INPUT
    if uncited:
        payload["assertions"][0]["evidence"] = []
    wire = global_input_payload(payload)
    claim = dict(zip(wire["assertion_columns"], wire["assertions"][0]))
    if uncited:
        assert claim["evidence_aliases"] == []
    else:
        assert claim["evidence_aliases"] is None
        fid = claim["source_fragment_id"]
        assert next(
            f
            for u in wire["source_units"]
            for f in u["source_fragments"]
            if f["source_fragment_id"] == fid
        )["citation_aliases"] == ["e1"]
    legacy = copy.deepcopy(payload)
    legacy.pop("evaluation_input_contract")
    assert "evidence_aliases" in global_input_payload(legacy)["assertions"][0]


@pytest.mark.parametrize("kind", ["scientific", "runtime_method", "report_structure"])
def test_only_runtime_claims_receive_trusted_method_record(canonical, kind):
    payload = copy.deepcopy(canonical)
    payload["evaluation_input_contract"] = EVALUATION_INPUT
    payload["host_runtime_record"] = {"queries": [{"status": "failed"}]}
    payload["assertions"][0]["kind"] = kind
    result = batch_payload(payload, payload["assertions"][:1])
    assert bool(result["host_runtime_record"]) is (kind == "runtime_method")
    assert result["original_supports"] == payload["original_supports"]
    assert result["source_units"] == payload["source_units"]


def test_claim_scope_rejects_the_whole_output_without_truncation(case):
    scope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )["envelope"]
    claims = [{"claim_text": "x" * 200} for _ in range(128)]
    check_inventory_envelope(claims, scope)
    oversized = [{"claim_text": "条件" * 100} for _ in range(128)]
    before = copy.deepcopy(oversized)
    with pytest.raises(EnvelopeExceeded):
        check_inventory_envelope(oversized, scope)
    assert oversized == before
    with pytest.raises(EnvelopeExceeded):
        check_inventory_envelope([{"claim_text": "x" * 2001}], scope)


def test_global_aliases_bind_native_schema_and_restore_only_identity_fields():
    payload = global_payload()
    payload.update(
        evaluation_input_contract=EVALUATION_INPUT,
        report_structure=[{"source_unit_id": "unit:1", "section": "Methods"}],
        expected_source_unit_ids=["unit:1"],
        analysis_claim_ids_by_section={"methods": ["a1"]},
    )
    native = {
        "type": "json_schema",
        "json_schema": {
            "schema": {
                "properties": {"source_unit_id": {"enum": ["unit:1"]}},
            }
        },
    }
    before = copy.deepcopy(payload)
    wire, schema = global_identity_projection(
        global_input_payload(payload), native, payload
    )
    assert payload == before
    assert set(wire["source_units"]) == {"v1"}
    assert schema["json_schema"]["schema"]["properties"]["source_unit_id"]["enum"] == [
        "v1"
    ]
    result = GlobalReview.model_validate(
        {
            "units": [
                {
                    "source_unit_id": "v1",
                    "complete": True,
                    "explanation": "v1 is text here.",
                }
            ],
            "conflicts": [],
            "issues": [],
            "coverage": [
                {
                    "requirement_id": "r",
                    "answered": True,
                    "claim_ids": ["a1"],
                    "explanation": "Literal a1.",
                }
            ],
        }
    )
    raw = result.model_dump_json()
    restored = restore_global_identities(result, payload)
    assert restored.units[0].source_unit_id == "unit:1"
    assert restored.units[0].explanation == "v1 is text here."
    assert result.model_dump_json() == raw
    result.units[0].source_unit_id = "unit:1"
    with pytest.raises(ValueError, match="alias_invalid"):
        restore_global_identities(result, payload)


def test_repair_targets_preserve_both_science_and_reviewer_diagnoses(case):
    unit = {
        "unit_id": "u",
        "path": ["methods", 0, 0],
        "source_start": 1,
        "source_end": 100,
    }
    targets = repair_targets(
        {"units": [unit]},
        [
            {"unit_id": "u", "explanation": "Incorrect model attribution."},
            {"unit_id": "u", "explanation": "Explain the comparison conditions."},
        ],
        case.body,
        case.summaries,
        merge=True,
    )
    assert targets["u"]["explanation"].splitlines() == [
        "Incorrect model attribution.",
        "Explain the comparison conditions.",
    ]
    scope = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )["envelope"]
    payload = {"current_body": case.body, "targets": targets}
    before = copy.deepcopy(payload)
    wire = repair_projection(payload, scope)
    assert payload == before and wire["current_body"] == case.body
    assert wire["targets"]["u"]["original_sha256"] == digest(targets["u"]["original"])
    assert "original" not in wire["targets"]["u"]
    assert nested_json_size(wire) < nested_json_size(payload) + 150
    targets["u"]["explanation"] = "X" * 13000
    with pytest.raises(EnvelopeExceeded):
        repair_projection(payload, scope)


def test_size_balancing_keeps_all_128_claims_and_caps_each_batch(canonical, case):
    payload = copy.deepcopy(canonical)
    payload["evaluation_input_contract"] = EVALUATION_INPUT
    payload["writer_envelope"] = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )["envelope"]
    payload["source_units"][0]["source_fragments"][0]["text"] += " [E:e1]"
    template = payload["assertions"][0]
    claims = [
        {
            **copy.deepcopy(template),
            "claim_id": f"claim_{i}",
            "claim_text": ("x" * 1700 if i < 16 else "Small") + str(i),
        }
        for i in range(128)
    ]
    payload["assertions"] = claims
    check_inventory_envelope(claims, payload["writer_envelope"])
    first = plan_batches(payload, EvalConfig())
    assert first == plan_batches(payload, EvalConfig())
    assert len(first) == 8 and all(len(b["assertions"]) == 16 for b in first)
    ids = [c["claim_id"] for batch in first for c in batch["assertions"]]
    assert len(ids) == len(set(ids)) == 128
    assert set(ids) == {c["claim_id"] for c in claims}
    assert (
        max(nested_json_size([c["claim_text"] for c in b["assertions"]]) for b in first)
        < 5000
    )
    with pytest.raises(ValueError, match="batch_capacity_exceeded"):
        plan_batches(payload, EvalConfig(strict_batch_size=4))


def test_batch_tables_preserve_prose_claims_and_exact_fragment_ownership(canonical):
    payload = copy.deepcopy(canonical)
    payload["evaluation_input_contract"] = EVALUATION_INPUT
    unit = payload["source_units"][0]
    unit["source_fragments"][0]["text"] += " [E:e1]"
    unit["citation_aliases"].append("e2")
    unit["source_fragments"].append(
        {
            "source_fragment_id": "neighbor-only",
            "text": "Only under a different dataset condition. [E:e2]",
            "citation_aliases": ["e2"],
        }
    )
    before = copy.deepcopy(payload)
    projected = batch_alias_input(payload)
    assert payload == before
    parent, fragments = {}, {}
    prose = []
    for u in projected["source_units"]:
        for row in u["source_fragments"]:
            f = dict(zip(projected["fragment_columns"], row))
            prose.append(f["prose"])
            if f["source_fragment_id"] is None:
                assert f["citation_aliases"] is None
            else:
                parent[f["source_fragment_id"]] = u["source_unit_id"]
                fragments[f["source_fragment_id"]] = f
    assert prose == [
        "A method has a source-bound limitation.",
        "Only under a different dataset condition.",
    ]
    restored = []
    for row in projected["assertions"]:
        claim = dict(zip(projected["assertion_columns"], row))
        fid = claim["source_fragment_id"]
        citations = claim.pop("evidence_aliases")
        if citations is None:
            citations = fragments[fid]["citation_aliases"]
        claim.update(
            source_unit_id=parent[fid],
            source_fragment_id="f1",
            evidence=[{"evidence_alias": a, "material_ref": a} for a in citations],
        )
        restored.append(claim)
    assert restored == payload["assertions"]
    assert projected["original_supports"] == payload["original_supports"]
    assert "report_layout" not in flat_evaluation_instruction(payload)
    assert "QUANTITATIVE ATTRIBUTION" in batch_prompt(payload)
    assert "SUPPORT CONTRACT" in batch_prompt(payload)


@pytest.mark.parametrize("fault", ["extra_claim_field", "duplicate_fragment", "union"])
def test_batch_tables_reject_lossy_or_ambiguous_inputs(canonical, fault):
    payload = copy.deepcopy(canonical)
    payload["evaluation_input_contract"] = EVALUATION_INPUT
    payload["source_units"][0]["source_fragments"][0]["text"] += " [E:e1]"
    if fault == "extra_claim_field":
        payload["assertions"][0]["unrecognized"] = "must not disappear"
    elif fault == "union":
        payload["source_units"][0]["citation_aliases"] = []
    else:
        unit = copy.deepcopy(payload["source_units"][0])
        unit["source_unit_id"] = "another-unit"
        payload["source_units"].append(unit)
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="evaluation_.*invalid"):
        batch_alias_input(payload)
    assert payload == before


def test_actual_batch_sizing_keeps_shared_source_pools_together(canonical, case):
    payload = copy.deepcopy(canonical)
    payload["evaluation_input_contract"] = EVALUATION_INPUT
    payload["writer_envelope"] = writer_budget_plan(
        case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v5"
    )["envelope"]
    unit = payload["source_units"][0]
    unit["citation_aliases"] = ["e1", "e2"]
    unit["source_fragments"] = [
        {
            "source_fragment_id": f"f{i}",
            "text": f"Method {i} has a qualified result. [E:e{i}]",
            "citation_aliases": [f"e{i}"],
        }
        for i in (1, 2)
    ]
    payload["cited_materials"]["e2"] = {"original_support_ref": "s2"}
    payload["original_supports"] = {
        f"s{i}": {"paper_id": f"paper-{i}", "text": str(i) * 5000} for i in (1, 2)
    }
    payload["support_passages"] = {f"p{i}": [f"e{i}", f"s{i}", 0, 5000] for i in (1, 2)}
    template = copy.deepcopy(payload["assertions"][0])
    payload["assertions"] = [
        {
            **copy.deepcopy(template),
            "claim_id": f"claim_{index}",
            "source_fragment_id": f"f{1 if index < 64 else 2}",
            "evidence": [
                {
                    "evidence_alias": f"e{1 if index < 64 else 2}",
                    "material_ref": f"e{1 if index < 64 else 2}",
                }
            ],
        }
        for index in range(128)
    ]
    before = copy.deepcopy(payload)
    batches = plan_batches(payload, EvalConfig())
    assert payload == before and len(batches) == 8
    assert all(len(batch["assertions"]) == 16 for batch in batches)
    assert all(len(batch["original_supports"]) == 1 for batch in batches)
    assert [c for batch in batches for c in batch["assertions"]] == payload[
        "assertions"
    ]
    for batch in batches:
        for claim in batch["assertions"]:
            alias = claim["evidence"][0]["evidence_alias"]
            sid = batch["cited_materials"][alias]["original_support_ref"]
            assert batch["original_supports"][sid] == payload["original_supports"][sid]


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        '{"fragments":{}}',
        '{"fragments":{"bad":{}}}',
        '{"fragments":{"f1":{},"f1":{}}}',
    ],
)
def test_inventory_alias_errors_never_guess_a_missing_host_identity(raw):
    units = [{"source_fragments": [{"source_fragment_id": "host:fragment"}]}]
    with pytest.raises(ValueError):
        restore_inventory_identities(raw, units)


def test_schema_projection_preserves_real_validation_constraints():
    from jsonschema import Draft202012Validator

    schema = {
        "type": "object",
        "title": "Generated annotation",
        "description": "Host prompt supplies the task instructions.",
        "$defs": {
            "Active": {"type": "string", "enum": ["a1"], "title": "Identity"},
            "Unused": {"type": "integer"},
        },
        "properties": {"claim_id": {"$ref": "#/$defs/Active"}},
        "required": ["claim_id"],
        "additionalProperties": False,
    }
    response_format = {"type": "json_schema", "json_schema": {"schema": schema}}
    projected = validation_schema_projection(response_format)["json_schema"]["schema"]
    assert set(projected["$defs"]) == {"Active"}
    for value in (
        {"claim_id": "a1"},
        {},
        {"claim_id": "bad"},
        {"claim_id": "a1", "extra": 1},
    ):
        assert Draft202012Validator(schema).is_valid(value) == Draft202012Validator(
            projected
        ).is_valid(value)
    assert response_format["json_schema"]["schema"] == schema
