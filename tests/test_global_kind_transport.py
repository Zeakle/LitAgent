"""Keep assigned kinds explicit without changing scientific evaluation authority."""

import copy
import json

import pytest
from test_evaluation_definition_contract import (
    V1,
    prompt_for,
    request_payload,
)
from test_evaluation_wire_projection import canonical  # noqa: F401

from litagent.eval import scoped
from litagent.eval.batched import global_input_payload
from litagent.writing_flat import global_identity_projection

KINDS = ("scientific", "report_structure", "runtime_method")
COLUMNS = (
    "sequence",
    "claim_id",
    "source_fragment_id",
    "claim_text",
    "evidence_aliases",
)
LEGACY_REQUEST_IDENTITY = (
    "878051b01f904e0b75aa2d955e897ffac3c80ae388cc50a0d5876a161a14b214"
)


def mixed_global(canonical):
    """Retain identical prose at distinct positions and interleaved assigned kinds."""
    value = copy.deepcopy(canonical)
    second = copy.deepcopy(value["source_units"][0])
    second["source_unit_id"] = "u2"
    second["source_fragments"][0]["source_fragment_id"] = "f2"
    value["source_units"].append(second)
    value["report_structure"].append({"source_unit_id": "u2", "section": "methods"})
    sample = value["assertions"][0]
    value["assertions"] = []
    for index, kind in enumerate(
        ("scientific", "runtime_method", "report_structure") * 2
    ):
        claim = copy.deepcopy(sample)
        claim.update(
            claim_id=f"original-{index}",
            kind=kind,
            claim_text='Same complete relation with 汉字, "quotes", and\tconditions.',
            source_unit_id="u1" if index < 3 else "u2",
            source_fragment_id="f1" if index < 3 else "f2",
        )
        if index % 2:
            claim["evidence"] = []
        value["assertions"].append(claim)
    return request_payload(value, "global")


def flatten(wire):
    """Read complete group names without assigning a default classification."""
    assert tuple(wire["assertion_columns"]) == COLUMNS
    result = []
    for kind, rows in wire["assertions_by_kind"].items():
        for row in rows:
            result.append({**dict(zip(COLUMNS, row, strict=True)), "kind": kind})
    return sorted(result, key=lambda row: row["sequence"])


def project(payload):
    """Exercise both production projections without a budget or runtime owner."""
    wire = global_input_payload(payload, explicit_kinds=True)
    return global_identity_projection(
        wire,
        scoped.evaluation_response_format(payload),
        payload,
        explicit_kinds=True,
    )


def test_actual_global_builder_transmits_each_original_kind_and_full_order(canonical):
    payload = mixed_global(canonical)
    before = copy.deepcopy(payload)
    messages, native = scoped.build_evaluation_request(
        prompt_for("global", payload), payload
    )
    wire = json.loads(messages[1]["content"])
    assert tuple(wire["assertions_by_kind"]) == KINDS
    assert "assertions" not in wire and "assertion_kind_default" not in wire
    rows = flatten(wire)
    assert [row["sequence"] for row in rows] == list(range(1, 7))
    assert [row["claim_id"] for row in rows] == [f"a{i}" for i in range(1, 7)]
    for row, original in zip(rows, before["assertions"], strict=True):
        assert row["kind"] == original["kind"]
        assert row["claim_text"] == original["claim_text"]
        assert row["source_fragment_id"] == (
            "f1" if original["source_unit_id"] == "u1" else "f2"
        )
        assert row["evidence_aliases"] == (None if original["evidence"] else [])
    assert len({row["claim_text"] for row in rows}) == 1
    assert len({row["source_fragment_id"] for row in rows}) == 2
    assert "null kind means scientific" not in messages[0]["content"]
    assert "Omitted\nkind means assertion_kind_default" not in messages[0]["content"]
    assert "assigned kind is its enclosing full group name" in messages[0]["content"]
    assert native["json_schema"]["strict"] is True
    assert payload == before


@pytest.mark.parametrize("kind", KINDS)
def test_full_group_names_include_empty_groups_without_changing_native_schema(
    canonical, kind
):
    payload = request_payload(canonical, "global")
    for claim in payload["assertions"]:
        claim["kind"] = kind
    before = copy.deepcopy(payload)
    explicit, schema = project(payload)
    legacy, legacy_schema = global_identity_projection(
        global_input_payload(payload),
        scoped.evaluation_response_format(payload),
        payload,
    )
    assert set(explicit["assertions_by_kind"]) == set(KINDS)
    assert len(explicit["assertions_by_kind"][kind]) == len(payload["assertions"])
    assert all(
        not rows for key, rows in explicit["assertions_by_kind"].items() if key != kind
    )
    assert schema == legacy_schema
    excluded = {
        "assertions",
        "assertions_by_kind",
        "assertion_columns",
        "assertion_kind_default",
    }
    assert {k: v for k, v in explicit.items() if k not in excluded} == {
        k: v for k, v in legacy.items() if k not in excluded
    }
    assert payload == before


@pytest.mark.parametrize("revision", [None, V1])
def test_absent_and_v1_global_requests_match_exact_prechange_bytes(canonical, revision):
    payload = request_payload(canonical, "global", revision=revision)
    messages, native = scoped.build_evaluation_request(
        prompt_for("global", payload), payload
    )
    assert scoped._request_identity(messages, native) == LEGACY_REQUEST_IDENTITY
    wire = json.loads(messages[1]["content"])
    assert "assertions_by_kind" not in wire
    assert wire["assertion_kind_default"] == "scientific"


@pytest.mark.parametrize("invalid", [None, 1, "true"])
def test_projection_flags_reject_non_bool_without_mutation(canonical, invalid):
    payload = mixed_global(canonical)
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="kind_configuration"):
        global_input_payload(payload, explicit_kinds=invalid)
    wire = global_input_payload(payload, explicit_kinds=True)
    with pytest.raises(ValueError, match="kind_configuration"):
        global_identity_projection(wire, None, payload, explicit_kinds=invalid)
    assert payload == before


@pytest.mark.parametrize("invalid", [None, False, "factual", ""])
def test_explicit_kind_never_uses_null_default_or_unknown_category(canonical, invalid):
    payload = mixed_global(canonical)
    payload["assertions"][0]["kind"] = invalid
    before = copy.deepcopy(payload)
    with pytest.raises(ValueError, match="evaluation_input_kind_invalid"):
        global_input_payload(payload, explicit_kinds=True)
    assert payload == before


@pytest.mark.parametrize(
    "fault",
    [
        "omit",
        "duplicate",
        "order",
        "kind",
        "text",
        "fragment",
        "citations",
        "columns",
        "default",
    ],
)
def test_grouping_refuses_incomplete_reordered_or_rebound_rows(canonical, fault):
    payload = mixed_global(canonical)
    wire = global_input_payload(payload, explicit_kinds=True)
    if fault == "omit":
        wire["assertions"].pop()
    elif fault == "duplicate":
        wire["assertions"][1] = copy.deepcopy(wire["assertions"][0])
    elif fault == "order":
        wire["assertions"].reverse()
    elif fault == "kind":
        wire["assertions"][0][3] = "report_structure"
    elif fault == "text":
        wire["assertions"][0][2] += " Missing condition."
    elif fault == "fragment":
        wire["assertions"][0][1] = "f2"
    elif fault == "citations":
        wire["assertions"][0][4] = []
    elif fault == "columns":
        wire["assertion_columns"].reverse()
    else:
        wire["assertion_kind_default"] = "scientific"
    before = copy.deepcopy(wire)
    with pytest.raises(ValueError, match="evaluation_global_kind_identity_invalid"):
        global_identity_projection(wire, None, payload, explicit_kinds=True)
    assert wire == before


@pytest.mark.parametrize(
    "field",
    ["global_output_contract", "report_input_contract", "evaluation_input_contract"],
)
def test_explicit_projection_rejects_wrong_stage_or_input_contract(canonical, field):
    payload = mixed_global(canonical)
    payload[field] = "unknown"
    with pytest.raises(ValueError, match="kind_configuration"):
        global_input_payload(payload, explicit_kinds=True)
    with pytest.raises(ValueError, match="kind_configuration"):
        global_identity_projection({}, None, payload, explicit_kinds=True)
