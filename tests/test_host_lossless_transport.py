"""Keep explicit lossless transport reversible and original repair positions exact."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from litagent.capacity import CapacityConfig
from litagent.capacity_repair import capacity_targets, edit_schema, normalize_edit_map
from litagent.eval.scoped import build_evaluation_request, flat_inventory_schema
from litagent.safety.planned_budget import make_writer_envelope
from litagent.writing_flat import (
    REFERENCE_SET_CONTRACT,
    TRANSPORT_REVISION,
    TRANSPORT_TABLE_INSTRUCTION,
    decode_field_table,
    encode,
    encode_field_table,
    inventory_transport_projection,
    parse,
    repair_projection,
    restore_reference_input,
    transport_identity,
)
from litagent.writing_v2 import BODY, WriterError, digest, schema_for


def envelope(selected=True):
    """Build the real locked-contract combination, without a private run archive."""
    return make_writer_envelope(
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
        repair_feedback=True,
        reference_contract=REFERENCE_SET_CONTRACT,
        **({"transport_revision": TRANSPORT_REVISION} if selected else {}),
    )


def inventory_payload(selected=True):
    """Use an independently constructed full ownership directory and schema."""
    units = [
        {
            "source_unit_id": "unit:one",
            "citation_aliases": ["e1"],
            "source_fragments": [
                {
                    "source_fragment_id": "fragment:one",
                    "text": "A changes representations. [E:e1]",
                    "citation_aliases": ["e1"],
                }
            ],
        }
    ]
    return {
        "writer_envelope": envelope(selected),
        "evaluation_input_contract": "flat-evaluation-input-v1",
        "inventory_output_contract": "host-flat-fragment-inventory-v1",
        "schema": flat_inventory_schema(units),
        "source_units": units,
        "allowed_citation_aliases": ["e1"],
        "report_layout": [
            {"text": "## Methods\n"},
            {
                "source_fragment_id": "fragment:one",
                "text": "A changes representations. [E:e1]",
            },
        ],
        "inventory_recovery": {
            "contract": "bounded-inventory-recovery-v2",
            "findings": [
                {
                    "origin": "global",
                    "issue_index": 0,
                    "source_unit_id": "unit:one",
                    "source_fragment_ids": ["fragment:one"],
                    "claim_ids": ["old:one"],
                    "code": "non_atomic",
                    "explanation": 'Keep all conditions, including \\"否定\\".',
                }
            ],
            "concern_claims": [
                {
                    "claim_id": "old:one",
                    "source_unit_id": "unit:one",
                    "source_fragment_id": "fragment:one",
                    "claim_text": "A changes representations.",
                    "kind": "scientific",
                    "correct_kind": "scientific",
                    "atomic": False,
                    "classification_valid": True,
                    "source_faithful": True,
                    "evidence_ids": ["source:one"],
                }
            ],
            "global_finding_bindings": [
                {
                    "finding_index": 0,
                    "code": "non_atomic",
                    "anchor_ids": ["claim:old:one"],
                    "source_unit_ids": ["unit:one"],
                    "explanation": "Keep all conditions.",
                }
            ],
        },
    }


def repair_case():
    """Map every replacement explicitly, including a merge within r1."""
    unit = {"text": "A complete finding.", "fact_card_refs": ["c1", "c2"]}
    body = {key: [{"sentences": [copy.deepcopy(unit)]}] for key in BODY}
    body[BODY[0]][0]["sentences"] *= 3
    all_targets = capacity_targets(body, {}, maximum=8)
    targets = {key: all_targets[key] for key in ("r1", "r2")}
    schema = edit_schema(
        schema_for("repair", ["c1", "c2", "c3"], targets), targets, mapping=True
    )
    canonical = {
        "outcome": "patched",
        "explanation": "Preserve every distinct finding.",
        "patches": [
            {
                "target_id": "r1",
                "sentences": [
                    copy.deepcopy(unit),
                    {**unit, "text": "The other finding."},
                ],
            },
            {"target_id": "r2", "sentences": [copy.deepcopy(unit)]},
        ],
        "edit_records": [
            {
                "unit_id": f"writer:{BODY[0]}:0:{i}",
                "replacement_indexes": [0 if i < 2 else 1],
                "reason": "Explicit relation.",
            }
            for i in range(3)
        ]
        + [
            {
                "unit_id": f"writer:{BODY[1]}:0:0",
                "replacement_indexes": [0],
                "reason": "Explicit relation.",
            }
        ],
    }
    wire = encode(canonical, schema)
    wire["patches"] = [wire["patches"][i] for i in (0, 2, 1)]
    for patch in wire["patches"]:
        patch["fact_card_refs"] = ["c1", "c2", "c1"]
    return body, targets, schema, canonical, wire


def selected_parse(raw, schema):
    """Select both independent contracts explicitly in the standalone parser."""
    return parse(
        raw,
        schema,
        reference_contract=REFERENCE_SET_CONTRACT,
        transport_revision=TRANSPORT_REVISION,
    )


def test_field_tables_preserve_heterogeneous_keys_and_every_json_value():
    rows = [
        {
            "text": 'Unicode 否定 / \\ / "quoted"',
            "optional": None,
            "flag": False,
            "values": [],
        },
        {"text": "Repeated text", "nested": {"flag": False, "empty": ""}},
        {"text": "Repeated text", "optional": 0},
        {},
        {"text": "Repeated text", "optional": None, "flag": False, "values": []},
    ]
    before = copy.deepcopy(rows)
    table = encode_field_table(rows)
    assert decode_field_table(table) == rows == before
    assert len(table["columns"]) == 4
    assert [row[0] for row in table["rows"]] == [0, 1, 2, 3, 0]
    assert decode_field_table(encode_field_table([])) == []
    table["rows"][1][-1]["flag"] = True
    assert rows == before


@pytest.mark.parametrize(
    "table",
    [
        None,
        [],
        {"columns": [], "rows": [], "extra": 1},
        {"columns": [["x", "x"]], "rows": [[0, 1, 2]]},
        {"columns": [["x"]], "rows": [[True, 1]]},
        {"columns": [["x"]], "rows": [[-1, 1]]},
        {"columns": [["x"]], "rows": [[1, 1]]},
        {"columns": [["x"]], "rows": [[0]]},
        {"columns": [["x"]], "rows": [[0, 1, 2]]},
        {"columns": [["x"], ["unused"]], "rows": [[0, 1]]},
        {"columns": [["x"], ["x"]], "rows": [[0, 1], [1, 2]]},
        {"columns": [[1]], "rows": [[0, 1]]},
        {"columns": [["x"]], "rows": [[0, float("nan")]]},
    ],
)
def test_corrupt_field_tables_are_rejected_without_partial_decode(table):
    with pytest.raises(ValueError, match="host_transport_table_invalid"):
        decode_field_table(table)


@pytest.mark.parametrize(
    "rows",
    [
        None,
        {},
        [None],
        [{1: "coercion"}],
        [{"x": (1,)}],
        [{"x": float("inf")}],
        [{"": 1}],
    ],
)
def test_encoder_never_coerces_non_json_input(rows):
    with pytest.raises(ValueError, match="host_transport_table_invalid"):
        encode_field_table(rows)


def test_real_builder_projects_after_alias_validation_and_preserves_schema():
    payload = inventory_payload()
    before = copy.deepcopy(payload)
    old = inventory_payload(False)
    messages, schema = build_evaluation_request(
        "Extract every assertion.",
        payload,
        recovery_evidence_aliases={"source:one": "e1"},
    )
    old_messages, old_schema = build_evaluation_request(
        "Extract every assertion.", old, recovery_evidence_aliases={"source:one": "e1"}
    )
    wire, old_wire = (
        json.loads(value[1]["content"]) for value in (messages, old_messages)
    )
    assert schema == old_schema
    assert payload == before
    assert wire["transport_revision"] == TRANSPORT_REVISION
    assert messages[0]["content"].count(TRANSPORT_TABLE_INSTRUCTION) == 1
    assert TRANSPORT_TABLE_INSTRUCTION not in old_messages[0]["content"]
    for name in ("findings", "concern_claims", "global_finding_bindings"):
        assert (
            decode_field_table(wire["inventory_recovery"][name])
            == old_wire["inventory_recovery"][name]
        )
    assert wire["report_layout"] == old_wire["report_layout"]
    assert wire["source_units"] == old_wire["source_units"]
    assert wire["inventory_recovery"]["contract"] == "bounded-inventory-recovery-v2"
    assert "transport_revision" not in old_wire
    payload["inventory_recovery"]["concern_claims"][0]["evidence_ids"] = ["foreign"]
    with pytest.raises(ValueError, match="inventory_recovery_alias_identity_invalid"):
        build_evaluation_request(
            "Extract every assertion.",
            payload,
            recovery_evidence_aliases={"source:one": "e1"},
        )


def test_projection_none_returns_exact_old_payload_without_mutation():
    value = {"unchanged": [False, None]}
    assert inventory_transport_projection(value) is value
    with pytest.raises(ValueError, match="host_transport_revision_invalid"):
        inventory_transport_projection(value, transport_revision="future")


def test_interleaved_repair_groups_stably_and_restores_all_received_occurrences():
    _, targets, schema, canonical, wire = repair_case()
    raw = json.dumps(wire, ensure_ascii=False)
    with pytest.raises(WriterError, match="writer_flat_target_order_invalid"):
        parse(raw, schema, reference_contract=REFERENCE_SET_CONTRACT)
    result = selected_parse(raw, schema)
    assert result["canonical"] == canonical
    assert (
        normalize_edit_map(result["canonical"], targets)["patches"]
        == canonical["patches"]
    )
    assert [r["wire_ordinal"] for r in result["bindings"]["units"]] == [0, 2, 1]
    assert result["bindings"]["transport_identity"] == transport_identity()
    assert (
        restore_reference_input(
            canonical, result["bindings"], transport_revision=TRANSPORT_REVISION
        )
        == result["received_canonical"]
    )
    for record in result["bindings"]["units"]:
        source = json.loads(raw[record["start"] : record["end"]])
        assert source["target_id"] == record["target_id"]
        span = record["target_id_span"]
        assert json.loads(raw[span["start"] : span["end"]]) == record["target_id"]
        assert [
            json.loads(raw[s["start"] : s["end"]]) for s in record["reference_spans"]
        ] == ["c1", "c2", "c1"]


@pytest.mark.parametrize(
    "fault",
    [
        "revision",
        "identity",
        "ordinal",
        "path",
        "target",
        "span",
        "length",
        "remove_marker",
    ],
)
def test_new_reference_bindings_reject_tamper_and_cross_revision(fault):
    _, _, schema, canonical, wire = repair_case()
    result = selected_parse(json.dumps(wire), schema)
    bindings = copy.deepcopy(result["bindings"])
    if fault == "revision":
        bindings["transport_revision"] = "future"
    elif fault == "identity":
        bindings["transport_identity"] = "0" * 64
    elif fault == "ordinal":
        bindings["units"][1]["wire_ordinal"] = 0
    elif fault == "path":
        bindings["units"][1]["wire_path"] = ["patches", 1]
    elif fault == "target":
        bindings["units"][0]["target_id"] = "r2"
    elif fault == "span":
        bindings["units"][0]["target_id_span"] = bindings["units"][0]["text_span"]
    elif fault == "length":
        bindings["raw_length"] = 1
    else:
        del bindings["transport_revision"]
        del bindings["transport_identity"]
    with pytest.raises(WriterError, match="writer_reference_binding_invalid"):
        restore_reference_input(
            canonical, bindings, transport_revision=TRANSPORT_REVISION
        )
    with pytest.raises(WriterError, match="writer_reference_binding_invalid"):
        restore_reference_input(canonical, result["bindings"])


def test_first_appearance_order_is_preserved_without_sorting_target_ids():
    _, _, schema, _, wire = repair_case()
    wire["patches"] = [wire["patches"][i] for i in (1, 0, 2)]
    result = selected_parse(json.dumps(wire), schema)
    assert [p["target_id"] for p in result["canonical"]["patches"]] == ["r2", "r1"]
    assert (
        restore_reference_input(
            result["canonical"],
            result["bindings"],
            transport_revision=TRANSPORT_REVISION,
        )
        == result["received_canonical"]
    )


def test_original_capacity_and_reference_permissions_precede_grouping():
    _, _, schema, _, wire = repair_case()
    wire["patches"][2]["fact_card_refs"] = ["c1"] * 4
    with pytest.raises(WriterError, match="writer_flat_schema_invalid"):
        selected_parse(json.dumps(wire), schema)
    wire["patches"][2]["fact_card_refs"] = ["unknown", "c1"]
    with pytest.raises(WriterError, match="writer_reference_invalid"):
        selected_parse(json.dumps(wire), schema)


def test_repair_projection_uses_exact_host_directory_without_duplicate_prose():
    body, targets, _, _, _ = repair_case()
    payload = {
        "current_body": body,
        "body_sha256": digest(body),
        "targets": targets,
        "cause": "capacity",
        "previous_local_failure": None,
    }
    before = copy.deepcopy(payload)
    wire = repair_projection(payload, envelope())
    assert payload == before
    assert wire["current_body"] == body
    assert wire["targets"]["r1"]["original_unit_ids"] == [
        f"writer:{BODY[0]}:0:{i}" for i in range(3)
    ]
    assert all("original" not in target for target in wire["targets"].values())
    old = repair_projection(payload, envelope(False))
    assert "transport_revision" not in old
    assert all("original_unit_ids" not in target for target in old["targets"].values())


def test_private_real_037_remains_invalid_after_mechanical_grouping():
    base = (
        Path(__file__).resolve().parents[1]
        / "artifacts/acceptance/reference-set-live-20261101/terminal-review"
    )
    raw_path = base / "vit-repair_0.original.raw.json"
    if not raw_path.exists():
        pytest.skip(
            "Private full 037 evidence is unavailable; public cases remain mandatory."
        )
    raw = raw_path.read_text()
    schema = json.loads((base / "vit-repair_0.exact-host-schema.json").read_text())
    targets = json.loads(
        (base / "vit-repair_0.exact-requested-targets.json").read_text()
    )
    before = hashlib.sha256(raw.encode()).hexdigest()
    with pytest.raises(WriterError, match="writer_flat_target_order_invalid"):
        parse(raw, schema, reference_contract=REFERENCE_SET_CONTRACT)
    result = selected_parse(raw, schema)
    assert len(result["bindings"]["units"]) == 43
    assert len(result["canonical"]["edit_records"]) == 43
    with pytest.raises(WriterError, match="writer_capacity_edit_position_invalid"):
        normalize_edit_map(result["canonical"], targets)
    assert (
        restore_reference_input(
            result["canonical"],
            result["bindings"],
            transport_revision=TRANSPORT_REVISION,
        )
        == result["received_canonical"]
    )
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == before


def test_private_real_037_complete_recovery_uses_actual_builder_and_full_output():
    """Replay all private rows when available; never substitute the proposed wire."""
    from types import SimpleNamespace

    from litagent.body_quality import NATIVE_OWNERSHIP_INSTRUCTION
    from litagent.eval.inventory_recovery import MIXED_CONTRACT, recovery_instruction
    from litagent.eval.scoped import (
        FLAT_INVENTORY_PROMPT,
        LOCATED_INVENTORY_INSTRUCTION,
        check_request_budget,
    )
    from litagent.safety.planned_budget import CONTEXT_LIMIT, _digest
    from litagent.writing_v2 import strict_request_reservation

    base = (
        Path(__file__).resolve().parents[1]
        / "artifacts/acceptance/reference-set-live-20261101/terminal-review"
    )
    path = base / "few-shot.actual-recovery-canonical-payload.json"
    if not path.exists():
        pytest.skip(
            "Private full 037 evidence is unavailable; public ownership tests remain mandatory."
        )
    payload = json.loads(path.read_text())
    recorded = json.loads(
        (base / "few-shot.actual-recovery-complete-wire.json").read_text()
    )
    before = copy.deepcopy(payload)
    fragments = {
        fragment["source_fragment_id"]: fragment
        for unit in payload["source_units"]
        for fragment in unit["source_fragments"]
    }
    aliases = {}
    for claim in payload["inventory_recovery"]["concern_claims"]:
        for eid, alias in zip(
            claim["evidence_ids"],
            fragments[claim["source_fragment_id"]]["citation_aliases"],
            strict=True,
        ):
            assert eid not in aliases or aliases[eid] == alias
            aliases[eid] = alias
    prompt = (
        FLAT_INVENTORY_PROMPT
        + LOCATED_INVENTORY_INSTRUCTION
        + recovery_instruction(MIXED_CONTRACT)
        + NATIVE_OWNERSHIP_INSTRUCTION
    )
    messages, native = build_evaluation_request(
        prompt, payload, recovery_evidence_aliases=aliases
    )
    assert messages == recorded["messages"]
    assert native == recorded["response_format"]
    assert strict_request_reservation(messages, 16384, native) == 127667
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_request_budget(
            SimpleNamespace(), messages, 16384, CONTEXT_LIMIT, response_format=native
        )
    selected = copy.deepcopy(payload)
    selected["writer_envelope"].update(
        transport_revision=TRANSPORT_REVISION, transport_identity=transport_identity()
    )
    selected["writer_envelope"]["identity"] = _digest(
        {k: v for k, v in selected["writer_envelope"].items() if k != "identity"}
    )
    projected, projected_native = build_evaluation_request(
        prompt, selected, recovery_evidence_aliases=aliases
    )
    old_wire = json.loads(messages[1]["content"])
    new_wire = json.loads(projected[1]["content"])
    assert projected_native == native
    assert payload == before
    for name, count in zip(
        ("findings", "concern_claims", "global_finding_bindings"),
        (51, 60, 45),
        strict=True,
    ):
        decoded = decode_field_table(new_wire["inventory_recovery"][name])
        assert decoded == old_wire["inventory_recovery"][name]
        assert len(decoded) == count
    assert new_wire["report_layout"] == old_wire["report_layout"]
    assert new_wire["source_units"] == old_wire["source_units"]
    assert (
        check_request_budget(
            SimpleNamespace(),
            projected,
            16384,
            CONTEXT_LIMIT,
            response_format=projected_native,
        )
        == strict_request_reservation(projected, 16384, native)
        < CONTEXT_LIMIT
    )
    for name in ("findings", "global_finding_bindings"):
        for row in selected["inventory_recovery"][name]:
            row["explanation"] = "Preserve this complete condition and negation. " * 44
            row["explanation"] = row["explanation"][:2000]
    long_messages, long_native = build_evaluation_request(
        prompt, selected, recovery_evidence_aliases=aliases
    )
    assert all(
        len(
            decode_field_table(
                json.loads(long_messages[1]["content"])["inventory_recovery"][name]
            )
        )
        == count
        for name, count in (("findings", 51), ("global_finding_bindings", 45))
    )
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_request_budget(
            SimpleNamespace(),
            long_messages,
            16384,
            CONTEXT_LIMIT,
            response_format=long_native,
        )


@pytest.mark.parametrize("stage", ["chapter", "summary"])
def test_non_repair_units_keep_order_under_the_same_explicit_revision(stage):
    from litagent.writing_v2 import body_section_schema

    field = "fact_card_refs" if stage == "chapter" else "body_unit_refs"
    refs = ["c1", "c2", "c3"]
    unit = {"text": "A complete condition remains.", field: refs[:2]}
    if stage == "chapter":
        schema = body_section_schema("methods", refs, maximum=8)
        canonical = {"methods": [{"sentences": [unit]}]}
    else:
        schema = schema_for("summary", refs)
        canonical = {
            key: [{"sentences": [copy.deepcopy(unit)]}]
            for key in ("abstract", "conclusion")
        }
    wire = encode(canonical, schema)
    for values in wire.values():
        if isinstance(values, list):
            for value in values:
                value[field] = ["c1", "c2", "c1"]
    new = selected_parse(json.dumps(wire), schema)
    old = parse(json.dumps(wire), schema, reference_contract=REFERENCE_SET_CONTRACT)
    assert new["canonical"] == old["canonical"] == canonical
    assert new["received_canonical"] == old["received_canonical"]
    assert (
        restore_reference_input(
            new["canonical"], new["bindings"], transport_revision=TRANSPORT_REVISION
        )
        == new["received_canonical"]
    )
    with pytest.raises(WriterError, match="writer_reference_binding_invalid"):
        restore_reference_input(
            old["canonical"], old["bindings"], transport_revision=TRANSPORT_REVISION
        )


def test_new_repair_directory_never_accepts_a_forged_body_or_revision():
    body, targets, _, _, _ = repair_case()
    payload = {
        "current_body": body,
        "body_sha256": "0" * 64,
        "targets": targets,
        "cause": "capacity",
        "previous_local_failure": None,
    }
    with pytest.raises(WriterError, match="writer_patch_identity_invalid"):
        repair_projection(payload, envelope())
    payload.update(body_sha256=digest(body), transport_revision="future")
    with pytest.raises(ValueError, match="host_transport_revision_invalid"):
        repair_projection(payload, envelope())
