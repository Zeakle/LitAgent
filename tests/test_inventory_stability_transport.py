"""Share stable inventory metadata without changing prose, ownership or replies."""

import copy
import json

import pytest

from litagent.eval.scoped import flat_inventory_schema
from litagent.quality import digest
from litagent.writing_flat import (
    INVENTORY_STABILITY_TRANSPORT_INSTRUCTION,
    decode_field_table,
    encode_field_table,
    evaluation_projection,
    inventory_identity_projection,
    inventory_stability_projection_identity,
    inventory_stability_transport_projection,
    inventory_transport_projection,
    reference_set_identity,
    restore_inventory_stability_transport,
    transport_identity,
)


def encoded(value):
    """Compare exact object order and every escaped character, not just equality."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def cell(columns, key):
    """Locate one real field-table cell without copying its nested values."""
    return columns.index(key) + 1


def shared_unit_and_fragments(projected, number):
    """Expose exact wire rows to make corruption controls hit the real codec."""
    table = projected["source_units"]
    row = table["rows"][number]
    fields = table["columns"][row[0]]
    return row, fields, row[cell(fields, "source_fragments")]


def plain_wire(*, tables=True):
    """Keep ordered citation differences, identical lists, empties and all prose."""
    rows = [
        {
            "source_fragment_id": "f1",
            "assertions": [
                {"kind": "scientific", "claim_text": '汉字 \\ "\t\n 数值2%仍是2%。'},
                {
                    "kind": "runtime_method",
                    "claim_text": "A conditional method remains.",
                },
            ],
            "exclusion_reason": "",
        },
        {
            "source_fragment_id": "f2",
            "assertions": [],
            "exclusion_reason": '空片段保留原理由 \\ "\n',
        },
        {
            "source_fragment_id": "f3",
            "assertions": [{"kind": "report_structure", "claim_text": "Scope only."}],
            "exclusion_reason": "",
        },
    ]
    findings = [
        {
            "source_unit_id": "v1",
            "source_fragment_ids": ["f0"],
            "claim_ids": ["p1"],
            "code": "non_atomic",
            "explanation": "No old scientific judgment is a gold label.",
        }
    ]
    concerns = [{"claim_id": "p1", "optional": None, "flag": False, "empty": ""}]
    bindings = [{"finding_index": 0, "anchor_ids": ["claim:p1"], "extra": "保留"}]
    return {
        "inventory_output_contract": "host-flat-fragment-inventory-v1",
        "evaluation_input_contract": "flat-evaluation-input-v1",
        "allowed_citation_aliases": ["e1", "e2"],
        "source_units": [
            {
                "source_unit_id": "v1",
                "citation_aliases": ["e1", "e2"],
                "source_fragments": [
                    {"source_fragment_id": "f0", "citation_aliases": ["e1", "e2"]},
                    {"source_fragment_id": "f1", "citation_aliases": ["e2", "e1"]},
                ],
            },
            {
                "source_unit_id": "v2",
                "citation_aliases": [],
                "source_fragments": [
                    {"source_fragment_id": "f2", "citation_aliases": []}
                ],
            },
            {
                "source_unit_id": "v3",
                "citation_aliases": ["e1", "e2"],
                "source_fragments": [
                    {"source_fragment_id": "f3", "citation_aliases": ["e1", "e2"]}
                ],
            },
        ],
        "report_layout": [
            {"text": "## Full report heading\n\n"},
            {"source_fragment_id": "f0", "prose": "Actual editable complete prose."},
            {"text": "\nAll intervening qualification survives.\n"},
            {"source_fragment_id": "f1", "prose": '全文 汉字 "\\\t\n and limitations.'},
            {"source_fragment_id": "f2", "prose": "—"},
            {"source_fragment_id": "f3", "prose": "A complete scope statement."},
        ],
        "inventory_recovery": {
            "contract": "bounded-inventory-recovery-v2",
            "findings": encode_field_table(findings) if tables else findings,
            "concern_claims": encode_field_table(concerns) if tables else concerns,
            "global_finding_bindings": (
                encode_field_table(bindings) if tables else bindings
            ),
            "preserved_fragments": rows,
            "preserved_fragments_sha256": "b" * 64,
            "source_report_sha256": "a" * 64,
        },
        "unrelated_metadata": {
            "raw_positions": [0, 12, 42],
            "false": False,
            "null": None,
        },
    }


@pytest.mark.parametrize("tables", [False, True])
def test_whole_wire_roundtrip_keeps_all_text_citations_tables_and_input(tables):
    original = plain_wire(tables=tables)
    before = encoded(original)
    projected = inventory_stability_transport_projection(original)
    assert encoded(original) == before
    assert projected is not original
    assert projected["report_layout"] == original["report_layout"]
    assert projected["unrelated_metadata"] == original["unrelated_metadata"]
    assert projected["citation_alias_sets"] == [[0, 1], [1, 0], []]
    assert projected["preserved_assertion_columns"] == ["kind", "claim_text"]
    _, _, fragments = shared_unit_and_fragments(projected, 0)
    fields = projected["source_fragment_columns"][fragments[1][0]]
    assert fragments[1][cell(fields, "citation_aliases_ref")] == 1
    _, _, empty = shared_unit_and_fragments(projected, 1)
    assert (
        empty[0][
            cell(
                projected["source_fragment_columns"][empty[0][0]],
                "citation_aliases_ref",
            )
        ]
        == 2
    )
    rows = decode_field_table(projected["inventory_recovery"]["preserved_fragments"])
    assert rows[0]["assertions"][0] == [
        "scientific",
        original["inventory_recovery"]["preserved_fragments"][0]["assertions"][0][
            "claim_text"
        ],
    ]
    assert rows[1]["assertions"] == []
    assert (
        rows[1]["exclusion_reason"]
        == original["inventory_recovery"]["preserved_fragments"][1]["exclusion_reason"]
    )
    for key in ("findings", "concern_claims", "global_finding_bindings"):
        assert (
            projected["inventory_recovery"][key] == original["inventory_recovery"][key]
        )
    assert encoded(restore_inventory_stability_transport(projected)) == before


def test_source_fieldsets_preserve_every_additional_value_and_object_order():
    original = plain_wire()
    original["source_units"][0]["additional_unit_context"] = {
        "null": None,
        "flag": False,
        "range": [12, 45],
    }
    fragment = original["source_units"][0]["source_fragments"][0]
    fragment["additional_fragment_context"] = '原文 "\\\t\n 与限制。'
    fragment["character_range"] = [12, 45]
    second = original["source_units"][0]["source_fragments"][1]
    second["different_empty_field"] = []
    before = encoded(original)
    projected = inventory_stability_transport_projection(original)
    assert len(projected["source_units"]["columns"]) == 2
    assert len(projected["source_fragment_columns"]) == 3
    assert encoded(restore_inventory_stability_transport(projected)) == before
    assert encoded(original) == before


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_index",
        "bool_index",
        "negative_index",
        "string_index",
        "catalog_missing",
        "catalog_type",
        "catalog_entry_type",
        "unknown_alias",
        "duplicate_alias",
        "duplicate_catalog",
        "unused_catalog",
        "catalog_reordered",
        "ordinal_bool",
        "ordinal_negative",
        "ordinal_float",
        "old_catalog_strings",
        "missing_columns",
        "columns_type",
        "columns_wrong",
        "both_reference_and_list",
        "nested_assertion_type",
        "nested_assertion_size",
        "nested_kind",
        "nested_text",
        "duplicate_fragment",
        "missing_preserved",
        "extra_preserved",
        "bad_table",
        "cross_fragment_reference",
        "parent_order_changed",
        "source_columns_missing",
        "source_columns_type",
        "duplicate_source_fieldset",
        "unused_source_fieldset",
        "old_source_list",
        "unit_reference_missing",
        "fragment_reference_missing",
        "fragment_row_size",
        "unit_table_extra_field",
    ],
)
def test_corrupt_shared_wire_rejects_without_guessing_or_mutating(fault):
    projected = inventory_stability_transport_projection(plain_wire())
    unit, unit_fields, fragments = shared_unit_and_fragments(projected, 0)
    fragment = fragments[1]
    fragment_fields = projected["source_fragment_columns"][fragment[0]]
    ref = cell(fragment_fields, "citation_aliases_ref")
    table = projected["inventory_recovery"]["preserved_fragments"]
    if fault == "unknown_index":
        fragment[ref] = 999
    elif fault == "bool_index":
        fragment[ref] = True
    elif fault == "negative_index":
        fragment[ref] = -1
    elif fault == "string_index":
        fragment[ref] = "1"
    elif fault == "catalog_missing":
        projected.pop("citation_alias_sets")
    elif fault == "catalog_type":
        projected["citation_alias_sets"] = {"0": [0]}
    elif fault == "catalog_entry_type":
        projected["citation_alias_sets"][1] = "e2"
    elif fault == "unknown_alias":
        projected["citation_alias_sets"][1] = [2]
    elif fault == "duplicate_alias":
        projected["citation_alias_sets"][1] = [1, 1]
    elif fault == "duplicate_catalog":
        projected["citation_alias_sets"].append([0, 1])
    elif fault == "unused_catalog":
        projected["citation_alias_sets"].append([0])
    elif fault == "catalog_reordered":
        projected["citation_alias_sets"][:2] = projected["citation_alias_sets"][:2][
            ::-1
        ]
        for n in range(len(projected["source_units"]["rows"])):
            source_unit, fields, fs = shared_unit_and_fragments(projected, n)
            index = cell(fields, "citation_aliases_ref")
            if source_unit[index] in (0, 1):
                source_unit[index] = 1 - source_unit[index]
            for f in fs:
                index = cell(
                    projected["source_fragment_columns"][f[0]], "citation_aliases_ref"
                )
                if f[index] in (0, 1):
                    f[index] = 1 - f[index]
    elif fault == "ordinal_bool":
        projected["citation_alias_sets"][1] = [True, 0]
    elif fault == "ordinal_negative":
        projected["citation_alias_sets"][1] = [-1, 0]
    elif fault == "ordinal_float":
        projected["citation_alias_sets"][1] = [1.0, 0]
    elif fault == "old_catalog_strings":
        projected["citation_alias_sets"] = [["e1", "e2"], ["e2", "e1"], []]
    elif fault == "missing_columns":
        projected.pop("preserved_assertion_columns")
    elif fault == "columns_type":
        projected["preserved_assertion_columns"] = "kind,claim_text"
    elif fault == "columns_wrong":
        projected["preserved_assertion_columns"] = ["claim_text", "kind"]
    elif fault == "both_reference_and_list":
        projected["source_fragment_columns"].append(
            [*fragment_fields, "citation_aliases"]
        )
        fragment[0] = len(projected["source_fragment_columns"]) - 1
        fragment.append(["e2", "e1"])
    elif fault == "nested_assertion_type":
        table["rows"][0][2] = [{"kind": "scientific", "claim_text": "bad"}]
    elif fault == "nested_assertion_size":
        table["rows"][0][2][0].append("extra")
    elif fault == "nested_kind":
        table["rows"][0][2][0][0] = "supported"
    elif fault == "nested_text":
        table["rows"][0][2][0][1] = 7
    elif fault == "duplicate_fragment":
        fragment[cell(fragment_fields, "source_fragment_id")] = "f0"
    elif fault == "missing_preserved":
        table["rows"].pop()
    elif fault == "extra_preserved":
        table["rows"].append(copy.deepcopy(table["rows"][0]))
    elif fault == "bad_table":
        table["rows"][0][0] = True
    elif fault == "cross_fragment_reference":
        fragment[ref] = 0
    elif fault == "parent_order_changed":
        unit[cell(unit_fields, "citation_aliases_ref")] = 1
    elif fault == "source_columns_missing":
        projected.pop("source_fragment_columns")
    elif fault == "source_columns_type":
        projected["source_fragment_columns"] = "source_fragment_id,citation_aliases_ref"
    elif fault == "duplicate_source_fieldset":
        projected["source_fragment_columns"].append(copy.deepcopy(fragment_fields))
    elif fault == "unused_source_fieldset":
        projected["source_fragment_columns"].append(["unused"])
    elif fault == "old_source_list":
        projected["source_units"] = plain_wire()["source_units"]
    elif fault == "unit_reference_missing":
        for fields in projected["source_units"]["columns"]:
            index = cell(fields, "citation_aliases_ref")
            for row in projected["source_units"]["rows"]:
                row.pop(index)
            fields.remove("citation_aliases_ref")
    elif fault == "fragment_reference_missing":
        index = cell(fragment_fields, "citation_aliases_ref")
        for n in range(len(projected["source_units"]["rows"])):
            _, _, fs = shared_unit_and_fragments(projected, n)
            for f in fs:
                f.pop(index)
        fragment_fields.remove("citation_aliases_ref")
    elif fault == "fragment_row_size":
        fragment.append("unexpected")
    else:
        projected["source_units"]["unexpected"] = "extra"
    before = encoded(projected)
    with pytest.raises(ValueError, match="inventory_stability_transport_invalid"):
        restore_inventory_stability_transport(projected)
    assert encoded(projected) == before


@pytest.mark.parametrize(
    "fault",
    [
        "parent_order",
        "parent_missing",
        "fragment_duplicate_citation",
        "unknown_citation",
        "baseline_hash",
        "numeric_baseline_hash",
        "preexisting_catalog",
        "preexisting_fragment_columns",
        "missing_baseline",
        "protected_wrong_order",
    ],
)
def test_encoder_checks_real_parent_union_and_complete_baseline_first(fault):
    original = plain_wire()
    unit = original["source_units"][0]
    if fault == "parent_order":
        unit["citation_aliases"] = ["e2", "e1"]
    elif fault == "parent_missing":
        unit.pop("citation_aliases")
    elif fault == "fragment_duplicate_citation":
        unit["source_fragments"][0]["citation_aliases"] = ["e1", "e1"]
    elif fault == "unknown_citation":
        unit["source_fragments"][0]["citation_aliases"] = ["e3"]
    elif fault == "baseline_hash":
        original["inventory_recovery"]["preserved_fragments_sha256"] = None
    elif fault == "numeric_baseline_hash":
        original["inventory_recovery"]["preserved_fragments_sha256"] = int("1" * 64)
    elif fault == "preexisting_catalog":
        original["citation_alias_sets"] = []
    elif fault == "preexisting_fragment_columns":
        original["source_fragment_columns"] = []
    elif fault == "missing_baseline":
        original["inventory_recovery"].pop("preserved_fragments")
    else:
        original["inventory_recovery"]["preserved_fragments"].reverse()
    before = encoded(original)
    with pytest.raises(ValueError, match="inventory_stability_transport_invalid"):
        inventory_stability_transport_projection(original)
    assert encoded(original) == before


def native_input():
    """Build an actual pre-alias native input with immutable host range metadata."""
    units = []
    for n, refs in enumerate((["e1", "e2"], ["e2", "e1"], [])):
        text = f'原文{n}: a full condition and numerical value 2%. "\\\n'
        text += " " + " ".join(f"[E:{a}]" for a in refs) if refs else ""
        units.append(
            {
                "source_unit_id": f"long-unit-{n}",
                "citation_aliases": refs,
                "source_fragments": [
                    {
                        "source_fragment_id": f"long-fragment-{n}",
                        "text": text,
                        "citation_aliases": refs,
                    }
                ],
                "scope_context": "Complete preceding context.",
            }
        )
    preserved = []
    for n in (1, 2):
        preserved.append(
            {
                "source_unit_id": f"long-unit-{n}",
                "source_fragment_id": f"long-fragment-{n}",
                "source_start": n * 100,
                "source_end": n * 100 + 80,
                "evidence_ids": ["b", "a"] if n == 1 else [],
                "assertions": (
                    [
                        {
                            "kind": "scientific",
                            "claim_text": "Keep the original2% and condition.",
                        }
                    ]
                    if n == 1
                    else []
                ),
                "exclusion_reason": "" if n == 1 else "Scope only.",
            }
        )
    recovery = {
        "contract": "bounded-inventory-recovery-v2",
        "source_report_sha256": "a" * 64,
        "preserved_fragments": preserved,
        "preserved_fragments_sha256": digest(preserved),
        "concern_claims": [
            {
                "claim_id": "old-1",
                "source_unit_id": "long-unit-0",
                "source_fragment_id": "long-fragment-0",
                "evidence_ids": ["a", "b"],
            }
        ],
        "findings": [
            {
                "source_unit_id": "long-unit-0",
                "source_fragment_ids": ["long-fragment-0"],
                "claim_ids": ["old-1"],
            }
        ],
        "global_finding_bindings": [],
    }
    payload = {
        "inventory_output_contract": "host-flat-fragment-inventory-v1",
        "evaluation_input_contract": "flat-evaluation-input-v1",
        "source_units": units,
        "allowed_citation_aliases": ["e1", "e2"],
        "inventory_recovery": recovery,
        "report_layout": [{"text": "## Methods\n"}]
        + [
            {
                "source_fragment_id": u["source_fragments"][0]["source_fragment_id"],
                "text": u["source_fragments"][0]["text"],
            }
            for u in units
        ],
    }
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "host_flat_fragment_inventory_v1",
            "strict": True,
            "schema": flat_inventory_schema(units),
        },
    }
    return evaluation_projection(payload), schema


def test_explicit_stable_projection_keeps_native_output_and_all_host_ranges():
    payload, schema = native_input()
    original, original_schema = encoded(payload), encoded(schema)
    aliases = {"a": "e1", "b": "e2"}
    old_wire, old_schema = inventory_identity_projection(
        payload, schema, evidence_aliases=aliases
    )
    false_wire, false_schema = inventory_identity_projection(
        payload, schema, evidence_aliases=aliases, stable=False
    )
    new_wire, new_schema = inventory_identity_projection(
        payload, schema, evidence_aliases=aliases, stable=True
    )
    assert encoded(old_wire) == encoded(false_wire)
    assert encoded(old_schema) == encoded(false_schema) == encoded(new_schema)
    assert encoded(restore_inventory_stability_transport(new_wire)) == encoded(old_wire)
    assert encoded(payload) == original and encoded(schema) == original_schema
    assert (
        payload["inventory_recovery"]["preserved_fragments"][0]["source_start"] == 100
    )
    assert "citation_alias_sets" not in old_wire
    # The existing three concern tables are outside the new encoding and survive.
    final = inventory_transport_projection(
        new_wire, transport_revision="host-lossless-transport-v1"
    )
    restored = restore_inventory_stability_transport(final)
    expected = inventory_transport_projection(
        old_wire, transport_revision="host-lossless-transport-v1"
    )
    assert encoded(restored) == encoded(expected)


def test_only_stability_identity_binds_new_instruction(monkeypatch):
    import litagent.writing_flat as flat
    from litagent.eval.inventory_recovery import mixed_inventory_recovery_identity

    before = (transport_identity(), reference_set_identity())
    old_mixed = mixed_inventory_recovery_identity()
    stable_mixed = mixed_inventory_recovery_identity(stable=True)
    stable_before = inventory_stability_projection_identity()
    monkeypatch.setattr(
        flat,
        "INVENTORY_STABILITY_TRANSPORT_INSTRUCTION",
        INVENTORY_STABILITY_TRANSPORT_INSTRUCTION + " changed",
    )
    assert inventory_stability_projection_identity() != stable_before
    assert (transport_identity(), reference_set_identity()) == before
    assert mixed_inventory_recovery_identity() == old_mixed
    assert mixed_inventory_recovery_identity(stable=True) != stable_mixed


@pytest.mark.parametrize("stable", [None, 0, 1, "true", [], {}])
def test_internal_selection_requires_real_bool(stable):
    payload, schema = native_input()
    with pytest.raises(ValueError, match="inventory_stability_transport_invalid"):
        inventory_identity_projection(
            payload, schema, evidence_aliases={"a": "e1", "b": "e2"}, stable=stable
        )
