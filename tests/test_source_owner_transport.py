"""Derive edit targets from exact source owners without changing authored bytes."""

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator
from test_source_map_wire import assert_field, source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
from litagent.capacity_repair import (
    SOURCE_OWNER_CONTRACT,
    normalize_source_map,
    original_unit_directory,
    source_edit_schema,
    source_map_identity,
)
from litagent.quality import digest
from litagent.safety.planned_budget import EnvelopeExceeded, nested_json_size
from litagent.writing_v2 import WriterError, provider_response_format, schema_for


@pytest.fixture
def owner_case(source_case):
    """Reuse independent positions and real canonical source-map consumers."""
    targets, old_schema, canonical = source_case
    schema = source_edit_schema(
        schema_for("repair", {"c1": {}}, targets),
        targets,
        contract=SOURCE_OWNER_CONTRACT,
    )
    return targets, old_schema, schema, canonical


def parse_owner(wire, schema, *, references=True, escaped=False):
    """Run the actual strict decoder and optional reference occurrence codec."""
    raw = json.dumps(wire, ensure_ascii=escaped)
    return raw, flat.parse(
        raw,
        schema,
        **(
            {
                "reference_contract": flat.REFERENCE_SET_CONTRACT,
                "transport_revision": flat.TRANSPORT_REVISION,
            }
            if references
            else {}
        ),
    )


@pytest.mark.parametrize("references", [False, True])
@pytest.mark.parametrize("escaped", [False, True])
def test_source_owner_roundtrip_binds_sources_without_fabricated_target_spans(
    owner_case, references, escaped
):
    targets, _, schema, canonical = owner_case
    wire = flat.encode(canonical, schema)
    assert set(wire["patches"][0]) == {"text", "fact_card_refs", "source_unit_ids"}
    raw, parsed = parse_owner(wire, schema, references=references, escaped=escaped)
    assert parsed["canonical"] == parsed["received_canonical"] == canonical
    bindings = parsed["source_relation_bindings"]
    directory_sha = digest(original_unit_directory(targets))
    assert bindings["contract"] == SOURCE_OWNER_CONTRACT
    assert bindings["wire_identity"] == flat.source_map_wire_identity(
        SOURCE_OWNER_CONTRACT
    )
    assert bindings["source_owner_directory_sha256"] == directory_sha
    assert bindings["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    for index, relation in enumerate(bindings["relations"]):
        assert relation["target_id"] == {
            "derived": True,
            "value": canonical["source_relations"][index]["target_id"],
            "source_owner_directory_sha256": directory_sha,
        }
        assert_field(raw, wire, relation["source_unit_ids"])
        for source in relation["source_ids"]:
            assert_field(raw, wire, source)
    for deletion in bindings["deleted_sources"]:
        for field in ("wire_binding", "unit_id", "reason"):
            assert_field(raw, wire, deletion[field])
    assert normalize_source_map(parsed["canonical"], targets)["removed_targets"] == [
        "r3"
    ]
    if references:
        for index, record in enumerate(parsed["bindings"]["units"]):
            assert "target_id_span" not in record
            assert record["target_id_derivation"] == {
                "rule": "unique-source-owner-v1",
                "source_owner_directory_sha256": directory_sha,
                "source_relation_index": index,
            }
        assert (
            flat.restore_reference_input(
                parsed["canonical"],
                parsed["bindings"],
                transport_revision=flat.TRANSPORT_REVISION,
            )
            == parsed["received_canonical"]
        )


def test_source_owner_native_schema_never_contains_host_directory(owner_case):
    _, _, schema, _ = owner_case
    before = copy.deepcopy(schema)
    native = provider_response_format("repair", schema, wire_contract="flat-writer-v1")
    encoded = json.dumps(native)
    assert "x-source-owner-directory" not in encoded
    assert "x-repair-mapping-contract" not in encoded
    assert (
        "target_id"
        not in native["json_schema"]["schema"]["properties"]["patches"]["items"][
            "properties"
        ]
    )
    assert schema == before


def test_source_owner_interleaving_repeated_text_uses_physical_source_ownership(
    owner_case,
):
    targets, _, schema, canonical = owner_case
    wire = flat.encode(canonical, schema)
    wire["patches"] = [wire["patches"][1], wire["patches"][0], wire["patches"][2]]
    _, parsed = parse_owner(wire, schema)
    assert [p["target_id"] for p in parsed["canonical"]["patches"]] == ["r2", "r1"]
    assert [
        r["replacement_index"] for r in parsed["canonical"]["source_relations"]
    ] == [0, 1, 0]
    assert [
        r["wire_ordinal"] for r in parsed["source_relation_bindings"]["relations"]
    ] == [0, 2, 1]
    assert normalize_source_map(parsed["canonical"], targets)["removed_targets"] == [
        "r3"
    ]
    assert (
        flat.restore_reference_input(
            parsed["canonical"],
            parsed["bindings"],
            transport_revision=flat.TRANSPORT_REVISION,
        )
        == parsed["received_canonical"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "string",
        "null",
        "bool",
        "unknown",
        "duplicate",
        "mixed",
        "extra_target",
    ],
)
def test_source_owner_bad_sources_or_redundant_target_are_not_repaired(
    owner_case, fault
):
    targets, _, schema, canonical = owner_case
    wire = flat.encode(canonical, schema)
    patch = wire["patches"][0]
    ids = original_unit_directory(targets)
    if fault == "empty":
        patch["source_unit_ids"] = []
    elif fault == "string":
        patch["source_unit_ids"] = ids["r1"][0]
    elif fault == "null":
        patch["source_unit_ids"] = None
    elif fault == "bool":
        patch["source_unit_ids"] = [True]
    elif fault == "unknown":
        patch["source_unit_ids"] = ["writer:unknown:0:0"]
    elif fault == "duplicate":
        patch["source_unit_ids"] *= 2
    elif fault == "mixed":
        patch["source_unit_ids"].append(ids["r2"][0])
    else:
        patch["target_id"] = "r1"
    before = copy.deepcopy(wire)
    with pytest.raises(WriterError):
        parse_owner(wire, schema)
    assert wire == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing_contract",
        "wrong_contract",
        "missing_directory",
        "overlap",
        "empty",
        "bool",
        "wrong_target",
        "source_enum",
        "deletion_enum",
    ],
)
def test_source_owner_host_directory_must_match_both_canonical_enums(owner_case, fault):
    _, _, schema, _ = owner_case
    if fault == "missing_contract":
        schema.pop("x-repair-mapping-contract")
    elif fault == "wrong_contract":
        schema["x-repair-mapping-contract"] = "capacity-source-map-v3"
    elif fault == "missing_directory":
        schema.pop("x-source-owner-directory")
    elif fault == "overlap":
        schema["x-source-owner-directory"]["r2"] = schema["x-source-owner-directory"][
            "r1"
        ]
    elif fault == "empty":
        schema["x-source-owner-directory"]["r1"] = []
    elif fault == "bool":
        schema["x-source-owner-directory"]["r1"] = [True]
    elif fault == "wrong_target":
        schema["x-source-owner-directory"]["r99"] = schema[
            "x-source-owner-directory"
        ].pop("r1")
    elif fault == "source_enum":
        schema["properties"]["source_relations"]["items"]["properties"][
            "source_unit_ids"
        ]["items"]["enum"].pop()
    else:
        schema["properties"]["deleted_sources"]["items"]["properties"]["unit_id"][
            "enum"
        ].pop()
    with pytest.raises(WriterError):
        flat.prepare(schema)


def test_source_owner_schema_survives_sorted_checkpoint_json(owner_case):
    _, _, schema, canonical = owner_case
    restored = json.loads(json.dumps(schema, sort_keys=True))
    wire = flat.encode(canonical, restored)
    _, parsed = parse_owner(wire, restored)
    assert parsed["canonical"] == canonical


def test_source_owner_producer_rejects_original_hash_drift(owner_case):
    targets, _, _, _ = owner_case
    targets["r1"]["original_sha256"] = "0" * 64
    with pytest.raises(WriterError, match="writer_patch_identity_invalid"):
        source_edit_schema(
            schema_for("repair", {"c1": {}}, targets),
            targets,
            contract=SOURCE_OWNER_CONTRACT,
        )


@pytest.mark.parametrize("fault", ["missing_original", "deletion_overlap"])
def test_source_owner_still_uses_real_complete_coverage_consumer(owner_case, fault):
    targets, _, schema, canonical = owner_case
    wire = flat.encode(canonical, schema)
    if fault == "missing_original":
        wire["patches"][0]["source_unit_ids"].pop()
    else:
        wire["deleted_sources"].append(
            {
                "unit_id": wire["patches"][0]["source_unit_ids"][0],
                "reason": "Declared deletion.",
            }
        )
    _, parsed = parse_owner(wire, schema)
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        normalize_source_map(parsed["canonical"], targets)


def test_source_owner_complete_deletion_has_no_fabricated_patch(owner_case):
    targets, _, schema, _ = owner_case
    wire = {
        "outcome": "patched",
        "explanation": "Explicitly remove redundant originals.",
        "patches": [],
        "deleted_sources": [
            {"unit_id": uid, "reason": "Redundant content."}
            for uid in original_unit_directory(targets)["r1"]
        ],
    }
    _, parsed = parse_owner(wire, schema)
    assert parsed["canonical"]["patches"] == []
    assert normalize_source_map(parsed["canonical"], targets)["removed_targets"] == [
        "r1"
    ]


def test_source_owner_complete_canonical_metadata_keeps_its_existing_limit(owner_case):
    targets, _, schema, canonical = owner_case
    _, parsed = parse_owner(flat.encode(canonical, schema), schema, escaped=True)
    size = nested_json_size(
        {
            key: parsed["canonical"][key]
            for key in ("source_relations", "deleted_sources")
        }
    )
    normalize_source_map(parsed["canonical"], targets, max_json_bytes=size)
    with pytest.raises(EnvelopeExceeded):
        normalize_source_map(parsed["canonical"], targets, max_json_bytes=size - 1)


@pytest.mark.parametrize(
    "fault",
    [
        "fake_span",
        "rule",
        "directory_sha",
        "relation_index",
        "bool_index",
        "missing_marker",
    ],
)
def test_source_owner_reference_restore_rejects_resealed_derivation(owner_case, fault):
    _, _, schema, canonical = owner_case
    _, parsed = parse_owner(flat.encode(canonical, schema), schema)
    bindings = parsed["bindings"]
    record = bindings["units"][0]
    if fault == "fake_span":
        record["target_id_span"] = {"start": 0, "end": 1}
    elif fault == "rule":
        record["target_id_derivation"]["rule"] = "guess-from-text"
    elif fault == "directory_sha":
        record["target_id_derivation"]["source_owner_directory_sha256"] = "0" * 64
    elif fault == "relation_index":
        record["target_id_derivation"]["source_relation_index"] = 1
    elif fault == "bool_index":
        record["target_id_derivation"]["source_relation_index"] = False
    else:
        bindings.pop("repair_mapping_contract")
    with pytest.raises(WriterError, match="writer_reference_binding_invalid"):
        flat.restore_reference_input(
            parsed["canonical"], bindings, transport_revision=flat.TRANSPORT_REVISION
        )


def test_source_owner_selection_never_upgrades_old_wire_or_default_identity(owner_case):
    _, old_schema, new_schema, canonical = owner_case
    old_wire = flat.encode(canonical, old_schema)
    new_wire = flat.encode(canonical, new_schema)
    with pytest.raises(WriterError, match="writer_flat_schema_invalid"):
        parse_owner(old_wire, new_schema)
    with pytest.raises(WriterError, match="writer_flat_schema_invalid"):
        parse_owner(new_wire, old_schema)
    assert (
        flat.source_map_wire_identity()
        == "03a9471fc5e12065a4624b364dd1bedd5f8edface3df25cf8edca9c731a60929"
    )
    assert (
        flat.source_map_wire_identity(SOURCE_OWNER_CONTRACT)
        != flat.source_map_wire_identity()
    )
    assert source_map_identity(SOURCE_OWNER_CONTRACT) != source_map_identity()
    assert (
        digest(flat.prepare(old_schema)["host_schema"])
        == "8697d94cba829d328de6eeb19820cf617ca8a861179dcb0dff92498ea7cfd1dd"
    )
    Draft202012Validator(flat.prepare(old_schema)["wire_schema"]).validate(old_wire)


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_identity",
        "missing_contract",
        "missing_capacity_rule",
        "unknown_map",
    ],
)
def test_source_owner_complete_selected_envelope_reaches_structured_wire_guard(
    case, fault
):
    """Exercise the real V4 envelope instead of only the isolated response codec."""
    from test_source_owner_pipeline import FLAGS

    from litagent.writing_v2 import writer_budget_plan

    envelope = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **FLAGS,
    )["envelope"]
    if fault is None:
        assert flat._guidance_wire_selected(envelope) is True
        return
    if fault == "missing_identity":
        envelope.pop("repair_guidance_wire_identity")
    elif fault == "missing_contract":
        envelope.pop("repair_guidance_contract")
    elif fault == "missing_capacity_rule":
        envelope.pop("repair_guidance_capacity_rule")
    else:
        envelope["repair_mapping_contract"] = "capacity-source-map-v5"
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat._guidance_wire_selected(envelope)
