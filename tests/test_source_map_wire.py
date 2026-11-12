"""Keep source lineage outside prose while deriving target-local positions."""

import copy
import hashlib
import json

import pytest
from jsonschema import Draft202012Validator, ValidationError

import litagent.writing_flat as flat
from litagent.capacity_repair import (
    SOURCE_MAP_CONTRACT,
    capacity_targets,
    edit_schema,
    normalize_source_map,
    original_unit_directory,
    source_edit_schema,
)
from litagent.quality import digest
from litagent.safety.planned_budget import EnvelopeExceeded, nested_json_size
from litagent.writing_v2 import BODY, WriterError, schema_for


@pytest.fixture
def source_case():
    """Create distinct positions with identical text, not scientific gold."""
    unit = {"text": "同文但位置独立。", "fact_card_refs": ["c1"]}
    body = {
        "title": "Source-map transport fixture",
        **{section: [{"sentences": [copy.deepcopy(unit)]}] for section in BODY},
    }
    body["introduction"][0]["sentences"].append(copy.deepcopy(unit))
    targets = capacity_targets(body, {}, maximum=66)
    directory = original_unit_directory(targets)
    schema = source_edit_schema(schema_for("repair", {"c1": {}}, targets), targets)
    value = {
        "outcome": "patched",
        "explanation": "仅用于结构契约，不代表科学支持。",
        "patches": [
            {"target_id": "r1", "sentences": [copy.deepcopy(unit)]},
            {
                "target_id": "r2",
                "sentences": [copy.deepcopy(unit), copy.deepcopy(unit)],
            },
        ],
        "source_relations": [
            {
                "target_id": "r1",
                "replacement_index": 0,
                "source_unit_ids": directory["r1"],
            },
            *[
                {
                    "target_id": "r2",
                    "replacement_index": index,
                    "source_unit_ids": directory["r2"],
                }
                for index in range(2)
            ],
        ],
        "deleted_sources": [
            {"unit_id": directory["r3"][0], "reason": "删除非核心重复项 r1。"}
        ],
    }
    return targets, schema, value


def parse_wire(wire, schema, *, references=True, escaped=False):
    """Use the real tokenizer, schemas, binding producer and reference guard."""
    raw = json.dumps(wire, ensure_ascii=escaped)
    result = flat.parse(
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
    return raw, result


def assert_field(raw, wire, field):
    """Verify original token ranges rather than searching repeated text."""
    expected = wire
    for part in field["wire_path"]:
        expected = expected[part]
    literal = raw[field["start"] : field["end"]]
    assert json.loads(literal) == expected
    assert field["value_sha256"] == digest(expected)
    assert field["raw_sha256"] == hashlib.sha256(literal.encode()).hexdigest()


def test_source_map_native_shape_has_no_model_managed_positions(source_case):
    targets, schema, value = source_case
    plan = flat.prepare(schema)
    wire = flat.encode(value, schema)
    Draft202012Validator(plan["wire_schema"]).validate(wire)
    assert plan["kind"] == "repair"
    assert plan["repair_mapping_contract"] == SOURCE_MAP_CONTRACT
    assert set(wire) == {"outcome", "explanation", "patches", "deleted_sources"}
    assert set(wire["patches"][0]) == {
        "target_id",
        "text",
        "fact_card_refs",
        "source_unit_ids",
    }
    assert plan["wire_schema"]["properties"]["patches"]["maxItems"] == 66
    assert "source_map_json_bytes" in json.dumps(plan["wire_schema"])
    assert "replacement_index" not in json.dumps(plan["wire_schema"])
    _, parsed = parse_wire(wire, schema)
    assert parsed["canonical"] == parsed["received_canonical"] == value
    normalized = normalize_source_map(parsed["canonical"], targets)
    assert normalized["removed_targets"] == ["r3"]
    assert [record["replacement_indexes"] for record in normalized["edit_records"]] == [
        [0],
        [0],
        [0, 1],
        [],
        [0],
        [0],
    ]
    assert normalized["patches"][-2:] == [
        {"target_id": tid, "sentences": targets[tid]["original"]}
        for tid in ("r4", "r5")
    ]
    assert parsed["source_relation_bindings"]["scientific_assessment"] == "not_verified"


@pytest.mark.parametrize("references", [False, True])
@pytest.mark.parametrize("escaped", [False, True])
def test_source_map_interleaving_has_exact_independent_raw_bindings(
    source_case, references, escaped
):
    _, schema, value = source_case
    wire = flat.encode(value, schema)
    wire["patches"] = [wire["patches"][1], wire["patches"][0], wire["patches"][2]]
    raw, parsed = parse_wire(wire, schema, references=references, escaped=escaped)
    canonical = parsed["canonical"]
    assert [p["target_id"] for p in canonical["patches"]] == ["r2", "r1"]
    assert [r["replacement_index"] for r in canonical["source_relations"]] == [0, 1, 0]
    assert all(
        set(unit) == {"text", "fact_card_refs"}
        for view in (canonical, parsed["received_canonical"])
        for patch in view["patches"]
        for unit in patch["sentences"]
    )
    bindings = parsed["source_relation_bindings"]
    assert bindings["contract"] == SOURCE_MAP_CONTRACT
    assert bindings["wire_identity"] == flat.source_map_wire_identity()
    assert bindings["raw_sha256"] == hashlib.sha256(raw.encode()).hexdigest()
    assert bindings["raw_length"] == len(raw)
    assert bindings["canonical_map_sha256"] == digest(
        {key: canonical[key] for key in ("source_relations", "deleted_sources")}
    )
    assert [entry["wire_ordinal"] for entry in bindings["relations"]] == [0, 2, 1]
    for index, entry in enumerate(bindings["relations"]):
        assert entry["canonical_path"] == ["source_relations", index]
        assert entry["canonical_relation_sha256"] == digest(
            canonical["source_relations"][index]
        )
        for name in ("target_id", "source_unit_ids"):
            assert_field(raw, wire, entry[name])
        for source_id in entry["source_ids"]:
            assert_field(raw, wire, source_id)
    for entry in bindings["deleted_sources"]:
        assert entry["canonical_deletion_sha256"] == digest(
            canonical["deleted_sources"][0]
        )
        for name in ("wire_binding", "unit_id", "reason"):
            assert_field(raw, wire, entry[name])
    assert "source_unit_ids" not in json.dumps(parsed["bindings"])


def test_source_map_reference_occurrences_never_include_lineage_fields(source_case):
    _, schema, value = source_case
    wire = flat.encode(value, schema)
    wire["patches"][0]["fact_card_refs"] = ["c1", "c1"]
    # This source schema deliberately allows two received paper references.
    schema["properties"]["patches"]["items"]["properties"]["sentences"]["items"][
        "properties"
    ]["fact_card_refs"]["maxItems"] = 2
    _, parsed = parse_wire(wire, schema)
    assert parsed["canonical"]["patches"][0]["sentences"][0]["fact_card_refs"] == ["c1"]
    assert parsed["received_canonical"]["patches"][0]["sentences"][0][
        "fact_card_refs"
    ] == ["c1", "c1"]
    assert (
        flat.restore_reference_input(
            parsed["canonical"],
            parsed["bindings"],
            transport_revision=flat.TRANSPORT_REVISION,
        )
        == parsed["received_canonical"]
    )
    assert parsed["source_relation_bindings"]["canonical_map_sha256"] == digest(
        {
            key: parsed["canonical"][key]
            for key in ("source_relations", "deleted_sources")
        }
    )


@pytest.mark.parametrize(
    "fault",
    [
        "index",
        "old_map",
        "root_relations",
        "unit_kind",
        "missing_source",
        "empty_source",
        "unknown_source",
        "reason_type",
        "extra_delete",
    ],
)
def test_source_map_rejects_mixed_or_invalid_native_shapes(source_case, fault):
    _, schema, value = source_case
    wire = flat.encode(value, schema)
    if fault == "index":
        wire["patches"][0]["replacement_index"] = 0
    elif fault == "old_map":
        wire["edit_records"] = []
    elif fault == "root_relations":
        wire["source_relations"] = []
    elif fault == "unit_kind":
        wire["patches"][0]["kind"] = "scientific"
    elif fault == "missing_source":
        wire["patches"][0].pop("source_unit_ids")
    elif fault == "empty_source":
        wire["patches"][0]["source_unit_ids"] = []
    elif fault == "unknown_source":
        wire["patches"][0]["source_unit_ids"] = ["writer:unknown:0:0"]
    elif fault == "reason_type":
        wire["deleted_sources"][0]["reason"] = False
    elif fault == "extra_delete":
        wire["deleted_sources"][0]["replacement_indexes"] = []
    before = copy.deepcopy(wire)
    with pytest.raises(WriterError):
        parse_wire(wire, schema)
    assert wire == before


@pytest.mark.parametrize(
    "fault",
    [
        "cross_target",
        "duplicate_source",
        "missing_original",
        "delete_linked",
        "duplicate_delete",
        "declined_with_patch",
    ],
)
def test_source_map_semantic_lineage_remains_a_separate_strict_consumer(
    source_case, fault
):
    targets, schema, value = source_case
    wire = flat.encode(value, schema)
    if fault == "cross_target":
        wire["patches"][0]["source_unit_ids"] = wire["patches"][1]["source_unit_ids"]
    elif fault == "duplicate_source":
        wire["patches"][0]["source_unit_ids"] *= 2
    elif fault == "missing_original":
        wire["patches"][0]["source_unit_ids"].pop()
    elif fault == "delete_linked":
        wire["deleted_sources"].append(
            {
                "unit_id": wire["patches"][0]["source_unit_ids"][0],
                "reason": "Repeated ownership.",
            }
        )
    elif fault == "duplicate_delete":
        wire["deleted_sources"] *= 2
    elif fault == "declined_with_patch":
        wire["outcome"] = "declined"
    _, parsed = parse_wire(wire, schema)
    before = copy.deepcopy(parsed)
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        normalize_source_map(parsed["canonical"], targets)
    assert parsed == before


def test_source_map_metadata_limit_counts_complete_host_map_without_prose(source_case):
    targets, schema, value = source_case
    _, parsed = parse_wire(flat.encode(value, schema), schema, escaped=True)
    canonical = parsed["canonical"]
    size = nested_json_size(
        {key: canonical[key] for key in ("source_relations", "deleted_sources")}
    )
    normalize_source_map(canonical, targets, max_json_bytes=size)
    with pytest.raises(EnvelopeExceeded) as raised:
        normalize_source_map(canonical, targets, max_json_bytes=size - 1)
    assert raised.value.diagnostics == [
        {
            "stage": "repair",
            "field": "source_map_json_bytes",
            "actual": size,
            "maximum": size - 1,
        }
    ]
    assert canonical == value


@pytest.mark.parametrize("fault", ["duplicate", "orphan", "missing", "bool_index"])
def test_source_map_encoder_refuses_nonbijective_host_relations(source_case, fault):
    _, schema, value = source_case
    if fault == "duplicate":
        value["source_relations"].append(copy.deepcopy(value["source_relations"][0]))
    elif fault == "orphan":
        value["source_relations"][0]["replacement_index"] = 2
    elif fault == "missing":
        value["source_relations"].pop()
    elif fault == "bool_index":
        value["source_relations"][0]["replacement_index"] = False
    with pytest.raises((WriterError, ValidationError)):
        flat.encode(value, schema)


def test_source_map_declined_is_empty_and_still_has_raw_metadata(source_case):
    targets, schema, _ = source_case
    value = {
        "outcome": "declined",
        "explanation": "Insufficient frozen material.",
        "patches": [],
        "source_relations": [],
        "deleted_sources": [],
    }
    _, parsed = parse_wire(flat.encode(value, schema), schema)
    assert parsed["canonical"] == value
    assert parsed["source_relation_bindings"]["relations"] == []
    assert normalize_source_map(value, targets)["patches"] == []


@pytest.mark.parametrize(
    "instruction",
    [
        "SOURCE_MAP_FLAT_INSTRUCTION",
        "SOURCE_MAP_TRANSPORT_INSTRUCTION",
        "SOURCE_MAP_CAPACITY_INSTRUCTION",
        "SOURCE_MAP_FIELD_DESCRIPTION",
    ],
)
def test_source_map_identity_isolated_from_legacy_instructions_and_schemas(
    source_case, monkeypatch, instruction
):
    targets, _, _ = source_case
    schema = edit_schema(
        schema_for("repair", {"c1": {}}, targets), targets, mapping=True
    )
    plan = flat.prepare(schema)
    assert "repair_mapping_contract" not in plan
    assert "source_unit_ids" not in json.dumps(plan["wire_schema"])
    assert (
        flat.transport_identity()
        == "bf34eeef27df5b7681a0fe6fad36758b69e5e059bd70d9c2447d1d086a83284b"
    )
    assert (
        flat.reference_set_identity()
        == "044e1d3757b38eac99d04baca3d3c5062c4f9fa4cf0c6f145303d93da9a29a83"
    )
    assert (
        digest(flat.INSTRUCTION)
        == "b3fe335223e35114169afd62578e30bb0ab25b6a85b66814ab5f25c26181f111"
    )
    assert (
        digest(flat.TRANSPORT_INSTRUCTION)
        == "811482df8d48b22496ce76326e157710d5f56f0dbd83c15a81426b227ec33230"
    )
    before = flat.source_map_wire_identity()
    monkeypatch.setattr(
        flat,
        instruction,
        getattr(flat, instruction) + " Exact rule mutation.",
    )
    assert flat.source_map_wire_identity() != before
    assert flat.prepare(schema) == plan
    assert (
        flat.transport_identity()
        == "bf34eeef27df5b7681a0fe6fad36758b69e5e059bd70d9c2447d1d086a83284b"
    )


def test_source_map_native_id_enums_have_exact_host_preserving_inverse(source_case):
    targets, schema, _ = source_case
    before = copy.deepcopy(schema)
    plan = flat.prepare(schema)
    original_ids = [
        uid for ids in original_unit_directory(targets).values() for uid in ids
    ]
    props = plan["wire_schema"]["properties"]
    source = props["patches"]["items"]["properties"]["source_unit_ids"]["items"]
    deletion = props["deleted_sources"]["items"]["properties"]["unit_id"]
    assert source == deletion == {"type": "string"}
    assert schema == before
    assert (
        digest(plan["host_schema"])
        == "8697d94cba829d328de6eeb19820cf617ca8a861179dcb0dff92498ea7cfd1dd"
    )
    restored = copy.deepcopy(plan["wire_schema"])
    restored["properties"]["patches"]["items"]["properties"]["source_unit_ids"][
        "items"
    ]["enum"] = original_ids
    restored["properties"]["deleted_sources"]["items"]["properties"]["unit_id"][
        "enum"
    ] = original_ids
    assert (
        digest(restored)
        == "99223ab6bd590f3f7c9ac6916c3eeff1794e2d5a4407a86b42f0f20a6dafde8e"
    )
    assert (
        plan["host_schema"]["properties"]["source_relations"]["items"]["properties"][
            "source_unit_ids"
        ]["items"]["enum"]
        == original_ids
    )
    assert (
        plan["host_schema"]["properties"]["deleted_sources"]["items"]["properties"][
            "unit_id"
        ]["enum"]
        == original_ids
    )


def test_source_map_native_mutation_cannot_pollute_host_or_other_plans(source_case):
    _, schema, _ = source_case
    before = copy.deepcopy(schema)
    plan = flat.prepare(schema)
    expected_host = copy.deepcopy(plan["host_schema"])
    fields = plan["wire_schema"]["properties"]
    fields["deleted_sources"]["items"]["properties"]["unit_id"]["enum"] = ["polluted"]
    fields["patches"]["items"]["properties"]["source_unit_ids"]["items"]["enum"] = [
        "polluted"
    ]
    fields["patches"]["items"]["properties"]["fact_card_refs"]["items"]["enum"] = [
        "polluted"
    ]
    assert plan["host_schema"] == expected_host
    assert schema == before
    assert flat.prepare(schema)["host_schema"] == expected_host
    assert "polluted" not in json.dumps(flat.prepare(schema))


@pytest.mark.parametrize("field", ["source", "deletion"])
@pytest.mark.parametrize("identifier", ["", "writer:unknown:0:0"])
def test_source_map_advisory_native_id_never_bypasses_host_enum(
    source_case, field, identifier
):
    _, schema, value = source_case
    wire = flat.encode(value, schema)
    if field == "source":
        wire["patches"][0]["source_unit_ids"] = [identifier]
    else:
        wire["deleted_sources"][0]["unit_id"] = identifier
    before = copy.deepcopy(wire)
    Draft202012Validator(flat.prepare(schema)["wire_schema"]).validate(wire)
    with pytest.raises(WriterError, match="writer_schema_invalid"):
        parse_wire(wire, schema)
    assert wire == before


@pytest.mark.parametrize("mapping", [False, True])
def test_source_map_native_dedup_does_not_change_legacy_plan_bytes(
    source_case, mapping
):
    targets, _, _ = source_case
    schema = schema_for("repair", {"c1": {}}, targets)
    if mapping:
        schema = edit_schema(schema, targets, mapping=True)
    expected = {
        False: "21fb8fdcb6f8620f900ed2958847844b7e94e5778955ef872e9a08176bfcefbb",
        True: "85e0eac534baed847cbc7a5509a551f16a3d2c1845e2cff0fdcc4b28a70e27de",
    }
    assert digest(flat.prepare(schema)) == expected[mapping]
