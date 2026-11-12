"""Keep declared source lineage strict while the host generates all positions."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from litagent import capacity_repair
from litagent.capacity_repair import (
    SOURCE_MAP_CONTRACT,
    SOURCE_MAP_HOST_REASON,
    capacity_targets,
    edit_schema,
    normalize_edit_map,
    normalize_source_map,
    original_unit_directory,
    source_edit_schema,
    source_map_identity,
    validate_edit_records,
)
from litagent.quality import digest
from litagent.safety.planned_budget import EnvelopeExceeded, nested_json_size
from litagent.writing_v2 import BODY, WriterError, classify_failure, schema_for


def unit(text):
    """Create content without declaring scientific validity."""
    return {"text": text, "fact_card_refs": ["c1"]}


@pytest.fixture
def targets():
    """Give one target several independently owned original units."""
    body = {
        "title": "Source relation fixture",
        **{section: [{"sentences": [unit(section)]}] for section in BODY},
    }
    body["methods"][0]["sentences"] = [unit("A"), unit("B"), unit("C")]
    return capacity_targets(body, {}, maximum=6)


def relation(tid, index, sources):
    """Represent an already host-enumerated canonical replacement relation."""
    return {
        "target_id": tid,
        "replacement_index": index,
        "source_unit_ids": list(sources),
    }


def response(targets, *, tid="r1", units=None, sources=None):
    """Change one target while leaving every other target unmentioned."""
    units = [unit("Replacement")] if units is None else units
    sources = original_unit_directory(targets)[tid] if sources is None else sources
    return {
        "outcome": "patched",
        "explanation": "Declared lineage, independently unverified.",
        "patches": [{"target_id": tid, "sentences": copy.deepcopy(units)}],
        "source_relations": [relation(tid, 0, sources)],
        "deleted_sources": [],
    }


def test_source_schema_has_only_clean_units_and_explicit_metadata(targets):
    """Do not expose host-generated positions as scientific unit properties."""
    base = schema_for("repair", {"c1": {}}, targets)
    before = copy.deepcopy(base)
    schema = source_edit_schema(base, targets)
    parsed = response(targets)
    Draft202012Validator(schema).validate(parsed)
    assert base == before
    assert list(schema["properties"]) == [
        "outcome",
        "explanation",
        "patches",
        "source_relations",
        "deleted_sources",
    ]
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    patch = schema["properties"]["patches"]["items"]["properties"]
    assert set(patch["sentences"]["items"]["properties"]) == {"text", "fact_card_refs"}
    assert schema["properties"]["source_relations"]["items"]["properties"][
        "source_unit_ids"
    ]["items"]["enum"] == [
        uid for ids in original_unit_directory(targets).values() for uid in ids
    ]


@pytest.mark.parametrize("mapping", [False, True])
def test_source_schema_rejects_legacy_mixture(targets, mapping):
    """New metadata cannot make an old position-bearing schema look selected."""
    old = edit_schema(
        schema_for("repair", {"c1": {}}, targets), targets, mapping=mapping
    )
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        source_edit_schema(old, targets)


def test_normalizer_preserves_untouched_targets_and_calls_legacy_guard(
    targets, monkeypatch
):
    """Derive old records only after every new source declaration is checked."""
    result = response(targets)
    before, original_targets = copy.deepcopy(result), copy.deepcopy(targets)
    seen = []
    real = capacity_repair.validate_edit_records

    def record_validation(value, frozen):
        seen.append(copy.deepcopy(value))
        return real(value, frozen)

    monkeypatch.setattr(capacity_repair, "validate_edit_records", record_validation)
    normalized = normalize_source_map(result, targets)
    assert len(seen) == 1
    assert set(normalized) == {
        "outcome",
        "explanation",
        "patches",
        "edit_records",
        "removed_targets",
    }
    assert normalized["edit_records"][0] == {
        "unit_id": "writer:introduction:0:0",
        "action": "rewritten",
        "reason": SOURCE_MAP_HOST_REASON,
        "replacement_indexes": [0],
    }
    assert normalized["removed_targets"] == []
    for patch in normalized["patches"][1:]:
        assert patch["sentences"] == targets[patch["target_id"]]["original"]
    assert all(
        record["reason"] == "Host-preserved unchanged target."
        for record in normalized["edit_records"][1:]
    )
    assert result == before and targets == original_targets


def test_sources_allow_merge_and_split_without_model_positions(targets):
    """Relation row order cannot change target-local replacement order."""
    a, b, c = original_unit_directory(targets)["r3"]
    result = response(
        targets, tid="r3", units=[unit("A+B"), unit("C first"), unit("C last")]
    )
    result["source_relations"] = [
        relation("r3", 2, [c]),
        relation("r3", 0, [b, a]),
        relation("r3", 1, [c]),
    ]
    normalized = normalize_source_map(result, targets)
    records = {r["unit_id"]: r for r in normalized["edit_records"]}
    assert records[a]["replacement_indexes"] == records[b]["replacement_indexes"] == [0]
    assert records[c]["replacement_indexes"] == [1, 2]
    assert all(records[uid]["action"] == "rewritten" for uid in (a, b, c))
    assert result["source_relations"][1]["source_unit_ids"] == [b, a]
    validate_edit_records(normalized, targets)


def test_explicit_deletion_preserves_reason_and_exact_retention(targets):
    """Deletion reasons stay model-authored; retention is a host comparison."""
    a, b, c = original_unit_directory(targets)["r3"]
    reason = "删除重复说明；保留“条件”与 Unicode 😀。"
    result = response(
        targets, tid="r3", units=[targets["r3"]["original"][2]], sources=[c]
    )
    result["deleted_sources"] = [
        {"unit_id": a, "reason": reason},
        {"unit_id": b, "reason": "Duplicate."},
    ]
    records = {
        r["unit_id"]: r for r in normalize_source_map(result, targets)["edit_records"]
    }
    assert records[a] == {
        "unit_id": a,
        "action": "removed",
        "reason": reason,
        "replacement_indexes": [],
    }
    assert records[c]["action"] == "retained"
    assert records[c]["replacement_indexes"] == [0]
    assert records[c]["reason"] == SOURCE_MAP_HOST_REASON


def test_all_originals_deleted_produces_removed_target_without_inference(targets):
    """An omitted patch alone never authorizes deleting its target."""
    result = response(targets)
    result["patches"], result["source_relations"] = [], []
    result["deleted_sources"] = [
        {"unit_id": uid, "reason": "Repeated non-core text."}
        for uid in original_unit_directory(targets)["r3"]
    ]
    normalized = normalize_source_map(result, targets)
    assert normalized["removed_targets"] == ["r3"]
    assert (
        next(p for p in normalized["patches"] if p["target_id"] == "r3")["sentences"]
        == []
    )
    validate_edit_records(normalized, targets)


def test_same_text_at_different_original_positions_is_not_implicit_coverage(targets):
    """Host IDs, rather than text matching, distinguish original obligations."""
    originals = [unit("Same"), unit("Same")]
    targets["r3"]["original"] = originals
    targets["r3"]["end"] = 2
    targets["r3"]["original_sha256"] = digest(originals)
    a, b = original_unit_directory(targets)["r3"]
    result = response(targets, tid="r3", units=[unit("Same")], sources=[a])
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        normalize_source_map(result, targets)
    result["source_relations"][0]["source_unit_ids"].append(b)
    normalized = normalize_source_map(result, targets)
    assert {r["unit_id"] for r in normalized["edit_records"]} >= {a, b}


@pytest.mark.parametrize(
    "fault",
    [
        "root_extra",
        "legacy_map",
        "root_type",
        "patches_type",
        "relations_type",
        "deletions_type",
        "outcome",
        "explanation",
        "patch_fields",
        "unknown_target",
        "duplicate_patch",
        "empty_units",
        "unit_type",
        "unit_metadata",
        "unit_text",
        "unit_references",
        "unit_reference_type",
        "relation_fields",
        "unknown_relation_target",
        "bool_position",
        "float_position",
        "negative_position",
        "high_position",
        "duplicate_relation",
        "missing_relation",
        "empty_sources",
        "duplicate_sources",
        "cross_target",
        "unknown_source",
        "source_type",
        "sources_type",
        "missing_original",
        "duplicate_deletion",
        "unknown_deletion",
        "mapped_and_deleted",
        "deletion_fields",
        "blank_reason",
        "long_reason",
        "reason_type",
        "deletion_id_type",
    ],
)
def test_source_map_rejects_untrusted_or_incomplete_relations(targets, fault):
    """No malformed metadata may become an accepted derived edit record."""
    result = response(targets)
    rel = result["source_relations"][0]
    uid = rel["source_unit_ids"][0]
    if fault == "root_extra":
        result["replacement_indexes"] = [0]
    elif fault == "legacy_map":
        result["edit_records"] = []
    elif fault == "root_type":
        result = []
    elif fault == "patches_type":
        result["patches"] = {}
    elif fault == "relations_type":
        result["source_relations"] = {}
    elif fault == "deletions_type":
        result["deleted_sources"] = {}
    elif fault == "outcome":
        result["outcome"] = "accepted"
    elif fault == "explanation":
        result["explanation"] = " "
    elif fault == "patch_fields":
        result["patches"][0]["source_unit_ids"] = [uid]
    elif fault == "unknown_target":
        result["patches"][0]["target_id"] = "r404"
    elif fault == "duplicate_patch":
        result["patches"].append(copy.deepcopy(result["patches"][0]))
    elif fault == "empty_units":
        result["patches"][0]["sentences"] = []
    elif fault == "unit_type":
        result["patches"][0]["sentences"] = ["text"]
    elif fault == "unit_metadata":
        result["patches"][0]["sentences"][0]["source_unit_ids"] = [uid]
    elif fault == "unit_text":
        result["patches"][0]["sentences"][0]["text"] = 1
    elif fault == "unit_references":
        result["patches"][0]["sentences"][0]["fact_card_refs"] = "c1"
    elif fault == "unit_reference_type":
        result["patches"][0]["sentences"][0]["fact_card_refs"] = [False]
    elif fault == "relation_fields":
        rel["reason"] = "Not a declared field."
    elif fault == "unknown_relation_target":
        rel["target_id"] = "r404"
    elif fault == "bool_position":
        rel["replacement_index"] = False
    elif fault == "float_position":
        rel["replacement_index"] = 0.0
    elif fault == "negative_position":
        rel["replacement_index"] = -1
    elif fault == "high_position":
        rel["replacement_index"] = 1
    elif fault == "duplicate_relation":
        result["source_relations"].append(copy.deepcopy(rel))
    elif fault == "missing_relation":
        result["source_relations"] = []
    elif fault == "empty_sources":
        rel["source_unit_ids"] = []
    elif fault == "duplicate_sources":
        rel["source_unit_ids"].append(uid)
    elif fault == "cross_target":
        rel["source_unit_ids"] = original_unit_directory(targets)["r2"]
    elif fault == "unknown_source":
        rel["source_unit_ids"] = ["writer:introduction:0:999"]
    elif fault == "source_type":
        rel["source_unit_ids"] = [[]]
    elif fault == "sources_type":
        rel["source_unit_ids"] = uid
    elif fault == "missing_original":
        result = response(
            targets, tid="r3", sources=original_unit_directory(targets)["r3"][:1]
        )
    else:
        removal = {
            "unit_id": original_unit_directory(targets)["r2"][0],
            "reason": "Duplicate.",
        }
        result["deleted_sources"] = [removal]
        if fault == "duplicate_deletion":
            result["deleted_sources"].append(copy.deepcopy(removal))
        elif fault == "unknown_deletion":
            removal["unit_id"] = "writer:unknown:0:0"
        elif fault == "mapped_and_deleted":
            removal["unit_id"] = uid
        elif fault == "deletion_fields":
            removal["action"] = "removed"
        elif fault == "blank_reason":
            removal["reason"] = " "
        elif fault == "long_reason":
            removal["reason"] = "删" * 257
        elif fault == "reason_type":
            removal["reason"] = False
        elif fault == "deletion_id_type":
            removal["unit_id"] = []
    before, frozen = copy.deepcopy(result), copy.deepcopy(targets)
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        normalize_source_map(result, targets)
    assert result == before and targets == frozen


def test_metadata_bound_counts_complete_canonical_object_and_json_escaping(targets):
    """The closing byte, host index and escaped text all contribute to the bound."""
    result = response(targets)
    result["deleted_sources"] = [
        {
            "unit_id": original_unit_directory(targets)["r2"][0],
            "reason": '引号"\\\n😀' * 10,
        }
    ]
    metadata = {key: result[key] for key in ("source_relations", "deleted_sources")}
    size = nested_json_size(metadata)
    assert size > len(json.dumps(metadata, ensure_ascii=False).encode())
    normalized = normalize_source_map(result, targets, max_json_bytes=size)
    before = copy.deepcopy(result)
    with pytest.raises(EnvelopeExceeded) as caught:
        normalize_source_map(result, targets, max_json_bytes=size - 1)
    assert caught.value.diagnostics == [
        {
            "stage": "repair",
            "field": "source_map_json_bytes",
            "actual": size,
            "maximum": size - 1,
        }
    ]
    assert classify_failure(caught.value)["family"] == "output_scope"
    assert normalize_source_map(result, targets) == normalized and result == before


@pytest.mark.parametrize("limit", [False, True, 0, -1, 1.5, "12288"])
def test_metadata_bound_requires_a_real_positive_integer(targets, limit):
    """No boolean or coercion may choose an output budget."""
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        normalize_source_map(response(targets), targets, max_json_bytes=limit)


def test_decline_is_empty_and_unchanged_output_keeps_existing_no_progress(targets):
    """A refusal cannot authorize mutations or consume hidden replacements."""
    result = response(targets)
    result.update(
        outcome="declined", patches=[], source_relations=[], deleted_sources=[]
    )
    assert normalize_source_map(result, targets) == {
        "outcome": "declined",
        "explanation": result["explanation"],
        "patches": [],
        "edit_records": [],
        "removed_targets": [],
    }
    result["outcome"] = "patched"
    with pytest.raises(WriterError, match="writer_repair_no_progress") as caught:
        normalize_source_map(result, targets)
    assert caught.value.family == "quality"
    result = response(targets)
    result["outcome"] = "declined"
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        normalize_source_map(result, targets)


@pytest.mark.parametrize(
    "mapping,expected",
    [
        (False, "729ea8753395dfcdf121eccc7df8cfe490b2c9ec514e963b08721924915a9c9a"),
        (True, "48e45c65011b6f84e124836a6411024e451a9922140ee3babb219d9ebb6fcc25"),
    ],
)
def test_legacy_schema_golden_and_original_global_index_failure_remain(
    mapping, expected
):
    """New source handling neither promotes old schema nor renumbers old replies."""
    body = {
        "title": "Directory fixture",
        **{section: [{"sentences": [unit(section)]}] for section in BODY},
    }
    targets = capacity_targets(body, {})
    schema = edit_schema(
        schema_for("repair", {"c1": {}}, targets), targets, mapping=mapping
    )
    assert digest(schema) == expected
    old = {
        "outcome": "patched",
        "explanation": "Global indexes remain invalid.",
        "patches": [
            {"target_id": tid, "sentences": copy.deepcopy(t["original"])}
            for tid, t in targets.items()
        ],
        "edit_records": [
            {
                "unit_id": ids[0],
                "reason": "Original map.",
                "replacement_indexes": [index],
            }
            for index, ids in enumerate(original_unit_directory(targets).values())
        ],
    }
    with pytest.raises(WriterError, match="writer_capacity_edit_position_invalid"):
        normalize_edit_map(old, targets)


def test_source_identity_binds_core_rules_without_modifying_legacy_text(monkeypatch):
    """A wire adapter can compose this stable core without a circular import."""
    before = source_map_identity()
    old = capacity_repair.MAP_GUIDANCE, capacity_repair.MAP_INSTRUCTION
    assert SOURCE_MAP_CONTRACT == "capacity-source-map-v3" and len(before) == 64
    monkeypatch.setattr(
        capacity_repair,
        "SOURCE_MAP_RULES",
        (*capacity_repair.SOURCE_MAP_RULES, "changed-rule"),
    )
    assert source_map_identity() != before
    assert old == (capacity_repair.MAP_GUIDANCE, capacity_repair.MAP_INSTRUCTION)


def test_single_original_private_view_preserves_input_and_validates_list_hash():
    """Only the local range view changes representation, never the frozen input."""
    original = unit("One exact original")
    targets = {
        "single": {
            "path": ["methods", 2, 3],
            "original": original,
            "original_sha256": digest(original),
            "maximum": 2,
        }
    }
    before = copy.deepcopy(targets)
    directory, views = capacity_repair._source_target_views(targets)
    assert views["single"]["original_sha256"] == digest([original])
    assert (
        original_unit_directory(views)
        == directory
        == {"single": ["writer:methods:2:3"]}
    )
    result = response(targets, tid="single", units=[original])
    normalized = normalize_source_map(result, targets)
    assert normalized["edit_records"][0]["action"] == "retained"
    assert targets == before


def test_source_schema_rejects_metadata_inside_scientific_unit(targets):
    """The canonical scientific schema cannot silently carry relation fields."""
    schema = schema_for("repair", {"c1": {}}, targets)
    item = schema["properties"]["patches"]["items"]["properties"]["sentences"]["items"]
    item["properties"]["source_unit_ids"] = {"type": "array"}
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        source_edit_schema(schema, targets)


@pytest.mark.parametrize("maximum", [False, 0, -1, "66"])
def test_source_target_maximum_is_not_coerced(targets, maximum):
    """Private capacity views retain the strict integer trust boundary."""
    targets["r1"]["maximum"] = maximum
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        normalize_source_map(response(targets), targets)


def test_complete_metadata_over_12288_is_retained_and_typed_not_truncated():
    """A structurally complete deletion declaration still has a finite byte cap."""
    originals = [unit(f"Original {index}") for index in range(45)]
    targets = {
        "range": {
            "path": ["methods", 0, 0],
            "end": 45,
            "maximum": 66,
            "original": originals,
            "original_sha256": digest(originals),
        }
    }
    result = {
        "outcome": "patched",
        "explanation": "This is not a scientific approval.",
        "patches": [],
        "source_relations": [],
        "deleted_sources": [
            {"unit_id": uid, "reason": "x" * 256}
            for uid in original_unit_directory(targets)["range"]
        ],
    }
    before = copy.deepcopy(result)
    with pytest.raises(EnvelopeExceeded) as caught:
        normalize_source_map(result, targets, max_json_bytes=12288)
    assert caught.value.diagnostics[0]["actual"] > 12288
    assert caught.value.diagnostics[0]["maximum"] == 12288
    assert result == before
