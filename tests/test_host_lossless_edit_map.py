"""Keep original directories exact while preserving strict edit-map accounting."""

import copy
import json
from pathlib import Path

import pytest

from litagent.capacity_repair import (
    capacity_targets,
    edit_schema,
    normalize_edit_map,
    original_unit_directory,
    validate_edit_records,
)
from litagent.quality import digest
from litagent.writing_flat import encode, parse
from litagent.writing_v2 import BODY, WriterError, schema_for


@pytest.fixture
def body():
    """Provide source positions without claiming scientific support for the text."""
    return {
        "title": "Directory fixture",
        **{
            section: [{"sentences": [{"text": section, "fact_card_refs": ["c1"]}]}]
            for section in BODY
        },
    }


def mapped_result(targets):
    """Return an explicit identity map for every frozen original position."""
    directory = original_unit_directory(targets)
    return {
        "outcome": "patched",
        "explanation": "Preserve each frozen unit at its exact position.",
        "patches": [
            {"target_id": tid, "sentences": copy.deepcopy(target["original"])}
            for tid, target in targets.items()
        ],
        "edit_records": [
            {
                "unit_id": uid,
                "reason": "Preserve this original.",
                "replacement_indexes": [index],
            }
            for ids in directory.values()
            for index, uid in enumerate(ids)
        ],
    }


@pytest.mark.parametrize(
    "mapping,expected",
    [
        (False, "729ea8753395dfcdf121eccc7df8cfe490b2c9ec514e963b08721924915a9c9a"),
        (True, "48e45c65011b6f84e124836a6411024e451a9922140ee3babb219d9ebb6fcc25"),
    ],
)
def test_directory_keeps_legacy_edit_schema_values_and_order(body, mapping, expected):
    """Freeze both old schema variants while sourcing IDs from one directory."""
    targets = capacity_targets(body, {})
    before = copy.deepcopy(targets)
    directory = original_unit_directory(targets)
    assert directory == {
        "r1": ["writer:introduction:0:0"],
        "r2": ["writer:taxonomy:0:0"],
        "r3": ["writer:methods:0:0"],
        "r4": ["writer:experiments:0:0"],
        "r5": ["writer:open_problems:0:0"],
    }
    schema = edit_schema(
        schema_for("repair", {"c1": {}}, targets), targets, mapping=mapping
    )
    assert digest(schema) == expected
    assert schema["properties"]["edit_records"]["items"]["properties"]["unit_id"][
        "enum"
    ] == [uid for ids in directory.values() for uid in ids]
    assert targets == before


def test_directory_binds_single_unit_summary_and_body_targets_by_position():
    """Keep equal text distinct and allow existing targets without optional SHAs."""
    unit = {"text": "Same text.", "body_unit_refs": ["u1"]}
    targets = {
        "summary": {"path": ["abstract", 2, 3], "original": copy.deepcopy(unit)},
        "body": {"path": ["methods", 2, 3], "original": copy.deepcopy(unit)},
        "next": {
            "path": ["abstract", 2, 4],
            "original": copy.deepcopy(unit),
            "original_sha256": digest(unit),
        },
    }
    assert original_unit_directory(targets) == {
        "summary": ["writer:abstract:2:3"],
        "body": ["writer:methods:2:3"],
        "next": ["writer:abstract:2:4"],
    }


def test_directory_preserves_nonzero_range_and_allows_adjacent_targets():
    """Bind explicit slices by absolute position without overlapping endpoints."""
    unit = {"text": "Same text at distinct positions.", "fact_card_refs": ["c1"]}
    first, second = [copy.deepcopy(unit), copy.deepcopy(unit)], [copy.deepcopy(unit)]
    targets = {
        "range": {
            "path": ["methods", 3, 2],
            "end": 4,
            "original": first,
            "original_sha256": digest(first),
        },
        "adjacent": {
            "path": ["methods", 3, 4],
            "end": 5,
            "original": second,
            "original_sha256": digest(second),
        },
    }
    assert original_unit_directory(targets) == {
        "range": ["writer:methods:3:2", "writer:methods:3:3"],
        "adjacent": ["writer:methods:3:4"],
    }


@pytest.mark.parametrize(
    "fault",
    [
        "empty",
        "target_id",
        "target_type",
        "path_missing",
        "path_tuple",
        "section",
        "negative_paragraph",
        "bool_paragraph",
        "negative_start",
        "bool_start",
        "bool_end",
        "empty_range",
        "original_missing",
        "original_count",
        "original_type",
        "original_unit_type",
        "original_sha",
        "snapshot_type",
        "snapshot_format",
        "snapshot_owner",
        "overlap",
    ],
)
def test_directory_rejects_forged_or_ambiguous_target_identity(body, fault):
    """Fail before a directory can authorize incorrect positions or ownership."""
    targets = capacity_targets(body, {})
    first = targets["r1"]
    if fault == "empty":
        targets.clear()
    elif fault == "target_id":
        targets[False] = targets.pop("r1")
    elif fault == "target_type":
        targets["r1"] = []
    elif fault == "path_missing":
        first.pop("path")
    elif fault == "path_tuple":
        first["path"] = tuple(first["path"])
    elif fault == "section":
        first["path"][0] = "appendix"
    elif fault == "negative_paragraph":
        first["path"][1] = -1
    elif fault == "bool_paragraph":
        first["path"][1] = True
    elif fault == "negative_start":
        first["path"][2] = -1
    elif fault == "bool_start":
        first["path"][2] = False
    elif fault == "bool_end":
        first["end"] = True
    elif fault == "empty_range":
        first["end"] = 0
    elif fault == "original_missing":
        first.pop("original")
    elif fault == "original_count":
        first["end"] += 1
    elif fault == "original_type":
        first["original"] = first["original"][0]
    elif fault == "original_unit_type":
        first["original"] = [False]
    elif fault == "original_sha":
        first["original"][0]["text"] += " Forged content."
    elif fault == "snapshot_type":
        first["snapshot_sha256"] = None
    elif fault == "snapshot_format":
        first["snapshot_sha256"] = "g" * 64
    elif fault == "snapshot_owner":
        first["snapshot_sha256"] = "0" * 64
    elif fault == "overlap":
        targets["other"] = copy.deepcopy(first)
    before = copy.deepcopy(targets)
    with pytest.raises(WriterError, match="writer_patch_identity_invalid"):
        original_unit_directory(targets)
    assert targets == before


def test_directory_rejects_list_original_for_single_unit_target():
    """Do not silently reinterpret legacy single-unit and explicit-range shapes."""
    with pytest.raises(WriterError, match="writer_patch_identity_invalid"):
        original_unit_directory({"x": {"path": ["conclusion", 0, 0], "original": []}})


def test_directory_flat_schema_and_mapping_consumers_share_original_positions(body):
    """Exercise the real flat encoder/parser and unchanged strict map consumer."""
    body["introduction"][0]["sentences"].append(
        {"text": "A separate original.", "fact_card_refs": ["c1"]}
    )
    targets = capacity_targets(body, {})
    result = mapped_result(targets)
    schema = edit_schema(
        schema_for("repair", {"c1": {}}, targets), targets, mapping=True
    )
    wire = encode(result, schema)
    parsed = parse(json.dumps(wire), schema)["canonical"]
    normalized = normalize_edit_map(parsed, targets)
    validate_edit_records(normalized, targets)
    assert [r["unit_id"] for r in normalized["edit_records"]] == [
        uid for ids in original_unit_directory(targets).values() for uid in ids
    ]
    assert all(r["action"] == "retained" for r in normalized["edit_records"])


def test_explicit_merge_and_deletion_preserve_all_unmentioned_targets(body):
    """Host preservation never substitutes for a changed target's missing map."""
    body["introduction"][0]["sentences"].append(
        {"text": "A distinct original.", "fact_card_refs": ["c1"]}
    )
    targets = capacity_targets(body, {})
    directory = original_unit_directory(targets)
    result = {
        "patches": [
            {
                "target_id": "r1",
                "sentences": [{"text": "Merged content.", "fact_card_refs": ["c1"]}],
            }
        ],
        "edit_records": [
            {"unit_id": uid, "reason": "Explicitly merged.", "replacement_indexes": [0]}
            for uid in directory["r1"]
        ],
    }
    normalized = normalize_edit_map(result, targets)
    assert [r["action"] for r in normalized["edit_records"][:2]] == [
        "rewritten",
        "rewritten",
    ]
    assert all(
        patch["sentences"] == targets[patch["target_id"]]["original"]
        for patch in normalized["patches"][1:]
    )
    result["patches"] = []
    for record in result["edit_records"]:
        record["replacement_indexes"] = []
    removed = normalize_edit_map(result, targets)
    assert removed["removed_targets"] == ["r1"]
    assert all(r["action"] == "removed" for r in removed["edit_records"][:2])


@pytest.mark.parametrize(
    "fault,code",
    [
        ("unknown_original", "writer_capacity_edit_coverage_invalid"),
        ("duplicate_original", "writer_capacity_edit_coverage_invalid"),
        ("missing_original", "writer_capacity_edit_coverage_invalid"),
        ("local_index", "writer_capacity_edit_position_invalid"),
        ("bool_index", "writer_capacity_edit_position_invalid"),
        ("duplicate_index", "writer_capacity_edit_position_invalid"),
        ("unknown_target", "writer_capacity_edit_coverage_invalid"),
        ("duplicate_target", "writer_capacity_edit_coverage_invalid"),
        ("orphan_replacement", "writer_capacity_new_unit_unaccounted"),
    ],
)
def test_directory_does_not_repair_invalid_explicit_mapping(body, fault, code):
    """Retain each independent strict-map guard after adding a host directory."""
    targets = capacity_targets(body, {})
    result = mapped_result(targets)
    first = result["edit_records"][0]
    if fault == "unknown_original":
        first["unit_id"] = "writer:introduction:0:99"
    elif fault == "duplicate_original":
        result["edit_records"].append(copy.deepcopy(first))
    elif fault == "missing_original":
        result["edit_records"].pop(0)
    elif fault == "local_index":
        first["replacement_indexes"] = [1]
    elif fault == "bool_index":
        first["replacement_indexes"] = [False]
    elif fault == "duplicate_index":
        first["replacement_indexes"] = [0, 0]
    elif fault == "unknown_target":
        result["patches"][0]["target_id"] = "unknown"
    elif fault == "duplicate_target":
        result["patches"].append(copy.deepcopy(result["patches"][0]))
    elif fault == "orphan_replacement":
        result["patches"][0]["sentences"].append(
            {"text": "Unmapped new content.", "fact_card_refs": ["c1"]}
        )
    before = copy.deepcopy((targets, result))
    with pytest.raises(WriterError, match=code):
        normalize_edit_map(result, targets)
    assert (targets, result) == before


def test_private_037_grouping_still_rejects_actual_indexes_and_missing_records():
    """Replay preserved private raw if present; never synthesize a repaired 037."""
    review = (
        Path(__file__).resolve().parents[1]
        / "artifacts/acceptance/reference-set-live-20261101/terminal-review"
    )
    names = [
        "vit-repair_0.original.raw.json",
        "vit-repair_0.exact-requested-targets.json",
        "vit-repair_0.exact-host-schema.json",
    ]
    if not all((review / name).is_file() for name in names):
        pytest.skip("037 private preserved raw is not part of the public test fixture")
    raw, targets, schema = [json.loads((review / name).read_text()) for name in names]
    assert len(raw["patches"]) == len(raw["edit_records"]) == 43
    directory = original_unit_directory(targets)
    assert sum(map(len, directory.values())) == 48
    with pytest.raises(WriterError, match="writer_flat_target_order_invalid"):
        parse(json.dumps(raw, ensure_ascii=False), schema)
    grouped = {}
    for patch in raw["patches"]:
        grouped.setdefault(patch["target_id"], []).append(patch)
    flat = copy.deepcopy(raw)
    flat["patches"] = [patch for units in grouped.values() for patch in units]
    canonical = parse(json.dumps(flat, ensure_ascii=False), schema)["canonical"]
    counts = {p["target_id"]: len(p["sentences"]) for p in canonical["patches"]}
    owner = {uid: tid for tid, ids in directory.items() for uid in ids}
    assert (
        sum(
            any(
                index >= counts[owner[r["unit_id"]]]
                for index in r["replacement_indexes"]
            )
            for r in raw["edit_records"]
        )
        == 38
    )
    missing = set(owner) - {record["unit_id"] for record in raw["edit_records"]}
    assert missing == {
        "writer:methods:7:0",
        "writer:methods:8:0",
        "writer:methods:9:0",
        "writer:methods:10:0",
        "writer:open_problems:8:0",
    }
    with pytest.raises(WriterError, match="writer_capacity_edit_position_invalid"):
        normalize_edit_map(canonical, targets)
    # Independent coverage isolation keeps the missing-five gate observable;
    # these empty replacements are a test fault, not a repaired candidate.
    coverage_only = copy.deepcopy(canonical)
    for patch in coverage_only["patches"]:
        patch["sentences"] = []
    for record in coverage_only["edit_records"]:
        record["replacement_indexes"] = []
    with pytest.raises(WriterError, match="writer_capacity_edit_coverage_invalid"):
        normalize_edit_map(coverage_only, targets)
