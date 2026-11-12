"""Authorize explicit capacity edits without treating unverified text as false."""

import copy

ACTION_CONTRACT = "capacity-edit-actions-v1"
MAP_CONTRACT = "capacity-edit-map-v2"
SOURCE_MAP_CONTRACT = "capacity-source-map-v3"
SOURCE_OWNER_CONTRACT = "capacity-source-map-v4"
SOURCE_MAP_HOST_REASON = "Host-derived from declared replacement sources."
SOURCE_MAP_RULES = (
    "source-ids-are-exact-authorized-original-positions-not-citations-v1",
    "one-nonempty-target-local-source-list-per-returned-unit-v1",
    "allow-one-source-to-many-units-and-many-sources-to-one-unit-v1",
    "reject-duplicate-relations-sources-deletions-and-cross-target-ownership-v1",
    "complete-changed-target-original-cover-with-disjoint-explicit-deletions-v1",
    "host-enumerates-target-local-order-never-searches-or-renumbers-model-indexes-v1",
    "preserve-unmentioned-targets-and-existing-no-progress-behavior-v1",
    "complete-nested-json-size-of-source-relations-and-deleted-sources-v1",
    "derive-legacy-edit-records-and-reuse-v2-normalization-and-validation-v1",
    "source-relations-are-lineage-declarations-not-scientific-support-v1",
    "private-single-original-range-view-preserves-id-and-rebinds-list-hash-v1",
)
GUIDANCE_VERSION = "bounded-capacity-map-feedback-v1"
CONTENT_GUIDANCE = (
    "Choose necessary distinct findings, not one expansion per paper or card. "
    "Use prior chapters to avoid repeating background, mechanisms or result lists. "
    "Keep mechanisms in methods and experimental comparisons in experiments. "
    "Abstract and conclusion each select a few central findings, not every method "
    "and dataset. Preserve core questions, owners, conditions and qualifications; "
    "do not compress independent facts into longer compound sentences. Unit "
    "targets are not assertion counts or evidence of scientific support."
)
MAP_GUIDANCE = (
    "Capacity diagnosis is not a scientific verdict. Remove only redundant or "
    "non-core elaboration; preserve core methods, conditions, numbers and sourced "
    "limitations. Return only changed targets. Account for every original of a "
    "changed target exactly once in edit_records. replacement_indexes are zero-based "
    "WITHIN that target's returned units, not the global patches array. Every new "
    "unit needs an original mapping. Example: one returned unit requires [0] on "
    "at least one original; two originals may both map to [0] when merged. [] "
    "means deleted, never merged. To delete all originals, return no patch for "
    "that target. Unmentioned targets remain unchanged. Full reassessment follows."
)
MAP_INSTRUCTION = (
    "Return patches only for changed targets. For each changed target, account for "
    "EVERY original unit exactly once in edit_records using unit_id, reason and "
    "replacement_indexes local to that target. Empty indexes explicitly delete "
    "that unit; a nonempty list maps its content to replacement units. Multiple "
    "originals may map to one merged replacement. Every replacement needs an "
    "original mapping. Unmentioned targets remain byte-for-byte unchanged. "
    "Do not return action labels or removed_targets. To remove a whole target, "
    "supply empty mappings for all its originals and no patches for that target."
)
ACTION_INSTRUCTION = (
    "retained: one unchanged replacement. rewritten: at least one replacement, "
    "including merged content; multiple originals may share an index. removed: "
    "content omitted, replacement_indexes MUST be []. Indexes are zero-based "
    "within the target. Merging is rewritten, never removed."
)


def guided_refinement(envelope):
    """Select the explicit v7 path without changing older writer contracts."""
    return (
        isinstance(envelope, dict)
        and "capacity" in envelope
        and envelope.get("repair_mapping_contract") == MAP_CONTRACT
        and envelope.get("global_findings_contract") == "anchored-global-findings-v2"
    )


def retryable_map_failure(failure, body, envelope):
    """Authorize only host-located orphan replacements on the unchanged body."""
    from litagent.quality import digest

    if (
        not guided_refinement(envelope)
        or not isinstance(body, dict)
        or type(envelope.get("body_max_units")) is not int
        or not isinstance(failure, dict)
        or failure.get("code") != "writer_capacity_new_unit_unaccounted"
        or failure.get("family") != "quality"
        or failure.get("source_body_sha256") != digest(body)
        or not isinstance(failure.get("diagnostics"), list)
        or not failure["diagnostics"]
    ):
        return False
    try:
        targets = capacity_targets(body, {})
    except (KeyError, TypeError, IndexError):
        return False
    seen = set()
    for issue in failure["diagnostics"]:
        if not isinstance(issue, dict):
            return False
        tid = issue.get("target_id")
        if not isinstance(tid, str) or tid not in targets or tid in seen:
            return False
        seen.add(tid)
        target = targets[tid]
        count, indexes = issue.get("replacement_count"), issue.get("unmapped_indexes")
        if (
            issue.get("path") != target["path"]
            or issue.get("end") != target["end"]
            or type(count) is not int
            or not 1 <= count <= envelope["body_max_units"]
            or not isinstance(indexes, list)
            or not indexes
            or any(type(i) is not int or not 0 <= i < count for i in indexes)
            or len(indexes) != len(set(indexes))
        ):
            return False
    return True


def capacity_targets(body, diagnostic, *, maximum=None):
    """Authorize disjoint paragraph ranges and bind every original unit."""
    from litagent.quality import digest
    from litagent.writing_v2 import BODY

    targets = {}
    for section in BODY:
        for pi, paragraph in enumerate(body[section]):
            originals = paragraph["sentences"]
            targets[f"r{len(targets)+1}"] = {
                "path": [section, pi, 0],
                "end": len(originals),
                "original": copy.deepcopy(originals),
                "original_sha256": digest(originals),
                "snapshot_sha256": digest(body),
                "maximum": len(originals) if maximum is None else maximum,
                "explanation": "Capacity refinement only; preserve core requirements. "
                + str(diagnostic.get("reason", "capacity")),
            }
    return targets


def original_unit_directory(targets) -> dict[str, list[str]]:
    """Bind each disjoint target to its exact original writer positions."""
    from litagent.quality import digest
    from litagent.writing_v2 import BODY, WriterError

    try:
        if not isinstance(targets, dict) or not targets:
            raise ValueError("target directory")
        directory, seen, snapshots = {}, set(), set()
        for tid, target in targets.items():
            if not isinstance(tid, str) or not tid or not isinstance(target, dict):
                raise ValueError("target identity")
            path = target["path"]
            if not isinstance(path, list) or len(path) != 3:
                raise ValueError("target path")
            section, pi, start = path
            if (
                section not in (*BODY, "abstract", "conclusion")
                or type(pi) is not int
                or pi < 0
                or type(start) is not int
                or start < 0
            ):
                raise ValueError("target position")
            end = target.get("end", start + 1)
            if type(end) is not int or end <= start:
                raise ValueError("target range")
            original = target["original"]
            if "end" in target:
                if (
                    not isinstance(original, list)
                    or len(original) != end - start
                    or any(not isinstance(unit, dict) for unit in original)
                ):
                    raise ValueError("target originals")
            elif not isinstance(original, dict):
                raise ValueError("target original")
            if "original_sha256" in target and target["original_sha256"] != digest(
                original
            ):
                raise ValueError("target original hash")
            if "snapshot_sha256" in target:
                snapshot = target["snapshot_sha256"]
                if (
                    not isinstance(snapshot, str)
                    or len(snapshot) != 64
                    or any(c not in "0123456789abcdef" for c in snapshot)
                ):
                    raise ValueError("target snapshot hash")
                snapshots.add(snapshot)
                if len(snapshots) != 1:
                    raise ValueError("target snapshot ownership")
            unit_ids = [f"writer:{section}:{pi}:{i}" for i in range(start, end)]
            # Equal text at different positions is distinct; overlapping paths
            # cannot authorize the same original under two target identities.
            if seen.intersection(unit_ids):
                raise ValueError("overlapping target ranges")
            seen.update(unit_ids)
            directory[tid] = unit_ids
        return directory
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise WriterError("writer_patch_identity_invalid") from exc


def edit_schema(schema, targets, *, mapping=False):
    """Require an explicit disposition for every authorized original unit."""
    from litagent.writing_v2 import _array, _closed

    schema = copy.deepcopy(schema)
    ids = [
        uid
        for unit_ids in original_unit_directory(targets).values()
        for uid in unit_ids
    ]
    properties = schema["properties"]
    properties["patches"]["items"]["properties"]["sentences"]["minItems"] = 0
    properties["patches"]["items"]["properties"]["sentences"]["maxItems"] = max(
        t["maximum"] for t in targets.values()
    )
    properties["removed_targets"] = _array(
        {"type": "string", "enum": list(targets)}, minimum=0, maximum=len(targets)
    )
    properties["edit_records"] = _array(
        _closed(
            {
                "unit_id": {"type": "string", "enum": ids},
                "action": {
                    "type": "string",
                    "enum": ["retained", "rewritten", "removed"],
                },
                "reason": {"type": "string", "minLength": 1, "maxLength": 256},
                "replacement_indexes": _array(
                    {
                        "type": "integer",
                        "enum": list(
                            range(max(t["maximum"] for t in targets.values()))
                        ),
                    },
                    minimum=0,
                    maximum=max(t["maximum"] for t in targets.values()),
                ),
            }
        ),
        minimum=0,
        maximum=len(ids),
    )
    schema["required"] = list(properties)
    if mapping:
        del properties["removed_targets"]
        record = properties["edit_records"]["items"]
        del record["properties"]["action"]
        record["required"].remove("action")
        schema["required"].remove("removed_targets")
    return schema


def normalize_edit_map(result, targets):
    """Derive dispositions from one complete map, preserving untouched targets."""
    from litagent.writing_v2 import WriterError

    result = copy.deepcopy(result)
    originals = {}
    for tid, target in targets.items():
        section, pi, start = target["path"]
        for i, unit in enumerate(target["original"], start):
            originals[f"writer:{section}:{pi}:{i}"] = (tid, unit)
    patches = {}
    for patch in result["patches"]:
        tid = patch["target_id"]
        if tid not in targets or tid in patches:
            raise WriterError("writer_capacity_edit_coverage_invalid")
        patches[tid] = patch["sentences"]
    records, seen, changed = [], set(), set(patches)
    for record in result["edit_records"]:
        uid = record["unit_id"]
        if uid not in originals or uid in seen:
            raise WriterError("writer_capacity_edit_coverage_invalid")
        seen.add(uid)
        tid, original = originals[uid]
        changed.add(tid)
        indexes = record["replacement_indexes"]
        replacement = patches.get(tid, [])
        if len(indexes) != len(set(indexes)) or any(
            type(i) is not int or not 0 <= i < len(replacement) for i in indexes
        ):
            raise WriterError("writer_capacity_edit_position_invalid")
        action = (
            "removed"
            if not indexes
            else (
                "retained"
                if len(indexes) == 1 and replacement[indexes[0]] == original
                else "rewritten"
            )
        )
        records.append({**record, "action": action})
    if not changed:
        raise WriterError("writer_repair_no_progress", "quality")
    for uid, (tid, original) in originals.items():
        if tid in changed:
            if uid not in seen:
                raise WriterError("writer_capacity_edit_coverage_invalid")
            patches.setdefault(tid, [])
        else:
            patches[tid] = copy.deepcopy(targets[tid]["original"])
            start = targets[tid]["path"][2]
            records.append(
                {
                    "unit_id": uid,
                    "action": "retained",
                    "reason": "Host-preserved unchanged target.",
                    "replacement_indexes": [int(uid.rsplit(":", 1)[1]) - start],
                }
            )
    result["patches"] = [
        {"target_id": tid, "sentences": patches[tid]} for tid in targets
    ]
    result["edit_records"] = records
    result["removed_targets"] = [tid for tid in targets if not patches[tid]]
    validate_edit_records(result, targets)
    return result


def validate_edit_records(result, targets):
    """Reject omitted originals, false retention claims and implicit deletion."""
    from litagent.writing_v2 import WriterError

    originals = {}
    patches = {p["target_id"]: p["sentences"] for p in result["patches"]}
    for tid, target in targets.items():
        section, pi, start = target["path"]
        for i, original in enumerate(target["original"], start):
            originals[f"writer:{section}:{pi}:{i}"] = (tid, original)
    records = result["edit_records"]
    if len(records) != len(originals) or {r["unit_id"] for r in records} != set(
        originals
    ):
        raise WriterError("writer_capacity_edit_coverage_invalid")
    covered = {tid: set() for tid in patches}
    for record in records:
        tid, original = originals[record["unit_id"]]
        indexes = record["replacement_indexes"]
        if len(indexes) != len(set(indexes)) or any(
            type(i) is not int or not 0 <= i < len(patches[tid]) for i in indexes
        ):
            raise WriterError("writer_capacity_edit_position_invalid")
        if record["action"] == "removed":
            if indexes:
                raise WriterError("writer_capacity_removal_invalid")
        elif not indexes or (
            record["action"] == "retained"
            and (len(indexes) != 1 or patches[tid][indexes[0]] != original)
        ):
            raise WriterError("writer_capacity_retention_invalid")
        covered[tid].update(indexes)
    missing = [
        {
            "target_id": tid,
            "path": targets[tid]["path"],
            "end": targets[tid]["end"],
            "replacement_count": len(patches[tid]),
            "unmapped_indexes": sorted(set(range(len(patches[tid]))) - indexes),
        }
        for tid, indexes in covered.items()
        if indexes != set(range(len(patches[tid])))
    ]
    if missing:
        raise WriterError("writer_capacity_new_unit_unaccounted", diagnostics=missing)


def source_map_identity(contract=SOURCE_MAP_CONTRACT):
    """Bind the opt-in source relation rules without importing wire adapters."""
    from litagent.quality import digest

    if contract == SOURCE_OWNER_CONTRACT:
        return digest(
            [
                SOURCE_OWNER_CONTRACT,
                source_map_identity(),
                "host-schema-exact-disjoint-source-owner-directory-v1",
                "derive-target-only-from-complete-nonempty-single-owner-source-list-v1",
                "no-model-target-id-or-fabricated-raw-target-span-v1",
            ]
        )
    if contract != SOURCE_MAP_CONTRACT:
        raise _source_map_invalid("mapping_contract")
    return digest(
        [
            SOURCE_MAP_CONTRACT,
            SOURCE_MAP_RULES,
            SOURCE_MAP_HOST_REASON,
            [
                "outcome",
                "explanation",
                "patches",
                "source_relations",
                "deleted_sources",
            ],
            ["target_id", "replacement_index", "source_unit_ids"],
            ["unit_id", "reason"],
            "closed-clean-text-and-one-reference-list-units-v1",
            "strict-integer-position-no-bool-or-coercion-v1",
            "deletion-reason-nonblank-max-256-characters-v1",
        ]
    )


def _source_map_invalid(reason, **details):
    """Locate invalid lineage without embedding scientific prose in errors."""
    from litagent.writing_v2 import WriterError

    return WriterError(
        "writer_capacity_source_map_invalid",
        diagnostics=[{"reason": reason, **details}],
    )


def _source_target_views(targets):
    """Validate frozen originals and make private range views for legacy checks."""
    from litagent.quality import digest

    directory = original_unit_directory(targets)
    views = copy.deepcopy(targets)
    for tid, target in views.items():
        if type(target.get("maximum")) is not int or target["maximum"] < 1:
            raise _source_map_invalid("target_maximum", target_id=tid)
        if "end" not in target:
            target["end"] = target["path"][2] + 1
            target["original"] = [target["original"]]
            if "original_sha256" in target:
                target["original_sha256"] = digest(target["original"])
    return directory, views


def source_edit_schema(schema, targets, *, contract=SOURCE_MAP_CONTRACT):
    """Declare clean replacements and separately bounded original-source links."""
    from litagent.writing_v2 import _array, _closed

    if contract not in (SOURCE_MAP_CONTRACT, SOURCE_OWNER_CONTRACT):
        raise _source_map_invalid("mapping_contract")
    directory, views = _source_target_views(targets)
    ids = [uid for units in directory.values() for uid in units]
    maximum = max(target["maximum"] for target in views.values())
    total = sum(target["maximum"] for target in views.values())
    result = copy.deepcopy(schema)
    try:
        properties = result["properties"]
        if set(properties) != {"outcome", "explanation", "patches"}:
            raise _source_map_invalid("mixed_schema")
        patch = properties["patches"]["items"]["properties"]
        if set(patch) != {"target_id", "sentences"}:
            raise _source_map_invalid("patch_schema")
        if set(patch["sentences"]["items"]["properties"]) not in (
            {"text", "fact_card_refs"},
            {"text", "body_unit_refs"},
        ):
            raise _source_map_invalid("unit_schema")
        properties["patches"]["minItems"] = 0
        patch["sentences"]["minItems"] = 1
        patch["sentences"]["maxItems"] = maximum
        properties["source_relations"] = _array(
            _closed(
                {
                    "target_id": {"type": "string", "enum": list(targets)},
                    "replacement_index": {
                        "type": "integer",
                        "enum": list(range(maximum)),
                    },
                    "source_unit_ids": _array(
                        {"type": "string", "enum": ids},
                        minimum=1,
                        maximum=len(ids),
                    ),
                }
            ),
            minimum=0,
            maximum=total,
        )
        properties["deleted_sources"] = _array(
            _closed(
                {
                    "unit_id": {"type": "string", "enum": ids},
                    "reason": {"type": "string", "minLength": 1, "maxLength": 256},
                }
            ),
            minimum=0,
            maximum=len(ids),
        )
        result["required"] = list(properties)
        if contract == SOURCE_OWNER_CONTRACT:
            result["x-repair-mapping-contract"] = contract
            result["x-source-owner-directory"] = copy.deepcopy(directory)
    except (KeyError, TypeError, AttributeError) as exc:
        raise _source_map_invalid("schema_shape") from exc
    return result


def normalize_source_map(result, targets, *, max_json_bytes=None):
    """Validate complete source declarations before deriving legacy positions."""
    from litagent.safety.planned_budget import EnvelopeExceeded, nested_json_size

    directory, views = _source_target_views(targets)
    if max_json_bytes is not None and (
        type(max_json_bytes) is not int or max_json_bytes < 1
    ):
        raise _source_map_invalid("metadata_limit")
    fields = {
        "outcome",
        "explanation",
        "patches",
        "source_relations",
        "deleted_sources",
    }
    if not isinstance(result, dict) or set(result) != fields:
        raise _source_map_invalid("root_fields")
    if (
        result["outcome"] not in ("patched", "declined")
        or not isinstance(result["explanation"], str)
        or not result["explanation"].strip()
        or any(
            type(result[key]) is not list for key in fields - {"outcome", "explanation"}
        )
    ):
        raise _source_map_invalid("root_values")
    patches = {}
    for patch in result["patches"]:
        if not isinstance(patch, dict) or set(patch) != {"target_id", "sentences"}:
            raise _source_map_invalid("patch_fields")
        tid, units = patch["target_id"], patch["sentences"]
        if (
            not isinstance(tid, str)
            or tid not in views
            or tid in patches
            or type(units) is not list
            or not 1 <= len(units) <= views[tid]["maximum"]
        ):
            raise _source_map_invalid("patch_target_or_count")
        for index, unit in enumerate(units):
            if not isinstance(unit, dict) or set(unit) not in (
                {"text", "fact_card_refs"},
                {"text", "body_unit_refs"},
            ):
                raise _source_map_invalid("unclean_unit", target_id=tid, index=index)
            refs = unit.get("fact_card_refs", unit.get("body_unit_refs"))
            if (
                not isinstance(unit["text"], str)
                or not unit["text"].strip()
                or type(refs) is not list
                or any(not isinstance(ref, str) or not ref for ref in refs)
            ):
                raise _source_map_invalid("unit_values", target_id=tid, index=index)
        patches[tid] = units
    owners = {uid: tid for tid, units in directory.items() for uid in units}
    positions, linked = {}, {}
    for relation in result["source_relations"]:
        if not isinstance(relation, dict) or set(relation) != {
            "target_id",
            "replacement_index",
            "source_unit_ids",
        }:
            raise _source_map_invalid("relation_fields")
        tid, index, sources = (
            relation["target_id"],
            relation["replacement_index"],
            relation["source_unit_ids"],
        )
        if (
            not isinstance(tid, str)
            or tid not in patches
            or type(index) is not int
            or not 0 <= index < len(patches[tid])
            or (tid, index) in positions
        ):
            raise _source_map_invalid("relation_position")
        if (
            type(sources) is not list
            or not sources
            or any(
                not isinstance(uid, str) or owners.get(uid) != tid for uid in sources
            )
            or len(sources) != len(set(sources))
        ):
            raise _source_map_invalid("relation_sources", target_id=tid, index=index)
        positions[tid, index] = sources
        for uid in sources:
            linked.setdefault(uid, []).append(index)
    if set(positions) != {
        (tid, index) for tid, units in patches.items() for index in range(len(units))
    }:
        raise _source_map_invalid("replacement_coverage")
    deleted = {}
    for removal in result["deleted_sources"]:
        if not isinstance(removal, dict) or set(removal) != {"unit_id", "reason"}:
            raise _source_map_invalid("deletion_fields")
        uid, reason = removal["unit_id"], removal["reason"]
        if (
            not isinstance(uid, str)
            or uid not in owners
            or uid in deleted
            or uid in linked
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 256
        ):
            raise _source_map_invalid("deletion_values")
        deleted[uid] = reason
    changed = set(patches) | {owners[uid] for uid in deleted}
    expected = {uid for uid, tid in owners.items() if tid in changed}
    if set(linked) | set(deleted) != expected:
        raise _source_map_invalid("original_coverage")
    # Count the complete received canonical metadata, including host positions,
    # before converting or sharing any IDs. Body text has its own existing bound.
    size = nested_json_size(
        {key: result[key] for key in ("source_relations", "deleted_sources")}
    )
    if max_json_bytes is not None and size > max_json_bytes:
        raise EnvelopeExceeded("repair", "source_map_json_bytes", size, max_json_bytes)
    legacy = {
        "outcome": result["outcome"],
        "explanation": result["explanation"],
        "patches": copy.deepcopy(result["patches"]),
        "edit_records": [
            {
                "unit_id": uid,
                "reason": deleted[uid] if uid in deleted else SOURCE_MAP_HOST_REASON,
                "replacement_indexes": sorted(linked.get(uid, [])),
            }
            for uid in owners
            if uid in expected
        ],
    }
    if result["outcome"] == "declined":
        if changed:
            raise _source_map_invalid("declined_with_changes")
        return {**legacy, "removed_targets": []}
    return normalize_edit_map(legacy, views)
