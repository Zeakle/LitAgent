"""Keep two complete diagnostic origins through the fixed lossless input codec."""

import base64
import copy
import json

import pytest
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_repair_task_view import (
    decoded_issue_rows,
    encode,
    issue_rows,
    rehash_view,
    replace_value,
    restore,
)
from test_semantic_conflict_identity import scope
from test_semantic_conflict_repair_pipeline import prepare_case, run_conflict
from test_semantic_conflict_wire import (
    EFFECT_KEYS,
    NEGATIVE,
    POSITIVE,
    build_captured_input,
    semantic_input,
)
from test_source_map_pipeline import products, sent_with
from test_source_owner_pipeline import FLAGS
from test_structured_repair_guidance import guidance_case  # noqa: F401
from test_structured_repair_guidance import context_for, rebind
from test_writer_v2 import case  # noqa: F401

import litagent.writing_flat as flat
import litagent.writing_v2 as writing
from litagent.eval.semantic_conflict import (
    EFFECT_CONTRACT,
    SEMANTIC_CONFLICT_POLICY,
    semantic_conflict_identity,
)
from litagent.quality import digest
from litagent.writing_v2 import WriterError

DIGEST_FIELDS = [
    "assessment_conflict_effect.decision_sha256",
    "assessment_conflict_effect.binding_sha256",
]
ASPECT_COLUMNS = ["aspect", "supported", "support_refs"]
DIGEST_MARKER = {
    "encoding": "python-base64-b85encode-32bytes-v1",
    "fields": DIGEST_FIELDS,
    "decoded_format": "64-lowercase-hex",
}


def catalog_origins(count=52, *, effects="both"):
    """Describe complete synthetic typed issues, never scientific gold labels."""
    origins = {}
    for origin in ("current", "prior_candidate"):
        issues = []
        for index in range(count):
            names = ["predicate", "conditions"]
            if index >= 22:
                names += ["quantities" if index % 2 else "causal_relation"]
            positive = index % 2 == 0
            issue = {
                "category": "evaluation",
                "claim_id": f"claim-{index}",
                "kind": "scientific",
                "claim_text": f"合成契约断言 {index}：Δ café e\u0301 / é 🧪，原字面条件保持。",
                "status": "unverifiable",
                "reason_code": "unverifiable",
                "local_error": "evaluation_decision_consistency_invalid",
                "target_ids": ["r1"],
                "fragment_refs": ["f1"],
                "body_paths": [],
                "support_basis_refs": ["s1"],
                "aspect_checks": [
                    {
                        "aspect": name,
                        "supported": not (positive and offset == 0),
                        "support_refs": [] if positive and offset == 0 else ["s1"],
                    }
                    for offset, name in enumerate(names)
                ],
                "support_validation": {
                    "status": "unverified",
                    "issue": "evaluation_decision_consistency_invalid",
                    "required_aspects": names,
                    "actual_aspects": names,
                    "context": "不同位置可共享字面模板，不能共享来源权限。",
                },
            }
            if effects in {"both", origin}:
                issue["assessment_conflict_effect"] = {
                    "contract": EFFECT_CONTRACT,
                    "policy": SEMANTIC_CONFLICT_POLICY,
                    "policy_identity": semantic_conflict_identity(),
                    "code": POSITIVE if positive else NEGATIVE,
                    "original_status": "supported" if positive else "unsupported",
                    "original_reason_code": (
                        "supported" if positive else "record_mismatch"
                    ),
                    "decision_sha256": digest([origin, index, "decision"]),
                    "binding_sha256": digest([origin, index, "binding"]),
                }
            issues.append(issue)
        origins[origin] = issues
    return origins


def projection(origins):
    """Use the real selected encoder while retaining independent original data."""
    return flat._semantic_conflict_template_tables(origins)


def test_all_104_issues_268_aspects_and_208_complete_hashes_round_trip():
    """Independent model-view decoding and host decoding preserve every value."""
    origins = catalog_origins()
    before = copy.deepcopy(origins)
    wire = projection(origins)
    assert wire["aspect_columns"] == ASPECT_COLUMNS
    assert wire["effect_digest_wire"] == DIGEST_MARKER
    assert flat._restore_semantic_conflict_template_tables(wire) == origins
    decoded = {name: decoded_issue_rows(wire, name) for name in origins}
    assert decoded == origins
    assert sum(len(rows) for rows in decoded.values()) == 104
    assert (
        sum(len(row["aspect_checks"]) for rows in decoded.values() for row in rows)
        == 268
    )
    hashes = [
        row["assessment_conflict_effect"][field]
        for rows in decoded.values()
        for row in rows
        for field in ("decision_sha256", "binding_sha256")
    ]
    assert len(hashes) == len(set(hashes)) == 208
    assert all(
        set(row["assessment_conflict_effect"]) == EFFECT_KEYS
        for rows in decoded.values()
        for row in rows
    )
    assert all(len(value) == 64 and value == value.lower() for value in hashes)
    templates = issue_rows(wire["issue_templates"])
    assert len(templates) == 52
    assert all(
        not {"decision_sha256", "binding_sha256"}
        & row["assessment_conflict_effect"].keys()
        for row in templates
    )
    for name in origins:
        occurrences = issue_rows(wire[name])
        assert len(occurrences) == 52
        assert all(
            len(encoded) == 40 and encoded.isascii()
            for row in occurrences
            for encoded in row["assessment_conflict_effect"].values()
        )
    assert origins == before


@pytest.mark.parametrize("effects", ["both", "current", "prior_candidate", "none"])
@pytest.mark.parametrize("prior", [False, True])
def test_effect_absence_and_single_origin_never_invent_missing_hashes(effects, prior):
    """Only an existing complete effect gets the two digest occurrence fields."""
    origins = catalog_origins(2, effects=effects)
    if not prior:
        origins["prior_candidate"] = None
    wire = projection(origins)
    assert flat._restore_semantic_conflict_template_tables(wire) == origins
    for name, expected in origins.items():
        assert decoded_issue_rows(wire, name) == expected
        if expected is None:
            continue
        for original, occurrence in zip(expected, issue_rows(wire[name]), strict=True):
            assert ("assessment_conflict_effect" in occurrence) is (
                "assessment_conflict_effect" in original
            )


def test_template_sharing_requires_complete_literal_equality_and_keeps_occurrences():
    """Same labels are interpreted within each origin's independent location table."""
    origins = catalog_origins(2)
    wire = projection(origins)
    assert len(issue_rows(wire["issue_templates"])) == 2
    assert [row["issue_template_index"] for row in issue_rows(wire["current"])] == [
        row["issue_template_index"] for row in issue_rows(wire["prior_candidate"])
    ]
    origins["prior_candidate"][0]["claim_text"] += " A different literal."
    changed = projection(origins)
    assert len(issue_rows(changed["issue_templates"])) == 3
    assert flat._restore_semantic_conflict_template_tables(changed) == origins


@pytest.mark.parametrize(
    "fault",
    [
        "missing_templates",
        "missing_marker",
        "unknown_root",
        "unknown_marker",
        "unknown_encoding",
        "wrong_digest_fields",
        "wrong_decoded_format",
        "aspect_order",
        "extra_aspect_column",
        "bool_index",
        "negative_index",
        "large_index",
        "extra_occurrence",
        "template_digest_collision",
        "group_bool",
        "field_collision",
        "bad_aspect_bool",
        "bad_aspect_length",
    ],
)
def test_closed_catalog_and_marker_boundaries_reject_malformed_views(fault):
    """A declared representation has one explicit inverse without format guessing."""
    wire = projection(catalog_origins(2))
    table = wire["issue_templates"]
    if fault == "missing_templates":
        wire.pop("issue_templates")
    elif fault == "missing_marker":
        wire.pop("effect_digest_wire")
    elif fault == "unknown_root":
        wire["future"] = False
    elif fault == "unknown_marker":
        wire["effect_digest_wire"]["future"] = False
    elif fault == "unknown_encoding":
        wire["effect_digest_wire"]["encoding"] = "guess-hex-or-base85"
    elif fault == "wrong_digest_fields":
        wire["effect_digest_wire"]["fields"].reverse()
    elif fault == "wrong_decoded_format":
        wire["effect_digest_wire"]["decoded_format"] = "short-hash"
    elif fault == "aspect_order":
        wire["aspect_columns"].reverse()
    elif fault == "extra_aspect_column":
        wire["aspect_columns"].append("explanation")
    elif fault in {"bool_index", "negative_index", "large_index"}:
        replace_value(
            wire["current"],
            0,
            "issue_template_index",
            {"bool_index": False, "negative_index": -1, "large_index": 200}[fault],
        )
    elif fault == "extra_occurrence":
        wire["current"]["issue_shared"]["future"] = False
    elif fault == "template_digest_collision":
        table["issue_shared"][DIGEST_FIELDS[0]] = "a" * 64
    elif fault == "group_bool":
        table["issues"][0][0] = False
    elif fault == "field_collision":
        key = next(iter(table["issue_shared"]))
        table["issue_columns"][0]["shared"][key] = copy.deepcopy(
            table["issue_shared"][key]
        )
    else:
        checks = copy.deepcopy(issue_rows(table)[0]["aspect_checks"])
        if fault == "bad_aspect_bool":
            checks[0][1] = 0
        else:
            checks[0].append("cannot omit")
        replace_value(table, 0, "aspect_checks", checks)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat._restore_semantic_conflict_template_tables(wire)


@pytest.mark.parametrize(
    "encoded",
    [
        "a" * 64,
        "A" * 64,
        "0" * 39,
        "0" * 41,
        " " * 40,
        "é" * 40,
        "~" * 40,
        None,
        False,
        [],
    ],
)
def test_base85_hashes_require_exact_canonical_32byte_scalars(encoded):
    """Hex strings and malformed Base85 do not become an implicit alternate codec."""
    wire = projection(catalog_origins(2))
    replace_value(wire["current"], 0, DIGEST_FIELDS[0], encoded)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        flat._restore_semantic_conflict_template_tables(wire)


@pytest.mark.parametrize(
    "fault",
    [
        "template_order",
        "occurrence_order",
        "valid_hash",
        "cross_origin_support",
        "cross_origin_location",
        "duplicate_occurrence",
    ],
)
def test_rehashed_catalog_cannot_replace_original_authority(guidance_case, fault):
    """Even a valid lossless inverse cannot authorize a changed occurrence or source."""
    args = semantic_input(guidance_case)
    wire, receipt = encode(args)
    assert restore(args, wire, receipt) == args[0]
    if fault == "template_order":
        wire["issue_templates"]["issues"].reverse()
    elif fault == "occurrence_order":
        wire["current"]["issues"].reverse()
    elif fault == "valid_hash":
        replace_value(
            wire["current"],
            0,
            DIGEST_FIELDS[1],
            base64.b85encode(bytes.fromhex("f" * 64)).decode("ascii"),
        )
    elif fault == "cross_origin_support":
        wire["prior_candidate"]["support_refs"]["s1"]["range"][0] = 1
    elif fault == "cross_origin_location":
        wire["prior_candidate"]["units"][0] = copy.deepcopy(wire["current"]["units"][0])
    else:
        wire["current"]["issues"].append(copy.deepcopy(wire["current"]["issues"][0]))
    rehash_view(wire, receipt)
    with pytest.raises(WriterError, match="writer_guidance_wire_invalid"):
        restore(args, wire, receipt)


def test_equal_literal_template_does_not_merge_distinct_origin_coordinates(
    guidance_case,
):
    """Identical f1/s1 labels retain separate report SHA, unit paths and source maps."""
    args = semantic_input(guidance_case)
    wire, receipt = encode(args)
    assert (
        wire["current"]["source_report_sha256"]
        != wire["prior_candidate"]["source_report_sha256"]
    )
    assert wire["current"]["units"] != wire["prior_candidate"]["units"]
    assert decoded_issue_rows(wire, "current")[0]["fragment_refs"] == ["f1"]
    assert decoded_issue_rows(wire, "prior_candidate")[0]["fragment_refs"] == ["f1"]
    assert restore(args, wire, receipt) == args[0]


@pytest.mark.parametrize("empty", [None, []])
def test_empty_current_origin_still_fails_original_request_authority(
    guidance_case, empty
):
    """A pure empty table does not grant an actual report-repair request."""
    case = copy.deepcopy(guidance_case)
    body, targets, packet = case[:3]
    packet.update(
        query="Bounded source review",
        language="en",
        requirements=[],
        chapter_recommendations={key: ["c1"] for key in writing.BODY},
    )
    current = None
    if empty == []:
        current = context_for(case, diagnostics=[])
        current["host_directory"]["fragments"] = {}
        rebind(current)
    with pytest.raises(WriterError):
        writing.repair_request_input(
            packet,
            body,
            {"r1": targets["r1"]},
            {"c1": "E1"},
            cause="evaluation",
            previous_failure=None,
            writing_plan=writing.RANGE_WRITING_PLAN,
            envelope=scope(),
            wire_contract="flat-writer-v1",
            guidance_context={"current": current, "prior_candidate": None},
        )


def test_unselected_complete_request_never_calls_the_new_template_codec(
    guidance_case, monkeypatch
):
    """New selected helpers and instructions leave the None request bytes unchanged."""

    def actual_request():
        (system, payload, schema), _, packet, envelope = build_captured_input(
            guidance_case, selected=False
        )
        system, payload = writing.prepare_planned_input(
            system, payload, packet, {"c1": "E1"}, writing.RANGE_WRITING_PLAN, envelope
        )
        return writing.request_messages(
            system, payload, schema, wire_contract="flat-writer-v1"
        ), writing.provider_response_format(
            "repair", schema, wire_contract="flat-writer-v1"
        )

    before = actual_request()
    legacy_identity = flat.repair_guidance_wire_identity(flat.REPAIR_TASK_VIEW_REVISION)

    def forbidden(*args, **kwargs):
        """A None owner cannot enter a newly selected representation."""
        raise AssertionError("The new template codec ran for an unselected owner.")

    monkeypatch.setattr(flat, "_semantic_conflict_template_tables", forbidden)
    monkeypatch.setattr(
        flat, "SEMANTIC_CONFLICT_TEMPLATE_INSTRUCTION", "New selected-only instruction."
    )
    assert actual_request() == before
    assert (
        flat.repair_guidance_wire_identity(flat.REPAIR_TASK_VIEW_REVISION)
        == legacy_identity
    )


@pytest.mark.asyncio
async def test_guidance_identity_change_isolates_old_plan_and_session_checkpoint(
    case, tmp_path, monkeypatch
):
    """Core policy identity stays equal while executable guidance changes the owner."""
    old_gateway, new_gateway = FlatGateway(case), FlatGateway(case)
    old_owner, old_budget, _ = await owner_for(old_gateway)
    new_owner, new_budget, _ = await owner_for(new_gateway)
    settings = {**FLAGS, "semantic_conflict_policy": SEMANTIC_CONFLICT_POLICY}
    try:
        with monkeypatch.context() as previous:
            previous.setattr(
                flat, "semantic_conflict_guidance_identity", lambda: "e" * 64
            )
            old_plan = writing.writer_budget_plan(
                case.packet,
                case.aliases,
                methodology={},
                budget_policy="planned-budget-v7",
                **settings,
            )
            old_budget.activate_plan(old_plan, path=tmp_path / "old-budget.json")
            old_session = make_session(old_owner, case, tmp_path)
            old_session.save()
            old_bytes = old_session.journal_path.read_bytes()
        new_plan = writing.writer_budget_plan(
            case.packet,
            case.aliases,
            methodology={},
            budget_policy="planned-budget-v7",
            **settings,
        )
        assert (
            old_plan["envelope"]["semantic_conflict_identity"]
            == new_plan["envelope"]["semantic_conflict_identity"]
        )
        assert (
            old_plan["envelope"]["builder_identity"]
            != new_plan["envelope"]["builder_identity"]
        )
        assert old_plan["identity"] != new_plan["identity"]
        new_budget.activate_plan(new_plan, path=tmp_path / "new-budget.json")
        before = new_budget.snapshot()
        with pytest.raises(WriterError, match="writer_checkpoint_identity_conflict"):
            make_session(new_owner, case, tmp_path)
        assert old_session.journal_path.read_bytes() == old_bytes
        assert new_budget.snapshot() == before
        assert not old_gateway.sent and not new_gateway.sent
    finally:
        await old_owner.close()
        await new_owner.close()


@pytest.mark.asyncio
async def test_two_actual_shared_repairs_reassess_complete_candidates_with_catalog(
    case, tmp_path, monkeypatch
):
    """Mock only external replies while real owners settle both repair requests."""
    prepare_case(case)
    report, gateway = await run_conflict(
        case, tmp_path, monkeypatch, failing_candidates=1
    )
    repairs = sent_with(gateway, "targets")
    assert len(repairs) == 2 and gateway.inventories == 3
    assert len(sent_with(gateway, "global_output_contract")) == 3
    for index, entry in enumerate(repairs):
        wire = entry["payload"]["repair_guidance"]
        assert wire["effect_digest_wire"] == DIGEST_MARKER
        assert (wire["prior_candidate"] is not None) is (index == 1)
        restored = flat._restore_semantic_conflict_template_tables(wire)
        for name in restored:
            assert decoded_issue_rows(wire, name) == restored[name]
        assert case.semantic_wrong in json.dumps(entry["payload"]["current_body"])
        assert "AnotherGhostNet" not in json.dumps(entry["payload"]["current_body"])
        for card in case.cards.values():
            assert card["source_quote"] in json.dumps(entry["request"])
    assert report["delivery"]["publishable"] is True
    metric = report["evaluation"]["faithfulness"]
    assert metric["score"] == 1.0 and metric["details"]["coverage"]["complete"] is True
    assert metric["details"]["unverified_count"] == 0
    _, journal, ledger = products(tmp_path)
    assert [candidate["committed"] for candidate in journal["candidates"]] == [
        False,
        True,
    ]
    assert ledger["usage_uncertain"] is False
    assert ledger["requests_sent"] == len(gateway.sent)
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())
