"""Replay source-map feedback across valid operational capacity revisions."""

import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
from test_host_lossless_identity import FLAGS
from test_writer_v2 import case  # noqa: F401

import litagent.eval.scoped as scoped
import litagent.writing_flat as flat
from litagent.body_quality import bound_repair_diagnostics
from litagent.capacity_repair import (
    capacity_targets,
    normalize_source_map,
    original_unit_directory,
    source_map_identity,
)
from litagent.quality import digest
from litagent.report_render import build_report_source_units
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_budget import _digest, validate_writer_envelope
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    bound_fragments,
    compile_document,
    current_repair_guidance_context,
    guidance_wire_fields,
    patch_provenance,
    repair_feedback_directory,
    repair_request_input,
    source_map_response_fields,
    support_validation_fields,
    writer_budget_plan,
)

MAP = "capacity-source-map-v3"
OLD_WIRE_ID = "88f75ad0eae1b398352faa39da1b79c460676bd783d4ea0dfadb049699dcf138"
OLD_MAP_ID = "d765ac91fe0dfbf283d0f0cfe939a87511970c17435900953fda10f03e28f3ce"
OLD_FEEDBACK_ID = "12dc1548f64557da7f6e9e63da70be509e6d1cdc3261e8ad698569ed6e7755f6"
LEGACY_PLAN_DIGEST = "b35cdb2b22425163eca9de67f41352fe201fdc433138d2433c29f98165f63d6f"
OLD_V3_BUILDER = "3f37e93d38589d47f23e287733154d0a10ed2aa9b6ecd4e65e38781b34049d5b"


def plan_for(case, mapping=None, *, wire=False):
    """Build actual v7 plans with explicit mapping choices, including real v2."""
    return writer_budget_plan(
        case.packet,
        case.aliases,
        **FLAGS,
        methodology={},
        budget_policy="planned-budget-v7",
        transport_revision="host-lossless-transport-v1",
        support_validation_revision="host-support-receipts-v2",
        repair_mapping_contract=mapping,
        **(
            {
                "repair_guidance_contract": "host-structured-repair-v1",
                "repair_guidance_wire_revision": "host-unit-ref-guidance-v1",
            }
            if wire
            else {}
        ),
    )


def capacity_envelope(envelope, assertions, batches=None):
    """Keep immutable source rules while declaring a separately valid capacity."""
    value = copy.deepcopy(envelope)
    value.update(
        model_assertions=assertions,
        assessment_batches=(assertions + 15) // 16 if batches is None else batches,
    )
    value["identity"] = _digest({k: v for k, v in value.items() if k != "identity"})
    return validate_writer_envelope(value)


def diagnostic_for(survey, bindings, section):
    """Create non-gold feedback at exact production body or summary positions."""
    fragments = bound_fragments(survey, bindings)
    native = {unit["unit_id"]: unit for unit in bindings["units"]}
    for unit in build_report_source_units(survey):
        for fragment in fragments.get(unit.unit_id, []):
            row = native[fragment["source_fragment_id"].split(":native:", 1)[1]]
            if row["path"][0] != section:
                continue
            start = unit.start + fragment["start"]
            end = unit.start + fragment["end"]
            quote = survey[start:end]
            return {
                "claim_id": "feedback-" + section,
                "statement_id": "feedback-" + section,
                "kind": "scientific",
                "correct_kind": "scientific",
                "claim_text": quote.split(" [E:", 1)[0],
                "source_unit_id": unit.unit_id,
                "source_fragment_id": fragment["source_fragment_id"],
                "source_quote": quote,
                "source_start": start,
                "source_end": end,
                "section": section,
                "evidence_ids": fragment["citation_aliases"],
                "local_error": None,
                "atomic": True,
                "classification_valid": True,
                "source_faithful": True,
                "status": "unsupported",
                "reason_code": "insufficient_support",
                "explanation": "Synthetic positional feedback, not scientific gold.",
                "support_basis": [],
                "aspect_checks": [],
                "alignment_status": "exact",
                "citation_scope": "exact",
                "diagnostic_source": "survey-evaluation-v3",
            }
    raise AssertionError("missing native fragment")


def feedback_case(
    case, current_count=128, prior_count=176, *, wire=True, prior_batches=None
):
    """Build both reports and their authentic raw/host receipts without an owner."""
    scope = plan_for(case, MAP, wire=wire)["envelope"]
    current = capacity_envelope(scope, current_count)
    prior_scope = capacity_envelope(scope, prior_count, prior_batches)
    document = {
        "packet": copy.deepcopy(case.packet),
        "aliases": copy.deepcopy(case.aliases),
        "body": copy.deepcopy(case.body),
        "summaries": copy.deepcopy(case.summaries),
        "generation_writing_plan": RANGE_WRITING_PLAN,
        "budget_envelope": current,
    }
    survey, bindings = compile_document(
        document["body"],
        document["summaries"],
        case.cards,
        case.aliases,
        "en",
        {},
        writing_plan=RANGE_WRITING_PLAN,
        envelope=current,
    )
    targets = capacity_targets(document["body"], {}, maximum=current["body_max_units"])
    capacity = {
        "reason": "capacity_remaining_workflow_unaffordable",
        "source_report_sha256": bindings["report_sha256"],
        "actual_assertions": 188,
        "target_assertions": 176,
        "target_batches": 11,
        "assertions_by_native_unit": {
            row["unit_id"]: 188 // len(bindings["units"])
            + (index < 188 % len(bindings["units"]))
            for index, row in enumerate(bindings["units"])
        },
    }
    captured = []
    context = (
        current_repair_guidance_context(
            survey,
            bindings,
            document,
            targets,
            [capacity],
            diagnostic_categories=["capacity"],
        )
        if wire
        else None
    )
    _, _, schema = repair_request_input(
        case.packet,
        document["body"],
        targets,
        case.aliases,
        cause="capacity",
        previous_failure=None,
        writing_plan=RANGE_WRITING_PLAN,
        envelope=current,
        wire_contract="flat-writer-v1",
        capacity_diagnostic=capacity,
        **(
            {
                "guidance_context": {"current": context, "prior_candidate": None},
                "guidance_wire_capture": captured.append,
            }
            if wire
            else {}
        ),
    )
    directory = original_unit_directory(targets)
    candidate_body = copy.deepcopy(document["body"])
    candidate_body[BODY[0]][0]["sentences"][0]["text"] += " Revised wording."
    canonical = {
        "outcome": "patched",
        "explanation": "Explicit original relations; no scientific verdict.",
        "patches": [],
        "source_relations": [],
        "deleted_sources": [],
    }
    for tid, target in targets.items():
        section, pi, _ = target["path"]
        canonical["patches"].append(
            {
                "target_id": tid,
                "sentences": copy.deepcopy(candidate_body[section][pi]["sentences"]),
            }
        )
        for index, uid in enumerate(directory[tid]):
            canonical["source_relations"].append(
                {"target_id": tid, "replacement_index": index, "source_unit_ids": [uid]}
            )
    raw = json.dumps(flat.encode(canonical, schema))
    parsed = flat.parse(
        raw,
        schema,
        reference_contract=current["reference_contract"],
        transport_revision=current["transport_revision"],
    )
    normalized = normalize_source_map(
        parsed["canonical"], targets, max_json_bytes=current["source_map_json_bytes"]
    )
    replacements = {row["target_id"]: row["sentences"] for row in normalized["patches"]}
    provenance = patch_provenance(
        document["body"], targets, replacements, candidate_body, cause="capacity"
    )
    candidate, candidate_bindings = compile_document(
        candidate_body,
        document["summaries"],
        case.cards,
        case.aliases,
        "en",
        {},
        writing_plan=RANGE_WRITING_PLAN,
        envelope=prior_scope,
    )
    guidance_record = None
    if wire:
        assert len(captured) == 1
        guidance_record = captured[0]
        guidance_record["authorized_targets"] = copy.deepcopy(targets)
        guidance_record["prior_replay_witness"] = None
        guidance_record["identity"] = digest(
            {k: v for k, v in guidance_record.items() if k != "identity"}
        )
    details = {
        "unsupported_claims": [
            diagnostic_for(candidate, candidate_bindings, section)
            for section in (BODY[0], "abstract")
        ]
    }
    prior = {
        "committed": False,
        "source_report_sha256": bindings["report_sha256"],
        "report_sha256": candidate_bindings["report_sha256"],
        "survey": candidate,
        "bindings": candidate_bindings,
        "writer_document": {
            **copy.deepcopy(document),
            "body": candidate_body,
            "budget_envelope": prior_scope,
        },
        "evaluation": {"faithfulness": {"details": details}},
        "repair_provenance": provenance,
        "repair_receipt": {
            "authorized_targets": copy.deepcopy(targets),
            **guidance_wire_fields(current),
            "guidance_wire_record": guidance_record,
            "raw": raw,
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "schema": schema,
            "canonical_sha256": digest(parsed["canonical"]),
            "wire_bindings_sha256": digest(parsed["bindings"]),
            "received_canonical_sha256": digest(parsed["received_canonical"]),
            "reference_contract": current["reference_contract"],
            "transport_revision": current["transport_revision"],
            "transport_identity": flat.transport_identity(),
            "original_unit_directory_sha256": digest(directory),
            "source_relation_bindings": copy.deepcopy(
                parsed["source_relation_bindings"]
            ),
            **support_validation_fields(current),
            **source_map_response_fields(current, parsed),
        },
    }
    failure = {
        "code": "writer_candidate_failed",
        "family": "quality",
        "candidate_sha256": prior["report_sha256"],
        "diagnostics": bound_repair_diagnostics(details),
    }
    return SimpleNamespace(
        survey=survey,
        document=document,
        bindings=bindings,
        prior=prior,
        failure=failure,
        targets=targets,
    )


def feedback(value):
    """Run the production replay against its complete positional inputs."""
    return repair_feedback_directory(
        value.document["packet"],
        value.survey,
        value.document,
        value.bindings,
        value.prior,
        value.failure,
        value.targets,
    )


@pytest.mark.parametrize("counts", [(128, 176), (176, 128), (176, 176)])
@pytest.mark.parametrize("wire", [False, True])
def test_legal_capacity_changes_keep_complete_current_prior_feedback(
    case, counts, wire
):
    """Keep exact body and summary lineage across both directions of capacity."""
    value = feedback_case(case, *counts, wire=wire)
    before = copy.deepcopy(value)
    result = feedback(value)
    assert result["current_report_sha256"] == value.bindings["report_sha256"]
    assert result["prior_report_sha256"] == value.prior["report_sha256"]
    assert result["current_report_sha256"] != result["prior_report_sha256"]
    assert len(result["fragments"]) == 2
    assert any(row["body_premise_paths"] for row in result["fragments"].values())
    for row in result["fragments"].values():
        assert (
            row["text"]
            == value.prior["survey"][row["source_start"] : row["source_end"]]
        )
        assert row["current_target_ids"]
    assert value == before


@pytest.fixture
def pinned_definition(monkeypatch):
    """Isolate mapping compatibility from deliberate later Evaluation revisions."""
    original = scoped.support_validation_identity

    def pinned(revision=scoped.SUPPORT_VALIDATION_REVISION):
        if revision == scoped.SUPPORT_VALIDATION_REVISION_V2:
            return "832c62a700de010d1ba7536139a8c6872071aa3cfb9036538cb35ed428025ffc"
        return original(revision)

    monkeypatch.setattr(scoped, "support_validation_identity", pinned)


@pytest.mark.parametrize("mapping", [None, "capacity-edit-map-v2"])
def test_actual_none_and_v2_plans_preserve_pinned_pre296_identity(
    case, mapping, pinned_definition
):
    """Pin the old Evaluation identity while checking actual mapping defaults."""
    plan = plan_for(case, mapping)
    assert digest(plan) == LEGACY_PLAN_DIGEST
    assert (
        plan["identity"]
        == "ca7ab4d75d001fdcab97caa87a30009f2454573f1fe0ad18f4ade29efe5b10f6"
    )
    assert (
        plan["envelope"]["identity"]
        == "07356c970b9e4eb8fc8e1fc799f17bfcbe4eed783854cc54919a0d2cf1a46a50"
    )
    assert (
        plan["envelope"]["builder_identity"]
        == "1078f251a9a6453e0ff8813365234ebbfbb2d4016ea690988779bd57b4b1be7d"
    )
    assert plan["envelope"]["repair_mapping_contract"] == "capacity-edit-map-v2"
    assert "source_map_identity" not in plan["envelope"]


def test_only_source_map_wire_rule_versions_runtime_not_pinned_fee_data(
    case, monkeypatch, pinned_definition
):
    """Reject old executable semantics while authenticating their unchanged data."""
    assert source_map_identity() == OLD_MAP_ID
    assert flat.repair_feedback_identity() == OLD_FEEDBACK_ID
    assert flat.source_map_wire_identity() != OLD_WIRE_ID
    scope = plan_for(case, MAP)["envelope"]
    assert scope["builder_identity"] != OLD_V3_BUILDER
    stale = copy.deepcopy(scope)
    stale.update(source_map_identity=OLD_WIRE_ID, builder_identity=OLD_V3_BUILDER)
    stale["identity"] = _digest({k: v for k, v in stale.items() if k != "identity"})
    assert stale["identity"] == (
        "c8c8627c999e06c4232dd99917d25eb64e61b191e08e4f75f0b0cfebce785e10"
    )
    before = copy.deepcopy(stale)
    with pytest.raises(ValueError, match="writer_envelope_source_map_invalid"):
        validate_writer_envelope(stale)

    def forbidden(*args, **kwargs):
        raise AssertionError("historical fee data must not call current runtime rules")

    monkeypatch.setattr(flat, "source_map_wire_identity", forbidden)
    assert accounting_envelope(stale).max_body_units == 66
    assert stale == before


@pytest.mark.parametrize("field", ["model_assertions", "assessment_batches"])
@pytest.mark.parametrize("coercion", [bool, float])
@pytest.mark.parametrize("origin", ["current", "prior"])
def test_operational_capacity_types_remain_strict_after_rehash(
    case, field, coercion, origin
):
    """Require each document's real integer counts even with a consistent hash."""
    value = feedback_case(case)
    scope = (
        value.document["budget_envelope"]
        if origin == "current"
        else value.prior["writer_document"]["budget_envelope"]
    )
    scope[field] = coercion(scope[field])
    scope["identity"] = _digest({k: v for k, v in scope.items() if k != "identity"})
    before = copy.deepcopy(value)
    with pytest.raises(WriterError, match="writer_feedback_identity_invalid"):
        feedback(value)
    assert value == before


@pytest.mark.parametrize(
    "field",
    [
        "source_identity",
        "builder_identity",
        "body_json_bytes",
        "request_caps",
        "source_map_json_bytes",
        "capacity",
        "source_map_identity",
        "unknown",
        "identity",
    ],
)
def test_nonoperational_envelope_drift_still_rejects(case, field):
    """Reject foreign static scopes even when their native bindings are valid."""
    value = feedback_case(case)
    scope = value.prior["writer_document"]["budget_envelope"]
    if field in {"source_identity", "builder_identity", "source_map_identity"}:
        scope[field] = "f" * 64
    elif field == "request_caps":
        scope[field]["repair"] += 1
    elif field == "source_map_json_bytes":
        for name in (
            "source_map_json_bytes",
            "diagnostic_json_bytes",
            "repair_guidance_json_bytes",
        ):
            scope[name] -= 1
    elif field == "capacity":
        scope[field]["max_assertions"] = 512
    elif field == "unknown":
        scope["untrusted_authority"] = 176
    elif field != "identity":
        scope[field] += 1
    scope["identity"] = (
        "0" * 64
        if field == "identity"
        else _digest({k: v for k, v in scope.items() if k != "identity"})
    )
    try:
        validate_writer_envelope(scope)
    except ValueError:
        pass
    else:
        # Recompile with the valid but foreign scope so a later native-binding
        # mismatch cannot accidentally stand in for the static ownership guard.
        prior_doc = value.prior["writer_document"]
        candidate, binding = compile_document(
            prior_doc["body"],
            prior_doc["summaries"],
            case.cards,
            case.aliases,
            "en",
            {},
            writing_plan=RANGE_WRITING_PLAN,
            envelope=scope,
        )
        assert candidate == value.prior["survey"]
        value.prior["bindings"] = binding
    before = copy.deepcopy(value)
    with pytest.raises(WriterError):
        feedback(value)
    assert value == before


@pytest.mark.parametrize(
    "fault",
    [
        "packet",
        "parent",
        "raw",
        "canonical",
        "targets",
        "position",
        "native_path",
        "wire",
    ],
)
def test_capacity_flexibility_preserves_actual_receipt_and_position_guards(case, fault):
    """Never trade authentic raw replies, exact ranges or lineage for capacity."""
    value = feedback_case(case)
    if fault == "packet":
        value.prior["writer_document"]["packet"]["query"] += " forged"
    elif fault == "parent":
        value.prior["source_report_sha256"] = "0" * 64
    elif fault == "raw":
        value.prior["repair_receipt"]["raw"] += " "
    elif fault == "canonical":
        value.prior["repair_receipt"]["canonical_sha256"] = "0" * 64
    elif fault == "targets":
        next(iter(value.prior["repair_receipt"]["authorized_targets"].values()))[
            "original_sha256"
        ] = ("0" * 64)
    elif fault == "position":
        for diagnostic in value.prior["evaluation"]["faithfulness"]["details"][
            "unsupported_claims"
        ]:
            diagnostic["source_start"] += 1
        value.failure["diagnostics"] = bound_repair_diagnostics(
            value.prior["evaluation"]["faithfulness"]["details"]
        )
    elif fault == "native_path":
        value.prior["bindings"]["units"][0]["path"][-1] += 1
    else:
        value.prior["repair_receipt"]["guidance_wire_record"]["payload_sha256"] = (
            "0" * 64
        )
    before = copy.deepcopy(value)
    with pytest.raises(WriterError):
        feedback(value)
    assert value == before


@pytest.mark.parametrize(
    "assertions,batches", [(127, 8), (1025, 65), (176, 10), (176, 65)]
)
def test_revised_operation_counts_keep_configuration_and_batch_bounds(
    case, assertions, batches
):
    """Keep configured bounds and enough assessment slots for every assertion."""
    value = feedback_case(case)
    scope = value.prior["writer_document"]["budget_envelope"]
    scope.update(model_assertions=assertions, assessment_batches=batches)
    scope["identity"] = _digest({k: v for k, v in scope.items() if k != "identity"})
    before = copy.deepcopy(value)
    with pytest.raises(WriterError, match="writer_feedback_identity_invalid"):
        feedback(value)
    assert value == before


def test_more_than_minimum_batches_is_still_an_independently_valid_envelope(case):
    """Accept legal slots without imposing an unproved minimum-batch equality."""
    value = feedback_case(case, prior_batches=12)
    before = copy.deepcopy(value)
    assert feedback(value)["prior_report_sha256"] == value.prior["report_sha256"]
    assert value == before


def test_equal_quotes_at_distinct_body_and_summary_positions_keep_separate_lineage(
    case,
):
    """Keep identical prose at different positions distinct and reject relocation."""
    revised = case.body[BODY[0]][0]["sentences"][0]["text"] + " Revised wording."
    case.summaries["abstract"][0]["sentences"][0].update(
        text=revised, body_unit_refs=["u1"]
    )
    value = feedback_case(case)
    rows = list(feedback(value)["fragments"].values())
    assert len(rows) == 2
    assert rows[0]["text"] == rows[1]["text"]
    assert rows[0]["source_start"] != rows[1]["source_start"]
    assert rows[0]["native_paths"] != rows[1]["native_paths"]
    diagnostics = value.prior["evaluation"]["faithfulness"]["details"][
        "unsupported_claims"
    ]
    diagnostics[1]["source_start"] = diagnostics[0]["source_start"]
    diagnostics[1]["source_end"] = diagnostics[0]["source_end"]
    value.failure["diagnostics"] = bound_repair_diagnostics(
        value.prior["evaluation"]["faithfulness"]["details"]
    )
    before = copy.deepcopy(value)
    with pytest.raises(WriterError, match="writer_feedback_identity_invalid"):
        feedback(value)
    assert value == before
