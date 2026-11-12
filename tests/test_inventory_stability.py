"""Keep recovery extraction content stable without inheriting scientific verdicts."""

import copy
import hashlib
import json
import re
from pathlib import Path

import pytest

from litagent.eval import inventory_recovery as recovery
from litagent.eval.scoped import (
    ScopedEvaluationError,
    build_source_fragments,
    flat_inventory_schema,
)
from litagent.evidence import extract_evidence_refs_ordered
from litagent.quality import digest
from litagent.report_render import build_report_source_units
from litagent.writing_flat import (
    TRANSPORT_REVISION,
    inventory_identity_projection,
    inventory_transport_projection,
)


@pytest.fixture
def portable043():
    """Load copied original fields, never a private run directory or a new verdict."""
    return json.loads(
        (Path(__file__).parent / "fixtures/run043-inventory-stability.json").read_text()
    )


def sample():
    """Build host ranges with repeated prose, distinct citations and an empty fragment."""
    survey = (
        "## Methods\n\nA changes embeddings and B adapts distance. [E:a]\n\n"
        "An unsupported result remains explicit. [E:b]\n\n"
        "An unsupported result remains explicit. [E:c]\n\nScope note only.\n"
    )
    claims, inventory, units = [], [], []
    aliases = {eid: f"e{i}" for i, eid in enumerate(("a", "b", "c"), 1)}
    for index, unit in enumerate(build_report_source_units(survey)):
        fragment = build_source_fragments(unit)[0]
        refs = extract_evidence_refs_ordered(fragment["source_quote"])
        empty = index == 3
        if not empty:
            claims.append(
                dict(
                    claim_id=f"claim_{index}",
                    kind="scientific",
                    claim_text=re.sub(r"\s*\[E:[^\]]+\]", "", fragment["source_quote"]),
                    source_unit_id=unit.unit_id,
                    source_fragment_id=fragment["source_fragment_id"],
                    source_start=unit.start + fragment["start"],
                    source_end=unit.start + fragment["end"],
                    source_quote=fragment["source_quote"],
                    evidence_ids=refs,
                    atomic=index != 0,
                    classification_valid=True,
                    source_faithful=True,
                    correct_kind="scientific",
                    explanation="Independent results require separate assertions.",
                    status="unsupported",
                )
            )
        inventory.append(
            dict(
                source_unit_id=unit.unit_id,
                source_fragment_id=fragment["source_fragment_id"],
                assertion_count=0 if empty else 1,
                exclusion_reason="Scope note, no assertion." if empty else "",
            )
        )
        units.append(
            dict(
                source_unit_id=unit.unit_id,
                source_fragments=[
                    dict(
                        source_fragment_id=fragment["source_fragment_id"],
                        text=fragment["source_quote"],
                        citation_aliases=[aliases[eid] for eid in refs],
                    )
                ],
            )
        )
    details = dict(
        assessment_stage="host_validated",
        inventory_check={"status": "failed"},
        source_report_sha256=hashlib.sha256(survey.encode()).hexdigest(),
        assertions=claims,
        global_issues=[],
        stages=dict(
            global_check="completed",
            requests=[
                {"status": "response_received", "usage": {"usage_available": True}}
            ],
            fragment_inventory=inventory,
        ),
    )
    return survey, details, units, aliases


def proposed(details):
    """Change only the authorized fragment; all protected content stays exact."""
    claims = copy.deepcopy(details["assertions"])
    claims[0]["claim_text"] = "A changes embeddings."
    return claims, copy.deepcopy(details["stages"]["fragment_inventory"])


def feedback_for(details, survey):
    return recovery.inventory_recovery_feedback(
        survey, details, contract=recovery.MIXED_CONTRACT, stable=True
    )


def test_selected_factory_preserves_unflagged_and_empty_fragments_without_verdicts():
    survey, details, _, _ = sample()
    before = copy.deepcopy(details)
    feedback = feedback_for(details, survey)
    rows = feedback["preserved_fragments"]
    assert details == before
    assert len(rows) == 3 and rows[-1]["assertions"] == []
    assert rows[-1]["exclusion_reason"] == "Scope note, no assertion."
    assert rows[0]["assertions"] == rows[1]["assertions"]
    assert rows[0]["source_start"] != rows[1]["source_start"]
    assert rows[0]["evidence_ids"] == ["b"] and rows[1]["evidence_ids"] == ["c"]
    assert feedback["preserved_fragments_sha256"] == digest(rows)
    assert all(
        set(c) == {"kind", "claim_text"} for row in rows for c in row["assertions"]
    )
    candidate, inventory = proposed(details)
    recovery.validate_recovered_inventory(
        candidate, feedback, fragment_inventory=inventory
    )


@pytest.mark.parametrize(
    "fault",
    [
        "drop",
        "soften",
        "kind",
        "range",
        "citation",
        "empty_reason",
        "duplicate",
        "missing_inventory",
    ],
)
def test_preserved_unsupported_content_and_empty_fragments_cannot_drift(fault):
    survey, details, _, _ = sample()
    feedback = feedback_for(details, survey)
    claims, inventory = proposed(details)
    if fault == "drop":
        claims.pop(1)
        inventory[1]["assertion_count"] = 0
    elif fault == "soften":
        claims[1]["claim_text"] = "A possible result."
    elif fault == "kind":
        claims[1]["kind"] = "report_structure"
    elif fault == "range":
        claims[1]["source_start"] += 1
    elif fault == "citation":
        claims[1]["evidence_ids"] = ["a"]
    elif fault == "empty_reason":
        inventory[-1]["exclusion_reason"] = "Different exclusion."
    elif fault == "duplicate":
        inventory.append(copy.deepcopy(inventory[1]))
    else:
        inventory = None
    with pytest.raises(ValueError, match="preserved_fragment_changed"):
        recovery.validate_recovered_inventory(
            claims, feedback, fragment_inventory=inventory
        )


def test_preserved_order_is_exact_but_decision_flags_and_claim_ids_are_not_inherited():
    survey, details, _, _ = sample()
    details["assertions"].insert(
        2,
        {
            **details["assertions"][1],
            "claim_id": "extra",
            "claim_text": "A second explicit unsupported fact.",
        },
    )
    details["stages"]["fragment_inventory"][1]["assertion_count"] = 2
    feedback = feedback_for(details, survey)
    claims, inventory = proposed(details)
    for index, claim in enumerate(claims):
        claim.update(claim_id=f"new_{index}", status="supported", atomic=False)
    recovery.validate_recovered_inventory(
        claims, feedback, fragment_inventory=inventory
    )
    claims[1], claims[2] = claims[2], claims[1]
    with pytest.raises(ValueError, match="preserved_fragment_changed"):
        recovery.validate_recovered_inventory(
            claims, feedback, fragment_inventory=inventory
        )


def test_missing_assertion_concern_authorizes_the_whole_actual_unit():
    survey, details, _, _ = sample()
    details["assertions"][0]["atomic"] = True
    details["global_issues"] = [
        dict(
            owner="inventory",
            code="missing_assertion",
            source_unit_id=details["assertions"][1]["source_unit_id"],
            claim_ids=[],
            explanation="An assertion may be missing.",
        )
    ]
    feedback = feedback_for(details, survey)
    assert feedback["findings"][0]["source_fragment_ids"] == [
        details["assertions"][1]["source_fragment_id"]
    ]
    assert details["assertions"][1]["source_fragment_id"] not in {
        r["source_fragment_id"] for r in feedback["preserved_fragments"]
    }


@pytest.mark.parametrize(
    "fault",
    [
        "digest",
        "hash_missing",
        "rows_missing",
        "duplicate",
        "extra",
        "bool_range",
        "wrong_assertion_shape",
    ],
)
def test_forged_baseline_is_not_an_optional_legacy_feedback(fault):
    survey, details, _, _ = sample()
    feedback = feedback_for(details, survey)
    rows = feedback["preserved_fragments"]
    if fault == "digest":
        feedback["preserved_fragments_sha256"] = "f" * 64
    elif fault == "hash_missing":
        feedback.pop("preserved_fragments_sha256")
    elif fault == "rows_missing":
        feedback.pop("preserved_fragments")
    elif fault == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif fault == "extra":
        rows[0]["status"] = "supported"
    elif fault == "bool_range":
        rows[0]["source_start"] = True
    else:
        rows[0]["assertions"][0]["status"] = "supported"
    if "preserved_fragments" in feedback and fault not in {"digest", "hash_missing"}:
        feedback["preserved_fragments_sha256"] = digest(rows)
    with pytest.raises(ValueError, match="stability_identity_invalid"):
        recovery.preserved_fragment_directory(feedback)


def projected():
    survey, details, units, aliases = sample()
    payload = dict(
        source_units=units,
        allowed_citation_aliases=list(aliases.values()),
        inventory_recovery=feedback_for(details, survey),
        inventory_output_contract="host-flat-fragment-inventory-v1",
    )
    native = {
        "type": "json_schema",
        "json_schema": {
            "name": "host_flat_fragment_inventory_v1",
            "strict": True,
            "schema": flat_inventory_schema(units),
        },
    }
    return payload, native, aliases


def test_native_projection_transmits_all_copyable_baseline_text_once_with_real_fN():
    payload, native, aliases = projected()
    before = copy.deepcopy((payload, native))
    wire, schema = inventory_identity_projection(
        payload, native, evidence_aliases=aliases
    )
    canonical = payload["inventory_recovery"]["preserved_fragments"]
    rows = wire["inventory_recovery"]["preserved_fragments"]
    assert len(rows) == 3 and [r["source_fragment_id"] for r in rows] == [
        "f2",
        "f3",
        "f4",
    ]
    assert all(
        set(r) == {"source_fragment_id", "assertions", "exclusion_reason"} for r in rows
    )
    assert [r["assertions"] for r in rows] == [r["assertions"] for r in canonical]
    assert (
        wire["inventory_recovery"]["preserved_fragments_sha256"]
        == payload["inventory_recovery"]["preserved_fragments_sha256"]
    )
    assert set(
        schema["json_schema"]["schema"]["properties"]["fragments"]["properties"]
    ) == {"f1", "f2", "f3", "f4"}
    transported = inventory_transport_projection(
        wire, transport_revision=TRANSPORT_REVISION
    )
    assert transported["inventory_recovery"]["preserved_fragments"] == rows
    assert (payload, native) == before


@pytest.mark.parametrize("fault", ["uid", "fid", "citation", "missing", "order"])
def test_projection_rejects_rehashed_foreign_or_incomplete_host_baseline(fault):
    payload, native, aliases = projected()
    feedback = payload["inventory_recovery"]
    rows = feedback["preserved_fragments"]
    if fault == "uid":
        rows[0]["source_unit_id"] = rows[1]["source_unit_id"]
    elif fault == "fid":
        rows[0]["source_fragment_id"] = "foreign"
    elif fault == "citation":
        rows[0]["evidence_ids"] = ["a"]
    elif fault == "missing":
        rows.pop()
    else:
        rows.reverse()
    feedback["preserved_fragments_sha256"] = digest(rows)
    with pytest.raises(ValueError, match="alias_identity_invalid"):
        inventory_identity_projection(payload, native, evidence_aliases=aliases)


def test_old_none_instruction_identity_and_feedback_stay_unchanged(
    portable043, monkeypatch
):
    survey, details, _, _ = sample()
    # The old manual fixture intentionally has no empty-fragment record.
    details["stages"]["fragment_inventory"].pop()
    old = recovery.inventory_recovery_feedback(
        survey, details, contract=recovery.MIXED_CONTRACT
    )
    assert "preserved_fragments" not in old
    assert (
        hashlib.sha256(
            recovery.recovery_instruction(recovery.MIXED_CONTRACT).encode()
        ).hexdigest()
        == portable043["legacy_mixed_instruction_sha256"]
    )
    legacy = recovery.mixed_inventory_recovery_identity()
    selected = recovery.mixed_inventory_recovery_identity(stable=True)
    monkeypatch.setattr(recovery, "INVENTORY_STABILITY_RULE", "changed selected rule")
    assert recovery.mixed_inventory_recovery_identity() == legacy
    assert recovery.mixed_inventory_recovery_identity(stable=True) != selected


def test_same_and_cycle_no_progress_remain_even_with_valid_stability_directory():
    survey, details, _, _ = sample()
    feedback = feedback_for(details, survey)
    inventory = details["stages"]["fragment_inventory"]
    with pytest.raises(ValueError, match="no_progress"):
        recovery.validate_recovered_inventory(
            details["assertions"], feedback, fragment_inventory=inventory
        )
    changed, inventory = proposed(details)
    feedback["seen_inventory_sha256"] = [recovery.inventory_identity(changed)]
    with pytest.raises(ValueError, match="no_progress"):
        recovery.validate_recovered_inventory(
            changed, feedback, fragment_inventory=inventory
        )


def test_stability_failure_locates_content_without_exposing_it():
    survey, details, _, _ = sample()
    feedback = feedback_for(details, survey)
    claims, inventory = proposed(details)
    claims[1]["claim_text"] = "A softened result."
    with pytest.raises(
        ScopedEvaluationError, match="preserved_fragment_changed"
    ) as caught:
        recovery.validate_recovered_inventory(
            claims, feedback, fragment_inventory=inventory
        )
    check = caught.value.details["inventory_stability_check"]
    assert isinstance(caught.value, ValueError)
    assert check["source_report_sha256"] == feedback["source_report_sha256"]
    assert check["source_fragment_id"] == claims[1]["source_fragment_id"]
    assert check["source_unit_id"] == claims[1]["source_unit_id"]
    assert check["expected_count"] == check["actual_count"] == 1
    assert check["expected_content_sha256"] != check["actual_content_sha256"]
    assert "A softened result." not in json.dumps(caught.value.details)
    assert "An unsupported result" not in json.dumps(caught.value.details)


def test_invalid_directory_keeps_typed_location_through_native_projection():
    payload, native, aliases = projected()
    feedback = payload["inventory_recovery"]
    row = feedback["preserved_fragments"][0]
    row["source_start"] = True
    feedback["preserved_fragments_sha256"] = digest(feedback["preserved_fragments"])
    with pytest.raises(
        ScopedEvaluationError, match="stability_identity_invalid"
    ) as caught:
        inventory_identity_projection(payload, native, evidence_aliases=aliases)
    check = caught.value.details["inventory_stability_check"]
    assert check["check"] == "directory_identity"
    assert check["source_fragment_id"] == row["source_fragment_id"]
    assert check["source_report_sha256"] == feedback["source_report_sha256"]
    assert check["expected_count"] == 1 and check["actual_count"] is None


def test_real043_split_results_cannot_recombine_when_unflagged(portable043):
    pair = portable043["pairs"][0]
    baseline = [r["assertion"] for r in pair["rounds"][1]]
    candidate = [r["assertion"] for r in pair["rounds"][2]]
    first = baseline[0]
    row = {
        key: first[key] for key in recovery.PRESERVED_FRAGMENT_FIELDS if key in first
    }
    row.update(
        assertions=[{k: c[k] for k in ("kind", "claim_text")} for c in baseline],
        exclusion_reason="",
    )
    feedback = dict(
        contract=recovery.MIXED_CONTRACT,
        previous_inventory_sha256=recovery.inventory_identity(baseline),
        findings=[],
        preserved_fragments=[row],
        preserved_fragments_sha256=digest([row]),
    )
    inventory = [
        dict(
            source_unit_id=first["source_unit_id"],
            source_fragment_id=first["source_fragment_id"],
            assertion_count=len(candidate),
            exclusion_reason="",
        )
    ]
    with pytest.raises(ValueError, match="preserved_fragment_changed"):
        recovery.validate_recovered_inventory(
            candidate, feedback, fragment_inventory=inventory
        )
    assert pair["rounds"][2][0]["decision"] is None


def test_real043_causal_chain_is_preserved_as_content_not_an_atomicity_gold(
    portable043,
):
    pair = portable043["pairs"][2]
    assertions = [r[0]["assertion"] for r in pair["rounds"]]
    assert len({c["claim_text"] for c in assertions}) == 1
    assert pair["rounds"][2][0]["decision"]["atomic"] is False
    # The stability guard neither splits conjunctions nor changes model verdicts.
    assert "by classifying" in pair["actual_citation"]["original_abstract"]
    assert portable043["provenance"]["not_human_gold"] is True


def test_real043_split_conditions_remain_exact_after_that_fragment_is_resolved(
    portable043,
):
    pair = portable043["pairs"][1]
    before = [r["assertion"] for r in pair["rounds"][1]]
    after = [r["assertion"] for r in pair["rounds"][2]]
    first = before[0]
    protected = {
        key: first[key] for key in recovery.PRESERVED_FRAGMENT_FIELDS if key in first
    }
    protected.update(
        assertions=[{key: c[key] for key in ("kind", "claim_text")} for c in before],
        exclusion_reason="",
    )
    # Another diagnosed fragment changes; the resolved proposal/absence split
    # stays exact and does not inherit any prior support verdict.
    editable = copy.deepcopy(portable043["pairs"][0]["rounds"][0][0]["assertion"])
    previous = recovery.inventory_identity([*before, editable])
    editable["claim_text"] = "A newly decomposed authorized assertion."
    feedback = dict(
        contract=recovery.MIXED_CONTRACT,
        previous_inventory_sha256=previous,
        findings=[{"source_fragment_ids": [editable["source_fragment_id"]]}],
        preserved_fragments=[protected],
        preserved_fragments_sha256=digest([protected]),
    )
    inventory = [
        dict(
            source_unit_id=first["source_unit_id"],
            source_fragment_id=first["source_fragment_id"],
            assertion_count=2,
            exclusion_reason="",
        )
    ]
    recovery.validate_recovered_inventory(
        [*after, editable], feedback, fragment_inventory=inventory
    )
    assert "相同 5-way shot 协议" in after[0]["claim_text"]
    assert "当前材料没有提供这些统一条件" in after[1]["claim_text"]
    after[0]["claim_text"] = after[0]["claim_text"].replace("5-way", "1-way")
    with pytest.raises(ValueError, match="preserved_fragment_changed"):
        recovery.validate_recovered_inventory(
            [*after, editable], feedback, fragment_inventory=inventory
        )
