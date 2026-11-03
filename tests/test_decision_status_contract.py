"""Keep status-pair legality separate from unvalidated scientific judgments."""

import copy
import hashlib
import itertools
import json
from pathlib import Path

import pytest
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_reference_set_identity import options_for
from test_support_receipt_identity import FLAGS, REVISION, selected_options
from test_writer_v2 import case  # noqa: F401

import litagent.eval.scoped as scoped
from litagent.eval.batched import (
    _validate_decision_consistency,
    assessment_schema,
    batch_prompt,
)
from litagent.frozen_writing import validate_frozen_writer_plan
from litagent.safety.planned_budget import PlannedBudget, validate_writer_envelope
from litagent.writing_v2 import WriterError, writer_budget_plan

FIXTURE_PATH = Path(__file__).parent / "fixtures/run041-status-decisions.json"
FIXTURE_SHA256 = "eb45b1d60e2f4ea8176a84077516ba80772052508c356b9785e839f0fadf31ed"
STATUSES = ("supported", "unsupported", "unverifiable")
REASONS = (
    "supported",
    "insufficient_support",
    "uncited_claim",
    "attribution_mismatch",
    "record_mismatch",
    "unverifiable",
)
LEGAL = {
    ("supported", "supported"),
    ("unverifiable", "unverifiable"),
    *(("unsupported", reason) for reason in REASONS[1:5]),
}
KINDS = ("scientific", "runtime_method", "report_structure")


@pytest.fixture(scope="module")
def raw041():
    """Expose only original semantic decision fields, with no scientific labels."""
    raw = FIXTURE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    return json.loads(raw)


def decision(status="unsupported", reason="insufficient_support", **changes):
    """Build protocol controls without claiming that any paper supports them."""
    return scoped.PassageDecision.model_validate(
        {
            "claim_id": "claim_1",
            "correct_kind": "scientific",
            "atomic": True,
            "classification_valid": True,
            "source_faithful": True,
            "status": status,
            "reason_code": reason,
            "explanation": "Controlled protocol response, not a scientific gold label.",
            "support_basis": [],
            "aspect_checks": [],
            **changes,
        }
    )


def host_inputs(value):
    """Use the same host consumer for complete and partial negative observations."""
    text = "A controlled assertion awaits scientific verification."
    claim = {
        "claim_id": value.claim_id,
        "kind": value.correct_kind,
        "claim_text": text,
        "source_unit_id": "u1",
        "source_fragment_id": "f1",
        "source_quote": text,
        "source_start": 0,
        "source_end": len(text),
        "evidence_ids": [],
        "local_error": "",
    }
    inputs = {
        "claims": [claim],
        "ledger": {},
        "aliases": {},
        "support_passages": {},
        "support_validation_revision": REVISION,
        "support_validation_identity": scoped.support_validation_identity(),
    }
    context = {
        "quality_policy": "strict_auto_v2",
        "_support_validation_revision": REVISION,
        "_host_validation_inputs": inputs,
    }
    progress = {
        "support_validation_revision": REVISION,
        "support_validation_identity": scoped.support_validation_identity(),
        "batches": [
            {
                "status": "completed",
                "claim_ids": [value.claim_id],
                "decisions": [value.model_dump()],
            }
        ],
    }
    return claim, context, progress


def complete_host(value, claim, context):
    """Call the production final validator, without bypassing its status guard."""
    return scoped.validate_assessment_decisions(
        [claim],
        [value],
        ledger={},
        aliases={},
        support_passages={},
        context=context,
        strict=True,
    )


def test_original_fixture_preserves_eight_provider_errors_without_gold_labels(raw041):
    assert raw041["science_gold_standard"] is False
    assert raw041["raw_provider_judgments_unvalidated"] is True
    assert set(raw041) == {
        "contract",
        "science_gold_standard",
        "raw_provider_judgments_unvalidated",
        "provenance",
        "decisions",
    }
    assert [d["claim_id"] for d in raw041["decisions"]] == [
        "claim_7",
        "claim_33",
        "claim_36",
        "claim_39",
        "claim_41",
        "claim_44",
        "claim_45",
        "claim_47",
    ]
    expected = set(scoped.PassageDecision.model_fields)
    for raw in raw041["decisions"]:
        assert set(raw) == expected
        assert raw["status"] == "unverifiable"
        assert raw["reason_code"] == "insufficient_support"
        assert raw["correct_kind"] == "scientific"
        assert raw["atomic"] and raw["classification_valid"] and raw["source_faithful"]
        assert raw["support_basis"] == []
        assert all(
            not a["supported"] and a["support_ids"] == [] for a in raw["aspect_checks"]
        )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("status,reason", tuple(itertools.product(STATUSES, REASONS)))
def test_all_eighteen_pairs_keep_existing_legality_and_typed_error(
    kind, status, reason
):
    value = decision(status, reason, correct_kind=kind)
    before = value.model_dump()
    if (status, reason) in LEGAL:
        assert scoped.validate_decision_status(value) is None
    else:
        with pytest.raises(ValueError, match="^evaluation_decision_status_invalid$"):
            scoped.validate_decision_status(value)
    assert value.model_dump() == before


@pytest.mark.parametrize("flag", ("atomic", "classification_valid", "source_faithful"))
def test_invalid_pair_cannot_be_hidden_by_an_invalid_scientific_flag(flag):
    value = decision("unverifiable", "insufficient_support", **{flag: False})
    claim, context, progress = host_inputs(value)
    for consume in (
        lambda: complete_host(value, claim, context),
        lambda: scoped.partial_assessment(progress, context),
    ):
        with pytest.raises(ValueError, match="^evaluation_decision_status_invalid$"):
            consume()


@pytest.mark.parametrize("index", range(8))
def test_raw_provider_pair_is_rejected_identically_by_partial_and_final(raw041, index):
    original = copy.deepcopy(raw041["decisions"][index])
    value = scoped.PassageDecision.model_validate(original)
    claim, context, progress = host_inputs(value)
    for consume in (
        lambda: scoped.validate_decision_status(value),
        lambda: complete_host(value, claim, context),
        lambda: scoped.partial_assessment(progress, context),
    ):
        with pytest.raises(ValueError, match="^evaluation_decision_status_invalid$"):
            consume()
    assert raw041["decisions"][index] == original


@pytest.mark.parametrize("index", range(8))
@pytest.mark.parametrize(
    "pair", (("unsupported", "insufficient_support"), ("unverifiable", "unverifiable"))
)
def test_separate_legal_controls_remain_negative_in_partial_and_final(
    raw041, index, pair
):
    original = copy.deepcopy(raw041["decisions"][index])
    raw = {**copy.deepcopy(original), "status": pair[0], "reason_code": pair[1]}
    value = scoped.PassageDecision.model_validate(raw)
    claim, context, progress = host_inputs(value)
    complete = complete_host(value, claim, context)
    partial = scoped.partial_assessment(progress, context)
    assert complete[0]["status"] == pair[0]
    assert partial["assertions"] == complete
    assert partial["overall_valid"] is False
    assert partial["supported_count"] == 0
    assert partial["unsupported_count"] == (pair[0] == "unsupported")
    assert partial["unverified_count"] == (pair[0] == "unverifiable")
    assert raw041["decisions"][index] == original


@pytest.mark.parametrize("index", range(8))
def test_legal_supported_pair_cannot_promote_original_negative_aspects(raw041, index):
    original = copy.deepcopy(raw041["decisions"][index])
    value = scoped.PassageDecision.model_validate(
        {**copy.deepcopy(original), "status": "supported", "reason_code": "supported"}
    )
    assert scoped.validate_decision_status(value) is None
    claim, _, _ = host_inputs(value)
    claim["required_support_aspects"] = [a.aspect for a in value.aspect_checks]
    with pytest.raises(
        scoped.ScopedEvaluationError, match="^evaluation_decision_consistency_invalid$"
    ) as caught:
        _validate_decision_consistency(claim, value, {})
    assert caught.value.details["decision_inconsistencies"][0]["subreason"] == (
        "positive_status_with_negative_required_aspect"
    )
    assert raw041["decisions"][index] == original


def without_descriptions(value):
    """Compare validation constraints independently of explanatory annotations."""
    if isinstance(value, dict):
        return {
            k: without_descriptions(v) for k, v in value.items() if k != "description"
        }
    if isinstance(value, list):
        return [without_descriptions(v) for v in value]
    return value


def test_schema_and_instruction_share_the_existing_six_pairs(canonical):
    expected = {
        "supported": ("supported",),
        "unsupported": REASONS[1:5],
        "unverifiable": ("unverifiable",),
    }
    assert scoped.DECISION_STATUS_REASONS == expected
    original = assessment_schema(canonical)
    before = copy.deepcopy(original)
    described = scoped.decision_status_schema(original)
    instruction = scoped.decision_status_instruction()
    assert instruction == ";".join(
        status + "/" + "|".join(reasons) for status, reasons in expected.items()
    )
    assert instruction not in scoped.decision_status_prompt()
    assert "schema" in scoped.decision_status_prompt()
    assert original == before
    assert without_descriptions(described) == without_descriptions(original)
    fields = described["$defs"]["PassageDecision"]["properties"]
    for field in ("status", "reason_code"):
        assert fields[field]["description"]
    assert fields["reason_code"]["description"] == instruction
    for reason in REASONS:
        assert reason in instruction


@pytest.mark.parametrize(
    "policy", ("planned-budget-v5", "planned-budget-v6", "planned-budget-v7")
)
def test_actual_status_rule_changes_invalidate_only_selected_plan_and_wire(
    case, canonical, monkeypatch, policy
):
    old_options = options_for(policy)
    old_plan = writer_budget_plan(
        case.packet, case.aliases, budget_policy=policy, methodology={}
    )
    old_payload = {**copy.deepcopy(canonical), "writer_envelope": old_plan["envelope"]}
    old_wire = scoped.build_evaluation_request(batch_prompt(old_payload), old_payload)
    old_fp = old_options.fingerprint()
    kwargs = {
        "budget_policy": "planned-budget-v7",
        "methodology": {},
        **FLAGS,
        "support_validation_revision": REVISION,
    }
    selected = selected_options()
    selected_fp = selected.fingerprint()
    selected_plan = writer_budget_plan(case.packet, case.aliases, **kwargs)
    selected_snapshot = PlannedBudget(selected_plan).snapshot()
    old_identity = scoped.support_validation_identity()
    rules = copy.deepcopy(scoped.DECISION_STATUS_REASONS)
    rules["unsupported"] = tuple(
        r for r in rules["unsupported"] if r != "record_mismatch"
    )
    monkeypatch.setattr(scoped, "DECISION_STATUS_REASONS", rules)
    assert scoped.support_validation_identity() != old_identity
    assert selected.fingerprint() != selected_fp
    current = writer_budget_plan(case.packet, case.aliases, **kwargs)
    assert current["identity"] != selected_plan["identity"]
    with pytest.raises(ValueError, match="support_validation"):
        validate_writer_envelope(selected_plan["envelope"])
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(selected_plan, current)
    owner = PlannedBudget(current)
    before = owner.snapshot()
    with pytest.raises(ValueError, match="planned_budget_identity_conflict"):
        owner.restore(selected_snapshot)
    assert owner.snapshot() == before
    assert old_options.fingerprint() == old_fp
    assert (
        writer_budget_plan(
            case.packet, case.aliases, budget_policy=policy, methodology={}
        )
        == old_plan
    )
    assert (
        scoped.build_evaluation_request(batch_prompt(old_payload), old_payload)
        == old_wire
    )
