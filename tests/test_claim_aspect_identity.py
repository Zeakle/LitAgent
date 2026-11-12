"""Bind versioned claim aspects without migrating historical response authority."""

import copy
from types import SimpleNamespace

import pytest
from test_concise_repair_prompt import FLAGS
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_joint_round_escrow import (
    _allocate,
    _amounts,
    _round_wires,
    _send_max,
    _settle_cap,
    _settle_initial_max,
)
from test_reference_set_identity import envelope as parent_envelope
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.eval import batched
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.quality import digest
from litagent.safety.planned_accounting import _envelope as accounting_envelope
from litagent.safety.planned_accounting import _plan as accounting_plan
from litagent.safety.planned_accounting import (
    validate_accounting_snapshot,
)
from litagent.safety.planned_budget import (
    JOINT_ROUND_ESCROW_CONTRACT,
    PlannedBudget,
    request_scope,
    validate_writer_envelope,
)
from litagent.writing_v2 import writer_budget_plan
from scripts import run_frozen_writing as campaign

PAIR = {"evaluation_response_contract", "evaluation_response_identity"}


def response_scope(contract=None, **changes):
    """Derive each pair through the real envelope producer on complete parents."""
    return parent_envelope(
        **{
            **FLAGS,
            "global_findings_contract": "anchored-global-findings-v2",
            "summary_constraints": True,
            "evaluation_response_contract": contract,
            **changes,
        }
    )


def frozen_plan(bundle, contract=None, **changes):
    """Keep frozen inputs and operational flags identical across response choices."""
    return plan_frozen_budget(
        bundle,
        budget_policy="planned-budget-v7",
        **{**FLAGS, "evaluation_response_contract": contract, **changes},
    )


def rehash(value):
    """Rehash data without granting a tampered pair current host authority."""
    value["identity"] = digest(
        {key: child for key, child in value.items() if key != "identity"}
    )
    return value


def test_default_identity_remains_v1_and_v2_is_a_distinct_explicit_choice():
    assert batched.EVALUATION_RESPONSE_CONTRACT == "host-bound-evaluation-v1"
    assert batched.EVALUATION_RESPONSE_CONTRACT_V2 == "host-bound-evaluation-v2"
    assert len(batched.EVALUATION_RESPONSE_CONTRACTS) == 2
    assert set(batched.EVALUATION_RESPONSE_CONTRACTS) == {
        batched.EVALUATION_RESPONSE_CONTRACT,
        batched.EVALUATION_RESPONSE_CONTRACT_V2,
    }
    assert batched.response_contract_identity() == batched.response_contract_identity(
        contract=batched.EVALUATION_RESPONSE_CONTRACT
    )
    assert batched.response_contract_identity() != batched.response_contract_identity(
        contract=batched.EVALUATION_RESPONSE_CONTRACT_V2
    )


@pytest.mark.parametrize("bad", [None, False, True, 0, 1, "", "future", [], {}])
def test_identity_rejects_values_outside_the_two_explicit_versions(bad):
    with pytest.raises(ValueError):
        batched.response_contract_identity(contract=bad)


def test_none_remains_absent_and_factory_derives_both_versioned_pairs():
    old = response_scope()
    assert response_scope(None) == old
    assert not PAIR.intersection(old)
    assert batched.selected_response_contract(old) is False
    for contract in batched.EVALUATION_RESPONSE_CONTRACTS:
        selected = response_scope(contract)
        before = copy.deepcopy(selected)
        assert selected["evaluation_response_contract"] == contract
        assert selected["evaluation_response_identity"] == (
            batched.response_contract_identity(contract=contract)
        )
        assert batched.selected_response_contract(selected) is True
        assert validate_writer_envelope(selected) == before
        assert selected == before
        for key in (
            "request_caps",
            "capacity",
            "body_max_units",
            "summary_max_units_per_section",
            "model_assertions",
            "source_identity",
        ):
            assert selected[key] == old[key]


@pytest.mark.parametrize(
    "contract", ["host-bound-evaluation-v1", "host-bound-evaluation-v2"]
)
@pytest.mark.parametrize(
    "fault", ["identity", "version", "strip_contract", "strip_identity"]
)
def test_tampered_pair_is_rejected_even_after_outer_identity_recalculation(
    contract, fault
):
    selected = response_scope(contract)
    if fault == "identity":
        selected["evaluation_response_identity"] = "0" * 64
    elif fault == "version":
        selected["evaluation_response_contract"] = (
            batched.EVALUATION_RESPONSE_CONTRACT_V2
            if contract == batched.EVALUATION_RESPONSE_CONTRACT
            else batched.EVALUATION_RESPONSE_CONTRACT
        )
    else:
        selected.pop(
            "evaluation_response_contract"
            if fault == "strip_contract"
            else "evaluation_response_identity"
        )
    rehash(selected)
    before = copy.deepcopy(selected)
    with pytest.raises(ValueError, match="evaluation_response"):
        batched.selected_response_contract(selected)
    with pytest.raises(ValueError, match="evaluation_response"):
        validate_writer_envelope(selected)
    assert selected == before


@pytest.mark.parametrize(
    "changes",
    [
        {"round_budget": False},
        {"capacity": None},
        {"classification_review": False},
        {"inventory_recovery": False},
        {"mixed_inventory_recovery": False},
        {"summary_constraints": False},
        {"global_findings_contract": "anchored-global-findings-v1"},
        {"reference_contract": None},
        {"transport_revision": None},
        {"support_validation_revision": "host-support-receipts-v1"},
    ],
)
def test_v2_never_infers_missing_parent_authority(changes):
    with pytest.raises(ValueError):
        response_scope(batched.EVALUATION_RESPONSE_CONTRACT_V2, **changes)


@pytest.mark.parametrize(
    "contract", ["host-bound-evaluation-v1", "host-bound-evaluation-v2"]
)
def test_frozen_factory_and_pure_reader_bind_the_same_version(frozen_bundle, contract):
    before = copy.deepcopy(frozen_bundle)
    plan = frozen_plan(frozen_bundle, contract)
    explicit_none = frozen_plan(frozen_bundle, None)
    assert plan["status"] == "admitted"
    assert plan["identity"] != explicit_none["identity"]
    assert plan["envelope"]["evaluation_response_identity"] == (
        batched.response_contract_identity(contract=contract)
    )
    assert plan["maximum_tokens"] == 4500000
    assert plan["maximum_requests"] == 64
    assert plan["repair_limit"] == plan["scope_bounds"]["shared_repairs"] == 2
    assert plan["context_reservation_limit"] == 117964
    assert plan["envelope"]["request_caps"]["assessment"] == 100000
    assert plan["stages"]["repair"]["output_tokens"] == 20000
    accounting_plan(plan)
    assert accounting_envelope(plan["envelope"]).max_body_units == 66
    assert frozen_bundle == before


def test_historical_fee_reader_accepts_pinned_data_without_current_aspect_code(
    monkeypatch,
):
    pinned = response_scope(batched.EVALUATION_RESPONSE_CONTRACT_V2)
    pinned["evaluation_response_identity"] = "f" * 64
    rehash(pinned)
    with pytest.raises(ValueError, match="evaluation_response"):
        validate_writer_envelope(pinned)
    monkeypatch.setattr(
        batched,
        "response_contract_identity",
        lambda **kwargs: pytest.fail("historical reader executed current schema code"),
    )
    assert accounting_envelope(pinned).max_body_units == 66


@pytest.mark.parametrize(
    "contract", ["host-bound-evaluation-v1", "host-bound-evaluation-v2"]
)
def test_manifest_version_is_explicit_and_does_not_add_public_options(contract):
    from litagent.api import SurveyRequest
    from litagent.workbench import RunOptions

    scope = response_scope(contract)
    manifest = {
        "writing_profile": campaign.ROUND_PROFILE,
        **FLAGS,
        "evaluation_response_contract": contract,
        "evaluation_response_identity": scope["evaluation_response_identity"],
    }
    assert campaign.evaluation_response_options({}) == {}
    assert campaign.evaluation_response_options(manifest) == {
        "evaluation_response_contract": contract
    }
    assert not PAIR.intersection(SurveyRequest.model_fields)
    assert not PAIR.intersection(RunOptions.model_fields)
    for field in PAIR:
        damaged = copy.deepcopy(manifest)
        damaged.pop(field)
        with pytest.raises(ValueError, match="frozen_evaluation_response"):
            campaign.evaluation_response_options(damaged)
    manifest["evaluation_response_identity"] = "0" * 64
    with pytest.raises(ValueError, match="frozen_evaluation_response"):
        campaign.evaluation_response_options(manifest)


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["missing_plan", "strip_selector", "wrong_identity"])
async def test_frozen_runtime_rejects_unbound_v2_before_worker_setup(
    frozen_bundle, fault
):
    contract = batched.EVALUATION_RESPONSE_CONTRACT_V2
    plan = frozen_plan(frozen_bundle, contract)
    options = {**FLAGS, "evaluation_response_contract": contract}
    if fault == "missing_plan":
        plan = None
    elif fault == "strip_selector":
        options.pop("evaluation_response_contract")
    else:
        plan["envelope"]["evaluation_response_identity"] = "0" * 64
        rehash(plan["envelope"])
        rehash(plan)
    worker = SimpleNamespace(
        _wired=False, _config=SimpleNamespace(capacity=CapacityConfig())
    )
    with pytest.raises(ValueError, match="evaluation_response"):
        await run_frozen(
            worker,
            frozen_bundle,
            "unstarted-aspect-contract",
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            **options,
        )
    assert vars(worker) == {"_wired": False, "_config": worker._config}


def test_v2_joint_maximum_usage_keeps_two_repairs_and_all_three_final_rounds(case):
    """Fund accounting boundary wires, separately from native schema size checks."""
    base = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        **{
            **FLAGS,
            "evaluation_response_contract": batched.EVALUATION_RESPONSE_CONTRACT_V2,
        },
    )
    from litagent.safety.planned_budget import with_joint_round_escrow

    plan = with_joint_round_escrow(base)
    assert plan["joint_round_escrow"]["contract"] == JOINT_ROUND_ESCROW_CONTRACT
    assert (
        plan["envelope"]["evaluation_response_contract"]
        == batched.EVALUATION_RESPONSE_CONTRACT_V2
    )
    assert plan["envelope"]["evaluation_response_identity"] == (
        batched.response_contract_identity(
            contract=batched.EVALUATION_RESPONSE_CONTRACT_V2
        )
    )
    owner = PlannedBudget(plan)
    domain = plan["joint_round_escrow"]["assessment"]
    amounts = _amounts(domain["reserved_tokens"], domain["max_requests"])
    assert max(amounts) <= domain["request_cap"] == 100000
    _settle_initial_max(owner)
    for index in range(3):
        values = _round_wires(owner, amounts, f"v2-max:{index}")
        allocation = _allocate(owner, values, report_sha=str(index + 1) * 64)
        assert allocation["round"]["remaining_repairs"] == 2 - index
        for slot, value in enumerate(values):
            _send_max(owner, value, f"v2-slot:{slot}")
        validate_accounting_snapshot(owner.snapshot(), expected_plan=plan)
        if index < 2:
            for stage in ("repair", "summary", "inventory"):
                _settle_cap(owner, stage, f"v2-candidate:{index}:{stage}")
    final = owner.snapshot()
    assert final["used_tokens"] == plan["required_tokens"] <= base["planned_tokens"]
    assert final["requests_sent"] == plan["planned_requests"] <= 64
    assert len(owner._stage_entries("repair")) == 2
    assert len(owner._stage_entries("global")) == 3
    assert all(
        entry["actual_tokens"] == entry["reserved_tokens"]
        for entry in owner.entries.values()
    )
    with request_scope("repair", "v2-third-repair"):
        with pytest.raises(ValueError, match="exceeded"):
            owner.reserve(plan["stages"]["repair"]["request_cap"])
