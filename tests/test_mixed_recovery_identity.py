"""Bind the explicit mixed-inventory contract without changing historical identities."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.config import AppConfig
from litagent.eval import inventory_recovery
from litagent.frozen_writing import (
    plan_frozen_budget,
    run_frozen,
    validate_frozen_writer_plan,
)
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import digest
from litagent.runner import LitAgent
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import (
    PlannedBudget,
    _digest,
    make_writer_envelope,
    request_scope,
    validate_writer_envelope,
    wire_reservation,
)
from litagent.workbench import RunOptions
from litagent.writing_v2 import RANGE_WRITING_PLAN, WriterError, writer_budget_plan
from scripts import run_frozen_writing as campaign


def options_for(policy="planned-budget-v7", **changes):
    """Match the legacy frozen choices while keeping recovery server-owned."""
    return RunOptions(
        **{
            "report_language": "zh-CN",
            "human_review": False,
            "review_structure": "academic_v2",
            "quality_policy": "strict_auto_v2",
            "verification": {"judge": False, "jev": False},
            "body_quality": "analysis_v1",
            "budget_policy": policy,
            "capacity_policy": (
                "legacy" if policy == "planned-budget-v5" else "adaptive-capacity-v1"
            ),
            "classification_review": policy == "planned-budget-v7",
            "generation_strategy": "body_summary_v3",
            **changes,
        }
    )


@pytest.fixture
def frozen_bundle(case):
    """Keep the real frozen planner's full material producer input."""
    return {
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "methodology": {},
        "extractions": [],
    }


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        (
            "planned-budget-v5",
            "540408b7ccc41272a9f935a3594a1b5a8c5e5a56452afe299b0d8ad186124103",
        ),
        (
            "planned-budget-v6",
            "b714b44c0e50f5c875a79e1267254bf84908f7c1a45f5810eceff1c6a96a540d",
        ),
        (
            "planned-budget-v7",
            "159b5e3d00e58fad526f5e8d63e66baf876626884fe01345a9d72598f429bc20",
        ),
    ],
)
def test_inactive_mixed_flag_preserves_historical_run_fingerprints(policy, expected):
    before = options_for(policy)
    assert before.mixed_inventory_recovery is False
    assert "mixed_inventory_recovery" not in before.model_dump(mode="json")
    assert before.fingerprint() == expected
    assert options_for(policy, mixed_inventory_recovery=False).fingerprint() == expected


def test_035_feedback_enabled_run_fingerprint_is_not_implicitly_upgraded():
    original = options_for(repair_feedback=True)
    assert original.fingerprint() == (
        "5a2b4d9cfea0c54e6e27ab1cb5bafa882b285b3aed79e21ec7d0dd49143b58d3"
    )
    assert (
        options_for(repair_feedback=True, mixed_inventory_recovery=False).fingerprint()
        == original.fingerprint()
    )


@pytest.mark.parametrize("wrong", [None, 0, 1, "true", [], {}])
def test_internal_mixed_option_is_strictly_boolean(wrong):
    with pytest.raises(ValidationError):
        options_for(mixed_inventory_recovery=wrong)


@pytest.mark.parametrize("policy", ["planned-budget-v5", "planned-budget-v6"])
def test_internal_mixed_option_requires_v7(policy):
    with pytest.raises(ValueError, match="mixed_inventory_recovery_requires"):
        options_for(policy, mixed_inventory_recovery=True)


def test_mixed_switch_is_not_a_public_survey_request_field():
    from litagent.api import SurveyRequest

    assert "mixed_inventory_recovery" not in SurveyRequest.model_fields
    assert (
        "mixed_inventory_recovery"
        not in SurveyRequest.model_json_schema()["properties"]
    )
    with pytest.raises(ValidationError):
        SurveyRequest(query="Compare the methods.", mixed_inventory_recovery=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_runner_uses_a_single_explicit_selection_for_both_task_flags(enabled):
    graph = TaskGraph()
    graph.add_task(
        SubTask(
            task_id="adversarial_review",
            description="Write the frozen report",
            agent_type="adversarial_review",
            input_data={},
        )
    )
    LitAgent._apply_run_options_to_graph(
        graph, options_for(mixed_inventory_recovery=enabled)
    )
    data = graph.tasks["adversarial_review"].input_data
    assert ("mixed_inventory_recovery" in data) is enabled
    assert ("inventory_recovery" in data) is enabled
    if enabled:
        assert data["mixed_inventory_recovery"] is data["inventory_recovery"] is True


def test_new_identity_binds_recovery_global_and_layout_without_mutating_v1(
    case, monkeypatch
):
    common = {
        "methodology": {},
        "budget_policy": "planned-budget-v7",
        "inventory_recovery": True,
    }
    old = writer_budget_plan(case.packet, case.aliases, **common)
    assert (
        writer_budget_plan(
            case.packet, case.aliases, **common, mixed_inventory_recovery=False
        )
        == old
    )
    current = writer_budget_plan(
        case.packet, case.aliases, **common, mixed_inventory_recovery=True
    )
    assert old["envelope"]["inventory_recovery_contract"] == inventory_recovery.CONTRACT
    assert current["envelope"]["inventory_recovery_contract"] == (
        inventory_recovery.MIXED_CONTRACT
    )
    assert current["identity"] != old["identity"]
    assert current["repair_limit"] == 2
    assert current["planned_requests"] <= 64
    assert current["maximum_tokens"] == 4500000
    old_run = options_for().fingerprint()
    new_run = options_for(mixed_inventory_recovery=True).fingerprint()
    assert new_run != old_run
    monkeypatch.setattr(
        inventory_recovery, "mixed_inventory_recovery_identity", lambda: "f" * 64
    )
    assert options_for().fingerprint() == old_run
    assert options_for(mixed_inventory_recovery=True).fingerprint() != new_run
    assert writer_budget_plan(case.packet, case.aliases, **common) == old
    assert (
        writer_budget_plan(
            case.packet, case.aliases, **common, mixed_inventory_recovery=True
        )["identity"]
        != current["identity"]
    )


def envelope(**changes):
    """Use the actual envelope producer with all required v7 ancestors."""
    return make_writer_envelope(
        **{
            "body_units": 66,
            "assertions": 128,
            "body_bytes": 32000,
            "source_identity": "a" * 64,
            "builder_identity": "b" * 64,
            "contract": "flat-writer-envelope-v1",
            "capacity": CapacityConfig(),
            "round_budget": True,
            "global_findings_contract": "anchored-global-findings-v2",
            "inventory_recovery": True,
            **changes,
        }
    )


def test_envelope_uses_the_same_field_and_rejects_unknown_contracts():
    old = envelope()
    assert envelope(mixed_inventory_recovery=False) == old
    new = envelope(mixed_inventory_recovery=True)
    assert set(new) == set(old)
    assert new["inventory_recovery_contract"] == inventory_recovery.MIXED_CONTRACT
    assert validate_writer_envelope(new) == new
    for value in ["bounded-inventory-recovery-v3", None, [], {}, True]:
        broken = copy.deepcopy(new)
        broken["inventory_recovery_contract"] = value
        broken["identity"] = _digest(
            {k: v for k, v in broken.items() if k != "identity"}
        )
        with pytest.raises(ValueError, match="writer_envelope_recovery_invalid"):
            validate_writer_envelope(broken)


@pytest.mark.parametrize(
    "changes",
    [
        {"inventory_recovery": False},
        {"inventory_recovery": 1},
        {"round_budget": False},
        {"capacity": None},
        {"mixed_inventory_recovery": 1},
        {"mixed_inventory_recovery": None},
    ],
)
def test_mixed_envelope_requires_strict_flag_and_recovery_ancestors(changes):
    with pytest.raises(ValueError, match="mixed_inventory_recovery_policy_invalid"):
        envelope(**{"mixed_inventory_recovery": True, **changes})


@pytest.mark.parametrize(
    "changes",
    [
        {"mixed_inventory_recovery": None},
        {"mixed_inventory_recovery": 1},
        {"mixed_inventory_recovery": "true"},
        {"inventory_recovery": False},
        {"inventory_recovery": 1},
        {"budget_policy": "planned-budget-v6"},
    ],
)
def test_frozen_mixed_input_rejected_before_material_access(changes):
    with pytest.raises(ValueError, match="mixed_inventory_recovery_policy_invalid"):
        plan_frozen_budget(
            {},
            **{
                "budget_policy": "planned-budget-v7",
                "inventory_recovery": True,
                "mixed_inventory_recovery": True,
                **changes,
            },
        )


@pytest.mark.asyncio
async def test_frozen_runtime_rejects_mixed_without_recovery_before_agent_wiring():
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="mixed_inventory_recovery_policy_invalid"):
        await run_frozen(
            agent,
            {},
            "offline-rejected",
            budget_policy="planned-budget-v7",
            mixed_inventory_recovery=True,
        )


def test_real_frozen_material_producer_selects_new_contract_without_mutation(
    frozen_bundle,
):
    before = copy.deepcopy(frozen_bundle)
    common = {
        "budget_policy": "planned-budget-v7",
        "inventory_recovery": True,
        "classification_review": True,
        "repair_feedback": True,
    }
    old = plan_frozen_budget(frozen_bundle, **common)
    assert (
        plan_frozen_budget(frozen_bundle, **common, mixed_inventory_recovery=False)
        == old
    )
    new = plan_frozen_budget(frozen_bundle, **common, mixed_inventory_recovery=True)
    assert (
        new["envelope"]["inventory_recovery_contract"]
        == inventory_recovery.MIXED_CONTRACT
    )
    assert frozen_bundle == before and new["identity"] != old["identity"]
    for expected, actual in [(old, new), (new, old)]:
        with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
            validate_frozen_writer_plan(expected, actual)
        owner = PlannedBudget(expected)
        snapshot = owner.snapshot()
        with pytest.raises(ValueError, match="identity_conflict"):
            owner.restore(PlannedBudget(actual).snapshot())
        assert owner.snapshot() == snapshot


def test_manifest_selection_requires_explicit_true_profile_and_recovery():
    old = {"writing_profile": campaign.ROUND_PROFILE, "inventory_recovery": True}
    assert campaign.mixed_inventory_recovery_options(old) == {}
    assert "mixed_inventory_recovery" not in campaign.manifest_options(old)
    selected = {**old, "mixed_inventory_recovery": True}
    assert campaign.mixed_inventory_recovery_options(selected) == {
        "mixed_inventory_recovery": True
    }
    assert campaign.manifest_options(selected)["mixed_inventory_recovery"] is True
    for value in [False, None, 1, "true", [], {}]:
        with pytest.raises(
            ValueError, match="mixed_inventory_recovery_configuration_invalid"
        ):
            campaign.mixed_inventory_recovery_options(
                {**old, "mixed_inventory_recovery": value}
            )
    for change in [
        {"inventory_recovery": False},
        {"writing_profile": campaign.FLAT_PROFILE},
    ]:
        with pytest.raises(
            ValueError, match="mixed_inventory_recovery_configuration_invalid"
        ):
            campaign.mixed_inventory_recovery_options({**selected, **change})


def test_cli_mixed_override_is_rejected_before_live_configuration(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["frozen", "--root", str(tmp_path), "--live", "--mixed-inventory-recovery"],
    )
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("Reached live configuration")
    )
    with pytest.raises(SystemExit) as caught:
        campaign.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_only_forwards_an_explicit_prepare_selection(
    monkeypatch, tmp_path, enabled
):
    calls = []
    monkeypatch.setattr(campaign, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(campaign, "profile_config", lambda config, profile: config)
    monkeypatch.setattr(
        campaign, "prepare", lambda *a, **kw: calls.append(kw) or {"status": "fixture"}
    )
    argv = ["frozen", "--root", str(tmp_path), "--prepare"]
    if enabled:
        argv.extend(
            [
                "--writing-profile",
                campaign.ROUND_PROFILE,
                "--inventory-recovery",
                "--mixed-inventory-recovery",
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    assert ("mixed_inventory_recovery" in calls[0]) is enabled
    if enabled:
        assert calls[0]["mixed_inventory_recovery"] is True
        assert calls[0]["inventory_recovery"] is True


def test_mixed_checkpoint_keeps_unknown_usage_and_no_replay(case):
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
    )
    owner = PlannedBudget(plan)
    amount = wire_reservation(
        [{"role": "user", "content": "Frozen inventory request."}], 16384
    )
    with request_scope("inventory", "unknown-mixed-inventory"):
        owner.reserve(amount)
        owner.bind_wire({"frozen_wire": "inventory-v2"})
        owner.begin_request()
        owner.charge_unknown_usage(amount)
        owner.release(amount)
    snapshot = owner.snapshot()
    entry = snapshot["entries"]["unknown-mixed-inventory"]
    assert entry["status"] == "interrupted_no_replay"
    assert entry["actual_tokens"] is None and entry["reserved_tokens"] == amount
    restored = CostBudget.from_snapshot(snapshot)
    assert restored.snapshot() == snapshot
    with request_scope("inventory", "unknown-mixed-inventory"):
        with pytest.raises(ValueError, match="no_replay"):
            restored.begin_request()


@pytest.mark.parametrize("enabled", [False, True])
def test_prepare_and_validate_rebuild_the_real_three_case_plans(
    frozen_bundle, tmp_path, monkeypatch, enabled
):
    """Mock archive I/O only; keep manifest, schema, planning and validation real."""
    monkeypatch.chdir(tmp_path)
    source_root = tmp_path / "artifacts/runs"
    source_root.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        (source_root / (run_id + ".json")).write_text(json.dumps({"source": run_id}))
    history_path = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )

    def read_material(path):
        return {
            **copy.deepcopy(frozen_bundle),
            "source_run_id": path.stem,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    monkeypatch.setattr(campaign, "read_bundle", read_material)
    monkeypatch.setattr(campaign, "source_paths", lambda: [])
    monkeypatch.setattr(campaign, "check_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    config = AppConfig(agent={}, logging={})
    config.adversarial.thinking_mode = "disabled"
    root = tmp_path / "identity-fixture"
    prepared = campaign.prepare(
        root,
        config,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        inventory_recovery=True,
        classification_review=True,
        **({"mixed_inventory_recovery": True} if enabled else {}),
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert campaign.validate(root, config) == manifest
    assert prepared["identity"] == manifest["identity"]
    assert ("mixed_inventory_recovery" in manifest) is enabled
    assert ("mixed_inventory_recovery" in manifest["options"]) is enabled
    assert ("mixed_inventory_recovery" in manifest["writing_parameters"]) is enabled
    expected_contract = (
        inventory_recovery.MIXED_CONTRACT if enabled else inventory_recovery.CONTRACT
    )
    for row in manifest["cases"]:
        plan = manifest["budget_plans"][row["case_id"]]
        assert plan["envelope"]["inventory_recovery_contract"] == expected_contract
        assert plan == plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=config.capacity,
            inventory_recovery=True,
            classification_review=True,
            **({"mixed_inventory_recovery": True} if enabled else {}),
        )
    assert manifest["limits"]["repairs"] == 2
    assert manifest["limits"]["retries"] == 0
    assert not (root / "state.json").exists()
