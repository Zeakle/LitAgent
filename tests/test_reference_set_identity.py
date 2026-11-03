"""Keep ordered reference handling explicit across frozen writer identities."""

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
    validate_writer_envelope,
)
from litagent.workbench import RunOptions
from litagent.writing_flat import REFERENCE_SET_CONTRACT
from litagent.writing_v2 import RANGE_WRITING_PLAN, WriterError, writer_budget_plan
from scripts import run_frozen_writing as campaign


def options_for(policy="planned-budget-v7", **changes):
    """Match historical frozen options before adding the one internal selector."""
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
def test_inactive_reference_contract_keeps_historical_fingerprints(policy, expected):
    old = options_for(policy)
    assert old.reference_contract is None
    assert "reference_contract" not in old.model_dump(mode="json")
    assert old.fingerprint() == expected
    assert options_for(policy, reference_contract=None).fingerprint() == expected


@pytest.mark.parametrize(
    "wrong", [False, True, 0, 1, "", "ordered-reference-set-v2", [], {}]
)
def test_reference_selector_rejects_unknown_or_coerced_values(wrong):
    with pytest.raises(ValidationError):
        options_for(reference_contract=wrong)


@pytest.mark.parametrize("policy", ["planned-budget-v5", "planned-budget-v6"])
def test_reference_selector_requires_the_explicit_v7_flat_combination(policy):
    with pytest.raises(
        ValueError, match="reference_contract_requires_planned_budget_v7"
    ):
        options_for(policy, reference_contract=REFERENCE_SET_CONTRACT)


def test_reference_selector_is_not_a_public_survey_parameter():
    from litagent.api import SurveyRequest

    assert "reference_contract" not in SurveyRequest.model_fields
    assert "reference_contract" not in SurveyRequest.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SurveyRequest(
            query="Explain the methods.", reference_contract=REFERENCE_SET_CONTRACT
        )


@pytest.mark.parametrize("selected", [False, True])
def test_runner_forwards_only_the_one_explicit_reference_selector(selected):
    graph = TaskGraph()
    graph.add_task(SubTask("write", "Write a report", "synthesis"))
    LitAgent._apply_run_options_to_graph(
        graph,
        options_for(reference_contract=REFERENCE_SET_CONTRACT if selected else None),
    )
    data = graph.tasks["write"].input_data
    assert ("reference_contract" in data) is selected
    if selected:
        assert data["reference_contract"] == REFERENCE_SET_CONTRACT


def envelope(**changes):
    """Build the actual flat round envelope without unrelated recovery flags."""
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
            **changes,
        }
    )


def test_envelope_optional_marker_has_no_inactive_serialized_form():
    old = envelope()
    assert envelope(reference_contract=None) == old
    assert "reference_contract" not in old
    new = envelope(reference_contract=REFERENCE_SET_CONTRACT)
    assert new["reference_contract"] == REFERENCE_SET_CONTRACT
    assert new["identity"] != old["identity"]
    assert validate_writer_envelope(new) == new
    for wrong in [None, False, [], {}, "ordered-reference-set-v2"]:
        broken = copy.deepcopy(new)
        broken["reference_contract"] = wrong
        broken["identity"] = _digest(
            {k: v for k, v in broken.items() if k != "identity"}
        )
        with pytest.raises(
            ValueError, match="writer_envelope_reference_contract_invalid"
        ):
            validate_writer_envelope(broken)


@pytest.mark.parametrize(
    "changes",
    [
        {"contract": "budgeted-writer-envelope-v1"},
        {"round_budget": False},
        {"round_budget": 1},
        {"capacity": None},
        {"reference_contract": "ordered-reference-set-v2"},
        {"reference_contract": True},
    ],
)
def test_reference_envelope_rejects_invalid_version_or_ancestors(changes):
    with pytest.raises(ValueError, match="reference_contract_policy_invalid"):
        envelope(**{"reference_contract": REFERENCE_SET_CONTRACT, **changes})


def test_reference_rule_identity_changes_only_explicit_runs_and_plans(
    case, monkeypatch
):
    import litagent.writing_flat as flat

    common = dict(
        methodology={},
        budget_policy="planned-budget-v7",
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        classification_review=True,
        repair_feedback=True,
    )
    old = writer_budget_plan(case.packet, case.aliases, **common)
    assert (
        writer_budget_plan(case.packet, case.aliases, **common, reference_contract=None)
        == old
    )
    selected = writer_budget_plan(
        case.packet, case.aliases, **common, reference_contract=REFERENCE_SET_CONTRACT
    )
    assert "reference_contract" not in old["envelope"]
    assert selected["envelope"]["reference_contract"] == REFERENCE_SET_CONTRACT
    assert selected["identity"] != old["identity"]
    assert selected["repair_limit"] == 2 and selected["maximum_tokens"] == 4500000
    assert selected["planned_requests"] <= 64
    old_run = options_for(
        repair_feedback=True, mixed_inventory_recovery=True
    ).fingerprint()
    new_options = options_for(
        repair_feedback=True,
        mixed_inventory_recovery=True,
        reference_contract=REFERENCE_SET_CONTRACT,
    )
    new_run = new_options.fingerprint()
    assert old_run != new_run
    monkeypatch.setattr(flat, "reference_set_identity", lambda: "f" * 64)
    assert (
        options_for(repair_feedback=True, mixed_inventory_recovery=True).fingerprint()
        == old_run
    )
    assert new_options.fingerprint() != new_run
    assert writer_budget_plan(case.packet, case.aliases, **common) == old
    assert (
        writer_budget_plan(
            case.packet,
            case.aliases,
            **common,
            reference_contract=REFERENCE_SET_CONTRACT,
        )["identity"]
        != selected["identity"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("selected", [False, True])
async def test_real_synthesis_caller_rebuilds_the_selected_reference_plan(
    case, tmp_path, monkeypatch, selected
):
    """Exercise the actual caller and planner before activation or any send."""
    import litagent.writing_v2 as writer

    class PlannedStop(Exception):
        """Stop after the real producer has built its immutable budget plan."""

    calls = []
    real_plan = writer.writer_budget_plan

    def build_then_stop(*args, **kwargs):
        """Keep actual planner validation while preventing downstream side effects."""
        plan = real_plan(*args, **kwargs)
        calls.append((kwargs, plan))
        raise PlannedStop

    monkeypatch.setattr(writer, "writer_budget_plan", build_then_stop)
    worker = SimpleNamespace(
        _repair_state_path_provider=lambda: tmp_path / "repairs.json",
        _run_id_provider=lambda: "offline-reference-caller",
        _llm=SimpleNamespace(_cost_budget=CostBudget(max_tokens=4500000)),
        _capacity_config=CapacityConfig(),
    )
    task = SimpleNamespace(
        input_data={
            "query": case.packet["query"],
            "generation_strategy": "body_summary_v3",
            "budget_policy": "planned-budget-v7",
            "report_language": "zh-CN",
            **({"reference_contract": REFERENCE_SET_CONTRACT} if selected else {}),
        }
    )
    with pytest.raises(PlannedStop):
        await writer.synthesize(
            worker, task, case.cards, case.selection, {}, case.packet["requirements"]
        )
    kwargs, plan = calls[0]
    assert ("reference_contract" in kwargs) is selected
    assert ("reference_contract" in plan["envelope"]) is selected
    if selected:
        assert kwargs["reference_contract"] == REFERENCE_SET_CONTRACT
        assert plan["envelope"]["reference_contract"] == REFERENCE_SET_CONTRACT
    assert worker._llm._cost_budget.snapshot()["used_tokens"] == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "budget_policy": "planned-budget-v6",
            "reference_contract": REFERENCE_SET_CONTRACT,
        },
        {
            "budget_policy": "planned-budget-v7",
            "reference_contract": "ordered-reference-set-v2",
        },
    ],
)
def test_frozen_planner_and_runtime_reject_invalid_selection_before_material_access(
    kwargs,
):
    with pytest.raises(ValueError, match="reference_contract_policy_invalid"):
        plan_frozen_budget({}, **kwargs)


@pytest.mark.asyncio
async def test_frozen_runtime_rejects_reference_override_before_wiring():
    with pytest.raises(ValueError, match="reference_contract_policy_invalid"):
        await run_frozen(
            SimpleNamespace(_wired=False),
            {},
            "offline-reference-rejected",
            budget_policy="planned-budget-v6",
            reference_contract=REFERENCE_SET_CONTRACT,
        )


def test_manifest_selector_is_explicit_versioned_and_round_only():
    old = {"writing_profile": campaign.ROUND_PROFILE}
    assert campaign.reference_contract_options(old) == {}
    assert "reference_contract" not in campaign.manifest_options(old)
    new = {**old, "reference_contract": REFERENCE_SET_CONTRACT}
    assert campaign.reference_contract_options(new) == {
        "reference_contract": REFERENCE_SET_CONTRACT
    }
    assert (
        campaign.manifest_options(new)["reference_contract"] == REFERENCE_SET_CONTRACT
    )
    for wrong in [None, False, 1, [], {}, "ordered-reference-set-v2"]:
        with pytest.raises(
            ValueError, match="reference_contract_configuration_invalid"
        ):
            campaign.reference_contract_options({**old, "reference_contract": wrong})
    with pytest.raises(ValueError, match="reference_contract_configuration_invalid"):
        campaign.reference_contract_options(
            {**new, "writing_profile": campaign.FLAT_PROFILE}
        )


def test_live_cli_rejects_reference_override_before_configuration(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "frozen",
            "--root",
            str(tmp_path),
            "--live",
            "--reference-contract",
            REFERENCE_SET_CONTRACT,
        ],
    )
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("Reached live configuration")
    )
    with pytest.raises(SystemExit) as caught:
        campaign.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("selected", [False, True])
def test_prepare_cli_forwards_only_an_explicit_selector(
    monkeypatch, tmp_path, selected
):
    calls = []
    monkeypatch.setattr(campaign, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(campaign, "profile_config", lambda config, profile: config)
    monkeypatch.setattr(
        campaign, "prepare", lambda *a, **kw: calls.append(kw) or {"status": "fixture"}
    )
    argv = ["frozen", "--root", str(tmp_path), "--prepare"]
    if selected:
        argv.extend(
            [
                "--writing-profile",
                campaign.ROUND_PROFILE,
                "--reference-contract",
                REFERENCE_SET_CONTRACT,
            ]
        )
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    assert ("reference_contract" in calls[0]) is selected
    if selected:
        assert calls[0]["reference_contract"] == REFERENCE_SET_CONTRACT


@pytest.mark.parametrize("selected", [False, True])
def test_real_frozen_prepare_validate_and_budget_rebuild_share_one_selector(
    case, tmp_path, monkeypatch, selected
):
    """Mock file material I/O only and keep the three-case planning chain real."""
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "artifacts/runs"
    source.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        (source / (run_id + ".json")).write_text(json.dumps({"source": run_id}))
    history = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    history.parent.mkdir(parents=True)
    history.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    bundle = {
        "query": case.packet["query"],
        "requirements": case.packet["requirements"],
        "selection": case.selection.to_dict(),
        "fact_cards_sha256": digest(case.cards),
        "methodology": {},
        "extractions": [],
    }
    monkeypatch.setattr(
        campaign,
        "read_bundle",
        lambda path: {
            **copy.deepcopy(bundle),
            "source_run_id": path.stem,
            "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )
    monkeypatch.setattr(campaign, "source_paths", lambda: [])
    monkeypatch.setattr(campaign, "check_profile", lambda *a, **kw: None)
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    config = AppConfig(agent={}, logging={})
    config.adversarial.thinking_mode = "disabled"
    root = tmp_path / "reference-identity-fixture"
    selection = {"reference_contract": REFERENCE_SET_CONTRACT} if selected else {}
    common = dict(
        inventory_recovery=True,
        mixed_inventory_recovery=True,
        classification_review=True,
        repair_feedback=True,
    )
    campaign.prepare(
        root,
        config,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        **common,
        **selection,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert campaign.validate(root, config) == manifest
    for container in [manifest, manifest["options"], manifest["writing_parameters"]]:
        assert ("reference_contract" in container) is selected
    for row in manifest["cases"]:
        plan = manifest["budget_plans"][row["case_id"]]
        assert ("reference_contract" in plan["envelope"]) is selected
        assert plan == plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=config.capacity,
            **common,
            **selection,
        )
        if selected:
            old = plan_frozen_budget(
                row["bundle"],
                budget_policy="planned-budget-v7",
                capacity=config.capacity,
                **common,
            )
            with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
                validate_frozen_writer_plan(plan, old)
            owner = PlannedBudget(plan)
            before = owner.snapshot()
            with pytest.raises(ValueError, match="identity_conflict"):
                owner.restore(PlannedBudget(old).snapshot())
            assert owner.snapshot() == before
    assert not (root / "state.json").exists()
    if selected:
        manifest["writing_parameters"].pop("reference_contract")
        manifest["identity"] = campaign.digest(
            {k: v for k, v in manifest.items() if k != "identity"}
        )
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="frozen_writing_parameters_conflict"):
            campaign.validate(root, config)
