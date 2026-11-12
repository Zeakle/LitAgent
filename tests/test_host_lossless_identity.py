"""Bind one opt-in lossless transport across frozen and runtime identities."""

import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from test_reference_set_identity import envelope, options_for
from test_writer_v2 import case  # noqa: F401

from litagent.config import AppConfig
from litagent.frozen_writing import (
    plan_frozen_budget,
    run_frozen,
    validate_frozen_writer_plan,
)
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import digest
from litagent.runner import LitAgent
from litagent.safety.planned_budget import (
    PlannedBudget,
    _digest,
    validate_writer_envelope,
)
from litagent.writing_v2 import RANGE_WRITING_PLAN, WriterError, writer_budget_plan
from scripts import run_frozen_writing as campaign

REVISION = "host-lossless-transport-v1"
REFERENCES = "ordered-reference-set-v1"
FLAGS = {
    "inventory_recovery": True,
    "mixed_inventory_recovery": True,
    "classification_review": True,
    "repair_feedback": True,
    "reference_contract": REFERENCES,
}
OPTION_FLAGS = {
    key: value for key, value in FLAGS.items() if key != "inventory_recovery"
}


def selected_options(**changes):
    """Select only the established v7 parents plus the new internal revision."""
    return options_for(**{**OPTION_FLAGS, "transport_revision": REVISION, **changes})


def selected_envelope(**changes):
    """Build the actual established round envelope and its derived identity."""
    return envelope(
        **{
            **FLAGS,
            "global_findings_contract": "anchored-global-findings-v2",
            "summary_constraints": True,
            "transport_revision": REVISION,
            **changes,
        }
    )


@pytest.mark.parametrize(
    "policy", ["planned-budget-v5", "planned-budget-v6", "planned-budget-v7"]
)
def test_none_revision_keeps_existing_options_and_wire_identity(policy):
    old = options_for(policy)
    explicit_none = options_for(policy, transport_revision=None)
    assert old.model_dump(mode="json") == explicit_none.model_dump(mode="json")
    assert old.fingerprint() == explicit_none.fingerprint()
    assert "transport_revision" not in old.model_dump(mode="json")


@pytest.mark.parametrize(
    "wrong", [False, True, 0, 1, "", "host-lossless-transport-v2", [], {}]
)
def test_revision_literal_rejects_invalid_or_coerced_values(wrong):
    with pytest.raises(ValidationError):
        selected_options(transport_revision=wrong)


@pytest.mark.parametrize(
    "change",
    [
        {
            "budget_policy": "planned-budget-v6",
            "classification_review": False,
            "mixed_inventory_recovery": False,
            "repair_feedback": False,
            "reference_contract": None,
        },
        {"generation_strategy": "body_summary_v2"},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
    ],
)
def test_revision_requires_every_existing_parent_contract(change):
    with pytest.raises(
        ValueError, match="transport_revision_requires_explicit_v7_contracts"
    ):
        selected_options(**change)


def test_revision_is_not_a_public_survey_parameter():
    from litagent.api import SurveyRequest

    assert "transport_revision" not in SurveyRequest.model_fields
    assert "transport_revision" not in SurveyRequest.model_json_schema()["properties"]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        SurveyRequest(query="Explain the public methods.", transport_revision=REVISION)


def test_derived_rule_identity_changes_only_selected_runs_and_envelopes(monkeypatch):
    import litagent.writing_flat as flat

    old = options_for(**OPTION_FLAGS)
    selected = selected_options()
    old_fingerprint, selected_fingerprint = old.fingerprint(), selected.fingerprint()
    scope = selected_envelope()
    assert scope["transport_revision"] == REVISION
    assert scope["transport_identity"] == flat.transport_identity()
    assert validate_writer_envelope(scope) == scope
    old_scope = selected_envelope(transport_revision=None)
    assert (
        "transport_revision" not in old_scope and "transport_identity" not in old_scope
    )
    monkeypatch.setattr(flat, "transport_identity", lambda: "f" * 64)
    assert old.fingerprint() == old_fingerprint
    assert selected.fingerprint() != selected_fingerprint
    assert selected_envelope(transport_revision=None) == old_scope
    with pytest.raises(ValueError, match="writer_envelope_transport_revision_invalid"):
        validate_writer_envelope(scope)


@pytest.mark.parametrize(
    "change",
    [
        {"transport_revision": None},
        {"transport_identity": None},
        {"transport_identity": "0" * 64},
        {"inventory_recovery_contract": "bounded-inventory-recovery-v1"},
        {"repair_feedback_contract": None},
        {"reference_contract": None},
    ],
)
def test_rehashed_envelope_cannot_hide_missing_transport_ancestors(change):
    scope = selected_envelope()
    scope.update(change)
    scope["identity"] = _digest(
        {key: value for key, value in scope.items() if key != "identity"}
    )
    with pytest.raises(ValueError):
        validate_writer_envelope(scope)


@pytest.mark.parametrize(
    "change",
    [
        {"transport_revision": "unknown"},
        {"contract": "budgeted-writer-envelope-v1"},
        {"round_budget": False},
        {"capacity": None},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
    ],
)
def test_envelope_rejects_bad_selection_before_material_construction(change):
    with pytest.raises(ValueError, match="transport_revision_policy_invalid"):
        selected_envelope(**change)


@pytest.mark.parametrize("selected", [False, True])
def test_runner_forwards_exactly_the_explicit_revision(selected):
    graph = TaskGraph()
    graph.add_task(SubTask("write", "Write the source-bound report", "synthesis"))
    options = selected_options(transport_revision=REVISION if selected else None)
    LitAgent._apply_run_options_to_graph(graph, options)
    assert ("transport_revision" in graph.tasks["write"].input_data) is selected
    if selected:
        assert graph.tasks["write"].input_data["transport_revision"] == REVISION


@pytest.mark.parametrize(
    "change",
    [
        {"transport_revision": "unknown"},
        {"budget_policy": "planned-budget-v6"},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
    ],
)
def test_frozen_plan_rejects_invalid_revision_before_reading_material(change):
    with pytest.raises(ValueError, match="transport_revision_policy_invalid"):
        plan_frozen_budget(
            {},
            **{
                "budget_policy": "planned-budget-v7",
                **FLAGS,
                "transport_revision": REVISION,
                **change,
            },
        )


@pytest.mark.asyncio
async def test_runtime_rejects_invalid_revision_before_agent_wiring():
    with pytest.raises(ValueError, match="transport_revision_policy_invalid"):
        await run_frozen(
            SimpleNamespace(_wired=False),
            {},
            "offline-rejected",
            budget_policy="planned-budget-v7",
            transport_revision=REVISION,
        )


def test_real_planner_and_checkpoint_identity_bind_only_explicit_transport(case):
    common = {"budget_policy": "planned-budget-v7", "methodology": {}, **FLAGS}
    old = writer_budget_plan(case.packet, case.aliases, **common)
    assert (
        writer_budget_plan(case.packet, case.aliases, **common, transport_revision=None)
        == old
    )
    selected = writer_budget_plan(
        case.packet, case.aliases, **common, transport_revision=REVISION
    )
    assert selected["status"] == "admitted" and selected["repair_limit"] == 2
    assert selected["maximum_tokens"] == 4500000 and selected["planned_requests"] <= 64
    assert selected["envelope"]["transport_revision"] == REVISION
    assert old["identity"] != selected["identity"]
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        validate_frozen_writer_plan(old, selected)
    restored = PlannedBudget(selected)
    before = restored.snapshot()
    with pytest.raises(ValueError, match="identity_conflict"):
        restored.restore(PlannedBudget(old).snapshot())
    assert restored.snapshot() == before


def manifest_fixture(**changes):
    """Construct only the selected server-side manifest parameter view."""
    return {
        "writing_profile": campaign.ROUND_PROFILE,
        **FLAGS,
        "transport_revision": REVISION,
        **changes,
    }


def test_manifest_revision_is_omitted_unless_selected_and_valid():
    old = manifest_fixture()
    old.pop("transport_revision")
    assert campaign.transport_revision_options(old) == {}
    assert "transport_revision" not in campaign.manifest_options(old)
    selected = manifest_fixture()
    assert campaign.transport_revision_options(selected) == {
        "transport_revision": REVISION
    }
    assert campaign.manifest_options(selected)["transport_revision"] == REVISION
    for changes in (
        {"transport_revision": None},
        {"transport_revision": False},
        {"transport_revision": "unknown"},
        {"inventory_recovery": False},
        {"mixed_inventory_recovery": False},
        {"repair_feedback": False},
        {"reference_contract": None},
        {"writing_profile": campaign.FLAT_PROFILE},
    ):
        with pytest.raises(
            ValueError, match="frozen_transport_revision_configuration_invalid"
        ):
            campaign.transport_revision_options(manifest_fixture(**changes))


def test_live_cli_cannot_override_transport_before_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(
        sys,
        "argv",
        ["frozen", "--root", str(tmp_path), "--live", "--transport-revision", REVISION],
    )
    monkeypatch.setattr(
        campaign, "load_config", lambda: pytest.fail("Reached live config")
    )
    with pytest.raises(SystemExit) as caught:
        campaign.main()
    assert caught.value.code == 2


@pytest.mark.parametrize("selected", [False, True])
def test_prepare_cli_forwards_only_selected_transport(monkeypatch, tmp_path, selected):
    calls = []
    monkeypatch.setattr(campaign, "load_config", lambda: SimpleNamespace())
    monkeypatch.setattr(campaign, "profile_config", lambda config, profile: config)
    monkeypatch.setattr(
        campaign, "prepare", lambda *args, **kwargs: calls.append(kwargs) or {}
    )
    argv = ["frozen", "--root", str(tmp_path), "--prepare"]
    if selected:
        argv += ["--transport-revision", REVISION]
    monkeypatch.setattr(sys, "argv", argv)
    campaign.main()
    assert ("transport_revision" in calls[0]) is selected


@pytest.mark.parametrize("selected", [False, True])
def test_real_prepare_validate_and_all_budget_rebuilds_share_transport(
    case, tmp_path, monkeypatch, selected
):
    monkeypatch.chdir(tmp_path)
    archive_root = tmp_path / "artifacts/runs"
    archive_root.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        (archive_root / (run_id + ".json")).write_text(json.dumps({"source": run_id}))
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
    monkeypatch.setattr(campaign, "check_profile", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        campaign.subprocess, "check_output", lambda *args, **kwargs: "a" * 40
    )
    config = AppConfig(agent={}, logging={})
    config.adversarial.thinking_mode = "disabled"
    root = tmp_path / "offline-lossless-identity"
    selector = {"transport_revision": REVISION} if selected else {}
    campaign.prepare(
        root,
        config,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        **FLAGS,
        **selector,
    )
    manifest = json.loads((root / "manifest.json").read_text())
    assert campaign.validate(root, config) == manifest
    for container in (manifest, manifest["options"], manifest["writing_parameters"]):
        assert ("transport_revision" in container) is selected
    for row in manifest["cases"]:
        plan = manifest["budget_plans"][row["case_id"]]
        assert ("transport_revision" in plan["envelope"]) is selected
        assert plan == plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=config.capacity,
            **FLAGS,
            **selector,
        )
    assert not (root / "state.json").exists()
    if selected:
        manifest["writing_parameters"].pop("transport_revision")
        manifest["identity"] = campaign.digest(
            {key: value for key, value in manifest.items() if key != "identity"}
        )
        (root / "manifest.json").write_text(json.dumps(manifest))
        with pytest.raises(ValueError, match="frozen_writing_parameters_conflict"):
            campaign.validate(root, config)
