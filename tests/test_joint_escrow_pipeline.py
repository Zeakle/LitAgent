"""Bind aggregate round escrows through frozen preparation and real consumers."""

import copy
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_scientific_repair_pipeline as scientific
import test_structured_repair_pipeline as structured
from model_config_helpers import load_previous_default_config as load_config
from test_concise_repair_prompt import FLAGS
from test_frozen_writer_plan import frozen_bundle  # noqa: F401
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.frozen_writing import plan_frozen_budget, run_frozen
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest
from litagent.safety.planned_budget import (
    JOINT_ROUND_ESCROW_CONTRACT,
    joint_round_escrow_identity,
    with_joint_round_escrow,
)
from litagent.writing_v2 import RANGE_WRITING_PLAN, WriterError, synthesize
from scripts import run_frozen_writing as campaign

RESOURCE_FIELDS = {"round_resource_contract", "round_resource_identity"}


def _plan(bundle, **changes):
    """Keep the selected writer stack identical across resource-only choices."""
    return plan_frozen_budget(
        bundle, budget_policy="planned-budget-v7", **{**FLAGS, **changes}
    )


def _rehash(value):
    """Rehash a tampered outer record without granting producer authority."""
    value["identity"] = digest(
        {key: child for key, child in value.items() if key != "identity"}
    )
    return value


def test_frozen_resource_selector_wraps_only_the_complete_outer_plan(frozen_bundle):
    """Omitted and explicit None keep every old field and identity unchanged."""
    before = copy.deepcopy(frozen_bundle)
    base = _plan(frozen_bundle)
    explicit_none = _plan(frozen_bundle, round_resource_contract=None)
    selected = _plan(frozen_bundle, round_resource_contract=JOINT_ROUND_ESCROW_CONTRACT)
    assert base == explicit_none
    assert "joint_round_escrow" not in base
    assert selected == with_joint_round_escrow(base)
    assert selected["envelope"] == base["envelope"]
    assert selected["planned_tokens"] == base["planned_tokens"]
    assert selected["maximum_tokens"] == base["maximum_tokens"] == 4500000
    assert selected["maximum_requests"] == base["maximum_requests"] == 64
    assert selected["scope_bounds"] == base["scope_bounds"]
    assert selected["repair_limit"] == base["repair_limit"] == 2
    assert selected["identity"] != base["identity"]
    descriptor = selected["joint_round_escrow"]
    assert descriptor["base_plan_identity"] == base["identity"]
    assert descriptor["rule_identity"] == joint_round_escrow_identity()
    assert descriptor["rounds"] == 3
    for stage, spec in base["stages"].items():
        if stage != "assessment":
            assert selected["stages"][stage] == spec
    assert frozen_bundle == before


@pytest.mark.parametrize("bad", [False, True, 0, 1, "", "future", [], {}])
def test_invalid_resource_selector_rejects_before_frozen_materials(bad):
    """A new resource mode is never inferred from arbitrary task-like values."""
    with pytest.raises(ValueError, match="round_resource_policy_invalid"):
        _plan({}, round_resource_contract=bad)


async def _run_pipeline(case, path, monkeypatch, *, selected, failures=0):
    """Replace external HTTP only; retain planner, owner, SDK and report exit."""
    path.mkdir(parents=True, exist_ok=True)
    gateway = structured.LongScientificGateway(case, failures, reviewer=True)
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    flags = dict(FLAGS)
    if selected:
        flags["round_resource_contract"] = JOINT_ROUND_ESCROW_CONTRACT
    monkeypatch.setattr(structured, "FLAGS", flags)
    structured.select_structured_policy(monkeypatch)
    report, returned_gateway = await run_feedback_case(
        case, path, monkeypatch, failures
    )
    assert returned_gateway is gateway
    state, journal, ledger = products(path)
    return SimpleNamespace(
        report=report, gateway=gateway, state=state, journal=journal, ledger=ledger
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failures", [0, 1, 2])
async def test_joint_escrow_reaches_real_owner_full_evaluation_and_report_exit(
    case, tmp_path, monkeypatch, failures
):
    """Resource selection never reduces full reassessment or shared repairs."""
    result = await _run_pipeline(
        case, tmp_path, monkeypatch, selected=True, failures=failures
    )
    report, gateway, journal, ledger = (
        result.report,
        result.gateway,
        result.journal,
        result.ledger,
    )
    plan = ledger["plan"]
    descriptor = plan["joint_round_escrow"]
    assert descriptor["contract"] == JOINT_ROUND_ESCROW_CONTRACT
    assert descriptor["rule_identity"] == joint_round_escrow_identity()
    assert descriptor["rounds"] == 3
    assert plan["stages"]["assessment"]["count"] == (
        descriptor["rounds"] * descriptor["assessment"]["max_requests"]
    )
    assert plan["stages"]["assessment"]["reserved_tokens"] == (
        descriptor["rounds"] * descriptor["assessment"]["reserved_tokens"]
    )
    assert plan["maximum_tokens"] == 4500000 and plan["maximum_requests"] == 64
    assert plan["repair_limit"] == plan["scope_bounds"]["shared_repairs"] == 2
    assert plan["stages"]["repair"]["output_tokens"] == 20000
    assert plan["envelope"]["request_caps"]["assessment"] == 100000
    expected = min(2, failures + 1)
    assert gateway.repair_count == len(sent_with(gateway, "targets")) == expected
    assert gateway.inventories == expected + 1
    assert len(sent_with(gateway, "body_units")) == expected + 1
    assert len(report["metadata"]["targeted_repair_state"]["attempts"]) == expected
    assert len(journal["candidates"]) == expected
    assert ledger["requests_sent"] == len(gateway.sent)
    assert ledger["call_count"] == len(ledger["entries"])
    assert all(entry["status"] == "settled" for entry in ledger["entries"].values())
    assert not ledger["usage_uncertain"]
    assert ledger["capacity_revisions"]
    for entry in gateway.sent:
        # The descriptor funds host operations; it grants no model authority.
        wire = json.dumps(entry["request"])
        assert "joint_round_escrow" not in wire
        assert "round_resource_contract" not in wire
    for entry in sent_with(gateway, "targets"):
        for card in case.cards.values():
            assert card["source_quote"] in json.dumps(entry["request"])
    assert report["delivery"]["publishable"] is (failures < 2)
    if failures < 2:
        details = report["evaluation"]["faithfulness"]["details"]
        assert details["coverage"]["complete"]
        assert details["stages"]["global_check"] == "completed"
        assert details["supported_count"] == details["factual_claim_count"] > 0
        assert details["unsupported_count"] == details["unverified_count"] == 0
        assert journal["candidates"][-1]["committed"]
        assert "FLOPs by" not in report["survey"]
    else:
        assert not any(candidate["committed"] for candidate in journal["candidates"])
        assert "FLOPs by 44%" in report["survey"]


@pytest.mark.asyncio
async def test_outer_resource_choice_changes_no_actual_model_material_or_schema(
    case, tmp_path, monkeypatch
):
    """Compare complete received SDK requests across otherwise identical runs."""
    results = []
    for label, selected in (("legacy", False), ("joint", True)):
        with monkeypatch.context() as scoped_patch:
            results.append(
                await _run_pipeline(
                    case, tmp_path / label, scoped_patch, selected=selected
                )
            )
    before, after = results
    assert "joint_round_escrow" not in before.ledger["plan"]
    assert after.ledger["plan"]["joint_round_escrow"]["base_plan_identity"] == (
        before.ledger["plan"]["identity"]
    )
    assert after.ledger["plan"]["envelope"] == before.ledger["plan"]["envelope"]
    assert [entry["request"] for entry in after.gateway.sent] == [
        entry["request"] for entry in before.gateway.sent
    ]
    assert after.report["survey"] == before.report["survey"]
    assert after.report["delivery"]["publishable"] is True
    assert before.report["delivery"]["publishable"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["missing_plan", "strip_selector", "strip_descriptor"]
)
async def test_frozen_runtime_rejects_partial_resource_selection_before_agent_use(
    frozen_bundle, fault
):
    plan = _plan(frozen_bundle, round_resource_contract=JOINT_ROUND_ESCROW_CONTRACT)
    marker = JOINT_ROUND_ESCROW_CONTRACT
    if fault == "missing_plan":
        plan = None
    elif fault == "strip_selector":
        marker = None
    else:
        plan.pop("joint_round_escrow")
        _rehash(plan)
    agent = SimpleNamespace(_wired=False)
    with pytest.raises(ValueError, match="round_resource_frozen_plan_required"):
        await run_frozen(
            agent,
            frozen_bundle,
            "unstarted-joint-resource",
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            round_resource_contract=marker,
            **FLAGS,
        )
    assert vars(agent) == {"_wired": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["base_plan_identity", "reserved_tokens"])
async def test_frozen_runtime_rebuilds_tampered_descriptor_before_model_setup(
    frozen_bundle, field
):
    plan = _plan(frozen_bundle, round_resource_contract=JOINT_ROUND_ESCROW_CONTRACT)
    if field == "base_plan_identity":
        plan["joint_round_escrow"][field] = "0" * 64
    else:
        plan["joint_round_escrow"]["assessment"][field] -= 1
    _rehash(plan)
    agent = SimpleNamespace(
        _wired=False, _config=SimpleNamespace(capacity=CapacityConfig())
    )
    with pytest.raises(WriterError, match="frozen_budget_plan_drift"):
        await run_frozen(
            agent,
            frozen_bundle,
            "unstarted-joint-tamper",
            budget_policy="planned-budget-v7",
            expected_budget_plan=plan,
            round_resource_contract=JOINT_ROUND_ESCROW_CONTRACT,
            **FLAGS,
        )
    assert agent._wired is False
    assert set(vars(agent)) == {"_wired", "_config"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", ["no_provider", "legacy_provider", "missing", "none", "forged"]
)
async def test_synthesis_derives_resource_authority_only_from_frozen_provider(
    case, frozen_bundle, fault
):
    plan = _plan(frozen_bundle, round_resource_contract=JOINT_ROUND_ESCROW_CONTRACT)
    worker = SimpleNamespace(_frozen_writer_plan_provider=lambda: plan)
    fields = {"round_resource_contract": JOINT_ROUND_ESCROW_CONTRACT}
    if fault == "no_provider":
        worker = SimpleNamespace()
    elif fault == "legacy_provider":
        legacy = _plan(frozen_bundle)
        worker = SimpleNamespace(_frozen_writer_plan_provider=lambda: legacy)
    elif fault == "missing":
        fields = {}
    elif fault == "none":
        fields["round_resource_contract"] = None
    else:
        fields["round_resource_contract"] = "future"
    task = SubTask(
        "synthesis",
        "Untrusted aggregate resource selection",
        "synthesis",
        input_data={
            "generation_strategy": "body_summary_v3",
            "budget_policy": "planned-budget-v7",
            **fields,
        },
    )
    with pytest.raises(WriterError, match="round_resource_task_owner_conflict"):
        await synthesize(worker, task, case.cards, case.selection, {}, [])


def test_resource_selection_remains_internal_and_manifest_pair_is_complete():
    from litagent.api import SurveyRequest
    from litagent.workbench import RunOptions

    assert not RESOURCE_FIELDS.intersection(SurveyRequest.model_fields)
    assert not RESOURCE_FIELDS.intersection(RunOptions.model_fields)
    assert campaign.round_resource_options({}) == {}
    manifest = {
        "writing_profile": campaign.ROUND_PROFILE,
        "round_resource_contract": JOINT_ROUND_ESCROW_CONTRACT,
        "round_resource_identity": joint_round_escrow_identity(),
    }
    assert campaign.round_resource_options(manifest) == {
        "round_resource_contract": JOINT_ROUND_ESCROW_CONTRACT
    }
    for field in RESOURCE_FIELDS:
        damaged = copy.deepcopy(manifest)
        damaged.pop(field)
        with pytest.raises(
            ValueError, match="frozen_round_resource_configuration_invalid"
        ):
            campaign.round_resource_options(damaged)


def test_portable_prepare_validate_binds_descriptor_and_rejects_manifest_strip(
    case, tmp_path, monkeypatch
):
    """Keep archive, plan, source sealing and manifest validation real offline."""
    monkeypatch.chdir(tmp_path)
    archives = tmp_path / "artifacts/runs"
    archives.mkdir(parents=True)
    for _, run_id in campaign.CASES:
        archive = {
            "run_id": run_id,
            "query": case.packet["query"],
            "events": [],
            "nodes": {
                "selection": {
                    "kind": "evidence.retrieve",
                    "output": case.selection.to_dict(),
                },
                "extract": {
                    "kind": "worker",
                    "task_id": "extract",
                    "agent_type": "extractor",
                    "output": [{"paper_id": "evit"}],
                },
            },
        }
        (archives / (run_id + ".json")).write_text(json.dumps(archive))
    baseline = (
        tmp_path
        / "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
    )
    baseline.parent.mkdir(parents=True)
    baseline.write_text(
        json.dumps({"cumulative": {"model_requests": 0, "model_known_tokens": 0}})
    )
    source = tmp_path / "portable_source.py"
    source.write_text('"""Portable joint-resource preparation fixture."""\n')
    monkeypatch.setattr(
        campaign, "source_paths", lambda: [source.relative_to(tmp_path)]
    )
    monkeypatch.setattr(campaign.subprocess, "check_output", lambda *a, **kw: "a" * 40)
    config = campaign.profile_config(load_config(), campaign.ROUND_PROFILE)
    root = tmp_path / "prepared"
    prepared = campaign.prepare(
        root,
        config,
        writing_profile=campaign.ROUND_PROFILE,
        writing_plan=RANGE_WRITING_PLAN,
        round_resource_contract=JOINT_ROUND_ESCROW_CONTRACT,
        **FLAGS,
    )
    manifest = campaign.validate(root, config)
    assert manifest["identity"] == prepared["identity"]
    for container in (manifest, manifest["writing_parameters"]):
        assert container["round_resource_contract"] == JOINT_ROUND_ESCROW_CONTRACT
        assert container["round_resource_identity"] == joint_round_escrow_identity()
    assert not RESOURCE_FIELDS.intersection(manifest["options"])
    for row in manifest["cases"]:
        base = plan_frozen_budget(
            row["bundle"],
            budget_policy="planned-budget-v7",
            capacity=config.capacity,
            **FLAGS,
        )
        assert manifest["budget_plans"][row["case_id"]] == with_joint_round_escrow(base)
    assert not (root / "state.json").exists()
    assert not (root / ".live.lock").exists()
    state = {"identity": manifest["identity"], "status": "prepared", "cases": []}
    state_before, manifest_before = copy.deepcopy(state), copy.deepcopy(manifest)
    manifest_bytes = (root / "manifest.json").read_bytes()
    with monkeypatch.context() as display_patch:
        display_patch.setattr(
            "litagent.safety.planned_budget.joint_round_escrow_identity",
            lambda: pytest.fail("read-only export invoked the current resource rule"),
        )
        public = campaign.export(root, state, manifest)
    assert state == state_before and manifest == manifest_before
    assert (root / "manifest.json").read_bytes() == manifest_bytes
    out = Path("benchmarks/survey/results") / root.name
    assert json.loads((out / "benchmark.json").read_text()) == public
    assert public["budget_plans"] == manifest["budget_plans"]
    assert public["planned"] == public["not_run"] == 3
    assert public["run"] == public["generated"] == public["evaluation_completed"] == 0
    assert public["passed"] == 0 and public["human_calibration"] == "not_validated"
    assert public["new_usage"]["requests"] == 0
    assert public["new_usage"]["known_tokens"] == 0
    with (out / "benchmark.csv").open() as stream:
        csv_rows = {row["case_id"]: row for row in csv.DictReader(stream)}
    markdown = (out / "benchmark.md").read_text()
    html = (out / "benchmark.html").read_text()
    embedded = html.split('<script id="data" type="application/json">', 1)[1].split(
        "</script>", 1
    )[0]
    assert json.loads(embedded) == public
    assert len(csv_rows) == len(public["cases"]) == 3
    assert "严格交付0/3" in markdown and "人工校准未验证" in markdown
    assert "未核验不等于通过" in html
    label = "Locked assessment domain per round"
    assert label in markdown and label in html
    for row in public["cases"]:
        case_id = row["case_id"]
        descriptor = public["budget_plans"][case_id]["joint_round_escrow"]
        assert descriptor == manifest["budget_plans"][case_id]["joint_round_escrow"]
        assert descriptor["contract"] == JOINT_ROUND_ESCROW_CONTRACT
        domain = descriptor["assessment"]
        expected = {
            "budget_round_max_requests": domain["max_requests"],
            "budget_assessment_request_cap": domain["request_cap"],
            "budget_round_reserved_tokens": domain["reserved_tokens"],
        }
        for field, value in expected.items():
            assert row[field] == value and int(csv_rows[case_id][field]) == value
        assert domain["reserved_tokens"] <= (
            domain["max_requests"] * domain["request_cap"]
        )
        assert row["status"] == row["evaluation"] == "not_run"
        assert row["publishable"] is False
        assert row["supported"] is row["unsupported"] is row["unverified"] is None
        assert csv_rows[case_id]["publishable"] == "False"
        assert csv_rows[case_id]["evaluation"] == "not_run"
        assert (
            f"|{case_id}|{domain['max_requests']}|{domain['request_cap']}|"
            f"{domain['reserved_tokens']}|"
        ) in markdown
        assert (
            f"<tr><td>{case_id}</td><td>{domain['max_requests']}</td>"
            f"<td>{domain['request_cap']}</td><td>{domain['reserved_tokens']}</td></tr>"
        ) in html
    assert not (root / "state.json").exists()
    assert not (root / ".live.lock").exists()
    for strip in ("identity", "both"):
        damaged = copy.deepcopy(manifest)
        fields = RESOURCE_FIELDS if strip == "both" else {"round_resource_identity"}
        for container in (damaged, damaged["writing_parameters"]):
            for field in fields:
                container.pop(field)
        _rehash(damaged)
        (root / "manifest.json").write_text(json.dumps(damaged))
        with pytest.raises((ValueError, WriterError)):
            campaign.validate(root, config)


def test_live_cli_cannot_change_resource_contract_before_loading_config(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_frozen_writing",
            "--live",
            "--root",
            str(tmp_path),
            "--round-resource-contract",
            JOINT_ROUND_ESCROW_CONTRACT,
        ],
    )
    monkeypatch.setattr(
        campaign,
        "load_config",
        lambda: pytest.fail("live resource override touched configuration"),
    )
    with pytest.raises(SystemExit) as exc:
        campaign.main()
    assert exc.value.code == 2
