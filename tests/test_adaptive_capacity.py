"""Check complete capacity reallocations through the durable production owner."""

import copy

import pytest
from model_config_helpers import load_previous_default_config as load_config
from pydantic import ValidationError
from test_evaluation_wire_projection import canonical  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import PlannedBudget, request_scope
from litagent.workbench import RunOptions
from litagent.writing_v2 import writer_budget_plan


def plan_for(case, capacity=None):
    """Use the real chapter builder and complete downstream reservation."""
    return writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v6",
        capacity=capacity,
    )


def settle(owner, stage, key, amount):
    """Settle one sent request using the same reserve/send/receipt lifecycle."""
    with request_scope(stage, key):
        owner.reserve(amount)
        owner.begin_request()
        owner.record({"prompt_tokens": amount, "completion_tokens": 0})


def settle_initial(owner, *, maximum=False):
    """Complete writing, summary, Reviewer and independent inventory once."""
    for stage, count in (
        ("body_initial", 1),
        ("body", 4),
        ("summary", 1),
        ("reviewer", 1),
        ("inventory", 1),
    ):
        for index in range(count):
            amount = owner.plan["stages"][stage]["request_cap"] if maximum else 1000
            settle(owner, stage, f"{stage}:{index}", amount)


def test_capacity_defaults_and_legacy_serialization():
    cfg = load_config()
    assert cfg.capacity is None and "capacity" not in cfg.model_dump()
    assert CapacityConfig().initial_assertions == 128
    assert CapacityConfig().max_assertions is None
    assert "capacity_policy" not in RunOptions().model_dump()
    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v6",
        capacity_policy="adaptive-capacity-v1",
        verification={"judge": False, "jev": False},
    )
    assert options.capacity_policy == "adaptive-capacity-v1"
    with pytest.raises(ValidationError):
        RunOptions(capacity_policy="adaptive-capacity-v1")


@pytest.mark.parametrize(
    "values",
    [
        {"batch_size": 17},
        {"max_assertions": 127},
        {"max_assessment_batches": 7},
        {"target_body_units": 67},
        {"initial_assertions": True},
        {"misspelled": 1},
    ],
)
def test_invalid_capacity_dependencies(values):
    with pytest.raises(ValidationError):
        CapacityConfig(**values)


def test_real_plan_locks_initial_path_and_configuration(case):
    plan = plan_for(case)
    assert plan["contract"] == "planned-budget-v6"
    assert plan["planned_requests"] == 41
    assert plan["envelope"]["capacity"] == CapacityConfig().model_dump()
    assert plan["maximum_tokens"] == 4500000
    assert plan["repair_limit"] == 2
    assert PlannedBudget(plan).plan == plan
    changed = plan_for(case, CapacityConfig(target_body_units=40))
    assert changed["identity"] != plan["identity"]


def test_settled_savings_expand_144_and_keep_both_complete_reevaluations(
    case, tmp_path
):
    initial = plan_for(case)
    owner = PlannedBudget(initial, path=tmp_path / "ledger.json")
    settle_initial(owner)
    allocation = owner.expand_capacity(144, 9)
    assert allocation["stages"]["assessment"]["count"] == 27
    assert allocation["stages"]["inventory"]["count"] == 3
    assert allocation["stages"]["repair"]["count"] == 2
    assert allocation["with_margin_tokens"] <= initial["planned_tokens"]
    assert owner.plan == initial
    assert CostBudget.from_snapshot(owner.snapshot()).snapshot() == owner.snapshot()
    # Spend the entire remaining legal path; final verification must still fit.
    for stage, spec in owner.effective_stages.items():
        done = len(owner._stage_entries(stage))
        for index in range(done, spec["count"]):
            settle(owner, stage, f"{stage}:{index}", spec["request_cap"])
    final = owner.snapshot()
    assert final["used_tokens"] == allocation["required_tokens"]
    assert final["used_tokens"] <= final["max_tokens"] == initial["planned_tokens"]
    assert final["requests_sent"] == 44


def test_no_savings_cannot_borrow_complete_repair_reservations(case):
    owner = PlannedBudget(plan_for(case))
    settle_initial(owner, maximum=True)
    before = owner.snapshot()
    with pytest.raises(ValueError, match="unaffordable"):
        owner.expand_capacity(256, 16)
    assert owner.snapshot() == before


def test_unknown_usage_cannot_release_tokens(case):
    owner = PlannedBudget(plan_for(case))
    with request_scope("body_initial", "unknown"):
        owner.reserve(1000)
        owner.begin_request()
        owner.charge_unknown_usage(1000)
    with pytest.raises(ValueError, match="unsafe"):
        owner.expand_capacity(144, 9)
    assert owner.snapshot()["entries"]["unknown"]["status"] == "interrupted_no_replay"


def test_revision_tampering_does_not_change_live_owner(case):
    owner = PlannedBudget(plan_for(case))
    settle_initial(owner)
    owner.expand_capacity(144, 9)
    before = owner.snapshot()
    damaged = copy.deepcopy(before)
    damaged["capacity_revisions"][0]["entries"]["body_initial:0"]["actual_tokens"] = 0
    with pytest.raises(ValueError, match="capacity_revision"):
        owner.restore(damaged)
    assert owner.snapshot() == before


def test_operational_limit_is_separate_from_initial_capacity(case):
    owner = PlannedBudget(plan_for(case, CapacityConfig(max_assertions=140)))
    settle_initial(owner)
    with pytest.raises(ValueError, match="operational"):
        owner.expand_capacity(144, 9)


@pytest.mark.parametrize("count", [129, 144, 240])
def test_real_batch_builders_keep_every_claim_above_initial_capacity(
    case, canonical, count
):
    from litagent.capacity_evaluation import adaptive_eval_config
    from litagent.config import EvalConfig
    from litagent.eval.batched import plan_batches

    owner = PlannedBudget(plan_for(case))
    payload = copy.deepcopy(canonical)
    payload["evaluation_input_contract"] = "flat-evaluation-input-v1"
    payload["source_units"][0]["source_fragments"][0]["text"] += " [E:e1]"
    payload["writer_envelope"] = owner.effective_envelope
    original = payload["assertions"][0]
    payload["assertions"] = [
        {**copy.deepcopy(original), "claim_id": f"claim_{i}"} for i in range(count)
    ]
    batches = plan_batches(payload, adaptive_eval_config(EvalConfig(), owner))
    restored = [c for b in batches for c in b["assertions"]]
    assert len(restored) == len({c["claim_id"] for c in restored}) == count
    assert sorted(restored, key=lambda c: c["claim_id"]) == sorted(
        payload["assertions"], key=lambda c: c["claim_id"]
    )
    assert all(len(b["assertions"]) <= 16 for b in batches)
    assert all(
        b["original_supports"]["s1"]["text"] == "Complete original conditions."
        for b in batches
    )
    with pytest.raises(ValueError, match="assertion_capacity"):
        plan_batches(payload, EvalConfig())


def test_public_request_accepts_only_the_explicit_server_policy_pair():
    from litagent.api import SurveyRequest

    values = dict(
        query="Explain methods.",
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v3",
        budget_policy="planned-budget-v6",
        capacity_policy="adaptive-capacity-v1",
        verification={"judge": False, "jev": False},
    )
    assert SurveyRequest(**values).budget_policy == "planned-budget-v6"
    with pytest.raises(ValidationError):
        SurveyRequest(**{**values, "capacity_policy": "legacy"})
    with pytest.raises(ValidationError):
        SurveyRequest(**{**values, "max_tokens": 99999999})


def test_capacity_resolver_does_not_coerce_false_to_default():
    from litagent.capacity import resolve_capacity

    with pytest.raises(ValidationError):
        resolve_capacity(False)


def test_configured_body_capacity_reaches_compile_and_edit_consumers(case):
    from litagent.capacity_repair import capacity_targets, edit_schema
    from litagent.writing_v2 import compile_document, schema_for

    capacity = CapacityConfig(max_body_units=130, initial_assertions=144)
    plan = plan_for(case, capacity)
    envelope = plan["envelope"]
    body = copy.deepcopy(case.body)
    original = body["methods"][0]["sentences"][0]
    body["methods"] = [
        {
            "sentences": [
                {**copy.deepcopy(original), "text": f"Finding {i}."} for i in range(126)
            ]
        }
    ]
    # This is local transport/compilation coverage, never a scientific verdict.
    report, bindings = compile_document(
        body, case.summaries, case.cards, case.aliases, "en", {}, envelope=envelope
    )
    assert report and len(bindings["units"]) == 132
    targets = capacity_targets(body, {})
    schema = edit_schema(schema_for("repair", case.aliases, targets), targets)
    indexes = schema["properties"]["edit_records"]["items"]["properties"][
        "replacement_indexes"
    ]
    assert indexes["maxItems"] == 126
    assert indexes["items"]["enum"] == list(range(126))
    body["methods"] = [{"sentences": [copy.deepcopy(original)]} for _ in range(126)]
    assert (
        len(
            compile_document(
                body,
                case.summaries,
                case.cards,
                case.aliases,
                "en",
                {},
                envelope=envelope,
            )[1]["units"]
        )
        == 132
    )


@pytest.mark.parametrize("round_policy", [False, True])
def test_capacity_exports_share_one_data_source_without_mutating_state(
    case, tmp_path, monkeypatch, round_policy
):
    import csv
    import json
    import re

    from litagent.contracts import build_config_summary
    from litagent.writing_v2 import RANGE_WRITING_PLAN
    from scripts import run_frozen_writing as flow

    profile = flow.ROUND_PROFILE if round_policy else flow.CAPACITY_PROFILE
    config = flow.profile_config(load_config(), profile)
    plan = (
        writer_budget_plan(
            case.packet, case.aliases, methodology={}, budget_policy="planned-budget-v7"
        )
        if round_policy
        else plan_for(case)
    )
    manifest = {
        "writing_profile": profile,
        "writing_plan": RANGE_WRITING_PLAN,
        "writer_wire_contract": "flat-writer-v1",
        "config": build_config_summary(config, "strict_auto_v2"),
        "identity": "f" * 64,
        "source_commit": "offline",
        "source_hashes": {},
        "history": {},
        "limits": flow.LIMITS,
        "budget_plans": {key: copy.deepcopy(plan) for key, _ in flow.CASES},
    }
    state = {
        "status": "stopped",
        "cases": [
            {
                "case_id": "few-shot-metric",
                "run_id": None,
                "status": "blocked",
                "generated": True,
                "evaluation": "not_complete",
                "publishable": False,
                "requests": 0,
                "known_tokens": 0,
                "unknown_hold": 0,
                "usage_complete": True,
                # This presentation-only fixture made no provider requests.
                "budget_snapshot": PlannedBudget(plan).snapshot(),
                "supported": 0,
                "unsupported": 0,
                "unverified": 178,
                "capacity": {
                    "status": "needs_refinement",
                    "actual_assertions": 184,
                    "actual_batches": 12,
                },
            }
        ],
    }
    if round_policy:
        state["cases"][0].update(
            global_findings_contract="located-global-findings-v1",
            assessment_stage="host_validated",
            inventory_status="failed",
            partial_assessment_stage="host_validated",
            partial_supported=61,
            partial_unsupported=26,
            partial_unverified=30,
        )
        state["cases"][0]["capacity"].update(
            round={"number": 1, "mode": "refine", "status": "superseded_unverified"},
            target_assertions=176,
            target_batches=11,
            assessment_reserved_tokens=820816,
        )
    before = copy.deepcopy(state)
    monkeypatch.chdir(tmp_path)
    flow.export(tmp_path / "view", state, manifest)
    out = tmp_path / "benchmarks/survey/results/view"
    data = json.loads((out / "benchmark.json").read_text())
    embedded = json.loads(
        re.search(
            r'<script id="data" type="application/json">(.*?)</script>',
            (out / "benchmark.html").read_text(),
            re.S,
        ).group(1)
    )
    rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert data == embedded and state == before
    assert data["passed"] == 0 and data["evaluation_completed"] == 0
    assert data["cases"][0]["capacity_initial_assertions"] == 128
    assert data["cases"][0]["capacity_actual_assertions"] == 184
    assert rows[0]["capacity_batches"] == "12"
    assert rows[0]["budget_locked_repairs"] == "2"
    assert rows[0]["budget_body_units"] == "66"
    if round_policy:
        assert rows[0]["partial_supported"] == "61"
        assert data["cases"][0]["partial_unverified"] == 30
        assert "部分诊断非最终分数" in (out / "benchmark.md").read_text()
        assert "部分诊断非最终分数" in (out / "benchmark.html").read_text()

    if round_policy:
        assert data["view_version"] == "frozen-writing-view-v13"
        assert rows[0]["capacity_target_assertions"] == "176"
        assert rows[0]["capacity_assessment_reservation"] == "820816"
        assert "superseded_unverified" in (out / "benchmark.md").read_text()
