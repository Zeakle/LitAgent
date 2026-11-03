"""Test durable future allocations through actual request and worker consumers."""

import asyncio
import copy
import csv
import json
from unittest.mock import patch

import httpx
import pytest
from test_native_responses import envelope, owner_for
from test_writer_thinking_profile import configured, manifest  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.llm.routing import ModelBoundClient
from litagent.safety.budget import CostBudget
from litagent.safety.planned_budget import (
    CONTEXT_LIMIT,
    PlannedBudget,
    build_budget_plan,
    complete_writer_plan,
    request_scope,
    wire_reservation,
)
from litagent.workbench import RunOptions, VerificationOptions
from litagent.writing_v2 import WriterSession, schema_for
from scripts import run_frozen_writing as campaign


def stages():
    """Use certified known-wire bounds for a controlled full consumer scenario."""
    return {
        key: {
            "count": count,
            "request_cap": cap,
            "output_tokens": output,
            "bound": "exact_wire",
        }
        for key, count, cap, output in (
            ("body", 5, 40000, 20000),
            ("summary", 3, 25000, 20000),
            ("reviewer", 1, 8000, 4096),
            ("repair", 2, 25000, 20000),
            ("inventory", 3, 18000, 16384),
            ("assessment", 24, 18000, 16384),
            ("global", 3, 18000, 16384),
        )
    }


def plan():
    """Return a controlled complete-path fixture with explicit wire ceilings."""
    return build_budget_plan(stages(), identity="controlled-complete-wire-v1")


def test_complete_plan_reserves_every_shared_repair_and_full_evaluation():
    """Reserve every full reassessment without assuming repairs will succeed."""
    value = plan()
    assert value["status"] == "admitted"
    assert value["planned_tokens"] <= 1000000
    assert value["planned_requests"] == 41
    assert value["required_tokens"] == sum(
        s["count"] * s["request_cap"] for s in stages().values()
    )
    assert value["planned_tokens"] >= value["required_tokens"] * 1.1
    assert value["stages"]["inventory"]["count"] == 3
    assert value["stages"]["assessment"]["count"] == 24


@pytest.mark.parametrize("fault", ["total", "count", "context", "unknown_future"])
def test_oversize_is_rejected_before_a_request(fault):
    """Reject total, context and count overflow without issuing a request."""
    value = stages()
    if fault == "count":
        value["assessment"]["count"] = 65
    elif fault == "context":
        value["body"]["request_cap"] = CONTEXT_LIMIT + 1
    elif fault == "total":
        value["assessment"]["request_cap"] = 100000
    else:
        result = complete_writer_plan(identity="unbounded-future-model-text")
        assert result["status"] == "preflight_rejected"
        assert all(
            s["bound"] == "context_upper_bound" for s in result["stages"].values()
        )
        return
    result = build_budget_plan(value, identity="negative")
    assert result["status"] == "preflight_rejected"
    with pytest.raises(ValueError, match="preflight_rejected"):
        PlannedBudget(result)


def test_full_maximum_path_cannot_consume_followup_allocations(tmp_path):
    """Spend every admitted slot at its bound and retain future stage allocations."""
    budget = PlannedBudget(plan(), path=tmp_path / "ledger.json")
    for stage, spec in stages().items():
        for index in range(spec["count"]):
            with request_scope(stage, f"{stage}:{index}"):
                amount = spec["request_cap"]
                budget.reserve(amount)
                before_send = json.loads((tmp_path / "ledger.json").read_text())
                assert (
                    before_send["entries"][f"{stage}:{index}"]["status"]
                    == "reserved_not_sent"
                )
                budget.begin_request()
                budget.record(
                    {
                        "prompt_tokens": amount - spec["output_tokens"],
                        "completion_tokens": spec["output_tokens"],
                    }
                )
                budget.release(amount)
        with request_scope(stage, stage + ":overflow"):
            with pytest.raises(ValueError, match="envelope_exceeded"):
                budget.reserve(1)
    assert budget.used == plan()["required_tokens"]
    assert budget.snapshot()["requests_sent"] == 41
    restored = CostBudget.from_snapshot(budget.snapshot())
    assert restored.snapshot() == budget.snapshot()
    assert budget.used < budget.snapshot()["max_tokens"]


def test_request_ownership_settlement_and_conflicting_duplicates(tmp_path):
    """Settle usage idempotently and reject conflicting or replayed identities."""
    budget = PlannedBudget(plan(), path=tmp_path / "ledger.json")
    with pytest.raises(ValueError, match="scope_missing"):
        budget.reserve(1)
    with request_scope("body", "body-1"):
        budget.reserve(1000)
        budget.begin_request()
        usage = {"prompt_tokens": 400, "completion_tokens": 500}
        budget.record(usage)
        budget.record(usage)
        budget.charge_unknown_usage(1000)
        assert budget.used == 900 and budget.call_count == 1
        assert not budget.stop_reason
        with pytest.raises(ValueError, match="identity_conflict"):
            budget.record({"prompt_tokens": 400, "completion_tokens": 501})
        budget.release(1000)
        with pytest.raises(ValueError, match="no_replay"):
            budget.reserve(1000)


@pytest.mark.parametrize("sent", [True, False])
def test_restart_holds_unsettled_reservation_without_replay(tmp_path, sent):
    """Preserve unsettled reservations across repeated process recovery."""
    path = tmp_path / "ledger.json"
    budget = PlannedBudget(plan(), path=path)
    with request_scope("inventory", "i"):
        budget.reserve(17500)
        if sent:
            budget.begin_request()
    restored = PlannedBudget(plan(), path=path)
    assert restored.stop_reason == "strict_usage_unknown"
    assert restored.used == 0  # Unknown is an occupied ceiling, not measured usage.
    assert restored.snapshot()["entries"]["i"]["reserved_tokens"] == 17500
    assert restored.snapshot()["requests_sent"] == int(sent)
    again = PlannedBudget(plan(), path=path)
    assert again.snapshot() == restored.snapshot()
    with request_scope("assessment", "must-not-send"):
        with pytest.raises(ValueError, match="unknown"):
            again.reserve(17500)


@pytest.mark.asyncio
async def test_concurrent_stage_requests_cannot_borrow_another_stage():
    """Keep async ownership isolated while settling concurrent requests."""
    budget = PlannedBudget(plan())
    observed = []

    async def send(stage, key, size):
        with request_scope(stage, key):
            budget.reserve(size)
            budget.begin_request()
            await asyncio.sleep(0)
            budget.record({"prompt_tokens": size, "completion_tokens": 0})
            budget.release(size)
            observed.append(key)

    await asyncio.gather(send("body", "b", 25000), send("assessment", "a", 18000))
    assert len(observed) == 2 and budget.used == 43000
    assert budget.snapshot()["requests_sent"] == 2


def test_legacy_budget_bytes_and_run_identity_keep_inactive_default():
    """Retain legacy serialized identities and reject unsafe policy combinations."""
    old = CostBudget()
    old.configure_strict()
    snap = old.snapshot()
    assert snap["schema_version"] == 1 and snap["strict_policy"] == "staged-budget-v1"
    assert CostBudget.from_snapshot(snap).snapshot() == snap
    assert "budget_policy" not in RunOptions().model_dump()
    explicit = RunOptions(budget_policy="legacy")
    assert explicit.fingerprint() == RunOptions().fingerprint()
    with pytest.raises(ValueError):
        RunOptions(budget_policy="planned-budget-v2")
    options = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v2",
        verification=VerificationOptions(judge=False, jev=False),
    )
    assert (
        options.model_copy(update={"budget_policy": "planned-budget-v2"}).fingerprint()
        != options.fingerprint()
    )


@pytest.mark.asyncio
async def test_actual_sdk_disabled_output_and_known_bill_once(case, tmp_path):
    """Send disabled thinking through the actual SDK and record complete usage once."""
    sent = []

    def handler(request):
        value = json.loads(request.content)
        sent.append(value)
        assert value["thinking"] == {"type": "disabled"}
        assert "reasoning" not in value
        assert value["max_output_tokens"] == 20000
        assert value["text"]["format"]["strict"] is True
        return httpx.Response(200, json=envelope(json.dumps(case.body)))

    owner, legacy, _ = await owner_for(handler)
    value = plan()
    legacy.activate_plan(value, path=tmp_path / "budget.json")
    session = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode="disabled",
        provider_api="responses",
    )
    try:
        result = await session.call(
            "body:introduction",
            "Return schema JSON.",
            case.packet,
            schema_for("body", case.aliases),
        )
        assert result["introduction"] == case.body["introduction"]
        assert legacy.used == 120
        assert legacy.snapshot()["requests_sent"] == 1 and len(sent) == 1
    finally:
        await owner.close()


def test_wire_measurement_counts_schema_and_complete_source_material():
    """Include native schemas and every source byte in request reservations."""
    messages = [{"role": "user", "content": "完整公开原文" * 1000}]
    schema = {"type": "json_schema", "json_schema": {"schema": {"type": "object"}}}
    assert wire_reservation(messages, 16384, schema) > wire_reservation(messages, 16384)
    assert (
        wire_reservation(messages, 16384, schema)
        > len(messages[0]["content"].encode()) + 16384
    )


@pytest.mark.asyncio
async def test_frozen_three_domain_admission_stops_before_client_creation(
    manifest, case, monkeypatch
):
    """Preflight all three valid materials before creating any paid runtime."""
    root, config = manifest
    from litagent.quality import digest

    prior_parser = campaign.read_bundle

    def valid_bundle(path):
        """Provide valid fixed sources; do not mock the budget planner."""
        return {
            **prior_parser(path),
            "selection": case.selection.to_dict(),
            "fact_cards_sha256": digest(case.cards),
            "requirements": case.packet["requirements"],
        }

    monkeypatch.setattr(campaign, "read_bundle", valid_bundle)
    target = root.with_name("planned-admission")
    config = campaign.profile_config(config, campaign.PLANNED_PROFILE)
    campaign.prepare(
        target,
        config,
        writing_profile=campaign.PLANNED_PROFILE,
        writing_plan=campaign.RANGE_WRITING_PLAN,
    )
    frozen = campaign.validate(target, config)
    assert len(frozen["budget_plans"]) == 3
    assert frozen["limits"]["total_tokens"] == 3000000
    with patch.object(
        campaign, "LitAgent", side_effect=AssertionError("must not initialize")
    ):
        result = await campaign.execute(target, config)
    assert result["status"] == "preflight_rejected" and result["cases"] == []
    out = campaign.Path("benchmarks/survey/results") / target.name
    public = json.loads((out / "benchmark.json").read_text())
    assert public["run"] == 0 and public["passed"] == 0 and public["not_run"] == 3
    assert public["new_usage"]["requests"] == 0
    assert public["budget_plans"] == frozen["budget_plans"]
    html = (out / "benchmark.html").read_text()
    assert (
        json.loads(
            html.split('<script id="data" type="application/json">')[1].split(
                "</script>"
            )[0]
        )
        == public
    )
    assert "预算准入" in (out / "benchmark.md").read_text()
    rows = list(csv.DictReader((out / "benchmark.csv").open()))
    assert len(rows) == 3
    assert all(r["budget_status"] == "preflight_rejected" for r in rows)


@pytest.mark.asyncio
@pytest.mark.parametrize("unsupported", [False, True])
async def test_real_sdk_evaluation_keeps_semantic_gate_independent(case, unsupported):
    """Budget admission never turns a wrong cited scientific assertion into support."""
    from test_writer_v2 import EvaluatorTransport

    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.quality import digest, strict_acceptance
    from litagent.writing_v2 import compile_document

    evaluator = EvaluatorTransport(unsupported=unsupported)

    async def handler(request):
        value = json.loads(request.content)
        assert request.url.path.endswith("chat/completions")
        answer = await evaluator.chat(
            value["messages"], **{k: v for k, v in value.items() if k != "messages"}
        )
        return httpx.Response(
            200,
            json={
                "id": "offline",
                "object": "chat.completion",
                "created": 0,
                "model": value["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": answer.content},
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 100,
                    "total_tokens": 120,
                },
            },
        )

    owner, budget, _ = await owner_for(handler)
    # This controlled seven-unit transcript contains one assessment batch.
    admitted = build_budget_plan(
        {
            stage: {
                "count": 1,
                "request_cap": CONTEXT_LIMIT,
                "output_tokens": 16384,
                "bound": "context_upper_bound",
            }
            for stage in ("inventory", "assessment", "global")
        },
        identity="controlled-one-batch-real-consumers",
    )
    budget.activate_plan(admitted)
    draft, bindings = compile_document(
        case.body, case.summaries, case.cards, case.aliases, "en"
    )
    context = {
        "query": case.packet["query"],
        "quality_policy": "strict_auto_v2",
        "body_quality": "analysis_v1",
        "report_language": "en",
        "review_structure": "academic_v2",
        "writing_bindings": bindings,
        "scope_notice_contract": "scope-notice-v1",
        "fact_cards": case.cards,
        "evidence_snapshot_sha256": digest(case.cards),
        "evidence": case.ledger,
        "coverage_requirements": case.packet["requirements"],
        "structure_checks": {"status": "passed"},
        "method_checks": {"status": "not_applicable"},
        "language_checks": {"status": "passed"},
    }
    try:
        details = await evaluate_scoped(
            llm=ModelBoundClient(owner, "lite", "evaluation"),
            survey=draft,
            context=context,
            config=EvalConfig(),
        )
        assert details["coverage"]["complete"]
        assert strict_acceptance(details)["status"] == (
            "failed" if unsupported else "passed"
        )
        assert budget.used == 360 and budget.snapshot()["requests_sent"] == 3
        assert all(
            e["status"] == "settled" and e["wire_sha256"]
            for e in budget.snapshot()["entries"].values()
        )
    finally:
        await owner.close()


@pytest.mark.parametrize("fault", ["schema", "output", "count", "scope"])
def test_complete_wire_and_stage_preflight_reject_before_spending(fault):
    """A changed request cannot consume an admitted reservation silently."""
    budget = PlannedBudget(plan())
    with request_scope("assessment", "a"):
        if fault == "schema":
            with pytest.raises(ValueError, match="envelope_exceeded"):
                budget.check_wire(
                    [{"role": "user", "content": "short"}],
                    16384,
                    {
                        "type": "json_schema",
                        "json_schema": {"schema": {"description": "x" * 20000}},
                    },
                )
        elif fault == "output":
            with pytest.raises(ValueError, match="envelope_exceeded"):
                budget.check_wire([{"role": "user", "content": "short"}], 1)
        elif fault == "count":
            with pytest.raises(ValueError, match="capacity_exceeded"):
                budget.check_stage_requests("assessment", [18000] * 25)
        else:
            with request_scope("unplanned", "bad"):
                with pytest.raises(ValueError, match="scope_missing"):
                    budget.reserve(1)
    assert budget.used == 0 and not budget.snapshot()["entries"]


def test_preparation_usage_is_not_reset_when_installing_a_plan():
    """A live Survey retains paid preparation usage at the writing boundary."""
    legacy = CostBudget()
    legacy.configure_strict()
    legacy.begin_request()
    legacy.record({"prompt_tokens": 5000, "completion_tokens": 1000})
    admitted = build_budget_plan(
        stages(), identity="prepared", spent_tokens=6000, spent_requests=1
    )
    legacy.activate_plan(admitted)
    assert legacy.used == 6000 and legacy.call_count == 1
    assert CostBudget.from_snapshot(legacy.snapshot()).snapshot() == legacy.snapshot()


@pytest.mark.parametrize("fault", ["totals", "entries", "closed"])
def test_rejected_restore_does_not_mutate_an_existing_owner(fault):
    """A bad checkpoint must not clear paid identities before raising."""
    budget = PlannedBudget(plan())
    with request_scope("body", "already-paid"):
        budget.reserve(40000)
        budget.begin_request()
        budget.record({"prompt_tokens": 100, "completion_tokens": 100})
    before = budget.snapshot()
    broken = copy.deepcopy(before)
    if fault == "totals":
        broken["entries"] = {}
    elif fault == "entries":
        broken["entries"]["already-paid"]["stage"] = "invalid"
    else:
        broken["closed_stages"] = ["invalid"]
    with pytest.raises(ValueError):
        budget.restore(broken)
    assert budget.snapshot() == before


@pytest.mark.parametrize("state", ["settled", "unknown", "closed"])
def test_older_valid_snapshot_cannot_regrant_consumed_capacity(state):
    """Do not clear paid or unresolved requests, or reopen a closed stage."""
    budget = PlannedBudget(plan())
    old = budget.snapshot()
    if state == "closed":
        budget.close_stage("body")
    else:
        with request_scope("body", "never-repeat"):
            budget.reserve(40000)
            budget.begin_request()
            if state == "unknown":
                budget.charge_unknown_usage(40000)
            else:
                budget.record({"prompt_tokens": 100, "completion_tokens": 100})
    before = budget.snapshot()
    with pytest.raises(ValueError, match="regression"):
        budget.restore(old)
    assert budget.snapshot() == before


def test_forward_restore_preserves_the_shared_owner_and_persistent_path(tmp_path):
    """Validate separately, then install a forward checkpoint on the same owner."""
    path = tmp_path / "durable.json"
    budget = PlannedBudget(plan(), path=path)
    forward = PlannedBudget(plan())
    with request_scope("body", "known-paid"):
        forward.reserve(40000)
        forward.begin_request()
        forward.record({"prompt_tokens": 100, "completion_tokens": 100})
    budget.restore(forward.snapshot())
    assert budget.snapshot() == forward.snapshot()
    assert json.loads(path.read_text()) == budget.snapshot()
    assert budget.path == path


@pytest.mark.parametrize("fault", ["request_id", "wire", "stage", "reservation"])
def test_same_totals_cannot_hide_a_consumed_request_identity_change(fault):
    """Equal counters do not authorize rewriting an already-paid wire identity."""
    budget = PlannedBudget(plan())
    with request_scope("body", "paid-wire"):
        budget.reserve(1000)
        budget.bind_wire({"frozen": "original-request"})
        budget.begin_request()
        budget.record({"prompt_tokens": 100, "completion_tokens": 100})
    before = budget.snapshot()
    rewritten = copy.deepcopy(before)
    entry = rewritten["entries"]["paid-wire"]
    if fault == "request_id":
        rewritten["entries"]["different-request"] = rewritten["entries"].pop(
            "paid-wire"
        )
    elif fault == "wire":
        entry["wire_sha256"] = "0" * 64
    elif fault == "stage":
        entry["stage"] = "summary"
    else:
        entry["reserved_tokens"] = 2000
    with pytest.raises(ValueError, match="regression"):
        budget.restore(rewritten)
    assert budget.snapshot() == before


def test_invalid_restore_cannot_write_or_invoke_the_shared_checkpoint(tmp_path):
    """Temporary validation has no durable side effects or callback ownership."""
    observed = []
    path = tmp_path / "durable.json"
    budget = PlannedBudget(plan(), path=path, checkpoint=observed.append)
    before = path.read_bytes()
    callbacks = len(observed)
    broken = budget.snapshot()
    broken["closed_stages"] = ["not-a-stage"]
    with pytest.raises(ValueError):
        budget.restore(broken)
    assert path.read_bytes() == before and len(observed) == callbacks


def test_fresh_shared_cost_owner_restores_v2_without_replacing_client_references():
    """Resume uses CostBudget.restore on the owner already wired into clients."""
    owner = CostBudget()
    owner.configure_strict()
    client_reference = owner
    previous = PlannedBudget(plan())
    with request_scope("body", "completed-before-restart"):
        previous.reserve(1000)
        previous.begin_request()
        previous.record({"prompt_tokens": 100, "completion_tokens": 100})
    owner.restore(previous.snapshot())
    assert client_reference is owner
    assert client_reference.snapshot() == previous.snapshot()
    with request_scope("body", "completed-before-restart"):
        with pytest.raises(ValueError, match="no_replay"):
            client_reference.reserve(1000)


def test_shared_owner_rejects_a_v2_snapshot_with_conflicting_preparation_usage():
    """Installing a checkpoint cannot erase this owner's existing paid usage."""
    owner = CostBudget()
    owner.configure_strict()
    owner.begin_request()
    owner.record({"prompt_tokens": 100, "completion_tokens": 100})
    before = owner.snapshot()
    with pytest.raises(ValueError):
        owner.restore(PlannedBudget(plan()).snapshot())
    assert owner.snapshot() == before


def test_shared_restore_keeps_matching_preparation_and_later_paid_usage():
    """Installing v2 retains both the live preparation prefix and frozen spend."""
    owner = CostBudget()
    owner.configure_strict()
    owner.begin_request()
    owner.record({"prompt_tokens": 5000, "completion_tokens": 1000})
    admitted = build_budget_plan(
        stages(), identity="prepared-resume", spent_tokens=6000, spent_requests=1
    )
    previous = PlannedBudget(admitted)
    with request_scope("body", "paid-after-preparation"):
        previous.reserve(1000)
        previous.begin_request()
        previous.record({"prompt_tokens": 100, "completion_tokens": 100})
    owner.restore(previous.snapshot())
    assert owner.used == 6200 and owner.call_count == 2
    assert owner.snapshot() == previous.snapshot()


@pytest.mark.parametrize("unsafe", ["reserved", "unknown", "stopped", "non_strict"])
def test_shared_restore_cannot_replace_an_unsafe_live_budget(unsafe):
    """A checkpoint cannot erase in-flight capacity or this owner's stop state."""
    owner = CostBudget()
    if unsafe != "non_strict":
        owner.configure_strict(halt_on_unknown_usage=True)
    if unsafe == "reserved":
        owner.reserve(1000)
    elif unsafe == "unknown":
        owner.charge_unknown_usage(1000)
    elif unsafe == "stopped":
        owner.stop_after_worker_failure()
    before = owner.snapshot()
    with pytest.raises(ValueError, match="activation_unsafe"):
        owner.restore(PlannedBudget(plan()).snapshot())
    assert owner.snapshot() == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing_plan",
        "limit",
        "boolean_count",
        "stop_reason",
        "closed_stage",
        "unknown_policy",
    ],
)
def test_invalid_snapshots_do_not_grant_fresh_capacity(fault):
    """Reject corrupt restore data rather than silently resetting its allowance."""
    snapshot = PlannedBudget(plan()).snapshot()
    if fault == "missing_plan":
        snapshot.pop("plan")
    elif fault == "limit":
        snapshot["max_tokens"] = 1000001
    elif fault == "boolean_count":
        snapshot["call_count"] = False
    elif fault == "stop_reason":
        snapshot["stop_reason"] = "ignore_previous_failure"
    elif fault == "closed_stage":
        snapshot["closed_stages"] = ["unplanned"]
    else:
        snapshot["strict_policy"] = "staged-budget-v1"
    with pytest.raises(ValueError):
        CostBudget.from_snapshot(snapshot)


def test_oversize_model_diagnostics_are_rejected_without_truncation():
    """Bound future diagnostic fields without weakening or editing a claim."""
    from litagent.safety.planned_budget import check_response_bounds

    result = {"decisions": [{"explanation": "条" * 2000}]}
    check_response_bounds(result)
    result["decisions"][0]["explanation"] += "件"
    original = copy.deepcopy(result)
    with pytest.raises(ValueError, match="response_size_exceeded"):
        check_response_bounds(result)
    assert result == original


@pytest.mark.asyncio
async def test_cancelled_sdk_request_keeps_unknown_hold_and_stops_followups(
    case, tmp_path
):
    """Cancellation preserves the durable outbound identity and forbids a retry."""
    sent = []

    async def handler(request):
        sent.append(request)
        raise asyncio.CancelledError()

    owner, legacy, _ = await owner_for(handler)
    legacy.activate_plan(plan(), path=tmp_path / "ledger.json")
    session = WriterSession(
        ModelBoundClient(owner, "pro", "generation"),
        tmp_path / "r.json",
        "r",
        case.packet,
        case.aliases,
        thinking_mode="disabled",
        provider_api="responses",
    )
    try:
        with pytest.raises(asyncio.CancelledError):
            await session.call(
                "body:introduction",
                "Return schema JSON.",
                case.packet,
                schema_for("body", case.aliases),
            )
        snapshot = legacy.snapshot()
        assert snapshot["requests_sent"] == 1 and snapshot["used_tokens"] == 0
        assert (
            snapshot["usage_uncertain"] and legacy.stop_reason == "strict_usage_unknown"
        )
        entry = next(iter(snapshot["entries"].values()))
        assert entry["reserved_tokens"] > 20000 and entry["wire_sha256"]
        restored = PlannedBudget(plan(), path=tmp_path / "ledger.json")
        with request_scope("body", "new-id"):
            with pytest.raises(ValueError, match="unknown"):
                restored.reserve(25000)
        assert len(sent) == 1
    finally:
        await owner.close()


def test_diagnostic_limits_do_not_truncate_or_reject_complete_source_quotes():
    """Scientific quotes include citation syntax and are not diagnostic prose."""
    from litagent.safety.planned_budget import check_response_bounds

    value = {"source_quote": "x" * 3000, "explanation": "y" * 2000}
    original = copy.deepcopy(value)
    check_response_bounds(value)
    assert value == original
