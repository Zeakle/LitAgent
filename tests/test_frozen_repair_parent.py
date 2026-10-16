"""Inherit exact paid attempt usage without resetting budgets or replaying parents."""

import copy
import json
from pathlib import Path

import pytest
from test_frozen_quality_repair import factories, source

from litagent.config import load_config
from litagent.observability.context import reset_task_id, set_task_id
from litagent.report_audit import digest
from litagent.safety.budget import CostBudget
from scripts.run_frozen_quality_repair import prepare, run_trial


def parent(tmp_path, attempts=2):
    src = source(tmp_path)
    original = json.loads(src.read_text())
    events = []
    for i in range(57):
        pair = copy.deepcopy(original["events"])
        pair[0]["data"]["operation_id"] = pair[1]["data"]["operation_id"] = f"old-{i}"
        events.extend(pair)
    original["events"] = events
    original["usage"]["total_tokens"] = 5700
    src.write_text(json.dumps(original))
    plan = prepare(src, load_config(), "parent-trial")["plan"]
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    budget.restore(plan["source_budget"])
    events = []
    for i in range(attempts):
        data = {
            "task_id": "evidence_rewrite" if i == 0 else "evaluation_inventory",
            "operation_id": f"parent-{i}",
            "prompt_tokens": 9,
            "completion_tokens": 1,
            "total_tokens": 10,
            "usage_available": True,
        }
        events.extend(
            [
                {
                    "event": "llm.start",
                    "data": {
                        "operation_id": data["operation_id"],
                        "task_id": data["task_id"],
                    },
                },
                {"event": "llm.complete", "data": data},
            ]
        )
        token = set_task_id(data["task_id"])
        try:
            budget.begin_request()
            budget.record(data)
        finally:
            reset_task_id(token)
    root = tmp_path / "parent"
    (root / "runs").mkdir(parents=True)
    report = {
        "survey": original["report"]["survey"],
        "metadata": {},
        "evaluation": {},
        "delivery": {"publishable": False},
    }
    state = {
        "status": "blocked",
        "binding": plan["binding"],
        "plan": plan,
        "budget": budget.snapshot(),
        "new_survey_requests": attempts,
        "new_survey_tokens": attempts * 10,
        "usage_complete": True,
        "reason": "contract_invalid",
        "report": report,
    }
    artifact = {
        "run_id": "parent-trial",
        "query": original["query"],
        "status": "completed",
        "events": events,
        "usage": {"total_tokens": attempts * 10},
        "report": report,
    }
    for filename, data in [
        ("checkpoint.json", state),
        ("input-snapshot.json", original),
        ("runs/parent-trial.json", artifact),
    ]:
        (root / filename).write_text(json.dumps(data))
    return src, root


def test_parent_usage_is_additive_and_limits_leave_room_for_complete_evaluation(
    tmp_path,
):
    src, root = parent(tmp_path)
    original = {str(p): p.read_bytes() for p in root.rglob("*.json")}
    prepared = prepare(src, load_config(), "child-trial", parent_root=root)
    plan = prepared["plan"]
    assert (
        plan["source_budget"]["requests_sent"] == 59
        and plan["source_budget"]["used_tokens"] == 5720
    )
    assert plan["source_budget"]["phase_usage"] == [5700, 0, 20]
    assert (
        plan["limits"]["new_survey_requests"] == 5
        and plan["limits"]["new_survey_tokens"] == 494280
    )
    assert (
        prepared["config"].eval.strict_max_claims == 32
        and prepared["config"].eval.strict_max_batches == 2
    )
    assert (
        plan["parent"]["binding"]
        == json.loads((root / "checkpoint.json").read_text())["binding"]
    )
    assert original == {str(p): p.read_bytes() for p in root.rglob("*.json")}
    assert (
        prepare(src, load_config(), "child-trial")["plan"]["binding"] != plan["binding"]
    )


@pytest.mark.parametrize(
    "fault",
    [
        "status",
        "usage_unknown",
        "usage_false",
        "missing_response",
        "duplicate",
        "wrong_total",
        "budget_reset",
        "delta",
        "source_sha",
        "input",
        "body",
        "submitted",
        "query",
        "phase",
        "exhausted",
    ],
)
def test_parent_disagreements_fail_before_preparing_new_transports(tmp_path, fault):
    src, root = parent(tmp_path)
    path = root / "checkpoint.json"
    s = json.loads(path.read_text())
    run = root / "runs/parent-trial.json"
    a = json.loads(run.read_text())
    if fault == "status":
        s["status"] = "started_no_replay"
    elif fault == "usage_unknown":
        a["events"][1]["data"]["usage_available"] = False
    elif fault == "usage_false":
        s["usage_complete"] = False
    elif fault == "missing_response":
        a["events"].pop()
    elif fault == "duplicate":
        a["events"].append(copy.deepcopy(a["events"][0]))
    elif fault == "wrong_total":
        a["usage"]["total_tokens"] += 1
    elif fault == "budget_reset":
        s["budget"] = copy.deepcopy(s["plan"]["source_budget"])
    elif fault == "delta":
        s["new_survey_tokens"] = 0
    elif fault == "source_sha":
        s["plan"]["source_artifact_sha256"] = "forged"
        s["plan"]["binding"] = digest(
            {k: v for k, v in s["plan"].items() if k != "binding"}
        )
        s["binding"] = s["plan"]["binding"]
    elif fault == "input":
        data = json.loads((root / "input-snapshot.json").read_text())
        data["query"] = "changed"
        (root / "input-snapshot.json").write_text(json.dumps(data))
    elif fault == "body":
        a["report"]["survey"] += "changed"
    elif fault == "submitted":
        s["reason"] = "committed"
        a["report"]["delivery"]["publishable"] = True
    elif fault == "query":
        a["query"] = "Changed"
    elif fault == "phase":
        a["events"][1]["data"]["task_id"] = "new_search"
    else:
        s["budget"]["requests_sent"] = 64
    path.write_text(json.dumps(s))
    run.write_text(json.dumps(a))
    with pytest.raises(ValueError):
        prepare(src, load_config(), "child", parent_root=root)


@pytest.mark.asyncio
async def test_dry_run_and_old_authorization_never_start_a_client(tmp_path):
    src, root = parent(tmp_path)
    deps, _ = factories()
    child = tmp_path / "child"
    result = await run_trial(
        src, load_config(), root=child, run_id="child-trial", parent_root=root, **deps
    )
    assert result["paid_calls"] == 0 and not child.exists()
    for f in deps.values():
        f.assert_not_called()
    parent_binding = json.loads((root / "checkpoint.json").read_text())["binding"]
    with pytest.raises(ValueError, match="authorization_required"):
        await run_trial(
            src,
            load_config(),
            root=child,
            run_id="child-trial",
            parent_root=root,
            live=True,
            authorization={"actor": "user", "binding": parent_binding},
            **deps,
        )
    for f in deps.values():
        f.assert_not_called()


@pytest.mark.asyncio
async def test_terminal_child_is_reused_without_adding_a_request(tmp_path):
    src, root = parent(tmp_path)
    deps, agent = factories()
    child = tmp_path / "child"
    plan = prepare(src, load_config(), "child", parent_root=root)["plan"]
    auth = {"actor": "user", "binding": plan["binding"]}
    first = await run_trial(
        src,
        load_config(),
        root=child,
        run_id="child",
        parent_root=root,
        live=True,
        authorization=auth,
        **deps,
    )
    second = await run_trial(
        src,
        load_config(),
        root=child,
        run_id="child",
        parent_root=root,
        live=True,
        authorization=auth,
        **deps,
    )
    assert first == second and first["budget"]["requests_sent"] == 59
    agent._attempt_evidence_rewrite.assert_awaited_once()
    assert deps["client_factory"].call_count == 1


def test_changed_parent_receipt_breaks_grandchild_identity(tmp_path):
    src, root = parent(tmp_path)
    prepared = prepare(src, load_config(), "child", parent_root=root)
    plan = prepared["plan"]
    child = tmp_path / "child"
    (child / "runs").mkdir(parents=True)
    state = {
        "status": "blocked",
        "plan": plan,
        "binding": plan["binding"],
        "budget": plan["source_budget"],
        "new_survey_requests": 0,
        "new_survey_tokens": 0,
        "usage_complete": True,
        "reason": "no_request",
    }
    artifact = {
        "run_id": "child",
        "status": "completed",
        "query": prepared["artifact"]["query"],
        "events": [],
        "usage": {"total_tokens": 0},
        "report": {
            "survey": prepared["artifact"]["report"]["survey"],
            "delivery": {"publishable": False},
        },
    }
    for name, data in [
        ("checkpoint.json", state),
        ("input-snapshot.json", prepared["artifact"]),
        ("runs/child.json", artifact),
    ]:
        (child / name).write_text(json.dumps(data))
    state["report"] = artifact["report"]
    (child / "checkpoint.json").write_text(json.dumps(state))
    grand = prepare(src, load_config(), "grandchild", parent_root=child)
    assert grand["plan"]["source_budget"]["requests_sent"] == 59
    original = json.loads((root / "runs/parent-trial.json").read_text())
    original["new_nonbudget_field"] = "Changed"
    (root / "runs/parent-trial.json").write_text(json.dumps(original))
    with pytest.raises(ValueError, match="receipt"):
        prepare(src, load_config(), "grandchild", parent_root=child)


def test_exhausted_request_capacity_never_spends_on_an_incomplete_plan(tmp_path):
    src, root = parent(tmp_path, attempts=6)
    with pytest.raises(ValueError, match="complete_evaluation_capacity"):
        prepare(src, load_config(), "child", parent_root=root)


def test_cross_attempt_duplicate_operation_is_not_billed_twice(tmp_path):
    src, root = parent(tmp_path)
    path = root / "runs/parent-trial.json"
    data = json.loads(path.read_text())
    data["events"][0]["data"]["operation_id"] = data["events"][1]["data"][
        "operation_id"
    ] = "old-0"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="lifecycle"):
        prepare(src, load_config(), "child", parent_root=root)


def test_parent_chain_cycle_is_rejected(tmp_path):
    src, root = parent(tmp_path)
    state = json.loads((root / "checkpoint.json").read_text())
    state["plan"]["parent"] = {"root": str(root.resolve())}
    state["plan"]["binding"] = digest(
        {k: v for k, v in state["plan"].items() if k != "binding"}
    )
    state["binding"] = state["plan"]["binding"]
    (root / "checkpoint.json").write_text(json.dumps(state))
    with pytest.raises(ValueError, match="cycle"):
        prepare(src, load_config(), "child", parent_root=root)
