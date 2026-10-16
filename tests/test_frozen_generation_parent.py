"""Preserve every paid attempt before extending a frozen validation campaign."""

import hashlib
import json
from pathlib import Path

import pytest
from test_frozen_generation import prepared

from litagent.observability.context import reset_task_id, set_task_id
from litagent.safety.budget import CostBudget
from scripts.run_frozen_generation import file_sha, prepare, run_trial, settle


def save(path, data):
    """Persist exact test receipts in the same shapes as the live supervisor."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def finish(root, plan, operation="trial-op-1", audit=False):
    """Record one failed, fully accounted generation without paid calls."""
    save(root / "execution-plan.json", plan)
    source = json.loads((root / "input-snapshot.json").read_text())
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    token = set_task_id("synthesis")
    try:
        budget.remaining()
        budget.begin_request()
        budget.record({"prompt_tokens": 9, "completion_tokens": 1})
    finally:
        reset_task_id(token)
    costs = {
        s: {"tokens": 0, "requests": 0, "usage_complete": True} for s in plan["limits"]
    }
    costs["survey"].update(tokens=10, requests=1)
    state = {
        "status": "failed",
        "binding": plan["binding"],
        "costs": costs,
        "budget": budget.snapshot(),
        "total_rounds": plan["rounds_before"] + 1,
        "report": None,
    }
    if audit:
        costs["judge"].update(tokens=20, requests=1)
        costs["jev"].update(tokens=30, requests=1)
        audit_result = {
            "source_run_id": plan["run_id"],
            "source_report_revision": 1,
            "source_report_sha256": hashlib.sha256(
                b"# Synthetic audit fixture"
            ).hexdigest(),
            "judge_usage": {
                "total_tokens": 20,
                "call_count": 1,
                "usage_complete": True,
            },
            "jev": {
                "usage": {"input_tokens": 30, "call_count": 1, "usage_complete": True}
            },
        }
        state["report"] = {
            "survey": "# Synthetic audit fixture",
            "delivery": {"publishable": False},
        }
        ap = root / "audit-result.json"
        save(ap, audit_result)
        state["audit_receipt"] = {"path": str(ap), "sha256": file_sha(ap)}
    state["cumulative_after"] = settle(plan, costs)
    save(root / "checkpoint.json", state)
    save(
        Path("artifacts/runs") / f"{plan['run_id']}.json",
        {
            "run_id": plan["run_id"],
            "query": source["question"],
            "status": "failed",
            "report": state["report"],
            "usage": {"total_tokens": 10},
            "events": [
                {
                    "event": "llm.start",
                    "data": {"operation_id": operation, "task_id": "synthesis"},
                },
                {
                    "event": "llm.complete",
                    "data": {
                        "operation_id": operation,
                        "task_id": "synthesis",
                        "prompt_tokens": 9,
                        "completion_tokens": 1,
                        "total_tokens": 10,
                        "usage_available": True,
                    },
                },
            ],
        },
    )
    return state


def child(parent, name):
    """Copy frozen inputs and add the real parent files to protected history."""
    root = parent.parent / name
    root.mkdir()
    (root / "input-snapshot.json").write_bytes(
        (parent / "input-snapshot.json").read_bytes()
    )
    protected = json.loads((parent / "protected-before.json").read_text())["hashes"]
    protected = dict(protected)
    for path in (parent / "checkpoint.json", parent / "execution-plan.json"):
        protected[str(path)] = file_sha(path)
    save(root / "protected-before.json", {"hashes": protected})
    return root


def parent_fixture(tmp_path, monkeypatch, **kwargs):
    """Use a real prepared identity, then account the known parent operation."""
    root, cfg, plan, inputs, history, grant = prepared(tmp_path)
    monkeypatch.chdir(tmp_path)
    state = finish(root, plan, **kwargs)
    next_root = child(root, "next")
    return root, next_root, cfg, history, grant, state


@pytest.mark.asyncio
async def test_real_failure_receipt_is_added_and_dry_run_sends_nothing(
    tmp_path, monkeypatch
):
    parent, root, cfg, history, grant, old = parent_fixture(tmp_path, monkeypatch)
    plan = prepare(root, cfg, "second", history, grant, parent_root=parent)
    assert plan["rounds_before"] == 2
    assert plan["cumulative_before"]["survey"]["charged"] == 110
    assert plan["cumulative_before"]["survey"]["requests"] == 2
    assert plan["remaining_rounds_after_reservation"] == 15
    assert plan["parent_trial"]["hashes"][
        str((parent / "checkpoint.json").resolve())
    ] == file_sha(parent / "checkpoint.json")
    assert (await run_trial(root, cfg, plan))["paid_calls"] == 0
    assert not (root / "checkpoint.json").exists()
    assert json.loads((parent / "checkpoint.json").read_text()) == old


def test_two_generations_inherit_without_refunding_or_reusing_operations(
    tmp_path, monkeypatch
):
    parent, root, cfg, history, grant, _ = parent_fixture(tmp_path, monkeypatch)
    plan = prepare(root, cfg, "second", history, grant, parent_root=parent)
    finish(root, plan, "trial-op-2")
    third = child(root, "third")
    plan3 = prepare(third, cfg, "third", history, grant, parent_root=root)
    assert plan3["rounds_before"] == 3
    assert plan3["cumulative_before"]["survey"]["charged"] == 120
    assert plan3["cumulative_before"]["survey"]["requests"] == 3


def test_protected_latest_parent_cannot_be_omitted(tmp_path, monkeypatch):
    parent, root, cfg, history, grant, _ = parent_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="latest_parent_required"):
        prepare(root, cfg, "second", history, grant)


def test_cannot_skip_latest_known_child(tmp_path, monkeypatch):
    parent, root, cfg, history, grant, _ = parent_fixture(tmp_path, monkeypatch)
    plan = prepare(root, cfg, "second", history, grant, parent_root=parent)
    finish(root, plan, "trial-op-2")
    third = child(root, "third")
    with pytest.raises(ValueError, match="latest_parent_required"):
        prepare(third, cfg, "third", history, grant, parent_root=parent)


@pytest.mark.parametrize(
    "fault",
    [
        "started",
        "unknown",
        "amount",
        "boolean",
        "requests",
        "duplicate_id",
        "missing_response",
        "input",
        "repeat_run_id",
    ],
)
def test_invalid_parent_cannot_grant_another_allowance(tmp_path, monkeypatch, fault):
    parent, root, cfg, history, grant, state = parent_fixture(tmp_path, monkeypatch)
    if fault == "started":
        state["status"] = "started_no_replay"
    elif fault == "unknown":
        state["costs"]["survey"]["usage_complete"] = False
    elif fault == "amount":
        state["cumulative_after"]["survey"]["charged"] = 100
    elif fault == "boolean":
        state["total_rounds"] = True
    elif fault == "requests":
        state["budget"]["requests_sent"] = 0
    elif fault in ("duplicate_id", "missing_response"):
        path = Path("artifacts/runs/new-trial.json")
        a = json.loads(path.read_text())
        if fault == "duplicate_id":
            for e in a["events"]:
                e["data"]["operation_id"] = "old-request"
        else:
            a["events"].pop()
        save(path, a)
    elif fault == "input":
        data = json.loads((root / "input-snapshot.json").read_text())
        data["question"] = "Different question"
        save(root / "input-snapshot.json", data)
    save(parent / "checkpoint.json", state)
    with pytest.raises(ValueError, match="parent_|history_|requires_new_identity"):
        prepare(
            root,
            cfg,
            "new-trial" if fault == "repeat_run_id" else "second",
            history,
            grant,
            parent_root=parent,
        )


def test_parent_cycle_is_rejected_before_another_attempt(tmp_path, monkeypatch):
    from litagent.report_audit import digest

    parent, root, cfg, history, grant, state = parent_fixture(tmp_path, monkeypatch)
    plan = json.loads((parent / "execution-plan.json").read_text())
    plan["parent_trial"] = {
        "root": str(parent),
        "binding": plan["binding"],
        "hashes": {},
    }
    plan["binding"] = digest({k: v for k, v in plan.items() if k != "binding"})
    state["binding"] = plan["binding"]
    save(parent / "execution-plan.json", plan)
    save(parent / "checkpoint.json", state)
    with pytest.raises(ValueError, match="parent_cycle"):
        prepare(root, cfg, "second", history, grant, parent_root=parent)


def test_completed_eighteenth_round_cannot_open_another_allowance(
    tmp_path, monkeypatch
):
    parent, cfg, _, _, history, grant = prepared(tmp_path)
    monkeypatch.chdir(tmp_path)
    baseline = json.loads(history.read_text())
    original = json.loads(
        Path(baseline["observations"][0]["artifact_path"]).read_text()
    )
    observations = []
    for index in range(17):
        record = json.loads(json.dumps(original))
        record["run_id"] = f"baseline-{index}"
        for event in record["events"]:
            event["data"]["operation_id"] = f"baseline-op-{index}"
        path = tmp_path / f"baseline-{index}.json"
        save(path, record)
        observations.append(
            {
                **baseline["observations"][0],
                "run_id": record["run_id"],
                "artifact_path": str(path),
                "artifact_sha256": "sha256:" + file_sha(path),
            }
        )
    baseline["observations"] = observations
    baseline["budgets"]["survey"].update(charged=1700, requests=17)
    save(history, baseline)
    plan = prepare(parent, cfg, "new-trial", history, grant)
    finish(parent, plan)
    root = child(parent, "next")
    with pytest.raises(ValueError, match="round_limit"):
        prepare(root, cfg, "second", history, grant, parent_root=parent)


def test_distinct_judge_jev_receipts_are_not_free_or_combined_with_survey(
    tmp_path, monkeypatch
):
    parent, root, cfg, history, grant, _ = parent_fixture(
        tmp_path, monkeypatch, audit=True
    )
    plan = prepare(root, cfg, "second", history, grant, parent_root=parent)
    assert plan["cumulative_before"]["survey"]["charged"] == 110
    assert plan["cumulative_before"]["judge"]["charged"] == 20
    assert plan["cumulative_before"]["jev"]["charged"] == 30
    audit = json.loads((parent / "audit-result.json").read_text())
    audit["judge_usage"]["total_tokens"] = 0
    save(parent / "audit-result.json", audit)
    with pytest.raises(ValueError, match="parent_audit_changed"):
        prepare(root, cfg, "second", history, grant, parent_root=parent)
