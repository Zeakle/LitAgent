"""Exercise native repair round limits, settlement and stopping decisions."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from test_writer_v2 import case  # noqa: F401

from litagent.config import load_config
from litagent.eval.engine_policy import configured_engine, native_writer_budget_plan
from litagent.eval.ragas_global_wire import UNIT_CONTRACT
from litagent.run_limits import configured_run_limits
from litagent.runner import RewriteOutcome
from litagent.safety.planned_budget import PlannedBudget, request_scope
from scripts import run_native_repair_rounds as rounds


@pytest.fixture
def budget(case):
    """Use the real format recovery and native Evaluation budget constructor."""
    config = load_config("config/agent-format-repair.yaml")
    plan = native_writer_budget_plan(
        case.packet,
        case.aliases,
        budget_policy="planned-budget-v7",
        semantic_engine=configured_engine(config),
        capacity=config.capacity,
        methodology={},
        run_limits=configured_run_limits(config),
    )
    assert plan["status"] == "admitted", plan
    history = PlannedBudget(plan).snapshot()
    return rounds.MultiRoundRepairBudget(history, plan, remaining_rounds=4)


def test_round_reservations_keep_format_and_future_rounds(budget):
    """A second repair cannot borrow the next round's slot."""
    assert budget.round_limits["format_repair"]["requests"] == 3
    assert budget.round_limits["format_summary"]["requests"] == 3
    assert budget.trial_limits["repair"]["requests"] == 4
    with pytest.raises(ValueError, match="round_not_open"):
        budget.check_stage_requests("repair", [1])
    for i in range(4):
        budget.begin_round()
        with request_scope("repair", f"repair:{i}"):
            budget.reserve(64)
            budget.bind_wire({"round": i})
            budget.begin_request()
            budget.record({"prompt_tokens": 40, "completion_tokens": 10})
        with pytest.raises(ValueError, match="allowance_exceeded"):
            budget.check_stage_requests("repair", [1])
        assert budget.end_round() == {
            "sent_by_stage": {"repair": 1},
            "settled_tokens": 50,
            "settled": True,
        }
    with pytest.raises(ValueError, match="limit_exhausted"):
        budget.begin_round()
    assert budget.snapshot()["used_tokens"] == 200
    with pytest.raises(ValueError, match="no_replay"):
        budget.restore(budget.snapshot())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("modes", "expected"),
    [
        (["reject", "commit"], "committed"),
        (["reject", "duplicate"], "native_multi_no_progress"),
        (["reject", "unknown"], "strict_usage_unknown"),
        (["reject", "technical"], "evaluation_provider_failed"),
        (["reject", "partial"], "native_multi_candidate_evaluation_incomplete"),
        (["reject", "regressed"], "writer_candidate_regressed"),
        (["reject"] * 4, "native_multi_round_limit_reached"),
    ],
)
async def test_repair_round_terminal_outcome(monkeypatch, budget, modes, expected):
    """Only a novel and completely reevaluated quality failure can continue."""
    baseline = {"faithfulness": {"score": None}}
    report = {
        "survey": "original",
        "evaluation": deepcopy(baseline),
        "quality": {"status": "failed"},
        "metadata": {"query": "question", "report_language": "en"},
    }
    decision = {"action": "combined"}
    checkpoint = []
    attempts = []

    async def attempt(**kwargs):
        assert kwargs["report_data"]["survey"] == "original"
        assert kwargs["initial_evaluation"] == baseline
        index = len(attempts)
        mode = modes[index]
        with request_scope("repair", f"repair:{index}"):
            budget.reserve(64)
            budget.bind_wire({"round": index})
            budget.begin_request()
            if mode == "unknown":
                budget.release(64)
            else:
                budget.record({"prompt_tokens": 40, "completion_tokens": 10})
        candidate = f"candidate {0 if mode == 'duplicate' else index}"
        evaluation = {
            "faithfulness": {
                "details": {
                    "source_report_sha256": rounds._candidate_sha(candidate),
                    "coverage": {"complete": mode != "partial"},
                    "stages": {
                        "global_check": "completed",
                        "global_input_contract": UNIT_CONTRACT,
                    },
                }
            }
        }
        if mode == "commit":
            report.update(survey=candidate, evaluation=evaluation)
        attempts.append(mode)
        reason = {
            "commit": "writer_candidate_committed",
            "technical": "evaluation_provider_failed",
            "regressed": "writer_candidate_regressed",
        }.get(mode, "writer_candidate_failed")
        return RewriteOutcome(
            True, mode == "commit", candidate, None, evaluation, None, reason
        )

    runner = SimpleNamespace(
        _cost_budget=budget,
        _build_eval_context=lambda **kwargs: {},
        _attempt_evidence_rewrite=attempt,
    )
    monkeypatch.setattr(rounds, "recovery_decision", lambda *_: decision)
    result = await rounds.run_repair_rounds(
        runner,
        report,
        {},
        decision,
        checkpoint=lambda rows: checkpoint.append(deepcopy(rows)),
    )
    assert attempts == modes
    assert result.reason == expected
    assert result.rounds == checkpoint[-1]
    assert [row["lineage_round"] for row in result.rounds] == list(
        range(2, 2 + len(modes))
    )
    assert all(
        snapshot[-1]["status"] == "started_no_replay" for snapshot in checkpoint[::2]
    )
    if expected != "committed":
        assert report["survey"] == "original"
        assert report["evaluation"] == baseline
