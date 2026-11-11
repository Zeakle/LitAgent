"""Run multiple funded native repairs without promoting rejected drafts."""

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from litagent.eval.ragas_global_wire import UNIT_CONTRACT
from litagent.eval.ragas_repair import recovery_decision
from litagent.quality import digest
from litagent.run_limits import FORMAT_JSON_POLICY
from litagent.runner import RewriteOutcome
from scripts.run_native_repair_trial import RepairTrialBudget


class MultiRoundRepairBudget(RepairTrialBudget):
    """Keep separate per-round reservations inside one frozen cumulative budget."""

    def __init__(self, history, plan, *, remaining_rounds, path=None):
        """Prove every remaining round fits before creating an execution ledger."""
        if type(remaining_rounds) is not int or not 1 <= remaining_rounds <= 4:
            raise ValueError("native_multi_round_limit_invalid")
        super().__init__(history, plan)
        self.remaining_rounds = remaining_rounds
        self._round_limits = self.trial_limits
        if (
            plan["envelope"].get("run_limits", {}).get("syntax_policy")
            == FORMAT_JSON_POLICY
        ):
            for stage in ("format_repair", "format_summary"):
                self._round_limits[stage] = {
                    "requests": 3,
                    "tokens": 3 * plan["stages"][stage]["request_cap"],
                }
        self._trial_limits = {
            stage: {field: amount * remaining_rounds for field, amount in cap.items()}
            for stage, cap in self._round_limits.items()
        }
        self._check_full_allowance(history)
        self._round_prefix = None
        self._rounds_started = 0
        if path is not None:
            self.path = Path(path)
            with self.path.open("x", encoding="utf-8") as stream:
                json.dump(self.snapshot(), stream, ensure_ascii=False)

    @property
    def round_limits(self):
        """Expose the frozen one-round reservation for the execution manifest."""
        return {stage: cap.copy() for stage, cap in self._round_limits.items()}

    def _check_full_allowance(self, history):
        """Check the worst legal path against the original funded stages and total."""
        plan = self.plan
        for stage, cap in self._trial_limits.items():
            entries = [e for e in history["entries"].values() if e["stage"] == stage]
            spec = plan["stages"][stage]
            if (
                stage in history["closed_stages"]
                or len(entries) + cap["requests"] > spec["count"]
                or sum(e["actual_tokens"] for e in entries) + cap["tokens"]
                > spec["reserved_tokens"]
            ):
                raise ValueError("native_multi_round_budget_insufficient")
        requests = history["requests_sent"] + sum(
            x["requests"] for x in self._trial_limits.values()
        )
        tokens = history["used_tokens"] + sum(
            x["tokens"] for x in self._trial_limits.values()
        )
        if requests > min(plan["planned_requests"], plan["maximum_requests"]) or (
            tokens > min(plan["planned_tokens"], plan["maximum_tokens"])
        ):
            raise ValueError("native_multi_round_budget_insufficient")

    def begin_round(self):
        """Open one round only after all earlier requests have settled."""
        if (
            self._round_prefix is not None
            or self._rounds_started >= self.remaining_rounds
        ):
            raise ValueError("native_multi_round_limit_exhausted")
        if (
            self.stop_reason
            or self.snapshot()["usage_uncertain"]
            or any(
                e["status"] != "settled" or not e["sent"] for e in self.entries.values()
            )
        ):
            raise ValueError("native_multi_previous_round_unsettled")
        self._round_prefix = frozenset(self.entries)
        self._rounds_started += 1

    def end_round(self):
        """Return this round's real send and settlement summary."""
        if self._round_prefix is None:
            raise ValueError("native_multi_round_not_open")
        entries = [v for k, v in self.entries.items() if k not in self._round_prefix]
        self._round_prefix = None
        return {
            "sent_by_stage": dict(Counter(e["stage"] for e in entries if e["sent"])),
            "settled_tokens": sum(
                e["actual_tokens"] for e in entries if e["status"] == "settled"
            ),
            "settled": all(e["status"] == "settled" and e["sent"] for e in entries),
        }

    def check_stage_requests(self, stage, amounts):
        """Refuse cross-round borrowing before the durable send boundary."""
        if self._round_prefix is None:
            raise ValueError("native_multi_round_not_open")
        amounts = tuple(amounts)
        super().check_stage_requests(stage, amounts)
        cap = self._round_limits[stage]
        current = [
            e
            for k, e in self.entries.items()
            if k not in self._round_prefix and e["stage"] == stage
        ]
        if (
            len(current) + len(amounts) > cap["requests"]
            or sum(e["reserved_tokens"] for e in current) + sum(amounts) > cap["tokens"]
        ):
            raise ValueError("native_multi_round_stage_allowance_exceeded")


@dataclass(frozen=True)
class RepairRoundResult:
    """Retain the final outcome and the auditable stopping decision."""

    outcome: RewriteOutcome | None
    rounds: list[dict]
    reason: str


def _candidate_sha(survey):
    """Hash the exact Markdown consumed by Evaluation."""
    return hashlib.sha256(survey.encode()).hexdigest()


def _round_disposition(outcome, accounting, budget, seen):
    """Allow another try only after a complete, novel, settled quality failure."""
    if (
        budget.stop_reason
        or budget.snapshot()["usage_uncertain"]
        or not accounting["settled"]
    ):
        return budget.stop_reason or "native_multi_round_unsettled"
    if outcome.committed:
        return "committed"
    if outcome.reason_code != "writer_candidate_failed":
        return outcome.reason_code
    if not outcome.candidate:
        return "native_multi_candidate_missing"
    sha = _candidate_sha(outcome.candidate)
    if sha in seen:
        return "native_multi_no_progress"
    evaluation = outcome.candidate_evaluation or {}
    details = evaluation.get("faithfulness", {}).get("details", {})
    stages = details.get("stages", {})
    if (
        details.get("source_report_sha256") != sha
        or details.get("execution_error")
        or details.get("component_errors")
        or not details.get("coverage", {}).get("complete")
        or stages.get("global_check") != "completed"
        or stages.get("global_input_contract") != UNIT_CONTRACT
    ):
        return "native_multi_candidate_evaluation_incomplete"
    seen.add(sha)
    return "continue_quality_repair"


async def run_repair_rounds(runner, report, results, first_decision, *, checkpoint):
    """Use the ordinary native repair consumer while keeping the draft immutable."""
    budget = runner._cost_budget
    if not isinstance(budget, MultiRoundRepairBudget):
        raise ValueError("native_multi_budget_required")
    metadata = report["metadata"]
    original_sha = _candidate_sha(report["survey"])
    original_evaluation = digest(report["evaluation"])
    seen = {original_sha}
    rounds = []
    outcome = None
    for index in range(budget.remaining_rounds):
        context = runner._build_eval_context(
            query=metadata["query"], survey=report["survey"], results=results
        )
        decision = recovery_decision(report, context, budget)
        if not index and decision != first_decision:
            raise ValueError("native_multi_first_decision_changed")
        if decision["action"] not in {"combined", "prose"}:
            return RepairRoundResult(outcome, rounds, decision["reason_code"])
        budget.begin_round()
        record = {
            "round": index + 1,
            "lineage_round": index + 2,
            "source_report_sha256": original_sha,
            "decision": decision,
            "status": "started_no_replay",
        }
        rounds.append(record)
        checkpoint(rounds)
        try:
            outcome = await runner._attempt_evidence_rewrite(
                query=metadata["query"],
                report_data=report,
                results=results,
                initial_evaluation=report["evaluation"],
                initial_quality=report["quality"],
                report_language=metadata["report_language"],
            )
        finally:
            accounting = budget.end_round()
        if not outcome.committed and (
            _candidate_sha(report["survey"]) != original_sha
            or digest(report["evaluation"]) != original_evaluation
        ):
            raise ValueError("native_multi_rejected_candidate_mutated_report")
        reason = _round_disposition(outcome, accounting, budget, seen)
        if reason == "continue_quality_repair" and index + 1 == budget.remaining_rounds:
            reason = "native_multi_round_limit_reached"
        record.update(
            status="completed",
            reason=reason,
            outcome_reason=outcome.reason_code,
            candidate_sha256=(
                _candidate_sha(outcome.candidate) if outcome.candidate else None
            ),
            candidate_evaluation_sha256=(
                digest(outcome.candidate_evaluation)
                if outcome.candidate_evaluation
                else None
            ),
            committed=outcome.committed,
            accounting=accounting,
        )
        checkpoint(rounds)
        if reason != "continue_quality_repair":
            return RepairRoundResult(outcome, rounds, reason)
    raise RuntimeError("native_multi_missing_terminal_outcome")
