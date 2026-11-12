"""Route native Evaluation recovery and project authenticated edit guidance."""

import asyncio
import copy
import hashlib

from litagent import review_policy
from litagent.quality import digest

VERSION = "survey-evaluation-ragas-v1"
RECOVERY = "ragas-recovery-v1"


def repetition_followup(original, evaluation, rewrite, envelope):
    """Admit one fully supported, exact-bound draft for a local repetition retry."""
    from litagent.eval.engine_policy import (
        KEYED_GLOBAL_LOCATED_OMISSIONS,
        KEYED_GLOBAL_QUOTED_OMISSIONS,
    )
    from litagent.quality import evaluated_repair_repetitions

    if (
        envelope.get("semantic_engine", {}).get("global_unit_response_contract")
        not in (KEYED_GLOBAL_QUOTED_OMISSIONS, KEYED_GLOBAL_LOCATED_OMISSIONS)
        or rewrite.reason_code != "writer_candidate_failed"
        or not rewrite.candidate
        or not rewrite.candidate_evaluation
        or not rewrite.candidate_metadata
        or not rewrite.candidate_bindings
    ):
        return False
    faithfulness = rewrite.candidate_evaluation.get("faithfulness") or {}
    details = faithfulness.get("details") or {}
    issues = details.get("global_issues") or []
    if (
        faithfulness.get("score") != 1.0
        or details.get("strict_acceptance", {}).get("failure_codes")
        != ["global_inconsistency"]
        or details.get("conflicts")
        or not issues
        or any(issue.get("code") != "repetition" for issue in issues)
        or details.get("writer_bindings") != rewrite.candidate_bindings
    ):
        return False
    try:
        return (
            evaluated_repair_repetitions(
                original, evaluation, rewrite.candidate, rewrite.candidate_evaluation
            )["status"]
            == "failed"
        )
    except (ValueError, KeyError, TypeError):
        return False


def preserves_repetition_findings(initial, final):
    """Require one supported occurrence of each prior scientific finding."""

    def findings(evaluation):
        return {
            (
                claim["section"],
                " ".join(claim["claim_text"].split()).strip(" 。.!！?？;；"),
                tuple(sorted(claim["evidence_ids"])),
            )
            for claim in evaluation["faithfulness"]["details"]["assertions"]
            if claim["kind"] == "scientific" and claim["status"] == "supported"
        }

    return findings(initial) <= findings(final)


def native_repair_diagnostics(details):
    """Bind native report findings to edits without reclassifying NLI verdicts."""
    from litagent.body_quality import bound_repair_diagnostics
    from litagent.eval.engine_policy import ANCHORED_GLOBAL_REVIEW

    diagnostics = bound_repair_diagnostics(details)
    stages = details.get("stages", {})
    if (
        details.get("evaluation_version") != VERSION
        or stages.get("global_findings_contract") != ANCHORED_GLOBAL_REVIEW
        or stages.get("global_check") != "completed"
        or stages.get("global_sections", {}).get("issues", {}).get("status") != "passed"
        or details.get("inventory_check", {}).get("status") != "passed"
    ):
        return diagnostics
    claims = {claim["claim_id"]: claim for claim in details["assertions"]}
    for finding in details["global_issues"]:
        if finding["owner"] != "report" or finding["code"] not in (
            "category_count",
            "repetition",
        ):
            continue
        for cid in finding["claim_ids"]:
            claim = claims.get(cid)
            if claim is None or claim["source_unit_id"] != finding["source_unit_id"]:
                raise ValueError("ragas_recovery_diagnostic_invalid")
            if claim["status"] == "supported" and claim["kind"] == "scientific":
                diagnostics.append(
                    {
                        **claim,
                        "diagnostic_owner": "report",
                        "reason_code": "evaluation_report_" + finding["code"],
                        "explanation": finding["explanation"],
                        "preserve_supported_claim": True,
                    }
                )
    return diagnostics


def recovery_summary(metadata):
    """Project the recorded decision and outcome without publishing source text."""
    from litagent.quality import remaining_repair_allowances

    decision = metadata.get("recovery_decision")
    if not isinstance(decision, dict) or decision.get("contract") != RECOVERY:
        return None
    state = metadata["targeted_repair_state"]
    outcome = metadata.get("evidence_rewrite") or {}
    return {
        "action": decision["action"],
        "decision_reason": decision["reason_code"],
        "outcome_reason": outcome.get("reason_code"),
        "attempted": outcome.get("attempted") is True,
        "committed": outcome.get("committed") is True,
        "budget_status": decision["budget"]["status"],
        "reassessment": (
            "completed"
            if (
                (outcome.get("candidate_evaluation") or {})
                .get("faithfulness", {})
                .get("details", {})
                .get("stages", {})
                .get("global_check")
                == "completed"
            )
            else "unverified"
        ),
        "remaining_attempts": state["limit"] - len(state["attempts"]),
        "remaining_by_category": remaining_repair_allowances(state),
    }


def sanitize_recovery(value):
    """Keep only bounded public recovery fields from archived results."""
    import re

    if not isinstance(value, dict) or value.get("action") not in {
        "stop",
        "prose",
        "combined",
        "inventory",
    }:
        return None
    result = {"action": value["action"]}
    for field in ("decision_reason", "outcome_reason"):
        code = value.get(field)
        result[field] = (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code)
            else None
        )
    result.update(
        attempted=value.get("attempted") is True,
        committed=value.get("committed") is True,
        reassessment=(
            "completed" if value.get("reassessment") == "completed" else "unverified"
        ),
    )
    budget = value.get("budget_status")
    if isinstance(budget, str) and budget in {"admitted", "rejected", "not_checked"}:
        result["budget_status"] = budget
    remaining = value.get("remaining_attempts")
    if type(remaining) is int and 0 <= remaining <= 11:
        result["remaining_attempts"] = remaining
    from litagent.run_limits import REPAIR_ALLOWANCES

    by_category = value.get("remaining_by_category")
    if (
        isinstance(by_category, dict)
        and (
            set(by_category) == set(REPAIR_ALLOWANCES)
            or set(by_category) == set(review_policy.ALLOWANCES)
        )
        and all(
            type(by_category[k]) is int and 0 <= by_category[k] <= n
            for k, n in (
                review_policy.ALLOWANCES
                if set(by_category) == set(review_policy.ALLOWANCES)
                else REPAIR_ALLOWANCES
            ).items()
        )
    ):
        result["remaining_by_category"] = dict(by_category)
    return result


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _settled_evaluation(stages, entries, round_identity):
    """Verify that every completed response belongs to a settled native request."""
    requests = stages["requests"]
    seen = set()
    for row in [
        *requests,
        {
            "phase": "global",
            "request_id": stages["global_request_id"],
            "status": stages["global_check"],
            "usage": stages["global_usage"],
        },
    ]:
        key, phase = row["request_id"], row["phase"]
        entry = entries[key]
        usage = row["usage"]
        _require(
            key not in seen
            and phase in {"inventory", "assessment", "global"}
            and row["status"] == "completed"
            and review_policy.parent_stage(entry["stage"]) == phase
            and entry["status"] == "settled"
            and entry["sent"] is True
            and entry.get("wire_sha256")
            and usage.get("usage_available") is True
            and usage["prompt_tokens"] + usage["completion_tokens"]
            == entry["actual_tokens"]
            and entry["usage_sha256"]
            == digest({k: usage[k] for k in ("prompt_tokens", "completion_tokens")})
            and (round_identity is None or key.startswith(round_identity + ":")),
            "ragas_recovery_receipt_invalid",
        )
        seen.add(key)


def _followup_budget(snapshot, *, edit, summary):
    """Preflight one whole round against the protected stage upper bounds."""
    from litagent.eval.engine_policy import phase_batches

    plan = snapshot["plan"]
    stages = plan["stages"]
    batches = phase_batches(plan["envelope"])
    counts = {"inventory": batches, "assessment": batches, "global": 1}
    if edit:
        counts["repair"] = 1
        if summary:
            counts["summary"] = 1
    reserved = {}
    for stage, count in counts.items():
        stage = review_policy.stage_name(
            plan.get("envelope"), stage, "evaluation_revision" if edit else "inventory"
        )
        spec = stages[stage]
        entries = [e for e in snapshot["entries"].values() if e["stage"] == stage]
        maximum = count * spec["request_cap"]
        _require(
            stage not in snapshot["closed_stages"]
            and len(entries) + count <= spec["count"]
            and sum(e["actual_tokens"] for e in entries) + maximum
            <= spec["reserved_tokens"],
            "ragas_recovery_followup_budget_insufficient",
        )
        reserved[stage] = {"requests": count, "tokens": maximum}
    _require(
        snapshot["requests_sent"] + sum(counts.values()) <= plan["maximum_requests"]
        and snapshot["used_tokens"] + sum(r["tokens"] for r in reserved.values())
        <= plan["planned_tokens"],
        "ragas_recovery_followup_budget_insufficient",
    )
    return reserved


def recovery_decision(report, context, budget):
    """Select one bounded recovery from current native results and paid receipts."""
    from litagent.eval.ragas_native import source_samples
    from litagent.eval.results import EvaluationRunResult
    from litagent.quality import remaining_repair_allowances, targeted_repair_state
    from litagent.run_limits import repair_allowances
    from litagent.safety.planned_accounting import validate_accounting_snapshot
    from litagent.safety.planned_budget import planned_backend
    from litagent.writing_v2 import BODY, repair_targets

    owner = planned_backend(budget)
    if owner is None or not owner.plan["envelope"].get("semantic_engine"):
        return None
    survey, meta = report["survey"], report["metadata"]
    evaluation = report["evaluation"]
    details = evaluation.get("faithfulness", {}).get("details", {})
    sha = hashlib.sha256(survey.encode()).hexdigest()
    decision = {
        "contract": RECOVERY,
        "action": "stop",
        "source_report_sha256": sha,
        "evaluation_sha256": digest(evaluation),
        "targets": [],
        "remaining_attempts": None,
        "budget": {"status": "not_checked"},
    }
    try:
        state = targeted_repair_state(
            meta["targeted_repair_state"], meta["targeted_repair_state"]["run_id"]
        )
        limit = owner.plan["repair_limit"]
        _require(limit == state["limit"], "ragas_recovery_state_invalid")
        balances = remaining_repair_allowances(state)
        _require(
            (balances is None) == (repair_allowances(owner.plan["envelope"]) is None),
            "ragas_recovery_state_invalid",
        )
        if balances is not None:
            decision["remaining_by_category"] = balances
        remaining = limit - len(state["attempts"])
        decision["remaining_attempts"] = remaining
        _require(
            not owner.stop_reason and not owner.snapshot()["usage_uncertain"],
            owner.stop_reason or "ragas_recovery_usage_unknown",
        )
        _require(
            not details.get("execution_error"),
            details.get("execution_error") or "ragas_recovery_execution_incomplete",
        )
        run = EvaluationRunResult.from_details(details, sha)
        stages = details["stages"]
        empty_inventory = (
            not run.assertions
            and not stages.get("batches")
            and all(
                stages.get(key) == 0
                for key in (
                    "expected_assertions",
                    "processed_assertions",
                    "expected_batches",
                )
            )
            and details["inventory_check"]["status"] == "failed"
        )
        _require(
            details["evaluation_version"] == VERSION
            and details["source_report_sha256"] == sha
            and stages.get("global_check") == "completed"
            and details["coverage"]["complete"] is True
            and (run.execution_complete() or empty_inventory)
            and not run.details.get("component_errors"),
            "ragas_recovery_execution_incomplete",
        )
        snapshot = validate_accounting_snapshot(
            owner.snapshot(), expected_plan=owner.plan
        )
        _require(
            all(
                e["status"] == "settled" and e["sent"]
                for e in snapshot["entries"].values()
            ),
            "ragas_recovery_receipt_invalid",
        )
        round_identity = stages.get("evaluation_identity")
        _require(round_identity is not None, "ragas_recovery_identity_invalid")
        _settled_evaluation(stages, snapshot["entries"], round_identity)
        bindings = meta["writing_bindings"]
        _require(
            context["writing_bindings"] == bindings == details["writer_bindings"]
            and context["evidence_snapshot_sha256"] == meta["evidence_snapshot_sha256"]
            and details["evidence_snapshot_sha256"] == meta["evidence_snapshot_sha256"]
            and context["fact_cards"] == meta["fact_cards"],
            "ragas_recovery_source_invalid",
        )
        _, units, _, samples = source_samples(survey, context)
        fragments = {s["source_fragment_id"]: s for s in samples}
        processed = set()
        for claim in run.assertions:
            sample = fragments[claim["source_fragment_id"]]
            _require(
                all(
                    claim[k] == sample[k]
                    for k in (
                        "source_unit_id",
                        "source_start",
                        "source_end",
                        "source_quote",
                        "evidence_ids",
                        "section",
                    )
                )
                and claim["status"] in {"supported", "unsupported"},
                "ragas_recovery_source_invalid",
            )
            processed.add(sample["id"])
        for phase, expected in (
            ("inventory", {s["id"] for s in samples}),
            ("assessment", processed),
        ):
            ids = [
                i
                for r in stages["requests"]
                if r["phase"] == phase
                for i in r["sample_ids"]
            ]
            _require(
                set(ids) == expected and len(ids) == len(expected),
                "ragas_recovery_execution_incomplete",
            )
        negative = [a for a in run.assertions if a["status"] == "unsupported"]
        _require(
            negative == details["unsupported_claims"],
            "ragas_recovery_diagnostic_invalid",
        )
        inventory = not run.inventory_valid()
        omissions = [
            x for x in details["global_issues"] if x.get("owner") == "inventory"
        ]
        decision["inventory_findings"] = {
            "omission_unit_ids": sorted(
                {issue["source_unit_id"] for issue in omissions}
            ),
            "conflicting_unit_ids": details["inventory_check"].get(
                "conflicting_unit_ids", []
            ),
        }
        _require(
            not inventory
            or (
                details["inventory_check"]["status"] == "failed"
                and all(
                    issue["code"] == "missing_assertion"
                    and not issue["claim_ids"]
                    and issue["source_unit_id"] in {u.unit_id for u in units}
                    for issue in omissions
                )
            ),
            "ragas_recovery_diagnostic_invalid",
        )
        diagnostics = negative if inventory else native_repair_diagnostics(details)
        review = meta.get("reviewer_diagnostics") or {}
        if (
            not inventory
            and not review_policy.enabled(owner.plan["envelope"])
            and review.get("source_report_sha256") == sha
        ):
            diagnostics.extend(review.get("issues", []))
        document = meta["writer_document"]
        targets = repair_targets(
            bindings, diagnostics, document["body"], document["summaries"], merge=True
        )
        if targets:
            _require(
                balances is None
                or balances.get("prose", balances.get("evaluation_revision", 0)) > 0,
                "ragas_recovery_prose_limit_reached",
            )
            action = "combined" if inventory else "prose"
            reason = (
                "ragas_recovery_omission_and_prose"
                if inventory
                else "ragas_recovery_prose_issue"
            )
        elif inventory:
            _require(
                (
                    balances["inventory"] > 0
                    if balances is not None
                    else not any(
                        a.get("purpose") == RECOVERY for a in state["attempts"]
                    )
                ),
                "ragas_recovery_inventory_limit_reached",
            )
            action, reason = "inventory", "ragas_recovery_inventory_issue"
        else:
            raise ValueError("ragas_recovery_no_editable_targets")
        _require(remaining > 0, "ragas_recovery_attempts_exhausted")
        decision["budget"] = {"status": "checking"}
        reserved = _followup_budget(
            snapshot,
            edit=bool(targets),
            summary=any(t["path"][0] in BODY for t in targets.values()),
        )
        decision.update(
            action=action,
            reason_code=reason,
            targets=list(targets),
            budget={"status": "admitted", "stages": reserved},
        )
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        reason = str(exc)
        decision["reason_code"] = (
            reason
            if reason.startswith(
                ("ragas_", "planned_", "evaluation_", "budget_", "strict_", "writer_")
            )
            else "ragas_recovery_identity_invalid"
        )
        if decision["budget"]["status"] == "checking":
            decision["budget"]["status"] = "rejected"
    return decision


async def recover_inventory(runner, *, query, report, results, decision):
    """Protect recovery reservations independently of authored revisions."""
    from litagent.safety.planned_budget import locked_writer_envelope

    if review_policy.enabled(locked_writer_envelope(runner._cost_budget)):
        with review_policy.partition("inventory"):
            return await _recover_inventory(
                runner, query=query, report=report, results=results, decision=decision
            )
    return await _recover_inventory(
        runner, query=query, report=report, results=results, decision=decision
    )


async def _recover_inventory(runner, *, query, report, results, decision):
    """Reassess an unchanged report once, charging the shared attempt before send."""
    from litagent.quality import (
        begin_targeted_repair,
        emit_targeted_repair,
        load_targeted_repairs,
        remaining_repair_allowances,
        save_targeted_repairs,
    )
    from litagent.runner import RewriteOutcome

    state = report["metadata"]["targeted_repair_state"]
    path = runner._targeted_repair_path()
    stored = load_targeted_repairs(path, runner._session_id)
    if stored != state:
        raise ValueError("strict_targeted_repair_checkpoint_conflict")
    state = copy.deepcopy(state)
    entry = None

    def before_send(request):
        nonlocal entry
        if entry is not None:
            return
        entry = begin_targeted_repair(
            state, "evaluation", digest(request), category="inventory"
        )
        entry.update(
            purpose=RECOVERY, source_report_sha256=decision["source_report_sha256"]
        )
        save_targeted_repairs(
            path,
            state,
            **{
                f"inventory_recovery_{entry['index']}": {
                    "decision": decision,
                    "first_request": request,
                }
            },
        )
        report["metadata"]["targeted_repair_state"] = state
        emit_targeted_repair(runner._trace_hook, state, "evaluation.inventory_recovery")

    try:
        evaluation = await runner._evaluate(
            query=query,
            survey=report["survey"],
            results=results,
            phase="post_rewrite",
            inventory_recovery=decision,
            recovery_on_send=before_send,
        )
    except BaseException as exc:
        if entry is not None:
            entry["status"] = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            save_targeted_repairs(path, state)
        raise
    details = evaluation.get("faithfulness", {}).get("details", {})
    complete = details.get("coverage", {}).get("complete") is True and not details.get(
        "execution_error"
    )
    valid = complete and details.get("inventory_check", {}).get("status") == "passed"
    balances = remaining_repair_allowances(state)
    retryable = (
        entry is not None
        and balances is not None
        and complete
        and details.get("inventory_check", {}).get("status") == "failed"
        and balances["inventory"] > 0
    )
    if entry is not None:
        entry.update(status="completed" if complete else "failed")
        save_targeted_repairs(
            path,
            state,
            **{
                f"inventory_recovery_{entry['index']}": {
                    "decision": decision,
                    "evaluation": evaluation,
                }
            },
        )
    return RewriteOutcome(
        entry is not None,
        False,
        report["survey"],
        None,
        evaluation,
        runner._derive_quality(evaluation),
        (
            "inventory_recovery_evaluated"
            if valid
            else (
                "inventory_recovery_retryable"
                if retryable
                else "inventory_recovery_failed"
            )
        ),
    )


def project_guidance(diagnostics, *, delegate, host_directory, **kwargs):
    """Validate original provenance before a private, lossless codec conversion.

    Only the internal diagnostic tag is adapted. The published packet restores
    the native tag and original hashes; source locations, scientific judgments,
    target permissions and retained explanations are never reinterpreted.
    """
    from litagent.writing_v2 import WriterError

    try:
        directory = copy.deepcopy(host_directory)
        if (
            directory["identity"]
            != digest({k: v for k, v in directory.items() if k != "identity"})
            or directory["diagnostics_sha256"] != digest(diagnostics)
            or len(directory["issues"]) != len(diagnostics)
        ):
            raise ValueError("native diagnostic identity")
        normalized = copy.deepcopy(diagnostics)
        for index, (raw, issue) in enumerate(
            zip(normalized, directory["issues"], strict=True)
        ):
            if issue["diagnostic_index"] != index or issue[
                "diagnostic_sha256"
            ] != digest(diagnostics[index]):
                raise ValueError("native issue identity")
            if raw.get("diagnostic_source") == VERSION:
                if issue["category"] != "evaluation":
                    raise ValueError("native issue category")
                raw["diagnostic_source"] = "survey-evaluation-v3"
                issue["diagnostic_sha256"] = digest(raw)
        directory["diagnostics_sha256"] = digest(normalized)
        directory["identity"] = digest(
            {k: v for k, v in directory.items() if k != "identity"}
        )
        projected = delegate(normalized, host_directory=directory, **kwargs)
        projected["context_identity"] = host_directory["identity"]
        projected["diagnostics_sha256"] = host_directory["diagnostics_sha256"]
        for issue, raw in zip(projected["issues"], diagnostics, strict=True):
            if raw.get("diagnostic_source") == VERSION:
                issue["diagnostic_source"] = VERSION
        return projected
    except (KeyError, TypeError, ValueError) as exc:
        raise WriterError("writer_guidance_identity_invalid", "identity") from exc
