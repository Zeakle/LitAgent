"""Coordinate editorial drafts independently of the final Evaluation owner."""

import copy
import json

from litagent import review_policy
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest, repair_attempts_remaining


def writer_session(client, path, run_id, document, *, trace_hook=None):
    """Reopen the same authenticated writer and raw-reference journal."""
    from litagent.writing_v2 import WriterSession

    envelope = document.get("budget_envelope")
    if review_policy.enabled(envelope) and (
        envelope["run_limits"].get("workflow_identity") != review_policy.identity()
    ):
        raise ValueError("review_workflow_identity_invalid")
    session = WriterSession(
        client,
        path,
        run_id,
        document["packet"],
        document["aliases"],
        trace_hook=trace_hook,
        thinking_mode=document.get("generation_thinking_mode", "disabled"),
        reasoning_effort=document.get("generation_reasoning_effort"),
        provider_api=document.get("generation_provider_api", "chat"),
        shape_recovery=document.get("shape_recovery_contract")
        == "lossless-array-reemission-v1",
        reemission_contract=document.get("generation_reemission_contract"),
        syntax_policy=document.get("generation_syntax_policy"),
        writing_plan=document.get("generation_writing_plan"),
    )
    if not review_policy.enabled(session.envelope) or envelope != session.envelope:
        raise ValueError("review_cycle_envelope_conflict")
    return session


def improvement(report, candidate, regressed):
    """Promote only a fully assessed reduction of exact-origin problem groups."""
    result = {"promote": False, "reason": "candidate_not_independently_verified"}
    try:
        old_eval = report["evaluation"]
        new_eval = candidate.candidate_evaluation
        old = old_eval["faithfulness"]["details"]
        new = new_eval["faithfulness"]["details"]
        if (
            not candidate.candidate
            or new.get("execution_error")
            or new.get("component_errors")
            or new["inventory_check"]["status"] != "passed"
            or not new["coverage"]["complete"]
            or new["stages"]["global_check"] != "completed"
            or regressed(
                old_eval,
                new_eval,
                original_survey=report["survey"],
                candidate_survey=candidate.candidate,
            )
        ):
            return result
        from litagent.eval.results import EvaluationRunResult

        # Every independently passed check must remain passed; score changes alone
        # cannot authorize a new working version.
        before_checks = EvaluationRunResult.from_details(
            old, report["metadata"]["writing_bindings"]["report_sha256"]
        ).checks()
        after_checks = EvaluationRunResult.from_details(
            new, candidate.candidate_bindings["report_sha256"]
        ).checks()
        if any(
            v.get("status") == "passed"
            and after_checks.get(k, {}).get("status") != "passed"
            for k, v in before_checks.items()
        ):
            return {**result, "reason": "candidate_check_regressed"}
        provenance = candidate.candidate_metadata["targeted_repair_state"]["attempts"][
            -1
        ]["provenance"]
        if provenance["identity"] != digest(
            {k: v for k, v in provenance.items() if k != "identity"}
        ):
            raise ValueError("candidate_lineage_invalid")
        if provenance["source_body_sha256"] != digest(
            report["metadata"]["writer_document"]["body"]
        ) or provenance["candidate_body_sha256"] != digest(
            candidate.candidate_metadata["writer_document"]["body"]
        ):
            raise ValueError("candidate_lineage_invalid")
        origins = {
            tuple(u["path"]): tuple(u["source_range"]) for u in provenance["units"]
        }

        def groups(details, bindings, mapped):
            found = set()

            def locate(item):
                matches = [
                    u
                    for u in bindings["units"]
                    if (
                        u["source_start"] <= item["source_start"] < u["source_end"]
                        and item["source_end"] <= u["source_end"]
                    )
                ]
                if len(matches) != 1:
                    raise ValueError("candidate_lineage_ambiguous")
                path = tuple(matches[0]["path"])
                if path[0] in {"abstract", "conclusion"}:
                    return (path[0],)
                return origins[path] if mapped else (*path, path[-1] + 1)

            for claim in details["assertions"]:
                if claim["status"] == "unsupported":
                    found.add((locate(claim), "scientific_support"))
                elif claim["status"] != "supported":
                    raise ValueError("candidate_incomplete")
            claims = {c["claim_id"]: c for c in details["assertions"]}
            for issue in details.get("global_issues", []):
                if not issue.get("claim_ids"):
                    raise ValueError("candidate_issue_without_lineage")
                for cid in issue["claim_ids"]:
                    found.add((locate(claims[cid]), issue["code"]))
            return found

        previous = groups(old, report["metadata"]["writing_bindings"], False)
        current = groups(new, candidate.candidate_bindings, True)
        promote = bool(current < previous)
        return {
            "promote": promote,
            "reason": "bound_problems_reduced" if promote else "no_bound_improvement",
            "previous_problem_groups": len(previous),
            "candidate_problem_groups": len(current),
        }
    except (KeyError, TypeError, ValueError, AttributeError):
        return {**result, "reason": "candidate_improvement_unproven"}


def _settled_rejected_editorial_patch(session, error):
    """Recognize a settled patch rejection that leaves the draft intact."""
    from litagent.safety.planned_budget import planned_backend
    from litagent.writing_v2 import classify_failure

    if (
        error.family != "protocol"
        or error.code not in {"writer_flat_schema_invalid", "writer_duplicate_key"}
        or not session.state["attempts"]
    ):
        return False
    attempt = session.state["attempts"][-1]
    if (
        attempt["category"] != "review_revision"
        or attempt["status"] != "failed"
        or attempt.get("error_code") != error.code
    ):
        return False
    name = f"repair_{attempt['index']}"
    failure = session.journal.get("failure", {})
    call = session.journal["calls"].get(name, {})
    if (
        failure.get("call") != name
        or failure.get("stage") != "repair"
        or failure.get("response_received") is not True
        or call.get("status") != "failed_no_replay"
        or call.get("provider_status") != "completed"
        or call.get("finish_reason") != "stop"
        or call.get("provider_error")
        or call.get("failure") != classify_failure(error)
    ):
        return False
    budget = planned_backend(getattr(session.llm, "_cost_budget", None))
    if budget is None or budget.stop_reason:
        return False
    settlement = budget.entries.get(f"{session.identity}:{name}", {})
    if (
        settlement.get("stage") != "review_revision/repair"
        or settlement.get("status") != "settled"
        or settlement.get("sent") is not True
    ):
        return False
    if error.code == "writer_duplicate_key":
        # Strict parsing rejected the whole response before applying any edit.
        return True
    if not error.diagnostics:
        return False
    try:
        reply = json.loads(call["raw"])
        for diagnostic in error.diagnostics:
            path = diagnostic.get("path", [])
            if (
                diagnostic.get("reason") != "minLength"
                or len(path) != 3
                or path[0] != "patches"
                or type(path[1]) is not int
                or path[1] < 0
                or path[2] != "text"
            ):
                return False
            patch = reply["patches"][path[1]]
            response = reply["target_responses"][patch["target_id"]]
            if patch["text"] != "" or response["action"] != "replace":
                return False
    except (IndexError, KeyError, TypeError, ValueError):
        return False
    return True


async def editorial_cycle(worker, task, source):
    """Review each valid draft, retaining disputes for independent Evaluation."""
    from litagent.writing_v2 import (
        BODY,
        WriterError,
        _split_summary_repair_guidance,
        compile_document,
        current_repair_guidance_context,
        refresh_summaries,
        repair_targets,
    )

    current = copy.deepcopy(source)
    doc = current["writer_document"]
    session = writer_session(
        worker._synthesis._llm,
        worker._synthesis._repair_state_path_provider(),
        worker._synthesis._run_id_provider(),
        doc,
        trace_hook=worker._trace_hook,
    )
    history = []
    seen = {current["writing_bindings"]["report_sha256"]}
    for round_number in range(4):
        worker._reviewer._writer_path = session.path
        worker._reviewer._writer_run_id = session.run_id
        review_task = SubTask(
            task_id=task.task_id,
            description="Review the current editorial draft",
            agent_type="reviewer",
            input_data={
                **task.input_data,
                "review_round": round_number,
                "upstream_results": {"synthesis": current},
            },
        )
        async with worker._sub_span(task.task_id, "reviewer", round_number + 1):
            review = await worker._reviewer.execute(review_task)
        # Reviewer owns the same journal; reopen it before the writer saves.
        session = writer_session(
            worker._synthesis._llm,
            worker._synthesis._repair_state_path_provider(),
            worker._synthesis._run_id_provider(),
            current["writer_document"],
            trace_hook=worker._trace_hook,
        )
        if (
            review["source_report_sha256"]
            != current["writing_bindings"]["report_sha256"]
        ):
            raise ValueError("review_cycle_report_conflict")
        row = {
            "round": round_number,
            "source_report_sha256": review["source_report_sha256"],
            "review": copy.deepcopy(review),
            "status": "reviewed",
        }
        history.append(row)
        current["reviewer_diagnostics"] = review
        session.journal["editorial_history"] = copy.deepcopy(history)
        session.save()
        if not review["issues"]:
            row["status"] = "review_accepted"
            break
        if repair_attempts_remaining(session.state, "review_revision") <= 0:
            row["status"] = "review_limit_reached"
            break
        doc = current["writer_document"]
        targets = repair_targets(
            current["writing_bindings"],
            review["issues"],
            doc["body"],
            doc["summaries"],
            merge=True,
        )
        if not targets:
            raise ValueError("review_cycle_unlocated_issues")
        issues, categories = review["issues"], ["reviewer"] * len(review["issues"])
        summary_guidance = None
        summary_only = all(t["path"][0] not in BODY for t in targets.values())
        if not summary_only:
            issues, categories, summary_guidance = _split_summary_repair_guidance(
                current["draft"], current["writing_bindings"], doc, issues, categories
            )
        guidance = {
            "current": current_repair_guidance_context(
                current["draft"],
                current["writing_bindings"],
                doc,
                targets,
                issues,
                diagnostic_categories=categories,
            ),
            "prior_candidate": None,
        }
        try:
            with review_policy.partition("review_revision"):
                edited = await session.patch(
                    doc["summaries"] if summary_only else doc["body"],
                    targets,
                    cause="review",
                    summary_body=doc["body"] if summary_only else None,
                    guidance_context=guidance,
                )
                row["responses"] = copy.deepcopy(
                    session.state["attempts"][-1].get("target_responses", {})
                )
                if edited == (doc["summaries"] if summary_only else doc["body"]):
                    row["status"] = "unchanged_with_response"
                    break
                if summary_only:
                    body, summaries = doc["body"], edited
                else:
                    body = edited
                    if summary_guidance:
                        name = "summary_repair_" + str(len(session.state["attempts"]))
                        session.journal.setdefault("summary_regeneration_guidance", {})[
                            name
                        ] = summary_guidance
                        session.save()
                    summaries = await refresh_summaries(
                        session, doc["body"], body, doc["summaries"], targets
                    )
                draft, bindings = compile_document(
                    body,
                    summaries,
                    current["fact_cards"],
                    doc["aliases"],
                    current["report_language"],
                    current.get("review_methodology"),
                    writing_plan=session.writing_plan,
                    envelope=session.envelope,
                )
            if bindings["report_sha256"] in seen:
                row["status"] = "repeated_candidate"
                break
            row.update(
                status="revised_unverified", candidate_sha256=bindings["report_sha256"]
            )
            seen.add(bindings["report_sha256"])
            current.update(
                draft=draft,
                writing_bindings=bindings,
                writer_document={**doc, "body": body, "summaries": summaries},
            )
            session.journal.setdefault("editorial_candidates", []).append(
                {
                    "source_report_sha256": row["source_report_sha256"],
                    "survey": draft,
                    "bindings": bindings,
                    "writer_document": current["writer_document"],
                    "evaluation_status": "unverified",
                    "publishable": False,
                }
            )
        except WriterError as exc:
            row.update(status="candidate_rejected", reason_code=exc.code)
            if exc.family != "quality":
                if not _settled_rejected_editorial_patch(session, exc):
                    raise
                # Preserve the failed call and spent allowance, but do not
                # present this discarded edit as a failure of the retained draft.
                session.journal.pop("failure")
            # An editorial disagreement or unusable edit cannot erase a valid draft.
            break
        finally:
            session.journal["editorial_history"] = copy.deepcopy(history)
            session.save()
    session.journal["editorial_history"] = copy.deepcopy(history)
    session.save()
    return {
        **current,
        "final_draft": current["draft"],
        "rounds": history,
        "editorial_history": history,
        "total_rounds": len(history),
        "final_score": review["score"],
        "accepted": review["accepted"],
        "targeted_repair_state": copy.deepcopy(session.state),
        "degradation_reasons": [],
    }


def result_projection(report):
    """Expose version-bound loop states without treating editorial advice as science."""
    import hashlib

    from litagent.quality import remaining_repair_allowances

    meta = report.get("metadata") or {}
    state = meta.get("targeted_repair_state") or {}
    if state.get("version") != review_policy.LEDGER_CONTRACT:
        return None
    remaining = remaining_repair_allowances(state)
    sha = hashlib.sha256(report.get("survey", "").encode()).hexdigest()
    reviews = [
        {
            "round": row["round"],
            "status": row["status"],
            "source_report_sha256": row["source_report_sha256"],
            "candidate_sha256": row.get("candidate_sha256"),
            "responses": {
                key: item["action"] for key, item in row.get("responses", {}).items()
            },
        }
        for row in meta.get("editorial_history", [])
    ]
    history = meta.get("evaluation_history") or {}

    def evaluated(number, evaluation, attempt=None):
        from litagent.eval.results import derive_quality

        attempt = attempt or {}
        metric = (evaluation or {}).get("faithfulness", {})
        details = metric.get("details", {})
        try:
            status = derive_quality(copy.deepcopy(evaluation))["status"]
        except (KeyError, TypeError, ValueError):
            status = "unverified"
        return {
            "round": number,
            "kind": (
                "evaluation"
                if number == 0 or evaluation
                else (
                    "repair_preflight"
                    if not attempt.get("attempted")
                    else "repair_incomplete"
                )
            ),
            "report_sha256": details.get("source_report_sha256"),
            "status": status,
            "metric_status": metric.get("status", "unverified"),
            "reason": attempt.get("reason_code"),
            "supported": details.get("supported_count"),
            "unsupported": details.get("unsupported_count"),
            "unverified": details.get("unverified_count"),
            "committed": attempt.get("committed", False),
            "promoted": attempt.get("working_draft_decision", {}).get("promote", False),
        }

    return {
        "contract": review_policy.CONTRACT,
        "report_sha256": sha,
        "allowances": {
            key: {
                "limit": limit,
                "used": limit - remaining[key],
                "remaining": remaining[key],
            }
            for key, limit in review_policy.ALLOWANCES.items()
        },
        "reviews": reviews,
        "evaluations": [evaluated(0, history.get("initial_evaluation"))]
        + [
            evaluated(i, a.get("candidate_evaluation"), a)
            for i, a in enumerate(history.get("attempts", []), 1)
        ],
        "working_report_sha256": meta.get("working_draft", {}).get("report_sha256"),
        "publishable": report.get("delivery", {}).get("publishable") is True,
    }


def public_projection(value, report_sha256):
    """Whitelist a loop summary and refuse stale version bindings for all exports."""
    import re

    def sha(value):
        return (
            value
            if isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            else None
        )

    def code(value):
        return (
            value
            if isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", value)
            else None
        )

    if (
        not isinstance(value, dict)
        or value.get("contract") != review_policy.CONTRACT
        or value.get("report_sha256") != report_sha256
        or not sha(report_sha256)
    ):
        return None
    try:
        allowances = {}
        for key, limit in review_policy.ALLOWANCES.items():
            item = value["allowances"][key]
            if (
                item["limit"] != limit
                or type(item["used"]) is not int
                or not 0 <= item["used"] <= limit
                or item["remaining"] != limit - item["used"]
            ):
                return None
            allowances[key] = {
                name: item[name] for name in ("limit", "used", "remaining")
            }
        reviews, evaluations = [], []
        for row in value["reviews"]:
            if (
                type(row["round"]) is not int
                or not 0 <= row["round"] < 4
                or not sha(row["source_report_sha256"])
            ):
                return None
            reviews.append(
                {
                    "round": row["round"],
                    "status": code(row["status"]) or "unverified",
                    "source_report_sha256": row["source_report_sha256"],
                    "candidate_sha256": sha(row.get("candidate_sha256")),
                    "responses": {
                        key: action
                        for key, action in row.get("responses", {}).items()
                        if re.fullmatch(r"t[1-9][0-9]*", key)
                        and action in {"replace", "keep", "cannot_resolve"}
                    },
                }
            )
        for row in value["evaluations"]:
            if type(row["round"]) is not int or not 0 <= row["round"] <= 12:
                return None
            evaluations.append(
                {
                    "round": row["round"],
                    "report_sha256": sha(row.get("report_sha256")),
                    "kind": (
                        row.get("kind")
                        if row.get("kind")
                        in {"evaluation", "repair_preflight", "repair_incomplete"}
                        else "evaluation"
                    ),
                    "status": (
                        row["status"]
                        if row.get("status")
                        in {"passed", "failed", "unverified", "error", "not_applicable"}
                        else "unverified"
                    ),
                    "metric_status": (
                        row.get("metric_status")
                        if row.get("metric_status")
                        in {"passed", "failed", "unverified", "error", "not_applicable"}
                        else "unverified"
                    ),
                    "reason": code(row.get("reason")),
                    **{
                        k: (
                            row.get(k)
                            if type(row.get(k)) is int and row[k] >= 0
                            else None
                        )
                        for k in ("supported", "unsupported", "unverified")
                    },
                    "committed": row.get("committed") is True,
                    "promoted": row.get("promoted") is True,
                }
            )
        return {
            "contract": review_policy.CONTRACT,
            "report_sha256": report_sha256,
            "allowances": allowances,
            "reviews": reviews,
            "evaluations": evaluations,
            "working_report_sha256": sha(value.get("working_report_sha256")),
            "publishable": value.get("publishable") is True,
        }
    except (KeyError, TypeError, ValueError):
        return None
