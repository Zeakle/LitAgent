"""Coordinate editorial drafts independently of the final Evaluation owner."""

import copy
import json

from litagent import review_policy
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import digest, repair_attempts_remaining


def writer_session(client, path, run_id, document, *, trace_hook=None):
    """Reopen the same authenticated writer and raw-reference journal."""
    from litagent.inline_citations import selected
    from litagent.writing_v2 import WriterSession

    envelope = document.get("budget_envelope")
    if review_policy.enabled(envelope) and (
        envelope["run_limits"].get("workflow_identity")
        != review_policy.identity(inline=selected(envelope))
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


PATCH_REJECTION_CONTRACT = "bound-patch-rejection-v1"


def reject_patch_candidate(session, error, retained, *, category):
    """Authenticate an unapplied conflicting patch and retain an independent draft."""
    import base64
    import hashlib

    from litagent import stage_repair
    from litagent.inline_citations import selected
    from litagent.llm.responses import (
        build_responses_request,
        decode_private_receipt,
        normalize_responses,
    )
    from litagent.safety.planned_budget import planned_backend
    from litagent.writing_v2 import WriterError, compile_document, review_payload

    if stage_repair.enabled(session.envelope):
        return stage_repair.reject_candidate(
            session, error, retained, category=category
        )

    if not selected(session.envelope) or error.code != "writer_patch_response_mismatch":
        return None
    try:
        attempt = session.state["attempts"][-1]
        bound = attempt["patch_request"]
        name = f"repair_{attempt['index']}"
        call = session.journal["calls"][name]
        owner = planned_backend(getattr(session.llm, "_cost_budget", None))
        if (
            category not in {"review_revision", "evaluation_revision"}
            or attempt["category"] != category
            or attempt["status"] != "failed"
            or attempt.get("error_code") != error.code
            or bound["contract"] != PATCH_REJECTION_CONTRACT
            or digest(bound) != attempt["patch_request_sha256"]
            or owner is None
            or owner.stop_reason
            or owner.snapshot()["usage_uncertain"]
            or any(e.get("status") != "settled" for e in owner.entries.values())
            or call["status"] != "response_received"
            or call["stage"] != "repair"
            or call["provider_status"] != "completed"
            or call["finish_reason"] != "stop"
            or call.get("provider_error")
        ):
            return None
        settlement = owner.entries[f"{session.identity}:{name}"]
        usage = call["usage"]
        amounts = {k: usage[k] for k in ("prompt_tokens", "completion_tokens")}
        if (
            settlement["stage"] != f"{category}/repair"
            or settlement["status"] != "settled"
            or settlement.get("sent") is not True
            or usage.get("usage_available") is not True
            or any(type(v) is not int or v < 0 for v in amounts.values())
            or settlement["usage_sha256"] != digest(amounts)
            or settlement["actual_tokens"] != sum(amounts.values())
        ):
            return None
        receipt = call["provider_receipt"]
        request, _ = decode_private_receipt(receipt)
        response = normalize_responses(
            base64.b64decode(receipt["response_body_base64"])
        )
        client = getattr(session.llm, "_base", session.llm)
        model = getattr(session.llm, "_model", client._model)
        extra = (
            None
            if session.thinking_mode == "provider_default"
            else {"thinking": {"type": session.thinking_mode}}
        )
        from litagent.llm.responses import RESPONSES_CONTRACT
        from litagent.llm.routing import ModelBoundClient

        args = {
            "provider_api": "responses",
            "response_format": bound["response_format"],
            "max_output_tokens": call["max_tokens"],
        }
        if isinstance(session.llm, ModelBoundClient):
            args["model"] = model
        if extra is not None:
            args["extra_body"] = extra
        identity = [
            bound["messages"],
            bound["response_format"],
            {"max_output_tokens": call["max_tokens"]},
            extra,
        ]
        if session.reasoning_effort is not None:
            args["reasoning_effort"] = session.reasoning_effort
            identity.append({"reasoning_effort": session.reasoning_effort})
        identity.append(["responses", RESPONSES_CONTRACT])
        if call["input_sha256"] != digest(identity) or settlement[
            "wire_sha256"
        ] != digest([bound["messages"], args, "responses"]):
            return None
        expected = build_responses_request(
            bound["messages"],
            model=model,
            output_limit=call["max_tokens"],
            response_format=bound["response_format"],
            temperature=client._temperature,
            extra_body=extra,
            reasoning_effort=session.reasoning_effort,
        )
        if (
            request != expected
            or receipt["status_code"] != 200
            or receipt["headers"].get("x-request-id") != call.get("request_id")
            or response["model"] != model
            or call["model"] != model
            or response["usage"] != usage
            or response["content"] != call["raw"]
            or response["provider_status"] != "completed"
            or response["provider_error"]
            or hashlib.sha256(call["raw"].encode()).hexdigest() != call["raw_sha256"]
            or digest(bound["schema"]) != call["schema_sha256"]
        ):
            return None
        parsed, _ = session.parse_wire_response(
            call["raw"], bound["schema"], {"stage": "repair"}
        )
        if digest(parsed) != call["canonical_sha256"]:
            return None
        try:
            review_policy.validate_responses(
                parsed, bound["targets"], bound["references"], diagnose=True
            )
        except WriterError as replayed:
            if (
                replayed.code != error.code
                or not replayed.diagnostics
                or replayed.diagnostics != error.diagnostics
            ):
                return None
        else:
            return None
        review_payload(retained)
        doc = retained["writer_document"]
        parent = doc["summaries"] if bound["summary_only"] else doc["body"]
        if digest(parent) != bound["body_sha256"]:
            return None
        text, bindings = compile_document(
            doc["body"],
            doc["summaries"],
            retained["fact_cards"],
            doc["aliases"],
            retained["report_language"],
            retained.get("review_methodology"),
            writing_plan=session.writing_plan,
            envelope=session.envelope,
        )
        if text != retained["draft"] or bindings != retained["writing_bindings"]:
            return None
        remaining = repair_attempts_remaining(session.state, category)
        decision = dict(
            contract=PATCH_REJECTION_CONTRACT,
            category=category,
            call=name,
            reason_code=error.code,
            diagnostics=copy.deepcopy(error.diagnostics),
            retained_report_sha256=bindings["report_sha256"],
            remaining=remaining,
            actual_tokens=sum(amounts.values()),
            action=(
                "evaluate_retained"
                if category == "review_revision"
                else "retry_evaluation_revision" if remaining else "stop"
            ),
        )
        records = session.journal.setdefault("patch_rejections", {})
        if name in records and records[name] != decision:
            return None
        records[name] = decision
        session.save()
        return decision
    except (KeyError, TypeError, ValueError, AttributeError, IndexError):
        return None


SUMMARY_REJECTION_CONTRACT = "bound-summary-rejection-v1"
SUMMARY_CITATION_ERRORS = frozenset(
    {
        "writer_inline_citation_unknown",
        "writer_inline_citation_position_invalid",
        "writer_inline_citation_without_prose",
    }
)


def summary_citation_diagnostics(session, raw, schema, units):
    """Inspect rejected bytes without turning them into an accepted response."""
    from litagent import inline_citations as citations
    from litagent.safety.planned_budget import check_response_bounds
    from litagent.writing_flat import _reference_schema, prepare
    from litagent.writing_v2 import (
        WriterError,
        _check_summary_capacity,
        _valid_text,
        check_writing_scope,
        parse_response,
    )

    wire_schema = prepare(schema)["wire_schema"]
    if session.reference_contract is not None:
        wire_schema = _reference_schema(wire_schema)
    value = parse_response(raw, wire_schema)
    canonical, diagnostics = {}, []
    for section in ("abstract", "conclusion"):
        paragraphs = []
        for ordinal, item in enumerate(value[section]):
            if item["paragraph_start"]:
                paragraphs.append({"sentences": []})
            if not paragraphs:
                raise WriterError("writer_flat_first_paragraph_required")
            _valid_text(item["text"])
            premises = list(dict.fromkeys(item["body_unit_refs"]))
            allowed = list(
                dict.fromkeys(
                    ref for uid in premises for ref in units[uid]["fact_card_refs"]
                )
            )
            path = [section, len(paragraphs) - 1, len(paragraphs[-1]["sentences"])]
            base = {
                "path": path,
                "wire_path": [section, ordinal, "text"],
                "body_unit_refs": premises,
                "allowed_card_refs": allowed,
            }
            for finding in citations.position_diagnostics(item["text"]):
                diagnostics.append(
                    {
                        **base,
                        **finding,
                        **(
                            {"in_frozen_pool": finding["reference"] in session.aliases}
                            if "reference" in finding
                            else {}
                        ),
                    }
                )
            for mark in citations.MARK.finditer(item["text"]):
                ref = mark[1]
                if ref not in allowed:
                    diagnostics.append(
                        {
                            **base,
                            "code": "writer_inline_citation_unknown",
                            "reference": ref,
                            "offset": mark.start(),
                            "in_frozen_pool": ref in session.aliases,
                            "reason": (
                                "outside_body_dependencies"
                                if ref in session.aliases
                                else "unknown_frozen_card"
                            ),
                        }
                    )
            paragraphs[-1]["sentences"].append(
                {
                    "text": item["text"],
                    "body_unit_refs": premises,
                }
            )
        canonical[section] = paragraphs
        _check_summary_capacity(
            [u["text"] for u in value[section]],
            session.packet["language"],
            envelope=session.envelope,
            section=section,
        )
    check_writing_scope(
        canonical, session.writing_plan, session.envelope, stage="summary"
    )
    check_response_bounds(canonical, envelope=session.envelope)
    return diagnostics


def reject_summary_candidate(session, error, body, summaries, retained, *, category):
    """Authenticate rejection and independently recompile the retained manuscript."""
    from litagent import stage_repair
    from litagent.writing_v2 import compile_document, patch_provenance, review_payload

    if stage_repair.enabled(session.envelope):
        return stage_repair.reject_candidate(
            session, error, retained, category=category, body=body, summaries=summaries
        )

    proof = settled_rejected_summary(session, error, body, summaries, category=category)
    if not proof:
        return None
    try:
        # Rebuild the evidence packet as well as the rendered citation bindings.
        review_payload(retained)
        doc = retained["writer_document"]
        attempt = session.state["attempts"][-1]
        provenance = attempt["provenance"]
        if provenance != patch_provenance(
            doc["body"],
            provenance["targets"],
            provenance["replacements"],
            body,
            cause="review" if category == "review_revision" else "evaluation",
        ):
            return None
        patch = session.journal["calls"][f"repair_{attempt['index']}"]
        receipt = attempt["provenance_receipt"]
        if (
            receipt["raw_sha256"] != patch["raw_sha256"]
            or receipt["raw"] != patch["raw"]
            or receipt["input_sha256"] != patch["input_sha256"]
            or receipt["canonical_sha256"] != patch["canonical_sha256"]
        ):
            return None
        text, bindings = compile_document(
            doc["body"],
            doc["summaries"],
            retained["fact_cards"],
            doc["aliases"],
            retained["report_language"],
            retained.get("review_methodology"),
            writing_plan=session.writing_plan,
            envelope=session.envelope,
        )
        if text != retained["draft"] or bindings != retained["writing_bindings"]:
            return None
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    remaining = repair_attempts_remaining(session.state, category)
    decision = {
        **proof,
        "contract": SUMMARY_REJECTION_CONTRACT,
        "action": (
            "evaluate_retained"
            if category == "review_revision"
            else "retry_evaluation_revision" if remaining else "stop"
        ),
        "retained_report_sha256": bindings["report_sha256"],
        "remaining": remaining,
    }
    decision["identity"] = digest(decision)
    records = session.journal.setdefault("summary_rejections", {})
    existing = records.get(decision["call"])
    if existing is not None and existing != decision:
        return None
    records[decision["call"]] = copy.deepcopy(decision)
    session.save()
    return decision


def settled_rejected_summary(
    session, error, body, summaries=None, *, category="review_revision"
):
    """Prove that this edit's summary was received and charged before rejection."""
    import base64
    import hashlib

    from litagent.inline_citations import references, selected
    from litagent.llm.responses import (
        build_responses_request,
        decode_private_receipt,
        normalize_responses,
    )
    from litagent.llm.routing import ModelBoundClient
    from litagent.safety.planned_budget import planned_backend
    from litagent.writing_v2 import (
        body_units,
        classify_failure,
        summary_request_input,
    )

    attempts = session.state["attempts"]
    compile_failure = summaries is not None
    if (
        not selected(session.envelope)
        or session.wire_contract != "flat-writer-v1"
        or error.family != "protocol"
        or category not in {"review_revision", "evaluation_revision"}
        or error.code not in SUMMARY_CITATION_ERRORS
        or (compile_failure and error.code != "writer_inline_citation_unknown")
        or not attempts
    ):
        return False
    attempt = attempts[-1]
    if (
        attempt.get("index") != len(attempts) - 1
        or attempt.get("category") != category
        or attempt.get("status") != "completed"
        or attempt.get("candidate_sha256") != digest(body)
    ):
        return False
    name = f"summary_repair_{len(attempts)}"
    classified = classify_failure(error)
    call = session.journal["calls"].get(name, {})
    failure = session.journal.get("failure")
    if compile_failure:
        if failure is not None or call.get("status") != "response_received":
            return False
    elif (
        failure
        != {**classified, "stage": "summary", "call": name, "response_received": True}
        or call.get("failure") != classified
        or call.get("status") != "failed_no_replay"
    ):
        return False
    if (
        call.get("stage") != "summary"
        or call.get("provider_api") != "responses"
        or call.get("provider_status") != "completed"
        or call.get("finish_reason") != "stop"
        or call.get("provider_error")
    ):
        return False
    owner = planned_backend(getattr(session.llm, "_cost_budget", None))
    if (
        owner is None
        or owner.stop_reason
        or owner.snapshot()["usage_uncertain"]
        or any(e.get("status") != "settled" for e in owner.entries.values())
    ):
        return False
    settlement = owner.entries.get(f"{session.identity}:{name}", {})
    usage = call.get("usage")
    if not isinstance(usage, dict):
        return False
    amounts = {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens")}
    if (
        settlement.get("stage") != f"{category}/summary"
        or settlement.get("status") != "settled"
        or settlement.get("sent") is not True
        or usage.get("usage_available") is not True
        or any(type(value) is not int or value < 0 for value in amounts.values())
        or settlement.get("usage_sha256") != digest(amounts)
        or settlement.get("actual_tokens") != sum(amounts.values())
    ):
        return False
    try:
        receipt = call["provider_receipt"]
        request, _ = decode_private_receipt(receipt)
        response = normalize_responses(
            base64.b64decode(receipt["response_body_base64"])
        )
        units = body_units(body)
        system, payload, schema = summary_request_input(
            session.packet, units, session.envelope
        )
        system, payload = session.planned_input(system, payload)
        messages = session.messages(system, payload, schema)
        native = session.response_format("summary", schema)
        args = {
            "provider_api": "responses",
            "response_format": native,
            "max_output_tokens": call["max_tokens"],
        }
        bound = isinstance(session.llm, ModelBoundClient)
        client = session.llm._base if bound else session.llm
        if bound:
            args["model"] = session.llm._model
        if session.thinking_mode != "provider_default":
            args["extra_body"] = {"thinking": {"type": session.thinking_mode}}
        if session.reasoning_effort is not None:
            args["reasoning_effort"] = session.reasoning_effort
        expected = build_responses_request(
            messages,
            model=args.get("model", client._model),
            output_limit=call["max_tokens"],
            response_format=native,
            temperature=client._temperature,
            extra_body=args.get("extra_body"),
            reasoning_effort=session.reasoning_effort,
        )
        if not (
            receipt["status_code"] == 200
            and receipt["headers"].get("x-request-id") == call.get("request_id")
            and request == expected
            and settlement.get("wire_sha256") == digest([messages, args, "responses"])
            and response["model"] == request["model"] == call["model"]
            and response["provider_status"] == "completed"
            and response["finish_reason"] == "stop"
            and response["provider_error"] is None
            and response["usage"] == usage
            and response["content"] == call["raw"]
            and hashlib.sha256(call["raw"].encode()).hexdigest() == call["raw_sha256"]
        ):
            return False
        diagnostics = summary_citation_diagnostics(session, call["raw"], schema, units)
        if not diagnostics:
            return False
        if not compile_failure:
            from litagent.writing_v2 import WriterError

            try:
                session.parse_wire_response(call["raw"], schema, {"stage": "summary"})
            except WriterError as replayed:
                if classify_failure(replayed) != classified:
                    return False
            else:
                return False
        else:
            from litagent.writing_v2 import WriterError

            parsed, _ = session.parse_wire_response(
                call["raw"], schema, {"stage": "summary"}
            )
            if parsed != summaries or digest(parsed) != call.get("canonical_sha256"):
                return False
            (diagnostic,) = error.diagnostics
            section, pi, si = diagnostic["path"]
            if section not in ("abstract", "conclusion") or any(
                type(index) is not int or index < 0 for index in (pi, si)
            ):
                return False
            unit = parsed[section][pi]["sentences"][si]
            allowed = {
                ref
                for uid in unit["body_unit_refs"]
                for ref in units[uid]["fact_card_refs"]
            }
            try:
                references(unit["text"], allowed)
            except WriterError as replayed:
                actual = [
                    {**d, "path": diagnostic["path"]} for d in replayed.diagnostics
                ]
                if replayed.code != error.code or actual != list(error.diagnostics):
                    return False
            else:
                return False
        return {
            "category": category,
            "call": name,
            "reason_code": error.code,
            "candidate_body_sha256": digest(body),
            "raw_sha256": call["raw_sha256"],
            "request_key": f"{session.identity}:{name}",
            "wire_sha256": settlement["wire_sha256"],
            "usage_sha256": settlement["usage_sha256"],
            "actual_tokens": settlement["actual_tokens"],
            "diagnostics": diagnostics,
        }

    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return False


async def editorial_cycle(worker, task, source):
    """Review each valid draft, retaining disputes for independent Evaluation."""
    from litagent import stage_repair
    from litagent.writing_v2 import (
        BODY,
        WriterError,
        _split_summary_repair_guidance,
        classify_failure,
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
    cached_review = None
    for round_number in range(4):
        reuse_review = (
            stage_repair.enabled(session.envelope) and cached_review is not None
        )
        async with worker._sub_span(
            task.task_id,
            "reviewer",
            round_number + 1,
            input_data={
                **task.input_data,
                "review_round": round_number,
                "upstream_results": {"synthesis": current},
                "reused": reuse_review,
            },
        ) as review_span:
            if not reuse_review:
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
                review = await worker._reviewer.execute(review_task)
                cached_review = review
            else:
                review = cached_review
            review_span["output"] = {**review, "reused": reuse_review}
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
            row["status"] = (
                "review_issues_unbound"
                if review.get("rejected_issues")
                else "review_accepted"
            )
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
        async with worker._sub_span(
            task.task_id,
            "synthesis",
            round_number + 2,
            input_data={
                "query": task.input_data.get("query", ""),
                "previous_draft": current["draft"],
                "source_report_sha256": current["writing_bindings"]["report_sha256"],
                "review": review,
                "targets": targets,
                "evidence_selection": current.get("evidence_selection"),
            },
        ) as revision_span:
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
                            name = "summary_repair_" + str(
                                len(session.state["attempts"])
                            )
                            session.journal.setdefault(
                                "summary_regeneration_guidance", {}
                            )[name] = summary_guidance
                            session.save()
                        summaries = None
                    try:
                        if not summary_only:
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
                        revision_span["output"] = {
                            "draft": draft,
                            "writing_bindings": bindings,
                            "promoted": False,
                        }
                    except WriterError as exc:
                        decision = (
                            None
                            if summary_only
                            else reject_summary_candidate(
                                session,
                                exc,
                                body,
                                summaries,
                                current,
                                category="review_revision",
                            )
                        )
                        if decision is None:
                            raise
                        # Both calls and the completed edit stay in the ledger.
                        failure = session.journal.pop("failure", None)
                        if failure is None:
                            failure = {
                                **classify_failure(exc),
                                "stage": "summary_compilation",
                                "call": f"summary_repair_{len(session.state['attempts'])}",
                                "response_received": True,
                            }
                        row.update(
                            status="candidate_rejected",
                            reason_code=exc.code,
                            summary_failure=failure,
                            summary_rejection=decision,
                        )
                        if (
                            stage_repair.enabled(session.envelope)
                            and decision["action"] != "stop"
                        ):
                            continue
                        break
                if bindings["report_sha256"] in seen:
                    row["status"] = "repeated_candidate"
                    break
                row.update(
                    status="revised_unverified",
                    candidate_sha256=bindings["report_sha256"],
                )
                cached_review = None
                if stage_repair.enabled(session.envelope):
                    session.journal.pop("pending_candidate", None)
                    session.journal.pop("candidate_feedback", None)
                seen.add(bindings["report_sha256"])
                current.update(
                    draft=draft,
                    writing_bindings=bindings,
                    writer_document={**doc, "body": body, "summaries": summaries},
                )
                revision_span["output"]["promoted"] = True
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
                decision = reject_patch_candidate(
                    session, exc, current, category="review_revision"
                )
                if decision is not None:
                    row["patch_rejection"] = decision
                    row["patch_failure"] = copy.deepcopy(session.journal.get("failure"))
                    session.journal.setdefault("rejected_failures", {})[
                        decision["call"]
                    ] = session.journal.pop("failure", {})
                    if (
                        stage_repair.enabled(session.envelope)
                        and decision["action"] != "stop"
                    ):
                        continue
                    break
                if stage_repair.enabled(session.envelope) and exc.code not in {
                    "writer_repair_declined",
                    "writer_repair_no_progress",
                }:
                    raise
                if exc.family != "quality":
                    if not _settled_rejected_editorial_patch(session, exc):
                        raise
                    # Preserve the failed call and spent allowance, but do not
                    # present this discarded edit as a failure of the retained draft.
                    session.journal.pop("failure")
                # An editorial disagreement or unusable edit cannot erase a valid draft.
                break
            finally:
                revision_span.setdefault("output", {}).update(
                    status=row["status"],
                    reason_code=row.get("reason_code"),
                )
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


def summary_rejection_projection(value):
    """Whitelist local rejection facts without exposing receipts or report text."""
    import re

    from litagent.stage_repair import public_rejection

    if (projected := public_rejection(value)) is not None:
        return projected

    if (
        not isinstance(value, dict)
        or value.get("contract") != SUMMARY_REJECTION_CONTRACT
    ):
        return None
    try:
        category = value["category"]
        if category not in {"review_revision", "evaluation_revision"}:
            return None
        if value["reason_code"] not in SUMMARY_CITATION_ERRORS:
            return None
        if not re.fullmatch(r"[a-f0-9]{64}", value["retained_report_sha256"]):
            return None
        if not re.fullmatch(r"summary_repair_[1-9][0-9]*", value["call"]):
            return None
        remaining = value["remaining"]
        if type(remaining) is not int or not 0 <= remaining <= 3:
            return None
        actual = value["actual_tokens"]
        if type(actual) is not int or actual < 0:
            return None
        rows = []
        for row in value["diagnostics"]:
            path = row["path"]
            if (
                not isinstance(path, list)
                or len(path) != 3
                or path[0] not in {"abstract", "conclusion"}
                or any(type(n) is not int or n < 0 for n in path[1:])
                or row["code"] not in SUMMARY_CITATION_ERRORS
            ):
                return None
            clean = {"path": list(path), "code": row["code"]}
            if "offset" in row:
                if type(row["offset"]) is not int or row["offset"] < 0:
                    return None
                clean["offset"] = row["offset"]
            if "reference" in row:
                if not re.fullmatch(r"c[1-9][0-9]*", row["reference"]):
                    return None
                clean["reference"] = row["reference"]
                if type(row.get("in_frozen_pool")) is not bool:
                    return None
                clean["in_frozen_pool"] = row["in_frozen_pool"]
                if row["code"] == "writer_inline_citation_unknown":
                    clean["reason"] = (
                        "outside_body_dependencies"
                        if row["in_frozen_pool"]
                        else "unknown_frozen_card"
                    )
            rows.append(clean)
        if not rows:
            return None
        action = (
            "evaluate_retained"
            if category == "review_revision"
            else "retry_evaluation_revision" if remaining else "stop"
        )
        return {
            "contract": SUMMARY_REJECTION_CONTRACT,
            "category": category,
            "action": action,
            "reason_code": value["reason_code"],
            "retained_report_sha256": value["retained_report_sha256"],
            "call": value["call"],
            "remaining": remaining,
            "actual_tokens": actual,
            "diagnostics": rows,
        }
    except (KeyError, TypeError, ValueError):
        return None


def patch_rejection_projection(value):
    """Expose only authenticated local conflict coordinates and resource facts."""
    import re

    from litagent.stage_repair import public_rejection

    if (projected := public_rejection(value)) is not None:
        return projected

    try:
        if value["contract"] != PATCH_REJECTION_CONTRACT:
            return None
        category = value["category"]
        remaining = value["remaining"]
        if (
            category not in {"review_revision", "evaluation_revision"}
            or type(remaining) is not int
            or not 0 <= remaining <= 3
            or type(value["actual_tokens"]) is not int
            or value["actual_tokens"] < 0
            or not re.fullmatch(r"[0-9a-f]{64}", value["retained_report_sha256"])
            or not re.fullmatch(r"repair_(?:0|[1-9][0-9]*)", value["call"])
        ):
            return None
        rows = []
        for row in value["diagnostics"]:
            if (
                not re.fullmatch(r"t[1-9][0-9]*", row["target_key"])
                or row["action"] not in {"replace", "keep", "cannot_resolve"}
                or row["reason"]
                not in {"patch_for_non_replace", "missing_replace_patch"}
                or not isinstance(row["target_id"], str)
                or len(row["target_id"]) > 300
            ):
                return None
            rows.append(
                {
                    k: row[k]
                    for k in (
                        "target_key",
                        "target_id",
                        "action",
                        "reason",
                        "patch_index",
                    )
                }
            )
        return {
            "contract": PATCH_REJECTION_CONTRACT,
            "category": category,
            "action": (
                "evaluate_retained"
                if category == "review_revision"
                else "retry_evaluation_revision" if remaining else "stop"
            ),
            "reason_code": "writer_patch_response_mismatch",
            "diagnostics": rows,
            **{
                k: value[k]
                for k in (
                    "call",
                    "remaining",
                    "actual_tokens",
                    "retained_report_sha256",
                )
            },
        }
    except (KeyError, TypeError, ValueError):
        return None


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
            "patch_rejection": patch_rejection_projection(row.get("patch_rejection")),
            "summary_rejection": summary_rejection_projection(
                row.get("summary_rejection")
            ),
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
            "patch_rejection": patch_rejection_projection(
                attempt.get("patch_rejection")
            ),
            "summary_rejection": summary_rejection_projection(
                attempt.get("summary_rejection")
            ),
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
                    "patch_rejection": patch_rejection_projection(
                        row.get("patch_rejection")
                    ),
                    "summary_rejection": summary_rejection_projection(
                        row.get("summary_rejection")
                    ),
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
                    "patch_rejection": patch_rejection_projection(
                        row.get("patch_rejection")
                    ),
                    "summary_rejection": summary_rejection_projection(
                        row.get("summary_rejection")
                    ),
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
