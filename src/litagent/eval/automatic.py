"""Freeze automatic review inputs and enforce independent semantic delivery gates."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path

from litagent.audit_context import SurveyAuditContext
from litagent.benchmark.survey_judge import _stable_sha256
from litagent.benchmark.survey_models import (
    AnchoredJudgeClaim,
    JudgeClaimAssessment,
    JudgeInventorySnapshot,
    JudgeUsage,
    SurveyJudgeMetrics,
)
from litagent.evidence import extract_evidence_refs_ordered
from litagent.quality import digest, strict_acceptance, validate_fact_snapshot
from litagent.report_render import build_report_source_units
from litagent.workbench import verification_enabled

VERSION = "automatic-acceptance-v2"
JUDGE_VERSION = "frozen-original-audit-v1"


def frozen_judge_fingerprint() -> str:
    """Bind both prompts and the host verification rules to automatic checkpoints."""
    from litagent.eval import scoped
    from litagent.eval.batched import GLOBAL_PROMPT

    return _stable_sha256(
        {
            "version": JUDGE_VERSION,
            "assessment": scoped.STRICT_ASSESSMENT_PROMPT,
            "support_contract": scoped.STRICT_BINDING_CONTRACT,
            "support_host_sha256": hashlib.sha256(
                Path(scoped.__file__).read_bytes()
            ).hexdigest(),
            "global": GLOBAL_PROMPT,
            "host": Path(__file__).read_text(encoding="utf-8"),
        }
    )


def freeze_automatic_inputs(
    report: dict, ledger: Mapping, *, run_id: str, revision: int = 1
) -> dict:
    """Bind complete original sources, evaluated assertions and host records to one SHA."""
    from litagent.eval.scoped import LOCAL_NUMERIC_CONTRACT

    survey = report["survey"]
    metadata = report.get("metadata", {})
    details = report.get("evaluation", {}).get("faithfulness", {}).get("details", {})
    evaluation = report.get("evaluation", {})
    if any(
        evaluation.get(metric, {}).get("passed") is not True
        or evaluation.get(metric, {}).get("skipped")
        for metric in ("citation_accuracy", "faithfulness", "internal_consistency")
    ):
        raise ValueError("automatic_evaluation_incomplete")
    if (
        details.get("quality_policy") not in {"strict_auto_v1", "strict_auto_v2"}
        or strict_acceptance(details)["status"] != "passed"
    ):
        raise ValueError("automatic_evaluation_not_passed")
    if (
        metadata.get("body_quality") == "analysis_v1"
        and details.get("body_analysis_contract") != "body-analysis-v1"
    ):
        raise ValueError("automatic_analysis_not_verified")
    cards = metadata.get("fact_cards", {})
    policy = details["quality_policy"]
    if metadata.get("writing_bindings", {}).get("scope_notice_contract"):
        from litagent.writing_passages import (
            SCOPE_NOTICE_CONTRACT,
            validate_scope_notice_bindings,
        )

        validate_scope_notice_bindings(
            survey, metadata["writing_bindings"], metadata.get("report_language", "en")
        )
        if details.get("scope_notice_contract") != SCOPE_NOTICE_CONTRACT:
            raise ValueError("automatic_scope_notice_contract_mismatch")
    if policy == "strict_auto_v2":
        from litagent.extraction_verification import verified_extraction_ledger
        from litagent.quality import build_fact_cards

        try:
            if verification_enabled(metadata, "jev"):
                ledger = verified_extraction_ledger(
                    metadata.get("extraction_verification") or {}, ledger, run_id=run_id
                )
            else:
                # Keep the selected original evidence, including uncited cards.
                selected_ids = {card["evidence_id"] for card in cards.values()}
                if not selected_ids or not selected_ids.issubset(ledger):
                    raise ValueError("selected evidence missing")
                ledger = {
                    eid: item for eid, item in ledger.items() if eid in selected_ids
                }
            if cards != build_fact_cards(ledger):
                raise ValueError("incomplete accepted card snapshot")
        except ValueError as exc:
            reason = (
                "automatic_extraction_not_passed"
                if verification_enabled(metadata, "jev")
                else "automatic_evidence_snapshot_invalid"
            )
            raise ValueError(reason) from exc
    validate_fact_snapshot(cards, ledger, metadata.get("evidence_snapshot_sha256"))
    units = {unit.unit_id: unit for unit in build_report_source_units(survey)}
    from litagent.eval.scoped import build_source_fragments

    fragments = {
        fragment["source_fragment_id"]: (unit, fragment)
        for unit in units.values()
        for fragment in build_source_fragments(unit)
    }
    assertions = details.get("assertions", [])
    ids = [c["claim_id"] for c in assertions]
    if not ids or len(ids) != len(set(ids)) or len(ids) > 128:
        raise ValueError("automatic_inventory_invalid")
    science = []
    for claim in assertions:
        unit = units.get(claim["source_unit_id"])
        start, end = claim["source_start"], claim["source_end"]
        fragment = fragments.get(claim.get("source_fragment_id"))
        if (
            unit is None
            or fragment is None
            or fragment[0].unit_id != unit.unit_id
            or fragment[1]["source_quote"] != claim["source_quote"]
            or unit.start + fragment[1]["start"] != start
            or not unit.start <= start < end <= unit.end
            or survey[start:end] != claim["source_quote"]
            or extract_evidence_refs_ordered(claim["source_quote"])
            != claim["evidence_ids"]
            or claim.get("local_error")
            or claim.get("kind")
            not in {"scientific", "runtime_method", "report_structure"}
        ):
            raise ValueError("automatic_source_fidelity_invalid")
        if claim["kind"] == "scientific":
            if not claim["evidence_ids"] or any(
                not isinstance(ledger.get(eid, {}).get("supporting_text"), str)
                or not ledger[eid]["supporting_text"].strip()
                for eid in claim["evidence_ids"]
            ):
                raise ValueError("automatic_original_support_missing")
            science.append(
                AnchoredJudgeClaim(
                    **{
                        key: claim[key]
                        for key in (
                            "claim_id",
                            "claim_text",
                            "source_unit_id",
                            "source_quote",
                            "source_start",
                            "source_end",
                            "evidence_ids",
                        )
                    }
                )
            )
    inventory = JudgeInventorySnapshot(
        survey_sha256="sha256:" + hashlib.sha256(survey.encode()).hexdigest(),
        source_units_sha256=_stable_sha256([u.__dict__ for u in units.values()]),
        ledger_sha256=_stable_sha256(dict(ledger)),
        inventory_sha256=_stable_sha256([c.model_dump(mode="json") for c in science]),
        claims=tuple(science),
    )
    context = SurveyAuditContext(
        query=report.get("query") or metadata.get("query") or "",
        report_language=metadata.get("report_language", "zh-CN"),
        review_structure=metadata.get("review_structure", "academic_v2"),
        review_methodology=metadata.get("review_methodology", {}),
    )
    return {
        "source_run_id": run_id,
        "source_report_revision": revision,
        "source_report_sha256": inventory.survey_sha256.removeprefix("sha256:"),
        "context": context.model_dump(mode="json"),
        "report": report,
        "ledger": dict(ledger),
        "artifact": {"run_id": run_id, "quality_policy": policy},
        "frozen_inventory": {
            **(
                {
                    "scope_notice_contract": details["scope_notice_contract"],
                    "scope_notice_bindings": metadata["writing_bindings"][
                        "scope_notices"
                    ],
                }
                if details.get("scope_notice_contract")
                else {}
            ),
            "version": JUDGE_VERSION,
            **(
                {"local_numeric_contract": LOCAL_NUMERIC_CONTRACT}
                if policy == "strict_auto_v2"
                else {}
            ),
            "inventory": inventory.model_dump(mode="json"),
            "assertions": assertions,
            "coverage_requirements": details["coverage_requirements"],
            "host_validation": {
                key: details.get(key)
                for key in (
                    "structure_checks",
                    "method_checks",
                    "language_checks",
                )
            },
        },
    }


async def evaluate_frozen_judge(
    judge,
    *,
    case,
    survey,
    ledger,
    frozen_inventory,
    checkpoint,
    save_checkpoint,
    before_call,
    context_max_tokens,
    total_budget_tokens,
):
    """Re-audit a prevalidated inventory and the whole report in at most two calls."""
    from litagent.eval.batched import (
        GLOBAL_CONSISTENCY_CONTRACT,
        GlobalReview,
        global_prompt,
        global_review_type,
        valid_limitations,
    )
    from litagent.eval.scoped import (
        LOCAL_NUMERIC_CONTRACT,
    )
    from litagent.eval.scoped import STRICT_ASSESSMENT_PROMPT as ASSESSMENT_PROMPT
    from litagent.eval.scoped import (
        Assessment,
        _require_ids,
        _scientific_materials,
        build_support_passages,
        required_support_aspects,
        support_passage_table,
        validate_support_basis,
    )
    from litagent.writing_passages import scope_notice_catalog

    state = checkpoint if checkpoint is not None else {}
    save = save_checkpoint or (lambda _: None)
    start_time = time.monotonic()
    usage = JudgeUsage(usage_complete=True)
    inventory = JudgeInventorySnapshot.model_validate(frozen_inventory["inventory"])
    binding = _stable_sha256(
        {
            "survey": survey,
            "ledger": dict(ledger),
            "frozen": frozen_inventory,
            "config": judge._config.fingerprint,
            "prompt": frozen_judge_fingerprint(),
        }
    )
    if state.get("binding", binding) != binding:
        raise ValueError("automatic_judge_identity_conflict")
    state["binding"] = binding
    state.setdefault("calls", [])
    if state["calls"]:
        raise ValueError("automatic_judge_restart_no_replay")
    if inventory.inventory_sha256 != _stable_sha256(
        [c.model_dump(mode="json") for c in inventory.claims]
    ):
        raise ValueError("automatic_judge_inventory_hash_mismatch")
    if inventory.survey_sha256 != "sha256:" + hashlib.sha256(
        survey.encode()
    ).hexdigest() or inventory.ledger_sha256 != _stable_sha256(dict(ledger)):
        raise ValueError("automatic_judge_input_mismatch")
    units = build_report_source_units(survey)
    assertions = frozen_inventory["assertions"]
    science = [c for c in assertions if c["kind"] == "scientific"]
    _require_ids(
        [c["claim_id"] for c in science],
        [c.claim_id for c in inventory.claims],
        "automatic_judge_inventory_invalid",
    )
    aliases = {eid: f"e{index}" for index, eid in enumerate(ledger, 1)}
    # Only original supports are presented; compressed extraction text is never evidence.
    original_ledger = {
        eid: {**value, "text": value.get("supporting_text", "")}
        for eid, value in ledger.items()
    }
    refs = list(dict.fromkeys(eid for c in science for eid in c["evidence_ids"]))
    materials, supports = _scientific_materials(
        original_ledger, refs, original_only=True
    )
    passages = build_support_passages(
        {eid: original_ledger[eid] for eid in refs},
        {eid: aliases[eid] for eid in refs},
        original_only=True,
    )
    science_units = {c["source_unit_id"] for c in science}
    payload = {
        "source_units": [
            {"source_unit_id": u.unit_id, "text": u.text}
            for u in units
            if u.unit_id in science_units
        ],
        "assertions": [
            {
                **c,
                "evidence": [
                    {"evidence_alias": aliases[eid], "material_ref": aliases[eid]}
                    for eid in c["evidence_ids"]
                ],
                "required_support_aspects": required_support_aspects(c["claim_text"]),
            }
            for c in science
        ],
        "cited_materials": {aliases[eid]: value for eid, value in materials.items()},
        "original_supports": supports,
        "support_passages": support_passage_table(passages, aliases, materials),
        "schema": Assessment.model_json_schema(),
    }

    async def call(prompt, value, phase):
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps(value, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        reserve = (
            len(json.dumps(messages, ensure_ascii=False).encode())
            + 1024
            + judge._config.max_tokens
        )
        if (
            reserve > int(context_max_tokens * 0.9)
            or usage.total_tokens + reserve > total_budget_tokens
        ):
            raise ValueError("automatic_judge_total_budget_exceeded")
        if len(state["calls"]) >= 2 or (before_call and not before_call()):
            raise ValueError("automatic_judge_request_budget_exceeded")
        entry = {
            "stage": phase,
            "status": "started_no_replay",
            "reserved_tokens": reserve,
            "usage_complete": False,
        }
        state["calls"].append(entry)
        save(state)
        usage.call_count += 1
        kwargs = {
            "max_tokens": judge._config.max_tokens,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "transport_max_attempts": 1,
        }
        if judge._config.thinking_mode != "provider_default":
            kwargs["extra_body"] = {"thinking": {"type": judge._config.thinking_mode}}
        response = await asyncio.wait_for(
            judge._llm.chat(messages, **kwargs), judge._config.timeout_seconds
        )
        raw_usage = response.usage or {}
        if raw_usage.get("usage_available") is False or not all(
            isinstance(raw_usage.get(k), int) and raw_usage[k] >= 0
            for k in ("prompt_tokens", "completion_tokens")
        ):
            usage.usage_complete = False
            usage.missing_usage_calls += 1
            raise ValueError("automatic_judge_usage_unknown")
        usage.prompt_tokens += raw_usage["prompt_tokens"]
        usage.completion_tokens += raw_usage["completion_tokens"]
        usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
        entry.update(
            status="completed",
            usage_complete=True,
            usage=raw_usage,
            actual_model=getattr(response, "model", None),
            response=response.content,
        )
        save(state)
        if usage.total_tokens > total_budget_tokens:
            raise ValueError("automatic_judge_total_budget_exceeded")
        return response.content

    try:
        assessed = Assessment.model_validate_json(
            await call(ASSESSMENT_PROMPT, payload, "science")
        )
        _require_ids(
            [d.claim_id for d in assessed.decisions],
            [c["claim_id"] for c in science],
            "automatic_judge_decision_coverage_invalid",
        )
        _require_ids(
            [u.source_unit_id for u in assessed.units],
            list(science_units),
            "automatic_judge_unit_coverage_invalid",
        )
        decisions = {d.claim_id: d for d in assessed.decisions}
        items = []
        for claim in science:
            decision = decisions[claim["claim_id"]]
            proof = validate_support_basis(
                claim,
                decision,
                original_ledger,
                aliases,
                passages=passages,
                original_only=True,
                local_numeric_bindings=frozen_inventory.get("local_numeric_contract")
                == LOCAL_NUMERIC_CONTRACT,
            )
            supported = (
                decision.status == "supported"
                and decision.source_faithful
                and decision.classification_valid
                and decision.atomic
                and decision.correct_kind == "scientific"
                and proof["status"] == "passed"
            )
            items.append(
                JudgeClaimAssessment(
                    claim_id=claim["claim_id"],
                    claim_text=claim["claim_text"],
                    source_faithful=decision.source_faithful,
                    supported=supported,
                    evidence_ids=claim["evidence_ids"],
                    explanation=decision.explanation
                    + (
                        ""
                        if supported
                        else " / " + proof.get("issue", "semantic_failure")
                    ),
                )
            )
        full_payload = {
            **(
                {
                    "host_scope_notice_catalog": scope_notice_catalog(
                        case.report_language
                    ),
                    "host_scope_notice_bindings": frozen_inventory.get(
                        "scope_notice_bindings", {}
                    ),
                }
                if frozen_inventory.get("scope_notice_contract")
                else {}
            ),
            **(
                {"global_consistency_contract": GLOBAL_CONSISTENCY_CONTRACT}
                if frozen_inventory.get("local_numeric_contract")
                == LOCAL_NUMERIC_CONTRACT
                else {}
            ),
            "source_units": [
                {"source_unit_id": u.unit_id, "text": u.text} for u in units
            ],
            "assertions": assertions,
            "coverage_requirements": frozen_inventory["coverage_requirements"],
            "host_runtime_record": case.review_methodology,
            "host_validation": frozen_inventory["host_validation"],
            "schema": global_review_type(
                {
                    "host_scope_notice_catalog": bool(
                        frozen_inventory.get("scope_notice_contract")
                    )
                }
            ).model_json_schema(),
        }
        global_result = global_review_type(full_payload).model_validate_json(
            await call(
                global_prompt(full_payload)
                + "\nAlso verify runtime-method statements against host_runtime_record; scientific statements are never exempt because of their section or wording.",
                full_payload,
                "full_report",
            )
        )
        _require_ids(
            [u.source_unit_id for u in global_result.units],
            [u.unit_id for u in units],
            "automatic_judge_full_unit_coverage_invalid",
        )
        _require_ids(
            [r.requirement_id for r in global_result.coverage],
            [r["requirement_id"] for r in frozen_inventory["coverage_requirements"]],
            "automatic_judge_core_coverage_invalid",
        )
        good = {item.claim_id for item in items if item.supported}
        passed = (
            valid_limitations(global_result, assertions, good)
            and all(u.complete for u in assessed.units + global_result.units)
            and not assessed.conflicts
            and not global_result.conflicts
            and not global_result.issues
            and all(
                r.answered and r.claim_ids and set(r.claim_ids).issubset(good)
                for r in global_result.coverage
            )
        )
        state["automatic_full_report_check"] = {
            "status": "passed" if passed else "failed",
            **global_result.model_dump(mode="json"),
        }
        save(state)
        supported_count = sum(item.supported for item in items)
        metrics = SurveyJudgeMetrics(
            status="completed",
            factual_claim_count=len(items),
            faithfulness=supported_count / len(items),
            unsupported_claim_rate=1 - supported_count / len(items),
            contradiction_rate=(
                0 if not global_result.conflicts and not assessed.conflicts else 1
            ),
            claim_assessments=items,
            inventory_snapshot=inventory,
            reason_codes=[] if passed else ["automatic_judge_full_report_failed"],
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        # Persist diagnostic identifiers; raw provider exceptions remain outside public output.
        reason = (
            str(exc)
            if str(exc).startswith("automatic_")
            else "automatic_judge_contract_invalid"
        )
        metrics = SurveyJudgeMetrics(status="failed", reason_codes=[reason])
        if any(c.get("usage_complete") is not True for c in state["calls"]):
            usage.usage_complete = False
    usage.elapsed_ms = int((time.monotonic() - start_time) * 1000)
    return metrics, usage


def automatic_acceptance(
    snapshot: dict, evaluation: dict, audit: dict | None = None
) -> dict:
    """Require exact complete denominators and all three semantic gates without averaging."""
    failures = []
    policy = (
        snapshot.get("report", {})
        .get("metadata", {})
        .get("quality_policy", "strict_auto_v1")
    )
    upfront = policy == "strict_auto_v2"
    metadata = snapshot.get("report", {}).get("metadata", {})
    optional = upfront and metadata.get("verification") is not None
    judge_required = not optional or verification_enabled(metadata, "judge")
    jev_required = not optional or verification_enabled(metadata, "jev")
    if not judge_required:
        audit = None
    details = evaluation.get("faithfulness", {}).get("details", {})
    if strict_acceptance(details)["status"] != "passed" or any(
        evaluation.get(metric, {}).get("passed") is not True
        or evaluation.get(metric, {}).get("skipped")
        for metric in ("citation_accuracy", "faithfulness", "internal_consistency")
    ):
        failures.append("evaluation_not_passed")
    frozen = snapshot.get("frozen_inventory", {})
    inventory = frozen.get("inventory", {})
    claims = inventory.get("claims", [])
    if optional and not claims:
        failures.append("automatic_inventory_missing")
    if not audit:
        audit = {}
        if judge_required:
            failures.append("judge_not_run")
        if not upfront:
            failures.append("jev_not_run")
    for key in ("source_run_id", "source_report_revision", "source_report_sha256"):
        if audit and audit.get(key) != snapshot.get(key):
            failures.append("audit_binding_mismatch")
    from litagent.report_audit import digest as compact_digest

    if audit and (
        audit.get("ledger_sha256") != compact_digest(snapshot.get("ledger", {}))
        or audit.get("source_snapshot_sha256") != compact_digest(snapshot)
    ):
        failures.append("audit_binding_mismatch")
    judge = audit.get("judge") or {}
    assessed = judge.get("claim_assessments", [])
    if (
        not claims
        or judge.get("status") != "completed"
        or judge.get("inventory_snapshot") != inventory
        or len(assessed) != len(claims)
        or {c.get("claim_id") for c in assessed} != {c["claim_id"] for c in claims}
        or any(
            c.get("supported") is not True or c.get("source_faithful") is not True
            for c in assessed
        )
        or judge.get("faithfulness") != 1
        or judge.get("contradiction_rate") != 0
        or audit.get("judge_full_report_check", {}).get("status") != "passed"
        or audit.get("judge_full_report_check", {}).get("issues")
        or audit.get("judge_full_report_check", {}).get("conflicts")
        or (audit.get("judge_usage") or {}).get("usage_complete") is not True
    ):
        if audit:
            failures.append("judge_not_passed")
    jev = audit.get("jev") or {}
    expected = {f"g:{c['claim_id']}": ("supported", None) for c in claims}
    for claim in claims:
        for index, eid in enumerate(claim["evidence_ids"]):
            expected[f"m:{claim['claim_id']}:{index}"] = ("contributes", eid)
    decisions = jev.get("decisions", [])
    if (
        jev.get("status") != "completed"
        or jev.get("actual_model") != "jev-1.13.0"
        or jev.get("usage", {}).get("usage_complete") is not True
        or jev.get("inventory_sha256") != inventory.get("inventory_sha256")
        or jev.get("ledger_sha256") != inventory.get("ledger_sha256")
        or jev.get("source_report_sha256") != snapshot.get("source_report_sha256")
        or len(jev.get("inputs", [])) != len(claims)
        or {i.get("claim_id") for i in jev.get("inputs", [])}
        != {c["claim_id"] for c in claims}
        or len(decisions) != len(expected)
        or {d.get("question_id") for d in decisions} != set(expected)
        or any(
            d.get("choice") != expected.get(d.get("question_id"), (None, None))[0]
            or d.get("evidence_id")
            != expected.get(d.get("question_id"), (None, None))[1]
            or not isinstance(d.get("confidence"), (int, float))
            or not math.isfinite(d.get("confidence", 0))
            or d.get("confidence", 0) < 0.9
            or d.get("uncertain") is not False
            or d.get("status") != "valid"
            for d in decisions
        )
        or any(i.get("status") != "ready" for i in jev.get("inputs", []))
    ):
        if audit and not upfront:
            failures.append("jev_not_passed")
    receipt = (
        snapshot.get("report", {}).get("metadata", {}).get("extraction_verification")
        or {}
    )
    if upfront:
        from litagent.extraction_verification import (
            LEDGER_CONTRACT,
            verified_extraction_ledger,
        )
        from litagent.quality import build_fact_cards

        if jev_required:
            try:
                accepted_ledger = verified_extraction_ledger(
                    receipt,
                    snapshot.get("ledger", {}),
                    run_id=snapshot.get("source_run_id"),
                )
                if snapshot.get("report", {}).get("metadata", {}).get(
                    "fact_cards"
                ) != build_fact_cards(accepted_ledger):
                    raise ValueError("incomplete accepted cards")
            except ValueError:
                failures.append("extraction_jev_not_passed")
    return {
        "version": (
            "automatic-acceptance-v4"
            if optional
            else "automatic-acceptance-v3" if upfront else VERSION
        ),
        **(
            {
                "verification": metadata["verification"],
                "verification_sha256": digest(metadata["verification"]),
            }
            if optional
            else {}
        ),
        **(
            {"extraction_ledger_contract": LEDGER_CONTRACT}
            if upfront and jev_required
            else {}
        ),
        **(
            {"body_quality": "analysis_v1"}
            if metadata.get("body_quality") == "analysis_v1"
            else {}
        ),
        "policy": policy,
        "actor": "system",
        "source_run_id": snapshot.get("source_run_id"),
        "revision": snapshot.get("source_report_revision"),
        "report_sha256": snapshot.get("source_report_sha256"),
        "inventory_sha256": inventory.get("inventory_sha256"),
        "fact_cards_sha256": digest(
            snapshot.get("report", {}).get("metadata", {}).get("fact_cards", {})
        ),
        "evaluation_sha256": digest(evaluation),
        **({"extraction_verification_sha256": digest(receipt)} if upfront else {}),
        "methodology_sha256": digest(
            snapshot.get("context", {}).get("review_methodology", {})
        ),
        "status": "passed" if not failures else "failed",
        "publishable": not failures,
        "failure_codes": list(dict.fromkeys(failures)),
        "gates": {
            "evaluation": (
                "passed" if "evaluation_not_passed" not in failures else "failed"
            ),
            "judge": (
                "disabled"
                if not judge_required
                else (
                    ("blocked" if "evaluation_not_passed" in failures else "not_run")
                    if not audit
                    else "passed" if "judge_not_passed" not in failures else "failed"
                )
            ),
            **(
                {
                    "extraction_jev": (
                        "disabled"
                        if not jev_required
                        else (
                            "failed"
                            if "extraction_jev_not_passed" in failures
                            else "passed"
                        )
                    )
                }
                if upfront
                else {
                    "jev": (
                        (
                            "blocked"
                            if "evaluation_not_passed" in failures
                            else "not_run"
                        )
                        if not audit
                        else "passed" if "jev_not_passed" not in failures else "failed"
                    )
                }
            ),
        },
        "audit_id": audit.get("audit_id"),
        "human_calibration": "unverified",
    }
