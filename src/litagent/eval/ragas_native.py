"""Evaluate exact cited units with RAGAS and keep report-quality checks independent."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
from dataclasses import dataclass

from litagent.body_quality import validate_analysis_decisions
from litagent.eval.batched import (
    global_review_schema,
    limitations_check,
    normalize_global_findings,
)
from litagent.eval.engine_policy import (
    ANCHORED_GLOBAL_REVIEW,
    KEYED_ASSESSMENT,
    KEYED_GLOBAL_ANCHORS,
    KEYED_GLOBAL_QUOTED_OMISSIONS,
    KEYED_GLOBAL_UNITS,
    anchored_native_review,
    native_unit_response_contract,
    unit_native_review,
)
from litagent.eval.ragas_adapter import CONTRACT, NativeRagasAdapter
from litagent.eval.ragas_adapter import _parse as _parse_ragas_json
from litagent.eval.ragas_adapter import planned_batches
from litagent.eval.responsibilities import (
    GLOBAL_INSTRUCTION,
    parse_global_response,
    project_global_schema,
)
from litagent.eval.results import EvaluationRunResult
from litagent.eval.scoped import (
    _managed_span,
    _section,
    build_source_fragments,
    check_request_budget,
)
from litagent.evidence import extract_evidence_refs_ordered
from litagent.operational_limits import PROFILE, budget_resource_profile
from litagent.quality import validate_fact_snapshot
from litagent.report_render import build_report_source_units
from litagent.safety.planned_budget import (
    locked_writer_envelope,
    planned_backend,
    request_scope,
)


@dataclass(frozen=True)
class NativeEvaluationRunResult(EvaluationRunResult):
    """Reuse delivery calculations while exposing the actual RAGAS implementation."""

    def metrics(self):
        """Publish RAGAS semantic results without host entity/number reinterpretation."""
        values = super().metrics()
        details = values["faithfulness"]["details"]
        details.update(
            implementation=CONTRACT, display_name="RAGAS 实际引用 Faithfulness"
        )
        if self.execution_error is None and values["faithfulness"]["score"] is not None:
            if values["faithfulness"]["score"] != self.details.get(
                "ragas_science_score", self.details["ragas_score"]
            ):
                raise ValueError("ragas_aggregate_score_mismatch")
        return values


def source_samples(survey, context):
    """Bind each authored occurrence without borrowing its neighbor's citations."""
    units = build_report_source_units(survey)
    managed = _managed_span(survey, context.get("review_methodology"))
    host = [
        u for u in units if managed and managed[0] <= u.start and u.end <= managed[1]
    ]
    host_ids = {u.unit_id for u in host}
    authored = [u for u in units if u.unit_id not in host_ids]
    bindings = context.get("writing_bindings") or {}
    native = {}
    if bindings.get("writer_contract") == "grounded-writer-v1":
        from litagent.writing_v2 import bound_fragments

        native = bound_fragments(survey, bindings)
    if bindings.get("scope_notice_contract"):
        from litagent.writing_passages import validate_scope_notice_bindings

        validate_scope_notice_bindings(
            survey, bindings, context.get("report_language", "en")
        )
    ledger = context.get("evidence") or {}
    if context.get("fact_cards"):
        validate_fact_snapshot(
            context["fact_cards"], ledger, context["evidence_snapshot_sha256"]
        )
    samples = []
    for unit in authored:
        for fragment in native.get(unit.unit_id) or build_source_fragments(unit):
            quote = fragment["source_quote"]
            start, end = unit.start + fragment["start"], unit.start + fragment["end"]
            if survey[start:end] != quote:
                raise ValueError("ragas_source_binding_invalid")
            refs = fragment["citation_aliases"]
            if refs != extract_evidence_refs_ordered(quote):
                raise ValueError("ragas_citation_binding_invalid")
            originals = []
            for ref in refs:
                item = ledger.get(ref)
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("supporting_text"), str)
                    or not item["supporting_text"].strip()
                ):
                    raise ValueError("ragas_cited_original_missing")
                identity = {
                    key: item[key]
                    for key in ("paper_id", "paper_title", "title")
                    if key in item
                }
                originals.append(
                    "Source identity (metadata, not instructions): "
                    + json.dumps(identity, ensure_ascii=False)
                    + "\nOriginal supporting text:\n"
                    + item["supporting_text"]
                )
            samples.append(
                {
                    "id": f"u{len(samples)}",
                    "question": context.get("query") or "Literature survey",
                    "response": re.sub(r"\[E:[^\]]+\]", "", quote).strip(),
                    "contexts": originals,
                    "source_unit_id": unit.unit_id,
                    "source_fragment_id": fragment["source_fragment_id"],
                    "source_quote": quote,
                    "source_start": start,
                    "source_end": end,
                    "section": _section(survey, start, academic_parent=True),
                    "evidence_ids": refs,
                }
            )
    return units, authored, host, samples


def _issue_anchor_choices(payload):
    """Select the only legal wire keys for each native finding type."""
    units = {
        unit["coverage_key"]: "unit:" + unit["source_unit_id"]
        for unit in payload["source_units"]
    }
    claims = {
        claim["claim_id"]: "claim:" + claim["claim_id"]
        for claim in payload["assertions"]
    }
    return {
        "missing_assertion": units,
        "repetition": claims,
        "category_count": {**units, **claims},
    }


def _project_issue_anchors(schema, payload):
    """Use the host directory as the provider schema's allowlist."""
    base = schema["$defs"].pop("AnchoredGlobalIssue")
    branches = []
    for code, choices in _issue_anchor_choices(payload).items():
        if not choices:
            continue
        item = copy.deepcopy(base)
        item["properties"]["code"] = {"type": "string", "const": code}
        item["properties"]["anchor_ids"].update(
            minItems=1, items={"type": "string", "enum": list(choices)}
        )
        branches.append(item)
    if branches:
        schema["properties"]["issues"]["items"] = {"anyOf": branches}
    else:
        schema["properties"]["issues"] = {"type": "array", "maxItems": 0}


def _fail_global_field(
    progress, field, reason, *, code="evaluation_global_finding_identity_invalid"
):
    """Identify the offending wire field without treating it as a finding."""
    progress["global_validation_error"] = {"field": field, "reason": reason}
    raise ValueError(code)


def _restore_issue_anchors(raw, payload, progress):
    """Restore issue coordinates only from keys issued for this request."""
    choices = _issue_anchor_choices(payload)
    issues = raw.get("issues")
    if not isinstance(issues, list):
        _fail_global_field(progress, "issues", "missing_or_invalid_list")
    for index, issue in enumerate(issues):
        path = f"issues[{index}]"
        if not isinstance(issue, dict) or set(issue) != {
            "code",
            "anchor_ids",
            "explanation",
        }:
            _fail_global_field(progress, path, "invalid_fields")
        code = issue["code"]
        if not isinstance(code, str) or code not in choices:
            _fail_global_field(progress, path + ".code", "unknown_code")
        explanation = issue["explanation"]
        if not isinstance(explanation, str) or not explanation.strip():
            _fail_global_field(progress, path + ".explanation", "invalid_explanation")
        anchors = issue["anchor_ids"]
        if not isinstance(anchors, list) or not anchors:
            _fail_global_field(
                progress, path + ".anchor_ids", "missing_or_invalid_keys"
            )
        allowed, restored, seen = choices[code], [], set()
        for pos, anchor in enumerate(anchors):
            field = f"{path}.anchor_ids[{pos}]"
            if not isinstance(anchor, str) or anchor not in allowed:
                _fail_global_field(progress, field, "unknown_or_wrong_role")
            if anchor in seen:
                _fail_global_field(progress, field, "duplicate_key")
            seen.add(anchor)
            restored.append(allowed[anchor])
        if (
            code == "missing_assertion"
            and payload.get("global_unit_response_contract")
            == KEYED_GLOBAL_QUOTED_OMISSIONS
        ):
            units = {u["coverage_key"]: u for u in payload["source_units"]}
            if len(anchors) != 1 or explanation not in re.sub(
                r"\[E:[^\]]+\]", "", units[anchors[0]]["text"]
            ):
                _fail_global_field(
                    progress, path + ".explanation", "omission_quote_outside_unit"
                )
            unit = units[anchors[0]]
            matched = [
                claim["claim_id"]
                for claim in payload["assertions"]
                if claim["claim_id"] in unit["extracted_claim_ids"]
                and claim["claim_text"].strip(" 。.!！?？;；")
                == explanation.strip(" 。.!！?？;；")
            ]
            if matched:
                # A self-contradictory review cannot establish complete coverage.
                progress.setdefault("omission_review_conflicts", []).append(
                    {"source_unit_id": unit["source_unit_id"], "claim_ids": matched}
                )
        issue["anchor_ids"] = restored


def global_review_request(
    survey,
    context,
    config,
    units,
    assertions,
    *,
    compact=False,
    force_compact=False,
    anchored=False,
    unit_view=False,
    unit_response_contract=None,
):
    """Build the exact provider request while retaining canonical host input."""
    if unit_response_contract is not None and (
        not anchored
        or unit_response_contract
        not in (KEYED_GLOBAL_UNITS, KEYED_GLOBAL_ANCHORS, KEYED_GLOBAL_QUOTED_OMISSIONS)
    ):
        raise ValueError("ragas_global_unit_response_contract_invalid")
    short_anchors = unit_response_contract in (
        KEYED_GLOBAL_ANCHORS,
        KEYED_GLOBAL_QUOTED_OMISSIONS,
    )
    requirements = context.get("coverage_requirements") or []
    payload = {
        "body_analysis_contract": "body-analysis-v1",
        "global_findings_contract": (
            ANCHORED_GLOBAL_REVIEW if anchored else "located-global-findings-v1"
        ),
        "source_units": [{"source_unit_id": u.unit_id, "text": u.text} for u in units],
        "assertions": [
            {
                key: claim[key]
                for key in (
                    "claim_id",
                    "claim_text",
                    "kind",
                    "section",
                    "source_unit_id",
                    "source_fragment_id",
                    "evidence_ids",
                    "status",
                )
            }
            for claim in assertions
        ],
        "report_structure": [
            {
                "source_unit_id": u.unit_id,
                "section": _section(survey, u.start, academic_parent=True),
            }
            for u in units
        ],
        "coverage_requirements": requirements,
        "review_methodology": context.get("review_methodology") or {},
    }
    claim_ids_by_unit = {u.unit_id: [] for u in units}
    for claim in assertions:
        claim_ids_by_unit[claim["source_unit_id"]].append(claim["claim_id"])
    for unit in payload["source_units"]:
        unit["extracted_claim_ids"] = claim_ids_by_unit[unit["source_unit_id"]]
    schema = project_global_schema(global_review_schema(payload))
    if unit_response_contract:
        payload["global_unit_response_contract"] = unit_response_contract
        unit_schema = schema["$defs"]["UnitDecision"]
        del unit_schema["properties"]["source_unit_id"]
        unit_schema["required"].remove("source_unit_id")
        properties = {}
        for index, unit in enumerate(payload["source_units"], 1):
            key = f"u{index}"
            unit["coverage_key"] = key
            properties[key] = {"$ref": "#/$defs/UnitDecision"}
        schema["properties"]["units"] = {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        }
        if short_anchors:
            _project_issue_anchors(schema, payload)
    fmt = {
        "type": "json_schema",
        "json_schema": {
            "name": "ragas_report_review",
            "strict": True,
            "schema": schema,
        },
    }
    messages = [
        {
            "role": "system",
            "content": GLOBAL_INSTRUCTION
            + (
                "\nUse each unit's local claims list for omission checks. "
                "Mark that unit incomplete only for a specific unrepresented "
                "proposition or qualifier. Never use paper evidence to rewrite the inventory. "
                if unit_view
                else "\nFor omissions compare each source unit with every statement listed "
                "in its extracted_claim_ids. Supported and unsupported statements both "
                "count as extracted. Only report a specific unrepresented proposition "
                "or qualifier; mark that unit incomplete. Check units with no claims "
                "too. Never use paper evidence to rewrite the inventory. "
            )
            + (
                (
                    "In issues use anchor_ids exactly as permitted by the schema. "
                    "For missing_assertion select the original unit's coverage_key (uN). "
                    "For repetition select the affected claim_id at each location. "
                    "For category_count select a coverage_key or claim_id. "
                    "Do not copy source_unit_id or add prefixes. "
                    if short_anchors
                    else "In issues use anchor_ids copied exactly from claim_id or source_unit_id. "
                    "Do not add prefixes. Give only source_unit_id anchors for missing assertions. "
                    "For repetition include the affected claim anchors at each location; "
                    "the host derives their parent units. "
                )
                + "Do not return source_unit_id or claim_ids inside issues. "
                "For each answered requirement list only "
                "the minimum sufficient supported scientific claims that together "
                "answer it. If supported claims do not cover every required part, "
                "mark it unanswered. Do not use the whole inventory as a witness."
                if anchored
                else "For omissions use the original source_unit_id and an empty claim_ids list."
            ),
        },
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]
    if unit_response_contract:
        messages[0]["content"] += (
            "\nReturn units as an object keyed by each source unit's coverage_key. "
            "Fill every required key with complete and explanation. "
        )
        if short_anchors:
            messages[0][
                "content"
            ] += "Use the same coverage_key for unit issue anchors."
        else:
            messages[0][
                "content"
            ] += "Use the original source_unit_id only for issue anchors."
    if unit_response_contract == KEYED_GLOBAL_QUOTED_OMISSIONS:
        messages[0]["content"] += (
            "\nInventory coverage is independent of verdicts. Compare the quoted "
            "report proposition to ALL extracted statements in that unit, including "
            "unsupported and translated equivalents. If it is represented, the unit "
            "is complete even when its paper support is insufficient. For each "
            "missing_assertion use exactly one unit key and put only the exact "
            "unrepresented clause from that unit in explanation, without citations."
        )
    if unit_view or compact or force_compact:
        from litagent.eval.ragas_global_wire import (
            INSTRUCTION,
            UNIT_INSTRUCTION,
            pack_global_input,
            unpack_global_input,
        )
        from litagent.safety.budget import strict_request_reservation

        limit = int(config.context_tokens * (1 - config.strict_context_margin))
        if (
            unit_view
            or force_compact
            or strict_request_reservation(messages, config.max_tokens, fmt) > limit
        ):
            wire = pack_global_input(payload, by_unit=unit_view)
            if unpack_global_input(wire) != payload:
                raise ValueError("ragas_global_projection_roundtrip_invalid")
            messages[0]["content"] += "\n" + (
                UNIT_INSTRUCTION if unit_view else INSTRUCTION
            )
            messages[1]["content"] = json.dumps(
                wire, ensure_ascii=False, separators=(",", ":")
            )
    return payload, messages, fmt


def parse_native_global_response(content, payload, progress):
    """Map exact native IDs to the shared anchor representation before validation."""
    anchored = payload.get("global_findings_contract") == ANCHORED_GLOBAL_REVIEW
    contract = payload.get("global_unit_response_contract")
    short_anchors = contract in (KEYED_GLOBAL_ANCHORS, KEYED_GLOBAL_QUOTED_OMISSIONS)
    keyed = contract == KEYED_GLOBAL_UNITS or short_anchors
    if keyed or anchored:
        try:
            raw = _parse_ragas_json(content)
        except ValueError as exc:
            if short_anchors:
                progress["global_validation_error"] = {
                    "field": "response",
                    "reason": str(exc).splitlines()[0],
                }
            raise
        if keyed:
            decisions = raw.get("units") if isinstance(raw, dict) else None
            directory = {
                unit["coverage_key"]: unit["source_unit_id"]
                for unit in payload["source_units"]
            }
            if not isinstance(decisions, dict):
                _fail_global_field(
                    progress,
                    "units",
                    "missing_or_invalid_object",
                    code="evaluation_global_unit_coverage_invalid",
                )
            missing = directory.keys() - decisions.keys()
            unknown = decisions.keys() - directory.keys()
            if missing or unknown:
                key = min(missing or unknown)
                _fail_global_field(
                    progress,
                    f"units.{key}",
                    "missing_key" if missing else "unknown_key",
                    code="evaluation_global_unit_coverage_invalid",
                )
            for key, value in decisions.items():
                if not isinstance(value, dict) or set(value) != {
                    "complete",
                    "explanation",
                }:
                    _fail_global_field(
                        progress,
                        f"units.{key}",
                        "invalid_fields",
                        code="evaluation_global_unit_coverage_invalid",
                    )
                if short_anchors and (
                    type(value["complete"]) is not bool
                    or not isinstance(value["explanation"], str)
                ):
                    _fail_global_field(
                        progress,
                        f"units.{key}",
                        "invalid_decision",
                        code="evaluation_global_unit_coverage_invalid",
                    )
            raw["units"] = [
                {**decisions[key], "source_unit_id": source_id}
                for key, source_id in directory.items()
            ]
        if short_anchors:
            _restore_issue_anchors(raw, payload, progress)
        elif anchored and isinstance(raw, dict) and isinstance(raw.get("issues"), list):
            aliases = {
                **{
                    u["source_unit_id"]: "unit:" + u["source_unit_id"]
                    for u in payload["source_units"]
                },
                **{
                    c["claim_id"]: "claim:" + c["claim_id"]
                    for c in payload["assertions"]
                },
            }
            for issue in raw["issues"]:
                if isinstance(issue, dict) and isinstance(
                    issue.get("anchor_ids"), list
                ):
                    # The shared parser still checks roles, unknown IDs and duplicates
                    # after both permitted spellings have the same canonical value.
                    issue["anchor_ids"] = [
                        (
                            aliases.get(anchor, anchor)
                            if isinstance(anchor, str)
                            else anchor
                        )
                        for anchor in issue["anchor_ids"]
                    ]
        content = json.dumps(raw, ensure_ascii=False, allow_nan=False)
    return normalize_global_findings(
        parse_global_response(content, payload, progress), payload
    )


async def _global_review(
    llm, survey, context, config, units, assertions, progress, *, force_compact=False
):
    """Reuse the existing global model without redoing RAGAS scientific NLI."""
    budget = getattr(llm, "_cost_budget", None)
    envelope = locked_writer_envelope(budget)
    payload, messages, fmt = global_review_request(
        survey,
        context,
        config,
        units,
        assertions,
        compact=budget_resource_profile(budget) == PROFILE,
        force_compact=force_compact,
        anchored=anchored_native_review(envelope),
        unit_view=unit_native_review(envelope),
        unit_response_contract=native_unit_response_contract(envelope),
    )
    key = hashlib.sha256(
        json.dumps([messages, fmt], ensure_ascii=False).encode()
    ).hexdigest()
    if progress.get("evaluation_identity") is not None:
        key = progress["evaluation_identity"] + ":" + key
    progress["global_check"] = "started_no_replay"
    checkpoint = context.get("evaluation_checkpoint")
    if checkpoint:
        checkpoint(progress)
    with request_scope("global", key):
        check_request_budget(
            llm,
            messages,
            config.max_tokens,
            int(config.context_tokens * (1 - config.strict_context_margin)),
            response_format=fmt,
        )
        response = await llm.chat(
            messages,
            response_format=fmt,
            max_output_tokens=config.max_tokens,
            temperature=0,
            extra_body={"thinking": {"type": "disabled"}},
            provider_api="responses",
            transport_max_attempts=1,
        )
    if (
        response.finish_reason not in (None, "stop")
        or getattr(response, "provider_status", None) == "incomplete"
    ):
        raise ValueError("ragas_global_incomplete_response")
    if not response.usage or response.usage.get("usage_available") is False:
        raise ValueError("ragas_global_usage_unknown")
    parsed = parse_native_global_response(response.content, payload, progress)
    supported = {a["claim_id"] for a in assertions if a["status"] == "supported"}
    complete_units = {u.source_unit_id for u in parsed.units if u.complete}
    omission_units = {
        x.source_unit_id for x in parsed.issues if x.code == "missing_assertion"
    }
    disputed_units = {
        row["source_unit_id"] for row in progress.get("omission_review_conflicts", [])
    }
    progress.update(
        global_check="completed", global_usage=response.usage, global_request_id=key
    )
    if payload["global_findings_contract"] == ANCHORED_GLOBAL_REVIEW:
        progress["global_findings_contract"] = ANCHORED_GLOBAL_REVIEW
    if unit_native_review(envelope):
        progress["global_input_contract"] = envelope["semantic_engine"][
            "global_input_contract"
        ]
    if payload.get("global_unit_response_contract"):
        progress["global_unit_response_contract"] = payload[
            "global_unit_response_contract"
        ]
    return {
        "conflicts": [x.model_dump() for x in parsed.conflicts],
        "core_coverage": [x.model_dump() for x in parsed.coverage],
        "global_issues": [{**x.model_dump(), "owner": x.owner} for x in parsed.issues],
        "inventory_check": {
            "conflicting_unit_ids": sorted(
                (complete_units & omission_units) | disputed_units
            ),
            "status": (
                "passed"
                if all(u.complete for u in parsed.units)
                and not any(x.owner == "inventory" for x in parsed.issues)
                else "failed"
            ),
        },
        "analysis_check": validate_analysis_decisions(
            parsed.analyses, assertions, supported, payload["report_structure"]
        ),
        "limitations_check": limitations_check(
            parsed, assertions, supported, payload["report_structure"], independent=True
        ),
    }


def _host_organization(statement, quote):
    """Accept only an exact, standalone description of this report's organization."""
    value = statement.strip().rstrip("。.")
    sentences = re.split(r"[。！？\n]", re.sub(r"\[E:[^\]]+\]", "", quote))
    if value not in {s.strip().rstrip(".") for s in sentences}:
        return False
    # Restrict exemptions to heading-like labels and an explicit self-reference.
    return bool(
        re.fullmatch(
            r"(?:本综述|本文)按[“「][\w\s、—–-]{2,60}[”」](?:展开|组织|解释[\w\s、-]{1,70})",
            value,
        )
        and not re.search(
            r"表明|证明|发现|优于|导致|提高|提升|降低|减少|依赖|可以|能够|可使|"
            r"用于|产生|实现|更高|更低|\d|[；;]",
            value,
        )
    )


def canonical_assertions(samples, *, host_organization=False):
    """Preserve every generated statement and any completed NLI without a score."""
    assertions, groups = [], {}
    for sample in samples:
        groups[sample["id"]] = []
        verdicts = sample.get("verdicts", [])
        for index, statement in enumerate(sample.get("statements", [])):
            verdict = verdicts[index] if index < len(verdicts) else None
            cid = f"claim_{len(assertions)}"
            groups[sample["id"]].append(cid)
            status = (
                "unverifiable"
                if verdict is None
                else "supported" if verdict["verdict"] == 1 else "unsupported"
            )
            organization = host_organization and _host_organization(
                statement, sample["source_quote"]
            )
            if organization and verdict is not None:
                status = "supported"
            assertions.append(
                {
                    **{
                        key: sample[key]
                        for key in (
                            "source_unit_id",
                            "source_fragment_id",
                            "source_quote",
                            "source_start",
                            "source_end",
                            "section",
                            "evidence_ids",
                        )
                    },
                    "claim_id": cid,
                    "kind": "report_structure" if organization else "scientific",
                    "claim_text": statement,
                    "atomic": True,
                    "classification_valid": True,
                    "source_faithful": True,
                    "diagnostic_source": "survey-evaluation-ragas-v1",
                    "local_error": None,
                    "correct_kind": (
                        "report_structure" if organization else "scientific"
                    ),
                    "statement_id": cid,
                    "alignment_status": "exact",
                    "citation_scope": "exact",
                    "status": status,
                    "reason_code": (
                        "ragas_nli_incomplete"
                        if verdict is None
                        else (
                            "supported"
                            if status == "supported"
                            else "insufficient_support"
                        )
                    ),
                    "explanation": (
                        "Exact report organization statement."
                        if organization
                        else (
                            "NLI not completed."
                            if verdict is None
                            else verdict["reason"]
                        )
                    ),
                    "support_basis": [],
                    "aspect_checks": [],
                }
            )
    return assertions, groups


def _check_native_admission(
    phase, samples, *, llm, survey, context, config, units, adapter, envelope, progress
):
    """Prove the current round fits without borrowing future rounds' slots."""
    from litagent.eval.engine_policy import phase_batches

    budget = planned_backend(getattr(llm, "_cost_budget", None))
    limit = int(config.context_tokens * (1 - config.strict_context_margin))
    state = progress.setdefault("capacity_admission", {})
    stage = phase
    try:
        if adapter.batch_size != envelope["semantic_engine"]["batch_size"]:
            raise ValueError("ragas_capacity_batch_mismatch")
        batches = planned_batches(
            phase,
            samples,
            config,
            adapter.batch_size,
            phase_batches(envelope),
            keyed_assessment=adapter.keyed_assessment,
        )
        reservations = []
        for ids, messages, fmt, _ in batches:
            with request_scope(phase, "ragas-preflight:" + ids[0]):
                reservations.append(
                    check_request_budget(
                        llm, messages, config.max_tokens, limit, response_format=fmt
                    )
                )
        if reservations:
            budget.check_stage_requests(phase, reservations)
        state[phase] = {"batch_reservations": reservations, "context_limit": limit}
        if phase == "assessment" and reservations:
            stage = "global"
            # Pending verdicts have the longest status label. The schema depends
            # on claim identities and sections, so either verdict only shrinks the wire.
            assertions, _ = canonical_assertions(
                samples,
                host_organization=native_unit_response_contract(envelope)
                == KEYED_GLOBAL_QUOTED_OMISSIONS,
            )
            _, messages, fmt = global_review_request(
                survey,
                context,
                config,
                units,
                assertions,
                force_compact=True,
                anchored=anchored_native_review(envelope),
                unit_view=unit_native_review(envelope),
                unit_response_contract=native_unit_response_contract(envelope),
            )
            from litagent.safety.budget import strict_request_reservation

            amount = strict_request_reservation(messages, config.max_tokens, fmt)
            state["global"] = {"reservation": amount, "context_limit": limit}
            with request_scope("global", "ragas-preflight:global"):
                check_request_budget(
                    llm, messages, config.max_tokens, limit, response_format=fmt
                )
            budget.check_stage_requests("global", [amount])
            state["complete_inventory"] = {
                "sample_count": len(samples),
                "statement_count": len(assertions),
            }
    except ValueError as exc:
        state["rejected_stage"] = stage
        state["reason"] = str(exc)
        if getattr(exc, "diagnostics", None):
            state["diagnostics"] = exc.diagnostics
        raise
    return [ids for ids, *_ in batches]


async def execute_native(*, llm, survey, context, config):
    """Run native RAGAS and full independent global review under one report identity."""
    sha = hashlib.sha256(survey.encode()).hexdigest()
    progress = {"engine": CONTRACT, "processed_assertions": 0, "batches": []}
    details = {
        "evaluation_version": "survey-evaluation-ragas-v1",
        "source_report_sha256": sha,
        "quality_policy": context.get("quality_policy", "legacy"),
        "body_analysis_contract": "body-analysis-v1",
        "coverage_requirements": context.get("coverage_requirements") or [],
        "structure_checks": context.get("structure_checks", {}),
        "method_checks": context.get("method_checks", {}),
        "language_checks": context.get("language_checks", {}),
        "assertions": [],
        "stages": progress,
        "coverage": {},
        "threshold": 0.8,
        "implementation": CONTRACT,
        **(
            {
                "scope_notice_contract": context["writing_bindings"][
                    "scope_notice_contract"
                ]
            }
            if (context.get("writing_bindings") or {}).get("scope_notice_contract")
            else {}
        ),
        "fact_card_map": {
            key: {
                field: card.get(field)
                for field in (
                    "evidence_id",
                    "paper_id",
                    "content_sha256",
                    "source_locator",
                    "source_version",
                )
            }
            for key, card in context.get("fact_cards", {}).items()
        },
    }
    error = None
    budget = getattr(llm, "_cost_budget", None)
    envelope = locked_writer_envelope(budget)
    engine = (envelope or {}).get("semantic_engine")
    keyed_assessment = (
        engine is None or engine.get("assessment_response_contract") == KEYED_ASSESSMENT
    )
    adapter = NativeRagasAdapter(llm, config, keyed_assessment=keyed_assessment)
    if keyed_assessment:
        progress["assessment_response_contract"] = KEYED_ASSESSMENT
    adapter.evaluation_identity = context.get("native_evaluation_identity")
    if adapter.evaluation_identity is not None:
        progress["evaluation_identity"] = adapter.evaluation_identity
    adapter.before_send = context.get("recovery_on_send")
    try:
        async with asyncio.timeout(
            config.total_timeout_seconds
            or config.timeout_seconds
            * (
                2
                * (
                    (len(build_report_source_units(survey)) + adapter.batch_size - 1)
                    // adapter.batch_size
                )
                + 1
            )
        ):
            all_units, units, host, samples = source_samples(survey, context)
            details["coverage"] = {
                "source_units": len(all_units),
                "processed_units": len(host),
                "model_source_units": len(units),
                "model_processed_units": 0,
                "host_processed_units": len(host),
            }
            checkpoint = context.get("evaluation_checkpoint")

            def save(value):
                """Retain transport state in the existing version-bound checkpoint."""
                progress.update(value)
                if checkpoint:
                    checkpoint(progress)

            adapter.checkpoint = save
            if budget_resource_profile(budget) == PROFILE and engine:

                def admit(phase, material):
                    """Keep all legal inventory until the full consumer preflight."""
                    return _check_native_admission(
                        phase,
                        material,
                        llm=llm,
                        survey=survey,
                        context=context,
                        config=config,
                        units=units,
                        adapter=adapter,
                        envelope=envelope,
                        progress=progress,
                    )

                adapter.admission = admit
            result = await adapter.score(samples)
            organization_enabled = (engine or {}).get(
                "global_unit_response_contract"
            ) == KEYED_GLOBAL_QUOTED_OMISSIONS
            assertions, claim_groups = canonical_assertions(
                result["samples"], host_organization=organization_enabled
            )
            scientific = [a for a in assertions if a["kind"] == "scientific"]
            details.update(
                validation_semantics="official-statement-generation-and-nli;host-locators;independent-global-omission-check",
                assertions=assertions,
                ragas_score=result["score"],
                ragas_version=result["ragas_version"],
                unsupported_claims=[
                    a for a in assertions if a["status"] == "unsupported"
                ],
                writer_bindings=context.get("writing_bindings", {}),
                evidence_snapshot_sha256=context.get("evidence_snapshot_sha256"),
            )
            if len(scientific) != len(assertions):
                details["ragas_science_score"] = (
                    sum(a["status"] == "supported" for a in scientific)
                    / len(scientific)
                    if scientific
                    else None
                )
            progress.update(
                expected_assertions=len(assertions),
                processed_assertions=len(assertions),
                expected_batches=sum(
                    r["phase"] == "assessment" for r in result["requests"]
                ),
                batches=[
                    {
                        "index": index,
                        "status": "completed",
                        "claim_ids": [
                            cid
                            for sample_id in request["sample_ids"]
                            for cid in claim_groups[sample_id]
                        ],
                    }
                    for index, request in enumerate(
                        r for r in result["requests"] if r["phase"] == "assessment"
                    )
                ],
            )
            details["coverage"].update(
                processed_units=len(all_units), model_processed_units=len(units)
            )
            details.update(
                await _global_review(
                    llm,
                    survey,
                    context,
                    config,
                    units,
                    assertions,
                    progress,
                    force_compact=adapter.admission is not None,
                )
            )
            details["coverage"]["complete"] = True
            if checkpoint:
                checkpoint(progress)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        error = (
            str(exc)
            if str(exc).startswith(("ragas_", "evaluation_", "planned_", "budget_"))
            else "ragas_execution_error:" + type(exc).__name__
        )
        details.update(execution_error=error, unverified_reason=error)
        details["assertions"], _ = canonical_assertions(adapter.samples)
        details["partial_samples"] = [
            {
                key: copy.deepcopy(sample.get(key))
                for key in ("id", "statements", "verdicts")
            }
            for sample in adapter.samples
        ]
        progress["expected_assertions"] = sum(
            len(s.get("statements", [])) for s in adapter.samples
        )
        progress["processed_assertions"] = sum(
            len(s.get("verdicts", [])) for s in adapter.samples
        )
        progress["execution_error"] = error
        if context.get("evaluation_checkpoint"):
            context["evaluation_checkpoint"](progress)
    run = EvaluationRunResult.from_details(details, sha, error=error)
    return NativeEvaluationRunResult(
        run.report_sha256, run.details, run.execution_error
    )


def completed_repair(details, survey):
    """Authorize bounded repair only from complete native results at exact locations."""
    from litagent.eval.ragas_repair import native_repair_diagnostics

    try:
        if (
            details.get("source_report_sha256")
            != hashlib.sha256(survey.encode()).hexdigest()
        ):
            return False
        run = EvaluationRunResult.from_details(details, details["source_report_sha256"])
        if not run.inventory_valid() or not native_repair_diagnostics(details):
            return False
        for claim in run.assertions:
            a, b = claim["source_start"], claim["source_end"]
            if (
                type(a) is not int
                or type(b) is not int
                or not 0 <= a < b <= len(survey)
            ):
                return False
            if (
                survey[a:b] != claim["source_quote"]
                or extract_evidence_refs_ordered(survey[a:b]) != claim["evidence_ids"]
            ):
                return False
        return True
    except (ValueError, KeyError, TypeError):
        return False


def funded_capacity(report, budget):
    """Authorize an over-128 inventory only with settled native stage escrows."""
    from litagent.eval.engine_policy import validate_engine
    from litagent.quality import digest
    from litagent.safety.planned_accounting import validate_accounting_snapshot
    from litagent.safety.planned_budget import planned_backend

    try:
        owner = planned_backend(budget)
        snapshot = validate_accounting_snapshot(
            owner.snapshot(), expected_plan=owner.plan
        )
        # This path authenticates settled results, including historical readbacks.
        validate_engine(owner.plan["envelope"]["semantic_engine"], archived=True)
        details = report["evaluation"]["faithfulness"]["details"]
        sha = hashlib.sha256(report["survey"].encode()).hexdigest()
        run = EvaluationRunResult.from_details(details, sha)
        if (
            not run.inventory_valid()
            or snapshot["usage_uncertain"]
            or snapshot["stop_reason"]
        ):
            raise ValueError("incomplete")
        entries = snapshot["entries"]
        if not entries or any(
            e["status"] != "settled" or not e["sent"] for e in entries.values()
        ):
            raise ValueError("unsettled")
        requests = details["stages"]["requests"]
        expected = [(r["request_id"], r["phase"]) for r in requests]
        expected.append((details["stages"]["global_request_id"], "global"))
        if not requests or any(r["status"] != "completed" for r in requests):
            raise ValueError("incomplete transport")
        for key, phase in expected:
            entry = entries[key]
            if entry["stage"] != phase or not entry.get("wire_sha256"):
                raise ValueError("stage identity")
        return {
            "contract": "host-automatic-capacity-ragas-v1",
            "source_report_sha256": sha,
            "budget_plan_identity": owner.plan["identity"],
            "budget_snapshot_sha256": digest(snapshot),
            "actual_assertions": len(run.assertions),
            "actual_batches": sum(r["phase"] == "assessment" for r in requests),
            "semantic_engine": owner.plan["envelope"]["semantic_engine"],
        }
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("automatic_capacity_binding_invalid") from exc
