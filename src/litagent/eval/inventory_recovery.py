"""Recover an invalid assertion inventory without editing the frozen report."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from contextvars import ContextVar

from litagent.quality import digest

CONTRACT = "bounded-inventory-recovery-v1"
MIXED_CONTRACT = "bounded-inventory-recovery-v2"
RECOVERY_CONTRACTS = frozenset((CONTRACT, MIXED_CONTRACT))
PRESERVED_FRAGMENT_FIELDS = (
    "source_unit_id",
    "source_fragment_id",
    "source_start",
    "source_end",
    "evidence_ids",
    "assertions",
    "exclusion_reason",
)
INVENTORY_STABILITY_RULE = (
    "Unflagged fragments preserve their ordered kind/claim_text and exclusion_reason "
    "exactly; source ranges and citations remain host-bound. Findings alone authorize "
    "fragment changes. Preserve content, never prior scientific verdicts."
)
_active: ContextVar[dict | None] = ContextVar("inventory_recovery", default=None)
INSTRUCTION = """
INVENTORY RECOVERY: the previous inventory was rejected for non-atomic assertions.
inventory_recovery.findings identifies extraction defects, NOT false report facts.
Independently re-extract the ENTIRE unchanged report using the SAME output schema.
Split independent mechanisms, results, limitations and method comparisons without
dropping propositions, negations, datasets, numbers, conditions or qualifiers.
Keep conditional relations intact. Preserve each proposition's own host fragment
and actual citations. Do not edit prose, borrow sources, or turn scientific facts
into organization. Previous verdicts are not evidence; all claims will be checked
again. Unflagged paragraphs, Abstract and Conclusion must also be fully processed.
"""
MIXED_INSTRUCTION = """
INVENTORY RECOVERY bounded-inventory-recovery-v2: independently re-extract the
ENTIRE unchanged report. Prior findings are UNRESOLVED concerns, not proven
defects, scientific evidence, or instructions to change the prose. Review each
actual host fragment, including unflagged text, Abstract and Conclusion, with
the SAME output schema. Reconsider classification and source fidelity; split
independent propositions without losing negation, entities, numbers, conditions
or qualifiers. A missing-assertion concern does not license inventing a claim.
Do not copy previous categories or verdicts, borrow another fragment's citations,
delete report text, or hide science as organization. All new assertions require
fresh actual-citation assessment and the complete global check. Repeating an
unchanged inventory or cycling to an earlier inventory is no progress.
pN refers only to prior advisory claims; vN identifies this request's source units.
Neither is a new output field or claim ID. Return the same fragment-keyed schema.
"""
MIXED_STABILITY_INSTRUCTION = """
INVENTORY RECOVERY bounded-inventory-recovery-v2: independently re-extract the
ENTIRE unchanged report. Prior findings are UNRESOLVED concerns, not proven
defects, scientific evidence, or instructions to change the prose. Review each
actual host fragment, including unflagged text, Abstract and Conclusion, with
the SAME complete output schema. preserved_fragments supplies the latest unflagged
baseline: copy its ordered kind/claim_text assertions and exclusion_reason exactly,
including empty fragments. Only fragments named by findings may change. This freezes
extraction content, never scientific verdicts; every assertion is assessed again.
Within the named fragments reconsider classification and source fidelity; split
independent propositions without losing negation, entities, numbers, conditions
or qualifiers. A missing-assertion concern does not license inventing a claim.
Within named fragments do not assume prior kinds/verdicts; preserved kinds are
content only, never validation. Never borrow another fragment's citations, delete
report text, or hide science as organization. All assertions require
fresh actual-citation assessment and the complete global check. Repeating an
unchanged inventory or cycling to an earlier inventory is no progress.
pN refers only to prior advisory claims; vN identifies this request's source units.
Neither is a new output field or claim ID. Return the same fragment-keyed schema.
"""


DEFINITION_RECOVERY_INSTRUCTION = """
INVENTORY RECOVERY: findings are unresolved extraction concerns, not proven report
errors, evidence or permission to edit prose. Use the same schema and shared
definitions to process the entire unchanged report. Only findings authorize changes
to named fragments; copy every preserved_fragments ordered kind/claim_text and
exclusion_reason exactly, including empty fragments. Preserved kinds freeze content,
not validation. Within flagged fragments independently reconsider extraction; never
invent a missing claim, borrow citations or copy previous verdicts. Every assertion
requires fresh actual-citation assessment and full global review. Repeating or
cycling inventories is no progress. pN is prior advisory identity, vN a source-unit
alias, neither a new output field or claim ID. Return the same fragment-keyed schema.
"""


def recovery_instruction(contract, *, stable=False, support_validation_revision=None):
    """Choose an explicit revision without changing historical instructions."""
    if type(stable) is not bool or stable and contract != MIXED_CONTRACT:
        raise ValueError("inventory_recovery_policy_invalid")
    if support_validation_revision is not None:
        from litagent.eval.scoped import (
            SUPPORT_VALIDATION_REVISION_V2,
            _check_support_revision,
        )

        _check_support_revision(support_validation_revision)
        if support_validation_revision == SUPPORT_VALIDATION_REVISION_V2:
            if contract != MIXED_CONTRACT or not stable:
                raise ValueError("inventory_recovery_policy_invalid")
            return DEFINITION_RECOVERY_INSTRUCTION
    if contract == CONTRACT:
        return INSTRUCTION
    if contract == MIXED_CONTRACT:
        return MIXED_STABILITY_INSTRUCTION if stable else MIXED_INSTRUCTION
    raise ValueError("inventory_recovery_policy_invalid")


def mixed_inventory_recovery_identity(*, stable=False):
    """Bind all request-side rules enabled by the new internal revision."""
    from litagent.eval.scoped import mixed_global_findings_identity
    from litagent.writing_flat import (
        inventory_stability_projection_identity,
        reply_layout_identity,
    )

    if type(stable) is not bool:
        raise ValueError("inventory_recovery_policy_invalid")
    previous = digest(
        [
            MIXED_CONTRACT,
            MIXED_INSTRUCTION,
            mixed_global_findings_identity(),
            reply_layout_identity(),
        ]
    )
    return (
        digest(
            [
                previous,
                MIXED_STABILITY_INSTRUCTION,
                INVENTORY_STABILITY_RULE,
                PRESERVED_FRAGMENT_FIELDS,
                inventory_stability_projection_identity(),
            ]
        )
        if stable
        else previous
    )


def inventory_identity(assertions):
    """Compare ordered extraction content, never previous scientific verdicts."""
    return digest(
        [
            [c["source_fragment_id"], c["kind"], c["claim_text"], c["evidence_ids"]]
            for c in assertions
        ]
    )


def inventory_recovery_feedback(survey, details, *, contract=CONTRACT, stable=False):
    """Keep v1 eligibility unchanged; explicitly opt into full mixed recovery."""
    if type(stable) is not bool or stable and contract != MIXED_CONTRACT:
        raise ValueError("inventory_recovery_policy_invalid")
    if contract == MIXED_CONTRACT:
        try:
            return _mixed_recovery_feedback(survey, details, stable=stable)
        except (KeyError, TypeError, AttributeError, StopIteration) as exc:
            raise ValueError("inventory_recovery_source_identity_invalid") from exc
    if contract != CONTRACT:
        raise ValueError("inventory_recovery_policy_invalid")
    return _legacy_recovery_feedback(survey, details)


def _legacy_recovery_feedback(survey, details):
    """Authorize only exact, host-validated non-atomic extraction findings."""
    import hashlib

    if (
        details.get("assessment_stage") != "host_validated"
        or details.get("inventory_check", {}).get("status") != "failed"
        or details.get("stages", {}).get("global_check") != "completed"
    ):
        return None
    sha = hashlib.sha256(survey.encode()).hexdigest()
    if details.get("source_report_sha256") != sha:
        raise ValueError("inventory_recovery_report_identity_invalid")
    claims = details.get("assertions", [])
    by_id = {c["claim_id"]: c for c in claims}
    if len(by_id) != len(claims):
        raise ValueError("inventory_recovery_claim_identity_invalid")
    findings = []
    from litagent.report_render import build_report_source_units

    units = {u.unit_id: u for u in build_report_source_units(survey)}
    from litagent.eval.scoped import build_source_fragments

    native = {}
    if (
        details.get("writer_bindings", {}).get("writer_contract")
        == "grounded-writer-v1"
    ):
        from litagent.writing_v2 import bound_fragments

        native = bound_fragments(survey, details["writer_bindings"])
    for issue in details.get("global_issues", []):
        if issue.get("owner") != "inventory":
            continue
        if issue.get("code") != "non_atomic":
            return None
        for cid in issue.get("claim_ids", []):
            claim = by_id.get(cid)
            if claim is None or claim["source_unit_id"] != issue["source_unit_id"]:
                raise ValueError("inventory_recovery_claim_identity_invalid")
            findings.append({**claim, "explanation": issue["explanation"]})
    if any(
        c.get("classification_valid") is False or c.get("source_faithful") is False
        for c in claims
    ):
        return None
    findings.extend(c for c in claims if c.get("atomic") is False)
    unique = {}
    for claim in findings:
        start, end = claim.get("source_start"), claim.get("source_end")
        unit = units.get(claim.get("source_unit_id"))
        fragment = (
            next(
                (
                    f
                    for f in (native.get(unit.unit_id) or build_source_fragments(unit))
                    if f["source_fragment_id"] == claim.get("source_fragment_id")
                ),
                None,
            )
            if unit is not None
            else None
        )
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(survey)
            or survey[start:end] != claim.get("source_quote")
            or unit is None
            or not unit.start <= start < end <= unit.end
            or fragment is None
            or (start, end)
            != (unit.start + fragment["start"], unit.start + fragment["end"])
        ):
            raise ValueError("inventory_recovery_source_identity_invalid")
        from litagent.evidence import extract_evidence_refs_ordered

        if extract_evidence_refs_ordered(survey[start:end]) != claim["evidence_ids"]:
            raise ValueError("inventory_recovery_citation_identity_invalid")
        unique[claim["claim_id"]] = {
            k: claim[k]
            for k in (
                "claim_id",
                "source_unit_id",
                "source_fragment_id",
                "source_start",
                "source_end",
                "source_quote",
                "claim_text",
                "evidence_ids",
                "explanation",
            )
        }
    if not unique:
        return None
    grouped = {}
    for claim in unique.values():
        key = claim["source_fragment_id"]
        group = grouped.setdefault(
            key,
            {
                "source_fragment_id": key,
                "source_unit_id": claim["source_unit_id"],
                "claim_ids": [],
                "code": "non_atomic",
            },
        )
        group["claim_ids"].append(claim["claim_id"])
    return {
        "contract": CONTRACT,
        "source_report_sha256": sha,
        "previous_inventory_sha256": inventory_identity(claims),
        # The full report is sent once. Audit the original positions/texts locally;
        # transmit their exact fragment identities instead of repeating each quote.
        "bindings_sha256": digest(list(unique.values())),
        "findings": list(grouped.values()),
    }


def _mixed_recovery_feedback(survey, details, *, stable=False):
    """Preserve all concerns, validating their complete host-owned basis first."""
    import hashlib

    from litagent.eval.scoped import build_source_fragments
    from litagent.evidence import extract_evidence_refs_ordered
    from litagent.report_render import build_report_source_units

    if not isinstance(survey, str) or not isinstance(details, dict):
        raise ValueError("inventory_recovery_source_identity_invalid")
    if (
        details.get("assessment_stage") != "host_validated"
        or details.get("inventory_check", {}).get("status") != "failed"
        or details.get("stages", {}).get("global_check") != "completed"
    ):
        return None
    sha = hashlib.sha256(survey.encode()).hexdigest()
    if details.get("source_report_sha256") != sha:
        raise ValueError("inventory_recovery_report_identity_invalid")
    stages = details["stages"]
    requests = stages.get("requests")
    if (
        not isinstance(requests, list)
        or not requests
        or any(
            not isinstance(r, dict)
            or r.get("status") != "response_received"
            or r.get("usage", {}).get("usage_available") is not True
            for r in requests
        )
    ):
        raise ValueError("inventory_recovery_usage_identity_invalid")
    claims = details.get("assertions")
    if not isinstance(claims, list) or not claims:
        raise ValueError("inventory_recovery_claim_identity_invalid")
    units = {u.unit_id: u for u in build_report_source_units(survey)}
    bindings = details.get("writer_bindings", {})
    native = {}
    if bindings:
        from litagent.writing_v2 import bound_fragments

        native = bound_fragments(survey, bindings)
    fragments = {
        f["source_fragment_id"]: (unit, f)
        for unit in units.values()
        for f in (native.get(unit.unit_id) or build_source_fragments(unit))
    }
    by_id = {}
    for c in claims:
        if not isinstance(c, dict) or not isinstance(c.get("claim_id"), str):
            raise ValueError("inventory_recovery_claim_identity_invalid")
        cid = c["claim_id"]
        if not cid or cid in by_id:
            raise ValueError("inventory_recovery_claim_identity_invalid")
        kinds = {"scientific", "runtime_method", "report_structure"}
        if (
            c.get("kind") not in kinds
            or c.get("correct_kind") not in kinds
            or any(
                type(c.get(key)) is not bool
                for key in ("atomic", "classification_valid", "source_faithful")
            )
            or not isinstance(c.get("explanation"), str)
        ):
            raise ValueError("inventory_recovery_claim_identity_invalid")
        unit, fragment = fragments.get(c.get("source_fragment_id"), (None, None))
        start, end = c.get("source_start"), c.get("source_end")
        if (
            unit is None
            or unit.unit_id != c.get("source_unit_id")
            or type(start) is not int
            or type(end) is not int
            or (start, end)
            != (unit.start + fragment["start"], unit.start + fragment["end"])
            or survey[start:end] != c.get("source_quote")
            or not isinstance(c.get("claim_text"), str)
            or not c["claim_text"].strip()
        ):
            raise ValueError("inventory_recovery_source_identity_invalid")
        if extract_evidence_refs_ordered(survey[start:end]) != c.get("evidence_ids"):
            raise ValueError("inventory_recovery_citation_identity_invalid")
        by_id[cid] = c

    inventory = stages.get("fragment_inventory")
    if not isinstance(inventory, list):
        raise ValueError("inventory_recovery_source_identity_invalid")
    seen_fragments = set()
    for row in inventory:
        fid = row.get("source_fragment_id")
        unit, _ = fragments.get(fid, (None, None))
        actual = [c for c in claims if c["source_fragment_id"] == fid]
        if (
            unit is None
            or fid in seen_fragments
            or row.get("source_unit_id") != unit.unit_id
            or type(row.get("assertion_count")) is not int
            or row["assertion_count"] != len(actual)
            or stable
            and not isinstance(row.get("exclusion_reason"), str)
        ):
            raise ValueError("inventory_recovery_source_identity_invalid")
        seen_fragments.add(fid)
    expected = (
        {f["source_fragment_id"] for fs in native.values() for f in fs}
        if native
        else set(fragments) if stable else {c["source_fragment_id"] for c in claims}
    )
    if seen_fragments != expected:
        raise ValueError("inventory_recovery_source_identity_invalid")

    findings, affected = [], set()
    issues = details.get("global_issues", [])
    if not isinstance(issues, list):
        raise ValueError("inventory_recovery_source_identity_invalid")
    allowed = {"non_atomic", "classification", "source_fidelity", "missing_assertion"}
    for index, issue in enumerate(issues):
        if issue.get("owner") not in {"inventory", "report"}:
            raise ValueError("inventory_recovery_claim_identity_invalid")
        if issue.get("owner") != "inventory":
            continue
        ids = issue.get("claim_ids")
        uid = issue.get("source_unit_id")
        if (
            issue.get("code") not in allowed
            or uid not in units
            or not isinstance(ids, list)
            or len(set(ids)) != len(ids)
            or (not ids and issue["code"] != "missing_assertion")
            or not isinstance(issue.get("explanation"), str)
            or not issue["explanation"].strip()
            or any(
                cid not in by_id or by_id[cid]["source_unit_id"] != uid for cid in ids
            )
        ):
            raise ValueError("inventory_recovery_claim_identity_invalid")
        fids = (
            list(dict.fromkeys(by_id[cid]["source_fragment_id"] for cid in ids))
            if ids
            else [fid for fid, (unit, _) in fragments.items() if unit.unit_id == uid]
        )
        if not fids:
            raise ValueError("inventory_recovery_source_identity_invalid")
        findings.append(
            {
                "origin": "global",
                "issue_index": index,
                "source_unit_id": uid,
                "source_fragment_ids": fids,
                "claim_ids": ids,
                "code": issue["code"],
                "explanation": issue["explanation"],
            }
        )
        affected.update(ids)
    for c in claims:
        for flag, code in (
            ("atomic", "non_atomic"),
            ("classification_valid", "classification"),
            ("source_faithful", "source_fidelity"),
        ):
            if c.get(flag) is not False:
                continue
            findings.append(
                {
                    "origin": "batch",
                    "source_unit_id": c["source_unit_id"],
                    "source_fragment_ids": [c["source_fragment_id"]],
                    "claim_ids": [c["claim_id"]],
                    "code": code,
                    "explanation": c["explanation"],
                }
            )
            affected.add(c["claim_id"])
    raw_findings = stages.get("global_finding_bindings", [])
    if not isinstance(raw_findings, list):
        raise ValueError("inventory_recovery_claim_identity_invalid")
    for index, finding in enumerate(raw_findings):
        if (
            not isinstance(finding, dict)
            or set(finding)
            != {"finding_index", "code", "anchor_ids", "explanation", "source_unit_ids"}
            or type(finding["finding_index"]) is not int
            or finding["finding_index"] != index
            or not isinstance(finding["code"], str)
            or not isinstance(finding["explanation"], str)
            or not isinstance(finding["anchor_ids"], list)
            or not finding["anchor_ids"]
        ):
            raise ValueError("inventory_recovery_claim_identity_invalid")
        parents = []
        for anchor in finding["anchor_ids"]:
            if not isinstance(anchor, str):
                raise ValueError("inventory_recovery_claim_identity_invalid")
            if anchor.startswith("claim:") and anchor[6:] in by_id:
                affected.add(anchor[6:])
                parents.append(by_id[anchor[6:]]["source_unit_id"])
            elif anchor.startswith("unit:") and anchor[5:] in units:
                parents.append(anchor[5:])
            else:
                raise ValueError("inventory_recovery_claim_identity_invalid")
        if list(dict.fromkeys(parents)) != finding["source_unit_ids"]:
            raise ValueError("inventory_recovery_claim_identity_invalid")
    if not findings:
        return None
    fields = (
        "claim_id",
        "source_unit_id",
        "source_fragment_id",
        "claim_text",
        "kind",
        "correct_kind",
        "atomic",
        "classification_valid",
        "source_faithful",
        "evidence_ids",
    )
    editable = {fid for finding in findings for fid in finding["source_fragment_ids"]}
    preserved = []
    for row in inventory if stable else []:
        fid = row["source_fragment_id"]
        if fid in editable:
            continue
        unit, fragment = fragments[fid]
        preserved.append(
            {
                "source_unit_id": unit.unit_id,
                "source_fragment_id": fid,
                "source_start": unit.start + fragment["start"],
                "source_end": unit.start + fragment["end"],
                "evidence_ids": extract_evidence_refs_ordered(
                    survey[
                        unit.start + fragment["start"] : unit.start + fragment["end"]
                    ]
                ),
                "assertions": [
                    {key: claim[key] for key in ("kind", "claim_text")}
                    for claim in claims
                    if claim["source_fragment_id"] == fid
                ],
                "exclusion_reason": row["exclusion_reason"],
            }
        )
    return {
        "contract": MIXED_CONTRACT,
        "source_report_sha256": sha,
        "previous_inventory_sha256": inventory_identity(claims),
        "bindings_sha256": digest(
            [
                [c["claim_id"], c["source_start"], c["source_end"], c["evidence_ids"]]
                for c in claims
            ]
        ),
        "writer_bindings_sha256": digest(bindings),
        "assessment_sha256": digest([claims, issues, requests]),
        "findings": findings,
        "concern_claims": [
            {k: c[k] for k in fields if k in c}
            for c in claims
            if c["claim_id"] in affected
        ],
        "global_finding_bindings": copy.deepcopy(raw_findings),
        **(
            {
                "preserved_fragments": preserved,
                "preserved_fragments_sha256": digest(preserved),
            }
            if stable
            else {}
        ),
    }


def _stability_error(code, feedback, *, check, expected=None, actual=None, claims=None):
    """Locate a rejected extraction without exposing any assertion text."""
    from litagent.eval.scoped import ScopedEvaluationError

    def text(value):
        return value if isinstance(value, str) else None

    def content_sha(row, assertions):
        if not row and assertions is None:
            return None
        try:
            return digest(
                {
                    "assertions": assertions,
                    "exclusion_reason": (row or {}).get("exclusion_reason"),
                }
            )
        except (TypeError, ValueError):
            return None

    expected = expected if isinstance(expected, dict) else {}
    actual = actual if isinstance(actual, dict) else {}
    before = expected.get("assertions")
    after = (
        [{key: claim.get(key) for key in ("kind", "claim_text")} for claim in claims]
        if isinstance(claims, list) and all(isinstance(c, dict) for c in claims)
        else None
    )
    return ScopedEvaluationError(
        code,
        {
            "inventory_stability_check": {
                "status": "failed",
                "reason_code": code,
                "check": check,
                "source_report_sha256": text(feedback.get("source_report_sha256")),
                "source_fragment_id": text(
                    expected.get("source_fragment_id", actual.get("source_fragment_id"))
                ),
                "source_unit_id": text(
                    expected.get("source_unit_id", actual.get("source_unit_id"))
                ),
                "actual_source_fragment_id": text(actual.get("source_fragment_id")),
                "actual_source_unit_id": text(actual.get("source_unit_id")),
                "expected_count": len(before) if isinstance(before, list) else None,
                "actual_count": len(claims) if isinstance(claims, list) else None,
                "expected_content_sha256": content_sha(expected, before),
                "actual_content_sha256": content_sha(actual, after),
            }
        },
    )


def preserved_fragment_directory(feedback):
    """Validate an optional host baseline without inferring any scientific verdict."""
    keys = ("preserved_fragments", "preserved_fragments_sha256")
    if not any(key in feedback for key in keys):
        return None
    row, check = None, "directory_digest"
    try:
        rows = feedback["preserved_fragments"]
        if not isinstance(rows, list) or feedback[
            "preserved_fragments_sha256"
        ] != digest(rows):
            raise ValueError("directory digest")
        editable = {
            fid
            for finding in feedback["findings"]
            for fid in finding["source_fragment_ids"]
        }
        seen = set()
        for row in rows:
            check = "directory_shape"
            if not isinstance(row, dict) or set(row) != set(PRESERVED_FRAGMENT_FIELDS):
                raise ValueError("directory shape")
            check = "directory_identity"
            fid = row["source_fragment_id"]
            if (
                any(
                    not isinstance(row[key], str) or not row[key]
                    for key in ("source_fragment_id", "source_unit_id")
                )
                or fid in seen
                or fid in editable
                or type(row["source_start"]) is not int
                or type(row["source_end"]) is not int
                or not 0 <= row["source_start"] < row["source_end"]
                or not isinstance(row["evidence_ids"], list)
                or any(
                    not isinstance(eid, str) or not eid for eid in row["evidence_ids"]
                )
                or not isinstance(row["assertions"], list)
                or not isinstance(row["exclusion_reason"], str)
            ):
                raise ValueError("directory identity")
            for claim in row["assertions"]:
                check = "assertion_identity"
                if (
                    not isinstance(claim, dict)
                    or set(claim) != {"kind", "claim_text"}
                    or claim["kind"]
                    not in {"scientific", "runtime_method", "report_structure"}
                    or not isinstance(claim["claim_text"], str)
                    or not claim["claim_text"].strip()
                ):
                    raise ValueError("assertion identity")
            seen.add(fid)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        raise _stability_error(
            "inventory_recovery_stability_identity_invalid",
            feedback,
            check=check,
            expected=row,
        ) from exc
    return rows


def validate_recovered_inventory(
    claims, feedback, *, fragment_inventory=None, allow_unchanged=False
):
    """Reject unrequested extraction drift, equivalent inventories and cycles."""
    if feedback.get("contract") != MIXED_CONTRACT:
        raise ValueError("inventory_recovery_policy_invalid")
    preserved = preserved_fragment_directory(feedback)
    if preserved is not None:
        expected_row, entry, selected, check = None, None, None, "fragment_inventory"
        try:
            if not isinstance(fragment_inventory, list):
                raise ValueError("fragment inventory missing")
            actual = {}
            for row in fragment_inventory:
                entry = row
                fid = row["source_fragment_id"]
                if fid in actual:
                    check = "duplicate_fragment"
                    raise ValueError("duplicate fragment")
                actual[fid] = row
            for row in preserved:
                expected_row, entry, selected = row, None, None
                fid = row["source_fragment_id"]
                check = "fragment_coverage"
                selected = [c for c in claims if c["source_fragment_id"] == fid]
                entry = actual[fid]
                check = "preserved_content_or_binding"
                if (
                    entry["source_unit_id"] != row["source_unit_id"]
                    or type(entry["assertion_count"]) is not int
                    or entry["assertion_count"] != len(selected)
                    or entry["exclusion_reason"] != row["exclusion_reason"]
                    or [
                        {key: c[key] for key in ("kind", "claim_text")}
                        for c in selected
                    ]
                    != row["assertions"]
                    or any(
                        any(
                            c[key] != row[key]
                            for key in (
                                "source_unit_id",
                                "source_start",
                                "source_end",
                                "evidence_ids",
                            )
                        )
                        for c in selected
                    )
                ):
                    raise ValueError("preserved fragment changed")
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            raise _stability_error(
                "inventory_recovery_preserved_fragment_changed",
                feedback,
                check=check,
                expected=expected_row,
                actual=entry,
                claims=selected,
            ) from exc
    previous = feedback["previous_inventory_sha256"]
    seen = feedback.get("seen_inventory_sha256", [])
    if not isinstance(seen, list) or any(
        not isinstance(value, str) or len(value) != 64 for value in seen
    ):
        raise ValueError("inventory_recovery_source_identity_invalid")
    if type(allow_unchanged) is not bool:
        raise ValueError("inventory_recovery_policy_invalid")
    if not allow_unchanged and inventory_identity(claims) in {previous, *seen}:
        raise ValueError("inventory_recovery_no_progress")


@contextmanager
def recovery_scope(feedback):
    """Explicitly identify the one repair slot used for complete re-extraction."""
    token = _active.set(feedback)
    try:
        yield
    finally:
        _active.reset(token)


def recovery_wire_binding(owner, messages, output_tokens, response_format):
    """Check a settled parent before binding a real Lite request to repair funds."""
    from litagent.capacity_rounds import request_proof

    feedback = _active.get()
    if feedback is None:
        return None
    parent = owner.allocation
    if (
        owner.plan["contract"] != "planned-budget-v7"
        or owner.plan["envelope"].get("inventory_recovery_contract")
        not in RECOVERY_CONTRACTS
        or owner._owner()[0] != "repair"
        or feedback.get("contract")
        != owner.plan["envelope"]["inventory_recovery_contract"]
        or output_tokens != 16384
        or not parent
        or parent["round"]["mode"] != "assess"
        or parent["round"]["report_sha256"] != feedback["source_report_sha256"]
        or any(e["status"] != "settled" for e in owner.entries.values())
        or len(owner._stage_entries("global"))
        != parent["round"]["offsets"]["global"] + 1
        or len(owner._stage_entries("assessment"))
        != parent["round"]["offsets"]["assessment"] + parent["round"]["batches"]
    ):
        raise ValueError("inventory_recovery_budget_identity_invalid")
    binding = {
        "contract": feedback["contract"],
        "parent_allocation": parent["identity"],
        "source_report_sha256": feedback["source_report_sha256"],
        "feedback_sha256": digest(feedback),
        "wire_sha256": request_proof(
            "repair", messages, output_tokens, response_format
        )["wire_sha256"],
    }
    owner._checked_inventory_recovery = (
        owner._owner(),
        binding,
        request_proof("repair", messages, output_tokens, response_format)[
            "reserved_tokens"
        ],
    )
    return binding


def validate_recovery_binding(owner, entry, previous):
    """Reject forged, stale or old-policy recovery boundaries during replay."""
    value = entry.get("inventory_recovery")
    if value is None:
        return False
    if (
        owner.plan["contract"] != "planned-budget-v7"
        or owner.plan["envelope"].get("inventory_recovery_contract")
        not in RECOVERY_CONTRACTS
        or entry.get("stage") != "repair"
        or not isinstance(value, dict)
        or set(value)
        != {
            "contract",
            "parent_allocation",
            "source_report_sha256",
            "feedback_sha256",
            "wire_sha256",
        }
        or value["contract"] != owner.plan["envelope"]["inventory_recovery_contract"]
        or not previous
        or value["parent_allocation"] != previous["identity"]
        or value["source_report_sha256"] != previous["round"]["report_sha256"]
        or any(
            not isinstance(value[k], str)
            or len(value[k]) != 64
            or any(c not in "0123456789abcdef" for c in value[k])
            for k in value
            if k != "contract"
        )
    ):
        raise ValueError("inventory_recovery_budget_identity_invalid")
    return True


async def recover_report_inventory(runner, *, report_data, results, query, feedback):
    """Consume one shared attempt, preserving prose and all failed evaluations."""
    from litagent.quality import (
        begin_targeted_repair,
        emit_targeted_repair,
        load_targeted_repairs,
        save_targeted_repairs,
        targeted_repair_state,
    )
    from litagent.runner import RewriteOutcome
    from litagent.safety.planned_budget import nested_json_size, planned_backend

    owner = planned_backend(runner._cost_budget)
    state = targeted_repair_state(
        report_data["metadata"].get("targeted_repair_state"), runner._session_id
    )
    stored = load_targeted_repairs(runner._targeted_repair_path(), runner._session_id)
    if stored != state:
        raise ValueError("strict_targeted_repair_checkpoint_conflict")
    from litagent.closeout import recovery_allowed

    if not recovery_allowed(owner.effective_envelope, state):
        return RewriteOutcome(
            False, False, None, None, None, None, "inventory_recovery_limit_reached"
        )
    feedback = {**feedback, "attempt_index": len(state["attempts"])}
    if feedback["contract"] == MIXED_CONTRACT:
        feedback["seen_inventory_sha256"] = list(
            dict.fromkeys(
                e["previous_inventory_sha256"]
                for e in state["attempts"]
                if e.get("purpose") == MIXED_CONTRACT
                and e.get("source_report_sha256") == feedback["source_report_sha256"]
                and e.get("previous_inventory_sha256")
            )
        )
    elif nested_json_size(feedback) > owner.effective_envelope["diagnostic_json_bytes"]:
        return RewriteOutcome(
            False,
            False,
            None,
            None,
            None,
            None,
            "inventory_recovery_diagnostic_capacity_exceeded",
        )
    entry = None

    def before_send(request):
        nonlocal entry
        entry = begin_targeted_repair(state, "evaluation", request["input_sha256"])
        entry.update(
            purpose=feedback["contract"],
            source_report_sha256=feedback["source_report_sha256"],
        )
        if feedback["contract"] == MIXED_CONTRACT:
            entry["previous_inventory_sha256"] = feedback["previous_inventory_sha256"]
        report_data["metadata"]["targeted_repair_state"] = state
        save_targeted_repairs(runner._targeted_repair_path(), state)
        emit_targeted_repair(runner._trace_hook, state, "evaluation.inventory_recovery")

    try:
        evaluation = await runner._evaluate(
            query=query,
            survey=report_data["survey"],
            results=results,
            phase="post_rewrite",
            inventory_recovery=feedback,
            recovery_on_send=before_send,
        )
    except BaseException as exc:
        if entry is not None:
            import asyncio

            entry["status"] = (
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            )
            save_targeted_repairs(runner._targeted_repair_path(), state)
        raise
    details = evaluation.get("faithfulness", {}).get("details", {})
    requests = details.get("stages", {}).get("requests", [])
    from litagent.closeout import profile_fields, recovery_summary

    if profile_fields(owner.effective_envelope):
        review = recovery_summary(feedback, evaluation)
        report_data["metadata"].setdefault("inventory_recovery_reviews", []).append(
            review
        )
        if entry is not None:
            entry["inventory_review"] = review
    if entry is not None:
        request = requests[0] if requests else {}
        entry.update(
            status=(
                "completed"
                if details.get("assessment_stage") == "host_validated"
                else "failed"
            ),
            usage=copy.deepcopy(request.get("usage", {})),
            outcome=details.get("unverified_reason")
            or details.get("skipped_reason")
            or "evaluated",
        )
        save_targeted_repairs(
            runner._targeted_repair_path(),
            state,
            **{
                f"inventory_recovery_{entry['index']}": {
                    "feedback": feedback,
                    "evaluation": evaluation,
                    "response": request.get("response"),
                }
            },
        )
    quality = runner._derive_quality(evaluation)
    terminal = details.get("skipped_reason")
    if terminal == "inventory_recovery_no_progress" or (
        feedback["contract"] == MIXED_CONTRACT
        and terminal == "inventory_recovery_request_capacity_exceeded"
    ):
        report_data["metadata"]["writer_failure"] = {
            "family": (
                "resource"
                if terminal == "inventory_recovery_request_capacity_exceeded"
                else "quality"
            ),
            "code": terminal,
            "source_report_sha256": feedback["source_report_sha256"],
        }
    valid = (
        details.get("assessment_stage") == "host_validated"
        and details.get("inventory_check", {}).get("status") == "passed"
        and details.get("stages", {}).get("global_check") == "completed"
    )
    retryable = (
        details.get("inventory_check", {}).get("status") == "failed"
        and details.get("assessment_stage") == "host_validated"
    )
    return RewriteOutcome(
        entry is not None,
        False,
        report_data["survey"],
        None,
        evaluation,
        quality,
        (
            terminal
            if feedback["contract"] == MIXED_CONTRACT
            and terminal == "inventory_recovery_request_capacity_exceeded"
            else (
                "inventory_recovery_evaluated"
                if valid
                else (
                    "inventory_recovery_retryable"
                    if retryable
                    else "inventory_recovery_failed"
                )
            )
        ),
    )
