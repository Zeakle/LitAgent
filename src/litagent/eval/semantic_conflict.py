"""Keep source-bound semantic conflicts distinct from model verdicts."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from collections.abc import Mapping

SEMANTIC_CONFLICT_POLICY = "host-unverified-continue-v1"
EFFECT_CONTRACT = "host-assessment-conflict-effect-v1"
CONFLICT_CODES = frozenset(
    {
        "positive_status_with_negative_required_aspect",
        "negative_status_with_complete_positive_diagnostics",
    }
)
POLICY_FIELDS = ("semantic_conflict_policy", "semantic_conflict_identity")
EFFECT_FIELDS = frozenset(
    {
        "contract",
        "policy",
        "policy_identity",
        "code",
        "original_status",
        "original_reason_code",
        "decision_sha256",
        "binding_sha256",
    }
)
_PARENTS = {
    "contract": "flat-writer-envelope-v1",
    "repair_mapping_contract": "capacity-source-map-v4",
    "global_findings_contract": "anchored-global-findings-v2",
    "inventory_recovery_contract": "bounded-inventory-recovery-v2",
    "summary_constraints_contract": "effective-summary-limits-v1",
    "classification_review_contract": "independent-assertion-kinds-v1",
    "repair_feedback_contract": "host-repair-feedback-v1",
    "reference_contract": "ordered-reference-set-v1",
    "transport_revision": "host-lossless-transport-v1",
    "support_validation_revision": "host-support-receipts-v2",
    "repair_guidance_contract": "host-structured-repair-v1",
    "repair_guidance_wire_revision": "host-repair-task-view-v1",
}


def _digest(value) -> str:
    """Use the existing host JSON identity serialization."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def semantic_conflict_identity() -> str:
    """Bind the complete selected policy while leaving support-v2 unchanged."""
    return _digest(
        [
            SEMANTIC_CONFLICT_POLICY,
            EFFECT_CONTRACT,
            sorted(CONFLICT_CODES),
            sorted(EFFECT_FIELDS),
            _PARENTS,
            "all-ids-source-owned-basis-exact-aspects-settled-before-two-codes-v1",
            "original-decisions-retained-effect-never-supports-single-denominator-v1",
            "full-all-batches-global-inventory-owner-replay-before-two-repairs-v1",
            *[
                inspect.getsource(function)
                for function in (
                    validate_settlement_wrapper,
                    validate_conflict_inputs,
                    claim_anchor,
                    make_effect,
                    validate_effect,
                    host_effects,
                )
            ],
        ]
    )


def selected_policy(envelope) -> bool:
    """Require an explicit immutable pair and the complete current parent stack."""
    if envelope is None:
        return False
    if not isinstance(envelope, Mapping):
        raise ValueError("evaluation_semantic_conflict_policy_invalid")
    if not any(key in envelope for key in POLICY_FIELDS):
        return False
    if (
        envelope.get(POLICY_FIELDS[0]) != SEMANTIC_CONFLICT_POLICY
        or envelope.get(POLICY_FIELDS[1]) != semantic_conflict_identity()
        or any(envelope.get(key) != value for key, value in _PARENTS.items())
        or not isinstance(envelope.get("capacity"), Mapping)
    ):
        raise ValueError("evaluation_semantic_conflict_policy_invalid")
    return True


def semantic_conflict_fields(envelope) -> dict:
    """Omit the pair entirely for every unselected historical path."""
    return (
        {key: envelope[key] for key in POLICY_FIELDS}
        if selected_policy(envelope)
        else {}
    )


def validate_settlement_wrapper(settlement, *, envelope, payload=None) -> dict:
    """Validate the locally bound receipt shape; owner authentication is separate."""
    if not selected_policy(envelope) or not isinstance(settlement, dict):
        raise ValueError("evaluation_semantic_conflict_settlement_invalid")
    expected = {
        "receipt",
        "report_sha256",
        "response_sha256",
        "input_sha256",
        "batch_payload_sha256",
    }
    if set(settlement) != expected or any(
        not _hash(settlement[key]) for key in expected - {"receipt"}
    ):
        raise ValueError("evaluation_semantic_conflict_settlement_invalid")
    receipt = settlement["receipt"]
    if (
        not isinstance(receipt, dict)
        or receipt.get("contract") != "host-settled-assessment-receipt-v1"
        or receipt.get("stage") != "assessment"
        or receipt.get("semantic_conflict_policy") != SEMANTIC_CONFLICT_POLICY
        or receipt.get("semantic_conflict_identity") != semantic_conflict_identity()
        or not _hash(receipt.get("identity"))
        or receipt["identity"]
        != _digest({k: v for k, v in receipt.items() if k != "identity"})
    ):
        raise ValueError("evaluation_semantic_conflict_settlement_invalid")
    if payload is not None and settlement["batch_payload_sha256"] != _digest(payload):
        raise ValueError("evaluation_semantic_conflict_settlement_invalid")
    return settlement


def validate_conflict_inputs(claim: dict, decision, payload: dict) -> None:
    """Check source permissions before allowing either semantic-conflict effect."""
    from litagent.eval.scoped import required_support_aspects

    def require(condition):
        if not condition:
            raise ValueError("evaluation_semantic_conflict_input_invalid")

    require(selected_policy(payload.get("writer_envelope")))
    require(
        claim.get("kind") == decision.correct_kind == "scientific"
        and decision.atomic is True
        and decision.source_faithful is True
        and decision.classification_valid is True
    )
    required = required_support_aspects(
        claim["claim_text"],
        applicability=True,
        support_validation_revision="host-support-receipts-v2",
    )
    require(claim.get("required_support_aspects") == required)
    names = [aspect.aspect for aspect in decision.aspect_checks]
    require(len(names) == len(set(names)) and set(names) == set(required))
    parents = {}
    for unit in payload["source_units"]:
        for fragment in unit["source_fragments"]:
            fid = fragment["source_fragment_id"]
            require(fid not in parents)
            parents[fid] = (unit["source_unit_id"], fragment)
    require(claim["source_fragment_id"] in parents)
    uid, fragment = parents[claim["source_fragment_id"]]
    require(uid == claim["source_unit_id"])
    cited = [entry["evidence_alias"] for entry in claim["evidence"]]
    require(bool(cited) and len(cited) == len(set(cited)))
    require(set(cited).issubset(fragment["citation_aliases"]))
    basis = [entry.support_id for entry in decision.support_basis]
    require(bool(basis) and len(basis) == len(set(basis)))
    for pid in basis:
        require(pid in payload["support_passages"])
        passage = payload["support_passages"][pid]
        require(isinstance(passage, (list, tuple)) and len(passage) in (4, 5))
        alias, source, start, end = passage[:4]
        require(alias in cited and alias in payload["cited_materials"])
        require(source in payload["original_supports"])
        original = payload["original_supports"][source]
        material = payload["cited_materials"][alias]
        require(isinstance(original, dict) and isinstance(material, dict))
        text = original.get("text")
        require(
            material.get("original_support_ref") == source
            and material.get("paper_id") == original.get("paper_id")
            and isinstance(text, str)
            and bool(text.strip())
            and type(start) is int
            and type(end) is int
            and 0 <= start < end <= len(text)
        )
        if len(passage) == 5:
            # The fifth field is a host display hint. Support still resolves to
            # the complete retained original range, never the short preview.
            require(
                type(passage[4]) is dict
                and set(passage[4]) == {"display_preview"}
                and passage[4]["display_preview"] == text[start : min(end, start + 80)]
            )
    for aspect in decision.aspect_checks:
        require(len(aspect.support_ids) == len(set(aspect.support_ids)))
        require(set(aspect.support_ids).issubset(basis))
        require(not aspect.supported or bool(aspect.support_ids))


def claim_anchor(claim: dict, *, aliases=None) -> list:
    """Use the same report-owned identity for batch and rich host assertions."""
    from litagent.eval.scoped import required_support_aspects

    if "evidence" in claim:
        cited = [entry["evidence_alias"] for entry in claim["evidence"]]
    else:
        if aliases is None:
            raise ValueError("evaluation_semantic_conflict_binding_invalid")
        cited = [aliases[eid] for eid in claim["evidence_ids"]]
    return [
        *[
            claim[key]
            for key in (
                "claim_id",
                "kind",
                "claim_text",
                "source_unit_id",
                "source_fragment_id",
            )
        ],
        cited,
        required_support_aspects(
            claim["claim_text"],
            applicability=True,
            support_validation_revision="host-support-receipts-v2",
        ),
    ]


def make_effect(claim, decision, *, code, settlement, aliases=None) -> dict:
    """Describe a conservative host effect without changing any model verdict."""
    if code not in CONFLICT_CODES:
        raise ValueError("evaluation_semantic_conflict_effect_invalid")
    original = decision.model_dump() if hasattr(decision, "model_dump") else decision
    checks = original["aspect_checks"]
    names = [value["aspect"] for value in checks]
    required = claim_anchor(claim, aliases=aliases)[-1]
    positive = code == "positive_status_with_negative_required_aspect"
    if (
        claim["kind"] != "scientific"
        or original["correct_kind"] != "scientific"
        or not all(
            original[key] is True
            for key in ("atomic", "classification_valid", "source_faithful")
        )
        or len(names) != len(set(names))
        or set(names) != set(required)
        or not checks
        or (
            positive
            and (
                original["status"] != "supported"
                or not any(not value["supported"] for value in checks)
            )
        )
        or (
            not positive
            and (
                original["status"] != "unsupported"
                or not all(value["supported"] for value in checks)
            )
        )
    ):
        raise ValueError("evaluation_semantic_conflict_effect_invalid")
    decision_sha = _digest(original)
    return {
        "contract": EFFECT_CONTRACT,
        "policy": SEMANTIC_CONFLICT_POLICY,
        "policy_identity": semantic_conflict_identity(),
        "code": code,
        "original_status": original["status"],
        "original_reason_code": original["reason_code"],
        "decision_sha256": decision_sha,
        "binding_sha256": _digest(
            [claim_anchor(claim, aliases=aliases), decision_sha, settlement]
        ),
    }


def validate_effect(effect, claim, decision, *, settlement, aliases=None) -> dict:
    """Reject altered or borrowed effects even when their status is conservative."""
    if not isinstance(effect, dict) or set(effect) != EFFECT_FIELDS:
        raise ValueError("evaluation_semantic_conflict_effect_invalid")
    expected = make_effect(
        claim, decision, code=effect["code"], settlement=settlement, aliases=aliases
    )
    if effect != expected:
        raise ValueError("evaluation_semantic_conflict_effect_invalid")
    return expected


def host_effects(progress, claims, decisions, *, aliases) -> dict:
    """Bind conservative effects to the received report, batch and raw verdict."""
    envelope = progress.get("writer_envelope")
    if not selected_policy(envelope):
        return {}
    if any(progress.get(key) != envelope[key] for key in POLICY_FIELDS):
        raise ValueError("evaluation_semantic_conflict_binding_invalid")
    indexed = {claim["claim_id"]: claim for claim in claims}
    effects, observed = {}, set()
    for batch in progress.get("batches", []):
        if batch.get("status") != "completed":
            continue
        settlement = validate_settlement_wrapper(
            batch.get("semantic_conflict_settlement"), envelope=envelope
        )
        matches = [
            entry
            for entry in progress.get("requests", [])
            if entry.get("phase") == "assessment"
            and entry.get("status") == "response_received"
            and entry.get("semantic_conflict_settlement") == settlement
        ]
        if (
            len(matches) != 1
            or settlement["report_sha256"] != progress.get("source_report_sha256")
            or settlement["batch_payload_sha256"] != batch.get("input_sha256")
            or settlement["input_sha256"] != matches[0].get("input_sha256")
            or settlement["response_sha256"]
            != hashlib.sha256(matches[0]["response"].encode()).hexdigest()
        ):
            raise ValueError("evaluation_semantic_conflict_binding_invalid")
        raw = {entry["claim_id"]: entry for entry in batch.get("decisions", [])}
        entries = batch.get("assessment_conflict_effects")
        if not isinstance(entries, list):
            raise ValueError("evaluation_semantic_conflict_effect_invalid")
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != EFFECT_FIELDS | {
                "claim_id"
            }:
                raise ValueError("evaluation_semantic_conflict_effect_invalid")
            cid = entry["claim_id"]
            if cid in observed or cid not in raw or cid not in batch["claim_ids"]:
                raise ValueError("evaluation_semantic_conflict_binding_invalid")
            observed.add(cid)
            if cid not in indexed:
                continue
            decision = decisions[cid]
            if raw[cid] != decision.model_dump():
                raise ValueError("evaluation_semantic_conflict_binding_invalid")
            effects[cid] = validate_effect(
                {key: value for key, value in entry.items() if key != "claim_id"},
                indexed[cid],
                decision,
                settlement=settlement,
                aliases=aliases,
            )
    return effects
