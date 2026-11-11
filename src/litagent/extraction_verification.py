"""Verify selected extraction candidates against their original sources before writing."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import re
from collections.abc import Mapping
from pathlib import Path

from litagent.benchmark.artifacts import BenchmarkArtifactRepository
from litagent.benchmark.citation_audit import JevAuditConfig, JevCitationAuditor
from litagent.benchmark.survey_models import AnchoredJudgeClaim, JudgeInventorySnapshot
from litagent.context.evidence_selector import EvidenceSelection
from litagent.quality import digest
from litagent.report_audit import atomic_json

VERSION = "jev-extraction-verification-v1"
ADMISSION_VERSION = "jev-extraction-admission-v2"
FALLBACK_POLICY = "verified_subset_v1"
_ADMISSION_V2_IDENTITY = (
    "47d56c886ccc4d8c2080eb7cc44e88d8a92a761caaf4462fdc32c9a0bec69a2c"
)
_FALLBACK_V1_IDENTITY = (
    "95f95e52f1554e30785d61eb0066cde88781b16c5445fe3e3db304010142b1e6"
)
_TRANSIENT_HTTP = {f"jev_http_{code}" for code in (429, 500, 502, 503, 504)}
RESOURCE_POLICY = "expanded-v1"
UPLIFT_RESOURCE_POLICY = "operational-plus20-v1"
_EXPANDED_V1_RESOURCE_IDENTITY = (
    "c45c98bc3c243550ed401d00abe53f039da42f68baae61ae926383c6591f0c3f"
)
_CONFIDENCE_V1_IDENTITY = (
    "9cd436df5355008cab3984b701468c8c3a3081d849028bb85c8e5969bb449f38"
)


def admission_timeout_ms(resource_policy=None) -> int:
    """Add all bounded Jev dispatch waves to the extraction task deadline."""
    config = extraction_config(resource_policy=resource_policy)
    waves = (config.max_requests + config.concurrency - 1) // config.concurrency
    deadline = int((waves * config.timeout_seconds + 10) * 1000)
    if resource_policy == UPLIFT_RESOURCE_POLICY:
        from litagent.operational_limits import increase

        deadline = max(deadline, increase(admission_timeout_ms(RESOURCE_POLICY)))
    return deadline


def admission_identity() -> str:
    """Preserve the original strict v2 identity; bind new policy separately."""
    return _ADMISSION_V2_IDENTITY


def fallback_identity() -> str:
    """Preserve receipts made before explicit confidence policies were introduced."""
    return _FALLBACK_V1_IDENTITY


def extraction_config(
    confidence_threshold=None, *, protocol=ADMISSION_VERSION, resource_policy=None
):
    """Keep omitted thresholds at the historical 0.9 and validate explicit values."""
    if confidence_threshold is not None and (
        protocol != ADMISSION_VERSION
        or type(confidence_threshold) not in (int, float)
        or not 0 <= confidence_threshold <= 1
    ):
        raise ValueError("extraction_confidence_threshold_invalid")
    if resource_policy is not None and (
        resource_policy not in (RESOURCE_POLICY, UPLIFT_RESOURCE_POLICY)
        or protocol != ADMISSION_VERSION
    ):
        raise ValueError("extraction_resource_policy_invalid")
    values = {"enabled": True}
    if confidence_threshold is not None:
        values["confidence_threshold"] = confidence_threshold
    if resource_policy:
        values.update(
            max_questions_per_batch=32,
            concurrency=4,
            max_requests=200,
            max_input_tokens=4_000_000,
        )
    if resource_policy == UPLIFT_RESOURCE_POLICY:
        values.update(
            resource_revision=UPLIFT_RESOURCE_POLICY,
            max_questions_per_batch=39,
            concurrency=5,
            max_requests=240,
            max_input_tokens=4_800_000,
            max_request_bytes=28_800,
            timeout_seconds=36.0,
        )
    return JevAuditConfig(**values)


def resource_limits(config: JevAuditConfig) -> dict:
    """Expose bounded resources without endpoints, secrets or scientific decisions."""
    return {
        key: getattr(config, key)
        for key in (
            "max_questions_per_batch",
            "concurrency",
            "max_requests",
            "max_input_tokens",
            "max_request_bytes",
            "timeout_seconds",
            "max_retries",
        )
    }


def confidence_identity() -> str:
    """Preserve the explicit-threshold receipts created before resource profiles."""
    return _CONFIDENCE_V1_IDENTITY


def resource_identity(policy=RESOURCE_POLICY) -> str:
    """Bind expanded resource behavior separately from historical threshold identity."""
    if policy == RESOURCE_POLICY:
        return _EXPANDED_V1_RESOURCE_IDENTITY
    if policy != UPLIFT_RESOURCE_POLICY:
        raise ValueError("extraction_resource_policy_invalid")
    from litagent.benchmark.citation_audit import _question_specs, _request_payload

    return digest(
        [
            policy,
            resource_limits(extraction_config(resource_policy=policy)),
            confidence_identity(),
            fallback_identity(),
            admission_identity(),
            sorted(_TRANSIENT_HTTP),
            *[
                inspect.getsource(fn)
                for fn in (
                    extraction_config,
                    resource_limits,
                    admission_timeout_ms,
                    _receipt_config,
                    _binding,
                    _classify,
                    _classify_available,
                    _fallback_summary,
                    _receipt_classification,
                    validate_extraction_receipt,
                    require_admission_policy,
                    ExtractionVerifier.verify_candidates,
                    ExtractionVerifier.verify,
                    _question_specs,
                    _request_payload,
                )
            ],
        ]
    )


def _stage(version: str) -> str:
    """Select one known question contract without upgrading legacy receipts."""
    if version not in (VERSION, ADMISSION_VERSION):
        raise ValueError("extraction_protocol_invalid")
    return "extraction_admission" if version == ADMISSION_VERSION else "extraction"


def _binding(
    version: str,
    run_id: str,
    selection: Mapping,
    config: JevAuditConfig,
    failure_policy: str | None = None,
    confidence_threshold: float | None = None,
    resource_policy: str | None = None,
) -> str:
    """Keep the historical digest while binding new admission behavior explicitly."""
    values = [version, run_id, selection, config.fingerprint]
    if version == ADMISSION_VERSION:
        values.append(admission_identity())
    if failure_policy is not None:
        if failure_policy != FALLBACK_POLICY or version != ADMISSION_VERSION:
            raise ValueError("extraction_failure_policy_invalid")
        values.append([failure_policy, fallback_identity()])
    if confidence_threshold is not None:
        values.append([confidence_threshold, confidence_identity()])
    if resource_policy is not None:
        values.append([resource_policy, resource_identity(resource_policy)])
    return digest(values)


def _receipt_config(receipt: Mapping) -> JevAuditConfig:
    """Authenticate the recorded threshold without applying today's defaults."""
    threshold = receipt.get("confidence_threshold")
    if "confidence_threshold" not in receipt:
        if "confidence_identity" in receipt:
            raise ValueError("extraction_confidence_identity_invalid")
    elif (
        threshold is None or receipt.get("confidence_identity") != confidence_identity()
    ):
        raise ValueError("extraction_confidence_identity_invalid")
    policy = receipt.get("resource_policy")
    if "resource_policy" in receipt and (
        policy not in (RESOURCE_POLICY, UPLIFT_RESOURCE_POLICY)
        or receipt.get("resource_identity") != resource_identity(policy)
    ):
        raise ValueError("extraction_resource_identity_invalid")
    if policy is None and any(
        k in receipt for k in ("resource_identity", "resource_limits")
    ):
        raise ValueError("extraction_resource_identity_invalid")
    config = extraction_config(
        threshold, protocol=receipt.get("version"), resource_policy=policy
    )
    if policy and receipt.get("resource_limits") != resource_limits(config):
        raise ValueError("extraction_resource_limits_invalid")
    return config


def _sha(value) -> str:
    """Use the citation auditor's canonical identities without trusting supplied hashes."""
    from litagent.benchmark.citation_audit import _stable_sha256

    return _stable_sha256(value)


def build_extraction_inventory(
    ledger: Mapping, *, version=VERSION
) -> JudgeInventorySnapshot:
    """Bind each extracted proposition and all semantic fields to exactly one original."""
    _stage(version)
    claims = []
    for eid, item in ledger.items():
        if (
            not isinstance(eid, str)
            or not isinstance(item, Mapping)
            or item.get("evidence_id") != eid
            or not isinstance(item.get("paper_id"), str)
            or not item["paper_id"].strip()
            or not isinstance(item.get("text"), str)
            or not item["text"].strip()
            or not isinstance(item.get("supporting_text"), str)
            or not item["supporting_text"].strip()
        ):
            raise ValueError("extraction_source_or_claim_missing")
        fields = item.get("fact_fields", {})
        if not isinstance(fields, Mapping) or any(
            value is not None and not isinstance(value, str)
            for value in fields.values()
        ):
            raise ValueError("extraction_fields_invalid")
        if version == ADMISSION_VERSION:
            from litagent.evidence import FACT_FIELD_NAMES

            if set(fields) - FACT_FIELD_NAMES:
                raise ValueError("extraction_fields_invalid")
        proposition = item["text"]
        nonempty = {key: val for key, val in fields.items() if val}
        if nonempty:
            proposition += (
                "\nThe following extracted fields describe this same finding; "
                "every field and its association must be entailed by the original: "
                + json.dumps(nonempty, ensure_ascii=False, sort_keys=True)
            )
        cid = "extract:" + digest([version, eid, item])[:24]
        claims.append(
            AnchoredJudgeClaim(
                claim_id=cid,
                claim_text=proposition,
                source_unit_id=eid,
                source_quote=proposition,
                source_start=0,
                source_end=len(proposition),
                evidence_ids=(eid,),
            )
        )
    if not claims:
        raise ValueError("extraction_candidates_empty")
    return JudgeInventorySnapshot(
        survey_sha256=_sha(dict(ledger)),
        source_units_sha256=_sha([c.source_unit_id for c in claims]),
        ledger_sha256=_sha(dict(ledger)),
        inventory_sha256=_sha([c.model_dump(mode="json") for c in claims]),
        claims=tuple(claims),
    )


def _classify(
    ledger: Mapping, result: Mapping, *, version=VERSION, config=None
) -> tuple[list[str], dict]:
    """Reject incomplete protocols; separate semantic negatives from transport failures."""
    config = config or extraction_config()
    inventory = build_extraction_inventory(ledger, version=version)
    from litagent.benchmark.citation_audit import build_citation_audit_inputs

    expected_inputs = build_citation_audit_inputs(
        inventory, ledger, artifact={"quality_policy": "strict_auto_v2"}
    )
    expected = {
        f"g:{claim.claim_id}": (claim.claim_id, "group", None)
        for claim in inventory.claims
    }
    expected.update(
        {
            f"m:{claim.claim_id}:0": (claim.claim_id, "member", claim.evidence_ids[0])
            for claim in inventory.claims
        }
    )
    if version == ADMISSION_VERSION:
        expected = {
            f"{prefix}:{claim.claim_id}": (claim.claim_id, kind, claim.evidence_ids[0])
            for claim in inventory.claims
            for prefix, kind in (("v", "claim_validity"), ("f", "claim_format"))
        }
    decisions = result.get("decisions", [])
    if (
        result.get("status") != "completed"
        or result.get("actual_model") != "jev-1.13.0"
        or result.get("reason_codes")
        or result.get("usage", {}).get("usage_complete") is not True
        or result.get("inventory_sha256") != inventory.inventory_sha256
        or result.get("ledger_sha256") != inventory.ledger_sha256
        or result.get("planned_question_count") != len(expected)
        or result.get("inputs") != [i.model_dump(mode="json") for i in expected_inputs]
        or len(decisions) != len(expected)
        or {d.get("question_id") for d in decisions} != set(expected)
    ):
        raise ValueError("extraction_verification_incomplete")
    from litagent.benchmark.citation_audit import (
        CitationAuditBatch,
        CitationAuditDecision,
        _question_specs,
        _validate_response,
    )

    specs = {
        spec.question_id: spec
        for spec in _question_specs(
            expected_inputs, verification_stage=_stage(version), candidate_ledger=ledger
        )
    }
    recovered = []
    usage_inputs = usage_outputs = 0
    batch_qids = []
    batches = result.get("batches", [])
    for value in batches:
        batch = CitationAuditBatch.model_validate(value)
        if batch.status != "completed" or batch.actual_model != "jev-1.13.0":
            raise ValueError("extraction_verification_incomplete")
        if any(qid not in specs for qid in batch.question_ids):
            raise ValueError("extraction_decision_invalid")
        batch_specs = [specs[qid] for qid in batch.question_ids]
        from litagent.benchmark.citation_audit import _request_payload

        if batch.request != _request_payload(batch_specs, "jev-1.13.0"):
            raise ValueError("extraction_request_binding_changed")
        parsed, usage, _ = _validate_response(batch.response, batch_specs, config)
        if not usage.usage_complete or usage != batch.usage:
            raise ValueError("extraction_usage_incomplete")
        recovered.extend(d.model_dump(mode="json") for d in parsed)
        usage_inputs += usage.input_tokens
        usage_outputs += usage.output_tokens
        batch_qids.extend(batch.question_ids)
    if (
        not batches
        or len(batch_qids) != len(expected)
        or set(batch_qids) != set(expected)
        or sorted(recovered, key=lambda d: d["question_id"])
        != sorted(decisions, key=lambda d: d["question_id"])
        or result["usage"].get("input_tokens") != usage_inputs
        or result["usage"].get("output_tokens") != usage_outputs
        or result["usage"].get("call_count") != len(batches)
        or usage_inputs > config.max_input_tokens
        or len(batches) > config.max_requests
    ):
        raise ValueError("extraction_decision_or_usage_invalid")
    decisions = {
        d["question_id"]: CitationAuditDecision.model_validate(d) for d in decisions
    }
    accepted, rejected = [], {}
    for claim in inventory.claims:
        eid = claim.evidence_ids[0]
        group = decisions[
            (
                f"v:{claim.claim_id}"
                if version == ADMISSION_VERSION
                else f"g:{claim.claim_id}"
            )
        ]
        member = decisions[
            (
                f"f:{claim.claim_id}"
                if version == ADMISSION_VERSION
                else f"m:{claim.claim_id}:0"
            )
        ]
        for decision in (group, member):
            if (decision.claim_id, decision.kind, decision.evidence_id) != expected[
                decision.question_id
            ]:
                raise ValueError("extraction_decision_invalid")
        reasons = []
        if version == ADMISSION_VERSION:
            if group.choice != "ok":
                reasons.append("extraction_claim_validity_not_ok")
            if member.choice != "ok":
                reasons.append("extraction_claim_format_not_ok")
        elif group.choice != "supported" or member.choice != "contributes":
            reasons.append("extraction_not_supported")
        if any(
            d.uncertain or d.confidence < config.confidence_threshold
            for d in (group, member)
        ):
            reasons.append("extraction_low_confidence")
        if reasons:
            rejected[eid] = reasons
        else:
            accepted.append(eid)
    return accepted, rejected


def _classify_available(
    ledger: Mapping, result: Mapping, *, config=None
) -> tuple[list[str], dict]:
    """Admit only replay-validated pairs after a completely accounted HTTP failure."""
    from litagent.benchmark.citation_audit import (
        CitationAuditResult,
        _failed_response_usage,
        _merge_usage,
        _pack_batches,
        _question_specs,
        _request_payload,
        _validate_response,
        build_citation_audit_inputs,
    )

    config = config or extraction_config()
    if result.get("status") == "completed":
        return _classify(ledger, result, version=ADMISSION_VERSION, config=config)
    audit = CitationAuditResult.model_validate(result)
    inventory = build_extraction_inventory(ledger, version=ADMISSION_VERSION)
    inputs = build_citation_audit_inputs(
        inventory, ledger, artifact={"quality_policy": "strict_auto_v2"}
    )
    specs = _question_specs(
        inputs, verification_stage="extraction_admission", candidate_ledger=ledger
    )
    planned, oversized = _pack_batches(specs, config)
    if (
        audit.status not in {"partial", "failed"}
        or not audit.usage.usage_complete
        or audit.inventory_sha256 != inventory.inventory_sha256
        or audit.ledger_sha256 != inventory.ledger_sha256
        or audit.inputs != inputs
        or audit.planned_question_count != len(specs)
        or audit.config_fingerprint != config.fingerprint
        or audit.requested_model != config.model
        or oversized
        or not audit.batches
        or len(audit.batches) > min(len(planned), config.max_requests)
    ):
        raise ValueError("extraction_fallback_ineligible")
    recovered, reasons, failed = [], [], 0
    for batch, expected in zip(audit.batches, planned):
        if (
            batch.question_ids != [s.question_id for s in expected]
            or batch.request != _request_payload(expected, config.model)
            or batch.requested_model != config.model
        ):
            raise ValueError("extraction_request_binding_changed")
        if batch.status == "completed":
            parsed, usage, model = _validate_response(batch.response, expected, config)
            if batch.reason_codes or batch.actual_model != model:
                raise ValueError("extraction_fallback_ineligible")
            recovered.extend(parsed)
        else:
            if (
                batch.status != "failed"
                or batch.error_type != "HTTPStatusError"
                or len(batch.reason_codes) != 1
                or batch.reason_codes[0] not in _TRANSIENT_HTTP
                or not isinstance(batch.response, dict)
                or batch.response.get("model") not in (None, config.model)
                or batch.actual_model != batch.response.get("model")
            ):
                raise ValueError("extraction_fallback_ineligible")
            usage = _failed_response_usage(batch.response)
            failed += 1
            reasons.extend(batch.reason_codes)
        if not usage.usage_complete or usage != batch.usage:
            raise ValueError("extraction_usage_incomplete")
    # Recompute from original responses, including the cost of failed attempts.
    merged = _merge_usage([batch.usage for batch in audit.batches])
    models = sorted({b.actual_model for b in audit.batches if b.actual_model})
    if (
        not failed
        or audit.reason_codes != list(dict.fromkeys(reasons))
        or audit.usage != merged
        or merged.call_count != len(audit.batches)
        or merged.input_tokens > config.max_input_tokens
        or audit.actual_models != models
        or audit.actual_model != (models[0] if len(models) == 1 else None)
        or len({d.question_id for d in recovered}) != len(recovered)
        or sorted(recovered, key=lambda d: d.question_id)
        != sorted(audit.decisions, key=lambda d: d.question_id)
        or audit.status != ("partial" if recovered else "failed")
    ):
        raise ValueError("extraction_fallback_ineligible")
    decisions = {d.question_id: d for d in recovered}
    accepted, rejected = [], {}
    for claim in inventory.claims:
        eid = claim.evidence_ids[0]
        pair = [decisions.get(f"{prefix}:{claim.claim_id}") for prefix in ("v", "f")]
        if any(d is None for d in pair):
            rejected[eid] = ["extraction_verification_unavailable"]
            continue
        failures = []
        for decision, kind, reason in zip(
            pair,
            ("claim_validity", "claim_format"),
            ("extraction_claim_validity_not_ok", "extraction_claim_format_not_ok"),
        ):
            if (decision.claim_id, decision.kind, decision.evidence_id) != (
                claim.claim_id,
                kind,
                eid,
            ):
                raise ValueError("extraction_decision_invalid")
            if decision.choice != "ok":
                failures.append(reason)
        if any(d.uncertain or d.confidence < config.confidence_threshold for d in pair):
            failures.append("extraction_low_confidence")
        if failures:
            rejected[eid] = failures
        else:
            accepted.append(eid)
    return accepted, rejected


def _fallback_summary(result: Mapping, accepted: list[str], rejected: Mapping) -> dict:
    """Keep service availability separate from semantic admission results."""
    return {
        "active": result.get("status") != "completed",
        "reason_codes": list(result.get("reason_codes", [])),
        "verified_count": len(accepted),
        "unverified_evidence_ids": [
            eid
            for eid, reasons in rejected.items()
            if "extraction_verification_unavailable" in reasons
        ],
    }


def _receipt_classification(receipt: Mapping) -> tuple[list[str], dict]:
    """Dispatch only explicitly versioned fallback receipts to partial validation."""
    config = _receipt_config(receipt)
    policy = receipt.get("failure_policy")
    if policy is None:
        return _classify(
            receipt["candidate_ledger"],
            receipt["jev"],
            version=receipt["version"],
            config=config,
        )
    if (
        policy != FALLBACK_POLICY
        or receipt.get("version") != ADMISSION_VERSION
        or receipt.get("failure_identity") != fallback_identity()
    ):
        raise ValueError("extraction_failure_policy_invalid")
    return _classify_available(
        receipt["candidate_ledger"], receipt["jev"], config=config
    )


def require_admission_policy(receipt: Mapping, verification: Mapping) -> None:
    """Reject a valid receipt made under a different run's failure semantics."""
    if receipt.get("failure_policy") != verification.get("extraction_failure_policy"):
        raise ValueError("extraction_failure_policy_mismatch")
    if receipt.get("confidence_threshold") != verification.get(
        "extraction_confidence_threshold"
    ):
        raise ValueError("extraction_confidence_threshold_mismatch")
    if receipt.get("resource_policy") != verification.get("extraction_resource_policy"):
        raise ValueError("extraction_resource_policy_mismatch")


def validate_extraction_receipt(receipt: Mapping, selected: Mapping) -> bool:
    """Recheck receipts against frozen original inputs rather than trusting passed flags."""
    try:
        candidates = receipt["candidate_ledger"]
        version = receipt.get("version")
        _stage(version)
        config = _receipt_config(receipt)
        accepted, rejected = _receipt_classification(receipt)
        fallback = (
            _fallback_summary(receipt["jev"], accepted, rejected)
            if receipt.get("failure_policy")
            else None
        )
        return (
            (
                version != ADMISSION_VERSION
                or receipt.get("admission_identity") == admission_identity()
            )
            and receipt.get("stage") == "extraction"
            and receipt.get("status")
            == ("degraded" if fallback and fallback["active"] else "completed")
            and receipt.get("fallback") == fallback
            and receipt.get("candidate_sha256") == digest(candidates)
            and receipt.get("binding")
            == _binding(
                version,
                receipt["source_run_id"],
                receipt["selection"],
                config,
                receipt.get("failure_policy"),
                receipt.get("confidence_threshold"),
                receipt.get("resource_policy"),
            )
            and receipt["selection"]["selected_items"] == candidates
            and receipt["jev"].get("source_run_id") == receipt["source_run_id"]
            and receipt["jev"].get("config_fingerprint") == config.fingerprint
            and receipt.get("accepted_evidence_ids") == accepted
            and receipt.get("rejected") == rejected
            and set(selected) == set(accepted)
            and all(selected[eid] == candidates[eid] for eid in selected)
        )
    except (KeyError, ValueError, TypeError, AttributeError):
        return False


LEDGER_CONTRACT = "accepted-extraction-ledger-v1"


def verified_extraction_ledger(
    receipt: Mapping, available: Mapping, *, run_id: str
) -> dict:
    """Return all accepted originals, never the global or merely cited subset."""
    try:
        accepted = receipt["accepted_evidence_ids"]
        if (
            receipt.get("source_run_id") != run_id
            or not isinstance(accepted, list)
            or not accepted
            or any(not isinstance(eid, str) for eid in accepted)
            or len(set(accepted)) != len(accepted)
        ):
            raise ValueError("invalid accepted identity")
        selected = {eid: available[eid] for eid in accepted}
        if not validate_extraction_receipt(receipt, selected):
            raise ValueError("invalid extraction receipt")
        return copy.deepcopy(selected)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("extraction_verified_ledger_invalid") from exc


def admitted_writing_ledger(
    receipt: Mapping, available: Mapping, cards: Mapping, *, run_id: str
) -> dict:
    """Bind a ranked writing subset to the complete dual-admission receipt."""
    from litagent.quality import build_fact_cards

    if receipt.get("version") != ADMISSION_VERSION:
        raise ValueError("extraction_admission_protocol_required")
    try:
        originals = {
            eid: receipt["candidate_ledger"][eid]
            for eid in receipt["accepted_evidence_ids"]
        }
        admitted = verified_extraction_ledger(receipt, originals, run_id=run_id)
        if any(
            eid not in admitted or item != admitted[eid]
            for eid, item in available.items()
        ):
            raise ValueError("unadmitted writing source")
        selected_ids = {card["evidence_id"] for card in cards.values()}
        if not selected_ids or not selected_ids.issubset(available):
            raise ValueError("writing source missing")
        selected = {eid: available[eid] for eid in available if eid in selected_ids}
        if build_fact_cards(selected) != cards:
            raise ValueError("writing cards changed")
        return copy.deepcopy(selected)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("extraction_admission_writing_snapshot_invalid") from exc


class ExtractionVerifier:
    """Own one bounded Jev extraction attempt per run with durable no-replay state."""

    def __init__(
        self,
        root: Path = Path("artifacts/extraction-verification"),
        *,
        auditor=None,
        trace_hook=None,
        protocol=VERSION,
    ):
        self.root = root
        self.auditor = auditor
        self.trace_hook = trace_hook
        _stage(protocol)
        self.protocol = protocol
        self._lock = asyncio.Lock()

    async def verify_candidates(
        self,
        candidates: Mapping,
        *,
        run_id: str,
        failure_policy: str | None = None,
        confidence_threshold: float | None = None,
        resource_policy: str | None = None,
    ):
        """Check all quarantined candidates before any formal ledger is returned."""
        if self.protocol != ADMISSION_VERSION:
            raise ValueError("extraction_admission_protocol_required")
        sections = (
            "introduction",
            "taxonomy",
            "methods",
            "experiments",
            "open_problems",
        )
        selection = EvidenceSelection(
            candidate_count=len(candidates),
            selected_items=copy.deepcopy(dict(candidates)),
            section_evidence_ids={key: [] for key in sections},
            method_by_section={},
            omitted_count=0,
            estimated_tokens=len(
                json.dumps(dict(candidates), ensure_ascii=False).encode()
            ),
        )
        admitted, receipt = await self.verify(
            selection,
            run_id=run_id,
            failure_policy=failure_policy,
            confidence_threshold=confidence_threshold,
            resource_policy=resource_policy,
        )
        return admitted.selected_items, receipt

    def _emit(self, event: str, data: dict) -> None:
        if self.trace_hook:
            try:
                self.trace_hook(event, data)
            except Exception:
                pass

    async def verify(
        self,
        selection: EvidenceSelection,
        *,
        run_id: str,
        failure_policy: str | None = None,
        confidence_threshold: float | None = None,
        resource_policy: str | None = None,
    ):
        """Filter only a completely checked selection and preserve all negative candidates."""
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is None:
            raise ValueError("extraction_run_id_invalid")
        if failure_policy is not None and (
            failure_policy != FALLBACK_POLICY or self.protocol != ADMISSION_VERSION
        ):
            raise ValueError("extraction_failure_policy_invalid")
        ledger = copy.deepcopy(selection.selected_items)
        config = extraction_config(
            confidence_threshold,
            protocol=self.protocol,
            resource_policy=resource_policy,
        )
        inventory = build_extraction_inventory(ledger, version=self.protocol)
        from litagent.benchmark.citation_audit import (
            _pack_batches,
            _question_specs,
            _request_payload,
            build_citation_audit_inputs,
        )

        inputs = build_citation_audit_inputs(
            inventory, ledger, artifact={"quality_policy": "strict_auto_v2"}
        )
        batches, rejected = _pack_batches(
            _question_specs(
                inputs,
                verification_stage=_stage(self.protocol),
                candidate_ledger=ledger,
            ),
            config,
        )
        estimated = sum(
            len(json.dumps(_request_payload(batch, config.model)).encode("utf-8")) + 64
            for batch in batches
        )
        if (
            rejected
            or len(batches) > config.max_requests
            or estimated > config.max_input_tokens
        ):
            raise ValueError("extraction_verification_budget_exceeded")
        binding = _binding(
            self.protocol,
            run_id,
            selection.to_dict(),
            config,
            failure_policy,
            confidence_threshold,
            resource_policy,
        )
        path = self.root / run_id / "receipt.json"
        async with self._lock:
            if path.exists():
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if receipt.get("binding") != binding:
                    raise ValueError("extraction_identity_conflict")
                if receipt.get("status") not in (
                    ("completed", "degraded") if failure_policy else ("completed",)
                ):
                    raise ValueError("extraction_interrupted_no_replay")
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    # An exclusive durable reservation also protects separate processes.
                    # A crash before the JSON commit still cannot create a second attempt.
                    (path.parent / "reserved").mkdir()
                except FileExistsError as exc:
                    raise ValueError("extraction_in_progress_no_replay") from exc
                receipt = {
                    "version": self.protocol,
                    **(
                        {"admission_identity": admission_identity()}
                        if self.protocol == ADMISSION_VERSION
                        else {}
                    ),
                    "stage": "extraction",
                    "source_run_id": run_id,
                    "binding": binding,
                    "candidate_ledger": ledger,
                    "selection": copy.deepcopy(selection.to_dict()),
                    "candidate_sha256": digest(ledger),
                    "status": "started_no_replay",
                    **(
                        {
                            "resource_policy": resource_policy,
                            "resource_identity": resource_identity(resource_policy),
                            "resource_limits": resource_limits(config),
                        }
                        if resource_policy is not None
                        else {}
                    ),
                    **(
                        {
                            "confidence_threshold": confidence_threshold,
                            "confidence_identity": confidence_identity(),
                        }
                        if confidence_threshold is not None
                        else {}
                    ),
                    **(
                        {
                            "failure_policy": failure_policy,
                            "failure_identity": fallback_identity(),
                        }
                        if failure_policy
                        else {}
                    ),
                }
                atomic_json(path, receipt)
                self._emit(
                    "extraction_verification.start",
                    {
                        "run_id": run_id,
                        "binding": binding,
                        "candidate_count": len(ledger),
                    },
                )
                auditor = self.auditor or JevCitationAuditor(
                    repository=BenchmarkArtifactRepository(self.root / run_id / "jev"),
                    trace_hook=self.trace_hook,
                )
                try:
                    result = await auditor.evaluate(
                        inventory,
                        ledger,
                        config,
                        artifact={
                            "run_id": run_id,
                            "quality_policy": "strict_auto_v2",
                            "extraction_protocol": self.protocol,
                            **(
                                {
                                    "extraction_failure_policy": failure_policy,
                                    "extraction_failure_identity": fallback_identity(),
                                }
                                if failure_policy
                                else {}
                            ),
                            **(
                                {"admission_identity": admission_identity()}
                                if self.protocol == ADMISSION_VERSION
                                else {}
                            ),
                        },
                    )
                    receipt["jev"] = result.model_dump(mode="json")
                    accepted, rejected = _receipt_classification(receipt)
                    fallback = (
                        _fallback_summary(receipt["jev"], accepted, rejected)
                        if failure_policy
                        else None
                    )
                    receipt.update(
                        status=(
                            "degraded"
                            if fallback and fallback["active"]
                            else "completed"
                        ),
                        accepted_evidence_ids=accepted,
                        rejected=rejected,
                    )
                    if fallback is not None:
                        receipt["fallback"] = fallback
                except BaseException as exc:
                    receipt.update(
                        status="failed_no_replay",
                        reason_code=(
                            str(exc)
                            if isinstance(exc, ValueError)
                            else type(exc).__name__
                        ),
                    )
                    atomic_json(path, receipt)
                    self._emit(
                        "extraction_verification.failed",
                        {
                            "run_id": run_id,
                            "reason_code": receipt["reason_code"],
                            "binding": binding,
                        },
                    )
                    raise
                atomic_json(path, receipt)
            accepted, rejected = _receipt_classification(receipt)
            filtered = copy.deepcopy(selection.to_dict())
            filtered["selected_items"] = {eid: ledger[eid] for eid in accepted}
            filtered["section_evidence_ids"] = {
                section: [eid for eid in ids if eid in accepted]
                for section, ids in filtered["section_evidence_ids"].items()
            }
            filtered["omitted_count"] = filtered["candidate_count"] - len(accepted)
            # Preserve original reservations; filtering never creates more writing budget.
            verified = EvidenceSelection.from_dict(filtered)
            if not validate_extraction_receipt(receipt, verified.selected_items):
                raise ValueError("extraction_receipt_invalid")
            self._emit(
                (
                    "extraction_verification.degraded"
                    if receipt.get("status") == "degraded"
                    else "extraction_verification.complete"
                ),
                {
                    "run_id": run_id,
                    "binding": binding,
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "usage": receipt["jev"]["usage"],
                    **({"fallback": receipt["fallback"]} if failure_policy else {}),
                },
            )
            return verified, receipt
