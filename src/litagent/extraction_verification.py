"""Verify selected extraction candidates against their original sources before writing."""

from __future__ import annotations

import asyncio
import copy
import hashlib
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


def _sha(value) -> str:
    """Use the citation auditor's canonical identities without trusting supplied hashes."""
    from litagent.benchmark.citation_audit import _stable_sha256

    return _stable_sha256(value)


def build_extraction_inventory(ledger: Mapping) -> JudgeInventorySnapshot:
    """Bind each extracted proposition and all semantic fields to exactly one original."""
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
        proposition = item["text"]
        nonempty = {key: val for key, val in fields.items() if val}
        if nonempty:
            proposition += (
                "\nThe following extracted fields describe this same finding; "
                "every field and its association must be entailed by the original: "
                + json.dumps(nonempty, ensure_ascii=False, sort_keys=True)
            )
        cid = "extract:" + digest([VERSION, eid, item])[:24]
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


def _classify(ledger: Mapping, result: Mapping) -> tuple[list[str], dict]:
    """Reject incomplete protocols; separate semantic negatives from transport failures."""
    inventory = build_extraction_inventory(ledger)
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
        for spec in _question_specs(expected_inputs, verification_stage="extraction")
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
        parsed, usage, _ = _validate_response(
            batch.response, batch_specs, JevAuditConfig()
        )
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
        or usage_inputs > 200000
        or len(batches) > 20
    ):
        raise ValueError("extraction_decision_or_usage_invalid")
    decisions = {
        d["question_id"]: CitationAuditDecision.model_validate(d) for d in decisions
    }
    accepted, rejected = [], {}
    for claim in inventory.claims:
        eid = claim.evidence_ids[0]
        group = decisions[f"g:{claim.claim_id}"]
        member = decisions[f"m:{claim.claim_id}:0"]
        for decision in (group, member):
            if (decision.claim_id, decision.kind, decision.evidence_id) != expected[
                decision.question_id
            ]:
                raise ValueError("extraction_decision_invalid")
        reasons = []
        if group.choice != "supported" or member.choice != "contributes":
            reasons.append("extraction_not_supported")
        if any(d.uncertain or d.confidence < 0.9 for d in (group, member)):
            reasons.append("extraction_low_confidence")
        if reasons:
            rejected[eid] = reasons
        else:
            accepted.append(eid)
    return accepted, rejected


def validate_extraction_receipt(receipt: Mapping, selected: Mapping) -> bool:
    """Recheck receipts against frozen original inputs rather than trusting passed flags."""
    try:
        candidates = receipt["candidate_ledger"]
        accepted, rejected = _classify(candidates, receipt["jev"])
        return (
            receipt.get("version") == VERSION
            and receipt.get("stage") == "extraction"
            and receipt.get("status") == "completed"
            and receipt.get("candidate_sha256") == digest(candidates)
            and receipt.get("binding")
            == digest(
                [
                    VERSION,
                    receipt["source_run_id"],
                    receipt["selection"],
                    JevAuditConfig(enabled=True).fingerprint,
                ]
            )
            and receipt["selection"]["selected_items"] == candidates
            and receipt["jev"].get("source_run_id") == receipt["source_run_id"]
            and receipt["jev"].get("config_fingerprint")
            == JevAuditConfig(enabled=True).fingerprint
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


class ExtractionVerifier:
    """Own one bounded Jev extraction attempt per run with durable no-replay state."""

    def __init__(
        self,
        root: Path = Path("artifacts/extraction-verification"),
        *,
        auditor=None,
        trace_hook=None,
    ):
        self.root = root
        self.auditor = auditor
        self.trace_hook = trace_hook
        self._lock = asyncio.Lock()

    def _emit(self, event: str, data: dict) -> None:
        if self.trace_hook:
            try:
                self.trace_hook(event, data)
            except Exception:
                pass

    async def verify(self, selection: EvidenceSelection, *, run_id: str):
        """Filter only a completely checked selection and preserve all negative candidates."""
        if re.fullmatch(r"[A-Za-z0-9_-]{1,128}", run_id) is None:
            raise ValueError("extraction_run_id_invalid")
        ledger = copy.deepcopy(selection.selected_items)
        config = JevAuditConfig(enabled=True)
        inventory = build_extraction_inventory(ledger)
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
            _question_specs(inputs, verification_stage="extraction"), config
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
        binding = digest([VERSION, run_id, selection.to_dict(), config.fingerprint])
        path = self.root / run_id / "receipt.json"
        async with self._lock:
            if path.exists():
                receipt = json.loads(path.read_text(encoding="utf-8"))
                if receipt.get("binding") != binding:
                    raise ValueError("extraction_identity_conflict")
                if receipt.get("status") != "completed":
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
                    "version": VERSION,
                    "stage": "extraction",
                    "source_run_id": run_id,
                    "binding": binding,
                    "candidate_ledger": ledger,
                    "selection": copy.deepcopy(selection.to_dict()),
                    "candidate_sha256": digest(ledger),
                    "status": "started_no_replay",
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
                            "extraction_protocol": VERSION,
                        },
                    )
                    receipt["jev"] = result.model_dump(mode="json")
                    accepted, rejected = _classify(ledger, receipt["jev"])
                    receipt.update(
                        status="completed",
                        accepted_evidence_ids=accepted,
                        rejected=rejected,
                    )
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
            accepted, rejected = _classify(ledger, receipt["jev"])
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
                "extraction_verification.complete",
                {
                    "run_id": run_id,
                    "binding": binding,
                    "accepted_count": len(accepted),
                    "rejected_count": len(rejected),
                    "usage": receipt["jev"]["usage"],
                },
            )
            return verified, receipt
