"""Authenticate the frozen writer inputs for an overlength Survey summary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval.results import BENCHMARK_FAITHFULNESS_POLICY
from litagent.llm.responses import decode_private_receipt, normalize_private_receipt
from litagent.quality import build_fact_cards, digest
from litagent.review_structure import build_methodology, prepare_writer_methodology
from litagent.writing_v2 import (
    BODY,
    SCOPE,
    WriterError,
    WriterSession,
    compile_document,
    effective_summary_limits,
    source_packet,
    summary_length_repair_targets,
)
from scripts.view_survey_latency import Campaign, read_campaign, read_json_object

_SYNTHESIS = "adversarial_review:synthesis:r1"
_SUMMARY = "summary_initial"
_CAPACITY_FAILURE = "writer_summary_capacity_exceeded"


def _require(condition, reason):
    """Report a bounded error without exposing private archive payloads."""
    if not condition:
        raise ValueError(reason)


def _event(archive, name, task_id):
    """Select the unique producer event belonging to the original task."""
    records = [
        event["data"]
        for event in archive["events"]
        if event["event"] == name and event["data"].get("task_id") == task_id
    ]
    _require(len(records) == 1, "summary_recovery_producer_ambiguous")
    return records[0]


def _verify_provider_call(call):
    """Authenticate the actual archived HTTP input and response bytes."""
    receipt = call["provider_receipt"]
    request, _ = decode_private_receipt(receipt)
    result = normalize_private_receipt(receipt)
    _require(
        receipt["status_code"] == 200
        and call["status"] == "response_received"
        and call["provider_status"] == result["provider_status"] == "completed"
        and call["finish_reason"] == result["finish_reason"] == "stop"
        and call["provider_error"] is None
        and not result["provider_error"]
        and call["raw"] == result["content"]
        and call["usage"] == result["usage"]
        and call["model"] == result["model"] == request["model"]
        and call["max_tokens"] == request["max_output_tokens"],
        "summary_recovery_provider_receipt_mismatch",
    )
    system, payload = request["input"]
    _require(
        system["role"] == "system"
        and payload["role"] == "user"
        and call["system_sha256"] == digest(system["content"])
        and call["payload_sha256"] == digest(json.loads(payload["content"])),
        "summary_recovery_provider_payload_mismatch",
    )


@dataclass(frozen=True)
class FrozenSummaryCase:
    """An immutable source checkpoint reconstructed from original run evidence."""

    row: dict
    archive: dict
    task_input: dict
    selection: EvidenceSelection
    cards: dict
    packet: dict
    aliases: dict
    methodology: dict
    body: dict
    summaries: dict
    journal: dict
    budget_plan: dict
    repair_state: dict
    summary_issues: list[dict]

    def session_kwargs(self):
        """Return the original options that participate in writer identity."""
        return {
            "thinking_mode": self.journal["generation_thinking_mode"],
            "reasoning_effort": self.journal["generation_reasoning_effort"],
            "provider_api": self.journal["generation_provider_api"],
            "syntax_policy": self.journal["syntax_policy"],
            "writing_plan": self.journal["writing_plan"]["contract"],
        }

    def validate_session(self, session: WriterSession) -> None:
        """Run production reference-lineage and exact source-version checks."""
        _require(
            session.identity == self.journal["identity"]
            and session.packet == self.packet
            and session.aliases == self.aliases
            and session.journal["calls"] == self.journal["calls"]
            and session.state["limit"] == self.repair_state["limit"]
            and not session.state["attempts"],
            "summary_recovery_writer_session_mismatch",
        )
        session.reference_input(self.body)
        session.reference_input(self.summaries)
        try:
            compile_document(
                self.body,
                self.summaries,
                self.cards,
                self.aliases,
                self.packet["language"],
                self.methodology,
                writing_plan=session.writing_plan,
                envelope=session.envelope,
            )
        except WriterError as exc:
            if exc.code != _CAPACITY_FAILURE:
                raise
        else:
            raise ValueError("summary_recovery_archive_has_no_overlength")

    def validate_writer_ledger_wire(self, ledger: dict) -> None:
        """Bind private writer wire receipts to an already validated ledger."""
        for name, call in self.journal["calls"].items():
            request, _ = decode_private_receipt(call["provider_receipt"])
            output_spec = request["text"]["format"]
            kwargs = {
                "provider_api": call["provider_api"],
                "model": request["model"],
                "max_output_tokens": request["max_output_tokens"],
                "response_format": {
                    "type": output_spec["type"],
                    "json_schema": {
                        k: v for k, v in output_spec.items() if k != "type"
                    },
                },
            }
            if "thinking" in request:
                kwargs["extra_body"] = {"thinking": request["thinking"]}
            for key in ("reasoning_effort", "responses_transport"):
                if call.get(key) is not None:
                    kwargs[key] = call[key]
            entry = ledger["entries"].get(f"{self.journal['identity']}:{name}", {})
            _require(
                entry.get("wire_sha256")
                == digest([request["input"], kwargs, call["provider_api"]]),
                "summary_recovery_historical_writer_wire_mismatch",
            )


def load_frozen_summary_case(
    campaign_root: Path | Campaign, slot_key: str
) -> FrozenSummaryCase:
    """Verify original Survey bytes and reconstruct one frozen writer checkpoint."""
    campaign = (
        campaign_root
        if isinstance(campaign_root, Campaign)
        else read_campaign(campaign_root)
    )
    row = next((item for item in campaign.rows if item.get("run_id") == slot_key), None)
    _require(
        row is not None
        and row["source"] == "replay"
        and row["delivery"] == "blocked"
        and row["policy"] == BENCHMARK_FAITHFULNESS_POLICY
        and not row["usage"]["usage_uncertain"],
        "summary_recovery_scope_invalid",
    )
    source = campaign.archives[slot_key]
    archive = source.read()
    journal = read_json_object(source.path.with_name("writer-revisions.json"))
    metadata = archive["report"]["metadata"]
    _require(
        archive["report"]["partial"] is True
        and metadata["writer_failure"]["code"]
        == journal["failure"]["code"]
        == _CAPACITY_FAILURE,
        "summary_recovery_original_failure_mismatch",
    )
    task_input = archive["nodes"][f"subspan:{_SYNTHESIS}"]["input"]["input"]
    _require(
        task_input["query"] == archive["query"]
        and task_input["report_language"] == metadata["report_language"],
        "summary_recovery_task_owner_mismatch",
    )
    selection = EvidenceSelection.from_dict(
        _event(archive, "evidence.retrieve.complete", _SYNTHESIS)
    )
    selection.writing_scope_policy = SCOPE
    for section in ("abstract", "conclusion"):
        selection.section_evidence_ids[section] = list(selection.selected_items)
        selection.method_by_section[section] = "selected_union"
    cards = build_fact_cards(selection.selected_items)
    packet, aliases = source_packet(
        cards,
        selection,
        task_input["query"],
        task_input["coverage_requirements"],
        task_input["report_language"],
    )
    methodology = prepare_writer_methodology(
        build_methodology(
            task_input["query"],
            task_input["methodology_results"],
            archive["config"]["effective"]["rag"]["content_mode"],
            source_outcomes=metadata["search_source_outcomes"],
            started_at=task_input["search_started_at"],
        ),
        selection=selection,
        cards=cards,
        extractions=_event(archive, "worker.complete", "extract")["output"],
        requested_methods=task_input["requested_methods"],
        upstream_results=task_input["methodology_results"],
    )
    budget_plan = _event(archive, "budget.plan", _SYNTHESIS)["budget_plan"]
    envelope = budget_plan["envelope"]
    _require(
        budget_plan == archive["budget_plan"]
        and digest(packet) == journal["writer_source_view_source_sha256"]
        and digest([packet, aliases, methodology]) == envelope["source_identity"],
        "summary_recovery_frozen_inputs_mismatch",
    )
    repair_state = metadata["targeted_repair_state"]
    _require(
        repair_state["run_id"] == slot_key and repair_state["attempts"] == [],
        "summary_recovery_original_repair_history_not_empty",
    )
    calls = journal["calls"]
    _require(
        list(calls) == [*(f"body:{section}" for section in BODY), _SUMMARY],
        "summary_recovery_writer_call_sequence_mismatch",
    )
    body = {}
    for section in BODY:
        call = calls[f"body:{section}"]
        _verify_provider_call(call)
        part = call["reference_canonical"]
        _require(
            set(part) == ({"title", section} if section == BODY[0] else {section})
            and digest(part) == journal["body_parts"][section]["candidate_sha256"],
            "summary_recovery_body_part_mismatch",
        )
        body.update(part)
    _verify_provider_call(calls[_SUMMARY])
    summaries = calls[_SUMMARY]["reference_canonical"]
    _require(
        all(
            digest(value) in journal["reference_inputs"] for value in (body, summaries)
        ),
        "summary_recovery_reference_inputs_missing",
    )
    _, issues = summary_length_repair_targets(
        summaries, packet["language"], effective_summary_limits(envelope)
    )
    _require(bool(issues), "summary_recovery_archive_has_no_overlength")
    return FrozenSummaryCase(
        row,
        archive,
        task_input,
        selection,
        cards,
        packet,
        aliases,
        methodology,
        body,
        summaries,
        journal,
        budget_plan,
        repair_state,
        issues,
    )
