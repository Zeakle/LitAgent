"""Audit existing future-input limits without generating or accepting a report."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval.scoped import AtomicFragmentAssertion
from litagent.frozen_writing import CASES, plan_frozen_budget, read_bundle
from litagent.quality import build_fact_cards, digest
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import CONTEXT_LIMIT, check_response_bounds
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    chapter_request_input,
    compile_document,
    prepare_planned_input,
    provider_response_format,
    request_messages,
    review_payload,
    source_packet,
    validate_body,
)


def scope_witness(bundle: dict) -> dict:
    """Demonstrate local shape validity without claiming scientific validity."""
    selection = EvidenceSelection.from_dict(bundle["selection"])
    cards = build_fact_cards(selection.selected_items)
    if bundle.get("fact_cards_sha256") != digest(cards):
        raise ValueError("frozen_writing_fact_card_identity_conflict")
    packet, aliases = source_packet(
        cards, selection, bundle["query"], bundle["requirements"], "zh-CN"
    )
    text = "Non-scientific scope witness. "
    text += "x" * (2000 - len(text))
    counts = dict(zip(BODY, (8, 10, 18, 16, 14)))
    body = {
        "title": "NON-SCIENTIFIC INPUT-SIZE WITNESS",
        **{
            section: [
                {
                    "sentences": [
                        {"text": text, "fact_card_refs": [next(iter(aliases))]}
                        for _ in range(count)
                    ]
                }
            ]
            for section, count in counts.items()
        },
    }
    units = validate_body(body, aliases, writing_plan=RANGE_WRITING_PLAN)
    previous = {k: copy.deepcopy(v) for k, v in body.items() if k != BODY[-1]}
    system, payload, schema = chapter_request_input(
        packet, aliases, BODY[-1], previous, previous, RANGE_WRITING_PLAN
    )
    system, payload = prepare_planned_input(
        system, payload, packet, aliases, RANGE_WRITING_PLAN
    )
    reservation = strict_request_reservation(
        request_messages(system, payload, schema),
        20000,
        provider_response_format("synthesis", schema),
    )
    summaries = {
        key: [
            {
                "sentences": [
                    {
                        "text": "This is a non-scientific input-size witness.",
                        "body_unit_refs": [next(iter(units))],
                    }
                ]
            }
        ]
        for key in ("abstract", "conclusion")
    }
    draft, bindings = compile_document(
        body,
        summaries,
        cards,
        aliases,
        methodology=bundle["methodology"],
        writing_plan=RANGE_WRITING_PLAN,
    )
    reviewed = review_payload(
        {
            "draft": draft,
            "writing_bindings": bindings,
            "writer_document": {"packet": packet, "aliases": aliases},
            "fact_cards": cards,
            "evidence_snapshot_sha256": digest(cards),
            "evidence_selection": selection.to_dict(),
            "coverage_requirements": bundle["requirements"],
            "report_language": "zh-CN",
        }
    )
    # This is a necessary part of a real request, never a complete admission cap.
    reviewer_minimum = strict_request_reservation(
        [
            {
                "role": "user",
                "content": json.dumps(
                    reviewed, ensure_ascii=False, separators=(",", ":")
                ),
            }
        ],
        4096,
    )
    native_assertion = AtomicFragmentAssertion(
        kind="scientific", claim_text="x" * 10000
    )
    check_response_bounds(native_assertion.model_dump())
    return {
        "kind": "locally-valid-non-scientific-future-scope-witness",
        "archive_sha256": bundle["archive_sha256"],
        "fact_cards_sha256": digest(cards),
        "local_body_schema_and_plan_valid": True,
        "local_report_and_binding_valid": True,
        "body_units": len(units),
        "characters_per_body_unit": len(text),
        "actual_card_refs_per_body_unit": 1,
        "report_native_units": len(bindings["units"]),
        "report_utf8_bytes": len(draft.encode()),
        "last_chapter_full_wire_reservation": reservation,
        "last_chapter_context_exceeded": reservation > CONTEXT_LIMIT,
        "reviewer_necessary_part_reservation": reviewer_minimum,
        "reviewer_necessary_part_context_exceeded": reviewer_minimum > CONTEXT_LIMIT,
        "context_reservation_limit": CONTEXT_LIMIT,
        "inventory_claim_text_schema_has_max_length": "maxLength"
        in AtomicFragmentAssertion.model_json_schema()["properties"]["claim_text"],
        "ten_thousand_character_claim_passes_current_size_check": True,
        "can_admit": False,
        "scientific_run": False,
        "new_paid_requests": 0,
    }


def build_review() -> dict:
    """Read only the fixed three archives and publish scalar evidence."""
    rows = []
    for case, archive in CASES:
        bundle = read_bundle(Path("artifacts/runs", archive + ".json"))
        row = scope_witness(bundle)
        plan = plan_frozen_budget(bundle, budget_policy="planned-budget-v3")
        rows.append(
            {
                "case_id": case,
                **row,
                "existing_complete_plan_status": plan["status"],
                "existing_complete_plan_required_tokens": plan["required_tokens"],
            }
        )
    result = {
        "kind": "offline-future-scope-audit-not-scientific-benchmark",
        "cases": rows,
        "new_paid_requests": 0,
        "can_admit": False,
        "maximum_tokens": 1000000,
        "phase_14_5": "in_progress",
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path("scripts/review_future_budget_scope.py"),
                Path("src/litagent/writing_v2.py"),
                Path("src/litagent/eval/scoped.py"),
                Path("src/litagent/safety/planned_budget.py"),
            )
        },
    }
    result["identity"] = digest(result)
    return result


def main() -> None:
    """Export a scalar-only scope audit without initializing a runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = build_review()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": review["cases"], "new_paid_requests": 0}))


if __name__ == "__main__":
    main()
