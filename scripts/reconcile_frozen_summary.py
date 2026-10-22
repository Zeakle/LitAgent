"""Read the latest summary receipt without creating a grant or an HTTP client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from litagent.report_audit import digest
from litagent.staged_generation import (
    prepare_summary_request,
    summary_reference_binding,
    validate_frozen_body_input,
)
from scripts import probe_frozen_summary as frozen
from scripts import probe_staged_generation as probe

PREDECESSOR_VERSION = "frozen-summary-protocol-v1"
PREDECESSOR_CONTRACT = "native-body-summary-v7"


def reconcile(parent: Path, tip: Path):
    """Advance the immutable v7 summary receipt and preserve its repair counter."""
    tip = tip.resolve()
    manifest = json.loads((tip / "manifest.json").read_text())
    if (
        manifest.get("version") != PREDECESSOR_VERSION
        or manifest.get("contract") != PREDECESSOR_CONTRACT
        or manifest.get("parent") != str(parent.resolve())
    ):
        raise ValueError("summary_accounting_latest_receipt_required")
    inputs, body, repair = frozen.frozen_inputs(parent, manifest["accounting_parent"])
    validate_frozen_body_input(body)
    if manifest.get("frozen_body") != body or manifest.get("repair_state") != repair:
        raise ValueError("summary_accounting_body_or_repair_changed")
    inputs = probe.reconcile_accounting(
        inputs,
        tip,
        predecessor_version=PREDECESSOR_VERSION,
        predecessor_contract=PREDECESSOR_CONTRACT,
        predecessor_requests=1,
        predecessor_frozen=body,
    )
    journals = list(tip.glob("generation.json.generation-*.json"))
    journal = json.loads(journals[0].read_text())
    checkpoint = tip / "generation.json"
    state = json.loads(checkpoint.read_text())
    binding = summary_reference_binding(body["body"])
    messages, response_format = prepare_summary_request(
        body["body"],
        binding["body_unit_count"],
        inputs["messages"],
        inputs["response_format"],
        short_refs=True,
        compact_body=True,
    )
    # Reproduce the explicitly supported historical v7 wire, never the current
    # generation contract or a new provider request.
    packet = json.loads(messages[1]["content"])
    packet["generation_contract"] = manifest["contract"]
    messages[1]["content"] = json.dumps(
        packet, ensure_ascii=False, separators=(",", ":")
    )
    # Summary-only continuation cannot spend or reset the shared body repairs.
    if (
        state != {"state": repair}
        or list(journal["steps"]) != ["summaries"]
        or any(journal.get("frozen_body", {}).get(k) != v for k, v in body.items())
        or journal.get("initial_repair_state_sha256") != digest(repair)
        or journal.get("summary_reference_binding") != binding
        or journal["steps"]["summaries"].get("input_sha256")
        != digest([messages, response_format])
    ):
        raise ValueError("summary_accounting_checkpoint_invalid")
    inputs["inputs"][str(checkpoint)] = probe.sha(checkpoint)
    return inputs, body, repair


def main():
    """Print identity and accounting only, with no source prose or credentials."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--tip", type=Path, required=True)
    args = parser.parse_args()
    inputs, body, repair = reconcile(args.parent, args.tip)
    print(
        json.dumps(
            {
                "accounting_binding": inputs["accounting_binding"],
                "budget": inputs["parent_budget"],
                "cumulative": inputs["historical"],
                "body_sha256": body["body_sha256"],
                "repair_attempts": len(repair["attempts"]),
                "new_paid_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
