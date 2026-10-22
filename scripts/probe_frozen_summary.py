"""Run one summary-only protocol call on a reconciled, already repaired body."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import subprocess
from pathlib import Path

from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.native_wire import array_from_native, canonical_from_array, unique
from litagent.quality import apply_native_repairs
from litagent.report_audit import atomic_json, digest
from litagent.staged_generation import (
    CONTRACT,
    FROZEN_BODY_CONTRACT,
    decode_typed_body,
    expand_body,
    prepare_summary_request,
    restore_body_repair_response,
    validate_body_document,
)
from scripts import probe_received_body as received
from scripts import probe_staged_generation as probe
from scripts.run_stage_model_quality import check_models, sha

VERSION = "frozen-summary-protocol-v1"
EXPECTED_CONTRACT = "native-body-summary-v7"
REQUESTS = 1


def source_hashes():
    """Pin the writer, complete receipt adapters and this single-call executor."""
    return {
        **received.source_hashes(),
        str(Path(__file__).resolve()): sha(Path(__file__)),
    }


def frozen_inputs(parent, accounting_parent):
    """Verify all paid receipts and reapply the original permitted repair exactly."""
    tip = Path(accounting_parent).resolve()
    m = json.loads((tip / "manifest.json").read_text())
    if (
        m.get("version") != received.VERSION
        or m.get("contract") != "native-body-summary-v5"
        or m.get("parent") != str(Path(parent).resolve())
    ):
        raise ValueError("frozen_summary_latest_receipt_required")
    inputs, prior, prior_repair = received.received_inputs(
        parent, m["accounting_parent"]
    )
    if m.get("repair_state") != prior_repair:
        raise ValueError("frozen_summary_initial_repair_changed")
    inputs = probe.reconcile_accounting(
        inputs,
        tip,
        predecessor_version=received.VERSION,
        predecessor_contract="native-body-summary-v5",
        predecessor_requests=2,
        predecessor_received=prior,
    )
    journal_path = next(tip.glob("generation.json.generation-*.json"))
    journal = json.loads(journal_path.read_text())
    checkpoint_path = tip / "generation.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    state = checkpoint["state"]
    attempt = checkpoint["staged_body_attempt_0"]
    if (
        journal["status"] != "failed_no_replay"
        or list(journal["steps"]) != ["body_repair_0", "summaries"]
        or journal.get("error_phase") != "summaries"
        or journal["received_body"].get("status") != "locally_repaired"
        or any(journal["received_body"].get(k) != v for k, v in prior.items())
        or state.get("run_id") != prior_repair["run_id"]
        or len(state.get("attempts", [])) != 1
        or state["attempts"][0]["status"] != "completed"
        or state["attempts"][0].get("response_sha256")
        != journal["steps"]["body_repair_0"]["raw_response_sha256"]
        or attempt["response"] != journal["steps"]["body_repair_0"]["raw_response"]
    ):
        raise ValueError("frozen_summary_repair_receipt_invalid")
    # Restore the canonical original independently from the earlier provider
    # bytes. A forged new candidate cannot become the base of this continuation.
    original_body = expand_body(
        decode_typed_body(
            json.loads(prior["raw_response"], object_pairs_hook=unique),
            journal["body_card_reference_binding"],
            allow_empty_references=True,
            block_layout=False,
        )
    )
    native, _ = canonical_from_array(original_body, body_only=True)
    if json.loads(attempt["candidate"]) != native:
        raise ValueError("frozen_summary_original_candidate_changed")
    restored = restore_body_repair_response(
        attempt["response"], attempt["wire_binding"], cards=inputs["cards"]
    )
    canonical, proof = apply_native_repairs(
        attempt["candidate"], restored, attempt["binding"], cards=inputs["cards"]
    )
    stored_proof = dict(attempt["repair_proof"])
    wire_sha = stored_proof.pop("wire_binding_sha256", None)
    if (
        hashlib.sha256(restored.encode()).hexdigest()
        != attempt["restored_response_sha256"]
        or canonical != attempt["applied_candidate"]
        or proof != stored_proof
        or wire_sha != attempt["wire_binding"]["sha256"]
        or state["attempts"][0].get("repair_proof") != attempt["repair_proof"]
    ):
        raise ValueError("frozen_summary_repair_proof_invalid")
    body, _ = validate_body_document(
        array_from_native(json.loads(canonical), body_only=True),
        inputs["cards"],
        "academic_v2",
        inputs["language"],
        EvidenceSelection.from_dict(inputs["selection"]),
    )
    frozen = {
        "contract": FROZEN_BODY_CONTRACT,
        "body": body,
        "body_sha256": digest(body),
        "source_binding": m["binding"],
        "source_checkpoint_sha256": sha(checkpoint_path),
        "repair_proof_sha256": digest(attempt["repair_proof"]),
    }
    frozen["sha256"] = digest(frozen)
    inputs["inputs"][str(checkpoint_path)] = sha(checkpoint_path)
    return inputs, frozen, state


def prepare(root, parent, accounting_parent):
    """Freeze a fresh grant without resetting earlier usage or repair attempts."""
    if CONTRACT != EXPECTED_CONTRACT:
        raise ValueError("frozen_summary_contract_migration_required")
    config = load_config()
    check_models(config)
    inputs, frozen, repair = frozen_inputs(parent, accounting_parent)
    m = {
        "version": VERSION,
        "contract": CONTRACT,
        "root": str(root.resolve()),
        "parent": str(parent.resolve()),
        "accounting_parent": str(accounting_parent.resolve()),
        "input": inputs,
        "frozen_body": frozen,
        "repair_state": repair,
        "limit": probe.LIMIT,
        "requests": REQUESTS,
        "output": probe.OUTPUT,
        "config": build_config_summary(config, "strict_auto_v2"),
        "runtime": probe.runtime_identity(),
        "hashes": source_hashes(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
    }
    m["frozen_code"] = {
        str(Path("frozen-code") / Path(p).relative_to(Path.cwd())): h
        for p, h in m["hashes"].items()
    }
    m["binding"] = digest(m)
    root.mkdir(parents=True, exist_ok=False)
    for relative, h in m["frozen_code"].items():
        original = Path.cwd() / Path(relative).relative_to("frozen-code")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        if sha(target) != h:
            raise ValueError("frozen_summary_source_changed")
    atomic_json(root / "manifest.json", m)
    return m


def validate(root):
    """Reject drift and expanded limits before any provider client is created."""
    m = json.loads((root / "manifest.json").read_text())
    config = load_config()
    check_models(config)
    inputs, frozen, repair = frozen_inputs(m["parent"], m["accounting_parent"])
    if (
        CONTRACT != EXPECTED_CONTRACT
        or m["contract"] != CONTRACT
        or m["version"] != VERSION
        or m["root"] != str(root.resolve())
        or m["binding"] != digest({k: v for k, v in m.items() if k != "binding"})
        or (m["limit"], m["requests"], m["output"])
        != (probe.LIMIT, REQUESTS, probe.OUTPUT)
        or (m["input"], m["frozen_body"], m["repair_state"]) != (inputs, frozen, repair)
        or m["config"] != build_config_summary(config, "strict_auto_v2")
        or m["runtime"] != probe.runtime_identity()
        or m["hashes"] != source_hashes()
        or m["frozen_code"]
        != {
            str(Path("frozen-code") / Path(p).relative_to(Path.cwd())): h
            for p, h in m["hashes"].items()
        }
        or any(sha(Path(p)) != h for p, h in inputs["inputs"].items())
        or any(
            not (root / p).is_file() or sha(root / p) != h
            for p, h in m["frozen_code"].items()
        )
    ):
        raise ValueError("frozen_summary_identity_invalid")
    return m, config


def preflight(m, budget):
    """Reserve the full source request and unchanged 20k output at the latest tip."""
    inputs = m["input"]
    body, count = validate_body_document(
        m["frozen_body"]["body"],
        inputs["cards"],
        "academic_v2",
        inputs["language"],
        EvidenceSelection.from_dict(inputs["selection"]),
    )
    messages, schema = prepare_summary_request(
        body,
        count,
        inputs["messages"],
        inputs["response_format"],
        short_refs=True,
        compact_body=True,
    )
    reservation = probe.strict_request_reservation(messages, probe.OUTPUT, schema)
    if reservation > 117964:
        raise ValueError("strict_context_budget_exceeded")
    token = probe.set_task_id("synthesis.summary_protocol")
    try:
        budget.reserve(reservation)
        budget.release(reservation)
    finally:
        probe.reset_task_id(token)
    return reservation


async def run(root, *, live=False):
    """Allow only one summary request; reuse terminal results without replay."""
    m, config = validate(root)
    budget = probe.ProbeBudget.from_snapshot(m["input"]["parent_budget"])
    budget.bind(REQUESTS)
    if live and (root / "state.json").exists():
        return await probe.execute(
            root, m, config, budget, frozen_body=copy.deepcopy(m["frozen_body"])
        )
    reservation = preflight(m, budget)
    if not live:
        return {
            "status": "planned",
            "binding": m["binding"],
            "reservation": reservation,
            "requests": REQUESTS,
            "limit": probe.LIMIT,
            "parent_budget": m["input"]["parent_budget"],
        }
    return await probe.execute(
        root, m, config, budget, frozen_body=copy.deepcopy(m["frozen_body"])
    )


def main():
    """Separate source freezing from execution of the newly bounded allowance."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--accounting-parent", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.parent is None or args.accounting_parent is None:
            parser.error("--prepare requires --parent and --accounting-parent")
        prepare(args.root, args.parent, args.accounting_parent)
    elif args.parent is not None or args.accounting_parent is not None:
        parser.error("parent options require --prepare")
    value = asyncio.run(run(args.root, live=args.live))
    print(json.dumps({k: v for k, v in value.items() if k != "diagnostics"}))


if __name__ == "__main__":
    main()
