"""Validate one received body repair without regenerating or accepting a Survey."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import subprocess
from pathlib import Path

from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.quality import (
    GroundedWritingError,
    load_targeted_repairs,
    targeted_repair_state,
)
from litagent.report_audit import atomic_json, digest
from litagent.staged_generation import (
    CONTRACT,
    RECEIVED_BODY_CONTRACT,
    card_reference_binding,
    decode_typed_body,
    expand_body,
    prepare_body_request,
    prepare_staged_body_repair,
    prepare_summary_request,
    unique,
    validate_body,
    validate_received_body,
)
from scripts import probe_staged_generation as probe
from scripts.run_stage_model_quality import check_models, sha

VERSION = "frozen-received-body-protocol-v1"
EXPECTED_CONTRACT = "native-body-summary-v5"


def source_hashes():
    """Freeze the entire writer and both explicit execution/accounting adapters."""
    return {
        **probe.source_hashes(),
        str(Path(__file__).resolve()): sha(Path(__file__)),
    }


def received_inputs(parent, accounting_parent):
    """Reconcile v2/v3/v4 charges and the exact still-unrepaired provider bytes."""
    parent, tip = Path(parent).resolve(), Path(accounting_parent).resolve()
    m = json.loads((tip / "manifest.json").read_text())
    if (
        m.get("version") != probe.TYPED_VERSION
        or m.get("contract") != "native-body-summary-v4"
        or m.get("parent") != str(parent)
        or not m.get("accounting_parent")
    ):
        raise ValueError("received_probe_latest_accounting_required")
    inputs = probe.frozen_inputs(parent, m["accounting_parent"])
    inputs = probe.reconcile_accounting(
        inputs,
        tip,
        predecessor_version=probe.TYPED_VERSION,
        predecessor_contract="native-body-summary-v4",
        predecessor_requests=probe.TYPED_REQUESTS,
    )
    journal_path = next(tip.glob("generation.json.generation-*.json"))
    journal = json.loads(journal_path.read_text())
    checkpoint = tip / "generation.json"
    repair = load_targeted_repairs(checkpoint, tip.name)
    original = json.loads(checkpoint.read_text())
    if (
        journal["status"] != "failed_no_replay"
        or set(journal["steps"]) != {"body"}
        or journal["steps"]["body"]["status"] != "response_received"
        or m.get("repair_state") != targeted_repair_state(None, tip.name)
        or repair != m["repair_state"]
        or original != {"state": repair}
    ):
        raise ValueError("received_probe_body_already_consumed")
    prepared, rf = prepare_body_request(
        inputs["messages"], inputs["response_format"], block_layout=False
    )
    expected_packet = json.loads(prepared[1]["content"])
    expected_packet["generation_contract"] = m["contract"]
    events = [
        json.loads(line) for line in (tip / "events.jsonl").read_text().splitlines()
    ]
    start = next(e["data"] for e in events if e["event"] == "llm.start")
    binding = card_reference_binding(
        inputs["cards"],
        # Use the same host scopes as both the original writer and the repair.
        probe_scope(inputs),
    )
    if (
        json.loads(start["messages"][1]["content"]) != expected_packet
        or start["response_format"] != rf
        or journal.get("body_card_reference_binding") != binding
    ):
        raise ValueError("received_probe_body_material_changed")
    received = {
        "contract": RECEIVED_BODY_CONTRACT,
        "raw_response": journal["steps"]["body"]["raw_response"],
        "raw_response_sha256": journal["steps"]["body"]["raw_response_sha256"],
        "source_binding": m["binding"],
        "source_journal_sha256": sha(journal_path),
        "reference_binding_sha256": digest(binding),
    }
    received["sha256"] = digest(received)
    validate_received_body(received, binding)
    inputs["inputs"][str(checkpoint)] = sha(checkpoint)
    return inputs, received, repair


def probe_scope(inputs):
    """Build permissions from the authoritative original evidence selection."""
    from litagent.quality import _writing_scopes

    return _writing_scopes(
        inputs["cards"], "academic_v2", inputs["language"], inputs["selection"]
    )


def prepare(root, parent, accounting_parent):
    """Claim no paid operation while freezing a new immutable continuation grant."""
    if CONTRACT != EXPECTED_CONTRACT:
        raise ValueError("received_probe_contract_migration_required")
    config = load_config()
    check_models(config)
    inputs, received, repair = received_inputs(parent, accounting_parent)
    value = {
        "version": VERSION,
        "contract": CONTRACT,
        "root": str(root.resolve()),
        "parent": str(parent.resolve()),
        "accounting_parent": str(accounting_parent.resolve()),
        "input": inputs,
        "received_body": received,
        "repair_state": repair,
        "limit": probe.LIMIT,
        "requests": probe.REQUESTS,
        "output": probe.OUTPUT,
        "config": build_config_summary(config, "strict_auto_v2"),
        "runtime": probe.runtime_identity(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "hashes": source_hashes(),
    }
    value["frozen_code"] = {
        str(Path("frozen-code") / Path(p).relative_to(Path.cwd())): h
        for p, h in value["hashes"].items()
    }
    value["binding"] = digest(value)
    root.mkdir(parents=True, exist_ok=False)
    for relative, h in value["frozen_code"].items():
        original = Path.cwd() / Path(relative).relative_to("frozen-code")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        if sha(target) != h:
            raise ValueError("received_probe_frozen_source_changed")
    atomic_json(root / "manifest.json", value)
    return value


def validate(root):
    """Reject source, material, accounting, permissions or allowance drift."""
    m = json.loads((root / "manifest.json").read_text())
    config = load_config()
    check_models(config)
    inputs, received, repair = received_inputs(m["parent"], m["accounting_parent"])
    if (
        CONTRACT != EXPECTED_CONTRACT
        or m["version"] != VERSION
        or m["contract"] != CONTRACT
        or m["root"] != str(root.resolve())
        or m["binding"] != digest({k: v for k, v in m.items() if k != "binding"})
        or (m["limit"], m["requests"], m["output"])
        != (probe.LIMIT, probe.REQUESTS, probe.OUTPUT)
        or m["input"] != inputs
        or m["received_body"] != received
        or m["repair_state"] != repair
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
        raise ValueError("received_probe_identity_invalid")
    return m, config


def preflight(m, budget):
    """Reserve full next inputs before authorizing the sole live execution path."""
    inputs = m["input"]
    _, rf = prepare_body_request(
        inputs["messages"], inputs["response_format"], block_layout=False
    )
    selection = EvidenceSelection.from_dict(inputs["selection"])
    raw = m["received_body"]["raw_response"]
    try:
        body, count = validate_body(
            raw,
            rf,
            inputs["cards"],
            "academic_v2",
            inputs["language"],
            selection,
            collect_diagnostics=True,
        )
    except GroundedWritingError as failure:
        from litagent.native_wire import canonical_from_array

        body = expand_body(
            decode_typed_body(
                json.loads(raw, object_pairs_hook=unique),
                card_reference_binding(inputs["cards"], probe_scope(inputs)),
                allow_empty_references=True,
                block_layout=False,
            )
        )
        native, _ = canonical_from_array(body, body_only=True)
        prepared, schema, _ = prepare_staged_body_repair(
            inputs["messages"],
            json.dumps(native, ensure_ascii=False),
            failure.diagnostics,
            cards=inputs["cards"],
            structure="academic_v2",
            language=inputs["language"],
            selection=selection,
        )
    else:
        prepared, schema = prepare_summary_request(
            body, count, inputs["messages"], inputs["response_format"]
        )
    reservation = probe.strict_request_reservation(prepared, probe.OUTPUT, schema)
    if reservation > 117964:
        raise ValueError("strict_context_budget_exceeded")
    token = probe.set_task_id("synthesis.received_protocol")
    try:
        budget.reserve(reservation)
        budget.release(reservation)
    finally:
        probe.reset_task_id(token)
    return reservation


async def run(root, *, live=False):
    """Reuse the received input exactly once under a new, fully accounted identity."""
    m, config = validate(root)
    budget = probe.ProbeBudget.from_snapshot(m["input"]["parent_budget"])
    budget.bind(m["requests"])
    # Terminal or uncertain attempts are reconciled before new request preflight.
    if live and (root / "state.json").exists():
        return await probe.execute(
            root, m, config, budget, received_body=copy.deepcopy(m["received_body"])
        )
    reservation = preflight(m, budget)
    if not live:
        return {
            "status": "planned",
            "binding": m["binding"],
            "limit": m["limit"],
            "requests": m["requests"],
            "next_reservation": reservation,
            "parent_budget": m["input"]["parent_budget"],
            "received_body_sha256": m["received_body"]["raw_response_sha256"],
        }
    return await probe.execute(
        root, m, config, budget, received_body=copy.deepcopy(m["received_body"])
    )


def main():
    """Prepare separately, then execute without permitting a body generation call."""
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
    result = asyncio.run(run(args.root, live=args.live))
    print(json.dumps({k: v for k, v in result.items() if k != "diagnostics"}))


if __name__ == "__main__":
    main()
