"""Isolate frozen inventory protocol failures with three explicit, non-retried probes."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
from pathlib import Path

from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.eval.scoped import (
    STRICT_INVENTORY_PROMPT,
    _strict_inventory_response,
    _strict_inventory_schema,
    check_request_budget,
    flat_inventory_response,
    flat_inventory_schema,
)
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import CompositeTraceHook, RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer
from litagent.quality import validate_fact_snapshot
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget
from scripts.run_frozen_generation import MODEL, ORIGIN, file_sha

VERSION = "frozen-inventory-diagnostic-v1"
CASES = ("short_nested", "short_flat", "full_flat")
FLAT_PROMPT = (
    "Extract every atomic assertion from each host fragment. Return ONLY the supplied "
    "native JSON schema: fragments is an object keyed by ALL exact supplied fragment IDs. "
    "Each value contains assertions (kind, claim_text) and exclusion_reason. Do not return "
    "unit keys, citations, offsets or invented keys. The host owns fragment-to-unit and "
    "citation mappings. Input source_units supplies context, not output nesting.\n"
    + "Extract ONLY assertions"
    + STRICT_INVENTORY_PROMPT.split("Extract ONLY assertions", 1)[1]
)


def prepare(root: Path, source: Path, config) -> dict:
    """Freeze a terminal source, all distinct probes and readable code before calls."""
    root.mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists() or (root / "checkpoint.json").exists():
        raise ValueError("inventory_diagnostic_identity_exists")
    if config.llm.base_url.rstrip("/") != ORIGIN or not config.llm.uses_single_model(
        MODEL
    ):
        raise ValueError("inventory_diagnostic_provider_invalid")
    manifest = json.loads((source / "manifest.json").read_text())
    state = json.loads((source / "comparison.json").read_text())
    terminal = json.loads((source / "terminal-summary.json").read_text())
    slot = source / "vit-baseline"
    case = json.loads((slot / "checkpoint.json").read_text())
    inputs = json.loads((slot / "input-snapshot.json").read_text())
    selected = EvidenceSelection.from_dict(inputs["evidence_selection"])
    metadata = case["report"]["metadata"]
    validate_fact_snapshot(
        metadata["fact_cards"],
        selected.selected_items,
        metadata["evidence_snapshot_sha256"],
    )
    if (
        digest({k: v for k, v in manifest.items() if k != "binding"})
        != manifest["binding"]
        or state["binding"] != manifest["binding"]
        or terminal["binding"] != manifest["binding"]
        or state["status"] != "stopped"
        or state.get("pending")
        or terminal["budgets"] != state["budgets"]
        or file_sha_text(case["report"]["survey"]) != case["report_sha256"]
        or terminal["report_sha256"] != case["report_sha256"]
        or metadata["fact_cards"] != inputs["fact_cards"]
        or metadata["evidence_snapshot_sha256"] != inputs["fact_cards_sha256"]
        or file_sha(Path(inputs["source_artifact"])) != inputs["source_artifact_sha256"]
    ):
        raise ValueError("inventory_diagnostic_source_invalid")
    for service, cumulative in terminal["cumulative_after"].items():
        before, current = (
            manifest["historical"]["services"][service],
            state["budgets"][service],
        )
        if cumulative != {
            "charged": before["charged"] + current["charged"],
            "requests": before["requests"] + current["requests"],
            "usage_complete": before["usage_complete"] and current["usage_complete"],
        }:
            raise ValueError("inventory_diagnostic_history_invalid")
    events = [
        json.loads(line) for line in (slot / "events.jsonl").read_text().splitlines()
    ]
    starts = [
        e["data"]
        for e in events
        if e["event"] == "llm.start"
        and e["data"].get("task_id") == "evaluation.initial"
    ]
    if len(starts) != 1:
        raise ValueError("inventory_diagnostic_source_request_invalid")
    original = json.loads(starts[0]["messages"][1]["content"])
    units = original["source_units"]
    flat_inventory_schema(units)
    hashes = {}
    for path in [
        source / n
        for n in ("manifest.json", "comparison.json", "terminal-summary.json")
    ] + [slot / n for n in ("checkpoint.json", "events.jsonl", "input-snapshot.json")]:
        hashes[str(path.resolve())] = file_sha(path)
    hashes[str(Path(inputs["source_artifact"]).resolve())] = inputs[
        "source_artifact_sha256"
    ]
    for path in [*sorted(Path("src/litagent").rglob("*.py")), Path(__file__)]:
        hashes[str(path.resolve())] = file_sha(path)
        target = root / "frozen-code" / path.resolve().relative_to(Path.cwd())
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    probes = []
    for name in CASES:
        selected = units[:1] if name.startswith("short") else units
        flat = name.endswith("flat")
        payload = copy.deepcopy(original)
        payload.update(
            source_units=selected,
            schema=(
                flat_inventory_schema(selected)
                if flat
                else _strict_inventory_schema(selected)
            ),
        )
        payload["inventory_output_contract"] = (
            "host-flat-fragment-inventory-v1" if flat else "host-fragment-inventory-v2"
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": payload["inventory_output_contract"].replace("-", "_"),
                "strict": True,
                "schema": payload["schema"],
            },
        }
        messages = [
            {
                "role": "system",
                "content": FLAT_PROMPT if flat else STRICT_INVENTORY_PROMPT,
            },
            {
                "role": "user",
                "content": json.dumps(
                    payload, ensure_ascii=False, separators=(",", ":")
                ),
            },
        ]
        probe = {
            "name": name,
            "flat": flat,
            "messages": messages,
            "response_format": response_format,
            "max_tokens": 16384,
            "source_units": selected,
        }
        probe["binding"] = digest(probe)
        atomic_json(root / f"{name}.json", probe)
        hashes[str((root / f"{name}.json").resolve())] = file_sha(root / f"{name}.json")
        probes.append({"name": name, "binding": probe["binding"]})
    plan = {
        "version": VERSION,
        "source": str(source.resolve()),
        "source_report_sha256": case["report_sha256"],
        "historical": terminal["cumulative_after"],
        "config": build_config_summary(config, "strict_auto_v2"),
        "model": MODEL,
        "origin": ORIGIN,
        "probes": probes,
        "hashes": hashes,
        "limits": {
            "tokens": 200000,
            "requests": 13,
            "diagnostic_requests": 3,
            "evaluation_requests": 10,
            "retries": 0,
        },
        "judge": False,
        "jev": False,
        "formal": False,
    }
    plan["binding"] = digest(plan)
    atomic_json(root / "manifest.json", plan)
    return plan


def file_sha_text(value: str) -> str:
    """Bind exact report bytes without normalizing the historical text."""
    import hashlib

    return hashlib.sha256(value.encode()).hexdigest()


async def run(root: Path, config, *, live=False, client_factory=None) -> dict:
    """Execute distinct diagnostic observations; never replay an uncertain attempt."""
    plan = json.loads((root / "manifest.json").read_text())
    if digest({k: v for k, v in plan.items() if k != "binding"}) != plan[
        "binding"
    ] or any(file_sha(Path(p)) != h for p, h in plan["hashes"].items()):
        raise ValueError("inventory_diagnostic_identity_changed")
    if build_config_summary(config, "strict_auto_v2") != plan["config"]:
        raise ValueError("inventory_diagnostic_config_changed")
    if not live:
        return {
            "status": "planned",
            "binding": plan["binding"],
            "paid_calls": 0,
            "limits": plan["limits"],
        }
    from litagent.benchmark.quality_suite import suite_lock

    with suite_lock(root / ".diagnostic.lock"):
        path = root / "checkpoint.json"
        if path.exists():
            old = json.loads(path.read_text())
            if old["binding"] != plan["binding"]:
                raise ValueError("inventory_diagnostic_checkpoint_conflict")
            if old.get("pending"):
                budget = CostBudget.from_snapshot(old["budget"])
                token = set_task_id("evaluation.diagnostic.recovery")
                try:
                    budget.begin_request()
                    budget.charge_unknown_usage(old["pending"]["reserved_tokens"])
                finally:
                    reset_task_id(token)
                old.update(
                    status="interrupted_no_replay",
                    budget=budget.snapshot(),
                    pending=None,
                )
                atomic_json(path, old)
            return old
        budget = CostBudget(max_tokens=plan["limits"]["tokens"])
        budget.configure_strict(halt_on_unknown_usage=True)
        state = {
            "binding": plan["binding"],
            "status": "running",
            "observations": [],
            "pending": None,
            "budget": budget.snapshot(),
        }
        atomic_json(path, state)
        tracer = None
        if client_factory is None and config.observability.enabled:
            tracer = LangFuseTracer(
                config.observability.langfuse_host,
                os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=root.name,
            )

        def journal(event, data):
            """Keep full materials local; the external sink receives metadata only."""
            with (root / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(
                    json.dumps(
                        {"event": event, "data": data}, ensure_ascii=False, default=str
                    )
                    + "\n"
                )

        hook = CompositeTraceHook(
            journal, RedactingTraceHook(tracer, "metadata_only") if tracer else None
        )
        client = (
            client_factory(budget, hook)
            if client_factory
            else OpenAICompatibleClient(
                ORIGIN,
                MODEL,
                cost_budget=budget,
                trace_hook=hook,
                transport_max_attempts=1,
            )
        )
        hook("survey.start", {"session_id": root.name, "phase": "inventory_diagnostic"})
        try:
            for spec in plan["probes"]:
                probe = json.loads((root / f"{spec['name']}.json").read_text())
                token = set_task_id("evaluation.diagnostic." + spec["name"])
                response = None
                reservation = 0
                reserved = False
                entry = {
                    "name": spec["name"],
                    "input_sha256": probe["binding"],
                    "status": "started_no_replay",
                }
                try:
                    reservation = check_request_budget(
                        client,
                        probe["messages"],
                        probe["max_tokens"],
                        int(131072 * 0.9),
                        response_format=probe["response_format"],
                    )
                    budget.reserve(reservation)
                    reserved = True
                    state.update(
                        pending={**entry, "reserved_tokens": reservation},
                        budget=budget.snapshot(),
                    )
                    atomic_json(path, state)
                    response = await asyncio.wait_for(
                        client.chat(
                            probe["messages"],
                            response_format=probe["response_format"],
                            max_tokens=probe["max_tokens"],
                            temperature=0,
                            transport_max_attempts=1,
                            extra_body={"thinking": {"type": "disabled"}},
                        ),
                        180,
                    )
                    entry.update(
                        usage=response.usage,
                        finish_reason=response.finish_reason,
                        request_id=response.request_id,
                        actual_model=response.model,
                    )
                    if (
                        response.model != MODEL
                        or response.usage.get("usage_available") is not True
                    ):
                        raise ValueError("inventory_diagnostic_model_or_usage_invalid")
                    atomic_json(
                        root / f"{spec['name']}-response.json",
                        {"content": response.content, **entry},
                    )
                    progress = {}
                    parser = (
                        flat_inventory_response
                        if probe["flat"]
                        else _strict_inventory_response
                    )
                    try:
                        inventory = parser(
                            response.content,
                            probe["source_units"],
                            progress,
                            research_gaps=True,
                        )
                        if response.finish_reason != "stop":
                            raise ValueError("evaluation_response_finish_invalid")
                        entry.update(
                            status="complete_inventory",
                            units=len(inventory.units),
                            fragments=len(progress["fragment_inventory"]),
                            assertions=sum(len(u.assertions) for u in inventory.units),
                        )
                    except (ValueError, TypeError, KeyError) as exc:
                        entry.update(
                            status="protocol_invalid",
                            reason_code=(
                                str(exc)
                                if str(exc).startswith("evaluation_")
                                else "evaluation_contract_invalid"
                            ),
                            diagnostics=getattr(exc, "details", {}),
                        )
                except BaseException as exc:
                    entry.update(
                        status="stopped",
                        reason_code=(
                            str(exc)
                            if isinstance(exc, ValueError)
                            else type(exc).__name__
                        ),
                    )
                    state["status"] = "stopped"
                    if isinstance(exc, asyncio.CancelledError):
                        raise
                finally:
                    if reserved:
                        budget.release(reservation)
                    reset_task_id(token)
                    state["observations"].append(entry)
                    state.update(pending=None, budget=budget.snapshot())
                    atomic_json(path, state)
                if state["status"] == "stopped":
                    break
            else:
                state["status"] = "completed"
            state["flat_full_inventory_complete"] = any(
                o["name"] == "full_flat" and o["status"] == "complete_inventory"
                for o in state["observations"]
            )
            state["scientific_verified"] = False
            hook(
                "survey.complete",
                {"status": state["status"], "total_tokens": budget.used},
            )
            if tracer:
                state["trace"] = {
                    "trace_id": tracer.trace_id,
                    "trace_url": tracer.trace_url,
                }
            atomic_json(path, state)
            return state
        finally:
            await client.close()
            hook.flush()


def main():
    """Default to a zero-call preview of the frozen three-probe design."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.prepare:
        if args.source is None:
            parser.error("--source is required with --prepare")
        prepare(args.root, args.source, config)
    result = asyncio.run(run(args.root, config, live=args.live))
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "status",
                    "binding",
                    "flat_full_inventory_complete",
                    "budget",
                    "limits",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
