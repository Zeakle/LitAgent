"""Validate one frozen staged writer attempt without accepting a Survey."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import CompositeTraceHook, RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer
from litagent.quality import (
    build_fact_cards,
    grounded_messages,
    load_targeted_repairs,
    prepare_grounded_request,
    render_grounded_report,
    save_targeted_repairs,
    targeted_repair_state,
)
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.staged_generation import (
    CONTRACT,
    generate_staged_report,
    prepare_body_request,
)
from scripts.run_stage_model_quality import MODELS, ORIGIN, check_models, sha

VERSION = "frozen-staged-protocol-v1"
LIMIT = 120000
REQUESTS = 2
OUTPUT = 20000
REGISTRY = Path("artifacts/acceptance/staged-protocol-grants")
ARTIFACTS = Path("artifacts")
LIVE_CONTRACT = "native-body-summary-v2"
ACCOUNTED_VERSION = "frozen-staged-protocol-v2"
ACCOUNTED_CONTRACT = "native-body-summary-v3"
TYPED_VERSION = "frozen-staged-protocol-v3"
TYPED_CONTRACT = "native-body-summary-v4"
TYPED_REQUESTS = 4


def runtime_identity():
    """Pin the interpreter and SDK versions used by the protocol experiment."""
    return {
        "python": sys.version,
        "packages": {
            p: version(p) for p in ("openai", "httpx", "pydantic", "jsonschema")
        },
    }


def source_hashes():
    """Cover all implementation files and both execution scripts, not a subset."""
    return {
        str(p.resolve()): sha(p)
        for p in [
            *sorted(Path("src/litagent").rglob("*.py")),
            Path(__file__),
            Path("scripts/run_stage_model_quality.py"),
        ]
    }


def rebuild(parent):
    """Reconstruct actual producer inputs and reconcile the completed parent calls."""
    summary_path = Path(parent) / "acceptance-summary.json"
    summary = json.loads(summary_path.read_text())
    archive_path = (
        ARTIFACTS / "runs" / (summary["suite_id"] + "-pilot-few-shot-metric.json")
    )
    archive = json.loads(archive_path.read_text())
    starts = [e["data"] for e in archive["events"] if e["event"] == "llm.start"]
    ends = [e["data"] for e in archive["events"] if e["event"] == "llm.complete"]
    if (
        summary["status"] != "stopped"
        or summary["new_unknown_usage"] is not False
        or sha(archive_path) != summary["source_archive_sha256"]
        or len(starts) != len(ends)
        or len(starts) != summary["new_requests_started"]
        or len({e["operation_id"] for e in starts}) != len(starts)
        or {e["operation_id"] for e in starts} != {e["operation_id"] for e in ends}
        or any(e.get("usage_available") is not True for e in ends)
        or sum(e["total_tokens"] for e in ends) != summary["new_known_tokens"]
    ):
        raise ValueError("staged_probe_parent_invalid")
    frame = starts[-1]
    packet = json.loads(frame["messages"][1]["content"])
    raw = next(
        n["output"]
        for n in archive["nodes"].values()
        if n.get("kind") == "evidence.retrieve"
    )
    selection = EvidenceSelection.from_dict(
        {
            k: raw[k]
            for k in (
                "candidate_count",
                "selected_items",
                "section_evidence_ids",
                "method_by_section",
                "omitted_count",
                "estimated_tokens",
                "ranking_decisions",
            )
        }
    )
    for key in ("abstract", "conclusion"):
        selection.section_evidence_ids[key] = list(selection.selected_items)
        selection.method_by_section[key] = "selected_union"
    all_cards = build_fact_cards(selection.selected_items)
    cards = {cid: all_cards[cid] for cid in packet["cards"]}
    messages, rf = prepare_grounded_request(
        grounded_messages(
            cards,
            "academic_v2",
            packet["report_language"],
            packet["research_question"],
            coverage=packet["coverage_requirements"],
            evidence_selection=selection,
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
            require_unit_citation_scope=True,
        ),
        array_wire=True,
    )
    # Reconcile the historical host packet before any newer transport encoding.
    rebuilt = json.loads(messages[1]["content"])
    rebuilt.update(
        generation_contract=packet["generation_contract"],
        generation_stage="body",
        generation_section_order=[
            "introduction",
            "taxonomy",
            "methods",
            "experiments",
            "open_problems",
        ],
    )
    if set(packet) != set(rebuilt) or any(
        packet[k] != rebuilt[k] for k in packet if k != "generation_contract"
    ):
        raise ValueError("staged_probe_input_changed")
    journals = list(
        (ARTIFACTS / "targeted-repairs").glob(
            archive["run_id"] + ".json.generation-*.json"
        )
    )
    if len(journals) != 1:
        raise ValueError("staged_probe_parent_journal_ambiguous")
    journal = json.loads(journals[0].read_text())
    budget = journal["budget"]
    if (
        journal["status"] != "failed_no_replay"
        or budget.get("usage_uncertain") is not False
        or budget["used_tokens"] != summary["new_known_tokens"]
        or budget["requests_sent"] != len(starts)
        or budget["max_tokens"] != 500000
        or budget.get("stop_reason") is not None
        or budget["used_tokens"] + LIMIT > 500000
        or len(starts) + REQUESTS > 64
    ):
        raise ValueError("staged_probe_parent_budget_invalid")
    CostBudget.from_snapshot(budget)
    return {
        "messages": messages,
        "response_format": rf,
        "cards": cards,
        "selection": selection.to_dict(),
        "language": packet["report_language"],
        "parent_budget": budget,
        "historical": summary["cumulative"],
        "inputs": {
            str(p.resolve()): sha(p) for p in (summary_path, archive_path, journals[0])
        },
        "parent_archive_sha256": sha(archive_path),
    }


def reconcile_accounting(
    inputs,
    accounting_parent,
    *,
    predecessor_version=VERSION,
    predecessor_contract=LIVE_CONTRACT,
):
    """Advance one explicitly versioned receipt without refunding paid calls."""
    root = Path(accounting_parent).resolve()
    paths = [
        root / name
        for name in (
            "manifest.json",
            "state.json",
            "acceptance-summary.json",
            "events.jsonl",
        )
    ]
    m, state, receipt = [json.loads(p.read_text()) for p in paths[:3]]
    events = [json.loads(line) for line in paths[3].read_text().splitlines()]
    grant = REGISTRY / (
        digest([inputs["parent_archive_sha256"], predecessor_contract]) + ".json"
    )
    if (
        m["binding"] != digest({k: v for k, v in m.items() if k != "binding"})
        or m["version"] != predecessor_version
        or m["contract"] != predecessor_contract
        or m["root"] != str(root)
        or m["input"] != inputs
        or (m["limit"], m["requests"], m["output"]) != (LIMIT, REQUESTS, OUTPUT)
        or json.loads(grant.read_text()) != {"binding": m["binding"], "root": str(root)}
        or state != receipt
        or state["binding"] != m["binding"]
        or state["status"] not in {"failed_no_replay", "protocol_completed"}
        or state["scientific_verdict"] != "not_evaluated"
        or state["unknown_hold"] != 0
    ):
        raise ValueError("staged_probe_accounting_identity_invalid")
    starts = [e["data"] for e in events if e["event"] == "llm.start"]
    ends = [e["data"] for e in events if e["event"] == "llm.complete"]
    n = len(starts)
    if (
        not 1 <= n <= REQUESTS
        or len(ends) != n
        or len({e["operation_id"] for e in starts}) != n
        or len({e["operation_id"] for e in ends}) != n
        or [e["operation_id"] for e in starts] != [e["operation_id"] for e in ends]
        or state["new_requests"] != n
        or any(
            e.get("model") != MODELS["generation"] or e.get("max_tokens") != OUTPUT
            for e in starts
        )
        or any(
            e.get("usage_available") is not True
            or e.get("actual_model") != MODELS["generation"]
            or any(
                type(e.get(k)) is not int or e[k] < 0
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            )
            or e["total_tokens"] != e["prompt_tokens"] + e["completion_tokens"]
            for e in ends
        )
    ):
        raise ValueError("staged_probe_accounting_calls_invalid")
    known = sum(e["total_tokens"] for e in ends)
    expected = dict(inputs["parent_budget"])
    expected.update(
        used_tokens=expected["used_tokens"] + known,
        call_count=expected["call_count"] + n,
        requests_sent=expected["requests_sent"] + n,
        phase_usage=[
            expected["phase_usage"][0],
            expected["phase_usage"][1] + known,
            expected["phase_usage"][2],
        ],
    )
    historical = dict(inputs["historical"])
    historical["model_requests"] += n
    historical["model_known_tokens"] += known
    if (
        known > LIMIT
        or state["new_known_tokens"] != known
        or state["budget"] != expected
        or expected.get("usage_uncertain") is not False
        or expected.get("stop_reason") is not None
        or expected.get("strict_phase") != 1
        or expected.get("halt_on_unknown_usage") is not True
        or state["cumulative"] != historical
        or expected["used_tokens"] + LIMIT > 500000
        or expected["requests_sent"] + REQUESTS > 64
    ):
        raise ValueError("staged_probe_accounting_budget_invalid")
    CostBudget.from_snapshot(expected)
    journals = list(root.glob("generation.json.generation-*.json"))
    if len(journals) != 1:
        raise ValueError("staged_probe_accounting_journal_invalid")
    journal = json.loads(journals[0].read_text())
    steps = list(journal["steps"].values())
    if (
        journal["contract"] != predecessor_contract
        or journal["status"] not in {"failed_no_replay", "completed"}
        or (
            journal.get("budget")
            if journal["status"] == "failed_no_replay"
            else steps[-1].get("budget") if steps else None
        )
        != expected
        or len(steps) != n
        or any(
            step.get("status") not in {"locally_valid", "response_received"}
            or step.get("raw_response_sha256")
            != hashlib.sha256(e["content"].encode()).hexdigest()
            or step.get("raw_response") != e["content"]
            or step.get("model") != e["actual_model"]
            or step.get("finish_reason") != e["finish_reason"]
            or step.get("usage")
            != {
                k: e[k]
                for k in ("prompt_tokens", "completion_tokens", "usage_available")
            }
            for step, e in zip(steps, ends)
        )
    ):
        raise ValueError("staged_probe_accounting_journal_invalid")
    running = dict(inputs["parent_budget"])
    for step, start, end in zip(steps, starts, ends):
        running = {
            **running,
            "used_tokens": running["used_tokens"] + end["total_tokens"],
            "call_count": running["call_count"] + 1,
            "requests_sent": running["requests_sent"] + 1,
            "phase_usage": [
                running["phase_usage"][0],
                running["phase_usage"][1] + end["total_tokens"],
                running["phase_usage"][2],
            ],
        }
        if step.get("budget") != running or step.get("input_sha256") != digest(
            [start["messages"], start["response_format"]]
        ):
            raise ValueError("staged_probe_accounting_step_invalid")
    return {
        **inputs,
        "parent_budget": expected,
        "historical": historical,
        "accounting_binding": m["binding"],
        "inputs": {
            **inputs["inputs"],
            **{str(p.resolve()): sha(p) for p in (*paths, grant, journals[0])},
        },
    }


def frozen_inputs(parent, accounting_parent=None):
    """Reconcile the fixed v2 then v3 chain before permitting a v4 allowance."""
    inputs = rebuild(parent)
    if not accounting_parent:
        return inputs
    m = json.loads((Path(accounting_parent) / "manifest.json").read_text())
    if (
        m.get("version") == ACCOUNTED_VERSION
        and m.get("contract") == ACCOUNTED_CONTRACT
    ):
        previous = Path(m["accounting_parent"])
        prior = json.loads((previous / "manifest.json").read_text())
        if (
            m.get("parent") != str(parent.resolve())
            or prior.get("version") != VERSION
            or prior.get("contract") != LIVE_CONTRACT
            or prior.get("accounting_parent")
        ):
            raise ValueError("staged_probe_accounting_chain_invalid")
        inputs = reconcile_accounting(inputs, previous)
        return reconcile_accounting(
            inputs,
            accounting_parent,
            predecessor_version=ACCOUNTED_VERSION,
            predecessor_contract=ACCOUNTED_CONTRACT,
        )
    return reconcile_accounting(inputs, accounting_parent)


def typed_predecessor(accounting_parent):
    """Require the latest completed v3 receipt for the new typed writer."""
    if not accounting_parent:
        return False
    m = json.loads((Path(accounting_parent) / "manifest.json").read_text())
    return (
        CONTRACT == TYPED_CONTRACT
        and m.get("version") == ACCOUNTED_VERSION
        and m.get("contract") == ACCOUNTED_CONTRACT
        and bool(m.get("accounting_parent"))
    )


def prepare(root, parent, accounting_parent=None):
    """Freeze the exact root, original evidence, code and bounded new allowance."""
    config = load_config()
    check_models(config)
    typed = typed_predecessor(accounting_parent)
    value = {
        "version": (
            TYPED_VERSION
            if typed
            else ACCOUNTED_VERSION if accounting_parent else VERSION
        ),
        "contract": CONTRACT,
        "root": str(root.resolve()),
        "parent": str(parent.resolve()),
        "input": frozen_inputs(parent, accounting_parent),
        "limit": LIMIT,
        "requests": TYPED_REQUESTS if typed else REQUESTS,
        "output": OUTPUT,
        "config": build_config_summary(config, "strict_auto_v2"),
        "runtime": runtime_identity(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "hashes": source_hashes(),
    }
    if accounting_parent:
        if CONTRACT != ACCOUNTED_CONTRACT and not typed:
            raise ValueError("staged_probe_new_contract_requires_accounting_migration")
        value["accounting_parent"] = str(accounting_parent.resolve())
        value["frozen_code"] = {
            str(Path("frozen-code") / Path(p).relative_to(Path.cwd())): h
            for p, h in value["hashes"].items()
        }
    if typed:
        value["repair_state"] = load_targeted_repairs(
            root / "generation.json", root.name
        )
        # Freeze the shared allowance before any request; it is not reset on restart.
        if value["repair_state"]["attempts"]:
            raise ValueError("staged_probe_repair_state_not_initial")
        if value["input"]["parent_budget"]["requests_sent"] + TYPED_REQUESTS > 64:
            raise ValueError("staged_probe_parent_budget_invalid")
    value["binding"] = digest(value)
    root.mkdir(parents=True, exist_ok=False)
    for relative, h in value.get("frozen_code", {}).items():
        original = Path.cwd() / Path(relative).relative_to("frozen-code")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original.read_bytes())
        if sha(target) != h:
            raise ValueError("staged_probe_frozen_source_changed")
    atomic_json(root / "manifest.json", value)
    return value


def validate(root):
    """Reject drift or expanded grants before constructing a transport."""
    m = json.loads((root / "manifest.json").read_text())
    config = load_config()
    check_models(config)
    typed = typed_predecessor(m.get("accounting_parent"))
    if (
        m["binding"] != digest({k: v for k, v in m.items() if k != "binding"})
        or m["root"] != str(root.resolve())
        or m["version"]
        != (
            TYPED_VERSION
            if typed
            else ACCOUNTED_VERSION if m.get("accounting_parent") else VERSION
        )
        or m["contract"] != CONTRACT
        or (m["limit"], m["requests"], m["output"])
        != (LIMIT, TYPED_REQUESTS if typed else REQUESTS, OUTPUT)
        or typed
        and m.get("repair_state") != targeted_repair_state(None, root.name)
        or m["config"] != build_config_summary(config, "strict_auto_v2")
        or m["runtime"] != runtime_identity()
        or m["hashes"] != source_hashes()
        or any(
            sha(Path(p)) != h
            for p, h in {**m["hashes"], **m["input"]["inputs"]}.items()
        )
        or m["input"] != frozen_inputs(Path(m["parent"]), m.get("accounting_parent"))
        or bool(m.get("accounting_parent"))
        and (
            m["contract"] != (TYPED_CONTRACT if typed else ACCOUNTED_CONTRACT)
            or m.get("frozen_code")
            != {
                str(Path("frozen-code") / Path(p).relative_to(Path.cwd())): h
                for p, h in m["hashes"].items()
            }
            or any(
                not (root / p).is_file() or sha(root / p) != h
                for p, h in m.get("frozen_code", {}).items()
            )
        )
    ):
        raise ValueError("staged_probe_identity_invalid")
    return m, config


class ProbeBudget(CostBudget):
    """Add a version-bound call ceiling to the unchanged inherited Survey budget."""

    def bind(self, requests=REQUESTS):
        """Capture inherited usage once, before reserving any new request."""
        self._probe_request_limit = requests
        self._probe_start = self.used
        self._probe_requests = self.snapshot()["requests_sent"]

    def reserve(self, tokens):
        """Reserve complete inputs and maximum output against both ceilings."""
        if self.used - self._probe_start + tokens > LIMIT:
            raise ValueError("staged_probe_token_limit")
        return super().reserve(tokens)

    def begin_request(self):
        """Count a sent request against the same non-resettable parent budget."""
        if (
            self.snapshot()["requests_sent"] - self._probe_requests
            >= self._probe_request_limit
        ):
            raise ValueError("staged_probe_request_limit")
        return super().begin_request()


def save_receipt(root, m, state):
    """Carry prior charges forward without promoting a protocol result to science."""
    cumulative = dict(m["input"]["historical"])
    cumulative["model_requests"] += state.get("new_requests", 0)
    cumulative["model_known_tokens"] += state.get("new_known_tokens", 0)
    cumulative["model_unknown_conservative_tokens"] += state.get("unknown_hold", 0)
    state["cumulative"] = cumulative
    atomic_json(root / "state.json", state)
    atomic_json(root / "acceptance-summary.json", state)
    return state


async def run(root, *, live=False):
    """Use production writer validation without publishing or accepting a report."""
    if live and CONTRACT != LIVE_CONTRACT:
        path = root / "manifest.json"
        manifest = json.loads(path.read_text()) if path.is_file() else {}
        if not (
            (
                CONTRACT == ACCOUNTED_CONTRACT
                and manifest.get("version") == ACCOUNTED_VERSION
                or CONTRACT == TYPED_CONTRACT
                and manifest.get("version") == TYPED_VERSION
            )
            and manifest.get("accounting_parent")
        ):
            raise ValueError("staged_probe_new_contract_requires_accounting_migration")
    m, config = validate(root)
    budget = ProbeBudget.from_snapshot(m["input"]["parent_budget"])
    budget.bind(m["requests"])
    messages, rf = prepare_body_request(
        m["input"]["messages"], m["input"]["response_format"]
    )
    reservation = strict_request_reservation(messages, OUTPUT, rf)
    task = set_task_id("synthesis.protocol_conformance")
    try:
        if reservation > 117964:
            raise ValueError("strict_context_budget_exceeded")
        budget.reserve(reservation)
        budget.release(reservation)
    finally:
        reset_task_id(task)
    if not live:
        return {
            "status": "planned",
            "binding": m["binding"],
            "limit": LIMIT,
            "requests": m["requests"],
            "parent_budget": m["input"]["parent_budget"],
            "body_reservation": reservation,
        }
    with suite_lock(root / "probe.lock"):
        path = root / "state.json"
        if path.exists():
            state = json.loads(path.read_text())
            if state["binding"] != m["binding"]:
                raise ValueError("staged_probe_state_conflict")
            if state["status"] == "started_no_replay":
                state.update(
                    status="interrupted_no_replay",
                    new_requests=m["requests"],
                    unknown_hold=max(0, LIMIT - state.get("new_known_tokens", 0)),
                )
            return save_receipt(root, m, state)
        REGISTRY.mkdir(parents=True, exist_ok=True)
        grant = REGISTRY / (
            digest([m["input"]["parent_archive_sha256"], CONTRACT]) + ".json"
        )
        # One explicit attempt per frozen parent and protocol, even across roots.
        with grant.open("x") as stream:
            json.dump({"binding": m["binding"], "root": m["root"]}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        state = {
            "binding": m["binding"],
            "status": "started_no_replay",
            "scientific_verdict": "not_evaluated",
        }
        atomic_json(path, state)

        def journal(event, data):
            with (root / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"event": event, "data": data}, ensure_ascii=False)
                    + "\n"
                )
                stream.flush()
                os.fsync(stream.fileno())
            if event in {"llm.start", "llm.complete"}:
                state["budget"] = budget.snapshot()
                if event == "llm.complete":
                    state["new_known_tokens"] = state.get(
                        "new_known_tokens", 0
                    ) + data.get("total_tokens", 0)
                atomic_json(path, state)

        tracer = LangFuseTracer(
            config.observability.langfuse_host,
            os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            os.getenv("LANGFUSE_SECRET_KEY", ""),
            trace_seed=root.name,
        )
        hook = CompositeTraceHook(
            journal, RedactingTraceHook(tracer, payload_mode="metadata_only")
        )
        owner = None
        inputs = m["input"]
        repair_state = None
        if m["version"] == TYPED_VERSION:
            repair_state = m["repair_state"]
            save_targeted_repairs(root / "generation.json", repair_state)
        token = set_task_id("synthesis.protocol_conformance")
        hook("survey.start", {"session_id": root.name, "query": "Protocol conformance"})
        try:
            owner = OpenAICompatibleClient(
                ORIGIN,
                MODELS["generation"],
                cost_budget=budget,
                trace_hook=hook,
                transport_max_attempts=1,
            )

            async def capture(request):
                body = json.loads(request.content)
                if (
                    str(request.url) != ORIGIN + "/chat/completions"
                    or request.method != "POST"
                    or body["model"] != MODELS["generation"]
                    or body.get("max_tokens") != OUTPUT
                    or body.get("thinking") != {"type": "disabled"}
                    or body["response_format"]["json_schema"]["name"]
                    not in {
                        "grounded_body_v2_rows",
                        "grounded_body_v3_refs",
                        "grounded_body_v4_typed",
                        "grounded_summaries_v1",
                        *(
                            {"native_body_repair_v6"}
                            if m["version"] == TYPED_VERSION
                            else set()
                        ),
                    }
                ):
                    raise ValueError("staged_probe_wire_invalid")

            owner._client._client.event_hooks["request"].append(capture)
            candidate, proof = await generate_staged_report(
                ModelBoundClient(owner, MODELS["generation"], "generation"),
                inputs["messages"],
                inputs["response_format"],
                cards=inputs["cards"],
                structure="academic_v2",
                language=inputs["language"],
                selection=EvidenceSelection.from_dict(inputs["selection"]),
                checkpoint_path=root / "generation.json",
                trace_hook=hook,
                parent_task_id="protocol",
                repair_state=repair_state,
                generation_options={
                    "temperature": 0.1,
                    "extra_body": {"thinking": {"type": "disabled"}},
                },
            )
            rendered = render_grounded_report(
                candidate,
                inputs["cards"],
                "academic_v2",
                inputs["language"],
                evidence_selection=inputs["selection"],
                require_native=True,
                require_full_sources=True,
                require_scope_notices=True,
                analysis_mode=True,
                require_unit_citation_scope=True,
            )
            atomic_json(
                root / "candidate.json",
                {"candidate": candidate, "proof": proof, "rendered": rendered},
            )
            state.update(
                status="protocol_completed",
                local_report_sha256=hashlib.sha256(rendered.encode()).hexdigest(),
            )
        except BaseException as exc:
            state.update(
                status="failed_no_replay", error_type=type(exc).__name__, error=str(exc)
            )
            if hasattr(exc, "diagnostics"):
                state["diagnostics"] = exc.diagnostics
            if isinstance(exc, json.JSONDecodeError):
                state["json_error"] = {
                    "offset": exc.pos,
                    "line": exc.lineno,
                    "column": exc.colno,
                }
        finally:
            snapshot = budget.snapshot()
            known = sum(
                json.loads(line)["data"].get("total_tokens", 0)
                for line in (root / "events.jsonl").read_text().splitlines()
                if json.loads(line)["event"] == "llm.complete"
            )
            state.update(
                budget=snapshot,
                new_requests=(
                    snapshot["requests_sent"] - inputs["parent_budget"]["requests_sent"]
                    if owner
                    else 0
                ),
                new_known_tokens=known,
                unknown_hold=(
                    snapshot["used_tokens"]
                    - inputs["parent_budget"]["used_tokens"]
                    - known
                    if owner
                    else 0
                ),
                trace_id=tracer.trace_id,
                trace_url=tracer.trace_url,
            )
            atomic_json(path, state)
            # Protocol completion never asserts scientific acceptance or delivery.
            hook(
                "survey.complete",
                {
                    "rounds": 0,
                    "accepted": False,
                    "quality_status": "unverified",
                    "delivery_status": "not_delivered",
                    "total_tokens": known,
                },
            )
            if owner is not None:
                await owner.close()
            hook.flush()
            reset_task_id(token)
        return save_receipt(root, m, state)


def main():
    """Require a distinct frozen preparation before one bounded live attempt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--accounting-parent", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.parent is None:
            parser.error("--prepare requires --parent")
        prepare(args.root, args.parent, args.accounting_parent)
    elif args.accounting_parent is not None:
        parser.error("--accounting-parent requires --prepare")
    result = asyncio.run(run(args.root, live=args.live))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"diagnostics"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
