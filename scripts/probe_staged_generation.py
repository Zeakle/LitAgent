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
    prepare_grounded_request,
    render_grounded_report,
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
    body_messages, _ = prepare_body_request(messages, rf)
    rebuilt = json.loads(body_messages[1]["content"])
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


def prepare(root, parent):
    """Freeze the exact root, original evidence, code and bounded new allowance."""
    config = load_config()
    check_models(config)
    value = {
        "version": VERSION,
        "contract": CONTRACT,
        "root": str(root.resolve()),
        "parent": str(parent.resolve()),
        "input": rebuild(parent),
        "limit": LIMIT,
        "requests": REQUESTS,
        "output": OUTPUT,
        "config": build_config_summary(config, "strict_auto_v2"),
        "runtime": runtime_identity(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "hashes": source_hashes(),
    }
    value["binding"] = digest(value)
    root.mkdir(parents=True, exist_ok=False)
    atomic_json(root / "manifest.json", value)
    return value


def validate(root):
    """Reject drift or expanded grants before constructing a transport."""
    m = json.loads((root / "manifest.json").read_text())
    config = load_config()
    check_models(config)
    if (
        m["binding"] != digest({k: v for k, v in m.items() if k != "binding"})
        or m["root"] != str(root.resolve())
        or m["version"] != VERSION
        or m["contract"] != CONTRACT
        or (m["limit"], m["requests"], m["output"]) != (LIMIT, REQUESTS, OUTPUT)
        or m["config"] != build_config_summary(config, "strict_auto_v2")
        or m["runtime"] != runtime_identity()
        or m["hashes"] != source_hashes()
        or any(
            sha(Path(p)) != h
            for p, h in {**m["hashes"], **m["input"]["inputs"]}.items()
        )
        or m["input"] != rebuild(Path(m["parent"]))
    ):
        raise ValueError("staged_probe_identity_invalid")
    return m, config


class ProbeBudget(CostBudget):
    """Add a two-call grant ceiling to the unchanged inherited Survey budget."""

    def bind(self):
        """Capture inherited usage once, before reserving any new request."""
        self._probe_start = self.used
        self._probe_requests = self.snapshot()["requests_sent"]

    def reserve(self, tokens):
        """Reserve complete inputs and maximum output against both ceilings."""
        if self.used - self._probe_start + tokens > LIMIT:
            raise ValueError("staged_probe_token_limit")
        return super().reserve(tokens)

    def begin_request(self):
        """Count a sent request against the same non-resettable parent budget."""
        if self.snapshot()["requests_sent"] - self._probe_requests >= REQUESTS:
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
    m, config = validate(root)
    budget = ProbeBudget.from_snapshot(m["input"]["parent_budget"])
    budget.bind()
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
            "requests": REQUESTS,
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
                    new_requests=REQUESTS,
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
                    not in {"grounded_body_v2_rows", "grounded_summaries_v1"}
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
            state.update(status="failed_no_replay", error_type=type(exc).__name__)
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
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        if args.parent is None:
            parser.error("--prepare requires --parent")
        prepare(args.root, args.parent)
    result = asyncio.run(run(args.root, live=args.live))
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in {"diagnostics"}},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
