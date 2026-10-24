"""Observe one unchanged full-input writer request without accepting a Survey."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path

from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.frozen_writing import read_bundle
from litagent.llm.client import OpenAICompatibleClient
from litagent.llm.routing import ModelBoundClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.quality import build_fact_cards
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget, strict_request_reservation
from litagent.writing_v2 import (
    CONTRACT,
    EXECUTION_CONTRACT,
    INSTRUCTION,
    PROVIDER_SCHEMA_CONTRACT,
    WriterSession,
    classify_failure,
    provider_response_format,
    request_messages,
    schema_for,
    source_packet,
    validate_body,
)
from scripts.run_frozen_writing import (
    accounting_history,
    runtime_identity,
    source_paths,
)
from scripts.run_stage_model_quality import MODELS, check_models, sha

VERSION = "writer-v2-protocol-observation-v3"
PARENT = Path("artifacts/acceptance/frozen-writer-three-domain-20261023-001")
LIMITS = {"tokens": 120000, "requests": 1, "output_tokens": 20000, "retries": 0}


def build_manifest(config):
    """Rebuild the fixed public input and compare every request field with its parent."""
    check_models(config)
    parent = json.loads((PARENT / "manifest.json").read_text())
    state = json.loads((PARENT / "state.json").read_text())
    if (
        parent["identity"]
        != digest({k: v for k, v in parent.items() if k != "identity"})
        or state["identity"] != parent["identity"]
        or state["status"] != "stopped"
        or len(state["cases"]) != 1
        or state["cases"][0]["known_tokens"] != 12477
        or not state["cases"][0]["usage_complete"]
    ):
        raise ValueError("writer_probe_parent_invalid")
    for path, expected in parent["source_hashes"].items():
        if sha(PARENT / "source" / path) != expected:
            raise ValueError("writer_probe_parent_snapshot_changed")
    row = parent["cases"][0]
    bundle_path = Path("artifacts/runs", row["bundle"]["source_run_id"] + ".json")
    bundle = read_bundle(bundle_path)
    if row["case_id"] != "few-shot-metric" or bundle != row["bundle"]:
        raise ValueError("writer_probe_parent_input_changed")
    archive_path = Path("artifacts/runs", state["cases"][0]["run_id"] + ".json")
    archive = json.loads(archive_path.read_text())
    starts = [e["data"] for e in archive["events"] if e["event"] == "llm.start"]
    ends = [e["data"] for e in archive["events"] if e["event"] == "llm.complete"]
    if (
        len(starts) != 1
        or len(ends) != 1
        or starts[0]["operation_id"] != ends[0]["operation_id"]
        or ends[0].get("usage_available") is not True
        or ends[0]["total_tokens"] != 12477
    ):
        raise ValueError("writer_probe_parent_usage_invalid")
    selection = EvidenceSelection.from_dict(bundle["selection"])
    packet, aliases = source_packet(
        build_fact_cards(selection.selected_items),
        selection,
        bundle["query"],
        bundle["requirements"],
        "zh-CN",
    )
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": json.dumps(packet, ensure_ascii=False)},
    ]
    local_schema = schema_for("body", aliases)
    original_rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "writer_v2_synthesis",
            "strict": True,
            "schema": local_schema,
        },
    }
    original = starts[0]
    received_path = Path(
        "artifacts/targeted-repairs",
        state["cases"][0]["run_id"] + ".json.writer-v2.json",
    )
    received = json.loads(received_path.read_text())["calls"]["body"]
    if (
        hashlib.sha256(received["raw"].encode()).hexdigest()
        != "4a101a71503f146da2bcec6b157414787d4e3096ae5036d481ce9428670c9b47"
        or received["finish_reason"] != "stop"
    ):
        raise ValueError("writer_probe_parent_response_changed")
    if (
        messages != original["messages"]
        or original_rf != original["response_format"]
        or original["max_tokens"] != LIMITS["output_tokens"]
        or original["model"] != MODELS["generation"]
        or original["temperature"] != config.llm.temperature
    ):
        raise ValueError("writer_probe_original_request_changed")
    rf = provider_response_format("synthesis", local_schema)
    messages = request_messages(INSTRUCTION, packet, local_schema)
    history, receipts = accounting_history(
        parent["history"],
        [
            PARENT,
            Path("artifacts/acceptance/writer-v2-protocol-20261023-001"),
            Path("artifacts/acceptance/frozen-writer-three-domain-20261023-002"),
            Path("artifacts/acceptance/writer-v2-schema-compat-20261023-001"),
            Path("artifacts/acceptance/frozen-writer-three-domain-20261023-003"),
            Path("artifacts/acceptance/frozen-writer-three-domain-20261024-004"),
            Path("artifacts/acceptance/frozen-writer-three-domain-20261024-005"),
            Path("artifacts/acceptance/frozen-writer-three-domain-20261024-006"),
            Path("artifacts/acceptance/frozen-writer-three-domain-20261024-007"),
        ],
    )
    files = [
        PARENT / "manifest.json",
        PARENT / "state.json",
        archive_path,
        bundle_path,
        received_path,
    ]
    manifest = {
        "version": VERSION,
        "contract": CONTRACT,
        "execution_contract": EXECUTION_CONTRACT,
        "provider_schema_contract": PROVIDER_SCHEMA_CONTRACT,
        "limits": dict(LIMITS),
        "runtime": runtime_identity(),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_hashes": {str(p): sha(p) for p in source_paths()},
        "parent_hashes": {str(p): sha(p) for p in files},
        "config": build_config_summary(config, "strict_auto_v2"),
        "messages": messages,
        "response_format": rf,
        "local_schema": local_schema,
        "changed_request_fields": [
            "response_format.json_schema.schema",
            "messages[0].content",
        ],
        "packet": packet,
        "aliases": aliases,
        "reservation": strict_request_reservation(
            messages, LIMITS["output_tokens"], rf
        ),
        "history": history,
        "accounting_receipts": receipts,
    }
    manifest["identity"] = digest(manifest)
    return manifest


def prepare(root, config):
    """Freeze a new one-request observation without reusing an old terminal suite."""
    if root.exists():
        raise ValueError("writer_probe_already_exists")
    manifest = build_manifest(config)
    root.mkdir(parents=True)
    atomic_json(root / "manifest.json", manifest)
    for path in manifest["source_hashes"]:
        target = root / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(path).read_bytes())
    return {
        "identity": manifest["identity"],
        "limits": manifest["limits"],
        "reservation": manifest["reservation"],
    }


def validate(root, config):
    """Reject source, input, runtime or allowance drift before constructing a client."""
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest != build_manifest(config):
        raise ValueError("writer_probe_identity_conflict")
    for path, expected in manifest["source_hashes"].items():
        if sha(root / "source" / path) != expected:
            raise ValueError("writer_probe_snapshot_changed")
    return manifest


def publish(root, result):
    """Publish metadata only; protocol success never becomes a scientific score."""
    public = {
        k: result.get(k)
        for k in (
            "identity",
            "status",
            "failure",
            "known_tokens",
            "requests",
            "unknown_hold",
            "usage_complete",
            "elapsed_ms",
            "trace_url",
            "raw_sha256",
            "finish_reason",
            "provider_schema_contract",
            "source_commit",
            "changed_request_fields",
            "limits",
            "input_binding_sha256",
        )
    }
    public.update(
        evaluation="not_run",
        publishable=False,
        scientific_score=None,
        judge="disabled",
        jev="disabled",
    )
    if public["failure"]:
        failure = public["failure"]
        # Provider errors may echo request material; public output carries codes only.
        code = failure.get("code", "")
        public["failure"] = {
            "family": failure.get("family"),
            "code": (
                code
                if re.fullmatch(r"[a-z][a-z0-9_]{0,100}", code)
                else "provider_or_transport_error"
            ),
        }
    atomic_json(root / "result-public.json", public)
    (root / "result-public.md").write_text(
        "# Writer protocol observation\n\n"
        f"Status: {public['status']}\n\n"
        f"Requests: {public['requests']}; known tokens: {public['known_tokens']}; "
        f"unknown reservation: {public['unknown_hold']}\n\n"
        "Scientific Evaluation: not run. Publishable: false.\n\n"
        f"Failure: `{(public.get('failure') or {}).get('code', 'none')}`\n",
    )
    return public


def capture_http(owner, root):
    """Keep exact response bytes local before the SDK parses them; omit request headers."""

    async def capture(response):
        await response.aread()
        for name, data in (
            ("http-request.body", response.request.content),
            ("http-response.body", response.content),
        ):
            temporary = root / (name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(root / name)
        atomic_json(
            root / "http-metadata.json",
            {
                "status_code": response.status_code,
                "request_id": response.headers.get("x-request-id"),
                "log_id": response.headers.get("x-tt-logid"),
                "request_sha256": hashlib.sha256(response.request.content).hexdigest(),
                "response_sha256": hashlib.sha256(response.content).hexdigest(),
            },
        )

    # This diagnostic pins the SDK version in its manifest and owns its HTTP client.
    http = owner._client._client
    http.follow_redirects = False
    http.event_hooks["response"].append(capture)


async def execute(root, config):
    """Persist the send boundary, make at most one call and never resume an unknown call."""
    with suite_lock(root / "execution.lock"):
        manifest = validate(root, config)
        path = root / "state.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["identity"] != manifest["identity"]:
                raise ValueError("writer_probe_state_conflict")
            if result["status"] == "started_no_replay":
                result.update(
                    status="interrupted_no_replay",
                    usage_complete=False,
                    unknown_hold=manifest["reservation"],
                    requests=1,
                )
                atomic_json(path, result)
            return publish(root, result)
        result = {
            "identity": manifest["identity"],
            "status": "started_no_replay",
            "known_tokens": 0,
            "unknown_hold": manifest["reservation"],
            "requests": 0,
            "usage_complete": False,
            "provider_schema_contract": manifest.get("provider_schema_contract"),
            "source_commit": manifest.get("source_commit"),
            "changed_request_fields": manifest.get("changed_request_fields"),
            "limits": dict(LIMITS),
            "input_binding_sha256": digest(
                [
                    manifest["messages"],
                    manifest["response_format"],
                    LIMITS["output_tokens"],
                    {"thinking": {"type": "disabled"}},
                ]
            ),
        }
        atomic_json(path, result)
        recorder = RunRecorder(
            root.name, manifest["packet"]["query"], ArchiveRepository()
        )
        recorder.set_config_summary(manifest["config"])
        tracer = (
            LangFuseTracer(
                config.observability.langfuse_host,
                os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=root.name,
            )
            if config.observability.enabled
            else None
        )
        hook = CompositeTraceHook(
            recorder,
            (
                RedactingTraceHook(tracer, payload_mode="metadata_only")
                if tracer
                else None
            ),
        )
        # Standalone diagnostics need the same parent lifecycle as worker runs.
        # The final quality remains unverified even when the protocol succeeds.
        hook(
            "survey.start",
            {"session_id": root.name, "query": manifest["packet"]["query"]},
        )
        budget = CostBudget(max_tokens=LIMITS["tokens"])
        budget.configure_strict(halt_on_unknown_usage=True)
        owner = OpenAICompatibleClient(
            config.llm.base_url,
            config.llm.model,
            max_tokens=LIMITS["output_tokens"],
            temperature=config.llm.temperature,
            cost_budget=budget,
            trace_hook=hook,
            transport_max_attempts=1,
            request_timeouts=config.llm.request_timeouts_seconds,
        )
        capture_http(owner, root)
        client = ModelBoundClient(owner, MODELS["generation"], "generation")
        token = set_task_id("synthesis_protocol")
        start = time.monotonic()
        failure = None
        session = None
        try:
            session = WriterSession(
                client,
                root / "request.json",
                root.name,
                manifest["packet"],
                manifest["aliases"],
                trace_hook=hook,
            )
            body = await session.call(
                "body",
                INSTRUCTION,
                manifest["packet"],
                manifest["local_schema"],
            )
            validate_body(body, manifest["aliases"])
            result["status"] = "protocol_completed"
        except BaseException as exc:
            failure = classify_failure(exc)
            result.update(status="blocked", failure=failure)
        finally:
            reset_task_id(token)
            await owner.close()
        hook(
            "survey.complete",
            {
                "accepted": False,
                "quality_status": "unverified",
                "delivery_status": "blocked",
                "total_tokens": (
                    budget.used if not budget.snapshot()["usage_uncertain"] else None
                ),
            },
        )
        if tracer:
            recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            tracer.flush()
        artifact = recorder.finalize(error=failure["code"] if failure else None)
        known = sum(
            e["data"]["total_tokens"]
            for e in artifact["events"]
            if e["event"] == "llm.complete" and e["data"].get("usage_available") is True
        )
        entry = session.journal["calls"].get("body", {}) if session else {}
        result.update(
            known_tokens=known,
            requests=budget.snapshot()["requests_sent"],
            unknown_hold=max(0, budget.used - known),
            usage_complete=not budget.snapshot()["usage_uncertain"],
            elapsed_ms=round((time.monotonic() - start) * 1000),
            trace_url=tracer.trace_url if tracer else None,
            raw_sha256=entry.get("raw_sha256"),
            finish_reason=entry.get("finish_reason"),
            budget=budget.snapshot(),
        )
        atomic_json(path, result)
        return publish(root, result)


def main():
    """Require explicit prepare or single-call execution for a new local observation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    result = (
        prepare(args.root, config)
        if args.prepare
        else asyncio.run(execute(args.root, config))
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
