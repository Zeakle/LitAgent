"""Test one ordinary-array encoding without generating or accepting a Survey."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import sys
from importlib.metadata import version
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema import ValidationError as SchemaValidationError

from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.llm.client import OpenAICompatibleClient
from litagent.observability.context import reset_task_id, set_task_id
from litagent.observability.recorder import CompositeTraceHook, RedactingTraceHook
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import atomic_json, digest
from litagent.safety.budget import CostBudget, strict_request_reservation

CONTRACT = "native-array-protocol-v1"
MODEL = "doubao-seed-2-1-pro-260915"
ORIGIN = "https://ark.cn-beijing.volces.com/api/v3"
LIMIT = 50000
REAL_CONTRACT = "native-inline-real-input-v1"
REAL_LIMIT = 120000
OUTPUT = 20000
BODY = ("introduction", "taxonomy", "methods", "experiments", "open_problems")
SECTIONS = (*BODY, "abstract", "conclusion")
BODY_FIELDS = {"kind", "text", "fact_card_ids", "scope_notice_id"}
SUMMARY_FIELDS = {"text", "body_references"}


def sha(path):
    """Hash immutable inputs without disclosing their contents."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def closed(value, keys):
    """Reject extra, missing and mixed transport fields before projecting."""
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError("array_wire_shape_invalid")


def unique(pairs):
    """Reject duplicate JSON keys rather than selecting their last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("array_wire_duplicate_key")
        result[key] = value
    return result


def array_wire_schema(native_schema):
    """Project known v22 containers; retain unit permissions and source definitions."""
    source = copy.deepcopy(native_schema)
    Draft202012Validator.check_schema(source)
    try:
        closed(source["properties"], {"title", "sections"})
        sections = source["properties"]["sections"]["properties"]
        closed(sections, SECTIONS)
        for section in sections.values():
            closed(section["properties"], {"first_block", "additional_blocks"})
            ref = section["properties"]["first_block"]
            if section["properties"]["additional_blocks"] != {
                "type": "array",
                "items": ref,
            }:
                raise ValueError("array_wire_source_schema_invalid")
            name = ref["$ref"].removeprefix("#/$defs/")
            block = source["$defs"][name]
            properties = block["properties"]
            closed(
                properties,
                {"heading", "type", "first_sentence", "additional_sentences"},
            )
            # First-position restrictions remain mandatory in the original local
            # schema after projection; ordinary items are only a transport shape.
            sentence = properties["additional_sentences"]["items"]
            block["properties"] = {
                "heading": properties["heading"],
                "type": properties["type"],
                "sentences": {"type": "array", "items": sentence},
            }
            block["required"] = list(block["properties"])
            section["properties"] = {"blocks": {"type": "array", "items": ref}}
            section["required"] = ["blocks"]
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("array_wire_source_schema_invalid") from exc
    Draft202012Validator.check_schema(source)
    return source


def _check_unit(unit, section):
    """Validate known value types while leaving scientific support unverified."""
    closed(unit, BODY_FIELDS if section in BODY else SUMMARY_FIELDS)
    if not isinstance(unit["text"], str):
        raise ValueError("array_wire_unit_invalid")
    if section in BODY:
        if (
            not isinstance(unit["kind"], str)
            or unit["kind"] not in {"scientific", "organization", "gap"}
            or not isinstance(unit["scope_notice_id"], str)
            or not isinstance(unit["fact_card_ids"], list)
            or any(not isinstance(cid, str) for cid in unit["fact_card_ids"])
        ):
            raise ValueError("array_wire_unit_invalid")
    elif not isinstance(unit["body_references"], list) or not unit["body_references"]:
        raise ValueError("array_wire_reference_invalid")


def array_from_native(document):
    """Encode canonical containers without editing unit values or ordering."""
    closed(document, {"title", "sections"})
    closed(document["sections"], SECTIONS)
    result = {"title": document["title"], "sections": {}}
    for key, section in document["sections"].items():
        closed(section, {"first_block", "additional_blocks"})
        if not isinstance(section["additional_blocks"], list):
            raise ValueError("array_wire_shape_invalid")
        blocks = []
        for block in [section["first_block"], *section["additional_blocks"]]:
            closed(block, {"heading", "type", "first_sentence", "additional_sentences"})
            if not isinstance(block["additional_sentences"], list):
                raise ValueError("array_wire_shape_invalid")
            units = [block["first_sentence"], *block["additional_sentences"]]
            for unit in units:
                _check_unit(unit, key)
            blocks.append(
                {
                    "heading": block["heading"],
                    "type": block["type"],
                    "sentences": copy.deepcopy(units),
                }
            )
        result["sections"][key] = {"blocks": blocks}
    return result


def canonical_from_array(document):
    """Restore exact indices and field values, recording only a format projection."""
    if isinstance(document, str):
        document = json.loads(document, object_pairs_hook=unique)
    closed(document, {"title", "sections"})
    if not isinstance(document["title"], str):
        raise ValueError("array_wire_shape_invalid")
    closed(document["sections"], SECTIONS)
    result, positions = {"title": document["title"], "sections": {}}, []
    for key, section in document["sections"].items():
        closed(section, {"blocks"})
        if not isinstance(section["blocks"], list) or not section["blocks"]:
            raise ValueError("array_wire_shape_invalid")
        blocks = []
        for bi, block in enumerate(section["blocks"]):
            closed(block, {"heading", "type", "sentences"})
            if not isinstance(block["heading"], str) or block["type"] not in {
                "paragraph",
                "list",
            }:
                raise ValueError("array_wire_shape_invalid")
            units = block["sentences"]
            if not isinstance(units, list) or not units:
                raise ValueError("array_wire_shape_invalid")
            for si, unit in enumerate(units):
                _check_unit(unit, key)
                positions.append(
                    {
                        "wire_path": f"sections.{key}.blocks.{bi}.sentences.{si}",
                        "canonical_path": f"sections.{key}."
                        + ("first_block" if bi == 0 else f"additional_blocks.{bi-1}")
                        + "."
                        + (
                            "first_sentence"
                            if si == 0
                            else f"additional_sentences.{si-1}"
                        ),
                        "unit_sha256": digest(unit),
                    }
                )
            blocks.append(
                {
                    "heading": block["heading"],
                    "type": block["type"],
                    "first_sentence": copy.deepcopy(units[0]),
                    "additional_sentences": copy.deepcopy(units[1:]),
                }
            )
        result["sections"][key] = {
            "first_block": blocks[0],
            "additional_blocks": blocks[1:],
        }
    for key in ("abstract", "conclusion"):
        for block in document["sections"][key]["blocks"]:
            for unit in block["sentences"]:
                seen = set()
                for ref in unit["body_references"]:
                    closed(ref, {"section_key", "block_index", "sentence_index"})
                    k, bi, si = (
                        ref["section_key"],
                        ref["block_index"],
                        ref["sentence_index"],
                    )
                    if (
                        k not in BODY
                        or type(bi) is not int
                        or type(si) is not int
                        or min(bi, si) < 0
                        or (k, bi, si) in seen
                    ):
                        raise ValueError("array_wire_reference_invalid")
                    seen.add((k, bi, si))
                    try:
                        target = document["sections"][k]["blocks"][bi]["sentences"][si]
                    except IndexError as exc:
                        raise ValueError("array_wire_reference_invalid") from exc
                    if target["kind"] != "scientific":
                        raise ValueError("array_wire_reference_invalid")
    return result, {
        "contract": CONTRACT,
        "wire_document_sha256": digest(document),
        "canonical_sha256": digest(result),
        "positions": positions,
        "scientific_verdict": "not_evaluated",
    }


def protocol_document(schema):
    """Build a non-scientific protocol fixture from actual source permissions."""
    result = {"title": "PROTOCOL_PROBE", "sections": {}}
    for key in SECTIONS:
        if key in BODY:
            ref = schema["properties"]["sections"]["properties"][key]["properties"][
                "blocks"
            ]["items"]["$ref"].split("/")[-1]
            unit_ref = schema["$defs"][ref]["properties"]["sentences"]["items"][
                "$ref"
            ].split("/")[-1]
            cid = schema["$defs"][unit_ref]["properties"]["fact_card_ids"]["items"][
                "enum"
            ][0]
            unit = {
                "text": "PROTOCOL_PROBE",
                "kind": "scientific",
                "fact_card_ids": [cid],
                "scope_notice_id": "none",
            }
        else:
            unit = {
                "text": "PROTOCOL_PROBE",
                "body_references": [
                    {"section_key": "methods", "block_index": 0, "sentence_index": 0}
                ],
            }
        block = {
            "heading": "PROTOCOL_PROBE" if key in BODY else "",
            "type": "paragraph",
            "sentences": [copy.deepcopy(unit) for _ in range(2 if key in BODY else 1)],
        }
        result["sections"][key] = {
            "blocks": [copy.deepcopy(block) for _ in range(2 if key in BODY else 1)]
        }
    return result


def prepare_protocol(root, parent, *, scenario=CONTRACT):
    """Freeze a single request and the completed parent accounting before execution."""
    summary_path = parent / "acceptance-summary.json"
    summary = json.loads(summary_path.read_text())
    archive_path = Path("artifacts/runs") / (
        summary["suite_id"] + "-pilot-few-shot-metric.json"
    )
    if (
        summary["status"] != "stopped"
        or summary["new_unknown_usage"] is not False
        or sha(archive_path) != summary["source_archive_sha256"]
    ):
        raise ValueError("array_protocol_parent_invalid")
    archive = json.loads(archive_path.read_text())
    starts = [e["data"] for e in archive["events"] if e["event"] == "llm.start"]
    ends = [e["data"] for e in archive["events"] if e["event"] == "llm.complete"]
    if (
        len(starts) != summary["new_requests_started"]
        or len(ends) != len(starts)
        or any(e["usage_available"] is not True for e in ends)
        or sum(e["total_tokens"] for e in ends) != summary["new_known_tokens"]
    ):
        raise ValueError("array_protocol_parent_usage_invalid")
    original = starts[-1]["response_format"]["json_schema"]["schema"]
    if scenario not in {CONTRACT, REAL_CONTRACT}:
        raise ValueError("array_protocol_scenario_invalid")
    real_input = scenario == REAL_CONTRACT
    limit = REAL_LIMIT if real_input else LIMIT
    if real_input:
        from litagent.body_quality import NATIVE_SCHEMA_ENCODING_CONTRACT
        from litagent.native_wire import inline_local_schema

        wire_schema = inline_local_schema(original)
    else:
        wire_schema = array_wire_schema(original)
        canonical, _ = canonical_from_array(protocol_document(wire_schema))
        Draft202012Validator(original).validate(canonical)
    reference = json.dumps(
        [{"section_key": "methods", "block_index": 0, "sentence_index": 0}],
        separators=(",", ":"),
    )
    messages = [
        {
            "role": "user",
            "content": "This is one protocol test, NOT a Survey or scientific evaluation. Output JSON only matching the native schema. All free text must be PROTOCOL_PROBE; summary headings must be empty. Use all seven sections. Each of the five body sections must contain exactly two blocks with exactly two sentences each. Body units: kind scientific, scope_notice_id none, and exactly the first allowed fact_card_id for that section. Each abstract/conclusion has one paragraph block and one sentence with body_references="
            + reference
            + ". No paper content, real findings, commentary or extra keys. Do not use first_block/additional_blocks or first_sentence/additional_sentences.",
        }
    ]
    if real_input:
        messages = copy.deepcopy(starts[-1]["messages"])
        packet = json.loads(messages[1]["content"])
        if packet.get("native_wire_contract") != "native-array-wire-v1":
            raise ValueError("array_protocol_parent_wire_invalid")
        packet["native_schema_encoding_contract"] = NATIVE_SCHEMA_ENCODING_CONTRACT
        messages[1]["content"] = json.dumps(
            packet, ensure_ascii=False, separators=(",", ":")
        )
    frame = {
        "messages": messages,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "native_array_protocol_v1",
                "strict": True,
                "schema": wire_schema,
            },
        },
        "max_tokens": OUTPUT,
    }
    if real_input:
        frame["response_format"] = copy.deepcopy(starts[-1]["response_format"])
        frame["response_format"]["json_schema"]["schema"] = wire_schema
    frame["reservation"] = strict_request_reservation(
        messages, OUTPUT, frame["response_format"]
    )
    if (
        frame["reservation"] > min(limit, 117964)
        or summary["new_known_tokens"] + limit > 500000
        or len(starts) + 1 > 64
    ):
        raise ValueError("array_protocol_budget_invalid")
    value = {
        "contract": scenario,
        "run_id": root.name,
        "root": str(root.resolve()),
        "model": MODEL,
        "origin": ORIGIN,
        "max_requests": 1,
        "max_tokens": limit,
        "retries": 0,
        "frame": frame,
        "canonical_schema": None if real_input else original,
        "original_array_schema": original if real_input else None,
        "parent": str(summary_path.resolve()),
        "parent_sha256": sha(summary_path),
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha(archive_path),
        "script_sha256": sha(__file__),
        "source_hashes": {
            str(p): sha(p)
            for p in sorted(
                (Path(__file__).resolve().parents[1] / "src/litagent").rglob("*.py")
            )
        },
        "script": str(Path(__file__).resolve()),
        "runtime": {
            "python": sys.version,
            "packages": {
                p: version(p) for p in ("openai", "httpx", "pydantic", "jsonschema")
            },
        },
        "historical": summary["cumulative"],
        "parent_requests": len(starts),
        "parent_tokens": summary["new_known_tokens"],
        "scientific_verdict": "not_applicable",
    }
    value["binding"] = digest(value)
    root.mkdir(parents=True, exist_ok=False)
    (root / "probe.py").write_bytes(Path(__file__).read_bytes())
    atomic_json(root / "probe-manifest.json", value)
    return value


def validate_manifest(root):
    """Reject changed script, parent receipt, schema or runtime without opening transport."""
    m = json.loads((root / "probe-manifest.json").read_text())
    if (
        digest({k: v for k, v in m.items() if k != "binding"}) != m["binding"]
        or m["contract"] not in {CONTRACT, REAL_CONTRACT}
        or m["run_id"] != root.name
        or m["root"] != str(root.resolve())
        or m["model"] != MODEL
        or m["origin"] != ORIGIN
        or m["max_requests"] != 1
        or m["max_tokens"] != (REAL_LIMIT if m["contract"] == REAL_CONTRACT else LIMIT)
        or m["retries"] != 0
        or m["script_sha256"] != sha(__file__)
        or m["runtime"]
        != {
            "python": sys.version,
            "packages": {
                p: version(p) for p in ("openai", "httpx", "pydantic", "jsonschema")
            },
        }
        or m["parent_sha256"] != sha(m["parent"])
        or m["archive_sha256"] != sha(m["archive"])
    ):
        raise ValueError("array_protocol_identity_conflict")
    if (
        any(
            not Path(p).is_file() or sha(p) != expected
            for p, expected in m["source_hashes"].items()
        )
        or sha(root / "probe.py") != m["script_sha256"]
    ):
        raise ValueError("array_protocol_source_changed")
    parent = json.loads(Path(m["parent"]).read_text())
    if (
        m["historical"] != parent["cumulative"]
        or m["parent_requests"] != parent["new_requests_started"]
        or m["parent_tokens"] != parent["new_known_tokens"]
    ):
        raise ValueError("array_protocol_parent_usage_invalid")
    if (
        type(m["parent_requests"]) is not int
        or not 0 <= m["parent_requests"] < 64
        or type(m["parent_tokens"]) is not int
        or not 0 <= m["parent_tokens"] <= 500000 - m["max_tokens"]
    ):
        raise ValueError("array_protocol_budget_invalid")
    if (
        type(m["frame"]["reservation"]) is not int
        or m["frame"]["max_tokens"] != OUTPUT
        or strict_request_reservation(
            m["frame"]["messages"], OUTPUT, m["frame"]["response_format"]
        )
        != m["frame"]["reservation"]
        or m["frame"]["reservation"] > min(m["max_tokens"], 117964)
    ):
        raise ValueError("array_protocol_budget_invalid")
    if m["contract"] == REAL_CONTRACT:
        from litagent.body_quality import NATIVE_SCHEMA_ENCODING_CONTRACT
        from litagent.native_wire import inline_local_schema

        archive = json.loads(Path(m["archive"]).read_text())
        start = [e["data"] for e in archive["events"] if e["event"] == "llm.start"][-1]
        expected_messages = copy.deepcopy(start["messages"])
        packet = json.loads(expected_messages[1]["content"])
        packet["native_schema_encoding_contract"] = NATIVE_SCHEMA_ENCODING_CONTRACT
        expected_messages[1]["content"] = json.dumps(
            packet, ensure_ascii=False, separators=(",", ":")
        )
        expected_format = copy.deepcopy(start["response_format"])
        expected_format["json_schema"]["schema"] = inline_local_schema(
            start["response_format"]["json_schema"]["schema"]
        )
        if (
            m["frame"]["messages"] != expected_messages
            or m["frame"]["response_format"] != expected_format
        ):
            raise ValueError("array_protocol_parent_input_changed")
        _real_cards(m)
    return m


def _real_cards(m):
    """Bind full raw selected evidence, not the lossy UI evidence projection."""
    from litagent.quality import build_fact_cards

    archive = json.loads(Path(m["archive"]).read_text())
    packet = json.loads(m["frame"]["messages"][1]["content"])
    selected = next(
        node["output"]["selected_items"]
        for node in archive["nodes"].values()
        if node.get("kind") == "evidence.retrieve"
    )
    ledger = (
        selected
        if isinstance(selected, dict)
        else {e["evidence_id"]: e for e in selected}
    )
    all_cards = build_fact_cards(ledger)
    cards = {cid: all_cards[cid] for cid in packet["cards"]}
    for cid, card in cards.items():
        card["allowed_section_keys"] = packet["cards"][cid]["allowed_section_keys"]
        source = packet["sources"][packet["cards"][cid]["source_ref"]]
        if (
            card["source_quote"] != source["source_quote"]
            or card["paper_id"] != source["paper_id"]
        ):
            raise ValueError("array_protocol_source_binding_invalid")
    return cards


def _finalize(root, m, state):
    """Interpret a durably received response locally, never by resending it."""
    row = state["received"]
    try:
        state["raw_response_sha256"] = hashlib.sha256(
            row["content"].encode()
        ).hexdigest()
        if (
            row["actual_model"] != MODEL
            or row["finish_reason"] != "stop"
            or not row["usage_available"]
        ):
            raise ValueError("array_protocol_response_identity_invalid")
        decoded = json.loads(row["content"], object_pairs_hook=unique)
        Draft202012Validator(
            m["frame"]["response_format"]["json_schema"]["schema"]
        ).validate(decoded)
        native, proof = canonical_from_array(decoded)
        if m["contract"] == REAL_CONTRACT:
            from litagent.quality import render_grounded_report

            packet = json.loads(m["frame"]["messages"][1]["content"])
            cards = _real_cards(m)
            rendered = render_grounded_report(
                json.dumps(native, ensure_ascii=False),
                cards,
                "academic_v2",
                packet["report_language"],
                require_native=True,
                require_source_quotes=True,
                require_full_sources=True,
                require_scope_notices=True,
                analysis_mode=True,
                require_unit_citation_scope=True,
            )
            state["local_report_sha256"] = hashlib.sha256(rendered.encode()).hexdigest()
            if array_from_native(native) != decoded:
                raise ValueError("array_protocol_projection_not_lossless")
        else:
            Draft202012Validator(m["canonical_schema"]).validate(native)
            if decoded != protocol_document(
                m["frame"]["response_format"]["json_schema"]["schema"]
            ):
                raise ValueError("array_protocol_fixed_fixture_mismatch")
        state.update(status="completed", projection_proof=proof)
    except SchemaValidationError as exc:
        state.update(
            status="protocol_failed",
            error_type=type(exc).__name__,
            schema_error={"path": list(exc.absolute_path), "validator": exc.validator},
        )
    except (ValueError, TypeError, KeyError, IndexError) as exc:
        state.update(
            status="protocol_failed",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        if isinstance(exc, json.JSONDecodeError):
            state["json_error"] = {
                "offset": exc.pos,
                "line": exc.lineno,
                "column": exc.colno,
            }
    except Exception as exc:
        state.update(status="protocol_failed", error_type=type(exc).__name__)
    atomic_json(root / "probe-state.json", state)
    return state


def save_receipt(root, m, state):
    """Carry every known fee or unknown allowance forward without adding a Survey."""
    row, budget = state.get("received", {}), state.get("budget", {})
    known = row.get("usage_available") is True
    tokens = (
        (int(row.get("prompt_tokens", 0)) + int(row.get("completion_tokens", 0)))
        if known
        else 0
    )
    requests = max(
        budget.get("requests_sent", 0), state.get("conservative_requests", 0)
    )
    hold = state.get("unknown_hold", 0)
    if (
        requests not in (0, 1)
        or tokens < 0
        or hold not in (0, m["max_tokens"])
        or tokens + hold > m["max_tokens"]
    ):
        raise ValueError("array_protocol_accounting_invalid")
    cumulative = copy.deepcopy(m["historical"])
    cumulative["model_requests"] += requests
    cumulative["model_known_tokens"] += tokens
    cumulative["model_unknown_conservative_tokens"] += hold
    receipt = {
        "status": state["status"],
        "binding": m["binding"],
        "new_requests": requests,
        "new_known_tokens": tokens,
        "new_unknown_hold": hold,
        "family_requests": m["parent_requests"] + requests,
        "family_known_tokens": m["parent_tokens"] + tokens,
        "family_unknown_hold": hold,
        "cumulative": cumulative,
        "scientific_verdict": "not_applicable",
        "new_survey_rounds": 0,
        "phase14_5": "in_progress",
        "trace_id": state.get("trace_id"),
        "trace_url": state.get("trace_url"),
    }
    path = root / "acceptance-summary.json"
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise ValueError("array_protocol_receipt_conflict")
    if not path.exists():
        atomic_json(path, receipt)
    return state


async def run_protocol(root, *, live=False):
    """Issue at most one known request; terminal or interrupted identities never replay."""
    m = validate_manifest(root)
    limit = m["max_tokens"]
    if not live:
        return {
            "status": "planned",
            "binding": m["binding"],
            "reservation": m["frame"]["reservation"],
            "paid_requests": 0,
        }
    config = load_config()
    if (
        config.llm.base_url.rstrip("/") != ORIGIN
        or config.llm.model_for("generation") != MODEL
    ):
        raise ValueError("array_protocol_model_configuration_invalid")
    with suite_lock(root / "probe.lock"):
        path = root / "probe-state.json"
        if m["contract"] == REAL_CONTRACT:
            # A grant belongs to the parent, not to a freely copied output root.
            grant = Path(m["parent"]).parent / "inline-protocol-owner.json"
            with suite_lock(grant.with_suffix(".lock")):
                owner_identity = {"binding": m["binding"], "root": m["root"]}
                if grant.exists() and json.loads(grant.read_text()) != owner_identity:
                    raise ValueError("array_protocol_parent_already_owned")
                if not grant.exists():
                    atomic_json(grant, owner_identity)
        if path.exists():
            state = json.loads(path.read_text())
            if state["binding"] != m["binding"]:
                raise ValueError("array_protocol_identity_conflict")
            if state["status"] in {"response_received", "completed", "protocol_failed"}:
                return save_receipt(root, m, _finalize(root, m, state))
            if state["status"] == "started_no_replay":
                state.update(
                    status="interrupted_no_replay",
                    unknown_hold=limit,
                    conservative_requests=1,
                )
                atomic_json(path, state)
            return save_receipt(root, m, state)
        state = {
            "binding": m["binding"],
            "status": "started_no_replay",
            "unknown_hold": limit,
            "conservative_requests": 1,
        }
        atomic_json(path, state)
        budget = CostBudget(max_tokens=limit)
        budget.configure_strict(halt_on_unknown_usage=True)

        def journal(event, data):
            """Persist known responses before local parsing and redact external trace I/O."""
            with (root / "events.jsonl").open("a") as stream:
                stream.write(
                    json.dumps({"event": event, "data": data}, ensure_ascii=False)
                    + "\n"
                )
            if event == "llm.complete":
                state.update(
                    received=data,
                    budget=budget.snapshot(),
                    status="response_received",
                    unknown_hold=0 if data["usage_available"] else limit,
                )
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
        hook(
            "survey.start",
            {"session_id": root.name, "query": "Protocol conformance only"},
        )
        state.update(trace_id=tracer.trace_id, trace_url=tracer.trace_url)
        atomic_json(path, state)
        owner = None
        wire = []
        try:
            owner = OpenAICompatibleClient(
                ORIGIN,
                MODEL,
                cost_budget=budget,
                trace_hook=hook,
                transport_max_attempts=1,
            )

            async def capture(request):
                """Capture safe wire fields; abort a second transport before sending."""
                if wire:
                    raise ValueError("array_protocol_second_request_forbidden")
                body = json.loads(request.content)
                wire.append(
                    {
                        "url": str(request.url),
                        "model": body["model"],
                        "thinking": body.get("thinking"),
                        "response_format": body.get("response_format"),
                        "max_tokens": body.get("max_tokens"),
                        "body_sha256": hashlib.sha256(request.content).hexdigest(),
                    }
                )
                atomic_json(root / "wire.json", wire)
                if (
                    str(request.url) != ORIGIN + "/chat/completions"
                    or request.method != "POST"
                    or body["model"] != MODEL
                    or body.get("max_tokens") != OUTPUT
                    or body.get("thinking") != {"type": "disabled"}
                    or body.get("response_format") != m["frame"]["response_format"]
                    or body.get("messages") != m["frame"]["messages"]
                    or body.get("temperature") != 0.1
                ):
                    raise ValueError("array_protocol_wire_mismatch")

            owner._client._client.event_hooks["request"].append(capture)
            frame = m["frame"]
            # This independent protocol evaluation owns the complete 50k grant;
            # it is not a preparation phase or a continuation of the failed Survey.
            token = set_task_id("evaluation.protocol_conformance")
            reserved = False
            try:
                budget.reserve(frame["reservation"])
                reserved = True
                await asyncio.wait_for(
                    owner.chat(
                        frame["messages"],
                        response_format=frame["response_format"],
                        max_tokens=OUTPUT,
                        temperature=0.1,
                        extra_body={"thinking": {"type": "disabled"}},
                        transport_max_attempts=1,
                    ),
                    180,
                )
            finally:
                if reserved:
                    budget.release(frame["reservation"])
                reset_task_id(token)
            state = _finalize(root, m, state)
        except BaseException as exc:
            state.update(
                status=(
                    "interrupted_no_replay"
                    if isinstance(exc, asyncio.CancelledError)
                    else "failed_no_replay"
                ),
                error_type=type(exc).__name__,
            )
            sent = budget.snapshot()["requests_sent"]
            state.update(
                conservative_requests=sent,
                unknown_hold=(
                    limit
                    if sent and not state.get("received", {}).get("usage_available")
                    else 0
                ),
            )
        finally:
            state.update(
                budget=budget.snapshot(),
                trace_id=tracer.trace_id,
                trace_url=tracer.trace_url,
            )
            atomic_json(path, state)
            if owner is not None:
                await owner.close()
            hook.flush()
        return save_receipt(root, m, state)


def main():
    """Separate immutable preparation from an explicitly bounded live invocation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--parent", type=Path)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument(
        "--scenario", choices=[CONTRACT, REAL_CONTRACT], default=CONTRACT
    )
    args = parser.parse_args()
    if args.prepare:
        if args.parent is None:
            parser.error("--prepare requires --parent")
        prepare_protocol(args.root, args.parent, scenario=args.scenario)
    result = asyncio.run(run_protocol(args.root, live=args.live))
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in {"received", "projection_proof"}
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
