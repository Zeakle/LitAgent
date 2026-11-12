"""Keep flat summary JSON guidance aligned with native and host consumers."""

import base64
import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_summary_body_view import current_scope
from test_writer_v2 import case  # noqa: F401

from litagent.format_repair import check_summary_retry_capacity, summary_retry_messages
from litagent.safety.budget import strict_request_reservation
from litagent.writer_source_view import (
    GROUNDING_INSTRUCTION,
    SUMMARY_JSON_CONTRACT,
    summary_json_guidance,
)
from litagent.writing_flat import parse, prepare
from litagent.writing_v2 import (
    RANGE_WRITING_PLAN,
    WriterError,
    body_units,
    compile_document,
    prepare_planned_input,
    provider_response_format,
    request_messages,
    summary_request_input,
)


def summary_messages(case):
    """Use actual frozen scope, payload and the final request builder."""
    scope = current_scope()
    system, payload, schema = summary_request_input(
        case.packet, body_units(case.body), scope
    )
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, scope
    )
    return (
        scope,
        payload,
        schema,
        request_messages(system, payload, schema, wire_contract="flat-writer-v1"),
    )


def test_actual_summary_contains_valid_values_instead_of_field_name_lists(case):
    scope, payload, schema, messages = summary_messages(case)
    system = messages[0]["content"]
    assert SUMMARY_JSON_CONTRACT in system
    assert "Native output type map" not in system
    assert "item keys" not in system
    assert "whole requested chapter" not in system
    assert "For patches return" not in system
    prefix = "JSON encoding example only: "
    example, _ = json.JSONDecoder().raw_decode(system.split(prefix, 1)[1])
    native = provider_response_format("summary", schema, wire_contract="flat-writer-v1")
    Draft202012Validator(native["json_schema"]["schema"]).validate(example)
    canonical = parse(json.dumps(example), schema)["canonical"]
    report, bindings = compile_document(
        case.body, canonical, case.cards, case.aliases, "en", envelope=scope
    )
    assert bindings["report_sha256"] and report
    assert json.loads(messages[1]["content"])["body_units"] == payload["body_units"]


def test_066_internal_json_is_still_rejected_without_guessing(case):
    _, _, schema, _ = summary_messages(case)
    original = '{"abstract":[{"text","body_unit_refs":["u1","u2","u5","u10"],"paragraph_start":true}],"conclusion":[]}'
    with pytest.raises(WriterError):
        parse(original, schema)


def test_example_must_obey_actual_schema_and_cannot_widen_authority(case):
    _, payload, schema, _ = summary_messages(case)
    flat = prepare(schema)["wire_schema"]
    invalid = copy.deepcopy(flat)
    invalid["properties"]["abstract"]["items"]["properties"]["body_unit_refs"][
        "items"
    ] = {"enum": ["unavailable"]}
    with pytest.raises(WriterError, match="summary_json_guidance_invalid"):
        summary_json_guidance(invalid, payload)
    legacy = {key: value for key, value in payload.items() if key != "body_sections"}
    assert summary_json_guidance(flat, legacy) is None
    assert summary_json_guidance(schema, payload) is None


def test_live_420_complete_summary_request_fits_original_format_reservation():
    """Rebuild the summary and retry for all frozen premises, without sending."""
    root = Path(
        "artifacts/acceptance/workbench-few-shot-420-20261112/runtime/"
        "targeted-repairs"
    )
    writer = root / "f0326ac4f8a3.json.writer-v2.json"
    ledger = root / "f0326ac4f8a3.json.budget-ledger.json"
    if not writer.is_file() or not ledger.is_file():
        pytest.skip("Local 420 summary and budget are unavailable")
    entry = json.loads(writer.read_bytes())["calls"]["summary_initial"]
    plan = json.loads(ledger.read_bytes())["plan"]
    original = json.loads(
        base64.b64decode(entry["provider_receipt"]["request_body_base64"])
    )
    payload = json.loads(original["input"][1]["content"])
    units = payload["body_units"]
    assert len(units) == 48 and "body_sections" not in payload
    scope = plan["envelope"]
    system, _, schema = summary_request_input(payload, units, scope)
    assert GROUNDING_INSTRUCTION in original["input"][0]["content"]
    system += "\n" + GROUNDING_INSTRUCTION
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    native = provider_response_format("summary", schema, wire_contract="flat-writer-v1")
    retry = summary_retry_messages(messages)
    assert retry is not messages
    assert retry[1:] == messages[1:] == original["input"][1:]
    assert native["json_schema"]["schema"] == original["text"]["format"]["schema"]
    output = original["max_output_tokens"]
    assert output == plan["stages"]["format_summary"]["output_tokens"]
    check_summary_retry_capacity(messages, native, output, scope)
    assert strict_request_reservation(retry, output, native) <= min(
        entry["reservation"], plan["stages"]["format_summary"]["request_cap"]
    )
