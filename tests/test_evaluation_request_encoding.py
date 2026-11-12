"""Keep native Evaluation schemas single and all scientific inputs complete."""

import copy
import json
from types import SimpleNamespace

import pytest

from litagent.body_quality import EVALUATION_ENCODING_CONTRACT
from litagent.config import EvalConfig
from litagent.eval.batched import request_size
from litagent.eval.scoped import (
    _record_request_rejection,
    _request_identity,
    build_evaluation_request,
    check_request_budget,
    evaluation_response_format,
)
from litagent.safety.budget import strict_request_reservation
from litagent.workbench import RunOptions


@pytest.mark.parametrize(
    "field,contract",
    [
        ("inventory_output_contract", "host-flat-fragment-inventory-v1"),
        ("assessment_output_contract", "host-batch-assessment-v1"),
        ("global_output_contract", "host-global-completeness-v1"),
    ],
)
def test_native_projection_retains_every_scientific_byte_and_schema(field, contract):
    """Only the duplicate top-level schema changes on the actual wire."""
    payload = {
        field: contract,
        "schema": {"type": "object", "properties": {"answer": {"type": "string"}}},
        "original_supports": {"a": {"text": "条件与结果。" * 10000}},
        "source_units": [{"id": "u", "text": "A result. [E:a]"}],
        "assertions": [{"id": "c", "quantity": 12, "condition": "dataset A"}],
    }
    before = copy.deepcopy(payload)
    messages, native = build_evaluation_request("Complete evidence only.", payload)
    wire = json.loads(messages[1]["content"])
    assert "schema" not in wire
    assert wire.pop("response_schema_contract") == EVALUATION_ENCODING_CONTRACT
    assert wire == {k: v for k, v in payload.items() if k != "schema"}
    assert native == evaluation_response_format(payload)
    assert native["json_schema"]["schema"] == payload["schema"]
    assert native["json_schema"]["strict"] is True
    assert payload == before
    size = strict_request_reservation(messages, 16384, native)
    assert request_size("Complete evidence only.", payload, 16384) == size
    assert (
        check_request_budget(None, messages, 16384, size, response_format=native)
        == size
    )
    with pytest.raises(ValueError, match="evaluation_context_budget_exceeded"):
        check_request_budget(None, messages, 16384, size - 1, response_format=native)


@pytest.mark.parametrize("contract", [None, "host-fragment-inventory-v2"])
def test_legacy_wire_bytes_remain_unchanged(contract):
    """Historical JSON-object and native v2 inventory keep their exact identities."""
    payload = {"schema": {"type": "object"}, "text": "旧记录"}
    if contract:
        payload["inventory_output_contract"] = contract
    messages, native = build_evaluation_request("Original prompt", payload)
    assert messages == [
        {"role": "system", "content": "Original prompt"},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]
    assert native == evaluation_response_format(payload)


def test_rejected_input_is_the_exact_planned_wire_and_keeps_full_sources():
    """Local refusal receipts use the same binding and reservation without sending."""
    payload = {
        "assessment_output_contract": "host-batch-assessment-v1",
        "schema": {"type": "object"},
        "original_supports": {"s": {"text": "Complete source. " * 10000}},
    }
    progress, events = {}, []
    llm = SimpleNamespace(_trace_hook=lambda event, data: events.append((event, data)))
    _record_request_rejection(
        llm,
        "Verify",
        payload,
        reason="evaluation_context_budget_exceeded",
        phase="assessment",
        config=EvalConfig(),
        progress=progress,
    )
    messages, native = build_evaluation_request("Verify", payload)
    record = progress["rejected_requests"][0]
    assert record["status"] == "rejected_not_sent"
    assert record["input_sha256"] == _request_identity(messages, native)
    assert record["budget"]["reservation"] == request_size("Verify", payload, 16384)
    assert events[0][1]["messages"] == messages
    assert events[0][1]["response_format"] == native
    assert (
        json.loads(messages[1]["content"])["original_supports"]
        == payload["original_supports"]
    )


def test_encoding_version_changes_only_analysis_run_identity(monkeypatch):
    """New wire contracts cannot reuse analysis checkpoints under an old run identity."""
    import litagent.body_quality as quality

    current = RunOptions(quality_policy="strict_auto_v2", body_quality="analysis_v1")
    legacy = RunOptions()
    current_before, legacy_before = current.fingerprint(), legacy.fingerprint()
    monkeypatch.setattr(quality, "EVALUATION_ENCODING_CONTRACT", "different-wire")
    assert current.fingerprint() != current_before
    assert legacy.fingerprint() == legacy_before
