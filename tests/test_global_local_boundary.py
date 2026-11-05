"""Keep exact byte limits inside the real global request without dropping content."""

import copy
import json
from pathlib import Path

import pytest
from test_evaluation_dimensions_identity import dimension_envelope
from test_global_local_input import flatten

from litagent.config import load_config
from litagent.eval import batched, scoped
from litagent.eval.results import CONTRACT
from litagent.output_capacity import configured_output_tokens
from litagent.safety.budget import strict_request_reservation
from litagent.safety.planned_budget import (
    check_inventory_envelope,
    check_writer_envelope,
    nested_json_size,
)
from litagent.writing_v2 import RANGE_WRITING_PLAN, compile_document


@pytest.mark.parametrize("references", [0, 1])
def test_full_operational_bytes_and_sparse_citations_fit_actual_global_request(
    references,
):
    fixture = json.loads(
        (
            Path(__file__).parent / f"fixtures/global-boundary-refs-{references}.json"
        ).read_text()
    )
    assert fixture["scientific_gold_standard"] is False
    envelope = dimension_envelope(
        evaluation_result_contract=CONTRACT,
        output_tokens=configured_output_tokens(
            load_config("config/expanded-capacity.yaml")
        ),
    )
    body, summaries = fixture["body"], fixture["summaries"]
    payload = copy.deepcopy(fixture["payload"])
    payload["writer_envelope"] = envelope
    payload["schema"] = batched.global_review_schema(payload)
    check_writer_envelope(body, envelope, stage="body")
    check_writer_envelope(summaries, envelope, stage="summary")
    check_inventory_envelope(payload["assertions"], envelope)
    assert nested_json_size(body) == envelope["body_json_bytes"] == 32000
    assert nested_json_size(summaries) == envelope["summary_json_bytes"] == 4000
    assert nested_json_size([c["claim_text"] for c in payload["assertions"]]) == 30000
    _, binding = compile_document(
        body,
        summaries,
        fixture["cards"],
        fixture["aliases"],
        "zh-CN",
        {},
        writing_plan=RANGE_WRITING_PLAN,
        envelope=envelope,
    )
    assert len(binding["units"]) == 70
    messages, native = scoped.build_evaluation_request(
        batched.global_prompt(payload, native_unit_ownership=True), payload
    )
    wire = json.loads(messages[1]["content"])
    assert len(flatten(wire)) == len(payload["assertions"]) == 128
    assert (
        sum(
            "fragment" in p
            for group in wire["report_sections"]
            for p in group["layout"]
        )
        == 70
    )
    assert strict_request_reservation(messages, 32768, native) <= 120000
    assert payload["assertions"] == fixture["payload"]["assertions"]
    # Capacity success grants no semantic validity or support to this synthetic text.
    assert all(c["kind"] == "scientific" for c in payload["assertions"])
