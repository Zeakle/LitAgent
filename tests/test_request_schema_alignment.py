"""Compare native producer permissions with the corresponding strict consumers."""

import copy
import json

import pytest
from jsonschema import Draft202012Validator
from test_writer_v2 import case  # noqa: F401

from litagent.agents.extraction_strategy import (
    _FACT_FIELDS,
    _grounded_claim_response_format,
    decode_grounded_claims,
)
from litagent.capacity import CapacityConfig
from litagent.writing_flat import parse, prepare
from litagent.writing_v2 import (
    RANGE_WRITING_PLAN,
    WriterError,
    body_units,
    check_summary_provider,
    prepare_planned_input,
    provider_response_format,
    request_messages,
    summary_request_input,
    writer_budget_plan,
)


def test_extractor_member_permissions_agree_with_native_and_decoder():
    allowed = {"paper_a": ["chunk_a"], "paper_b": ["chunk_b"]}
    native = _grounded_claim_response_format(allowed)
    validator = Draft202012Validator(native["json_schema"]["schema"])
    records = {
        name: {
            "claim_records": [
                {
                    "finding": "A source-bound finding.",
                    "source_chunk_key": keys[0],
                    "confidence": None,
                    "fact_fields": dict.fromkeys(_FACT_FIELDS),
                }
            ]
        }
        for name, keys in allowed.items()
    }
    reply = {"papers": records}
    validator.validate(reply)
    for name, keys in allowed.items():
        decoded, _ = decode_grounded_claims(
            records[name], {keys[0]: "Complete original source."}
        )
        assert len(decoded["claim_records"]) == 1
    invalid = copy.deepcopy(reply)
    invalid["papers"]["paper_a"]["claim_records"][0]["source_chunk_key"] = "chunk_b"
    assert not validator.is_valid(invalid)
    with pytest.raises(ValueError, match="extraction_grounded_source_invalid"):
        decode_grounded_claims(
            invalid["papers"]["paper_a"], {"chunk_a": "Complete original source."}
        )
    invalid = copy.deepcopy(reply)
    invalid["papers"]["paper_a"]["claim_records"][0]["confidence"] = True
    assert not validator.is_valid(invalid)
    with pytest.raises(ValueError, match="extraction_grounded_confidence_invalid"):
        decode_grounded_claims(
            invalid["papers"]["paper_a"], {"chunk_a": "Complete original source."}
        )


@pytest.mark.parametrize("maximum", [1, 2, 3])
def test_summary_final_provider_description_and_host_share_effective_limits(
    case, maximum
):
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        capacity=CapacityConfig(max_summary_units=maximum),
    )
    envelope = plan["envelope"]
    system, payload, schema = summary_request_input(
        case.packet, body_units(case.body), envelope
    )
    system, payload = prepare_planned_input(
        system, payload, case.packet, case.aliases, RANGE_WRITING_PLAN, envelope
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    native = provider_response_format("summary", schema, wire_contract="flat-writer-v1")
    check_summary_provider(native, envelope)
    wire = json.loads(messages[1]["content"])
    assert wire["summary_constraints"]["max_units"] == maximum
    assert "target_sentence_units" not in wire["summary_limits_per_section"]
    validator = Draft202012Validator(native["json_schema"]["schema"])
    reply = {
        section: [
            {
                "text": "An explicitly source-bound finding.",
                "body_unit_refs": ["u3"],
                "paragraph_start": True,
            }
        ]
        for section in ("abstract", "conclusion")
    }
    validator.validate(reply)
    parse(json.dumps(reply), schema)
    for section in reply:
        assert (
            prepare(schema)["wire_schema"]["properties"][section]["maxItems"] == maximum
        )
        assert (
            json.dumps({"maxItems": maximum, "minItems": 1})
            in native["json_schema"]["schema"]["properties"][section]["description"]
        )
    # Cardinality is intentionally host-enforced on the provider's schema subset.
    invalid = copy.deepcopy(reply)
    invalid["conclusion"] *= maximum + 1
    with pytest.raises(WriterError, match="writer_flat_schema_invalid"):
        parse(json.dumps(invalid), schema)
    broken = copy.deepcopy(native)
    broken["json_schema"]["schema"]["properties"]["conclusion"][
        "description"
    ] = "Unrelated limit"
    with pytest.raises(WriterError, match="summary"):
        check_summary_provider(broken, envelope)
