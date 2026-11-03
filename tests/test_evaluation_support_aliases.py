"""Keep wire passage names reversible without weakening scientific validation."""

import copy
import json

import pytest
from test_evaluation_wire_projection import canonical  # noqa: F401

from litagent.eval.batched import (
    assessment_schema,
    batch_alias_input,
    batch_payload,
    batch_prompt,
    batch_support_aliases,
    restore_batch_support_ids,
)
from litagent.eval.scoped import Assessment, build_evaluation_request
from litagent.safety.budget import strict_request_reservation


@pytest.fixture
def long_passages(canonical):
    """Retain complete passage values while exercising opaque long identities."""
    value = copy.deepcopy(canonical)
    passage = value["support_passages"]["p1"]
    value["support_passages"] = {
        "passage:" + str(i).zfill(64): copy.deepcopy(passage) for i in range(20)
    }
    value["schema"] = assessment_schema(value)
    return value


def decision(pid):
    """Supply a typed response independently of scientific verdict acceptance."""
    return Assessment.model_validate(
        {
            "units": [],
            "conflicts": [],
            "decisions": [
                {
                    "claim_id": "claim_0",
                    "correct_kind": "scientific",
                    "atomic": True,
                    "classification_valid": True,
                    "source_faithful": True,
                    "status": "supported",
                    "reason_code": "supported",
                    "explanation": "A controlled decision, not a scientific proof.",
                    "support_basis": [{"support_id": pid}],
                    "aspect_checks": [
                        {
                            "aspect": "predicate",
                            "supported": True,
                            "support_ids": [pid],
                            "explanation": "Controlled support selection.",
                        }
                    ],
                }
            ],
        }
    )


def test_passage_values_and_schema_restore_exactly(long_passages):
    before = copy.deepcopy(long_passages)
    aliases = batch_support_aliases(before)
    projected = batch_alias_input(long_passages)
    assert len(aliases) == len(set(aliases.values())) == len(before["support_passages"])
    assert {aliases[k]: v for k, v in projected["support_passages"].items()} == (
        before["support_passages"]
    )
    schema = copy.deepcopy(projected["schema"])
    for node in (
        schema["$defs"]["SupportSelection"]["properties"]["support_id"],
        schema["$defs"]["AspectCheck"]["properties"]["support_ids"]["items"],
    ):
        assert node["enum"] == list(aliases)
        node["enum"] = [aliases[k] for k in node["enum"]]
    assert schema == before["schema"]
    for key in before.keys() - {"assertions", "schema", "support_passages"}:
        assert projected[key] == before[key]
    assert before == long_passages


def test_schema_and_response_use_same_short_names_without_changing_raw(long_passages):
    messages, response_format = build_evaluation_request(
        batch_prompt(long_passages), long_passages
    )
    wire = json.loads(messages[1]["content"])
    assert set(wire["support_passages"]) == set(
        response_format["json_schema"]["schema"]["$defs"]["SupportSelection"][
            "properties"
        ]["support_id"]["enum"]
    )
    response = decision("p1")
    raw = response.model_dump_json()
    decoded = restore_batch_support_ids(response, long_passages)
    canonical_id = batch_support_aliases(long_passages)["p1"]
    assert decoded.decisions[0].support_basis[0].support_id == canonical_id
    assert decoded.decisions[0].aspect_checks[0].support_ids == [canonical_id]
    assert response.model_dump_json() == raw
    assert decoded is not response


@pytest.mark.parametrize("pid", ["p999", "passage:" + "0" * 64, "p01"])
def test_unknown_or_mixed_response_identity_never_falls_back(long_passages, pid):
    response = decision(pid)
    raw = response.model_dump()
    with pytest.raises(ValueError, match="response_support_identity_invalid"):
        restore_batch_support_ids(response, long_passages)
    assert response.model_dump() == raw


def test_unknown_aspect_identity_is_rejected_even_with_valid_basis(long_passages):
    response = decision("p1")
    response.decisions[0].aspect_checks[0].support_ids = ["p999"]
    with pytest.raises(ValueError, match="response_support_identity_invalid"):
        restore_batch_support_ids(response, long_passages)


def test_corrupt_canonical_support_enum_is_not_rewritten_into_validity(long_passages):
    invalid = copy.deepcopy(long_passages)
    invalid["schema"]["$defs"]["SupportSelection"]["properties"]["support_id"][
        "enum"
    ] = ["not-a-host-passage"]
    with pytest.raises(ValueError, match="input_support_schema_invalid"):
        batch_alias_input(invalid)


@pytest.mark.parametrize("table", [None, [], {"": []}, {1: []}])
def test_invalid_passage_table_identity_is_explicit(canonical, table):
    canonical["support_passages"] = table
    with pytest.raises(ValueError, match="input_support_identity_invalid"):
        batch_support_aliases(canonical)


def test_long_identity_savings_include_changed_native_schema(long_passages):
    old = copy.deepcopy(long_passages)
    old.pop("assessment_input_contract")
    messages, schema = build_evaluation_request(batch_prompt(old), old)
    projected, new_schema = build_evaluation_request(
        batch_prompt(long_passages), long_passages
    )
    assert strict_request_reservation(messages, 16384, schema) > (
        strict_request_reservation(projected, 16384, new_schema) + 3000
    )
    assert schema != new_schema


def test_fallback_retains_both_original_schema_and_support_names(canonical):
    single = batch_payload(canonical, canonical["assertions"][:1])
    old = copy.deepcopy(single)
    old.pop("assessment_input_contract")
    assert build_evaluation_request(batch_prompt(single), single) == (
        build_evaluation_request(batch_prompt(old), old)
    )
    response = decision("p1")
    assert restore_batch_support_ids(response, single) is response


def test_empty_support_table_keeps_uncited_science_and_negative_response(canonical):
    for claim in canonical["assertions"]:
        claim["evidence"] = []
    canonical["support_passages"] = {}
    canonical["schema"] = assessment_schema(canonical)
    projected = batch_alias_input(canonical)
    assert projected["support_passages"] == {}
    assert len(projected["assertions"]) == 16
    response = decision("unknown")
    response.decisions[0].status = "unsupported"
    response.decisions[0].reason_code = "uncited_claim"
    response.decisions[0].support_basis = []
    response.decisions[0].aspect_checks[0].support_ids = []
    assert restore_batch_support_ids(response, canonical) == response
