"""Require host-owned locations and citations without treating them as support."""

import json

import pytest
from test_batched_evaluation import BatchLLM, materials

from litagent.config import EvalConfig
from litagent.eval.scoped import (
    ScopedEvaluationError,
    _strict_inventory_response,
    _strict_inventory_schema,
    build_source_fragments,
    evaluate_scoped,
)
from litagent.quality import build_fact_cards, digest
from litagent.report_render import build_report_source_units


def inputs(text=None):
    """Provide genuine host partitions with independently cited source sentences."""
    text = text or (
        "# Report\n\n## Methods\n\nAlpha uses patches. [E:e1] [E:e2] "
        "Beta uses distillation. [E:e3]\n\n## Comparison\n\n"
        "This review compares the methods."
    )
    return [
        {
            "source_unit_id": u.unit_id,
            "source_fragments": [
                {
                    "source_fragment_id": f["source_fragment_id"],
                    "text": f["source_quote"],
                    "citation_aliases": f["citation_aliases"],
                }
                for f in build_source_fragments(u)
            ],
        }
        for u in build_report_source_units(text)
    ]


def response(units):
    """Use the model's semantic-only wire contract with every required key."""
    return {
        "units": {
            u["source_unit_id"]: {
                f["source_fragment_id"]: {
                    "assertions": [{"kind": "scientific", "claim_text": f["text"]}],
                    "exclusion_reason": "",
                }
                for f in u["source_fragments"]
            }
            for u in units
        }
    }


@pytest.mark.parametrize(
    "text",
    [
        None,
        "# 报告\n\n## 方法\n\n甲采用图像分块 [E:e1]；乙采用蒸馏 [E:e2]。",
        "# Report\n\n## Results\n\n| Method | Result |\n|---|---|\n"
        "| Alpha | 80% [E:e1] |\n| Beta | 75% [E:e2] |",
    ],
)
def test_host_binds_every_fragment_in_source_order_and_its_exact_citations(text):
    units = inputs(text)
    value = response(units)
    # Transport order cannot change the canonical host inventory.
    value["units"] = dict(reversed(list(value["units"].items())))
    for uid, fragments in value["units"].items():
        value["units"][uid] = dict(reversed(list(fragments.items())))
    progress = {}
    inventory = _strict_inventory_response(json.dumps(value), units, progress)
    assert [u.source_unit_id for u in inventory.units] == [
        u["source_unit_id"] for u in units
    ]
    actual = [a for u in inventory.units for a in u.assertions]
    expected = [f for u in units for f in u["source_fragments"]]
    assert [a.source_fragment_id for a in actual] == [
        f["source_fragment_id"] for f in expected
    ]
    assert [a.evidence_aliases for a in actual] == [
        f["citation_aliases"] for f in expected
    ]
    assert len(progress["fragment_inventory"]) == len(expected)
    schema = _strict_inventory_schema(units)
    assert set(schema["properties"]["units"]["required"]) == {
        u["source_unit_id"] for u in units
    }
    assert set(schema["$defs"]["AtomicFragmentAssertion"]["properties"]) == {
        "kind",
        "claim_text",
    }


def test_shared_citations_cover_each_atomic_claim_and_uncited_science_stays_uncited():
    units = inputs()
    value = response(units)
    first = units[0]["source_fragments"][0]
    partition = value["units"][units[0]["source_unit_id"]][first["source_fragment_id"]]
    partition["assertions"] = [
        {"kind": "scientific", "claim_text": "Alpha uses patches"},
        {"kind": "scientific", "claim_text": "Alpha has a patch representation"},
    ]
    inventory = _strict_inventory_response(json.dumps(value), units, {})
    assert [a.evidence_aliases for a in inventory.units[0].assertions[:2]] == [
        ["e1", "e2"],
        ["e1", "e2"],
    ]
    assert inventory.units[-1].assertions[0].evidence_aliases == []


@pytest.mark.parametrize(
    "case",
    [
        "missing_unit",
        "extra_unit",
        "missing_fragment",
        "extra_fragment",
        "other_unit_fragment",
        "legacy_shape",
        "unit_array",
        "root_extra",
        "fragment_array",
        "invalid_kind",
        "empty_claim",
        "blank_claim",
        "unexplained_empty",
        "conflicting_exclusion",
        "citation_field",
        "identity_field",
        "position_field",
    ],
)
def test_missing_partitions_and_model_owned_metadata_fail_without_normalization(case):
    units = inputs()
    value = response(units)
    uid = units[0]["source_unit_id"]
    fid = units[0]["source_fragments"][0]["source_fragment_id"]
    fragment = value["units"][uid][fid]
    if case == "missing_unit":
        del value["units"][uid]
    elif case == "extra_unit":
        value["units"]["unknown-unit"] = {}
    elif case == "missing_fragment":
        del value["units"][uid][fid]
    elif case == "extra_fragment":
        value["units"][uid]["unknown-fragment"] = fragment
    elif case == "other_unit_fragment":
        wrong = units[-1]["source_fragments"][0]["source_fragment_id"]
        value["units"][uid][wrong] = value["units"][uid].pop(fid)
    elif case in {"legacy_shape", "unit_array"}:
        value["units"] = [{"source_unit_id": uid, "assertions": []}]
    elif case == "root_extra":
        value["source_quote"] = "invented"
    elif case == "fragment_array":
        value["units"][uid] = [fragment]
    elif case == "invalid_kind":
        fragment["assertions"][0]["kind"] = "exempt"
    elif case in {"empty_claim", "blank_claim"}:
        fragment["assertions"][0]["claim_text"] = "" if case == "empty_claim" else " "
    elif case == "unexplained_empty":
        fragment["assertions"] = []
    elif case == "conflicting_exclusion":
        fragment["exclusion_reason"] = "Ignore the science"
    else:
        field = {
            "citation_field": "evidence_aliases",
            "identity_field": "source_fragment_id",
            "position_field": "source_start",
        }[case]
        fragment["assertions"][0][field] = "invented"
    with pytest.raises(ValueError):
        _strict_inventory_response(json.dumps(value), units, {})


@pytest.mark.parametrize("duplicate", ["unit", "fragment", "semantic_field"])
def test_duplicate_json_keys_are_rejected_at_every_level(duplicate):
    units = inputs()
    value = response(units)
    if duplicate == "unit":
        raw = '{"units":' + json.dumps(value["units"]) + ',"units":{}}'
    elif duplicate == "fragment":
        uid = units[0]["source_unit_id"]
        fid = units[0]["source_fragments"][0]["source_fragment_id"]
        raw = '{"units":{' + json.dumps(uid) + ":{" + json.dumps(fid) + ":{},"
        raw += json.dumps(fid) + ":{}}}}"
    else:
        raw = json.dumps(value).replace(
            '"kind": "scientific"', '"kind":"scientific","kind":"report_structure"', 1
        )
    with pytest.raises(ValueError, match="evaluation_inventory_duplicate_key"):
        _strict_inventory_response(raw, units, {})


def test_partial_exclusion_is_retained_for_independent_global_review():
    units = inputs("# Report\n\nAlpha uses patches. [E:e1] Overview.")
    value = response(units)
    fragment = units[0]["source_fragments"][1]
    value["units"][units[0]["source_unit_id"]][fragment["source_fragment_id"]] = {
        "assertions": [],
        "exclusion_reason": "Controlled exclusion, independently checked later",
    }
    progress = {}
    inventory = _strict_inventory_response(json.dumps(value), units, progress)
    assert inventory.units[0].assertions and not inventory.units[0].exclusion_reason
    assert progress["fragment_inventory"][1]["assertion_count"] == 0
    assert progress["fragment_inventory"][1]["exclusion_reason"]


@pytest.mark.asyncio
async def test_valid_native_keys_do_not_approve_a_claim_borrowed_from_another_unit():
    class BorrowedClaim(BatchLLM):
        async def chat(self, messages, **kwargs):
            result = await super().chat(messages, **kwargs)
            payload = self.calls[-1]
            data = json.loads(result.content)
            if "allowed_citation_aliases" in payload:
                last = list(data["units"].values())[-1]
                next(iter(last.values()))["assertions"][0][
                    "claim_text"
                ] = "ViT uses image patches."
            elif "cited_materials" in payload:
                data["decisions"][-1]["source_faithful"] = False
            result.content = json.dumps(data)
            return result

    _, context = materials(2)
    context["evidence"]["a1"].update(
        text="DeiT uses distillation.", supporting_text="DeiT uses distillation."
    )
    context["fact_cards"] = build_fact_cards(context["evidence"])
    context["evidence_snapshot_sha256"] = digest(context["fact_cards"])
    survey = (
        "# Report\n\nViT uses image patches. [E:a0]\n\nDeiT uses distillation. [E:a1]"
    )
    llm = BorrowedClaim()
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_classification_or_source_invalid"
    ) as error:
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    details = error.value.details
    wrong = details["assertions"][-1]
    assert wrong["evidence_ids"] == ["a1"]
    assert "DeiT" in wrong["source_quote"] and wrong["status"] != "supported"
    assert details["coverage"]["complete"] is False
    assert details["coverage"]["model_processed_fragments"] == 2
    assert (
        llm.calls[-1]["fragment_inventory"] == details["stages"]["fragment_inventory"]
    )


@pytest.mark.asyncio
async def test_old_response_is_rejected_before_scientific_calls_or_retries():
    class OldResponse(BatchLLM):
        async def chat(self, messages, **kwargs):
            result = await super().chat(messages, **kwargs)
            result.content = '{"units":[{"source_unit_id":"old","assertions":[]}]}'
            return result

    survey, context = materials(1)
    llm = OldResponse()
    with pytest.raises(
        ScopedEvaluationError, match="evaluation_inventory_shape_invalid"
    ):
        await evaluate_scoped(
            llm=llm, survey=survey, context=context, config=EvalConfig()
        )
    assert len(llm.calls) == 1 and llm.batches == 0
