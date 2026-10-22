"""Bind reference repairs to rejected rows without accepting scientific correctness."""

import asyncio
import copy
import hashlib
import json

import httpx
import pytest
from jsonschema import ValidationError
from openai import AsyncOpenAI
from test_staged_body_diagnostics import repair_inputs
from test_staged_native_generation import provider_body, provider_summaries

from litagent import staged_generation as staged
from litagent.llm.client import OpenAICompatibleClient
from litagent.native_wire import array_from_native
from litagent.observability.context import reset_task_id, set_task_id
from litagent.quality import (
    _writing_scopes,
    begin_targeted_repair,
    save_targeted_repairs,
    targeted_repair_state,
)
from litagent.safety.budget import CostBudget
from litagent.workbench import RunOptions


@pytest.fixture
def reference_case(repair_inputs):
    """Reproduce two references missing from a narrower methods menu."""
    x = copy.deepcopy(repair_inputs)
    sentences = x["body"]["sections"]["methods"]["blocks"][0]["sentences"]
    sentences[0]["text"] = "On DeiT-S, EViT improves speed by 50%."
    sentences.append(
        {
            **copy.deepcopy(sentences[0]),
            "text": "On DeiT-B, EViT improves speed by 20%.",
        }
    )
    args = dict(
        cards=x["cards"],
        structure="academic_v2",
        language="en",
        selection=x["selected"],
    )
    prepared, rf = staged.prepare_body_request(x["messages"], x["rf"])
    binding = staged.card_reference_binding(
        x["cards"], _writing_scopes(x["cards"], "academic_v2", "en", x["selected"])
    )
    wire = provider_body(x["body"], x["cards"], selection=x["selected"])
    staged.validate_body(json.dumps(wire), rf, **args)
    valid = wire["methods"][0]["fact_card_refs"][0]
    other = next(
        ref
        for ref, cid in binding["reading_cards"].items()
        if cid not in binding["section_cards"]["methods"].values()
    )
    invalid = "m" + other[1:]
    assert invalid not in binding["section_cards"]["methods"]
    for row in wire["methods"][:2]:
        row["fact_card_refs"].append(invalid)
    raw = json.dumps(wire, ensure_ascii=False)
    with pytest.raises(staged.BodyReferenceError) as e:
        staged.validate_body(raw, rf, collect_diagnostics=True, **args)
    failure = e.value
    response = {
        "can_repair": True,
        "patches": {slot: [valid] for slot in failure.targets},
    }
    return {
        **x,
        "args": args,
        "body_rf": rf,
        "prepared_body": prepared,
        "binding": binding,
        "wire": wire,
        "raw": raw,
        "failure": failure,
        "valid_ref": valid,
        "invalid_ref": invalid,
        "reference_response": response,
    }


def test_diagnostics_keep_every_original_path_and_complete_material(reference_case):
    x = reference_case
    f = x["failure"]
    assert [d["path"] for d in f.diagnostics] == [
        ["methods", 0, "fact_card_refs", 1],
        ["methods", 1, "fact_card_refs", 1],
    ]
    assert f.raw_response_sha256 == hashlib.sha256(x["raw"].encode()).hexdigest()
    prepared, rf = staged.prepare_body_reference_repair(x["prepared_body"], f)
    packet = json.loads(prepared[1]["content"])
    original = json.loads(x["prepared_body"][1]["content"])
    for field in (
        "sources",
        "cards",
        "research_question",
        "section_source_index",
        "analysis_plan",
        "coverage_requirements",
    ):
        assert packet[field] == original[field]
    assert packet["rejected_body"] == x["wire"]
    assert (
        packet["reference_repair"]["targets"]["r0"]["original"]
        == x["wire"]["methods"][0]
    )
    assert rf["json_schema"]["name"] == "body_reference_repair_v1"
    assert "Scientific text" in prepared[0]["content"]


def test_patch_preserves_all_text_layout_notices_and_unaffected_references(
    reference_case,
):
    x = reference_case
    before = copy.deepcopy(x["wire"])
    raw, proof = staged.apply_body_reference_repair(
        x["raw"],
        json.dumps(x["reference_response"]),
        x["failure"],
        x["body_rf"],
        x["binding"],
    )
    repaired = json.loads(raw)
    for i in range(2):
        before["methods"][i]["fact_card_refs"] = [x["valid_ref"]]
    assert repaired == before
    body, _ = staged.validate_body(raw, x["body_rf"], **x["args"])
    assert body == x["body"]
    assert proof["scientific_verdict"] == "not_evaluated"
    assert proof["changed_paths"] == [
        ["methods", 0, "fact_card_refs"],
        ["methods", 1, "fact_card_refs"],
    ]
    assert x["wire"]["methods"][0]["fact_card_refs"][-1] == x["invalid_ref"]


@pytest.mark.parametrize(
    "fault",
    [
        "extra_target",
        "missing_target",
        "text",
        "empty",
        "duplicate",
        "unknown",
        "reading_key",
        "other_section",
        "null",
        "false",
        "string_bool",
        "malformed",
        "duplicate_json",
    ],
)
def test_bad_or_unavailable_patches_never_change_rejected_bytes(reference_case, fault):
    x = reference_case
    r = copy.deepcopy(x["reference_response"])
    original = x["raw"]
    if fault == "extra_target":
        r["patches"]["other"] = [x["valid_ref"]]
    elif fault == "missing_target":
        r["patches"].pop("r1")
    elif fault == "text":
        r["text"] = "weakened claim"
    elif fault == "empty":
        r["patches"]["r0"] = []
    elif fault == "duplicate":
        r["patches"]["r0"] = [x["valid_ref"]] * 2
    elif fault in {"unknown", "reading_key", "other_section"}:
        r["patches"]["r0"] = [
            {"unknown": x["invalid_ref"], "reading_key": "c1", "other_section": "o1"}[
                fault
            ]
        ]
    elif fault == "null":
        r["patches"]["r0"] = None
    elif fault == "false":
        r["can_repair"] = False
        r["patches"] = {k: None for k in r["patches"]}
    elif fault == "string_bool":
        r["can_repair"] = "true"
    raw = json.dumps(r)
    if fault == "malformed":
        raw = "{"
    if fault == "duplicate_json":
        raw = raw.replace('"can_repair": true', '"can_repair":false,"can_repair":true')
    with pytest.raises((ValueError, ValidationError)):
        staged.apply_body_reference_repair(
            original, raw, x["failure"], x["body_rf"], x["binding"]
        )
    assert x["raw"] == original


@pytest.mark.parametrize("fault", ["raw", "schema", "binding", "targets", "diagnostic"])
def test_stale_or_tampered_identities_fail_closed(reference_case, fault):
    x = reference_case
    raw = x["raw"]
    rf = copy.deepcopy(x["body_rf"])
    binding = copy.deepcopy(x["binding"])
    failure = x["failure"]
    if fault == "raw":
        raw += " "
    elif fault == "schema":
        rf["json_schema"]["name"] = "other"
    elif fault == "binding":
        binding["section_cards"]["methods"][x["invalid_ref"]] = "invented"
    elif fault == "targets":
        failure.targets["r0"]["allowed_references"].append(x["invalid_ref"])
    else:
        failure.diagnostics[0]["path"][1] = 5
    with pytest.raises(ValueError, match="binding_invalid"):
        staged.apply_body_reference_repair(
            raw, json.dumps(x["reference_response"]), failure, rf, binding
        )


@pytest.mark.parametrize(
    "fault",
    [
        "extra_field",
        "bad_type",
        "block_skip",
        "duplicate_json",
        "schema_widening",
        "legacy_codec",
    ],
)
def test_only_narrow_v11_enum_errors_are_repairable(reference_case, fault):
    x = reference_case
    wire = copy.deepcopy(x["wire"])
    rf = copy.deepcopy(x["body_rf"])
    raw = x["raw"]
    expected = (ValueError, ValidationError)
    if fault == "extra_field":
        wire["methods"][0]["untrusted"] = "ignored?"
    elif fault == "bad_type":
        wire["methods"][0]["text"] = 42
    elif fault == "block_skip":
        wire["methods"][0]["block_index"] = 1
    elif fault == "schema_widening":
        rf["json_schema"]["schema"]["properties"]["methods"]["items"]["properties"][
            "fact_card_refs"
        ]["items"]["enum"].append(x["invalid_ref"])
    elif fault == "legacy_codec":
        rf["json_schema"]["name"] = "grounded_body_v4_typed"
    raw = json.dumps(wire)
    if fault == "duplicate_json":
        raw = raw.replace('"title":', '"title":"duplicate","title":', 1)
    with pytest.raises(expected) as e:
        staged.validate_body(raw, rf, collect_diagnostics=True, **x["args"])
    assert not isinstance(e.value, staged.BodyReferenceError)


def test_default_and_v10_remain_strict_without_schema_repair(
    reference_case, monkeypatch
):
    x = reference_case
    with pytest.raises(ValidationError):
        staged.validate_body(x["raw"], x["body_rf"], **x["args"])
    current = RunOptions(
        quality_policy="strict_auto_v2",
        body_quality="analysis_v1",
        generation_strategy="body_summary_v1",
    ).fingerprint()
    single = RunOptions().fingerprint()
    monkeypatch.setattr(staged, "CONTRACT", "native-body-summary-v10")
    assert RunOptions().fingerprint() == single
    assert (
        RunOptions(
            quality_policy="strict_auto_v2",
            body_quality="analysis_v1",
            generation_strategy="body_summary_v1",
        ).fingerprint()
        != current
    )
    with pytest.raises(ValidationError):
        staged.validate_body(
            x["raw"], x["body_rf"], collect_diagnostics=True, **x["args"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "budget",
        "exhausted",
        "cancel",
        "unknown",
        "refusal",
        "invalid_response",
        "scientific",
        "prior_attempt",
    ],
)
async def test_real_sdk_generation_repair_summary_and_failure_boundaries(
    reference_case, tmp_path, fault
):
    x = reference_case
    budget = CostBudget()
    budget.configure_strict(halt_on_unknown_usage=True)
    state = targeted_repair_state(None, "reference-offline")
    path = tmp_path / "repair.json"
    if fault in {"exhausted", "prior_attempt"}:
        for _ in range(2 if fault == "exhausted" else 1):
            e = begin_targeted_repair(state, "native_initial", "a" * 64)
            e["status"] = "failed"
    if fault == "scientific":
        x["wire"]["methods"][0][
            "text"
        ] = "On DeiT-S, EViT improves speed by 50%; on DeiT-B, EViT improves speed by 20%."
    save_targeted_repairs(path, state)
    calls = []

    async def transport(req):
        packet = json.loads(req.content)
        calls.append(packet)
        name = packet["response_format"]["json_schema"]["name"]
        if name == "grounded_body_v5_blocks":
            content = x["wire"]
            if fault == "budget":
                budget._used_tokens = 300000
                budget._phase_usage[1] = 300000
        elif name == "body_reference_repair_v1":
            if fault == "cancel":
                raise asyncio.CancelledError()
            content = x["reference_response"]
            if fault == "refusal":
                content = {
                    "can_repair": False,
                    "patches": {k: None for k in content["patches"]},
                }
            if fault == "invalid_response":
                content = {
                    "can_repair": True,
                    "patches": {k: [x["invalid_ref"]] for k in content["patches"]},
                }
        elif name == "native_body_repair_v7_refs":
            assert fault == "scientific"
            content = {"can_repair": False, "patches": {}}
        else:
            assert "summaries" in name
            content = provider_summaries(x["summaries"], packet)
        result = {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "offline",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": json.dumps(content)},
                    "finish_reason": "stop",
                }
            ],
        }
        if not (fault == "unknown" and name == "body_reference_repair_v1"):
            result["usage"] = {
                "prompt_tokens": 10,
                "completion_tokens": 10,
                "total_tokens": 20,
            }
        return httpx.Response(200, json=result)

    client = OpenAICompatibleClient(
        "https://offline.invalid/v1",
        "offline",
        api_key="offline",
        cost_budget=budget,
        transport_max_attempts=1,
    )
    await client._client.close()
    client._client = AsyncOpenAI(
        api_key="offline",
        base_url="https://offline.invalid/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
    )
    token = set_task_id("synthesis.reference-offline")
    args = dict(
        cards=x["cards"],
        structure="academic_v2",
        language="en",
        selection=x["selected"],
        checkpoint_path=path,
        repair_state=state,
    )
    try:
        if fault not in {None, "prior_attempt"}:
            with pytest.raises((ValueError, ValidationError, asyncio.CancelledError)):
                await staged.generate_staged_report(
                    client, x["messages"], x["rf"], **args
                )
        else:
            candidate, proof = await staged.generate_staged_report(
                client, x["messages"], x["rf"], **args
            )
            complete = array_from_native(json.loads(candidate))
            assert {
                "title": complete["title"],
                "sections": {k: complete["sections"][k] for k in staged.BODY},
            } == x["body"]
            assert proof["scientific_verdict"] == "not_evaluated"
        names = [c["response_format"]["json_schema"]["name"] for c in calls]
        expected = (
            1
            if fault in {"budget", "exhausted"}
            else 3 if fault in {None, "prior_attempt", "scientific"} else 2
        )
        assert len(calls) == expected
        assert budget.snapshot()["requests_sent"] == expected
        if fault in {"cancel", "unknown"}:
            assert budget.snapshot()["usage_uncertain"] and budget.used > 20
        if fault == "budget":
            assert not state["attempts"]
        elif fault in {"exhausted", "prior_attempt", "scientific"}:
            assert len(state["attempts"]) == 2
        else:
            assert len(state["attempts"]) == 1
        journal = json.loads(next(tmp_path.glob("*.generation-*.json")).read_text())
        assert journal["steps"]["body"]["raw_response"] == json.dumps(x["wire"])
        assert len(journal["body_reference_failure"]["diagnostics"]) == 2
        count = len(calls)
        with pytest.raises(ValueError, match="restart_no_replay"):
            await staged.generate_staged_report(client, x["messages"], x["rf"], **args)
        assert len(calls) == count
    finally:
        reset_task_id(token)
        await client.close()
