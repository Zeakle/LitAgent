"""Test opt-in shape re-emission without accepting repaired local JSON."""

import asyncio
import copy
import json

import httpx
import pytest
from model_config_helpers import load_previous_default_config as load_config
from test_native_responses import envelope, owner_for, session_for
from test_writer_v2 import BODY, Provider, case  # noqa: F401

from litagent.observability.context import reset_task_id, set_task_id
from litagent.quality import begin_targeted_repair, load_targeted_repairs
from litagent.writing_v2 import (
    SHAPE_RECOVERY_CONTRACT,
    WriterError,
    WriterSession,
    array_reemission_target,
    body_section_schema,
)
from scripts import run_frozen_writing as campaign


def test_profile_and_identity_are_explicit_without_mutating_legacy(case, tmp_path):
    """A server-only behavior switch cannot resume or change a prior session."""
    config = load_config()
    before = config.model_dump()
    changed = campaign.profile_config(config, campaign.SHAPE_PROFILE)
    campaign.check_profile(changed, campaign.SHAPE_PROFILE)
    assert config.model_dump() == before
    assert campaign.profile_parameters(campaign.SHAPE_PROFILE) == {
        "native_provider_api": "responses",
        "native_reasoning_effort": "low",
        "native_shape_recovery": True,
    }
    assert "native_shape_recovery" not in campaign.profile_parameters(
        campaign.RESPONSES_PROFILE
    )
    assert "shape_recovery_contract" not in campaign.profile_description(
        campaign.RESPONSES_PROFILE, {}
    )
    assert (
        campaign.profile_description(campaign.SHAPE_PROFILE, {})[
            "shape_recovery_contract"
        ]
        == SHAPE_RECOVERY_CONTRACT
    )
    provider = Provider(case)
    path = tmp_path / "old.json"
    old = WriterSession(provider, path, "r", case.packet, case.aliases)
    explicit = WriterSession(
        provider, path, "r", case.packet, case.aliases, shape_recovery=False
    )
    assert old.identity == explicit.identity
    old.save()
    with pytest.raises(WriterError, match="identity_conflict"):
        WriterSession(
            provider, path, "r", case.packet, case.aliases, shape_recovery=True
        )


@pytest.mark.parametrize(
    "fault", ["json", "duplicate", "missing", "unknown", "empty", "text_type", "valid"]
)
def test_only_complete_valid_items_can_authorize_array_reemission(case, fault):
    """The comparison target never invents fields, sources or semantic content."""
    value = {"taxonomy": copy.deepcopy(case.body["taxonomy"][0])}
    if fault == "missing":
        value["taxonomy"].pop("sentences")
    elif fault == "unknown":
        value["taxonomy"]["sentences"][0]["fact_card_refs"] = ["c999"]
    elif fault == "empty":
        value["taxonomy"]["sentences"] = []
    elif fault == "text_type":
        value["taxonomy"]["sentences"][0]["text"] = ["not a string"]
    elif fault == "valid":
        value = {"taxonomy": case.body["taxonomy"]}
    raw = json.dumps(value)
    if fault == "json":
        raw = raw[:-1]
    elif fault == "duplicate":
        raw = '{"taxonomy":{},' + raw[1:]
    with pytest.raises(WriterError):
        array_reemission_target(raw, body_section_schema("taxonomy", case.aliases))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault", [None, "text", "ref", "extra", "repeat", "json", "cancel", "unknown_usage"]
)
async def test_reemission_has_separate_receipt_shared_count_and_no_replay(
    case, tmp_path, fault
):
    """Only a new valid, verbatim model response may enter the chapter compiler."""
    sent = []

    def handler(request):
        wire = json.loads(request.content)
        data = json.loads(wire["input"][1]["content"])
        sent.append(data)
        for key, value in case.packet.items():
            assert data[key] == value
        section = data["requested_body_section"]
        result = {
            k: copy.deepcopy(case.body[k])
            for k in wire["text"]["format"]["schema"]["properties"]
        }
        correction = "shape_reemission" in data
        if section == "taxonomy" and not correction:
            result[section] = result[section][0]
        if correction:
            assert len(sent) == 3
            assert wire["max_output_tokens"] == 20000
            assert data["shape_reemission"]["paths"] == [["taxonomy"]]
            assert data["prior_chapters_unverified"] == {
                "title": case.body["title"],
                "introduction": case.body["introduction"],
            }
            assert data["rejected_response"]["taxonomy"] == case.body["taxonomy"][0]
            if fault == "cancel":
                raise asyncio.CancelledError()
            if fault == "text":
                result[section][0]["sentences"][0]["text"] += " Unauthorized rewrite."
            elif fault == "ref":
                result[section][0]["sentences"][0]["fact_card_refs"] = [
                    case.by_eid["a"]
                ]
            elif fault == "extra":
                result["methods"] = case.body["methods"]
            elif fault == "repeat":
                result[section] = result[section][0]
        raw = "{" if correction and fault == "json" else json.dumps(result)
        response = envelope(raw)
        if correction and fault == "unknown_usage":
            response.pop("usage")
        return httpx.Response(200, json=response)

    owner, budget, _ = await owner_for(handler)
    session = session_for(owner, case, tmp_path, shape_recovery=True)
    token = set_task_id("synthesis")
    try:
        if fault is None:
            assert await session.initial_body() == case.body
            assert len(sent) == 6
            assert "failure" not in session.journal
            assert (
                session.journal["body_parts"]["taxonomy"]["shape_repair_call"]
                == "repair_0"
            )
        else:
            with pytest.raises((ValueError, asyncio.CancelledError)):
                await session.initial_body()
            assert len(sent) == 3
            assert list(session.journal["body_parts"]) == ["introduction"]
        assert len(session.state["attempts"]) == 1
        assert session.state["attempts"][0]["stage"] == "native_initial"
        assert session.state["attempts"][0]["kind"] == SHAPE_RECOVERY_CONTRACT
        assert session.journal["calls"]["body:taxonomy"]["status"] == "failed_no_replay"
        assert session.journal["calls"]["body:taxonomy"]["raw"] == json.dumps(
            {"taxonomy": case.body["taxonomy"][0]}
        )
        if fault not in {"cancel", "unknown_usage"}:
            assert budget.used == 120 * len(sent)
            assert not budget.snapshot()["usage_uncertain"]
            assert budget.snapshot()["phase_usage"] == [0, budget.used, 0]
        else:
            assert budget.snapshot()["usage_uncertain"]
            assert budget.used > 240
        assert budget.snapshot()["requests_sent"] == len(sent)
        assert "summary_initial" not in session.journal["calls"]
        before = session.journal_path.read_bytes()
        with pytest.raises(WriterError, match="ineligible"):
            await session.reemit_array_shape(
                "taxonomy",
                sent[1],
                body_section_schema("taxonomy", case.aliases),
                "body:taxonomy",
            )
        assert session.journal_path.read_bytes() == before
        restarted = session_for(owner, case, tmp_path, shape_recovery=True)
        with pytest.raises(WriterError, match="no_replay"):
            await restarted.initial_body()
        assert session.journal_path.read_bytes() == before
    finally:
        reset_task_id(token)
        await owner.close()


@pytest.mark.asyncio
async def test_two_shapes_exhaust_the_same_semantic_repair_ledger(case, tmp_path):
    """Neither subsequent chapters nor evaluation may gain a third attempt."""
    sent = []

    def handler(request):
        wire = json.loads(request.content)
        data = json.loads(wire["input"][1]["content"])
        sent.append(data)
        section = data["requested_body_section"]
        result = {
            k: copy.deepcopy(case.body[k])
            for k in wire["text"]["format"]["schema"]["properties"]
        }
        if section in BODY[1:4] and "shape_reemission" not in data:
            result[section] = result[section][0]
        return httpx.Response(200, json=envelope(json.dumps(result)))

    owner, _, _ = await owner_for(handler)
    session = session_for(owner, case, tmp_path, shape_recovery=True)
    try:
        with pytest.raises(WriterError, match="writer_schema_invalid") as error:
            await session.initial_body()
        assert error.value.family == "protocol"
        assert error.value.diagnostics[0]["reason"] == "writer_repair_limit"
        state = load_targeted_repairs(session.path, "r")
        assert len(state["attempts"]) == 2
        assert all(a["status"] == "completed" for a in state["attempts"])
        assert len(sent) == 6 and "repair_2" not in session.journal["calls"]
        with pytest.raises(ValueError, match="limit_exceeded"):
            begin_targeted_repair(state, "evaluation", "a" * 64)
        targets = {
            "u1": {
                "path": ["introduction", 0, 0],
                "original": copy.deepcopy(case.body["introduction"][0]["sentences"][0]),
            }
        }
        with pytest.raises(WriterError, match="writer_repair_limit") as patch_error:
            await session.patch(case.body, targets, cause="Scientific repair")
        assert patch_error.value.family == "quality"
        assert len(sent) == 6 and len(session.state["attempts"]) == 2
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("near_request_limit", [False, True])
async def test_failed_reemission_preflight_spends_no_attempt_or_extra_request(
    case, tmp_path, near_request_limit
):
    """Downstream reserves and the shared request ceiling survive protocol recovery."""
    sent = []

    def handler(request):
        wire = json.loads(request.content)
        sent.append(wire)
        data = json.loads(wire["input"][1]["content"])
        assert "shape_reemission" not in data
        section = data["requested_body_section"]
        result = {
            k: copy.deepcopy(case.body[k])
            for k in wire["text"]["format"]["schema"]["properties"]
        }
        response = envelope(json.dumps(result))
        if section == "taxonomy":
            result[section] = result[section][0]
            response = envelope(json.dumps(result))
            if not near_request_limit:
                response["usage"].update(input_tokens=19000, total_tokens=19100)
        return httpx.Response(200, json=response)

    owner, budget, _ = await owner_for(
        handler, allowance=500000 if near_request_limit else 50000
    )
    if near_request_limit:
        for _ in range(55):
            budget.begin_request()
    session = session_for(owner, case, tmp_path, shape_recovery=True)
    token = set_task_id("synthesis")
    try:
        with pytest.raises(ValueError, match="budget"):
            await session.initial_body()
        assert len(sent) == 2
        assert not session.state["attempts"]
        assert "repair_0" not in session.journal["calls"]
        assert session.journal["failure"]["request_sent"] is False
        assert budget.snapshot()["requests_sent"] == (57 if near_request_limit else 2)
        assert not budget.snapshot()["usage_uncertain"]
    finally:
        reset_task_id(token)
        await owner.close()


def test_failed_shared_attempt_accounting_survives_without_a_report(
    tmp_path, monkeypatch
):
    """Failure before report assembly must not make a consumed attempt disappear."""
    from litagent.quality import save_targeted_repairs, targeted_repair_state

    monkeypatch.chdir(tmp_path)
    path = tmp_path / "artifacts/targeted-repairs/r.json"
    state = targeted_repair_state(None, "r")
    attempt = begin_targeted_repair(state, "native_initial", "a" * 64)
    attempt.update(
        status="failed",
        kind=SHAPE_RECOVERY_CONTRACT,
        error_code="writer_schema_invalid",
    )
    save_targeted_repairs(path, state)
    assert campaign.repair_accounting("r") == {
        "repairs": 1,
        "repair_history": [
            {
                "index": 0,
                "stage": "native_initial",
                "status": "failed",
                "kind": SHAPE_RECOVERY_CONTRACT,
                "error_code": "writer_schema_invalid",
            }
        ],
    }
    assert campaign.repair_accounting("absent") is None
    with pytest.raises(ValueError, match="identity"):
        campaign.repair_accounting("../r")
