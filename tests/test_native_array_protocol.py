"""Verify array transport projection and one-shot SDK accounting without science claims."""

import asyncio
import copy
import hashlib
import json
import shutil
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator, ValidationError
from openai import AsyncOpenAI
from test_body_analysis import analysis_document
from test_writing_passages import facts

from litagent.llm import client as client_module
from litagent.quality import grounded_messages, prepare_grounded_request
from litagent.report_audit import atomic_json
from scripts import probe_native_array_wire as probe


def native_schema(cards=None):
    """Exercise the real current schema producer and all chapter source permissions."""
    _, rf = prepare_grounded_request(
        grounded_messages(
            facts() if cards is None else cards,
            "academic_v2",
            "en",
            "EViT conditions",
            use_full_sources=True,
            use_scope_notices=True,
            analysis_mode=True,
            require_unit_citation_scope=True,
        )
    )
    return rf["json_schema"]["schema"]


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Bind a completed local parent with real source schema and public accounting."""
    monkeypatch.chdir(tmp_path)
    parent = tmp_path / "parent"
    parent.mkdir()
    archive = tmp_path / "artifacts/runs/parent-pilot-few-shot-metric.json"
    archive.parent.mkdir(parents=True)
    archive.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "event": "llm.start",
                        "data": {
                            "response_format": {
                                "json_schema": {"schema": native_schema()}
                            }
                        },
                    },
                    {
                        "event": "llm.complete",
                        "data": {"total_tokens": 100, "usage_available": True},
                    },
                ]
            }
        )
    )
    history = {
        "initial_generation_rounds": 28,
        "model_requests": 526,
        "model_known_tokens": 2812458,
        "model_unknown_conservative_tokens": 1003467,
        "jev_requests": 67,
        "jev_conservative_input_tokens": 756769,
        "judge_requests": 0,
        "jev_usage_complete": False,
    }
    atomic_json(
        parent / "acceptance-summary.json",
        {
            "suite_id": "parent",
            "status": "stopped",
            "new_unknown_usage": False,
            "source_archive_sha256": probe.sha(archive),
            "new_requests_started": 1,
            "new_known_tokens": 100,
            "cumulative": history,
        },
    )
    root = tmp_path / "protocol"
    manifest = probe.prepare_protocol(root, parent)
    return root, manifest


def test_array_roundtrip_retains_all_values_indices_and_original_constraints():
    """Keep exact findings, negation, values, ordering and summary dependencies."""
    cards = facts("EViT gains 50% on DeiT-S, but not 90% under condition B.")
    original = analysis_document(cards)
    block = original["sections"]["methods"]["first_block"]
    block["additional_sentences"] = [
        {
            **block["first_sentence"],
            "text": "Second condition is not the first. 低样本设定，50%。",
        }
    ]
    original["sections"]["methods"]["additional_blocks"] = [copy.deepcopy(block)]
    original["sections"]["abstract"]["first_block"]["first_sentence"][
        "body_references"
    ] = [{"section_key": "methods", "block_index": 1, "sentence_index": 1}]
    wire = probe.array_from_native(original)
    decoded, proof = probe.canonical_from_array(json.dumps(wire, ensure_ascii=False))
    assert decoded == original and probe.array_from_native(decoded) == wire
    assert proof["scientific_verdict"] == "not_evaluated"
    assert len(proof["positions"]) == 10
    assert proof["positions"][-1]["unit_sha256"]
    schema = native_schema(cards)
    array_schema = probe.array_wire_schema(schema)
    Draft202012Validator(array_schema).validate(wire)
    Draft202012Validator(schema).validate(decoded)
    assert set(array_schema["$defs"]["b0"]["properties"]) == {
        "heading",
        "type",
        "sentences",
    }
    assert schema["$defs"]["b0"]["properties"]["first_sentence"]


@pytest.mark.parametrize(
    "fault",
    [
        "extra",
        "mixed",
        "empty_blocks",
        "empty_sentences",
        "missing_section",
        "unit_type",
        "body_field",
        "reference",
        "bool_index",
    ],
)
def test_invalid_array_shape_never_discards_or_guesses_fields(fault):
    """Reject malformed structures without discarding unknown or protected fields."""
    document = probe.protocol_document(probe.array_wire_schema(native_schema()))
    if fault == "extra":
        document["private_extra"] = "not ignored"
    elif fault == "mixed":
        document["sections"]["methods"]["first_block"] = {}
    elif fault == "empty_blocks":
        document["sections"]["methods"]["blocks"] = []
    elif fault == "empty_sentences":
        document["sections"]["methods"]["blocks"][0]["sentences"] = []
    elif fault == "missing_section":
        document["sections"].pop("abstract")
    elif fault == "unit_type":
        document["sections"]["methods"]["blocks"][0]["sentences"][0]["kind"] = {}
    elif fault == "body_field":
        document["sections"]["methods"]["blocks"][0]["sentences"][0]["invented"] = "x"
    else:
        document["sections"]["abstract"]["blocks"][0]["sentences"][0][
            "body_references"
        ][0]["sentence_index"] = (True if fault == "bool_index" else 999)
    before = copy.deepcopy(document)
    with pytest.raises(ValueError):
        probe.canonical_from_array(document)
    assert document == before


def test_original_first_scientific_rule_remains_mandatory_after_wire_projection():
    """Keep local first-position constraints and duplicate-key rejection."""
    schema = native_schema()
    wire_schema = probe.array_wire_schema(schema)
    doc = probe.protocol_document(wire_schema)
    unit = doc["sections"]["open_problems"]["blocks"][0]["sentences"][0]
    unit.update(kind="gap", text="", fact_card_ids=[], scope_notice_id="evidence_limit")
    Draft202012Validator(wire_schema).validate(doc)
    native, _ = probe.canonical_from_array(doc)
    with pytest.raises(ValidationError):
        Draft202012Validator(schema).validate(native)
    encoded = json.dumps(doc).replace(
        '"title": "PROTOCOL_PROBE"', '"title":"one","title":"two"'
    )
    with pytest.raises(ValueError, match="duplicate"):
        probe.canonical_from_array(encoded)


def test_actual_truncated_response_is_not_accepted_by_adding_a_brace():
    """Preserve the real protocol failure and reject heuristic completion."""
    fixture = json.loads(
        Path("tests/fixtures/native-array-json-failure-real.json").read_text()
    )
    text = fixture["response"]
    assert hashlib.sha256(text.encode()).hexdigest() == fixture["response_sha256"]
    assert (
        fixture["completion_tokens"] < fixture["max_tokens"]
        and fixture["finish_reason"] == "stop"
    )
    with pytest.raises(json.JSONDecodeError):
        probe.canonical_from_array(text)
    synthetic = json.loads(text + "}")
    with pytest.raises(ValidationError):
        Draft202012Validator(fixture["schema"]).validate(synthetic)
    with pytest.raises(ValueError):
        probe.canonical_from_array(synthetic)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [None, "json", "schema", "model", "usage", "cancel", "timeout", "length", "text"],
)
async def test_real_sdk_one_request_budget_and_no_replay(prepared, monkeypatch, fault):
    """Exercise actual SDK, accounting, locks and recovery with only HTTP mocked."""
    root, m = prepared
    monkeypatch.setenv("LLM_API_KEY", "offline-secret")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    calls = []
    entered, release = asyncio.Event(), asyncio.Event()

    async def transport(request):
        """Return a controlled response or transport fault at the actual SDK boundary."""
        body = json.loads(request.content)
        calls.append(body)
        assert (
            body["model"] == probe.MODEL
            and body["max_tokens"] == 20000
            and body["thinking"] == {"type": "disabled"}
        )
        assert (
            body["response_format"] == m["frame"]["response_format"]
            and body["response_format"]["json_schema"]["strict"]
        )
        assert "EViT" not in json.dumps(body["messages"])
        started = json.loads((root / "probe-state.json").read_text())
        assert (
            started["status"] == "started_no_replay"
            and started["unknown_hold"] == 50000
        )
        if fault is None:
            entered.set()
            await release.wait()
        if fault == "cancel":
            raise asyncio.CancelledError()
        if fault == "timeout":
            raise httpx.ReadTimeout("offline")
        doc = probe.protocol_document(body["response_format"]["json_schema"]["schema"])
        if fault == "schema":
            doc["sections"].pop("methods")
        if fault == "text":
            doc["title"] = "Not the fixed protocol fixture"
        response = {
            "id": "offline",
            "object": "chat.completion",
            "created": 0,
            "model": "wrong-model" if fault == "model" else probe.MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": (
                            "invalid JSON" if fault == "json" else json.dumps(doc)
                        ),
                    },
                    "finish_reason": "length" if fault == "length" else "stop",
                }
            ],
        }
        if fault != "usage":
            response["usage"] = {
                "prompt_tokens": 30,
                "completion_tokens": 20,
                "total_tokens": 50,
            }
        return httpx.Response(200, json=response)

    monkeypatch.setattr(
        client_module,
        "AsyncOpenAI",
        lambda **kwargs: AsyncOpenAI(
            **kwargs,
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(transport)),
        ),
    )
    assert (await probe.run_protocol(root))["paid_requests"] == 0
    if fault is None:
        pending = asyncio.create_task(probe.run_protocol(root, live=True))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            with pytest.raises(OSError):
                await probe.run_protocol(root, live=True)
        finally:
            release.set()
        result = await pending
    else:
        result = await probe.run_protocol(root, live=True)
    assert len(calls) == 1 and result["budget"]["requests_sent"] == 1
    assert result["status"] == (
        "completed"
        if fault is None
        else (
            "interrupted_no_replay"
            if fault == "cancel"
            else (
                "failed_no_replay"
                if fault in {"usage", "timeout"}
                else "protocol_failed"
            )
        )
    )
    receipt = json.loads((root / "acceptance-summary.json").read_text())
    assert (
        receipt["new_survey_rounds"] == 0
        and receipt["cumulative"]["initial_generation_rounds"] == 28
    )
    assert (
        receipt["cumulative"]["model_requests"] == 527
        and receipt["family_requests"] == 2
    )
    assert receipt["new_unknown_hold"] == (
        50000 if fault in {"usage", "cancel", "timeout"} else 0
    )
    assert receipt["new_known_tokens"] == (
        0 if fault in {"usage", "cancel", "timeout"} else 50
    )
    before = (root / "acceptance-summary.json").read_bytes()
    assert await probe.run_protocol(root, live=True) == result
    assert len(calls) == 1 and before == (root / "acceptance-summary.json").read_bytes()
    assert "offline-secret" not in "".join(p.read_text() for p in root.glob("*.json"))
    if fault is None:
        received = copy.deepcopy(result)
        received["status"] = "response_received"
        atomic_json(root / "probe-state.json", received)
        assert (await probe.run_protocol(root, live=True))["status"] == "completed"
        assert (
            len(calls) == 1
            and (root / "acceptance-summary.json").read_bytes() == before
        )


@pytest.mark.asyncio
async def test_restart_with_unknown_response_never_constructs_transport(
    prepared, monkeypatch
):
    """Keep the whole unknown grant and refuse transport reconstruction."""
    root, m = prepared
    atomic_json(
        root / "probe-state.json",
        {"status": "started_no_replay", "binding": m["binding"]},
    )

    def forbidden(*args, **kwargs):
        """Fail if recovery attempts to construct a model transport."""
        raise AssertionError("transport constructed")

    monkeypatch.setattr(probe, "OpenAICompatibleClient", forbidden)
    state = await probe.run_protocol(root, live=True)
    assert state["status"] == "interrupted_no_replay" and state["unknown_hold"] == 50000
    assert (
        json.loads((root / "acceptance-summary.json").read_text())["cumulative"][
            "model_unknown_conservative_tokens"
        ]
        == 1053467
    )


def test_identity_and_budget_rejected_before_transport(prepared):
    """Reject changed frozen parent inputs before any paid attempt."""
    root, m = prepared
    assert m["frame"]["reservation"] <= 50000 and m["frame"]["max_tokens"] == 20000
    (Path(m["parent"])).write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        probe.validate_manifest(root)
    assert not (root / "probe-state.json").exists()


def test_copying_a_prepared_identity_cannot_create_another_paid_attempt(prepared):
    """Bind the request identity to its sole output directory."""
    root, _ = prepared
    clone = root.parent / "other-protocol"
    shutil.copytree(root, clone)
    with pytest.raises(ValueError, match="identity"):
        probe.validate_manifest(clone)
    assert not (clone / "probe-state.json").exists()


def test_frozen_script_snapshot_change_is_rejected(prepared):
    """Detect changed executable snapshots before dispatch."""
    root, _ = prepared
    path = root / "probe.py"
    path.write_text(path.read_text() + "\n# changed snapshot\n")
    with pytest.raises(ValueError, match="source_changed"):
        probe.validate_manifest(root)
    assert not (root / "probe-state.json").exists()


@pytest.mark.parametrize("unknown", [None, True, "false"])
def test_unknown_parent_usage_cannot_be_reinterpreted_as_known(prepared, unknown):
    """Require explicit complete usage rather than truthy or missing values."""
    root, m = prepared
    parent = Path(m["parent"])
    value = json.loads(parent.read_text())
    value["new_unknown_usage"] = unknown
    atomic_json(parent, value)
    with pytest.raises(ValueError, match="parent_invalid"):
        probe.prepare_protocol(root.parent / "another-root", parent.parent)


@pytest.mark.parametrize("field", ["output", "reservation", "parent_requests"])
def test_rebound_input_still_cannot_increase_explicit_limits(prepared, field):
    """Enforce independent numeric limits even after a manifest is rebound."""
    root, m = prepared
    if field == "output":
        m["frame"]["max_tokens"] = 20001
    elif field == "reservation":
        m["frame"]["reservation"] = 50001
    else:
        m["parent_requests"] = 64
    m["binding"] = probe.digest({k: v for k, v in m.items() if k != "binding"})
    atomic_json(root / "probe-manifest.json", m)
    with pytest.raises(ValueError, match="budget_invalid|parent_usage_invalid"):
        probe.validate_manifest(root)
    assert not (root / "probe-state.json").exists()


def test_protocol_prompt_uses_a_valid_json_reference_example(prepared):
    """Do not ask the model to copy pseudo-JSON into a strict response."""
    _, m = prepared
    example = m["frame"]["messages"][0]["content"].split("body_references=", 1)[1]
    refs, _ = json.JSONDecoder().raw_decode(example)
    assert refs == [{"section_key": "methods", "block_index": 0, "sentence_index": 0}]
