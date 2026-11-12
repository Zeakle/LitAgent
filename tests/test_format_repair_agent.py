"""Exercise model-proposed format edits and the actual Responses consumer."""

import base64
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from test_native_responses import envelope, owner_for
from test_unavailable_cost_exports import display_case, exported
from test_writer_v2 import case

from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.eval.engine_policy import configured_engine
from litagent.eval.results import CONTRACT as RESULT_CONTRACT
from litagent.format_repair import FormatRepairAgent, validate_recovery_record
from litagent.frozen_writing import FrozenGuardedClient, plan_frozen_budget
from litagent.llm.client import ProviderResponseError
from litagent.llm.routing import ModelBoundClient
from litagent.output_capacity import configured_output_tokens
from litagent.quality import build_fact_cards
from litagent.run_limits import FORMAT_JSON_POLICY, FORMAT_MODELS, configured_run_limits
from litagent.safety.planned_accounting import validate_accounting_snapshot
from litagent.writing_flat import encode, parse
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    WriterSession,
    body_section_schema,
    chapter_request_input,
    source_packet,
    writer_format_status,
)


def edit(first, last, replacement=""):
    return {"start": first, "end": last, "replacement": replacement}


@pytest.mark.parametrize("problem", ["extra", "separator", "trailing", "fence", "eof"])
def test_patch_keeps_literal_offsets(case, problem):
    schema = body_section_schema("methods", case.aliases)
    text = '证据、标点 },] 和转义 "。'
    canonical = {"methods": [{"sentences": [{"text": text, "fact_card_refs": ["c1"]}]}]}
    valid = json.dumps(encode(canonical, schema), ensure_ascii=False)
    if problem == "extra":
        raw, edits = valid + "}", [edit(len(valid), len(valid) + 1)]
    elif problem == "trailing":
        pos = len(valid) - 1
        raw, edits = valid[:pos] + "," + valid[pos:], [edit(pos, pos + 1)]
    elif problem == "separator":
        pos = valid.index(', "fact_card_refs"')
        raw, edits = valid[:pos] + valid[pos + 1 :], [edit(pos, pos, ",")]
    elif problem == "fence":
        prefix = "```json\n"
        raw = prefix + valid + "\n```"
        edits = [edit(0, len(prefix)), edit(len(prefix) + len(valid), len(raw))]
    else:
        raw = valid[:-2]
        edits = [edit(len(raw), len(raw), valid[-2:])]
    parsed = parse(raw, schema, format_patch=edits)
    assert parsed["canonical"] == canonical
    span = parsed["bindings"]["units"][0]["text_span"]
    assert json.loads(raw[span["start"] : span["end"]]) == text
    assert parsed["normalized_raw"] == valid


@pytest.mark.parametrize(
    "attack", ["move_array", "change_number", "duplicate", "missing_unit_close"]
)
def test_patch_rejects_ownership_or_literal_changes(attack):
    if attack == "move_array":
        raw = '{"units":[{"text":"a","refs":[]},{"text":"b","refs":["c1"]}]}'
        pos = raw.index("[]")
        edits = [edit(pos, pos + 2, "[")]
    elif attack == "change_number":
        raw = '{"n":12}}'
        edits = [edit(5, 7, "13"), edit(8, 9)]
    elif attack == "duplicate":
        raw = '{"n":12,"n":13}}'
        edits = [edit(len(raw) - 1, len(raw))]
    else:
        raw = '{"units":[{"text":"a","refs":["c1"]'
        edits = [edit(len(raw), len(raw), "}]}")]
    with pytest.raises(ValueError):
        FormatRepairAgent(raw).apply(edits)


@pytest.mark.parametrize("response", ["original", "regeneration"])
def test_411_root_closure_passes_original_provider_schema(response):
    journal = Path(
        "artifacts/acceptance/workbench-few-shot-411-20261111/runtime/"
        "targeted-repairs/e2f5eed4f5fa.json.writer-v2.json"
    )
    if not journal.is_file():
        pytest.skip("411 provider responses are available only in the local archive")
    source = json.loads(journal.read_bytes())["calls"]["repair_1"]["format_recovery"]
    original = source["original"]
    provider_request = json.loads(
        base64.b64decode(original["provider_receipt"]["request_body_base64"])
    )
    schema = provider_request["text"]["format"]["schema"]
    raw = original["raw"] if response == "original" else source["attempts"][-1]["raw"]
    agent = FormatRepairAgent(raw)
    value, _, candidate, _ = agent.apply(agent.root_closure_edits())
    assert candidate == raw + "}"
    Draft202012Validator(schema).validate(value)


@pytest.fixture(scope="module")
def archive():
    root = Path("artifacts/acceptance/ragas-native-single-20261109-075")
    names = ("execution.json", "repair.json.writer-v2.json", "final-budget.json")
    if not all((root / name).is_file() for name in names):
        pytest.skip("075 history is available only in the local archive")
    originals = {name: (root / name).read_bytes() for name in names}
    execution, journal, accounting = [json.loads(originals[name]) for name in names]
    validate_accounting_snapshot(accounting, expected_plan=accounting["plan"])
    cfg = load_config("config/agent-format-repair.yaml")
    plan = plan_frozen_budget(
        execution["bundle"],
        budget_policy="planned-budget-v7",
        capacity=cfg.capacity,
        output_tokens=configured_output_tokens(cfg),
        preparation_budget_snapshot=execution["preparation_snapshot"],
        semantic_engine=configured_engine(cfg),
        run_limits=configured_run_limits(cfg),
        **profile_flags(PROFILE),
        evaluation_result_contract=RESULT_CONTRACT,
    )
    assert plan["status"] == "admitted"
    assert plan["repair_limit"] == 11
    assert plan["maximum_requests"] == 226
    assert plan["maximum_tokens"] == 47_420_000
    assert plan["planned_requests"] == accounting["plan"]["planned_requests"] + 66
    yield execution, journal, plan
    assert originals == {name: (root / name).read_bytes() for name in names}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "valid",
        "local_eof",
        "invalid_local_candidate",
        "lite",
        "pro",
        "regeneration",
        "failure",
        "unknown",
    ],
)
async def test_075_recovery_with_sdk_budget_and_replay(tmp_path, archive, path):
    execution, journal, plan = archive
    bundle = execution["bundle"]
    selection = EvidenceSelection.from_dict(bundle["selection"])
    packet, aliases = source_packet(
        build_fact_cards(selection.selected_items),
        selection,
        bundle["query"],
        bundle["requirements"],
        "zh-CN",
    )
    name = "body:experiments"
    raw = journal["calls"][name]["raw"]
    prefix = {}
    for section in BODY[: BODY.index("experiments")]:
        prefix.update(journal["calls"]["body:" + section]["reference_canonical"])
    system, payload, schema = chapter_request_input(
        packet,
        aliases,
        "experiments",
        prefix,
        prefix,
        RANGE_WRITING_PLAN,
        plan["envelope"],
    )
    requests = []
    initial = {
        "valid": raw[:-1],
        "local_eof": raw[:-2],
        "invalid_local_candidate": "{",
    }.get(path, raw)

    def handler(request):
        request_body = json.loads(request.content)
        index = len(requests)
        requests.append(request_body)
        if index == 0:
            answer = initial
        elif index == 3:
            assert request_body == requests[0]
            answer = raw if path == "failure" else raw[:-1]
        elif (path, index) in {("lite", 1), ("pro", 2)}:
            answer = json.dumps(
                {
                    "outcome": "patch",
                    "edits": [edit(len(raw) - 1, len(raw))],
                }
            )
        elif path == "pro":
            answer = "{"
        else:
            answer = '{"outcome":"cannot_repair","edits":[]}'
        kwargs = {"usage": None} if path == "unknown" and index == 1 else {}
        return httpx.Response(
            200, json=envelope(answer, model=request_body["model"], **kwargs)
        )

    owner, _, _ = await owner_for(handler)
    owner._cost_budget.restore(execution["preparation_snapshot"])
    owner._cost_budget.activate_plan(plan)

    guards = []

    def session():
        return WriterSession(
            FrozenGuardedClient(
                ModelBoundClient(owner, FORMAT_MODELS[1], "generation"),
                lambda: guards.append("checked"),
            ),
            tmp_path / "repair.json",
            "075-agent-offline",
            packet,
            aliases,
            provider_api="responses",
            writing_plan=RANGE_WRITING_PLAN,
            syntax_policy=FORMAT_JSON_POLICY,
        )

    try:
        writer = session()
        if path == "unknown":
            with pytest.raises(ProviderResponseError, match="strict_usage_incomplete"):
                await writer.call(name, system, payload, schema)
        elif path == "failure":
            with pytest.raises(WriterError, match="writer_json_invalid"):
                await writer.call(name, system, payload, schema)
        else:
            value = await writer.call(name, system, payload, schema)
            assert len(writer.reference_input(value)[1]) == 10
            assert session().reference_input(value) == writer.reference_input(value)
        expected_calls = {
            "valid": 1,
            "local_eof": 1,
            "invalid_local_candidate": 4,
            "lite": 2,
            "pro": 3,
            "regeneration": 4,
            "failure": 4,
            "unknown": 2,
        }
        assert len(requests) == expected_calls[path]
        assert len(guards) >= len(requests)
        if len(requests) > 1:
            assert requests[1]["model"] == FORMAT_MODELS[0]
        if len(requests) > 2:
            assert requests[2]["model"] == FORMAT_MODELS[1]
        status = writer_format_status(writer.journal)
        if path == "valid":
            assert status["calls"] == []
        else:
            entry = writer.journal["calls"][name]
            recovery = entry["format_recovery"]
            assert recovery["original"]["raw"] == initial
            validate_recovery_record(entry)
            if path == "local_eof":
                assert entry["raw"] == initial
                assert recovery["attempts"] == [
                    {"kind": "local_eof", "status": "accepted", "sent": False}
                ]
                assert status["recovered_count"] == 1
                assert status["requests_used"] == status["actual_tokens"] == 0
                recovery["attempts"][0]["sent"] = True
                with pytest.raises(WriterError, match="writer_format_receipt_conflict"):
                    validate_recovery_record(entry)
                recovery["attempts"][0]["sent"] = False
            if path in {"lite", "pro", "local_eof"}:
                entry["format_patch"]["candidate"] += " "
                with pytest.raises(WriterError, match="writer_format_receipt_conflict"):
                    validate_recovery_record(entry)
        with pytest.raises(WriterError, match="writer_request_no_replay"):
            await writer.call(name, system, payload, schema)
        snapshot = owner._cost_budget._planned_backend.snapshot()
        validate_accounting_snapshot(snapshot, expected_plan=plan)
        assert (
            snapshot["requests_sent"] == plan["spent_requests"] + expected_calls[path]
        )
        assert snapshot["usage_uncertain"] == (path == "unknown")
    finally:
        await owner.close()


def test_workbench_export_keeps_format_paths_and_budget_separate(display_case):
    root, manifest, state, _ = display_case
    status = {
        "policy": FORMAT_JSON_POLICY,
        "recovered_count": 0,
        "regenerated_count": 1,
        "requests_used": 3,
        "requests_remaining": 63,
        "reserved_tokens": 14014242,
        "actual_tokens": 521,
        "unknown_tokens": 0,
        "calls": [
            {
                "call": "body:experiments",
                "stage": "synthesis",
                "status": "regenerated",
                "kinds": [
                    "lite · declined",
                    "pro · rejected",
                    "regeneration · accepted",
                ],
                "reason": None,
                "attempts": [
                    {"kind": "lite", "status": "declined"},
                    {"kind": "pro", "status": "rejected"},
                    {"kind": "regeneration", "status": "accepted"},
                ],
            }
        ],
    }
    state["cases"][0]["writer_format"] = status
    public, rows, markdown, html = exported(root, state, manifest)
    assert (
        public["cases"][0]["writer_format"]
        == json.loads(rows[0]["writer_format"])
        == status
    )
    for value in (
        "regenerated",
        "lite · declined",
        "pro · rejected",
        "14014242",
        "521",
    ):
        assert value in markdown and value in html
