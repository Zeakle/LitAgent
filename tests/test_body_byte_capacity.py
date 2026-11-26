"""Exercise serialized-body refinement without relaxing scientific acceptance."""

import copy
import json
from pathlib import Path

import httpx
import pytest
import test_scientific_repair_pipeline as scientific
import test_structured_repair_pipeline as structured
from test_evaluation_dimensions_pipeline import separated_peer
from test_repair_feedback import run_feedback_case
from test_separate_review_pipeline import execute
from test_source_map_pipeline import products
from test_workbench_survey_repair import WorkbenchGateway
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent import capacity_recovery
from litagent.capacity import CapacityConfig
from litagent.capacity_recovery import CONTRACT as INITIAL_CAPACITY_RECOVERY
from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.eval.results import BENCHMARK_FAITHFULNESS_POLICY, CONTRACT
from litagent.llm.responses import decode_private_receipt
from litagent.quality import digest
from litagent.safety.planned_budget import (
    EnvelopeExceeded,
    nested_json_size,
    planned_backend,
)
from litagent.writing_flat import _canonical_task_payload, parse
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    WriterError,
    body_byte_capacity_error,
    body_byte_progress,
    chapter_request_input,
    prepare_planned_input,
    repair_request_input,
    request_messages,
    writer_budget_plan,
)


@pytest.mark.parametrize("case_index", [0, 1])
def test_real068_bodies_preserve_the_original_byte_failure(source_case, case_index):
    fixture = json.loads(
        (Path(__file__).parent / "fixtures/body-byte-capacity-068.json").read_text()
    )
    assert fixture["science_verified"] is False
    row = fixture["cases"][case_index]
    body = row["body"]
    before = copy.deepcopy(body)
    assert nested_json_size(body) == row["expected_bytes"] > row["limit"]
    assert sum(len(p["sentences"]) for k in BODY for p in body[k]) == row["unit_count"]
    env = plan_for(source_case)["envelope"]
    exc = EnvelopeExceeded(
        "body", "nested_json_bytes", nested_json_size(body), row["limit"]
    )
    staged = body_byte_capacity_error(exc, body, RANGE_WRITING_PLAN, env)
    assert staged.candidate == body and body == before
    # Eligibility is not acceptance: the original hard scope still rejects it.
    from litagent.safety.planned_budget import check_writer_envelope

    with pytest.raises(EnvelopeExceeded, match="writer_envelope_exceeded"):
        check_writer_envelope(body, env, stage="body")


def test_model_plan_omits_host_labels_without_dropping_future_scientific_rules():
    from litagent.writer_source_view import (
        _CHAPTER_HOST_PLAN_FIELDS,
        _model_chapter_plan,
    )

    normative = {
        "body_max_units": 66,
        "core_coverage_required": True,
        "future_scientific_rule": "retain",
    }
    full = {**_CHAPTER_HOST_PLAN_FIELDS, **normative}
    original = copy.deepcopy(full)
    projected = _model_chapter_plan(full)
    assert projected == normative and full == original
    assert _model_chapter_plan(projected) == projected
    for key in _CHAPTER_HOST_PLAN_FIELDS:
        damaged = {**full, key: "unreviewed-value"}
        with pytest.raises(ValueError, match="writer_source_view_invalid"):
            _model_chapter_plan(damaged)


def plan_for(case, *, capacity=None):
    """Use the actual server profile and complete budget builder."""
    flags = profile_flags(PROFILE)
    # The frozen entry point adds escrow to this base writer plan. It is not
    # a writer_budget_plan keyword; the full consumer test exercises both.
    flags.pop("round_resource_contract")
    return writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        capacity=capacity,
        evaluation_result_contract=CONTRACT,
        **flags,
    )


def test_byte_progress_keeps_the_exact_prefix_and_legacy_wire_unchanged(source_case):
    plan = plan_for(source_case)
    env = plan["envelope"]
    prefix = {"title": source_case.body["title"], BODY[0]: source_case.body[BODY[0]]}
    before = copy.deepcopy(prefix)
    progress = body_byte_progress(prefix, BODY[1], RANGE_WRITING_PLAN, env)
    assert progress["used"] == nested_json_size(prefix)
    assert progress["remaining"] == env["body_json_bytes"] - progress["used"]
    assert progress["minimum_remaining_framing"] > 0
    assert 0 < progress["recommended_text_allowance"] < progress["remaining"]
    _, payload, _ = chapter_request_input(
        source_case.packet,
        source_case.aliases,
        BODY[1],
        prefix,
        prefix,
        RANGE_WRITING_PLAN,
        env,
    )
    assert payload["writing_plan_progress"]["json_bytes"] == progress
    assert prefix == before
    _, old, _ = chapter_request_input(
        source_case.packet,
        source_case.aliases,
        BODY[1],
        prefix,
        prefix,
        RANGE_WRITING_PLAN,
    )
    assert "json_bytes" not in old["writing_plan_progress"]
    assert body_byte_progress(prefix, BODY[1], RANGE_WRITING_PLAN, None) is None


@pytest.mark.parametrize(
    "field", ["body.units", "unit.characters", "nested_json_bytes"]
)
@pytest.mark.parametrize("complete", [False, True])
def test_only_complete_byte_overflow_can_stage_refinement(source_case, field, complete):
    capacity = CapacityConfig(body_json_bytes=nested_json_size(source_case.body) - 1)
    env = plan_for(source_case, capacity=capacity)["envelope"]
    body = copy.deepcopy(source_case.body)
    if not complete:
        body.pop(BODY[-1])
    exc = EnvelopeExceeded(
        "body", field, nested_json_size(body), env["body_json_bytes"]
    )
    error = body_byte_capacity_error(exc, body, RANGE_WRITING_PLAN, env)
    assert (error is not None) is (complete and field == "nested_json_bytes")
    if error:
        assert error.family == "quality" and error.candidate == body
        assert error.candidate is not body and error.diagnostics == exc.diagnostics
    assert body_byte_capacity_error(exc, body, RANGE_WRITING_PLAN, None) is None


class ByteGateway(scientific.ScientificRepairGateway):
    """Change HTTP output only; leave all writer, accounting and evaluation code real."""

    def __init__(self, case, mode):
        super().__init__(case, 0)
        self.mode = mode
        self.capacity_calls = 0

    def answer(self, payload, native_schema):
        payload = _canonical_task_payload(payload)
        if payload.get("cause") != "capacity":
            value = super().answer(payload, native_schema)
            if payload.get("requested_body_section") == BODY[-1]:
                value["units"][0]["text"] += " " * 1200
            return value
        self.capacity_calls += 1
        self.repair_count += 1
        if self.mode == "declined":
            return {
                "outcome": "declined",
                "explanation": "Cannot safely refine.",
                "patches": [],
                "deleted_sources": [],
            }
        patches = []
        for target in payload["targets"].values():
            section, pi, start = target["path"]
            for i, uid in enumerate(target["original_unit_ids"]):
                unit = copy.deepcopy(
                    payload["current_body"][section][pi]["sentences"][start + i]
                )
                unit["text"] = unit["text"].rstrip()
                if self.mode == "persistent" and section == BODY[-1]:
                    # A changed but still over-limit candidate may use only the
                    # remaining shared attempt, always from the same original.
                    unit["text"] += " " * (1200 - self.capacity_calls)
                patches.append({**unit, "source_unit_ids": [uid]})
        return {
            "outcome": "patched",
            "explanation": "Remove only excess whitespace; keep every scientific premise and reference.",
            "patches": patches,
            "deleted_sources": [],
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["valid", "persistent", "declined"])
async def test_real_byte_refinement_shares_two_edits_and_full_scientific_reassessment(
    source_case, tmp_path, monkeypatch, mode
):
    config = load_config("config/expanded-capacity.yaml")
    config.capacity = CapacityConfig(body_json_bytes=1600)
    gateway = separated_peer(ByteGateway(source_case, mode))
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    monkeypatch.setattr(
        structured,
        "FLAGS",
        {**profile_flags(PROFILE), "evaluation_result_contract": CONTRACT},
    )
    structured.select_structured_policy(monkeypatch)
    import litagent.frozen_writing as frozen

    actual_plan = frozen.plan_frozen_budget

    def selected_plan(*args, **kwargs):
        kwargs.setdefault("capacity", config.capacity)
        return actual_plan(*args, **kwargs)

    monkeypatch.setattr(frozen, "plan_frozen_budget", selected_plan)
    report, _ = await run_feedback_case(
        source_case, tmp_path, monkeypatch, 0, config=config
    )
    stored, journal, budget = products(tmp_path)
    state = stored["state"]
    assert not budget["usage_uncertain"] and budget["requests_sent"] <= 64
    assert budget["plan"]["repair_limit"] == 2
    assert 1 <= len(state["attempts"]) <= 2
    assert state["attempts"][0]["cause"] == "capacity"
    assert gateway.capacity_calls >= 1
    assert journal["capacity_candidates"][0]["original"][BODY[-1]][0]["sentences"][0][
        "text"
    ].endswith(" " * 1200)
    if mode == "valid":
        assert len(state["attempts"]) == 2  # Capacity, then real scientific correction.
        assert gateway.capacity_calls == 1 and gateway.inventories == 2
        assert report["delivery"]["publishable"]
        assert report["evaluation"]["faithfulness"]["score"] == 1.0
        assert journal["candidates"][-1]["committed"]
        assert nested_json_size(report["metadata"]["writer_document"]["body"]) <= 1600
        initial = report["metadata"]["evaluation_history"]["initial_evaluation"][
            "faithfulness"
        ]["details"]
        assert initial["unsupported_count"] == 1
    else:
        assert not report["delivery"]["publishable"]
        assert not report["metadata"].get("writer_document")
        assert not journal["candidates"]
        assert gateway.inventories == 0
        assert gateway.capacity_calls == (2 if mode == "persistent" else 1)
        if mode == "persistent":
            attempts = journal["byte_capacity_attempts"]
            assert len(attempts) == 2
            assert (
                attempts[0]["source_body_sha256"] == attempts[1]["source_body_sha256"]
            )
            assert attempts[0]["candidate_sha256"] != attempts[1]["candidate_sha256"]
            assert all(a["scientific_assessment"] == "not_verified" for a in attempts)


class SourceOwnerGateway(WorkbenchGateway):
    """Replay source-owner errors at the HTTP boundary."""

    def __init__(self, case, scenario):
        super().__init__(case, 0)
        self.scenario = scenario
        self.refinements = []

    def answer(self, payload, native_schema):
        payload = _canonical_task_payload(payload)
        if payload.get("cause") != "capacity":
            value = super().answer(payload, native_schema)
            if payload.get("requested_body_section") == BODY[-1]:
                value["units"][0]["text"] += " " * 1200
            return value
        self.capacity_calls += 1
        self.refinements.append(copy.deepcopy(payload))
        patches = []
        for target in payload["targets"].values():
            section, paragraph, start = target["path"]
            units = payload["current_body"][section][paragraph]["sentences"]
            for offset, uid in enumerate(target["original_unit_ids"]):
                unit = units[start + offset]
                patches.append(
                    {**unit, "text": unit["text"].rstrip(), "source_unit_ids": [uid]}
                )
        return {
            "outcome": "patched",
            "explanation": "Remove trailing whitespace while preserving the draft.",
            "patches": patches,
            "deleted_sources": [],
        }

    async def __call__(self, request):
        response = await super().__call__(request)
        if request.url.host != "offline.invalid" or not request.url.path.endswith(
            "/responses"
        ):
            return response
        messages = json.loads(request.content)["input"]
        if json.loads(messages[-1]["content"]).get("cause") != "capacity":
            return response
        if self.capacity_calls != 1 and self.scenario not in {"repeat", "exhausted"}:
            return response
        data = response.json()
        content = data["output"][-1]["content"][0]
        value = json.loads(content["text"])
        value["patches"][0]["source_unit_ids"].extend(
            value["patches"][1]["source_unit_ids"]
        )
        value["patches"][2]["source_unit_ids"] = ["methods:0:0"]
        if self.scenario == "exhausted":
            value["patches"][2]["text"] += f" Candidate {self.capacity_calls}."
        if self.scenario == "unknown":
            data.pop("usage")
        content["text"] = json.dumps(value)
        return httpx.Response(200, json=data)


def test_real_118_replays_two_distinct_source_ownership_errors():
    """Use the original reply and real schema without changing the archive."""
    root = Path("artifacts/targeted-repairs")
    journal_file = root / "118c0b5229f3.json.writer-v2.json"
    if not journal_file.is_file():
        pytest.skip("requires the local immutable 118 writer journal")
    before = journal_file.read_bytes()
    journal = json.loads(before)
    plan = json.loads((root / "118c0b5229f3.json.budget-plan.json").read_bytes())
    call = journal["calls"]["repair_0"]
    record = journal["capacity_candidates"][0]
    guide = call["guidance_wire_record"]
    packet = guide["authority"]["packet"]
    _, _, schema = repair_request_input(
        packet,
        record["original"],
        record["targets"],
        list(packet["cards"]),
        cause="capacity",
        previous_failure=guide["arguments"]["previous_failure"],
        writing_plan=RANGE_WRITING_PLAN,
        envelope=plan["envelope"],
        wire_contract="flat-writer-v1",
        capacity_diagnostic=guide["arguments"]["capacity_diagnostic"],
        guidance_context=guide["authority"]["guidance_context"],
    )
    assert digest(schema) == call["schema_sha256"]
    findings = capacity_recovery.source_findings(call["raw"], schema, record["targets"])
    assert [
        (row["wire_path"][1], row["reason"], row["authorized_target_ids"])
        for row in findings["diagnostics"]
    ] == [
        (24, "multiple_authorized_targets", ["r25", "r26"]),
        (27, "unknown_sources", []),
    ]
    assert findings["diagnostics"][1]["source_unit_ids"] == ["open_problems:2:0"]
    assert not capacity_recovery.enabled(plan["envelope"])
    assert journal_file.read_bytes() == before


def test_benchmark_archive_keeps_original_regeneration_wire_and_source_validation():
    root = Path(
        "artifacts/benchmarks/survey/latency-faithfulness-95-final-run/snapshots/"
        "replay-few-shot-metric-external-abstract-r2"
    )
    journal_path = root / "writer-revisions.json"
    run_path = root / "run.json"
    if not (journal_path.is_file() and run_path.is_file()):
        pytest.skip("requires the local immutable Benchmark archive")
    journal = json.loads(journal_path.read_bytes())
    envelope = json.loads(run_path.read_bytes())["budget_plan"]["envelope"]
    call = journal["calls"]["repair_0"]
    record = journal["capacity_candidates"][0]
    guide = call["guidance_wire_record"]
    packet = guide["authority"]["packet"]
    aliases = list(packet["cards"])
    system, payload, schema = repair_request_input(
        packet,
        record["original"],
        record["targets"],
        aliases,
        cause="capacity",
        previous_failure=guide["arguments"]["previous_failure"],
        writing_plan=RANGE_WRITING_PLAN,
        envelope=envelope,
        wire_contract="flat-writer-v1",
        capacity_diagnostic=guide["arguments"]["capacity_diagnostic"],
        guidance_context=guide["authority"]["guidance_context"],
    )
    system, payload = prepare_planned_input(
        system, payload, packet, aliases, RANGE_WRITING_PLAN, envelope
    )
    messages = request_messages(system, payload, schema, wire_contract="flat-writer-v1")
    regeneration = call["format_recovery"]["attempts"][-1]
    assert regeneration["kind"] == "regeneration"
    for entry in (call, regeneration):
        request, _ = decode_private_receipt(entry["provider_receipt"])
        assert request["input"] == messages
    assert "source_owner_instruction" not in payload["previous_local_failure"]
    findings = capacity_recovery.source_findings(
        regeneration["raw"], schema, record["targets"]
    )
    assert [item["source_unit_ids"] for item in findings["diagnostics"]] == [
        ["open_problems", "3", "0"]
    ]
    with pytest.raises(WriterError, match="writer_capacity_source_map_invalid"):
        parse(regeneration["raw"], schema)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scenario", ["fixed", "benchmark", "repeat", "exhausted", "legacy", "unknown"]
)
async def test_live_byte_refinement_retries_only_authentic_source_owner_failures(
    case, tmp_path, monkeypatch, scenario
):
    config = load_config("config/agent-format-repair.yaml")
    assert config.safety.planned_run_limits.initial_capacity_recovery == (
        INITIAL_CAPACITY_RECOVERY
    )
    if scenario == "legacy":
        config.safety.planned_run_limits.initial_capacity_recovery = None
    if scenario == "benchmark":
        config.eval.benchmark_faithfulness_policy = BENCHMARK_FAITHFULNESS_POLICY
    config.capacity = config.capacity.model_copy(update={"body_json_bytes": 1600})
    if scenario == "fixed":
        authenticate = capacity_recovery.authenticate_rejection

        def check_authority(session, name, body, targets, failure):
            owner = planned_backend(session.llm._cost_budget)
            entry = owner.entries[f"{session.identity}:{name}"]
            assert entry["stage"] == "repair"
            with monkeypatch.context() as fault:
                fault.setitem(entry, "stage", "capacity/repair")
                assert authenticate(session, name, body, targets, failure) is None
            with monkeypatch.context() as fault:
                usage = session.journal["calls"][name]["usage"]
                fault.setitem(
                    usage, "completion_tokens", usage["completion_tokens"] + 1
                )
                assert authenticate(session, name, body, targets, failure) is None
            different = {**body, "title": "A different manuscript"}
            assert authenticate(session, name, different, targets, failure) is None
            return authenticate(session, name, body, targets, failure)

        monkeypatch.setattr(
            capacity_recovery, "authenticate_rejection", check_authority
        )
    gateway = SourceOwnerGateway(case, scenario)
    report = await execute(case, tmp_path, monkeypatch, gateway, config=config)
    root = tmp_path / "artifacts/targeted-repairs"
    (journal_file,) = root.glob("*.json.writer-v2.json")
    (repair_file,) = root.glob("????????????.json")
    (budget_file,) = root.glob("*.json.budget-ledger.json")
    journal = json.loads(journal_file.read_bytes())
    stored = json.loads(repair_file.read_bytes())
    budget = json.loads(budget_file.read_bytes())
    attempts = stored["state"]["attempts"]
    record = journal["capacity_candidates"][0]
    rejections = record.get("source_map_rejections", [])
    expected = {
        "fixed": 2,
        "benchmark": 2,
        "repeat": 2,
        "exhausted": 3,
        "legacy": 1,
        "unknown": 1,
    }[scenario]
    assert gateway.capacity_calls == expected, journal.get("failure")
    assert all(a["cause"] == "capacity" for a in attempts[:expected])
    sent_failures = [row["previous_local_failure"] for row in gateway.refinements]
    if scenario == "benchmark":
        assert all(
            '["open_problems:3:0"]' in row["source_owner_instruction"]
            and "Never split" in row["source_owner_instruction"]
            for row in sent_failures
        )
    else:
        assert all("source_owner_instruction" not in row for row in sent_failures)
    if scenario == "unknown":
        assert budget["usage_uncertain"]
        assert not rejections and gateway.global_calls == 0
        assert not report["delivery"]["publishable"]
        return
    assert all(entry["status"] == "settled" for entry in budget["entries"].values())
    assert len(rejections) == (
        0
        if scenario == "legacy"
        else 1 if scenario in {"fixed", "benchmark"} else expected
    )
    for row in rejections:
        assert row["body_sha256"] == record["snapshot_sha256"]
        assert [item["reason"] for item in row["diagnostics"]] == [
            "multiple_authorized_targets",
            "unknown_sources",
        ]
    if scenario in {"fixed", "benchmark"}:
        assert gateway.global_calls >= 1
        assert record["status"] == "locally_valid_unverified"
        assert (
            report["evaluation"]["faithfulness"]["details"]["stages"]["global_check"]
            == "completed"
        )
        assert all(
            attempt["source_body_sha256"] == record["snapshot_sha256"]
            for attempt in attempts[:expected]
        )
        found = gateway.refinements[1]["repair_guidance"]
        assert "multiple_authorized_targets" in json.dumps(found)
        assert "unknown_sources" in json.dumps(found)
        assert "methods:0:0" not in json.dumps(found)
    else:
        assert not report["delivery"]["publishable"]
        assert gateway.global_calls == 0
        assert not journal["candidates"]
        if rejections:
            assert rejections[-1]["next_action"] == (
                "stop_repeated_candidate"
                if scenario == "repeat"
                else "stop_capacity_exhausted"
            )
