"""Exercise serialized-body refinement without relaxing scientific acceptance."""

import copy
import json
from pathlib import Path

import pytest
import test_scientific_repair_pipeline as scientific
import test_structured_repair_pipeline as structured
from test_evaluation_dimensions_pipeline import separated_peer
from test_repair_feedback import run_feedback_case
from test_source_map_pipeline import products
from test_writer_source_view_pipeline import source_case  # noqa: F401
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.closeout import PROFILE, profile_flags
from litagent.config import load_config
from litagent.eval.results import CONTRACT
from litagent.safety.planned_budget import EnvelopeExceeded, nested_json_size
from litagent.writing_flat import _canonical_task_payload
from litagent.writing_v2 import (
    BODY,
    RANGE_WRITING_PLAN,
    body_byte_capacity_error,
    body_byte_progress,
    chapter_request_input,
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
