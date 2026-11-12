"""Authenticate conservative conflict effects before shared report repair."""

import copy
import json

import pytest
import test_scientific_repair_pipeline as scientific
import test_structured_repair_pipeline as structured
from test_repair_feedback import run_feedback_case
from test_repair_task_view import decoded_issue_rows
from test_report_repair_eligibility import (
    SemanticGateway,
    corrupt_observation,
    prepare_case,
)
from test_source_map_pipeline import products, sent_with
from test_writer_v2 import case  # noqa: F401

from litagent.eval.semantic_conflict import SEMANTIC_CONFLICT_POLICY
from litagent.quality import digest
from litagent.writing_flat import REPAIR_TASK_VIEW_REVISION


class ConflictGateway(SemanticGateway):
    """Alter only transmitted replies while preserving real consumers and owners."""

    def __init__(
        self,
        case,
        *,
        direction="negative",
        failing_candidates=0,
        fault=None,
        negative_reason="insufficient_support",
    ):
        super().__init__(case, failing_candidates, fault=fault)
        original = self.evaluator.chat

        async def chat(messages, **kwargs):
            response = await original(messages, **kwargs)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            value = json.loads(response.content)
            claims = {c["claim_id"]: c for c in payload["assertions"]}
            for decision in value["decisions"]:
                claim = claims[decision["claim_id"]]
                if case.semantic_wrong not in claim["claim_text"]:
                    continue
                decision["explanation"] = (
                    "LOCAL_CONFLICT_EXPLANATION_" + decision["claim_id"]
                )
                for aspect in decision["aspect_checks"]:
                    aspect["explanation"] = (
                        "LOCAL_CONFLICT_ASPECT_"
                        + decision["claim_id"]
                        + aspect["aspect"]
                    )
                if direction == "negative":
                    decision.update(status="unsupported", reason_code=negative_reason)
                else:
                    decision.update(status="supported", reason_code="supported")
                    decision["aspect_checks"][0]["supported"] = False
                if fault == "atomic":
                    decision["atomic"] = False
                if fault == "extra_aspect":
                    # Keep the protocol fault introduced by the lower transport fixture.
                    assert len(decision["aspect_checks"]) > len(
                        claim["required_support_aspects"]
                    )
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


async def run_conflict(
    case,
    tmp_path,
    monkeypatch,
    *,
    direction="negative",
    failing_candidates=0,
    fault=None,
    selected=True,
    negative_reason="insufficient_support",
):
    """Run actual frozen synthesis, reviewer, report, evaluation and shared repair."""
    flags = {
        **structured.FLAGS,
        "repair_mapping_contract": "capacity-source-map-v4",
        "support_validation_revision": "host-support-receipts-v2",
        "repair_guidance_wire_revision": REPAIR_TASK_VIEW_REVISION,
    }
    if selected:
        flags["semantic_conflict_policy"] = SEMANTIC_CONFLICT_POLICY
    monkeypatch.setattr(structured, "FLAGS", flags)
    structured.select_structured_policy(monkeypatch)
    gateway = ConflictGateway(
        case,
        direction=direction,
        failing_candidates=failing_candidates,
        fault=fault,
        negative_reason=negative_reason,
    )
    monkeypatch.setattr(scientific, "ScientificRepairGateway", lambda *_: gateway)
    report, _ = await run_feedback_case(case, tmp_path, monkeypatch, failing_candidates)
    return report, gateway


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reason",
    [
        "insufficient_support",
        "uncited_claim",
        "attribution_mismatch",
        "record_mismatch",
    ],
)
async def test_every_legal_negative_reason_reaches_authenticated_repair_and_full_reevaluation(
    case, tmp_path, monkeypatch, reason
):
    """A legal status pair must not fail at a later typed-guidance consumer."""
    prepare_case(case)
    report, gateway = await run_conflict(
        case, tmp_path, monkeypatch, negative_reason=reason
    )
    initial = report["metadata"]["evaluation_history"]["initial_evaluation"][
        "faithfulness"
    ]["details"]
    effects = [
        value["assessment_conflict_effect"]
        for value in initial["assertions"]
        if value.get("assessment_conflict_effect")
    ]
    assert effects and all(value["original_reason_code"] == reason for value in effects)
    assert initial["stages"]["global_check"] == "completed"
    repair = sent_with(gateway, "targets")[0]["payload"]
    projected = [
        row["assessment_conflict_effect"]
        for row in decoded_issue_rows(repair["repair_guidance"], "current")
        if row.get("assessment_conflict_effect")
    ]
    assert projected and all(
        value["original_reason_code"] == reason for value in projected
    )
    assert gateway.repair_count == 1
    assert len(sent_with(gateway, "global_output_contract")) == 2
    assert report["delivery"]["publishable"] is True
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["coverage"]["complete"] is True
    assert final["supported_count"] == final["factual_claim_count"]
    assert final["unsupported_count"] == final["unverified_count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["negative", "positive"])
async def test_complete_conflict_keeps_raw_and_repairs_only_prose(
    case, tmp_path, monkeypatch, direction
):
    """A fully assessed effect offers review, never support for the original."""
    prepare_case(case)
    report, gateway = await run_conflict(
        case, tmp_path, monkeypatch, direction=direction
    )
    initial = report["metadata"]["evaluation_history"]["initial_evaluation"]
    details = initial["faithfulness"]["details"]
    effects = [a for a in details["assertions"] if "assessment_conflict_effect" in a]
    assert effects and all(a["status"] == "unverifiable" for a in effects)
    assert details["coverage"]["complete"] is False
    assert details["stages"]["global_check"] == "completed"
    assert details["stages"]["processed_assertions"] == len(details["assertions"])
    original_status = "unsupported" if direction == "negative" else "supported"
    raw = {
        d["claim_id"]: d for b in details["stages"]["batches"] for d in b["decisions"]
    }
    for assertion in effects:
        assert raw[assertion["claim_id"]]["status"] == original_status
        assert "assessment_conflict_effect" not in raw[assertion["claim_id"]]
        assert (
            assertion["assessment_conflict_effect"]["original_status"]
            == original_status
        )
    assert gateway.repair_count == 1 and gateway.inventories == 2
    assert len(sent_with(gateway, "global_output_contract")) == 2
    assert report["delivery"]["publishable"] is True
    final = report["evaluation"]["faithfulness"]["details"]
    assert final["coverage"]["complete"] is True
    assert final["supported_count"] == final["factual_claim_count"]
    assert final["unverified_count"] == 0
    _, journal, ledger = products(tmp_path)
    assert journal["candidates"][0]["committed"] is True
    assert ledger["usage_uncertain"] is False
    repair = sent_with(gateway, "targets")[0]["payload"]
    assert (
        repair["writing_envelope"]["semantic_conflict_policy"]
        == SEMANTIC_CONFLICT_POLICY
    )
    rows = decoded_issue_rows(repair["repair_guidance"], "current")
    projected = [row for row in rows if "assessment_conflict_effect" in row]
    assert projected and all(row["aspect_checks"] for row in projected)
    if direction == "negative":
        assert all(
            check["supported"] for row in projected for check in row["aspect_checks"]
        )
    assert "LOCAL_CONFLICT_EXPLANATION_" not in json.dumps(repair)
    assert "LOCAL_CONFLICT_ASPECT_" not in json.dumps(repair)
    for call in journal["calls"].values():
        assert call["semantic_conflict_policy"] == SEMANTIC_CONFLICT_POLICY
        assert (
            call["semantic_conflict_identity"] == journal["semantic_conflict_identity"]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_candidates", [0, 1, 2])
async def test_effect_shares_two_original_based_attempts_and_full_reassessment(
    case, tmp_path, monkeypatch, failing_candidates
):
    """Every failed candidate remains local, and no third HTTP slot appears."""
    prepare_case(case)
    report, gateway = await run_conflict(
        case, tmp_path, monkeypatch, failing_candidates=failing_candidates
    )
    expected = min(2, failing_candidates + 1)
    repairs = sent_with(gateway, "targets")
    assert gateway.repair_count == len(repairs) == expected
    assert gateway.inventories == expected + 1
    assert len(sent_with(gateway, "global_output_contract")) == expected + 1
    for repair in repairs:
        body = json.dumps(repair["payload"]["current_body"])
        assert case.semantic_wrong in body and "AnotherGhostNet" not in body
    state, journal, ledger = products(tmp_path)
    assert len(state["state"]["attempts"]) == expected
    assert len(journal["candidates"]) == expected
    assert journal["candidates"][-1]["committed"] is (failing_candidates < 2)
    assert report["delivery"]["publishable"] is (failing_candidates < 2)
    assert ledger["usage_uncertain"] is False
    if failing_candidates == 2:
        assert case.semantic_wrong in report["survey"]
        assert "AnotherGhostNet" not in report["survey"]


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["negative", "positive"])
@pytest.mark.parametrize(
    "fault",
    [
        "atomic",
        "classification",
        "source_fidelity",
        "extra_aspect",
        "missing_aspect",
        "duplicate_aspect",
        "unknown_basis",
        "wrong_paper_basis",
        "global_incomplete",
        "global_json",
    ],
)
async def test_effect_cannot_hide_invalid_or_incomplete_diagnostics(
    case, tmp_path, monkeypatch, direction, fault
):
    """A typed semantic conflict cannot replace source or completeness checks."""
    prepare_case(case)
    if fault == "wrong_paper_basis":
        case.body["taxonomy"][0]["sentences"][0] = {
            "text": case.ledger["a"]["supporting_text"],
            "fact_card_refs": [case.by_eid["a"]],
        }
    report, gateway = await run_conflict(
        case, tmp_path, monkeypatch, direction=direction, fault=fault
    )
    assert gateway.repair_count == 0 and not sent_with(gateway, "targets")
    assert report["delivery"]["publishable"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    [
        "source_sha",
        "binding_sha",
        "request_input_sha",
        "batch_input_sha",
        "provider_usage",
        "missing_batch",
        "partial_batch",
        "missing_id",
        "duplicate_id",
        "global_unfinished",
        "effect_code",
        "effect_binding",
        "effect_raw_status",
        "effect_settlement",
        "batch_effect",
    ],
)
async def test_authenticated_effect_rejects_changed_observations_before_http(
    case, tmp_path, monkeypatch, fault
):
    """Use the real eligibility predicate on original and corrupted observations."""
    import litagent.writing_v2 as writing

    prepare_case(case)
    original = writing._semantic_conflict_repair_eligible
    seen = []

    def checked(runner, **arguments):
        assert original(runner, **arguments) is True
        before = digest(arguments["initial_evaluation"])
        if not fault.startswith("effect_") and fault != "batch_effect":
            changed = corrupt_observation(arguments, fault)
        else:
            changed = {
                **arguments,
                "initial_evaluation": copy.deepcopy(arguments["initial_evaluation"]),
            }
            details = changed["initial_evaluation"]["faithfulness"]["details"]
            assertion = next(
                a for a in details["assertions"] if "assessment_conflict_effect" in a
            )
            effect = assertion["assessment_conflict_effect"]
            if fault == "effect_code":
                effect["code"] = "an_unrecognized_conflict"
            elif fault == "effect_binding":
                effect["binding_sha256"] = "0" * 64
            elif fault == "effect_raw_status":
                effect["original_status"] = "supported"
            elif fault == "effect_settlement":
                request = next(
                    r
                    for r in details["stages"]["requests"]
                    if "semantic_conflict_settlement" in r
                )
                request["semantic_conflict_settlement"]["response_sha256"] = "0" * 64
            else:
                batch = next(
                    b
                    for b in details["stages"]["batches"]
                    if b.get("assessment_conflict_effects")
                )
                batch["assessment_conflict_effects"][0]["binding_sha256"] = "0" * 64
        result = original(runner, **changed)
        assert digest(arguments["initial_evaluation"]) == before
        seen.append(result)
        return result

    monkeypatch.setattr(writing, "_semantic_conflict_repair_eligible", checked)
    report, gateway = await run_conflict(case, tmp_path, monkeypatch)
    assert seen == [False]
    assert gateway.repair_count == 0 and report["delivery"]["publishable"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize("direction", ["negative", "positive"])
async def test_unselected_v2_keeps_original_consistency_stop(
    case, tmp_path, monkeypatch, direction
):
    """The old complete V2 stack cannot acquire the new effect or repair scope."""
    prepare_case(case)
    report, gateway = await run_conflict(
        case, tmp_path, monkeypatch, direction=direction, selected=False
    )
    assert gateway.repair_count == 0 and report["delivery"]["publishable"] is False
    details = report["evaluation"]["faithfulness"]["details"]
    progress = json.loads(
        next((tmp_path / "evaluation").glob("*/progress.json")).read_text()
    )["progress"]
    assert (
        progress["batches"][0]["reason_code"]
        == "evaluation_decision_consistency_invalid"
    )
    assert not any(
        "assessment_conflict_effect" in a for a in details.get("assertions", [])
    )


@pytest.mark.parametrize("changed", ["instruction", "effect_projection", "task_view"])
def test_selected_guidance_identity_tracks_actual_rules_without_legacy_drift(
    monkeypatch, changed
):
    """New rule edits invalidate selected plans while old default hashes stay fixed."""
    from litagent import writing_flat as flat

    legacy = (
        flat.structured_repair_guidance_identity(),
        flat.repair_guidance_wire_identity(),
        flat.source_map_wire_identity(),
    )
    selected = flat.semantic_conflict_guidance_identity()

    def altered_projection(*args, **kwargs):
        """Provide a different actual source unit for the identity negative control."""
        raise AssertionError("Identity checks never execute this projection.")

    if changed == "instruction":
        monkeypatch.setattr(
            flat, "ASSESSMENT_CONFLICT_REVIEW_INSTRUCTION", "Changed rule."
        )
    else:
        name = (
            "_assessment_conflict_guidance"
            if changed == "effect_projection"
            else "_repair_task_view_build"
        )
        monkeypatch.setattr(flat, name, altered_projection)
    assert flat.semantic_conflict_guidance_identity() != selected
    assert (
        flat.structured_repair_guidance_identity(),
        flat.repair_guidance_wire_identity(),
        flat.source_map_wire_identity(),
    ) == legacy
