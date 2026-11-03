"""Reject malformed decisions through the real SDK and durable batch consumers."""

import copy
import json

import pytest
from test_decision_status_contract import FIXTURE_PATH
from test_flat_writer_pipeline import FlatGateway
from test_native_responses import owner_for
from test_native_wire_session import make_session
from test_support_receipt_identity import FLAGS, REVISION
from test_writer_v2 import case  # noqa: F401

from litagent.capacity import CapacityConfig
from litagent.config import EvalConfig
from litagent.eval.scoped import (
    ScopedEvaluationError,
    decision_status_instruction,
    decision_status_prompt,
    evaluate_scoped,
)
from litagent.llm.routing import ModelBoundClient
from litagent.quality import digest, strict_acceptance
from litagent.writing_v2 import compile_document, writer_budget_plan


class StatusGateway(FlatGateway):
    """Change external JSON only; retain SDK decoding, aliases and all guards."""

    def __init__(self, case, *, bad_batch=None, verdict=None):
        """Retain the real client and alter only controlled provider responses."""
        super().__init__(case)
        self.assessment_calls = 0
        self.bad_batch = bad_batch
        self.verdict = verdict
        self.injected_decision = None
        original = self.evaluator.chat

        async def chat(messages, **options):
            """Emit the scripted verdict at the existing mocked HTTP boundary."""
            response = await original(messages, **options)
            payload = self.evaluator.base.calls[-1]
            if "cited_materials" not in payload:
                return response
            self.assessment_calls += 1
            value = json.loads(response.content)
            if self.assessment_calls == self.bad_batch:
                raw = copy.deepcopy(
                    json.loads(FIXTURE_PATH.read_text())["decisions"][0]
                )
                # Only host identity is substituted: this is a protocol replay,
                # not an assertion that the archived explanation fits this paper.
                raw["claim_id"] = value["decisions"][0]["claim_id"]
                value["decisions"][0] = raw
                self.injected_decision = copy.deepcopy(raw)
            elif self.verdict in ("unsupported", "unverifiable"):
                for row in value["decisions"]:
                    row.update(
                        status=self.verdict,
                        reason_code=(
                            "insufficient_support"
                            if self.verdict == "unsupported"
                            else "unverifiable"
                        ),
                        explanation="Controlled negative, not a scientific gold label.",
                        support_basis=[],
                    )
                    for aspect in row["aspect_checks"]:
                        aspect.update(supported=False, support_ids=[])
            response.content = json.dumps(value)
            return response

        self.evaluator.chat = chat


async def evaluate_sdk(case, tmp_path, gateway, *, selected=True):
    """Generate real flat calls before exercising the exact assessment pipeline."""
    owner, budget, events = await owner_for(gateway)
    plan = writer_budget_plan(
        case.packet,
        case.aliases,
        methodology={},
        budget_policy="planned-budget-v7",
        capacity=CapacityConfig(
            target_body_units=5,
            max_body_units=5,
            max_summary_units=1,
            initial_assertions=8,
            batch_size=2,
            max_assertions=8,
            max_assessment_batches=4,
        ),
        **FLAGS,
        **({"support_validation_revision": REVISION} if selected else {}),
    )
    assert plan["status"] == "admitted"
    budget.activate_plan(plan, path=tmp_path / "budget.json")
    session = make_session(owner, case, tmp_path)
    checkpoints = []
    try:
        body = await session.initial_body()
        summary = await session.summaries(body, "summary_initial")
        survey, bindings = compile_document(
            body, summary, case.cards, case.aliases, "en", envelope=session.envelope
        )
        await session.call(
            "reviewer",
            "Review the complete draft.",
            {"report_layout": []},
            {
                "type": "object",
                "properties": {
                    "issues": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 128,
                    },
                    "explanation": {"type": "string"},
                },
                "required": ["issues", "explanation"],
                "additionalProperties": False,
            },
            stage="reviewer",
            max_tokens=4096,
        )
        context = {
            "query": case.packet["query"],
            "quality_policy": "strict_auto_v2",
            "body_quality": "analysis_v1",
            "report_language": "en",
            "review_structure": "academic_v2",
            "writing_bindings": bindings,
            "scope_notice_contract": "scope-notice-v1",
            "fact_cards": case.cards,
            "evidence_snapshot_sha256": digest(case.cards),
            "evidence": case.ledger,
            "coverage_requirements": case.packet["requirements"],
            "structure_checks": {"status": "passed"},
            "method_checks": {"status": "not_applicable"},
            "language_checks": {"status": "passed"},
            "evaluation_checkpoint": lambda row: checkpoints.append(copy.deepcopy(row)),
        }
        result, error = None, None
        try:
            result = await evaluate_scoped(
                llm=ModelBoundClient(owner, "lite", "evaluation"),
                survey=survey,
                context=context,
                config=EvalConfig(strict_batch_size=2),
            )
        except ScopedEvaluationError as exc:
            error = exc
        return result, error, checkpoints, budget.snapshot(), events
    finally:
        await owner.close()


def assert_actual_cost(snapshot, gateway):
    """All returned provider calls stay settled, regardless of semantic rejection."""
    entries = list(snapshot["entries"].values())
    assert len(entries) == len(gateway.sent)
    assert snapshot["call_count"] == snapshot["requests_sent"] == len(gateway.sent)
    assert snapshot["used_tokens"] == 120 * len(gateway.sent)
    assert snapshot["usage_uncertain"] is False
    assert all(row["status"] == "settled" and row["sent"] for row in entries)
    assert all(row["actual_tokens"] == 120 for row in entries)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_batch", (1, 2))
async def test_selected_bad_batch_is_settled_but_never_completed_or_followed(
    case, tmp_path, bad_batch
):
    gateway = StatusGateway(case, bad_batch=bad_batch)
    result, error, checkpoints, snapshot, _ = await evaluate_sdk(
        case, tmp_path, gateway
    )
    assert result is None
    assert str(error) == "evaluation_decision_status_invalid"
    details = error.details
    stages = details["stages"]
    assert stages["expected_batches"] > bad_batch
    assert gateway.assessment_calls == bad_batch
    assert len(stages["batches"]) == bad_batch
    assert all(row["status"] == "completed" for row in stages["batches"][:-1])
    invalid = stages["batches"][-1]
    assert invalid["status"] == "invalid_response"
    assert invalid["reason_code"] == "evaluation_decision_status_invalid"
    assert invalid["usage"]["usage_available"] is True
    assert "decisions" not in invalid
    assert stages["processed_assertions"] == sum(
        len(row["claim_ids"]) for row in stages["batches"][:-1]
    )
    assert not any("global_output_contract" in row["payload"] for row in gateway.sent)
    assert stages.get("global_check") != "completed"
    assert details["coverage"]["complete"] is False
    assert details.get("score") is None
    assert details["partial_assessment"]["overall_valid"] is False
    assert (
        details["partial_assessment"]["processed_assertions"]
        == stages["processed_assertions"]
    )
    assert strict_acceptance(details)["status"] == "failed"
    assert checkpoints[-1]["batches"][-1]["status"] == "invalid_response"
    assert all(
        checkpoint["batches"][bad_batch - 1]["status"] != "completed"
        for checkpoint in checkpoints
        if len(checkpoint.get("batches", [])) >= bad_batch
    )
    assert not any("targets" in row["payload"] for row in gateway.sent)
    assert_actual_cost(snapshot, gateway)
    assessments = [row for row in gateway.sent if "support_passages" in row["payload"]]
    assert len(assessments) == bad_batch
    for row in assessments:
        prompt = row["request"]["messages"][0]["content"]
        assert prompt.count(decision_status_prompt()) == 1
        assert decision_status_instruction() not in prompt
        fields = row["request"]["response_format"]["json_schema"]["schema"]["$defs"][
            "PassageDecision"
        ]["properties"]
        assert fields["reason_code"]["description"] == decision_status_instruction()
        assert fields["status"]["description"]


@pytest.mark.asyncio
async def test_none_retains_late_guard_and_unannotated_real_wire(case, tmp_path):
    gateway = StatusGateway(case, bad_batch=1)
    result, error, _, snapshot, _ = await evaluate_sdk(
        case, tmp_path, gateway, selected=False
    )
    assert result is None and str(error) == "evaluation_decision_status_invalid"
    stages = error.details["stages"]
    assert stages["global_check"] == "completed"
    assert gateway.assessment_calls == stages["expected_batches"]
    assert all(row["status"] == "completed" for row in stages["batches"])
    for row in gateway.sent:
        if "support_passages" not in row["payload"]:
            continue
        assert (
            decision_status_instruction()
            not in row["request"]["messages"][0]["content"]
        )
        assert decision_status_prompt() not in row["request"]["messages"][0]["content"]
        fields = row["request"]["response_format"]["json_schema"]["schema"]["$defs"][
            "PassageDecision"
        ]["properties"]
        assert "description" not in fields["status"]
        assert "description" not in fields["reason_code"]
    assert_actual_cost(snapshot, gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ("supported", "unsupported", "unverifiable"))
async def test_legal_pairs_complete_but_negative_or_unknown_never_deliver(
    case, tmp_path, verdict
):
    gateway = StatusGateway(case, verdict=verdict)
    result, error, _, snapshot, _ = await evaluate_sdk(case, tmp_path, gateway)
    assert error is None
    assert result["coverage"]["complete"] is (verdict != "unverifiable")
    assert result["stages"]["global_check"] == "completed"
    assert all(row["status"] == "completed" for row in result["stages"]["batches"])
    assert strict_acceptance(result)["status"] == (
        "passed" if verdict == "supported" else "failed"
    )
    if verdict == "unsupported":
        assert result["unsupported_count"] == result["factual_claim_count"]
    elif verdict == "unverifiable":
        assert result["unverified_count"] == result["factual_claim_count"]
        assert result["score"] is None
        assert result["unverified_reason"] == "evaluation_assertions_unverifiable"
    assert_actual_cost(snapshot, gateway)
