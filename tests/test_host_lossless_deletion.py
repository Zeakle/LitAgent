"""Preserve explicit capacity deletions across canonical and raw-reference views."""

import copy
import json

import pytest
from test_host_lossless_pipeline import setup
from test_reference_set_pipeline import DuplicateGateway
from test_writer_v2 import case  # noqa: F401

from litagent.capacity_repair import capacity_targets, original_unit_directory
from litagent.quality import digest
from litagent.writing_v2 import (
    BODY,
    WriterError,
    body_units,
    refresh_summaries,
)


class WholeTargetDeletionGateway(DuplicateGateway):
    """Omit only a target whose originals are all explicitly mapped to deletion."""

    def __init__(self, case, *, missing_record=False, unsupported=False):
        super().__init__(case, unsupported=unsupported)
        self.missing_record = missing_record
        self.deleted_target = None

    def answer(self, payload, native_schema):
        if payload.get("cause") == "capacity":
            tid, target = next(
                (tid, target)
                for tid, target in payload["targets"].items()
                if target["path"][:2] == [BODY[0], 1]
            )
            self.deleted_target = tid
            records = [
                {
                    "unit_id": uid,
                    "replacement_indexes": [],
                    "reason": "Remove this redundant elaboration only.",
                }
                for uid in target["original_unit_ids"]
            ]
            if self.missing_record:
                records.pop()
            return {
                "outcome": "patched",
                "explanation": "Explicitly remove the redundant paragraph.",
                "patches": [],
                "edit_records": records,
            }
        value = super().answer(payload, native_schema)
        if payload.get("requested_body_section") == BODY[0]:
            first = copy.deepcopy(value["units"][0])
            first["paragraph_start"] = True
            second = copy.deepcopy(first)
            second.update(
                text="Redundant elaboration of the same source.", paragraph_start=False
            )
            value["units"].extend([first, second])
        return value


@pytest.mark.asyncio
async def test_explicit_whole_target_deletion_preserves_untouched_received_units_and_refreshes_summary(
    case, tmp_path
):
    gateway = WholeTargetDeletionGateway(case)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        old_summary = await session.summaries(body, "summary_initial")
        received, original_origins = session.reference_input(body)
        original_body = copy.deepcopy(body)
        targets = capacity_targets(body, {}, maximum=session.envelope["body_max_units"])
        sent_before = len(gateway.sent)
        candidate = await session.patch(
            body, targets, cause="capacity", claim_count=144
        )
        assert body == original_body
        assert len(candidate[BODY[0]]) == 1
        assert candidate[BODY[0]][0] == body[BODY[0]][0]
        assert all(candidate[section] == body[section] for section in BODY[1:])
        actual, origins = session.reference_input(candidate)
        assert actual[BODY[0]] == received[BODY[0]][:1]
        assert all(actual[section] == received[section] for section in BODY[1:])
        assert len(origins) == len(session.reference_paths(candidate)) == 5
        assert all(origin["call"].startswith("body:") for origin in origins)
        assert len(gateway.sent) == sent_before + 1
        attempt = session.state["attempts"][-1]
        deleted = gateway.deleted_target
        assert attempt["replacement_lengths"][deleted] == 0
        assert attempt["provenance_receipt"][
            "original_unit_directory_sha256"
        ] == digest(original_unit_directory(targets))
        call = session.journal["calls"]["repair_0"]
        assert json.loads(call["raw"])["patches"] == []
        assert call["wire_bindings"]["units"] == []
        assert len(attempt["edit_records"]) == len(original_origins)
        refreshed = await refresh_summaries(
            session, body, candidate, old_summary, targets
        )
        assert session.reference_input(refreshed)[0]
        assert all(
            ref in body_units(candidate)
            for paragraphs in refreshed.values()
            for paragraph in paragraphs
            for unit in paragraph["sentences"]
            for ref in unit["body_unit_refs"]
        )
        assert len(session.state["attempts"]) == 1
    finally:
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["supported", "unsupported", "core_missing"])
async def test_deleted_target_summary_reviewer_and_complete_evaluation_keep_strict_gate(
    case, tmp_path, outcome
):
    """Exercise actual consumers after a structural deletion, with only HTTP mocked."""
    from litagent.agents.reviewer import ReviewerWorker
    from litagent.config import EvalConfig
    from litagent.eval.scoped import evaluate_scoped
    from litagent.llm.routing import ModelBoundClient
    from litagent.orchestrator.task_graph import SubTask
    from litagent.quality import strict_acceptance
    from litagent.writing_v2 import (
        EOF_CONTAINER_POLICY,
        PROMPT_CONTRACT,
        RANGE_WRITING_PLAN,
        compile_document,
    )

    gateway = WholeTargetDeletionGateway(case, unsupported=outcome == "unsupported")
    if outcome == "core_missing":
        actual_evaluator = gateway.evaluator.chat

        async def missing_core(messages, **options):
            response = await actual_evaluator(messages, **options)
            value = json.loads(response.content)
            if "coverage" in value:
                value["coverage"][0].update(
                    answered=False,
                    claim_ids=[],
                    explanation="The required comparison remains unanswered.",
                )
                response.content = json.dumps(value)
            return response

        gateway.evaluator.chat = missing_core
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        summary = await session.summaries(body, "summary_initial")
        targets = capacity_targets(body, {}, maximum=session.envelope["body_max_units"])
        candidate = await session.patch(
            body, targets, cause="capacity", claim_count=144
        )
        refreshed = await refresh_summaries(session, body, candidate, summary, targets)
        survey, bindings = compile_document(
            candidate,
            refreshed,
            case.cards,
            case.aliases,
            "en",
            {},
            writing_plan=RANGE_WRITING_PLAN,
            envelope=session.envelope,
        )
        source = {
            "draft": survey,
            "writing_bindings": bindings,
            "writer_document": {
                "packet": case.packet,
                "aliases": case.aliases,
                "body": candidate,
                "summaries": refreshed,
                "budget_envelope": session.envelope,
                "generation_prompt_contract": PROMPT_CONTRACT,
                "generation_provider_api": "responses",
                "generation_syntax_policy": EOF_CONTAINER_POLICY,
                "generation_writing_plan": RANGE_WRITING_PLAN,
                "writer_wire_contract": "flat-writer-v1",
            },
            "report_language": "en",
            "fact_cards": case.cards,
            "evidence_snapshot_sha256": digest(case.cards),
            "evidence_selection": case.selection.to_dict(),
            "coverage_requirements": case.packet["requirements"],
            "review_methodology": {},
        }
        reviewer = ReviewerWorker(ModelBoundClient(owner, "lite", "reviewer"))
        reviewer._writer_path = session.path
        reviewer._writer_run_id = session.run_id
        reviewed = await reviewer.execute(
            SubTask(
                "reviewer",
                "Review the complete candidate after explicit capacity deletion.",
                "reviewer",
                input_data={
                    "generation_strategy": "body_summary_v3",
                    "query": case.packet["query"],
                    "budget_policy": "planned-budget-v7",
                    "upstream_results": {"synthesis": source},
                },
            )
        )
        assert reviewed["accepted"]
        assert reviewed["source_report_sha256"] == bindings["report_sha256"]
        details = await evaluate_scoped(
            llm=ModelBoundClient(owner, "lite", "evaluation"),
            survey=survey,
            context={
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
            },
            config=EvalConfig(),
        )
        assert details["assessment_stage"] == "host_validated"
        assert details["inventory_check"]["status"] == "passed"
        assert details["coverage"]["complete"]
        assert details["stages"]["global_check"] == "completed"
        verdict = strict_acceptance(details)
        if outcome == "supported":
            assert details["supported_count"] == details["factual_claim_count"] > 0
            assert verdict["status"] == "passed"
        elif outcome == "unsupported":
            assert details["unsupported_count"] > 0
            assert verdict["status"] == "failed"
        else:
            assert details["supported_count"] == details["factual_claim_count"] > 0
            assert verdict["status"] == "failed"
            assert "core_coverage_unverified" in verdict["failure_codes"]
        assert len(session.state["attempts"]) == 1
        assert not owner._cost_budget.snapshot()["usage_uncertain"]
        calls = [row["payload"] for row in gateway.sent]
        assert any("report_layout" in row and "cards" in row for row in calls)
        assert any("allowed_citation_aliases" in row for row in calls)
        assert any("cited_materials" in row for row in calls)
        assert any(
            "coverage_requirements" in row and "allowed_citation_aliases" not in row
            for row in calls
        )
    finally:
        await owner.close()


@pytest.mark.asyncio
async def test_whole_target_deletion_with_missing_original_record_stays_rejected(
    case, tmp_path
):
    gateway = WholeTargetDeletionGateway(case, missing_record=True)
    owner, session = await setup(case, tmp_path, gateway)
    try:
        body = await session.initial_body()
        received_before = session.reference_input(body)
        targets = capacity_targets(body, {}, maximum=session.envelope["body_max_units"])
        with pytest.raises(WriterError, match="writer_capacity_edit_coverage_invalid"):
            await session.patch(body, targets, cause="capacity", claim_count=144)
        assert session.reference_input(body) == received_before
        assert len(session.state["attempts"]) == 1
        assert not session.journal["candidates"]
    finally:
        await owner.close()
