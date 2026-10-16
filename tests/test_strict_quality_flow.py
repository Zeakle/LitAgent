"""Exercise structured generation, transactional repairs and human calibration gates."""

import copy
import json
from types import MethodType
from unittest.mock import AsyncMock

import pytest
from test_strict_quality import draft, to_native, valid_details

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask, TaskGraph
from litagent.quality import (
    build_fact_cards,
    coverage_requirements,
    prepare_grounded_request,
)
from litagent.review_structure import METHOD_BEGIN, prepare_authored_body
from litagent.runner import LitAgent, RewriteOutcome
from litagent.workbench import RunOptions
from scripts.build_strict_quality_calibration import calibration_gate


@pytest.mark.asyncio
@pytest.mark.parametrize("language", ["en", "zh-CN"])
async def test_real_synthesis_and_all_revision_messages_share_frozen_fact_contract(
    language,
):
    item = {
        "evidence_id": "a",
        "paper_id": "evit",
        "text": "EViT improves speed.",
        "supporting_text": "EViT improves speed.",
        "chunk_key": "abstract",
    }
    facts = build_fact_cards({"a": item})
    report = draft(facts, language)
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(
        content=json.dumps(to_native(report, facts)), model="offline", usage={}
    )
    selection = EvidenceSelection(
        candidate_count=1,
        selected_items={"a": item},
        section_evidence_ids={
            k: ["a"]
            for k in [
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            ]
        },
        method_by_section={
            k: "offline"
            for k in [
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            ]
        },
        omitted_count=0,
        estimated_tokens=1,
    )
    selector = AsyncMock()
    selector.select.return_value = selection
    worker = SynthesisWorker(llm, evidence_selector=selector)
    task = SubTask(
        "adversarial_review",
        "strict body",
        "synthesis",
        input_data={
            "query": "EViT",
            "quality_policy": "strict_v1",
            "review_structure": "academic_v2",
            "report_language": language,
        },
    )
    result = await worker.execute(task)
    assert result["quality_policy"] == "strict_v1" and result["fact_cards"] == facts
    assert (
        result["draft"].count(METHOD_BEGIN) == 1 and result["draft"].count("[E:a]") == 7
    )
    kwargs = {
        "quality_policy": "strict_v1",
        "fact_cards": facts,
        "review_structure": "academic_v2",
        "report_language": language,
        "review_methodology": result["review_methodology"],
        "coverage_requirements": result["coverage_requirements"],
    }
    revised = worker.revise(result["draft"], "located issue", selection, **kwargs)
    repaired = worker.rewrite_with_evidence(
        result["draft"],
        "unused packet",
        "located diagnostic",
        evidence_selection=selection.to_dict(),
        **kwargs,
    )
    initial = json.loads(llm.chat.await_args.args[0][1]["content"])
    assert initial["report_language"] == language
    assert initial["writing_contract"] == "grounded-writing-v14"
    for messages in [revised, repaired]:
        packet = json.loads(messages[1]["content"])
        prepared, revision_format = prepare_grounded_request(messages)
        assert json.loads(prepared[1]["content"])["cards"] == initial["cards"] and list(
            packet["sources"].values()
        ) == list(initial["sources"].values())
        assert METHOD_BEGIN not in packet["current_draft"]
        assert revision_format == llm.chat.await_args.kwargs["response_format"]
        assert (
            json.loads(prepared[1]["content"])["section_contract"]
            == initial["section_contract"]
        )
        assert packet["coverage_requirements"] == initial["coverage_requirements"]
        assert packet["coverage_matrix"] == initial["coverage_matrix"]
        assert packet["summary_rules"] == initial["summary_rules"]
        assert packet["report_language"] == initial["report_language"]
    llm.chat.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome, expected",
    [
        ("quality_still_failed", 2),
        ("metric_regressed", 1),
        ("rewrite_run_budget_exceeded", 1),
        ("no_progress", 1),
    ],
)
async def test_strict_repair_bound_preserves_failed_drafts_and_stops_on_blockers(
    outcome, expected
):
    agent = LitAgent(load_config())
    graph = TaskGraph()
    graph.add_task(SubTask("adversarial_review", "write", "adversarial_review"))
    graph.mark_running("adversarial_review")
    graph.mark_done("adversarial_review", {})
    details = valid_details()
    details["assertions"][0]["status"] = "unsupported"
    evaluation = {
        "citation_accuracy": {"passed": True},
        "faithfulness": {"passed": False, "details": details},
    }
    initial = {"survey": "original", "metadata": {}}
    agent._extract_report = lambda *args: copy.deepcopy(initial)
    agent._evaluate = AsyncMock(return_value=evaluation)
    attempts = []

    async def repair(self, **kwargs):
        attempts.append(kwargs["report_data"]["survey"])
        body = (
            "original"
            if outcome == "no_progress"
            else f"failed candidate {len(attempts)}"
        )
        return RewriteOutcome(
            True,
            False,
            body,
            None,
            evaluation,
            {"status": "failed"},
            "quality_still_failed" if outcome == "no_progress" else outcome,
        )

    agent._attempt_evidence_rewrite = MethodType(repair, agent)
    report, _ = await agent._prepare_report(
        query="question",
        graph=graph,
        results={"adversarial_review": {"final_draft": "original"}},
        options=RunOptions(quality_policy="strict_v1"),
        search_outcomes=[],
        degradation_codes=[],
        cancellation=None,
    )
    assert len(attempts) == expected and report["survey"] == "original"
    assert not report["delivery"]["publishable"]
    assert len(report["metadata"]["evidence_rewrite_attempts"]) == expected
    if expected == 2:
        assert attempts[1] == "failed candidate 1"


def calibration_fixture():
    """Create explicitly synthetic human-labelled rows for metric contract tests only."""
    rows = []
    for domain in ["vit", "nerf", "fewshot"]:
        for split in ["dev", "holdout"]:
            for i in range(20):
                gold = "supported" if i < 10 else "unsupported"
                rows.append(
                    {
                        "sample_id": f"{domain}-{split}-{i}",
                        "domain": domain,
                        "split": split,
                        "original_source": {"paper_id": f"{domain}-{split}"},
                        "cited_source": {"paper_id": f"{domain}-{split}"},
                        "gold": gold,
                        "annotations": [
                            {
                                "reviewer": reviewer,
                                "human_attested": True,
                                "support_status": gold,
                                "atomic": True,
                                "source_faithful": True,
                            }
                            for reviewer in ["fixture-human-a", "fixture-human-b"]
                        ],
                    }
                )
    return {"samples": rows}


def test_calibration_requires_two_distinct_humans_and_reports_uncertainty():
    dataset = calibration_fixture()
    predictions = {r["sample_id"]: r["gold"] for r in dataset["samples"]}
    result = calibration_gate(dataset, predictions)
    assert result["status"] == "passed"
    assert all(
        g["zero_false_accept_one_sided_95_upper"] > 0.1 for g in result["groups"]
    )
    dataset["samples"][0]["annotations"][1]["reviewer"] = "fixture-human-a"
    assert calibration_gate(dataset, predictions)["status"] == "annotation_pending"


def test_calibration_source_leaks_or_unadjudicated_disagreements_cannot_pass():
    dataset = calibration_fixture()
    predictions = {r["sample_id"]: r["gold"] for r in dataset["samples"]}
    dataset["samples"][0]["annotations"][1]["support_status"] = "unsupported"
    assert calibration_gate(dataset, predictions)["status"] == "annotation_pending"
    dataset["samples"][0]["cited_source"]["paper_id"] = "vit-holdout"
    with pytest.raises(ValueError, match="source_split_leak"):
        calibration_gate(dataset, predictions)


@pytest.mark.asyncio
async def test_strict_external_trace_keeps_full_sources_only_in_local_recorder(
    tmp_path,
):
    from litagent.config import ObservabilityConfig
    from litagent.interaction_store import SqliteReviewRepository
    from litagent.interactive_service import InteractiveRunService
    from litagent.observability.recorder import ArchiveRepository

    repository = SqliteReviewRepository(tmp_path / "db")
    await repository.initialize()
    await repository.create_run(
        "strict-trace", "question", RunOptions(quality_policy="strict_v1")
    )
    captured = []
    service = InteractiveRunService(
        load_config().model_copy(
            update={"observability": ObservabilityConfig(enabled=False)}
        ),
        repository,
        archive_repository=ArchiveRepository(tmp_path / "archive"),
        external_trace_factory=lambda *_: lambda event, data: captured.append(
            (event, data)
        ),
    )
    hook = await service._trace_hook("strict-trace", "question")
    hook(
        "evaluation.batch",
        {
            "phase": "initial",
            "input_sha256": "a" * 64,
            "batch_states": ["completed"],
            "source_quote": "FULL ORIGINAL PDF TEXT",
            "error": "FULL ORIGINAL PDF TEXT",
            "error_type": "TimeoutError",
            "messages": [{"content": "FULL ORIGINAL PDF TEXT"}],
        },
    )
    assert captured[0][1] == {
        "phase": "initial",
        "input_sha256": "a" * 64,
        "batch_states": ["completed"],
        "error_type": "TimeoutError",
        "error": "TimeoutError",
    }
    events = await repository.list_events("strict-trace")
    assert events[-1]["data"]["source_quote"] == "FULL ORIGINAL PDF TEXT"


def test_calibration_development_must_retain_both_positive_and_negative_examples():
    dataset = calibration_fixture()
    for row in dataset["samples"]:
        if row["split"] == "dev":
            row["gold"] = "supported"
            for vote in row["annotations"]:
                vote["support_status"] = "supported"
    assert (
        calibration_gate(
            dataset, {r["sample_id"]: r["gold"] for r in dataset["samples"]}
        )["status"]
        == "annotation_pending"
    )


def test_calibration_domains_follow_source_roster_even_when_manifest_is_reordered():
    import hashlib

    from scripts.build_strict_quality_calibration import build_package

    source = "ViT uses image patches to model visual representations for recognition. The method compares models under explicit training conditions."
    manifest = {
        "papers": [
            {
                "paper_id": f"paper{i}",
                "abstract": source,
                "abstract_sha256": hashlib.sha256(source.encode()).hexdigest(),
            }
            for i in range(20)
        ]
    }
    roster = {
        "cases": [
            {"domain": domain, "relevant_paper_ids": [f"paper{i}" for i in indices]}
            for domain, indices in [
                ("vision_transformer", range(10)),
                ("nerf", range(10, 20)),
            ]
        ]
    }
    previous = {
        "samples": [
            {
                "domain": "fewshot",
                "source_artifact": "offline-fixture.json",
                "original_supports": [
                    {"paper_id": f"fewshot{i}", "supporting_text": source}
                ],
            }
            for i in range(10)
        ]
    }
    first = build_package(manifest, previous, roster)
    manifest["papers"].reverse()
    second = build_package(manifest, previous, roster)
    assert first == second
    assert first["summary"]["domain_split_counts"] == {
        d: {"dev": 20, "holdout": 20} for d in ["vit", "nerf", "fewshot"]
    }
    assert all(
        not row["annotations"] and row["gold"] is None for row in first["samples"]
    )
