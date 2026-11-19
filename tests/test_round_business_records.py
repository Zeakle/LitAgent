"""Verify real phase producers, durable snapshots and workbench consumers."""

import asyncio
import hashlib
import json

import pytest
from test_adversarial import FakeEvidenceSelector, _make_task_with_data
from test_writer_v2 import case  # noqa: F401

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.planner import SurveyPlanner
from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.llm.client import MockLLMClient
from litagent.observability.recorder import ArchiveRepository, RunRecorder


async def record_legacy_rounds(tmp_path, *, failure=False):
    """Run real workers with deterministic model replies and a real recorder."""
    review = {
        "score": 0.5,
        "strengths": [],
        "weaknesses": ["Missing comparison"],
        "issues": [
            {"section": "intro", "issue": "Add comparison", "severity": "major"}
        ],
        "missing_coverage": [],
        "verdict": "revise",
        "evidence_compliance": {
            "passed": True,
            "unsupported_claims": [],
            "unknown_evidence_ids": [],
        },
    }
    accepted = {**review, "score": 0.9, "verdict": "accept", "issues": []}

    class OfflineLLM(MockLLMClient):
        """Fail only the revision call when testing the retained draft path."""

        async def chat(self, messages, **kwargs):
            if failure and self._call_count == 2:
                raise RuntimeError("revision_transport_failed")
            return await super().chat(messages, **kwargs)

    llm = OfflineLLM(
        responses=[
            "Initial draft [E:p1:claim:0].\n",
            json.dumps(review),
            "Revised draft [E:p1:claim:0].\nComparison added.",
            json.dumps(accepted),
        ]
    )
    recorder = RunRecorder(
        "round-inspection", "Compare approaches", ArchiveRepository(tmp_path)
    )
    worker = AdversarialReviewWorker(
        llm,
        SynthesisWorker(llm, evidence_selector=FakeEvidenceSelector()),
        ReviewerWorker(llm),
        trace_hook=recorder,
    )
    task = _make_task_with_data()
    task.input_data["query"] = "Compare approaches"
    result = await worker.execute(task)
    task.input_data["query"] = "mutated after execution"
    artifact = recorder.finalize(
        {"survey": result["final_draft"], "delivery": {"status": "ready"}}
    )
    return artifact, result


@pytest.mark.asyncio
async def test_real_legacy_rounds_preserve_full_io_and_exact_previous_draft(tmp_path):
    artifact, result = await record_legacy_rounds(tmp_path)
    nodes = artifact["nodes"]
    s1 = nodes["subspan:adversarial_review:synthesis:r1"]
    r1 = nodes["subspan:adversarial_review:reviewer:r1"]
    s2 = nodes["subspan:adversarial_review:synthesis:r2"]
    r2 = nodes["subspan:adversarial_review:reviewer:r2"]
    assert s1["input"]["input"]["query"] == "Compare approaches"
    assert "Initial draft" in s1["output"]["draft"]
    assert s2["input"]["input"]["previous_draft"] == r1["input"]["input"]["draft"]
    assert s2["input"]["input"]["review"] == r1["output"]
    assert (
        r2["input"]["input"]["draft"] == s2["output"]["draft"] == result["final_draft"]
    )
    assert (
        s2["output"]["report_sha256"]
        == hashlib.sha256(result["final_draft"].encode()).hexdigest()
    )
    assert s2["output"]["promoted"] is True
    restored = ArchiveRepository(tmp_path).get("round-inspection")
    assert restored["nodes"] == nodes


@pytest.mark.asyncio
async def test_failed_revision_keeps_input_without_fabricating_success_output(tmp_path):
    artifact, result = await record_legacy_rounds(tmp_path, failure=True)
    node = artifact["nodes"]["subspan:adversarial_review:synthesis:r2"]
    assert node["status"] == "failed"
    assert node["error"] == "revision_transport_failed"
    assert node["output"] is None
    assert node["input"]["input"]["previous_draft"] == result["final_draft"]
    assert "subspan:adversarial_review:reviewer:r2" not in artifact["nodes"]


@pytest.mark.asyncio
async def test_cancelled_round_retains_source_and_never_returns_partial_draft(tmp_path):
    recorder = RunRecorder("cancel", "query", ArchiveRepository(tmp_path))
    llm = MockLLMClient()
    worker = AdversarialReviewWorker(
        llm, SynthesisWorker(llm), ReviewerWorker(llm), trace_hook=recorder
    )
    with pytest.raises(asyncio.CancelledError):
        async with worker._sub_span(
            "write", "synthesis", 2, input_data={"previous_draft": "old"}
        ) as span:
            span["output"] = {"draft": "partial"}
            raise asyncio.CancelledError
    node = recorder.snapshot()["nodes"]["subspan:write:synthesis:r2"]
    assert node["status"] == "cancelled" and node["output"] is None
    assert node["input"]["input"]["previous_draft"] == "old"


@pytest.mark.asyncio
async def test_query_rewrite_records_original_query_and_host_selected_fields(tmp_path):
    recorder = RunRecorder(
        "query", "Compare Alpha and Beta methods", ArchiveRepository(tmp_path)
    )
    planner = SurveyPlanner(
        llm=MockLLMClient(
            responses=[
                json.dumps(
                    {
                        "sub_queries": ["Alpha original paper", "Beta original paper"],
                        "named_methods": ["Alpha", "Beta", "Invented"],
                        "coverage_constraints": ["Compare", "Invented constraint"],
                    }
                )
            ]
        ),
        trace_hook=recorder,
    )
    await planner.plan("Compare Alpha and Beta methods")
    node = recorder.snapshot()["nodes"]["subspan:query_decomposition"]
    assert node["input"]["input"]["query"] == "Compare Alpha and Beta methods"
    assert node["output"]["sub_queries"] == [
        "Alpha original paper",
        "Beta original paper",
    ]
    assert node["output"]["named_methods"] == ["Alpha", "Beta"]
    assert node["output"]["coverage_constraints"] == ["Compare"]


@pytest.mark.asyncio
async def test_native_editorial_pipeline_archives_each_exact_draft(
    case, tmp_path, monkeypatch
):
    from test_separate_review_pipeline import SeparateGateway, execute

    report = await execute(
        case, tmp_path, monkeypatch, SeparateGateway(case, reviews=1)
    )
    artifact = ArchiveRepository(tmp_path / "archive").list()[0]
    nodes = artifact["nodes"]
    s1, r1, s2, r2 = [
        nodes[f"subspan:adversarial_review:{role}:r{number}"]
        for role, number in (
            ("synthesis", 1),
            ("reviewer", 1),
            ("synthesis", 2),
            ("reviewer", 2),
        )
    ]
    assert (
        r1["input"]["input"]["upstream_results"]["synthesis"]["draft"]
        == s1["output"]["draft"]
    )
    assert (
        r1["output"]["source_report_sha256"]
        == hashlib.sha256(s1["output"]["draft"].encode()).hexdigest()
    )
    assert s2["input"]["input"]["previous_draft"] == s1["output"]["draft"]
    assert (
        r2["input"]["input"]["upstream_results"]["synthesis"]["draft"]
        == s2["output"]["draft"]
    )
    assert s1["output"]["draft"] != s2["output"]["draft"]
    evaluation = nodes["subspan:evaluation.initial"]
    assert evaluation["input"]["input"]["survey"] == s2["output"]["draft"]
    assert (
        evaluation["input"]["input"]["report_sha256"]
        == r2["output"]["source_report_sha256"]
    )
    assert report["delivery"]["publishable"] is True
