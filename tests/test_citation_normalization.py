"""Verify citation formatting repairs without expanding evidence authority."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from litagent.evidence import normalize_evidence_citations


@pytest.mark.parametrize("separator", [",", ";", ", ", ";\n\t"])
def test_compounds_preserve_prose_order_duplicates_and_offsets(separator):
    marker = f"[E:e2{separator}E:e1{separator}E:e2]"
    text = f"Before {marker} after."
    result = normalize_evidence_citations(text, {"e1", "e2"})
    assert result.text == "Before [E:e2] [E:e1] [E:e2] after."
    assert result.changes[0].start == 7
    assert text[result.changes[0].start : result.changes[0].end] == marker
    assert result.changes[0].original == marker
    assert not result.rejected
    again = normalize_evidence_citations(result.text, {"e1", "e2"})
    assert again.text == result.text
    assert not again.changes


@pytest.mark.parametrize(
    ("marker", "reason"),
    [
        ("[E:e1, E:unknown]", "unknown_evidence_ids"),
        ("[E:e1, e2]", "unsupported_citation_syntax"),
        ("[E:e1", "unclosed_citation"),
        ("[E:e1, [E:e2; E:e1]]", "nested_citation"),
        ("[outer [E:e1; E:e2]]", "nested_citation"),
        ("[E:e1,, E:e2]", "unsupported_citation_syntax"),
    ],
)
def test_rejected_markers_are_preserved_atomically(marker, reason):
    result = normalize_evidence_citations(marker, {"e1", "e2"})
    assert result.text == marker
    assert not result.changes
    assert result.rejected[0].reason_code == reason


def test_rejection_does_not_prevent_independent_repair():
    text = "A [E:e1, E:unknown]. B [E:e1; E:e2]."
    result = normalize_evidence_citations(text, {"e1", "e2"})
    assert result.text == "A [E:e1, E:unknown]. B [E:e1] [E:e2]."
    assert len(result.rejected) == len(result.changes) == 1


def test_long_ids_are_matched_exactly_and_standard_markers_are_untouched():
    eid = "arxiv:2003.08934:section-2:claim:0123456789ab"
    text = f"A [E:{eid}, E:e2]. B [E:{eid[:-1]}, E:e2]. [E:unknown]"
    result = normalize_evidence_citations(text, {eid, "e2"})
    assert result.text == f"A [E:{eid}] [E:e2]. B [E:{eid[:-1]}, E:e2]. [E:unknown]"
    assert len(result.changes) == len(result.rejected) == 1


@pytest.mark.asyncio
async def test_malformed_citation_fails_even_at_numeric_threshold():
    from litagent.eval.citation import CitationEvaluator

    result = await CitationEvaluator(AsyncMock()).evaluate(
        "[E:e1] [E:e2] [E:e3] [E:e4] [E:e1, E:e2]",
        {"evidence": {f"e{i}": {} for i in range(1, 5)}},
    )
    assert result.score == 0.8
    assert result.passed is False


@pytest.mark.asyncio
async def test_legacy_audit_cannot_pass_malformed_markers():
    from litagent.eval.citation import CitationEvaluator
    from litagent.llm.client import MockLLMClient

    result = await CitationEvaluator(MockLLMClient(['{"cited": []}'])).evaluate(
        "Claim [E:e1, E:e2]", {"papers": [{"title": "Paper"}]}
    )
    assert result.score == 1.0
    assert result.passed is False
    assert result.details["malformed_evidence_refs"]


@pytest.mark.parametrize("text", ["", "Plain text.\nSame text!", "[ordinary bracket]"])
def test_non_citation_text_is_unchanged(text):
    result = normalize_evidence_citations(text, set())
    assert result.text == text
    assert not result.changes and not result.rejected


def test_rewrite_validator_rejects_malformed_markers():
    from litagent.config import AgentConfig, AppConfig, LoggingConfig
    from litagent.runner import LitAgent

    result = LitAgent(
        AppConfig(agent=AgentConfig(), logging=LoggingConfig())
    )._validate_rewrite_candidate(
        original="# Title\n\n## Methods\nClaim [E:e1].",
        candidate="# Title\n\n## Methods\nChanged claim [E:e1] [E:e1, E:e2].",
        evidence_ledger={"e1": {}, "e2": {}},
    )
    assert not result.accepted
    assert "malformed_evidence_refs" in result.reason_codes


def _selection():
    """Provide a selected subset smaller than the extraction ledger."""
    from litagent.context.evidence_selector import EvidenceSelection

    return EvidenceSelection(
        candidate_count=3,
        selected_items={
            eid: {"evidence_id": eid, "paper_id": "p1", "text": eid}
            for eid in ("e1", "e2")
        },
        section_evidence_ids={"methods": ["e1", "e2"]},
        method_by_section={"methods": "lexical_fallback"},
        omitted_count=1,
        estimated_tokens=20,
    )


@pytest.mark.asyncio
async def test_initial_draft_uses_selected_ids_and_preserves_raw_trace():
    from litagent.agents.synthesis import SynthesisWorker
    from litagent.llm.client import MockLLMClient
    from litagent.orchestrator.task_graph import SubTask

    raw = "A [E:e1, E:e2]. B [E:e1, E:e3]."
    events = []
    worker = SynthesisWorker(
        MockLLMClient([raw]),
        evidence_selector=SimpleNamespace(select=AsyncMock(return_value=_selection())),
        trace_hook=lambda event, data: events.append((event, data)),
    )
    result = await worker.execute(
        SubTask(
            task_id="synthesis",
            agent_type="synthesis",
            description="test",
            input_data={
                "upstream_results": {
                    "extract": [
                        {
                            "paper_id": "p1",
                            "title": "Paper",
                            "evidence_items": [
                                {"evidence_id": eid, "text": eid}
                                for eid in ("e1", "e2", "e3")
                            ],
                        }
                    ]
                }
            },
        )
    )
    assert result["draft"] == "A [E:e1] [E:e2]. B [E:e1, E:e3]."
    assert events[0][1]["input"]["text"] == raw
    assert (
        events[-1][1]["output"]["rejected"][0]["reason_code"] == "unknown_evidence_ids"
    )


@pytest.mark.asyncio
async def test_each_revision_is_normalized_before_review():
    from litagent.agents.adversarial import AdversarialReviewWorker
    from litagent.llm.client import MockLLMClient
    from litagent.orchestrator.task_graph import SubTask

    selection = _selection()
    synthesis = SimpleNamespace(
        execute=AsyncMock(
            return_value={
                "draft": "Original [E:e1]",
                "evidence_selection": selection.to_dict(),
            }
        ),
        revise=Mock(return_value=[{"role": "user", "content": "revise"}]),
    )
    revise = {
        "score": 0.5,
        "verdict": "revise",
        "evidence_compliance": {"passed": False},
    }
    accept = {
        "score": 0.9,
        "verdict": "accept",
        "evidence_compliance": {"passed": True},
    }
    reviewer = SimpleNamespace(
        execute=AsyncMock(return_value=revise),
        review_revision=AsyncMock(side_effect=[revise, accept]),
    )
    worker = AdversarialReviewWorker(
        MockLLMClient(["First [E:e1, E:e2]", "Second [E:e2; E:e1]"]),
        synthesis,
        reviewer,
    )
    result = await worker.execute(
        SubTask(task_id="adv", agent_type="adversarial_review", description="test")
    )
    assert [call.args[0] for call in reviewer.review_revision.await_args_list] == [
        "First [E:e1] [E:e2]",
        "Second [E:e2] [E:e1]",
    ]
    assert result["accepted"] is True
    assert result["total_rounds"] == 3


def test_trace_pairs_are_unique_and_recorder_keeps_raw_and_normalized_text(tmp_path):
    from litagent.observability.citations import normalize_citations_traced
    from litagent.observability.recorder import ArchiveRepository, RunRecorder

    recorder = RunRecorder("test", "query", ArchiveRepository(tmp_path))
    for _ in range(2):
        normalize_citations_traced(
            "A [E:e1, E:e2]",
            {"e1", "e2"},
            trace_hook=recorder,
            parent_task_id="adv",
            stage="revision",
            round_num=2,
        )
    artifact = recorder.snapshot()
    nodes = list(artifact["nodes"].values())
    assert len(nodes) == 2
    assert all(n["input"]["input"]["text"] == "A [E:e1, E:e2]" for n in nodes)
    assert all(n["output"]["text"] == "A [E:e1] [E:e2]" for n in nodes)
    assert all(n["status"] == "completed" for n in nodes)
    assert not any(event["event"].startswith("llm.") for event in artifact["events"])
    assert artifact["usage"]["total_tokens"] == 0


def test_normalization_does_not_hide_unsupported_or_unknown_claims():
    from litagent.agents.reviewer import ReviewerWorker
    from litagent.llm.client import MockLLMClient

    normalized = normalize_evidence_citations(
        "Claim [E:e1, E:e2]. [E:unknown]", {"e1", "e2"}
    )
    unsupported = [{"claim": "Claim", "reason": "Evidence does not support it"}]
    review = ReviewerWorker(MockLLMClient())._enforce_evidence_contract(
        normalized.text,
        _selection(),
        {
            "score": 0.9,
            "verdict": "accept",
            "evidence_compliance": {
                "passed": True,
                "unsupported_claims": unsupported,
                "unknown_evidence_ids": [],
            },
        },
    )
    preserved = review["evidence_compliance"]["unsupported_claims"]
    assert [
        {"claim": item["claim"], "reason": item["reason"]} for item in preserved
    ] == unsupported
    assert preserved[0]["existing_evidence_ids"] == ["e1", "e2", "unknown"]
    assert preserved[0]["source_quote"] == normalized.text
    assert review["evidence_compliance"]["unknown_evidence_ids"] == ["unknown"]
    assert review["evidence_compliance"]["passed"] is False
    assert review["verdict"] == "revise"


def test_trace_failure_does_not_change_output_or_generation_usage(tmp_path):
    from litagent.benchmark.survey_metrics import _event_count
    from litagent.observability.citations import normalize_citations_traced
    from litagent.observability.recorder import ArchiveRepository, RunRecorder

    recorder = RunRecorder("test", "query", ArchiveRepository(tmp_path))
    raw = "Claim [E:e1, E:e2]"
    recorder(
        "llm.complete",
        {
            "operation_id": "call-1",
            "content": raw,
            "prompt_tokens": 7,
            "completion_tokens": 3,
            "total_tokens": 10,
        },
    )
    result = normalize_citations_traced(
        raw, {"e1", "e2"}, trace_hook=recorder, parent_task_id="adv", stage="revision"
    )
    artifact = recorder.snapshot()
    assert artifact["events"][0]["data"]["content"] == raw
    assert artifact["usage"]["total_tokens"] == 10
    assert _event_count(artifact, "llm.complete", "llm.failed") == 1
    broken_hook = Mock(side_effect=RuntimeError("trace unavailable"))
    assert (
        normalize_citations_traced(
            raw,
            {"e1", "e2"},
            trace_hook=broken_hook,
            parent_task_id="adv",
            stage="revision",
        )
        == result
    )


def test_runtime_fingerprint_includes_serialization_behavior(monkeypatch):
    from litagent.benchmark import survey_runner
    from litagent.benchmark.survey_metrics import METRICS_VERSION

    captured = []
    monkeypatch.setattr(
        survey_runner,
        "_hash_files",
        lambda paths: captured.extend(paths) or "fingerprint",
    )
    assert survey_runner.runtime_prompt_source_fingerprint() == "fingerprint"
    relative = {
        path.relative_to(
            survey_runner.Path(survey_runner.__file__).resolve().parents[1]
        ).as_posix()
        for path in captured
    }
    assert {
        "evidence.py",
        "runner.py",
        "eval/citation.py",
        "observability/citations.py",
    } <= relative
    assert METRICS_VERSION == "survey-metrics-v4-paired-downstream"
