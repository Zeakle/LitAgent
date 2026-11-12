"""Bind section permissions and rejected candidates without changing evidence."""

import copy
import hashlib
import json
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator
from test_observability import _tracer_with_mock
from test_strict_quality import draft, to_native

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.context.evidence_selector import EvidenceSelection
from litagent.llm.client import LLMResponse
from litagent.observability.recorder import (
    ArchiveRepository,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    GroundedWritingError,
    build_fact_cards,
    grounded_messages,
    render_grounded_report,
    render_grounded_report_traced,
)
from litagent.review_structure import assemble_methodology, structure_sections
from litagent.safety.budget import CostBudget


def materials(structure="academic_v2", language="en"):
    """Use distinct evidence from one paper to reject paper-level scope widening."""
    ledger = {
        "a": {
            "evidence_id": "a",
            "paper_id": "evit",
            "text": "EViT improves speed.",
            "supporting_text": "EViT improves speed.",
            "chunk_key": "abstract",
        },
        "b": {
            "evidence_id": "b",
            "paper_id": "evit",
            "text": "EViT reports a dataset-specific result.",
            "supporting_text": "EViT reports a dataset-specific result.",
            "chunk_key": "results",
        },
    }
    cards = build_fact_cards(ledger)
    aid, bid = list(cards)
    scope = {
        k: ["a"]
        for k, _ in structure_sections(structure, language)
        if k != "run_methodology"
    }
    scope["methods"] = ["b"]
    scope["experiments"] = ["a", "b"]
    selection = EvidenceSelection(
        2, ledger, scope, {k: "lexical_fallback" for k in scope}, 0, 1
    )
    report = draft({aid: cards[aid]}, language)
    next(s for s in report["sections"] if s["section_key"] == "methods")["blocks"][0][
        "sentences"
    ][0]["fact_card_ids"] = [bid]
    return cards, selection, report


def unit(report, section):
    """Select the synthetic atomic unit by its template key."""
    return next(s for s in report["sections"] if s["section_key"] == section)["blocks"][
        0
    ]["sentences"][0]


@pytest.mark.parametrize(
    "structure", ["academic_v1", "academic_v2", "method_transparent_v1"]
)
@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_dynamic_schema_reverse_permissions_and_host_share_exact_scope(
    structure, language
):
    cards, selection, report = materials(structure, language)
    before = copy.deepcopy((cards, selection.to_dict(), report))
    packet = json.loads(
        grounded_messages(
            cards, structure, language, "Compare EViT", evidence_selection=selection
        )[1]["content"]
    )
    validator = Draft202012Validator(packet["grounded_report_schema"])
    validator.check_schema(validator.schema)
    validator.validate(report)
    assert (
        render_grounded_report(
            json.dumps(report), cards, structure, language, evidence_selection=selection
        ).count("[E:")
        == 7
    )
    aid, bid = list(cards)
    assert "methods" not in packet["cards"][aid]["allowed_section_keys"]
    assert packet["cards"][bid]["allowed_section_keys"] == ["methods", "experiments"]
    assert "finding_candidate" not in packet["cards"][aid]
    assert {s["source_quote"] for s in packet["sources"].values()} == {
        c["source_quote"] for c in cards.values()
    }
    for section in packet["section_contract"]:
        key = section["section_key"]
        assert set(section["allowed_fact_card_ids"]) == {
            cid
            for cid, c in packet["cards"].items()
            if key in c["allowed_section_keys"]
        }
    unit(report, "introduction")["fact_card_ids"] = [bid]
    assert list(validator.iter_errors(report))
    with pytest.raises(
        GroundedWritingError, match="strict_writing_out_of_scope_fact_card"
    ):
        render_grounded_report(
            json.dumps(report), cards, structure, language, evidence_selection=selection
        )
    assert cards == before[0] and selection.to_dict() == before[1]


@pytest.mark.parametrize("kind", ["scientific", "organization", "gap"])
def test_complete_diagnostics_bind_all_bad_units_to_unchanged_candidate(kind):
    cards, selection, report = materials()
    aid, bid = list(cards)
    unit(report, "introduction").update(kind=kind, fact_card_ids=[bid])
    unit(report, "taxonomy")["fact_card_ids"] = [aid, bid]
    unit(report, "open_problems")["fact_card_ids"] = [bid]
    raw = json.dumps(report)
    with pytest.raises(GroundedWritingError) as caught:
        render_grounded_report(
            raw, cards, "academic_v2", "en", evidence_selection=selection
        )
    err = caught.value
    assert err.candidate_sha256 == hashlib.sha256(raw.encode()).hexdigest()
    assert [d["location"] for d in err.diagnostics] == [
        "introduction.blocks.0.sentences.0",
        "taxonomy.blocks.0.sentences.0",
        "open_problems.blocks.0.sentences.0",
    ]
    assert all(
        d["invalid_fact_card_ids"] == [bid] and d["allowed_fact_card_ids"] == [aid]
        for d in err.diagnostics
    )
    assert err.diagnostics[1]["existing_evidence_ids"] == ["a", "b"]
    assert all(
        d["candidate_text"] == unit(report, d["section_key"])["text"]
        for d in err.diagnostics
    )
    assert json.dumps(report) == raw


@pytest.mark.parametrize(
    "mode,code",
    [
        ("unknown", "strict_writing_unknown_fact_card"),
        ("duplicate", "strict_writing_duplicate_fact_card"),
        ("uncited", "strict_writing_uncited_science"),
    ],
)
def test_schema_and_host_reject_unknown_duplicate_and_uncited_units(mode, code):
    cards, selection, report = materials()
    aid = next(iter(cards))
    unit(report, "experiments")["fact_card_ids"] = {
        "unknown": ["f:unknown"],
        "duplicate": [aid, aid],
        "uncited": [],
    }[mode]
    schema = json.loads(
        grounded_messages(
            cards, "academic_v2", "en", "EViT", evidence_selection=selection
        )[1]["content"]
    )["grounded_report_schema"]
    assert list(Draft202012Validator(schema).iter_errors(report))
    with pytest.raises(GroundedWritingError, match=code) as caught:
        render_grounded_report(
            json.dumps(report), cards, "academic_v2", "en", evidence_selection=selection
        )
    assert caught.value.diagnostics[0]["location"] == "experiments.blocks.0.sentences.0"


@pytest.mark.parametrize("missing", [False, True])
def test_empty_or_missing_chapter_is_gap_only_without_borrowing_other_sources(missing):
    cards, selection, report = materials()
    if missing:
        selection.section_evidence_ids.pop("methods")
    else:
        selection.section_evidence_ids["methods"] = []
    packet = json.loads(
        grounded_messages(
            cards, "academic_v2", "en", "EViT", evidence_selection=selection
        )[1]["content"]
    )
    validator = Draft202012Validator(packet["grounded_report_schema"])
    assert list(validator.iter_errors(report))
    with pytest.raises(GroundedWritingError):
        render_grounded_report(
            json.dumps(report), cards, "academic_v2", "en", evidence_selection=selection
        )
    unit(report, "methods").update(
        kind="gap",
        text="This run lacks selected evidence for the required mechanism.",
        fact_card_ids=[],
    )
    validator.validate(report)
    assert render_grounded_report(
        json.dumps(report), cards, "academic_v2", "en", evidence_selection=selection
    )
    unit(report, "methods")["kind"] = "scientific"
    assert list(validator.iter_errors(report))
    with pytest.raises(GroundedWritingError, match="strict_writing_uncited_science"):
        render_grounded_report(
            json.dumps(report), cards, "academic_v2", "en", evidence_selection=selection
        )


@pytest.mark.parametrize("scope", [None, [], {"methods": "a"}, {"methods": [1]}])
def test_invalid_explicit_scope_never_grants_global_permissions(scope):
    cards, _, report = materials()
    for call in [
        lambda: grounded_messages(
            cards,
            "academic_v2",
            "en",
            "EViT",
            evidence_selection={"section_evidence_ids": scope},
        ),
        lambda: render_grounded_report(
            json.dumps(report),
            cards,
            "academic_v2",
            "en",
            evidence_selection={"section_evidence_ids": scope},
        ),
    ]:
        with pytest.raises(ValueError, match="strict_writing_scope_contract_invalid"):
            call()


@pytest.mark.parametrize("mode", ["order", "extra"])
def test_schema_cannot_change_template_order_or_add_chapters(mode):
    cards, selection, report = materials()
    packet = json.loads(
        grounded_messages(
            cards, "academic_v2", "en", "EViT", evidence_selection=selection
        )[1]["content"]
    )
    if mode == "order":
        report["sections"][1:3] = reversed(report["sections"][1:3])
    else:
        report["sections"].append(copy.deepcopy(report["sections"][-1]))
    assert list(
        Draft202012Validator(packet["grounded_report_schema"]).iter_errors(report)
    )
    with pytest.raises(ValueError, match="strict_writing_section_contract_invalid"):
        render_grounded_report(
            json.dumps(report), cards, "academic_v2", "en", evidence_selection=selection
        )


def test_diagnostics_persist_in_worker_and_archive_but_trace_is_metadata_only(tmp_path):
    cards, selection, report = materials()
    unit(report, "introduction")["fact_card_ids"] = [list(cards)[1]]
    repo = ArchiveRepository(tmp_path)
    recorder = RunRecorder("scope", "EViT", repo)
    tracer, log = _tracer_with_mock()
    public = []

    def sink(event, data):
        public.append((event, data))
        tracer(event, data)

    redacted = RedactingTraceHook(sink, "metadata_only")

    def hook(event, data):
        recorder(event, data)
        redacted(event, data)

    hook("survey.start", {"query": "EViT"})
    hook("worker.start", {"task_id": "adv", "agent_type": "adversarial_review"})
    with pytest.raises(GroundedWritingError):
        render_grounded_report_traced(
            json.dumps(report),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            trace_hook=hook,
            parent_task_id="adv",
            stage="initial",
        )
    hook(
        "worker.failed",
        {"task_id": "adv", "error": "strict_writing_out_of_scope_fact_card"},
    )
    stored = recorder.finalize(error="offline failure", terminal_status="failed")
    local = stored["nodes"]["worker:adv"]["metadata"]["writing_validation_failures"][0]
    assert (
        local["diagnostics"][0]["candidate_text"]
        == unit(report, "introduction")["text"]
    )
    assert json.loads(repo.path_for("scope").read_text())["nodes"]["worker:adv"][
        "metadata"
    ]["writing_validation_failures"] == [local]
    safe = next(d for e, d in public if e == "writing.validation.failed")
    assert (
        safe["candidate_sha256"] == local["candidate_sha256"]
        and safe["diagnostic_count"] == 1
    )
    assert "diagnostics" not in safe and "candidate_text" not in str(safe)
    metadata = next(
        d["metadata"]["writing_validation"]
        for e, d in log
        if e == "update" and "writing_validation" in d.get("metadata", {})
    )
    assert metadata["diagnostic_count"] == 1 and "diagnostics" not in metadata


def test_broken_trace_hook_does_not_mask_validation_or_create_success():
    cards, selection, report = materials()
    unit(report, "introduction")["fact_card_ids"] = [list(cards)[1]]

    def broken(*args):
        raise RuntimeError("offline trace unavailable")

    with pytest.raises(
        GroundedWritingError, match="strict_writing_out_of_scope_fact_card"
    ):
        render_grounded_report_traced(
            json.dumps(report),
            cards,
            "academic_v2",
            "en",
            evidence_selection=selection,
            trace_hook=broken,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_actual_initial_generation_emits_bound_diagnostic_without_retry(policy):
    cards, selection, report = materials()
    unit(report, "introduction")["fact_card_ids"] = [list(cards)[1]]
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(json.dumps(to_native(report, cards)))
    selector = AsyncMock()
    selector.select.return_value = selection
    events = []
    worker = SynthesisWorker(
        llm, evidence_selector=selector, trace_hook=lambda e, d: events.append((e, d))
    )
    with pytest.raises(GroundedWritingError):
        await worker.execute(
            SubTask(
                "adv",
                "offline",
                "synthesis",
                input_data={
                    "query": "EViT",
                    "quality_policy": policy,
                    "review_structure": "academic_v2",
                    "report_language": "en",
                },
            )
        )
    llm.chat.assert_awaited_once()
    assert llm.chat.await_args.kwargs["response_format"]["type"] == "json_schema"
    event = next(d for e, d in events if e == "writing.validation.failed")
    assert event["task_id"] == "adv" and event["phase"] == "initial"
    assert event["diagnostic_count"] == 1
    assert (
        event["candidate_sha256"]
        == hashlib.sha256(json.dumps(to_native(report, cards)).encode()).hexdigest()
    )


@pytest.mark.asyncio
async def test_actual_revision_rejects_candidate_preserves_original_and_stops_budget():
    cards, selection, report = materials()
    original = assemble_methodology(
        render_grounded_report(
            json.dumps(report), cards, "academic_v2", "en", evidence_selection=selection
        ),
        "academic_v2",
        "en",
        {},
    )
    revised = copy.deepcopy(report)
    unit(revised, "introduction")["fact_card_ids"] = [list(cards)[1]]
    llm = AsyncMock()
    llm.chat.return_value = LLMResponse(json.dumps(to_native(revised, cards)))
    budget = CostBudget(500000)
    budget.configure_strict(halt_on_unknown_usage=True)
    llm._cost_budget = budget
    synthesis = SynthesisWorker(llm)
    synthesis.execute = AsyncMock(
        return_value={
            "draft": original,
            "quality_policy": "strict_auto_v1",
            "fact_cards": cards,
            "coverage_requirements": [
                {"requirement_id": "question", "kind": "question", "text": "EViT"}
            ],
            "evidence_selection": selection.to_dict(),
        }
    )
    reviewer = AsyncMock()
    reviewer.execute.return_value = {
        "score": 0.5,
        "verdict": "revise",
        "bound_feedback": [
            {
                "source_quote": original,
                "existing_evidence_ids": ["a", "b"],
                "issue": "Located review",
            }
        ],
        "evidence_compliance": {"passed": False},
        "language_compliance": {"passed": True},
    }
    events = []
    worker = AdversarialReviewWorker(
        llm,
        synthesis,
        reviewer,
        max_rounds=2,
        trace_hook=lambda e, d: events.append((e, d)),
    )
    result = await worker.execute(
        SubTask(
            "adv",
            "offline",
            "adversarial_review",
            input_data={
                "query": "EViT",
                "quality_policy": "strict_auto_v1",
                "review_structure": "academic_v2",
                "report_language": "en",
            },
        )
    )
    assert (
        result["final_draft"] == original
        and budget.stop_reason == "strict_worker_failed"
    )
    llm.chat.assert_awaited_once()
    reviewer.review_revision.assert_not_awaited()
    assert llm.chat.await_args.kwargs["response_format"]["type"] == "json_schema"
    event = next(d for e, d in events if e == "writing.validation.failed")
    assert event["phase"] == "revision" and event["diagnostic_count"] == 1
