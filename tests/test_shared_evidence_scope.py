"""Global source de-duplication must not deny legitimate strict chapter reuse."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from litagent.agents.synthesis import SynthesisWorker
from litagent.config import load_config
from litagent.context.budget import BudgetManager
from litagent.context.evidence_selector import (
    EvidenceSelection,
    EvidenceSelector,
    SectionSpec,
)
from litagent.llm.client import LLMResponse
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import (
    build_fact_cards,
    digest,
    grounded_messages,
    prepare_grounded_request,
)
from scripts.run_frozen_quality_repair import _frozen_repair_scope

SECTIONS = (
    SectionSpec("introduction", "Representation", "network"),
    SectionSpec("methods", "Representation", "network"),
)


def ledger():
    return {
        k: {
            "evidence_id": k,
            "paper_id": p,
            "paper_title": "Representation",
            "text": text,
            "supporting_text": text,
        }
        for k, p, text in [
            (
                "a",
                "foundation",
                "A fully connected network maps coordinates to density and radiance.",
            ),
            (
                "b",
                "variant",
                "The variant improves performance on the measured scenes.",
            ),
        ]
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("max_items", [1, 2])
async def test_selected_source_can_enter_both_chapters_without_double_counting(
    max_items,
):
    selector = EvidenceSelector(
        None, BudgetManager(), top_k_per_section=1, max_items=max_items, per_paper_cap=1
    )
    result = await selector.select(
        "network", ledger(), SECTIONS, max_tokens=16000, reuse_selected=True
    )
    assert result.section_evidence_ids == {"introduction": ["a"], "methods": ["a"]}
    assert len(result.selected_items) == 1
    assert any(
        d["reason_code"] == "reused_selected" and d["section"] == "methods"
        for d in result.ranking_decisions
    )
    assert (
        EvidenceSelection.from_dict(result.to_dict()).section_evidence_ids
        == result.section_evidence_ids
    )


@pytest.mark.asyncio
async def test_legacy_keeps_distinct_section_allocation():
    selector = EvidenceSelector(None, BudgetManager(), top_k_per_section=1, max_items=2)
    result = await selector.select("network", ledger(), SECTIONS, max_tokens=16000)
    assert result.section_evidence_ids == {"introduction": ["a"], "methods": ["b"]}


@pytest.mark.asyncio
async def test_reuse_is_charged_to_xml_scope_budget():
    selector = EvidenceSelector(None, BudgetManager(), top_k_per_section=1, max_items=1)
    first = await selector.select(
        "network", ledger(), SECTIONS[:1], max_tokens=16000, reuse_selected=True
    )
    result = await selector.select(
        "network",
        ledger(),
        SECTIONS,
        max_tokens=first.estimated_tokens,
        reuse_selected=True,
    )
    assert result.section_evidence_ids["introduction"] == ["a"]
    assert result.section_evidence_ids["methods"] == []
    assert any(
        d["reason_code"] == "token_budget"
        and d["section"] == "methods"
        and d["evidence_id"] == "a"
        for d in result.ranking_decisions
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["strict_v1", "strict_auto_v1"])
async def test_actual_synthesis_enables_strict_reuse(policy):
    items = ledger()
    selection = EvidenceSelection(
        2,
        items,
        {
            "introduction": ["a"],
            "taxonomy": ["a"],
            "methods": ["a"],
            "experiments": ["b"],
            "open_problems": ["b"],
        },
        {
            k: "lexical_fallback"
            for k in [
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
            ]
        },
        0,
        100,
    )
    selector = SimpleNamespace(select=AsyncMock(return_value=selection))
    llm = SimpleNamespace(
        chat=AsyncMock(return_value=LLMResponse("{}", usage={"usage_available": True}))
    )
    worker = SynthesisWorker(llm, evidence_selector=selector)
    worker._get_extractions = lambda _: []
    task = SubTask(
        "synthesis",
        "write",
        "synthesis",
        input_data={
            "query": "network",
            "quality_policy": policy,
            "review_structure": "academic_v2",
            "report_language": "en",
        },
    )
    with pytest.raises(ValueError):
        await worker.execute(task)
    assert selector.select.await_args.kwargs["reuse_selected"] is True
    llm.chat.assert_awaited_once()


def frozen():
    items = ledger()
    cards = build_fact_cards(items)
    scope = {
        "valid": True,
        "candidate_count": 2,
        "selected_count": 2,
        "section_evidence_ids": {
            "introduction": ["a"],
            "taxonomy": ["a"],
            "methods": ["b"],
            "experiments": ["b"],
            "open_problems": ["b"],
            "abstract": ["a", "b"],
            "conclusion": ["a", "b"],
        },
        "method_by_section": {
            k: (
                "selected_union"
                if k in {"abstract", "conclusion"}
                else "lexical_fallback"
            )
            for k in [
                "introduction",
                "taxonomy",
                "methods",
                "experiments",
                "open_problems",
                "abstract",
                "conclusion",
            ]
        },
        "omitted_count": 0,
        "estimated_tokens": 100,
    }
    decisions = [
        {
            "section": key,
            "evidence_id": eid,
            "rank": i + 1,
            "reason_code": (
                "duplicate_selected" if eid == "a" and key == "methods" else "selected"
            ),
        }
        for key in [
            "introduction",
            "taxonomy",
            "methods",
            "experiments",
            "open_problems",
        ]
        for i, eid in enumerate(["a", "b"])
    ]
    return {
        "query": "network",
        "report": {
            "metadata": {"fact_cards": cards, "evidence_snapshot_sha256": digest(cards)}
        },
        "nodes": {
            "write": {
                "kind": "worker",
                "task_id": "adversarial_review",
                "output": {"evidence_selection": scope},
            }
        },
        "events": [
            {
                "event": "evidence.retrieve.complete",
                "data": {
                    "selected_items": items,
                    "section_evidence_ids": {
                        k: copy.deepcopy(v)
                        for k, v in scope["section_evidence_ids"].items()
                        if k not in {"abstract", "conclusion"}
                    },
                    "candidate_count": 2,
                    "ranking_decisions": decisions,
                },
            }
        ],
    }


def test_frozen_rank_one_becomes_legal_in_method_schema_without_changing_cards():
    artifact = frozen()
    before = copy.deepcopy(artifact)
    config = load_config()
    config.context.evidence_top_k_per_section = 1
    correction = _frozen_repair_scope(artifact, config)
    assert correction["selection"]["section_evidence_ids"]["methods"] == ["a"]
    assert correction["selected_source_count"] == 2 and artifact == before
    cards = artifact["report"]["metadata"]["fact_cards"]
    messages = grounded_messages(
        cards,
        "academic_v2",
        "en",
        "network",
        evidence_selection=correction["selection"],
    )
    prepared, fmt = prepare_grounded_request(messages)
    packet = __import__("json").loads(prepared[1]["content"])
    cid = next(k for k, c in cards.items() if c["evidence_id"] == "a")
    assert "methods" in packet["cards"][cid]["allowed_section_keys"]
    assert cid in str(fmt["json_schema"]["schema"])


@pytest.mark.parametrize(
    "fault",
    [
        "unknown_selected",
        "source_changed",
        "duplicate_rank",
        "unknown_section",
        "missing_rank",
        "mismatched_summary",
    ],
)
def test_frozen_invalid_routes_do_not_silently_authorize_material(fault):
    artifact = frozen()
    record = artifact["events"][0]["data"]
    if fault == "unknown_selected":
        record["selected_items"]["forged"] = {
            "evidence_id": "forged",
            "paper_id": "new",
            "supporting_text": "new",
        }
    elif fault == "source_changed":
        record["selected_items"]["a"]["supporting_text"] = "Changed source"
    elif fault == "duplicate_rank":
        record["ranking_decisions"][1]["rank"] = 1
    elif fault == "unknown_section":
        record["ranking_decisions"][0]["section"] = "forged"
    elif fault == "missing_rank":
        record["ranking_decisions"][0].pop("rank")
    else:
        artifact["nodes"]["write"]["output"]["evidence_selection"][
            "section_evidence_ids"
        ]["introduction"] = ["b"]
    with pytest.raises(ValueError):
        _frozen_repair_scope(artifact, load_config())


def test_history_without_ranking_record_keeps_compatibility():
    assert _frozen_repair_scope({"events": []}, load_config()) is None
