"""Bind budget preflight and synthesis to one immutable host-method producer."""

import copy
from unittest.mock import AsyncMock

import pytest

from litagent.agents.synthesis import SynthesisWorker
from litagent.context.evidence_selector import EvidenceSelection
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import build_fact_cards
from litagent.review_structure import (
    method_material_coverage,
    prepare_writer_methodology,
)


@pytest.fixture
def material():
    """Keep actual source identity separate from extraction and search metadata."""
    item = {
        "evidence_id": "a",
        "paper_id": "evit",
        "text": "EViT reduces computation on DeiT-S.",
        "supporting_text": "EViT reduces computation on DeiT-S.",
        "chunk_key": "abstract",
        "content_scope": "abstract",
    }
    sections = {
        key: ["a"]
        for key in (
            "introduction",
            "taxonomy",
            "methods",
            "experiments",
            "open_problems",
        )
    }
    selection = EvidenceSelection(
        candidate_count=3,
        selected_items={"a": item},
        section_evidence_ids=sections,
        method_by_section={key: "lexical_fallback" for key in sections},
        omitted_count=2,
        estimated_tokens=100,
    )
    extractions = [
        {
            "paper_id": "evit",
            "fulltext_acquisition": {
                "status": "failed",
                "reason_code": "fulltext_unavailable",
                "full_document_reviewed": False,
            },
            "evidence_items": [item],
        }
    ]
    return {
        "record": {"queries": [{"query": "EViT", "status": "completed"}]},
        "selection": selection,
        "cards": build_fact_cards(selection.selected_items),
        "extractions": extractions,
        "requested_methods": ["EViT", "Evo-ViT"],
        "upstream_results": {"extract": extractions},
    }


def test_prepared_record_matches_original_strict_producer_and_is_idempotent(material):
    """Preserve original valid disclosure values without sharing nested storage."""
    original = copy.deepcopy(material)
    prepared = prepare_writer_methodology(**material)
    expected = copy.deepcopy(material["record"])
    expected["evidence_counts"] = {"candidate": 3, "selected": 1}
    expected["named_method_material_coverage"] = method_material_coverage(
        material["requested_methods"],
        material["upstream_results"],
        material["selection"].selected_items,
    )
    expected["fulltext_acquisition"] = {
        "evit": material["extractions"][0]["fulltext_acquisition"]
    }
    expected["content_mode"] = "abstract_only"
    expected["full_text_review"] = (
        "not performed; no selected fulltext evidence available"
    )
    assert prepared == expected
    assert prepare_writer_methodology(**{**material, "record": prepared}) == prepared
    assert material == original
    prepared["queries"][0]["query"] = "changed"
    prepared["fulltext_acquisition"]["evit"]["status"] = "changed"
    assert material == original


@pytest.mark.parametrize("scope", ["abstract", "selected_fulltext"])
def test_prepared_fulltext_scope_only_describes_selected_blocks(material, scope):
    """Acquisition success never implies complete document review."""
    material["selection"].selected_items["a"]["content_scope"] = scope
    material["cards"] = build_fact_cards(material["selection"].selected_items)
    material["extractions"][0]["fulltext_acquisition"]["status"] = "available"
    result = prepare_writer_methodology(**material)
    assert result["content_mode"] == (
        "abstract_and_selected_fulltext"
        if scope == "selected_fulltext"
        else "abstract_only"
    )
    assert result["full_text_review"] == (
        "selected blocks only; complete document review not performed"
        if scope == "selected_fulltext"
        else "not performed; no selected fulltext evidence available"
    )


def test_missing_acquisition_and_upstream_are_not_fabricated(material):
    """Retain absent host fields and explicitly empty acquisition data."""
    material["extractions"] = [{"paper_id": "evit"}]
    material["upstream_results"] = None
    result = prepare_writer_methodology(**material)
    assert result["fulltext_acquisition"] == {}
    assert "content_mode" not in result
    assert "full_text_review" not in result
    coverage = result["named_method_material_coverage"]["methods"]
    assert coverage["EViT"]["selected"] is True
    assert coverage["EViT"]["extract"] is False
    assert all(value is False for value in coverage["Evo-ViT"].values())


@pytest.mark.parametrize(
    ("field", "invalid", "code"),
    [
        ("record", None, "writer_envelope_methodology_required"),
        ("record", [], "writer_methodology_record_invalid"),
        ("selection", {}, "writer_methodology_selection_invalid"),
        ("cards", [], "writer_methodology_fact_cards_invalid"),
        ("extractions", "extract", "writer_methodology_extractions_invalid"),
        ("extractions", [None], "writer_methodology_extractions_invalid"),
        ("requested_methods", "EViT", "writer_methodology_requested_methods_invalid"),
        ("requested_methods", [None], "writer_methodology_requested_methods_invalid"),
        ("requested_methods", [""], "writer_methodology_requested_methods_invalid"),
        ("upstream_results", [], "writer_methodology_upstream_results_invalid"),
        ("upstream_results", {1: []}, "writer_methodology_upstream_results_invalid"),
        (
            "extractions",
            [{"fulltext_acquisition": {"status": "available"}}],
            "writer_methodology_acquisition_invalid",
        ),
    ],
)
def test_invalid_inputs_have_explicit_errors_without_mutation(
    material, field, invalid, code
):
    """Reject malformed host input before model transport can be initialized."""
    material[field] = invalid
    before = copy.deepcopy(material)
    with pytest.raises(ValueError, match=f"^{code}$"):
        prepare_writer_methodology(**material)
    assert material == before


@pytest.mark.parametrize(
    "field", ["source_quote", "paper_id", "content_sha256", "source_start", "method"]
)
def test_fact_card_lineage_and_candidate_fields_cannot_drift(material, field):
    """A valid-looking card ID cannot replace source or extracted candidate data."""
    next(iter(material["cards"].values()))[field] = "changed"
    before = copy.deepcopy(material)
    with pytest.raises(ValueError, match="writer_methodology_fact_cards_invalid"):
        prepare_writer_methodology(**material)
    assert material == before


@pytest.mark.parametrize(
    ("field", "value"), [("source_start", False), ("source_end", True)]
)
def test_boolean_offsets_cannot_impersonate_integer_source_identity(
    material, field, value
):
    """JSON identity distinguishes bool offsets from equal Python integers."""
    material["selection"].selected_items["a"].update(text="x", supporting_text="x")
    material["cards"] = build_fact_cards(material["selection"].selected_items)
    next(iter(material["cards"].values()))[field] = value
    before = copy.deepcopy(material)
    with pytest.raises(ValueError, match="writer_methodology_fact_cards_invalid"):
        prepare_writer_methodology(**material)
    assert material == before


@pytest.mark.parametrize("count", [True, -1, 0])
def test_invalid_selection_counts_fail_before_preparation(material, count):
    """Counts are real evidence counts, not booleans or inconsistent metadata."""
    material["selection"].candidate_count = count
    with pytest.raises(ValueError, match="writer_methodology_selection_invalid"):
        prepare_writer_methodology(**material)


@pytest.mark.asyncio
async def test_actual_synthesis_uses_shared_v4_preparation(material, monkeypatch):
    """Inspect the real worker's prepared boundary without purchasing a model call."""
    before = copy.deepcopy(material)
    selector = AsyncMock()
    selector.select.return_value = copy.deepcopy(material["selection"])
    expected = prepare_writer_methodology(**material)
    captured = {}

    async def capture(worker, task, cards, selection, methodology, requirements):
        captured["methodology"] = copy.deepcopy(methodology)
        assert cards == material["cards"]
        return {"prepared": methodology}

    monkeypatch.setattr("litagent.writing_v2.synthesize", capture)
    worker = SynthesisWorker(
        AsyncMock(),
        evidence_selector=selector,
        methodology_provider=lambda task: material["record"],
        frozen_writer_plan_provider=lambda: {"expected": "server-only"},
    )
    task = SubTask(
        "synthesis",
        "write",
        "synthesis",
        input_data={
            "query": "Explain EViT and Evo-ViT.",
            "quality_policy": "strict_auto_v2",
            "generation_strategy": "body_summary_v2",
            "budget_policy": "planned-budget-v4",
            "review_structure": "academic_v2",
            "requested_methods": material["requested_methods"],
            "verification": {"judge": False, "jev": False},
            "upstream_results": material["upstream_results"],
        },
    )
    assert (await worker.execute(task))["prepared"] == expected
    assert captured["methodology"] == expected
    assert worker._frozen_writer_plan_provider() == {"expected": "server-only"}
    assert material == before
