"""Bind the actual compact worker producer to verified frozen evidence."""

import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_extraction_verification import selection, verify

from litagent.agents.adversarial import AdversarialReviewWorker
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import build_fact_cards, digest
from scripts.run_frozen_generation import validate_frozen_worker_evidence


async def actual_worker(tmp_path, monkeypatch):
    """Use the real export path rather than inventing selected_items in its summary."""
    selected, receipt, _ = await verify(tmp_path, monkeypatch)
    cards = build_fact_cards(selected.selected_items)
    synth = SimpleNamespace(
        execute=AsyncMock(
            return_value={
                "draft": "# Review\n\nEViT improves speed by 50% on DeiT-S. [E:a]",
                "quality_policy": "strict_auto_v2",
                "evidence_selection": selected.to_dict(),
                "fact_cards": cards,
                "evidence_snapshot_sha256": digest(cards),
                "extraction_verification": receipt,
            }
        )
    )
    reviewer = SimpleNamespace(
        execute=AsyncMock(
            return_value={
                "score": 1.0,
                "verdict": "accept",
                "issues": [],
                "evidence_compliance": {
                    "passed": True,
                    "unsupported_claims": [],
                    "unknown_evidence_ids": [],
                },
                "language_compliance": {"passed": True},
            }
        )
    )
    worker = AdversarialReviewWorker(None, synth, reviewer, max_rounds=1)
    result = await worker.execute(
        SubTask(
            "review",
            "offline",
            "adversarial_review",
            input_data={"quality_policy": "strict_auto_v2"},
        )
    )
    assert "selected_items" not in result["evidence_selection"]
    inputs = {
        "fact_cards": build_fact_cards(selection().selected_items),
        "evidence_selection": selection().to_dict(),
    }
    return inputs, result, selected


@pytest.mark.asyncio
async def test_real_worker_summary_and_exact_full_snapshot_bind_same_evidence(
    tmp_path, monkeypatch
):
    inputs, result, selected = await actual_worker(tmp_path, monkeypatch)
    before = copy.deepcopy(result)
    assert validate_frozen_worker_evidence(result, inputs) == selected.selected_items
    assert result == before
    result["evidence_selection"] = selected.to_dict()
    assert validate_frozen_worker_evidence(result, inputs) == selected.selected_items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "count",
        "bool_count",
        "section",
        "method",
        "rejected",
        "card",
        "card_missing",
        "receipt",
        "source_input",
        "full_snapshot",
        "digest",
        "duplicate_section",
        "foreign_selection",
    ],
)
async def test_invalid_materials_or_summary_never_pass_by_projection(
    tmp_path, monkeypatch, mutation
):
    inputs, result, selected = await actual_worker(tmp_path, monkeypatch)
    if mutation == "missing":
        result["evidence_selection"].pop("valid")
    elif mutation == "count":
        result["evidence_selection"]["selected_count"] += 1
    elif mutation == "bool_count":
        result["evidence_selection"]["selected_count"] = True
    elif mutation == "section":
        result["evidence_selection"]["section_evidence_ids"]["methods"].append("b")
    elif mutation == "method":
        result["evidence_selection"]["method_by_section"]["methods"] = "invented"
    elif mutation == "rejected":
        result["fact_cards"] = inputs["fact_cards"]
    elif mutation == "card":
        next(iter(result["fact_cards"].values()))["source_quote"] = "another paper"
    elif mutation == "card_missing":
        result["fact_cards"] = {}
    elif mutation == "receipt":
        result["extraction_verification"]["accepted_evidence_ids"].append("b")
    elif mutation == "source_input":
        next(iter(inputs["fact_cards"].values()))["source_quote"] = "changed input"
    elif mutation == "digest":
        from litagent.report_audit import digest as plan_digest

        result["evidence_snapshot_sha256"] = plan_digest(result["fact_cards"])
    elif mutation == "duplicate_section":
        result["evidence_selection"]["section_evidence_ids"]["methods"].append("a")
    elif mutation == "foreign_selection":
        inputs["evidence_selection"]["estimated_tokens"] += 1
    else:
        result["evidence_selection"] = selected.to_dict()
        result["evidence_selection"]["selected_items"][
            "b"
        ] = selection().selected_items["b"]
    with pytest.raises(ValueError, match="frozen_generation_result_evidence_changed"):
        validate_frozen_worker_evidence(result, inputs)


@pytest.mark.asyncio
async def test_real_trial_reaches_unique_report_preparation_after_compact_export(
    tmp_path, monkeypatch
):
    from test_frozen_generation_jev import trial

    from litagent.runner import LitAgent

    prepare_report = AsyncMock(side_effect=ValueError("offline_after_freeze"))
    monkeypatch.setattr(LitAgent, "_prepare_report", prepare_report)
    _, _, _, _, state, artifact, _, _, _ = await trial(
        tmp_path, monkeypatch, valid_writer=True
    )
    prepare_report.assert_awaited_once()
    worker = prepare_report.call_args.kwargs["results"]["adversarial_review"]
    assert "selected_items" not in worker["evidence_selection"]
    assert worker["evidence_selection"]["valid"] is True
    assert (
        state["reason"] == "frozen_generation_error:ValueError"
        and not artifact["report"]
    )
    assert state["costs"]["judge"]["requests"] == 0
    assert state["costs"]["survey"]["usage_complete"]
